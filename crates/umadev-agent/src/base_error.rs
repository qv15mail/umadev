//! Base-failure classifier — turns a raw base failure (an idle hang, a non-zero
//! exit, a stderr tail, a JSON-RPC error) into a typed [`BaseFailure`] plus an
//! actionable, per-base, i18n diagnosis the user can act on.
//!
//! Today a base failure surfaced only a raw stderr tail (or nothing), so a hung
//! `claude` with a bad key read as a cause-less "base session idle". This module
//! is the "errors are data" layer: [`classify`] pattern-matches the evidence we
//! actually captured (exit status string + stderr tail + an extra reason/JSON-RPC
//! string), and [`actionable_message`] names WHAT failed + HOW to fix it, per
//! base, as a localized string.
//!
//! Design notes:
//! - **Pure + dependency-free + fail-open.** [`classify`] is a total function over
//!   three optional strings; it never touches the filesystem or network, never
//!   panics, and empty / unrecognised input collapses to [`BaseFailure::Unknown`]
//!   (which the mint points map back to today's behaviour). No `regex` dep — plain
//!   `str` scanning keeps the crate light.
//! - **Ordered cascade, most specific first.** Auth (401/403) is checked before a
//!   generic rate limit; a JSON-RPC `-32001` / `529` "overloaded" before network;
//!   the first family to match wins. A captured non-zero exit with no textual
//!   match degrades to [`BaseFailure::Exited`] carrying the code, never a panic.
//! - **Wiring is centralised.** The two failure mint points
//!   (`director_loop::enrich_idle_reason` for the `/run` path,
//!   `umadev-tui`'s `enrich_base_failure` for the chat path) call [`classify`]
//!   FIRST, PREPEND [`actionable_message`], and KEEP the raw stderr tail appended
//!   as the technical detail — so power users still see the verbatim base error.

/// What kind of base failure the captured evidence points to.
///
/// Every variant maps to a per-base, actionable [`actionable_message`].
/// [`BaseFailure::Unknown`] is the fail-open floor: the mint point keeps today's
/// bare-reason behaviour (no actionable line prepended).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseFailure {
    /// Not logged in / unauthorized / bad-or-expired API key (401/403). The user
    /// must re-auth the base; UmaDev cannot fix this itself.
    Auth,
    /// The base hit a rate limit / quota / usage cap (429). Transient — retry or
    /// switch model.
    RateLimit,
    /// The base model endpoint could not be reached. `ssl` is `true` when the
    /// failure is specifically an SSL/TLS/certificate verification problem (proxy
    /// or corporate CA), `false` for a plain connectivity failure (refused /
    /// reset / timeout / DNS).
    Network {
        /// `true` ⇒ SSL/TLS/certificate verification failure (distinct fix:
        /// proxy / `NODE_EXTRA_CA_CERTS`), `false` ⇒ plain connectivity failure.
        ssl: bool,
    },
    /// The prompt / conversation exceeded the model's maximum context length.
    Context,
    /// The base is overloaded / at capacity / busy (529, codex JSON-RPC
    /// `-32001`). Transient — retry or switch model/base.
    Overloaded,
    /// The base process exited non-zero and nothing else matched; carries the
    /// captured exit code (`-1` when the process died without a parseable code,
    /// e.g. killed by a signal).
    Exited(i32),
    /// Nothing matched — the fail-open floor. The mint point keeps today's bare
    /// reason (no actionable line).
    Unknown,
}

/// Classify a base failure from the evidence we actually captured.
///
/// - `exit_status` — the formatted [`std::process::ExitStatus`] of a non-zero
///   exit (the mint points pass this ONLY when the process exited unsuccessfully,
///   so a present value always means "the process failed").
/// - `stderr_tail` — the base's last stderr lines (where a broken model/login
///   config writes its error; it never goes to stdout).
/// - `extra` — any additional reason string we have, e.g. the base's own failure
///   reason or a JSON-RPC error object (`{"code":-32001,"message":"overloaded"}`).
///
/// Pure + fail-open: all-`None` input → [`BaseFailure::Unknown`]; never panics.
#[must_use]
pub fn classify(
    exit_status: Option<&str>,
    stderr_tail: Option<&str>,
    extra: Option<&str>,
) -> BaseFailure {
    // Build one lowercased haystack from the textual evidence (stderr + extra).
    let mut hay = String::new();
    if let Some(s) = stderr_tail {
        hay.push_str(&s.to_ascii_lowercase());
        hay.push(' ');
    }
    if let Some(s) = extra {
        hay.push_str(&s.to_ascii_lowercase());
    }
    let hay = hay.as_str();

    // Ordered, most-specific first. The first family to fire wins.
    if is_auth(hay) {
        return BaseFailure::Auth;
    }
    if is_rate_limit(hay) {
        return BaseFailure::RateLimit;
    }
    if is_overloaded(hay) {
        return BaseFailure::Overloaded;
    }
    if is_context(hay) {
        return BaseFailure::Context;
    }
    if let Some(ssl) = is_network(hay) {
        return BaseFailure::Network { ssl };
    }

    // No textual family matched. A captured non-zero exit is still a hard
    // failure — carry its code so the message can name it.
    if let Some(es) = exit_status {
        return BaseFailure::Exited(parse_exit_code(es).unwrap_or(-1));
    }

    BaseFailure::Unknown
}

/// The per-base, actionable, localized diagnosis for a [`BaseFailure`].
///
/// Returns a short imperative line that names the fix for THIS base (so a `claude`
/// auth failure says "run `claude /login`" while `codex` says "check codex
/// login"). [`BaseFailure::Unknown`] returns an empty string — the caller then
/// falls back to today's generic reason (no actionable line prepended).
#[must_use]
pub fn actionable_message(f: &BaseFailure, backend: &str) -> String {
    match f {
        BaseFailure::Auth => umadev_i18n::tl(auth_key(backend)).to_string(),
        BaseFailure::RateLimit => umadev_i18n::tl("base.fail.ratelimit").to_string(),
        BaseFailure::Overloaded => umadev_i18n::tl("base.fail.overloaded").to_string(),
        BaseFailure::Network { ssl: false } => umadev_i18n::tl("base.fail.network").to_string(),
        BaseFailure::Network { ssl: true } => umadev_i18n::tl("base.fail.network.ssl").to_string(),
        BaseFailure::Context => umadev_i18n::tl("base.fail.context").to_string(),
        BaseFailure::Exited(code) => umadev_i18n::tlf("base.fail.exited", &[&code.to_string()]),
        BaseFailure::Unknown => String::new(),
    }
}

/// Pick the per-base i18n key for an auth failure. Falls back to a base-agnostic
/// key for an unknown / empty backend id.
fn auth_key(backend: &str) -> &'static str {
    match backend {
        "claude-code" | "claude" => "base.fail.auth.claude",
        "codex" => "base.fail.auth.codex",
        "opencode" => "base.fail.auth.opencode",
        _ => "base.fail.auth.generic",
    }
}

// ---------------------------------------------------------------------------
// Family detectors — each a pure substring scan over the lowercased haystack.
// ---------------------------------------------------------------------------

/// Not logged in / unauthorized / bad-or-expired key (401/403).
fn is_auth(hay: &str) -> bool {
    const MARKERS: &[&str] = &[
        "unauthorized",
        "unauthenticated",
        "authentication",
        "authorization failed",
        "auth failed",
        "auth error",
        "401",
        "403",
        "forbidden",
        "api key",
        "api-key",
        "apikey",
        "x-api-key",
        "not logged in",
        "not authenticated",
        "please log in",
        "please login",
        "log in to",
        "/login",
        "login required",
        "logged out",
        "invalid key",
        "invalid token",
        "invalid credentials",
        "credential",
        "token expired",
        "expired token",
        "token has expired",
        "session expired",
    ];
    MARKERS.iter().any(|m| hay.contains(m))
}

/// Rate limit / quota / usage cap (429).
fn is_rate_limit(hay: &str) -> bool {
    const MARKERS: &[&str] = &[
        "rate limit",
        "rate-limit",
        "ratelimit",
        "rate_limit",
        "429",
        "too many requests",
        "quota",
        "usage limit",
    ];
    MARKERS.iter().any(|m| hay.contains(m))
}

/// Overloaded / at capacity / busy (529, codex JSON-RPC `-32001`).
fn is_overloaded(hay: &str) -> bool {
    const MARKERS: &[&str] = &[
        "overloaded",
        "overload",
        "529",
        "-32001",
        "at capacity",
        "over capacity",
        "capacity",
        "server is busy",
        "service is busy",
    ];
    MARKERS.iter().any(|m| hay.contains(m))
}

/// Prompt / conversation exceeded the model's maximum context length.
fn is_context(hay: &str) -> bool {
    const MARKERS: &[&str] = &[
        "context length",
        "context window",
        "maximum context",
        "max context",
        "context_length_exceeded",
        "prompt is too long",
        "prompt too long",
        "input is too long",
        "too many tokens",
        "token limit",
        "maximum number of tokens",
        "exceeds the maximum",
        "reduce the length",
    ];
    MARKERS.iter().any(|m| hay.contains(m))
}

/// Network reachability failure. Returns `Some(true)` for an SSL/TLS/cert
/// problem, `Some(false)` for a plain connectivity failure, `None` if neither.
///
/// SSL markers are checked FIRST so a cert failure is reported with its distinct
/// fix (proxy / `NODE_EXTRA_CA_CERTS`) rather than a generic "check your network".
fn is_network(hay: &str) -> Option<bool> {
    const SSL_MARKERS: &[&str] = &[
        "ssl",
        "tls",
        "certificate",
        "self-signed",
        "self signed",
        "unable to verify",
        "unable to get local issuer",
        "cert_",
        "err_cert",
        "x509",
        "handshake failed",
        "ssl_error",
        "sslerror",
    ];
    const NET_MARKERS: &[&str] = &[
        "connection refused",
        "connection reset",
        "econnrefused",
        "econnreset",
        "etimedout",
        "timed out",
        "timeout",
        "unable to connect",
        "could not connect",
        "failed to connect",
        "cannot connect",
        "enotfound",
        "getaddrinfo",
        "network is unreachable",
        "no route to host",
        "name resolution",
        "temporary failure in name resolution",
        "socket hang up",
        "network error",
        "connection error",
    ];
    if SSL_MARKERS.iter().any(|m| hay.contains(m)) {
        return Some(true);
    }
    if NET_MARKERS.iter().any(|m| hay.contains(m)) {
        return Some(false);
    }
    None
}

/// Parse the first (optionally signed) integer out of a formatted
/// [`std::process::ExitStatus`] string (e.g. `"exit status: 2"` →
/// `Some(2)`, `"signal: 9 (SIGKILL)"` → `Some(9)`). `None` when no digit is
/// present.
fn parse_exit_code(s: &str) -> Option<i32> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = if i > 0 && bytes[i - 1] == b'-' {
                i - 1
            } else {
                i
            };
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            return s[start..j].parse::<i32>().ok();
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_unknown() {
        // Fail-open floor: nothing captured → Unknown, no panic.
        assert_eq!(classify(None, None, None), BaseFailure::Unknown);
        assert_eq!(classify(None, Some(""), Some("")), BaseFailure::Unknown);
    }

    #[test]
    fn auth_from_varied_markers() {
        assert_eq!(
            classify(None, Some("error: invalid x-api-key"), None),
            BaseFailure::Auth
        );
        assert_eq!(
            classify(None, Some("Error 401 Unauthorized"), None),
            BaseFailure::Auth
        );
        assert_eq!(
            classify(None, Some("you are not logged in, run claude /login"), None),
            BaseFailure::Auth
        );
        assert_eq!(
            classify(None, Some("authentication token has expired"), None),
            BaseFailure::Auth
        );
        // A 403 is auth, not a generic rate limit (order check).
        assert_eq!(
            classify(None, Some("403 forbidden"), None),
            BaseFailure::Auth
        );
    }

    #[test]
    fn rate_limit_from_varied_markers() {
        assert_eq!(
            classify(None, Some("429 Too Many Requests"), None),
            BaseFailure::RateLimit
        );
        assert_eq!(
            classify(None, Some("You have hit the rate limit"), None),
            BaseFailure::RateLimit
        );
        assert_eq!(
            classify(None, Some("usage limit reached for this org"), None),
            BaseFailure::RateLimit
        );
        assert_eq!(
            classify(None, Some("quota exceeded"), None),
            BaseFailure::RateLimit
        );
    }

    #[test]
    fn overloaded_including_codex_jsonrpc_minus_32001() {
        // The codex overloaded surface: a JSON-RPC error object with code -32001.
        assert_eq!(
            classify(
                None,
                None,
                Some(r#"jsonrpc error: {"code":-32001,"message":"overloaded"}"#)
            ),
            BaseFailure::Overloaded
        );
        assert_eq!(
            classify(None, Some("HTTP 529 overloaded"), None),
            BaseFailure::Overloaded
        );
        assert_eq!(
            classify(None, Some("the server is at capacity, try again"), None),
            BaseFailure::Overloaded
        );
    }

    #[test]
    fn network_plain_vs_ssl() {
        // Plain connectivity → ssl:false.
        assert_eq!(
            classify(None, Some("connect ECONNREFUSED 127.0.0.1:443"), None),
            BaseFailure::Network { ssl: false }
        );
        assert_eq!(
            classify(None, Some("getaddrinfo ENOTFOUND api.example.com"), None),
            BaseFailure::Network { ssl: false }
        );
        assert_eq!(
            classify(None, Some("request timed out"), None),
            BaseFailure::Network { ssl: false }
        );
        // SSL/cert → ssl:true (the distinct fix path).
        assert_eq!(
            classify(None, Some("unable to verify the first certificate"), None),
            BaseFailure::Network { ssl: true }
        );
        assert_eq!(
            classify(None, Some("SELF_SIGNED_CERT_IN_CHAIN"), None),
            BaseFailure::Network { ssl: true }
        );
        assert_eq!(
            classify(None, Some("SSL handshake failed"), None),
            BaseFailure::Network { ssl: true }
        );
    }

    #[test]
    fn context_overflow() {
        assert_eq!(
            classify(None, Some("prompt is too long: 250000 tokens"), None),
            BaseFailure::Context
        );
        assert_eq!(
            classify(
                None,
                Some("This model's maximum context length is 200000 tokens"),
                None
            ),
            BaseFailure::Context
        );
        assert_eq!(
            classify(None, Some("context_length_exceeded"), None),
            BaseFailure::Context
        );
    }

    #[test]
    fn exited_carries_the_code_when_nothing_else_matches() {
        // A non-zero exit with no recognisable text → Exited(code).
        assert_eq!(
            classify(Some("exit status: 2"), Some("something opaque"), None),
            BaseFailure::Exited(2)
        );
        // Killed by a signal (no exit code) → still a hard exit; -1 sentinel when
        // a code can't be parsed, but the signal number parses here.
        assert_eq!(
            classify(Some("signal: 9 (SIGKILL)"), None, None),
            BaseFailure::Exited(9)
        );
        // Present exit string with no digits → Exited(-1) sentinel, never a panic.
        assert_eq!(
            classify(Some("killed"), None, None),
            BaseFailure::Exited(-1)
        );
    }

    #[test]
    fn text_match_wins_over_exit_code() {
        // Even with a non-zero exit, a recognised stderr family takes precedence
        // (the cause is more actionable than "exited N").
        assert_eq!(
            classify(Some("exit status: 1"), Some("error: invalid api key"), None),
            BaseFailure::Auth
        );
    }

    #[test]
    fn actionable_message_is_per_base_for_auth() {
        // The auth message names the fix for THIS base — distinct key per backend.
        assert_eq!(
            actionable_message(&BaseFailure::Auth, "claude-code"),
            umadev_i18n::tl("base.fail.auth.claude")
        );
        assert_eq!(
            actionable_message(&BaseFailure::Auth, "codex"),
            umadev_i18n::tl("base.fail.auth.codex")
        );
        assert_eq!(
            actionable_message(&BaseFailure::Auth, "opencode"),
            umadev_i18n::tl("base.fail.auth.opencode")
        );
        // An unknown / empty backend falls back to the base-agnostic key.
        assert_eq!(
            actionable_message(&BaseFailure::Auth, ""),
            umadev_i18n::tl("base.fail.auth.generic")
        );
        // And the per-base keys are actually different strings.
        assert_ne!(
            actionable_message(&BaseFailure::Auth, "claude-code"),
            actionable_message(&BaseFailure::Auth, "codex")
        );
    }

    #[test]
    fn actionable_message_maps_each_variant_to_its_key() {
        assert_eq!(
            actionable_message(&BaseFailure::RateLimit, "codex"),
            umadev_i18n::tl("base.fail.ratelimit")
        );
        assert_eq!(
            actionable_message(&BaseFailure::Overloaded, "codex"),
            umadev_i18n::tl("base.fail.overloaded")
        );
        assert_eq!(
            actionable_message(&BaseFailure::Network { ssl: false }, "codex"),
            umadev_i18n::tl("base.fail.network")
        );
        assert_eq!(
            actionable_message(&BaseFailure::Network { ssl: true }, "codex"),
            umadev_i18n::tl("base.fail.network.ssl")
        );
        assert_eq!(
            actionable_message(&BaseFailure::Context, "codex"),
            umadev_i18n::tl("base.fail.context")
        );
        // Exited names the code via a positional placeholder.
        let m = actionable_message(&BaseFailure::Exited(137), "codex");
        assert!(m.contains("137"), "exit message names the code: {m}");
        // Unknown is empty → the mint point keeps today's generic reason.
        assert_eq!(actionable_message(&BaseFailure::Unknown, "codex"), "");
    }

    #[test]
    fn parse_exit_code_extracts_first_integer() {
        assert_eq!(parse_exit_code("exit status: 2"), Some(2));
        assert_eq!(parse_exit_code("signal: 9 (SIGKILL)"), Some(9));
        assert_eq!(parse_exit_code("exit code: 137"), Some(137));
        assert_eq!(parse_exit_code("no digits here"), None);
    }
}
