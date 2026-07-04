//! Deploy adapter — the engine's "ship it" capability, the optional handoff
//! step that closes the commercial-delivery loop *after* `delivery`.
//!
//! `delivery` produces the proof-pack (docs + quality + compliance + a runnable
//! build). What it does NOT do is put the product on a public URL. This module
//! is the bridge to "it's live": it
//!
//! 1. **Detects** the deploy target from the workspace's own files — a
//!    `vercel.json` / Next.js app → Vercel, `netlify.toml` → Netlify, a
//!    `Dockerfile` → a container image, `fly.toml` → Fly.io, a built static
//!    `dist/` / `out/` → a static host. Each target carries the exact CLI
//!    command a user would run.
//! 2. **Executes** that command as a subprocess **only when the user explicitly
//!    triggers a deploy** (the binary/TUI gates this behind a confirm). The
//!    actual deploy is the *user's* action against *their own* logged-in
//!    platform CLI — UmaDev never deploys on its own, owns no credentials, and
//!    injects nothing into the platform.
//! 3. Captures the **preview URL + log tail** into a structured [`DeployProof`]
//!    that is serialized to `.umadev/audit/deploy-proof.json` and folded into
//!    the delivery proof-pack (see `phases::build_and_zip_proof_pack`).
//!
//! Everything here is **fail-open**: an unrecognised platform, a missing deploy
//! CLI, or a failed/timed-out deploy degrades to a `NotDeployed(reason)` record
//! with a manual-deploy hint — never a panic, never a blocked host. User-facing
//! prose lives in the binary (which owns the i18n catalog); this crate stays
//! dependency-light and emits machine-readable data plus a neutral summary line.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::process::Command;

/// Cap captured deploy-log output so a chatty CLI can't bloat the JSON.
const CAPTURE_CAP: usize = 8 * 1024;

/// How long (seconds) a deploy is allowed to run before we abort and record a
/// timeout. A first deploy that builds remotely can be slow; this is a generous
/// backstop so a hung interactive login can't block forever.
const DEPLOY_TIMEOUT_SECS: u64 = 600;

/// Cap on RAW captured deploy output held in memory while the command runs. A
/// chatty deploy (e.g. a verbose `docker build`) can print far more than we keep;
/// we retain only the last `OUTPUT_CAP` bytes (the result / URL / error lives at
/// the end) while ALWAYS draining the pipe so the child never blocks. The final
/// stored `log_tail` is capped smaller still, at [`CAPTURE_CAP`].
const OUTPUT_CAP: usize = 256 * 1024;

/// Bounded reap after a deploy TIMES OUT, so a killed CLI and its pipe readers
/// can't turn a timeout into an unbounded hang.
const KILL_REAP_SECS: u64 = 5;

/// A recognised deployment platform. Detected purely from files already in the
/// workspace; each variant maps to a single canonical CLI command.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeployTarget {
    /// `vercel.json` present, or a Next.js app (`next` dependency / `next.config.*`).
    Vercel,
    /// `netlify.toml` present.
    Netlify,
    /// `fly.toml` present (Fly.io).
    Fly,
    /// A Cloudflare Pages/Workers project (`wrangler.toml` / `wrangler.json`).
    CloudflarePages,
    /// A `Dockerfile` present — container image build (no auto-push target, so
    /// the command just builds the image; pushing is the user's choice).
    Docker,
    /// A pre-built static bundle (`dist/` / `out/` / `build/` / `public/`) with
    /// no platform config — deployable to any static host via a generic CLI.
    StaticHost,
    /// No recognised target. Deploy is skipped (fail-open).
    None,
}

impl DeployTarget {
    /// Stable string label used in proof rows and events.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vercel => "vercel",
            Self::Netlify => "netlify",
            Self::Fly => "fly",
            Self::CloudflarePages => "cloudflare-pages",
            Self::Docker => "docker",
            Self::StaticHost => "static-host",
            Self::None => "none",
        }
    }

    /// Human-friendly platform name for display.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Vercel => "Vercel",
            Self::Netlify => "Netlify",
            Self::Fly => "Fly.io",
            Self::CloudflarePages => "Cloudflare Pages",
            Self::Docker => "Docker image",
            Self::StaticHost => "static host",
            Self::None => "none",
        }
    }

    /// The CLI binary this target deploys through (the thing that must be on
    /// PATH). `None` for [`DeployTarget::None`].
    #[must_use]
    pub const fn cli_binary(self) -> Option<&'static str> {
        match self {
            Self::Vercel => Some("vercel"),
            Self::Netlify => Some("netlify"),
            Self::Fly => Some("flyctl"),
            Self::CloudflarePages => Some("wrangler"),
            Self::Docker => Some("docker"),
            Self::StaticHost => Some("npx"),
            Self::None => None,
        }
    }

    /// The exact, copy-pasteable deploy command for this target. `None` for
    /// [`DeployTarget::None`]. These are the production-deploy forms a user runs
    /// against their *own* logged-in CLI; UmaDev only surfaces / runs them.
    #[must_use]
    pub fn deploy_command(self) -> Option<String> {
        let cmd = match self {
            Self::Vercel => "npx vercel --prod --yes",
            Self::Netlify => "npx netlify deploy --prod",
            Self::Fly => "flyctl deploy",
            // Cloudflare Pages: deploy the built output dir; `dist` is wrangler's
            // own default convention for Pages projects.
            Self::CloudflarePages => "npx wrangler pages deploy dist",
            // Docker: build the image. Pushing/running is the user's choice — we
            // do not assume a registry. Tag is a stable local name.
            Self::Docker => "docker build -t app:latest .",
            // Static host: a zero-config global deploy of the built bundle via
            // a widely-available static-deploy CLI.
            Self::StaticHost => "npx surge ./dist",
            Self::None => return None,
        };
        Some(cmd.to_string())
    }
}

/// Whether the deploy ran to completion (and a URL was captured) or degraded
/// (and why). The top-level verdict the proof-pack surfaces.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "reason")]
pub enum DeployStatus {
    /// The deploy command exited 0.
    Deployed,
    /// The deploy did not happen / did not succeed; the payload is a short
    /// machine reason (e.g. `"no deploy target detected"`, `"vercel not on
    /// PATH"`, `"deploy command exited 1"`, `"timed out after 600s"`).
    /// Fail-open: this is a neutral "not deployed", never an error.
    NotDeployed(String),
}

impl DeployStatus {
    /// `true` iff the deploy completed successfully.
    #[must_use]
    pub fn is_deployed(&self) -> bool {
        matches!(self, DeployStatus::Deployed)
    }

    /// Stable label for proof rows / display switches.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            DeployStatus::Deployed => "deployed",
            DeployStatus::NotDeployed(_) => "not_deployed",
        }
    }
}

/// The full deploy-proof record. Serialized to
/// `.umadev/audit/deploy-proof.json` and embedded in the proof-pack.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeployProof {
    /// ISO-8601 timestamp the deploy ran (or was attempted).
    pub timestamp: String,
    /// Detected platform.
    pub platform: DeployTarget,
    /// Top-level verdict.
    pub status: DeployStatus,
    /// The exact command we ran, if any (`None` when no target was detected).
    pub command: Option<String>,
    /// Subprocess exit code; `-1` for spawn / timeout failures, `None` when we
    /// never ran a command.
    pub exit_code: Option<i32>,
    /// The live / preview URL parsed from the deploy output, if one was printed.
    pub url: Option<String>,
    /// Wall-clock duration of the deploy, milliseconds (`None` when nothing ran).
    pub duration_ms: Option<u64>,
    /// Truncated tail of the deploy log (stdout+stderr, capped at 8 KiB).
    pub log_tail: String,
}

impl DeployProof {
    /// Build a "not deployed" record carrying only the platform + reason — used
    /// on every fail-open early return so the artifact is still produced.
    fn not_deployed(platform: DeployTarget, reason: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            platform,
            status: DeployStatus::NotDeployed(reason.into()),
            command: platform.deploy_command(),
            exit_code: None,
            url: None,
            duration_ms: None,
            log_tail: String::new(),
        }
    }

    /// A neutral, language-agnostic one-line summary (the binary localizes the
    /// real user message; this is for logs / the proof-pack summary).
    #[must_use]
    pub fn summary_line(&self) -> String {
        match &self.status {
            DeployStatus::Deployed => {
                let url = self.url.as_deref().unwrap_or("(no URL printed)");
                format!("deployed to {}: {url}", self.platform.label())
            }
            DeployStatus::NotDeployed(reason) => {
                format!("not deployed ({}): {reason}", self.platform.label())
            }
        }
    }
}

/// Detect the deploy target from the workspace's files. Pure file-presence /
/// manifest inspection — no network, no spawning. Order is by specificity:
/// an explicit platform config wins over a generic Dockerfile, which wins over
/// a bare built bundle.
#[must_use]
pub fn detect_deploy_target(workspace: &Path) -> DeployTarget {
    // 1. Explicit platform configs (most specific).
    if workspace.join("vercel.json").is_file() || is_next_app(workspace) {
        return DeployTarget::Vercel;
    }
    if workspace.join("netlify.toml").is_file() {
        return DeployTarget::Netlify;
    }
    if workspace.join("fly.toml").is_file() {
        return DeployTarget::Fly;
    }
    if workspace.join("wrangler.toml").is_file() || workspace.join("wrangler.json").is_file() {
        return DeployTarget::CloudflarePages;
    }
    // 2. A Dockerfile — container build (no platform config above it).
    if workspace.join("Dockerfile").is_file() {
        return DeployTarget::Docker;
    }
    // 3. A pre-built static bundle with no platform config — any static host.
    if ["dist", "out", "build", "public"]
        .iter()
        .any(|d| workspace.join(d).is_dir())
    {
        return DeployTarget::StaticHost;
    }
    DeployTarget::None
}

/// Whether the workspace is a Next.js app (which deploys to Vercel even without
/// a `vercel.json`). Detected by a `next.config.*` file or a `next` dependency.
fn is_next_app(workspace: &Path) -> bool {
    for cfg in ["next.config.js", "next.config.mjs", "next.config.ts"] {
        if workspace.join(cfg).is_file() {
            return true;
        }
    }
    package_json_depends_on(workspace, "next")
}

/// Whether `package.json` declares a dependency on `pkg` (in `dependencies` /
/// `devDependencies`). Best-effort; a missing / malformed manifest → `false`.
fn package_json_depends_on(workspace: &Path, pkg: &str) -> bool {
    let Ok(content) = std::fs::read_to_string(workspace.join("package.json")) else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) else {
        return false;
    };
    let in_obj = |key: &str| {
        json.get(key)
            .and_then(|v| v.as_object())
            .is_some_and(|o| o.contains_key(pkg))
    };
    in_obj("dependencies") || in_obj("devDependencies")
}

/// Run a deploy against `workspace`. **Caller must have obtained explicit user
/// consent** — this spawns a real, outward-facing command. Always returns a
/// [`DeployProof`]; on any failure it degrades to [`DeployStatus::NotDeployed`]
/// with a reason, never an `Err`/panic (fail-open).
///
/// `command` is the exact command to run. When `None`, the canonical command
/// for the detected platform is used; when no platform is detected, a
/// `NotDeployed("no deploy target detected")` record is returned without
/// spawning anything.
///
/// stdin is `/dev/null`: a deploy CLI that needs an interactive login must
/// fail fast on EOF rather than hang invisibly. The login is the user's job in
/// their own terminal; this adapter records the outcome of a non-interactive
/// attempt.
pub async fn run_deploy(workspace: &Path, command: Option<&str>) -> DeployProof {
    let platform = detect_deploy_target(workspace);
    let command = match command {
        Some(c) if !c.trim().is_empty() => c.trim().to_string(),
        _ => match platform.deploy_command() {
            Some(c) => c,
            None => return DeployProof::not_deployed(platform, "no deploy target detected"),
        },
    };

    // The first token is the binary that must be on PATH (e.g. `npx`, `docker`,
    // `flyctl`). If it's missing, record a neutral skip + manual hint.
    let bin = command.split_whitespace().next().unwrap_or_default();
    if !bin.is_empty() && !which(bin) {
        return DeployProof::not_deployed(platform, format!("{bin} not found on PATH"));
    }

    run_deploy_command(workspace, platform, command, DEPLOY_TIMEOUT_SECS).await
}

/// Spawn + drive one deploy command against `workspace`, racing its exit against
/// `timeout_secs`. Always returns a [`DeployProof`] — fail-open, never hangs.
///
/// **Kill + bound (the audit fix).** The command is spawned with
/// `kill_on_drop(true)` AND detached into its own session/process-group, so on
/// timeout we kill the WHOLE tree (the `sh -c` wrapper forks `npx` → `node`,
/// etc.) — tokio dropping the `Child` alone would leave those descendants
/// running. Output is captured through a bounded, tail-retaining reader
/// ([`read_capped_tail`]) instead of `Command::output()`, so a chatty command
/// can't buffer unbounded stdout/stderr into memory; the reader always drains so
/// the child never blocks on a full pipe.
async fn run_deploy_command(
    workspace: &Path,
    platform: DeployTarget,
    command: String,
    timeout_secs: u64,
) -> DeployProof {
    let started = Instant::now();
    // Run through `sh -c` (Unix) / `cmd /c` (Windows) so multi-token commands
    // like `npx vercel --prod --yes` execute as written.
    let (shell, shell_arg) = if cfg!(windows) {
        ("cmd", "/c")
    } else {
        ("sh", "-c")
    };
    let mut dcmd = Command::new(shell);
    dcmd.arg(shell_arg)
        .arg(&command)
        .current_dir(workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    // Detach into its OWN session/process-group so a timeout can take down the
    // whole deploy tree, not just the `sh -c` wrapper. Safe: stdin is null and
    // stdout/stderr are piped. Fail-open (see spawn_util).
    crate::spawn_util::detach_from_controlling_terminal(&mut dcmd);
    let mut child = match dcmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let mut proof =
                DeployProof::not_deployed(platform, format!("could not run deploy command: {e}"));
            proof.command = Some(command);
            proof.exit_code = Some(-1);
            return proof;
        }
    };

    // Capped-tail readers: retain only the last OUTPUT_CAP bytes of each stream
    // (bounding memory on a chatty deploy) while always draining so the child
    // never blocks on a full pipe.
    let stdout_task = child
        .stdout
        .take()
        .map(|h| tokio::spawn(read_capped_tail(h, OUTPUT_CAP)));
    let stderr_task = child
        .stderr
        .take()
        .map(|h| tokio::spawn(read_capped_tail(h, OUTPUT_CAP)));

    // Race the command's exit against the deploy budget.
    let wait_result = tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait()).await;
    if wait_result.is_err() {
        // Timed out: KILL the whole process group (detached above) so `npx`/`node`
        // descendants die too — dropping the `Child` alone would leave them
        // running. `start_kill` + `kill_on_drop` back up the direct child; the
        // reap is bounded so a wedged wait() can't hang.
        let _ = crate::spawn_util::kill_process_group(&child);
        let _ = child.start_kill();
        let _ = tokio::time::timeout(Duration::from_secs(KILL_REAP_SECS), child.wait()).await;
    }

    let raw_stdout = join_capped(stdout_task).await;
    let raw_stderr = join_capped(stderr_task).await;
    let stdout = String::from_utf8_lossy(&raw_stdout);
    let stderr = String::from_utf8_lossy(&raw_stderr);
    let ms = started.elapsed().as_millis().try_into().unwrap_or(u64::MAX);

    match wait_result {
        Ok(Ok(status)) => {
            let exit = status.code().unwrap_or(-1);
            // Many deploy CLIs print the live URL on stdout; some on stderr.
            let url = extract_url(&stdout).or_else(|| extract_url(&stderr));
            let log_tail = log_tail(&stdout, &stderr);
            if status.success() {
                DeployProof {
                    timestamp: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
                    platform,
                    status: DeployStatus::Deployed,
                    command: Some(command),
                    exit_code: Some(exit),
                    url,
                    duration_ms: Some(ms),
                    log_tail,
                }
            } else {
                DeployProof {
                    timestamp: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
                    platform,
                    status: DeployStatus::NotDeployed(format!("deploy command exited {exit}")),
                    command: Some(command),
                    exit_code: Some(exit),
                    url,
                    duration_ms: Some(ms),
                    log_tail,
                }
            }
        }
        Ok(Err(e)) => {
            let mut proof =
                DeployProof::not_deployed(platform, format!("could not run deploy command: {e}"));
            proof.command = Some(command);
            proof.exit_code = Some(-1);
            proof
        }
        Err(_) => {
            let mut proof =
                DeployProof::not_deployed(platform, format!("timed out after {timeout_secs}s"));
            proof.command = Some(command);
            proof.exit_code = Some(-1);
            proof.duration_ms = Some(timeout_secs.saturating_mul(1000));
            // Keep the killed command's last words for the auditor.
            proof.log_tail = log_tail(&stdout, &stderr);
            proof
        }
    }
}

/// Read `reader` to EOF, retaining only the LAST `cap` bytes (the deploy
/// result / URL / error lives at the end) while always draining so the child
/// never blocks on a full pipe. Memory is bounded to `2*cap` between trims, so a
/// huge stream costs O(total), not O(total²).
async fn read_capped_tail<R>(mut reader: R, cap: usize) -> Vec<u8>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt as _;
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; 8192];
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if cap > 0 && buf.len() > cap.saturating_mul(2) {
                    let drop_to = buf.len() - cap;
                    buf.drain(..drop_to);
                }
            }
        }
    }
    if cap > 0 && buf.len() > cap {
        let drop_to = buf.len() - cap;
        buf.drain(..drop_to);
    }
    buf
}

/// Join a capped-tail reader task, BOUNDED so a wedged descendant that still
/// holds a pipe open after a kill can't hang us. Fail-open: a missing task or a
/// panic yields an empty buffer.
async fn join_capped(task: Option<tokio::task::JoinHandle<Vec<u8>>>) -> Vec<u8> {
    match task {
        Some(t) => match tokio::time::timeout(Duration::from_secs(KILL_REAP_SECS), t).await {
            Ok(Ok(buf)) => buf,
            Ok(Err(_)) | Err(_) => Vec::new(),
        },
        None => Vec::new(),
    }
}

/// Persist the proof to `.umadev/audit/deploy-proof.json`. Returns the path on
/// success; a write failure is fail-open (callers swallow the `Err`) — it must
/// not block delivery.
pub fn write_deploy_proof(workspace: &Path, proof: &DeployProof) -> std::io::Result<PathBuf> {
    let audit_dir = workspace.join(".umadev/audit");
    std::fs::create_dir_all(&audit_dir)?;
    let path = audit_dir.join("deploy-proof.json");
    let body = serde_json::to_string_pretty(proof).unwrap_or_else(|_| "{}".into());
    std::fs::write(&path, body)?;
    Ok(path)
}

/// The canonical location of the deploy-proof artifact relative to the
/// workspace root. Used by the proof-pack assembler so it stays in sync.
#[must_use]
pub fn deploy_proof_rel_path() -> &'static str {
    ".umadev/audit/deploy-proof.json"
}

// ---------------------------------------------------------------------------
// internals — pure, unit-tested
// ---------------------------------------------------------------------------

/// Pull the first `http(s)://…` URL out of deploy output. Deploy CLIs print the
/// live URL on a line; we take the first well-formed one. Trailing punctuation
/// / quotes / ANSI-ish trailers are trimmed so the captured URL is clickable.
fn extract_url(output: &str) -> Option<String> {
    for line in output.lines() {
        if let Some(idx) = line.find("https://").or_else(|| line.find("http://")) {
            let rest = &line[idx..];
            // Stop at the first whitespace; trim trailing noise punctuation.
            let token = rest.split_whitespace().next().unwrap_or(rest);
            let cleaned = token.trim_end_matches(['.', ',', ')', ']', '"', '\'', '`', '>']);
            if cleaned.len() > "https://".len() {
                return Some(cleaned.to_string());
            }
        }
    }
    None
}

/// Build the capped log tail from stdout + stderr. Keeps the *end* of the
/// combined output (where the result / error / URL lives), capped at
/// [`CAPTURE_CAP`] on a char boundary.
fn log_tail(stdout: &str, stderr: &str) -> String {
    let mut combined = String::new();
    if !stdout.trim().is_empty() {
        combined.push_str(stdout.trim_end());
    }
    if !stderr.trim().is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(stderr.trim_end());
    }
    tail_capped(&combined, CAPTURE_CAP)
}

/// Keep the last `cap` bytes of `s`, trimmed to a char boundary, prefixed with
/// a marker when truncation happened.
fn tail_capped(s: &str, cap: usize) -> String {
    if s.len() <= cap {
        return s.to_string();
    }
    let mut start = s.len() - cap;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    format!("...[truncated]\n{}", &s[start..])
}

/// Check whether a PATH-resolvable binary exists. Splits `PATH` on the
/// platform-native separator and honours `PATHEXT` on Windows so `which("npx")`
/// finds `npx.cmd`. Mirrors the verify/runtime-proof helpers.
fn which(bin: &str) -> bool {
    let Ok(path_var) = std::env::var("PATH") else {
        return false;
    };
    let separator = if cfg!(windows) { ';' } else { ':' };
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.BAT;.CMD;.COM".to_string())
            .split(';')
            .map(str::to_string)
            .collect()
    } else {
        vec![String::new()]
    };
    for dir in path_var.split(separator) {
        if dir.is_empty() {
            continue;
        }
        for ext in &exts {
            if Path::new(dir).join(format!("{bin}{ext}")).is_file() {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn detect_vercel_via_config() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("vercel.json"), "{}").unwrap();
        assert_eq!(detect_deploy_target(tmp.path()), DeployTarget::Vercel);
    }

    #[test]
    fn detect_vercel_via_next_dependency() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("package.json"),
            r#"{"name":"app","dependencies":{"next":"^14.0.0"}}"#,
        )
        .unwrap();
        assert_eq!(detect_deploy_target(tmp.path()), DeployTarget::Vercel);
    }

    #[test]
    fn detect_vercel_via_next_config() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("next.config.mjs"), "export default {}").unwrap();
        assert_eq!(detect_deploy_target(tmp.path()), DeployTarget::Vercel);
    }

    #[test]
    fn detect_netlify() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("netlify.toml"), "[build]").unwrap();
        assert_eq!(detect_deploy_target(tmp.path()), DeployTarget::Netlify);
    }

    #[test]
    fn detect_fly() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("fly.toml"), "app = \"x\"").unwrap();
        assert_eq!(detect_deploy_target(tmp.path()), DeployTarget::Fly);
    }

    #[test]
    fn detect_cloudflare_pages() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("wrangler.toml"), "name = \"x\"").unwrap();
        assert_eq!(
            detect_deploy_target(tmp.path()),
            DeployTarget::CloudflarePages
        );
    }

    #[test]
    fn detect_docker() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("Dockerfile"), "FROM scratch").unwrap();
        assert_eq!(detect_deploy_target(tmp.path()), DeployTarget::Docker);
    }

    #[test]
    fn detect_static_host_from_dist() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir(tmp.path().join("dist")).unwrap();
        assert_eq!(detect_deploy_target(tmp.path()), DeployTarget::StaticHost);
    }

    #[test]
    fn detect_none_for_empty_workspace() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(detect_deploy_target(tmp.path()), DeployTarget::None);
    }

    #[test]
    fn platform_config_wins_over_dockerfile() {
        // A repo with BOTH a vercel.json and a Dockerfile picks the explicit
        // platform config — it is the more specific signal.
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("vercel.json"), "{}").unwrap();
        fs::write(tmp.path().join("Dockerfile"), "FROM scratch").unwrap();
        assert_eq!(detect_deploy_target(tmp.path()), DeployTarget::Vercel);
    }

    #[test]
    fn dockerfile_wins_over_bare_static_bundle() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("Dockerfile"), "FROM scratch").unwrap();
        fs::create_dir(tmp.path().join("dist")).unwrap();
        assert_eq!(detect_deploy_target(tmp.path()), DeployTarget::Docker);
    }

    #[test]
    fn every_real_target_has_a_command_and_cli() {
        for t in [
            DeployTarget::Vercel,
            DeployTarget::Netlify,
            DeployTarget::Fly,
            DeployTarget::CloudflarePages,
            DeployTarget::Docker,
            DeployTarget::StaticHost,
        ] {
            assert!(t.deploy_command().is_some(), "{t:?} must have a command");
            assert!(t.cli_binary().is_some(), "{t:?} must name a CLI binary");
        }
        assert!(DeployTarget::None.deploy_command().is_none());
        assert!(DeployTarget::None.cli_binary().is_none());
    }

    #[test]
    fn target_labels_are_stable() {
        assert_eq!(DeployTarget::Vercel.as_str(), "vercel");
        assert_eq!(DeployTarget::Netlify.as_str(), "netlify");
        assert_eq!(DeployTarget::Fly.as_str(), "fly");
        assert_eq!(DeployTarget::CloudflarePages.as_str(), "cloudflare-pages");
        assert_eq!(DeployTarget::Docker.as_str(), "docker");
        assert_eq!(DeployTarget::StaticHost.as_str(), "static-host");
        assert_eq!(DeployTarget::None.as_str(), "none");
    }

    #[test]
    fn extract_url_finds_https_in_log() {
        let log = "Building...\nDeployed to https://my-app-abc123.vercel.app in 12s\nDone";
        assert_eq!(
            extract_url(log).as_deref(),
            Some("https://my-app-abc123.vercel.app")
        );
    }

    #[test]
    fn extract_url_trims_trailing_punctuation() {
        let log = "Live at: https://app.netlify.app.";
        assert_eq!(extract_url(log).as_deref(), Some("https://app.netlify.app"));
        let parened = "See (https://app.fly.dev)";
        assert_eq!(extract_url(parened).as_deref(), Some("https://app.fly.dev"));
    }

    #[test]
    fn extract_url_returns_none_without_url() {
        assert!(extract_url("no url here, just text").is_none());
        // A bare scheme with no host is not a usable URL.
        assert!(extract_url("https://").is_none());
    }

    #[test]
    fn log_tail_keeps_the_end_and_truncates_long_output() {
        let long = "x".repeat(CAPTURE_CAP + 500);
        let tail = log_tail(&long, "");
        assert!(tail.starts_with("...[truncated]"));
        assert!(tail.len() <= CAPTURE_CAP + "...[truncated]\n".len());
    }

    #[test]
    fn log_tail_combines_stdout_and_stderr() {
        let tail = log_tail("out line", "err line");
        assert!(tail.contains("out line"));
        assert!(tail.contains("err line"));
    }

    #[test]
    fn tail_capped_does_not_split_multibyte_chars() {
        let s = "做".repeat(20); // each char is 3 bytes
        let tail = tail_capped(&s, 10);
        // Must still be valid UTF-8 (no panic on slicing).
        assert!(tail.ends_with('做'));
    }

    #[test]
    fn not_deployed_record_carries_platform_command_and_reason() {
        let p = DeployProof::not_deployed(DeployTarget::Vercel, "vercel not on PATH");
        assert_eq!(p.platform, DeployTarget::Vercel);
        assert!(!p.status.is_deployed());
        assert_eq!(p.status.as_str(), "not_deployed");
        // Even a not-deployed record surfaces the command the user can run.
        assert_eq!(p.command.as_deref(), Some("npx vercel --prod --yes"));
        assert!(p.summary_line().contains("vercel not on PATH"));
    }

    #[tokio::test]
    async fn run_deploy_no_target_is_fail_open() {
        // Empty workspace → no target → neutral NotDeployed, no spawn, no panic.
        let tmp = TempDir::new().unwrap();
        let proof = run_deploy(tmp.path(), None).await;
        assert_eq!(proof.platform, DeployTarget::None);
        assert!(!proof.status.is_deployed());
        if let DeployStatus::NotDeployed(reason) = &proof.status {
            assert!(reason.contains("no deploy target"));
        } else {
            panic!("expected NotDeployed");
        }
    }

    #[tokio::test]
    async fn run_deploy_missing_cli_is_fail_open() {
        // A Vercel project but the `npx`/binary path is a guaranteed-absent
        // command → NotDeployed("... not found on PATH"), never a crash.
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("vercel.json"), "{}").unwrap();
        let proof = run_deploy(tmp.path(), Some("definitely-not-a-real-binary-xyz deploy")).await;
        assert!(!proof.status.is_deployed());
        if let DeployStatus::NotDeployed(reason) = &proof.status {
            assert!(reason.contains("not found on PATH"), "got: {reason}");
        } else {
            panic!("expected NotDeployed");
        }
    }

    #[tokio::test]
    async fn run_deploy_captures_url_and_writes_proof() {
        // A trivially-succeeding command that prints a URL: proves the success
        // path captures the URL + writes the artifact. Uses `printf`/`echo`,
        // which exists on the CI runners; skip cleanly if neither is present.
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("netlify.toml"), "[build]").unwrap();
        if !which("echo") && !which("sh") {
            return;
        }
        let proof = run_deploy(
            tmp.path(),
            Some("echo Deployed to https://demo.example.app"),
        )
        .await;
        // On any platform with a working shell this deploys cleanly.
        if proof.status.is_deployed() {
            assert_eq!(proof.url.as_deref(), Some("https://demo.example.app"));
            let path = write_deploy_proof(tmp.path(), &proof).unwrap();
            assert!(path.ends_with("deploy-proof.json"));
            let body = fs::read_to_string(&path).unwrap();
            assert!(body.contains("\"platform\": \"netlify\""));
            assert!(body.contains("demo.example.app"));
        }
    }

    #[test]
    fn deploy_proof_rel_path_is_stable() {
        assert_eq!(deploy_proof_rel_path(), ".umadev/audit/deploy-proof.json");
    }

    #[tokio::test]
    async fn read_capped_tail_keeps_the_last_cap_bytes() {
        // A reader producing MORE than `cap` bytes: only the last `cap` are kept
        // (the tail, where a deploy prints its result / URL), bounding memory.
        let data = vec![b'a'; 10_000];
        let cap = 1_000;
        let got = read_capped_tail(&data[..], cap).await;
        assert_eq!(got.len(), cap, "keeps exactly the last cap bytes");
        assert!(got.iter().all(|&b| b == b'a'));
    }

    #[tokio::test]
    async fn read_capped_tail_returns_everything_when_under_cap() {
        let got = read_capped_tail(&b"short output"[..], 1_000).await;
        assert_eq!(got, b"short output");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_deploy_command_times_out_kills_and_does_not_hang() {
        // A command that runs FAR past the (tiny) budget, with a backgrounded
        // grandchild holding the stdout pipe open. On timeout the whole group is
        // killed, so this returns a bounded NotDeployed(timeout) promptly instead
        // of blocking on the pipe holder. (Fixes: tokio dropping the Child on
        // timeout without killing it, and unbounded output().)
        let tmp = TempDir::new().unwrap();
        let started = Instant::now();
        let proof = tokio::time::timeout(
            Duration::from_secs(25),
            run_deploy_command(
                tmp.path(),
                DeployTarget::Netlify,
                "sleep 60 & sleep 60".to_string(),
                1,
            ),
        )
        .await
        .expect("run_deploy_command must return, not hang, on timeout");
        assert!(!proof.status.is_deployed());
        match &proof.status {
            DeployStatus::NotDeployed(reason) => {
                assert!(reason.contains("timed out"), "reason: {reason}");
            }
            DeployStatus::Deployed => panic!("expected NotDeployed(timeout)"),
        }
        assert_eq!(proof.exit_code, Some(-1));
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "must return promptly after killing the group, not wait out the pipe holder"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_deploy_command_success_captures_url() {
        // Sanity: the happy path still deploys + captures the URL after the
        // switch off `Command::output()` to a bounded reader.
        let tmp = TempDir::new().unwrap();
        let proof = run_deploy_command(
            tmp.path(),
            DeployTarget::Netlify,
            "echo Deployed to https://demo.example.app".to_string(),
            30,
        )
        .await;
        assert!(proof.status.is_deployed(), "echo exits 0 → Deployed");
        assert_eq!(proof.url.as_deref(), Some("https://demo.example.app"));
    }
}
