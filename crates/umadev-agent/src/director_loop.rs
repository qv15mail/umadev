//! Real-time director orchestration loop — Wave 3 of
//! `docs/AGENT_WIELDS_BASE_ARCHITECTURE.md` §5 (+ §3).
//!
//! Wave 2 ([`crate::director`]) gave the director a clean Rust API for its four
//! team levers — [`crate::director::summon`] / [`crate::director::review`] /
//! [`crate::director::verify`] / [`crate::director::checkpoint`] — and described
//! them in the director's prompt. But on the default agentic `/run` path the base
//! ran its OWN internal tool loop inside a single `complete_streaming` call, so
//! UmaDev could never INTERCEPT a tool request mid-turn — the levers were a
//! capability the director could read about but not actually CALL.
//!
//! Wave 3 closes that gap. It moves `/run` onto a [`BaseSession`] **event pump**
//! (the same `while let Some(ev) = session.next_event()` shape the continuous
//! driver uses) so UmaDev sits between the director and its tools and can MEDIATE
//! each lever request: execute the corresponding [`crate::director`] function and
//! re-inject the factual result so the director keeps orchestrating — live,
//! on-the-spot, exactly as a real senior director would.
//!
//! ## How a lever request travels (the mediation contract)
//!
//! UmaDev does **not** vendor a host tool-registration SDK (the bases expose no
//! uniform custom-tool/MCP surface we may touch — see the workspace anti-rules and
//! the "don't touch the `*_session` internals" Wave 3 scope). So the director
//! requests a lever through a **structured marker** it emits in its OWN text — a
//! single line of the shape
//!
//! ```text
//! <<<umadev:summon role="frontend-engineer" mode="serial" instruction="build the login page">>>
//! <<<umadev:review kind="quality">>>
//! <<<umadev:verify kind="build-test">>>
//! <<<umadev:checkpoint question="ship to prod?">>>
//! ```
//!
//! The pump accumulates the director's text for the turn, and at
//! [`SessionEvent::TurnDone`] it [`parse_markers`]es every marker the turn
//! emitted. For each one it runs the matching [`crate::director`] function
//! ([`mediate_one`]), formats a compact factual result ([`format_*_result`]), and
//! — if any lever fired — re-injects ALL the results as the next turn's directive
//! ([`results_directive`]) so the director reads what its team produced and
//! decides the next move. A turn that emits NO marker is a normal end-of-build
//! turn → the loop settles.
//!
//! Native base tool-mechanism note: surfacing the four levers as FIRST-CLASS base
//! tools the base calls structurally (claude `can_use_tool` for a custom tool name
//! / an MCP tool) is the long-term ideal, but no host CLI exposes a uniform,
//! UmaDev-mediable custom-tool hook today without modifying the host sessions
//! (out of Wave 3 scope). The structured-marker channel is the portable
//! lowest-common-denominator that works identically on claude / codex / opencode,
//! so Wave 3 ships real-time scheduling through it; a future wave may upgrade a
//! base that grows a mediable native hook to channel ① with NO change to the
//! [`crate::director`] functions this loop calls.
//!
//! ## Floor preserved (every Wave 2 invariant still holds — see [`crate::director`])
//!
//! 1. **Single-writer.** Only a [`SummonMode::Serial`] summon mutates the
//!    workspace, on the MAIN session under the run-lock the caller already holds;
//!    parallel summons + every review run on isolated read-only forks. The pump
//!    runs the levers SERIALLY (one at a time, in marker order) so two serial
//!    doers can never write at once.
//! 2. **Objective floor untouched.** [`crate::director::verify`] is a deterministic
//!    reality check; review verdicts stay advisory. The source-present hard-gate
//!    ([`crate::acceptance::source_files`]) still runs at the boundary in the
//!    caller, unchanged.
//! 3. **Governance + audit.** A serial summon drives [`crate::continuous::drive_rework_turn`],
//!    which governs + audits every write exactly like a phase turn. The PreToolUse
//!    hook still fires under everything.
//! 4. **No new endpoint.** Every lever runs over the SAME borrowed brain.
//! 5. **Fail-open.** A malformed marker is ignored (the director just gets no
//!    result for it); a lever that errors re-injects an error line (the director
//!    handles it); a base that emits no markers degrades to a plain agentic turn
//!    (Wave 1 behaviour — the director works alone). The loop can NEVER wedge.
//! 6. **Reversible.** This loop is reached only on the DEFAULT `/run` path; the
//!    legacy fixed pipeline (`UMADEV_LEGACY_PIPELINE=1`) is untouched.
//!
//! The loop is **bounded** by [`MAX_DIRECTOR_ROUNDS`] so a director that keeps
//! emitting levers without ever finishing can't spin forever — after the cap the
//! loop ends and the caller's objective hard-gate has the final say on reality.

use std::sync::Arc;

use umadev_runtime::{ApprovalDecision, BaseSession, SessionEvent, StreamEvent, TurnStatus};

use crate::director::{
    self, CheckpointDecision, ReviewResult, SummonMode, VerifyKind, VerifyResult,
};
use crate::events::{EngineEvent, EventSink};
use crate::runner::RunOptions;
use crate::trust::requires_confirmation;

/// The hard ceiling on director↔team orchestration rounds in one `/run`. Each
/// round is: the director drives a turn, it may fire team levers, we mediate them
/// and re-inject the results, and the director continues. The director normally
/// finishes well within this; the cap only stops a pathological loop where the
/// director keeps requesting levers without ever calling the build done. After the
/// cap the loop ends gracefully (the caller's source-present hard-gate is the
/// objective backstop). Kept generous so a real multi-seat product build (PM →
/// architect → frontend → backend → QA, each maybe a rework round) has room.
const MAX_DIRECTOR_ROUNDS: usize = 24;

/// The marker prefix the director emits to request a team lever. Chosen to be
/// visually distinct and extremely unlikely to occur in ordinary prose / code, so
/// [`parse_markers`] never false-positives on the director's narration.
const MARKER_OPEN: &str = "<<<umadev:";
/// The marker close.
const MARKER_CLOSE: &str = ">>>";

/// How the director loop settled — mirrors the caller's existing director outcome
/// but lives in the agent crate so both the CLI and the TUI share ONE loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectorLoopOutcome {
    /// The director finished its build (its last turn emitted no team lever and
    /// ended cleanly / truncated-but-accepted). The caller then runs the objective
    /// source-present hard-gate to verify reality.
    Done {
        /// The director's final assistant text — the caller reads it for a
        /// "claimed a build" check against the real source files.
        reply: String,
    },
    /// The session died / a turn failed — an honest hard stop, never disguised as
    /// success. Carries a machine-true reason.
    Failed(String),
}

/// One parsed team-lever request the director emitted via a structured marker.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LeverRequest {
    /// Delegate a slice to a named seat (serial doer / parallel reviewer).
    Summon {
        role: String,
        instruction: String,
        mode: SummonMode,
    },
    /// Convene a cross-review team over the current blackboard.
    Review { kind: crate::continuous::ReviewKind },
    /// Run an objective reality check.
    Verify { kind: VerifyKind },
    /// Pause for the user when the decision is theirs (bounded by the trust tier).
    Checkpoint { question: String },
}

/// Drive an explicit `/run` (full product build) through the **real-time director
/// loop** — the Wave 3 engine. ONE live [`BaseSession`] is the director's brain;
/// the director plans + delegates LIVE by emitting team-lever markers that this
/// loop mediates (executing the matching [`crate::director`] function and
/// re-injecting the factual result), so the director truly orchestrates its team
/// on the spot rather than working alone.
///
/// `first_directive` is the goal framing the caller already built (e.g.
/// [`crate::experts::director_build_directive`]); the loop sends it as the opening
/// turn, then pumps the director's events. The caller owns the session lifetime
/// (and the run-lock) and `end()`s it after this returns.
///
/// Floor preserved (see the module docs): single-writer, governance + audit,
/// advisory review, objective verify, fail-open, no endpoint. The loop never
/// blocks the host — every failure mode degrades to a graceful settle.
pub async fn drive_director_loop(
    session: &mut dyn BaseSession,
    options: &RunOptions,
    events: &Arc<dyn EventSink>,
    first_directive: String,
) -> DirectorLoopOutcome {
    // The opening turn carries the goal framing AND the marker-syntax capability,
    // so the director knows HOW to call a team lever even on a path whose host
    // session set no system prompt (the CLI `drive_director_run` path). On the TUI
    // path the system prompt ALSO carries it (via `experts::director_with_team_tools`)
    // — a harmless restatement, and the single source of the wording is this module.
    let mut next_directive = format!("{first_directive}\n\n---\n{}", director_loop_capability());
    let mut last_reply = String::new();

    for round in 0..MAX_DIRECTOR_ROUNDS {
        // Drive ONE director turn, accumulating its text so we can parse the
        // team-lever markers it emitted once the turn ends.
        let turn = match drive_one_turn(session, options, events, next_directive).await {
            Ok(t) => t,
            Err(reason) => return DirectorLoopOutcome::Failed(reason),
        };
        last_reply = turn.text.clone();

        // Parse every team-lever marker this turn emitted. None → a normal
        // end-of-work turn: the director is done orchestrating, settle.
        let levers = parse_markers(&turn.text);
        if levers.is_empty() {
            return DirectorLoopOutcome::Done { reply: last_reply };
        }

        // Mediate each lever SERIALLY (in marker order) so two serial doers can
        // never write at once (single-writer). Collect the factual results.
        let mut results: Vec<String> = Vec::new();
        let mut paused_for_user = false;
        for lever in levers {
            let (line, pause) = mediate_one(session, options, events, lever).await;
            results.push(line);
            if pause {
                paused_for_user = true;
            }
        }

        // A checkpoint that genuinely paused for the user ends the loop here: the
        // caller surfaces the question and the user drives the next step (a
        // revise / continue re-enters with their answer). We still report Done so
        // the boundary hard-gate runs over whatever was built so far.
        if paused_for_user {
            return DirectorLoopOutcome::Done { reply: last_reply };
        }

        // Re-inject the team's results so the director reads what actually
        // happened and decides its next move — the heart of real-time scheduling.
        let _ = round; // (round is the loop bound; the directive itself is lean)
        next_directive = results_directive(&results);
    }

    // Hit the round cap: the director kept orchestrating without settling. End
    // gracefully — the caller's objective source-present hard-gate decides reality.
    events.emit(EngineEvent::Note(
        "team · director loop reached its round budget — settling (objective hard-gate decides reality)"
            .to_string(),
    ));
    DirectorLoopOutcome::Done { reply: last_reply }
}

/// One director turn's observable result.
struct TurnResult {
    /// The accumulated assistant text (markers included — `parse_markers` strips
    /// them out). The caller reads it for the "claimed a build" hard-gate.
    text: String,
}

/// Send one directive and pump the director's event stream to its `TurnDone`,
/// forwarding tool calls + text to the live sink (the SAME `WorkerStream` render
/// path the pipeline uses), answering approvals via the always-on irreversible
/// floor, and accumulating the assistant text. Returns the turn's text, or `Err`
/// with a machine-true reason on a failed / dead turn (fail-open: the caller maps
/// it to a hard stop, never a panic).
async fn drive_one_turn(
    session: &mut dyn BaseSession,
    options: &RunOptions,
    events: &Arc<dyn EventSink>,
    directive: String,
) -> Result<TurnResult, String> {
    if let Err(e) = session.send_turn(directive).await {
        return Err(format!("session send: {e}"));
    }
    let mut text = String::new();
    loop {
        let Some(ev) = session.next_event().await else {
            // `None` = the session ended (process dead / EOF). Per the BaseSession
            // contract, treat as a failed turn — fail-open, no panic.
            return Err("base session ended mid-turn".to_string());
        };
        match ev {
            SessionEvent::TextDelta(delta) => {
                text.push_str(&delta);
                events.emit(EngineEvent::WorkerStream {
                    event: StreamEvent::Text { delta },
                });
            }
            SessionEvent::ToolCall { name, input } => {
                // Surface what the base actually DID (the source of truth). The
                // governance hook governs the write itself in real time; here we
                // render the tool row for live progress.
                let detail = tool_call_target(&input);
                events.emit(EngineEvent::WorkerStream {
                    event: StreamEvent::ToolUse { name, detail },
                });
            }
            SessionEvent::ToolResult { ok, summary } => {
                events.emit(EngineEvent::WorkerStream {
                    event: StreamEvent::ToolResult { ok, summary },
                });
            }
            SessionEvent::NeedApproval {
                req_id,
                action,
                target,
            } => {
                // Always-on irreversible floor: deny an irreversible action even
                // headless (the same floor the `auto` tier can't skip), allow the
                // rest so a headless build isn't wedged waiting on a human.
                let decision = if requires_confirmation(options.mode, &action, &target) {
                    events.emit(EngineEvent::Note(umadev_i18n::tlf(
                        "continuous.dangerous_action_denied",
                        &[&action, &target],
                    )));
                    ApprovalDecision::Deny
                } else {
                    ApprovalDecision::Allow
                };
                if let Err(e) = session.respond(&req_id, decision).await {
                    return Err(format!("session respond: {e}"));
                }
            }
            SessionEvent::TurnDone { status } => match status {
                // Completed / Truncated → accept the turn (the deterministic floor
                // downstream is the real stop signal; forcing a fail here would
                // hard-stop a build that may have produced usable output).
                TurnStatus::Completed | TurnStatus::Truncated => {
                    return Ok(TurnResult { text });
                }
                TurnStatus::Interrupted => return Err("director turn interrupted".to_string()),
                TurnStatus::Failed(reason) => return Err(reason),
            },
        }
    }
}

/// Best-effort human-readable target of a base tool call (a file path / command)
/// for the live tool row — fail-open to an empty string on any unexpected shape.
fn tool_call_target(input: &serde_json::Value) -> String {
    for key in ["file_path", "path", "command", "url", "pattern"] {
        if let Some(s) = input.get(key).and_then(serde_json::Value::as_str) {
            return s.to_string();
        }
    }
    String::new()
}

/// Execute ONE mediated team lever and return `(result_line, paused_for_user)`.
/// The result line is a compact, factual summary the director reads next round;
/// `paused_for_user` is `true` only for a checkpoint that genuinely paused (the
/// caller then surfaces the question and ends the loop). Every branch is fail-open:
/// a lever that degrades returns a truthful "couldn't / not done" line, never an
/// error that wedges the loop.
async fn mediate_one(
    session: &mut dyn BaseSession,
    options: &RunOptions,
    events: &Arc<dyn EventSink>,
    lever: LeverRequest,
) -> (String, bool) {
    match lever {
        LeverRequest::Summon {
            role,
            instruction,
            mode,
        } => {
            let r = director::summon(session, options, events, &role, &instruction, mode).await;
            (format_summon_result(&r), false)
        }
        LeverRequest::Review { kind } => {
            let r = director::review(session, options, events, kind).await;
            (format_review_result(kind, &r), false)
        }
        LeverRequest::Verify { kind } => {
            let r = director::verify(options, events, kind).await;
            (format_verify_result(kind, &r), false)
        }
        LeverRequest::Checkpoint { question } => {
            let decision = director::checkpoint(options, events, &question);
            match decision {
                CheckpointDecision::AutoProceed => (
                    format!("checkpoint(\"{question}\"): auto-approved (full-autonomy tier) — proceed."),
                    false,
                ),
                CheckpointDecision::AskUser => (
                    format!(
                        "checkpoint(\"{question}\"): PAUSED for the user — the build stops here until \
                         they answer; do not proceed past this decision on your own."
                    ),
                    true,
                ),
            }
        }
    }
}

/// Format a [`crate::director::summon`] result as a compact factual line.
fn format_summon_result(r: &director::SummonResult) -> String {
    match &r.verdict {
        // A parallel (reviewing) seat returned an opinion.
        Some(v) => {
            let mut line = format!(
                "summon {}({}): {} — accepts={}",
                r.role,
                SummonMode::Parallel.as_str(),
                if r.done {
                    "reviewed"
                } else {
                    "could not review (fail-open accept)"
                },
                v.accepts
            );
            if !v.blocking.is_empty() {
                line.push_str(&format!("; must-fix: {}", v.blocking.join("; ")));
            }
            line
        }
        // A serial doer mutated the workspace.
        None => format!(
            "summon {}({}): {}",
            r.role,
            SummonMode::Serial.as_str(),
            if r.done {
                "did the work (turn completed — verify the files landed)"
            } else {
                "turn did not complete (degraded — decide whether to retry)"
            }
        ),
    }
}

/// Format a [`crate::director::review`] result as a compact factual line.
fn format_review_result(kind: crate::continuous::ReviewKind, r: &ReviewResult) -> String {
    let kind_id = review_kind_id(kind);
    if r.seats == 0 {
        return format!("review {kind_id}: no team convened for this task (lean / nothing to review) — proceed.");
    }
    if r.blocking.is_empty() {
        return format!(
            "review {kind_id}: {} seat(s) reviewed, all accept — no must-fix.",
            r.seats
        );
    }
    format!(
        "review {kind_id}: {} seat(s) reviewed, {} must-fix finding(s): {}",
        r.seats,
        r.blocking.len(),
        r.blocking.join("; ")
    )
}

/// Format a [`crate::director::verify`] result as a compact factual line.
fn format_verify_result(kind: VerifyKind, r: &VerifyResult) -> String {
    let kind_id = kind.as_str();
    if !r.available {
        let why = r.evidence.first().cloned().unwrap_or_default();
        return format!(
            "verify {kind_id}: skipped (nothing to check){}.",
            fmt_suffix(&why)
        );
    }
    let head = if r.passed {
        format!("verify {kind_id}: PASSED")
    } else {
        format!("verify {kind_id}: FAILED")
    };
    if r.evidence.is_empty() {
        format!("{head}.")
    } else {
        format!("{head} — {}", r.evidence.join("; "))
    }
}

/// `" — <why>"` when `why` is non-empty, else empty — keeps the verify line tidy.
fn fmt_suffix(why: &str) -> String {
    if why.trim().is_empty() {
        String::new()
    } else {
        format!(" — {}", why.trim())
    }
}

/// Stable lowercase id for a review node kind (for the result line the director
/// reads). Mirrors `continuous::kind_phase_label` without importing a private fn.
fn review_kind_id(kind: crate::continuous::ReviewKind) -> &'static str {
    use crate::continuous::ReviewKind::{Docs, Preview, Quality};
    match kind {
        Docs => "docs",
        Preview => "preview",
        Quality => "quality",
    }
}

/// Build the next directive: the factual results of every lever the director just
/// fired, framed as "here's what your team produced — decide the next move." Lean
/// (no role priming — the director already holds the full build context in this
/// one continuous session), command-style so the director keeps acting rather than
/// narrating.
fn results_directive(results: &[String]) -> String {
    let mut body = String::new();
    for r in results {
        body.push_str("- ");
        body.push_str(r);
        body.push('\n');
    }
    format!(
        "Your team reported back. Here is what actually happened (objective results, \
         not your memory):\n{body}\nNow decide the next move as the director: if a \
         reviewer raised must-fix issues, send the relevant seat back to fix them \
         (summon again); if a verify failed, fix the cause; if everything checks out \
         and the goal is met, finish and report honestly what was built. You may fire \
         more team levers, or do work yourself, or call it done — your call. When the \
         build is genuinely complete and verified, end your turn WITHOUT emitting any \
         further marker."
    )
}

// ───────────────────────────────────────────────────────────────────────────
// Marker parsing — the director→UmaDev request channel
// ───────────────────────────────────────────────────────────────────────────

/// Parse every team-lever marker the director emitted in `text`. A marker is a
/// `<<<umadev:<verb> key="value" …>>>` line; unknown verbs / malformed markers are
/// skipped (fail-open — the director simply gets no result for a garbled request).
/// Returns the levers in emission order so the loop mediates them deterministically.
fn parse_markers(text: &str) -> Vec<LeverRequest> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find(MARKER_OPEN) {
        let after_open = &rest[open + MARKER_OPEN.len()..];
        let Some(close_rel) = after_open.find(MARKER_CLOSE) else {
            break; // an unterminated marker → stop (fail-open, ignore the tail)
        };
        let body = &after_open[..close_rel];
        if let Some(lever) = parse_one_marker(body) {
            out.push(lever);
        }
        // Advance past this marker's close.
        rest = &after_open[close_rel + MARKER_CLOSE.len()..];
    }
    out
}

/// Parse ONE marker body (`<verb> key="value" …`) into a [`LeverRequest`].
/// Returns `None` for an unknown verb or a request that lacks its required field,
/// so a malformed marker fail-opens to "no lever" rather than a bogus action.
fn parse_one_marker(body: &str) -> Option<LeverRequest> {
    let body = body.trim();
    // The verb is the first whitespace-delimited token.
    let (verb, attrs_str) = match body.split_once(char::is_whitespace) {
        Some((v, a)) => (v, a),
        None => (body, ""),
    };
    let attrs = parse_attrs(attrs_str);
    let get = |k: &str| {
        attrs
            .iter()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v.clone())
    };

    match verb.trim().to_ascii_lowercase().as_str() {
        "summon" => {
            let role = get("role")?;
            if role.trim().is_empty() {
                return None;
            }
            let instruction = get("instruction").unwrap_or_default();
            let mode = match get("mode").as_deref().map(str::trim) {
                Some("parallel" | "review") => SummonMode::Parallel,
                // Default + "serial" / anything else → a serial doer (the common
                // case: delegate work). Fail-open to the safe single-writer mode.
                _ => SummonMode::Serial,
            };
            Some(LeverRequest::Summon {
                role,
                instruction,
                mode,
            })
        }
        "review" => {
            let kind = parse_review_kind(get("kind").as_deref());
            Some(LeverRequest::Review { kind })
        }
        "verify" => {
            let kind = parse_verify_kind(get("kind").as_deref())?;
            Some(LeverRequest::Verify { kind })
        }
        "checkpoint" => {
            let question = get("question").or_else(|| get("q")).unwrap_or_default();
            Some(LeverRequest::Checkpoint { question })
        }
        _ => None, // unknown verb → fail-open ignore
    }
}

/// Map a `kind="…"` to a [`crate::continuous::ReviewKind`]; defaults to Quality
/// (the most common director review — vetting delivered code) on absence/unknown.
fn parse_review_kind(kind: Option<&str>) -> crate::continuous::ReviewKind {
    use crate::continuous::ReviewKind::{Docs, Preview, Quality};
    match kind.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("docs") => Docs,
        Some("preview" | "frontend") => Preview,
        // "quality" / unknown / absent → Quality.
        _ => Quality,
    }
}

/// Map a `kind="…"` to a [`VerifyKind`]; `None` for an unknown kind so the verify
/// marker is skipped rather than running an arbitrary check.
fn parse_verify_kind(kind: Option<&str>) -> Option<VerifyKind> {
    match kind.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("build-test" | "build" | "test" | "build_test") => Some(VerifyKind::BuildTest),
        Some("contract" | "coverage") => Some(VerifyKind::Contract),
        Some("source-present" | "source" | "source_present") => Some(VerifyKind::SourcePresent),
        // Absent kind → default to the cheapest real reality check (source-present)
        // so a bare `<<<umadev:verify>>>` still does something useful.
        None | Some("") => Some(VerifyKind::SourcePresent),
        _ => None,
    }
}

/// Parse `key="value"` attribute pairs from a marker's attribute string. Tolerant:
/// values are double-quoted; an unquoted bareword value is also accepted; a `>`
/// inside a quoted value is fine (we only split on the marker close upstream).
/// Returns pairs in source order. Pure + total — never panics.
fn parse_attrs(s: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Skip whitespace.
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        // Read the key up to '=' or whitespace.
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'=' && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let key = s[key_start..i].trim().to_string();
        // Skip whitespace before '='.
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'=' {
            i += 1; // consume '='
                    // Skip whitespace after '='.
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            let value = if i < bytes.len() && (bytes[i] == b'"' || bytes[i] == b'\'') {
                let quote = bytes[i];
                i += 1; // consume opening quote
                let val_start = i;
                while i < bytes.len() && bytes[i] != quote {
                    i += 1;
                }
                let v = s[val_start..i].to_string();
                if i < bytes.len() {
                    i += 1; // consume closing quote
                }
                v
            } else {
                // Bareword value up to whitespace.
                let val_start = i;
                while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                s[val_start..i].to_string()
            };
            if !key.is_empty() {
                out.push((key, value));
            }
        } else if !key.is_empty() {
            // A bare flag with no value — record it with an empty value.
            out.push((key, String::new()));
        }
    }
    out
}

/// The director-loop capability block — the Wave 3 prompt surface. Tells the
/// director EXACTLY how to actually CALL a team lever now (the marker syntax), on
/// top of the Wave 2 capability description (which says WHAT the levers are). This
/// is what turns the Wave 2 "you have these abilities" into Wave 3 "and here is how
/// you invoke one, live." Framed as the director's own judgement — a small goal may
/// need no marker at all (just do it), a real product several.
#[must_use]
pub fn director_loop_capability() -> &'static str {
    "HOW TO ACTUALLY CALL A TEAM LEVER (real-time — you invoke it, the shell runs \
     it and reports the result back to you next turn):\n\
     Emit a single marker line of EXACTLY this form, then end your turn so the \
     result comes back:\n\
     - Delegate work to a seat (it writes the files, serially): \
     <<<umadev:summon role=\"frontend-engineer\" mode=\"serial\" instruction=\"build the login page\">>>\n\
     - Get a second opinion from a seat (read-only, returns a verdict): \
     <<<umadev:summon role=\"security-engineer\" mode=\"parallel\" instruction=\"audit the auth surface\">>>\n\
     - Convene the cross-review team: <<<umadev:review kind=\"quality\">>> (kinds: docs / preview / quality)\n\
     - Objectively check reality: <<<umadev:verify kind=\"build-test\">>> (kinds: build-test / contract / source-present)\n\
     - Pause for the user when the call is theirs: <<<umadev:checkpoint question=\"ship to prod?\">>>\n\
     Roles: product-manager / architect / uiux-designer / frontend-engineer / \
     backend-engineer / qa-engineer / security-engineer / devops-engineer. After you \
     emit one or more markers, END YOUR TURN; the results come back and you decide the \
     next move. When the build is genuinely done and verified, end your turn with NO \
     marker. Use the levers like a real director — on YOUR judgement, proportionate; a \
     trivial change may need none (just do it yourself)."
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::RecordingSink;
    use crate::trust::TrustMode;
    use umadev_runtime::{SessionError, SessionEvent, TurnStatus};

    fn opts(root: &std::path::Path) -> RunOptions {
        RunOptions {
            project_root: root.to_path_buf(),
            requirement: "做一个登录系统".to_string(),
            slug: "demo".to_string(),
            model: String::new(),
            backend: String::new(),
            design_system: String::new(),
            seed_template: String::new(),
            mode: TrustMode::Auto,
            strict_coverage: false,
        }
    }

    fn sink() -> (Arc<dyn EventSink>, RecordingSink) {
        let rec = RecordingSink::default();
        (Arc::new(rec.clone()), rec)
    }

    // ── A scripted fake BaseSession: each `send_turn` loads the next scripted
    // batch of events (a turn). Forks emit a fixed JSON verdict so a parallel
    // summon / review gets a verdict. `next_event` drains the current batch. ──
    #[derive(Clone)]
    struct FakeSession {
        /// One event-batch per upcoming MAIN turn, consumed front-to-back.
        turns: std::collections::VecDeque<Vec<SessionEvent>>,
        /// The currently-draining batch.
        current: std::collections::VecDeque<SessionEvent>,
        /// Directives the MAIN session received, in order (asserted by tests).
        sent: Arc<std::sync::Mutex<Vec<String>>>,
        /// Whether `fork()` succeeds.
        can_fork: bool,
        /// JSON a forked judge turn emits.
        fork_reply: String,
        /// `true` once this is a forked (read-only) session.
        is_fork: bool,
    }

    impl FakeSession {
        fn new(turns: Vec<Vec<SessionEvent>>, can_fork: bool, fork_reply: &str) -> Self {
            Self {
                turns: turns.into_iter().collect(),
                current: std::collections::VecDeque::new(),
                sent: Arc::new(std::sync::Mutex::new(Vec::new())),
                can_fork,
                fork_reply: fork_reply.to_string(),
                is_fork: false,
            }
        }
        fn sent_handle(&self) -> Arc<std::sync::Mutex<Vec<String>>> {
            Arc::clone(&self.sent)
        }
    }

    #[async_trait::async_trait]
    impl BaseSession for FakeSession {
        async fn fork(&mut self) -> Result<Box<dyn BaseSession>, SessionError> {
            if !self.can_fork {
                return Err(SessionError::ForkUnsupported("test".into()));
            }
            let mut f = self.clone();
            f.is_fork = true;
            f.current.clear();
            f.turns.clear();
            Ok(Box::new(f))
        }
        async fn send_turn(&mut self, directive: String) -> Result<(), SessionError> {
            if self.is_fork {
                // A forked judge turn emits its JSON verdict then ends.
                self.current = [
                    SessionEvent::TextDelta(self.fork_reply.clone()),
                    SessionEvent::TurnDone {
                        status: TurnStatus::Completed,
                    },
                ]
                .into_iter()
                .collect();
                return Ok(());
            }
            self.sent.lock().unwrap().push(directive);
            self.current = self
                .turns
                .pop_front()
                .unwrap_or_else(|| {
                    vec![SessionEvent::TurnDone {
                        status: TurnStatus::Completed,
                    }]
                })
                .into_iter()
                .collect();
            Ok(())
        }
        async fn next_event(&mut self) -> Option<SessionEvent> {
            self.current.pop_front()
        }
        async fn respond(
            &mut self,
            _req_id: &str,
            _decision: ApprovalDecision,
        ) -> Result<(), SessionError> {
            Ok(())
        }
        async fn interrupt(&mut self) -> Result<(), SessionError> {
            Ok(())
        }
        async fn end(&mut self) -> Result<(), SessionError> {
            Ok(())
        }
    }

    fn text_turn(s: &str) -> Vec<SessionEvent> {
        vec![
            SessionEvent::TextDelta(s.to_string()),
            SessionEvent::TurnDone {
                status: TurnStatus::Completed,
            },
        ]
    }

    // ── Marker parsing ────────────────────────────────────────────────────

    #[test]
    fn parses_a_summon_marker_with_attrs() {
        let levers = parse_markers(
            "I'll delegate this.\n<<<umadev:summon role=\"frontend-engineer\" mode=\"serial\" instruction=\"build the login form\">>>\nDone for now.",
        );
        assert_eq!(levers.len(), 1);
        assert_eq!(
            levers[0],
            LeverRequest::Summon {
                role: "frontend-engineer".to_string(),
                instruction: "build the login form".to_string(),
                mode: SummonMode::Serial,
            }
        );
    }

    #[test]
    fn parses_multiple_markers_in_order() {
        let levers = parse_markers(
            "<<<umadev:summon role=\"architect\" instruction=\"design it\">>> then \
             <<<umadev:review kind=\"quality\">>> and \
             <<<umadev:verify kind=\"build-test\">>>",
        );
        assert_eq!(levers.len(), 3);
        assert!(matches!(levers[0], LeverRequest::Summon { .. }));
        assert!(matches!(
            levers[1],
            LeverRequest::Review {
                kind: crate::continuous::ReviewKind::Quality
            }
        ));
        assert!(matches!(
            levers[2],
            LeverRequest::Verify {
                kind: VerifyKind::BuildTest
            }
        ));
    }

    #[test]
    fn summon_defaults_to_serial_and_parallel_is_recognised() {
        let s = parse_markers("<<<umadev:summon role=\"qa\">>>");
        assert_eq!(
            s[0],
            LeverRequest::Summon {
                role: "qa".to_string(),
                instruction: String::new(),
                mode: SummonMode::Serial,
            }
        );
        let p = parse_markers(
            "<<<umadev:summon role=\"qa\" mode=\"parallel\" instruction=\"review\">>>",
        );
        assert!(matches!(
            p[0],
            LeverRequest::Summon {
                mode: SummonMode::Parallel,
                ..
            }
        ));
    }

    #[test]
    fn unknown_verb_and_malformed_markers_are_skipped_fail_open() {
        // Unknown verb → skipped.
        assert!(parse_markers("<<<umadev:teleport role=\"x\">>>").is_empty());
        // Summon with no role → skipped (required field).
        assert!(parse_markers("<<<umadev:summon instruction=\"x\">>>").is_empty());
        // Verify with an unknown kind → skipped.
        assert!(parse_markers("<<<umadev:verify kind=\"vibes\">>>").is_empty());
        // Unterminated marker → ignored, no panic.
        assert!(parse_markers("<<<umadev:summon role=\"x\"").is_empty());
        // Plain prose with no marker → no levers.
        assert!(parse_markers("Just building the app, no markers here.").is_empty());
    }

    #[test]
    fn verify_and_review_kinds_default_sensibly() {
        // Bare verify → source-present (cheapest real check).
        assert_eq!(
            parse_markers("<<<umadev:verify>>>")[0],
            LeverRequest::Verify {
                kind: VerifyKind::SourcePresent
            }
        );
        // Bare review → quality.
        assert_eq!(
            parse_markers("<<<umadev:review>>>")[0],
            LeverRequest::Review {
                kind: crate::continuous::ReviewKind::Quality
            }
        );
    }

    #[test]
    fn checkpoint_marker_carries_the_question() {
        let l = parse_markers("<<<umadev:checkpoint question=\"deploy to prod?\">>>");
        assert_eq!(
            l[0],
            LeverRequest::Checkpoint {
                question: "deploy to prod?".to_string()
            }
        );
    }

    // ── The real-time loop ────────────────────────────────────────────────

    #[tokio::test]
    async fn director_summons_a_serial_doer_in_real_time() {
        // Turn 1: the director emits a serial summon. The loop mediates it (driving
        // the doer turn), then re-injects the result. Turn 2 (the doer's): no
        // marker → completes. Turn 3 (the director reads results): no marker → done.
        let tmp = tempfile::TempDir::new().unwrap();
        let (events, _rec) = sink();
        let turns = vec![
            // director plan turn → fires a serial summon
            text_turn(
                "Planning. <<<umadev:summon role=\"frontend-engineer\" instruction=\"build it\">>>",
            ),
            // the summoned doer's turn (driven by director::summon) → completes
            text_turn("built the login form"),
            // director reads the result → no further marker → done
            text_turn("All set — the login form is built and verified."),
        ];
        let mut sess = FakeSession::new(turns, false, "");
        let sent = sess.sent_handle();
        let o = opts(tmp.path());

        let outcome = drive_director_loop(&mut sess, &o, &events, "GO".to_string()).await;
        match outcome {
            DirectorLoopOutcome::Done { reply } => {
                assert!(reply.contains("All set"), "final reply carried");
            }
            other @ DirectorLoopOutcome::Failed(_) => panic!("expected Done, got {other:?}"),
        }
        let sent = sent.lock().unwrap();
        // The opening directive, then the doer directive (carries the role +
        // instruction), then the results re-injection.
        assert!(sent[0].contains("GO"), "opening directive sent");
        assert!(
            sent.iter()
                .any(|d| d.contains("frontend-engineer") && d.contains("build it")),
            "the serial doer was driven with the role + instruction: {sent:?}"
        );
        assert!(
            sent.iter().any(|d| d.contains("Your team reported back")),
            "the result was re-injected so the director could read it: {sent:?}"
        );
    }

    #[tokio::test]
    async fn director_runs_a_review_and_gets_blocking_back() {
        // The director convenes a quality review; a seat raises a blocking finding;
        // the loop re-injects it as a factual must-fix line the director reads.
        let tmp = tempfile::TempDir::new().unwrap();
        let (events, _rec) = sink();
        let reply = r#"{"accepts": false, "blocking": ["登录失败路径无测试"]}"#;
        let turns = vec![
            text_turn("<<<umadev:review kind=\"quality\">>>"),
            text_turn("Acknowledged — I'll address the gap."),
        ];
        let mut sess = FakeSession::new(turns, true, reply);
        let sent = sess.sent_handle();
        let o = opts(tmp.path());

        let outcome = drive_director_loop(&mut sess, &o, &events, "GO".to_string()).await;
        assert!(matches!(outcome, DirectorLoopOutcome::Done { .. }));
        let sent = sent.lock().unwrap();
        assert!(
            sent.iter()
                .any(|d| d.contains("登录失败路径无测试") && d.contains("must-fix")),
            "the blocking finding was re-injected as a must-fix line: {sent:?}"
        );
    }

    #[tokio::test]
    async fn director_verifies_reality_in_real_time() {
        // The director fires a source-present verify; with no source on disk it
        // comes back FAILED — a factual reality line the director reads.
        let tmp = tempfile::TempDir::new().unwrap();
        let (events, _rec) = sink();
        let turns = vec![
            text_turn("<<<umadev:verify kind=\"source-present\">>>"),
            text_turn("Right, nothing built yet — I'll implement it."),
        ];
        let mut sess = FakeSession::new(turns, false, "");
        let sent = sess.sent_handle();
        let o = opts(tmp.path());

        let _ = drive_director_loop(&mut sess, &o, &events, "GO".to_string()).await;
        let sent = sent.lock().unwrap();
        assert!(
            sent.iter()
                .any(|d| d.contains("verify source-present") && d.contains("FAILED")),
            "the verify reality result was re-injected: {sent:?}"
        );
    }

    #[tokio::test]
    async fn checkpoint_pauses_for_the_user_and_ends_the_loop() {
        // In a guarded tier, a checkpoint genuinely pauses: the loop ends (Done)
        // so the caller surfaces the question, and does NOT keep driving turns.
        let tmp = tempfile::TempDir::new().unwrap();
        let (events, rec) = sink();
        let mut o = opts(tmp.path());
        o.mode = TrustMode::Guarded;
        let turns = vec![
            text_turn("<<<umadev:checkpoint question=\"deploy to prod?\">>>"),
            // This turn should NOT run (the loop paused). If it did, the test for
            // the directive count below would see an extra send.
            text_turn("should not reach here"),
        ];
        let mut sess = FakeSession::new(turns, false, "");
        let sent = sess.sent_handle();

        let outcome = drive_director_loop(&mut sess, &o, &events, "GO".to_string()).await;
        assert!(matches!(outcome, DirectorLoopOutcome::Done { .. }));
        // Exactly ONE main directive was sent (the opening turn); the loop paused
        // for the user instead of driving the next turn.
        assert_eq!(
            sent.lock().unwrap().len(),
            1,
            "the loop paused, no further turn"
        );
        let evs = rec.events();
        assert!(
            evs.iter()
                .any(|e| matches!(e, EngineEvent::Note(n) if n.contains("checkpoint"))),
            "the checkpoint surfaced a note: {evs:?}"
        );
    }

    #[tokio::test]
    async fn no_marker_first_turn_is_a_plain_agentic_build_fail_open() {
        // A director (or a base that doesn't understand markers) that just builds
        // and ends with no marker → the loop settles after one turn (Wave 1
        // behaviour: the director worked alone). This is the fail-open degrade.
        let tmp = tempfile::TempDir::new().unwrap();
        let (events, _rec) = sink();
        let turns = vec![text_turn("I built the whole thing directly. Done.")];
        let mut sess = FakeSession::new(turns, false, "");
        let sent = sess.sent_handle();
        let o = opts(tmp.path());

        let outcome = drive_director_loop(&mut sess, &o, &events, "GO".to_string()).await;
        match outcome {
            DirectorLoopOutcome::Done { reply } => assert!(reply.contains("Done")),
            other @ DirectorLoopOutcome::Failed(_) => panic!("expected Done, got {other:?}"),
        }
        assert_eq!(
            sent.lock().unwrap().len(),
            1,
            "no marker → one turn, no re-injection (pure agentic)"
        );
    }

    #[tokio::test]
    async fn dead_session_is_a_failed_outcome_not_a_panic() {
        // A session that ends mid-turn (next_event → None with no TurnDone) is an
        // honest Failed outcome — fail-open, never a panic.
        let tmp = tempfile::TempDir::new().unwrap();
        let (events, _rec) = sink();
        // A turn whose batch has a text delta but NO TurnDone → next_event drains
        // to None mid-turn.
        let turns = vec![vec![SessionEvent::TextDelta("partial".to_string())]];
        let mut sess = FakeSession::new(turns, false, "");
        let o = opts(tmp.path());

        let outcome = drive_director_loop(&mut sess, &o, &events, "GO".to_string()).await;
        assert!(
            matches!(outcome, DirectorLoopOutcome::Failed(_)),
            "a dead session is a Failed outcome: {outcome:?}"
        );
    }

    #[test]
    fn capability_block_teaches_the_marker_syntax() {
        let c = director_loop_capability();
        assert!(c.contains("<<<umadev:summon"));
        assert!(c.contains("<<<umadev:review"));
        assert!(c.contains("<<<umadev:verify"));
        assert!(c.contains("<<<umadev:checkpoint"));
        // Frames it as the director's own judgement, not a forced chain.
        let lower = c.to_lowercase();
        assert!(lower.contains("your judgement") || lower.contains("real director"));
    }

    #[test]
    fn parse_attrs_handles_quotes_whitespace_and_barewords() {
        let a = parse_attrs("role=\"frontend\"  mode=serial  instruction=\"build it now\"");
        assert_eq!(a.len(), 3);
        assert_eq!(a[0], ("role".to_string(), "frontend".to_string()));
        assert_eq!(a[1], ("mode".to_string(), "serial".to_string()));
        assert_eq!(
            a[2],
            ("instruction".to_string(), "build it now".to_string())
        );
    }
}
