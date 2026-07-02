//! Own fd 0 and turn raw bytes into [`crossterm::event::Event`]s (UX maturity
//! roadmap §2, P1) — the runtime half of the byte-tokenizer root fix.
//!
//! A blocking thread reads raw bytes from stdin into a [`tokio::sync::mpsc`]
//! channel (honouring `#![forbid(unsafe_code)]` — `std::io::stdin().lock()` on a
//! thread, never a `libc::read`). [`OwnedInput`] feeds those bytes through the
//! [`Tokenizer`] + [`Decoder`] and exposes a single async [`OwnedInput::next`]
//! that yields one `Event` at a time, so the existing event loop's
//! key/mouse/paste/resize arm consumes it unchanged.
//!
//! Three things the owned source arms internally:
//! - **bytes** — `mpsc` chunk → `tokenizer.feed` → `decoder` → event queue;
//! - **a 50 ms ESC-flush timer** — armed only while the tokenizer holds a
//!   buffered incomplete escape (a lone `\x1b`); on fire, if more bytes are
//!   already queued the continuation is ingested instead (the re-arm trick), and
//!   only a genuinely idle FD flushes the buffered `\x1b` as a real Esc;
//! - **SIGWINCH** — owning fd 0 means crossterm's `Event::Resize` (which it
//!   derived from SIGWINCH) is gone, so we install our own safe `tokio::signal`
//!   handler and synthesize `Event::Resize` from [`crossterm::terminal::size`].
//!
//! The legacy `crossterm::EventStream` path is retained behind
//! [`legacy_input_from_env`] (`UMADEV_LEGACY_INPUT=1`) so a tokenizer bug in the
//! field is one env var away from reverting. It is also the default on Windows:
//! console keys such as Esc and arrows can arrive as Windows input records rather
//! than stdin bytes, and crossterm's native backend is the correct reader there.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crossterm::event::{Event, EventStream};
use futures::StreamExt;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use super::decode::{Decoder, InputEvent};
use super::tokenize::Tokenizer;

/// Default lone-ESC flush timeout. Deferred-verdict window: a real Esc resolves
/// within this long; a split arrow's continuation arrives far sooner (within the
/// same input burst), so it completes the sequence before the timer fires.
const DEFAULT_ESC_FLUSH_MS: u64 = 50;

/// Read-chunk size for the stdin reader thread. One `read()` returns whatever is
/// available up to this; a large paste arrives in a few chunks, not byte-by-byte.
const READ_CHUNK: usize = 4096;

/// Whether to use the legacy `crossterm::EventStream` input path instead of the
/// owned byte-tokenizer (`UMADEV_LEGACY_INPUT=1`). The de-risk escape hatch: the
/// owned tokenizer is the DEFAULT, but a field bug reverts with one env var.
#[must_use]
pub fn legacy_input_from_env() -> bool {
    std::env::var("UMADEV_LEGACY_INPUT").is_ok_and(|v| {
        let v = v.trim();
        v == "1" || v.eq_ignore_ascii_case("true")
    })
}

/// The ESC-flush timeout, env-overridable via `UMADEV_ESC_FLUSH_MS` and clamped
/// to a sane `1..=1000` ms range (a `0` would flush every lone ESC instantly and
/// resurrect the phantom-Esc race; a huge value would make Esc feel laggy).
fn esc_flush_interval() -> Duration {
    let ms = std::env::var("UMADEV_ESC_FLUSH_MS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&v| (1..=1000).contains(&v))
        .unwrap_or(DEFAULT_ESC_FLUSH_MS);
    Duration::from_millis(ms)
}

/// Spawn the blocking stdin reader thread and return the byte channel.
///
/// `std::io::stdin().lock()` on a dedicated thread (NOT `libc::read` — no
/// `unsafe`). Each `read()` returns the bytes currently available; we forward
/// them to the async side. Fail-open: EOF, a send error (receiver dropped), or
/// a non-interrupt read error all end the thread cleanly; the receiver then sees
/// the channel close and the input source degrades gracefully (the rest of the
/// app keeps running). On process exit a thread still blocked in `read()` is
/// reaped by the OS — it is intentionally detached, never joined.
fn spawn_stdin_reader() -> UnboundedReceiver<Vec<u8>> {
    let (tx, rx): (UnboundedSender<Vec<u8>>, _) = tokio::sync::mpsc::unbounded_channel();
    // If the thread can't even be spawned, `spawn` consumes + drops the closure
    // (and its captured `tx`), so the channel closes immediately and the receiver
    // degrades gracefully (fail-open: input is dead, the app still runs).
    let _ = std::thread::Builder::new()
        .name("umadev-stdin".into())
        .spawn(move || {
            use std::io::Read as _;
            let stdin = std::io::stdin();
            let mut lock = stdin.lock();
            let mut buf = [0u8; READ_CHUNK];
            loop {
                match lock.read(&mut buf) {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        if tx.send(buf[..n].to_vec()).is_err() {
                            break; // receiver gone
                        }
                    }
                    // A signal interrupted the read — just loop and try again.
                    Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(_) => break,
                }
            }
        });
    rx
}

/// The SIGWINCH (terminal-resize) signal stream the owned source selects on. On
/// non-unix there is no such signal; the field is `None` and the arm stays inert
/// (mirrors the SIGCONT R5 handling in the event loop).
#[cfg(unix)]
type WinchSignal = tokio::signal::unix::Signal;
/// See [`WinchSignal`] — the non-unix placeholder.
#[cfg(not(unix))]
type WinchSignal = ();

/// Register the SIGWINCH listener (terminal resize). tokio installs the handler
/// safely (no `unsafe`, no work in signal context). `None` on non-unix / if
/// registration fails (fail-open: resize self-heal just won't fire).
fn register_winch_signal() -> Option<WinchSignal> {
    #[cfg(unix)]
    {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::window_change()).ok()
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// Await the next SIGWINCH. Inert (never resolves) on non-unix / if registration
/// failed, so the select! arm stays dormant rather than busy-looping.
async fn next_winch(sig: &mut Option<WinchSignal>) {
    #[cfg(unix)]
    {
        match sig.as_mut() {
            Some(s) => {
                let _ = s.recv().await;
            }
            None => std::future::pending::<()>().await,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = sig;
        std::future::pending::<()>().await;
    }
}

/// P2 — parse a **DECRPM reply** to the startup DEC-2026 (synchronized output)
/// DECRQM probe: `\x1b[?2026;<n>$y`. `n = 1` (set) or `2` (reset) mean the
/// terminal implements the mode → `Some(true)`; any other recognized reply
/// value (`0` = not recognized, `3`/`4` = permanently locked) → `Some(false)`;
/// bytes that are not a 2026 DECRPM at all → `None` (not our reply). The event
/// loop sends the query once at startup and reads the verdict via
/// [`InputSource::take_sync_output_reply`]; routing the reply through the ONE
/// owned tokenizer (instead of a second stdin reader) is what keeps it from
/// racing the input stream or leaking as keystrokes.
fn decrpm_2026_verdict(bytes: &[u8]) -> Option<bool> {
    let n = bytes
        .strip_prefix(b"\x1b[?2026;")?
        .strip_suffix(b"$y")?
        .iter()
        .try_fold(0u32, |acc, &b| {
            b.is_ascii_digit()
                .then(|| acc.saturating_mul(10) + u32::from(b - b'0'))
        })?;
    Some(n == 1 || n == 2)
}

/// Sleep until `deadline`, or never (when `None`) — so the ESC-flush arm is a
/// plain always-enabled select! branch (no precondition) that simply parks when
/// no flush is pending.
async fn sleep_until_opt(deadline: Option<Instant>) {
    match deadline {
        Some(d) => {
            // Saturating: a past deadline fires immediately.
            tokio::time::sleep_until(tokio::time::Instant::from_std(d)).await;
        }
        None => std::future::pending::<()>().await,
    }
}

/// The owned stdin input source: byte channel + tokenizer + decoder + a ready
/// event queue + the ESC-flush deadline + the SIGWINCH listener.
pub struct OwnedInput {
    /// Raw byte chunks from the reader thread.
    rx: UnboundedReceiver<Vec<u8>>,
    /// Boundary tokenizer (persistent buffer across reads).
    tokenizer: Tokenizer,
    /// Token → event decoder (persistent paste state).
    decoder: Decoder,
    /// Decoded events ready to hand out, one per [`OwnedInput::next`] call.
    queue: VecDeque<Event>,
    /// When `Some`, the instant at which a buffered lone-ESC should flush to a
    /// real Esc (unless its continuation arrives first).
    esc_deadline: Option<Instant>,
    /// The configured ESC-flush window.
    esc_interval: Duration,
    /// SIGWINCH listener (resize) — `None` on non-unix / registration failure.
    winch: Option<WinchSignal>,
    /// Whether the reader channel has closed (thread ended). Disables the recv
    /// arm so the source parks instead of busy-looping on `None`.
    closed: bool,
    /// P2 — the captured verdict of the startup DEC-2026 DECRQM probe, parked
    /// here when the DECRPM reply flows through the decoder (see
    /// [`decrpm_2026_verdict`]). `None` until (unless) the terminal answers;
    /// drained one-shot by [`OwnedInput::take_sync_output_reply`].
    sync_output_reply: Option<bool>,
}

impl OwnedInput {
    /// Create the owned source: spawn the reader thread, register SIGWINCH.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rx: spawn_stdin_reader(),
            tokenizer: Tokenizer::for_stdin(),
            decoder: Decoder::new(),
            queue: VecDeque::new(),
            esc_deadline: None,
            esc_interval: esc_flush_interval(),
            winch: register_winch_signal(),
            closed: false,
            sync_output_reply: None,
        }
    }

    /// Decode-side event sink shared by [`Self::ingest`] and
    /// [`Self::flush_escape`]: captures the DEC-2026 DECRPM probe reply (P2 —
    /// consumed here, never surfaced as input) and enqueues everything that maps
    /// to a real terminal event. Every other [`InputEvent::Response`] stays
    /// dropped exactly as before.
    fn enqueue(&mut self, ev: InputEvent) {
        if let InputEvent::Response(bytes) = &ev {
            if let Some(verdict) = decrpm_2026_verdict(bytes) {
                self.sync_output_reply = Some(verdict);
            }
        }
        if let Some(event) = ev.into_event() {
            self.queue.push_back(event);
        }
    }

    /// P2 — take the captured DECRPM verdict for the startup synchronized-output
    /// probe, if the terminal has answered. One-shot (`None` after the first
    /// take); the event loop polls this until its probe deadline, then falls
    /// back to the env allowlist.
    #[must_use]
    pub fn take_sync_output_reply(&mut self) -> Option<bool> {
        self.sync_output_reply.take()
    }

    /// Feed a byte chunk through the tokenizer + decoder, enqueueing events, then
    /// (re)arm or disarm the ESC-flush deadline.
    fn ingest(&mut self, bytes: &[u8]) {
        for token in self.tokenizer.feed(bytes) {
            for ev in self.decoder.feed_token(token) {
                self.enqueue(ev);
            }
        }
        self.update_esc_deadline();
    }

    /// Force-flush a buffered incomplete escape (the lone-ESC → Esc verdict).
    fn flush_escape(&mut self) {
        for token in self.tokenizer.flush() {
            for ev in self.decoder.feed_token(token) {
                self.enqueue(ev);
            }
        }
        self.esc_deadline = None;
    }

    /// (Re)arm the flush deadline iff an incomplete escape is buffered. Reset on
    /// every ingest so a still-incomplete sequence extends the window (mirrors a
    /// mature reference terminal's re-arm-on-each-input behaviour).
    fn update_esc_deadline(&mut self) {
        if self.tokenizer.has_pending_escape() {
            self.esc_deadline = Some(Instant::now() + self.esc_interval);
        } else {
            self.esc_deadline = None;
        }
    }

    /// Yield the next input event, awaiting bytes / the ESC-flush timer /
    /// SIGWINCH as needed. Returns `None` only on a hard end (never for an idle
    /// terminal); a closed reader channel parks instead so the rest of the loop
    /// keeps running.
    pub async fn next(&mut self) -> Option<std::io::Result<Event>> {
        loop {
            if let Some(event) = self.queue.pop_front() {
                return Some(Ok(event));
            }
            let deadline = self.esc_deadline;
            tokio::select! {
                chunk = self.rx.recv(), if !self.closed => {
                    match chunk {
                        Some(bytes) => self.ingest(&bytes),
                        None => self.closed = true,
                    }
                }
                () = sleep_until_opt(deadline) => {
                    // The flush timer fired. Re-arm trick: if a continuation is
                    // already queued (a heavy render blocked the loop past the
                    // timeout), ingest it instead of flushing — so a split arrow
                    // completes and no phantom Esc surfaces. Only a genuinely
                    // idle FD flushes the buffered lone ESC as a real Esc.
                    self.esc_deadline = None;
                    match self.rx.try_recv() {
                        Ok(bytes) => self.ingest(&bytes),
                        Err(_) => self.flush_escape(),
                    }
                }
                () = next_winch(&mut self.winch) => {
                    if let Ok((cols, rows)) = crossterm::terminal::size() {
                        self.queue.push_back(Event::Resize(cols, rows));
                    }
                }
            }
        }
    }
}

impl Default for OwnedInput {
    fn default() -> Self {
        Self::new()
    }
}

impl InputEvent {
    /// Convert a decoded event to a crossterm [`Event`] for the unified event
    /// loop. A [`InputEvent::Response`] (terminal query reply) maps to `None` —
    /// it is dropped, never surfaced as input.
    fn into_event(self) -> Option<Event> {
        match self {
            InputEvent::Key(k) => Some(Event::Key(k)),
            InputEvent::Mouse(m) => Some(Event::Mouse(m)),
            InputEvent::Paste(p) => Some(Event::Paste(p)),
            InputEvent::Focus(true) => Some(Event::FocusGained),
            InputEvent::Focus(false) => Some(Event::FocusLost),
            InputEvent::Resize(c, r) => Some(Event::Resize(c, r)),
            InputEvent::Response(_) => None,
        }
    }
}

/// The event-loop input source: the owned byte tokenizer (default) or the
/// legacy `crossterm::EventStream` (escape hatch). Both expose one async
/// [`InputSource::next`] returning `Option<io::Result<Event>>`, so the event
/// loop's `select!` arm is identical for either — the gate is a clean branch at
/// setup, never a per-event check.
pub enum InputSource {
    /// The owned byte-tokenizer source (default).
    Owned(Box<OwnedInput>),
    /// The legacy crossterm stream (`UMADEV_LEGACY_INPUT=1`).
    Legacy(Box<EventStream>),
}

impl InputSource {
    /// Construct the source per the escape-hatch env gate.
    #[must_use]
    pub fn from_env() -> Self {
        if cfg!(windows) || legacy_input_from_env() {
            InputSource::Legacy(Box::new(EventStream::new()))
        } else {
            InputSource::Owned(Box::<OwnedInput>::default())
        }
    }

    /// Whether this is the owned tokenizer path. Used to bypass the legacy
    /// `MouseSeqFilter` backstop (the tokenizer subsumes it, and re-buffering a
    /// resolved Esc through the filter would re-introduce the very Esc latency
    /// the root fix removes).
    #[must_use]
    pub fn is_owned(&self) -> bool {
        matches!(self, InputSource::Owned(_))
    }

    /// Await the next terminal event.
    pub async fn next(&mut self) -> Option<std::io::Result<Event>> {
        match self {
            InputSource::Owned(o) => o.next().await,
            InputSource::Legacy(s) => s.next().await,
        }
    }

    /// P2 — take the terminal's DECRPM answer to the startup synchronized-output
    /// probe, if it has arrived (one-shot). Always `None` on the legacy path:
    /// crossterm's parser owns stdin there and has no lane for the reply, which
    /// is exactly why the event loop only SENDS the probe on the owned path and
    /// falls back to the env allowlist at the deadline otherwise.
    #[must_use]
    pub fn take_sync_output_reply(&mut self) -> Option<bool> {
        match self {
            InputSource::Owned(o) => o.take_sync_output_reply(),
            InputSource::Legacy(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct EnvRestore {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvRestore {
        fn capture(key: &'static str) -> Self {
            Self {
                key,
                prev: std::env::var(key).ok(),
            }
        }

        fn set(&self, value: &str) {
            std::env::set_var(self.key, value);
        }

        fn remove(&self) {
            std::env::remove_var(self.key);
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            match self.prev.as_ref() {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn legacy_input_env_gate() {
        let _guard = ENV_LOCK.lock().unwrap();
        let env = EnvRestore::capture("UMADEV_LEGACY_INPUT");
        env.remove();
        assert!(!legacy_input_from_env(), "default is the owned tokenizer");
        env.set("1");
        assert!(legacy_input_from_env(), "=1 selects the legacy path");
        env.set("true");
        assert!(legacy_input_from_env(), "=true also selects legacy");
        env.set("0");
        assert!(!legacy_input_from_env(), "=0 stays on the owned path");
    }

    #[tokio::test]
    #[cfg(not(windows))]
    async fn input_source_defaults_to_owned_off_windows() {
        let _guard = ENV_LOCK.lock().unwrap();
        let env = EnvRestore::capture("UMADEV_LEGACY_INPUT");
        env.remove();
        assert!(InputSource::from_env().is_owned());
    }

    #[tokio::test]
    #[cfg(windows)]
    async fn input_source_defaults_to_legacy_on_windows() {
        let _guard = ENV_LOCK.lock().unwrap();
        let env = EnvRestore::capture("UMADEV_LEGACY_INPUT");
        env.remove();
        assert!(
            !InputSource::from_env().is_owned(),
            "Windows console Esc/arrows are native input records, not stdin bytes"
        );
    }

    #[test]
    fn esc_flush_interval_clamps() {
        let _guard = ENV_LOCK.lock().unwrap();
        let env = EnvRestore::capture("UMADEV_ESC_FLUSH_MS");
        env.remove();
        assert_eq!(
            esc_flush_interval(),
            Duration::from_millis(DEFAULT_ESC_FLUSH_MS)
        );
        env.set("0");
        assert_eq!(
            esc_flush_interval(),
            Duration::from_millis(DEFAULT_ESC_FLUSH_MS),
            "0 is rejected (clamped to default)"
        );
        env.set("120");
        assert_eq!(esc_flush_interval(), Duration::from_millis(120));
        env.set("999999");
        assert_eq!(
            esc_flush_interval(),
            Duration::from_millis(DEFAULT_ESC_FLUSH_MS),
            "out-of-range is rejected"
        );
    }

    /// A bare `OwnedInput` around a hand-made byte channel — no stdin reader
    /// thread, no SIGWINCH — so the decode/capture path is testable hermetically.
    fn owned_for_test() -> (OwnedInput, UnboundedSender<Vec<u8>>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (
            OwnedInput {
                rx,
                tokenizer: Tokenizer::for_stdin(),
                decoder: Decoder::new(),
                queue: VecDeque::new(),
                esc_deadline: None,
                esc_interval: Duration::from_millis(DEFAULT_ESC_FLUSH_MS),
                winch: None,
                closed: false,
                sync_output_reply: None,
            },
            tx,
        )
    }

    #[test]
    fn decrpm_2026_verdict_parses_supported_and_unsupported() {
        // n=1 (set) and n=2 (reset) both mean the mode is implemented.
        assert_eq!(decrpm_2026_verdict(b"\x1b[?2026;1$y"), Some(true));
        assert_eq!(decrpm_2026_verdict(b"\x1b[?2026;2$y"), Some(true));
        // n=0 = not recognized → unsupported.
        assert_eq!(decrpm_2026_verdict(b"\x1b[?2026;0$y"), Some(false));
        assert_eq!(decrpm_2026_verdict(b"\x1b[?2026;4$y"), Some(false));
        // A DECRPM for a DIFFERENT mode, a DA1 reply, or ordinary keys are not
        // ours — `None`, never a false verdict.
        assert_eq!(decrpm_2026_verdict(b"\x1b[?2004;1$y"), None);
        assert_eq!(decrpm_2026_verdict(b"\x1b[?62;1c"), None);
        assert_eq!(decrpm_2026_verdict(b"hello"), None);
        // Garbage where the digit should be → not a verdict.
        assert_eq!(decrpm_2026_verdict(b"\x1b[?2026;x$y"), None);
    }

    #[test]
    fn sync_probe_reply_is_captured_and_never_leaks_as_input() {
        let (mut oi, _tx) = owned_for_test();
        oi.ingest(b"\x1b[?2026;1$y");
        assert!(
            oi.queue.is_empty(),
            "the DECRPM reply must be consumed, never surfaced as keystrokes"
        );
        assert_eq!(oi.take_sync_output_reply(), Some(true), "verdict captured");
        assert_eq!(oi.take_sync_output_reply(), None, "the take is one-shot");
    }

    #[test]
    fn sync_probe_reply_split_across_reads_and_mixed_with_typing_still_resolves() {
        // The reply can straddle a read boundary and be followed by real input
        // in the same chunk — the tokenizer reassembles the sequence, the
        // verdict is captured, and ONLY the real keys surface.
        let (mut oi, _tx) = owned_for_test();
        oi.ingest(b"\x1b[?2026;");
        assert!(oi.queue.is_empty(), "an incomplete reply emits nothing");
        oi.ingest(b"0$yhi");
        assert_eq!(oi.take_sync_output_reply(), Some(false));
        let keys: Vec<Event> = std::mem::take(&mut oi.queue).into();
        assert_eq!(
            keys.len(),
            2,
            "exactly the two real keystrokes surface, none of the reply: {keys:?}"
        );
    }

    #[test]
    fn input_event_into_event_maps_surface() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        assert!(matches!(
            InputEvent::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)).into_event(),
            Some(Event::Key(_))
        ));
        assert!(matches!(
            InputEvent::Paste("x".into()).into_event(),
            Some(Event::Paste(_))
        ));
        assert!(matches!(
            InputEvent::Focus(true).into_event(),
            Some(Event::FocusGained)
        ));
        assert!(matches!(
            InputEvent::Focus(false).into_event(),
            Some(Event::FocusLost)
        ));
        assert!(matches!(
            InputEvent::Resize(80, 24).into_event(),
            Some(Event::Resize(80, 24))
        ));
        // A terminal response is dropped (not surfaced as input).
        assert!(InputEvent::Response(b"\x1b[?1c".to_vec())
            .into_event()
            .is_none());
    }
}
