//! Retrieval-quality feedback bridge — threads WHICH curated-knowledge chunks
//! were surfaced into a build step out to that step's PASS/FAIL outcome, so the
//! cross-project usefulness prior ([`umadev_knowledge::usefulness`]) can self-tune
//! the ranking over time.
//!
//! ## The seam (modelled on surfaced-lesson-identities)
//!
//! Knowledge is retrieved per step deep inside firmware composition
//! ([`crate::phases::seat_scoped_knowledge_digest`] /
//! [`crate::phases::agentic_knowledge_digest`]) — far from where the step's
//! deterministic acceptance verdict is finally known
//! ([`crate::director_loop`]). Rather than thread the retrieved chunk ids through
//! every call in between, this mirrors EXACTLY what the lessons layer already does
//! for surfaced lesson identities (`lessons::record_surfaced_identities` →
//! `read_surfaced_identities`): the retrieval site drops a small, bounded,
//! OVERWRITE-most-recent snapshot of the surfaced chunk keys into the project's
//! `_raw` dir, and the outcome seam (`self_evolve::reward_on_pass` /
//! `penalise_on_fail`) reads it back and folds it into the SAME feedback call.
//!
//! ## Contract
//!
//! - **Transient + project-local.** The snapshot is per-step scratch state (like
//!   the surfaced-identities snapshot); only the durable usefulness PRIOR it feeds
//!   is cross-project (under the user home).
//! - **Bounded.** At most [`MAX_TRACKED_CHUNKS`] chunk keys are snapshotted per
//!   step.
//! - **Fail-open + deterministic.** Every IO here is best-effort: a missing dir,
//!   an unreadable/corrupt snapshot, or no home dir all degrade to "no feedback"
//!   — never a panic, never a changed step outcome. No brain consult.
//! - **A side effect of the verdict, never a driver of it.** Recording usefulness
//!   is invoked AFTER acceptance is computed; it changes only future RANKING.

use std::path::Path;

use crate::lessons::RAW_DIR;

/// The per-step snapshot of surfaced knowledge-chunk keys, written by the
/// retrieval site and read at the outcome seam. Lives beside the surfaced
/// lesson-identity snapshot in `.umadev/learned/_raw/`.
pub const SURFACED_CHUNKS_FILE: &str = "surfaced-chunks.json";

/// Hard cap on how many chunk keys one step's snapshot retains — the step only
/// injects a handful of chunks, and this bounds the outcome record either way.
pub const MAX_TRACKED_CHUNKS: usize = 12;

/// Full path to the surfaced-chunks snapshot for a project.
fn snapshot_path(project_root: &Path) -> std::path::PathBuf {
    project_root.join(RAW_DIR).join(SURFACED_CHUNKS_FILE)
}

/// Snapshot the `(path, section)` keys of the chunks a step just surfaced,
/// OVERWRITING any prior snapshot (only the MOST RECENT surfacing is what a later
/// verify outcome can attribute to — exactly the surfaced-identities policy).
/// Bounded to [`MAX_TRACKED_CHUNKS`]; fail-open (any IO / serialize error is
/// swallowed). An empty input clears nothing extra — it simply writes an empty
/// list so a stale snapshot from a previous step cannot be mis-attributed.
pub fn record_surfaced_chunks(project_root: &Path, keys: &[(String, String)]) {
    let raw_dir = project_root.join(RAW_DIR);
    let _ = std::fs::create_dir_all(&raw_dir);
    let bounded: Vec<&(String, String)> = keys.iter().take(MAX_TRACKED_CHUNKS).collect();
    if let Ok(json) = serde_json::to_string(&bounded) {
        let _ = std::fs::write(raw_dir.join(SURFACED_CHUNKS_FILE), json);
    }
}

/// Read the most recently surfaced chunk keys (written by
/// [`record_surfaced_chunks`]). Fail-open: a missing/corrupt snapshot yields an
/// empty vec (no feedback), never an error.
#[must_use]
pub fn read_surfaced_chunks(project_root: &Path) -> Vec<(String, String)> {
    std::fs::read_to_string(snapshot_path(project_root))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Fold the surfaced chunks of a step into the cross-project usefulness prior
/// with the step's outcome (`helpful = true` on a PASS, `false` on a FAIL). Reads
/// the project-local snapshot, records into the home store (or `home`, when given
/// — used by tests). A no-op when nothing was surfaced. Fail-open + deterministic.
fn apply(project_root: &Path, home: Option<&Path>, helpful: bool) {
    let keys = read_surfaced_chunks(project_root);
    if keys.is_empty() {
        return; // nothing surfaced → nothing to attribute (never touches the store)
    }
    match home {
        Some(h) => umadev_knowledge::usefulness::record_chunk_outcomes_in(h, &keys, helpful),
        None => umadev_knowledge::usefulness::record_chunk_outcomes(&keys, helpful),
    }
}

/// A step PASSED: reward the knowledge chunks that were in front of the doer.
/// Best-effort, fail-open, deterministic — never changes the step outcome.
pub fn reward_surfaced_chunks(project_root: &Path) {
    apply(project_root, None, true);
}

/// A step FAILED: demote the knowledge chunks that were in front of the doer.
/// Best-effort, fail-open, deterministic — never changes the step outcome.
pub fn penalise_surfaced_chunks(project_root: &Path) {
    apply(project_root, None, false);
}

#[cfg(test)]
mod tests {
    use super::*;
    use umadev_knowledge::usefulness::{UsefulnessStore, MIN_SAMPLES, NEUTRAL_WEIGHT};

    fn key(path: &str, section: &str) -> (String, String) {
        (path.to_string(), section.to_string())
    }

    #[test]
    fn snapshot_round_trips_and_is_bounded() {
        let tmp = tempfile::TempDir::new().unwrap();
        let keys: Vec<(String, String)> = (0..(MAX_TRACKED_CHUNKS + 5))
            .map(|i| key(&format!("f{i}.md"), "S"))
            .collect();
        record_surfaced_chunks(tmp.path(), &keys);
        let back = read_surfaced_chunks(tmp.path());
        assert_eq!(
            back.len(),
            MAX_TRACKED_CHUNKS,
            "snapshot is bounded to MAX_TRACKED_CHUNKS"
        );
        assert_eq!(back[0], key("f0.md", "S"));
    }

    #[test]
    fn read_is_fail_open_on_a_missing_snapshot() {
        let tmp = tempfile::TempDir::new().unwrap();
        assert!(read_surfaced_chunks(tmp.path()).is_empty());
    }

    #[test]
    fn a_passing_step_rewards_its_surfaced_chunks() {
        let project = tempfile::TempDir::new().unwrap();
        let home = tempfile::TempDir::new().unwrap();
        record_surfaced_chunks(project.path(), &[key("security/login.md", "OAuth")]);
        // Reward across enough passing steps to cross the sample gate.
        for _ in 0..MIN_SAMPLES {
            apply(project.path(), Some(home.path()), true);
        }
        let store = UsefulnessStore::load_from(home.path());
        assert!(
            store.weight_for("security/login.md", "OAuth") > NEUTRAL_WEIGHT,
            "a chunk surfaced for passing steps gains usefulness"
        );
    }

    #[test]
    fn a_failing_step_penalises_its_surfaced_chunks() {
        let project = tempfile::TempDir::new().unwrap();
        let home = tempfile::TempDir::new().unwrap();
        record_surfaced_chunks(project.path(), &[key("security/login.md", "OAuth")]);
        for _ in 0..MIN_SAMPLES {
            apply(project.path(), Some(home.path()), false);
        }
        let store = UsefulnessStore::load_from(home.path());
        assert!(
            store.weight_for("security/login.md", "OAuth") < NEUTRAL_WEIGHT,
            "a chunk surfaced for failing steps loses usefulness"
        );
    }

    #[test]
    fn no_surfaced_chunks_is_a_no_op_on_the_store() {
        let project = tempfile::TempDir::new().unwrap();
        let home = tempfile::TempDir::new().unwrap();
        // No snapshot written → apply must not create or touch the home store.
        apply(project.path(), Some(home.path()), true);
        assert!(
            UsefulnessStore::load_from(home.path()).is_empty(),
            "an empty snapshot records nothing (fail-open no-op)"
        );
    }
}
