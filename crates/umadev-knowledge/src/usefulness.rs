//! Retrieval-quality feedback — a per-chunk **usefulness prior** that lets the
//! curated-knowledge ranking SELF-TUNE from build outcomes.
//!
//! Static BM25 / vector / RRF ranking answers "which chunk best matches the
//! query?" but never learns from what happened after a chunk was injected. This
//! module adds a thin, bounded, fail-open memory: a chunk that preceded a CLEAN
//! step earns a usefulness boost; one that preceded a FAILURE is demoted. Over
//! many runs the corpus's own track record nudges ranking, WITHOUT discarding
//! lexical/semantic relevance — the prior is a *multiplicative* weight blended on
//! top of the BM25/vector score in [`crate::retrieve`], never a replacement.
//!
//! ## Where it lives (cross-project)
//!
//! The curated `knowledge/` corpus is shared across every project, so its track
//! record is too: the store is a single JSON file under the user home
//! (`~/.umadev/knowledge-usefulness.json`), NOT per-project. A per-project
//! transient "which chunks were surfaced for this step" snapshot lives in the
//! agent crate (mirroring its surfaced-lesson-identity snapshot); this module
//! only owns the durable, cross-project prior it feeds.
//!
//! ## Conservatism contract
//!
//! - **Sample-gated.** Below [`MIN_SAMPLES`] observations a chunk's weight is
//!   NEUTRAL (`1.0`) — a single observation never moves ranking, and a fresh
//!   corpus (no observations at all) ranks byte-for-byte as before.
//! - **Bounded weight.** Once well-sampled the weight stays within
//!   `[WEIGHT_MIN, WEIGHT_MAX]` (`0.3..=1.2`) — a proven-helpful chunk lifts, a
//!   proven-harmful one sinks, but relevance still dominates.
//! - **Bounded store.** At most [`MAX_ENTRIES`] chunk keys are retained
//!   (least-recently-updated evicted first, deterministically).
//! - **Fail-open.** A missing / corrupt / unwritable store degrades to the
//!   neutral prior (today's static ranking) — never a panic, never an error.
//! - **Deterministic.** Pure integer bookkeeping + a fixed weight map; no clock
//!   read decides ranking, no brain consult, reproducible run-to-run.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Minimum outcome observations (`helpful + harmful`) before a chunk's prior may
/// move its rank. Below this the weight is neutral `1.0`, so a single observation
/// can never dominate and an unobserved corpus is unchanged.
pub const MIN_SAMPLES: u32 = 3;

/// Hard cap on distinct chunk keys the store retains (bounded memory). When
/// exceeded, the least-recently-updated entries are evicted first.
pub const MAX_ENTRIES: usize = 4096;

/// Defensive cap on how many chunk keys ONE outcome record processes, so a
/// caller can never explode the store in a single call (the caller already caps
/// the per-step snapshot, this is belt-and-suspenders).
pub const MAX_RECORD_BATCH: usize = 64;

/// Lowest multiplicative weight a proven-HARMFUL chunk can sink to.
const WEIGHT_MIN: f32 = 0.3;
/// Highest multiplicative weight a proven-HELPFUL chunk can rise to.
const WEIGHT_MAX: f32 = 1.2;

/// The neutral weight applied to an unobserved / thinly-sampled chunk — leaves
/// the BM25/vector ranking exactly as it was.
pub const NEUTRAL_WEIGHT: f32 = 1.0;

/// Store filename under the `.umadev` state dir in the user home.
const USEFULNESS_FILE: &str = "knowledge-usefulness.json";
/// The `.umadev` state subdir under the user home the store file lives in.
const STATE_SUBDIR: &str = ".umadev";

/// The helpful / harmful tally for one chunk, plus a monotone update stamp used
/// purely for deterministic eviction (NOT a wall clock).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct ChunkStat {
    /// Times this chunk was surfaced into a step that then PASSED.
    #[serde(default)]
    helpful: u32,
    /// Times this chunk was surfaced into a step that then FAILED.
    #[serde(default)]
    harmful: u32,
    /// Store-local monotone sequence at the last update — the eviction key.
    #[serde(default)]
    updated: u64,
}

/// The per-chunk usefulness prior, keyed by chunk identity
/// (`corpus-relative path` + `section heading`). A durable, cross-project map
/// loaded fail-open from `~/.umadev/knowledge-usefulness.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsefulnessStore {
    /// Monotone counter stamped onto every touched entry, so eviction has a
    /// deterministic recency order without reading a clock.
    #[serde(default)]
    seq: u64,
    /// `chunk_key -> tally`. Bounded to [`MAX_ENTRIES`].
    #[serde(default)]
    entries: HashMap<String, ChunkStat>,
}

/// Compose the stable identity key for a chunk: `path` + `section`, joined by a
/// unit-separator that cannot appear in either field. This is the SAME identity
/// both the snapshot writer and the ranking reader key on, so they always agree.
#[must_use]
pub fn chunk_key(path: &str, section: &str) -> String {
    format!("{path}\u{1f}{section}")
}

/// Resolve the store file path under an explicit home dir.
fn usefulness_path(home: &Path) -> PathBuf {
    home.join(STATE_SUBDIR).join(USEFULNESS_FILE)
}

/// Resolve the user home dir the cross-project store lives under. Honors an
/// explicit `UMADEV_HOME` override first (so callers + tests can redirect it),
/// then `HOME` (Unix) / `USERPROFILE` (Windows). `None` when none is set —
/// callers then no-op (fail-open).
fn usefulness_home() -> Option<PathBuf> {
    std::env::var("UMADEV_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOME").ok())
        .or_else(|| std::env::var("USERPROFILE").ok())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

impl UsefulnessStore {
    /// Load the cross-project store from the user home. Fail-open: no home, a
    /// missing file, or a corrupt/unreadable blob all yield an EMPTY store (every
    /// weight then neutral — today's static ranking), never an error.
    #[must_use]
    pub fn load() -> Self {
        usefulness_home().map_or_else(Self::default, |home| Self::load_from(&home))
    }

    /// Load the store from an explicit home dir (the durable file is at
    /// `<home>/.umadev/knowledge-usefulness.json`). Fail-open to an empty store.
    /// Exposed so the record bridge + tests can point at a temp home.
    #[must_use]
    pub fn load_from(home: &Path) -> Self {
        let path = usefulness_path(home);
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str::<Self>(&t).ok())
            .unwrap_or_default()
    }

    /// The multiplicative usefulness weight for a chunk, in `[WEIGHT_MIN,
    /// WEIGHT_MAX]`. NEUTRAL (`1.0`) when the chunk is unobserved or has fewer
    /// than [`MIN_SAMPLES`] observations (so a single observation never
    /// dominates and a fresh corpus is unchanged). Otherwise a linear map of the
    /// helpful ratio: all-helpful → `WEIGHT_MAX`, all-harmful → `WEIGHT_MIN`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn weight_for(&self, path: &str, section: &str) -> f32 {
        let Some(stat) = self.entries.get(&chunk_key(path, section)) else {
            return NEUTRAL_WEIGHT;
        };
        let total = stat.helpful.saturating_add(stat.harmful);
        if total < MIN_SAMPLES {
            return NEUTRAL_WEIGHT; // sample-gated: thin evidence stays neutral
        }
        let ratio = f64::from(stat.helpful) / f64::from(total);
        let span = f64::from(WEIGHT_MAX) - f64::from(WEIGHT_MIN);
        let w = (f64::from(WEIGHT_MIN) + span * ratio) as f32;
        w.clamp(WEIGHT_MIN, WEIGHT_MAX)
    }

    /// Record an outcome for a batch of surfaced chunk keys: `helpful = true`
    /// increments each chunk's helpful tally, `false` the harmful tally. Bounded
    /// ([`MAX_RECORD_BATCH`] keys per call) and self-capping ([`MAX_ENTRIES`]).
    /// Pure integer bookkeeping — deterministic, never fails.
    pub fn record(&mut self, keys: &[(String, String)], helpful: bool) {
        for (path, section) in keys.iter().take(MAX_RECORD_BATCH) {
            let key = chunk_key(path, section);
            self.seq = self.seq.saturating_add(1);
            let stat = self.entries.entry(key).or_default();
            if helpful {
                stat.helpful = stat.helpful.saturating_add(1);
            } else {
                stat.harmful = stat.harmful.saturating_add(1);
            }
            stat.updated = self.seq;
        }
        self.enforce_cap();
    }

    /// Evict least-recently-updated entries down to [`MAX_ENTRIES`]. Tiebreak on
    /// the key so eviction is deterministic even when `updated` stamps collide.
    fn enforce_cap(&mut self) {
        if self.entries.len() <= MAX_ENTRIES {
            return;
        }
        let mut items: Vec<(u64, String)> = self
            .entries
            .iter()
            .map(|(k, v)| (v.updated, k.clone()))
            .collect();
        // Oldest (smallest updated) first; key breaks ties.
        items.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let remove = self.entries.len() - MAX_ENTRIES;
        for (_, k) in items.into_iter().take(remove) {
            self.entries.remove(&k);
        }
    }

    /// Persist the store to an explicit home dir via an atomic temp+rename write.
    /// Fail-open: an unmakeable dir or a write error is swallowed (the prior just
    /// doesn't advance) — never a panic, never an error surfaced to a caller.
    pub fn save_to(&self, home: &Path) {
        let path = usefulness_path(home);
        let Some(parent) = path.parent() else {
            return;
        };
        if std::fs::create_dir_all(parent).is_err() {
            return;
        }
        let Ok(body) = serde_json::to_string(self) else {
            return;
        };
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, body.as_bytes()).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }

    /// Number of distinct chunk keys tracked (for tests / introspection).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store has no observations yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Record a step outcome for the surfaced chunk keys into the cross-project home
/// store: load → record → save, all fail-open. `helpful = true` on a PASS,
/// `false` on a FAIL. A no-op when there are no keys or no home dir — so it never
/// touches disk (and never pollutes a test home) unless there is real signal.
pub fn record_chunk_outcomes(keys: &[(String, String)], helpful: bool) {
    if keys.is_empty() {
        return;
    }
    let Some(home) = usefulness_home() else {
        return;
    };
    record_chunk_outcomes_in(&home, keys, helpful);
}

/// Explicit-home variant of [`record_chunk_outcomes`] — the durable file is at
/// `<home>/.umadev/knowledge-usefulness.json`. The bridge in the agent crate +
/// tests use this so the cross-project store can be redirected to a temp home.
pub fn record_chunk_outcomes_in(home: &Path, keys: &[(String, String)], helpful: bool) {
    if keys.is_empty() {
        return;
    }
    let mut store = UsefulnessStore::load_from(home);
    store.record(keys, helpful);
    store.save_to(home);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(path: &str, section: &str) -> (String, String) {
        (path.to_string(), section.to_string())
    }

    #[test]
    fn unobserved_chunk_is_neutral() {
        let store = UsefulnessStore::default();
        assert!((store.weight_for("a.md", "S") - NEUTRAL_WEIGHT).abs() < f32::EPSILON);
    }

    #[test]
    fn a_single_observation_does_not_move_ranking() {
        // One helpful observation is below MIN_SAMPLES → still neutral, so a single
        // outcome can never dominate the ranking.
        let mut store = UsefulnessStore::default();
        store.record(&[key("a.md", "S")], true);
        assert!(
            (store.weight_for("a.md", "S") - NEUTRAL_WEIGHT).abs() < f32::EPSILON,
            "a single observation must stay neutral (sample-gated)"
        );
    }

    #[test]
    fn well_sampled_helpful_chunk_lifts_weight() {
        let mut store = UsefulnessStore::default();
        for _ in 0..MIN_SAMPLES {
            store.record(&[key("a.md", "S")], true);
        }
        let w = store.weight_for("a.md", "S");
        assert!(
            w > NEUTRAL_WEIGHT,
            "all-helpful must lift above neutral: {w}"
        );
        assert!(
            (w - WEIGHT_MAX).abs() < 1e-4,
            "all-helpful maps to WEIGHT_MAX"
        );
    }

    #[test]
    fn well_sampled_harmful_chunk_sinks_weight() {
        let mut store = UsefulnessStore::default();
        for _ in 0..MIN_SAMPLES {
            store.record(&[key("bad.md", "S")], false);
        }
        let w = store.weight_for("bad.md", "S");
        assert!(
            w < NEUTRAL_WEIGHT,
            "all-harmful must sink below neutral: {w}"
        );
        assert!(
            (w - WEIGHT_MIN).abs() < 1e-4,
            "all-harmful maps to WEIGHT_MIN"
        );
    }

    #[test]
    fn weight_stays_within_bounds_for_mixed_signal() {
        let mut store = UsefulnessStore::default();
        store.record(&[key("m.md", "S")], true);
        store.record(&[key("m.md", "S")], true);
        store.record(&[key("m.md", "S")], false);
        let w = store.weight_for("m.md", "S");
        assert!(
            (WEIGHT_MIN..=WEIGHT_MAX).contains(&w),
            "weight in bounds: {w}"
        );
    }

    #[test]
    fn record_round_trips_through_an_explicit_home() {
        let home = tempfile::TempDir::new().unwrap();
        for _ in 0..MIN_SAMPLES {
            record_chunk_outcomes_in(home.path(), &[key("a.md", "S")], true);
        }
        let store = UsefulnessStore::load_from(home.path());
        assert!(
            store.weight_for("a.md", "S") > NEUTRAL_WEIGHT,
            "persisted helpful observations lift the weight on reload"
        );
    }

    #[test]
    fn a_passing_step_gains_and_a_failing_step_loses_usefulness() {
        let home = tempfile::TempDir::new().unwrap();
        // A chunk in front of passing steps climbs; a different chunk in front of
        // failing steps sinks — the two diverge.
        for _ in 0..MIN_SAMPLES {
            record_chunk_outcomes_in(home.path(), &[key("good.md", "S")], true);
            record_chunk_outcomes_in(home.path(), &[key("bad.md", "S")], false);
        }
        let store = UsefulnessStore::load_from(home.path());
        assert!(store.weight_for("good.md", "S") > store.weight_for("bad.md", "S"));
        assert!(store.weight_for("good.md", "S") > NEUTRAL_WEIGHT);
        assert!(store.weight_for("bad.md", "S") < NEUTRAL_WEIGHT);
    }

    #[test]
    fn load_is_fail_open_on_a_corrupt_store() {
        let home = tempfile::TempDir::new().unwrap();
        let path = usefulness_path(home.path());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{ this is not valid json ][").unwrap();
        let store = UsefulnessStore::load_from(home.path());
        assert!(
            store.is_empty(),
            "a corrupt store loads as empty (fail-open)"
        );
        assert!((store.weight_for("x.md", "S") - NEUTRAL_WEIGHT).abs() < f32::EPSILON);
    }

    #[test]
    fn record_is_fail_open_on_a_missing_home() {
        // A home that cannot be created (a FILE where the dir must be) must not
        // panic; the outcome is simply dropped.
        let tmp = tempfile::TempDir::new().unwrap();
        let file_as_home = tmp.path().join("iam-a-file");
        std::fs::write(&file_as_home, b"x").unwrap();
        record_chunk_outcomes_in(&file_as_home, &[key("a.md", "S")], true);
        // No panic == pass; the store never materialised.
        assert!(UsefulnessStore::load_from(&file_as_home).is_empty());
    }

    #[test]
    fn empty_keys_never_touch_disk() {
        let home = tempfile::TempDir::new().unwrap();
        record_chunk_outcomes_in(home.path(), &[], true);
        assert!(
            !usefulness_path(home.path()).exists(),
            "an empty batch must not create the store file"
        );
    }

    #[test]
    fn store_is_bounded_and_evicts_oldest() {
        let mut store = UsefulnessStore::default();
        // Insert well over the cap; the store must never exceed MAX_ENTRIES.
        for i in 0..(MAX_ENTRIES + 50) {
            store.record(&[key(&format!("f{i}.md"), "S")], true);
        }
        assert!(store.len() <= MAX_ENTRIES, "store size stays bounded");
        // The most recently inserted key survived; the very first was evicted.
        assert!(store
            .entries
            .contains_key(&chunk_key(&format!("f{}.md", MAX_ENTRIES + 49), "S")));
        assert!(!store.entries.contains_key(&chunk_key("f0.md", "S")));
    }

    #[test]
    fn record_batch_is_capped() {
        let mut store = UsefulnessStore::default();
        let keys: Vec<(String, String)> = (0..(MAX_RECORD_BATCH + 20))
            .map(|i| key(&format!("f{i}.md"), "S"))
            .collect();
        store.record(&keys, true);
        assert_eq!(
            store.len(),
            MAX_RECORD_BATCH,
            "one record call processes at most MAX_RECORD_BATCH keys"
        );
    }
}
