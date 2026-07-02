//! Self-evolution wiring for the DEFAULT director-loop path — the memory
//! side-effects of a plan step's acceptance verdict that turn UmaDev's memory
//! from *capture + frequency + recall* into genuine *evolution*.
//!
//! ## Why this exists (the stranded machinery)
//!
//! The learning primitives — trust reward/penalty ([`crate::lessons::apply_dev_error_trust`]
//! / [`crate::lessons::apply_trust_for_identities`]), pitfall-resolved
//! ([`crate::lessons::mark_pitfalls_resolved`]), base-reflected correction
//! strategies ([`crate::lessons::recurring_pitfall_for_error`] →
//! [`crate::lessons::reflection_prompt`] → [`crate::lessons::record_pitfall_strategy`]),
//! failure-time recall ([`crate::lessons::lessons_for_error`]), and the
//! base-judged delivery reconcile ([`crate::lessons::reconcile_candidates`] →
//! [`crate::lessons::sediment_lessons_with_judge`]) — all EXIST and are exercised,
//! but were only ever wired into the LEGACY single-shot runner
//! (`crate::runner`). On the shipped default path (`crate::director_loop`) a
//! lesson's trust never moved, a pitfall was never marked resolved, and a
//! reflection never fired. This module RE-WIRES that same machinery onto the
//! default path; it designs no new memory mechanism.
//!
//! ## Invariant: a SIDE EFFECT of the verdict, never a driver of it
//!
//! Every function here is invoked AFTER UmaDev has already computed a step's
//! acceptance verdict on the deterministic floor. Nothing here changes that
//! verdict, drives loop control, or gates the run:
//!
//! - **Trust / pitfall writes are best-effort.** A store read/write error is a
//!   no-op (the underlying `lessons` mutators are fail-open); the step outcome is
//!   never affected.
//! - **The reflection + reconcile brain consults fork READ-ONLY and fail-open.** A
//!   failed/wedged fork, an offline brain, a timeout, or an empty reply leaves
//!   memory unchanged and never blocks the step (the SAME `fork() → ForkConsult`
//!   seam the critics + fact-extraction backstop use).
//! - **Bounded.** Reflection runs at most ONCE per recurring error signature per
//!   run (a run-scoped set the caller threads); the delivery reconcile spends at
//!   most [`MAX_RECONCILE_CALLS`] base consults.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use umadev_runtime::BaseSession;

use crate::continuous::{fork_with_timeout, ForkConsult};
use crate::events::{EngineEvent, EventSink};
use crate::lessons;

/// Hard cap on how many fresh lessons the delivery reconcile spends a base consult
/// on, so a large corpus can't explode delivery latency. Newest candidates first
/// (`reconcile_candidates` is already newest-first). Mirrors the legacy runner's
/// `MAX_RECONCILE_CALLS`.
const MAX_RECONCILE_CALLS: usize = 8;

/// Trust PENALTY side-effect of a step whose acceptance verdict FAILED — the
/// recalled lessons were in front of the doer and the step did not pass, so their
/// trust is nudged down (asymmetric: a fail pushes harder than a pass lifts).
///
/// Two channels, exactly as the legacy runner fed them at its failure site:
/// - the dev-error reflux keyed on the ACTUAL failure evidence
///   ([`lessons::apply_dev_error_trust`] with `passed = false`), and
/// - the surfaced NON-pitfall / belief lessons the step's recall snapshotted
///   ([`lessons::read_surfaced_identities`] → [`lessons::apply_trust_for_identities`]).
///
/// Best-effort + fail-open: empty inputs / an unreadable store adjust nothing.
/// Deterministic (no brain consult).
pub(crate) fn penalise_on_fail(root: &Path, failure_evidence: &[String]) {
    let _ = lessons::apply_dev_error_trust(root, failure_evidence, false);
    let ids = lessons::read_surfaced_identities(root);
    let _ = lessons::apply_trust_for_identities(root, &ids, false);
    // Retrieval-quality feedback (same seam as the lesson-trust reflux): the
    // curated-knowledge chunks surfaced into this step were in front of the doer
    // and it did NOT pass — demote their cross-project usefulness prior so future
    // ranking trusts them less. Fail-open + deterministic; a no-op when nothing was
    // surfaced. Only changes future RANKING, never this step's outcome.
    crate::knowledge_feedback::penalise_surfaced_chunks(root);
}

/// Trust REWARD (+ pitfall-resolved) side-effect of a step whose acceptance verdict
/// PASSED — the recalled lessons were in play and the gate then passed.
///
/// - ALWAYS rewards the surfaced NON-pitfall / belief lessons: they were recalled
///   into the step and it passed ([`lessons::apply_trust_for_identities`] with
///   `passed = true`).
/// - On a RECOVERY (this pass followed a recorded failing round, so
///   `recovered_from` carries that round's error evidence) it ALSO rewards the
///   dev-error pitfall whose recorded fix just held ([`lessons::apply_dev_error_trust`]
///   with `passed = true`) and marks it resolved ([`lessons::mark_pitfalls_resolved`]) —
///   the strongest signal that a pitfall's fix is effective. A clean first-pass
///   (`recovered_from` empty) skips this half: nothing failed, so nothing is
///   "resolved".
///
/// Best-effort + fail-open + deterministic. Never changes the step outcome.
pub(crate) fn reward_on_pass(root: &Path, recovered_from: &[String]) {
    let ids = lessons::read_surfaced_identities(root);
    let _ = lessons::apply_trust_for_identities(root, &ids, true);
    if !recovered_from.is_empty() {
        let _ = lessons::apply_dev_error_trust(root, recovered_from, true);
        let _ = lessons::mark_pitfalls_resolved(root, recovered_from);
    }
    // Retrieval-quality feedback (same seam as the lesson-trust reward): the
    // curated-knowledge chunks surfaced into this step were in front of the doer
    // and it PASSED — lift their cross-project usefulness prior so future ranking
    // surfaces them sooner. Fail-open + deterministic; a no-op when nothing was
    // surfaced. Only changes future RANKING, never this step's outcome.
    crate::knowledge_feedback::reward_surfaced_chunks(root);
}

/// Reflection: on a TRUE recurrence of a pitfall (its recorded fix already failed
/// and it came back), spend ONE cheap read-only fork consult to design a DIFFERENT
/// higher-level corrective strategy, and record it on the pitfall so later recall
/// ([`lessons::lessons_for_error`]) surfaces it instead of the bare template line.
///
/// Gated + bounded + fail-open:
/// - [`lessons::recurring_pitfall_for_error`] returns `Some` ONLY on a genuine
///   recurrence (a first failure stays on the cheap template path — no consult, no
///   cost).
/// - `reflected` is a run-scoped set of signatures already attempted this run; a
///   signature is inserted BEFORE the consult so reflection fires AT MOST ONCE per
///   recurring signature per run even if the consult fails.
/// - The consult forks READ-ONLY ([`fork_with_timeout`] + [`ForkConsult`]); a
///   failed/wedged fork, an offline brain, a timeout, or an empty reply degrades to
///   "no strategy recorded" (`false`) — never an error, never a blocked step.
///
/// Returns `true` iff a new strategy was recorded. `failure_detail` is the step's
/// failing evidence (joined), classified to a signature internally.
pub(crate) async fn reflect_on_recurring_failure(
    session: &mut dyn BaseSession,
    root: &Path,
    events: &Arc<dyn EventSink>,
    failure_detail: &str,
    reflected: &mut HashSet<String>,
) -> bool {
    let Some(recurring) = lessons::recurring_pitfall_for_error(root, failure_detail) else {
        return false; // not a true recurrence → stay on the cheap template path
    };
    let sig = recurring.signature.clone();
    // Bounded: at most one reflection ATTEMPT per recurring signature per run.
    // Insert before the consult so even a failed consult never retries this run.
    if !reflected.insert(sig.clone()) {
        return false;
    }
    let (system, user) = lessons::reflection_prompt(&recurring);
    let fork = fork_with_timeout(session).await;
    let consult = ForkConsult::new(fork);
    let reply = consult
        .judge_text("mem-reflect", format!("{system}\n\n{user}"))
        .await;
    consult.end().await;
    let Some(text) = reply.filter(|t| !t.trim().is_empty()) else {
        return false; // offline / no fork / empty reply → leave the template path
    };
    let recorded = lessons::record_pitfall_strategy(root, &sig, &text);
    if recorded {
        events.emit(EngineEvent::Note(
            "[learned] 同类踩坑反复出现，已让底座反思生成一个不同的高层纠错策略并记录，下次自动规避。"
                .to_string(),
        ));
    }
    recorded
}

/// Base-judged memory reconcile at delivery — the evolution half of the learning
/// loop the plain append-sediment (`crate::phases::run_delivery`) leaves undone.
///
/// For each fresh lesson vs. its most-similar priors, ask the brain (read-only
/// fork) whether it ADD / UPDATE / INVALIDATE / NOOPs, then re-sediment with that
/// decision map so the corpus is CURATED instead of purely appended. Ported from
/// the legacy runner's `evolve_memory_at_delivery` (its reconcile pass), driven
/// through the read-only fork seam instead of a main-session turn so it never
/// disturbs the just-finished build's session.
///
/// Bounded ([`MAX_RECONCILE_CALLS`]) + fail-open at every step: no candidates → a
/// no-op (never even forks); a failed/offline fork → every consult returns `None`,
/// no decision is applied, and the plain-append behaviour already in place stands.
pub(crate) async fn reconcile_at_delivery(
    session: &mut dyn BaseSession,
    root: &Path,
    events: &Arc<dyn EventSink>,
) {
    let candidates = lessons::reconcile_candidates(root);
    if candidates.is_empty() {
        return; // nothing fresh to reconcile → no fork, no cost
    }
    // ONE read-only fork drives every bounded consult (each `judge_text` is a fresh
    // turn on the same forked session). A fork that couldn't open routes every
    // consult to `None` → no decisions → the reconcile is a no-op (fail-open).
    let fork = fork_with_timeout(session).await;
    let consult = ForkConsult::new(fork);
    let mut decisions: std::collections::HashMap<
        (String, String, String),
        lessons::ReconcileDecision,
    > = std::collections::HashMap::new();
    for (fresh, similar) in candidates.iter().take(MAX_RECONCILE_CALLS) {
        let (system, user) = lessons::reconcile_prompt(fresh, similar);
        if let Some(reply) = consult
            .judge_text("mem-reconcile", format!("{system}\n\n{user}"))
            .await
            .filter(|t| !t.trim().is_empty())
        {
            let id = (
                fresh.domain.clone(),
                fresh.title.clone(),
                fresh.first_seen.clone(),
            );
            decisions.insert(id, lessons::parse_reconcile_decision(&reply));
        }
    }
    consult.end().await;
    if decisions.is_empty() {
        return; // offline / no confident verdicts → leave the append-only corpus
    }
    let judge = move |fresh: &lessons::Lesson, _similar: &[lessons::Lesson]| {
        let id = (
            fresh.domain.clone(),
            fresh.title.clone(),
            fresh.first_seen.clone(),
        );
        decisions
            .get(&id)
            .copied()
            .unwrap_or(lessons::ReconcileDecision::Noop)
    };
    let _ = lessons::sediment_lessons_with_judge(root, Some(&judge));
    events.emit(EngineEvent::Note(
        "[learned] 交付前整理记忆库：让底座对相似旧教训做了 ADD/UPDATE/INVALIDATE 判定，已合并并淘汰过期条目。"
            .to_string(),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::NullSink;
    use crate::lessons::{
        apply_dev_error_trust, capture_dev_errors, capture_quality_failures, read_raw_lessons,
        relevant_lessons_for_prompt, DEV_ERRORS_FILE, NEUTRAL_TRUST,
    };
    use crate::phases::QualityCheck;
    use std::collections::VecDeque;
    use umadev_runtime::{ApprovalDecision, SessionError, SessionEvent, TurnStatus};

    fn sink() -> Arc<dyn EventSink> {
        Arc::new(NullSink)
    }

    // ── A scripted fake base session whose read-only fork answers each turn with a
    // fixed reply (so a consult gets a deterministic strategy / verdict). The MAIN
    // session is never driven in these unit tests — only forked. `can_fork = false`
    // exercises the fail-open path. ──

    struct ForkBrain {
        reply: String,
        pending: VecDeque<SessionEvent>,
    }
    #[async_trait::async_trait]
    impl BaseSession for ForkBrain {
        async fn send_turn(&mut self, _d: String) -> Result<(), SessionError> {
            // Refill on every turn so multiple sequential consults each get a reply.
            self.pending = [
                SessionEvent::TextDelta(self.reply.clone()),
                SessionEvent::TurnDone {
                    status: TurnStatus::Completed,
                    usage: None,
                },
            ]
            .into_iter()
            .collect();
            Ok(())
        }
        async fn next_event(&mut self) -> Option<SessionEvent> {
            self.pending.pop_front()
        }
        async fn respond(&mut self, _r: &str, _d: ApprovalDecision) -> Result<(), SessionError> {
            Ok(())
        }
        async fn interrupt(&mut self) -> Result<(), SessionError> {
            Ok(())
        }
        async fn end(&mut self) -> Result<(), SessionError> {
            Ok(())
        }
    }

    struct Brain {
        reply: String,
        can_fork: bool,
    }
    impl Brain {
        fn forking(reply: &str) -> Self {
            Self {
                reply: reply.to_string(),
                can_fork: true,
            }
        }
        fn no_fork() -> Self {
            Self {
                reply: String::new(),
                can_fork: false,
            }
        }
    }
    #[async_trait::async_trait]
    impl BaseSession for Brain {
        async fn fork(&mut self) -> Result<Box<dyn BaseSession>, SessionError> {
            if !self.can_fork {
                return Err(SessionError::ForkUnsupported("test".into()));
            }
            Ok(Box::new(ForkBrain {
                reply: self.reply.clone(),
                pending: VecDeque::new(),
            }))
        }
        async fn send_turn(&mut self, _d: String) -> Result<(), SessionError> {
            Ok(())
        }
        async fn next_event(&mut self) -> Option<SessionEvent> {
            None
        }
        async fn respond(&mut self, _r: &str, _d: ApprovalDecision) -> Result<(), SessionError> {
            Ok(())
        }
        async fn interrupt(&mut self) -> Result<(), SessionError> {
            Ok(())
        }
        async fn end(&mut self) -> Result<(), SessionError> {
            Ok(())
        }
    }

    /// One failed quality check → one persisted NON-pitfall (Failure) lesson.
    fn failing_check() -> QualityCheck {
        QualityCheck {
            name: "coverage".to_string(),
            category: "quality".to_string(),
            description: "test".to_string(),
            status: "failed".to_string(),
            score: 20,
            details: "coverage below the bar for the login system".to_string(),
            weight: 2.0,
        }
    }

    // ── Trust: reward on pass, penalise on fail (surfaced non-pitfall lessons) ──

    #[test]
    fn reward_on_pass_lifts_the_recalled_nonpitfall_trust() {
        let tmp = tempfile::TempDir::new().unwrap();
        let req = "做一个登录系统";
        // Seed one non-pitfall lesson, then RECALL it so its identity is snapshotted
        // as "surfaced" — exactly what `drive_build_step`'s top-of-step recall does.
        capture_quality_failures(tmp.path(), &[failing_check()], "demo", req);
        let _ = relevant_lessons_for_prompt(tmp.path(), req); // writes surfaced snapshot

        let trust_of = |t: &Path| {
            read_raw_lessons(t, "quality-failures.jsonl")
                .into_iter()
                .next()
                .map(|l| l.trust())
        };
        let before = trust_of(tmp.path()).unwrap();
        assert!(
            (before - NEUTRAL_TRUST).abs() < f32::EPSILON,
            "seeds at neutral"
        );
        // A clean pass (no recovery) still rewards the recalled lesson's trust.
        reward_on_pass(tmp.path(), &[]);
        assert!(
            trust_of(tmp.path()).unwrap() > before,
            "a passing step must lift the recalled lesson's trust"
        );
    }

    #[test]
    fn penalise_on_fail_sinks_the_recalled_and_the_matching_pitfall() {
        let tmp = tempfile::TempDir::new().unwrap();
        let req = "做一个登录系统";
        capture_quality_failures(tmp.path(), &[failing_check()], "demo", req);
        // A dev-error pitfall whose signature matches the failing evidence below.
        let err = "Error: Cannot find module 'lodash'".to_string();
        capture_dev_errors(tmp.path(), std::slice::from_ref(&err), "demo", req);
        let _ = relevant_lessons_for_prompt(tmp.path(), req); // snapshot surfaced ids

        let qf_trust = |t: &Path| {
            read_raw_lessons(t, "quality-failures.jsonl")
                .into_iter()
                .next()
                .map(|l| l.trust())
                .unwrap()
        };
        let pit_trust = |t: &Path| {
            read_raw_lessons(t, DEV_ERRORS_FILE)
                .into_iter()
                .find(|l| l.signature == "dependency/module-not-found/lodash")
                .map(|l| l.trust())
                .unwrap()
        };
        let qf0 = qf_trust(tmp.path());
        let pit0 = pit_trust(tmp.path());
        penalise_on_fail(tmp.path(), std::slice::from_ref(&err));
        assert!(
            qf_trust(tmp.path()) < qf0,
            "a failing step sinks the recalled non-pitfall lesson's trust"
        );
        assert!(
            pit_trust(tmp.path()) < pit0,
            "a failing step sinks the matching dev-error pitfall's trust"
        );
    }

    #[test]
    fn reward_on_recovery_marks_the_pitfall_resolved() {
        let tmp = tempfile::TempDir::new().unwrap();
        let req = "做一个登录系统";
        let err = "Error: Cannot find module 'lodash'".to_string();
        capture_dev_errors(tmp.path(), std::slice::from_ref(&err), "demo", req);
        // Recovery: reward + resolve keyed on the failing round's evidence.
        reward_on_pass(tmp.path(), std::slice::from_ref(&err));
        let pit = read_raw_lessons(tmp.path(), DEV_ERRORS_FILE)
            .into_iter()
            .find(|l| l.signature == "dependency/module-not-found/lodash")
            .unwrap();
        assert!(
            pit.efficacy.as_ref().is_some_and(|e| e.proven_fix),
            "a recovery marks the recovered pitfall's fix proven"
        );
    }

    // ── Reflection: fires once on a true recurrence, fail-open otherwise ──

    /// Seed a RECURRING pitfall (its recorded fix already failed and it came back)
    /// so `recurring_pitfall_for_error` gates a reflection.
    fn seed_recurring_pitfall(root: &Path) {
        let err = "Error: Cannot find module 'lodash'".to_string();
        // Capture it, then feed a fail signal so it escalates to Recurring: inject
        // (surface) it, then have it fail again after the warning.
        capture_dev_errors(root, std::slice::from_ref(&err), "demo", "需求");
        // Mark it warned-then-recurred via the public injection + capture cycle:
        let _ = relevant_lessons_for_prompt(root, "lodash module");
        capture_dev_errors(root, std::slice::from_ref(&err), "demo", "需求");
        // A fail signal in play keeps its trust honest (not required for gating).
        let _ = apply_dev_error_trust(root, std::slice::from_ref(&err), false);
    }

    #[tokio::test]
    async fn reflection_records_a_strategy_on_a_true_recurrence_and_is_bounded() {
        let tmp = tempfile::TempDir::new().unwrap();
        seed_recurring_pitfall(tmp.path());
        // Precondition: the store gates a reflection (Recurring status).
        let recurring =
            lessons::recurring_pitfall_for_error(tmp.path(), "Error: Cannot find module 'lodash'");
        assert!(
            recurring.is_some(),
            "fixture must be a true recurrence to exercise reflection"
        );

        let strategy = "Pin lodash in package.json and run a clean lockfile install.";
        let mut brain = Brain::forking(strategy);
        let mut reflected: HashSet<String> = HashSet::new();
        let first = reflect_on_recurring_failure(
            &mut brain,
            tmp.path(),
            &sink(),
            "Error: Cannot find module 'lodash'",
            &mut reflected,
        )
        .await;
        assert!(first, "a true recurrence records a reflected strategy");
        let stored = read_raw_lessons(tmp.path(), DEV_ERRORS_FILE)
            .into_iter()
            .find(|l| l.signature == "dependency/module-not-found/lodash")
            .unwrap();
        assert_eq!(
            stored.efficacy.as_ref().unwrap().next_strategy,
            strategy,
            "record_pitfall_strategy is reached and persists the base strategy"
        );

        // Bounded: a SECOND call for the same signature this run is a no-op.
        let second = reflect_on_recurring_failure(
            &mut brain,
            tmp.path(),
            &sink(),
            "Error: Cannot find module 'lodash'",
            &mut reflected,
        )
        .await;
        assert!(
            !second,
            "reflection fires at most once per signature per run"
        );
    }

    #[tokio::test]
    async fn reflection_is_fail_open_when_the_fork_fails() {
        let tmp = tempfile::TempDir::new().unwrap();
        seed_recurring_pitfall(tmp.path());
        let mut brain = Brain::no_fork(); // offline / no fork
        let mut reflected: HashSet<String> = HashSet::new();
        let recorded = reflect_on_recurring_failure(
            &mut brain,
            tmp.path(),
            &sink(),
            "Error: Cannot find module 'lodash'",
            &mut reflected,
        )
        .await;
        assert!(!recorded, "a fork failure records nothing, never panics");
        let stored = read_raw_lessons(tmp.path(), DEV_ERRORS_FILE)
            .into_iter()
            .find(|l| l.signature == "dependency/module-not-found/lodash")
            .unwrap();
        assert!(
            stored
                .efficacy
                .as_ref()
                .is_none_or(|e| e.next_strategy.is_empty()),
            "no strategy is recorded when the consult can't run"
        );
    }

    #[tokio::test]
    async fn reflection_abstains_on_a_non_recurrence() {
        let tmp = tempfile::TempDir::new().unwrap();
        // A first-ever failure (Active, not Recurring) must NOT trigger a consult.
        let err = "Error: Cannot find module 'lodash'".to_string();
        capture_dev_errors(tmp.path(), std::slice::from_ref(&err), "demo", "需求");
        let mut brain = Brain::forking("unused strategy");
        let mut reflected: HashSet<String> = HashSet::new();
        let recorded =
            reflect_on_recurring_failure(&mut brain, tmp.path(), &sink(), &err, &mut reflected)
                .await;
        assert!(
            !recorded,
            "a first failure stays on the cheap template path"
        );
        assert!(reflected.is_empty(), "a non-recurrence never claims a slot");
    }

    // ── Delivery reconcile: fail-open + no-op guards ──

    #[tokio::test]
    async fn reconcile_at_delivery_no_candidates_never_forks() {
        let tmp = tempfile::TempDir::new().unwrap();
        // Empty corpus → no candidates → the reconcile returns before forking.
        let mut brain = Brain::forking("UPDATE");
        reconcile_at_delivery(&mut brain, tmp.path(), &sink()).await;
        // Nothing to assert beyond "did not panic / did not create a corpus".
        assert!(read_raw_lessons(tmp.path(), "quality-failures.jsonl").is_empty());
    }

    #[tokio::test]
    async fn reconcile_at_delivery_is_fail_open_when_the_fork_fails() {
        let tmp = tempfile::TempDir::new().unwrap();
        // Two similar non-pitfall lessons → a real reconcile candidate exists.
        capture_quality_failures(tmp.path(), &[failing_check()], "demo", "登录系统");
        // A distinct-enough second failure that still shares the domain/keywords.
        let mut c2 = failing_check();
        c2.details = "coverage still below the bar for the login flow".to_string();
        capture_quality_failures(tmp.path(), &[c2], "demo", "登录系统的表单");
        let before = read_raw_lessons(tmp.path(), "quality-failures.jsonl");

        // Fork fails (offline) → every consult is None → nothing invalidated.
        let mut brain = Brain::no_fork();
        reconcile_at_delivery(&mut brain, tmp.path(), &sink()).await;
        let after = read_raw_lessons(tmp.path(), "quality-failures.jsonl");
        assert_eq!(
            before.iter().filter(|l| !l.invalidated).count(),
            after.iter().filter(|l| !l.invalidated).count(),
            "an offline fork reconciles nothing (fail-open); no lesson invalidated"
        );
    }
}
