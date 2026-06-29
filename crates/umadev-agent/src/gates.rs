//! Confirmation gates — UD-FLOW-002 / UD-FLOW-003.

use serde::{Deserialize, Serialize};

/// Which gate this represents.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gate {
    /// Before `research` — the worker generated clarifying questions; wait
    /// for the user to answer them before the pipeline continues. The answers
    /// enrich the requirement so research/docs land closer to intent.
    ClarifyGate,
    /// After `docs` phase — wait for explicit user approval of PRD/ARCH/UIUX.
    DocsConfirm,
    /// After `frontend` phase — wait for explicit user approval of preview.
    PreviewConfirm,
}

impl Gate {
    /// Canonical id persisted to `workflow-state.json#active_gate`.
    #[must_use]
    pub const fn id_str(self) -> &'static str {
        match self {
            Self::ClarifyGate => "clarify",
            Self::DocsConfirm => "docs_confirm",
            Self::PreviewConfirm => "preview_confirm",
        }
    }

    /// Inverse of [`Gate::id_str`]: parse a persisted gate id back into the
    /// typed enum. Case-insensitive + whitespace-tolerant; returns `None`
    /// for unknown ids (fail-open). Replaces the ad-hoc string matches the
    /// CLI previously sprinkled across `main.rs`. Mirrors
    /// `umadev_spec::Gate::from_id` so both Gate types stay parseable
    /// from the same persisted strings.
    #[must_use]
    pub fn from_id(id: &str) -> Option<Self> {
        match id.trim().to_ascii_lowercase().as_str() {
            "clarify" => Some(Self::ClarifyGate),
            "docs_confirm" => Some(Self::DocsConfirm),
            "preview_confirm" => Some(Self::PreviewConfirm),
            _ => None,
        }
    }
}

/// What the user did at the gate.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum GateOutcome {
    /// User said `确认 / 通过 / 继续 / lgtm / approve / ...`.
    Approved,
    /// User requested revisions (free-form).
    Revise(String),
    /// User explicitly cancelled the pipeline.
    Cancelled,
}

/// The semantic decision a structured-gate option maps onto. Each maps to the
/// EXISTING gate flow — there is **no new decision machinery**: `Approve` drives
/// the confirm/continue path, `Revise`/`AddMore` drop into the existing
/// free-text revise path (the picker is a nicer front-end to it), and `Cancel`
/// aborts the run. UD-FLOW-002 / UD-FLOW-003.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateDecision {
    /// Approve the gate — the existing confirm/continue path.
    Approve,
    /// Request revisions — drops into the existing free-text revise path.
    Revise,
    /// Supplement / add more — a revise-class follow-up with an "add more" framing.
    AddMore,
    /// Cancel the run.
    Cancel,
}

impl GateDecision {
    /// Stable id (persistence / tests).
    #[must_use]
    pub const fn id_str(self) -> &'static str {
        match self {
            Self::Approve => "approve",
            Self::Revise => "revise",
            Self::AddMore => "add_more",
            Self::Cancel => "cancel",
        }
    }

    /// The i18n key the UI localizes into this option's picker label. Carried as
    /// a key (not a localized string) so the runner can attach a structured
    /// choice without knowing the user's locale — the TUI resolves it at render
    /// time (a non-key string passed through `t()` is returned verbatim, so a
    /// caller may also supply a literal label).
    #[must_use]
    pub const fn label_key(self) -> &'static str {
        match self {
            Self::Approve => "gate.choice.confirm",
            Self::Revise => "gate.choice.revise",
            Self::AddMore => "gate.choice.add_more",
            Self::Cancel => "gate.choice.cancel",
        }
    }
}

/// One labeled option in a structured gate choice (2–4 per choice).
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct GateChoiceOption {
    /// The display label — an i18n KEY (localized by the UI via `t()`) or a
    /// literal string (passed through verbatim). See [`GateDecision::label_key`].
    pub label: String,
    /// Which existing gate decision picking this option drives.
    pub decision: GateDecision,
}

/// A structured choice surfaced when a gate opens: a question + 2–4 labeled
/// options the UI renders as a picker (↑↓ / number keys + Enter). Free-text
/// stays an always-available fallback — the picker never replaces it.
///
/// **Fail-open:** an empty `options` list (or a `None` choice on the gate event)
/// means "no structured choice" → the UI falls back to the existing free-form
/// gate exactly as before.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct GateChoice {
    /// The question shown above the options — an i18n key or a literal (same
    /// resolution rule as [`GateChoiceOption::label`]).
    pub question: String,
    /// The 2–4 options.
    pub options: Vec<GateChoiceOption>,
}

impl GateChoice {
    /// The STANDARD structured choice for a gate, carried as i18n keys so it is
    /// locale-free (the UI localizes at render). Returns `None` for a gate that
    /// has no standard approve/revise choice (the clarify gate collects free-form
    /// answers, not a decision) so the caller falls back to the free-form gate.
    #[must_use]
    pub fn standard(gate: Gate) -> Option<Self> {
        let (question, decisions): (&str, &[GateDecision]) = match gate {
            Gate::DocsConfirm => (
                "gate.choice.docs.question",
                &[
                    GateDecision::Approve,
                    GateDecision::Revise,
                    GateDecision::AddMore,
                ],
            ),
            Gate::PreviewConfirm => (
                "gate.choice.preview.question",
                &[
                    GateDecision::Approve,
                    GateDecision::Revise,
                    GateDecision::AddMore,
                ],
            ),
            // The clarify gate is an answer-collection surface, not an
            // approve/revise decision → no standard picker (free-form, unchanged).
            Gate::ClarifyGate => return None,
        };
        Some(Self {
            question: question.to_string(),
            options: decisions
                .iter()
                .map(|d| GateChoiceOption {
                    label: d.label_key().to_string(),
                    decision: *d,
                })
                .collect(),
        })
    }

    /// Whether this choice has at least one option — the fail-open guard the UI
    /// checks before rendering a picker (an empty choice → free-form gate).
    #[must_use]
    pub fn is_renderable(&self) -> bool {
        !self.options.is_empty()
    }
}

const APPROVAL_TOKENS: &[&str] = &[
    "确认", "通过", "继续", "approved", "approve", "lgtm", "ship it", "ok",
];

/// Classify a free-form user reply into a gate outcome.
///
/// UD-FLOW-002 rules:
/// - exact match against `APPROVAL_TOKENS` (case-insensitive, trimmed) → Approved
/// - "cancel" / "取消" / "重来" → Cancelled
/// - everything else → Revise(text)
#[must_use]
pub fn classify_reply(reply: &str) -> GateOutcome {
    let lower = reply.trim().to_lowercase();
    if lower.is_empty() {
        return GateOutcome::Revise(String::new());
    }
    if APPROVAL_TOKENS
        .iter()
        .any(|t| t.eq_ignore_ascii_case(&lower))
    {
        return GateOutcome::Approved;
    }
    if matches!(lower.as_str(), "cancel" | "取消" | "重来" | "restart") {
        return GateOutcome::Cancelled;
    }
    GateOutcome::Revise(reply.trim().to_string())
}

/// Heuristic: does this base reply CLAIM it made code changes? Used by the director
/// build loop to decide whether an honesty/QC read is even warranted (a pure
/// chat/plan answer that touched no files has nothing to QC), and — at the app
/// boundary — to anchor a "claimed-but-no-diff" warning. Deliberately broad and
/// bilingual; a false positive only adds an advisory check, never blocks anything
/// (the source-present floor is itself fail-open). Lives here, the agent crate's
/// reply-classification home, so the TUI's public wrapper has ONE source of truth.
#[must_use]
pub fn claims_code_changes(text: &str) -> bool {
    // English change verbs. Matched as substrings (`t.contains(k)`), so a root
    // covers its inflections: `build` → building/built (kept explicit for clarity),
    // `wrote` → rewrote, `set up` → "set up the route". The build-loop directive
    // literally says "build it", so a base answering "I built …/wrote …/scaffolded
    // …/wired up …" MUST register as a code claim — otherwise the honesty QC + the
    // source-present hard-gate are skipped over a possibly-hallucinated "done".
    const EN: &[&str] = &[
        "refactor",
        "added",
        "changed",
        "edited",
        "created",
        "updated",
        "modified",
        "removed",
        "deleted",
        "implemented",
        "renamed",
        "rewrote",
        "replaced",
        "inserted",
        // The most common "I did the work" verbs — aligned with the /run
        // directive's own "build it" wording (P1-3).
        "build", // building / rebuilt / "I'll build" → also "built" (substring)
        "built",
        "wrote",
        "wired",
        "scaffolded",
        "generated",
        "coded",
        "developed",
        "set up",
    ];
    // Chinese change verbs (no case folding needed).
    const ZH: &[&str] = &[
        "重构",
        "新增",
        "删除",
        "修改",
        "实现",
        "修复",
        "改了",
        "改动",
        "更新",
        "增加",
        "移除",
        "重命名",
        "替换",
        "已添加",
        "已修改",
        "写入",
        "创建",
    ];
    let t = text.to_lowercase();
    if EN.iter().any(|k| t.contains(k)) {
        return true;
    }
    ZH.iter().any(|k| text.contains(k))
}

/// Heuristic: does this base reply show it ALREADY ran the project's build/test
/// THIS turn and it PASSED? Used by the director's auto-QC to skip UmaDev's own
/// *duplicate* full build/test read (an `npm install` + build can be minutes) when
/// the base's body — which holds the tools — already ran it green inside its turn.
///
/// **Conservative by contract (no correctness regression):** this returns `true`
/// ONLY when the reply both (a) names a build/test/lint run AND (b) reports it
/// passed, AND (c) shows NO failure signal. Anything ambiguous — no mention, a
/// vague "done", or any whiff of a failure/error — returns `false`, so UmaDev
/// falls back to running its OWN objective read (the prior behaviour). A false
/// negative just re-runs a check we could have trusted (slower, still correct); we
/// never skip on a false positive that hides a real failure. Bilingual; matched as
/// lowercased substrings.
#[must_use]
pub fn base_ran_build_test_clean(text: &str) -> bool {
    let t = text.to_lowercase();

    // (c) Any failure signal vetoes the skip — if the base mentions a failing
    // build/test anywhere in its reply, UmaDev must run its own read to see it.
    const FAILURE: &[&str] = &[
        "fail",
        "failing",
        "failed",
        "error",
        "errored",
        "broke",
        "broken",
        "did not pass",
        "didn't pass",
        "does not pass",
        "doesn't pass",
        "not passing",
        "red",
        "exit code 1",
        "exit 1",
        "panic",
        "测试失败",
        "构建失败",
        "编译失败",
        "报错",
        "未通过",
        "没通过",
        "不通过",
    ];
    if FAILURE.iter().any(|k| t.contains(k)) {
        return false;
    }

    // (a) names a build/test/lint run AND (b) reports it passed/green. Require a
    // PASS phrase that co-locates the run with a success word so a bare "looks good"
    // (no actual run) does NOT qualify.
    const PASS_EN: &[&str] = &[
        "tests pass",
        "tests passing",
        "all tests pass",
        "all tests passing",
        "test suite passes",
        "tests are passing",
        "tests green",
        "build passes",
        "build passed",
        "build succeeded",
        "build succeeds",
        "builds successfully",
        "built successfully",
        "compiles cleanly",
        "compiled successfully",
        "lint passes",
        "lint passed",
        "lint clean",
        "checks pass",
        "all checks pass",
        "ci passes",
        "test and build pass",
        "build and test pass",
    ];
    const PASS_ZH: &[&str] = &[
        "测试通过",
        "测试全部通过",
        "测试全绿",
        "构建通过",
        "构建成功",
        "编译通过",
        "编译成功",
        "检查通过",
        "全部通过",
        "校验通过",
    ];
    PASS_EN.iter().any(|k| t.contains(k)) || PASS_ZH.iter().any(|k| text.contains(k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claims_code_changes_detects_change_verbs_bilingually() {
        assert!(claims_code_changes(
            "I created app.ts and updated the route"
        ));
        assert!(claims_code_changes("已实现登录表单，新增了失败路径测试"));
        // A pure chat / plan answer with no change verb → no claim.
        assert!(!claims_code_changes(
            "Here's how I'd approach it conceptually — nothing touched."
        ));
        assert!(!claims_code_changes("这是我的思路，我先和你确认一下方案"));
    }

    #[test]
    fn claims_code_changes_recognises_build_verbs() {
        // P1-3: the /run directive says "build it", so the base's most common "done"
        // phrasings ("I built …", "wrote …", "scaffolded …", "wired up …", "set up …")
        // MUST count as a code claim, or the honesty QC + source-present hard-gate are
        // skipped over a possibly-hallucinated build.
        for claim in [
            "I built the login page and wrote the tests. All done.",
            "Built the app end to end.",
            "Scaffolded the project and wired up the routes.",
            "Generated the API client and coded the form handler.",
            "Developed the dashboard and set up the auth flow.",
            "I'll build it now and report back.",
        ] {
            assert!(claims_code_changes(claim), "should claim a build: {claim}");
        }
        // Still no false positive on a pure plan / discussion (no build verb).
        assert!(!claims_code_changes(
            "Let me first discuss the trade-offs of each option before touching anything."
        ));
    }

    #[test]
    fn base_ran_build_test_clean_detects_a_passed_run_bilingually() {
        for claim in [
            "I implemented it and ran the tests — all tests pass.",
            "Built the app; the build succeeded and lint passes.",
            "Ran cargo test, the test suite passes cleanly.",
            "构建成功,测试全部通过,可以交付了。",
            "我跑了一遍,编译通过、测试通过。",
        ] {
            assert!(
                base_ran_build_test_clean(claim),
                "should read as a clean self-run: {claim}"
            );
        }
    }

    #[test]
    fn base_ran_build_test_clean_is_false_on_failure_or_ambiguity() {
        // A failure signal ANYWHERE vetoes the skip — UmaDev must run its own read.
        for txt in [
            "Tests pass for the model layer but the integration test failed.",
            "Build succeeded but lint is failing on two files.",
            "构建成功,但有一个测试失败了。",
            "编译通过,不过跑测试时报错了。",
        ] {
            assert!(
                !base_ran_build_test_clean(txt),
                "a failure signal must veto the skip: {txt}"
            );
        }
        // Ambiguous "done" with no explicit passed-run → no skip (conservative).
        for txt in [
            "Done — implemented the login form and the route.",
            "Looks good, the page renders.",
            "实现完了,你看一下。",
            "",
        ] {
            assert!(
                !base_ran_build_test_clean(txt),
                "ambiguous reply must NOT trigger the skip: {txt}"
            );
        }
    }

    #[test]
    fn approval_tokens_match() {
        for t in [
            "确认", "通过", "继续", "approved", "Approve", "LGTM", "ship it",
        ] {
            assert!(matches!(classify_reply(t), GateOutcome::Approved), "{t}");
        }
    }

    #[test]
    fn cancel_tokens_match() {
        for t in ["cancel", "取消", "重来", "restart"] {
            assert!(matches!(classify_reply(t), GateOutcome::Cancelled), "{t}");
        }
    }

    #[test]
    fn revise_default() {
        let out = classify_reply("把图标库换成 lucide");
        if let GateOutcome::Revise(text) = out {
            assert!(text.contains("lucide"));
        } else {
            panic!("expected Revise");
        }
    }

    #[test]
    fn empty_reply_is_revise_with_empty_text() {
        assert!(matches!(classify_reply(""), GateOutcome::Revise(s) if s.is_empty()));
    }

    #[test]
    fn standard_choice_is_present_for_confirm_gates_and_absent_for_clarify() {
        // docs/preview confirm gates carry a 3-option approve/revise/add-more
        // choice (locale-free i18n keys); the clarify gate has no standard picker.
        for gate in [Gate::DocsConfirm, Gate::PreviewConfirm] {
            let c = GateChoice::standard(gate).expect("confirm gate has a choice");
            assert!(c.is_renderable());
            assert_eq!(c.options.len(), 3);
            assert_eq!(c.options[0].decision, GateDecision::Approve);
            assert_eq!(c.options[1].decision, GateDecision::Revise);
            assert_eq!(c.options[2].decision, GateDecision::AddMore);
            // Labels are carried as i18n KEYS, not localized strings.
            assert_eq!(c.options[0].label, "gate.choice.confirm");
        }
        assert!(GateChoice::standard(Gate::ClarifyGate).is_none());
    }

    #[test]
    fn empty_choice_is_not_renderable_fail_open() {
        let empty = GateChoice {
            question: "q".to_string(),
            options: vec![],
        };
        assert!(!empty.is_renderable());
    }

    #[test]
    fn gate_decision_ids_and_label_keys_are_stable() {
        for d in [
            GateDecision::Approve,
            GateDecision::Revise,
            GateDecision::AddMore,
            GateDecision::Cancel,
        ] {
            assert!(!d.id_str().is_empty());
            assert!(d.label_key().starts_with("gate.choice."));
        }
    }

    #[test]
    fn gate_from_id_roundtrips_and_is_case_insensitive() {
        for g in [Gate::ClarifyGate, Gate::DocsConfirm, Gate::PreviewConfirm] {
            assert_eq!(Gate::from_id(g.id_str()), Some(g));
        }
        assert_eq!(Gate::from_id("Docs_Confirm"), Some(Gate::DocsConfirm));
        assert_eq!(
            Gate::from_id("  preview_confirm  "),
            Some(Gate::PreviewConfirm)
        );
        assert_eq!(Gate::from_id("nope"), None);
        assert_eq!(Gate::from_id(""), None);
    }
}
