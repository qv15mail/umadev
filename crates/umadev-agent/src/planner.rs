//! Dynamic phase planner — the "dynamic agent" layer.
//!
//! UmaDev's canonical pipeline is the full nine-phase chain
//! ([`umadev_spec::PHASE_CHAIN`]). Forcing EVERY task through all nine phases is
//! exactly the rigidity the SOTA agent literature warns against: a fixed
//! workflow is the right call for *well-defined* work, but a one-line bug fix
//! does not need similar-product research + three core documents + two human
//! confirmation gates + a delivery proof-pack. That rigidity is what makes a
//! pipeline feel "weak" on small or narrow tasks.
//!
//! This module classifies the requirement and tailors WHICH phases run, while
//! (1) preserving the canonical ORDER, and (2) keeping the confirm gates
//! whenever their guarded phase actually runs and the task is heavyweight
//! enough to warrant a human checkpoint.
//!
//! The classifier is deterministic (bilingual zh/en keyword + intent
//! heuristics) so it needs no model call and is fully unit-tested. A
//! brain-assisted refinement can layer on top later without changing this
//! contract. **Fail-open:** an unrecognised requirement falls back to the full
//! [`TaskKind::Greenfield`] pipeline — the planner never produces *fewer*
//! phases than the safe default by accident.

use umadev_spec::Phase;

/// The kind of work a requirement describes. Inferred deterministically by
/// [`classify`]; drives the tailored [`PhasePlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    /// A new product / app from scratch — the full pipeline.
    Greenfield,
    /// Frontend / UI only — skip the backend phase.
    FrontendOnly,
    /// Backend / API / data only — skip the frontend phase + its preview gate.
    BackendOnly,
    /// A small bug fix — skip research / docs / gates; go straight to a lean
    /// implement + quality pass.
    Bugfix,
    /// A refactor / cleanup — the same lean path as a bug fix.
    Refactor,
    /// Docs / research / planning only — no code phases.
    DocsOnly,
    /// A trivial change — a one-line tweak, a style nudge, a tiny script. The
    /// LIGHTEST path of all: a lean clarify-spec → implement → verify, with no
    /// research / docs / two confirm gates / delivery proof-pack. This is the
    /// answer to "the full nine phases are too heavy for a small task": the
    /// planner can auto-suggest it (see [`classify`]), and `umadev quick`
    /// forces it regardless of classification.
    Light,
}

impl TaskKind {
    /// Stable identifier for logs and workflow state.
    #[must_use]
    pub fn id(self) -> &'static str {
        match self {
            TaskKind::Greenfield => "greenfield",
            TaskKind::FrontendOnly => "frontend_only",
            TaskKind::BackendOnly => "backend_only",
            TaskKind::Bugfix => "bugfix",
            TaskKind::Refactor => "refactor",
            TaskKind::DocsOnly => "docs_only",
            TaskKind::Light => "light",
        }
    }

    /// The ordered phases for this kind — always an order-preserving subset of
    /// [`umadev_spec::PHASE_CHAIN`]. A confirm gate is included only when the
    /// phase it guards runs AND the task is heavyweight enough to warrant a
    /// human checkpoint (the lean bug-fix / refactor paths skip the gates).
    #[must_use]
    pub fn phases(self) -> Vec<Phase> {
        use Phase::{
            Backend, Delivery, Docs, DocsConfirm, Frontend, PreviewConfirm, Quality, Research, Spec,
        };
        match self {
            TaskKind::Greenfield => vec![
                Research,
                Docs,
                DocsConfirm,
                Spec,
                Frontend,
                PreviewConfirm,
                Backend,
                Quality,
                Delivery,
            ],
            TaskKind::FrontendOnly => vec![
                Research,
                Docs,
                DocsConfirm,
                Spec,
                Frontend,
                PreviewConfirm,
                Quality,
                Delivery,
            ],
            TaskKind::BackendOnly => {
                vec![
                    Research,
                    Docs,
                    DocsConfirm,
                    Spec,
                    Backend,
                    Quality,
                    Delivery,
                ]
            }
            // Lean fast paths: no research/docs ceremony, no gates, no delivery
            // proof-pack — just plan the change, implement it, gate on quality.
            TaskKind::Bugfix | TaskKind::Refactor => vec![Spec, Frontend, Backend, Quality],
            TaskKind::DocsOnly => vec![Research, Docs, DocsConfirm],
            // The lightest path — for a trivial change the full nine phases are
            // pure overhead. A lean clarify-lite `Spec` → implement
            // (`Frontend` + `Backend`, whichever the change touches) → `Quality`
            // verify. No research, no three core docs, no two confirm gates, no
            // delivery proof-pack. Governance still applies on every write.
            TaskKind::Light => vec![Spec, Frontend, Backend, Quality],
        }
    }

    /// Whether this is the lightweight fast track (trivial work). The runner
    /// drives a [`TaskKind::Light`] plan through [`crate::AgentRunner::run_light`]
    /// in a single shot rather than the gate-anchored three-block walk.
    #[must_use]
    pub fn is_light(self) -> bool {
        matches!(self, TaskKind::Light)
    }
}

/// A tailored, ordered plan of phases for a specific requirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhasePlan {
    /// The inferred task kind.
    pub kind: TaskKind,
    /// Ordered phases to execute — a subset of [`umadev_spec::PHASE_CHAIN`].
    pub phases: Vec<Phase>,
    /// Human-readable reason, shown to the user for transparency.
    pub rationale: String,
}

impl PhasePlan {
    /// Whether `phase` is part of this plan.
    #[must_use]
    pub fn includes(&self, phase: Phase) -> bool {
        self.phases.contains(&phase)
    }

    /// Phases from the canonical chain that this plan skips.
    #[must_use]
    pub fn skipped(&self) -> Vec<Phase> {
        umadev_spec::PHASE_CHAIN
            .iter()
            .copied()
            .filter(|p| !self.phases.contains(p))
            .collect()
    }
}

/// Classify `requirement` and produce a tailored [`PhasePlan`]. Deterministic,
/// bilingual (zh / en), fail-open to [`TaskKind::Greenfield`].
#[must_use]
pub fn plan(requirement: &str) -> PhasePlan {
    let kind = classify(requirement);
    let mut phases = kind.phases();
    // A simple (Light) build rarely needs BOTH a frontend and a backend. Trim the
    // surface the requirement never mentions so a pure static page doesn't pay for
    // an empty backend phase (~25% of a lean run was a do-nothing Backend turn),
    // and a small script doesn't pay for an empty frontend phase. Only trims when
    // the requirement is clearly one-sided; an ambiguous Light build keeps both.
    // Never touches a gated plan (Greenfield/FrontendOnly/BackendOnly already pick
    // their phases) — only the otherwise-fixed lean Light list.
    if kind == TaskKind::Light {
        let q = requirement.to_lowercase();
        let fe = mentions_frontend(&q);
        let be = mentions_backend(&q);
        if fe && !be {
            phases.retain(|p| *p != Phase::Backend);
        } else if be && !fe {
            phases.retain(|p| *p != Phase::Frontend);
        }
    }
    PhasePlan {
        kind,
        phases,
        rationale: rationale_for(kind),
    }
}

/// `true` when the requirement names a frontend / UI surface (used to trim a lean
/// plan's surplus phase). Distinctive tokens only — shared with [`classify`].
fn mentions_frontend(q: &str) -> bool {
    const FE: &[&str] = &[
        "前端",
        "界面",
        "页面",
        "单页",
        "样式",
        "组件",
        "布局",
        "静态页",
        "落地页",
        "frontend",
        "tailwind",
        "react",
        "vue",
        "html",
        "css",
        " ui",
        "single page",
        "static page",
        "landing page",
    ];
    FE.iter().any(|n| q.contains(n))
}

/// `true` when the requirement names a backend / server / data surface — shared
/// with [`classify`].
fn mentions_backend(q: &str) -> bool {
    const BE: &[&str] = &[
        "后端",
        "接口",
        "数据库",
        "服务端",
        "数据表",
        "鉴权",
        "脚本",
        "命令行",
        "backend",
        "graphql",
        "fastapi",
        "express",
        "微服务",
        "script",
        "cli",
        "api",
        "server",
    ];
    BE.iter().any(|n| q.contains(n))
}

/// Deterministic intent classification. Order matters: the narrowest intents
/// (bug fix, refactor, docs-only) are matched before the broad frontend /
/// backend split, which is matched before the greenfield default. Needles are
/// chosen to be distinctive (Chinese terms + multi-character English tokens) to
/// avoid substring false positives.
#[must_use]
pub fn classify(requirement: &str) -> TaskKind {
    let q = requirement.to_lowercase();
    let has = |needles: &[&str]| needles.iter().any(|n| q.contains(n));

    // 1. Bug fix — the narrowest, fastest path.
    if has(&[
        "修复",
        "修一下",
        "修个",
        "报错",
        "bug",
        "fixbug",
        "fix the",
        "fix a",
        "crash",
        "不工作",
        "失效",
        "坏了",
        "崩溃",
        "报错",
        "闪退",
        "hotfix",
    ]) {
        return TaskKind::Bugfix;
    }
    // 2. Refactor / cleanup.
    if has(&[
        "重构",
        "refactor",
        "整理代码",
        "优化代码",
        "clean up",
        "cleanup",
        "拆分模块",
        "tidy up",
        "代码结构",
    ]) {
        return TaskKind::Refactor;
    }
    // 3. Docs / research / planning only.
    if has(&[
        "写文档",
        "出文档",
        "只做调研",
        "research only",
        "只要文档",
        "写个方案",
        "写 prd",
        "写prd",
        "需求文档",
        "技术方案",
        "调研报告",
        "docs only",
    ]) {
        return TaskKind::DocsOnly;
    }
    // 4. Trivial change — the lightest of all. A one-line tweak, a tiny style
    //    nudge, a small script: the full nine phases are pure overhead. Needles
    //    are deliberately NARROW (explicit "small/tiny/trivial" markers + tiny
    //    artefacts) so a real feature never silently downgrades to Light; an
    //    ambiguous request still falls through to the heavyweight default.
    if has(&[
        "小改",
        "小修改",
        "微调",
        "改个文案",
        "改文案",
        "改个文字",
        "改个颜色",
        "改颜色",
        "改个样式",
        "改一行",
        "加个日志",
        "小脚本",
        "写个脚本",
        "small tweak",
        "tiny tweak",
        "minor tweak",
        "quick change",
        "trivial change",
        "one-liner",
        "one liner",
        "small script",
        "tiny script",
        "tweak the copy",
        "change the text",
        "rename ",
        "bump the version",
        "typo",
    ]) {
        return TaskKind::Light;
    }

    // 5. Frontend vs backend split (distinctive tokens only).
    let frontend = has(&[
        "前端",
        "界面",
        "页面",
        "样式",
        "组件",
        "布局",
        "frontend",
        "tailwind",
        "react",
        "vue",
        "落地页",
    ]);
    let backend = has(&[
        "后端",
        "接口",
        "数据库",
        "服务端",
        "数据表",
        "鉴权",
        "backend",
        "graphql",
        "fastapi",
        "express",
        "微服务",
    ]);

    // 5.5. A genuinely SIMPLE small build — the lightweight path. "做一个简单的
    //      待办清单单页应用,纯前端" should NOT pay for research + three full core
    //      documents + two confirm gates; it should go spec → implement → verify.
    //      This is a SCOPED downgrade, guarded on BOTH sides:
    //      (a) it fires only when an explicit "this is small" signal is present
    //          (简单/单页/demo/小工具/静态页/single page/…), AND
    //      (b) it NEVER fires when a heavyweight-product signal is present
    //          (登录/auth/数据库/支付/SaaS/平台/多页/multi-module/…) — those keep
    //          their full pipeline because the research + docs + gates ARE the value.
    //      Fail-open: with no explicit-simple signal OR any heavy signal present,
    //      we fall through to the existing FrontendOnly / BackendOnly / Greenfield
    //      defaults, so a real product is never mis-downgraded.
    if is_simple_build(&q) && !has_heavy_signal(&q) {
        return TaskKind::Light;
    }

    if frontend && !backend {
        return TaskKind::FrontendOnly;
    }
    if backend && !frontend {
        return TaskKind::BackendOnly;
    }

    // 6. Default — a full product build.
    TaskKind::Greenfield
}

/// Whether `q` (already lowercased) carries an EXPLICIT "this is a small build"
/// signal — the positive half of the lightweight-build heuristic. These are
/// markers a user adds to say "keep it small": an explicit smallness word
/// (简单 / 小 / mini / demo / toy / quick), a single-page / static-page shape, or
/// a "tiny tool / small app" framing. A bare "做一个待办应用" carries NO such
/// signal, so it stays on the full pipeline (a real product) — only when the
/// user actually scoped it down do we consider the light path.
fn is_simple_build(q: &str) -> bool {
    let has = |needles: &[&str]| needles.iter().any(|n| q.contains(n));
    has(&[
        // Explicit smallness (zh).
        "简单的",
        "简易",
        "简单小",
        "极简",
        "单页应用",
        "单页面",
        "单个页面",
        "一个小",
        "小工具",
        "小demo",
        "小 demo",
        "小项目",
        "静态页",
        "静态网页",
        "纯静态",
        "练手",
        "小练习",
        "玩具项目",
        // Explicit smallness (en).
        "simple ",
        "single page",
        "single-page",
        "one page",
        "one-page",
        "static page",
        "static html",
        "tiny app",
        "small app",
        "little app",
        "mini app",
        "small tool",
        "tiny tool",
        "demo app",
        "toy app",
        "toy project",
        "just a simple",
        "quick demo",
        "quick prototype",
        "basic html",
    ])
}

/// Whether `q` (already lowercased) carries a HEAVYWEIGHT-product signal — the
/// negative half of the lightweight-build heuristic. Any of these means "this is
/// a real product even if phrased casually" (auth, persistence, payments,
/// multi-module / multi-page surface, an explicit commercial / production /
/// platform framing), so the light path is VETOED and the full pipeline stands.
/// This is the guardrail that keeps "做一个带邮箱登录的 SaaS 数据分析仪表盘" on
/// `Greenfield` no matter how the smallness words read.
fn has_heavy_signal(q: &str) -> bool {
    let has = |needles: &[&str]| needles.iter().any(|n| q.contains(n));
    has(&[
        // Persistence / accounts / payments / auth — anything that needs a backend
        // surface, a data model, or a security posture is NOT a light build.
        "登录",
        "注册",
        "账号",
        "账户",
        "鉴权",
        "权限",
        "数据库",
        "持久化",
        "后端",
        "服务端",
        "接口",
        "api",
        "支付",
        "订单",
        "结算",
        "上传",
        "实时",
        "推送",
        "auth",
        "login",
        "signup",
        "sign up",
        "sign-in",
        "oauth",
        "database",
        "backend",
        "server-side",
        "payment",
        "checkout",
        "stripe",
        // Commercial / scale / multi-surface framing — the research + docs + gates
        // are the product value here, so never downgrade.
        "商业级",
        "可上线",
        "上线",
        "生产级",
        "saas",
        "平台",
        "多页",
        "多模块",
        "后台管理",
        "管理系统",
        "仪表盘",
        "dashboard",
        "platform",
        "production",
        "multi-page",
        "multi page",
        "multi-module",
        "enterprise",
        "scalable",
    ])
}

/// Derive the governance [`ProjectContext`](umadev_governance::ProjectContext)
/// for a run — the signal that lets the rule engine skip server/security-surface
/// rules (CSP, clickjacking, structured logging, HSTS, HTTPS-redirect, CSRF,
/// token-context RNG) for a project that has no such surface.
///
/// This adds **no** model call. It reads three sources the run already has, and
/// is **conservative and fail-open**: it only returns the lenient
/// [`ProjectContext::static_frontend`](umadev_governance::ProjectContext::static_frontend)
/// when EVERY signal agrees the project is a static, frontend-only build with no
/// server/data/auth surface. ANY signal of a backend → the strict
/// [`ProjectContext::unknown`](umadev_governance::ProjectContext::unknown)
/// (assume a surface might exist), so a real backend/auth project is never
/// under-governed by accident.
///
/// Signals:
/// 1. **Task kind** — only [`TaskKind::FrontendOnly`] / [`TaskKind::Light`] are
///    candidates. [`TaskKind::Greenfield`] / [`TaskKind::BackendOnly`] and the
///    lean bug-fix/refactor paths always stay strict.
/// 2. **Requirement text** — any heavyweight signal (auth / database / payment /
///    api / backend / dashboard / platform — see [`has_heavy_signal`]) vetoes
///    the lenient context.
/// 3. **Architecture doc + produced source** — if the architecture doc declares
///    an API surface or data model, or any produced source file carries server
///    evidence (a listener, an API route, a backend framework import, auth/token
///    handling), the project HAS a surface → strict.
#[must_use]
pub fn derive_project_context(
    requirement: &str,
    project_root: &std::path::Path,
    slug: &str,
) -> umadev_governance::ProjectContext {
    let strict = umadev_governance::ProjectContext::unknown();

    // Signal 1: task kind must be a frontend-only / light build.
    let kind = classify(requirement);
    if !matches!(kind, TaskKind::FrontendOnly | TaskKind::Light) {
        return strict;
    }

    // Signal 2: no heavyweight (auth/db/payment/api/backend/…) requirement words.
    if has_heavy_signal(&requirement.to_lowercase()) {
        return strict;
    }

    // Signal 3a: the architecture doc must NOT declare a server/data surface.
    // Read it best-effort; absent/empty doc is fine for a light build (it often
    // has no architecture doc at all), so absence is NOT a veto here — the other
    // signals already established "frontend-only, no heavy words".
    let arch = std::fs::read_to_string(project_root.join(format!("output/{slug}-architecture.md")))
        .unwrap_or_default()
        .to_lowercase();
    if doc_declares_server_surface(&arch) {
        return strict;
    }

    // Signal 3b: no produced source file may carry server evidence. If even one
    // does, the project grew a backend → strict. (Reuses the governance kernel's
    // per-file server-evidence detector by scanning each file through it.)
    if any_source_has_server_surface(project_root) {
        return strict;
    }

    umadev_governance::ProjectContext::static_frontend()
}

/// Whether an architecture doc (already lowercased) declares a real server /
/// data / auth surface — an API section/table, a data model, an auth scheme, or
/// a backend framework. Conservative: any such marker means "has a surface".
fn doc_declares_server_surface(arch_lower: &str) -> bool {
    if arch_lower.trim().is_empty() {
        return false;
    }
    [
        "## api",
        "api surface",
        "## data model",
        "数据模型",
        "数据库",
        "database",
        "endpoint",
        "/api/",
        "auth",
        "鉴权",
        "session",
        "会话",
        "jwt",
        "token",
        "backend",
        "服务端",
        "后端",
    ]
    .iter()
    .any(|needle| arch_lower.contains(needle))
}

/// Whether ANY produced source file carries its own server / security surface,
/// per the governance kernel's per-file detector. Scans through the public
/// `scan_content`-adjacent path: a file is a server surface iff governing it as
/// a static frontend would NOT skip the surface rules. Bounded + fail-open.
fn any_source_has_server_surface(project_root: &std::path::Path) -> bool {
    // A file that the static-frontend context would STILL govern at full
    // strictness is, by definition, one that carries server evidence. We probe
    // that by checking a server-surface rule under the lenient context: if it
    // would fire, the file has a surface. Cheaper + equivalent: rely on the
    // detector indirectly by scanning a benign HTML-shaped probe per file is not
    // possible from here, so we read each file and look for the same evidence the
    // kernel uses, via the public lenient/strict differential.
    let files = crate::acceptance::source_files(project_root);
    for f in files.iter().take(400) {
        let Ok(content) = std::fs::read_to_string(f) else {
            continue;
        };
        let rel = f
            .strip_prefix(project_root)
            .unwrap_or(f)
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        if umadev_governance::file_has_server_surface(&rel, &content) {
            return true;
        }
    }
    false
}

/// Parse a user-supplied phase name into a typed [`Phase`], for `umadev redo
/// <phase>` / `/redo <phase>`. Case-insensitive and whitespace-tolerant, and
/// accepts the common friendly aliases a user is likely to type (`fe`/`ui` for
/// frontend, `be`/`api` for backend, `qa` for quality, etc.) in addition to the
/// canonical [`Phase::id`] strings. Returns `None` for anything unrecognised so
/// the caller can show the valid set — fail-open, never panics.
#[must_use]
pub fn phase_from_id(name: &str) -> Option<Phase> {
    match name
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-'], "_")
        .as_str()
    {
        "research" => Some(Phase::Research),
        "docs" | "doc" | "documents" => Some(Phase::Docs),
        "docs_confirm" | "docsconfirm" => Some(Phase::DocsConfirm),
        "spec" | "plan" => Some(Phase::Spec),
        "frontend" | "fe" | "ui" | "front" => Some(Phase::Frontend),
        "preview_confirm" | "previewconfirm" | "preview" => Some(Phase::PreviewConfirm),
        "backend" | "be" | "api" | "back" => Some(Phase::Backend),
        "quality" | "qa" | "quality_gate" => Some(Phase::Quality),
        "delivery" | "deliver" | "release" => Some(Phase::Delivery),
        _ => None,
    }
}

/// The phase names a user can pass to `redo`, in canonical chain order — used
/// to build a friendly "valid phases: …" error when [`phase_from_id`] rejects.
#[must_use]
pub fn redoable_phase_ids() -> Vec<&'static str> {
    umadev_spec::PHASE_CHAIN.iter().map(|p| p.id()).collect()
}

/// An **advisory prior** the director may read — Wave 3 of
/// `docs/AGENT_WIELDS_BASE_ARCHITECTURE.md` §3 (planner DEMOTED from "decides the
/// fixed phase list" to "an advisory prior").
///
/// In the Wave 3 model the director (thinking through the base) decides on the
/// spot how to get a goal done — who to bring in, in what order, how much process.
/// The planner no longer DICTATES that route; it only offers a *suggestion* the
/// director can take or ignore. This function renders the classifier's read of a
/// requirement as a SHORT, explicitly-advisory hint: "this looks like X; a goal
/// like this usually benefits from Y — but YOU judge what THIS one needs." It
/// deliberately uses non-binding language and never names a forced phase chain.
///
/// Deterministic + fail-open: it reuses [`classify`] (no model call), so an
/// unrecognised requirement falls back to the [`TaskKind::Greenfield`] advisory —
/// the most thorough suggestion, which the director is still free to trim.
#[must_use]
pub fn advisory_prior(requirement: &str) -> String {
    let kind = classify(requirement);
    // The non-binding "you might consider" framing per inferred kind. Each line
    // describes the SHAPE of work a goal like this usually warrants, then hands the
    // call back to the director — it is a prior, not an order.
    let hint = match kind {
        TaskKind::Greenfield => {
            "This reads like a full product build from scratch. A goal this size \
             usually benefits from framing the requirements (PM) and an approach \
             (architect) before splitting frontend / backend, then a QA + security \
             pass — but scale that to what THIS goal actually needs."
        }
        TaskKind::FrontendOnly => {
            "This reads like frontend / UI work. It usually centres on the design \
             system + the components, with a quick contract check that the calls \
             line up — little or no backend. Bring in a designer / frontend seat as \
             you see fit."
        }
        TaskKind::BackendOnly => {
            "This reads like backend / API / data work. It usually centres on the \
             routes + data model + validation, with the frontend contract kept in \
             mind — little or no UI. Bring in an architect / backend seat as you \
             see fit."
        }
        TaskKind::Bugfix => {
            "This reads like a focused bug fix. It usually wants a minimal, targeted \
             change and a check that the original failure path is actually gone — no \
             research / docs ceremony. Often one engineer end to end."
        }
        TaskKind::Refactor => {
            "This reads like a refactor / cleanup. It usually wants a structural \
             change that keeps behaviour identical — lean on the existing tests to \
             prove nothing broke. No new docs ceremony."
        }
        TaskKind::DocsOnly => {
            "This reads like docs / research / planning, not code. It usually wants \
             the writing done well and a checkpoint with the user before any \
             implementation."
        }
        TaskKind::Light => {
            "This reads like a small, scoped change. It usually wants you to just do \
             it directly — implement + a quick verify — with none of the research / \
             docs / gate ceremony a full product needs."
        }
    };
    format!(
        "[advisory — a prior you may use or ignore] {hint} The plan is YOURS to \
         decide; this is only a read of the goal, not a route you must follow."
    )
}

/// Build a plan that FORCES the lightweight fast track regardless of how the
/// requirement classifies. This is what `umadev quick` / `/quick` use: the user
/// has explicitly asked for the lean path, so we skip classification and pin
/// [`TaskKind::Light`]. The deterministic classifier still drives the default
/// `umadev run` path, where a trivial requirement is auto-suggested into Light
/// but the user can override by running the full pipeline instead.
#[must_use]
pub fn plan_light(requirement: &str) -> PhasePlan {
    let _ = requirement; // reserved for future per-requirement light tailoring
    let kind = TaskKind::Light;
    PhasePlan {
        kind,
        phases: kind.phases(),
        rationale: rationale_for(kind),
    }
}

/// The subset of `plan`'s skipped phases that are safe to skip TODAY within the
/// runner's gate-anchored three-block structure with zero downstream risk:
/// `Delivery` — the final phase, which runs AFTER the quality gate, so skipping
/// it (a lean bug-fix / refactor needs no deploy proof-pack) cannot affect any
/// gate or quality check. `Research` / `Backend` / `Frontend` and the lean
/// gate-skipping paths interact with later phases (the quality gate filters by
/// check name, not phase) and so are deferred to the full plan-driven runner
/// walk — the planner never claims a skip it does not actually perform.
#[must_use]
pub fn gate_safe_skips(plan: &PhasePlan) -> Vec<Phase> {
    plan.skipped()
        .into_iter()
        .filter(|p| matches!(p, Phase::Delivery))
        .collect()
}

/// One-line rationale per kind (localised at the call site is overkill here;
/// the runner surfaces this verbatim as a transparency note).
// Honest, advisory descriptions of how the task was classified. They describe
// the FOCUS, not a literal phase-skip — today the runner only auto-skips the
// Delivery phase (via gate_safe_skips); the rest of the pipeline still runs and
// pauses at its gates, so these must not promise skips that don't happen.
fn rationale_for(kind: TaskKind) -> String {
    match kind {
        TaskKind::Greenfield => "全新产品 — 走完整九阶段管线".to_string(),
        TaskKind::FrontendOnly => "偏前端 — 重点在前端实现与预览确认".to_string(),
        TaskKind::BackendOnly => "偏后端 — 重点在后端实现与前后端契约对齐".to_string(),
        TaskKind::Bugfix => "小修复 — 聚焦定位与最小改动,文档从简".to_string(),
        TaskKind::Refactor => "重构 — 聚焦结构调整、保持行为不变".to_string(),
        TaskKind::DocsOnly => "文档/调研为主 — 在文档确认门停下,由你决定是否继续实现".to_string(),
        TaskKind::Light => {
            "轻量档 — 极简流程:澄清简版→实现→验证,跳过调研/三文档/两道确认门/交付物料包".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use umadev_spec::Phase;

    #[test]
    fn classifies_bugfix() {
        assert_eq!(classify("修复登录页的 bug"), TaskKind::Bugfix);
        assert_eq!(classify("登录一直报错,帮我修一下"), TaskKind::Bugfix);
        assert_eq!(classify("the app crashes on submit"), TaskKind::Bugfix);
    }

    #[test]
    fn classifies_refactor() {
        assert_eq!(classify("重构 app.rs 拆分模块"), TaskKind::Refactor);
        assert_eq!(classify("refactor the auth module"), TaskKind::Refactor);
    }

    #[test]
    fn classifies_docs_only() {
        assert_eq!(classify("先写需求文档"), TaskKind::DocsOnly);
        assert_eq!(classify("写个方案给我看看"), TaskKind::DocsOnly);
    }

    #[test]
    fn classifies_frontend_and_backend() {
        assert_eq!(classify("做一个前端落地页"), TaskKind::FrontendOnly);
        assert_eq!(classify("build a React component"), TaskKind::FrontendOnly);
        assert_eq!(classify("写一个后端接口"), TaskKind::BackendOnly);
        assert_eq!(
            classify("a GraphQL backend with auth"),
            TaskKind::BackendOnly
        );
    }

    #[test]
    fn frontend_and_backend_together_is_greenfield() {
        // Mentions both sides → a full build, not a one-sided task.
        assert_eq!(
            classify("做一个带前端和后端的电商网站"),
            TaskKind::Greenfield
        );
    }

    #[test]
    fn defaults_to_greenfield() {
        assert_eq!(classify("做一个待办事项应用"), TaskKind::Greenfield);
        assert_eq!(classify("帮我做个 SaaS 产品"), TaskKind::Greenfield);
    }

    #[test]
    fn greenfield_runs_the_full_chain() {
        let p = plan("做一个电商平台");
        assert_eq!(p.kind, TaskKind::Greenfield);
        assert_eq!(p.phases, umadev_spec::PHASE_CHAIN.to_vec());
        assert!(p.skipped().is_empty());
    }

    #[test]
    fn bugfix_skips_research_docs_and_gates() {
        let p = plan("修复一个报错");
        assert_eq!(p.kind, TaskKind::Bugfix);
        assert!(!p.includes(Phase::Research));
        assert!(!p.includes(Phase::Docs));
        assert!(!p.includes(Phase::DocsConfirm));
        assert!(!p.includes(Phase::PreviewConfirm));
        assert!(!p.includes(Phase::Delivery));
        // …but still plans + quality-gates the change.
        assert!(p.includes(Phase::Spec));
        assert!(p.includes(Phase::Quality));
        let skipped = p.skipped();
        assert!(skipped.contains(&Phase::Research));
    }

    #[test]
    fn frontend_only_skips_backend_keeps_preview_gate() {
        let p = plan("做一个前端页面");
        assert!(p.includes(Phase::Frontend));
        assert!(p.includes(Phase::PreviewConfirm));
        assert!(!p.includes(Phase::Backend));
    }

    #[test]
    fn backend_only_skips_frontend_and_preview_gate() {
        let p = plan("写一个后端 graphql 接口");
        assert!(p.includes(Phase::Backend));
        assert!(!p.includes(Phase::Frontend));
        assert!(!p.includes(Phase::PreviewConfirm));
        // Docs gate still applies (it's a heavyweight build).
        assert!(p.includes(Phase::DocsConfirm));
    }

    #[test]
    fn gate_safe_skips_is_delivery_only_today() {
        // A bug fix plan skips many phases, but only Delivery is wired as a
        // zero-risk skip today (it runs after the quality gate).
        let p = plan("修复一个报错");
        assert_eq!(gate_safe_skips(&p), vec![Phase::Delivery]);
        // Greenfield skips nothing.
        assert!(gate_safe_skips(&plan("做一个电商网站")).is_empty());
    }

    #[test]
    fn classifies_trivial_as_light() {
        assert_eq!(classify("帮我改个文案"), TaskKind::Light);
        assert_eq!(classify("这里微调一下间距"), TaskKind::Light);
        assert_eq!(classify("写个脚本批量重命名文件"), TaskKind::Light);
        assert_eq!(
            classify("a small tweak to the header copy"),
            TaskKind::Light
        );
        assert_eq!(classify("just a typo in the readme"), TaskKind::Light);
        // Ordering note: a request phrased as a "fix" matches the narrower
        // Bugfix lean path FIRST — both are lean, so this is fine.
        assert_eq!(classify("fix a typo in the readme"), TaskKind::Bugfix);
    }

    #[test]
    fn non_trivial_does_not_downgrade_to_light() {
        // A real feature / product must NOT silently become Light.
        assert_eq!(classify("做一个待办事项应用"), TaskKind::Greenfield);
        assert_eq!(classify("做一个前端落地页"), TaskKind::FrontendOnly);
        assert_eq!(classify("写一个后端接口"), TaskKind::BackendOnly);
    }

    #[test]
    fn simple_single_page_build_is_light() {
        // The dogfood case: an explicitly-simple, single-page, pure-frontend build
        // must take the lightweight path (spec -> implement -> verify), NOT the
        // full research + three-docs + gate pipeline that took 24 minutes.
        assert_eq!(
            classify("做一个简单的待办清单单页应用,纯前端 HTML+CSS+JS,支持添加删除"),
            TaskKind::Light
        );
        assert_eq!(classify("做一个简单的计算器单页面"), TaskKind::Light);
        assert_eq!(classify("写一个静态页展示个人简介"), TaskKind::Light);
        assert_eq!(
            classify("a simple single-page todo app, pure HTML/CSS/JS"),
            TaskKind::Light
        );
        assert_eq!(
            classify("build a tiny demo app, just a static page"),
            TaskKind::Light
        );
        assert_eq!(classify("做一个小工具帮我格式化 JSON"), TaskKind::Light);
    }

    #[test]
    fn heavy_signal_vetoes_the_light_downgrade() {
        // The boundary the planner must NOT cross: a smallness word does NOT make a
        // real product lean. Any auth / database / payment / SaaS / platform / dashboard
        // signal keeps the FULL pipeline — research + three docs + gates are its value.
        assert_eq!(
            classify("做一个带邮箱登录的 SaaS 数据分析仪表盘,要能上线"),
            TaskKind::Greenfield
        );
        // "简单的" present but a database + login veto the light path → it routes to
        // a heavyweight bucket (never the lean Light path).
        assert_ne!(
            classify("做一个简单的博客平台,带登录和数据库"),
            TaskKind::Light
        );
        // "single page" present but it's a real dashboard with auth + a backend api
        // → stays heavyweight (never the lean Light path).
        assert_ne!(
            classify("a single page dashboard with user login and a backend api"),
            TaskKind::Light
        );
        // A simple-sounding but payment-bearing build is not light (it routes to a
        // heavyweight path — the exact one depends on the surface words, but it must
        // never be the lean Light path).
        assert_ne!(
            classify("做一个简单的小商城,支持下单和支付"),
            TaskKind::Light
        );
    }

    #[test]
    fn light_classification_skips_research_docs_and_gates() {
        // The whole point of the downgrade: a Light-classified simple build plans
        // straight to spec -> implement -> verify, with NO research / docs / gates.
        let p = plan("做一个简单的待办清单单页应用,纯前端 HTML+CSS+JS,支持添加删除");
        assert_eq!(p.kind, TaskKind::Light);
        assert!(!p.includes(Phase::Research));
        assert!(!p.includes(Phase::Docs));
        assert!(!p.includes(Phase::DocsConfirm));
        assert!(!p.includes(Phase::PreviewConfirm));
        assert!(!p.includes(Phase::Delivery));
        // …but still plans the implementation + the quality (hard) gate.
        assert!(p.includes(Phase::Spec));
        assert!(p.includes(Phase::Frontend));
        assert!(p.includes(Phase::Quality));
    }

    #[test]
    fn classification_boundary_simple_vs_complex_samples() {
        // A contrast battery: each LEFT sample is genuinely small (Light); each
        // RIGHT sample is a real product (full pipeline). The classifier must split
        // them correctly — simple stays light, complex stays full.
        let light = [
            "做一个简单的倒计时单页应用",
            "写个静态网页放我的简历",
            "a small tool to convert markdown to html",
            "做一个极简的番茄钟,纯前端",
        ];
        for r in light {
            assert_eq!(classify(r), TaskKind::Light, "should be Light: {r}");
        }
        let heavy = [
            "做一个在线协作文档平台,带账号和实时同步",
            "构建一个电商网站,商品、购物车、支付、后台管理",
            "build a multi-page SaaS with authentication and a postgres database",
            "做一个带后端接口和数据库的待办应用,支持多用户登录",
        ];
        for r in heavy {
            assert_ne!(classify(r), TaskKind::Light, "should NOT be Light: {r}");
        }
    }

    #[test]
    fn bugfix_classification_for_broken_button() {
        // The task's required bug-fix boundary sample.
        assert_eq!(classify("修复登录按钮点击没反应"), TaskKind::Bugfix);
    }

    #[test]
    fn light_plan_is_the_lean_subset_no_gates() {
        // Whether reached by classification or forced via `plan_light`, a Light
        // plan skips research/docs/both gates/delivery and keeps spec+quality.
        for p in [plan("帮我改个文案"), plan_light("anything at all")] {
            assert_eq!(p.kind, TaskKind::Light);
            assert!(p.kind.is_light());
            assert!(p.includes(Phase::Spec));
            assert!(p.includes(Phase::Quality));
            assert!(!p.includes(Phase::Research));
            assert!(!p.includes(Phase::Docs));
            assert!(!p.includes(Phase::DocsConfirm));
            assert!(!p.includes(Phase::PreviewConfirm));
            assert!(!p.includes(Phase::Delivery));
        }
    }

    #[test]
    fn advisory_prior_is_explicitly_non_binding_per_kind() {
        // The Wave 3 demotion: the planner returns an ADVISORY prior the director
        // may ignore — never a forced phase chain. Every kind's hint must read as a
        // suggestion (the "advisory" / "may use or ignore" framing) and hand the
        // plan back to the director.
        for req in [
            "做一个电商平台",          // Greenfield
            "做一个前端落地页",        // FrontendOnly
            "写一个后端 graphql 接口", // BackendOnly
            "修复一个报错",            // Bugfix
            "重构 app.rs 拆分模块",    // Refactor
            "先写需求文档",            // DocsOnly
            "帮我改个文案",            // Light
        ] {
            let a = advisory_prior(req);
            let lower = a.to_lowercase();
            assert!(
                lower.contains("advisory"),
                "the prior is explicitly advisory for `{req}`: {a}"
            );
            assert!(
                lower.contains("yours to decide") || lower.contains("not a route you must"),
                "the prior hands the plan back to the director for `{req}`: {a}"
            );
            // It must NOT prescribe the literal fixed phase chain.
            assert!(
                !lower.contains("research -> docs") && !a.contains("研究→文档"),
                "the prior must not name a forced phase chain for `{req}`: {a}"
            );
        }
    }

    #[test]
    fn advisory_prior_reflects_the_classification() {
        // A greenfield prior mentions the full-build shape; a light prior mentions
        // doing it directly — so the director's read matches the goal's size.
        assert!(advisory_prior("做一个电商平台")
            .to_lowercase()
            .contains("full product"));
        assert!(advisory_prior("帮我改个文案")
            .to_lowercase()
            .contains("just do it"));
    }

    #[test]
    fn phase_from_id_parses_canonical_and_aliases() {
        assert_eq!(phase_from_id("frontend"), Some(Phase::Frontend));
        assert_eq!(phase_from_id("  FE "), Some(Phase::Frontend));
        assert_eq!(phase_from_id("backend"), Some(Phase::Backend));
        assert_eq!(phase_from_id("api"), Some(Phase::Backend));
        assert_eq!(phase_from_id("QA"), Some(Phase::Quality));
        assert_eq!(
            phase_from_id("preview-confirm"),
            Some(Phase::PreviewConfirm)
        );
        assert_eq!(phase_from_id("plan"), Some(Phase::Spec));
        // Every canonical id round-trips.
        for p in umadev_spec::PHASE_CHAIN {
            assert_eq!(phase_from_id(p.id()), Some(*p), "{}", p.id());
        }
        assert_eq!(phase_from_id("nonsense"), None);
        assert_eq!(phase_from_id(""), None);
    }

    #[test]
    fn plan_light_forces_light_for_any_requirement() {
        // `plan_light` ignores classification — even a greenfield ask is pinned
        // to Light when the user explicitly chose the fast track.
        let p = plan_light("做一个完整的电商平台");
        assert_eq!(p.kind, TaskKind::Light);
    }

    #[test]
    fn every_plan_preserves_canonical_order() {
        for req in [
            "做一个电商网站",
            "做个前端页面",
            "写后端接口",
            "修复 bug",
            "重构代码",
            "写需求文档",
            "改个文案",
        ] {
            let p = plan(req);
            // The plan's phases appear in the same relative order as PHASE_CHAIN.
            let chain: Vec<Phase> = umadev_spec::PHASE_CHAIN.to_vec();
            let mut last = None;
            for ph in &p.phases {
                let idx = chain.iter().position(|c| c == ph).unwrap();
                if let Some(prev) = last {
                    assert!(idx > prev, "phase {ph:?} out of canonical order in {req}");
                }
                last = Some(idx);
            }
        }
    }

    // ---- derive_project_context ----------------------------------------

    #[test]
    fn context_lenient_for_simple_static_frontend() {
        // An explicitly-simple, pure-frontend build with no backend artifacts →
        // the lenient static-frontend context (surface rules skipped).
        let tmp = tempfile::TempDir::new().unwrap();
        let ctx = derive_project_context(
            "做一个简单的待办清单单页应用,纯前端 HTML+CSS+JS",
            tmp.path(),
            "todo",
        );
        assert!(
            ctx.static_frontend_only,
            "a proven static frontend should get the lenient context"
        );
    }

    #[test]
    fn context_strict_for_greenfield_product() {
        // A full product (Greenfield) always stays strict.
        let tmp = tempfile::TempDir::new().unwrap();
        let ctx = derive_project_context("做一个电商平台", tmp.path(), "shop");
        assert!(
            !ctx.static_frontend_only,
            "a greenfield product must stay strict"
        );
    }

    #[test]
    fn context_strict_when_requirement_has_heavy_signal() {
        // "简单的" (simple) present but a login + database veto the lenient path —
        // even if it classified frontend-ish, the heavy signal keeps it strict.
        let tmp = tempfile::TempDir::new().unwrap();
        let ctx =
            derive_project_context("做一个简单的前端页面,带用户登录和数据库", tmp.path(), "app");
        assert!(
            !ctx.static_frontend_only,
            "auth/database words must keep governance strict"
        );
    }

    #[test]
    fn context_strict_when_arch_doc_declares_a_surface() {
        // A simple-frontend requirement, but the architecture doc grew an API
        // surface → strict (the doc proves a backend exists).
        let tmp = tempfile::TempDir::new().unwrap();
        let out = tmp.path().join("output");
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(
            out.join("todo-architecture.md"),
            "# Architecture\n\n## API surface\n\n| Method | Path |\n|---|---|\n",
        )
        .unwrap();
        let ctx = derive_project_context("做一个简单的待办清单单页应用,纯前端", tmp.path(), "todo");
        assert!(
            !ctx.static_frontend_only,
            "an architecture doc with an API surface proves a backend → strict"
        );
    }

    #[test]
    fn context_strict_when_a_source_file_has_server_evidence() {
        // A simple-frontend requirement, but a produced source file boots a
        // server → strict (the project grew a backend).
        let tmp = tempfile::TempDir::new().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        let listen = format!("{}.{}(3000)", "app", "listen");
        std::fs::write(
            src.join("server.ts"),
            format!("const app = express(); {listen};"),
        )
        .unwrap();
        let ctx = derive_project_context("做一个简单的待办清单单页应用,纯前端", tmp.path(), "todo");
        assert!(
            !ctx.static_frontend_only,
            "a produced server file proves a backend → strict"
        );
    }

    #[test]
    fn doc_surface_detection() {
        assert!(doc_declares_server_surface("## api surface\n| GET | /x |"));
        assert!(doc_declares_server_surface("uses a postgres database"));
        assert!(doc_declares_server_surface("需要鉴权"));
        assert!(!doc_declares_server_surface(""));
        assert!(!doc_declares_server_surface(
            "just a color palette and a few buttons"
        ));
    }

    #[test]
    fn lean_frontend_build_skips_the_empty_backend_phase() {
        // A simple pure-frontend page is Light, but should not pay for a
        // do-nothing Backend phase (~25% of the run was an empty backend turn).
        let p = plan("做一个简单的番茄钟计时器单页应用,纯前端,开始/暂停/重置");
        assert_eq!(p.kind, TaskKind::Light);
        assert!(
            p.includes(Phase::Frontend),
            "a frontend page keeps Frontend"
        );
        assert!(
            !p.includes(Phase::Backend),
            "a pure frontend page drops Backend"
        );
        assert!(p.includes(Phase::Spec) && p.includes(Phase::Quality));
    }

    #[test]
    fn lean_script_build_skips_the_empty_frontend_phase() {
        // The mirror: a small backend/script build drops the empty Frontend phase.
        let p = plan("写一个简单的脚本,读取 csv 文件统计行数");
        assert_eq!(p.kind, TaskKind::Light);
        assert!(!p.includes(Phase::Frontend), "a script drops Frontend");
        assert!(p.includes(Phase::Backend), "a script keeps Backend");
    }

    #[test]
    fn lean_ambiguous_build_keeps_both_surfaces() {
        // No clear one-sidedness → keep both (conservative, never strands work).
        let p = plan("做一个简单的小工具");
        assert_eq!(p.kind, TaskKind::Light);
        assert!(p.includes(Phase::Frontend) && p.includes(Phase::Backend));
    }
}
