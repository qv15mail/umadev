//! Project-level configuration overrides via `.umadevrc`.
//!
//! Users can create a `.umadevrc` file (TOML) in the project root to
//! customize UmaDev behavior per-project without modifying the global
//! config.
//!
//! ```toml
//! # .umadevrc
//! [quality]
//! threshold = 85              # override quality gate pass threshold
//! skip_checks = ["dark_mode"] # skip specific quality checks
//!
//! [pipeline]
//! skip_phases = ["research"]  # skip research if you already did it
//! max_review_rounds = 2       # limit review→fix cycles
//!
//! [experts]
//! custom_knowledge = "team-standards/" # extra knowledge directory
//!
//! [knowledge]
//! enabled = true                       # use structured BM25 retrieval (default)
//! engine = "bm25"                      # "bm25" (offline) or "hybrid" (+ OpenAI embeddings)
//! top_k = 6                            # chunks injected per phase
//!
//! [model]
//! provider = "deepseek"                # use this named provider (defined in
//!                                       # ~/.umadev/config.toml) instead of
//!                                       # the global default. Empty = disable
//!                                       # any custom provider for this project.
//!
//! [codex]
//! sandbox_mode = "workspace-write"     # codex launch sandbox: read-only |
//!                                       # workspace-write (default, safe) |
//!                                       # danger-full-access. The default
//!                                       # blocks local dev servers (npm start
//!                                       # for React / Electron) and git
//!                                       # commits; set danger-full-access to
//!                                       # allow them (high-risk — you accept
//!                                       # the risk to your system environment).
//! ```

use std::path::Path;

use serde::Deserialize;

/// Project-level overrides from `.umadevrc`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProjectConfig {
    /// Quality gate overrides.
    #[serde(default)]
    pub quality: QualityConfig,
    /// Pipeline behavior overrides.
    #[serde(default)]
    pub pipeline: PipelineConfig,
    /// Expert knowledge overrides.
    #[serde(default)]
    pub experts: ExpertsConfig,
    /// Knowledge-base / RAG retrieval overrides.
    #[serde(default)]
    pub knowledge: KnowledgeConfig,
    /// Custom-model (API provider) overrides.
    #[serde(default)]
    pub model: ModelConfig,
    /// Codex base launch-sandbox overrides.
    #[serde(default)]
    pub codex: CodexConfig,
}

/// Quality gate customization.
#[derive(Debug, Clone, Deserialize)]
pub struct QualityConfig {
    /// Minimum score to pass (default 90).
    #[serde(default = "default_threshold")]
    pub threshold: u32,
    /// Check names to skip (e.g. `dark_mode`).
    #[serde(default)]
    pub skip_checks: Vec<String>,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            threshold: default_threshold(),
            skip_checks: Vec::new(),
        }
    }
}

fn default_threshold() -> u32 {
    90
}

/// Pipeline behavior customization.
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineConfig {
    /// Phases to skip (e.g. `research` if you already did it).
    #[serde(default)]
    pub skip_phases: Vec<String>,
    /// Max review→fix rounds per document (default 3).
    #[serde(default = "default_review_rounds")]
    pub max_review_rounds: usize,
    /// Strict spec-coverage enforcement (default `false`). When `true`, the
    /// spec phase's FR→task coverage check BLOCKS the pipeline (pausing at
    /// `spec`) if any PRD functional requirement has no covering task, instead
    /// of merely emitting an advisory note. Default `false` keeps the existing
    /// warn-only behaviour so a partial breakdown never silently halts a run.
    /// Overridable per run by the `UMADEV_STRICT_COVERAGE=1` environment flag.
    #[serde(default)]
    pub strict_coverage: bool,
    /// Auto-approve all pipeline gates without waiting for user input
    /// (default `true`). UmaDev's pipeline is designed to run fully
    /// autonomously — like Claude Code's `/goal` mode. The gates
    /// (`docs_confirm`, `preview_confirm`) still fire as checkpoints
    /// in the event stream and status bar, but they don't block
    /// execution. Set to `false` to restore manual gate approval.
    #[serde(default = "default_auto_approve")]
    pub auto_approve_gates: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            skip_phases: Vec::new(),
            max_review_rounds: default_review_rounds(),
            strict_coverage: false,
            auto_approve_gates: default_auto_approve(),
        }
    }
}

fn default_auto_approve() -> bool {
    true
}

fn default_review_rounds() -> usize {
    3
}

/// Expert knowledge customization.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ExpertsConfig {
    /// Additional knowledge directory (relative to project root).
    /// Files in this dir are injected alongside built-in expert knowledge.
    #[serde(default)]
    pub custom_knowledge: Option<String>,
}

/// Custom-model (API provider) overrides.
///
/// A project can pin which named provider (defined in the user's
/// `~/.umadev/config.toml`) runs this project's pipeline. This lets one
/// machine use different models for different projects — e.g. a cheap model
/// for throwaway experiments, a strong one for the production repo.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelConfig {
    /// Name of the provider to use for this project, overriding the global
    /// `default_provider`. An empty string means "explicitly use no custom
    /// provider" (fall back to host CLI / offline). `None` (field absent)
    /// means "no opinion — use the global setting."
    #[serde(default)]
    pub provider: Option<String>,
}

/// Codex base launch-sandbox customization (`.umadevrc` `[codex]` section).
///
/// Codex launches its workspace in one of three sandbox tiers. UmaDev keeps the
/// safe `workspace-write` baseline by default (write the project dir, but no
/// network and no arbitrary system access) — behaviour is UNCHANGED when this
/// section is absent. A full-stack project that must boot a local dev server
/// (`npm start` for React / Electron) or run `git commit` needs the relaxed
/// `danger-full-access` tier; advanced users opt in explicitly and accept the
/// risk to their system environment.
#[derive(Debug, Clone, Deserialize)]
pub struct CodexConfig {
    /// One of `read-only` / `workspace-write` (default) / `danger-full-access`.
    /// An unknown / garbage value fails open to `workspace-write` (never an
    /// error), so a typo can never silently widen the sandbox.
    #[serde(default = "default_codex_sandbox")]
    pub sandbox_mode: String,
}

impl Default for CodexConfig {
    fn default() -> Self {
        Self {
            sandbox_mode: default_codex_sandbox(),
        }
    }
}

impl CodexConfig {
    /// Resolve the configured string to the typed [`CodexSandbox`], failing open
    /// to [`CodexSandbox::WorkspaceWrite`] on any unrecognised value.
    #[must_use]
    pub fn resolved_sandbox(&self) -> CodexSandbox {
        CodexSandbox::parse_fail_open(&self.sandbox_mode)
    }
}

fn default_codex_sandbox() -> String {
    CodexSandbox::WorkspaceWrite.as_codex_arg().to_string()
}

/// The three Codex launch sandbox tiers, matching codex's own `--sandbox` enum
/// (and the JSON-RPC `sandbox` param).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodexSandbox {
    /// Read-only: the base can read the workspace but write nothing.
    ReadOnly,
    /// Write the workspace only — codex's default and UmaDev's safe baseline.
    #[default]
    WorkspaceWrite,
    /// No sandbox: full filesystem + network + process access. Required to boot
    /// local dev servers / run `git`; high-risk, opt-in only.
    DangerFullAccess,
}

impl CodexSandbox {
    /// Parse a config / env string fail-open. Recognises codex's canonical
    /// kebab-case ids (and is lenient about case / `_`↔`-`); anything else —
    /// including an empty or garbage value — resolves to [`Self::WorkspaceWrite`]
    /// so a typo can never silently widen the sandbox.
    #[must_use]
    pub fn parse_fail_open(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "read-only" | "readonly" => Self::ReadOnly,
            "danger-full-access" | "danger-full" | "full-access" => Self::DangerFullAccess,
            // "workspace-write" + every unrecognised value → the safe default.
            _ => Self::WorkspaceWrite,
        }
    }

    /// The canonical codex `--sandbox` / JSON-RPC `sandbox` value.
    #[must_use]
    pub fn as_codex_arg(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::DangerFullAccess => "danger-full-access",
        }
    }

    /// `true` for the high-risk full-access tier — drives the loud startup
    /// warning and the forced non-interactive approval pairing.
    #[must_use]
    pub fn is_high_risk(self) -> bool {
        matches!(self, Self::DangerFullAccess)
    }
}

/// Knowledge-base (RAG) retrieval customization.
///
/// Controls the [`umadev_knowledge`] retrieval engine that replaces the
/// legacy "keyword-sort the folder" approach. Default uses BM25 (offline,
/// zero-dependency); setting `engine = "hybrid"` additionally enables the
/// optional OpenAI embeddings layer when `OPENAI_EMBED_KEY` is present.
#[derive(Debug, Clone, Deserialize)]
pub struct KnowledgeConfig {
    /// Whether structured retrieval is enabled (default `true`). Set
    /// `false` to fall back to the legacy keyword-scoring path.
    #[serde(default = "default_knowledge_enabled")]
    pub enabled: bool,
    /// Retrieval engine: `"bm25"` (default) or `"hybrid"`.
    #[serde(default = "default_knowledge_engine")]
    pub engine: String,
    /// How many knowledge chunks to inject per phase (default 6).
    #[serde(default = "default_knowledge_top_k")]
    pub top_k: usize,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            enabled: default_knowledge_enabled(),
            engine: default_knowledge_engine(),
            top_k: default_knowledge_top_k(),
        }
    }
}

fn default_knowledge_enabled() -> bool {
    true
}

fn default_knowledge_engine() -> String {
    // Hybrid (BM25 + vector RRF fusion) is the commercial default; it degrades
    // to pure BM25 automatically when no embedding backend is reachable, so it
    // is always safe. Set `engine = "bm25"` to force offline lexical-only.
    "hybrid".to_string()
}

fn default_knowledge_top_k() -> usize {
    6
}

/// Read `.umadevrc` from the project root. Returns `Default` if missing
/// or malformed (fail-soft, same as UserConfig).
#[must_use]
pub fn load_project_config(project_root: &Path) -> ProjectConfig {
    let path = project_root.join(".umadevrc");
    let Ok(body) = std::fs::read_to_string(&path) else {
        return ProjectConfig::default();
    };
    let mut cfg: ProjectConfig = toml::from_str(&body).unwrap_or_default();
    // Validate the knowledge engine: only "bm25" and "hybrid" are legal.
    // Unknown values (e.g. "quantum") silently fall back to "bm25" so a
    // typo never breaks retrieval.
    if cfg.knowledge.engine != "bm25" && cfg.knowledge.engine != "hybrid" {
        if !cfg.knowledge.engine.is_empty() {
            tracing::warn!(
                "knowledge.engine = {:?} is not bm25 or hybrid — falling back to bm25",
                cfg.knowledge.engine
            );
        }
        cfg.knowledge.engine = "bm25".to_string();
    }
    // Clamp quality threshold and top_k to sensible bounds.
    cfg.quality.threshold = cfg.quality.threshold.min(100);
    cfg.knowledge.top_k = cfg.knowledge.top_k.clamp(1, 50);
    // Normalise the codex sandbox to a canonical kebab id; an unrecognised
    // value fails open to the safe `workspace-write` baseline (never widens the
    // sandbox). Mirrors the knowledge.engine fail-open above.
    let raw_sandbox = cfg.codex.sandbox_mode.clone();
    let resolved_sandbox = CodexSandbox::parse_fail_open(&raw_sandbox);
    let normalized_sandbox = raw_sandbox.trim().to_ascii_lowercase().replace('_', "-");
    if resolved_sandbox == CodexSandbox::WorkspaceWrite
        && !normalized_sandbox.is_empty()
        && normalized_sandbox != "workspace-write"
    {
        tracing::warn!(
            "codex.sandbox_mode = {:?} is not read-only/workspace-write/danger-full-access \
             — falling back to workspace-write",
            raw_sandbox
        );
    }
    cfg.codex.sandbox_mode = resolved_sandbox.as_codex_arg().to_string();
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn invalid_engine_falls_back_to_bm25() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".umadevrc"),
            "[knowledge]\nengine = \"quantum\"\ntop_k = 999\n",
        )
        .unwrap();
        let cfg = load_project_config(tmp.path());
        assert_eq!(cfg.knowledge.engine, "bm25", "quantum must fall back");
        assert_eq!(cfg.knowledge.top_k, 50, "top_k must clamp to 50");
    }

    #[test]
    fn valid_hybrid_engine_preserved() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".umadevrc"),
            "[knowledge]\nengine = \"hybrid\"\ntop_k = 8\n",
        )
        .unwrap();
        let cfg = load_project_config(tmp.path());
        assert_eq!(cfg.knowledge.engine, "hybrid");
        assert_eq!(cfg.knowledge.top_k, 8);
    }

    #[test]
    fn threshold_clamped_to_100() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".umadevrc"), "[quality]\nthreshold = 999\n").unwrap();
        let cfg = load_project_config(tmp.path());
        assert_eq!(cfg.quality.threshold, 100);
    }

    #[test]
    fn codex_sandbox_unset_defaults_to_workspace_write() {
        let tmp = TempDir::new().unwrap();
        // No `.umadevrc` at all → default, behaviour UNCHANGED.
        let cfg = load_project_config(tmp.path());
        assert_eq!(
            cfg.codex.resolved_sandbox(),
            CodexSandbox::WorkspaceWrite,
            "missing config must keep the safe baseline"
        );
        // A `.umadevrc` that omits the [codex] section → still the default.
        std::fs::write(tmp.path().join(".umadevrc"), "[quality]\nthreshold = 80\n").unwrap();
        let cfg = load_project_config(tmp.path());
        assert_eq!(cfg.codex.resolved_sandbox(), CodexSandbox::WorkspaceWrite);
        assert_eq!(cfg.codex.sandbox_mode, "workspace-write");
    }

    #[test]
    fn codex_sandbox_danger_full_access_parses() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".umadevrc"),
            "[codex]\nsandbox_mode = \"danger-full-access\"\n",
        )
        .unwrap();
        let cfg = load_project_config(tmp.path());
        assert_eq!(cfg.codex.resolved_sandbox(), CodexSandbox::DangerFullAccess);
        assert!(cfg.codex.resolved_sandbox().is_high_risk());
        assert_eq!(cfg.codex.sandbox_mode, "danger-full-access");
    }

    #[test]
    fn codex_sandbox_read_only_parses() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".umadevrc"),
            "[codex]\nsandbox_mode = \"read-only\"\n",
        )
        .unwrap();
        let cfg = load_project_config(tmp.path());
        assert_eq!(cfg.codex.resolved_sandbox(), CodexSandbox::ReadOnly);
        assert!(!cfg.codex.resolved_sandbox().is_high_risk());
    }

    #[test]
    fn codex_sandbox_garbage_falls_back_to_workspace_write() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".umadevrc"),
            "[codex]\nsandbox_mode = \"yolo-root\"\n",
        )
        .unwrap();
        let cfg = load_project_config(tmp.path());
        assert_eq!(
            cfg.codex.resolved_sandbox(),
            CodexSandbox::WorkspaceWrite,
            "a garbage value must fail open to the safe baseline, never error"
        );
        // The stored string is normalised so downstream consumers see the canonical id.
        assert_eq!(cfg.codex.sandbox_mode, "workspace-write");
    }

    #[test]
    fn codex_sandbox_parse_fail_open_is_lenient() {
        assert_eq!(
            CodexSandbox::parse_fail_open("DANGER_FULL_ACCESS"),
            CodexSandbox::DangerFullAccess
        );
        assert_eq!(
            CodexSandbox::parse_fail_open("  Read-Only  "),
            CodexSandbox::ReadOnly
        );
        assert_eq!(
            CodexSandbox::parse_fail_open(""),
            CodexSandbox::WorkspaceWrite
        );
        // Round-trip: every tier maps to codex's canonical kebab arg.
        assert_eq!(CodexSandbox::ReadOnly.as_codex_arg(), "read-only");
        assert_eq!(
            CodexSandbox::WorkspaceWrite.as_codex_arg(),
            "workspace-write"
        );
        assert_eq!(
            CodexSandbox::DangerFullAccess.as_codex_arg(),
            "danger-full-access"
        );
    }

    #[test]
    fn default_config_has_sane_values() {
        let cfg = ProjectConfig::default();
        assert_eq!(cfg.quality.threshold, 90);
        assert_eq!(cfg.pipeline.max_review_rounds, 3);
        assert!(cfg.pipeline.skip_phases.is_empty());
        // Knowledge defaults: enabled, hybrid engine (degrades to BM25 with no
        // embedding key), top_k 6.
        assert!(cfg.knowledge.enabled);
        assert_eq!(cfg.knowledge.engine, "hybrid");
        assert_eq!(cfg.knowledge.top_k, 6);
    }

    #[test]
    fn load_from_missing_file_returns_default() {
        let tmp = TempDir::new().unwrap();
        let cfg = load_project_config(tmp.path());
        assert_eq!(cfg.quality.threshold, 90);
    }

    #[test]
    fn load_parses_toml() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".umadevrc"),
            "[quality]\nthreshold = 80\nskip_checks = [\"dark_mode\"]\n\n[pipeline]\nmax_review_rounds = 2\n",
        )
        .unwrap();
        let cfg = load_project_config(tmp.path());
        assert_eq!(cfg.quality.threshold, 80);
        assert_eq!(cfg.quality.skip_checks, vec!["dark_mode"]);
        assert_eq!(cfg.pipeline.max_review_rounds, 2);
    }

    #[test]
    fn knowledge_section_parses() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".umadevrc"),
            "[knowledge]\nengine = \"hybrid\"\ntop_k = 10\n",
        )
        .unwrap();
        let cfg = load_project_config(tmp.path());
        assert_eq!(cfg.knowledge.engine, "hybrid");
        assert_eq!(cfg.knowledge.top_k, 10);
        assert!(cfg.knowledge.enabled); // defaults to true when omitted
    }

    #[test]
    fn knowledge_disabled_parses() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".umadevrc"),
            "[knowledge]\nenabled = false\n",
        )
        .unwrap();
        let cfg = load_project_config(tmp.path());
        assert!(!cfg.knowledge.enabled);
    }

    #[test]
    fn model_section_parses() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".umadevrc"),
            "[model]\nprovider = \"deepseek\"\n",
        )
        .unwrap();
        let cfg = load_project_config(tmp.path());
        assert_eq!(cfg.model.provider.as_deref(), Some("deepseek"));
    }

    #[test]
    fn model_section_disabled_via_empty_string() {
        let tmp = TempDir::new().unwrap();
        // Empty string = "explicitly use no provider for this project".
        std::fs::write(tmp.path().join(".umadevrc"), "[model]\nprovider = \"\"\n").unwrap();
        let cfg = load_project_config(tmp.path());
        assert_eq!(cfg.model.provider.as_deref(), Some(""));
    }

    #[test]
    fn model_section_absent_is_none() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".umadevrc"), "[quality]\nthreshold = 80\n").unwrap();
        let cfg = load_project_config(tmp.path());
        assert!(cfg.model.provider.is_none());
    }
}
