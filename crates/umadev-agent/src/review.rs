//! PR-ready review report — turn the evidence UmaDev already computed into the
//! single artifact a reviewer reads first.
//!
//! Most generated PRs hand a reviewer raw code and a one-line title. This module
//! flips that: it asserts, with a citation to a concrete file/number for each
//! claim, that the change is safe to merge — **CI was not weakened** (no test
//! files deleted, no `it.skip` / `#[ignore]` introduced in the diff), the
//! **API contract** holds (frontend↔backend alignment), **acceptance gaps** are
//! enumerated (planned endpoints with no implementation), the **governance +
//! security scans** verdicts, the **runtime evidence** (the app actually
//! booted + answered), and a **rollback** hint. The reviewer sees exactly what
//! was checked and what to look at by hand.
//!
//! Everything is **fail-open + deterministic + reuse**: no new model endpoint,
//! no new heavy deps. Each section reads an artifact UmaDev already produced
//! (`umadev-contract`, `acceptance`, `coverage`, the quality gate JSON, the
//! `security` scan, the `runtime_proof` JSON) and degrades to an honest "not
//! available" line rather than fabricating a pass. The git-diff CI check is the
//! one live probe; a missing/!git repo simply downgrades to "could not diff".

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::phases::QualityReport;
use crate::security::SecurityScan;

/// One assertion in the review report: a claim, the verdict, and the concrete
/// evidence backing it. Kept as data (not just rendered text) so the renderer is
/// a pure function over it and the unit tests can assert on the structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewClaim {
    /// Short title (e.g. "CI integrity").
    pub title: String,
    /// `pass` | `warn` | `fail` | `info` — drives the checkbox glyph.
    pub verdict: Verdict,
    /// The human-readable assertion, already carrying its evidence citation.
    pub detail: String,
}

/// Verdict class for a review claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Asserted and verified.
    Pass,
    /// Asserted with a caveat the reviewer should glance at.
    Warn,
    /// A real problem the reviewer must resolve before merge.
    Fail,
    /// Context, no judgement (e.g. rollback instructions).
    Info,
}

impl Verdict {
    /// Markdown checkbox / glyph for the claim line.
    fn glyph(self) -> &'static str {
        match self {
            Verdict::Pass => "[x]",
            Verdict::Warn => "[!]",
            Verdict::Fail => "[ ]",
            Verdict::Info => "[i]",
        }
    }
}

/// The assembled review report: an ordered list of claims plus the slug.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewReport {
    /// Project slug (filename stem).
    pub slug: String,
    /// Ordered claims, top to bottom.
    pub claims: Vec<ReviewClaim>,
}

impl ReviewReport {
    /// `true` iff no claim is a hard `Fail` — i.e. nothing blocks merge from
    /// UmaDev's deterministic checks (a human reviewer still has the final say).
    #[must_use]
    pub fn mergeable(&self) -> bool {
        !self.claims.iter().any(|c| c.verdict == Verdict::Fail)
    }
}

/// Workspace-relative path of the review report.
#[must_use]
pub fn review_report_rel_path(slug: &str) -> String {
    format!("output/{slug}-review-report.md")
}

/// Build the review report by reading every artifact UmaDev already produced.
/// Pure assembly + a single git-diff probe — fail-open throughout: a missing
/// artifact yields an honest "not available" claim, never a panic.
#[must_use]
pub fn build_review_report(project_root: &Path, slug: &str) -> ReviewReport {
    let claims = vec![
        ci_integrity_claim(project_root),
        contract_claim(project_root, slug),
        acceptance_claim(project_root, slug),
        coverage_claim(project_root, slug),
        quality_claim(project_root, slug),
        security_claim(project_root),
        runtime_claim(project_root),
        rollback_claim(project_root, slug),
    ];

    ReviewReport {
        slug: slug.to_string(),
        claims,
    }
}

/// Build, render, and write the review report to `output/<slug>-review-report.md`.
/// Returns the written path.
pub fn write_review_report(project_root: &Path, slug: &str) -> std::io::Result<PathBuf> {
    let report = build_review_report(project_root, slug);
    let md = render_review_md(&report);
    let path = project_root.join(review_report_rel_path(slug));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, md)?;
    Ok(path)
}

// =====================================================================
// per-claim builders (each reuses an already-computed data source)
// =====================================================================

/// CI integrity — the headline reviewer fear: "did this change quietly weaken
/// the test suite?". We diff against the repo HEAD and assert no test file was
/// deleted and no test was newly skipped/ignored in the added lines. Fail-open:
/// not a git repo / no HEAD → `Info` ("could not diff"), never a false `Fail`.
fn ci_integrity_claim(project_root: &Path) -> ReviewClaim {
    let Some(diff) = git_diff(project_root) else {
        return ReviewClaim {
            title: "CI integrity".to_string(),
            verdict: Verdict::Info,
            detail: "Could not produce a git diff (not a repo, or no prior commit) — \
                     CI-weakening could not be checked automatically; review test changes by hand."
                .to_string(),
        };
    };
    let signals = scan_ci_weakening(&diff);
    if signals.is_empty() {
        ReviewClaim {
            title: "CI integrity".to_string(),
            verdict: Verdict::Pass,
            detail: "No test files deleted and no tests newly skipped/ignored in the diff \
                     (scanned `git diff HEAD` for removed test files + added skip markers)."
                .to_string(),
        }
    } else {
        ReviewClaim {
            title: "CI integrity".to_string(),
            verdict: Verdict::Fail,
            detail: format!(
                "CI may have been weakened ({} signal(s)): {}. Confirm these are intentional \
                 before merge.",
                signals.len(),
                signals
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        }
    }
}

/// Frontend↔backend API contract — reuse `umadev-contract` exactly as the
/// quality gate does: parse the architecture API table, extract the real
/// frontend calls, cross-validate.
fn contract_claim(project_root: &Path, slug: &str) -> ReviewClaim {
    let arch = read(project_root.join(format!("output/{slug}-architecture.md")));
    if arch.trim().is_empty() {
        return ReviewClaim {
            title: "API contract".to_string(),
            verdict: Verdict::Info,
            detail: "No architecture doc to derive a contract from (docs-only stage).".to_string(),
        };
    }
    let arch_spec = umadev_contract::parse_architecture(&arch, slug);
    let calls = umadev_contract::extract_frontend_calls(project_root);
    let violations = umadev_contract::validate_frontend_vs_contract(&calls, &arch_spec);
    if violations.is_empty() {
        ReviewClaim {
            title: "API contract".to_string(),
            verdict: Verdict::Pass,
            detail: format!(
                "All {} extracted frontend call(s) align with the {} endpoint(s) in \
                 `output/{slug}-architecture.md` (UD-CODE-003).",
                calls.len(),
                arch_spec.len()
            ),
        }
    } else {
        ReviewClaim {
            title: "API contract".to_string(),
            verdict: Verdict::Warn,
            detail: format!(
                "{} frontend↔contract mismatch(es): {}.",
                violations.len(),
                violations
                    .iter()
                    .take(4)
                    .map(|v| v.detail.clone())
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        }
    }
}

/// Acceptance gaps — reuse `acceptance::task_acceptance_gaps`: planned endpoints
/// with no implementation evidence in the workspace.
fn acceptance_claim(project_root: &Path, slug: &str) -> ReviewClaim {
    let gaps = crate::acceptance::task_acceptance_gaps(project_root, slug);
    if gaps.is_empty() {
        ReviewClaim {
            title: "Acceptance".to_string(),
            verdict: Verdict::Pass,
            detail: "Every endpoint planned in the architecture API table has implementation \
                     evidence in the source tree (no acceptance gaps)."
                .to_string(),
        }
    } else {
        ReviewClaim {
            title: "Acceptance".to_string(),
            verdict: Verdict::Warn,
            detail: format!(
                "{} planned endpoint(s) have NO implementation found: {}.",
                gaps.len(),
                gaps.iter().take(4).cloned().collect::<Vec<_>>().join("; ")
            ),
        }
    }
}

/// Requirement coverage — reuse `coverage::uncovered_requirements`: PRD `FR-NNN`
/// ids cited by no task/plan.
fn coverage_claim(project_root: &Path, slug: &str) -> ReviewClaim {
    let uncovered = crate::coverage::uncovered_requirements(project_root, slug);
    if uncovered.is_empty() {
        ReviewClaim {
            title: "Requirement coverage".to_string(),
            verdict: Verdict::Pass,
            detail: "Every PRD functional requirement (FR-NNN) is cited by the execution plan \
                     or a task (no orphaned requirements)."
                .to_string(),
        }
    } else {
        ReviewClaim {
            title: "Requirement coverage".to_string(),
            verdict: Verdict::Warn,
            detail: format!(
                "{} PRD requirement(s) cited by no task: {}.",
                uncovered.len(),
                uncovered.join(", ")
            ),
        }
    }
}

/// Quality gate + governance — reuse the persisted quality-gate JSON (which
/// already folds in the governance block-event counts and the design scans).
fn quality_claim(project_root: &Path, slug: &str) -> ReviewClaim {
    let Some(report) = read_quality(project_root, slug) else {
        return ReviewClaim {
            title: "Quality gate".to_string(),
            verdict: Verdict::Info,
            detail: "No quality gate report yet (runs at the `quality` phase).".to_string(),
        };
    };
    let failed: Vec<&str> = report
        .checks
        .iter()
        .filter(|c| c.status == "failed")
        .map(|c| c.name.as_str())
        .collect();
    if report.passed && failed.is_empty() {
        ReviewClaim {
            title: "Quality gate".to_string(),
            verdict: Verdict::Pass,
            detail: format!(
                "Quality gate PASSED at {}/100 across {} checks \
                 (see `output/{slug}-quality-gate.json`).",
                report.total_score,
                report.checks.len()
            ),
        }
    } else {
        ReviewClaim {
            title: "Quality gate".to_string(),
            verdict: if report.passed {
                Verdict::Warn
            } else {
                Verdict::Fail
            },
            detail: format!(
                "Quality gate at {}/100 ({}); failing check(s): {}.",
                report.total_score,
                if report.passed {
                    "passed with warnings"
                } else {
                    "BELOW threshold"
                },
                if failed.is_empty() {
                    "none (sub-threshold weighted score)".to_string()
                } else {
                    failed.join(", ")
                }
            ),
        }
    }
}

/// Security scan — reuse the persisted `.umadev/audit/security-scan.json`.
fn security_claim(project_root: &Path) -> ReviewClaim {
    let path = project_root.join(crate::security::security_scan_rel_path());
    let Some(scan) = read(path).pipe_opt(|s| serde_json::from_str::<SecurityScan>(s).ok()) else {
        return ReviewClaim {
            title: "Security scan".to_string(),
            verdict: Verdict::Info,
            detail: "No security scan recorded yet (runs at the `delivery` phase).".to_string(),
        };
    };
    if !scan.any_ran() {
        return ReviewClaim {
            title: "Security scan".to_string(),
            verdict: Verdict::Info,
            detail: format!(
                "No security scanners available on this machine — {} \
                 (install gitleaks / npm-audit / cargo-audit / pip-audit to enable).",
                scan.summary_line()
            ),
        };
    }
    if scan.has_findings() {
        ReviewClaim {
            title: "Security scan".to_string(),
            verdict: Verdict::Warn,
            detail: format!(
                "{} (see `.umadev/audit/security-scan.json`).",
                scan.summary_line()
            ),
        }
    } else {
        ReviewClaim {
            title: "Security scan".to_string(),
            verdict: Verdict::Pass,
            detail: format!(
                "{} (see `.umadev/audit/security-scan.json`).",
                scan.summary_line()
            ),
        }
    }
}

/// Runtime evidence — reuse the persisted `.umadev/audit/runtime-proof.json`.
fn runtime_claim(project_root: &Path) -> ReviewClaim {
    let path = project_root.join(crate::runtime_proof::runtime_proof_rel_path());
    let Some(proof) =
        read(path).pipe_opt(|s| serde_json::from_str::<crate::runtime_proof::RuntimeProof>(s).ok())
    else {
        return ReviewClaim {
            title: "Runtime evidence".to_string(),
            verdict: Verdict::Info,
            detail: "No runtime proof recorded (`umadev verify --runtime` not run).".to_string(),
        };
    };
    if proof.status.is_verified() {
        let ok = proof.routes.iter().filter(|r| r.ok).count();
        ReviewClaim {
            title: "Runtime evidence".to_string(),
            verdict: Verdict::Pass,
            detail: format!(
                "App booted and answered: {} (route checks {}/{} OK; see \
                 `.umadev/audit/runtime-proof.json`).",
                proof.summary_line(),
                ok,
                proof.routes.len()
            ),
        }
    } else {
        ReviewClaim {
            title: "Runtime evidence".to_string(),
            verdict: Verdict::Info,
            detail: format!("Runtime not exercised: {}.", proof.summary_line()),
        }
    }
}

/// Rollback hint — always present (`Info`). Names the proof-pack + checkpoint
/// surfaces a reviewer would use to revert. We point at the shadow-checkpoint
/// repo when it exists, and always at plain `git revert`.
fn rollback_claim(project_root: &Path, slug: &str) -> ReviewClaim {
    let has_checkpoints = project_root.join(".umadev/checkpoints.git/HEAD").exists();
    let checkpoint_line = if has_checkpoints {
        " UmaDev checkpoints exist — `umadev rollback` rewinds files to a pre-phase snapshot."
    } else {
        ""
    };
    ReviewClaim {
        title: "Rollback".to_string(),
        verdict: Verdict::Info,
        detail: format!(
            "To revert: `git revert <merge-commit>` (or reset the feature branch).{checkpoint_line} \
             The full evidence bundle is in `release/proof-pack-{slug}-*.zip` for post-merge audit."
        ),
    }
}

// =====================================================================
// rendering
// =====================================================================

/// Render the report to PR-ready markdown. Pure function over [`ReviewReport`].
#[must_use]
pub fn render_review_md(report: &ReviewReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Review report — {}\n\n", report.slug));
    let verdict = if report.mergeable() {
        "No blocking issues from automated checks — ready for human review."
    } else {
        "Blocking issue(s) detected — resolve the failing claim(s) before merge."
    };
    out.push_str(&format!("> {verdict}\n\n"));
    out.push_str(
        "Generated by UmaDev from the run's own evidence. Each claim cites the file or number \
         it is derived from; nothing here is asserted without a source.\n\n",
    );
    out.push_str("## Reviewer checklist\n\n");
    for c in &report.claims {
        out.push_str(&format!(
            "- {} **{}** — {}\n",
            c.verdict.glyph(),
            c.title,
            c.detail
        ));
    }
    out.push_str(
        "\n## Legend\n\n\
         - `[x]` verified · `[!]` verified with a caveat · `[ ]` blocking, must resolve · \
         `[i]` context / not applicable\n",
    );
    out
}

// =====================================================================
// helpers
// =====================================================================

fn read(path: PathBuf) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

/// Tiny `Option`-combinator so the claim builders read top-down. Returns `None`
/// when the source string is empty (no artifact) OR the mapper returns `None`.
trait PipeOpt {
    fn pipe_opt<T>(self, f: impl FnOnce(&str) -> Option<T>) -> Option<T>;
}
impl PipeOpt for String {
    fn pipe_opt<T>(self, f: impl FnOnce(&str) -> Option<T>) -> Option<T> {
        if self.trim().is_empty() {
            None
        } else {
            f(&self)
        }
    }
}

fn read_quality(project_root: &Path, slug: &str) -> Option<QualityReport> {
    let body = read(project_root.join(format!("output/{slug}-quality-gate.json")));
    if body.trim().is_empty() {
        return None;
    }
    serde_json::from_str(&body).ok()
}

/// `git diff HEAD` (with rename detection) over the workspace, or `None` when
/// it's not a usable git repo. Fail-open: any spawn error / non-zero exit → None.
fn git_diff(project_root: &Path) -> Option<String> {
    let out = Command::new("git")
        .args(["diff", "--find-renames", "HEAD"])
        .current_dir(project_root)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Scan a unified diff for signals that CI/test coverage was weakened:
/// 1. a test FILE was deleted (`deleted file ... <test-path>`), or
/// 2. a test was newly skipped/ignored in the ADDED lines.
///
/// Returns a list of human descriptions (empty == clean). Pure + testable.
#[must_use]
pub fn scan_ci_weakening(diff: &str) -> Vec<String> {
    let mut signals = Vec::new();
    let mut cur_file = String::new();
    let mut pending_delete = false;

    // Markers that, when ADDED (a `+` line), disable a test.
    const SKIP_MARKERS: &[&str] = &[
        "it.skip",
        "describe.skip",
        "test.skip",
        "xit(",
        "xdescribe(",
        "#[ignore]",
        "#[ignore ",
        "@pytest.mark.skip",
        "@unittest.skip",
        "@Disabled",
        "@Ignore",
        "t.Skip(",
    ];

    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            // `a/path b/path` — take the b-side path.
            cur_file = rest.split(" b/").nth(1).unwrap_or("").trim().to_string();
            pending_delete = false;
            continue;
        }
        if line.starts_with("deleted file mode") {
            pending_delete = true;
            continue;
        }
        if pending_delete && is_test_path(&cur_file) {
            signals.push(format!("deleted test file `{cur_file}`"));
            pending_delete = false;
            continue;
        }
        // Added line introducing a skip marker (exclude the `+++` header).
        if line.starts_with('+') && !line.starts_with("+++") {
            let added = &line[1..];
            for m in SKIP_MARKERS {
                if added.contains(m) {
                    signals.push(format!(
                        "added skip/ignore (`{}`) in `{}`",
                        m.trim_end_matches(['(', ' ']),
                        if cur_file.is_empty() {
                            "<file>"
                        } else {
                            &cur_file
                        }
                    ));
                    break;
                }
            }
        }
    }
    signals
}

/// Heuristic: does this path look like a test file? (Matches the common
/// conventions across the stacks UmaDev targets.)
fn is_test_path(path: &str) -> bool {
    let p = path.to_ascii_lowercase();
    p.contains("/tests/")
        || p.contains("/test/")
        || p.contains("__tests__")
        || p.ends_with("_test.go")
        || p.ends_with("_test.py")
        || p.ends_with("test.ts")
        || p.ends_with("test.tsx")
        || p.ends_with("test.js")
        || p.ends_with("test.jsx")
        || p.ends_with(".test.ts")
        || p.ends_with(".test.tsx")
        || p.ends_with(".test.js")
        || p.ends_with(".spec.ts")
        || p.ends_with(".spec.tsx")
        || p.ends_with(".spec.js")
        || p.ends_with("_spec.rb")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn clean_diff_has_no_signals() {
        let diff = "diff --git a/src/lib.rs b/src/lib.rs\n\
                    @@ -1 +1,2 @@\n\
                    +pub fn add(a: i32, b: i32) -> i32 { a + b }\n";
        assert!(scan_ci_weakening(diff).is_empty());
    }

    #[test]
    fn detects_deleted_test_file() {
        let diff = "diff --git a/src/api.test.ts b/src/api.test.ts\n\
                    deleted file mode 100644\n\
                    index abc..000\n\
                    --- a/src/api.test.ts\n\
                    +++ /dev/null\n";
        let s = scan_ci_weakening(diff);
        assert_eq!(s.len(), 1);
        assert!(s[0].contains("deleted test file"));
        assert!(s[0].contains("api.test.ts"));
    }

    #[test]
    fn detects_added_skip_markers() {
        let diff = "diff --git a/tests/login_test.py b/tests/login_test.py\n\
                    @@ -1,3 +1,4 @@\n\
                    +@pytest.mark.skip\n\
                    +def test_login(): ...\n";
        let s = scan_ci_weakening(diff);
        assert!(s.iter().any(|x| x.contains("skip/ignore")));
    }

    #[test]
    fn detects_rust_ignore_in_added_lines_only() {
        // An `#[ignore]` that already existed (context line, no `+`) is fine;
        // only a freshly ADDED one counts.
        let added = "diff --git a/src/lib.rs b/src/lib.rs\n+#[ignore]\n+fn t() {}\n";
        assert_eq!(scan_ci_weakening(added).len(), 1);
        let context = "diff --git a/src/lib.rs b/src/lib.rs\n #[ignore]\n fn t() {}\n";
        assert!(scan_ci_weakening(context).is_empty());
    }

    #[test]
    fn deleting_a_non_test_file_is_not_a_signal() {
        let diff = "diff --git a/README.md b/README.md\n\
                    deleted file mode 100644\n";
        assert!(scan_ci_weakening(diff).is_empty());
    }

    #[test]
    fn is_test_path_matches_conventions() {
        assert!(is_test_path("src/foo.test.ts"));
        assert!(is_test_path("spec/models/user_spec.rb"));
        assert!(is_test_path("pkg/handler_test.go"));
        assert!(is_test_path("app/__tests__/Button.jsx"));
        assert!(!is_test_path("src/main.rs"));
        assert!(!is_test_path("docs/readme.md"));
    }

    #[test]
    fn render_is_pure_and_lists_every_claim() {
        let report = ReviewReport {
            slug: "demo".to_string(),
            claims: vec![
                ReviewClaim {
                    title: "CI integrity".to_string(),
                    verdict: Verdict::Pass,
                    detail: "no weakening".to_string(),
                },
                ReviewClaim {
                    title: "Acceptance".to_string(),
                    verdict: Verdict::Fail,
                    detail: "1 gap".to_string(),
                },
            ],
        };
        assert!(!report.mergeable());
        let md = render_review_md(&report);
        assert!(md.contains("# Review report — demo"));
        assert!(md.contains("CI integrity"));
        assert!(md.contains("Acceptance"));
        assert!(md.contains("[x]"));
        assert!(md.contains("[ ]")); // the failing claim
        assert!(md.contains("Blocking issue"));
    }

    #[test]
    fn build_on_bare_workspace_is_fail_open() {
        // No artifacts at all → every data-driven claim degrades to Info/Pass,
        // nothing panics, and the report is renderable + writable.
        let tmp = TempDir::new().unwrap();
        let report = build_review_report(tmp.path(), "demo");
        assert_eq!(report.claims.len(), 8);
        // Rollback claim is always present.
        assert!(report.claims.iter().any(|c| c.title == "Rollback"));
        let path = write_review_report(tmp.path(), "demo").unwrap();
        assert!(path.exists());
        assert!(fs::read_to_string(&path).unwrap().contains("Review report"));
    }

    #[test]
    fn acceptance_gap_surfaces_as_warn_claim() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("output")).unwrap();
        fs::write(
            tmp.path().join("output/demo-architecture.md"),
            "# API\n\n\
             | Method | Path | Description | Auth |\n\
             |---|---|---|---|\n\
             | GET | /api/widgets | list widgets | none |\n",
        )
        .unwrap();
        let report = build_review_report(tmp.path(), "demo");
        let accept = report
            .claims
            .iter()
            .find(|c| c.title == "Acceptance")
            .unwrap();
        // No source implements /api/widgets → a gap → Warn.
        assert_eq!(accept.verdict, Verdict::Warn);
        assert!(accept.detail.contains("/api/widgets"));
    }
}
