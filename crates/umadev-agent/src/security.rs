//! Pre-PR security scan gate — shell out to whatever scanners are already on
//! the machine, never bundle one.
//!
//! Two scan families, picked by the project's stack:
//! - **secrets** — a leaked-key scanner over the whole tree (`gitleaks`).
//! - **dependencies** — the package-manager's own advisory audit, chosen by the
//!   lockfile present (`npm audit` / `cargo audit` / `pip-audit`).
//!
//! Everything here is **fail-open by contract** (the same rule the governance
//! kernel follows): a missing tool, a non-zero exit we can't parse, a spawn
//! error, or a timeout all collapse to a `skipped`/`error` row with a short
//! reason — never a panic, never a hard block on the pipeline. A scan we could
//! not run is recorded as "not run", not as "clean". The result is written to
//! `.umadev/audit/security-scan.json` and folded into the proof-pack + the
//! review report so a PR reviewer can see exactly what was (and was not) checked.
//!
//! We deliberately do NOT vendor a scanner or add a Rust advisory-DB dep: the
//! value is in surfacing the customer's OWN installed tooling's verdict as
//! reviewable evidence, with zero new heavy transitive deps.

use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use chrono::Utc;
use serde::{Deserialize, Serialize};

/// Workspace-relative path of the persisted scan result.
const SCAN_REL_PATH: &str = ".umadev/audit/security-scan.json";

/// Hard wall-clock ceiling for any single scanner. A scanner that hangs must
/// not wedge delivery — we kill it and record a `timeout` skip.
const SCAN_TIMEOUT: Duration = Duration::from_secs(120);

/// The outcome class of one scanner invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanStatus {
    /// The scanner ran and found nothing actionable.
    Clean,
    /// The scanner ran and reported one or more findings.
    Findings,
    /// The scanner was not run (tool absent / not applicable to this stack).
    Skipped,
    /// The scanner was attempted but could not complete (spawn error, timeout,
    /// unparseable output). Treated as "not verified", never as "clean".
    Error,
}

impl ScanStatus {
    /// Stable label for display / audit rows.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ScanStatus::Clean => "clean",
            ScanStatus::Findings => "findings",
            ScanStatus::Skipped => "skipped",
            ScanStatus::Error => "error",
        }
    }
}

/// One scanner's result row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanResult {
    /// Which scanner produced this row (`gitleaks` / `npm-audit` / …).
    pub tool: String,
    /// What it checks (`secrets` / `dependencies`).
    pub category: String,
    /// Outcome class.
    pub status: ScanStatus,
    /// Count of findings (0 unless `status == Findings`).
    pub findings: u32,
    /// Short human reason — why it was skipped, or a one-line finding summary.
    pub detail: String,
}

impl ScanResult {
    fn skipped(tool: &str, category: &str, reason: impl Into<String>) -> Self {
        Self {
            tool: tool.to_string(),
            category: category.to_string(),
            status: ScanStatus::Skipped,
            findings: 0,
            detail: reason.into(),
        }
    }
}

/// The full pre-PR security scan report. Serialized to
/// `.umadev/audit/security-scan.json` and embedded in the proof-pack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityScan {
    /// ISO-8601 timestamp the scan ran.
    pub timestamp: String,
    /// Per-scanner rows (one per attempted scanner, including skips).
    pub results: Vec<ScanResult>,
}

impl SecurityScan {
    /// `true` iff at least one scanner actually ran (clean OR findings) — i.e.
    /// the scan produced real signal rather than skipping everything.
    #[must_use]
    pub fn any_ran(&self) -> bool {
        self.results
            .iter()
            .any(|r| matches!(r.status, ScanStatus::Clean | ScanStatus::Findings))
    }

    /// Total findings across all scanners that ran.
    #[must_use]
    pub fn total_findings(&self) -> u32 {
        self.results.iter().map(|r| r.findings).sum()
    }

    /// `true` iff any scanner reported findings — the PR-blocking signal a
    /// reviewer cares about (the gate itself stays fail-open; this is advisory).
    #[must_use]
    pub fn has_findings(&self) -> bool {
        self.results
            .iter()
            .any(|r| r.status == ScanStatus::Findings)
    }

    /// A neutral one-line summary for logs / the review report.
    #[must_use]
    pub fn summary_line(&self) -> String {
        let ran = self
            .results
            .iter()
            .filter(|r| matches!(r.status, ScanStatus::Clean | ScanStatus::Findings))
            .count();
        let skipped = self
            .results
            .iter()
            .filter(|r| r.status == ScanStatus::Skipped)
            .count();
        if ran == 0 {
            return "security scan: no scanners available (all skipped)".to_string();
        }
        let findings = self.total_findings();
        if findings == 0 {
            format!("security scan: {ran} scanner(s) ran, no findings ({skipped} skipped)")
        } else {
            format!(
                "security scan: {findings} finding(s) across {ran} scanner(s) ({skipped} skipped)"
            )
        }
    }
}

/// Workspace-relative path of the persisted security-scan artifact.
#[must_use]
pub fn security_scan_rel_path() -> &'static str {
    SCAN_REL_PATH
}

/// Run the pre-PR security scan over `project_root`. Always returns a
/// [`SecurityScan`] — every scanner is fail-open, so a machine with no scanners
/// installed yields an all-`skipped` report rather than an error. Detects which
/// scanners apply from the lockfiles present and which are actually on `PATH`.
#[must_use]
pub fn run_security_scan(project_root: &Path) -> SecurityScan {
    let mut results = Vec::new();

    // --- secrets: gitleaks over the whole tree -------------------------------
    results.push(scan_secrets(project_root));

    // --- dependencies: the package manager's own audit, by lockfile ----------
    // A polyglot repo can have more than one stack; we run each that applies and
    // whose tool is installed. Stacks with no lockfile are silently not added
    // (no skip row) — only an applicable-but-missing tool earns a visible skip.
    for dep in dependency_scanners(project_root) {
        results.push(dep);
    }

    SecurityScan {
        timestamp: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        results,
    }
}

/// Run the scan and persist it to `.umadev/audit/security-scan.json`. Returns
/// the written path on success. Fail-open: a write error is swallowed and the
/// in-memory report is still returned via the `Ok`/`Err` split so callers can
/// surface it regardless.
pub fn write_security_scan(
    project_root: &Path,
    scan: &SecurityScan,
) -> std::io::Result<std::path::PathBuf> {
    let path = project_root.join(SCAN_REL_PATH);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json =
        serde_json::to_string_pretty(scan).unwrap_or_else(|_| "{\"results\":[]}".to_string());
    std::fs::write(&path, json)?;
    Ok(path)
}

/// Whether `tool` is resolvable on `PATH`. Uses the platform's `which`/`where`
/// so a self-test never depends on running the scanner itself. Fail-open:
/// any spawn error → "not found".
fn tool_on_path(tool: &str) -> bool {
    let probe = if cfg!(windows) { "where" } else { "which" };
    Command::new(probe)
        .arg(tool)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run `cmd` with `args` in `cwd`, capped at [`SCAN_TIMEOUT`]. Returns
/// `(exit_code, combined_output)` on completion, or `None` on spawn failure /
/// timeout. The thread-less timeout is a poll loop on `try_wait` — we avoid
/// pulling tokio into a synchronous, fail-open scan path.
fn run_capped(cmd: &str, args: &[&str], cwd: &Path) -> Option<(i32, String)> {
    use std::process::Stdio;
    let mut child = Command::new(cmd)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = child.wait_with_output().ok()?;
                let mut combined = String::from_utf8_lossy(&out.stdout).into_owned();
                combined.push_str(&String::from_utf8_lossy(&out.stderr));
                return Some((status.code().unwrap_or(-1), combined));
            }
            Ok(None) => {
                if start.elapsed() > SCAN_TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => return None,
        }
    }
}

// =====================================================================
// secrets — gitleaks
// =====================================================================

/// Scan the working tree for leaked secrets via `gitleaks`. Skipped when the
/// tool is absent. `gitleaks detect` exits non-zero (1) when leaks are found
/// and 0 when clean, so the exit code IS the verdict.
fn scan_secrets(project_root: &Path) -> ScanResult {
    const TOOL: &str = "gitleaks";
    const CAT: &str = "secrets";
    if !tool_on_path(TOOL) {
        return ScanResult::skipped(TOOL, CAT, "gitleaks not installed");
    }
    // `--no-banner --redact` keeps output terse and never echoes the secret;
    // `--no-git` scans the working tree even without a git history (a fresh
    // scaffold may not be a repo yet).
    let args = [
        "detect",
        "--no-banner",
        "--redact",
        "--no-git",
        "--source",
        ".",
    ];
    let Some((code, out)) = run_capped(TOOL, &args, project_root) else {
        return ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Error,
            findings: 0,
            detail: "gitleaks did not complete (spawn error or timeout)".to_string(),
        };
    };
    if code == 0 {
        return ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Clean,
            findings: 0,
            detail: "no leaked secrets detected".to_string(),
        };
    }
    // Non-zero: gitleaks found leaks. Count "leaks found" lines for a rough
    // tally; the redacted output never carries the secret value itself.
    let n = count_gitleaks_findings(&out);
    ScanResult {
        tool: TOOL.to_string(),
        category: CAT.to_string(),
        status: ScanStatus::Findings,
        findings: n,
        detail: format!("gitleaks reported {n} leaked secret(s) — see scanner output"),
    }
}

/// Count secret findings from gitleaks output. It prints one `Finding:` block
/// per leak (and a trailing `leaks found: N` summary on newer versions). We
/// prefer the explicit summary line; otherwise fall back to counting blocks.
fn count_gitleaks_findings(out: &str) -> u32 {
    // Newer gitleaks: `... leaks found: 3` (in the WRN/INF summary line).
    for line in out.lines() {
        let lower = line.to_ascii_lowercase();
        if let Some(idx) = lower.find("leaks found:") {
            let tail = &line[idx + "leaks found:".len()..];
            if let Some(n) = tail.split_whitespace().next().and_then(|t| t.parse().ok()) {
                return n;
            }
        }
    }
    // Fallback: count `Finding:` markers (older text output).
    let blocks = out
        .lines()
        .filter(|l| l.trim_start().starts_with("Finding:"))
        .count();
    u32::try_from(blocks.max(1)).unwrap_or(1)
}

// =====================================================================
// dependencies — npm audit / cargo audit / pip-audit
// =====================================================================

/// Build the dependency-scan rows for whichever stacks this repo has. Only adds
/// a row for a stack whose lockfile is present; an applicable-but-uninstalled
/// tool produces a visible `skipped` row (so the reviewer knows it was relevant
/// but couldn't run).
fn dependency_scanners(project_root: &Path) -> Vec<ScanResult> {
    let mut out = Vec::new();
    if project_root.join("package-lock.json").is_file()
        || project_root.join("package.json").is_file()
    {
        out.push(scan_npm_audit(project_root));
    }
    if project_root.join("Cargo.lock").is_file() {
        out.push(scan_cargo_audit(project_root));
    }
    if project_root.join("requirements.txt").is_file()
        || project_root.join("pyproject.toml").is_file()
        || project_root.join("poetry.lock").is_file()
    {
        out.push(scan_pip_audit(project_root));
    }
    out
}

/// `npm audit --json` over an npm project. Parses the `metadata.vulnerabilities`
/// totals when present.
fn scan_npm_audit(project_root: &Path) -> ScanResult {
    const TOOL: &str = "npm-audit";
    const CAT: &str = "dependencies";
    if !tool_on_path("npm") {
        return ScanResult::skipped(TOOL, CAT, "npm not installed");
    }
    let Some((_code, out)) = run_capped("npm", &["audit", "--json"], project_root) else {
        return ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Error,
            findings: 0,
            detail: "npm audit did not complete (spawn error or timeout)".to_string(),
        };
    };
    parse_npm_audit(&out)
}

/// Parse `npm audit --json` output into a result row. Pure (testable without
/// npm): reads `.metadata.vulnerabilities.total`. Fail-open: unparseable JSON
/// (e.g. npm printed an error, or there's no lockfile) → `error`.
fn parse_npm_audit(out: &str) -> ScanResult {
    const TOOL: &str = "npm-audit";
    const CAT: &str = "dependencies";
    let Ok(v) = serde_json::from_str::<serde_json::Value>(out.trim()) else {
        return ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Error,
            findings: 0,
            detail: "npm audit output was not JSON (no lockfile, or npm error)".to_string(),
        };
    };
    // npm v7+ shape: metadata.vulnerabilities = {info,low,moderate,high,critical,total}.
    let total = v
        .get("metadata")
        .and_then(|m| m.get("vulnerabilities"))
        .and_then(|x| x.get("total"))
        .and_then(serde_json::Value::as_u64)
        .map(|n| u32::try_from(n).unwrap_or(u32::MAX));
    match total {
        Some(0) => ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Clean,
            findings: 0,
            detail: "no known vulnerable dependencies".to_string(),
        },
        Some(n) => {
            let (high, crit) = npm_high_crit(&v);
            ScanResult {
                tool: TOOL.to_string(),
                category: CAT.to_string(),
                status: ScanStatus::Findings,
                findings: n,
                detail: format!(
                    "{n} vulnerable dependency advisory(ies) ({high} high, {crit} critical)"
                ),
            }
        }
        None => ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Error,
            findings: 0,
            detail: "npm audit JSON lacked a vulnerability total".to_string(),
        },
    }
}

/// Pull the high/critical sub-counts from an npm-audit JSON value (best-effort).
fn npm_high_crit(v: &serde_json::Value) -> (u64, u64) {
    let vulns = v.get("metadata").and_then(|m| m.get("vulnerabilities"));
    let high = vulns
        .and_then(|x| x.get("high"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let crit = vulns
        .and_then(|x| x.get("critical"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    (high, crit)
}

/// `cargo audit --json` over a Rust project. `cargo-audit` exits non-zero when
/// vulnerabilities are found; the JSON carries the count.
fn scan_cargo_audit(project_root: &Path) -> ScanResult {
    const TOOL: &str = "cargo-audit";
    const CAT: &str = "dependencies";
    // `cargo audit` is a cargo subcommand; probe the `cargo-audit` shim binary.
    if !tool_on_path("cargo-audit") && !tool_on_path("cargo") {
        return ScanResult::skipped(TOOL, CAT, "cargo-audit not installed");
    }
    let Some((_code, out)) = run_capped("cargo", &["audit", "--json"], project_root) else {
        return ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Error,
            findings: 0,
            detail: "cargo audit did not complete (spawn error or timeout)".to_string(),
        };
    };
    parse_cargo_audit(&out)
}

/// Parse `cargo audit --json` output. Pure (testable without cargo-audit):
/// reads `.vulnerabilities.count`. A missing `cargo-audit` subcommand prints a
/// non-JSON cargo error → `skipped` (the tool isn't really installed).
fn parse_cargo_audit(out: &str) -> ScanResult {
    const TOOL: &str = "cargo-audit";
    const CAT: &str = "dependencies";
    let trimmed = out.trim();
    let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        // `error: no such command: audit` → the shim is genuinely missing.
        if trimmed.contains("no such command") || trimmed.contains("not installed") {
            return ScanResult::skipped(TOOL, CAT, "cargo-audit subcommand not installed");
        }
        return ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Error,
            findings: 0,
            detail: "cargo audit output was not JSON".to_string(),
        };
    };
    let count = v
        .get("vulnerabilities")
        .and_then(|x| x.get("count"))
        .and_then(serde_json::Value::as_u64)
        .map(|n| u32::try_from(n).unwrap_or(u32::MAX));
    match count {
        Some(0) | None if v.get("vulnerabilities").is_some() => ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Clean,
            findings: 0,
            detail: "no RustSec advisories for locked dependencies".to_string(),
        },
        Some(n) => ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Findings,
            findings: n,
            detail: format!("{n} RustSec advisory(ies) against Cargo.lock"),
        },
        None => ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Error,
            findings: 0,
            detail: "cargo audit JSON lacked a vulnerability count".to_string(),
        },
    }
}

/// `pip-audit --format json` over a Python project.
fn scan_pip_audit(project_root: &Path) -> ScanResult {
    const TOOL: &str = "pip-audit";
    const CAT: &str = "dependencies";
    if !tool_on_path("pip-audit") {
        return ScanResult::skipped(TOOL, CAT, "pip-audit not installed");
    }
    let Some((_code, out)) = run_capped("pip-audit", &["--format", "json"], project_root) else {
        return ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Error,
            findings: 0,
            detail: "pip-audit did not complete (spawn error or timeout)".to_string(),
        };
    };
    parse_pip_audit(&out)
}

/// Parse `pip-audit --format json` output. Pure (testable without pip-audit):
/// pip-audit emits either a top-level array of dependency records (each with a
/// `vulns` array) or a `{ "dependencies": [...] }` object depending on version.
/// We sum the `vulns` across all records.
fn parse_pip_audit(out: &str) -> ScanResult {
    const TOOL: &str = "pip-audit";
    const CAT: &str = "dependencies";
    let Ok(v) = serde_json::from_str::<serde_json::Value>(out.trim()) else {
        return ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Error,
            findings: 0,
            detail: "pip-audit output was not JSON".to_string(),
        };
    };
    // Accept both the bare-array and the {dependencies:[...]} shapes.
    let deps = v
        .as_array()
        .cloned()
        .or_else(|| {
            v.get("dependencies")
                .and_then(serde_json::Value::as_array)
                .cloned()
        })
        .unwrap_or_default();
    let mut total: u32 = 0;
    for d in &deps {
        if let Some(vulns) = d.get("vulns").and_then(serde_json::Value::as_array) {
            total = total.saturating_add(u32::try_from(vulns.len()).unwrap_or(0));
        }
    }
    if total == 0 {
        ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Clean,
            findings: 0,
            detail: "no known vulnerable Python dependencies".to_string(),
        }
    } else {
        ScanResult {
            tool: TOOL.to_string(),
            category: CAT.to_string(),
            status: ScanStatus::Findings,
            findings: total,
            detail: format!("{total} vulnerable Python dependency advisory(ies)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn empty_repo_skips_everything_fail_open() {
        // A bare temp dir has no lockfiles and (in CI) likely no gitleaks, so the
        // scan must produce only skip/error rows and NEVER panic or block.
        let tmp = TempDir::new().unwrap();
        let scan = run_security_scan(tmp.path());
        // gitleaks row is always attempted (secrets apply to any tree).
        assert!(scan.results.iter().any(|r| r.category == "secrets"));
        // No lockfiles → no dependency rows added.
        assert!(scan.results.iter().all(|r| r.category != "dependencies"));
        // Whatever happened, the report is well-formed and serializable.
        assert!(serde_json::to_string(&scan).is_ok());
    }

    #[test]
    fn write_then_read_roundtrips() {
        let tmp = TempDir::new().unwrap();
        let scan = SecurityScan {
            timestamp: "2026-06-22T00:00:00Z".to_string(),
            results: vec![ScanResult {
                tool: "gitleaks".to_string(),
                category: "secrets".to_string(),
                status: ScanStatus::Clean,
                findings: 0,
                detail: "clean".to_string(),
            }],
        };
        let path = write_security_scan(tmp.path(), &scan).unwrap();
        assert!(path.ends_with("security-scan.json"));
        let back: SecurityScan =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(back, scan);
    }

    #[test]
    fn npm_audit_clean_parse() {
        let json = r#"{"metadata":{"vulnerabilities":{"info":0,"low":0,"moderate":0,"high":0,"critical":0,"total":0}}}"#;
        let r = parse_npm_audit(json);
        assert_eq!(r.status, ScanStatus::Clean);
        assert_eq!(r.findings, 0);
    }

    #[test]
    fn npm_audit_findings_parse() {
        let json = r#"{"metadata":{"vulnerabilities":{"info":1,"low":2,"moderate":0,"high":3,"critical":1,"total":7}}}"#;
        let r = parse_npm_audit(json);
        assert_eq!(r.status, ScanStatus::Findings);
        assert_eq!(r.findings, 7);
        assert!(r.detail.contains("3 high"));
        assert!(r.detail.contains("1 critical"));
    }

    #[test]
    fn npm_audit_garbage_is_error_not_clean() {
        // Fail-open contract: unparseable output must NOT read as "clean".
        let r = parse_npm_audit("npm ERR! could not read lockfile");
        assert_eq!(r.status, ScanStatus::Error);
        assert_eq!(r.findings, 0);
    }

    #[test]
    fn cargo_audit_parse_variants() {
        let clean = r#"{"vulnerabilities":{"found":false,"count":0,"list":[]}}"#;
        assert_eq!(parse_cargo_audit(clean).status, ScanStatus::Clean);
        let found = r#"{"vulnerabilities":{"found":true,"count":2,"list":[{},{}]}}"#;
        let r = parse_cargo_audit(found);
        assert_eq!(r.status, ScanStatus::Findings);
        assert_eq!(r.findings, 2);
        // Missing subcommand → skipped, not error.
        assert_eq!(
            parse_cargo_audit("error: no such command: `audit`").status,
            ScanStatus::Skipped
        );
    }

    #[test]
    fn pip_audit_parse_both_shapes() {
        let bare =
            r#"[{"name":"flask","vulns":[{"id":"X"},{"id":"Y"}]},{"name":"jinja2","vulns":[]}]"#;
        let r = parse_pip_audit(bare);
        assert_eq!(r.status, ScanStatus::Findings);
        assert_eq!(r.findings, 2);
        let obj = r#"{"dependencies":[{"name":"flask","vulns":[]}]}"#;
        assert_eq!(parse_pip_audit(obj).status, ScanStatus::Clean);
        assert_eq!(parse_pip_audit("not json").status, ScanStatus::Error);
    }

    #[test]
    fn gitleaks_finding_count() {
        assert_eq!(
            count_gitleaks_findings("WRN leaks found: 4\nINF scan completed"),
            4
        );
        // Fallback: count Finding: blocks (older text output).
        assert_eq!(
            count_gitleaks_findings("Finding: AKIA...\nFinding: ghp_..."),
            2
        );
        // Non-zero exit but no parseable count → at least 1.
        assert_eq!(count_gitleaks_findings("something went wrong"), 1);
    }

    #[test]
    fn summary_and_rollups() {
        let scan = SecurityScan {
            timestamp: String::new(),
            results: vec![
                ScanResult {
                    tool: "gitleaks".into(),
                    category: "secrets".into(),
                    status: ScanStatus::Clean,
                    findings: 0,
                    detail: String::new(),
                },
                ScanResult {
                    tool: "npm-audit".into(),
                    category: "dependencies".into(),
                    status: ScanStatus::Findings,
                    findings: 3,
                    detail: String::new(),
                },
                ScanResult::skipped("pip-audit", "dependencies", "absent"),
            ],
        };
        assert!(scan.any_ran());
        assert!(scan.has_findings());
        assert_eq!(scan.total_findings(), 3);
        assert!(scan.summary_line().contains("3 finding"));
    }

    #[test]
    fn all_skipped_summary() {
        let scan = SecurityScan {
            timestamp: String::new(),
            results: vec![ScanResult::skipped("gitleaks", "secrets", "absent")],
        };
        assert!(!scan.any_ran());
        assert!(!scan.has_findings());
        assert!(scan.summary_line().contains("no scanners available"));
    }
}
