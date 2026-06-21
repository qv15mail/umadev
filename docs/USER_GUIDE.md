# UmaDev User Guide

## What is UmaDev?

UmaDev is a **governance rail around the AI coding base you already use**. It drives your already-logged-in base CLI — exactly three are first-class: **Claude Code, Codex, OpenCode** — through a 9-phase commercial delivery pipeline, keeping the base's output on track against real company standards. (Want a different model? That is the base's job — route it to a third-party / local model in the base's own config; UmaDev does not add new drivers for that.)

UmaDev itself does NOT write code, and it is **not a fully autonomous director that replaces the engineer**. The brain stays in the base; UmaDev tells the base WHAT to produce at each phase, checks the result, and leaves an evidence trail. It is still early-stage and governance-first — best validated on real projects.

## Quick Start

```bash
# 1. Install
npm install -g umadev

# 2. Initialize a project
cd my-project
umadev init

# 3. Launch the TUI
umadev
```

On first launch, pick your base (one of the three AI coding CLIs you've already logged into: Claude Code, Codex, or OpenCode). Then type your requirement and press Enter.

## The 9-Phase Pipeline

```
research → docs → ⏸ docs_confirm → spec → frontend → ⏸ preview_confirm → backend → quality → delivery
```

| Phase | What happens | Expert role |
|---|---|---|
| research | Competitive analysis, user discovery, design direction | Product Researcher |
| docs | PRD + Architecture + UI/UX design system | PM + Architect + Designer |
| docs_confirm | **GATE** — you review the 3 docs before coding starts | You |
| spec | Sprint breakdown, coding standards, task list | Engineering Manager |
| frontend | Base implements frontend with approved design tokens | Frontend Lead |
| preview_confirm | **GATE** — you review the frontend before backend | You |
| backend | Base implements API routes, database, auth, tests | Backend Lead |
| quality | 17 automated checks + 5-dimension visual review | QA Lead |
| delivery | Proof-pack zip with README + compliance mapping | Release Engineer |

> The 9 phases target a full commercial-grade delivery. Small tasks have a lighter path: declare the task type with `/kind` (full-stack / frontend-only / backend-only / bugfix / refactor) and UmaDev trims the phases — a bugfix is not pushed through the whole PRD / architecture / UIUX chain.

## TUI Commands

### Base
| Command | Description |
|---|---|
| `/claude` | Switch to Claude Code CLI |
| `/codex` | Switch to Codex CLI |
| `/opencode` | Switch to OpenCode CLI |
| `/offline` | Offline templates — internal CI / no-base fallback, not a product mode |

### Design
| Command | Description |
|---|---|
| `/design` | Browse available design systems |
| `/design <name>` | Select a design system |
| `/template <name>` | Select a seed template |
| `/model <id>` | Set the AI model |

### Pipeline
| Command | Description |
|---|---|
| `/continue` or `c` | Approve the current gate |
| `/revise <text>` | Request changes at a gate |
| `/run [slug] <req>` | Start a new run |
| `/redo` | Re-run current requirement |
| `/diff <artifact>` | View an artifact (prd/architecture/uiux) |

### Inspect
| Command | Description |
|---|---|
| `/status` | Pipeline progress + quality score |
| `/export` | Export proof-pack |
| `/config` | View all settings |
| `/knowledge` | Browse knowledge files |
| `/doctor` | Self-test |
| `/verify` | Workspace conformance |

### General
| Command | Description |
|---|---|
| `/help` | All commands |
| `/clear` | Clear chat history |
| `/quit` | Exit |

## Design Systems

UmaDev ships 5 design systems. Select one before running to get deterministic visual output:

| Name | Best for |
|---|---|
| `modern-minimal` | SaaS, dev tools, dashboards |
| `editorial-clean` | Blogs, content sites, portfolios |
| `tech-utility` | CLI companions, monitoring, data tools |
| `soft-warm` | Consumer apps, education, wellness |
| `bold-geometric` | Brand launches, creative agencies |

## Seed Templates

| Name | Structure |
|---|---|
| `saas-landing` | Nav → Hero → Trust → Features → Pricing → FAQ → Footer |
| `dashboard` | Sidebar + KPI cards + Charts + Data table |
| `blog-content` | Featured article + Grid + Newsletter |
| `e-commerce` | Gallery + Product info + Variants + Reviews + Related |
| `auth-system` | Login + Signup + Forgot + MFA + Reset |
| `settings-page` | Sidebar tabs + Profile + Security + Billing |
| `docs-site` | Sidebar nav + Content + Code blocks + Search |

## Configuration

### `.umadevrc` (project-level)

```toml
[quality]
threshold = 85              # quality gate pass threshold (default: 90)
skip_checks = ["dark_mode"] # skip specific checks

[pipeline]
skip_phases = ["research"]  # skip phases you don't need
max_review_rounds = 2       # limit auto-fix cycles (default: 3)

[experts]
custom_knowledge = "team-standards/"  # additional knowledge directory
```

### `~/.umadev/config.toml` (user-level)

```toml
backend = "claude-code"
model = "claude-sonnet-4-6"
design_system = "modern-minimal"
seed_template = "dashboard"
```

## Quality Gate

UmaDev runs 17 automated checks:

| Category | Checks |
|---|---|
| Artifacts | Research, PRD, Architecture, UIUX — content structure validation |
| Cross-reference | PRD↔Architecture route alignment, API URL consistency |
| Code quality | Emoji check, hardcoded colors, anti-AI-slop patterns |
| Design | UIUX token count, dark mode presence, design system completeness |
| Evidence | Audit log, tool-call log, discovery section |
| Depth | Acceptance criteria count, API route count |

## Expert Knowledge

Each pipeline phase is backed by a specialist's methodology:

| Expert | Knowledge | Used in |
|---|---|---|
| Product Manager | RICE scoring, AC format, edge cases, HEART metrics | Research, PRD |
| Architect | API design standards, security checklist (OWASP), auth patterns | Architecture |
| UI/UX Designer | Token architecture, interaction principles, WCAG 2.1, responsive | UIUX, Frontend |
| Frontend Lead | Component architecture, state management, error handling, performance | Frontend |
| Backend Lead | API handler pattern, database practices, JWT flow, logging standards | Backend |
| QA Lead | Test pyramid, AC→test conversion, pre-release checklist | Quality |
| DevOps | CI/CD pipeline, Docker, monitoring, rollback strategy | Delivery |

## FAQ

**Q: Do I need an API key?**
No. UmaDev drives your already-logged-in AI coding CLI. It uses your existing subscription.

**Q: What if the base times out?**
UmaDev retries once. If it still fails, it falls back to an offline template with TODO markers. You can `/redo` to try again.

**Q: Can I customize the quality checks?**
Yes, via `.umadevrc`. Set `skip_checks` to disable specific checks, or `threshold` to change the pass score.

**Q: Does it work offline?**
Offline is a fallback, not the product. Without a base reachable, it generates structured templates with TODO markers — useful for planning, CI smoke tests, or demos, but not a substitute for real development. Real delivery always runs through one of the three bases.
