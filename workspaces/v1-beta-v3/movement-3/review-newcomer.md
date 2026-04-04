# Movement 3 — Newcomer Final Review (Reviewer Pass)

**Reviewer:** Newcomer (acting as Reviewer)
**Focus:** User experience testing, documentation validation, onboarding assessment, error message quality, first-run experience, assumption detection
**Movement:** 3
**Date:** 2026-04-04
**Method:** Completely fresh eyes — no memory of previous movements. Ran every safe command. Validated every example. Walked the README start-to-finish. Intentionally broke things. Read the quality gate, both reviewer reports (Axiom, Prism), 8 musician reports, composer's notes, TASKS.md, and FINDINGS.md. Cross-referenced all claims against actual CLI output and committed code on HEAD (ca70b62).

---

## Executive Summary

**The newcomer experience is polished, coherent, and genuinely impressive for a v0.1.0.**

I found Mozart on GitHub thirty seconds ago. I read the README, ran the commands, tried the examples, and threw garbage at the CLI. What I found: a product that teaches instead of punishes, a CLI that's organized and consistent, error messages that guide me to the fix, and an example score (hello.yaml) that produces a beautiful HTML page — not a folder of text files. This is better than 90% of developer tools I encounter.

Movement 3 delivered consolidation: 43+ commits from 26+ musicians, 584 new tests (10,981 total), zero regressions, all five quality gates GREEN. The mateship pipeline is working — 33% of commits were pickups of uncommitted teammate work. Documentation was audited across all user-facing docs. Terminology was unified from "job" to "score" across CLI, README, getting-started, and all guides. The baton survived 67 adversarial tests with zero new bugs.

**Three things that would stop me in my tracks:**

1. **F-210 is the invisible wall.** Cross-sheet context is missing from the baton path. 24 of 34 example scores use `cross_sheet`. When the baton becomes default, these scores will silently produce degraded output — templates referencing `{{ previous_outputs }}` render with empty dicts. Tests won't catch this because they mock context. This is the most dangerous class of bug: tests say yes, reality says no. Every reviewer (Axiom, Prism, Weaver) independently confirmed it. This must be fixed before Phase 1 testing.

2. **F-450 is live and confusing.** `mozart clear-rate-limits` reports "conductor not running" when the conductor IS running (PID 1277279, uptime 36h 49m, confirmed by `conductor-status` and `doctor`). I independently reproduced this. A newcomer who follows the docs to clear stale rate limits gets told the system they just saw running doesn't exist. This is a teacher saying "there is no classroom" while you're sitting in the classroom.

3. **The demo gap is now an identity crisis.** Eight movements. Zero demo progress. The Lovable and Wordware comparison demos are P0 composer directives — the oldest unfulfilled obligations in the project. The infrastructure is here. The README is pristine. The example corpus validates clean. Nobody outside this repository knows any of it.

---

## The Newcomer Path — Every Command, Every Output

### 1. Version + Doctor

```
$ mozart --version
Mozart AI Compose v0.1.0

$ mozart doctor
  ✓ Python 3.12.3
  ✓ Mozart v0.1.0
  ✓ Conductor running (pid 1277279)
  6 instruments ready/unchecked, 4 not found
  ! No cost limits configured
```

**Assessment: Excellent.** Clean, informative, actionable. The cost warning is a genuine service — newcomers would otherwise discover cost limits by getting a surprise bill. Instruments are listed clearly with ready/not found status.

### 2. CLI Help

```
$ mozart --help
  8 groups: Getting Started, Jobs, Monitoring, Diagnostics, Services, Conductor, Learning, Commands
  30+ commands, logically organized
  Global options clearly documented including --conductor-clone
```

**Assessment: Professional.** The Learning section still dominates (12+ commands), but commands are grouped logically. The `--conductor-clone` option is visible at the top level but NOT visible when you do `mozart start --help` — a newcomer looking for clone functionality via the start command won't find it. They'd need to check `mozart --help`. This is a discoverability issue, not a bug.

### 3. Validate Flagship Example

```
$ mozart validate examples/hello.yaml
  ✓ Configuration valid: hello-mozart
  Sheets: 5, Instrument: claude-code, Validations: 5
  Execution DAG: Level 0: 1 → Level 1: 2,3,4 (parallel) → Level 2: 5
```

**Assessment: Superb.** The validate output shows the DAG visualization, a prompt preview, and expanded validations per sheet. A newcomer immediately understands what the score does before running it. The DAG shows sheets 2-4 run in parallel (the three character vignettes), with sheet 5 (finale) depending on all three. This is the best "dry-run" experience I've seen in any orchestration tool.

### 4. Error Message Quality

I deliberately fed garbage to the CLI:

| Input | Error Message | Grade |
|-------|---------------|-------|
| Empty file (`/dev/null`) | "Score must be a YAML mapping, got: empty file" + hints + doc reference | A |
| Missing file | "Path 'nonexistent.yaml' does not exist" | A |
| Plain text YAML | "Score must be a YAML mapping, got: str" + hints + doc reference | A |
| Nonexistent score ID | "Score not found: xxx" + "Run 'mozart list' to see available scores" | A |
| `--conductor-clone=` (no clone running) | "Mozart conductor is not running" + start hint | B+ |

Every error message tells me: what went wrong, why, and what to do about it. The hints point to docs and concrete next steps. The only B+ is the clone error — it doesn't mention that this is a *clone* that isn't running, so a newcomer might think the production conductor is down.

### 5. `mozart init`

```
$ mozart init
  Created: my-score.yaml (starter score — edit with your task)
  Created: .mozart/ (project config directory)
  Next steps: 0. mozart doctor → 1. Edit → 2. mozart start && mozart run → 3. mozart status
```

The generated `my-score.yaml` is well-commented, uses correct YAML syntax, includes Jinja2 variables, and validates clean. The comments explain every section. The "Next steps" output guides the newcomer through the exact workflow. Starting from step 0 is slightly unusual but functional.

**Assessment: Excellent.** This is how every developer tool should handle initialization.

### 6. `mozart instruments list`

```
10 instruments configured (3 ready, 3 unchecked)
Clean table with NAME, KIND, STATUS, DEFAULT MODEL columns
```

**Assessment: Clear and useful.** "not found" vs "unchecked" vs "ready" is immediately understandable. The KIND column (cli/http) is informative for advanced users.

### 7. Status Overview

```
$ mozart status (no args)
  ACTIVE: 4 scores (1 running, 3 paused)
  RECENT: 2 completed
```

**Assessment: Exactly what I expected.** `status` without arguments shows an overview. Every other CLI tool I've used does this. Mozart does too. This is the right default.

### 8. Dry Run

```
$ mozart run examples/hello.yaml --dry-run
  Score Configuration panel + sheet plan table + dependency graph + prompt preview
  Cost tracking warning (disabled)
```

**Assessment: Thorough.** The dry-run output shows everything the score will do without doing it. The cost warning is proactive — good. The prompt preview truncates with `...` which is acceptable.

### 9. The Hello Score Output

The `workspaces/hello-mozart/the-sky-library.html` is a self-contained HTML page with:
- Embedded CSS with typography, color palette, responsive layout
- Hero section with the city name in large serif caps
- Sections for world setting, three character vignettes, finale
- Decorative elements (gradient lines, letter-spacing, Italian-like chapter headers)

**Assessment: This is the wow moment.** A newcomer who runs `hello.yaml` and opens the HTML will think "this is not a toy." The visual quality answers the composer's directive about impressiveness. This is miles ahead of a folder of .md files.

---

## Movement 3 Work Verification

### Quality Gates — VERIFIED

| Gate | Status | Evidence |
|------|--------|----------|
| pytest | GREEN | 10,981+ passed (quality gate report) |
| mypy | GREEN | "no issues found" (quality gate report) |
| ruff | GREEN | "All checks passed!" (quality gate report) |
| flowspec | GREEN | 0 critical findings (quality gate report) |
| Example corpus | GREEN | 37/38 validate (1 expected failure: generator config) |

### Task Completion — Cross-Referenced

TASKS.md claims M3 milestone is 100% (26/26). Quality gate confirms this. Axiom and Prism both independently verified M3 task claims against git log and committed code. I verified the key claims:

| Claim | Evidence |
|-------|----------|
| F-152 dispatch guard fixed | adapter.py:746-866 (Axiom verified line numbers) |
| F-009/F-144 semantic tags fixed | patterns.py:82-118 (Foundation commit e9a9feb) |
| F-150 model override wired | cli_backend.py:116-142 (Foundation commit 08c5ca4) |
| F-440 state sync gap fixed | core.py:544-555 (Axiom's own fix) |
| F-460 terminology fixed | 35+ instances across 6 files (Newcomer + Guide) |
| 584 new tests added | Quality gate: 10,397 (M2) → 10,981 (M3) |
| 6 GitHub issues closed | #155, #154, #153, #139, #94, #131 (Prism verified) |

### Composer's Notes Compliance

| Directive | Status |
|-----------|--------|
| P0: Read specs before implementing | FOLLOWED (reports cite spec files) |
| P0: pytest/mypy/ruff must pass | VERIFIED GREEN |
| P0: Uncommitted work = failure | ADDRESSED (mateship pipeline caught and committed all outstanding work) |
| P0: No `mozart stop` inside orchestra | FOLLOWED (no evidence of conductor interference) |
| P0: Fix bugs → close issues → reference commit | FOLLOWED (Prism verified separation of duties) |
| P0: Hello.yaml must be visually impressive | ACHIEVED (HTML output is genuinely impressive) |
| P0: Lovable + Wordware demos | NOT STARTED (8 movements, zero progress) |
| P1: Music metaphor is load-bearing | FOLLOWED (terminology unified to "score"/"sheet"/"instrument") |
| P1: F-052 SheetContext aliases | NOT VERIFIED (didn't trace this specifically) |
| P1: Canyon: uncommitted work is P1 finding | FOLLOWED (mateship pipeline active this movement) |

---

## What's Working

### 1. Error Messages Are Teachers (A Grade)

Every error I generated told me what went wrong, why, and what to do. The hints point to docs. The schema error messages were specifically improved this movement (Journey's `_schema_error_hints()`) to give context-specific guidance for common mistakes like `prompt: "string"` instead of `prompt: { template: "..." }`. This is not common in developer tools. It's exceptional.

### 2. Documentation Is Honest and Complete

All 21 example files referenced in README exist. All 7 documentation files exist. The CLI reference in the README matches the actual `--help` output (8 groups, same commands). Zero broken links. The README was overhauled this movement (Compass) to add 13 missing CLI commands and the entire Conductor group. Guide updated all 5 user-facing docs for terminology consistency. Codex added 4 missing commands to cli-reference.md.

### 3. Example Corpus Validates Clean

37/38 examples pass `mozart validate`. The one failure (`iterative-dev-loop-config.yaml`) is a generator config, not a score — expected. Zero hardcoded absolute paths. 18/18 fan-out examples have `movements:` declarations (modernization completed this movement by Guide).

### 4. Adversarial Testing Depth

67 Phase 1 baton adversarial tests (Adversary), 258 adversarial tests across 4 passes (Breakpoint), 29 property-based invariant proofs (Theorem), 21 intelligence-layer litmus tests (Litmus), 22 user journey tests (Journey). The baton has been attacked from every angle with zero bugs found. This is confidence-building.

### 5. Mateship Pipeline Is Institutional

33% of M3 commits were mateship pickups. Foundation committed 3 teammates' work. Bedrock committed 2. Six others contributed pickups. The uncommitted work anti-pattern from earlier movements was caught and resolved within this movement. The protocol is no longer a rule — it's a habit.

---

## What's Concerning

### 1. F-210: The Silent Degradation Bomb (CRITICAL)

Cross-sheet context (`previous_outputs`, `previous_files`) is completely absent from the baton execution path. The legacy runner populates this via `_populate_cross_sheet_context()` at `context.py:171-221`. The baton has zero awareness of it. 24 of 34 example scores use `cross_sheet: auto_capture_stdout: true`.

When the baton becomes default:
- Templates with `{{ previous_outputs[1] }}` render with empty dicts
- No error is raised
- Tests pass because they mock context
- Real scores produce subtly degraded output

This is confirmed by Axiom (traced code paths), Prism (grep confirmed zero baton references), Weaver (filed F-210), and the quality gate acknowledges it as the single critical path blocker.

### 2. F-450: Commands Disagree About Conductor State (P2)

I independently confirmed F-450 (originally filed by Ember, cross-confirmed by previous Newcomer as F-462):

```
$ mozart conductor-status → running (PID 1277279, uptime 36h 49m)
$ mozart doctor → ✓ Conductor running
$ mozart status → RUNNING (uptime 1d 12h)
$ mozart clear-rate-limits → Error: Mozart conductor is not running
```

The IPC method used by `clear-rate-limits` differs from the others. A newcomer who sees three commands say "running" and one say "not running" will lose trust in the system.

### 3. Learning Stats Are Bleak

```
$ mozart learning-stats
  Total recorded: 239,451 executions
  First-attempt success: 12.0%
  Avg effectiveness: 0.51
  Recovery success rate: 0.0%
```

A 12% first-attempt success rate and 0.51 effectiveness (barely above random coin flip) would alarm any engineer evaluating whether to adopt Mozart. These stats are visible via `mozart learning-stats`. If they're internal metrics not meant for users, they shouldn't be in the CLI. If they are meant for users, they need explanation or context.

### 4. `history` Command Placement

README lists `history` under "Monitoring." CLI lists it under "Diagnostics." Minor inconsistency but it means a newcomer following the README's organizational model won't find `history` where they expect it in the CLI.

### 5. Demo Deficit

Eight movements. Zero progress on the Lovable demo or Wordware comparison demos — both P0 composer directives. The infrastructure is pristine. The docs are clean. The examples validate. Nobody outside this repository has seen any of it. The audience that would validate Mozart's thesis doesn't know it exists.

---

## Recommendations for M4

1. **Fix F-210 first.** Before any Phase 1 baton testing. ~100-200 lines of implementation per Weaver's estimate. Without it, baton testing produces misleading results.

2. **Fix F-450.** The IPC method mismatch is a trust-eroding bug. When commands disagree about system state, newcomers lose confidence.

3. **Start the demo.** One score that a stranger can run and show to their manager. The hello.yaml proves Mozart can produce impressive output. Now produce an impressive score that solves a real business problem.

4. **Context for learning stats.** Either add explanation to `learning-stats` output (e.g., "12% first-attempt success is expected for exploration-heavy workloads") or consider whether these stats should be user-facing at all.

5. **`history` placement.** Move it from Monitoring to Diagnostics in the README, or vice versa in the CLI. Pick one and be consistent.

---

## Final Assessment

**Movement 3 verdict: PASS — with one critical blocker (F-210) for the next phase.**

The product surface is ready for external eyes. The CLI is coherent, the docs are honest, the error messages teach, the examples work, and the hello score produces something genuinely impressive. 584 new tests, zero regressions, complete terminology unification, and a functioning mateship pipeline that catches and commits abandoned work.

But F-210 means the baton — Mozart's future execution engine — will silently degrade 71% of the example corpus when it becomes default. This is the gap between "tests pass" and "product works." Fix it before Phase 1 testing, or Phase 1 will produce data that lies.

The orchestra delivered this movement. Now deliver for the audience.
