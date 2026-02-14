# Issue Solver Score — Design Document

**Date:** 2026-02-14
**Status:** Approved

---

## Overview

A self-looping Mozart score that reads a project's roadmap (with issue list and dependency graph), auto-selects the next eligible GitHub issue, plans implementation phases, executes them with quality gates, reviews the work, updates docs, commits, and self-chains to the next issue.

Two deliverables:
1. **New Mozart feature:** `skip_when_command` — command-based conditional sheet execution
2. **Issue solver score:** `examples/issue-solver.yaml` — uses the feature

---

## Part 1: New Feature — `skip_when_command`

### Problem

`skip_when` today evaluates Python expressions against sheet state (`sheets[N].validation_passed`). It cannot read workspace files or run commands. The issue solver needs a planning sheet to declare "this issue needs 2 phases" and have later sheets skip based on that.

### Design

New field on `SheetConfig`:

```yaml
sheet:
  skip_when_command:
    8:
      command: 'grep -q "TOTAL_PHASES: [1]$" "{workspace}/03-plan.md"'
      description: "Skip phase 2 if plan has only 1 phase"
    9:
      command: 'grep -q "TOTAL_PHASES: [1]$" "{workspace}/03-plan.md"'
      description: "Skip phase 2 completion if plan has only 1 phase"
```

**Semantics:** Command returns 0 = **skip the sheet**. Non-zero = **run the sheet**. ("Skip when this command succeeds.")

**Model:**
```python
class SkipWhenCommand(BaseModel):
    command: str
    description: str | None = None
    timeout_seconds: float = Field(default=10.0, gt=0, le=60)
```

**Implementation in `_should_skip_sheet()`:**
1. Check existing `skip_when` (expression) first — unchanged
2. If not skipped, check `skip_when_command` for this sheet number
3. Template-expand `{workspace}` in the command (same as validations)
4. Run via `asyncio.create_subprocess_shell` with timeout
5. Return 0 → skip with reason from `description`
6. Non-zero → run the sheet
7. Timeout/error → log warning, run the sheet (fail-open for safety)

**Why a separate field:** Expression conditions are instant (no subprocess). Command conditions have I/O cost and timeout concerns. Different validation, different error handling. Backwards-compatible.

### Testing

| Layer | What | File |
|-------|------|------|
| Unit | `SkipWhenCommand` model validation | `tests/test_config.py` (extend) |
| Unit | `_should_skip_sheet()` with commands — mock subprocess | `tests/test_skip_when_command.py` |
| Functional | Lifecycle run verifying sheets skip correctly | `tests/test_skip_when_command.py` |
| E2E | Minimal score where sheets skip based on file content | `tests/test_skip_when_command_e2e.py` |

---

## Part 2: Issue Solver Score

### Architecture

17 stages, fan-out on stage 12 (3x parallel), 19 concrete sheets:

```
┌─────────────────────────────────────────────────────────┐
│  ANALYSIS                                                │
│                                                          │
│  Stage 1:  Read roadmap + dependency graph → select      │
│  Stage 2:  Deep investigation                            │
│  Stage 3:  Phase planning + write verify.sh              │
├─────────────────────────────────────────────────────────┤
│  IMPLEMENTATION (4 phases, each fix + completion)        │
│                                                          │
│  Stage 4:  Phase 1 — Fix                                 │
│  Stage 5:  Phase 1 — Completion pass                     │
│  Stage 6:  Phase 2 — Fix         [SKIP if < 2 phases]   │
│  Stage 7:  Phase 2 — Completion  [SKIP if < 2 phases]   │
│  Stage 8:  Phase 3 — Fix         [SKIP if < 3 phases]   │
│  Stage 9:  Phase 3 — Completion  [SKIP if < 3 phases]   │
│  Stage 10: Phase 4 — Fix         [SKIP if < 4 phases]   │
│  Stage 11: Phase 4 — Completion  [SKIP if < 4 phases]   │
├─────────────────────────────────────────────────────────┤
│  QUALITY                                                 │
│                                                          │
│  Stage 12: Fan-out review (3x parallel)                  │
│            [Functional] [E2E/Smoke] [Code Quality]       │
│  Stage 13: Review synthesis → fix findings               │
│  Stage 14: Update docs                                   │
├─────────────────────────────────────────────────────────┤
│  SHIP                                                    │
│                                                          │
│  Stage 15: Final verification                            │
│  Stage 16: Commit & push                                 │
│  Stage 17: Close issue + self-chain gate                 │
└─────────────────────────────────────────────────────────┘
```

### Issue Selection Intelligence (Stage 1)

1. Read `roadmap_file` for dependency graph and tier ordering
2. Query `gh issue list --label "$issue_label" --state open`
3. Cross-reference: only select issues whose dependencies are closed
4. Pick highest tier priority (tier-1 before tier-2)
5. Check `{workspace}/archive/` to skip recently-attempted issues
6. Claim with `gh issue comment`

### Phase Planning (Stage 3)

Breaks the issue into 1-4 implementation phases. Writes:
- `{workspace}/03-plan.md` — detailed phase plan with `TOTAL_PHASES: N` marker
- `{workspace}/verify.sh` — acceptance checks as executable shell script

### Implementation Phases (Stages 4-11)

Each phase follows the quality-continuous pattern:
- **Fix pass:** Target 70%+ completion of the phase's work items
- **Completion pass:** Finish deferred items, run code simplification

Phases 2-4 conditionally skip via `skip_when_command` based on `TOTAL_PHASES` in the plan.

After each phase's completion pass, `verify.sh` runs to check incremental progress (OK to fail partially — phases build incrementally).

### Verification Runner (verify.sh)

Written by Stage 3, specific to the issue. Structure:
```bash
#!/bin/bash
set -euo pipefail
PASS=0; FAIL=0; TOTAL=0
check() {
  TOTAL=$((TOTAL + 1))
  if eval "$2" >/dev/null 2>&1; then
    PASS=$((PASS + 1)); echo "  PASS: $1"
  else
    FAIL=$((FAIL + 1)); echo "  FAIL: $1"
  fi
}
echo "=== Verification: Issue #N ==="
# Functional checks, unit test checks, integration checks, smoke checks
# ...
echo "=== Results: $PASS/$TOTAL passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
```

### Fan-Out Review (Stage 12)

Three parallel reviewers:
- **Instance 1 (Functional):** Writes pytest tests targeting the changes
- **Instance 2 (E2E/Smoke):** Writes smoke test script, runs integration checks
- **Instance 3 (Code Quality):** Wolf prevention, silent failure audit, style review

### Final Verification (Stage 15)

All must pass:
1. `verify.sh` (acceptance checks)
2. `pytest` (full suite including new tests)
3. `ruff check` (lint)
4. `mypy` (type check)

### Self-Chain Lifecycle

```yaml
on_success:
  - type: run_job
    job_path: "examples/issue-solver.yaml"
    detached: true
    fresh: true

concert:
  enabled: true
  max_chain_depth: 30
  cooldown_between_jobs_seconds: 300

workspace_lifecycle:
  archive_on_fresh: true
  max_archives: 30
```

**Termination gate (Stage 17):** A validation that FAILS when no open roadmap issues remain with satisfied dependencies, preventing empty self-chains.

### Validation System

| Stage | Validations | Purpose |
|-------|------------|---------|
| 1 | file_exists + issue number marker | Selection happened |
| 3 | file_exists verify.sh + executable check | Plan + verifier written |
| 4-5 | file_exists + 70% completion | Phase 1 done |
| 6-11 | Same per phase (conditional) | Phases 2-4 |
| 12 | file_exists per review instance | Reviews written |
| 15 | verify.sh + tests + lint pass | Everything passes |
| 16 | Recent commit + push verified | Shipped |
| 17 | Chain gate | Stops loop |

### Configurable Variables

```yaml
variables:
  roadmap_file: "docs/plans/2026-02-14-roadmap-features.md"
  issue_label: "roadmap"
  test_command: "pytest -x -q --tb=short"
  lint_command: "ruff check src/"
  typecheck_command: "mypy src/ --ignore-missing-imports"
  project_root: "."
```

### Testing

| Layer | What | File |
|-------|------|------|
| Schema | `mozart validate examples/issue-solver.yaml` | `tests/test_validation_checks.py` |
| Dry run | 19-sheet plan renders correctly | Manual + CI |
| Functional | Jinja templates render for each stage/instance | `tests/test_score_templates.py` |

---

## Dependencies

- `skip_when_command` feature must be implemented before the score can use it
- Score file (`examples/issue-solver.yaml`) depends on the feature being in the codebase

## Implementation Order

1. Implement `skip_when_command` feature (model, lifecycle, tests)
2. Write the issue solver score
3. Validate and test end-to-end

---

*Approved: 2026-02-14*
