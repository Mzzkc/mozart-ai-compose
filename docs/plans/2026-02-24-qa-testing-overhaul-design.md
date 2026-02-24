# QA Testing Overhaul — Design Document

**Date:** 2026-02-24
**Status:** Approved
**Author:** Claude + Emzi

---

## Problem Statement

Mozart has 6,684 tests across 170 test files with 86% line coverage — but the testing
posture is weak:

- **Hollow coverage**: 714 `MagicMock()` instances that return truthy by default, hiding
  real failures. Tests "pass" without testing behavior.
- **Flaky patterns**: 16 test files use `asyncio.sleep` for coordination. 7 tight timing
  assertions (`assert elapsed < 0.01`). 20+ test functions with no assertions.
- **No property-based testing**: No hypothesis, no fuzzing, no adversarial input generation.
- **No CI pipeline**: Only a docs deployment workflow. Tests run locally or not at all.
- **No coverage enforcement**: `fail_under = 0` in pyproject.toml.
- **No overnight smoke testing**: No long-running integration tests against a real conductor.
- **Coverage gaps in critical modules**: Dashboard monitor at 15.9%, CLI validate at 19.6%,
  TUI at 27-52%, daemon manager at 72%.

## Goals

1. **80%+ meaningful coverage** across all modules (not hollow line coverage)
2. **Full adversarial testing** with hypothesis property-based tests for all data models
3. **Zero flaky tests** — no asyncio.sleep, no tight timing, no PID assumptions
4. **Overnight smoke testing** against a real conductor via cron
5. **Enforceable quality standards** via meta-tests that catch regressions

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Score structure | Module-by-module suite | Focused, cheap per run, composable |
| Adversarial approach | Full (hypothesis + parametrize) | Thoroughness over cost savings |
| Overnight execution | Mozart score + cron | Leverages existing infrastructure |
| Existing test handling | Audit & overhaul | Improve signal-to-noise, don't just pile on |

---

## Architecture

### Score Suite

```
scores/qa/
├── qa-foundation.yaml      # Infrastructure: deps, fixtures, conftest, coverage config
├── qa-core.yaml            # Config models, checkpoint, fan-out, templating, errors
├── qa-execution.yaml       # Runner, lifecycle, recovery, cost, backends
├── qa-daemon.yaml          # Manager, scheduler, IPC, event bus, health
├── qa-cli.yaml             # Commands, output, pause/resume, conductor commands
├── qa-dashboard.yaml       # Routes, SSE, auth, job control, security
├── qa-validation.yaml      # Validation checks, rendering, reporter, self-healing
├── qa-integration.yaml     # Cross-module e2e workflows
└── qa-overnight.yaml       # Long-running smoke tests (cron-triggered)
```

### Execution Model

1. `qa-foundation` runs first (always) — installs deps, creates fixtures, sets standards
2. Module scores (core, execution, daemon, cli, dashboard, validation) are independent —
   can run in any order, sequentially, or as a concert
3. `qa-integration` runs after all module scores (depends on their work)
4. `qa-overnight` is separate — designed to run via cron nightly

### Each Module Score: Internal Structure

| Stage | Purpose | Sheets |
|-------|---------|--------|
| 1 | **Audit** — Classify each test as strong/weak/flaky/useless, measure coverage | 1 |
| 2 | **Overhaul** — (fan-out x3) Fix flaky, add property-based, add adversarial | 3 |
| 3 | **Merge** — Resolve conflicts from parallel work | 1 |
| 4 | **Coverage gate** — Verify >= 80%, write targeted tests for gaps | 1 |
| 5 | **Commit** — Stage, commit, push | 1 |

---

## Score Details

### 1. qa-foundation.yaml

**Purpose:** Set up testing infrastructure for the entire suite.

**Stage 1: Dependency setup**
- Add to `[project.optional-dependencies].dev`:
  - `hypothesis>=6.0.0` (property-based testing)
  - `pytest-xdist>=3.0.0` (parallel test execution)
  - `pytest-randomly>=3.0.0` (detect order-dependent tests)
- Run `pip install -e ".[dev]"`
- Update `pyproject.toml`: set `fail_under = 80`

**Stage 2: Conftest overhaul**
- Create `tests/conftest_adversarial.py` with:
  - Hypothesis profiles (ci = fast, nightly = thorough)
  - Property-based strategies for all core Pydantic models
  - Adversarial string generators (unicode, null bytes, path traversal, SQL injection, huge)
  - Adversarial numeric generators (0, -1, MAX_INT, NaN, Inf)
  - `strict_mock()` helper: raises on unexpected calls (unlike MagicMock truthy defaults)
- Update `tests/conftest.py`:
  - Add autouse fixture detecting asyncio.sleep in tests (warning)
  - Add fixture detecting shared mutable state

**Stage 3: Test quality markers**
- Register markers: `adversarial`, `smoke`, `overnight`, `property_based`
- Create `tests/test_quality_gate.py` — meta-test that:
  - Scans all test files for flaky patterns
  - Fails if any test has no assertions
  - Reports mock quality

**Stage 4: Coverage baseline**
- Run full suite, capture per-module coverage
- Create `.coverage-baseline.json`
- Validate infrastructure works

**Stage 5: Commit**

### 2-7. Module Scores (qa-core, qa-execution, qa-daemon, qa-cli, qa-dashboard, qa-validation)

Each follows the template structure. Module-specific scopes:

| Score | Source Scope | Key Adversarial Targets |
|-------|-------------|------------------------|
| qa-core | `core/config/`, `core/checkpoint.py`, `core/fan_out.py`, `core/errors/`, `prompts/` | Garbage YAML config, checkpoint corruption, circular fan-out deps, Jinja injection |
| qa-execution | `execution/runner/`, `execution/escalation.py` | Broken backends, concurrent sheets, partial state recovery, escalation edge cases |
| qa-daemon | `daemon/` (all 20+ modules) | Malformed IPC JSON, 0/negative scheduler priorities, concurrent event bus, health timeouts |
| qa-cli | `cli/` | Garbage CLI args, huge/empty output rendering, pause/resume races |
| qa-dashboard | `dashboard/` | Dropped SSE connections, expired auth, rate limit under load, XSS in job names |
| qa-validation | `validation/`, `healing/` | Circular regex, malformed YAML, all remedies failing simultaneously |

### 8. qa-integration.yaml

Cross-module end-to-end tests. Not mocked — tests real code paths:

- Config parse → validate → runner setup → execute sheet → checkpoint → resume → complete
- Error in sheet → retry → exhausted → self-healing → retry again
- Fan-out expansion → parallel execution → fan-in collection
- Prelude/cadenza injection → template rendering → backend execution

### 9. qa-overnight.yaml

Long-running smoke tests against a real conductor.

| Stage | Test | Duration |
|-------|------|----------|
| 1 | Setup: start conductor, verify health | 1 min |
| 2 | Basic job: submit simple-sheet.yaml, complete | 5 min |
| 3 | Pause/resume: submit, pause, verify, resume, complete | 10 min |
| 4 | Concurrent: 2 simultaneous jobs | 15 min |
| 5 | Error recovery: failing job with retry + heal | 10 min |
| 6 | Crash recovery: kill -9 conductor, restart, resume | 10 min |
| 7 | Cleanup: stop conductor, verify no orphans | 2 min |
| 8 | Report: generate results, notify on failure | 1 min |

**Total: ~55 min**

Cron entry: `0 2 * * *` (2am daily)

Results archived in date-stamped workspace. Failures trigger desktop notifications.

---

## Test Quality Standards

### Mandatory (enforced by quality gate meta-test)

1. No `asyncio.sleep()` for coordination — polling loops with deadlines
2. No timing assertions < 30s — generous bounds only
3. No bare `MagicMock()` — must use `spec=RealClass` or `create_autospec()`
4. Every test function has at least one `assert` or `pytest.raises`
5. No module-level mutable state in tests — fixtures with teardown
6. Property-based tests for every Pydantic model
7. Adversarial inputs for every parser

### Anti-patterns (will fail quality gate)

1. Tests that pass when implementation removed (testing mocks not behavior)
2. `assert isinstance(result, SomeType)` without checking values
3. `assert len(result) > 0` without checking content
4. Tests that mock the thing they're testing

### Test Classification Rubric

- **Strong**: Tests behavior with meaningful assertions, handles edge cases, no timing deps
- **Weak**: Bare MagicMock, trivial assertions, tests implementation not behavior
- **Flaky**: asyncio.sleep coordination, tight timing, PID assumptions, shared state
- **Useless**: No assertions, `assert True`, import-only checks

**Target distribution:** 0 useless, 0 flaky, <10% weak, >90% strong

---

## Coverage Targets

| Level | Target |
|-------|--------|
| Overall project | >= 80% |
| Per-module (each score enforces) | >= 80% |
| Critical paths (config, IPC, state) | >= 95% |

### Current Gaps (baseline from 2026-02-24)

| Module | Current | Target |
|--------|---------|--------|
| dashboard/routes/monitor.py | 15.9% | 80% |
| cli/commands/validate.py | 19.6% | 80% |
| cli/commands/top.py | 27.0% | 80% |
| tui/app.py | 42.2% | 80% |
| tui/reader.py | 52.0% | 80% |
| daemon/profiler/strace_manager.py | 56.1% | 80% |
| cli/commands/status.py | 62.0% | 80% |
| cli/commands/diagnose.py | 65.7% | 80% |
| execution/escalation.py | 68.5% | 80% |
| daemon/manager.py | 72.3% | 80% |

---

## Estimated Cost & Duration

| Score | Sheets | Est. Cost | Est. Duration |
|-------|--------|-----------|---------------|
| qa-foundation | 5 | $3-5 | 30 min |
| qa-core | 7 | $8-12 | 45 min |
| qa-execution | 7 | $8-12 | 45 min |
| qa-daemon | 7 | $12-18 | 60 min |
| qa-cli | 7 | $8-12 | 45 min |
| qa-dashboard | 7 | $8-12 | 45 min |
| qa-validation | 7 | $5-8 | 30 min |
| qa-integration | 5 | $5-8 | 30 min |
| qa-overnight | 8 | $2-3 | 55 min |
| **Total** | **60** | **$60-90** | **~6 hrs** |

Module scores can run in parallel (concert), reducing wall-clock time to ~2-3 hours.

---

*Created: 2026-02-24*
