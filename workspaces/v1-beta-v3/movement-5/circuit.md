# Movement 5 — Circuit Report

**Date:** 2026-04-05
**Focus:** Systems integration, debugging, observability

---

## Summary

Three findings resolved, one mitigated, 14 TDD tests written, 7 existing tests updated. Two systems-level bugs fixed at their architectural root, not their symptoms. Meditation written.

---

## Work Completed

### F-149 RESOLVED (P1): Backpressure Cross-Instrument Rejection

**Root cause:** `BackpressureController.should_accept_job()` (`backpressure.py:163`) delegated to `current_level()` which escalated to `PressureLevel.HIGH` when `self._rate_coordinator.active_limits` was non-empty. HIGH caused all new job submissions to be rejected — regardless of which instrument was rate-limited. A rate limit on `claude-cli` blocked jobs targeting `gemini-cli`.

**Fix:** Separated job-level gating from sheet-level dispatch:
- `should_accept_job()` now only checks resource pressure (memory >85%, process limit, monitor degraded). Rate limits ignored.
- `rejection_reason()` returns `"resource"` or `None`. The `"rate_limit"` return value is eliminated.
- `current_level()` unchanged — sheet-level dispatch still factors rate limits for pacing.
- Manager (`manager.py:595`) simplified: the `reason == "rate_limit"` → PENDING queue path removed. Jobs go straight through.

**Architecture principle:** Job-level gating = system health (memory, processes). Sheet-level dispatch = per-instrument concerns (rate limits, model availability). Each concern at its correct scope.

**Evidence:**
- `tests/test_f149_cross_instrument_rejection.py`: 10 TDD tests — core behavior (rate limits don't block jobs), resource pressure still works, sheet dispatch still considers rate limits, critical paths unaffected.
- Updated 3 tests in `test_daemon_backpressure.py`, 2 in `test_rate_limit_pending.py`, 1 in `test_m4_adversarial_breakpoint.py`, 1 in `test_litmus_intelligence.py` — all previously asserted the OLD (incorrect) behavior.
- `python -m pytest tests/test_daemon_backpressure.py tests/test_f149_cross_instrument_rejection.py` → 67 passed.

**Files changed:**
- `src/mozart/daemon/backpressure.py:163-212` — `should_accept_job()` and `rejection_reason()` rewritten
- `src/mozart/daemon/manager.py:595-606` — rate_limit→PENDING path removed
- `tests/test_f149_cross_instrument_rejection.py` — NEW, 10 tests
- `tests/test_daemon_backpressure.py` — 3 tests updated
- `tests/test_rate_limit_pending.py` — 2 tests updated
- `tests/test_m4_adversarial_breakpoint.py` — 1 test updated
- `tests/test_litmus_intelligence.py` — 1 test updated

### F-451 RESOLVED (P2): Diagnose Can't Find Completed Jobs

**Root cause:** `diagnose` (`diagnose.py:733`) caught `JobSubmissionError` from the conductor and immediately exited with "Score not found" — even when a workspace flag was provided. The `status` command handles this case by falling back to filesystem search, but `diagnose` didn't.

**Fix:**
- When conductor returns `JobSubmissionError` and `-w` workspace is provided, fall back to `_find_job_state_direct()` filesystem search (`diagnose.py:738-742`).
- When no workspace provided, error hints now mention `-w` flag.
- The `-w` flag is now visible (not hidden) in `diagnose --help`.

**Evidence:**
- `tests/test_f451_diagnose_workspace_fallback.py`: 4 TDD tests — fallback works, no-workspace still errors, filesystem failure propagates, hints mention -w.
- `python -m pytest tests/test_f451_diagnose_workspace_fallback.py` → 4 passed.

**Files changed:**
- `src/mozart/cli/commands/diagnose.py:657-753` — workspace fallback, -w unhidden, hint updated
- `tests/test_f451_diagnose_workspace_fallback.py` — NEW, 4 tests

### F-471 MITIGATED (P2): Pending Jobs Lost on Restart

**Analysis:** F-149's fix eliminated the primary trigger for PENDING jobs (rate-limit rejection). Since `should_accept_job()` no longer returns False for rate limits, the `_queue_pending_job` path in `submit_job()` is no longer reachable for the rate-limit case. The PENDING infrastructure remains for potential future use (resource-pressure queueing) but the architectural gap that F-471 described is moot for the production case.

### Meditation

Written to `meditations/circuit.md`. Theme: the system between the signals — how correct subsystems compose into incorrect behavior at boundaries, and how discontinuity resets the filters that familiarity builds.

---

## Mateship

- Verified all M5 commits: Foundation (D-026), Canyon (D-027), Blueprint (F-430, F-202), Maverick (F-470, F-431, user variables).
- All 23 teammate tests pass (`test_f271_mcp_disable.py`, `test_f255_2_live_states.py`, `test_d027_baton_default.py`, `test_user_variables_in_validations.py`).

---

## Quality Verification

```
pytest tests/ (targeted: 94 passed, 0 failed)
mypy src/ — clean (no errors)
ruff check src/ — all checks passed
```

---

## Patterns Observed

**Correct subsystems, incorrect composition.** F-149 is the fifth instance of this pattern (after F-065, F-068, D-024, the finding-fix pipeline). Each component in the backpressure system was individually correct. The rate coordinator accurately tracked limits. The pressure controller correctly mapped limits to levels. The job gate correctly rejected at HIGH. But the implicit assumption — "any rate limit means system-wide pressure" — was wrong. The bug lived in the space between components, not in any of them.

**Scope separation solves composition bugs.** The fix wasn't to add complexity (instrument-aware checking at every level). It was to simplify — to recognize that job-level gating and sheet-level dispatch are different concerns operating at different scopes. Resource pressure is system-wide; rate limits are per-instrument. Once the scopes were separated, the bug disappeared and the code got simpler.

**The gap between commands.** F-451 is a UX observability gap. `status -w` finds the job. `diagnose` can't. The user's natural debugging path breaks. This class of bug — inconsistency between related commands — is easy to miss because each command works correctly in isolation. You only find it by walking the user's path.
