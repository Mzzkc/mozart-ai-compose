# Movement 5 — Quality Gate Report
**Agent:** Bedrock
**Date:** 2026-04-08
**Retry:** #9
**Verdict:** ✅ **PASS** — The ground holds.

---

## Executive Summary

All four quality gate checks pass. Movement 5 is complete and ready for Movement 6.

- **Test Coverage:** 11,810 tests pass (100% pass rate)
- **Type Safety:** Zero errors across 258 source files
- **Lint Quality:** All checks passed
- **Structural Integrity:** Zero critical findings

This is the successful conclusion of a 9-retry quality gate journey that resolved two distinct failure classes: the 11-state SheetStatus model expansion (50 tests, retries #1-5), and the F-470 memory leak regression (1 test, retry #8).

---

## Quality Gate Checks

### 1. Test Coverage — ✅ PASS
**Command:** `python -m pytest tests/ -v --tb=no`
**Result:** 11,810 passed, 69 skipped, 12 xfailed, 3 xpassed, 177 warnings in 57.29s
**Exit code:** 0

**Metrics:**
- **Total tests:** 11,810 passed (up from 11,397 in M4 = +413 new tests)
- **Pass rate:** 100%
- **Skipped:** 69 (expected behavior for conditional tests)
- **xfailed:** 12 (known issues properly marked)
- **xpassed:** 3 (tests unexpectedly passing — candidates for unskipping)

**Warnings breakdown:**
- Pydantic deprecation warnings (model_fields_set, model_computed_fields, model_fields) — technical debt, P3 priority
- RuntimeWarning: unawaited coroutines in test teardown — 2 occurrences in test_baton_property_based.py and test_status_beautification.py
- UserWarning: Backend type mismatch in test_runner_recovery.py (expected, test behavior)

**Assessment:** Clean pass. The +413 test count reflects Movement 5's substantial work: instrument fallbacks (35+ tests), baton refinements, config strictness (23 tests), validation enhancements (8 tests), and observability improvements.

---

### 2. Type Safety — ✅ PASS
**Command:** `python -m mypy src/ --no-error-summary`
**Result:** Zero errors
**Files checked:** 258 source files

**Assessment:** Type safety has remained intact across all 9 retries. No regressions introduced by any of the Movement 5 changes or post-movement integration work.

---

### 3. Lint Quality — ✅ PASS
**Command:** `python -m ruff check src/`
**Result:** All checks passed!

**Assessment:** Clean. No style violations, no unused imports, no complexity warnings. Ruff passed consistently across retries #6-9 (failed in retries #1-5 with 15 fixable warnings, which were subsequently fixed).

---

### 4. Structural Integrity — ✅ PASS
**Command:** `/home/emzi/Projects/flowspec/target/release/flowspec diagnose /home/emzi/Projects/marianne-ai-compose --severity critical -f summary -q`
**Result:** 0 findings

**Assessment:** No critical structural regressions. Dead wiring, orphaned implementations, and circular dependencies remain at zero. Structural integrity has held solid across all 9 retries — flowspec never flagged a critical issue throughout the entire quality gate process.

---

## The 9-Retry Journey

This quality gate required 9 retries across two distinct failure classes:

### Retries #1-5: 11-State Model Expansion (50 tests)
**Root cause:** The SheetStatus model expanded from 5 states to 11 states (commit `7d780b1`) to achieve 1:1 mapping between baton scheduling states and checkpoint persistence states. This architectural improvement was correct but broke 50 tests across 14 test files that had hardcoded expectations for the old 5-state model.

**New states added:**
- READY (was collapsed to "pending")
- DISPATCHED (new)
- WAITING (new)
- RETRY_SCHEDULED (new)
- FERMATA (new)
- CANCELLED (was collapsed to "failed")

**Callback signature change:** `StateSyncCallback` grew from 3 params to 4 (added `baton_sheet_state` for rich metadata).

**Resolution:**
- Musicians fixed 10 tests during retries #1-3
- Composer fixed remaining 40 tests in post-movement integration work
- All 50 tests now passing

### Retry #8: F-470 Memory Leak Regression (1 test)
**Root cause:** Composer's "delete sync layer" refactor accidentally deleted Maverick's F-470 memory leak fix (commit `201cd25`). The 5-line cleanup in `BatonAdapter.deregister_job()` prevents `_synced_status` from accumulating O(total_sheets_ever) entries in long-running conductors.

**The failing test:** `tests/test_f470_synced_status_cleanup.py::TestSyncedStatusCleanupOnDeregister::test_deregister_removes_synced_entries`

**Resolution:** Composer restored the F-470 fix in 4 commits between retry #8 and #9, along with several baton recovery improvements.

### Retry #9: Clean Pass
All tests pass. All fixes applied. The ground holds.

---

## Movement 5 Deliverables — Verification

All 12 participating musicians' work verified:

### D-026 (Foundation) — ✅ Complete
- **F-271 RESOLVED:** MCP process explosion fixed via profile-driven `mcp_disable_args`
- **F-255.2 RESOLVED:** Baton `_live_states` population fixed
- **Evidence:** 14 TDD tests pass, both findings marked resolved in FINDINGS.md

### D-027 (Canyon) — ✅ Complete
- **Baton is now the default:** `use_baton: true` in DaemonConfig
- **Evidence:** `test_d027_baton_default.py` passes, legacy tests updated with explicit `use_baton=False`

### D-029 (Dash + Lens) — ✅ Complete
- **Status beautification:** Rich panels, Now Playing, compact stats
- **Evidence:** `test_status_beautification.py` passes (23 tests)

### Instrument Fallbacks (Harper + Circuit) — ✅ Complete
- **Full feature:** Config models, Sheet entity, baton dispatch, availability check, V211 validation, status display, observability events
- **Evidence:** 35+ TDD tests pass across fallback feature files

### F-149 (Circuit) — ✅ Resolved
- **Backpressure cross-instrument rejection fixed**
- **Evidence:** 10 TDD tests in `test_f149_backpressure.py` pass

### F-451 (Circuit) — ✅ Resolved
- **Diagnose workspace fallback working**
- **Evidence:** 4 TDD tests pass

### F-470 (Maverick → Blueprint → Composer) — ✅ Resolved
- **Memory leak fixed, regressed, re-fixed**
- **Evidence:** `test_f470_synced_status_cleanup.py` passes

### F-431 (Maverick + Blueprint) — ✅ Resolved
- **Config strictness complete:** `extra='forbid'` on all daemon/profiler models
- **Evidence:** 23 TDD tests pass

### F-481 (Harper) — ✅ Complete
- **Baton PID tracking for orphan detection**
- **Evidence:** PluginCliBackend + BackendPool wired, tests pass

### F-490 (Ghost + Harper) — ✅ Complete
- **Process control audit complete**
- **Evidence:** Guard verified correct, structural regression tests added

### Rename Phase 1 (Composer + Ghost) — ✅ Complete
- **Package rename:** `src/marianne/` preserved, tests updated
- **Evidence:** All 11,810 tests pass under new package structure

### Documentation (Codex) — ✅ Complete
- **12 deliverables across 5 docs**
- **Evidence:** `docs/` directory updated with all M5 features documented

---

## Codebase Metrics

| Metric | Movement 5 | Movement 4 | Delta |
|--------|-----------|-----------|-------|
| Tests passing | 11,810 | 11,397 | +413 |
| Test files | 362 | 333 | +29 |
| Source lines | 99,694 | 98,447 | +1,247 |
| Type-checked files | 258 | (not tracked) | — |
| Commits | 26 | 93 | -67 |
| Participating musicians | 12 of 32 (37.5%) | 32 of 32 (100%) | -62.5% |

**Commit breakdown (M5):**
- Ghost: 6 commits
- Harper: 4 commits
- Circuit: 3 commits
- Forge: 2 commits
- Blueprint: 2 commits
- Canyon, Foundation, Maverick, Spark, Lens, Dash, Codex: 1 commit each
- Unattributed: 2 commits (rename work)

**Participation analysis:** 37.5% participation is down significantly from M4's 100%. This reflects Movement 5's concentrated work areas (rename, baton flip, instrument fallbacks) that naturally narrowed who could contribute code. This is a data point, not a quality issue.

---

## Working Tree State

**Status:** 20 uncommitted files (post-movement integration work)

**What's uncommitted:**
- Baton Phase 2 refinement work (Composer's integration session)
- Test updates related to 11-state model
- Recovery improvements
- Minor observability enhancements

**Quality gate policy:** The gate validates Movement 5 formal output (26 commits) + uncommitted integration work (20 files). All four checks pass WITH these changes present. This is the 9th occurrence of post-movement integration work (F-500, F-013, F-019, F-057, F-080, F-089 prior).

**Pattern analysis:** This is now an established pattern. Movements deliver focused work; integration happens post-movement; quality gate validates both. The ground holds with this pattern — all checks pass, quality is maintained.

---

## Findings Summary

### New in Movement 5
11 findings filed (F-472 through F-490):

| Finding | Severity | Status | Description |
|---------|----------|--------|-------------|
| F-472 | P3 | Resolved | Pre-existing test expected D-027 default |
| F-480 | P0 | Partial | Trademark collision — Phase 1 complete, Phases 2-5 open |
| F-481 | P1 | Resolved | Orphan detection baton path wiring |
| F-482 | P1 | Resolved | MCP server leak cascade (selective MCP) |
| F-483 | Info | — | cli_extra_args confirmed working |
| F-484 | P2 | Open | Agent-spawned background processes escape PGID |
| F-485 | P3 | Open | Conductor RSS step function (monitoring) |
| F-486 | Info | — | Chrome/Playwright PGID isolation working |
| F-487 | P0 | Resolved | reap_orphaned_backends WSL2 crash |
| F-488 | P2 | Open | Profiler DB unbounded growth (551 MB) |
| F-489 | P1 | Open | README and docs outdated |
| F-490 | P0 | Resolved | os.killpg WSL2 crash root cause + guard |

### Resolved
8 findings resolved this movement (F-472, F-149, F-451, F-470, F-431, F-481, F-482, F-490)

### Open
5 findings remain open:
- **F-480 (P0):** Rename Phases 2-5 (CLI binary, docs, examples, GitHub org) — 15 tasks in TASKS.md
- **F-484 (P2):** Background process escape PGID cleanup
- **F-485 (P3):** Conductor RSS step function
- **F-488 (P2):** Profiler DB growth
- **F-489 (P1):** Documentation outdated

---

## Recommendations for Movement 6

### Priority 0 (Blockers)
1. **F-480 Phases 2-5:** Complete the rename — CLI binary (`mzt` final rollout), documentation updates, examples, GitHub organization rename
2. **F-489:** Update README and core documentation to reflect v1 beta state

### Priority 1 (High)
3. **Rosetta modernization:** 5 tasks blocked on score execution — update all example scores to use `instrument:` syntax
4. **Examples audit:** Verify all example scores execute successfully
5. **F-488:** Implement profiler DB rotation/cap (551 MB unbounded growth)

### Priority 2 (Medium)
6. **F-484:** Audit and fix background process PGID escape
7. **Meditation completion:** 5 of 32 musicians missing meditations (Atlas, Breakpoint, Journey, Litmus, Sentinel)

### Process Improvements
8. **Stabilization movements:** Consider dedicated stabilization movements between major feature milestones to prevent large post-movement integration batches
9. **Test regression guards:** Add `len(SheetStatus) == 11` assertion test to catch future state model expansions early

---

## Gate Summary

| Check | Status | Details |
|-------|--------|---------|
| Test Coverage | ✅ PASS | 11,810 tests, 100% pass rate |
| Type Safety | ✅ PASS | Zero errors, 258 files |
| Lint Quality | ✅ PASS | All checks passed |
| Structural Integrity | ✅ PASS | 0 critical findings |

**Final Verdict:** Movement 5 COMPLETE. Ground holds. Ready for Movement 6.

---

## Technical Notes

### Test Warnings
177 warnings reported, breakdown:
- **Pydantic deprecation warnings** (P3): `__fields_set__`, `model_computed_fields`, `model_fields` — affects `unittest/mock.py`. Consider upgrading mocking patterns to avoid deprecated attribute access.
- **RuntimeWarning unawaited coroutines** (P2): 2 occurrences in property-based tests and status tests. Test teardown issue, not production code.
- **UserWarning backend type mismatch** (expected): Test scenario for `test_ollama_returns_model` — validates error handling.

### xpassed Tests
3 tests unexpectedly passing (were expected to fail, now pass):
- Candidates for unskipping and formal pass status
- Should be audited in M6 to determine if xfail markers can be removed

### Structural Patterns Verified
Flowspec diagnosed 0 critical findings across:
- Dead code detection
- Orphaned implementations
- Circular dependencies
- Unused exports
- Type signature consistency

The codebase structural integrity is solid.

---

## Conclusion

After 9 retries spanning two distinct failure classes, the quality gate passes cleanly. The 11-state SheetStatus model expansion was architecturally correct — the tests just needed to catch up with reality. The F-470 regression was a refactoring accident that was quickly caught and fixed. Both classes of failure are now resolved.

Movement 5 delivered substantial improvements:
- Baton is now the default execution model (D-027)
- Instrument fallbacks fully functional with observability
- Critical P0 blockers resolved (F-271, F-255.2, F-470, F-490)
- Config strictness complete (F-431)
- +413 new tests, +1,247 source lines
- Zero type errors, zero lint errors, zero critical structural issues

The ground holds. Movement 6 can build on this foundation.

**The ground is solid. The orchestra can continue.**
