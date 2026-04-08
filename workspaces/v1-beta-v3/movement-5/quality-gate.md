# Movement 5 — Quality Gate Report (Retry #8)

**Agent:** Bedrock
**Date:** 2026-04-08
**Verdict:** ❌ **FAIL**

## Summary

Movement 5 does NOT pass quality gate validation. All quality checks pass EXCEPT pytest, which has 1 test failure caused by a regression introduced in post-movement refactoring work. The failure is in F-470's TDD test suite — a memory leak fix that was correctly implemented in Movement 5 but accidentally deleted during subsequent baton Phase 2 refactoring.

## Validation Results

### ✅ Type Safety: PASS

```bash
cd /home/emzi/Projects/marianne-ai-compose && python -m mypy src/ --no-error-summary
```

**Result:** Clean. Zero errors.
**Evidence:** No output from mypy (clean run).

### ✅ Lint Quality: PASS

```bash
cd /home/emzi/Projects/marianne-ai-compose && python -m ruff check src/
```

**Result:** All checks passed.
**Evidence:** "All checks passed!" message, exit code 0.

### ✅ Structural Integrity: PASS

```bash
/home/emzi/Projects/flowspec/target/release/flowspec diagnose /home/emzi/Projects/marianne-ai-compose --severity critical -f summary -q
```

**Result:** 0 critical findings.
**Evidence:** "Diagnostics: 0 finding(s)\n\nNo findings."

### ❌ Test Suite: FAIL

```bash
cd /home/emzi/Projects/marianne-ai-compose && python -m pytest tests/ -x -q --tb=short
```

**Result:** 1 test failure (11,700+ tests passed before failure).
**Failure:** `tests/test_f470_synced_status_cleanup.py::TestSyncedStatusCleanupOnDeregister::test_deregister_removes_synced_entries`

**Error:**
```
AssertionError: Leaked entries: {('abc', 0), ('abc', 3), ('abc', 2), ('abc', 4), ('abc', 1)}
assert {('abc', 0), ...), ('abc', 4)} == set()
```

**File:** `/home/emzi/Projects/marianne-ai-compose/tests/test_f470_synced_status_cleanup.py:41`

## Root Cause Analysis

### The Regression

**F-470** was correctly fixed by Maverick in Movement 5 (commit `201cd25`, 2026-04-05):

```python
# F-470: Clean up state-diff dedup cache to prevent memory leak
self._synced_status = {
    k: v for k, v in self._synced_status.items() if k[0] != job_id
}
```

This fix was added to `BatonAdapter.deregister_job()` at line 518-521 in `src/mozart/daemon/baton/adapter.py`. The fix included 5 TDD tests that all passed.

**The regression** occurred in commit `01e4cdb` (2026-04-08, Composer):

> refactor(baton): delete sync layer, add per-sheet stale timeout (Phase 2)

This commit removed 217 lines of sync infrastructure from `adapter.py`. The commit message stated:

> "Kept compat attributes (_state_sync_callback, _synced_status) and identity wrappers (baton_to_checkpoint_status, mapping dicts) for tests that reference them"

The `_synced_status` dict was kept, but the cleanup code in `deregister_job()` was accidentally deleted. The refactor preserved the data structure but removed the deallocation logic.

### Evidence Chain

1. **201cd25** (Movement 5, Maverick): F-470 fix added, test passes
2. **01e4cdb** (Post-movement, Composer): Sync layer refactor removes fix
3. **Current HEAD** (5162ddb): Test fails with exact same symptom F-470 originally addressed

**Verified by:**
```bash
git show 201cd25 -- src/mozart/daemon/baton/adapter.py | grep -A 5 "_synced_status"
# Shows fix present

git show 01e4cdb:src/marianne/daemon/baton/adapter.py | grep -A 30 "def deregister_job"
# Shows fix absent
```

### Impact Assessment

**Severity:** P1 (High)
**Category:** Memory leak regression
**User Impact:** Long-running conductors accumulate O(total_sheets_ever) entries in `_synced_status`. For a conductor running 1000 jobs × 10 sheets each = 10,000 stale entries never freed.

**System Impact:**
- Dict size grows unbounded
- Memory pressure on long-running daemons
- Dict lookup performance degrades (O(n) linear scan on iteration)

**Breaking Change:** No. The leak is silent — jobs complete correctly, memory just accumulates.

## Current State

**Working tree:** Uncommitted changes present (post-movement work in progress):
- `src/marianne/daemon/baton/adapter.py` (stale check interception logic)
- `src/marianne/daemon/baton/core.py`
- `workspaces/v1-beta-v3/FINDINGS.md`
- `workspaces/v1-beta-v3/TASKS.md`
- `plugins` submodule

**Movement 5 formal work:** 26 commits from 12 musicians, all committed and merged.

**Post-movement work:** 11 commits from Composer (refactoring, bug fixes, feature additions) — work in progress, not yet committed.

## Fix Path

### Option 1: Restore F-470 Fix (1 line, 0 minutes)

Add back the missing cleanup in `src/marianne/daemon/baton/adapter.py:deregister_job()`:

```python
def deregister_job(self, job_id: str) -> None:
    """Remove a job from the adapter and baton."""
    # ... existing cleanup code ...

    # F-470: Clean up state-diff dedup cache to prevent memory leak
    self._synced_status = {
        k: v for k, v in self._synced_status.items() if k[0] != job_id
    }

    _logger.info("adapter.job_deregistered", extra={"job_id": job_id})
```

The 5 existing TDD tests from Movement 5 will verify the fix:
- `test_deregister_removes_synced_entries` (currently failing)
- `test_deregister_preserves_other_jobs`
- `test_deregister_large_scale_cleanup`
- `test_deregister_empty_cache_is_noop`
- `test_deregister_with_mixed_statuses`

### Option 2: Commit or Revert Uncommitted Work

The quality gate cannot pass while uncommitted work exists. Either:
1. Commit the in-progress work (with failing test)
2. Stash the uncommitted work and re-run gate on clean M5 HEAD
3. Fix F-470 regression, then commit everything together

**Recommended:** Option 1 (restore fix) + commit all work as a single "baton Phase 2" commit.

## Comparison to Previous Retries

**Retries #1-5:** 50 test failures from 11-state SheetStatus model expansion (architectural change, tests not updated).

**Retry #8 (this session):** 1 test failure from F-470 regression (refactoring accident).

The 50-test failures from retries #1-5 are GONE. Either:
1. The 11-state model work was reverted
2. The tests were fixed
3. The uncommitted work includes those fixes

Evidence from git log shows `b3e6a08 fix(baton): 13 fixes + Phase 2 unified state model` — the Composer fixed the state model issues post-movement.

## Recommendations

### Immediate (This Movement)

1. **Escalate to Composer:** The quality gate cannot pass while uncommitted work exists. The work in progress includes the F-470 regression.
2. **File F-504:** Document the F-470 regression in FINDINGS.md (P1, "F-470 fix deleted in 01e4cdb refactor").
3. **Add to TASKS.md:** "Restore F-470 cleanup in deregister_job (P1)" for next movement.

### Structural (Next Movement)

1. **Regression guard:** Add a test that counts lines in critical methods (e.g., `deregister_job` must have >15 lines). When a refactor removes code, the test fails and forces review.
2. **Refactor protocol:** When deleting >100 lines, run the full test suite on the refactored code before moving to the next change. Catch regressions at the source.
3. **Sync layer deletion audit:** Commit 01e4cdb deleted 217 lines. Run a focused audit: what other cleanup code was deleted? Check all `pop()`, `clear()`, and dict comprehension removals.

### Process

The pattern here is the 8th occurrence of the uncommitted work anti-pattern (F-013, F-019, F-057, F-080, F-089, F-500, F-501, now F-504). The Composer's integration work (11 commits, 18,504 insertions in M5 alone) happens post-movement, outside the coordination structure.

**Root cause (systemic):** Large integration work exceeds movement coordination capacity. The fix isn't discipline — it's process:
1. Plan large integration as dedicated movements (e.g., "Movement 5.5: Baton Phase 2")
2. Break integration into incremental commits during movements (daily/per-feature commits)
3. Use separate branches for integration, merge when complete

## Gate Verdict

**Type safety:** ✅ Intact (mypy clean)
**Lint quality:** ✅ Intact (ruff clean)
**Structural integrity:** ✅ Intact (flowspec 0 critical)
**Test coverage:** ❌ Broken (1 regression failure)

**Overall:** ❌ **Movement 5 quality gate FAILS**

The ground does not hold. The regression is fixable in 1 line. The uncommitted work makes the gate ambiguous — we're testing Movement 5 formal output + 11 commits of post-movement work. Until the working tree is clean, the gate result is unreliable.

**Next movement:** Restore F-470 fix, commit all post-movement work, re-run gate on clean state.

---

**Metadata:**
- Report word count: 1,347
- Tests run: ~11,700 before failure
- Failure count: 1
- Regression introduced: commit 01e4cdb
- Regression detected: retry #8
- Time between fix and regression: 3 days (2026-04-05 → 2026-04-08)
