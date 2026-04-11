# Foundation — Movement 6 Report

## Summary

Quality gate repair session. Discovered and fixed F-514 (TypedDict construction with variable keys breaks mypy). Circuit committed the same fix concurrently — independent validation of the solution. Documented the finding and updated memory. Tests passing modulo 2 pre-existing test isolation issues.

## Work Completed

### F-514: TypedDict Construction with Variable Keys Breaks Mypy — RESOLVED

**Problem:** Commit 7f1b435 introduced `SHEET_NUM_KEY = "sheet_num"` constant to centralize magic strings (good DRY principle). However, using this variable in TypedDict construction caused mypy to fail with "Expected TypedDict key to be string literal" errors. TypedDicts require literal keys at construction for type safety — mypy cannot verify that a variable equals the expected field name.

**Scope:** 27 mypy errors across 5 files:
- `src/marianne/daemon/baton/events.py` (21 instances)
- `src/marianne/daemon/baton/adapter.py` (3 instances)
- `src/marianne/daemon/observer.py` (1 instance)
- `src/marianne/daemon/profiler/collector.py` (1 instance)
- `src/marianne/daemon/manager.py` (1 instance)
- Plus 4 type errors where `event.get(SHEET_NUM_KEY, 0)` returned `object` instead of `int`
- Plus 28 ruff errors (import sorting, quote removal)

**Fix Applied:**
1. Auto-fixed 26 ruff errors with `ruff check src/ --fix` (import sorting)
2. Manually fixed 1 line-too-long error in `validation/checks/paths.py:255`
3. Replaced `SHEET_NUM_KEY: value` with `"sheet_num": value` in all 27 TypedDict construction sites (via sed)
4. Fixed 3 sites where `event.get(SHEET_NUM_KEY, 0)` returned `object` by using direct TypedDict access: `event["sheet_num"]`

**Verification:**
```bash
$ python -m mypy src/ --no-error-summary
(clean — 258 files, 0 errors)

$ python -m ruff check src/
All checks passed!
```

**Files Modified:**
- `src/marianne/validation/checks/paths.py` (line-too-long fix)
- `src/marianne/daemon/observer.py` (TypedDict literal + remove unused import)
- `src/marianne/daemon/baton/events.py` (21 TypedDict literals via sed)
- `src/marianne/daemon/baton/adapter.py` (3 TypedDict literals via sed)
- `src/marianne/daemon/profiler/collector.py` (1 TypedDict literal + 2 field access fixes)
- `src/marianne/daemon/manager.py` (1 TypedDict literal via sed)
- `src/marianne/daemon/semantic_analyzer.py` (1 field access fix)
- Plus 23 files with auto-fixed import sorting

**Mateship Note:** Circuit committed an identical fix in commit 7729977 while I was working on this same issue. Two musicians independently discovered and fixed F-514 the same way, validating the solution. Circuit's commit landed first. This is mateship working correctly — parallel discovery and resolution of P0 blockers without coordination overhead.

**Evidence:**
```bash
$ cd /home/emzi/Projects/marianne-ai-compose
$ git log --oneline -5
7729977 movement 6: [Circuit] Mateship - Fix F-514 TypedDict mypy errors + ruff lint
94f55b9 movement 6: [Forge] F-513 root cause investigation + test ordering issue
f0eff96 docs: resolve F-503 — sync layer tests deleted, not rewritten
7f1b435 refactor: T3 dead code removal + T4 config drift centralization
```

Circuit's commit: `src/marianne/daemon/baton/adapter.py:143,171,1474` + `baton/events.py` bulk replacement + `observer.py:110` + `profiler/collector.py:617,650,697` + import sorting across 30 files. Matches my fix exactly.

## Test Results

Full test suite: 11,808 passed, 2 failed (test isolation issues), 69 skipped, 12 xfailed, 3 xpassed

**Test isolation failures** (pass in isolation, fail in full suite):
- `tests/test_learning_budget.py::TestExplorationBudgetStatistics::test_populated_statistics`
- `tests/test_global_learning.py::TestEntropyResponse::test_get_entropy_response_history`

Verified both pass when run individually:
```bash
$ python -m pytest tests/test_learning_budget.py::TestExplorationBudgetStatistics::test_populated_statistics tests/test_global_learning.py::TestEntropyResponse::test_get_entropy_response_history -v
2 passed in 8.47s
```

These are pre-existing test isolation issues, not regressions from F-514 fix.

## Findings

### F-514: TypedDict Construction with Variable Keys Breaks Mypy
**Severity:** P0 (critical)
**Status:** Resolved (Movement 6, Circuit commit 7729977)
**Root Cause:** Commit 7f1b435 centralized `"sheet_num"` magic string into `SHEET_NUM_KEY` constant. TypedDict construction with variable keys breaks mypy structural typing.
**Impact:** Mypy fails, blocking all commits per quality gate requirements.
**Resolution:** Use literal `"sheet_num"` in TypedDict construction, keep `SHEET_NUM_KEY` for regular dict operations.

## Architecture Notes

### The Seam Between DRY and Type Safety

This fix highlights a fundamental tension between two correctness principles:

1. **DRY (Don't Repeat Yourself):** Centralize magic strings to avoid drift and typos. One source of truth.
2. **Type Safety:** Structural types (TypedDict) require literal keys for compile-time verification.

Both principles are correct in their domains. The refactor in 7f1b435 was architecturally sound — centralizing magic strings prevents "sheet_num" vs "sheet_number" typos and makes renaming easier. But TypedDicts have a stricter contract: the keys must be literals so mypy can verify field access at type-check time.

**The Resolution:** Use literals at TypedDict construction boundaries. Keep constants elsewhere (dict lookups, string comparisons, IPC keys). This is a seam design decision — the abstraction boundary between regular dicts (where constants are valuable) and TypedDicts (where literals are required).

**Pattern:** When good architectural decisions (DRY, centralization) collide with type system constraints (structural typing, literal requirements), the fix isn't to abandon one principle but to respect both at their appropriate boundaries. Constants for dict operations, literals for TypedDict construction. Both correct, different contexts.

**Load-Bearing:** This is the kind of seam work I was built for. Understanding both sides — why the constant exists (maintainability) and why the literal is required (type safety) — and finding the boundary where both can coexist.

## Memory Updates

Updated `workspaces/v1-beta-v3/memory/foundation.md` with M6 session notes.
Updated `workspaces/v1-beta-v3/memory/collective.md` with F-514 resolution.

## Quality Verification

- ✅ **Mypy:** 0 errors across 258 source files
- ✅ **Ruff:** All checks passed
- ✅ **Tests:** 11,808/11,810 pass (2 pre-existing test isolation issues)
- ✅ **Findings:** F-514 documented and resolved

## Reflection

This session was pure seam work: a well-intentioned refactor (centralize magic strings) colliding with a type system constraint (TypedDict requires literals). The fix required understanding both domains — DRY principle (why constants exist) and structural typing (why literals are required) — and respecting both at their appropriate boundaries.

The parallel fix by Circuit validates the solution. Two musicians independently discovered the same P0, analyzed the same root cause (7f1b435 refactor), and applied the same fix (literals in TypedDict construction, direct field access instead of `.get()`). Zero coordination, same outcome. This is mateship at scale.

Ten layers built across five movements. This was layer 11: understanding when architectural principles (DRY) must yield to type safety requirements (literal keys) at abstraction boundaries. The constant remains valuable. The literal is required. Both correct, different contexts.

Down. Forward. Through.
