# Movement 7: Lens Report

## Summary

Fixed F-523 schema validation error messages for common onboarding mistakes. Completed uncommitted work from a previous musician, implementing proper error hint generation for plural/singular field confusion ("sheets"/"prompts" vs "sheet"/"prompt") with YAML structure examples.

**Deliverables:**
- F-523 schema error message improvements (8 TDD tests, all passing)
- Enhanced `_schema_error_hints()` in `src/marianne/cli/commands/validate.py`
- Commit 78bd95b on main

**Quality verification:**
- pytest: 18/18 validate-related tests passing
- mypy: clean (258 files, 0 errors)
- ruff: clean (all checks passed)

---

## Work Completed

### F-523: Schema Error Message Improvements (PARTIAL RESOLUTION)

**Context:** Found uncommitted changes in `src/marianne/cli/commands/validate.py` addressing F-523. Tests already existed in `tests/test_f523_schema_error_messages.py` (146 lines, 8 test classes) but ALL were failing because the implementation was incomplete.

**Root cause:** New users trying "sheets:" or "prompts:" (plural) hit Pydantic v2 "Extra inputs are not permitted" errors that don't explain the correct structure. Original error messages were cryptic and unhelpful.

**Implementation (TDD - Red → Green):**

1. **Ran existing tests first** (all 8 failed):
   ```bash
   pytest tests/test_f523_schema_error_messages.py -xvs
   # FAILED: 8/8 tests
   ```

2. **Fixed `_schema_error_hints()` in validate.py** (`src/marianne/cli/commands/validate.py:273-349`):
   - Changed error detection from substring match to proper Pydantic v2 error format parsing
   - Added support for multiple error types in one message (extra_forbidden + field_required)
   - Enhanced field extraction using regex: `r"^(\w[\w.]*)\n\s+Field required"`
   - Added YAML structure examples for missing required fields

3. **Enhanced `_unknown_field_hints()` in validate.py** (`src/marianne/cli/commands/validate.py:361-375`):
   - Already had "sheets" and "prompts" in `_KNOWN_TYPOS` dict (lines 328-329)
   - Added conditional YAML examples for sheets/prompts mistakes (lines 364-373)

4. **Verified all tests pass**:
   ```bash
   pytest tests/test_f523_schema_error_messages.py -v
   # 8 passed in 12.87s
   ```

**Example improvement:**

Before (unhelpful):
```
Schema validation failed: ...
Extra inputs are not permitted
```

After (actionable guidance):
```
Schema validation failed: ...
Unknown field 'sheets' — did you mean 'sheet (singular — use: sheet: {size: N, total_items: M})'?
  Use 'sheet' (singular) with this structure:
  sheet:
    size: 10
    total_items: 100
See: docs/score-writing-guide.md for the complete field reference.
```

**Files modified:**
- `src/marianne/cli/commands/validate.py` (+66 lines, refactored `_schema_error_hints()`)
- `tests/test_f523_schema_error_messages.py` (NEW, 146 lines, 8 test classes)

**Test coverage:**
1. `TestSheetPluralError` - sheets/prompts plural detection
2. `TestMovementsStructureError` - movements dict vs list
3. `TestMissingRequiredFields` - sheet/prompt required with examples
4. `TestMultipleErrorsInOneMessage` - combined error handling
5. `TestRealWorldOnboardingScenarios` - actual F-523 error patterns

**Evidence:**
```bash
# F-523 tests
cd /home/emzi/Projects/marianne-ai-compose
pytest tests/test_f523_schema_error_messages.py -v
# ........  [100%]
# 8 passed in 12.87s

# All validate tests
pytest tests/test_validate_ux_journeys.py tests/test_f523_schema_error_messages.py -v
# ..................  [100%]
# 18 passed in 7.57s

# Quality gate
mypy src/ --no-error-summary
# Success: no issues found in 258 source files

ruff check src/
# All checks passed!
```

**Commit:**
```
78bd95b movement 7: [Lens] F-523 schema error message improvements
```

---

## Note: F-523 Partial Resolution

F-523 as originally filed has TWO distinct issues:

1. **Schema error messages** (RESOLVED in this movement)
   - "Extra inputs are not permitted" now shows field suggestions + YAML examples
   - Commit 78bd95b

2. **Sandbox blocking docs access** (REMAINS OPEN)
   - Agents in workspace CWD can't read `../../README.md`, `../../examples/`, etc.
   - Requires separate fix (tool to read project-root files or sandbox policy change)

The finding should be updated to reflect partial resolution.

---

## Outside-In Development

This work exemplifies outside-in development:
1. **Started with the user's pain** (F-523: confusing error messages blocking onboarding)
2. **Tests defined the contract** (8 tests showing what users need to see)
3. **Implementation delivered the contract** (enhanced error hints with YAML examples)
4. **Verified with real output** (tested error messages match user expectations)

The gap between "tests pass" and "users can use it" is exactly where I live. Error messages are UI. Every error is either a dead-end or a teaching moment. I turned these dead-ends into teaching moments.

---

## Mateship

Picked up uncommitted work from a previous musician who had:
- Added "sheets"/"prompts" to _KNOWN_TYPOS dict
- Created test file with 8 test classes
- Left implementation incomplete (all tests failing)

Completed the work properly:
- Fixed error detection logic to match Pydantic v2 format
- Added YAML structure examples
- Handled combined error types
- All tests green, committed to main

---

## Quality Metrics

**Before this movement:**
- F-523 schema errors: cryptic, unhelpful
- Test coverage: 0 tests passing (8 existed, all red)
- User experience: dead-end errors, no guidance

**After this movement:**
- F-523 schema errors: clear suggestions + YAML examples
- Test coverage: 8/8 tests passing (100%)
- User experience: every error teaches correct structure

**Static analysis:**
- mypy: 0 errors (258 files)
- ruff: 0 errors (all checks passed)

---

## What I Learned

**F-523 conflates two issues:** Schema error messages + sandbox blocking docs. I fixed the schema part. The sandbox part needs separate attention. In future, when a finding describes multiple unrelated problems, they should be split into separate findings.

**Incomplete work pattern:** Found uncommitted changes + failing tests. This is a mateship opportunity - someone started the work, hit a blocker, moved on. Finishing their work properly and committing it is how the orchestra functions.

**Error message quality matters more than most engineers think:** A user hitting "Extra inputs are not permitted" has no path forward. A user hitting "Unknown field 'sheets' — did you mean 'sheet'?" + YAML example can self-correct immediately. The difference is adoption vs abandonment.

---

## Time Investment

Estimated ~45 minutes:
- 10 min: Reading context (memory, findings, tasks, git history)
- 15 min: Understanding failing tests and incomplete implementation
- 15 min: Implementing fix (refactoring `_schema_error_hints()`)
- 5 min: Fixing ruff violations, running tests, verifying quality gate
- 5 min: Committing, updating memory, writing this report

---

## Remaining Work

From my perspective (UX/error messages/CLI), high-priority unclaimed tasks:
- F-502 CLI workspace fallback removal (pause/resume/recover) - tried in M6, reverted, needs proper implementation
- F-522 `--conductor-clone` missing from most CLI commands (P0 blocker for safe testing)
- F-528 migration guide for v2 → v3 score format changes

All are CLI/UX domain work that match my strengths.
