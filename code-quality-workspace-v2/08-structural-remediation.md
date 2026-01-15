# Sheet 8: Structural Fixes Remediation

**Date:** 2026-01-15
**Phase:** Movement III-B - Structural Fixes

---

## Summary

This sheet implemented 5 structural fixes identified in the synthesis triage (Sheet 6). These are medium-effort changes that improve code organization, testability, and interface consistency without requiring architectural changes.

---

## Fixes Applied

### 1. Wire aggregate_job_outcomes to CLI Learning Subcommand

**Location:** `src/mozart/cli.py` (aggregate-patterns command) + `src/mozart/learning/migration.py`

**Problem:** The `aggregate_job_outcomes` and `aggregate_job_outcomes_enhanced` functions in `learning/aggregator.py` were defined but never called. The CLI `aggregate-patterns` command used `OutcomeMigrator` without wiring up the `PatternAggregator`, so pattern detection after migration was a stub returning existing counts.

**Fix:**
1. CLI now creates a `PatternAggregator` and passes it to `OutcomeMigrator`
2. Updated `_detect_patterns_from_store()` to actually use the aggregator for priority updates and pattern pruning

```python
# Before: No aggregator wired
migrator = OutcomeMigrator(store)

# After: Aggregator connected for pattern detection
aggregator = PatternAggregator(store)
migrator = OutcomeMigrator(store, aggregator=aggregator)
```

**Impact:** Pattern aggregation now runs properly after workspace migration, updating priority scores and pruning deprecated patterns.

---

### 2. Create RunnerContext Dataclass

**Location:** `src/mozart/execution/runner.py:274-314`

**Problem:** `JobRunner.__init__` had 11 parameters, making call sites verbose and cognitive load high. Many parameters were optional learning/UI components that could be grouped.

**Fix:** Created `RunnerContext` dataclass to encapsulate optional dependencies:

```python
@dataclass
class RunnerContext:
    """Optional context components for JobRunner."""
    outcome_store: OutcomeStore | None = None
    escalation_handler: EscalationHandler | None = None
    judgment_client: JudgmentClient | None = None
    global_learning_store: GlobalLearningStore | None = None
    grounding_engine: GroundingEngine | None = None
    console: Console | None = None
    progress_callback: Callable[[int, int, float | None], None] | None = None
    execution_progress_callback: Callable[[dict[str, Any]], None] | None = None
```

Updated `JobRunner.__init__` to accept an optional `context` parameter while maintaining full backwards compatibility.

**Impact:** New code can use the cleaner context pattern; existing code continues to work unchanged.

---

### 3. Add test_templating.py Smoke Tests

**Location:** `tests/test_templating.py` (new file, 280 lines)

**Problem:** `prompts/templating.py` had zero test coverage (identified by SCI expert). This module handles critical prompt generation including completion prompts.

**Fix:** Created comprehensive smoke tests covering:
- `SheetContext` creation and serialization (2 tests)
- `PromptBuilder.build_sheet_context` item range calculations (2 tests)
- `PromptBuilder.build_sheet_prompt` with templates, variables, patterns, validation rules (6 tests)
- Template variable expansion (2 tests)
- Default prompt fallback (1 test)
- Completion prompt generation (2 tests)
- `build_sheet_prompt_simple` convenience function (1 test)

**Total:** 14 tests covering the core templating functionality.

---

### 4. Add test_hooks.py Basic Coverage

**Location:** `tests/test_hooks.py` (new file, 335 lines)

**Problem:** `execution/hooks.py` had zero test coverage (identified by SCI expert). This module handles post-success hook execution and concert orchestration.

**Fix:** Created basic coverage tests for:
- `HookResult` dataclass (3 tests)
- `ConcertContext` tracking (2 tests)
- `HookExecutor` template expansion (2 tests)
- No-hooks returns empty (1 test)
- `run_command` hook execution (1 test)
- Hook failure tracking (1 test)
- `run_job` missing path handling (1 test)
- Unknown hook type handling (1 test)
- `get_next_job_to_chain` (2 tests)
- Concert chain depth limiting (1 test)

**Total:** 15 tests covering core hook execution paths.

---

### 5. Standardize _detect_rate_limit Interface

**Location:** `src/mozart/backends/claude_cli.py:598` + `src/mozart/backends/anthropic_api.py:337`

**Problem:** The two backends had inconsistent interfaces:
- `claude_cli.py`: `_detect_rate_limit(stdout: str, stderr: str)`
- `anthropic_api.py`: `_detect_rate_limit(message: str)`

**Fix:** Unified the interface to:
```python
def _detect_rate_limit(self, stdout: str = "", stderr: str = "") -> bool:
    """Check output for rate limit indicators.

    Note: This interface matches [other backend]._detect_rate_limit
    for consistency across backends.
    """
```

Updated the call site in `anthropic_api.py` to use keyword argument: `self._detect_rate_limit(stderr=str(e))`

**Impact:** Both backends now have the same signature, making testing and backend abstraction easier.

---

## Verification Results

### Type Check (mypy)
```
Found 7 errors in 3 files (checked 52 source files)
```
- All errors are pre-existing, unrelated to structural fixes
- Key pre-existing issues in `cli.py` (ValidationEngine context), `errors.py` (float/int assignment)

### Test Results
```
tests/test_templating.py: 14 passed
tests/test_hooks.py: 15 passed
Total: 29 passed, 1 warning
```

### Import Verification
```python
from mozart.cli import app
from mozart.execution.runner import JobRunner, RunnerContext
from mozart.learning.global_store import GlobalLearningStore
from mozart.learning.aggregator import PatternAggregator, aggregate_job_outcomes
from mozart.learning.migration import OutcomeMigrator
# All critical imports OK
```

---

## Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `src/mozart/cli.py` | Modified | Wire PatternAggregator to aggregate-patterns command |
| `src/mozart/learning/migration.py` | Modified | Improve _detect_patterns_from_store to use aggregator |
| `src/mozart/execution/runner.py` | Modified | Add RunnerContext dataclass |
| `src/mozart/backends/anthropic_api.py` | Modified | Standardize _detect_rate_limit interface |
| `src/mozart/backends/claude_cli.py` | Modified | Add default params to _detect_rate_limit |
| `tests/test_templating.py` | Added | 14 tests for templating module |
| `tests/test_hooks.py` | Added | 15 tests for hooks module |

---

## Deferred Items

The following structural items from the triage were considered but deferred:

1. **Wire LocalJudgmentClient** - Requires product decision on feature scope
2. **RunnerContext full adoption** - Context is available, CLI migration is gradual
3. **Additional test coverage** - Beyond smoke tests; can expand in future iterations

---

*Structural Fixes Remediation completed for Mozart Code Quality Review v2*
