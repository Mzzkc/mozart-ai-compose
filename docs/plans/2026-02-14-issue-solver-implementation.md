# Issue Solver — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the `skip_when_command` feature (#71) and the issue solver score (#72) as designed in `docs/plans/2026-02-14-issue-solver-design.md`.

**Architecture:** Two deliverables: (1) a new `SkipWhenCommand` Pydantic model + lifecycle integration that lets sheets skip based on shell command exit codes, and (2) a 17-stage self-chaining score (`examples/issue-solver.yaml`) that uses the feature. TDD throughout.

**Tech Stack:** Python 3.12+, Pydantic v2, asyncio, pytest, YAML (Mozart score format)

---

### Task 1: Add `SkipWhenCommand` model

**Files:**
- Modify: `src/mozart/core/config/execution.py` (add after `ValidationRule` class, ~line 402)

**Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
from mozart.core.config import SkipWhenCommand

class TestSkipWhenCommand:
    """Tests for SkipWhenCommand model."""

    def test_defaults(self):
        """Test default values."""
        cmd = SkipWhenCommand(command="grep -q DONE file.txt")
        assert cmd.command == "grep -q DONE file.txt"
        assert cmd.description is None
        assert cmd.timeout_seconds == 10.0

    def test_custom_values(self):
        """Test custom values."""
        cmd = SkipWhenCommand(
            command="test -f output.txt",
            description="Skip if output exists",
            timeout_seconds=30.0,
        )
        assert cmd.description == "Skip if output exists"
        assert cmd.timeout_seconds == 30.0

    def test_timeout_must_be_positive(self):
        """Test timeout_seconds must be > 0."""
        with pytest.raises(ValidationError):
            SkipWhenCommand(command="echo hi", timeout_seconds=0)

    def test_timeout_max_60(self):
        """Test timeout_seconds capped at 60."""
        with pytest.raises(ValidationError):
            SkipWhenCommand(command="echo hi", timeout_seconds=61)

    def test_command_required(self):
        """Test command field is required."""
        with pytest.raises(ValidationError):
            SkipWhenCommand()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::TestSkipWhenCommand -v`
Expected: FAIL with `ImportError` — `SkipWhenCommand` doesn't exist yet

**Step 3: Write minimal implementation**

Add to `src/mozart/core/config/execution.py` at the end of the file:

```python
class SkipWhenCommand(BaseModel):
    """A command-based conditional skip rule for sheet execution.

    When the command exits 0, the sheet is SKIPPED.
    When the command exits non-zero, the sheet RUNS.
    On timeout or error, the sheet RUNS (fail-open for safety).

    The ``command`` field supports ``{workspace}`` template expansion,
    following the same pattern as validation commands.
    """

    command: str = Field(
        description="Shell command to evaluate. Exit 0 = skip the sheet. "
        "Supports {workspace} template expansion.",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable reason for the skip condition",
    )
    timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        le=60,
        description="Maximum seconds to wait for command (fail-open on timeout)",
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::TestSkipWhenCommand -v`
Expected: PASS (all 5 tests)

**Step 5: Export from `__init__.py`**

Add `SkipWhenCommand` to the import in `src/mozart/core/config/__init__.py`:

In the execution imports block (~line 19), add `SkipWhenCommand` to the import list.
In `__all__` (~line 83), add `"SkipWhenCommand"` to the Execution section.

**Step 6: Commit**

```bash
git add src/mozart/core/config/execution.py src/mozart/core/config/__init__.py tests/test_config.py
git commit -m "feat(config): add SkipWhenCommand model (#71)"
```

---

### Task 2: Add `skip_when_command` field to `SheetConfig`

**Files:**
- Modify: `src/mozart/core/config/job.py:49-98` (SheetConfig class)
- Modify: `tests/test_config.py` (TestSheetConfig class)

**Step 1: Write the failing test**

Add to `TestSheetConfig` in `tests/test_config.py`:

```python
    def test_skip_when_command_default_empty(self):
        """Test skip_when_command defaults to empty dict."""
        config = SheetConfig(size=1, total_items=5)
        assert config.skip_when_command == {}

    def test_skip_when_command_accepts_rules(self):
        """Test skip_when_command accepts SkipWhenCommand per sheet."""
        config = SheetConfig(
            size=1, total_items=10,
            skip_when_command={
                8: {"command": 'grep -q "TOTAL_PHASES: [1]$" "{workspace}/plan.md"',
                    "description": "Skip phase 2 if plan has only 1 phase"},
                9: {"command": 'grep -q "TOTAL_PHASES: [1]$" "{workspace}/plan.md"'},
            },
        )
        assert 8 in config.skip_when_command
        assert config.skip_when_command[8].command.startswith("grep")
        assert config.skip_when_command[8].description == "Skip phase 2 if plan has only 1 phase"
        assert config.skip_when_command[9].description is None

    def test_skip_when_command_in_jobconfig(self):
        """Test skip_when_command works in full JobConfig."""
        config = JobConfig.model_validate({
            "name": "test",
            "sheet": {
                "size": 1, "total_items": 5,
                "skip_when_command": {
                    3: {"command": "test -f /tmp/skip", "description": "test"},
                },
            },
            "prompt": {"template": "{{ sheet_num }}"},
        })
        assert 3 in config.sheet.skip_when_command
        assert config.sheet.skip_when_command[3].timeout_seconds == 10.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::TestSheetConfig::test_skip_when_command_default_empty tests/test_config.py::TestSheetConfig::test_skip_when_command_accepts_rules tests/test_config.py::TestSheetConfig::test_skip_when_command_in_jobconfig -v`
Expected: FAIL — `skip_when_command` not a field on SheetConfig

**Step 3: Write minimal implementation**

In `src/mozart/core/config/job.py`, add `SkipWhenCommand` to the import from `execution` (line 19-27):

```python
from mozart.core.config.execution import (
    CircuitBreakerConfig,
    CostLimitConfig,
    ParallelConfig,
    RateLimitConfig,
    RetryConfig,
    SkipWhenCommand,
    StaleDetectionConfig,
    ValidationRule,
)
```

Add the field to `SheetConfig` after `skip_when` (after line 87):

```python
    # Command-based conditional execution (GH#71)
    skip_when_command: dict[int, SkipWhenCommand] = Field(
        default_factory=dict,
        description=(
            "Command-based conditional skip rules. Map of sheet_num -> SkipWhenCommand. "
            "The command is run via shell; exit 0 = skip the sheet, non-zero = run it. "
            "Supports {workspace} template expansion in the command string. "
            "On timeout or error, the sheet runs (fail-open). "
            "Example: {8: {command: 'grep -q \"PHASES: 1\" plan.md', description: 'Skip phase 2'}}"
        ),
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::TestSheetConfig -v`
Expected: PASS (all SheetConfig tests including new ones)

**Step 5: Commit**

```bash
git add src/mozart/core/config/job.py tests/test_config.py
git commit -m "feat(config): add skip_when_command field to SheetConfig (#71)"
```

---

### Task 3: Implement `_should_skip_sheet` command support

**Files:**
- Modify: `src/mozart/execution/runner/lifecycle.py:652-700` (`_should_skip_sheet` method)
- Create: `tests/test_skip_when_command.py`

**Step 1: Write the failing unit tests**

Create `tests/test_skip_when_command.py` with tests covering:

1. `test_no_command_conditions_returns_none` — No skip_when_command → None
2. `test_sheet_not_in_conditions_returns_none` — Sheet not configured → None
3. `test_command_exits_0_skips_sheet` — `true` command → skip
4. `test_command_exits_nonzero_runs_sheet` — `false` command → run
5. `test_workspace_template_expansion` — `{workspace}` replaced in command
6. `test_workspace_expansion_with_grep` — Real grep on tmp_path file
7. `test_timeout_fails_open` — `sleep 10` with 0.1s timeout → run (fail-open)
8. `test_command_error_fails_open` — `/nonexistent/binary` → run (fail-open)
9. `test_expression_skip_checked_first` — Both skip_when and skip_when_command on same sheet; expression takes priority
10. `test_description_used_in_skip_reason` — Description appears in reason string
11. `test_no_description_uses_command_in_reason` — Without description, command shown

Each test creates a mock runner with a JobConfig containing `skip_when_command` and calls the async `_should_skip_sheet` method directly.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_skip_when_command.py -v`
Expected: FAIL — `_should_skip_sheet` is sync, not async; doesn't check `skip_when_command`

**Step 3: Implement the feature**

Modify `_should_skip_sheet` in `src/mozart/execution/runner/lifecycle.py:652-700`:

1. Change signature from `def` to `async def`
2. Keep the existing expression-based `skip_when` check as Phase 1 (unchanged logic)
3. Add Phase 2: check `skip_when_command` for this sheet number
4. Template-expand `{workspace}` in the command string using `str.replace()`
5. Run command via `asyncio.create_subprocess_shell` with stdout/stderr devnull
6. Use `asyncio.wait_for` with `timeout_seconds` from the config
7. Exit 0 → return skip reason (description or command string)
8. Non-zero → return None (run the sheet)
9. TimeoutError → log warning, kill process, return None (fail-open)
10. Exception → log warning, return None (fail-open)

Update the call site at line 724 from sync to await:
```python
skip_reason = await self._should_skip_sheet(next_sheet, state)
```

Search for any other call sites of `_should_skip_sheet` in the parallel execution path and update those too.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_skip_when_command.py -v`
Expected: PASS (all 11 tests)

**Step 5: Run full test suite to check for regressions**

Run: `pytest tests/test_config.py tests/test_skip_when_command.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/mozart/execution/runner/lifecycle.py tests/test_skip_when_command.py
git commit -m "feat(lifecycle): implement skip_when_command evaluation (#71)"
```

---

### Task 4: E2E test for `skip_when_command`

**Files:**
- Create: `tests/test_skip_when_command_e2e.py`

This test validates the full lifecycle: config with `skip_when_command` loads from YAML, roundtrips through serialize/deserialize, and validates correctly.

**Step 1: Write the E2E test**

Create `tests/test_skip_when_command_e2e.py` with tests:

1. `test_config_roundtrip_with_skip_when_command` — Write YAML with skip_when_command, load it, verify fields, dump and validate roundtrip
2. `test_validate_command_validates_skip_when_command` — JobConfig.model_validate accepts skip_when_command configs
3. `test_skip_when_command_with_both_skip_types` — Config can have both skip_when and skip_when_command on different sheets

**Step 2: Run test**

Run: `pytest tests/test_skip_when_command_e2e.py -v`
Expected: PASS (all 3 tests)

**Step 3: Commit**

```bash
git add tests/test_skip_when_command_e2e.py
git commit -m "test(e2e): add skip_when_command end-to-end tests (#71)"
```

---

### Task 5: Type check and lint

**Step 1: Run mypy**

Run: `mypy src/mozart/core/config/execution.py src/mozart/core/config/job.py src/mozart/execution/runner/lifecycle.py --ignore-missing-imports`
Expected: PASS (no errors)

**Step 2: Run ruff**

Run: `ruff check src/mozart/core/config/execution.py src/mozart/core/config/job.py src/mozart/execution/runner/lifecycle.py`
Expected: PASS

**Step 3: Run full test suite**

Run: `pytest tests/ -x -q --tb=short`
Expected: PASS (no regressions)

**Step 4: Commit if any fixes were needed**

```bash
git add -u
git commit -m "fix: address type/lint issues in skip_when_command (#71)"
```

---

### Task 6: Write the issue solver score

**Files:**
- Create: `examples/issue-solver.yaml`

This is the 17-stage, 19-sheet (with fan-out) self-chaining score. It is the primary deliverable of #72.

**Step 1: Write the score file**

Create `examples/issue-solver.yaml` following the architecture from the design doc:

- **Stages 1-3 (Analysis):** Read roadmap, investigate issue, plan phases + write verify.sh
- **Stages 4-11 (Implementation):** 4 phases × (fix + completion), phases 2-4 conditionally skipped via `skip_when_command`
- **Stage 12 (Quality):** Fan-out 3x parallel review (functional, e2e/smoke, code quality)
- **Stage 13 (Synthesis):** Review synthesis + fix findings
- **Stage 14 (Docs):** Update documentation
- **Stage 15 (Final verification):** verify.sh + pytest + ruff + mypy
- **Stage 16 (Ship):** Commit & push
- **Stage 17 (Chain gate):** Close issue + self-chain termination

Key YAML features to include:
- `skip_when_command` for conditional phase execution (stages 6-11 skip based on `TOTAL_PHASES` in plan)
- `fan_out: {12: 3}` for parallel review
- `on_success` hook for self-chaining with `fresh: true`, `detached: true`
- `concert` config with `max_chain_depth: 30`, `cooldown_between_jobs_seconds: 300`
- `workspace_lifecycle` with `archive_on_fresh: true`, `max_archives: 30`
- `cross_sheet` for passing outputs between stages
- `validations` per stage with staged execution (stage 1 = syntax, stage 2 = tests)
- Configurable `variables` for roadmap_file, issue_label, test/lint/typecheck commands
- `dependencies` DAG for execution ordering
- Each prompt uses Jinja conditionals (`{% if stage == N %}`) for stage-specific instructions

**Step 2: Validate the score**

Run: `mozart validate examples/issue-solver.yaml`
Expected: Valid (possibly with warnings for missing workspace — that's OK for a template)

**Step 3: Dry run**

Run: `mozart run examples/issue-solver.yaml --dry-run`
Expected: Shows 19 concrete sheets (17 stages, stage 12 expanded to 3 sheets)

**Step 4: Commit**

```bash
git add examples/issue-solver.yaml
git commit -m "feat(score): add issue-solver self-chaining score (#72)"
```

---

### Task 7: Write score template rendering tests

**Files:**
- Create: `tests/test_score_templates.py`

Tests that the issue-solver score's Jinja templates render correctly for each stage and instance.

**Step 1: Write the tests**

Create `tests/test_score_templates.py` with tests:

1. `test_score_loads` — Score loads from YAML without errors
2. `test_total_stages` — Score has 17 logical stages
3. `test_total_sheets_with_fanout` — Score has 19 concrete sheets
4. `test_fan_out_stage_12` — Stage 12 has 3 fan-out instances
5. `test_skip_when_command_targets_valid_sheets` — All skip targets are within range
6. `test_dependencies_are_valid` — All dependency references are valid sheet numbers
7. `test_self_chain_configured` — on_success hook present
8. `test_concert_configured` — Concert config has reasonable depth
9. `test_workspace_lifecycle_configured` — Archive on fresh is enabled
10. `test_has_validations` — Score has validation rules
11. `test_template_renders_for_all_sheets` — Jinja template renders for every sheet number without errors

**Step 2: Run the tests**

Run: `pytest tests/test_score_templates.py -v`
Expected: PASS (all tests, depends on Task 6 being complete)

**Step 3: Commit**

```bash
git add tests/test_score_templates.py
git commit -m "test: add issue-solver score template tests (#72)"
```

---

### Task 8: Final verification and push

**Step 1: Run full test suite**

Run: `pytest tests/ -x -q --tb=short`
Expected: All tests pass

**Step 2: Run type checker**

Run: `mypy src/ --ignore-missing-imports`
Expected: No new errors

**Step 3: Run linter**

Run: `ruff check src/`
Expected: Clean

**Step 4: Validate all example scores**

Run: `for f in examples/*.yaml; do echo "=== $f ===" && mozart validate "$f"; done`
Expected: All valid

**Step 5: Push**

```bash
git push origin main
```

---

## Implementation Order & Dependencies

```
Task 1 (SkipWhenCommand model)
  └→ Task 2 (SheetConfig field)
       └→ Task 3 (lifecycle implementation)
            └→ Task 4 (E2E tests)
                 └→ Task 5 (type check + lint)
                      └→ Task 6 (issue-solver score)
                           └→ Task 7 (score template tests)
                                └→ Task 8 (final verification)
```

Tasks 1-5 deliver `skip_when_command` (#71).
Tasks 6-7 deliver the issue solver score (#72).
Task 8 verifies everything together.
