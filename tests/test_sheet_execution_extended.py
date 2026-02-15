"""Extended tests for SheetExecutionMixin — targeting uncovered methods.

This module fills gaps in test_sheet_execution.py, covering:
- _classify_success_outcome (static)
- _is_escalation_available
- _should_enter_completion_mode
- _count_new_validations (static)
- _extract_agent_feedback
- _log_incomplete_validations
- _run_preflight_checks
- _enforce_cost_limits
- _check_execution_guards
- _populate_cross_sheet_context
- _capture_cross_sheet_files
- _build_sheet_context (edge cases)
- _prepare_sheet_execution (integration-level)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mozart.backends.base import ExecutionResult
from mozart.core.checkpoint import (
    CheckpointState,
    OutcomeCategory,
)
from mozart.core.config import JobConfig
from mozart.execution.runner.models import (
    FatalError,
    GracefulShutdownError,
    SheetExecutionMode,
)

# ---------------------------------------------------------------------------
# Fixtures (reusing the pattern from test_sheet_execution.py)
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, overrides: dict[str, Any] | None = None) -> JobConfig:
    """Build a minimal JobConfig for testing."""
    base: dict[str, Any] = {
        "name": "test-job",
        "description": "Unit test job",
        "workspace": str(tmp_path / "workspace"),
        "backend": {"type": "claude_cli", "skip_permissions": True},
        "sheet": {"size": 5, "total_items": 10},
        "prompt": {"template": "Process sheet {{ sheet_num }}."},
        "retry": {
            "max_retries": 3,
            "max_completion_attempts": 2,
            "base_delay_seconds": 1.0,
            "exponential_base": 2.0,
            "max_delay_seconds": 60.0,
            "jitter": False,
            "completion_threshold_percent": 60,
        },
        "validations": [],
    }
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                base[k].update(v)
            else:
                base[k] = v
    (tmp_path / "workspace").mkdir(exist_ok=True)
    return JobConfig(**base)


class _MockMixin:
    """Minimal mock of the mixin host attributes expected by SheetExecutionMixin."""

    def __init__(self, config: JobConfig) -> None:
        from rich.console import Console

        from mozart.core.errors import ErrorClassifier
        from mozart.core.logging import get_logger
        from mozart.execution.preflight import PreflightChecker
        from mozart.execution.retry_strategy import AdaptiveRetryStrategy, RetryStrategyConfig
        from mozart.prompts.templating import PromptBuilder

        self.config = config
        self.backend = MagicMock()
        self.backend.execute = AsyncMock(
            return_value=ExecutionResult(
                success=True, exit_code=0, stdout="done", stderr="", duration_seconds=5.0,
            )
        )
        self.backend.set_output_log_path = MagicMock()

        self.state_backend = AsyncMock()
        self.state_backend.save = AsyncMock()

        self.console = Console(quiet=True)
        self.outcome_store = None
        self.escalation_handler = None
        self.checkpoint_handler = None
        self.judgment_client = None

        self.prompt_builder = PromptBuilder(config.prompt)
        self.error_classifier = ErrorClassifier.from_config([])
        self.preflight_checker = PreflightChecker(
            workspace=config.workspace,
            working_directory=config.workspace,
        )

        self._logger = get_logger("test")
        self._circuit_breaker = None
        self._global_learning_store = None
        self._grounding_engine = None
        self._healing_coordinator = None
        self._retry_strategy = AdaptiveRetryStrategy(
            config=RetryStrategyConfig(
                base_delay=1.0,
                max_delay=60.0,
                exponential_base=2.0,
                jitter_factor=0.0,
            ),
        )

        self._current_sheet_num: int | None = None
        self._execution_progress_snapshots: list[dict[str, Any]] = []
        self._last_progress_monotonic: float = 0.0
        self._current_sheet_patterns: list[str] = []
        self._applied_pattern_ids: list[str] = []
        self._exploration_pattern_ids: list[str] = []
        self._exploitation_pattern_ids: list[str] = []
        self._shutdown_requested = False
        self._summary = None
        self._self_healing_enabled = False

    async def _interruptible_sleep(self, seconds: float) -> None:
        pass

    def _query_relevant_patterns(
        self, job_id: str, sheet_num: int, context_tags: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        return [], []

    async def _record_pattern_feedback(self, pattern_ids: Any, context: Any) -> None:
        pass

    async def _try_self_healing(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def _handle_rate_limit(self, state: Any, error_code: str = "E101", suggested_wait_seconds: Any = None) -> None:
        pass

    async def _track_cost(self, result: Any, sheet_state: Any, state: Any) -> None:
        pass

    def _check_cost_limits(self, sheet_state: Any, state: Any) -> tuple[bool, str | None]:
        return False, None


from mozart.execution.runner.context import ContextBuildingMixin
from mozart.execution.runner.recovery import RecoveryMixin
from mozart.execution.runner.sheet import SheetExecutionMixin


class _TestableSheetMixin(_MockMixin, SheetExecutionMixin, ContextBuildingMixin, RecoveryMixin):
    """Concrete class that combines the mixin with mock infrastructure."""
    pass


@pytest.fixture
def mixin(tmp_path: Path) -> _TestableSheetMixin:
    config = _make_config(tmp_path)
    return _TestableSheetMixin(config)


def _make_state(job_id: str = "test-job", total_sheets: int = 2) -> CheckpointState:
    """Build a minimal CheckpointState."""
    return CheckpointState(
        job_id=job_id,
        job_name="test-job",
        total_sheets=total_sheets,
    )


def _make_mock_vr(results: list[MagicMock] | None = None) -> MagicMock:
    """Build a mock SheetValidationResult."""
    vr = MagicMock()
    vr.results = results or []
    vr.all_passed = True
    vr.executed_pass_percentage = 100.0
    vr.pass_percentage = 100.0
    vr.get_passed_results.return_value = results or []
    vr.get_failed_results.return_value = []
    vr.to_dict_list.return_value = []
    vr.get_semantic_summary.return_value = {"dominant_category": "missing_output"}
    vr.get_actionable_hints.return_value = []
    return vr


# ===========================================================================
# Tests: _classify_success_outcome (static method)
# ===========================================================================


class TestClassifySuccessOutcome:
    """Tests for the static _classify_success_outcome method."""

    def test_success_without_retry_zero_attempts(self) -> None:
        """0 normal attempts + 0 completion = first try success."""
        outcome, first = SheetExecutionMixin._classify_success_outcome(0, 0)
        assert outcome == OutcomeCategory.SUCCESS_FIRST_TRY
        assert first is True

    def test_success_without_retry_one_attempt(self) -> None:
        """1 normal attempt + 0 completion = first try success."""
        outcome, first = SheetExecutionMixin._classify_success_outcome(1, 0)
        assert outcome == OutcomeCategory.SUCCESS_FIRST_TRY
        assert first is True

    def test_retry_success(self) -> None:
        """2 normal attempts + 0 completion = retry success."""
        outcome, first = SheetExecutionMixin._classify_success_outcome(2, 0)
        assert outcome == OutcomeCategory.SUCCESS_RETRY
        assert first is False

    def test_completion_success(self) -> None:
        """Any completion attempts > 0 = completion success."""
        outcome, first = SheetExecutionMixin._classify_success_outcome(1, 1)
        assert outcome == OutcomeCategory.SUCCESS_COMPLETION
        assert first is False

    def test_high_retry_with_completion(self) -> None:
        """Many retries + completion = completion success (completion takes priority)."""
        outcome, first = SheetExecutionMixin._classify_success_outcome(5, 3)
        assert outcome == OutcomeCategory.SUCCESS_COMPLETION
        assert first is False


# ===========================================================================
# Tests: _is_escalation_available
# ===========================================================================


class TestIsEscalationAvailable:
    """Tests for the _is_escalation_available method."""

    def test_both_enabled_and_handler_present(self, mixin: _TestableSheetMixin) -> None:
        mixin.config.learning.escalation_enabled = True
        mixin.escalation_handler = MagicMock()
        assert mixin._is_escalation_available() is True

    def test_enabled_but_no_handler(self, mixin: _TestableSheetMixin) -> None:
        mixin.config.learning.escalation_enabled = True
        mixin.escalation_handler = None
        assert mixin._is_escalation_available() is False

    def test_disabled_with_handler(self, mixin: _TestableSheetMixin) -> None:
        mixin.config.learning.escalation_enabled = False
        mixin.escalation_handler = MagicMock()
        assert mixin._is_escalation_available() is False

    def test_both_disabled(self, mixin: _TestableSheetMixin) -> None:
        mixin.config.learning.escalation_enabled = False
        mixin.escalation_handler = None
        assert mixin._is_escalation_available() is False


# ===========================================================================
# Tests: _should_enter_completion_mode
# ===========================================================================


class TestShouldEnterCompletionMode:
    """Tests for the _should_enter_completion_mode method."""

    def test_above_threshold_with_attempts_remaining(self, mixin: _TestableSheetMixin) -> None:
        # threshold is 60, max_completion_attempts is 2
        assert mixin._should_enter_completion_mode(pass_pct=75.0, completion_attempts=0) is True

    def test_at_threshold_returns_false(self, mixin: _TestableSheetMixin) -> None:
        """At exactly the threshold (not above), should be False."""
        assert mixin._should_enter_completion_mode(pass_pct=60.0, completion_attempts=0) is False

    def test_below_threshold_returns_false(self, mixin: _TestableSheetMixin) -> None:
        assert mixin._should_enter_completion_mode(pass_pct=50.0, completion_attempts=0) is False

    def test_above_threshold_but_exhausted(self, mixin: _TestableSheetMixin) -> None:
        assert mixin._should_enter_completion_mode(pass_pct=75.0, completion_attempts=2) is False


# ===========================================================================
# Tests: _count_new_validations (static method)
# ===========================================================================


class TestCountNewValidations:
    """Tests for the static _count_new_validations method."""

    def _make_rule(self, condition: str | None = None) -> MagicMock:
        rule = MagicMock()
        rule.condition = condition
        vr = MagicMock()
        vr.rule = rule
        return vr

    def test_no_condition_assumed_sheet_1(self) -> None:
        """Rules with no condition are assumed to originate at sheet 1."""
        results = [self._make_rule(None), self._make_rule(None)]
        vr = MagicMock()
        vr.results = results
        # At sheet 1, all no-condition rules are "new" (origin 1 >= sheet_num 1)
        count = SheetExecutionMixin._count_new_validations(vr, sheet_num=1)
        assert count == 2

    def test_no_condition_at_sheet_2(self) -> None:
        """At sheet 2, rules with origin 1 are inherited (not new)."""
        results = [self._make_rule(None)]
        vr = MagicMock()
        vr.results = results
        count = SheetExecutionMixin._count_new_validations(vr, sheet_num=2)
        assert count == 0

    def test_condition_with_sheet_num_threshold(self) -> None:
        """Rules with 'sheet_num >= 3' are new at sheet 3, inherited at sheet 4."""
        rule_s3 = self._make_rule("sheet_num >= 3")
        rule_s1 = self._make_rule(None)
        vr = MagicMock()
        vr.results = [rule_s3, rule_s1]
        # At sheet 3: rule_s3 (origin 3 >= 3 = True), rule_s1 (origin 1 >= 3 = False)
        assert SheetExecutionMixin._count_new_validations(vr, sheet_num=3) == 1

    def test_condition_with_stage_threshold(self) -> None:
        """Rules with 'stage >= 2' are recognized."""
        rule = self._make_rule("stage >= 2")
        vr = MagicMock()
        vr.results = [rule]
        assert SheetExecutionMixin._count_new_validations(vr, sheet_num=2) == 1
        assert SheetExecutionMixin._count_new_validations(vr, sheet_num=3) == 0

    def test_condition_with_exact_match(self) -> None:
        """Rules with 'sheet_num == 5' are recognized."""
        rule = self._make_rule("sheet_num == 5")
        vr = MagicMock()
        vr.results = [rule]
        assert SheetExecutionMixin._count_new_validations(vr, sheet_num=5) == 1
        assert SheetExecutionMixin._count_new_validations(vr, sheet_num=6) == 0


# ===========================================================================
# Tests: _extract_agent_feedback
# ===========================================================================


class TestExtractAgentFeedback:
    """Tests for the _extract_agent_feedback method."""

    def test_disabled_does_nothing(self, mixin: _TestableSheetMixin) -> None:
        mixin.config.feedback.enabled = False
        sheet_state = MagicMock()
        sheet_state.agent_feedback = None
        mixin._extract_agent_feedback(sheet_state, "some output")
        assert sheet_state.agent_feedback is None

    def test_empty_stdout_does_nothing(self, mixin: _TestableSheetMixin) -> None:
        mixin.config.feedback.enabled = True
        sheet_state = MagicMock()
        sheet_state.agent_feedback = None
        mixin._extract_agent_feedback(sheet_state, "")
        assert sheet_state.agent_feedback is None

    def test_json_format_extraction(self, mixin: _TestableSheetMixin) -> None:
        mixin.config.feedback.enabled = True
        mixin.config.feedback.format = "json"
        # Default pattern captures content between FEEDBACK markers
        pattern = mixin.config.feedback.pattern
        feedback_data = {"quality": "good", "score": 8}
        stdout = f"start\n<FEEDBACK>{json.dumps(feedback_data)}</FEEDBACK>\nend"
        # Update pattern to match the test data
        mixin.config.feedback.pattern = r"<FEEDBACK>(.*?)</FEEDBACK>"
        sheet_state = MagicMock()
        sheet_state.agent_feedback = None
        mixin._extract_agent_feedback(sheet_state, stdout)
        assert sheet_state.agent_feedback == feedback_data

    def test_text_format_extraction(self, mixin: _TestableSheetMixin) -> None:
        mixin.config.feedback.enabled = True
        mixin.config.feedback.format = "text"
        mixin.config.feedback.pattern = r"<FEEDBACK>(.*?)</FEEDBACK>"
        stdout = "start\n<FEEDBACK>This worked well</FEEDBACK>\nend"
        sheet_state = MagicMock()
        sheet_state.agent_feedback = None
        mixin._extract_agent_feedback(sheet_state, stdout)
        assert sheet_state.agent_feedback == {"text": "This worked well"}

    def test_invalid_json_logs_warning(self, mixin: _TestableSheetMixin) -> None:
        mixin.config.feedback.enabled = True
        mixin.config.feedback.format = "json"
        mixin.config.feedback.pattern = r"<FEEDBACK>(.*?)</FEEDBACK>"
        stdout = "<FEEDBACK>not-valid-json{</FEEDBACK>"
        sheet_state = MagicMock()
        sheet_state.agent_feedback = None
        mixin._extract_agent_feedback(sheet_state, stdout)
        # agent_feedback stays None on parse failure
        assert sheet_state.agent_feedback is None

    def test_no_match_does_nothing(self, mixin: _TestableSheetMixin) -> None:
        mixin.config.feedback.enabled = True
        mixin.config.feedback.pattern = r"<FEEDBACK>(.*?)</FEEDBACK>"
        stdout = "no feedback markers here"
        sheet_state = MagicMock()
        sheet_state.agent_feedback = None
        mixin._extract_agent_feedback(sheet_state, stdout)
        assert sheet_state.agent_feedback is None


# ===========================================================================
# Tests: _log_incomplete_validations
# ===========================================================================


class TestLogIncompleteValidations:
    """Tests for the _log_incomplete_validations method."""

    def test_returns_counts_and_percentage(self, mixin: _TestableSheetMixin) -> None:
        passed = [MagicMock(), MagicMock()]
        failed_rule = MagicMock()
        failed_rule.rule.description = "Output file missing"
        failed = [failed_rule]
        vr = MagicMock()
        vr.get_passed_results.return_value = passed
        vr.get_failed_results.return_value = failed
        vr.executed_pass_percentage = 66.7

        passed_count, failed_count, pass_pct = mixin._log_incomplete_validations(1, vr)
        assert passed_count == 2
        assert failed_count == 1
        assert pass_pct == 66.7

    def test_all_failed(self, mixin: _TestableSheetMixin) -> None:
        failed_rule = MagicMock()
        failed_rule.rule.description = "Check failed"
        vr = MagicMock()
        vr.get_passed_results.return_value = []
        vr.get_failed_results.return_value = [failed_rule]
        vr.executed_pass_percentage = 0.0

        passed_count, failed_count, pass_pct = mixin._log_incomplete_validations(1, vr)
        assert passed_count == 0
        assert failed_count == 1
        assert pass_pct == 0.0


# ===========================================================================
# Tests: _enforce_cost_limits
# ===========================================================================


class TestEnforceCostLimits:
    """Tests for the _enforce_cost_limits method."""

    @pytest.mark.asyncio
    async def test_disabled_does_nothing(self, mixin: _TestableSheetMixin) -> None:
        mixin.config.cost_limits.enabled = False
        state = _make_state()
        result = ExecutionResult(success=True, exit_code=0, stdout="", stderr="", duration_seconds=1.0)
        sheet_state = MagicMock()
        # Should not raise
        await mixin._enforce_cost_limits(result, sheet_state, state, 1)

    @pytest.mark.asyncio
    async def test_raises_on_cost_exceeded(self, mixin: _TestableSheetMixin) -> None:
        mixin.config.cost_limits.enabled = True
        mixin._check_cost_limits = MagicMock(return_value=(True, "Max cost $5.00 exceeded"))
        state = _make_state()
        state.total_estimated_cost = 6.0
        result = ExecutionResult(success=True, exit_code=0, stdout="", stderr="", duration_seconds=1.0)
        sheet_state = MagicMock()

        with pytest.raises(GracefulShutdownError, match="Cost limit exceeded"):
            await mixin._enforce_cost_limits(result, sheet_state, state, 1)

        assert state.cost_limit_reached is True

    @pytest.mark.asyncio
    async def test_no_raise_when_under_limit(self, mixin: _TestableSheetMixin) -> None:
        mixin.config.cost_limits.enabled = True
        mixin._check_cost_limits = MagicMock(return_value=(False, None))
        state = _make_state()
        result = ExecutionResult(success=True, exit_code=0, stdout="", stderr="", duration_seconds=1.0)
        sheet_state = MagicMock()
        # Should not raise
        await mixin._enforce_cost_limits(result, sheet_state, state, 1)


# ===========================================================================
# Tests: _check_execution_guards
# ===========================================================================


class TestCheckExecutionGuards:
    """Tests for the _check_execution_guards method."""

    @pytest.mark.asyncio
    async def test_no_guards_returns_false(self, mixin: _TestableSheetMixin) -> None:
        """No circuit breaker or cross-workspace = proceed normally."""
        result = await mixin._check_execution_guards(sheet_num=1)
        assert result is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks(self, mixin: _TestableSheetMixin) -> None:
        """Circuit breaker OPEN state blocks execution."""
        cb = AsyncMock()
        cb.can_execute.return_value = False
        cb.time_until_retry.return_value = 0.1
        cb.get_state.return_value = MagicMock(value="open")
        mixin._circuit_breaker = cb

        result = await mixin._check_execution_guards(sheet_num=1)
        assert result is True
        cb.can_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_allows(self, mixin: _TestableSheetMixin) -> None:
        """Circuit breaker CLOSED state allows execution."""
        cb = AsyncMock()
        cb.can_execute.return_value = True
        mixin._circuit_breaker = cb

        result = await mixin._check_execution_guards(sheet_num=1)
        assert result is False

    @pytest.mark.asyncio
    async def test_cross_workspace_rate_limit_honored(self, mixin: _TestableSheetMixin) -> None:
        """Cross-workspace rate limit blocks execution when active."""
        mixin.config.circuit_breaker.cross_workspace_coordination = True
        mixin.config.circuit_breaker.honor_other_jobs_rate_limits = True

        store = MagicMock()
        store.is_rate_limited.return_value = (True, 0.1)
        mixin._global_learning_store = store

        result = await mixin._check_execution_guards(sheet_num=1)
        # Returns False because cross-workspace rate limit uses sleep + continue, not True
        # (it doesn't short-circuit the caller's loop, just delays)
        assert result is False


# ===========================================================================
# Tests: _populate_cross_sheet_context
# ===========================================================================


class TestPopulateCrossSheetContext:
    """Tests for the _populate_cross_sheet_context method."""

    def test_auto_capture_stdout_from_prior_sheets(self, mixin: _TestableSheetMixin, tmp_path: Path) -> None:
        config = _make_config(tmp_path, overrides={
            "cross_sheet": {
                "auto_capture_stdout": True,
                "lookback_sheets": 0,
                "max_output_chars": 500,
            },
        })
        mixin.config = config

        state = _make_state(total_sheets=3)
        # Populate prior sheet states with stdout
        state.mark_sheet_started(1)
        sheet1 = state.sheets[1]
        sheet1.stdout_tail = "Sheet 1 output here"
        state.mark_sheet_started(2)
        sheet2 = state.sheets[2]
        sheet2.stdout_tail = "Sheet 2 output here"

        context = MagicMock()
        context.previous_outputs = {}
        context.previous_files = {}

        mixin._populate_cross_sheet_context(context, state, sheet_num=3, cross_sheet=config.cross_sheet)
        assert 1 in context.previous_outputs
        assert 2 in context.previous_outputs
        assert context.previous_outputs[1] == "Sheet 1 output here"

    def test_lookback_limits_scope(self, mixin: _TestableSheetMixin, tmp_path: Path) -> None:
        config = _make_config(tmp_path, overrides={
            "cross_sheet": {
                "auto_capture_stdout": True,
                "lookback_sheets": 1,
                "max_output_chars": 500,
            },
        })
        mixin.config = config

        state = _make_state(total_sheets=3)
        state.mark_sheet_started(1)
        state.sheets[1].stdout_tail = "Sheet 1 output"
        state.mark_sheet_started(2)
        state.sheets[2].stdout_tail = "Sheet 2 output"

        context = MagicMock()
        context.previous_outputs = {}
        context.previous_files = {}

        mixin._populate_cross_sheet_context(context, state, sheet_num=3, cross_sheet=config.cross_sheet)
        # lookback_sheets=1 → only sheet 2 (max(1, 3-1) = 2)
        assert 1 not in context.previous_outputs
        assert 2 in context.previous_outputs

    def test_truncates_long_output(self, mixin: _TestableSheetMixin, tmp_path: Path) -> None:
        config = _make_config(tmp_path, overrides={
            "cross_sheet": {
                "auto_capture_stdout": True,
                "lookback_sheets": 0,
                "max_output_chars": 10,
            },
        })
        mixin.config = config

        state = _make_state(total_sheets=2)
        state.mark_sheet_started(1)
        state.sheets[1].stdout_tail = "This is a very long output string"

        context = MagicMock()
        context.previous_outputs = {}
        context.previous_files = {}

        mixin._populate_cross_sheet_context(context, state, sheet_num=2, cross_sheet=config.cross_sheet)
        assert context.previous_outputs[1].endswith("... [truncated]")
        # First 10 chars + truncation marker
        assert context.previous_outputs[1].startswith("This is a ")


# ===========================================================================
# Tests: _capture_cross_sheet_files
# ===========================================================================


class TestCaptureCrossSheetFiles:
    """Tests for the _capture_cross_sheet_files method."""

    def test_captures_matching_files(self, mixin: _TestableSheetMixin, tmp_path: Path) -> None:
        config = _make_config(tmp_path, overrides={
            "cross_sheet": {
                "capture_files": ["*.txt"],
                "max_output_chars": 500,
            },
        })
        mixin.config = config

        # Create a file in workspace
        workspace = tmp_path / "workspace"
        test_file = workspace / "output.txt"
        test_file.write_text("file content here")

        state = _make_state()
        state.started_at = datetime(2020, 1, 1, tzinfo=UTC)  # Old start so file isn't stale

        context = MagicMock()
        context.previous_files = {}

        mixin._capture_cross_sheet_files(context, state, sheet_num=2, cross_sheet=config.cross_sheet)
        assert str(test_file) in context.previous_files
        assert context.previous_files[str(test_file)] == "file content here"

    def test_skips_stale_files(self, mixin: _TestableSheetMixin, tmp_path: Path) -> None:
        config = _make_config(tmp_path, overrides={
            "cross_sheet": {
                "capture_files": ["*.txt"],
                "max_output_chars": 500,
            },
        })
        mixin.config = config

        workspace = tmp_path / "workspace"
        test_file = workspace / "old.txt"
        test_file.write_text("old content")

        state = _make_state()
        # Set started_at to future so the file is considered stale
        state.started_at = datetime(2099, 1, 1, tzinfo=UTC)

        context = MagicMock()
        context.previous_files = {}

        mixin._capture_cross_sheet_files(context, state, sheet_num=2, cross_sheet=config.cross_sheet)
        assert str(test_file) not in context.previous_files

    def test_truncates_large_files(self, mixin: _TestableSheetMixin, tmp_path: Path) -> None:
        config = _make_config(tmp_path, overrides={
            "cross_sheet": {
                "capture_files": ["*.txt"],
                "max_output_chars": 10,
            },
        })
        mixin.config = config

        workspace = tmp_path / "workspace"
        test_file = workspace / "big.txt"
        test_file.write_text("a" * 100)

        state = _make_state()
        state.started_at = datetime(2020, 1, 1, tzinfo=UTC)

        context = MagicMock()
        context.previous_files = {}

        mixin._capture_cross_sheet_files(context, state, sheet_num=2, cross_sheet=config.cross_sheet)
        assert context.previous_files[str(test_file)].endswith("... [truncated]")


# ===========================================================================
# Tests: _run_preflight_checks
# ===========================================================================


class TestRunPreflightChecks:
    """Tests for the _run_preflight_checks method."""

    def test_returns_preflight_result(self, mixin: _TestableSheetMixin) -> None:
        state = _make_state()
        state.mark_sheet_started(1)
        result = mixin._run_preflight_checks(
            prompt="Process items 1-5",
            sheet_context={"sheet_num": 1, "workspace": str(mixin.config.workspace)},
            sheet_num=1,
            state=state,
        )
        assert result is not None
        assert hasattr(result, "has_errors")
        assert hasattr(result, "prompt_metrics")

    def test_stores_metrics_in_sheet_state(self, mixin: _TestableSheetMixin) -> None:
        state = _make_state()
        state.mark_sheet_started(1)
        result = mixin._run_preflight_checks(
            prompt="Process items 1-5",
            sheet_context={"sheet_num": 1, "workspace": str(mixin.config.workspace)},
            sheet_num=1,
            state=state,
        )
        sheet_state = state.sheets[1]
        # Preflight stores metrics dict and warnings list on SheetState
        assert isinstance(sheet_state.prompt_metrics, dict)
        assert sheet_state.prompt_metrics["estimated_tokens"] >= 0
        assert isinstance(sheet_state.preflight_warnings, list)


# ===========================================================================
# Tests: _build_sheet_context edge cases
# ===========================================================================


class TestBuildSheetContextEdgeCases:
    """Additional tests for _build_sheet_context covering edge cases."""

    def test_context_has_fan_out_metadata(self, mixin: _TestableSheetMixin) -> None:
        """Context should include fan_out metadata."""
        context = mixin._build_sheet_context(sheet_num=1)
        assert hasattr(context, "stage")
        assert hasattr(context, "instance")
        assert hasattr(context, "fan_count")
        assert hasattr(context, "total_stages")

    def test_first_sheet_no_cross_sheet(self, mixin: _TestableSheetMixin) -> None:
        """First sheet with cross_sheet config but no prior data should work."""
        state = _make_state(total_sheets=3)
        context = mixin._build_sheet_context(sheet_num=1, state=state)
        # Should not crash — no prior sheets to cross-reference
        assert context is not None

    def test_context_with_cross_sheet_config(self, mixin: _TestableSheetMixin, tmp_path: Path) -> None:
        """When cross_sheet is configured and state has prior data, context is enriched."""
        config = _make_config(tmp_path, overrides={
            "cross_sheet": {
                "auto_capture_stdout": True,
                "lookback_sheets": 0,
                "max_output_chars": 500,
            },
        })
        mixin.config = config
        mixin.prompt_builder = __import__("mozart.prompts.templating", fromlist=["PromptBuilder"]).PromptBuilder(config.prompt)

        state = _make_state(total_sheets=3)
        state.mark_sheet_started(1)
        state.sheets[1].stdout_tail = "Sheet 1 output"

        context = mixin._build_sheet_context(sheet_num=2, state=state)
        assert 1 in context.previous_outputs


# ===========================================================================
# Tests: _prepare_sheet_execution (integration-level)
# ===========================================================================


class TestPrepareSheetExecution:
    """Integration-level tests for _prepare_sheet_execution."""

    @pytest.mark.asyncio
    async def test_success_returns_setup_triple(self, mixin: _TestableSheetMixin) -> None:
        state = _make_state()
        setup, context, engine = await mixin._prepare_sheet_execution(state, sheet_num=1)
        assert setup.original_prompt is not None
        assert setup.current_prompt == setup.original_prompt
        assert setup.current_mode == SheetExecutionMode.NORMAL
        assert context is not None
        assert engine is not None

    @pytest.mark.asyncio
    async def test_preflight_error_raises_fatal(self, mixin: _TestableSheetMixin) -> None:
        """When preflight has fatal errors, raises FatalError."""
        from mozart.execution.preflight import PreflightResult, PromptMetrics

        # Mock preflight_checker to return errors
        bad_result = PreflightResult(
            errors=["Critical: workspace does not exist"],
            warnings=[],
            prompt_metrics=PromptMetrics(
                character_count=0,
                estimated_tokens=0,
                line_count=0,
                has_file_references=False,
            ),
        )
        mixin.preflight_checker.check = MagicMock(return_value=bad_result)

        state = _make_state()
        with pytest.raises(FatalError, match="Preflight check failed"):
            await mixin._prepare_sheet_execution(state, sheet_num=1)
