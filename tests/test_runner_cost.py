"""Tests for CostMixin (src/mozart/execution/runner/cost.py)."""

from __future__ import annotations

import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from mozart.backends.base import ExecutionResult
from mozart.core.checkpoint import CheckpointState, SheetState
from mozart.core.config.execution import CostLimitConfig
from mozart.execution.runner.cost import CostMixin
from mozart.execution.runner.models import RunSummary


# ---------------------------------------------------------------------------
# Concrete test class that inherits CostMixin
# ---------------------------------------------------------------------------


class _CostRunner(CostMixin):
    """Concrete test class inheriting CostMixin with stubbed base attributes."""

    def __init__(
        self,
        cost_limits: CostLimitConfig | None = None,
        circuit_breaker: object | None = None,
        summary: RunSummary | None = None,
    ) -> None:
        # Provide the attributes CostMixin expects from base.py
        config = MagicMock()
        config.cost_limits = cost_limits or CostLimitConfig(enabled=False)
        self.config = config

        self.console = MagicMock()
        self._logger = MagicMock()
        self._circuit_breaker = circuit_breaker
        self._summary = summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_JOB_ID = "test-cost-job"
_JOB_NAME = "test-cost"
_TOTAL_SHEETS = 5


def _make_sheet_state(sheet_num: int = 1) -> SheetState:
    return SheetState(sheet_num=sheet_num)


def _make_checkpoint_state() -> CheckpointState:
    return CheckpointState(
        job_id=_JOB_ID,
        job_name=_JOB_NAME,
        total_sheets=_TOTAL_SHEETS,
    )


def _make_summary() -> RunSummary:
    return RunSummary(
        job_id=_JOB_ID,
        job_name=_JOB_NAME,
        total_sheets=_TOTAL_SHEETS,
    )


def _make_result(
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    tokens_used: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> ExecutionResult:
    return ExecutionResult(
        success=True,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=1.0,
        exit_code=0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokens_used=tokens_used,
    )


# ---------------------------------------------------------------------------
# Tests: _track_cost
# ---------------------------------------------------------------------------


class TestTrackCostExactTokens:
    """_track_cost with exact input_tokens and output_tokens (API backend)."""

    @pytest.mark.asyncio
    async def test_returns_exact_counts_and_full_confidence(self) -> None:
        runner = _CostRunner()
        result = _make_result(input_tokens=1000, output_tokens=500)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        inp, out, cost, conf = await runner._track_cost(result, sheet_state, state)

        assert inp == 1000
        assert out == 500
        assert conf == 1.0

    @pytest.mark.asyncio
    async def test_calculates_cost_with_default_rates(self) -> None:
        # Default rates: 0.003 per 1k input, 0.015 per 1k output
        runner = _CostRunner()
        result = _make_result(input_tokens=2000, output_tokens=1000)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        inp, out, cost, conf = await runner._track_cost(result, sheet_state, state)

        expected = (2000 / 1000 * 0.003) + (1000 / 1000 * 0.015)
        assert cost == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_calculates_cost_with_custom_rates(self) -> None:
        # Opus pricing: 0.015 input, 0.075 output
        limits = CostLimitConfig(
            enabled=False,
            cost_per_1k_input_tokens=0.015,
            cost_per_1k_output_tokens=0.075,
        )
        runner = _CostRunner(cost_limits=limits)
        result = _make_result(input_tokens=2000, output_tokens=1000)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        inp, out, cost, conf = await runner._track_cost(result, sheet_state, state)

        expected = (2000 / 1000 * 0.015) + (1000 / 1000 * 0.075)
        assert cost == pytest.approx(expected)


class TestTrackCostLegacyTokensUsed:
    """_track_cost with deprecated tokens_used field."""

    @pytest.mark.asyncio
    async def test_returns_legacy_confidence(self) -> None:
        runner = _CostRunner()
        result = _make_result(tokens_used=500)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inp, out, cost, conf = await runner._track_cost(result, sheet_state, state)

        assert conf == 0.85

    @pytest.mark.asyncio
    async def test_emits_deprecation_warning(self) -> None:
        runner = _CostRunner()
        result = _make_result(tokens_used=500)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await runner._track_cost(result, sheet_state, state)

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert "tokens_used" in str(deprecation_warnings[0].message)

    @pytest.mark.asyncio
    async def test_estimates_input_as_2x_output(self) -> None:
        runner = _CostRunner()
        result = _make_result(tokens_used=500)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            inp, out, cost, conf = await runner._track_cost(result, sheet_state, state)

        assert out == 500
        assert inp == 1000  # 2x heuristic


class TestTrackCostEstimatedTokens:
    """_track_cost with no token fields (CLI backend fallback)."""

    @pytest.mark.asyncio
    async def test_returns_estimated_confidence(self) -> None:
        runner = _CostRunner()
        result = _make_result(stdout="Hello world output", stderr="")
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        inp, out, cost, conf = await runner._track_cost(result, sheet_state, state)

        assert conf == 0.7

    @pytest.mark.asyncio
    async def test_estimates_from_char_count(self) -> None:
        runner = _CostRunner()
        stdout = "a" * 400  # 400 chars => 100 output tokens
        result = _make_result(stdout=stdout, stderr="")
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        inp, out, cost, conf = await runner._track_cost(result, sheet_state, state)

        assert out == 100  # 400 / 4
        assert inp == 200  # 2x heuristic

    @pytest.mark.asyncio
    async def test_minimum_one_token_on_empty_output(self) -> None:
        runner = _CostRunner()
        result = _make_result(stdout="", stderr="")
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        inp, out, cost, conf = await runner._track_cost(result, sheet_state, state)

        assert out == 1  # min 1 token
        assert inp == 2  # 2x heuristic

    @pytest.mark.asyncio
    async def test_combines_stdout_and_stderr(self) -> None:
        runner = _CostRunner()
        # 200 chars stdout + 200 chars stderr = 400 chars => 100 tokens
        result = _make_result(stdout="a" * 200, stderr="b" * 200)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        inp, out, cost, conf = await runner._track_cost(result, sheet_state, state)

        assert out == 100
        assert inp == 200


class TestTrackCostStateUpdates:
    """_track_cost correctly updates sheet_state, checkpoint state, and summary."""

    @pytest.mark.asyncio
    async def test_updates_sheet_state(self) -> None:
        runner = _CostRunner()
        result = _make_result(input_tokens=1000, output_tokens=500)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        await runner._track_cost(result, sheet_state, state)

        assert sheet_state.input_tokens == 1000
        assert sheet_state.output_tokens == 500
        assert sheet_state.estimated_cost is not None
        assert sheet_state.estimated_cost > 0
        assert sheet_state.cost_confidence == 1.0

    @pytest.mark.asyncio
    async def test_accumulates_on_sheet_state(self) -> None:
        """Calling _track_cost twice accumulates token counts on the same sheet."""
        runner = _CostRunner()
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        result1 = _make_result(input_tokens=100, output_tokens=50)
        result2 = _make_result(input_tokens=200, output_tokens=100)

        await runner._track_cost(result1, sheet_state, state)
        await runner._track_cost(result2, sheet_state, state)

        assert sheet_state.input_tokens == 300
        assert sheet_state.output_tokens == 150

    @pytest.mark.asyncio
    async def test_confidence_uses_latest_not_min(self) -> None:
        """Confidence tracks latest value, not minimum across calls."""
        runner = _CostRunner()
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        # First call: estimated (0.7 confidence)
        result1 = _make_result(stdout="a" * 100, stderr="")
        await runner._track_cost(result1, sheet_state, state)
        assert sheet_state.cost_confidence == 0.7

        # Second call: exact (1.0 confidence)
        result2 = _make_result(input_tokens=100, output_tokens=50)
        await runner._track_cost(result2, sheet_state, state)
        assert sheet_state.cost_confidence == 1.0

    @pytest.mark.asyncio
    async def test_updates_checkpoint_state(self) -> None:
        runner = _CostRunner()
        result = _make_result(input_tokens=1000, output_tokens=500)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        await runner._track_cost(result, sheet_state, state)

        assert state.total_input_tokens == 1000
        assert state.total_output_tokens == 500
        assert state.total_estimated_cost > 0

    @pytest.mark.asyncio
    async def test_accumulates_on_checkpoint_state(self) -> None:
        """Multiple _track_cost calls accumulate on checkpoint state."""
        runner = _CostRunner()
        state = _make_checkpoint_state()

        result1 = _make_result(input_tokens=100, output_tokens=50)
        result2 = _make_result(input_tokens=200, output_tokens=100)

        sheet1 = _make_sheet_state(sheet_num=1)
        sheet2 = _make_sheet_state(sheet_num=2)

        await runner._track_cost(result1, sheet1, state)
        await runner._track_cost(result2, sheet2, state)

        assert state.total_input_tokens == 300
        assert state.total_output_tokens == 150

    @pytest.mark.asyncio
    async def test_updates_summary(self) -> None:
        summary = _make_summary()
        runner = _CostRunner(summary=summary)
        result = _make_result(input_tokens=1000, output_tokens=500)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        await runner._track_cost(result, sheet_state, state)

        assert summary.total_input_tokens == 1000
        assert summary.total_output_tokens == 500
        assert summary.total_estimated_cost > 0

    @pytest.mark.asyncio
    async def test_no_summary_does_not_crash(self) -> None:
        """When _summary is None, _track_cost still works."""
        runner = _CostRunner(summary=None)
        result = _make_result(input_tokens=100, output_tokens=50)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        # Should not raise
        inp, out, cost, conf = await runner._track_cost(result, sheet_state, state)
        assert inp == 100


class TestTrackCostCircuitBreaker:
    """_track_cost calls circuit_breaker.record_cost when present."""

    @pytest.mark.asyncio
    async def test_calls_record_cost(self) -> None:
        cb = AsyncMock()
        runner = _CostRunner(circuit_breaker=cb)
        result = _make_result(input_tokens=1000, output_tokens=500)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        inp, out, cost, conf = await runner._track_cost(result, sheet_state, state)

        cb.record_cost.assert_awaited_once_with(1000, 500, cost)

    @pytest.mark.asyncio
    async def test_no_circuit_breaker_does_not_crash(self) -> None:
        """When _circuit_breaker is None, _track_cost does not crash."""
        runner = _CostRunner(circuit_breaker=None)
        result = _make_result(input_tokens=100, output_tokens=50)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        # Should not raise
        await runner._track_cost(result, sheet_state, state)


# ---------------------------------------------------------------------------
# Tests: _check_cost_limits
# ---------------------------------------------------------------------------


class TestCheckCostLimitsDisabled:

    def test_disabled_returns_false(self) -> None:
        limits = CostLimitConfig(enabled=False)
        runner = _CostRunner(cost_limits=limits)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        exceeded, reason = runner._check_cost_limits(sheet_state, state)

        assert exceeded is False
        assert reason is None


class TestCheckCostLimitsNoLimitsConfigured:
    """Enabled but within-bounds limits return (False, None)."""

    def test_within_limits_returns_false(self) -> None:
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_sheet=100.0,  # High enough to never trigger
        )
        runner = _CostRunner(cost_limits=limits)
        sheet_state = _make_sheet_state()
        sheet_state.estimated_cost = 0.01
        state = _make_checkpoint_state()

        exceeded, reason = runner._check_cost_limits(sheet_state, state)

        assert exceeded is False
        assert reason is None


class TestCheckCostLimitsPerSheet:
    """_check_cost_limits enforces per-sheet cost limits."""

    def test_exceeded_returns_true_with_reason(self) -> None:
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_sheet=1.00,
        )
        runner = _CostRunner(cost_limits=limits)
        sheet_state = _make_sheet_state()
        sheet_state.estimated_cost = 1.50  # Over the $1.00 limit
        state = _make_checkpoint_state()

        exceeded, reason = runner._check_cost_limits(sheet_state, state)

        assert exceeded is True
        assert reason is not None
        assert "Sheet cost" in reason
        assert "$1.50" in reason or "1.5" in reason
        assert "$1.00" in reason

    def test_within_limit_returns_false(self) -> None:
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_sheet=5.00,
        )
        runner = _CostRunner(cost_limits=limits)
        sheet_state = _make_sheet_state()
        sheet_state.estimated_cost = 2.50
        state = _make_checkpoint_state()

        exceeded, reason = runner._check_cost_limits(sheet_state, state)

        assert exceeded is False
        assert reason is None

    def test_none_estimated_cost_treated_as_zero(self) -> None:
        """When sheet_state.estimated_cost is None, treat as 0.0."""
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_sheet=1.00,
        )
        runner = _CostRunner(cost_limits=limits)
        sheet_state = _make_sheet_state()
        # estimated_cost defaults to None
        assert sheet_state.estimated_cost is None
        state = _make_checkpoint_state()

        exceeded, reason = runner._check_cost_limits(sheet_state, state)

        assert exceeded is False
        assert reason is None


class TestCheckCostLimitsPerJob:
    """_check_cost_limits enforces per-job cost limits."""

    def test_exceeded_returns_true_with_reason(self) -> None:
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_job=10.00,
        )
        runner = _CostRunner(cost_limits=limits)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()
        state.total_estimated_cost = 12.50  # Over the $10.00 limit

        exceeded, reason = runner._check_cost_limits(sheet_state, state)

        assert exceeded is True
        assert reason is not None
        assert "Job cost" in reason
        assert "$10.00" in reason

    def test_within_limit_returns_false(self) -> None:
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_job=100.00,
        )
        runner = _CostRunner(cost_limits=limits)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()
        state.total_estimated_cost = 50.00

        exceeded, reason = runner._check_cost_limits(sheet_state, state)

        assert exceeded is False
        assert reason is None

    def test_per_sheet_checked_before_per_job(self) -> None:
        """When both limits are exceeded, per-sheet is reported (checked first)."""
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_sheet=1.00,
            max_cost_per_job=10.00,
        )
        runner = _CostRunner(cost_limits=limits)
        sheet_state = _make_sheet_state()
        sheet_state.estimated_cost = 2.00  # Over sheet limit
        state = _make_checkpoint_state()
        state.total_estimated_cost = 15.00  # Also over job limit

        exceeded, reason = runner._check_cost_limits(sheet_state, state)

        assert exceeded is True
        assert "Sheet cost" in reason  # Sheet limit hit first


class TestCheckCostLimitsWarningThreshold:
    """_check_cost_limits emits warning when approaching job limit."""

    def test_emits_warning_at_threshold(self) -> None:
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_job=100.00,
            warn_at_percent=80.0,
        )
        runner = _CostRunner(cost_limits=limits)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()
        # 85% of $100 = $85, above the 80% threshold
        state.total_estimated_cost = 85.00
        state.cost_limit_reached = False

        exceeded, reason = runner._check_cost_limits(sheet_state, state)

        # Not exceeded (85 < 100), but warning should be emitted
        assert exceeded is False
        assert reason is None

        # Logger and console should have been called for the warning
        runner._logger.warning.assert_called_once()
        call_args = runner._logger.warning.call_args
        assert call_args[0][0] == "cost.warning_threshold"

        runner.console.print.assert_called_once()
        printed = runner.console.print.call_args[0][0]
        assert "Cost warning" in printed

    def test_no_warning_below_threshold(self) -> None:
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_job=100.00,
            warn_at_percent=80.0,
        )
        runner = _CostRunner(cost_limits=limits)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()
        # 50% of $100 = $50, below the 80% threshold
        state.total_estimated_cost = 50.00
        state.cost_limit_reached = False

        exceeded, reason = runner._check_cost_limits(sheet_state, state)

        assert exceeded is False
        runner._logger.warning.assert_not_called()
        runner.console.print.assert_not_called()

    def test_no_warning_when_cost_limit_already_reached(self) -> None:
        """Warning is suppressed when cost_limit_reached is already True."""
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_job=100.00,
            warn_at_percent=80.0,
        )
        runner = _CostRunner(cost_limits=limits)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()
        state.total_estimated_cost = 85.00
        state.cost_limit_reached = True

        runner._check_cost_limits(sheet_state, state)

        runner._logger.warning.assert_not_called()
        runner.console.print.assert_not_called()

    def test_custom_warn_percent(self) -> None:
        """Custom warn_at_percent is respected."""
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_job=100.00,
            warn_at_percent=50.0,
        )
        runner = _CostRunner(cost_limits=limits)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()
        # 55% of $100 = $55, above the 50% threshold
        state.total_estimated_cost = 55.00
        state.cost_limit_reached = False

        runner._check_cost_limits(sheet_state, state)

        runner._logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# Integration-style tests: _track_cost + _check_cost_limits together
# ---------------------------------------------------------------------------


class TestCostTrackingIntegration:
    """End-to-end cost tracking and limit checking."""

    @pytest.mark.asyncio
    async def test_track_then_check_exceeds_sheet_limit(self) -> None:
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_sheet=0.01,
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
        )
        runner = _CostRunner(cost_limits=limits)
        # 10000 output tokens at $0.015/1k = $0.15, exceeds $0.01 limit
        result = _make_result(input_tokens=5000, output_tokens=10000)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        await runner._track_cost(result, sheet_state, state)
        exceeded, reason = runner._check_cost_limits(sheet_state, state)

        assert exceeded is True
        assert "Sheet cost" in reason

    @pytest.mark.asyncio
    async def test_track_then_check_within_limits(self) -> None:
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_sheet=10.00,
            max_cost_per_job=100.00,
        )
        runner = _CostRunner(cost_limits=limits)
        result = _make_result(input_tokens=100, output_tokens=50)
        sheet_state = _make_sheet_state()
        state = _make_checkpoint_state()

        await runner._track_cost(result, sheet_state, state)
        exceeded, reason = runner._check_cost_limits(sheet_state, state)

        assert exceeded is False
        assert reason is None

    @pytest.mark.asyncio
    async def test_cumulative_job_cost_exceeds_limit(self) -> None:
        """Multiple sheet executions accumulate and eventually exceed job limit."""
        limits = CostLimitConfig(
            enabled=True,
            max_cost_per_job=0.05,
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
        )
        runner = _CostRunner(cost_limits=limits)
        state = _make_checkpoint_state()

        # Execute 5 sheets, each with 2000 output tokens
        # Each sheet cost: (4000/1000 * 0.003) + (2000/1000 * 0.015) = 0.012 + 0.030 = 0.042
        for i in range(1, 6):
            result = _make_result(input_tokens=4000, output_tokens=2000)
            sheet_state = _make_sheet_state(sheet_num=i)
            await runner._track_cost(result, sheet_state, state)

        # After 5 sheets: total = 5 * 0.042 = 0.21, exceeds $0.05
        final_sheet = _make_sheet_state(sheet_num=5)
        exceeded, reason = runner._check_cost_limits(final_sheet, state)

        assert exceeded is True
        assert "Job cost" in reason
