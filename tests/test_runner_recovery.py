"""Tests for RecoveryMixin (src/mozart/execution/runner/recovery.py)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mozart.backends.base import ExecutionResult
from mozart.core.checkpoint import CheckpointState
from mozart.core.config import JobConfig
from mozart.core.errors import ClassificationResult, ClassifiedError, ErrorClassifier
from mozart.core.errors.codes import ErrorCategory, ErrorCode
from mozart.execution.runner.models import FatalError
from mozart.execution.runner.recovery import RecoveryMixin


# ---------------------------------------------------------------------------
# Helpers — minimal concrete class that inherits RecoveryMixin
# ---------------------------------------------------------------------------


def _make_config(
    *,
    backend_type: str = "claude_cli",
    cli_model: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    wait_minutes: int = 60,
    max_waits: int = 24,
    max_quota_waits: int = 48,
    base_delay_seconds: float = 5.0,
    exponential_base: float = 2.0,
    max_delay_seconds: float = 300.0,
    jitter: bool = True,
) -> JobConfig:
    return JobConfig.model_validate({
        "name": "test-recovery",
        "description": "Unit test config",
        "backend": {
            "type": backend_type,
            "cli_model": cli_model,
            "model": model,
        },
        "sheet": {"size": 3, "total_items": 9},
        "prompt": {"template": "Do something on sheet {sheet_num}"},
        "rate_limit": {
            "wait_minutes": wait_minutes,
            "max_waits": max_waits,
            "max_quota_waits": max_quota_waits,
        },
        "retry": {
            "base_delay_seconds": base_delay_seconds,
            "exponential_base": exponential_base,
            "max_delay_seconds": max_delay_seconds,
            "jitter": jitter,
        },
    })


def _make_state(
    *,
    last_completed_sheet: int = 0,
    rate_limit_waits: int = 0,
    quota_waits: int = 0,
    job_id: str = "test-job-001",
) -> CheckpointState:
    return CheckpointState(
        job_id=job_id,
        job_name="test-recovery",
        total_sheets=5,
        last_completed_sheet=last_completed_sheet,
        rate_limit_waits=rate_limit_waits,
        quota_waits=quota_waits,
    )


class _RecoveryHarness(RecoveryMixin):
    """Thin test harness that provides the attributes RecoveryMixin expects."""

    def __init__(self, config: JobConfig) -> None:
        self.config = config
        self.backend = MagicMock()
        self.state_backend = MagicMock()
        self.console = MagicMock()
        self._logger = MagicMock()
        self._global_learning_store = None
        self._healing_coordinator = None
        self.error_classifier = ErrorClassifier()
        self.rate_limit_callback = None

    async def _interruptible_sleep(self, seconds: float) -> None:
        """No-op sleep for tests."""


# ---------------------------------------------------------------------------
# _resolve_wait_duration
# ---------------------------------------------------------------------------


class TestResolveWaitDuration:

    def test_returns_suggested_when_positive(self) -> None:
        h = _RecoveryHarness(_make_config(wait_minutes=60))
        assert h._resolve_wait_duration(42.5) == 42.5

    def test_falls_back_to_config_when_none(self) -> None:
        h = _RecoveryHarness(_make_config(wait_minutes=5))
        assert h._resolve_wait_duration(None) == 300.0  # 5 * 60

    def test_falls_back_to_config_when_zero(self) -> None:
        h = _RecoveryHarness(_make_config(wait_minutes=10))
        assert h._resolve_wait_duration(0) == 600.0  # 10 * 60

    def test_falls_back_to_config_when_negative(self) -> None:
        h = _RecoveryHarness(_make_config(wait_minutes=2))
        assert h._resolve_wait_duration(-5.0) == 120.0  # 2 * 60

    def test_returns_small_positive_suggested(self) -> None:
        h = _RecoveryHarness(_make_config(wait_minutes=60))
        assert h._resolve_wait_duration(0.001) == 0.001


# ---------------------------------------------------------------------------
# _infer_active_sheet_num
# ---------------------------------------------------------------------------


class TestInferActiveSheetNum:
    """Tests for _infer_active_sheet_num (static method)."""

    def test_returns_last_completed_plus_one(self) -> None:
        state = _make_state(last_completed_sheet=3)
        assert RecoveryMixin._infer_active_sheet_num(state) == 4

    def test_no_sheets_completed_returns_one(self) -> None:
        state = _make_state(last_completed_sheet=0)
        assert RecoveryMixin._infer_active_sheet_num(state) == 1

    def test_all_sheets_completed(self) -> None:
        """When all sheets are done, returns total_sheets + 1 (one past end)."""
        state = _make_state(last_completed_sheet=5)
        assert RecoveryMixin._infer_active_sheet_num(state) == 6


# ---------------------------------------------------------------------------
# _get_retry_delay
# ---------------------------------------------------------------------------


class TestGetRetryDelay:
    """Tests for _get_retry_delay with exponential backoff and jitter."""

    _RANDOM_PATCH = "mozart.execution.runner.recovery.random.random"

    @staticmethod
    def _harness(*, jitter: bool = False, **kwargs: Any) -> _RecoveryHarness:
        return _RecoveryHarness(_make_config(jitter=jitter, **kwargs))

    def test_attempt_1_base_delay(self) -> None:
        """base=5, exp=2, attempt 1: 5 * 2^0 = 5."""
        assert self._harness(base_delay_seconds=5.0)._get_retry_delay(1) == 5.0

    def test_attempt_2_exponential(self) -> None:
        """base=5, exp=2, attempt 2: 5 * 2^1 = 10."""
        assert self._harness(base_delay_seconds=5.0)._get_retry_delay(2) == 10.0

    def test_attempt_3_exponential(self) -> None:
        """base=5, exp=2, attempt 3: 5 * 2^2 = 20."""
        assert self._harness(base_delay_seconds=5.0)._get_retry_delay(3) == 20.0

    def test_capped_at_max_delay(self) -> None:
        """Attempt 10: 5 * 2^9 = 2560, capped to 100."""
        h = self._harness(base_delay_seconds=5.0, max_delay_seconds=100.0)
        assert h._get_retry_delay(10) == 100.0

    def test_jitter_disabled_returns_exact_value(self) -> None:
        assert self._harness(base_delay_seconds=10.0)._get_retry_delay(1) == 10.0

    def test_jitter_enabled_applies_multiplier(self) -> None:
        """Delay multiplied by (0.5 + random()). random()=0.3 => multiplier=0.8."""
        h = self._harness(jitter=True, base_delay_seconds=10.0)
        with patch(self._RANDOM_PATCH, return_value=0.3):
            assert h._get_retry_delay(1) == pytest.approx(8.0)

    def test_jitter_with_random_zero(self) -> None:
        """random()=0.0 => multiplier=0.5 (minimum jitter)."""
        h = self._harness(jitter=True, base_delay_seconds=20.0)
        with patch(self._RANDOM_PATCH, return_value=0.0):
            assert h._get_retry_delay(1) == pytest.approx(10.0)

    def test_jitter_with_random_one(self) -> None:
        """random()~=1.0 => multiplier~=1.5 (maximum jitter)."""
        h = self._harness(jitter=True, base_delay_seconds=20.0)
        with patch(self._RANDOM_PATCH, return_value=0.9999):
            assert h._get_retry_delay(1) == pytest.approx(29.998, rel=1e-3)

    def test_jitter_applied_after_cap(self) -> None:
        """Jitter applies to capped value: cap=50, random()=0.5 => 50*(0.5+0.5)=50."""
        h = self._harness(jitter=True, base_delay_seconds=5.0, max_delay_seconds=50.0)
        with patch(self._RANDOM_PATCH, return_value=0.5):
            assert h._get_retry_delay(20) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# _get_effective_model
# ---------------------------------------------------------------------------


class TestGetEffectiveModel:

    @staticmethod
    def _model_for(**kwargs: Any) -> str | None:
        return _RecoveryHarness(_make_config(**kwargs))._get_effective_model()

    def test_claude_cli_returns_cli_model(self) -> None:
        assert self._model_for(backend_type="claude_cli", cli_model="claude-opus-4-6") == "claude-opus-4-6"

    def test_claude_cli_returns_none_when_cli_model_unset(self) -> None:
        assert self._model_for(backend_type="claude_cli", cli_model=None) is None

    def test_anthropic_api_returns_model(self) -> None:
        assert self._model_for(backend_type="anthropic_api", model="claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"

    def test_ollama_returns_model(self) -> None:
        assert self._model_for(backend_type="ollama", model="llama3.1:8b") == "llama3.1:8b"


# ---------------------------------------------------------------------------
# _classify_error
# ---------------------------------------------------------------------------


class TestClassifyError:
    """_classify_error returns .primary from the full ClassificationResult."""

    def test_returns_primary_from_classification(self) -> None:
        harness = _RecoveryHarness(_make_config())

        primary = ClassifiedError(
            category=ErrorCategory.RATE_LIMIT,
            message="Rate limit hit",
            error_code=ErrorCode.RATE_LIMIT_API,
        )
        mock_result = ClassificationResult(
            primary=primary,
            secondary=[],
            confidence=1.0,
        )

        with patch.object(harness, "_classify_execution", return_value=mock_result):
            result = ExecutionResult(
                success=False, stdout="", stderr="rate limit", duration_seconds=1.0
            )
            classified = harness._classify_error(result)

        assert classified is primary
        assert classified.category == ErrorCategory.RATE_LIMIT
        assert classified.error_code == ErrorCode.RATE_LIMIT_API


# ---------------------------------------------------------------------------
# _log_rate_limit_event
# ---------------------------------------------------------------------------


class TestLogRateLimitEvent:
    """Tests for _log_rate_limit_event -- logging and max-wait enforcement."""

    @staticmethod
    def _harness(**kwargs: Any) -> _RecoveryHarness:
        return _RecoveryHarness(_make_config(**kwargs))

    # --- Rate limit (is_quota=False) ---

    def test_rate_limit_below_max_does_not_raise(self) -> None:
        h = self._harness(max_waits=24)
        state = _make_state(rate_limit_waits=5)
        h._log_rate_limit_event(state, is_quota=False, wait_seconds=60.0)

    def test_rate_limit_at_max_raises_fatal(self) -> None:
        h = self._harness(max_waits=10)
        state = _make_state(rate_limit_waits=10)
        with pytest.raises(FatalError, match="Exceeded maximum rate limit waits"):
            h._log_rate_limit_event(state, is_quota=False, wait_seconds=60.0)

    def test_rate_limit_above_max_raises_fatal(self) -> None:
        h = self._harness(max_waits=5)
        state = _make_state(rate_limit_waits=7)
        with pytest.raises(FatalError, match="Exceeded maximum rate limit waits"):
            h._log_rate_limit_event(state, is_quota=False, wait_seconds=60.0)

    def test_rate_limit_logs_warning(self) -> None:
        h = self._harness(max_waits=24)
        state = _make_state(rate_limit_waits=3)
        h._log_rate_limit_event(state, is_quota=False, wait_seconds=120.0)

        h._logger.warning.assert_called_once()
        call_args = h._logger.warning.call_args
        assert call_args[0][0] == "rate_limit.detected"
        assert call_args[1]["wait_count"] == 3
        assert call_args[1]["max_waits"] == 24

    def test_rate_limit_console_output(self) -> None:
        h = self._harness(max_waits=24)
        state = _make_state(rate_limit_waits=2)
        h._log_rate_limit_event(state, is_quota=False, wait_seconds=300.0)

        h.console.print.assert_called_once()
        msg = h.console.print.call_args[0][0]
        assert "Rate limited" in msg
        assert "2/24" in msg

    # --- Quota exhaustion (is_quota=True) ---

    def test_quota_below_max_does_not_raise(self) -> None:
        h = self._harness(max_quota_waits=48)
        state = _make_state(quota_waits=10)
        h._log_rate_limit_event(state, is_quota=True, wait_seconds=3600.0)

    def test_quota_at_max_raises_fatal(self) -> None:
        h = self._harness(max_quota_waits=5)
        state = _make_state(quota_waits=5)
        with pytest.raises(FatalError, match="Exceeded maximum quota exhaustion waits"):
            h._log_rate_limit_event(state, is_quota=True, wait_seconds=3600.0)

    def test_quota_above_max_raises_fatal(self) -> None:
        h = self._harness(max_quota_waits=3)
        state = _make_state(quota_waits=4)
        with pytest.raises(FatalError, match="Exceeded maximum quota exhaustion waits"):
            h._log_rate_limit_event(state, is_quota=True, wait_seconds=3600.0)

    def test_quota_logs_warning(self) -> None:
        h = self._harness(max_quota_waits=48)
        state = _make_state(quota_waits=7)
        h._log_rate_limit_event(state, is_quota=True, wait_seconds=3600.0)

        h._logger.warning.assert_called_once()
        call_args = h._logger.warning.call_args
        assert call_args[0][0] == "quota_exhausted.detected"
        assert call_args[1]["wait_count"] == 7
        assert call_args[1]["max_quota_waits"] == 48

    def test_quota_console_output(self) -> None:
        h = self._harness(max_quota_waits=48)
        state = _make_state(quota_waits=1)
        h._log_rate_limit_event(state, is_quota=True, wait_seconds=1800.0)

        h.console.print.assert_called_once()
        msg = h.console.print.call_args[0][0]
        assert "Token quota exhausted" in msg
        assert "1/48" in msg

    # --- Edge cases ---

    def test_rate_limit_one_wait_allowed_at_zero(self) -> None:
        """With max_waits=1, first wait (count=0) does not raise."""
        h = self._harness(max_waits=1)
        state = _make_state(rate_limit_waits=0)
        h._log_rate_limit_event(state, is_quota=False, wait_seconds=60.0)

    def test_rate_limit_one_wait_raises_at_one(self) -> None:
        """With max_waits=1, second wait (count=1) raises."""
        h = self._harness(max_waits=1)
        state = _make_state(rate_limit_waits=1)
        with pytest.raises(FatalError):
            h._log_rate_limit_event(state, is_quota=False, wait_seconds=60.0)

    def test_fatal_error_includes_max_in_message(self) -> None:
        h = self._harness(max_waits=7)
        state = _make_state(rate_limit_waits=7)
        with pytest.raises(FatalError, match=r"\(7\)"):
            h._log_rate_limit_event(state, is_quota=False, wait_seconds=60.0)

    def test_quota_fatal_error_includes_max_in_message(self) -> None:
        h = self._harness(max_quota_waits=12)
        state = _make_state(quota_waits=12)
        with pytest.raises(FatalError, match=r"\(12\)"):
            h._log_rate_limit_event(state, is_quota=True, wait_seconds=3600.0)

    def test_wait_minutes_calculation(self) -> None:
        """Logged wait_minutes equals wait_seconds / 60."""
        h = self._harness(max_waits=100)
        state = _make_state(rate_limit_waits=1)
        h._log_rate_limit_event(state, is_quota=False, wait_seconds=180.0)

        call_kwargs = h._logger.warning.call_args[1]
        assert call_kwargs["wait_minutes"] == pytest.approx(3.0)
