"""Tests for mozart.daemon.backpressure module.

Covers BackpressureController: pressure level assessment, can_start_sheet()
protocol, should_accept_job() gating, gate() convenience method, and
integration with ResourceMonitor and RateLimitCoordinator.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from mozart.daemon.backpressure import (
    BackpressureController,
    PressureLevel,
    _LEVEL_DELAYS,
)
from mozart.daemon.config import ResourceLimitConfig
from mozart.daemon.monitor import ResourceMonitor
from mozart.daemon.rate_coordinator import RateLimitCoordinator


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def resource_config() -> ResourceLimitConfig:
    """Resource config with 1000MB limit for easy percentage math."""
    return ResourceLimitConfig(
        max_memory_mb=1000,
        max_processes=20,
        max_api_calls_per_minute=10,
    )


@pytest.fixture
def monitor(resource_config: ResourceLimitConfig) -> ResourceMonitor:
    """ResourceMonitor with no manager (standalone)."""
    return ResourceMonitor(resource_config)


@pytest.fixture
def coordinator() -> RateLimitCoordinator:
    """Fresh rate limit coordinator."""
    return RateLimitCoordinator()


@pytest.fixture
def controller(
    monitor: ResourceMonitor,
    coordinator: RateLimitCoordinator,
) -> BackpressureController:
    """BackpressureController wired to monitor and coordinator."""
    return BackpressureController(monitor, coordinator)


# ─── PressureLevel Enum ──────────────────────────────────────────────


class TestPressureLevel:
    """Tests for PressureLevel enum."""

    def test_all_levels_have_delays(self):
        """Every PressureLevel has a corresponding delay entry."""
        for level in PressureLevel:
            assert level in _LEVEL_DELAYS

    def test_delays_increase_with_severity(self):
        """Higher pressure levels have longer delays."""
        ordered = [
            PressureLevel.NONE,
            PressureLevel.LOW,
            PressureLevel.MEDIUM,
            PressureLevel.HIGH,
            PressureLevel.CRITICAL,
        ]
        for i in range(len(ordered) - 1):
            assert _LEVEL_DELAYS[ordered[i]] <= _LEVEL_DELAYS[ordered[i + 1]]

    def test_none_has_zero_delay(self):
        assert _LEVEL_DELAYS[PressureLevel.NONE] == 0.0

    def test_values_are_strings(self):
        """Enum values are lowercase strings for structured logging."""
        for level in PressureLevel:
            assert isinstance(level.value, str)
            assert level.value == level.value.lower()


# ─── current_level() ─────────────────────────────────────────────────


class TestCurrentLevel:
    """Tests for BackpressureController.current_level()."""

    def test_low_memory_returns_none(self, controller: BackpressureController):
        """Below 50% memory → NONE."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=400.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.NONE

    def test_50_percent_memory_returns_low(self, controller: BackpressureController):
        """Between 50-70% memory → LOW."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=550.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.LOW

    def test_70_percent_memory_returns_medium(self, controller: BackpressureController):
        """Between 70-85% memory → MEDIUM."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=750.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.MEDIUM

    def test_85_percent_memory_returns_high(self, controller: BackpressureController):
        """Between 85-95% memory → HIGH."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=900.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.HIGH

    def test_95_percent_memory_returns_critical(self, controller: BackpressureController):
        """>95% memory → CRITICAL."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=960.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.CRITICAL

    def test_not_accepting_work_returns_critical(
        self, controller: BackpressureController,
    ):
        """Monitor not accepting work → CRITICAL regardless of memory."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=100.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=False,
        ):
            assert controller.current_level() == PressureLevel.CRITICAL

    def test_active_rate_limits_bump_to_high(
        self, controller: BackpressureController,
        coordinator: RateLimitCoordinator,
    ):
        """Active rate limits → at least HIGH even with low memory."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=100.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            # Simulate an active rate limit
            import time
            coordinator._active_limits["claude_cli"] = time.monotonic() + 60
            assert controller.current_level() == PressureLevel.HIGH


# ─── can_start_sheet() — BackpressureChecker protocol ────────────────


class TestCanStartSheet:
    """Tests for BackpressureChecker protocol conformance."""

    @pytest.mark.asyncio
    async def test_none_pressure_allows_immediately(
        self, controller: BackpressureController,
    ):
        """At NONE pressure, sheet is allowed with zero delay."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=100.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            allowed, delay = await controller.can_start_sheet()
            assert allowed is True
            assert delay == 0.0

    @pytest.mark.asyncio
    async def test_low_pressure_allows_with_delay(
        self, controller: BackpressureController,
    ):
        """At LOW pressure, sheet is allowed with a small delay."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=550.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            allowed, delay = await controller.can_start_sheet()
            assert allowed is True
            assert delay == _LEVEL_DELAYS[PressureLevel.LOW]

    @pytest.mark.asyncio
    async def test_medium_pressure_allows_with_larger_delay(
        self, controller: BackpressureController,
    ):
        """At MEDIUM pressure, sheet is allowed but with a significant delay."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=750.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            allowed, delay = await controller.can_start_sheet()
            assert allowed is True
            assert delay == _LEVEL_DELAYS[PressureLevel.MEDIUM]

    @pytest.mark.asyncio
    async def test_high_pressure_allows_with_long_delay(
        self, controller: BackpressureController,
    ):
        """At HIGH pressure, sheet is still allowed but with a long delay."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=900.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            allowed, delay = await controller.can_start_sheet()
            assert allowed is True
            assert delay == _LEVEL_DELAYS[PressureLevel.HIGH]

    @pytest.mark.asyncio
    async def test_critical_pressure_rejects(
        self, controller: BackpressureController,
    ):
        """At CRITICAL pressure, sheet is rejected."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=960.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            allowed, delay = await controller.can_start_sheet()
            assert allowed is False
            assert delay == _LEVEL_DELAYS[PressureLevel.CRITICAL]

    @pytest.mark.asyncio
    async def test_returns_tuple_of_bool_float(
        self, controller: BackpressureController,
    ):
        """Return type matches BackpressureChecker protocol."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=100.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            result = await controller.can_start_sheet()
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bool)
            assert isinstance(result[1], float)


# ─── should_accept_job() ─────────────────────────────────────────────


class TestShouldAcceptJob:
    """Tests for job-level gating."""

    def test_accepts_at_none(self, controller: BackpressureController):
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=100.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.should_accept_job() is True

    def test_accepts_at_low(self, controller: BackpressureController):
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=550.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.should_accept_job() is True

    def test_accepts_at_medium(self, controller: BackpressureController):
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=750.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.should_accept_job() is True

    def test_rejects_at_high(self, controller: BackpressureController):
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=900.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.should_accept_job() is False

    def test_rejects_at_critical(self, controller: BackpressureController):
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=960.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.should_accept_job() is False


# ─── Scheduler integration ───────────────────────────────────────────


class TestSchedulerIntegration:
    """Tests verifying BackpressureController satisfies BackpressureChecker."""

    @pytest.mark.asyncio
    async def test_protocol_compatibility(
        self, controller: BackpressureController,
    ):
        """BackpressureController can be used where BackpressureChecker is expected."""
        from mozart.daemon.scheduler import BackpressureChecker

        # Protocol structural check — must have can_start_sheet
        assert hasattr(controller, "can_start_sheet")
        assert asyncio.iscoroutinefunction(controller.can_start_sheet)

    @pytest.mark.asyncio
    async def test_wires_into_scheduler(self):
        """Controller can be set on the scheduler via set_backpressure()."""
        from mozart.daemon.config import DaemonConfig
        from mozart.daemon.scheduler import GlobalSheetScheduler

        config = DaemonConfig(max_concurrent_sheets=5, max_concurrent_jobs=3)
        scheduler = GlobalSheetScheduler(config)

        resource_config = ResourceLimitConfig(
            max_memory_mb=1000, max_processes=20,
        )
        monitor = ResourceMonitor(resource_config)
        coordinator = RateLimitCoordinator()
        bp = BackpressureController(monitor, coordinator)

        # This should not raise — type is compatible
        scheduler.set_backpressure(bp)
        assert scheduler._backpressure is bp


# ─── Probe Failure → CRITICAL (D006) ──────────────────────────────


class TestProbeFailureBackpressure:
    """Tests that probe failure causes CRITICAL backpressure (fail-closed)."""

    def test_probe_failure_returns_critical(self, controller: BackpressureController):
        """When memory probe returns None, pressure is CRITICAL."""
        with patch.object(
            ResourceMonitor, "current_memory_mb", return_value=None,
        ):
            assert controller.current_level() == PressureLevel.CRITICAL

    def test_degraded_monitor_returns_critical(
        self, controller: BackpressureController, monitor: ResourceMonitor,
    ):
        """When monitor is degraded, pressure is CRITICAL."""
        monitor._degraded = True
        with patch.object(
            ResourceMonitor, "current_memory_mb", return_value=100.0,
        ):
            assert controller.current_level() == PressureLevel.CRITICAL
        monitor._degraded = False  # cleanup

    @pytest.mark.asyncio
    async def test_probe_failure_rejects_sheets(
        self, controller: BackpressureController,
    ):
        """Probe failure causes can_start_sheet to reject."""
        with patch.object(
            ResourceMonitor, "current_memory_mb", return_value=None,
        ):
            allowed, delay = await controller.can_start_sheet()
            assert allowed is False

    def test_probe_failure_rejects_jobs(
        self, controller: BackpressureController,
    ):
        """Probe failure causes should_accept_job to reject."""
        with patch.object(
            ResourceMonitor, "current_memory_mb", return_value=None,
        ):
            assert controller.should_accept_job() is False


# ─── P013: Exact Boundary Value Tests ─────────────────────────────────


class TestBoundaryValues:
    """Exact boundary tests for current_level() thresholds.

    With max_memory_mb=1000:
      - 50% = 500MB: boundary between NONE and LOW (>50% → LOW)
      - 70% = 700MB: boundary between LOW and MEDIUM (>70% → MEDIUM)
      - 85% = 850MB: boundary between MEDIUM and HIGH (>85% → HIGH)
      - 95% = 950MB: boundary between HIGH and CRITICAL (>95% → CRITICAL)

    All thresholds use strict `>`, so exact boundary values belong
    to the LOWER level.
    """

    def test_exactly_50_pct_is_none(self, controller: BackpressureController):
        """500MB is exactly 50% — should be NONE (threshold is >50%)."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=500.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.NONE

    def test_just_above_50_pct_is_low(self, controller: BackpressureController):
        """501MB is >50% — should be LOW."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=501.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.LOW

    def test_exactly_70_pct_is_low(self, controller: BackpressureController):
        """700MB is exactly 70% — should be LOW (threshold is >70%)."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=700.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.LOW

    def test_just_above_70_pct_is_medium(self, controller: BackpressureController):
        """701MB is >70% — should be MEDIUM."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=701.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.MEDIUM

    def test_exactly_85_pct_is_medium(self, controller: BackpressureController):
        """850MB is exactly 85% — should be MEDIUM (threshold is >85%)."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=850.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.MEDIUM

    def test_just_above_85_pct_is_high(self, controller: BackpressureController):
        """851MB is >85% — should be HIGH."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=851.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.HIGH

    def test_exactly_95_pct_is_high(self, controller: BackpressureController):
        """950MB is exactly 95% — should be HIGH (threshold is >95%)."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=950.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.HIGH

    def test_just_above_95_pct_is_critical(self, controller: BackpressureController):
        """951MB is >95% — should be CRITICAL."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=951.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.CRITICAL

    def test_zero_memory_is_none(self, controller: BackpressureController):
        """0MB usage — should be NONE."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=0.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.NONE

    def test_exactly_at_limit_is_critical(self, controller: BackpressureController):
        """1000MB (100%) — should be CRITICAL."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=1000.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.current_level() == PressureLevel.CRITICAL


# ─── P010: Rate Coordinator via Public API ─────────────────────────────


class TestRateCoordinatorPublicAPI:
    """Tests that backpressure reads rate limits through the public API.

    Validates P010: previous tests injected rate limits via private
    ``_active_limits`` dict.  These tests use ``report_rate_limit()``
    (the public write path) to prove the full
    report → active_limits → current_level() chain.
    """

    @pytest.mark.asyncio
    async def test_reported_limit_escalates_pressure(
        self, controller: BackpressureController,
        coordinator: RateLimitCoordinator,
    ):
        """A rate limit reported via public API escalates pressure to HIGH."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=100.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            # Before: no limits, low memory → NONE
            assert controller.current_level() == PressureLevel.NONE

            # Report via public API
            await coordinator.report_rate_limit(
                backend_type="claude_cli",
                wait_seconds=60.0,
                job_id="job-a",
                sheet_num=1,
            )

            # After: active limit → HIGH
            assert controller.current_level() == PressureLevel.HIGH

    @pytest.mark.asyncio
    async def test_public_api_limit_blocks_job_submission(
        self, controller: BackpressureController,
        coordinator: RateLimitCoordinator,
    ):
        """Rate limit via public API causes should_accept_job() to return False."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=100.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            assert controller.should_accept_job() is True

            await coordinator.report_rate_limit(
                backend_type="openai",
                wait_seconds=30.0,
                job_id="job-b",
                sheet_num=2,
            )

            # HIGH pressure → reject new jobs
            assert controller.should_accept_job() is False


# ─── P013: Rate Limit Expiry Transitions ───────────────────────────────


class TestRateLimitExpiryTransitions:
    """Tests that pressure level transitions back down after rate limits expire.

    Validates P013: once a rate limit expires, active_limits becomes
    empty, and current_level() should return to the level determined
    solely by memory pressure (not remain stuck at HIGH).
    """

    @pytest.mark.asyncio
    async def test_pressure_drops_after_limit_expires(
        self, controller: BackpressureController,
        coordinator: RateLimitCoordinator,
    ):
        """After rate limit expires, pressure returns to NONE (low memory)."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=100.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            # Report a very short limit
            await coordinator.report_rate_limit(
                backend_type="claude_cli",
                wait_seconds=0.02,
                job_id="job-a",
                sheet_num=1,
            )

            # Immediately: HIGH
            assert controller.current_level() == PressureLevel.HIGH

            # Wait for expiry
            await asyncio.sleep(0.03)

            # After expiry: back to NONE
            assert controller.current_level() == PressureLevel.NONE

    @pytest.mark.asyncio
    async def test_job_accepted_after_limit_expires(
        self, controller: BackpressureController,
        coordinator: RateLimitCoordinator,
    ):
        """Job submission resumes after rate limit expires."""
        with patch.object(
            ResourceMonitor, "_get_memory_usage_mb", return_value=100.0,
        ), patch.object(
            ResourceMonitor, "is_accepting_work", return_value=True,
        ):
            await coordinator.report_rate_limit(
                backend_type="claude_cli",
                wait_seconds=0.02,
                job_id="job-a",
                sheet_num=1,
            )

            assert controller.should_accept_job() is False

            await asyncio.sleep(0.03)

            assert controller.should_accept_job() is True
