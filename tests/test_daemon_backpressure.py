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
