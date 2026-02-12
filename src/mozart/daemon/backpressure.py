"""Backpressure — adaptive load management.

Monitors system resource pressure (memory, rate limits) and throttles
sheet dispatch to prevent overload.  Implements the ``BackpressureChecker``
protocol defined in ``scheduler.py`` so the ``GlobalSheetScheduler`` can
consult backpressure before dispatching each sheet.

Also exposes ``should_accept_job()`` for the ``JobManager`` to reject
new submissions when the system is critically stressed.

Lock ordering (daemon-wide):
  1. GlobalSheetScheduler._lock
  2. RateLimitCoordinator._lock
  3. BackpressureController  (no internal lock — reads are atomic)
  4. CentralLearningStore._lock  (future — Stage 5)
"""

from __future__ import annotations

import asyncio
from enum import Enum

from mozart.core.logging import get_logger
from mozart.daemon.monitor import ResourceMonitor
from mozart.daemon.rate_coordinator import RateLimitCoordinator

_logger = get_logger("daemon.backpressure")


class PressureLevel(Enum):
    """Graduated pressure levels for adaptive load management."""

    NONE = "none"          # All systems go
    LOW = "low"            # Slight delay between sheets
    MEDIUM = "medium"      # Significant delay, warn on new jobs
    HIGH = "high"          # Reject new jobs, wait for relief
    CRITICAL = "critical"  # Emergency: cancel lowest-priority jobs


# Delay in seconds per pressure level, applied before each sheet dispatch
_LEVEL_DELAYS: dict[PressureLevel, float] = {
    PressureLevel.NONE: 0.0,
    PressureLevel.LOW: 2.0,
    PressureLevel.MEDIUM: 10.0,
    PressureLevel.HIGH: 30.0,
    PressureLevel.CRITICAL: 60.0,
}


class BackpressureController:
    """Manages system load through adaptive backpressure.

    Assesses memory usage (as a fraction of the configured limit) and
    active rate limits to determine a ``PressureLevel``.  The scheduler
    calls ``can_start_sheet()`` before dispatching each sheet and gets
    back a (allowed, delay) tuple.

    No internal lock is needed because:
    - ``ResourceMonitor`` methods are thread/coroutine-safe.
    - ``RateLimitCoordinator.active_limits`` is a property that
      reads a dict snapshot.
    - ``PressureLevel`` assessment is a pure function of those reads.
    """

    def __init__(
        self,
        monitor: ResourceMonitor,
        rate_coordinator: RateLimitCoordinator,
    ) -> None:
        self._monitor = monitor
        self._rate_coordinator = rate_coordinator

    def current_level(self) -> PressureLevel:
        """Assess current pressure level from resource metrics.

        Thresholds (memory as % of ``ResourceLimitConfig.max_memory_mb``):
          - probe failure or monitor degraded  → CRITICAL (fail-closed)
          - >95% or monitor not accepting work → CRITICAL
          - >85% or any active rate limit      → HIGH
          - >70%                               → MEDIUM
          - >50%                               → LOW
          - otherwise                          → NONE
        """
        current_mem = self._monitor.current_memory_mb()
        if current_mem is None or self._monitor.is_degraded:
            return PressureLevel.CRITICAL

        max_mem = max(self._monitor.max_memory_mb, 1)
        memory_pct = current_mem / max_mem

        if memory_pct > 0.95 or not self._monitor.is_accepting_work():
            return PressureLevel.CRITICAL
        if memory_pct > 0.85 or self._rate_coordinator.active_limits:
            return PressureLevel.HIGH
        if memory_pct > 0.70:
            return PressureLevel.MEDIUM
        if memory_pct > 0.50:
            return PressureLevel.LOW
        return PressureLevel.NONE

    # ─── BackpressureChecker protocol ──────────────────────────────

    async def can_start_sheet(self) -> tuple[bool, float]:
        """Whether the scheduler may dispatch a sheet, and any delay.

        Satisfies the ``BackpressureChecker`` protocol used by
        ``GlobalSheetScheduler.next_sheet()``.

        Returns:
            ``(allowed, delay_seconds)``.  At CRITICAL level the sheet
            is rejected (``allowed=False``).  At lower levels a positive
            delay is returned to slow dispatch.
        """
        level = self.current_level()
        delay = _LEVEL_DELAYS[level]

        if level == PressureLevel.CRITICAL:
            _logger.warning(
                "backpressure.sheet_rejected",
                level=level.value,
            )
            return False, delay

        if delay > 0:
            _logger.info(
                "backpressure.sheet_delayed",
                level=level.value,
                delay_seconds=delay,
            )

        return True, delay

    # ─── Job-level gating ──────────────────────────────────────────

    def should_accept_job(self) -> bool:
        """Whether to accept new job submissions.

        Returns ``False`` at HIGH or CRITICAL pressure to prevent
        further resource consumption.
        """
        level = self.current_level()
        if level in (PressureLevel.HIGH, PressureLevel.CRITICAL):
            _logger.info(
                "backpressure.job_rejected",
                level=level.value,
            )
            return False
        return True

    # ─── Sheet-level gate (convenience) ────────────────────────────

    async def gate(self) -> None:
        """Adaptive delay based on current pressure.

        Convenience method — call before starting each sheet when
        not using the scheduler's built-in backpressure integration.
        """
        level = self.current_level()
        delay = _LEVEL_DELAYS[level]
        if delay > 0:
            _logger.info(
                "backpressure.delay",
                level=level.value,
                delay_seconds=delay,
            )
            await asyncio.sleep(delay)


__all__ = ["BackpressureController", "PressureLevel"]
