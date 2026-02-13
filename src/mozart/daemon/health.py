"""Health check infrastructure for daemon readiness and liveness probes.

Provides ``HealthChecker`` with two probe types:

- **Liveness**: Is the daemon process alive and responsive?  Always returns
  OK if the daemon can execute the handler at all.
- **Readiness**: Is the daemon ready to accept new jobs?  Checks resource
  limits via ``ResourceMonitor`` to implement backpressure signaling.

These are registered as JSON-RPC methods (``daemon.health``, ``daemon.ready``)
so clients (CLI, orchestrators) can query them over the Unix socket.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from mozart.core.logging import get_logger

if TYPE_CHECKING:
    from mozart.daemon.manager import JobManager
    from mozart.daemon.monitor import ResourceMonitor

_logger = get_logger("daemon.health")


class HealthChecker:
    """Daemon health and readiness probes.

    Parameters
    ----------
    manager:
        The ``JobManager`` instance for job count queries.
    monitor:
        The ``ResourceMonitor`` instance for resource threshold checks.
    """

    def __init__(
        self,
        manager: JobManager,
        monitor: ResourceMonitor,
        *,
        start_time: float | None = None,
    ) -> None:
        self._manager = manager
        self._monitor = monitor
        self._start_time = start_time or time.monotonic()

    async def liveness(self) -> dict[str, Any]:
        """Is the daemon process alive and responsive?

        This is the cheapest possible check — if the daemon can execute
        this handler and return a response, it's alive.  No resource
        checks or I/O are performed.
        """
        return {
            "status": "ok",
            "pid": os.getpid(),
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "shutting_down": self._manager.shutting_down,
        }

    async def readiness(self) -> dict[str, Any]:
        """Is the daemon ready to accept new jobs?

        Checks resource thresholds via the monitor and job failure rate
        via the manager.  Returns ``"ready"`` when resources are within
        limits and the failure rate is not elevated, ``"not_ready"``
        otherwise.
        """
        snapshot = await self._monitor.check_now()
        accepting = self._monitor.is_accepting_work()
        failure_elevated = self._manager.failure_rate_elevated

        shutting_down = self._manager.shutting_down
        is_ready = accepting and not shutting_down and not failure_elevated
        return {
            "status": "ready" if is_ready else "not_ready",
            "running_jobs": self._manager.running_count,
            "memory_mb": round(snapshot.memory_usage_mb, 1),
            "child_processes": snapshot.child_process_count,
            "accepting_work": is_ready,
            "shutting_down": shutting_down,
            "failure_rate_elevated": failure_elevated,
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
        }


__all__ = ["HealthChecker"]
