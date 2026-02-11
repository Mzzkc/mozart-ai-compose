"""Resource monitoring for the Mozart daemon.

Periodic background checks on memory, child process count, and zombie
detection.  Emits structured log warnings when limits are approached and
triggers backpressure (e.g. job cancellation) when hard limits are exceeded.

Uses psutil when available; falls back to /proc on Linux.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mozart.core.logging import get_logger
from mozart.daemon.config import ResourceLimitConfig

if TYPE_CHECKING:
    from mozart.daemon.manager import JobManager
    from mozart.daemon.pgroup import ProcessGroupManager

_logger = get_logger("daemon.monitor")


@dataclass
class ResourceSnapshot:
    """Point-in-time resource usage reading."""

    timestamp: float
    memory_usage_mb: float
    child_process_count: int
    running_jobs: int
    active_sheets: int
    zombie_pids: list[int] = field(default_factory=list)

    @property
    def memory_percent(self) -> float:
        """Memory usage as percentage of limit (requires limit from caller)."""
        return 0.0  # Computed externally via _compute_percent

    @property
    def processes_percent(self) -> float:
        """Process count as percentage of limit (requires limit from caller)."""
        return 0.0  # Computed externally via _compute_percent


def _compute_percent(current: float, limit: float) -> float:
    """Compute percentage of limit, clamped to [0, 100]."""
    if limit <= 0:
        return 0.0
    return min((current / limit) * 100.0, 100.0)


class ResourceMonitor:
    """Periodic resource monitoring for the daemon.

    Checks memory usage, child process count, and zombie processes
    on a configurable interval.  Emits structured log warnings when
    approaching limits and can trigger hard actions (job cancellation)
    when hard limits are exceeded.
    """

    # Threshold percentages
    WARN_THRESHOLD = 80.0
    HARD_THRESHOLD = 95.0

    def __init__(
        self,
        config: ResourceLimitConfig,
        manager: JobManager | None = None,
        pgroup: ProcessGroupManager | None = None,
    ) -> None:
        self._config = config
        self._manager = manager
        self._pgroup = pgroup
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self, interval_seconds: float = 15.0) -> None:
        """Start the periodic monitoring loop."""
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(interval_seconds))
        _logger.info("monitor.started", interval=interval_seconds)

    async def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        _logger.info("monitor.stopped")

    async def check_now(self) -> ResourceSnapshot:
        """Run an immediate resource check and return snapshot."""
        running_jobs = 0
        active_sheets = 0
        if self._manager is not None:
            running_jobs = self._manager.running_count
            active_sheets = self._manager.active_sheet_count

        snapshot = ResourceSnapshot(
            timestamp=time.monotonic(),
            memory_usage_mb=self._get_memory_usage_mb(),
            child_process_count=self._get_child_process_count(),
            running_jobs=running_jobs,
            active_sheets=active_sheets,
            zombie_pids=self._check_for_zombies(),
        )
        return snapshot

    # ─── Internal ─────────────────────────────────────────────────────

    async def _loop(self, interval: float) -> None:
        """Periodic monitoring loop."""
        while self._running:
            try:
                snapshot = await self.check_now()
                await self._evaluate(snapshot)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception:
                _logger.exception("monitor.check_failed")
                await asyncio.sleep(interval)

    async def _evaluate(self, snapshot: ResourceSnapshot) -> None:
        """Log warnings and enforce hard limits based on snapshot."""
        mem_pct = _compute_percent(
            snapshot.memory_usage_mb, self._config.max_memory_mb,
        )
        proc_pct = _compute_percent(
            snapshot.child_process_count, self._config.max_processes,
        )

        # Memory warnings / enforcement
        if mem_pct >= self.HARD_THRESHOLD:
            _logger.error(
                "monitor.memory_critical",
                usage_mb=snapshot.memory_usage_mb,
                limit_mb=self._config.max_memory_mb,
                percent=round(mem_pct, 1),
            )
            await self._enforce_memory_limit()
        elif mem_pct >= self.WARN_THRESHOLD:
            _logger.warning(
                "monitor.memory_warning",
                usage_mb=snapshot.memory_usage_mb,
                limit_mb=self._config.max_memory_mb,
                percent=round(mem_pct, 1),
            )

        # Process count warnings / enforcement
        if proc_pct >= self.HARD_THRESHOLD:
            _logger.error(
                "monitor.processes_critical",
                count=snapshot.child_process_count,
                limit=self._config.max_processes,
                percent=round(proc_pct, 1),
            )
        elif proc_pct >= self.WARN_THRESHOLD:
            _logger.warning(
                "monitor.processes_warning",
                count=snapshot.child_process_count,
                limit=self._config.max_processes,
                percent=round(proc_pct, 1),
            )

        # Zombie reaping
        if snapshot.zombie_pids:
            _logger.warning(
                "monitor.zombies_detected",
                zombie_pids=snapshot.zombie_pids,
                count=len(snapshot.zombie_pids),
            )

        # Periodic orphan cleanup via process group manager
        if self._pgroup is not None:
            self._pgroup.cleanup_orphans()

    async def _enforce_memory_limit(self) -> None:
        """Cancel the oldest running job when memory exceeds hard limit."""
        if self._manager is None:
            return
        # Find oldest running job and cancel it
        jobs = await self._manager.list_jobs()
        running = [j for j in jobs if j.get("status") == "running"]
        if running:
            oldest = min(running, key=lambda j: j.get("submitted_at", 0))
            job_id = oldest.get("job_id")
            if job_id:
                _logger.warning("monitor.cancelling_oldest_job", job_id=job_id)
                await self._manager.cancel_job(job_id)

    # ─── System Probes ────────────────────────────────────────────────

    @staticmethod
    def _get_memory_usage_mb() -> float:
        """Get current RSS memory in MB.

        Uses psutil if available, falls back to /proc/self/status.
        """
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except (ImportError, Exception):
            pass
        # Fallback: /proc/self/status (Linux only)
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # kB -> MB
        except (OSError, ValueError):
            pass
        return 0.0

    @staticmethod
    def _get_child_process_count() -> int:
        """Count child processes in our process group.

        Uses psutil if available, falls back to /proc iteration.
        """
        try:
            import psutil

            current = psutil.Process()
            return len(current.children(recursive=True))
        except (ImportError, Exception):
            pass
        # Fallback: /proc iteration (Linux only)
        my_pid = os.getpid()
        count = 0
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry}/status") as f:
                        for line in f:
                            if line.startswith("PPid:"):
                                ppid = int(line.split()[1])
                                if ppid == my_pid:
                                    count += 1
                                break
                except (OSError, ValueError):
                    continue
        except OSError:
            pass
        return count

    @staticmethod
    def _check_for_zombies() -> list[int]:
        """Detect and reap zombie child processes."""
        zombies: list[int] = []
        try:
            import psutil

            current = psutil.Process()
            for child in current.children(recursive=True):
                try:
                    if child.status() == psutil.STATUS_ZOMBIE:
                        zombies.append(child.pid)
                        os.waitpid(child.pid, os.WNOHANG)
                except (psutil.NoSuchProcess, ChildProcessError):
                    continue
        except ImportError:
            # Fallback: try waitpid for any child
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                    if pid == 0:
                        break
                    if os.WIFSIGNALED(status) or os.WIFEXITED(status):
                        zombies.append(pid)
                except ChildProcessError:
                    break
        except Exception:
            pass
        return zombies


__all__ = ["ResourceMonitor", "ResourceSnapshot"]
