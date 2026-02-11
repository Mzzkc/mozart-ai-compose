"""Cross-job rate limit coordination.

When any job hits a rate limit, ALL jobs using that backend are notified
to back off.  Much faster than the SQLite cross-process approach in
``mozart.learning.store.rate_limits`` since everything is in-process.

Satisfies the ``RateLimitChecker`` protocol defined in ``scheduler.py``
so the ``GlobalSheetScheduler`` can query limits before dispatching.

Lock ordering (daemon-wide):
  1. GlobalSheetScheduler._lock
  2. RateLimitCoordinator._lock   ← this module
  3. BackpressureController  (lock-free — reads are atomic)
  4. CentralLearningStore._lock    (future — Stage 5)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from mozart.core.logging import get_logger

_logger = get_logger("daemon.rate_coordinator")


@dataclass
class RateLimitEvent:
    """A single rate limit event reported by a job."""

    backend_type: str
    detected_at: float
    suggested_wait_seconds: float
    job_id: str
    sheet_num: int


class RateLimitCoordinator:
    """In-memory rate limit coordination across all daemon jobs.

    When any job hits a rate limit, ALL jobs using that backend
    are notified to back off.  The coordinator tracks per-backend
    resume times and exposes both sync and async query methods.

    The async ``is_rate_limited`` method satisfies the
    ``RateLimitChecker`` protocol so the scheduler can consult
    limits before dispatching sheets.

    The ``wait_if_limited`` method blocks the caller until the
    backend is clear — useful for runners that want to wait
    inline rather than re-queue.
    """

    def __init__(self) -> None:
        self._events: list[RateLimitEvent] = []
        self._active_limits: dict[str, float] = {}  # backend_type → resume_at (monotonic)
        self._lock = asyncio.Lock()

    # ─── Reporting ─────────────────────────────────────────────────

    async def report_rate_limit(
        self,
        backend_type: str,
        wait_seconds: float,
        job_id: str,
        sheet_num: int,
    ) -> None:
        """Report a rate limit hit — all jobs using this backend back off.

        If a limit is already active for the backend, the resume time
        is extended to whichever is later (existing or newly reported).

        Args:
            backend_type: Backend that was rate-limited (e.g. ``"claude_cli"``).
            wait_seconds: Suggested wait duration in seconds.
            job_id: Job that encountered the limit.
            sheet_num: Sheet that triggered the limit.
        """
        now = time.monotonic()
        async with self._lock:
            resume_at = now + wait_seconds
            self._active_limits[backend_type] = max(
                self._active_limits.get(backend_type, 0.0),
                resume_at,
            )
            self._events.append(RateLimitEvent(
                backend_type=backend_type,
                detected_at=now,
                suggested_wait_seconds=wait_seconds,
                job_id=job_id,
                sheet_num=sheet_num,
            ))
            # Prune events older than 1 hour
            cutoff = now - 3600
            self._events = [e for e in self._events if e.detected_at > cutoff]

        _logger.warning(
            "rate_limit.reported",
            backend=backend_type,
            wait_seconds=wait_seconds,
            job_id=job_id,
            sheet_num=sheet_num,
        )

    # ─── Querying (satisfies RateLimitChecker protocol) ────────────

    async def is_rate_limited(
        self,
        backend_type: str,
        model: str | None = None,
    ) -> tuple[bool, float]:
        """Check if a backend is currently rate-limited.

        Satisfies the ``RateLimitChecker`` protocol used by
        ``GlobalSheetScheduler.next_sheet()``.

        The ``model`` parameter is accepted for protocol compatibility
        but currently unused — limits are tracked per backend type.

        Args:
            backend_type: Backend to check.
            model: Unused; accepted for protocol compatibility.

        Returns:
            ``(is_limited, seconds_remaining)``.  When not limited,
            ``seconds_remaining`` is ``0.0``.
        """
        del model  # accepted for RateLimitChecker protocol compatibility
        async with self._lock:
            resume_at = self._active_limits.get(backend_type, 0.0)
        remaining = resume_at - time.monotonic()
        if remaining > 0:
            return True, remaining
        return False, 0.0

    # ─── Waiting ───────────────────────────────────────────────────

    async def wait_if_limited(self, backend_type: str) -> float:
        """Block until the backend is no longer rate-limited.

        Returns the number of seconds actually waited (``0.0`` if the
        backend was not limited).
        """
        async with self._lock:
            resume_at = self._active_limits.get(backend_type, 0.0)

        wait_time = resume_at - time.monotonic()
        if wait_time > 0:
            _logger.info(
                "rate_limit.waiting",
                backend=backend_type,
                seconds=round(wait_time, 1),
            )
            await asyncio.sleep(wait_time)
            return wait_time
        return 0.0

    # ─── Sync helpers ──────────────────────────────────────────────

    def is_limited_sync(self, backend_type: str) -> bool:
        """Non-async check — usable outside the event loop."""
        return self._active_limits.get(backend_type, 0.0) > time.monotonic()

    @property
    def active_limits(self) -> dict[str, float]:
        """Currently active limits as ``{backend_type: seconds_remaining}``."""
        now = time.monotonic()
        return {
            backend: round(resume_at - now, 1)
            for backend, resume_at in self._active_limits.items()
            if resume_at > now
        }

    @property
    def recent_events(self) -> list[RateLimitEvent]:
        """Events from the last hour (most recent first)."""
        now = time.monotonic()
        cutoff = now - 3600
        return sorted(
            (e for e in self._events if e.detected_at > cutoff),
            key=lambda e: e.detected_at,
            reverse=True,
        )


__all__ = ["RateLimitCoordinator", "RateLimitEvent"]
