"""Job manager for the Mozart daemon.

Maps job IDs to asyncio.Tasks, enforces concurrency limits via semaphore,
routes IPC requests to JobService, and cancels all tasks on shutdown.
"""

from __future__ import annotations

import asyncio
import os
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections.abc import Coroutine
from typing import Any

import mozart
from mozart.core.logging import get_logger
from mozart.daemon.backpressure import BackpressureController
from mozart.daemon.config import DaemonConfig
from mozart.daemon.exceptions import JobSubmissionError
from mozart.daemon.job_service import JobService
from mozart.daemon.learning_hub import LearningHub
from mozart.daemon.monitor import ResourceMonitor
from mozart.daemon.output import StructuredOutput
from mozart.daemon.rate_coordinator import RateLimitCoordinator
from mozart.daemon.scheduler import GlobalSheetScheduler
from mozart.daemon.task_utils import log_task_exception
from mozart.daemon.types import JobRequest, JobResponse

_logger = get_logger("daemon.manager")


class DaemonJobStatus(str, Enum):
    """Status values for daemon-managed jobs.

    Inherits from ``str`` so ``meta.status`` serializes directly as
    a plain string in JSON/dict output — no ``.value`` calls needed.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobMeta:
    """Metadata tracked per job in the manager."""

    job_id: str
    config_path: Path
    workspace: Path
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    status: DaemonJobStatus = DaemonJobStatus.QUEUED
    error_message: str | None = None
    error_traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for JSON-RPC responses."""
        result: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "config_path": str(self.config_path),
            "workspace": str(self.workspace),
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
        }
        if self.error_traceback:
            result["error_traceback"] = self.error_traceback
        return result


class JobManager:
    """Manages concurrent job execution within the daemon.

    Wraps JobService with task tracking and concurrency control.
    Each submitted job becomes an asyncio.Task that the manager
    tracks from start to completion/cancellation.
    """

    def __init__(
        self,
        config: DaemonConfig,
        *,
        start_time: float | None = None,
        monitor: ResourceMonitor | None = None,
    ) -> None:
        self._config = config
        self._start_time = start_time or time.monotonic()

        # Phase 3: Centralized learning hub.
        # Single GlobalLearningStore shared across all jobs — pattern
        # discoveries in Job A are instantly available to Job B.
        self._learning_hub = LearningHub()

        # Deferred to start() where the learning hub's store is available.
        self._service: JobService | None = None
        self._jobs: dict[str, asyncio.Task[Any]] = {}
        self._job_meta: dict[str, JobMeta] = {}
        self._concurrency_semaphore = asyncio.Semaphore(
            config.max_concurrent_jobs,
        )
        self._shutting_down = False
        self._shutdown_event = asyncio.Event()
        self._recent_failures: deque[float] = deque()

        # Phase 3: Global sheet scheduler for cross-job coordination.
        # Infrastructure is built and tested but not yet wired into the
        # execution path.  Currently, jobs run monolithically via
        # JobService.start_job().  When wired, _run_job_task() will
        # decompose jobs into sheets, register them via register_job(),
        # and use next_sheet()/mark_complete() for per-sheet dispatch.
        self._scheduler = GlobalSheetScheduler(config)

        # Phase 3: Cross-job rate limit coordination.
        # Built and tested; wired into the scheduler so next_sheet()
        # skips rate-limited backends.  Not yet active because the
        # scheduler itself is not yet driving execution.  When wired,
        # job runners or backends will call report_rate_limit() to
        # feed data into the coordinator.
        self._rate_coordinator = RateLimitCoordinator()
        self._scheduler.set_rate_limiter(self._rate_coordinator)

        # Phase 3: Backpressure controller.
        # Uses a single ResourceMonitor instance shared with DaemonProcess
        # for both periodic monitoring and point-in-time backpressure checks.
        # When no monitor is injected (e.g. unit tests), a standalone one
        # is created that only does point-in-time reads.
        self._monitor = monitor or ResourceMonitor(config.resource_limits, manager=self)
        self._backpressure = BackpressureController(
            self._monitor, self._rate_coordinator,
        )
        self._scheduler.set_backpressure(self._backpressure)

    # ─── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start daemon subsystems (learning hub, monitor, etc.)."""
        await self._learning_hub.start()
        # Create service with shared store now that the hub is initialized
        self._service = JobService(
            output=StructuredOutput(),
            global_learning_store=self._learning_hub.store,
            rate_limit_callback=self._on_rate_limit,
        )
        _logger.info(
            "manager.started",
            scheduler_status="instantiated_not_wired",
            scheduler_note="Phase 3 scheduler and rate coordinator are built "
            "and tested but not yet driving execution. Jobs run "
            "monolithically via JobService.",
        )

    @property
    def _checked_service(self) -> JobService:
        """Get the job service, raising if not yet started."""
        if self._service is None:
            raise RuntimeError("JobManager not started — call start() first")
        return self._service

    # ─── RPC Handlers ─────────────────────────────────────────────────

    async def submit_job(self, request: JobRequest) -> JobResponse:
        """Validate config, create task, return immediately."""
        if self._shutting_down:
            return JobResponse(
                job_id="",
                status="rejected",
                message="Daemon is shutting down",
            )

        if not self._backpressure.should_accept_job():
            return JobResponse(
                job_id="",
                status="rejected",
                message="System under high pressure — try again later",
            )

        job_id = f"{request.config_path.stem}-{uuid.uuid4().hex[:8]}"

        # Validate config exists
        if not request.config_path.exists():
            return JobResponse(
                job_id=job_id,
                status="rejected",
                message=f"Config file not found: {request.config_path}",
            )

        workspace = request.workspace or Path(f"workspace/{job_id}")

        meta = JobMeta(
            job_id=job_id,
            config_path=request.config_path,
            workspace=workspace,
        )
        self._job_meta[job_id] = meta

        task = asyncio.create_task(
            self._run_job_task(job_id, request),
            name=f"job-{job_id}",
        )
        self._jobs[job_id] = task
        task.add_done_callback(lambda t: self._on_task_done(job_id, t))

        _logger.info(
            "job.submitted",
            job_id=job_id,
            config_path=str(request.config_path),
        )

        return JobResponse(
            job_id=job_id,
            status="accepted",
            message=f"Job queued (concurrency limit: {self._config.max_concurrent_jobs})",
        )

    async def get_job_status(self, job_id: str, workspace: Path | None = None) -> dict[str, Any]:
        """Get status of a specific job."""
        meta = self._job_meta.get(job_id)
        if meta is None:
            raise JobSubmissionError(f"Job '{job_id}' not found")

        result = meta.to_dict()

        # If running, try to get deeper status from state backend
        if meta.status == DaemonJobStatus.RUNNING and workspace:
            state = await self._checked_service.get_status(meta.job_id, workspace)
            if state:
                result["current_sheet"] = state.current_sheet
                result["total_sheets"] = state.total_sheets
                result["last_completed_sheet"] = state.last_completed_sheet

        return result

    async def pause_job(self, job_id: str, workspace: Path | None = None) -> bool:
        """Send pause signal to a running job."""
        meta = self._job_meta.get(job_id)
        if meta is None:
            raise JobSubmissionError(f"Job '{job_id}' not found")
        if meta.status != DaemonJobStatus.RUNNING:
            raise JobSubmissionError(
                f"Job '{job_id}' is {meta.status}, not running"
            )

        ws = workspace or meta.workspace
        return await self._checked_service.pause_job(meta.job_id, ws)

    async def resume_job(self, job_id: str, workspace: Path | None = None) -> JobResponse:
        """Resume a paused job by creating a new task.

        If an old task for this job is still running (e.g., not yet fully
        paused), it is cancelled before the new resume task is created to
        prevent detached/duplicate execution.
        """
        meta = self._job_meta.get(job_id)
        if meta is None:
            raise JobSubmissionError(f"Job '{job_id}' not found")

        # Cancel stale task to prevent detached execution
        old_task = self._jobs.pop(job_id, None)
        if old_task is not None and not old_task.done():
            old_task.cancel()
            _logger.info("job.resume_cancelled_stale_task", job_id=job_id)

        ws = workspace or meta.workspace
        meta.status = DaemonJobStatus.QUEUED

        task = asyncio.create_task(
            self._resume_job_task(job_id, ws),
            name=f"job-resume-{job_id}",
        )
        self._jobs[job_id] = task
        task.add_done_callback(lambda t: self._on_task_done(job_id, t))

        return JobResponse(
            job_id=job_id,
            status="accepted",
            message="Job resume queued",
        )

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job task.

        Deregisters the job from the global sheet scheduler to remove
        any pending sheets from the global queue on cancellation.
        """
        task = self._jobs.get(job_id)
        if task is None:
            return False

        task.cancel()
        meta = self._job_meta.get(job_id)
        if meta:
            meta.status = DaemonJobStatus.CANCELLED

        await self._scheduler.deregister_job(job_id)
        _logger.info("job.cancelled", job_id=job_id)
        return True

    async def list_jobs(self) -> list[dict[str, Any]]:
        """List all tracked jobs and their states."""
        result: list[dict[str, Any]] = []
        for meta in self._job_meta.values():
            result.append(meta.to_dict())
        return result

    async def get_daemon_status(self) -> dict[str, Any]:
        """Build daemon status summary.

        Returns all fields required by the ``DaemonStatus`` Pydantic model
        so ``DaemonClient.status()`` can deserialize without crashing.
        """
        mem = self._monitor.current_memory_mb()
        return {
            "pid": os.getpid(),
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "running_jobs": self.running_count,
            "total_jobs_active": self.active_job_count,
            "memory_usage_mb": round(mem, 1) if mem is not None else 0.0,
            "version": getattr(mozart, "__version__", "0.1.0"),
        }

    # ─── Shutdown ─────────────────────────────────────────────────────

    async def shutdown(self, graceful: bool = True) -> None:
        """Cancel all running jobs, optionally waiting for sheets.

        Deregisters all active jobs from the global sheet scheduler
        to clean up any pending sheets, running-sheet tracking, and
        dependency data before the daemon exits.
        """
        self._shutting_down = True

        if graceful:
            timeout = self._config.shutdown_timeout_seconds
            _logger.info(
                "manager.shutting_down",
                graceful=True,
                timeout=timeout,
                running_jobs=self.running_count,
            )

            # Wait for running tasks to complete (up to timeout)
            running = [t for t in self._jobs.values() if not t.done()]
            if running:
                _, pending = await asyncio.wait(
                    running, timeout=timeout,
                )
                for task in pending:
                    task.cancel()
                if pending:
                    results = await asyncio.gather(*pending, return_exceptions=True)
                    for result in results:
                        if isinstance(result, BaseException):
                            _logger.warning(
                                "manager.shutdown_task_exception",
                                error=str(result),
                                error_type=type(result).__name__,
                            )
        else:
            _logger.info("manager.shutting_down", graceful=False)
            for task in self._jobs.values():
                if not task.done():
                    task.cancel()
            if self._jobs:
                results = await asyncio.gather(
                    *self._jobs.values(), return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        _logger.warning(
                            "manager.shutdown_task_exception",
                            error=str(result),
                            error_type=type(result).__name__,
                        )

        # Deregister all known jobs from the scheduler to clean up
        # heap entries, running-sheet tracking, and dependency data.
        # Uses _job_meta (not _jobs) because task done-callbacks may
        # have already cleared entries from _jobs during cancellation.
        for job_id in list(self._job_meta.keys()):
            await self._scheduler.deregister_job(job_id)

        self._jobs.clear()

        # Stop centralized learning hub (final persist + cleanup)
        await self._learning_hub.stop()

        self._shutdown_event.set()
        _logger.info("manager.shutdown_complete")

    async def wait_for_shutdown(self) -> None:
        """Block until shutdown is complete."""
        await self._shutdown_event.wait()

    # ─── Properties ───────────────────────────────────────────────────

    @property
    def shutting_down(self) -> bool:
        """Whether the manager is in the process of shutting down."""
        return self._shutting_down

    @property
    def running_count(self) -> int:
        """Number of currently running jobs."""
        return sum(
            1 for m in self._job_meta.values() if m.status == DaemonJobStatus.RUNNING
        )

    @property
    def active_job_count(self) -> int:
        """Number of active jobs (proxy for sheet count until Phase 3).

        Returns ``running_count`` since the scheduler is not yet wired
        into the execution path.  When Phase 3 wires per-sheet dispatch,
        this should be replaced with a true sheet-level count from
        ``self._scheduler.active_count``.
        """
        # TODO(Phase 3): return self._scheduler.active_count when wired
        return self.running_count

    _FAILURE_RATE_WINDOW = 60.0  # seconds
    _FAILURE_RATE_THRESHOLD = 3

    @property
    def failure_rate_elevated(self) -> bool:
        """Whether the recent job failure rate is elevated.

        Returns True if more than ``_FAILURE_RATE_THRESHOLD`` unexpected
        exceptions have occurred within the last ``_FAILURE_RATE_WINDOW``
        seconds.  Used by ``HealthChecker.readiness()`` to degrade the
        health signal when systemic failures are occurring.
        """
        now = time.monotonic()
        cutoff = now - self._FAILURE_RATE_WINDOW
        # Prune expired entries from the left
        while self._recent_failures and self._recent_failures[0] < cutoff:
            self._recent_failures.popleft()
        return len(self._recent_failures) > self._FAILURE_RATE_THRESHOLD

    @property
    def notifications_degraded(self) -> bool:
        """Whether notification delivery is degraded (forwarded from JobService)."""
        if self._service is None:
            return False
        return self._service.notifications_degraded

    @property
    def scheduler(self) -> GlobalSheetScheduler:
        """Access the global sheet scheduler for cross-job coordination."""
        return self._scheduler

    @property
    def rate_coordinator(self) -> RateLimitCoordinator:
        """Access the rate limit coordinator for cross-job rate limiting."""
        return self._rate_coordinator

    @property
    def backpressure(self) -> BackpressureController:
        """Access the backpressure controller for load management."""
        return self._backpressure

    @property
    def learning_hub(self) -> LearningHub:
        """Access the centralized learning hub."""
        return self._learning_hub

    # ─── Internal ─────────────────────────────────────────────────────

    async def _on_rate_limit(
        self,
        backend_type: str,
        wait_seconds: float,
        job_id: str,
        sheet_num: int,
    ) -> None:
        """Forward rate limit events from runners to the coordinator.

        Wiring prerequisite (P025): This callback is passed through
        JobService → RunnerContext → RecoveryMixin._handle_rate_limit()
        so that rate limit detections from any running job feed into the
        daemon's centralized RateLimitCoordinator.  The coordinator
        then informs the scheduler to skip the limited backend.
        """
        await self._rate_coordinator.report_rate_limit(
            backend_type=backend_type,
            wait_seconds=wait_seconds,
            job_id=job_id,
            sheet_num=sheet_num,
        )

    async def _run_managed_task(
        self,
        job_id: str,
        coro: Coroutine[Any, Any, DaemonJobStatus | None],
        *,
        start_event: str = "job.started",
        fail_event: str = "job.failed",
    ) -> None:
        """Shared lifecycle wrapper for job tasks.

        Acquires the concurrency semaphore, tracks status transitions,
        and handles CancelledError / TimeoutError / Exception uniformly.

        Jobs are guarded by ``job_timeout_seconds`` — if a job exceeds
        this wall-clock limit, it is cancelled with FAILED status and a
        descriptive error message.

        Args:
            job_id: The job being executed.
            coro: Awaitable that performs the actual work. May return a
                ``DaemonJobStatus`` to override the default COMPLETED
                status on success (e.g. PAUSED).
            start_event: Structlog event name for the start log.
            fail_event: Structlog event name for the failure log.
        """
        meta = self._job_meta[job_id]
        timeout = self._config.job_timeout_seconds

        async with self._concurrency_semaphore:
            meta.status = DaemonJobStatus.RUNNING
            meta.started_at = time.monotonic()
            _logger.info(start_event, job_id=job_id, timeout_seconds=timeout)

            try:
                result_status = await asyncio.wait_for(coro, timeout=timeout)
                if isinstance(result_status, DaemonJobStatus):
                    meta.status = result_status
                else:
                    meta.status = DaemonJobStatus.COMPLETED
                _logger.info(
                    "job.paused" if meta.status == DaemonJobStatus.PAUSED
                    else "job.completed",
                    job_id=job_id,
                )

            except asyncio.TimeoutError:
                meta.status = DaemonJobStatus.FAILED
                elapsed = time.monotonic() - (meta.started_at or 0)
                meta.error_message = (
                    f"Job exceeded timeout of {timeout:.0f}s "
                    f"(ran for {elapsed:.0f}s)"
                )
                self._recent_failures.append(time.monotonic())
                _logger.error(
                    "job.timeout",
                    job_id=job_id,
                    timeout_seconds=timeout,
                    elapsed_seconds=round(elapsed, 1),
                )

            except asyncio.CancelledError:
                meta.status = DaemonJobStatus.CANCELLED
                _logger.info("job.cancelled_during_execution", job_id=job_id)
                raise

            except Exception as exc:
                meta.status = DaemonJobStatus.FAILED
                meta.error_message = str(exc)
                meta.error_traceback = traceback.format_exc()
                self._recent_failures.append(time.monotonic())
                _logger.exception(fail_event, job_id=job_id)

    async def _run_job_task(self, job_id: str, request: JobRequest) -> None:
        """Task coroutine that runs a single job.

        Currently runs jobs monolithically via ``JobService.start_job()``.

        TODO(Phase 3 — scheduler integration): Replace monolithic execution
        with per-sheet dispatch through ``self._scheduler``:
          1. Parse sheets from config: ``config.sheets``
          2. Build SheetInfo list + dependency DAG
          3. ``await self._scheduler.register_job(job_id, sheets, deps)``
          4. Dispatch loop: ``entry = await self._scheduler.next_sheet()``
          5. Spawn per-sheet tasks, call ``mark_complete()`` on finish
          6. On error/cancel: ``await self._scheduler.deregister_job(job_id)``
        """

        async def _execute() -> DaemonJobStatus:
            from mozart.core.checkpoint import JobStatus
            from mozart.core.config import JobConfig

            config = JobConfig.from_yaml(request.config_path)
            if request.workspace:
                config = config.model_copy(
                    update={"workspace": request.workspace},
                )

            summary = await self._checked_service.start_job(
                config,
                fresh=request.fresh,
                start_sheet=request.start_sheet,
                self_healing=request.self_healing,
                self_healing_auto_confirm=request.self_healing_auto_confirm,
                dry_run=request.dry_run,
            )

            if summary.final_status == JobStatus.PAUSED:
                return DaemonJobStatus.PAUSED
            return DaemonJobStatus.COMPLETED

        await self._run_managed_task(job_id, _execute())

    async def _resume_job_task(self, job_id: str, workspace: Path) -> None:
        """Task coroutine that resumes a paused job."""

        async def _execute() -> None:
            await self._checked_service.resume_job(job_id, workspace)

        await self._run_managed_task(
            job_id, _execute(),
            start_event="job.resuming",
            fail_event="job.resume_failed",
        )

    def _on_task_done(self, job_id: str, task: asyncio.Task[Any]) -> None:
        """Callback when a job task completes (success, error, or cancel)."""
        self._jobs.pop(job_id, None)
        log_task_exception(task, _logger, "job.task_failed")

        exc = task.exception() if not task.cancelled() else None
        if exc:
            meta = self._job_meta.get(job_id)
            if meta and meta.status == DaemonJobStatus.RUNNING:
                meta.status = DaemonJobStatus.FAILED
                meta.error_message = str(exc)

        self._prune_job_history()

    # ─── Wiring Adapters (Phase 3 prep) ─────────────────────────────

    @staticmethod
    def _build_sheet_infos(
        job_id: str,
        config: Any,
    ) -> list[Any]:
        """Build a list of ``SheetInfo`` objects from a ``JobConfig``.

        Translates config-layer sheet definitions into scheduler-layer
        ``SheetInfo`` dataclasses.  Each sheet gets the backend type and
        model from the job config, so the scheduler can forward them to
        the rate limiter for per-model tracking.

        Stub — returns an empty list until Phase 3 wires per-sheet dispatch.

        Args:
            job_id: The daemon's job identifier.
            config: A ``JobConfig`` instance (typed as Any to avoid import
                    in this stub phase).

        Returns:
            A list of ``SheetInfo`` objects, one per concrete sheet.
        """
        # TODO(Phase 3 wiring): Implement translation from
        # config.sheet → SheetInfo list, using config.backend.type
        # and config.backend.model / config.backend.cli_model.
        return []

    @staticmethod
    def _build_dependency_map(
        config: Any,
    ) -> dict[int, set[int]]:
        """Build a sheet dependency DAG from a ``JobConfig``.

        Translates ``config.sheet.dependencies`` (``dict[int, list[int]]``)
        into the scheduler's format (``dict[int, set[int]]``).

        Stub — returns an empty dict until Phase 3 wires per-sheet dispatch.

        Args:
            config: A ``JobConfig`` instance.

        Returns:
            Dependency map: ``{sheet_num: {prerequisite_sheet_nums}}``.
        """
        # TODO(Phase 3 wiring): return {
        #     sn: set(deps) for sn, deps in config.sheet.dependencies.items()
        # }
        return {}

    def _prune_job_history(self) -> None:
        """Evict oldest terminal jobs when history exceeds max_job_history."""
        max_history = self._config.max_job_history
        terminal = sorted(
            (
                (jid, m) for jid, m in self._job_meta.items()
                if m.status in (
                    DaemonJobStatus.COMPLETED,
                    DaemonJobStatus.FAILED,
                    DaemonJobStatus.CANCELLED,
                )
            ),
            key=lambda x: x[1].submitted_at,
        )
        excess = len(terminal) - max_history
        if excess > 0:
            pruned_ids = [jid for jid, _ in terminal[:excess]]
            for jid in pruned_ids:
                self._job_meta.pop(jid, None)
            _logger.debug(
                "manager.job_history_pruned",
                pruned_count=excess,
                oldest_pruned=pruned_ids[0],
            )


__all__ = ["DaemonJobStatus", "JobManager", "JobMeta"]
