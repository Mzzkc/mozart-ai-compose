"""Job manager for the Mozart daemon.

Maps job IDs to asyncio.Tasks, enforces concurrency limits via semaphore,
routes IPC requests to JobService, and cancels all tasks on shutdown.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
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
from mozart.daemon.types import JobRequest, JobResponse

_logger = get_logger("daemon.manager")


@dataclass
class JobMeta:
    """Metadata tracked per job in the manager."""

    job_id: str
    config_path: Path
    workspace: Path
    submitted_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    status: str = "queued"  # queued, running, completed, failed, cancelled


class JobManager:
    """Manages concurrent job execution within the daemon.

    Wraps JobService with task tracking and concurrency control.
    Each submitted job becomes an asyncio.Task that the manager
    tracks from start to completion/cancellation.
    """

    def __init__(self, config: DaemonConfig) -> None:
        self._config = config

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

        # Phase 3: Global sheet scheduler for cross-job coordination.
        # Routes sheet execution through priority queue with fair-share.
        # The semaphore above gates job-level entry; the scheduler gates
        # sheet-level concurrency across all jobs.
        self._scheduler = GlobalSheetScheduler(config)

        # Phase 3: Cross-job rate limit coordination.
        # When any job hits a rate limit, all jobs using that backend
        # back off. Wired into the scheduler so next_sheet() skips
        # rate-limited backends.
        self._rate_coordinator = RateLimitCoordinator()
        self._scheduler.set_rate_limiter(self._rate_coordinator)

        # Phase 3: Backpressure controller.
        # Monitors memory/rate-limit pressure and throttles or rejects
        # sheet dispatch and job submissions when the system is stressed.
        # NOTE: This ResourceMonitor is used only for point-in-time checks
        # by BackpressureController (e.g. current memory usage). It does NOT
        # run a periodic monitoring loop. The separate monitor created in
        # DaemonProcess.run() handles periodic monitoring + orphan cleanup.
        self._monitor = ResourceMonitor(config.resource_limits, manager=self)
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
        )
        _logger.info("manager.started")

    @property
    def _svc(self) -> JobService:
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

        result: dict[str, Any] = {
            "job_id": meta.job_id,
            "status": meta.status,
            "config_path": str(meta.config_path),
            "workspace": str(meta.workspace),
            "submitted_at": meta.submitted_at,
            "started_at": meta.started_at,
        }

        # If running, try to get deeper status from state backend
        if meta.status == "running" and workspace:
            state = await self._svc.get_status(meta.job_id, workspace)
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
        if meta.status != "running":
            raise JobSubmissionError(
                f"Job '{job_id}' is {meta.status}, not running"
            )

        ws = workspace or meta.workspace
        return await self._svc.pause_job(meta.job_id, ws)

    async def resume_job(self, job_id: str, workspace: Path | None = None) -> JobResponse:
        """Resume a paused job by creating a new task."""
        meta = self._job_meta.get(job_id)
        if meta is None:
            raise JobSubmissionError(f"Job '{job_id}' not found")

        ws = workspace or meta.workspace
        meta.status = "queued"

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
        """Cancel a running job task."""
        task = self._jobs.get(job_id)
        if task is None:
            return False

        task.cancel()
        meta = self._job_meta.get(job_id)
        if meta:
            meta.status = "cancelled"

        _logger.info("job.cancelled", job_id=job_id)
        return True

    async def list_jobs(self) -> list[dict[str, Any]]:
        """List all tracked jobs and their states."""
        result: list[dict[str, Any]] = []
        for meta in self._job_meta.values():
            result.append({
                "job_id": meta.job_id,
                "status": meta.status,
                "config_path": str(meta.config_path),
                "workspace": str(meta.workspace),
                "submitted_at": meta.submitted_at,
                "started_at": meta.started_at,
            })
        return result

    async def get_daemon_status(self) -> dict[str, Any]:
        """Build daemon status summary."""
        return {
            "pid": os.getpid(),
            "running_jobs": self.running_count,
            "total_sheets_active": self.active_sheet_count,
            "version": getattr(mozart, "__version__", "0.1.0"),
        }

    # ─── Shutdown ─────────────────────────────────────────────────────

    async def shutdown(self, graceful: bool = True) -> None:
        """Cancel all running jobs, optionally waiting for sheets."""
        self._shutting_down = True

        if graceful:
            timeout = getattr(
                self._config, "shutdown_timeout_seconds", 300.0,
            )
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
                    await asyncio.gather(*pending, return_exceptions=True)
        else:
            _logger.info("manager.shutting_down", graceful=False)
            for task in self._jobs.values():
                if not task.done():
                    task.cancel()
            if self._jobs:
                await asyncio.gather(
                    *self._jobs.values(), return_exceptions=True,
                )

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
    def running_count(self) -> int:
        """Number of currently running jobs."""
        return sum(
            1 for m in self._job_meta.values() if m.status == "running"
        )

    @property
    def active_sheet_count(self) -> int:
        """Total active sheets across all jobs.

        Uses the GlobalSheetScheduler's active count when available,
        falling back to running_count as a proxy.
        """
        if self._scheduler.active_count > 0:
            return self._scheduler.active_count
        return self.running_count

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

    async def _run_job_task(self, job_id: str, request: JobRequest) -> None:
        """Task coroutine that runs a single job."""
        meta = self._job_meta[job_id]

        async with self._concurrency_semaphore:
            meta.status = "running"
            meta.started_at = time.monotonic()
            _logger.info("job.started", job_id=job_id)

            try:
                from mozart.core.config import JobConfig

                config = JobConfig.from_yaml(request.config_path)
                if request.workspace:
                    config = config.model_copy(
                        update={"workspace": request.workspace},
                    )

                await self._svc.start_job(
                    config,
                    fresh=request.fresh,
                    self_healing=request.self_healing,
                    self_healing_auto_confirm=request.self_healing_auto_confirm,
                )
                meta.status = "completed"
                _logger.info("job.completed", job_id=job_id)

            except asyncio.CancelledError:
                meta.status = "cancelled"
                _logger.info("job.cancelled_during_execution", job_id=job_id)
                raise

            except Exception:
                meta.status = "failed"
                _logger.exception("job.failed", job_id=job_id)

    async def _resume_job_task(self, job_id: str, workspace: Path) -> None:
        """Task coroutine that resumes a paused job."""
        meta = self._job_meta[job_id]

        async with self._concurrency_semaphore:
            meta.status = "running"
            meta.started_at = time.monotonic()
            _logger.info("job.resuming", job_id=job_id)

            try:
                await self._svc.resume_job(job_id, workspace)
                meta.status = "completed"
                _logger.info("job.completed", job_id=job_id)

            except asyncio.CancelledError:
                meta.status = "cancelled"
                raise

            except Exception:
                meta.status = "failed"
                _logger.exception("job.resume_failed", job_id=job_id)

    def _on_task_done(self, job_id: str, task: asyncio.Task[Any]) -> None:
        """Callback when a job task completes (success, error, or cancel)."""
        self._jobs.pop(job_id, None)

        exc = task.exception() if not task.cancelled() else None
        if exc:
            meta = self._job_meta.get(job_id)
            if meta and meta.status == "running":
                meta.status = "failed"

        self._prune_job_history()

    def _prune_job_history(self) -> None:
        """Evict oldest terminal jobs when history exceeds max_job_history."""
        max_history = self._config.max_job_history
        terminal = sorted(
            (
                (jid, m) for jid, m in self._job_meta.items()
                if m.status in ("completed", "failed", "cancelled")
            ),
            key=lambda x: x[1].submitted_at,
        )
        excess = len(terminal) - max_history
        if excess > 0:
            for jid, _ in terminal[:excess]:
                self._job_meta.pop(jid, None)


__all__ = ["JobManager", "JobMeta"]
