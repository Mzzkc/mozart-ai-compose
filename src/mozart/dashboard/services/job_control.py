"""Job lifecycle control service."""
from __future__ import annotations

import asyncio
import os
import signal
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from mozart.core.checkpoint import CheckpointState, JobStatus
from mozart.core.config import JobConfig
from mozart.core.logging import get_logger

if TYPE_CHECKING:
    from mozart.state.base import StateBackend

logger = get_logger("job_control")


@dataclass
class JobStartResult:
    """Result of starting a job."""
    job_id: str
    job_name: str
    status: str
    workspace: Path
    total_sheets: int
    pid: int | None = None


@dataclass
class JobActionResult:
    """Result of a job action (pause/resume/cancel)."""
    success: bool
    job_id: str
    status: str
    message: str


class JobControlService:
    """Service for controlling job lifecycle."""

    def __init__(self, state_backend: StateBackend, workspace_root: Path | None = None):
        self._state_backend = state_backend
        self._workspace_root = workspace_root or Path.cwd()
        self._running_processes: dict[str, asyncio.subprocess.Process] = {}

    async def start_job(
        self,
        config_path: Path | None = None,
        config_content: str | None = None,
        workspace: Path | None = None,
        start_sheet: int = 1,
        self_healing: bool = False,
    ) -> JobStartResult:
        """Start a new Mozart job execution.

        Args:
            config_path: Path to YAML config file.
            config_content: Inline YAML config content.
            workspace: Override workspace directory.
            start_sheet: Starting sheet number.
            self_healing: Enable self-healing mode.

        Returns:
            JobStartResult with job details and process info.

        Raises:
            ValueError: If neither config_path nor config_content provided.
            FileNotFoundError: If config_path doesn't exist.
            RuntimeError: If job execution fails to start.
        """
        if not config_path and not config_content:
            raise ValueError("Must provide either config_path or config_content")

        if config_path and not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Generate unique job ID
        job_id = str(uuid.uuid4())[:8]  # Short ID for readability

        try:
            # Build command arguments - using create_subprocess_exec for security
            cmd_args = [sys.executable, "-m", "mozart.cli", "run"]

            if config_content:
                # For inline content, we'll write to a temp file
                # This ensures proper YAML parsing through the CLI
                import tempfile
                temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml', text=True)
                try:
                    with open(temp_fd, 'w') as f:
                        f.write(config_content)
                    cmd_args.append(temp_path)
                finally:
                    os.close(temp_fd)  # Close the file descriptor
            else:
                cmd_args.append(str(config_path))

            # Add optional arguments
            if workspace:
                cmd_args.extend(["--workspace", str(workspace)])
            if start_sheet > 1:
                cmd_args.extend(["--start-sheet", str(start_sheet)])
            if self_healing:
                cmd_args.append("--self-healing")

            # Start subprocess with individual arguments (no shell=True)
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._workspace_root),
            )

            # Track the process
            self._running_processes[job_id] = process

            # Parse config to get job details
            if config_content:
                config = JobConfig.from_yaml_string(config_content)
            else:
                # config_path is guaranteed to exist due to earlier check
                assert config_path is not None, "config_path should not be None here"
                config = JobConfig.from_yaml(config_path)

            # Determine workspace
            if not workspace:
                workspace = Path(config.workspace) if config.workspace else self._workspace_root

            logger.info(
                "job_started",
                job_id=job_id,
                job_name=config.name,
                pid=process.pid,
                workspace=str(workspace),
                total_sheets=config.sheet.total_sheets,
                start_sheet=start_sheet,
                self_healing=self_healing,
            )

            return JobStartResult(
                job_id=job_id,
                job_name=config.name,
                status=JobStatus.RUNNING.value,
                workspace=workspace,
                total_sheets=config.sheet.total_sheets,
                pid=process.pid,
            )

        except Exception as e:
            # Clean up on failure
            if job_id in self._running_processes:
                process = self._running_processes.pop(job_id)
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except (TimeoutError, ProcessLookupError):
                    pass  # Process already dead or won't die

            logger.error(
                "job_start_failed",
                job_id=job_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise RuntimeError(f"Failed to start job: {e}") from e

    async def pause_job(self, job_id: str) -> JobActionResult:
        """Pause a running job by sending SIGSTOP.

        Args:
            job_id: Job identifier.

        Returns:
            JobActionResult with operation status.
        """
        state = await self._state_backend.load(job_id)
        if not state:
            return JobActionResult(
                success=False,
                job_id=job_id,
                status=JobStatus.FAILED.value,
                message=f"Job not found: {job_id}"
            )

        if state.status != JobStatus.RUNNING:
            return JobActionResult(
                success=False,
                job_id=job_id,
                status=state.status.value,
                message=f"Job is not running (status: {state.status.value})"
            )

        pid = await self.get_job_pid(job_id)
        if not pid:
            return JobActionResult(
                success=False,
                job_id=job_id,
                status=state.status.value,
                message="No process ID found for job"
            )

        try:
            os.kill(pid, signal.SIGSTOP)

            # Update state to paused
            state.mark_job_paused()
            await self._state_backend.save(state)

            logger.info(
                "job_paused",
                job_id=job_id,
                pid=pid,
            )

            return JobActionResult(
                success=True,
                job_id=job_id,
                status=JobStatus.PAUSED.value,
                message=f"Job {job_id} paused successfully"
            )

        except ProcessLookupError:
            # Process already dead - mark as zombie
            state.mark_zombie_detected("Process not found during pause")
            await self._state_backend.save(state)

            return JobActionResult(
                success=False,
                job_id=job_id,
                status=JobStatus.PAUSED.value,
                message=f"Process not found (marked as zombie): {job_id}"
            )

        except PermissionError:
            return JobActionResult(
                success=False,
                job_id=job_id,
                status=state.status.value,
                message=f"Permission denied to pause job {job_id}"
            )

        except OSError as e:
            logger.error(
                "job_pause_failed",
                job_id=job_id,
                pid=pid,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return JobActionResult(
                success=False,
                job_id=job_id,
                status=state.status.value,
                message=f"Failed to pause job: {e}"
            )

    async def resume_job(self, job_id: str) -> JobActionResult:
        """Resume a paused job by sending SIGCONT.

        Args:
            job_id: Job identifier.

        Returns:
            JobActionResult with operation status.
        """
        state = await self._state_backend.load(job_id)
        if not state:
            return JobActionResult(
                success=False,
                job_id=job_id,
                status=JobStatus.FAILED.value,
                message=f"Job not found: {job_id}"
            )

        if state.status != JobStatus.PAUSED:
            return JobActionResult(
                success=False,
                job_id=job_id,
                status=state.status.value,
                message=f"Job is not paused (status: {state.status.value})"
            )

        pid = await self.get_job_pid(job_id)
        if not pid:
            # Try to restart the job instead of resuming a dead process
            return await self._restart_job_execution(job_id, state)

        try:
            os.kill(pid, signal.SIGCONT)

            # Update state to running
            state.status = JobStatus.RUNNING
            state.updated_at = datetime.now()
            await self._state_backend.save(state)

            logger.info(
                "job_resumed",
                job_id=job_id,
                pid=pid,
            )

            return JobActionResult(
                success=True,
                job_id=job_id,
                status=JobStatus.RUNNING.value,
                message=f"Job {job_id} resumed successfully"
            )

        except ProcessLookupError:
            # Process is dead - attempt restart
            return await self._restart_job_execution(job_id, state)

        except PermissionError:
            return JobActionResult(
                success=False,
                job_id=job_id,
                status=state.status.value,
                message=f"Permission denied to resume job {job_id}"
            )

        except OSError as e:
            logger.error(
                "job_resume_failed",
                job_id=job_id,
                pid=pid,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return JobActionResult(
                success=False,
                job_id=job_id,
                status=state.status.value,
                message=f"Failed to resume job: {e}"
            )

    async def cancel_job(self, job_id: str) -> JobActionResult:
        """Cancel a job by sending SIGTERM.

        Args:
            job_id: Job identifier.

        Returns:
            JobActionResult with operation status.
        """
        state = await self._state_backend.load(job_id)
        if not state:
            return JobActionResult(
                success=False,
                job_id=job_id,
                status=JobStatus.FAILED.value,
                message=f"Job not found: {job_id}"
            )

        if state.status in (JobStatus.COMPLETED, JobStatus.CANCELLED):
            return JobActionResult(
                success=False,
                job_id=job_id,
                status=state.status.value,
                message=f"Job already finished (status: {state.status.value})"
            )

        pid = await self.get_job_pid(job_id)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)

                # Give process time to handle SIGTERM gracefully
                await asyncio.sleep(2.0)

                # Check if process is still running
                try:
                    os.kill(pid, 0)  # Check if process exists
                    # Still running - send SIGKILL
                    os.kill(pid, signal.SIGKILL)
                    logger.warning(
                        "job_force_killed",
                        job_id=job_id,
                        pid=pid,
                        reason="SIGTERM timeout"
                    )
                except ProcessLookupError:
                    # Process exited gracefully
                    pass

            except ProcessLookupError:
                # Process already dead
                pass
            except (PermissionError, OSError) as e:
                logger.error(
                    "job_kill_failed",
                    job_id=job_id,
                    pid=pid,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )

        # Update state to cancelled regardless of kill success
        state.status = JobStatus.CANCELLED
        state.completed_at = datetime.now()
        state.updated_at = datetime.now()
        state.pid = None  # Clear PID
        await self._state_backend.save(state)

        # Clean up process tracking
        if job_id in self._running_processes:
            del self._running_processes[job_id]

        logger.info(
            "job_cancelled",
            job_id=job_id,
            pid=pid,
        )

        return JobActionResult(
            success=True,
            job_id=job_id,
            status=JobStatus.CANCELLED.value,
            message=f"Job {job_id} cancelled successfully"
        )

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job record from state.

        Args:
            job_id: Job identifier.

        Returns:
            True if deleted, False if not found or still running.
        """
        state = await self._state_backend.load(job_id)
        if not state:
            return False

        # Prevent deletion of running jobs
        if state.status == JobStatus.RUNNING:
            pid = await self.get_job_pid(job_id)
            if pid:
                try:
                    os.kill(pid, 0)  # Check if process exists
                    # Process is still running
                    logger.warning(
                        "delete_blocked_running_job",
                        job_id=job_id,
                        pid=pid,
                    )
                    return False
                except ProcessLookupError:
                    # Process is dead but state shows running - safe to delete
                    pass

        # Delete from backend
        deleted = await self._state_backend.delete(job_id)

        # Clean up process tracking
        if job_id in self._running_processes:
            del self._running_processes[job_id]

        if deleted:
            logger.info(
                "job_deleted",
                job_id=job_id,
                status=state.status.value,
            )

        return deleted

    async def get_job_pid(self, job_id: str) -> int | None:
        """Get PID of running job process.

        Args:
            job_id: Job identifier.

        Returns:
            Process ID if found and alive, None otherwise.
        """
        # Check tracked processes first
        if job_id in self._running_processes:
            process = self._running_processes[job_id]
            if process.returncode is None:  # Process still running
                return process.pid

        # Check state backend
        state = await self._state_backend.load(job_id)
        if state and state.pid:
            try:
                os.kill(state.pid, 0)  # Check if process exists
                return state.pid
            except ProcessLookupError:
                # Process is dead but recorded in state
                logger.debug(
                    "dead_pid_in_state",
                    job_id=job_id,
                    pid=state.pid,
                )
                return None

        return None

    async def _restart_job_execution(
        self,
        job_id: str,
        state: CheckpointState
    ) -> JobActionResult:
        """Restart job execution from current state.

        Used when resuming a job whose process is dead.
        """
        try:
            # Use mozart resume command with parameterized arguments
            cmd_args = [sys.executable, "-m", "mozart.cli", "resume", job_id]

            # Add workspace if available in state
            if state.worktree_path:
                cmd_args.extend(["--workspace", str(state.worktree_path)])

            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._workspace_root),
            )

            # Track the new process
            self._running_processes[job_id] = process

            # Update state
            state.status = JobStatus.RUNNING
            state.pid = process.pid
            state.updated_at = datetime.now()
            await self._state_backend.save(state)

            logger.info(
                "job_restarted",
                job_id=job_id,
                new_pid=process.pid,
            )

            return JobActionResult(
                success=True,
                job_id=job_id,
                status=JobStatus.RUNNING.value,
                message=f"Job {job_id} restarted with new process"
            )

        except Exception as e:
            logger.error(
                "job_restart_failed",
                job_id=job_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return JobActionResult(
                success=False,
                job_id=job_id,
                status=state.status.value,
                message=f"Failed to restart job: {e}"
            )
