"""Job lifecycle control service — conductor-only proxy.

Every operation routes through the conductor via ``DaemonClient`` IPC.
If the conductor is not running, all operations fail with clear errors.
There is no subprocess fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from marianne.core.config import JobConfig
from marianne.core.logging import get_logger
from marianne.daemon.exceptions import DaemonNotRunningError
from marianne.daemon.ipc.client import DaemonClient
from marianne.daemon.types import JobRequest

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
    via_daemon: bool = True


@dataclass
class JobActionResult:
    """Result of a job action (pause/resume/cancel)."""

    success: bool
    job_id: str
    status: str
    message: str
    via_daemon: bool = True


@dataclass
class ProcessHealth:
    """Process health check result (derived from conductor state)."""

    pid: int | None
    is_alive: bool
    is_zombie_state: bool
    process_exists: bool
    cpu_percent: float | None = None
    memory_mb: float | None = None
    uptime_seconds: float | None = None


class JobControlService:
    """Conductor-only proxy for job lifecycle control.

    Parameters
    ----------
    daemon_client:
        A ``DaemonClient`` connected to the conductor's Unix socket.
    """

    def __init__(self, daemon_client: DaemonClient) -> None:
        if daemon_client is None:
            raise ValueError(
                "DaemonClient is required — the dashboard requires a running conductor."
            )
        self._client = daemon_client

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    async def start_job(
        self,
        config_path: Path | None = None,
        config_content: str | None = None,
        workspace: Path | None = None,
        start_sheet: int = 1,
        self_healing: bool = False,
    ) -> JobStartResult:
        """Submit a new job to the conductor.

        Only file-based configs are supported (the conductor requires a
        path it can resolve).  Inline ``config_content`` is written to a
        temporary file and the path is forwarded.

        Args:
            config_path: Path to YAML config file.
            config_content: Inline YAML config content (written to temp file).
            workspace: Override workspace directory.
            start_sheet: Starting sheet number.
            self_healing: Enable self-healing mode.

        Returns:
            JobStartResult with job details.

        Raises:
            ValueError: If neither config_path nor config_content provided.
            FileNotFoundError: If config_path doesn't exist.
            RuntimeError: If the conductor rejects the submission.
        """
        if not config_path and not config_content:
            raise ValueError("Must provide either config_path or config_content")

        resolved_path = config_path
        temp_path: str | None = None

        if config_content:
            import os
            import tempfile

            fd, temp_path = tempfile.mkstemp(suffix=".yaml", text=True)
            os.fchmod(fd, 0o600)
            try:
                with open(fd, "w") as f:
                    f.write(config_content)
            except Exception:
                os.close(fd)
                raise
            resolved_path = Path(temp_path)

        assert resolved_path is not None

        if config_path and not config_content:
            resolved = config_path.resolve()
            if resolved.suffix not in (".yaml", ".yml"):
                raise ValueError(f"Config path must be a YAML file (.yaml/.yml): {config_path}")
            if ".." in config_path.parts:
                raise ValueError(f"Config path must not contain '..' traversal: {config_path}")
            if not resolved.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            request = JobRequest(
                config_path=resolved_path.resolve(),
                workspace=workspace.resolve() if workspace else None,
                self_healing=self_healing,
                start_sheet=start_sheet if start_sheet > 1 else None,
            )
            response = await self._client.submit_job(request)

            config = (
                JobConfig.from_yaml_string(config_content)
                if config_content
                else JobConfig.from_yaml(resolved_path)
            )

            ws = workspace or (Path(config.workspace) if config.workspace else Path.cwd())

            logger.info(
                "job_submitted_to_conductor",
                job_id=response.job_id,
                job_name=config.name,
                status=response.status,
            )

            return JobStartResult(
                job_id=response.job_id,
                job_name=config.name,
                status=response.status,
                workspace=ws,
                total_sheets=config.sheet.total_sheets,
            )

        except DaemonNotRunningError:
            raise RuntimeError("Conductor not running. Start it with: mzt start") from None
        except Exception as e:
            if isinstance(e, (ValueError, FileNotFoundError)):
                raise
            raise RuntimeError(f"Failed to submit job to conductor: {e}") from e
        finally:
            if temp_path is not None:
                import os

                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    async def pause_job(self, job_id: str) -> JobActionResult:
        """Pause a running job via the conductor."""
        try:
            await self._client.pause_job(job_id, "")
            return JobActionResult(
                success=True,
                job_id=job_id,
                status="paused",
                message=f"Pause request sent to conductor for job {job_id}",
            )
        except DaemonNotRunningError:
            raise RuntimeError("Conductor not running.") from None

    async def resume_job(self, job_id: str) -> JobActionResult:
        """Resume a paused job via the conductor."""
        try:
            await self._client.resume_job(job_id, "")
            return JobActionResult(
                success=True,
                job_id=job_id,
                status="running",
                message=f"Resume request sent to conductor for job {job_id}",
            )
        except DaemonNotRunningError:
            raise RuntimeError("Conductor not running.") from None

    async def cancel_job(self, job_id: str) -> JobActionResult:
        """Cancel a running or paused job via the conductor."""
        try:
            await self._client.cancel_job(job_id, "")
            return JobActionResult(
                success=True,
                job_id=job_id,
                status="cancelled",
                message=f"Cancel request sent to conductor for job {job_id}",
            )
        except DaemonNotRunningError:
            raise RuntimeError("Conductor not running.") from None

    async def delete_job(self, job_id: str) -> bool:
        """Delete a terminal job from the conductor registry."""
        try:
            result = await self._client.clear_jobs(job_ids=[job_id])
            deleted: bool = bool(result.get("deleted", 0))
            if deleted:
                logger.info("job_deleted", job_id=job_id)
            return deleted
        except DaemonNotRunningError:
            raise RuntimeError("Conductor not running.") from None

    # ------------------------------------------------------------------
    # Process health (for MCP compatibility)
    # ------------------------------------------------------------------

    async def verify_process_health(self, job_id: str) -> ProcessHealth:
        """Check job health by querying conductor state.

        Returns a ``ProcessHealth`` derived from the job's ``CheckpointState``
        as reported by the conductor.
        """
        try:
            status_data = await self._client.get_job_status(job_id, "")
            from marianne.core.checkpoint import CheckpointState, JobStatus

            state = CheckpointState(**status_data)

            is_terminal = state.status in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            )

            return ProcessHealth(
                pid=state.pid,
                is_alive=not is_terminal,
                is_zombie_state=state.is_zombie() if not is_terminal else False,
                process_exists=not is_terminal,
                uptime_seconds=None,
                cpu_percent=None,
                memory_mb=None,
            )
        except DaemonNotRunningError:
            return ProcessHealth(
                pid=None,
                is_alive=False,
                is_zombie_state=False,
                process_exists=False,
            )
