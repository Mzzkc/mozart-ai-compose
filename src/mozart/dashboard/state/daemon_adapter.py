"""StateBackend adapter that reads job state from the running daemon via IPC.

Wraps ``DaemonClient`` to satisfy the ``StateBackend`` ABC so the dashboard
can consume live daemon data without touching the filesystem directly.
All write methods raise ``NotImplementedError`` — the dashboard is read-only.
"""

from __future__ import annotations

import logging
from typing import Any

from mozart.core.checkpoint import CheckpointState, JobStatus, SheetStatus
from mozart.daemon.exceptions import DaemonError
from mozart.daemon.ipc.client import DaemonClient
from mozart.state.base import StateBackend
from mozart.utils.time import utc_now

_logger = logging.getLogger(__name__)


class DaemonStateAdapter(StateBackend):
    """Read-only ``StateBackend`` backed by live daemon IPC calls.

    Parameters
    ----------
    client:
        An already-configured ``DaemonClient`` instance.
    """

    def __init__(self, client: DaemonClient) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    async def load(self, job_id: str) -> CheckpointState | None:
        """Load state for a job from the daemon.

        Returns ``None`` if the job is not found (``DaemonError``).
        """
        try:
            data = await self._client.get_job_status(job_id, "")
            return CheckpointState(**data)
        except DaemonError:
            _logger.debug("load_job_not_found", extra={"job_id": job_id})
            return None

    async def list_jobs(self) -> list[CheckpointState]:
        """List all jobs by querying the daemon roster then enriching each."""
        roster: list[dict[str, Any]] = await self._client.list_jobs()
        results: list[CheckpointState] = []

        for entry in roster:
            job_id = entry.get("job_id", "")
            try:
                data = await self._client.get_job_status(job_id, "")
                results.append(CheckpointState(**data))
            except DaemonError:
                _logger.debug(
                    "list_jobs_fallback",
                    extra={"job_id": job_id},
                )
                # Construct a minimal CheckpointState from roster data
                results.append(
                    CheckpointState(
                        job_id=job_id,
                        job_name=job_id,
                        total_sheets=1,
                        status=JobStatus(entry.get("status", "pending")),
                        created_at=entry.get("submitted_at") or utc_now(),
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Write methods — not supported (dashboard is read-only)
    # ------------------------------------------------------------------

    async def save(self, state: CheckpointState) -> None:
        raise NotImplementedError("Dashboard is read-only")

    async def delete(self, job_id: str) -> bool:
        raise NotImplementedError("Dashboard is read-only")

    async def get_next_sheet(self, job_id: str) -> int | None:
        raise NotImplementedError("Dashboard is read-only")

    async def mark_sheet_status(
        self,
        job_id: str,
        sheet_num: int,
        status: SheetStatus,
        error_message: str | None = None,
    ) -> None:
        raise NotImplementedError("Dashboard is read-only")

    async def close(self) -> None:
        """No-op — DaemonClient uses per-request connections."""
