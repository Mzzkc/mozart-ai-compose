"""JSON file-based state backend.

Simple state storage using JSON files, similar to the .review-state
approach in the original bash script.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from mozart.core.checkpoint import CheckpointState, SheetStatus
from mozart.core.logging import get_logger
from mozart.state.base import StateBackend

# Module-level logger for state operations
_logger = get_logger("state.json")


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


class JsonStateBackend(StateBackend):
    """JSON file-based state storage.

    Stores each job's state in a separate JSON file within the state directory.
    File naming: {state_dir}/{job_id}.json
    """

    def __init__(self, state_dir: Path):
        """Initialize JSON backend.

        Args:
            state_dir: Directory to store state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _get_state_file(self, job_id: str) -> Path:
        """Get the state file path for a job."""
        # Sanitize job_id for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_id)
        return self.state_dir / f"{safe_id}.json"

    async def load(self, job_id: str) -> CheckpointState | None:
        """Load state from JSON file.

        Automatically detects and recovers zombie jobs (RUNNING status but
        process dead). When a zombie is detected, the state is updated to
        PAUSED and saved before returning.
        """
        state_file = self._get_state_file(job_id)
        if not state_file.exists():
            _logger.debug("state_file_not_found", job_id=job_id, path=str(state_file))
            return None

        try:
            with open(state_file) as f:
                data = json.load(f)
            state = CheckpointState.model_validate(data)

            # Check for zombie state and auto-recover
            if state.is_zombie():
                _logger.warning(
                    "zombie_auto_recovery",
                    job_id=job_id,
                    pid=state.pid,
                    status=state.status.value,
                )
                state.mark_zombie_detected(
                    reason="Detected on state load - process no longer running"
                )
                # Save the recovered state
                await self.save(state)

            _logger.debug(
                "checkpoint_loaded",
                job_id=job_id,
                status=state.status.value,
                last_completed_sheet=state.last_completed_sheet,
                total_sheets=state.total_sheets,
            )
            return state
        except json.JSONDecodeError as e:
            # Corrupted state file - log and return None
            _logger.error(
                "checkpoint_corruption_detected",
                job_id=job_id,
                path=str(state_file),
                error_type="json_decode",
                error=str(e),
            )
            return None
        except ValueError as e:
            # Validation error - log and return None
            _logger.error(
                "checkpoint_corruption_detected",
                job_id=job_id,
                path=str(state_file),
                error_type="validation",
                error=str(e),
            )
            return None

    async def save(self, state: CheckpointState) -> None:
        """Save state to JSON file."""
        state.updated_at = _utc_now()
        state_file = self._get_state_file(state.job_id)

        # Write atomically using temp file + rename
        temp_file = state_file.with_suffix(".json.tmp")
        with open(temp_file, "w") as f:
            json.dump(
                state.model_dump(mode="json"),
                f,
                indent=2,
                default=str,  # Handle datetime serialization
            )
        temp_file.rename(state_file)

        _logger.info(
            "checkpoint_saved",
            job_id=state.job_id,
            status=state.status.value,
            last_completed_sheet=state.last_completed_sheet,
            total_sheets=state.total_sheets,
            path=str(state_file),
        )

    async def delete(self, job_id: str) -> bool:
        """Delete state file."""
        state_file = self._get_state_file(job_id)
        if state_file.exists():
            state_file.unlink()
            return True
        return False

    async def list_jobs(self) -> list[CheckpointState]:
        """List all jobs with state files."""
        states = []
        for state_file in self.state_dir.glob("*.json"):
            if state_file.suffix == ".json" and not state_file.name.endswith(".tmp"):
                try:
                    with open(state_file) as f:
                        data = json.load(f)
                    states.append(CheckpointState.model_validate(data))
                except (json.JSONDecodeError, ValueError):
                    continue
        return sorted(states, key=lambda s: s.updated_at, reverse=True)

    async def get_next_sheet(self, job_id: str) -> int | None:
        """Get next sheet from state."""
        state = await self.load(job_id)
        if state is None:
            return 1  # Start from beginning if no state
        return state.get_next_sheet()

    async def mark_sheet_status(
        self,
        job_id: str,
        sheet_num: int,
        status: SheetStatus,
        error_message: str | None = None,
    ) -> None:
        """Update sheet status in state."""
        state = await self.load(job_id)
        if state is None:
            raise ValueError(f"No state found for job {job_id}")

        if status == SheetStatus.COMPLETED:
            state.mark_sheet_completed(sheet_num)
        elif status == SheetStatus.FAILED:
            state.mark_sheet_failed(sheet_num, error_message or "Unknown error")
        elif status == SheetStatus.IN_PROGRESS:
            state.mark_sheet_started(sheet_num)

        await self.save(state)
