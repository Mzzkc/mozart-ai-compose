"""JSON file-based state backend.

Simple state storage using JSON files, similar to the .review-state
approach in the original bash script.
"""

import json
from datetime import datetime
from pathlib import Path

from mozart.core.checkpoint import BatchStatus, CheckpointState
from mozart.state.base import StateBackend


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
        """Load state from JSON file."""
        state_file = self._get_state_file(job_id)
        if not state_file.exists():
            return None

        try:
            with open(state_file) as f:
                data = json.load(f)
            return CheckpointState.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            # Corrupted state file - log and return None
            import logging
            logging.warning(f"Failed to load state from {state_file}: {e}")
            return None

    async def save(self, state: CheckpointState) -> None:
        """Save state to JSON file."""
        state.updated_at = datetime.utcnow()
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

    async def get_next_batch(self, job_id: str) -> int | None:
        """Get next batch from state."""
        state = await self.load(job_id)
        if state is None:
            return 1  # Start from beginning if no state
        return state.get_next_batch()

    async def mark_batch_status(
        self,
        job_id: str,
        batch_num: int,
        status: BatchStatus,
        error_message: str | None = None,
    ) -> None:
        """Update batch status in state."""
        state = await self.load(job_id)
        if state is None:
            raise ValueError(f"No state found for job {job_id}")

        if status == BatchStatus.COMPLETED:
            state.mark_batch_completed(batch_num)
        elif status == BatchStatus.FAILED:
            state.mark_batch_failed(batch_num, error_message or "Unknown error")
        elif status == BatchStatus.IN_PROGRESS:
            state.mark_batch_started(batch_num)

        await self.save(state)
