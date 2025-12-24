"""Abstract base for state backends."""

from abc import ABC, abstractmethod

from mozart.core.checkpoint import BatchStatus, CheckpointState


class StateBackend(ABC):
    """Abstract base class for state storage backends.

    Implementations handle persistence of job state for resumable execution.
    """

    @abstractmethod
    async def load(self, job_id: str) -> CheckpointState | None:
        """Load state for a job.

        Args:
            job_id: Unique job identifier

        Returns:
            CheckpointState if found, None otherwise
        """
        ...

    @abstractmethod
    async def save(self, state: CheckpointState) -> None:
        """Save job state.

        Args:
            state: Checkpoint state to persist
        """
        ...

    @abstractmethod
    async def delete(self, job_id: str) -> bool:
        """Delete state for a job.

        Args:
            job_id: Unique job identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    async def list_jobs(self) -> list[CheckpointState]:
        """List all jobs with state.

        Returns:
            List of all checkpoint states
        """
        ...

    @abstractmethod
    async def get_next_batch(self, job_id: str) -> int | None:
        """Get the next batch to process for a job.

        Args:
            job_id: Unique job identifier

        Returns:
            Next batch number, or None if complete
        """
        ...

    @abstractmethod
    async def mark_batch_status(
        self,
        job_id: str,
        batch_num: int,
        status: BatchStatus,
        error_message: str | None = None,
    ) -> None:
        """Update status of a specific batch.

        Args:
            job_id: Unique job identifier
            batch_num: Batch number to update
            status: New status
            error_message: Optional error message for failures
        """
        ...

    async def infer_state_from_artifacts(
        self,
        job_id: str,
        workspace: str,
        artifact_pattern: str,
    ) -> int | None:
        """Infer last completed batch from artifact files.

        Fallback when state file is missing - checks for output files.
        Based on the fallback logic in run-batch-review.sh.

        Args:
            job_id: Unique job identifier
            workspace: Workspace directory path
            artifact_pattern: Glob pattern for batch artifacts (e.g., "batch*-security-report.md")

        Returns:
            Inferred last completed batch number, or None
        """
        import re
        from pathlib import Path

        workspace_path = Path(workspace)
        if not workspace_path.exists():
            return None

        # Find all matching artifact files
        artifacts = list(workspace_path.glob(artifact_pattern))
        if not artifacts:
            return None

        # Extract batch numbers and find max
        batch_nums = []
        for artifact in artifacts:
            match = re.search(r"batch(\d+)", artifact.name)
            if match:
                batch_nums.append(int(match.group(1)))

        return max(batch_nums) if batch_nums else None
