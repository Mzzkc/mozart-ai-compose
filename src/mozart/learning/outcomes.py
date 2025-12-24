"""Outcome recording and pattern detection for learning."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mozart.core.checkpoint import BatchStatus


@dataclass
class BatchOutcome:
    """Structured outcome data from a completed batch execution.

    This dataclass captures all relevant information about a batch execution
    for learning and pattern detection purposes.
    """

    batch_id: str
    job_id: str
    validation_results: list[dict[str, Any]]  # Serialized ValidationResult data
    execution_duration: float
    retry_count: int
    completion_mode_used: bool
    final_status: BatchStatus
    validation_pass_rate: float
    first_attempt_success: bool
    patterns_detected: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@runtime_checkable
class OutcomeStore(Protocol):
    """Protocol for outcome storage backends.

    Provides async methods for recording outcomes and querying
    for similar past outcomes to inform execution decisions.
    """

    async def record(self, outcome: BatchOutcome) -> None:
        """Record a batch outcome for future learning."""
        ...

    async def query_similar(
        self, context: dict[str, Any], limit: int = 10
    ) -> list[BatchOutcome]:
        """Query for similar past outcomes based on context."""
        ...

    async def get_patterns(self, job_name: str) -> list[str]:
        """Get detected patterns for a specific job."""
        ...


class JsonOutcomeStore:
    """JSON-file based outcome store implementation.

    Stores outcomes in a JSON file with atomic saves,
    following the same pattern as JsonStateBackend.
    """

    def __init__(self, store_path: Path) -> None:
        """Initialize the JSON outcome store.

        Args:
            store_path: Path to the JSON file for storing outcomes.
        """
        self.store_path = store_path
        self._outcomes: list[BatchOutcome] = []

    async def record(self, outcome: BatchOutcome) -> None:
        """Record a batch outcome to the store.

        Args:
            outcome: The batch outcome to record.
        """
        self._outcomes.append(outcome)
        await self._save()

    async def query_similar(
        self, context: dict[str, Any], limit: int = 10
    ) -> list[BatchOutcome]:
        """Query for similar past outcomes.

        Currently returns recent outcomes for the same job_id.
        Future: implement semantic similarity matching.

        Args:
            context: Context dict containing job_id and other metadata.
            limit: Maximum number of outcomes to return.

        Returns:
            List of similar batch outcomes.
        """
        await self._load()
        job_id = context.get("job_id")
        if not job_id:
            return self._outcomes[-limit:]

        matching = [o for o in self._outcomes if o.job_id == job_id]
        return matching[-limit:]

    async def get_patterns(self, job_name: str) -> list[str]:
        """Get detected patterns for a job.

        Args:
            job_name: The job name to get patterns for.

        Returns:
            List of pattern strings detected across outcomes.
        """
        await self._load()
        patterns: set[str] = set()
        for outcome in self._outcomes:
            if outcome.job_id == job_name:
                patterns.update(outcome.patterns_detected)
        return list(patterns)

    async def _save(self) -> None:
        """Save outcomes to JSON file with atomic write."""
        import json
        import tempfile

        data = {
            "outcomes": [
                {
                    "batch_id": o.batch_id,
                    "job_id": o.job_id,
                    "validation_results": o.validation_results,
                    "execution_duration": o.execution_duration,
                    "retry_count": o.retry_count,
                    "completion_mode_used": o.completion_mode_used,
                    "final_status": o.final_status.value,
                    "validation_pass_rate": o.validation_pass_rate,
                    "first_attempt_success": o.first_attempt_success,
                    "patterns_detected": o.patterns_detected,
                    "timestamp": o.timestamp.isoformat(),
                }
                for o in self._outcomes
            ]
        }

        # Atomic write: write to temp file, then rename
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.store_path.parent,
            suffix=".tmp",
            delete=False,
        ) as f:
            json.dump(data, f, indent=2)
            temp_path = Path(f.name)

        temp_path.rename(self.store_path)

    async def _load(self) -> None:
        """Load outcomes from JSON file."""
        import json

        if not self.store_path.exists():
            return

        with open(self.store_path) as f:
            data = json.load(f)

        self._outcomes = [
            BatchOutcome(
                batch_id=o["batch_id"],
                job_id=o["job_id"],
                validation_results=o["validation_results"],
                execution_duration=o["execution_duration"],
                retry_count=o["retry_count"],
                completion_mode_used=o["completion_mode_used"],
                final_status=BatchStatus(o["final_status"]),
                validation_pass_rate=o["validation_pass_rate"],
                first_attempt_success=o["first_attempt_success"],
                patterns_detected=o.get("patterns_detected", []),
                timestamp=datetime.fromisoformat(o["timestamp"]),
            )
            for o in data.get("outcomes", [])
        ]
