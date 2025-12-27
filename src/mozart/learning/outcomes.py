"""Outcome recording and pattern detection for learning."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mozart.core.checkpoint import SheetStatus

if TYPE_CHECKING:
    from mozart.learning.patterns import DetectedPattern


@dataclass
class SheetOutcome:
    """Structured outcome data from a completed sheet execution.

    This dataclass captures all relevant information about a sheet execution
    for learning and pattern detection purposes.
    """

    sheet_id: str
    job_id: str
    validation_results: list[dict[str, Any]]  # Serialized ValidationResult data
    execution_duration: float
    retry_count: int
    completion_mode_used: bool
    final_status: SheetStatus
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

    async def record(self, outcome: SheetOutcome) -> None:
        """Record a sheet outcome for future learning."""
        ...

    async def query_similar(
        self, context: dict[str, Any], limit: int = 10
    ) -> list[SheetOutcome]:
        """Query for similar past outcomes based on context."""
        ...

    async def get_patterns(self, job_name: str) -> list[str]:
        """Get detected patterns for a specific job."""
        ...

    async def get_relevant_patterns(
        self, context: dict[str, Any], limit: int = 5
    ) -> list[str]:
        """Get pattern descriptions relevant to the given context."""
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
        self._outcomes: list[SheetOutcome] = []

    async def record(self, outcome: SheetOutcome) -> None:
        """Record a sheet outcome to the store.

        After recording, if there are enough outcomes (>= 5), pattern
        detection is run and patterns_detected is populated on the outcome.

        Args:
            outcome: The sheet outcome to record.
        """
        self._outcomes.append(outcome)

        # Detect patterns after accumulating enough data
        if len(self._outcomes) >= 5:
            patterns = await self.detect_patterns()
            # Populate patterns_detected with top pattern descriptions
            outcome.patterns_detected = [
                p.to_prompt_guidance() for p in patterns[:3]
            ]

        await self._save()

    async def query_similar(
        self, context: dict[str, Any], limit: int = 10
    ) -> list[SheetOutcome]:
        """Query for similar past outcomes.

        Currently returns recent outcomes for the same job_id.
        Future: implement semantic similarity matching.

        Args:
            context: Context dict containing job_id and other metadata.
            limit: Maximum number of outcomes to return.

        Returns:
            List of similar sheet outcomes.
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

    async def detect_patterns(self) -> list["DetectedPattern"]:
        """Detect patterns from all recorded outcomes.

        Uses PatternDetector to analyze historical outcomes and
        identify recurring patterns that can inform future executions.

        Returns:
            List of DetectedPattern objects sorted by confidence.
        """
        from mozart.learning.patterns import PatternDetector

        await self._load()
        if not self._outcomes:
            return []

        detector = PatternDetector(self._outcomes)
        return detector.detect_all()

    async def get_relevant_patterns(
        self,
        context: dict[str, Any],
        limit: int = 5,
    ) -> list[str]:
        """Get pattern descriptions relevant to the given context.

        This method detects patterns, matches them to the context,
        and returns human-readable descriptions suitable for prompt injection.

        Args:
            context: Context dict containing job_id, sheet_num, validation_types, etc.
            limit: Maximum number of patterns to return.

        Returns:
            List of pattern description strings for prompt injection.
        """
        from mozart.learning.patterns import PatternApplicator, PatternMatcher

        # Detect all patterns
        all_patterns = await self.detect_patterns()
        if not all_patterns:
            return []

        # Match patterns to context
        matcher = PatternMatcher(all_patterns)
        matched = matcher.match(context, limit=limit)

        # Convert to prompt-ready descriptions
        applicator = PatternApplicator(matched)
        return applicator.get_pattern_descriptions()

    async def _save(self) -> None:
        """Save outcomes to JSON file with atomic write."""
        import json
        import tempfile

        data = {
            "outcomes": [
                {
                    "sheet_id": o.sheet_id,
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
            SheetOutcome(
                sheet_id=o["sheet_id"],
                job_id=o["job_id"],
                validation_results=o["validation_results"],
                execution_duration=o["execution_duration"],
                retry_count=o["retry_count"],
                completion_mode_used=o["completion_mode_used"],
                final_status=SheetStatus(o["final_status"]),
                validation_pass_rate=o["validation_pass_rate"],
                first_attempt_success=o["first_attempt_success"],
                patterns_detected=o.get("patterns_detected", []),
                timestamp=datetime.fromisoformat(o["timestamp"]),
            )
            for o in data.get("outcomes", [])
        ]
