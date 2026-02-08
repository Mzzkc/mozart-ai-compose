"""Result Synthesizer for parallel sheet execution (v18 evolution).

This module implements the "gather" phase of fan-out/gather pattern for
parallel execution. When sheets run in parallel, their content outputs
need to be synthesized into a unified result for downstream consumers.

Key features:
- Content reference extraction from parallel sheet outputs
- Synthesis strategies (merge, summarize, pass_through)
- Persistent synthesis state for checkpointing
- Integration with ParallelBatchResult

The ResultSynthesizer works with the existing ParallelExecutor to combine
outputs after batch completion.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from mozart.core.checkpoint import SynthesisResultDict

from mozart.core.logging import get_logger
from mozart.utils.time import utc_now

# Module logger
_logger = get_logger("synthesizer")


class SynthesisStrategy(str, Enum):
    """Strategy for synthesizing parallel sheet outputs.

    Attributes:
        MERGE: Combine all outputs into a single result (default).
        SUMMARIZE: Create a summary of all outputs.
        PASS_THROUGH: No synthesis, keep individual outputs separate.
    """

    MERGE = "merge"
    SUMMARIZE = "summarize"
    PASS_THROUGH = "pass_through"


@dataclass
class SynthesisConfig:
    """Configuration for result synthesis.

    Attributes:
        strategy: How to combine parallel outputs.
        include_metadata: Whether to include synthesis metadata.
        max_content_bytes: Maximum content size to synthesize (prevents OOM).
        fail_on_partial: If True, fail synthesis when some sheets failed.
        detect_conflicts: If True, run conflict detection before synthesis.
        conflict_key_filter: If provided, only check these keys for conflicts.
        fail_on_conflict: If True, fail synthesis when conflicts detected.
    """

    strategy: SynthesisStrategy = SynthesisStrategy.MERGE
    include_metadata: bool = True
    max_content_bytes: int = 1024 * 1024  # 1MB default
    fail_on_partial: bool = False
    detect_conflicts: bool = False
    conflict_key_filter: list[str] | None = None
    fail_on_conflict: bool = False


@dataclass
class SynthesisResult:
    """Result of synthesizing parallel sheet outputs.

    Tracks the synthesis state for a single batch of parallel sheets.
    This is persisted in checkpoint state for resume capability.

    Attributes:
        batch_id: Unique identifier for this synthesis batch.
        sheets: Sheet numbers included in this synthesis.
        strategy: Strategy used for synthesis.
        status: Current synthesis status.
        created_at: When synthesis started.
        completed_at: When synthesis completed (None if not complete).
        sheet_outputs: Map of sheet_num -> output reference (path or content).
        synthesized_content: The final synthesized result (if complete).
        error_message: Error description if synthesis failed.
        metadata: Additional metadata about the synthesis.
    """

    batch_id: str
    sheets: list[int] = field(default_factory=list)
    strategy: SynthesisStrategy = SynthesisStrategy.MERGE
    status: str = "pending"  # pending, ready, done, failed
    created_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    sheet_outputs: dict[int, str] = field(default_factory=dict)
    synthesized_content: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    conflict_detection: dict[str, Any] | None = None
    """Conflict detection result if conflict detection was enabled."""

    @property
    def is_complete(self) -> bool:
        """True if synthesis has completed (success or failure)."""
        return self.status in ("done", "failed")

    @property
    def is_success(self) -> bool:
        """True if synthesis completed successfully."""
        return self.status == "done"

    def to_dict(self) -> SynthesisResultDict:
        """Serialize to dictionary for persistence."""
        return {
            "batch_id": self.batch_id,
            "sheets": self.sheets,
            "strategy": self.strategy.value,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "sheet_outputs": self.sheet_outputs,
            "synthesized_content": self.synthesized_content,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "conflict_detection": self.conflict_detection,
        }  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, data: SynthesisResultDict) -> "SynthesisResult":
        """Deserialize from dictionary."""
        created_at = data.get("created_at")
        completed_at = data.get("completed_at")

        return cls(
            batch_id=data["batch_id"],
            sheets=data.get("sheets", []),
            strategy=SynthesisStrategy(data.get("strategy", "merge")),
            status=data.get("status", "pending"),
            created_at=datetime.fromisoformat(created_at) if created_at else utc_now(),
            completed_at=datetime.fromisoformat(completed_at) if completed_at else None,
            sheet_outputs=data.get("sheet_outputs", {}),
            synthesized_content=data.get("synthesized_content"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
            conflict_detection=data.get("conflict_detection"),
        )


class ResultSynthesizer:
    """Synthesizes outputs from parallel sheet executions.

    The synthesizer works with ParallelBatchResult to combine content
    outputs after a batch of parallel sheets completes.

    Example:
        ```python
        synthesizer = ResultSynthesizer(config)
        result = synthesizer.prepare_synthesis(batch_result, state)
        if result.status == "ready":
            result = synthesizer.execute_synthesis(result)
        ```

    Attributes:
        config: Synthesis configuration.
    """

    def __init__(self, config: SynthesisConfig | None = None):
        """Initialize the synthesizer.

        Args:
            config: Synthesis configuration. Uses defaults if None.
        """
        self.config = config or SynthesisConfig()
        self._logger = _logger

    def prepare_synthesis(
        self,
        batch_sheets: list[int],
        completed_sheets: list[int],
        failed_sheets: list[int],
        sheet_outputs: dict[int, str],
    ) -> SynthesisResult:
        """Prepare synthesis from batch results.

        Creates a SynthesisResult ready for execution. Does not actually
        run the synthesis - call execute_synthesis() for that.

        Args:
            batch_sheets: All sheet numbers in the batch.
            completed_sheets: Sheets that completed successfully.
            failed_sheets: Sheets that failed.
            sheet_outputs: Map of sheet_num -> output content/reference.

        Returns:
            SynthesisResult ready for synthesis (status="ready") or
            indicating failure (status="failed") if validation fails.
        """
        import uuid

        batch_id = f"batch_{uuid.uuid4().hex[:8]}"

        result = SynthesisResult(
            batch_id=batch_id,
            sheets=batch_sheets,
            strategy=self.config.strategy,
        )

        # Record outputs for completed sheets
        for sheet_num in completed_sheets:
            if sheet_num in sheet_outputs:
                result.sheet_outputs[sheet_num] = sheet_outputs[sheet_num]

        # Check if we can proceed with synthesis
        if failed_sheets and self.config.fail_on_partial:
            result.status = "failed"
            result.error_message = f"Synthesis requires all sheets: {len(failed_sheets)} failed"
            self._logger.warning(
                "synthesizer.partial_failure",
                batch_id=batch_id,
                failed_sheets=failed_sheets,
            )
            return result

        if not result.sheet_outputs:
            result.status = "failed"
            result.error_message = "No sheet outputs to synthesize"
            self._logger.warning(
                "synthesizer.no_outputs",
                batch_id=batch_id,
            )
            return result

        # Run conflict detection if enabled
        if self.config.detect_conflicts and len(result.sheet_outputs) >= 2:
            conflict_result = self._detect_conflicts(result.sheet_outputs)
            result.conflict_detection = conflict_result.to_dict()

            if conflict_result.has_conflicts:
                self._logger.warning(
                    "synthesizer.conflicts_detected",
                    batch_id=batch_id,
                    conflict_count=len(conflict_result.conflicts),
                    error_count=conflict_result.error_count,
                )

                if self.config.fail_on_conflict:
                    result.status = "failed"
                    result.error_message = (
                        f"Conflict detection found {len(conflict_result.conflicts)} "
                        f"conflicts ({conflict_result.error_count} errors)"
                    )
                    return result

        result.status = "ready"
        result.metadata = {
            "batch_size": len(batch_sheets),
            "completed_count": len(completed_sheets),
            "failed_count": len(failed_sheets),
            "outputs_captured": len(result.sheet_outputs),
            "conflict_detection_enabled": self.config.detect_conflicts,
        }

        self._logger.info(
            "synthesizer.prepared",
            batch_id=batch_id,
            sheets=batch_sheets,
            outputs_captured=len(result.sheet_outputs),
        )

        return result

    def _detect_conflicts(
        self,
        sheet_outputs: dict[int, str],
    ) -> "ConflictDetectionResult":
        """Run conflict detection on sheet outputs.

        Args:
            sheet_outputs: Map of sheet_num -> output content.

        Returns:
            ConflictDetectionResult with any conflicts found.
        """
        # Import here to use the ConflictDetector defined later
        detector = ConflictDetector(
            key_filter=self.config.conflict_key_filter,
            strict_mode=self.config.fail_on_conflict,
        )
        return detector.detect_conflicts(sheet_outputs)

    def execute_synthesis(self, result: SynthesisResult) -> SynthesisResult:
        """Execute synthesis on prepared result.

        Combines sheet outputs according to the configured strategy.

        Args:
            result: SynthesisResult with status="ready".

        Returns:
            Updated SynthesisResult with synthesis complete.
        """
        if result.status != "ready":
            msg = (
                f"execute_synthesis() called with status='{result.status}' "
                f"(expected 'ready') for batch '{result.batch_id}'"
            )
            self._logger.error(
                "synthesizer.invalid_state",
                batch_id=result.batch_id,
                status=result.status,
            )
            raise ValueError(msg)

        try:
            if result.strategy == SynthesisStrategy.MERGE:
                synthesized = self._merge_outputs(result.sheet_outputs)
            elif result.strategy == SynthesisStrategy.SUMMARIZE:
                synthesized = self._summarize_outputs(result.sheet_outputs)
            else:  # PASS_THROUGH
                synthesized = self._pass_through_outputs(result.sheet_outputs)

            # Check size limit
            if len(synthesized.encode("utf-8")) > self.config.max_content_bytes:
                result.status = "failed"
                result.error_message = (
                    f"Synthesized content exceeds limit: "
                    f"{len(synthesized.encode('utf-8'))} > {self.config.max_content_bytes}"
                )
                self._logger.warning(
                    "synthesizer.size_exceeded",
                    batch_id=result.batch_id,
                    size=len(synthesized.encode("utf-8")),
                    limit=self.config.max_content_bytes,
                )
                return result

            result.synthesized_content = synthesized
            result.status = "done"
            result.completed_at = utc_now()

            self._logger.info(
                "synthesizer.complete",
                batch_id=result.batch_id,
                strategy=result.strategy.value,
                content_size=len(synthesized),
            )

        except Exception as e:
            result.status = "failed"
            result.error_message = f"Synthesis failed: {e}"
            self._logger.error(
                "synthesizer.error",
                batch_id=result.batch_id,
                error=str(e),
            )

        return result

    def _merge_outputs(self, outputs: dict[int, str]) -> str:
        """Merge outputs by concatenating in sheet order.

        Args:
            outputs: Map of sheet_num -> content.

        Returns:
            Merged content with sheet separators.
        """
        parts: list[str] = []
        for sheet_num in sorted(outputs.keys()):
            content = outputs[sheet_num]
            if self.config.include_metadata:
                parts.append(f"--- Sheet {sheet_num} ---\n{content}")
            else:
                parts.append(content)

        separator = "\n\n" if self.config.include_metadata else "\n"
        return separator.join(parts)

    def _summarize_outputs(self, outputs: dict[int, str]) -> str:
        """Create a summary of outputs.

        For now, this creates a simple summary. Future versions could
        use AI to generate a proper summary.

        Args:
            outputs: Map of sheet_num -> content.

        Returns:
            Summary string.
        """
        lines = [
            f"Synthesis Summary ({len(outputs)} sheets):",
            "",
        ]

        for sheet_num in sorted(outputs.keys()):
            content = outputs[sheet_num]
            # First line or truncated preview
            preview = content.split("\n")[0][:100] if content else "(empty)"
            lines.append(f"  Sheet {sheet_num}: {preview}")

        lines.append("")
        lines.append(f"Total content size: {sum(len(c) for c in outputs.values())} chars")

        return "\n".join(lines)

    def _pass_through_outputs(self, outputs: dict[int, str]) -> str:
        """Pass through outputs as-is (JSON serialized).

        Args:
            outputs: Map of sheet_num -> content.

        Returns:
            JSON-serialized outputs.
        """
        import json

        return json.dumps(
            {str(k): v for k, v in outputs.items()},
            indent=2,
        )


def synthesize_batch(
    batch_sheets: list[int],
    completed_sheets: list[int],
    failed_sheets: list[int],
    sheet_outputs: dict[int, str],
    config: SynthesisConfig | None = None,
) -> SynthesisResult:
    """Convenience function to synthesize a batch in one call.

    Args:
        batch_sheets: All sheet numbers in the batch.
        completed_sheets: Sheets that completed successfully.
        failed_sheets: Sheets that failed.
        sheet_outputs: Map of sheet_num -> output content.
        config: Optional synthesis configuration.

    Returns:
        SynthesisResult with synthesis complete.
    """
    synthesizer = ResultSynthesizer(config)
    result = synthesizer.prepare_synthesis(
        batch_sheets, completed_sheets, failed_sheets, sheet_outputs
    )

    if result.status == "ready":
        result = synthesizer.execute_synthesis(result)

    return result


# =============================================================================
# Parallel Output Conflict Detection (v20 evolution)
# =============================================================================


@dataclass
class OutputConflict:
    """Represents a conflicting key-value pair between parallel sheet outputs.

    When parallel sheets produce outputs with the same key but different values,
    this represents a potential inconsistency that may need resolution before
    synthesis.

    Attributes:
        key: The key that has conflicting values.
        sheet_a: First sheet number in the conflict.
        value_a: Value from sheet A.
        sheet_b: Second sheet number in the conflict.
        value_b: Value from sheet B.
        severity: Conflict severity (warning, error).
    """

    key: str
    sheet_a: int
    value_a: str
    sheet_b: int
    value_b: str
    severity: str = "warning"

    def format_message(self) -> str:
        """Format as human-readable message."""
        return (
            f"Conflict on '{self.key}': "
            f"sheet {self.sheet_a}='{self.value_a}' vs "
            f"sheet {self.sheet_b}='{self.value_b}'"
        )


@dataclass
class ConflictDetectionResult:
    """Result of parallel output conflict detection.

    Tracks conflicts detected before synthesis merging.

    Attributes:
        sheets_analyzed: Sheets that were analyzed for conflicts.
        conflicts: List of detected conflicts.
        keys_checked: Total number of unique keys checked.
        checked_at: When the check was performed.
    """

    sheets_analyzed: list[int] = field(default_factory=list)
    conflicts: list[OutputConflict] = field(default_factory=list)
    keys_checked: int = 0
    checked_at: datetime = field(default_factory=utc_now)

    @property
    def has_conflicts(self) -> bool:
        """True if any conflicts were detected."""
        return len(self.conflicts) > 0

    @property
    def error_count(self) -> int:
        """Count of error-severity conflicts."""
        return sum(1 for c in self.conflicts if c.severity == "error")

    @property
    def warning_count(self) -> int:
        """Count of warning-severity conflicts."""
        return sum(1 for c in self.conflicts if c.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "sheets_analyzed": self.sheets_analyzed,
            "conflicts": [
                {
                    "key": c.key,
                    "sheet_a": c.sheet_a,
                    "value_a": c.value_a,
                    "sheet_b": c.sheet_b,
                    "value_b": c.value_b,
                    "severity": c.severity,
                }
                for c in self.conflicts
            ],
            "keys_checked": self.keys_checked,
            "checked_at": self.checked_at.isoformat(),
            "has_conflicts": self.has_conflicts,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


class ConflictDetector:
    """Detects conflicts in parallel sheet outputs before synthesis.

    Analyzes outputs from parallel sheets to identify cases where the same
    key has different values, which may indicate inconsistent results that
    need resolution before merging.

    Uses KeyVariableExtractor from validation module to parse key-value
    pairs from sheet outputs.

    Example:
        ```python
        detector = ConflictDetector()
        result = detector.detect_conflicts({
            1: "STATUS: complete\\nVERSION: 1.0",
            2: "STATUS: failed\\nVERSION: 1.0",  # STATUS conflicts!
        })
        if result.has_conflicts:
            for conflict in result.conflicts:
                print(conflict.format_message())
        ```
    """

    def __init__(
        self,
        key_filter: list[str] | None = None,
        strict_mode: bool = False,
    ) -> None:
        """Initialize detector.

        Args:
            key_filter: If provided, only check these keys for conflicts.
            strict_mode: If True, all conflicts are errors. If False, warnings.
        """
        # Import here to avoid circular import
        from mozart.execution.validation import KeyVariableExtractor

        self.extractor = KeyVariableExtractor(key_filter=key_filter)
        self.strict_mode = strict_mode
        self._logger = _logger

    def detect_conflicts(
        self,
        sheet_outputs: dict[int, str],
    ) -> ConflictDetectionResult:
        """Detect conflicts across parallel sheet outputs.

        Args:
            sheet_outputs: Map of sheet_num -> output content.

        Returns:
            ConflictDetectionResult with any conflicts found.
        """
        result = ConflictDetectionResult(
            sheets_analyzed=sorted(sheet_outputs.keys()),
        )

        if len(sheet_outputs) < 2:
            # Need at least 2 sheets to have conflicts
            return result

        # Extract key-value pairs from each sheet
        sheet_variables: dict[int, dict[str, str]] = {}
        for sheet_num, content in sheet_outputs.items():
            variables = self.extractor.extract(content)
            sheet_variables[sheet_num] = {v.key: v.value for v in variables}

        # Count total unique keys across all sheets
        all_keys: set[str] = set()
        for vars_dict in sheet_variables.values():
            all_keys.update(vars_dict.keys())
        result.keys_checked = len(all_keys)

        # Compare all pairs of sheets
        sheets_sorted = sorted(sheet_outputs.keys())
        for i, sheet_a in enumerate(sheets_sorted):
            for sheet_b in sheets_sorted[i + 1:]:
                self._compare_sheets(
                    sheet_a, sheet_variables.get(sheet_a, {}),
                    sheet_b, sheet_variables.get(sheet_b, {}),
                    result,
                )

        if result.has_conflicts:
            self._logger.warning(
                "conflict_detector.conflicts_found",
                sheets=result.sheets_analyzed,
                conflict_count=len(result.conflicts),
                error_count=result.error_count,
                warning_count=result.warning_count,
            )
        else:
            self._logger.debug(
                "conflict_detector.no_conflicts",
                sheets=result.sheets_analyzed,
                keys_checked=result.keys_checked,
            )

        return result

    def _compare_sheets(
        self,
        sheet_a: int,
        vars_a: dict[str, str],
        sheet_b: int,
        vars_b: dict[str, str],
        result: ConflictDetectionResult,
    ) -> None:
        """Compare variables between two sheets for conflicts."""
        # Find common keys
        common_keys = set(vars_a.keys()) & set(vars_b.keys())

        for key in common_keys:
            value_a = vars_a[key]
            value_b = vars_b[key]

            # Case-insensitive comparison for flexibility
            if value_a.lower() != value_b.lower():
                result.conflicts.append(OutputConflict(
                    key=key,
                    sheet_a=sheet_a,
                    value_a=value_a,
                    sheet_b=sheet_b,
                    value_b=value_b,
                    severity="error" if self.strict_mode else "warning",
                ))


def detect_parallel_conflicts(
    sheet_outputs: dict[int, str],
    key_filter: list[str] | None = None,
    strict_mode: bool = False,
) -> ConflictDetectionResult:
    """Convenience function to detect conflicts in one call.

    Args:
        sheet_outputs: Map of sheet_num -> output content.
        key_filter: If provided, only check these keys.
        strict_mode: If True, all conflicts are errors.

    Returns:
        ConflictDetectionResult with any conflicts found.
    """
    detector = ConflictDetector(key_filter=key_filter, strict_mode=strict_mode)
    return detector.detect_conflicts(sheet_outputs)
