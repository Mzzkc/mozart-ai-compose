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
    """

    strategy: SynthesisStrategy = SynthesisStrategy.MERGE
    include_metadata: bool = True
    max_content_bytes: int = 1024 * 1024  # 1MB default
    fail_on_partial: bool = False


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

    @property
    def is_complete(self) -> bool:
        """True if synthesis has completed (success or failure)."""
        return self.status in ("done", "failed")

    @property
    def is_success(self) -> bool:
        """True if synthesis completed successfully."""
        return self.status == "done"

    def to_dict(self) -> dict[str, Any]:
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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SynthesisResult":
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

        result.status = "ready"
        result.metadata = {
            "batch_size": len(batch_sheets),
            "completed_count": len(completed_sheets),
            "failed_count": len(failed_sheets),
            "outputs_captured": len(result.sheet_outputs),
        }

        self._logger.info(
            "synthesizer.prepared",
            batch_id=batch_id,
            sheets=batch_sheets,
            outputs_captured=len(result.sheet_outputs),
        )

        return result

    def execute_synthesis(self, result: SynthesisResult) -> SynthesisResult:
        """Execute synthesis on prepared result.

        Combines sheet outputs according to the configured strategy.

        Args:
            result: SynthesisResult with status="ready".

        Returns:
            Updated SynthesisResult with synthesis complete.
        """
        if result.status != "ready":
            self._logger.warning(
                "synthesizer.invalid_state",
                batch_id=result.batch_id,
                status=result.status,
            )
            return result

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
