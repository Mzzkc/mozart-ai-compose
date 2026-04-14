"""Job execution summary and completion types.

Contains types used across CLI, daemon, and execution layers
to represent job completion state. These types are the public contract between
the execution engine (baton/runner) and the rest of the system.

Canonical definitions:
- JobCompletionSummary: marianne.core.models
- FatalError, RateLimitExhaustedError, GracefulShutdownError: marianne.core.errors.exceptions
- GroundingDecisionContext, SheetExecutionMode: defined here
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marianne.execution.grounding import GroundingResult

# Re-export canonical types for backward compatibility
from marianne.core.errors.exceptions import (  # noqa: F401
    FatalError,
    GracefulShutdownError,
    RateLimitExhaustedError,
)
from marianne.core.models import JobCompletionSummary  # noqa: F401

# RunSummary is an alias for the Pydantic v2 JobCompletionSummary model.
# All existing code using RunSummary continues to work — constructors,
# field access, and property access are backward compatible.
RunSummary = JobCompletionSummary


@dataclass
class GroundingDecisionContext:
    """Context from grounding hooks for completion mode decisions.

    Encapsulates grounding results to inform decision-making about
    whether to retry, complete, or escalate.
    """

    passed: bool
    message: str
    confidence: float = 1.0
    should_escalate: bool = False
    recovery_guidance: str | None = None
    hooks_executed: int = 0

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))

    @classmethod
    def from_results(cls, results: list[GroundingResult]) -> GroundingDecisionContext:
        """Build context from grounding results list."""
        if not results:
            return cls(passed=True, message="No grounding hooks executed", hooks_executed=0)

        passed = all(r.passed for r in results)
        confidences = [r.confidence for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
        should_escalate = any(r.should_escalate for r in results)

        failed = [r for r in results if not r.passed]
        recovery_guidance = None
        if failed:
            guidance_parts = [r.recovery_guidance for r in failed if r.recovery_guidance]
            if guidance_parts:
                recovery_guidance = "; ".join(guidance_parts)

        if passed:
            message = f"All {len(results)} grounding check(s) passed"
        else:
            failures = ", ".join(f"{r.hook_name}: {r.message}" for r in failed)
            message = f"{len(failed)}/{len(results)} grounding check(s) failed: {failures}"

        return cls(
            passed=passed,
            message=message,
            confidence=avg_confidence,
            should_escalate=should_escalate,
            recovery_guidance=recovery_guidance,
            hooks_executed=len(results),
        )

    @classmethod
    def disabled(cls) -> GroundingDecisionContext:
        """Create context when grounding is disabled."""
        return cls(passed=True, message="Grounding not enabled", hooks_executed=0)


class SheetExecutionMode(str, Enum):
    """Mode of sheet execution."""

    NORMAL = "normal"
    COMPLETION = "completion"
    RETRY = "retry"
    ESCALATE = "escalate"
