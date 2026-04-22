"""Execution layer for Marianne jobs.

Contains validation primitives and shared types used by the baton-based
execution engine in ``daemon/baton/``. The legacy retry/circuit-breaker/
runner modules have been removed (Pre-Instrument Execution Atlas,
Phase 6) — their responsibilities now live inline in the baton.
"""

from marianne.core.errors.exceptions import FatalError
from marianne.core.summary import SheetExecutionMode
from marianne.execution.retry_strategy import (
    AdaptiveRetryStrategy,
    ErrorRecord,
    RetryPattern,
    RetryRecommendation,
    RetryStrategyConfig,
)
from marianne.execution.validation import (
    FileModificationTracker,
    SheetValidationResult,
    ValidationEngine,
    ValidationResult,
)

__all__ = [
    "AdaptiveRetryStrategy",
    "SheetExecutionMode",
    "SheetValidationResult",
    "ErrorRecord",
    "FatalError",
    "FileModificationTracker",
    "RetryPattern",
    "RetryRecommendation",
    "RetryStrategyConfig",
    "ValidationEngine",
    "ValidationResult",
]
