"""Validation framework for sheet outputs.

Re-exports all public names from the subpackage modules so that
existing imports like ``from mozart.execution.validation import X``
continue to work after the monolith split.
"""

from mozart.execution.validation.engine import ValidationEngine
from mozart.execution.validation.history import FailureHistoryStore, HistoricalFailure
from mozart.execution.validation.models import (
    FileModificationTracker,
    SheetValidationResult,
    ValidationResult,
)
from mozart.execution.validation.semantic import (
    KeyVariable,
    KeyVariableExtractor,
    SemanticConsistencyChecker,
    SemanticConsistencyResult,
    SemanticInconsistency,
)

__all__ = [
    "FailureHistoryStore",
    "FileModificationTracker",
    "HistoricalFailure",
    "KeyVariable",
    "KeyVariableExtractor",
    "SemanticConsistencyChecker",
    "SemanticConsistencyResult",
    "SemanticInconsistency",
    "SheetValidationResult",
    "ValidationEngine",
    "ValidationResult",
]
