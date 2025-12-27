"""Learning module for outcome recording, pattern detection, and judgment."""

from mozart.learning.judgment import (
    JudgmentClient,
    JudgmentProvider,
    JudgmentQuery,
    JudgmentResponse,
    LocalJudgmentClient,
)
from mozart.learning.outcomes import JsonOutcomeStore, OutcomeStore, SheetOutcome
from mozart.learning.patterns import (
    DetectedPattern,
    PatternApplicator,
    PatternDetector,
    PatternMatcher,
    PatternType,
)

__all__ = [
    # Outcomes
    "SheetOutcome",
    "OutcomeStore",
    "JsonOutcomeStore",
    # Patterns
    "PatternType",
    "DetectedPattern",
    "PatternDetector",
    "PatternMatcher",
    "PatternApplicator",
    # Judgment
    "JudgmentQuery",
    "JudgmentResponse",
    "JudgmentProvider",
    "JudgmentClient",
    "LocalJudgmentClient",
]
