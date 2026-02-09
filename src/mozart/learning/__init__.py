"""Learning module for outcome recording, pattern detection, and judgment.

This module provides:
- Outcome recording and pattern detection (local workspace)
- Global learning store for cross-workspace pattern persistence
- Pattern weighting with recency/effectiveness decay
- Pattern aggregation for merging outcomes into global patterns
- Error learning hooks for adaptive wait times
- Migration from workspace-local to global store
- Judgment client for LLM-based decision making
"""

from mozart.learning.aggregator import (
    AggregationResult,
    PatternAggregator,
    aggregate_job_outcomes,
)
from mozart.learning.error_hooks import (
    ErrorLearningConfig,
    ErrorLearningContext,
    ErrorLearningHooks,
    record_error_recovery,
)
from mozart.learning.global_store import (
    DEFAULT_GLOBAL_STORE_PATH,
    ErrorRecoveryRecord,
    ExecutionRecord,
    GlobalLearningStore,
    PatternRecord,
    get_global_store,
)
from mozart.learning.judgment import (
    JudgmentClient,
    JudgmentQuery,
    JudgmentResponse,
    LocalJudgmentClient,
)
from mozart.learning.migration import (
    MigrationResult,
    OutcomeMigrator,
    check_migration_status,
    migrate_existing_outcomes,
)
from mozart.learning.outcomes import JsonOutcomeStore, OutcomeStore, SheetOutcome
from mozart.learning.patterns import (
    DetectedPattern,
    PatternApplicator,
    PatternDetector,
    PatternMatcher,
    PatternType,
)
from mozart.learning.weighter import (
    PatternWeighter,
    calculate_priority,
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
    # Global Store
    "GlobalLearningStore",
    "get_global_store",
    "DEFAULT_GLOBAL_STORE_PATH",
    "ExecutionRecord",
    "PatternRecord",
    "ErrorRecoveryRecord",
    # Aggregation
    "PatternAggregator",
    "AggregationResult",
    "aggregate_job_outcomes",
    # Weighting
    "PatternWeighter",
    "calculate_priority",
    # Error Learning Hooks
    "ErrorLearningConfig",
    "ErrorLearningContext",
    "ErrorLearningHooks",
    "record_error_recovery",
    # Migration
    "MigrationResult",
    "OutcomeMigrator",
    "check_migration_status",
    "migrate_existing_outcomes",
    # Judgment
    "JudgmentQuery",
    "JudgmentResponse",
    "JudgmentClient",
    "LocalJudgmentClient",
]
