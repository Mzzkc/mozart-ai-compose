"""Tests for global learning components.

Tests the Movement III/IV implementation:
- GlobalLearningStore (SQLite-based global pattern store)
- PatternWeighter (priority calculation with decay)
- PatternAggregator (pattern merging and conflict resolution)
"""

import tempfile
from collections.abc import Generator
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mozart.core.checkpoint import SheetStatus
from mozart.learning.aggregator import (
    AggregationResult,
    EnhancedAggregationResult,
    EnhancedPatternAggregator,
    PatternAggregator,
)
from mozart.learning.global_store import (
    GlobalLearningStore,
    PatternRecord,
)
from mozart.learning.outcomes import SheetOutcome
from mozart.learning.patterns import (
    DetectedPattern,
    ExtractedPattern,
    OutputPatternExtractor,
    PatternType,
)
from mozart.learning.weighter import PatternWeighter, WeightingConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path() -> Path:
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Path(f.name)


@pytest.fixture
def global_store(temp_db_path: Path) -> Generator[GlobalLearningStore, None, None]:
    """Create a GlobalLearningStore with a temporary database."""
    store = GlobalLearningStore(temp_db_path)
    yield store
    # Cleanup
    if temp_db_path.exists():
        temp_db_path.unlink()


@pytest.fixture
def sample_outcome() -> SheetOutcome:
    """Create a sample sheet outcome for testing."""
    return SheetOutcome(
        sheet_id="test-job-sheet1",
        job_id="test-job",
        validation_results=[
            {"rule_type": "file_exists", "passed": True, "confidence": 1.0},
            {"rule_type": "content_contains", "passed": True, "confidence": 0.9},
        ],
        execution_duration=30.0,
        retry_count=0,
        completion_mode_used=False,
        final_status=SheetStatus.COMPLETED,
        validation_pass_rate=1.0,
        first_attempt_success=True,
    )


@pytest.fixture
def sample_failed_outcome() -> SheetOutcome:
    """Create a sample failed outcome for testing."""
    return SheetOutcome(
        sheet_id="test-job-sheet2",
        job_id="test-job",
        validation_results=[
            {"rule_type": "file_exists", "passed": False, "confidence": 1.0},
        ],
        execution_duration=45.0,
        retry_count=2,
        completion_mode_used=True,
        final_status=SheetStatus.FAILED,
        validation_pass_rate=0.0,
        first_attempt_success=False,
    )


# =============================================================================
# TestGlobalLearningStore
# =============================================================================


class TestGlobalLearningStore:
    """Tests for GlobalLearningStore class."""

    def test_store_initialization(self, global_store: GlobalLearningStore) -> None:
        """Test that store initializes correctly."""
        assert global_store.db_path.exists()

    def test_record_outcome(
        self,
        global_store: GlobalLearningStore,
        sample_outcome: SheetOutcome,
    ) -> None:
        """Test recording an outcome to the store."""
        workspace_path = Path("/tmp/test-workspace")
        exec_id = global_store.record_outcome(
            outcome=sample_outcome,
            workspace_path=workspace_path,
            model="claude-sonnet-4",
        )

        assert exec_id is not None
        assert len(exec_id) == 36  # UUID length

    def test_record_pattern(self, global_store: GlobalLearningStore) -> None:
        """Test recording a pattern to the store."""
        pattern_id = global_store.record_pattern(
            pattern_type="validation_failure",
            pattern_name="file_exists:missing",
            description="File not created during execution",
            context_tags=["validation:file_exists"],
            suggested_action="Ensure file is created before completion",
        )

        assert pattern_id is not None
        assert len(pattern_id) == 16  # Truncated hash

    def test_pattern_merge_increments_count(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that recording same pattern increments occurrence count."""
        # Record same pattern twice
        global_store.record_pattern(
            pattern_type="validation_failure",
            pattern_name="file_exists:missing",
        )
        global_store.record_pattern(
            pattern_type="validation_failure",
            pattern_name="file_exists:missing",
        )

        patterns = global_store.get_patterns(min_priority=0.0)
        file_pattern = next(
            (p for p in patterns if p.pattern_name == "file_exists:missing"),
            None,
        )

        assert file_pattern is not None
        assert file_pattern.occurrence_count == 2

    def test_record_error_recovery(self, global_store: GlobalLearningStore) -> None:
        """Test recording an error recovery."""
        record_id = global_store.record_error_recovery(
            error_code="E103",
            suggested_wait=60.0,
            actual_wait=90.0,
            recovery_success=True,
            model="claude-sonnet-4",
        )

        assert record_id is not None

    def test_get_learned_wait_time_insufficient_samples(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that learned wait time returns None with insufficient samples."""
        # Record only 2 recoveries (min_samples=3)
        for _ in range(2):
            global_store.record_error_recovery(
                error_code="E103",
                suggested_wait=60.0,
                actual_wait=90.0,
                recovery_success=True,
            )

        wait_time = global_store.get_learned_wait_time("E103")
        assert wait_time is None

    def test_get_learned_wait_time_sufficient_samples(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that learned wait time is calculated with sufficient samples."""
        # Record 5 successful recoveries with varying wait times
        waits = [60.0, 90.0, 120.0, 80.0, 70.0]
        for wait in waits:
            global_store.record_error_recovery(
                error_code="E103",
                suggested_wait=60.0,
                actual_wait=wait,
                recovery_success=True,
            )

        wait_time = global_store.get_learned_wait_time("E103")
        assert wait_time is not None
        # Should be between min (60) and average (84)
        assert 60.0 <= wait_time <= 84.0

    def test_get_execution_stats(
        self,
        global_store: GlobalLearningStore,
        sample_outcome: SheetOutcome,
        sample_failed_outcome: SheetOutcome,
    ) -> None:
        """Test getting execution statistics."""
        workspace_path = Path("/tmp/test-workspace")
        global_store.record_outcome(sample_outcome, workspace_path)
        global_store.record_outcome(sample_failed_outcome, workspace_path)

        stats = global_store.get_execution_stats()

        assert stats["total_executions"] == 2
        assert stats["first_attempt_success_rate"] == 0.5
        assert stats["unique_workspaces"] == 1

    def test_workspace_hash_stability(self) -> None:
        """Test that workspace hash is stable for same path."""
        path = Path("/home/user/project/workspace")
        hash1 = GlobalLearningStore.hash_workspace(path)
        hash2 = GlobalLearningStore.hash_workspace(path)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_clear_all(
        self,
        global_store: GlobalLearningStore,
        sample_outcome: SheetOutcome,
    ) -> None:
        """Test clearing all data from the store."""
        workspace_path = Path("/tmp/test-workspace")
        global_store.record_outcome(sample_outcome, workspace_path)
        global_store.record_pattern("test", "test:pattern")

        global_store.clear_all()

        stats = global_store.get_execution_stats()
        assert stats["total_executions"] == 0
        assert stats["total_patterns"] == 0


# =============================================================================
# TestPatternWeighter
# =============================================================================


class TestPatternWeighter:
    """Tests for PatternWeighter class."""

    @pytest.fixture
    def weighter(self) -> PatternWeighter:
        """Create a default pattern weighter."""
        return PatternWeighter()

    def test_effectiveness_with_few_samples(self, weighter: PatternWeighter) -> None:
        """Test that effectiveness returns neutral prior with few samples."""
        effectiveness = weighter.calculate_effectiveness(
            success_count=1,
            failure_count=1,
        )

        # With < 3 samples, should return neutral 0.5
        assert effectiveness == 0.5

    def test_effectiveness_with_enough_samples(self, weighter: PatternWeighter) -> None:
        """Test effectiveness calculation with sufficient samples."""
        effectiveness = weighter.calculate_effectiveness(
            success_count=8,
            failure_count=2,
        )

        # (8 + 0.5) / (10 + 1) = 8.5 / 11 ≈ 0.77
        assert 0.75 <= effectiveness <= 0.80

    def test_recency_factor_recent(self, weighter: PatternWeighter) -> None:
        """Test recency factor for recent patterns."""
        now = datetime.now()
        last_confirmed = now - timedelta(days=1)

        recency = weighter.calculate_recency_factor(last_confirmed, now)

        # Should be very close to 1.0
        assert recency > 0.99

    def test_recency_factor_old(self, weighter: PatternWeighter) -> None:
        """Test recency factor for old patterns."""
        now = datetime.now()
        last_confirmed = now - timedelta(days=90)  # 3 months

        recency = weighter.calculate_recency_factor(last_confirmed, now)

        # Should be around 0.9^3 = 0.729
        assert 0.70 <= recency <= 0.75

    def test_frequency_factor(self, weighter: PatternWeighter) -> None:
        """Test frequency factor calculation."""
        assert weighter.calculate_frequency_factor(0) == 0.0
        assert 0.1 <= weighter.calculate_frequency_factor(1) <= 0.2
        assert 0.5 <= weighter.calculate_frequency_factor(10) <= 0.6
        assert weighter.calculate_frequency_factor(100) == 1.0

    def test_priority_calculation(self, weighter: PatternWeighter) -> None:
        """Test full priority calculation."""
        now = datetime.now()
        priority = weighter.calculate_priority(
            occurrence_count=10,
            led_to_success_count=8,
            led_to_failure_count=2,
            last_confirmed=now - timedelta(days=7),
            variance=0.1,
        )

        # Should be in reasonable range
        assert 0.0 <= priority <= 1.0

    def test_is_deprecated_insufficient_data(self, weighter: PatternWeighter) -> None:
        """Test that patterns with insufficient data are not deprecated."""
        is_deprecated = weighter.is_deprecated(
            led_to_success_count=0,
            led_to_failure_count=2,
        )

        # Not enough data to deprecate
        assert is_deprecated is False

    def test_is_deprecated_low_effectiveness(self, weighter: PatternWeighter) -> None:
        """Test that low effectiveness patterns are deprecated."""
        is_deprecated = weighter.is_deprecated(
            led_to_success_count=1,
            led_to_failure_count=9,  # 10% effectiveness
        )

        # Should be deprecated (below 0.3 threshold)
        assert is_deprecated is True

    def test_classify_uncertainty_epistemic(self, weighter: PatternWeighter) -> None:
        """Test epistemic uncertainty classification."""
        classification = weighter.classify_uncertainty(variance=0.2)
        assert classification == "epistemic"

    def test_classify_uncertainty_aleatoric(self, weighter: PatternWeighter) -> None:
        """Test aleatoric uncertainty classification."""
        classification = weighter.classify_uncertainty(variance=0.5)
        assert classification == "aleatoric"

    def test_calculate_variance(self, weighter: PatternWeighter) -> None:
        """Test variance calculation from outcomes."""
        # All same outcomes = 0 variance
        same = weighter.calculate_variance([True, True, True, True])
        assert same == 0.0

        # 50/50 split = max variance
        split = weighter.calculate_variance([True, False, True, False])
        assert 0.24 <= split <= 0.26

    def test_custom_config(self) -> None:
        """Test weighter with custom configuration."""
        config = WeightingConfig(
            decay_rate_per_month=0.2,  # 20% decay
            effectiveness_threshold=0.4,
        )
        weighter = PatternWeighter(config)

        now = datetime.now()
        last_confirmed = now - timedelta(days=30)

        recency = weighter.calculate_recency_factor(last_confirmed, now)

        # Should be around 0.8 (20% decay)
        assert 0.79 <= recency <= 0.81


# =============================================================================
# TestPatternAggregator
# =============================================================================


class TestPatternAggregator:
    """Tests for PatternAggregator class."""

    @pytest.fixture
    def aggregator(
        self, global_store: GlobalLearningStore
    ) -> PatternAggregator:
        """Create a pattern aggregator with test store."""
        return PatternAggregator(global_store)

    def test_aggregate_empty_outcomes(
        self, aggregator: PatternAggregator
    ) -> None:
        """Test aggregating empty outcomes list."""
        result = aggregator.aggregate_outcomes(
            outcomes=[],
            workspace_path=Path("/tmp/test"),
        )

        assert isinstance(result, AggregationResult)
        assert result.outcomes_recorded == 0
        assert result.patterns_detected == 0

    def test_aggregate_single_outcome(
        self,
        aggregator: PatternAggregator,
        sample_outcome: SheetOutcome,
    ) -> None:
        """Test aggregating a single outcome."""
        result = aggregator.aggregate_outcomes(
            outcomes=[sample_outcome],
            workspace_path=Path("/tmp/test"),
            model="claude-sonnet-4",
        )

        assert result.outcomes_recorded == 1
        # Single outcome may not produce patterns (need >= 2 for most patterns)
        assert result.patterns_detected >= 0

    def test_aggregate_multiple_outcomes_detects_patterns(
        self,
        aggregator: PatternAggregator,
    ) -> None:
        """Test that aggregating multiple outcomes detects patterns."""
        # Create outcomes with recurring validation failures
        outcomes = []
        for i in range(5):
            outcomes.append(
                SheetOutcome(
                    sheet_id=f"test-job-sheet{i}",
                    job_id="test-job",
                    validation_results=[
                        {"rule_type": "file_exists", "passed": False},
                    ],
                    execution_duration=30.0,
                    retry_count=1,
                    completion_mode_used=False,
                    final_status=SheetStatus.FAILED,
                    validation_pass_rate=0.0,
                    first_attempt_success=False,
                )
            )

        result = aggregator.aggregate_outcomes(
            outcomes=outcomes,
            workspace_path=Path("/tmp/test"),
        )

        assert result.outcomes_recorded == 5
        # Should detect validation failure pattern
        assert result.patterns_detected >= 1
        assert result.patterns_merged >= 1
        assert result.priorities_updated is True

    def test_merge_with_conflict_resolution(
        self, aggregator: PatternAggregator
    ) -> None:
        """Test pattern merging with conflict resolution."""
        existing = PatternRecord(
            id="test-id",
            pattern_type="validation_failure",
            pattern_name="file_exists:missing",
            description="Old pattern",
            occurrence_count=5,
            first_seen=datetime.now() - timedelta(days=30),
            last_seen=datetime.now() - timedelta(days=10),
            last_confirmed=datetime.now() - timedelta(days=10),
            led_to_success_count=2,
            led_to_failure_count=3,
            effectiveness_score=0.4,
            variance=0.1,
            suggested_action="Old action",
            context_tags=["validation:file_exists"],
            priority_score=0.5,
        )

        new = DetectedPattern(
            pattern_type=PatternType.VALIDATION_FAILURE,
            description="New pattern",
            frequency=3,
            success_rate=0.8,
            last_seen=datetime.now(),
            context_tags=["validation:file_exists"],
        )

        merged = aggregator.merge_with_conflict_resolution(existing, new)

        # Sum occurrence counts
        assert merged["occurrence_count"] == 8  # 5 + 3

        # Weighted average effectiveness
        expected_eff = (0.4 * 5 + 0.8 * 3) / 8
        assert abs(merged["effectiveness_score"] - expected_eff) < 0.01

        # Higher effectiveness new action should be used
        assert "New pattern" in str(merged["suggested_action"])

    def test_prune_deprecated_patterns(
        self,
        global_store: GlobalLearningStore,
        aggregator: PatternAggregator,
    ) -> None:
        """Test pruning deprecated patterns."""
        # Record a low-effectiveness pattern with enough applications
        with global_store._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO patterns (
                    id, pattern_type, pattern_name,
                    occurrence_count, first_seen, last_seen, last_confirmed,
                    led_to_success_count, led_to_failure_count,
                    effectiveness_score, variance, context_tags, priority_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "deprecated-pattern",
                    "test",
                    "test:deprecated",
                    10,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    1,  # Only 1 success
                    9,  # 9 failures = 10% effectiveness
                    0.1,
                    0.0,
                    "[]",
                    0.5,
                ),
            )

        deprecated_count = aggregator.prune_deprecated_patterns()

        assert deprecated_count == 1

        # Check pattern was deprecated (priority set to 0)
        patterns = global_store.get_patterns(min_priority=0.0)
        deprecated = next(
            (p for p in patterns if p.id == "deprecated-pattern"), None
        )
        assert deprecated is not None
        assert deprecated.priority_score == 0.0


# =============================================================================
# TestAggregationResult
# =============================================================================


class TestAggregationResult:
    """Tests for AggregationResult class."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = AggregationResult()

        assert result.outcomes_recorded == 0
        assert result.patterns_detected == 0
        assert result.patterns_merged == 0
        assert result.priorities_updated is False
        assert result.errors == []

    def test_repr(self) -> None:
        """Test string representation."""
        result = AggregationResult()
        result.outcomes_recorded = 5
        result.patterns_detected = 3
        result.patterns_merged = 2

        repr_str = repr(result)

        assert "5" in repr_str
        assert "3" in repr_str
        assert "2" in repr_str


# =============================================================================
# TestLearningActivationIntegration
# =============================================================================


class TestLearningActivationIntegration:
    """Integration tests for Learning Activation features.

    Tests the end-to-end flow of:
    - Pattern querying from global store
    - Similar execution tracking
    - Optimal execution window analysis
    - Workspace clustering
    """

    def test_get_similar_executions_empty(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test get_similar_executions with no data."""
        executions = global_store.get_similar_executions()
        assert executions == []

    def test_get_similar_executions_with_filters(
        self,
        global_store: GlobalLearningStore,
        sample_outcome: SheetOutcome,
    ) -> None:
        """Test get_similar_executions with job_hash filter."""
        workspace_path = Path("/tmp/test-workspace")
        global_store.record_outcome(sample_outcome, workspace_path)

        # Get job hash for the recorded outcome
        job_hash = GlobalLearningStore.hash_job(sample_outcome.job_id)

        # Query similar executions
        executions = global_store.get_similar_executions(job_hash=job_hash)
        assert len(executions) == 1
        assert executions[0].job_hash == job_hash

    def test_get_optimal_execution_window_no_data(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test optimal execution window with no recovery data."""
        window = global_store.get_optimal_execution_window()

        assert window["optimal_hours"] == []
        assert window["avoid_hours"] == []
        assert window["confidence"] == 0.0
        assert window["sample_count"] == 0

    def test_get_optimal_execution_window_with_data(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test optimal execution window identifies patterns."""
        # Record successful recoveries at hour 10 (should become optimal)
        for _ in range(5):
            with global_store._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO error_recoveries
                    (id, error_code, suggested_wait, actual_wait,
                     recovery_success, recorded_at, model, time_of_day)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(datetime.now().timestamp()),
                        "E101",
                        60.0,
                        60.0,
                        True,  # Success
                        datetime.now().isoformat(),
                        None,
                        10,  # Hour 10
                    ),
                )

        # Record failed recoveries at hour 14 (should become avoid)
        for _ in range(5):
            with global_store._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO error_recoveries
                    (id, error_code, suggested_wait, actual_wait,
                     recovery_success, recorded_at, model, time_of_day)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(datetime.now().timestamp()) + str(_),
                        "E101",
                        60.0,
                        60.0,
                        False,  # Failure
                        datetime.now().isoformat(),
                        None,
                        14,  # Hour 14
                    ),
                )

        window = global_store.get_optimal_execution_window()

        assert 10 in window["optimal_hours"]
        assert 14 in window["avoid_hours"]
        assert window["confidence"] > 0
        assert window["sample_count"] == 10

    def test_workspace_clustering(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test workspace cluster assignment and retrieval."""
        workspace_hash = "abc123"
        cluster_id = "high-success-cluster"

        # Initially no cluster
        assert global_store.get_workspace_cluster(workspace_hash) is None

        # Assign to cluster
        global_store.assign_workspace_cluster(workspace_hash, cluster_id)

        # Should now return cluster ID
        assert global_store.get_workspace_cluster(workspace_hash) == cluster_id

        # Should appear in similar workspaces
        similar = global_store.get_similar_workspaces(cluster_id)
        assert workspace_hash in similar

    def test_integration_pattern_query_and_record(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test full integration: record pattern, query, and verify."""
        # Record a pattern
        pattern_id = global_store.record_pattern(
            pattern_type="validation_failure",
            pattern_name="integration_test_pattern",
            description="Test pattern for integration",
            context_tags=["test"],
            suggested_action="Review the validation",
        )

        # Query patterns
        patterns = global_store.get_patterns(min_priority=0.0)

        # Verify pattern exists
        test_pattern = next(
            (p for p in patterns if p.id == pattern_id), None
        )
        assert test_pattern is not None
        assert test_pattern.description == "Test pattern for integration"

        # Record pattern application
        workspace_path = Path("/tmp/test")
        outcome = SheetOutcome(
            sheet_id="test-sheet",
            job_id="test-job",
            validation_results=[],
            execution_duration=10.0,
            retry_count=0,
            completion_mode_used=False,
            final_status=SheetStatus.COMPLETED,
            validation_pass_rate=1.0,
            first_attempt_success=True,
        )
        exec_id = global_store.record_outcome(outcome, workspace_path)

        # Record that pattern was applied and improved outcome
        app_id = global_store.record_pattern_application(
            pattern_id=pattern_id,
            execution_id=exec_id,
            outcome_improved=True,
            retry_count_before=1,
            retry_count_after=0,
        )

        assert app_id is not None

        # Verify pattern effectiveness was updated
        patterns = global_store.get_patterns(min_priority=0.0)
        updated_pattern = next(
            (p for p in patterns if p.id == pattern_id), None
        )
        assert updated_pattern is not None
        assert updated_pattern.led_to_success_count == 1


# =============================================================================
# TestSelectivePatternRetrieval
# =============================================================================


class TestSelectivePatternRetrieval:
    """Tests for Selective Pattern Retrieval (SPR) - context_tags filtering.

    Evolution v7: Tests the context_tags filtering capability in get_patterns().
    """

    def test_get_patterns_filters_by_context_tags(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that get_patterns filters by context_tags correctly."""
        # Create patterns with different context tags
        global_store.record_pattern(
            pattern_type="validation_failure",
            pattern_name="sheet_specific",
            description="Pattern for sheet 5",
            context_tags=["sheet:5", "job:test-job"],
        )
        global_store.record_pattern(
            pattern_type="validation_failure",
            pattern_name="different_context",
            description="Pattern for sheet 10",
            context_tags=["sheet:10", "job:other-job"],
        )
        global_store.record_pattern(
            pattern_type="validation_failure",
            pattern_name="no_tags",
            description="Pattern without tags",
            context_tags=None,
        )

        # Query with context_tags that match first pattern
        patterns = global_store.get_patterns(
            min_priority=0.0,
            context_tags=["sheet:5"],
        )

        # Should only return the pattern with matching tag
        assert len(patterns) == 1
        assert patterns[0].pattern_name == "sheet_specific"

    def test_get_patterns_any_tag_matches(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that ANY tag match returns the pattern (OR semantics)."""
        global_store.record_pattern(
            pattern_type="test_pattern",
            pattern_name="multi_tag_pattern",
            description="Pattern with multiple tags",
            context_tags=["sheet:1", "sheet:2", "job:my-job"],
        )

        # Query with a tag that matches one of the pattern's tags
        patterns = global_store.get_patterns(
            min_priority=0.0,
            context_tags=["sheet:2", "unrelated:tag"],
        )

        assert len(patterns) == 1
        assert patterns[0].pattern_name == "multi_tag_pattern"

    def test_get_patterns_no_match_returns_empty(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that non-matching context_tags returns empty list."""
        global_store.record_pattern(
            pattern_type="test_pattern",
            pattern_name="tagged_pattern",
            description="Pattern with tags",
            context_tags=["sheet:5", "job:specific-job"],
        )

        # Query with non-matching tags
        patterns = global_store.get_patterns(
            min_priority=0.0,
            context_tags=["sheet:99", "job:nonexistent"],
        )

        assert len(patterns) == 0

    def test_get_patterns_none_context_tags_returns_all(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that context_tags=None returns all patterns (no filtering)."""
        global_store.record_pattern(
            pattern_type="test_pattern",
            pattern_name="pattern_with_tags",
            context_tags=["sheet:5"],
        )
        global_store.record_pattern(
            pattern_type="test_pattern",
            pattern_name="pattern_without_tags",
            context_tags=None,
        )

        # Query without context_tags filter
        patterns = global_store.get_patterns(min_priority=0.0, context_tags=None)

        # Should return all patterns
        assert len(patterns) >= 2
        pattern_names = {p.pattern_name for p in patterns}
        assert "pattern_with_tags" in pattern_names
        assert "pattern_without_tags" in pattern_names

    def test_get_patterns_empty_list_same_as_none(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that context_tags=[] behaves same as context_tags=None."""
        global_store.record_pattern(
            pattern_type="test_pattern",
            pattern_name="any_pattern",
            context_tags=["some:tag"],
        )

        # Empty list should NOT filter
        patterns_empty = global_store.get_patterns(min_priority=0.0, context_tags=[])
        patterns_none = global_store.get_patterns(min_priority=0.0, context_tags=None)

        # Both should return the same results
        assert len(patterns_empty) == len(patterns_none)

    def test_get_patterns_combines_type_and_tag_filters(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that pattern_type and context_tags filters work together."""
        global_store.record_pattern(
            pattern_type="type_a",
            pattern_name="type_a_tagged",
            context_tags=["sheet:5"],
        )
        global_store.record_pattern(
            pattern_type="type_b",
            pattern_name="type_b_tagged",
            context_tags=["sheet:5"],
        )
        global_store.record_pattern(
            pattern_type="type_a",
            pattern_name="type_a_different_tag",
            context_tags=["sheet:10"],
        )

        # Query with both filters
        patterns = global_store.get_patterns(
            pattern_type="type_a",
            min_priority=0.0,
            context_tags=["sheet:5"],
        )

        # Should only match type_a with sheet:5 tag
        assert len(patterns) == 1
        assert patterns[0].pattern_name == "type_a_tagged"


# =============================================================================
# TestOutputPatternExtractor
# =============================================================================


class TestOutputPatternExtractor:
    """Tests for OutputPatternExtractor class (Sheet 5 implementation)."""

    @pytest.fixture
    def extractor(self) -> OutputPatternExtractor:
        """Create an OutputPatternExtractor instance."""
        return OutputPatternExtractor()

    def test_empty_output_returns_empty_list(
        self, extractor: OutputPatternExtractor
    ) -> None:
        """Test that empty output returns no patterns."""
        assert extractor.extract_from_output("") == []
        assert extractor.extract_from_output("   ") == []
        assert extractor.extract_from_output(None) == []  # type: ignore[arg-type]

    def test_extract_rate_limit_pattern(
        self, extractor: OutputPatternExtractor
    ) -> None:
        """Test extraction of rate limit patterns."""
        output = "Error: rate limit exceeded for API calls"
        patterns = extractor.extract_from_output(output)

        assert len(patterns) == 1
        assert patterns[0].pattern_name == "rate_limit"
        assert "rate limit" in patterns[0].matched_text.lower()
        assert patterns[0].confidence == 0.95

    def test_extract_import_error_pattern(
        self, extractor: OutputPatternExtractor
    ) -> None:
        """Test extraction of import error patterns."""
        output = """Traceback (most recent call last):
  File "test.py", line 1, in <module>
    import nonexistent_module
ModuleNotFoundError: No module named 'nonexistent_module'"""
        patterns = extractor.extract_from_output(output)

        # Should find both traceback and import_error
        pattern_names = {p.pattern_name for p in patterns}
        assert "traceback" in pattern_names
        assert "import_error" in pattern_names

    def test_extract_permission_denied(
        self, extractor: OutputPatternExtractor
    ) -> None:
        """Test extraction of permission denied patterns."""
        output = "Error: Permission denied: /etc/passwd"
        patterns = extractor.extract_from_output(output)

        assert len(patterns) == 1
        assert patterns[0].pattern_name == "permission_denied"
        assert patterns[0].confidence == 0.95

    def test_extract_file_not_found(
        self, extractor: OutputPatternExtractor
    ) -> None:
        """Test extraction of file not found patterns."""
        output = "FileNotFoundError: [Errno 2] No such file or directory: 'missing.txt'"
        patterns = extractor.extract_from_output(output)

        assert len(patterns) >= 1
        file_not_found = next(
            (p for p in patterns if p.pattern_name == "file_not_found"), None
        )
        assert file_not_found is not None
        assert file_not_found.confidence == 0.95

    def test_extract_timeout_pattern(
        self, extractor: OutputPatternExtractor
    ) -> None:
        """Test extraction of timeout patterns."""
        output = "Connection timed out after 30s"
        patterns = extractor.extract_from_output(output)

        assert len(patterns) == 1
        assert patterns[0].pattern_name == "timeout"

    def test_extract_multiple_patterns(
        self, extractor: OutputPatternExtractor
    ) -> None:
        """Test extraction of multiple different patterns."""
        output = """ImportError: No module named 'requests'
Connection refused: localhost:8080
TypeError: 'NoneType' object is not callable"""
        patterns = extractor.extract_from_output(output)

        pattern_names = {p.pattern_name for p in patterns}
        assert "import_error" in pattern_names
        assert "connection_refused" in pattern_names
        assert "type_error" in pattern_names
        assert len(patterns) >= 3

    def test_line_context_extraction(
        self, extractor: OutputPatternExtractor
    ) -> None:
        """Test that context lines are extracted correctly."""
        output = """Line 1: Starting process
Line 2: Loading config
Line 3: ImportError: No module named 'flask'
Line 4: Process failed
Line 5: Cleanup complete"""
        patterns = extractor.extract_from_output(output)

        import_pattern = next(
            (p for p in patterns if p.pattern_name == "import_error"), None
        )
        assert import_pattern is not None
        assert import_pattern.line_number == 3
        assert len(import_pattern.context_before) == 2
        assert "Loading config" in import_pattern.context_before[-1]
        assert len(import_pattern.context_after) == 2
        assert "Process failed" in import_pattern.context_after[0]

    def test_source_parameter(
        self, extractor: OutputPatternExtractor
    ) -> None:
        """Test that source parameter is correctly set."""
        output = "Permission denied"
        stdout_patterns = extractor.extract_from_output(output, source="stdout")
        stderr_patterns = extractor.extract_from_output(output, source="stderr")

        assert stdout_patterns[0].source == "stdout"
        assert stderr_patterns[0].source == "stderr"

    def test_deduplication_same_line(
        self, extractor: OutputPatternExtractor
    ) -> None:
        """Test that duplicate patterns at same line are deduplicated."""
        # Rate limit appears multiple times on same line
        output = "rate limit rate limit rate limit"
        patterns = extractor.extract_from_output(output)

        # Should only have one rate_limit pattern
        rate_limit_count = sum(
            1 for p in patterns if p.pattern_name == "rate_limit"
        )
        assert rate_limit_count == 1

    def test_get_pattern_summary(
        self, extractor: OutputPatternExtractor
    ) -> None:
        """Test pattern summary generation."""
        output = """TypeError: bad arg
TypeError: another bad arg
ImportError: missing module"""
        patterns = extractor.extract_from_output(output)

        summary = extractor.get_pattern_summary(patterns)

        assert "type_error" in summary
        assert summary["type_error"] == 2
        assert "import_error" in summary
        assert summary["import_error"] == 1

    def test_confidence_scoring(
        self, extractor: OutputPatternExtractor
    ) -> None:
        """Test that confidence scores are correctly assigned."""

        # High confidence patterns
        high_conf = ["rate limit", "ImportError", "Permission denied"]
        # Lower confidence patterns
        lower_conf = ["Traceback (most recent call last)"]

        for text in high_conf:
            patterns = extractor.extract_from_output(text)
            assert patterns[0].confidence >= 0.90

        for text in lower_conf:
            patterns = extractor.extract_from_output(text)
            assert patterns[0].confidence < 0.90

    def test_case_insensitive_matching(
        self, extractor: OutputPatternExtractor
    ) -> None:
        """Test that matching is case insensitive where appropriate."""
        variants = ["RATE LIMIT", "Rate Limit", "rate limit", "RaTe LiMiT"]

        for variant in variants:
            patterns = extractor.extract_from_output(variant)
            assert len(patterns) >= 1
            assert any(p.pattern_name == "rate_limit" for p in patterns)


# =============================================================================
# TestExtractedPattern
# =============================================================================


class TestExtractedPattern:
    """Tests for ExtractedPattern dataclass (Sheet 5 implementation)."""

    def test_extracted_pattern_creation(self) -> None:
        """Test basic ExtractedPattern creation."""
        pattern = ExtractedPattern(
            pattern_name="rate_limit",
            matched_text="rate limit exceeded",
            line_number=42,
        )

        assert pattern.pattern_name == "rate_limit"
        assert pattern.matched_text == "rate limit exceeded"
        assert pattern.line_number == 42
        assert pattern.context_before == []
        assert pattern.context_after == []
        assert pattern.confidence == 0.8  # Default
        assert pattern.source == "stdout"  # Default

    def test_extracted_pattern_with_context(self) -> None:
        """Test ExtractedPattern with context lines."""
        pattern = ExtractedPattern(
            pattern_name="import_error",
            matched_text="ImportError: No module named 'x'",
            line_number=5,
            context_before=["import sys", "import os"],
            context_after=["# More code", "exit(1)"],
            confidence=0.95,
            source="stderr",
        )

        assert len(pattern.context_before) == 2
        assert len(pattern.context_after) == 2
        assert pattern.confidence == 0.95
        assert pattern.source == "stderr"


# =============================================================================
# TestPatternTypeEnum
# =============================================================================


class TestPatternTypeEnumOutputPattern:
    """Tests for OUTPUT_PATTERN enum value (Sheet 5 implementation)."""

    def test_output_pattern_enum_exists(self) -> None:
        """Test that OUTPUT_PATTERN exists in PatternType enum."""
        assert hasattr(PatternType, "OUTPUT_PATTERN")
        assert PatternType.OUTPUT_PATTERN.value == "output_pattern"

    def test_all_pattern_types_accessible(self) -> None:
        """Test that all pattern types are accessible."""
        expected_types = [
            "VALIDATION_FAILURE",
            "RETRY_SUCCESS",
            "COMPLETION_MODE",
            "FIRST_ATTEMPT_SUCCESS",
            "HIGH_CONFIDENCE",
            "LOW_CONFIDENCE",
            "SEMANTIC_FAILURE",
            "OUTPUT_PATTERN",
        ]

        for ptype in expected_types:
            assert hasattr(PatternType, ptype)


# =============================================================================
# TestErrorCodePatterns
# =============================================================================


class TestErrorCodePatterns:
    """Tests for _detect_error_code_patterns() method (Sheet 6 implementation)."""

    def test_detect_error_code_patterns_empty_outcomes(self) -> None:
        """Test error code detection with empty outcomes."""
        from mozart.learning.patterns import PatternDetector

        detector = PatternDetector([])
        patterns = detector._detect_error_code_patterns()
        assert patterns == []

    def test_detect_error_code_patterns_no_error_codes(self) -> None:
        """Test error code detection when outcomes have no error codes."""
        from mozart.learning.patterns import PatternDetector

        outcomes = [
            SheetOutcome(
                sheet_id="test-1",
                job_id="test-job",
                validation_results=[
                    {"rule_type": "file_exists", "passed": True}
                ],
                execution_duration=30.0,
                retry_count=0,
                completion_mode_used=False,
                final_status=SheetStatus.COMPLETED,
                validation_pass_rate=1.0,
                first_attempt_success=True,
            )
        ]

        detector = PatternDetector(outcomes)
        patterns = detector._detect_error_code_patterns()
        assert patterns == []

    def test_detect_error_code_patterns_single_occurrence(self) -> None:
        """Test error code detection with single occurrence (not pattern)."""
        from mozart.learning.patterns import PatternDetector

        outcomes = [
            SheetOutcome(
                sheet_id="test-1",
                job_id="test-job",
                validation_results=[
                    {"rule_type": "file_exists", "passed": False, "error_code": "E009"}
                ],
                execution_duration=30.0,
                retry_count=0,
                completion_mode_used=False,
                final_status=SheetStatus.FAILED,
                validation_pass_rate=0.0,
                first_attempt_success=False,
            )
        ]

        detector = PatternDetector(outcomes)
        patterns = detector._detect_error_code_patterns()
        # Single occurrence should not create a pattern
        assert patterns == []

    def test_detect_error_code_patterns_recurring(self) -> None:
        """Test error code detection with recurring error codes."""
        from mozart.learning.patterns import PatternDetector

        outcomes = [
            SheetOutcome(
                sheet_id="test-1",
                job_id="test-job",
                validation_results=[
                    {"rule_type": "file_exists", "passed": False, "error_code": "E009"}
                ],
                execution_duration=30.0,
                retry_count=0,
                completion_mode_used=False,
                final_status=SheetStatus.FAILED,
                validation_pass_rate=0.0,
                first_attempt_success=False,
            ),
            SheetOutcome(
                sheet_id="test-2",
                job_id="test-job",
                validation_results=[
                    {"rule_type": "file_exists", "passed": False, "error_code": "E009"}
                ],
                execution_duration=45.0,
                retry_count=1,
                completion_mode_used=False,
                final_status=SheetStatus.FAILED,
                validation_pass_rate=0.0,
                first_attempt_success=False,
            ),
        ]

        detector = PatternDetector(outcomes)
        patterns = detector._detect_error_code_patterns()

        # Should have at least one pattern for E009
        assert len(patterns) >= 1
        error_pattern = next(
            (p for p in patterns if "E009" in p.description), None
        )
        assert error_pattern is not None
        assert error_pattern.frequency >= 2
        assert "error_code:E009" in error_pattern.context_tags

    def test_detect_error_code_confidence_scaling(self) -> None:
        """Test that error code pattern confidence scales with frequency."""
        from mozart.learning.patterns import PatternDetector

        # Create 5 outcomes with same error code
        outcomes = []
        for i in range(5):
            outcomes.append(
                SheetOutcome(
                    sheet_id=f"test-{i}",
                    job_id="test-job",
                    validation_results=[
                        {"rule_type": "file_exists", "passed": False, "error_code": "E103"}
                    ],
                    execution_duration=30.0,
                    retry_count=0,
                    completion_mode_used=False,
                    final_status=SheetStatus.FAILED,
                    validation_pass_rate=0.0,
                    first_attempt_success=False,
                )
            )

        detector = PatternDetector(outcomes)
        patterns = detector._detect_error_code_patterns()

        error_pattern = next(
            (p for p in patterns if "E103" in p.description), None
        )
        assert error_pattern is not None
        # Higher frequency should have higher confidence
        assert error_pattern.confidence >= 0.85

    def test_error_patterns_in_detect_all(self) -> None:
        """Test that error code patterns are included in detect_all()."""
        from mozart.learning.patterns import PatternDetector

        outcomes = [
            SheetOutcome(
                sheet_id="test-1",
                job_id="test-job",
                validation_results=[
                    {"rule_type": "file_exists", "passed": False, "error_code": "E201"}
                ],
                execution_duration=30.0,
                retry_count=0,
                completion_mode_used=False,
                final_status=SheetStatus.FAILED,
                validation_pass_rate=0.0,
                first_attempt_success=False,
            ),
            SheetOutcome(
                sheet_id="test-2",
                job_id="test-job",
                validation_results=[
                    {"rule_type": "file_exists", "passed": False, "error_code": "E201"}
                ],
                execution_duration=45.0,
                retry_count=1,
                completion_mode_used=False,
                final_status=SheetStatus.FAILED,
                validation_pass_rate=0.0,
                first_attempt_success=False,
            ),
        ]

        detector = PatternDetector(outcomes)
        all_patterns = detector.detect_all()

        # Should include error code pattern
        error_pattern = next(
            (p for p in all_patterns if "E201" in p.description), None
        )
        assert error_pattern is not None


# =============================================================================
# TestEnhancedPatternAggregator
# =============================================================================


class TestEnhancedPatternAggregator:
    """Tests for EnhancedPatternAggregator class (Sheet 6 implementation)."""

    @pytest.fixture
    def enhanced_aggregator(
        self, global_store: GlobalLearningStore
    ) -> EnhancedPatternAggregator:
        """Create an enhanced pattern aggregator with test store."""
        return EnhancedPatternAggregator(global_store)

    def test_aggregate_empty_outcomes(
        self, enhanced_aggregator: EnhancedPatternAggregator
    ) -> None:
        """Test enhanced aggregation with empty outcomes list."""
        result = enhanced_aggregator.aggregate_with_all_sources(
            outcomes=[],
            workspace_path=Path("/tmp/test"),
        )

        assert isinstance(result, EnhancedAggregationResult)
        assert result.outcomes_recorded == 0
        assert result.patterns_detected == 0
        assert result.output_patterns == []
        assert result.output_pattern_summary == {}

    def test_aggregate_with_output_patterns(
        self, enhanced_aggregator: EnhancedPatternAggregator
    ) -> None:
        """Test enhanced aggregation extracts output patterns."""
        # Create outcome with stdout containing patterns
        outcome = SheetOutcome(
            sheet_id="test-1",
            job_id="test-job",
            validation_results=[
                {"rule_type": "file_exists", "passed": False}
            ],
            execution_duration=30.0,
            retry_count=0,
            completion_mode_used=False,
            final_status=SheetStatus.FAILED,
            validation_pass_rate=0.0,
            first_attempt_success=False,
        )
        # Add stdout_tail with error patterns
        outcome.stdout_tail = """Error: rate limit exceeded
Traceback (most recent call last):
  File "test.py", line 10
ImportError: No module named 'requests'"""

        result = enhanced_aggregator.aggregate_with_all_sources(
            outcomes=[outcome],
            workspace_path=Path("/tmp/test"),
        )

        assert result.outcomes_recorded == 1
        assert len(result.output_patterns) >= 1
        assert "rate_limit" in result.output_pattern_summary or \
               "import_error" in result.output_pattern_summary or \
               "traceback" in result.output_pattern_summary

    def test_enhanced_aggregation_result_repr(self) -> None:
        """Test EnhancedAggregationResult string representation."""
        result = EnhancedAggregationResult()
        result.outcomes_recorded = 5
        result.patterns_detected = 10
        result.patterns_merged = 8
        result.output_patterns = []  # empty for test

        repr_str = repr(result)

        assert "outcomes=5" in repr_str
        assert "detected=10" in repr_str
        assert "merged=8" in repr_str
        assert "output_patterns=0" in repr_str


# =============================================================================
# TestFullDataCollectionPipeline
# =============================================================================


class TestFullDataCollectionPipeline:
    """Integration tests for full learning data collection flow (Sheet 6).

    Tests the complete pipeline from SheetOutcome creation through
    pattern extraction, aggregation, and global store storage.
    """

    def test_full_data_collection_pipeline(
        self, temp_db_path: Path
    ) -> None:
        """Test complete learning data collection flow."""
        # 1. Create outcomes with realistic stdout/stderr and error codes
        outcomes = [
            SheetOutcome(
                sheet_id="test:1",
                job_id="test-job",
                validation_results=[
                    {
                        "rule_type": "file_exists",
                        "passed": False,
                        "error_code": "E009",
                        "failure_category": "missing",
                        "failure_reason": "File not created in workspace",
                    }
                ],
                execution_duration=30.0,
                retry_count=1,
                completion_mode_used=False,
                final_status=SheetStatus.FAILED,
                validation_pass_rate=0.0,
                first_attempt_success=False,
            ),
            SheetOutcome(
                sheet_id="test:2",
                job_id="test-job",
                validation_results=[
                    {
                        "rule_type": "file_exists",
                        "passed": False,
                        "error_code": "E009",
                        "failure_category": "missing",
                        "failure_reason": "File not created in workspace",
                    }
                ],
                execution_duration=45.0,
                retry_count=2,
                completion_mode_used=True,
                final_status=SheetStatus.FAILED,
                validation_pass_rate=0.0,
                first_attempt_success=False,
            ),
            SheetOutcome(
                sheet_id="test:3",
                job_id="test-job",
                validation_results=[
                    {
                        "rule_type": "content_contains",
                        "passed": True,
                    }
                ],
                execution_duration=20.0,
                retry_count=0,
                completion_mode_used=False,
                final_status=SheetStatus.COMPLETED,
                validation_pass_rate=1.0,
                first_attempt_success=True,
            ),
        ]

        # Add stdout_tail with error patterns to first outcome
        outcomes[0].stdout_tail = (
            "Error: FileNotFoundError: config.yaml not found\n"
            "Traceback (most recent call last):\n"
            "  File 'main.py', line 10\n"
        )
        outcomes[1].stdout_tail = (
            "FileNotFoundError: config.yaml not found\n"
            "Process failed\n"
        )

        # 2. Run enhanced aggregator
        store = GlobalLearningStore(temp_db_path)
        aggregator = EnhancedPatternAggregator(store)
        result = aggregator.aggregate_with_all_sources(
            outcomes, Path("/tmp/test-workspace")
        )

        # 3. Verify outcomes recorded
        assert result.outcomes_recorded == 3

        # 4. Verify patterns extracted (including output patterns)
        assert result.patterns_detected >= 1

        # 5. Verify output patterns extracted
        assert len(result.output_patterns) >= 1
        # Check that file_not_found pattern was detected from stdout
        pattern_names = {p.pattern_name for p in result.output_patterns}
        assert "file_not_found" in pattern_names or "traceback" in pattern_names

        # 6. Verify stored in global store
        patterns = store.get_patterns(min_priority=0.0)
        assert len(patterns) >= 1

        # 7. Verify error code pattern was detected
        # Check that E009 pattern exists (from recurring error code)
        error_pattern = next(
            (p for p in patterns if p.description and "E009" in p.description), None
        )
        assert error_pattern is not None

        # Cleanup
        if temp_db_path.exists():
            temp_db_path.unlink()

    def test_pipeline_with_semantic_patterns(
        self, temp_db_path: Path
    ) -> None:
        """Test pipeline detects semantic patterns from failure categories."""
        outcomes = [
            SheetOutcome(
                sheet_id=f"test:{i}",
                job_id="test-job",
                validation_results=[
                    {
                        "rule_type": "file_exists",
                        "passed": False,
                        "failure_category": "stale",
                        "failure_reason": "File not modified recently",
                    }
                ],
                execution_duration=30.0,
                retry_count=0,
                completion_mode_used=False,
                final_status=SheetStatus.FAILED,
                validation_pass_rate=0.0,
                first_attempt_success=False,
            )
            for i in range(3)  # 3 outcomes with same failure category
        ]

        store = GlobalLearningStore(temp_db_path)
        aggregator = EnhancedPatternAggregator(store)
        result = aggregator.aggregate_with_all_sources(
            outcomes, Path("/tmp/test-workspace")
        )

        # Verify semantic patterns detected
        assert result.patterns_detected >= 1

        # Check patterns in store include semantic failure
        patterns = store.get_patterns(min_priority=0.0)
        stale_pattern = next(
            (
                p for p in patterns
                if p.description and "stale" in p.description.lower()
            ),
            None,
        )
        assert stale_pattern is not None

        # Cleanup
        if temp_db_path.exists():
            temp_db_path.unlink()

    def test_import_verification(self) -> None:
        """Verify all learning module imports work correctly."""
        # Test all key imports
        from mozart.learning import (
            aggregator,
            global_store,
            outcomes,
            patterns,
            weighter,
        )

        # Verify key classes are accessible
        assert hasattr(patterns, "OutputPatternExtractor")
        assert hasattr(patterns, "ExtractedPattern")
        assert hasattr(patterns, "PatternDetector")
        assert hasattr(patterns, "PatternType")
        assert hasattr(aggregator, "EnhancedPatternAggregator")
        assert hasattr(aggregator, "EnhancedAggregationResult")
        assert hasattr(outcomes, "SheetOutcome")
        assert hasattr(global_store, "GlobalLearningStore")
        assert hasattr(weighter, "PatternWeighter")
