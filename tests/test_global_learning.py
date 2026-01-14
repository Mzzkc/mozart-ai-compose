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
from mozart.learning.aggregator import AggregationResult, PatternAggregator
from mozart.learning.global_store import (
    GlobalLearningStore,
    PatternRecord,
)
from mozart.learning.outcomes import SheetOutcome
from mozart.learning.patterns import DetectedPattern, PatternType
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
