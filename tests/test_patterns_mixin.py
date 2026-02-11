"""Tests for PatternsMixin pattern management and learning activation.

Tests the patterns mixin methods in isolation:
- _query_relevant_patterns(): Query patterns from global store
- _record_pattern_feedback(): Record pattern application outcomes
- _assess_failure_risk(): Assess failure risk from historical data

These tests exercise the pattern selection and feedback loop that
enables cross-workspace learning (v22 Metacognitive Pattern Reflection).
"""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from mozart.core.config import JobConfig
from mozart.execution.runner import JobRunner
from mozart.execution.runner.patterns import PatternFeedbackContext

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_config() -> JobConfig:
    """Create a sample job configuration for testing."""
    return JobConfig.model_validate({
        "name": "test-patterns-job",
        "description": "Test job for patterns mixin tests",
        "backend": {
            "type": "claude_cli",
            "skip_permissions": True,
            "cli_model": "claude-sonnet-4-20250514",
        },
        "sheet": {
            "size": 10,
            "total_items": 30,
        },
        "prompt": {
            "template": "Process sheet {{ sheet_num }}.",
        },
        "learning": {
            "enabled": True,
            "exploration_rate": 0.1,
            "exploration_min_priority": 0.1,
            "auto_apply": {
                "enabled": True,
                "trust_threshold": 0.8,
                "require_validated_status": False,
                "max_patterns_per_sheet": 3,
                "log_applications": True,
            },
        },
    })


@pytest.fixture
def sample_config_no_auto_apply(sample_config: JobConfig) -> JobConfig:
    """Config with auto_apply disabled."""
    config_dict = sample_config.model_dump()
    config_dict["learning"]["auto_apply"]["enabled"] = False
    return JobConfig.model_validate(config_dict)


@pytest.fixture
def mock_backend() -> AsyncMock:
    """Create a mock backend."""
    backend = AsyncMock()
    backend.health_check = AsyncMock(return_value=True)
    return backend


@pytest.fixture
def mock_state_backend() -> AsyncMock:
    """Create a mock state backend."""
    backend = AsyncMock()
    backend.load = AsyncMock(return_value=None)
    backend.save = AsyncMock()
    backend.list_jobs = AsyncMock(return_value=[])
    return backend


def _create_mock_pattern(
    pattern_id: str,
    name: str,
    effectiveness: float = 0.8,
    priority: float = 0.7,
    occurrence_count: int = 10,
    trust_score: float = 0.85,
) -> MagicMock:
    """Create a mock pattern record."""
    pattern = MagicMock()
    pattern.id = pattern_id
    pattern.pattern_name = name
    pattern.description = f"Description for {name}"
    pattern.effectiveness_score = effectiveness
    pattern.priority_score = priority
    pattern.occurrence_count = occurrence_count
    pattern.trust_score = trust_score
    return pattern


@pytest.fixture
def mock_global_store() -> MagicMock:
    """Create a mock global learning store."""
    store = MagicMock()

    # Default: return some patterns
    store.get_patterns = MagicMock(return_value=[
        _create_mock_pattern("p1", "Pattern One", effectiveness=0.9, priority=0.8),
        _create_mock_pattern("p2", "Pattern Two", effectiveness=0.5, priority=0.4),
    ])
    store.get_patterns_for_auto_apply = MagicMock(return_value=[])
    store.record_pattern_application = MagicMock()
    store.update_success_factors = MagicMock()
    store.get_execution_stats = MagicMock(return_value={
        "total_executions": 100,
        "first_attempt_success_rate": 0.75,
    })
    store.is_rate_limited = MagicMock(return_value=(False, None))

    return store


@pytest.fixture
def runner(
    sample_config: JobConfig,
    mock_backend: AsyncMock,
    mock_state_backend: AsyncMock,
) -> JobRunner:
    """Create a JobRunner instance for testing."""
    return JobRunner(
        config=sample_config,
        backend=mock_backend,
        state_backend=mock_state_backend,
    )


@pytest.fixture
def runner_with_global_store(
    sample_config: JobConfig,
    mock_backend: AsyncMock,
    mock_state_backend: AsyncMock,
    mock_global_store: MagicMock,
) -> JobRunner:
    """Create a JobRunner with global learning store."""
    runner = JobRunner(
        config=sample_config,
        backend=mock_backend,
        state_backend=mock_state_backend,
    )
    runner._global_learning_store = mock_global_store
    return runner


# =============================================================================
# TestQueryRelevantPatterns
# =============================================================================


class TestQueryRelevantPatterns:
    """Tests for _query_relevant_patterns() method."""

    def test_returns_empty_without_global_store(self, runner: JobRunner) -> None:
        """Test that it returns empty lists without global store."""
        descriptions, pattern_ids = runner._query_relevant_patterns(
            job_id="test-job",
            sheet_num=1,
        )

        assert descriptions == []
        assert pattern_ids == []

    def test_returns_patterns_from_global_store(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that patterns are fetched from global store."""
        descriptions, pattern_ids = runner_with_global_store._query_relevant_patterns(
            job_id="test-job",
            sheet_num=1,
        )

        assert len(descriptions) == 2
        assert pattern_ids == ["p1", "p2"]
        mock_global_store.get_patterns.assert_called()

    def test_includes_context_tags_in_query(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that context tags are passed to global store."""
        runner_with_global_store._query_relevant_patterns(
            job_id="test-job",
            sheet_num=3,
            context_tags=["error:rate_limit", "phase:execution"],
        )

        # Should be called with context_tags
        call_kwargs = mock_global_store.get_patterns.call_args.kwargs
        assert "context_tags" in call_kwargs
        assert "error:rate_limit" in call_kwargs["context_tags"]
        assert "phase:execution" in call_kwargs["context_tags"]

    def test_auto_generates_context_tags(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that context tags are auto-generated if not provided."""
        runner_with_global_store._query_relevant_patterns(
            job_id="test-job",
            sheet_num=5,
        )

        # Should auto-generate sheet and job tags
        call_kwargs = mock_global_store.get_patterns.call_args.kwargs
        assert "context_tags" in call_kwargs
        tags = call_kwargs["context_tags"]
        assert any("sheet:5" in str(t) for t in tags)
        assert any("job:test-job" in str(t) for t in tags)

    def test_tracks_exploration_vs_exploitation(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that patterns are categorized as exploration or exploitation."""
        # Pattern with low priority (<0.3) should be exploration
        mock_global_store.get_patterns.return_value = [
            _create_mock_pattern("high", "High Priority", priority=0.8),
            _create_mock_pattern("low", "Low Priority", priority=0.2),
        ]

        runner_with_global_store._query_relevant_patterns(
            job_id="test-job",
            sheet_num=1,
        )

        # Check tracking lists
        # High priority pattern should be exploitation
        assert "high" in runner_with_global_store._exploitation_pattern_ids
        # Low priority pattern should be exploration
        assert "low" in runner_with_global_store._exploration_pattern_ids

    def test_handles_global_store_error_gracefully(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that errors from global store are handled gracefully."""
        mock_global_store.get_patterns.side_effect = sqlite3.OperationalError("Database error")

        # Should not raise, should return empty
        descriptions, pattern_ids = runner_with_global_store._query_relevant_patterns(
            job_id="test-job",
            sheet_num=1,
        )

        assert descriptions == []
        assert pattern_ids == []

    def test_includes_effectiveness_in_description(
        self,
        runner_with_global_store: JobRunner,
    ) -> None:
        """Test that pattern descriptions include effectiveness info."""
        descriptions, _ = runner_with_global_store._query_relevant_patterns(
            job_id="test-job",
            sheet_num=1,
        )

        # Descriptions should contain effectiveness percentages
        assert any("90%" in desc for desc in descriptions)  # Pattern One: 0.9
        assert any("50%" in desc for desc in descriptions)  # Pattern Two: 0.5

    def test_includes_occurrence_count_in_description(
        self,
        runner_with_global_store: JobRunner,
    ) -> None:
        """Test that pattern descriptions include occurrence count."""
        descriptions, _ = runner_with_global_store._query_relevant_patterns(
            job_id="test-job",
            sheet_num=1,
        )

        # Descriptions should contain "seen Nx"
        assert any("seen 10x" in desc for desc in descriptions)

    def test_auto_apply_patterns_included_first(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that auto-apply patterns are included first."""
        auto_apply_pattern = _create_mock_pattern(
            "auto", "Auto Apply Pattern",
            effectiveness=0.95,
            trust_score=0.9,
        )
        mock_global_store.get_patterns_for_auto_apply.return_value = [auto_apply_pattern]

        descriptions, pattern_ids = runner_with_global_store._query_relevant_patterns(
            job_id="test-job",
            sheet_num=1,
        )

        # Auto-apply pattern should be first
        assert pattern_ids[0] == "auto"
        # 0.95 effectiveness -> "✓" indicator
        assert "✓" in descriptions[0]
        assert "effective" in descriptions[0]


# =============================================================================
# TestRecordPatternFeedback
# =============================================================================


class TestRecordPatternFeedback:
    """Tests for _record_pattern_feedback() method."""

    @pytest.mark.asyncio
    async def test_does_nothing_without_global_store(
        self,
        runner: JobRunner,
    ) -> None:
        """Test that recording does nothing without global store."""
        # Should not raise
        await runner._record_pattern_feedback(
            pattern_ids=["p1", "p2"],
            ctx=PatternFeedbackContext(
                validation_passed=True,
                first_attempt_success=True,
                sheet_num=1,
            ),
        )

    @pytest.mark.asyncio
    async def test_does_nothing_with_empty_pattern_ids(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that recording does nothing with empty pattern list."""
        await runner_with_global_store._record_pattern_feedback(
            pattern_ids=[],
            ctx=PatternFeedbackContext(
                validation_passed=True,
                first_attempt_success=True,
                sheet_num=1,
            ),
        )

        mock_global_store.record_pattern_application.assert_not_called()

    @pytest.mark.asyncio
    async def test_records_pattern_application_for_each_pattern(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that each pattern gets a feedback record."""
        await runner_with_global_store._record_pattern_feedback(
            pattern_ids=["p1", "p2", "p3"],
            ctx=PatternFeedbackContext(
                validation_passed=True,
                first_attempt_success=True,
                sheet_num=1,
            ),
        )

        # Should call record_pattern_application 3 times
        assert mock_global_store.record_pattern_application.call_count == 3

    @pytest.mark.asyncio
    async def test_records_pattern_led_to_success_true_on_success(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that pattern_led_to_success=True when validation passed and first attempt."""
        await runner_with_global_store._record_pattern_feedback(
            pattern_ids=["p1"],
            ctx=PatternFeedbackContext(
                validation_passed=True,
                first_attempt_success=True,
                sheet_num=1,
            ),
        )

        call_kwargs = mock_global_store.record_pattern_application.call_args.kwargs
        assert call_kwargs["pattern_led_to_success"] is True

    @pytest.mark.asyncio
    async def test_records_pattern_led_to_success_false_on_failure(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that pattern_led_to_success=False when validation failed."""
        await runner_with_global_store._record_pattern_feedback(
            pattern_ids=["p1"],
            ctx=PatternFeedbackContext(
                validation_passed=False,
                first_attempt_success=False,
                sheet_num=1,
            ),
        )

        call_kwargs = mock_global_store.record_pattern_application.call_args.kwargs
        assert call_kwargs["pattern_led_to_success"] is False

    @pytest.mark.asyncio
    async def test_records_pattern_led_to_success_false_on_retry(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that pattern_led_to_success=False when not first attempt."""
        await runner_with_global_store._record_pattern_feedback(
            pattern_ids=["p1"],
            ctx=PatternFeedbackContext(
                validation_passed=True,
                first_attempt_success=False,  # Required retry
                sheet_num=1,
            ),
        )

        call_kwargs = mock_global_store.record_pattern_application.call_args.kwargs
        assert call_kwargs["pattern_led_to_success"] is False

    @pytest.mark.asyncio
    async def test_passes_grounding_confidence(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that grounding confidence is passed to global store."""
        await runner_with_global_store._record_pattern_feedback(
            pattern_ids=["p1"],
            ctx=PatternFeedbackContext(
                validation_passed=True,
                first_attempt_success=True,
                sheet_num=1,
                grounding_confidence=0.85,
            ),
        )

        call_kwargs = mock_global_store.record_pattern_application.call_args.kwargs
        assert call_kwargs["grounding_confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_updates_success_factors_on_success(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that success factors are updated when pattern succeeds."""
        await runner_with_global_store._record_pattern_feedback(
            pattern_ids=["p1"],
            ctx=PatternFeedbackContext(
                validation_passed=True,
                first_attempt_success=True,
                sheet_num=1,
                validation_types=["file_exists", "content_regex"],
                error_categories=["rate_limit"],
                prior_sheet_status="completed",
            ),
        )

        mock_global_store.update_success_factors.assert_called_once()
        call_kwargs = mock_global_store.update_success_factors.call_args.kwargs
        assert call_kwargs["pattern_id"] == "p1"
        assert call_kwargs["validation_types"] == ["file_exists", "content_regex"]
        assert call_kwargs["error_categories"] == ["rate_limit"]
        assert call_kwargs["prior_sheet_status"] == "completed"

    @pytest.mark.asyncio
    async def test_does_not_update_success_factors_on_failure(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that success factors are NOT updated when pattern fails."""
        await runner_with_global_store._record_pattern_feedback(
            pattern_ids=["p1"],
            ctx=PatternFeedbackContext(
                validation_passed=False,
                first_attempt_success=False,
                sheet_num=1,
            ),
        )

        mock_global_store.update_success_factors.assert_not_called()

    @pytest.mark.asyncio
    async def test_distinguishes_exploration_from_exploitation(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that application mode is tracked correctly."""
        # Pre-populate tracking lists
        runner_with_global_store._exploration_pattern_ids = ["p_explore"]
        runner_with_global_store._exploitation_pattern_ids = ["p_exploit"]

        await runner_with_global_store._record_pattern_feedback(
            pattern_ids=["p_explore", "p_exploit"],
            ctx=PatternFeedbackContext(
                validation_passed=True,
                first_attempt_success=True,
                sheet_num=1,
            ),
        )

        # Check first call (exploration pattern)
        first_call = mock_global_store.record_pattern_application.call_args_list[0]
        assert first_call.kwargs["application_mode"] == "exploration"

        # Check second call (exploitation pattern)
        second_call = mock_global_store.record_pattern_application.call_args_list[1]
        assert second_call.kwargs["application_mode"] == "exploitation"

    @pytest.mark.asyncio
    async def test_handles_errors_gracefully(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that errors don't block execution."""
        mock_global_store.record_pattern_application.side_effect = (
            sqlite3.OperationalError("DB error")
        )

        # Should not raise
        await runner_with_global_store._record_pattern_feedback(
            pattern_ids=["p1"],
            ctx=PatternFeedbackContext(
                validation_passed=True,
                first_attempt_success=True,
                sheet_num=1,
            ),
        )


# =============================================================================
# TestAssessFailureRisk
# =============================================================================


class TestAssessFailureRisk:
    """Tests for _assess_failure_risk() method."""

    def test_returns_unknown_without_global_store(
        self,
        runner: JobRunner,
    ) -> None:
        """Test that it returns unknown risk without global store."""
        assessment = runner._assess_failure_risk(job_id="test", sheet_num=1)

        assert assessment["risk_level"] == "unknown"
        assert assessment["confidence"] == 0.0

    def test_returns_unknown_with_insufficient_data(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that it returns unknown with insufficient executions."""
        mock_global_store.get_execution_stats.return_value = {
            "total_executions": 5,  # Less than 10
            "first_attempt_success_rate": 0.8,
        }

        assessment = runner_with_global_store._assess_failure_risk(
            job_id="test", sheet_num=1
        )

        assert assessment["risk_level"] == "unknown"
        assert "insufficient data" in assessment["factors"][0]

    def test_returns_low_risk_with_high_success_rate(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that high success rate yields low risk."""
        mock_global_store.get_execution_stats.return_value = {
            "total_executions": 100,
            "first_attempt_success_rate": 0.85,  # > 0.7
        }

        assessment = runner_with_global_store._assess_failure_risk(
            job_id="test", sheet_num=1
        )

        assert assessment["risk_level"] == "low"
        assert assessment["confidence"] > 0

    def test_returns_medium_risk_with_moderate_success_rate(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that moderate success rate yields medium risk."""
        mock_global_store.get_execution_stats.return_value = {
            "total_executions": 50,
            "first_attempt_success_rate": 0.55,  # Between 0.4 and 0.7
        }

        assessment = runner_with_global_store._assess_failure_risk(
            job_id="test", sheet_num=1
        )

        assert assessment["risk_level"] == "medium"

    def test_returns_high_risk_with_low_success_rate(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that low success rate yields high risk."""
        mock_global_store.get_execution_stats.return_value = {
            "total_executions": 100,
            "first_attempt_success_rate": 0.25,  # < 0.4
        }

        assessment = runner_with_global_store._assess_failure_risk(
            job_id="test", sheet_num=1
        )

        assert assessment["risk_level"] == "high"

    def test_returns_high_risk_with_active_rate_limit(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that active rate limit increases risk to high."""
        mock_global_store.get_execution_stats.return_value = {
            "total_executions": 100,
            "first_attempt_success_rate": 0.8,  # Would be low risk
        }
        mock_global_store.is_rate_limited.return_value = (True, 120.0)

        assessment = runner_with_global_store._assess_failure_risk(
            job_id="test", sheet_num=1
        )

        assert assessment["risk_level"] == "high"
        assert any("rate limit" in f for f in assessment["factors"])

    def test_includes_recommendations_for_high_risk(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that high risk includes recommendations."""
        mock_global_store.get_execution_stats.return_value = {
            "total_executions": 100,
            "first_attempt_success_rate": 0.2,
        }

        assessment = runner_with_global_store._assess_failure_risk(
            job_id="test", sheet_num=1
        )

        assert assessment["risk_level"] == "high"
        assert len(assessment["recommended_adjustments"]) > 0

    def test_handles_errors_gracefully(
        self,
        runner_with_global_store: JobRunner,
        mock_global_store: MagicMock,
    ) -> None:
        """Test that errors return unknown risk."""
        mock_global_store.get_execution_stats.side_effect = sqlite3.OperationalError("DB error")

        assessment = runner_with_global_store._assess_failure_risk(
            job_id="test", sheet_num=1
        )

        assert assessment["risk_level"] == "unknown"
        assert "failed" in assessment["factors"][0]
