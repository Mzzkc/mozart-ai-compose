"""Tests for CLI learning commands.

Covers the 10 learning subcommands: learning-stats, learning-insights,
learning-activity, patterns-list, patterns-why, patterns-budget,
patterns-entropy, entropy-status, learning-drift, learning-epistemic-drift.

All commands use deferred imports (inside function body), so we patch
at the source module: mozart.learning.global_store.get_global_store
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from mozart.cli import app

runner = CliRunner()

# Patch target for get_global_store (deferred import at source)
_GS_PATCH = "mozart.learning.global_store.get_global_store"
# Patch target for GlobalLearningStore class (used directly in learning-insights)
_GS_CLS_PATCH = "mozart.learning.global_store.GlobalLearningStore"
# Patch target for check_migration_status
_MIGRATION_PATCH = "mozart.learning.migration.check_migration_status"


def _make_mock_store() -> MagicMock:
    """Create a mock GlobalLearningStore with common return values."""
    store = MagicMock()
    store.get_execution_stats.return_value = {
        "total_executions": 42,
        "first_attempt_success_rate": 0.85,
        "total_patterns": 7,
        "avg_pattern_effectiveness": 0.72,
        "unique_workspaces": 3,
        "total_error_recoveries": 5,
        "avg_recovery_success_rate": 0.80,
    }
    store.get_patterns.return_value = []
    store.get_optimal_execution_window.return_value = {
        "optimal_hours": [9, 10, 14],
        "avoid_hours": [2, 3],
        "confidence": 0.75,
    }
    store.get_similar_executions.return_value = []
    store.get_drifting_patterns.return_value = []
    store.get_pattern_drift_summary.return_value = {}
    store.get_epistemic_drifting_patterns.return_value = []
    store.get_epistemic_drift_summary.return_value = {}
    # Return None so the command shows "no records" message (avoids comparison ops)
    store.get_exploration_budget.return_value = None
    store.get_exploration_budget_history.return_value = []
    store.get_exploration_budget_statistics.return_value = {
        "avg_budget": 0.0,
        "min_budget": 0.0,
        "max_budget": 0.0,
        "total_adjustments": 0,
        "floor_enforcements": 0,
        "boost_count": 0,
        "decay_count": 0,
    }
    store.calculate_pattern_entropy.return_value = MagicMock(
        shannon_entropy=2.5,
        max_possible_entropy=3.0,
        diversity_index=0.83,
        unique_pattern_count=7,
        effective_pattern_count=5,
        total_applications=42,
        dominant_pattern_share=0.35,
        threshold_exceeded=False,
    )
    store.get_pattern_entropy_history.return_value = []
    store.check_entropy_response_needed.return_value = False
    store.get_entropy_response_history.return_value = []
    store.get_entropy_response_statistics.return_value = {
        "total_responses": 0,
        "avg_entropy_at_trigger": 0.0,
        "budget_boosts": 0,
        "quarantine_revisits": 0,
        "last_response": None,
    }
    store.get_last_entropy_response.return_value = None
    return store


class TestLearningStatsCommand:
    """Tests for the learning-stats command."""

    @patch(_MIGRATION_PATCH)
    @patch(_GS_PATCH)
    def test_learning_stats_human_output(self, mock_get_store, mock_migration):
        """learning-stats shows human-readable statistics."""
        mock_store = _make_mock_store()
        mock_get_store.return_value = mock_store
        mock_migration.return_value = {"pending_workspaces": 0, "migrated": True}

        result = runner.invoke(app, ["learning-stats"])

        assert result.exit_code == 0
        assert "42" in result.stdout  # total_executions

    @patch(_MIGRATION_PATCH)
    @patch(_GS_PATCH)
    def test_learning_stats_json_output(self, mock_get_store, mock_migration):
        """learning-stats --json returns valid JSON."""
        mock_store = _make_mock_store()
        mock_get_store.return_value = mock_store
        mock_migration.return_value = {"pending_workspaces": 0, "migrated": True}

        result = runner.invoke(app, ["learning-stats", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "executions" in data
        assert data["executions"]["total"] == 42


class TestLearningInsightsCommand:
    """Tests for the learning-insights command."""

    @patch(_GS_CLS_PATCH)
    def test_insights_no_patterns(self, mock_cls):
        """learning-insights shows message when no patterns exist."""
        mock_store = MagicMock()
        mock_store.get_patterns.return_value = []
        mock_cls.return_value = mock_store

        result = runner.invoke(app, ["learning-insights"])

        assert result.exit_code == 0
        assert "No patterns" in result.stdout

    @patch(_GS_CLS_PATCH)
    def test_insights_with_patterns(self, mock_cls):
        """learning-insights shows pattern table with data."""
        pattern = MagicMock()
        pattern.pattern_type = "output_pattern"
        pattern.description = "Test pattern description"
        pattern.occurrence_count = 5
        pattern.effectiveness_score = 0.85

        mock_store = MagicMock()
        mock_store.get_patterns.return_value = [pattern]
        mock_cls.return_value = mock_store

        result = runner.invoke(app, ["learning-insights"])

        assert result.exit_code == 0
        assert "output_pattern" in result.stdout


class TestLearningActivityCommand:
    """Tests for the learning-activity command."""

    @patch(_GS_PATCH)
    def test_activity_default_hours(self, mock_get_store):
        """learning-activity shows recent activity."""
        mock_store = _make_mock_store()
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["learning-activity"])

        assert result.exit_code == 0

    @patch(_GS_PATCH)
    def test_activity_json_output(self, mock_get_store):
        """learning-activity --json returns valid JSON."""
        mock_store = _make_mock_store()
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["learning-activity", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert isinstance(data, dict)


class TestPatternsListCommand:
    """Tests for the patterns-list command."""

    @patch(_GS_PATCH)
    def test_patterns_list_empty(self, mock_get_store):
        """patterns-list shows message when no patterns."""
        mock_store = _make_mock_store()
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["patterns-list"])

        assert result.exit_code == 0

    @patch(_GS_PATCH)
    def test_patterns_list_with_data(self, mock_get_store):
        """patterns-list shows patterns in table."""
        pattern = MagicMock()
        pattern.id = "pat-001"
        pattern.pattern_type = "retry_pattern"
        pattern.pattern_name = "Test Retry"
        pattern.description = "A test retry pattern"
        pattern.occurrence_count = 10
        pattern.effectiveness_score = 0.9
        pattern.priority_score = 0.85
        pattern.context_tags = {"test"}
        pattern.quarantine_status = MagicMock(value="active")
        pattern.trust_score = 0.95

        mock_store = MagicMock()
        mock_store.get_patterns.return_value = [pattern]
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["patterns-list"])

        assert result.exit_code == 0


class TestPatternsBudgetCommand:
    """Tests for the patterns-budget command."""

    @patch(_GS_PATCH)
    def test_budget_shows_status(self, mock_get_store):
        """patterns-budget shows budget information."""
        mock_store = _make_mock_store()
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["patterns-budget"])

        assert result.exit_code == 0


class TestPatternsEntropyCommand:
    """Tests for the patterns-entropy command."""

    @patch(_GS_PATCH)
    def test_entropy_shows_metrics(self, mock_get_store):
        """patterns-entropy shows entropy metrics."""
        mock_store = _make_mock_store()
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["patterns-entropy"])

        assert result.exit_code == 0


class TestLearningDriftCommand:
    """Tests for the learning-drift command."""

    @patch(_GS_PATCH)
    def test_drift_no_drifting_patterns(self, mock_get_store):
        """learning-drift shows message when no drift detected."""
        mock_store = _make_mock_store()
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["learning-drift"])

        assert result.exit_code == 0

    @patch(_GS_PATCH)
    def test_drift_json_output(self, mock_get_store):
        """learning-drift --json returns valid JSON."""
        mock_store = _make_mock_store()
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["learning-drift", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert isinstance(data, dict)


class TestEpistemicDriftCommand:
    """Tests for the learning-epistemic-drift command."""

    @patch(_GS_PATCH)
    def test_epistemic_drift_no_data(self, mock_get_store):
        """learning-epistemic-drift shows message with no data."""
        mock_store = _make_mock_store()
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["learning-epistemic-drift"])

        assert result.exit_code == 0


class TestEntropyStatusCommand:
    """Tests for the entropy-status command."""

    @patch(_GS_PATCH)
    def test_entropy_status_shows_info(self, mock_get_store):
        """entropy-status shows entropy response status."""
        mock_store = _make_mock_store()
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["entropy-status"])

        assert result.exit_code == 0
