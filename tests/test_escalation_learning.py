"""Tests for Escalation Learning Loop (v11 Evolution).

Tests the escalation learning system:
- EscalationDecisionRecord model
- record_escalation_decision() persistence
- get_escalation_history() query
- get_similar_escalation() pattern matching
- update_escalation_outcome() feedback loop
"""

import tempfile
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import pytest

from mozart.learning.global_store import (
    EscalationDecisionRecord,
    GlobalLearningStore,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    yield path
    # Cleanup
    if path.exists():
        path.unlink()


@pytest.fixture
def global_store(temp_db_path: Path) -> Generator[GlobalLearningStore, None, None]:
    """Create a GlobalLearningStore with a temporary database."""
    store = GlobalLearningStore(temp_db_path)
    yield store


# =============================================================================
# TestEscalationDecisionRecord
# =============================================================================


class TestEscalationDecisionRecord:
    """Tests for EscalationDecisionRecord dataclass."""

    def test_record_creation(self) -> None:
        """Test basic record creation."""
        record = EscalationDecisionRecord(
            id="test-id-123",
            job_hash="abc123",
            sheet_num=5,
            confidence=0.45,
            action="retry",
            guidance="Try again with more context",
            validation_pass_rate=60.0,
            retry_count=2,
        )

        assert record.id == "test-id-123"
        assert record.job_hash == "abc123"
        assert record.sheet_num == 5
        assert record.confidence == 0.45
        assert record.action == "retry"
        assert record.guidance == "Try again with more context"
        assert record.validation_pass_rate == 60.0
        assert record.retry_count == 2
        assert record.outcome_after_action is None
        assert record.model is None

    def test_record_with_outcome(self) -> None:
        """Test record with outcome after action."""
        record = EscalationDecisionRecord(
            id="test-id-456",
            job_hash="def456",
            sheet_num=3,
            confidence=0.35,
            action="skip",
            guidance="Skip this validation-heavy sheet",
            validation_pass_rate=30.0,
            retry_count=4,
            outcome_after_action="skipped",
            model="claude-sonnet-4",
        )

        assert record.outcome_after_action == "skipped"
        assert record.model == "claude-sonnet-4"

    def test_record_defaults(self) -> None:
        """Test that optional fields have correct defaults."""
        record = EscalationDecisionRecord(
            id="test-id",
            job_hash="hash",
            sheet_num=1,
            confidence=0.5,
            action="abort",
            guidance=None,
            validation_pass_rate=50.0,
            retry_count=0,
        )

        assert record.guidance is None
        assert record.outcome_after_action is None
        assert record.model is None
        assert isinstance(record.recorded_at, datetime)


# =============================================================================
# TestRecordEscalationDecision
# =============================================================================


class TestRecordEscalationDecision:
    """Tests for record_escalation_decision() method."""

    def test_record_escalation_basic(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test basic escalation recording."""
        record_id = global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=3,
            confidence=0.42,
            action="retry",
            validation_pass_rate=55.0,
            retry_count=1,
        )

        assert record_id is not None
        assert len(record_id) == 36  # UUID length

    def test_record_escalation_with_guidance(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test escalation recording with guidance."""
        record_id = global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=5,
            confidence=0.38,
            action="modify_prompt",
            validation_pass_rate=40.0,
            retry_count=2,
            guidance="Add more specific instructions for file creation",
        )

        # Verify by retrieving
        history = global_store.get_escalation_history(job_id="test-job")
        assert len(history) == 1
        assert history[0].id == record_id
        assert history[0].guidance == "Add more specific instructions for file creation"

    def test_record_escalation_with_model(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test escalation recording with model specified."""
        global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=1,
            confidence=0.55,
            action="skip",
            validation_pass_rate=65.0,
            retry_count=3,
            model="claude-sonnet-4",
        )

        history = global_store.get_escalation_history(job_id="test-job")
        assert len(history) == 1
        assert history[0].model == "claude-sonnet-4"

    def test_record_multiple_escalations(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test recording multiple escalation decisions."""
        for i in range(5):
            global_store.record_escalation_decision(
                job_id="test-job",
                sheet_num=i + 1,
                confidence=0.3 + (i * 0.1),
                action="retry" if i % 2 == 0 else "skip",
                validation_pass_rate=float(i * 10),
                retry_count=i,
            )

        history = global_store.get_escalation_history(job_id="test-job")
        assert len(history) == 5


# =============================================================================
# TestGetEscalationHistory
# =============================================================================


class TestGetEscalationHistory:
    """Tests for get_escalation_history() method."""

    def test_empty_history(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test empty history returns empty list."""
        history = global_store.get_escalation_history()
        assert history == []

    def test_filter_by_job(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test filtering by job ID."""
        # Record for two different jobs
        global_store.record_escalation_decision(
            job_id="job-a",
            sheet_num=1,
            confidence=0.4,
            action="retry",
            validation_pass_rate=50.0,
            retry_count=1,
        )
        global_store.record_escalation_decision(
            job_id="job-b",
            sheet_num=1,
            confidence=0.5,
            action="skip",
            validation_pass_rate=60.0,
            retry_count=2,
        )

        # Filter by job-a
        history_a = global_store.get_escalation_history(job_id="job-a")
        assert len(history_a) == 1
        assert history_a[0].action == "retry"

        # Filter by job-b
        history_b = global_store.get_escalation_history(job_id="job-b")
        assert len(history_b) == 1
        assert history_b[0].action == "skip"

    def test_filter_by_action(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test filtering by action type."""
        actions = ["retry", "skip", "abort", "retry", "modify_prompt"]
        for i, action in enumerate(actions):
            global_store.record_escalation_decision(
                job_id=f"job-{i}",
                sheet_num=i,
                confidence=0.4,
                action=action,
                validation_pass_rate=50.0,
                retry_count=1,
            )

        # Filter by retry
        retries = global_store.get_escalation_history(action="retry")
        assert len(retries) == 2

        # Filter by skip
        skips = global_store.get_escalation_history(action="skip")
        assert len(skips) == 1

    def test_history_ordered_by_recency(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that history is ordered by recency (newest first)."""
        import time

        for i in range(3):
            global_store.record_escalation_decision(
                job_id="test-job",
                sheet_num=i + 1,
                confidence=0.4 + (i * 0.1),
                action="retry",
                validation_pass_rate=50.0,
                retry_count=i,
            )
            time.sleep(0.01)  # Ensure different timestamps

        history = global_store.get_escalation_history(job_id="test-job")

        # Newest (sheet 3) should be first
        assert history[0].sheet_num == 3
        assert history[2].sheet_num == 1

    def test_history_respects_limit(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that limit parameter is respected."""
        for i in range(10):
            global_store.record_escalation_decision(
                job_id="test-job",
                sheet_num=i,
                confidence=0.4,
                action="retry",
                validation_pass_rate=50.0,
                retry_count=i,
            )

        history = global_store.get_escalation_history(limit=5)
        assert len(history) == 5


# =============================================================================
# TestGetSimilarEscalation
# =============================================================================


class TestGetSimilarEscalation:
    """Tests for get_similar_escalation() method."""

    def test_no_similar_when_empty(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test returns empty when no escalations exist."""
        similar = global_store.get_similar_escalation(
            confidence=0.5,
            validation_pass_rate=50.0,
        )
        assert similar == []

    def test_finds_similar_by_confidence(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test finding similar escalations by confidence."""
        # Record with confidence 0.45
        global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=1,
            confidence=0.45,
            action="retry",
            validation_pass_rate=50.0,
            retry_count=1,
        )

        # Search with confidence 0.50 (within default tolerance of 0.15)
        similar = global_store.get_similar_escalation(
            confidence=0.50,
            validation_pass_rate=50.0,
        )

        assert len(similar) == 1
        assert similar[0].confidence == 0.45

    def test_finds_similar_by_pass_rate(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test finding similar escalations by pass rate."""
        # Record with pass rate 60%
        global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=1,
            confidence=0.5,
            action="skip",
            validation_pass_rate=60.0,
            retry_count=2,
        )

        # Search with pass rate 55% (within default tolerance of 15%)
        similar = global_store.get_similar_escalation(
            confidence=0.5,
            validation_pass_rate=55.0,
        )

        assert len(similar) == 1
        assert similar[0].validation_pass_rate == 60.0

    def test_excludes_outside_tolerance(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that escalations outside tolerance are excluded."""
        # Record with confidence 0.2
        global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=1,
            confidence=0.2,
            action="abort",
            validation_pass_rate=20.0,
            retry_count=3,
        )

        # Search with confidence 0.8 (way outside default 0.15 tolerance)
        similar = global_store.get_similar_escalation(
            confidence=0.8,
            validation_pass_rate=80.0,
        )

        assert similar == []

    def test_prioritizes_successful_outcomes(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that successful outcomes are returned first."""
        # Record failed outcome first
        global_store.record_escalation_decision(
            job_id="job-1",
            sheet_num=1,
            confidence=0.45,
            action="retry",
            validation_pass_rate=50.0,
            retry_count=1,
            outcome_after_action="failed",
        )

        # Record successful outcome second
        global_store.record_escalation_decision(
            job_id="job-2",
            sheet_num=1,
            confidence=0.45,
            action="skip",
            validation_pass_rate=50.0,
            retry_count=1,
            outcome_after_action="success",
        )

        # Search should return success first
        similar = global_store.get_similar_escalation(
            confidence=0.45,
            validation_pass_rate=50.0,
        )

        assert len(similar) == 2
        assert similar[0].outcome_after_action == "success"
        assert similar[1].outcome_after_action == "failed"

    def test_custom_tolerance(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test custom tolerance parameters."""
        # Record with confidence 0.3
        global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=1,
            confidence=0.3,
            action="retry",
            validation_pass_rate=30.0,
            retry_count=1,
        )

        # Search with tight tolerance (should not match)
        similar_tight = global_store.get_similar_escalation(
            confidence=0.5,
            validation_pass_rate=50.0,
            confidence_tolerance=0.05,
            pass_rate_tolerance=5.0,
        )
        assert similar_tight == []

        # Search with loose tolerance (should match)
        similar_loose = global_store.get_similar_escalation(
            confidence=0.5,
            validation_pass_rate=50.0,
            confidence_tolerance=0.25,
            pass_rate_tolerance=25.0,
        )
        assert len(similar_loose) == 1


# =============================================================================
# TestUpdateEscalationOutcome
# =============================================================================


class TestUpdateEscalationOutcome:
    """Tests for update_escalation_outcome() method."""

    def test_update_outcome_success(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test updating escalation outcome."""
        record_id = global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=1,
            confidence=0.4,
            action="retry",
            validation_pass_rate=50.0,
            retry_count=1,
        )

        # Update outcome
        updated = global_store.update_escalation_outcome(
            escalation_id=record_id,
            outcome_after_action="success",
        )

        assert updated is True

        # Verify update
        history = global_store.get_escalation_history(job_id="test-job")
        assert history[0].outcome_after_action == "success"

    def test_update_nonexistent_returns_false(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that updating nonexistent record returns False."""
        updated = global_store.update_escalation_outcome(
            escalation_id="nonexistent-id",
            outcome_after_action="failed",
        )

        assert updated is False

    def test_update_multiple_outcomes(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test updating multiple different outcomes."""
        outcomes = [
            ("retry", "success"),
            ("skip", "skipped"),
            ("abort", "aborted"),
        ]

        record_ids = []
        for action, _ in outcomes:
            record_id = global_store.record_escalation_decision(
                job_id="test-job",
                sheet_num=1,
                confidence=0.4,
                action=action,
                validation_pass_rate=50.0,
                retry_count=1,
            )
            record_ids.append(record_id)

        # Update all outcomes
        for i, (_, outcome) in enumerate(outcomes):
            global_store.update_escalation_outcome(record_ids[i], outcome)

        # Verify all updates
        history = global_store.get_escalation_history(job_id="test-job")
        expected_outcomes = {o[1] for o in outcomes}
        actual_outcomes = {h.outcome_after_action for h in history}
        assert actual_outcomes == expected_outcomes


# =============================================================================
# TestEscalationLearningIntegration
# =============================================================================


class TestEscalationLearningIntegration:
    """Integration tests for escalation learning flow."""

    def test_full_escalation_learning_flow(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test complete escalation learning workflow."""
        # 1. Record initial escalation with retry action
        record_id_1 = global_store.record_escalation_decision(
            job_id="evolution-job",
            sheet_num=5,
            confidence=0.42,
            action="retry",
            validation_pass_rate=55.0,
            retry_count=2,
            guidance="User chose to retry with more context",
            model="claude-sonnet-4",
        )

        # 2. Update with successful outcome
        global_store.update_escalation_outcome(record_id_1, "success")

        # 3. New similar escalation occurs
        similar = global_store.get_similar_escalation(
            confidence=0.45,  # Similar to 0.42
            validation_pass_rate=58.0,  # Similar to 55%
        )

        # 4. Should find the previous successful escalation
        assert len(similar) == 1
        assert similar[0].action == "retry"
        assert similar[0].outcome_after_action == "success"
        assert similar[0].guidance == "User chose to retry with more context"

        # 5. Record the new escalation (using the guidance)
        _record_id_2 = global_store.record_escalation_decision(
            job_id="new-job",
            sheet_num=3,
            confidence=0.45,
            action="retry",  # Following the pattern
            validation_pass_rate=58.0,
            retry_count=1,
            guidance="Following pattern from similar escalation",
        )

        # 6. Verify both are in history
        all_history = global_store.get_escalation_history()
        assert len(all_history) == 2

        # 7. Verify can filter by action
        retry_history = global_store.get_escalation_history(action="retry")
        assert len(retry_history) == 2

    def test_schema_v3_migration(self, temp_db_path: Path) -> None:
        """Test that schema v3 creates escalation_decisions table."""
        # Create store (triggers schema creation)
        store = GlobalLearningStore(temp_db_path)

        # Verify table exists by querying it
        record_id = store.record_escalation_decision(
            job_id="test",
            sheet_num=1,
            confidence=0.5,
            action="retry",
            validation_pass_rate=50.0,
            retry_count=0,
        )

        # Should not raise and should return valid ID
        assert record_id is not None

        # Verify schema version is 5 (upgraded in v12 evolution)
        assert store.SCHEMA_VERSION == 5

    def test_clear_all_includes_escalations(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that clear_all() clears escalation decisions."""
        # Record some escalations
        for i in range(3):
            global_store.record_escalation_decision(
                job_id=f"job-{i}",
                sheet_num=i,
                confidence=0.5,
                action="retry",
                validation_pass_rate=50.0,
                retry_count=0,
            )

        # Verify they exist
        assert len(global_store.get_escalation_history()) == 3

        # Clear all
        global_store.clear_all()

        # Verify escalations are cleared
        assert global_store.get_escalation_history() == []


# =============================================================================
# TestEscalationOutcomeUpdate (v13 Evolution)
# =============================================================================


class TestEscalationOutcomeUpdate:
    """Tests for v13 escalation outcome update integration.

    These tests verify that the escalation feedback loop is properly closed
    when sheet execution reaches a final state after an escalation.
    """

    def test_outcome_updated_to_success(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that outcome is updated to 'success' when sheet succeeds after escalation."""
        # Record escalation decision
        record_id = global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=1,
            confidence=0.45,
            action="retry",
            validation_pass_rate=60.0,
            retry_count=2,
            guidance="Retrying with more context",
        )

        # Simulate success outcome
        updated = global_store.update_escalation_outcome(
            escalation_id=record_id,
            outcome_after_action="success",
        )

        assert updated is True

        # Verify the outcome was recorded
        history = global_store.get_escalation_history(job_id="test-job")
        assert len(history) == 1
        assert history[0].outcome_after_action == "success"

    def test_outcome_updated_to_failed_on_retry_exhausted(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that outcome is updated to 'failed' when retries are exhausted."""
        record_id = global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=1,
            confidence=0.35,
            action="retry",
            validation_pass_rate=40.0,
            retry_count=3,
        )

        # Simulate retry exhausted
        updated = global_store.update_escalation_outcome(
            escalation_id=record_id,
            outcome_after_action="failed",
        )

        assert updated is True

        history = global_store.get_escalation_history(job_id="test-job")
        assert history[0].outcome_after_action == "failed"

    def test_outcome_updated_to_skipped(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that outcome is updated to 'skipped' when user skips sheet."""
        record_id = global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=1,
            confidence=0.30,
            action="skip",
            validation_pass_rate=30.0,
            retry_count=2,
        )

        updated = global_store.update_escalation_outcome(
            escalation_id=record_id,
            outcome_after_action="skipped",
        )

        assert updated is True

        history = global_store.get_escalation_history(job_id="test-job")
        assert history[0].outcome_after_action == "skipped"

    def test_outcome_updated_to_aborted(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that outcome is updated to 'aborted' when user aborts job."""
        record_id = global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=1,
            confidence=0.20,
            action="abort",
            validation_pass_rate=20.0,
            retry_count=4,
        )

        updated = global_store.update_escalation_outcome(
            escalation_id=record_id,
            outcome_after_action="aborted",
        )

        assert updated is True

        history = global_store.get_escalation_history(job_id="test-job")
        assert history[0].outcome_after_action == "aborted"

    def test_outcome_not_updated_when_no_record_id(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that updating with invalid ID returns False."""
        # This mimics the case where no escalation occurred
        updated = global_store.update_escalation_outcome(
            escalation_id="nonexistent-id",
            outcome_after_action="success",
        )

        assert updated is False

    def test_outcome_idempotent_update(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that outcome can be updated multiple times (idempotent)."""
        record_id = global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=1,
            confidence=0.45,
            action="retry",
            validation_pass_rate=50.0,
            retry_count=1,
        )

        # First update
        global_store.update_escalation_outcome(record_id, "failed")
        history = global_store.get_escalation_history(job_id="test-job")
        assert history[0].outcome_after_action == "failed"

        # Second update (idempotent - just overwrites)
        updated = global_store.update_escalation_outcome(record_id, "success")
        assert updated is True
        history = global_store.get_escalation_history(job_id="test-job")
        assert history[0].outcome_after_action == "success"

    def test_outcome_correlation_with_action(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that outcome data correlates with action taken.

        This verifies the learning value - that we can analyze which
        actions lead to which outcomes.
        """
        # Record multiple escalations with different actions and outcomes
        outcomes = [
            ("retry", "success"),
            ("retry", "failed"),
            ("skip", "skipped"),
            ("abort", "aborted"),
        ]

        record_ids = []
        for action, _ in outcomes:
            record_id = global_store.record_escalation_decision(
                job_id="correlation-test",
                sheet_num=1,
                confidence=0.40,
                action=action,
                validation_pass_rate=50.0,
                retry_count=1,
            )
            record_ids.append(record_id)

        # Update all outcomes
        for i, (_, outcome) in enumerate(outcomes):
            global_store.update_escalation_outcome(record_ids[i], outcome)

        # Verify we can analyze the results
        history = global_store.get_escalation_history(job_id="correlation-test")
        assert len(history) == 4

        # Count outcomes by action (for learning analysis)
        retry_outcomes = [h.outcome_after_action for h in history if h.action == "retry"]
        assert set(retry_outcomes) == {"success", "failed"}

        skip_outcomes = [h.outcome_after_action for h in history if h.action == "skip"]
        assert skip_outcomes == ["skipped"]

        abort_outcomes = [h.outcome_after_action for h in history if h.action == "abort"]
        assert abort_outcomes == ["aborted"]


# =============================================================================
# TestRunnerEscalationOutcomeIntegration (v13 Evolution)
# =============================================================================


class TestRunnerEscalationOutcomeIntegration:
    """Integration tests for runner's _update_escalation_outcome method.

    v13 Evolution: Escalation Feedback Loop - these tests verify that the
    JobRunner correctly calls update_escalation_outcome at all escalation
    outcome points (success, failed, skipped, aborted).
    """

    def test_update_escalation_outcome_returns_early_if_no_store(self) -> None:
        """Test that _update_escalation_outcome returns early if global learning store is None."""
        from unittest.mock import MagicMock

        from mozart.core.checkpoint import SheetState, SheetStatus
        from mozart.execution.runner import JobRunner

        # Create a mock runner with no global learning store
        runner = MagicMock(spec=JobRunner)
        runner._global_learning_store = None
        runner._logger = MagicMock()

        # Create sheet state with escalation record ID
        sheet_state = SheetState(
            sheet_num=1,
            status=SheetStatus.COMPLETED,
            outcome_data={"escalation_record_id": "test-record-123"},
        )

        # Call the method directly (unbound)
        JobRunner._update_escalation_outcome(runner, sheet_state, "success", 1)

        # Should not log anything (early return)
        runner._logger.info.assert_not_called()
        runner._logger.warning.assert_not_called()

    def test_update_escalation_outcome_returns_early_if_no_record_id(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that _update_escalation_outcome returns early if no escalation_record_id."""
        from unittest.mock import MagicMock

        from mozart.core.checkpoint import SheetState, SheetStatus
        from mozart.execution.runner import JobRunner

        runner = MagicMock(spec=JobRunner)
        runner._global_learning_store = global_store
        runner._logger = MagicMock()

        # Create sheet state WITHOUT escalation record ID
        sheet_state = SheetState(
            sheet_num=1,
            status=SheetStatus.COMPLETED,
            outcome_data={},  # No escalation_record_id
        )

        JobRunner._update_escalation_outcome(runner, sheet_state, "success", 1)

        # Should not call update_escalation_outcome
        runner._logger.info.assert_not_called()

    def test_update_escalation_outcome_returns_early_if_outcome_data_none(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that _update_escalation_outcome handles outcome_data being None."""
        from unittest.mock import MagicMock

        from mozart.core.checkpoint import SheetState, SheetStatus
        from mozart.execution.runner import JobRunner

        runner = MagicMock(spec=JobRunner)
        runner._global_learning_store = global_store
        runner._logger = MagicMock()

        # Create sheet state with outcome_data=None (default)
        sheet_state = SheetState(
            sheet_num=1,
            status=SheetStatus.COMPLETED,
            outcome_data=None,
        )

        JobRunner._update_escalation_outcome(runner, sheet_state, "success", 1)

        # Should not call update_escalation_outcome
        runner._logger.info.assert_not_called()

    def test_update_escalation_outcome_calls_store_correctly(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that _update_escalation_outcome calls the store with correct parameters."""
        from unittest.mock import MagicMock

        from mozart.core.checkpoint import SheetState, SheetStatus
        from mozart.execution.runner import JobRunner

        # First, record an escalation decision to get a valid ID
        record_id = global_store.record_escalation_decision(
            job_id="test-job",
            sheet_num=1,
            confidence=0.45,
            action="retry",
            validation_pass_rate=60.0,
            retry_count=2,
        )

        runner = MagicMock(spec=JobRunner)
        runner._global_learning_store = global_store
        runner._logger = MagicMock()

        # Create sheet state with the actual escalation record ID
        sheet_state = SheetState(
            sheet_num=1,
            status=SheetStatus.COMPLETED,
            outcome_data={"escalation_record_id": record_id},
        )

        JobRunner._update_escalation_outcome(runner, sheet_state, "success", 1)

        # Verify outcome was updated
        history = global_store.get_escalation_history(job_id="test-job")
        assert len(history) == 1
        assert history[0].outcome_after_action == "success"

        # Verify info was logged
        runner._logger.info.assert_called_once()
        call_args = runner._logger.info.call_args
        assert call_args[0][0] == "escalation.outcome_updated"

    def test_update_escalation_outcome_logs_warning_on_not_found(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test that _update_escalation_outcome logs warning when record not found."""
        from unittest.mock import MagicMock

        from mozart.core.checkpoint import SheetState, SheetStatus
        from mozart.execution.runner import JobRunner

        runner = MagicMock(spec=JobRunner)
        runner._global_learning_store = global_store
        runner._logger = MagicMock()

        # Create sheet state with nonexistent escalation record ID
        sheet_state = SheetState(
            sheet_num=1,
            status=SheetStatus.COMPLETED,
            outcome_data={"escalation_record_id": "nonexistent-record-id"},
        )

        JobRunner._update_escalation_outcome(runner, sheet_state, "success", 1)

        # Verify warning was logged
        runner._logger.warning.assert_called_once()
        call_args = runner._logger.warning.call_args
        assert call_args[0][0] == "escalation.outcome_update_not_found"

    def test_update_escalation_outcome_logs_warning_on_exception(self) -> None:
        """Test that _update_escalation_outcome logs warning on exception."""
        from unittest.mock import MagicMock

        from mozart.core.checkpoint import SheetState, SheetStatus
        from mozart.execution.runner import JobRunner

        # Create a mock store that raises an exception
        mock_store = MagicMock()
        mock_store.update_escalation_outcome.side_effect = Exception("DB error")

        runner = MagicMock(spec=JobRunner)
        runner._global_learning_store = mock_store
        runner._logger = MagicMock()

        sheet_state = SheetState(
            sheet_num=1,
            status=SheetStatus.COMPLETED,
            outcome_data={"escalation_record_id": "test-record-123"},
        )

        JobRunner._update_escalation_outcome(runner, sheet_state, "success", 1)

        # Verify warning was logged
        runner._logger.warning.assert_called_once()
        call_args = runner._logger.warning.call_args
        assert call_args[0][0] == "escalation.outcome_update_failed"

    def test_update_escalation_outcome_success_after_retry(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test escalation outcome updated to 'success' when sheet succeeds after retry."""
        from unittest.mock import MagicMock

        from mozart.core.checkpoint import SheetState, SheetStatus
        from mozart.execution.runner import JobRunner

        # Record escalation with retry action
        record_id = global_store.record_escalation_decision(
            job_id="retry-success-job",
            sheet_num=5,
            confidence=0.42,
            action="retry",
            validation_pass_rate=55.0,
            retry_count=2,
            guidance="User chose retry",
        )

        runner = MagicMock(spec=JobRunner)
        runner._global_learning_store = global_store
        runner._logger = MagicMock()

        sheet_state = SheetState(
            sheet_num=5,
            status=SheetStatus.COMPLETED,
            outcome_data={"escalation_record_id": record_id},
        )

        # Simulate success outcome after retry
        JobRunner._update_escalation_outcome(runner, sheet_state, "success", 5)

        history = global_store.get_escalation_history(job_id="retry-success-job")
        assert len(history) == 1
        assert history[0].action == "retry"
        assert history[0].outcome_after_action == "success"

    def test_update_escalation_outcome_failed_on_retry_exhausted(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test escalation outcome updated to 'failed' when retries exhausted."""
        from unittest.mock import MagicMock

        from mozart.core.checkpoint import SheetState, SheetStatus
        from mozart.execution.runner import JobRunner

        record_id = global_store.record_escalation_decision(
            job_id="retry-failed-job",
            sheet_num=3,
            confidence=0.35,
            action="retry",
            validation_pass_rate=40.0,
            retry_count=3,
        )

        runner = MagicMock(spec=JobRunner)
        runner._global_learning_store = global_store
        runner._logger = MagicMock()

        sheet_state = SheetState(
            sheet_num=3,
            status=SheetStatus.FAILED,
            outcome_data={"escalation_record_id": record_id},
        )

        JobRunner._update_escalation_outcome(runner, sheet_state, "failed", 3)

        history = global_store.get_escalation_history(job_id="retry-failed-job")
        assert history[0].outcome_after_action == "failed"

    def test_update_escalation_outcome_skipped(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test escalation outcome updated to 'skipped' when user skips."""
        from unittest.mock import MagicMock

        from mozart.core.checkpoint import SheetState, SheetStatus
        from mozart.execution.runner import JobRunner

        record_id = global_store.record_escalation_decision(
            job_id="skip-job",
            sheet_num=2,
            confidence=0.30,
            action="skip",
            validation_pass_rate=30.0,
            retry_count=2,
        )

        runner = MagicMock(spec=JobRunner)
        runner._global_learning_store = global_store
        runner._logger = MagicMock()

        sheet_state = SheetState(
            sheet_num=2,
            status=SheetStatus.SKIPPED,
            outcome_data={"escalation_record_id": record_id},
        )

        JobRunner._update_escalation_outcome(runner, sheet_state, "skipped", 2)

        history = global_store.get_escalation_history(job_id="skip-job")
        assert history[0].outcome_after_action == "skipped"

    def test_update_escalation_outcome_aborted(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test escalation outcome updated to 'aborted' when user aborts."""
        from unittest.mock import MagicMock

        from mozart.core.checkpoint import SheetState, SheetStatus
        from mozart.execution.runner import JobRunner

        record_id = global_store.record_escalation_decision(
            job_id="abort-job",
            sheet_num=1,
            confidence=0.20,
            action="abort",
            validation_pass_rate=20.0,
            retry_count=4,
        )

        runner = MagicMock(spec=JobRunner)
        runner._global_learning_store = global_store
        runner._logger = MagicMock()

        sheet_state = SheetState(
            sheet_num=1,
            status=SheetStatus.FAILED,
            outcome_data={"escalation_record_id": record_id},
        )

        JobRunner._update_escalation_outcome(runner, sheet_state, "aborted", 1)

        history = global_store.get_escalation_history(job_id="abort-job")
        assert history[0].outcome_after_action == "aborted"

    def test_escalation_feedback_loop_complete_workflow(
        self, global_store: GlobalLearningStore
    ) -> None:
        """Test complete escalation feedback loop: record → update → query.

        This is an end-to-end test that validates the entire feedback loop
        works correctly when called through the runner integration.
        """
        from unittest.mock import MagicMock

        from mozart.core.checkpoint import SheetState, SheetStatus
        from mozart.execution.runner import JobRunner

        # 1. Simulate escalation recording (what _handle_escalation does)
        record_id = global_store.record_escalation_decision(
            job_id="e2e-feedback-job",
            sheet_num=7,
            confidence=0.45,
            action="retry",
            validation_pass_rate=60.0,
            retry_count=1,
            guidance="Suggested retry with more context",
            model="claude-sonnet-4",
        )

        # 2. Create runner mock
        runner = MagicMock(spec=JobRunner)
        runner._global_learning_store = global_store
        runner._logger = MagicMock()

        # 3. Create sheet state as it would be after escalation handling
        sheet_state = SheetState(
            sheet_num=7,
            status=SheetStatus.COMPLETED,
            outcome_data={"escalation_record_id": record_id},
        )

        # 4. Update outcome (what runner does after sheet completion)
        JobRunner._update_escalation_outcome(runner, sheet_state, "success", 7)

        # 5. Verify the feedback loop is complete
        history = global_store.get_escalation_history(job_id="e2e-feedback-job")
        assert len(history) == 1
        assert history[0].id == record_id
        assert history[0].action == "retry"
        assert history[0].outcome_after_action == "success"
        assert history[0].model == "claude-sonnet-4"

        # 6. Verify similar escalation lookup now returns this data
        similar = global_store.get_similar_escalation(
            confidence=0.45,
            validation_pass_rate=60.0,
        )
        assert len(similar) == 1
        assert similar[0].outcome_after_action == "success"
