"""Tests for mozart.core.checkpoint module."""

from datetime import datetime

import pytest

from mozart.core.checkpoint import (
    BatchState,
    BatchStatus,
    CheckpointState,
    JobStatus,
)


class TestBatchStatus:
    """Tests for BatchStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert BatchStatus.PENDING == "pending"
        assert BatchStatus.IN_PROGRESS == "in_progress"
        assert BatchStatus.COMPLETED == "completed"
        assert BatchStatus.FAILED == "failed"
        assert BatchStatus.SKIPPED == "skipped"


class TestBatchState:
    """Tests for BatchState model."""

    def test_default_state(self):
        """Test default batch state."""
        state = BatchState(batch_num=1)
        assert state.batch_num == 1
        assert state.status == BatchStatus.PENDING
        assert state.started_at is None
        assert state.completed_at is None
        assert state.attempt_count == 0
        assert state.validation_passed is None
        assert state.completion_attempts == 0

    def test_learning_fields(self):
        """Test learning metadata fields (Phase 1)."""
        state = BatchState(
            batch_num=1,
            confidence_score=0.85,
            first_attempt_success=True,
            outcome_category="success_first_try",
            learned_patterns=["pattern1", "pattern2"],
        )
        assert state.confidence_score == 0.85
        assert state.first_attempt_success is True
        assert state.outcome_category == "success_first_try"
        assert len(state.learned_patterns) == 2

    def test_confidence_score_bounds(self):
        """Test confidence score must be between 0 and 1."""
        # Valid values
        BatchState(batch_num=1, confidence_score=0.0)
        BatchState(batch_num=1, confidence_score=1.0)
        BatchState(batch_num=1, confidence_score=0.5)

        # Invalid values should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            BatchState(batch_num=1, confidence_score=1.5)

        with pytest.raises(Exception):
            BatchState(batch_num=1, confidence_score=-0.1)


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_values(self):
        """Test all job status values exist."""
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.PAUSED == "paused"
        assert JobStatus.CANCELLED == "cancelled"


class TestCheckpointState:
    """Tests for CheckpointState model."""

    def _create_state(
        self, job_id: str = "test-job", job_name: str = "Test", total_batches: int = 3
    ) -> CheckpointState:
        """Helper to create a CheckpointState."""
        state = CheckpointState(
            job_id=job_id,
            job_name=job_name,
            total_batches=total_batches,
            batches={i: BatchState(batch_num=i) for i in range(1, total_batches + 1)},
        )
        return state

    def test_create_state(self):
        """Test creating checkpoint state for a new job."""
        state = self._create_state(
            job_id="test-job-123",
            job_name="Test Job",
            total_batches=5,
        )
        assert state.job_id == "test-job-123"
        assert state.job_name == "Test Job"
        assert state.total_batches == 5
        assert state.status == JobStatus.PENDING
        assert state.last_completed_batch == 0
        assert len(state.batches) == 5

    def test_batches_initialized(self):
        """Test all batches are initialized with PENDING status."""
        state = self._create_state(total_batches=3)
        for i in range(1, 4):
            assert i in state.batches
            assert state.batches[i].status == BatchStatus.PENDING

    def test_get_next_batch_initial(self):
        """Test get_next_batch returns 1 for new job."""
        state = self._create_state(total_batches=3)
        assert state.get_next_batch() == 1

    def test_get_next_batch_after_completion(self):
        """Test get_next_batch returns correct batch after completion."""
        state = self._create_state(total_batches=3)
        state.mark_batch_started(1)
        state.mark_batch_completed(1)
        assert state.get_next_batch() == 2

    def test_get_next_batch_all_completed(self):
        """Test get_next_batch returns None when all complete."""
        state = self._create_state(total_batches=2)
        state.mark_batch_started(1)
        state.mark_batch_completed(1)
        state.mark_batch_started(2)
        state.mark_batch_completed(2)
        assert state.get_next_batch() is None

    def test_mark_batch_started(self):
        """Test marking a batch as started."""
        state = self._create_state(total_batches=3)
        state.mark_batch_started(1)
        assert state.batches[1].status == BatchStatus.IN_PROGRESS
        assert state.batches[1].started_at is not None
        assert state.batches[1].attempt_count == 1
        assert state.current_batch == 1
        assert state.status == JobStatus.RUNNING

    def test_mark_batch_completed(self):
        """Test marking a batch as completed."""
        state = self._create_state(total_batches=3)
        state.mark_batch_started(1)
        state.mark_batch_completed(1, validation_passed=True)

        assert state.batches[1].status == BatchStatus.COMPLETED
        assert state.batches[1].completed_at is not None
        assert state.batches[1].validation_passed is True
        assert state.last_completed_batch == 1

    def test_mark_batch_failed(self):
        """Test marking a batch as failed."""
        state = self._create_state(total_batches=3)
        state.mark_batch_started(1)
        state.mark_batch_failed(1, error_message="Test error", error_category="unknown")

        assert state.batches[1].status == BatchStatus.FAILED
        assert state.batches[1].error_message == "Test error"
        assert state.batches[1].error_category == "unknown"

    def test_job_completes_when_all_batches_done(self):
        """Test job status updates to COMPLETED when all batches are done."""
        state = self._create_state(total_batches=2)
        state.mark_batch_started(1)
        state.mark_batch_completed(1)
        assert state.status == JobStatus.RUNNING

        state.mark_batch_started(2)
        state.mark_batch_completed(2)
        assert state.status == JobStatus.COMPLETED
        assert state.completed_at is not None

    def test_retry_tracking(self):
        """Test retry attempt tracking."""
        state = self._create_state(total_batches=1)
        # First attempt
        state.mark_batch_started(1)
        assert state.batches[1].attempt_count == 1

        # Retry
        state.mark_batch_started(1)
        assert state.batches[1].attempt_count == 2

    def test_get_progress(self):
        """Test progress tracking."""
        state = self._create_state(total_batches=5)
        state.mark_batch_started(1)
        state.mark_batch_completed(1)
        state.mark_batch_started(2)
        state.mark_batch_completed(2)

        completed, total = state.get_progress()
        assert completed == 2
        assert total == 5

    def test_get_progress_percent(self):
        """Test progress percentage calculation."""
        state = self._create_state(total_batches=4)
        state.mark_batch_started(1)
        state.mark_batch_completed(1)
        state.mark_batch_started(2)
        state.mark_batch_completed(2)

        assert state.get_progress_percent() == 50.0


class TestCheckpointStateSerialization:
    """Tests for CheckpointState serialization."""

    def test_to_dict(self):
        """Test converting state to dictionary."""
        state = CheckpointState(
            job_id="test-job",
            job_name="Test",
            total_batches=2,
            batches={i: BatchState(batch_num=i) for i in range(1, 3)},
        )
        data = state.model_dump()
        assert data["job_id"] == "test-job"
        assert data["total_batches"] == 2
        assert "batches" in data

    def test_from_dict(self):
        """Test loading state from dictionary."""
        state = CheckpointState(
            job_id="test-job",
            job_name="Test",
            total_batches=2,
            batches={i: BatchState(batch_num=i) for i in range(1, 3)},
        )
        state.mark_batch_started(1)
        state.mark_batch_completed(1)

        data = state.model_dump()
        loaded = CheckpointState.model_validate(data)

        assert loaded.job_id == state.job_id
        assert loaded.last_completed_batch == 1
        assert loaded.batches[1].status == BatchStatus.COMPLETED
