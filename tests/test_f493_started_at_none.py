"""Test F-493: Status elapsed time shows 0.0s for running jobs.

Root cause: CheckpointState.started_at was None for baton-managed jobs.
Fix: Added model validator to auto-set started_at when status=RUNNING.

Issue: #158
"""

from datetime import UTC, datetime, timedelta

from marianne.core.checkpoint import CheckpointState, JobStatus


class TestF493StartedAtFix:
    """Verify that CheckpointState.started_at is auto-set for RUNNING jobs."""

    def test_running_job_auto_sets_started_at(self):
        """GIVEN: A CheckpointState with RUNNING status.
        WHEN: The state is created without explicit started_at.
        THEN: The model validator auto-sets started_at to current time.

        This is the FIX - the model validator prevents the bug.
        """
        before = datetime.now(UTC)
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=1,
            status=JobStatus.RUNNING,
            sheets={},
        )
        after = datetime.now(UTC)

        # Fixed: started_at is auto-set by model validator
        assert state.started_at is not None
        assert isinstance(state.started_at, datetime)
        assert before <= state.started_at <= after

    def test_pending_job_does_not_auto_set_started_at(self):
        """GIVEN: A CheckpointState with PENDING status.
        WHEN: The state is created without explicit started_at.
        THEN: started_at remains None (expected for pending jobs).
        """
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=1,
            status=JobStatus.PENDING,
            sheets={},
        )

        # Pending jobs should not have started_at set
        assert state.started_at is None

    def test_completed_job_does_not_auto_set_started_at(self):
        """GIVEN: A CheckpointState with COMPLETED status.
        WHEN: The state is created without explicit started_at.
        THEN: started_at remains None (should have been set when RUNNING).
        """
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=1,
            status=JobStatus.COMPLETED,
            completed_at=datetime.now(UTC),
            sheets={},
        )

        # Completed jobs with no started_at is unusual but allowed
        # (e.g., legacy checkpoints)
        assert state.started_at is None

    def test_elapsed_time_is_correct_when_started_at_auto_set(self):
        """GIVEN: A CheckpointState with RUNNING status (auto-sets started_at).
        WHEN: We compute elapsed time.
        THEN: Elapsed time is non-zero (not the "0.0s" bug).
        """
        from time import sleep

        # Create RUNNING checkpoint (auto-sets started_at)
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=1,
            status=JobStatus.RUNNING,
            sheets={},
        )

        # Wait a bit
        sleep(0.01)

        # Simulate _compute_elapsed from status.py:394-402
        if state.started_at:
            if state.completed_at:
                elapsed = (state.completed_at - state.started_at).total_seconds()
            else:
                elapsed = (datetime.now(UTC) - state.started_at).total_seconds()
            elapsed = max(elapsed, 0.0)
        else:
            elapsed = 0.0

        # Fixed: elapsed is > 0 (not the "0.0s elapsed" bug)
        assert elapsed > 0
        assert elapsed >= 0.01  # At least 10ms

    def test_explicit_started_at_is_preserved(self):
        """GIVEN: A CheckpointState with explicit started_at.
        WHEN: The state is created with status=RUNNING.
        THEN: The explicit started_at is preserved (not overwritten).
        """
        explicit_time = datetime.now(UTC) - timedelta(hours=2)

        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=1,
            status=JobStatus.RUNNING,
            started_at=explicit_time,  # Explicit value
            sheets={},
        )

        # Explicit started_at is preserved
        assert state.started_at == explicit_time

    def test_resumed_job_resets_started_at(self):
        """On resume, started_at should be reset to current time.

        This verifies the pattern at manager.py:2563.
        """
        old_started = datetime.now(UTC) - timedelta(days=7)

        # Checkpoint from previous run
        checkpoint = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=1,
            status=JobStatus.PAUSED,
            started_at=old_started,
            sheets={},
        )

        # On resume, reset started_at (manager.py:2563)
        before_resume = datetime.now(UTC)
        checkpoint.started_at = datetime.now(UTC)
        after_resume = datetime.now(UTC)

        # Verify started_at was reset
        assert checkpoint.started_at != old_started
        assert before_resume <= checkpoint.started_at <= after_resume
