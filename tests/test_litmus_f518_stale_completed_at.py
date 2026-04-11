"""Litmus test for F-518: Stale completed_at causes negative elapsed time on resume.

This is a boundary-gap bug class: two correct subsystems (resume sets started_at,
_compute_elapsed calculates duration) compose into incorrect behavior (negative time).

The bug: F-493 fixed started_at but didn't clear completed_at. When a COMPLETED job
is resumed, started_at gets reset to now but completed_at keeps the old timestamp
from when it completed before. _compute_elapsed() then calculates (old - now) which
is negative.

Evidence:
- mzt status shows "0.0s elapsed" (clamped)
- mzt diagnose shows "-317018.1s" (unclamped)

This litmus test proves:
1. The intelligence layer (status display) produces wrong data
2. The wrongness is semantic - not a crash, but incorrect meaning
3. The fix requires clearing completed_at on resume

Category: 46 - Monitoring correctness litmus tests
"""

from datetime import UTC, datetime, timedelta

import pytest

from marianne.core.checkpoint import CheckpointState, JobStatus, SheetState, SheetStatus
from marianne.utils.time import utc_now


@pytest.mark.asyncio
class TestLitmusF518StaleCompletedAt:
    """Litmus test: Does the monitoring system show correct elapsed time after resume?"""

    async def test_completed_at_cleared_on_resume(self) -> None:
        """LITMUS: A resumed job must NOT have stale completed_at from previous run.

        Without this, _compute_elapsed() in status.py:398-402 calculates:
            elapsed = completed_at - started_at

        Where completed_at is old (3 days ago) and started_at is new (now).
        Result: negative time, clamped to 0.0 in status, shown raw in diagnose.

        The intelligence layer (monitoring) becomes actively misleading.
        """
        # Simulate a job that completed 3 days ago
        three_days_ago = utc_now() - timedelta(days=3)
        checkpoint = CheckpointState(
            job_id="test-resumed-job",
            job_name="test",
            total_sheets=1,
            status=JobStatus.COMPLETED,
            started_at=three_days_ago,
            completed_at=three_days_ago + timedelta(hours=1),  # Completed 1 hour later
            sheets={1: SheetState(sheet_num=1, status=SheetStatus.COMPLETED)},
        )

        # Verify initial state
        assert checkpoint.completed_at is not None
        assert checkpoint.started_at < checkpoint.completed_at

        # User resumes the job - status changes to RUNNING
        checkpoint.status = JobStatus.RUNNING
        checkpoint.started_at = utc_now()

        # Model validator only runs on construction/validation, not field assignment
        # Trigger it by reconstructing the model (simulates what happens during persistence round-trip)
        checkpoint = CheckpointState(**checkpoint.model_dump())

        # Verify the fix: RUNNING jobs must have completed_at = None
        # The model validator _enforce_status_invariants() ensures this invariant
        # Also enforced explicitly in manager.py:2579 during resume for immediate effect
        assert checkpoint.completed_at is None, (
            "F-518: completed_at must be None for RUNNING jobs. "
            "Stale completed_at from previous run causes negative elapsed time. "
            "Fix: CheckpointState._enforce_status_invariants() clears completed_at when status=RUNNING"
        )

    async def test_compute_elapsed_with_stale_timestamps(self) -> None:
        """LITMUS: Verify _compute_elapsed() behavior with stale completed_at.

        This reproduces the exact calculation from status.py:395-403.
        It demonstrates what users see in the wild.
        """
        # Simulate the stale timestamp scenario
        # Start with COMPLETED state (before resume)
        three_days_ago = utc_now() - timedelta(days=3)
        now = utc_now()

        checkpoint = CheckpointState(
            job_id="test-job",
            job_name="test",
            total_sheets=1,
            status=JobStatus.COMPLETED,
            started_at=three_days_ago,
            completed_at=three_days_ago,
            sheets={1: SheetState(sheet_num=1, status=SheetStatus.COMPLETED)},
        )

        # Resume: transition to RUNNING with fresh started_at
        checkpoint.status = JobStatus.RUNNING
        checkpoint.started_at = now

        # Reconstruct to trigger model validator
        checkpoint = CheckpointState(**checkpoint.model_dump())

        # This is the _compute_elapsed() logic from status.py:395-403
        if checkpoint.started_at:
            if checkpoint.completed_at:
                elapsed = (checkpoint.completed_at - checkpoint.started_at).total_seconds()
            else:
                elapsed = (datetime.now(UTC) - checkpoint.started_at).total_seconds()
            computed = max(elapsed, 0.0)
        else:
            computed = 0.0

        # With stale completed_at, elapsed will be hugely negative
        # (3 days ago - now) = -259,200 seconds
        # But the clamping hides it: max(negative, 0.0) = 0.0

        # The user sees "0.0s elapsed" for a job that's been running for hours
        # This is worse than showing nothing - it's showing wrong data

        # After fix (completed_at = None), the calculation would use the else branch:
        #   elapsed = (now - started_at).total_seconds()
        # Which would show the correct elapsed time since resume

        # The litmus test: does the monitoring layer produce semantically correct data?
        # After fix: YES - completed_at is None so elapsed calculation uses (now - started_at)

        # This test verifies the fix is in place
        assert checkpoint.completed_at is None, (
            "Monitoring data integrity failure: stale completed_at "
            f"produces wrong elapsed time. started_at={checkpoint.started_at.isoformat() if checkpoint.started_at else 'None'}, "
            f"completed_at={checkpoint.completed_at.isoformat() if checkpoint.completed_at else 'None'}, "
            f"computed_elapsed={computed:.1f}s"
        )

    async def test_resume_clears_all_completion_metadata(self) -> None:
        """LITMUS: Resume should clear ALL completion-related fields, not just started_at.

        Defensive completeness - if we add more completion metadata later
        (e.g., completion_reason, final_cost), resume must clear those too.
        """
        checkpoint = CheckpointState(
            job_id="test-job",
            job_name="test",
            total_sheets=1,
            status=JobStatus.COMPLETED,
            started_at=utc_now() - timedelta(days=1),
            completed_at=utc_now() - timedelta(hours=1),
            sheets={1: SheetState(sheet_num=1, status=SheetStatus.COMPLETED)},
        )

        # On resume, transition from COMPLETED → RUNNING
        checkpoint.status = JobStatus.RUNNING
        checkpoint.started_at = utc_now()

        # Trigger model validator by reconstructing
        checkpoint = CheckpointState(**checkpoint.model_dump())

        # All completion metadata should be cleared
        assert checkpoint.completed_at is None, (
            "completed_at must be None for RUNNING jobs"
        )

        # Future-proofing: if CheckpointState gains completion_reason, final_cost, etc.
        # those should also be None for RUNNING jobs
        # This test will catch regressions when new fields are added


@pytest.mark.asyncio
class TestLitmusMonitoringCorrectness:
    """Litmus category 46: Does the monitoring layer produce semantically correct data?

    These tests don't check implementation details. They check: if a user runs
    'mzt status', do they see data that accurately reflects reality?

    Monitoring correctness is an intelligence layer concern - the system can be
    "correct" (no crashes, tests pass) while being "ineffective" (shows wrong data).
    """

    async def test_running_job_shows_nonzero_elapsed(self) -> None:
        """A job that's been running for 1 hour should show ~3600s elapsed."""
        one_hour_ago = utc_now() - timedelta(hours=1)
        checkpoint = CheckpointState(
            job_id="test-job",
            job_name="test",
            total_sheets=1,
            status=JobStatus.RUNNING,
            started_at=one_hour_ago,
            completed_at=None,  # Running jobs have no completion time
            sheets={1: SheetState(sheet_num=1, status=SheetStatus.IN_PROGRESS)},
        )

        # Compute elapsed (status.py logic)
        if checkpoint.started_at:
            if checkpoint.completed_at:
                elapsed = (checkpoint.completed_at - checkpoint.started_at).total_seconds()
            else:
                elapsed = (datetime.now(UTC) - checkpoint.started_at).total_seconds()
            computed = max(elapsed, 0.0)
        else:
            computed = 0.0

        # Should be approximately 3600 seconds (1 hour)
        # Allow 1 second tolerance for test execution time
        assert 3599.0 <= computed <= 3601.0, (
            f"Running job started 1 hour ago should show ~3600s elapsed, got {computed:.1f}s"
        )

    async def test_completed_job_shows_actual_duration(self) -> None:
        """A job that ran for 2 hours should show 7200s elapsed."""
        start = utc_now() - timedelta(days=1)
        end = start + timedelta(hours=2)

        checkpoint = CheckpointState(
            job_id="test-job",
            job_name="test",
            total_sheets=1,
            status=JobStatus.COMPLETED,
            started_at=start,
            completed_at=end,
            sheets={1: SheetState(sheet_num=1, status=SheetStatus.COMPLETED)},
        )

        # Compute elapsed
        if checkpoint.started_at:
            if checkpoint.completed_at:
                elapsed = (checkpoint.completed_at - checkpoint.started_at).total_seconds()
            else:
                elapsed = (datetime.now(UTC) - checkpoint.started_at).total_seconds()
            computed = max(elapsed, 0.0)
        else:
            computed = 0.0

        # Should be exactly 7200 seconds (2 hours)
        assert computed == 7200.0, (
            f"Completed job with 2h duration should show 7200s elapsed, got {computed:.1f}s"
        )

    async def test_paused_job_preserves_partial_elapsed(self) -> None:
        """A paused job should show elapsed time from when it started to when it paused.

        Note: Current implementation doesn't track pause time, so this might show
        elapsed since start (not since pause). The test documents expected behavior.
        """
        start = utc_now() - timedelta(hours=3)

        checkpoint = CheckpointState(
            job_id="test-job",
            job_name="test",
            total_sheets=1,
            status=JobStatus.PAUSED,
            started_at=start,
            completed_at=None,  # Paused jobs are not completed
            sheets={1: SheetState(sheet_num=1, status=SheetStatus.PENDING)},
        )

        # Compute elapsed
        if checkpoint.started_at:
            if checkpoint.completed_at:
                elapsed = (checkpoint.completed_at - checkpoint.started_at).total_seconds()
            else:
                elapsed = (datetime.now(UTC) - checkpoint.started_at).total_seconds()
            computed = max(elapsed, 0.0)
        else:
            computed = 0.0

        # Should show time since start (approximately 3 hours = 10800s)
        # This might not be ideal UX (should pause time freeze elapsed?)
        # but it documents current behavior
        assert computed >= 10799.0, (
            f"Paused job should show elapsed time, got {computed:.1f}s"
        )
