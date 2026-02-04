"""Simplified test for runner pause functionality."""
import tempfile
from pathlib import Path

import pytest
from mozart.core.checkpoint import CheckpointState, JobStatus
from mozart.execution.runner import JobRunner


def test_pause_signal_detection():
    """Test basic pause signal detection."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_workspace = Path(temp_dir)

        # Create a simple state
        state = CheckpointState(
            job_id="test-123",
            job_name="test",
            total_sheets=3,
            last_completed_sheet=0,
            status=JobStatus.RUNNING,
        )

        # Create a minimal runner instance (we're only testing the pause detection method)
        runner = JobRunner(
            config=None,  # We'll only test methods that don't need config
            backend=None,
            state_backend=None,
        )

        # Mock the config workspace
        runner.config = type('MockConfig', (), {'workspace': str(temp_workspace)})()

        # Test 1: No pause signal file - should return False
        assert not runner._check_pause_signal(state)

        # Test 2: Create pause signal file - should return True
        pause_signal_file = temp_workspace / f".mozart-pause-{state.job_id}"
        pause_signal_file.touch()
        assert runner._check_pause_signal(state)

        # Test 3: Clear pause signal - should remove file
        runner._clear_pause_signal(state)
        assert not pause_signal_file.exists()

        print("All basic pause signal tests passed!")


if __name__ == "__main__":
    test_pause_signal_detection()