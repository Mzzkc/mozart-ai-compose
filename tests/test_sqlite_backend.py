"""Tests for SQLite state backend."""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mozart.core.checkpoint import SheetStatus, CheckpointState, JobStatus
from mozart.state.sqlite_backend import SQLiteStateBackend


@pytest.fixture
async def sqlite_backend(tmp_path: Path) -> SQLiteStateBackend:
    """Create a SQLite backend with temp database."""
    db_path = tmp_path / "state.db"
    backend = SQLiteStateBackend(db_path)
    return backend


@pytest.fixture
def sample_state() -> CheckpointState:
    """Create a sample checkpoint state."""
    return CheckpointState(
        job_id="test-job-123",
        job_name="Test Job",
        total_sheets=5,
        status=JobStatus.PENDING,
    )


class TestSQLiteBackendBasics:
    """Test basic CRUD operations."""

    async def test_save_and_load_job(
        self, sqlite_backend: SQLiteStateBackend, sample_state: CheckpointState
    ) -> None:
        """Test saving and loading a job state."""
        await sqlite_backend.save(sample_state)

        loaded = await sqlite_backend.load(sample_state.job_id)

        assert loaded is not None
        assert loaded.job_id == sample_state.job_id
        assert loaded.job_name == sample_state.job_name
        assert loaded.total_sheets == sample_state.total_sheets
        assert loaded.status == JobStatus.PENDING

    async def test_load_nonexistent_returns_none(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test loading a non-existent job returns None."""
        result = await sqlite_backend.load("nonexistent-job")
        assert result is None

    async def test_delete_job(
        self, sqlite_backend: SQLiteStateBackend, sample_state: CheckpointState
    ) -> None:
        """Test deleting a job state."""
        await sqlite_backend.save(sample_state)

        # Verify it exists
        assert await sqlite_backend.load(sample_state.job_id) is not None

        # Delete it
        result = await sqlite_backend.delete(sample_state.job_id)
        assert result is True

        # Verify it's gone
        assert await sqlite_backend.load(sample_state.job_id) is None

    async def test_delete_nonexistent_returns_false(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test deleting non-existent job returns False."""
        result = await sqlite_backend.delete("nonexistent-job")
        assert result is False

    async def test_list_jobs(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test listing all jobs."""
        # Create multiple jobs
        for i in range(3):
            state = CheckpointState(
                job_id=f"job-{i}",
                job_name=f"Job {i}",
                total_sheets=5,
            )
            await sqlite_backend.save(state)

        jobs = await sqlite_backend.list_jobs()

        assert len(jobs) == 3
        job_ids = {j.job_id for j in jobs}
        assert job_ids == {"job-0", "job-1", "job-2"}


class TestBatchOperations:
    """Test batch-level operations."""

    async def test_get_next_sheet_new_job(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test get_next_sheet returns 1 for new job."""
        result = await sqlite_backend.get_next_sheet("new-job")
        assert result == 1

    async def test_get_next_sheet_with_state(
        self, sqlite_backend: SQLiteStateBackend, sample_state: CheckpointState
    ) -> None:
        """Test get_next_sheet returns correct batch number."""
        sample_state.last_completed_sheet = 2
        await sqlite_backend.save(sample_state)

        result = await sqlite_backend.get_next_sheet(sample_state.job_id)
        assert result == 3

    async def test_mark_sheet_in_progress(
        self, sqlite_backend: SQLiteStateBackend, sample_state: CheckpointState
    ) -> None:
        """Test marking a batch as in progress."""
        await sqlite_backend.save(sample_state)

        await sqlite_backend.mark_sheet_status(
            sample_state.job_id, 1, SheetStatus.IN_PROGRESS
        )

        loaded = await sqlite_backend.load(sample_state.job_id)
        assert loaded is not None
        assert loaded.status == JobStatus.RUNNING
        assert loaded.current_sheet == 1
        assert 1 in loaded.sheets
        assert loaded.sheets[1].status == SheetStatus.IN_PROGRESS

    async def test_mark_sheet_completed(
        self, sqlite_backend: SQLiteStateBackend, sample_state: CheckpointState
    ) -> None:
        """Test marking a batch as completed."""
        sample_state.mark_sheet_started(1)
        await sqlite_backend.save(sample_state)

        await sqlite_backend.mark_sheet_status(
            sample_state.job_id, 1, SheetStatus.COMPLETED
        )

        loaded = await sqlite_backend.load(sample_state.job_id)
        assert loaded is not None
        assert loaded.last_completed_sheet == 1
        assert loaded.sheets[1].status == SheetStatus.COMPLETED

    async def test_mark_sheet_failed(
        self, sqlite_backend: SQLiteStateBackend, sample_state: CheckpointState
    ) -> None:
        """Test marking a batch as failed with error message."""
        sample_state.mark_sheet_started(1)
        await sqlite_backend.save(sample_state)

        await sqlite_backend.mark_sheet_status(
            sample_state.job_id, 1, SheetStatus.FAILED, "Rate limit exceeded"
        )

        loaded = await sqlite_backend.load(sample_state.job_id)
        assert loaded is not None
        assert loaded.sheets[1].status == SheetStatus.FAILED
        assert loaded.sheets[1].error_message == "Rate limit exceeded"

    async def test_mark_sheet_nonexistent_job_raises(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test marking batch on non-existent job raises ValueError."""
        with pytest.raises(ValueError, match="No state found"):
            await sqlite_backend.mark_sheet_status(
                "nonexistent", 1, SheetStatus.COMPLETED
            )


class TestSheetStatePreservation:
    """Test that all SheetState fields are preserved."""

    async def test_batch_learning_fields_preserved(
        self, sqlite_backend: SQLiteStateBackend, sample_state: CheckpointState
    ) -> None:
        """Test that learning metadata fields are preserved."""
        from mozart.core.checkpoint import SheetState

        sample_state.sheets[1] = SheetState(
            sheet_num=1,
            status=SheetStatus.COMPLETED,
            confidence_score=0.85,
            learned_patterns=["pattern1", "pattern2"],
            similar_outcomes_count=5,
            first_attempt_success=True,
            outcome_category="success_first_try",
            outcome_data={"key": "value", "nested": {"a": 1}},
        )
        await sqlite_backend.save(sample_state)

        loaded = await sqlite_backend.load(sample_state.job_id)
        assert loaded is not None
        batch = loaded.sheets[1]

        assert batch.confidence_score == 0.85
        assert batch.learned_patterns == ["pattern1", "pattern2"]
        assert batch.similar_outcomes_count == 5
        assert batch.first_attempt_success is True
        assert batch.outcome_category == "success_first_try"
        assert batch.outcome_data == {"key": "value", "nested": {"a": 1}}

    async def test_batch_validation_fields_preserved(
        self, sqlite_backend: SQLiteStateBackend, sample_state: CheckpointState
    ) -> None:
        """Test that validation fields are preserved."""
        from mozart.core.checkpoint import SheetState

        sample_state.sheets[1] = SheetState(
            sheet_num=1,
            status=SheetStatus.COMPLETED,
            validation_passed=True,
            validation_details=[{"type": "file_exists", "passed": True}],
            passed_validations=["File check", "Content check"],
            failed_validations=["Size check"],
            last_pass_percentage=66.7,
        )
        await sqlite_backend.save(sample_state)

        loaded = await sqlite_backend.load(sample_state.job_id)
        assert loaded is not None
        batch = loaded.sheets[1]

        assert batch.validation_passed is True
        assert batch.validation_details == [{"type": "file_exists", "passed": True}]
        assert batch.passed_validations == ["File check", "Content check"]
        assert batch.failed_validations == ["Size check"]
        assert batch.last_pass_percentage == 66.7


class TestExecutionHistory:
    """Test execution history recording."""

    async def test_record_execution(
        self, sqlite_backend: SQLiteStateBackend, sample_state: CheckpointState
    ) -> None:
        """Test recording an execution attempt."""
        await sqlite_backend.save(sample_state)

        record_id = await sqlite_backend.record_execution(
            job_id=sample_state.job_id,
            sheet_num=1,
            attempt_num=1,
            prompt="Test prompt",
            output="Test output",
            exit_code=0,
            duration_seconds=5.5,
        )

        assert record_id > 0

    async def test_get_execution_history(
        self, sqlite_backend: SQLiteStateBackend, sample_state: CheckpointState
    ) -> None:
        """Test retrieving execution history."""
        await sqlite_backend.save(sample_state)

        # Record multiple executions
        for i in range(3):
            await sqlite_backend.record_execution(
                job_id=sample_state.job_id,
                sheet_num=1,
                attempt_num=i + 1,
                prompt=f"Prompt {i}",
                output=f"Output {i}",
                exit_code=0 if i == 2 else 1,
                duration_seconds=float(i + 1),
            )

        history = await sqlite_backend.get_execution_history(
            sample_state.job_id, sheet_num=1
        )

        assert len(history) == 3
        # Most recent first
        assert history[0]["attempt_num"] == 3
        assert history[0]["exit_code"] == 0

    async def test_get_execution_history_all_batches(
        self, sqlite_backend: SQLiteStateBackend, sample_state: CheckpointState
    ) -> None:
        """Test retrieving execution history for all batches."""
        await sqlite_backend.save(sample_state)

        # Record for multiple batches
        for batch in range(1, 4):
            await sqlite_backend.record_execution(
                job_id=sample_state.job_id,
                sheet_num=batch,
                attempt_num=1,
                exit_code=0,
            )

        history = await sqlite_backend.get_execution_history(sample_state.job_id)

        assert len(history) == 3


class TestJobStatistics:
    """Test job statistics calculation."""

    async def test_get_job_statistics(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test getting job statistics."""
        state = CheckpointState(
            job_id="stats-job",
            job_name="Stats Test",
            total_sheets=10,
            last_completed_sheet=7,
            total_retry_count=3,
        )
        await sqlite_backend.save(state)

        # Record some executions
        for i in range(1, 8):
            await sqlite_backend.record_execution(
                job_id="stats-job",
                sheet_num=i,
                attempt_num=1,
                exit_code=0,
                duration_seconds=2.0,
            )

        stats = await sqlite_backend.get_job_statistics("stats-job")

        assert stats["total_sheets"] == 10
        assert stats["completed_sheets"] == 7
        assert stats["success_rate"] == 70.0
        assert stats["total_retries"] == 3
        assert stats["total_executions"] == 7
        assert stats["avg_duration_seconds"] == 2.0

    async def test_get_job_statistics_nonexistent(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test getting statistics for non-existent job."""
        stats = await sqlite_backend.get_job_statistics("nonexistent")
        assert stats == {}


class TestQueryJobs:
    """Test job querying for dashboard."""

    async def test_query_jobs_all(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test querying all jobs."""
        # Create jobs with different statuses
        for i, status in enumerate([JobStatus.COMPLETED, JobStatus.RUNNING, JobStatus.FAILED]):
            state = CheckpointState(
                job_id=f"query-job-{i}",
                job_name=f"Query Job {i}",
                total_sheets=5,
                status=status,
            )
            await sqlite_backend.save(state)

        results = await sqlite_backend.query_jobs()

        assert len(results) == 3

    async def test_query_jobs_by_status(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test querying jobs filtered by status."""
        for i, status in enumerate([JobStatus.COMPLETED, JobStatus.RUNNING, JobStatus.FAILED]):
            state = CheckpointState(
                job_id=f"status-job-{i}",
                job_name=f"Status Job {i}",
                total_sheets=5,
                status=status,
            )
            await sqlite_backend.save(state)

        results = await sqlite_backend.query_jobs(status=JobStatus.RUNNING)

        assert len(results) == 1
        assert results[0]["status"] == "running"

    async def test_query_jobs_since(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test querying jobs updated since a time."""
        # Create a job
        state = CheckpointState(
            job_id="recent-job",
            job_name="Recent Job",
            total_sheets=5,
        )
        await sqlite_backend.save(state)

        # Query with time in the past
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        results = await sqlite_backend.query_jobs(since=past_time)

        assert len(results) == 1

        # Query with time in the future
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        results = await sqlite_backend.query_jobs(since=future_time)

        assert len(results) == 0


class TestConfigSnapshot:
    """Test config_snapshot storage for resume (Task 3)."""

    async def test_config_snapshot_persisted(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test that config_snapshot is saved and loaded correctly."""
        config_data = {
            "name": "test-job",
            "workspace": "/tmp/workspace",
            "sheet": {"size": 5, "total_items": 25},
            "prompt": {"template": "Process batch {{sheet_num}}"},
        }

        state = CheckpointState(
            job_id="config-test-job",
            job_name="Config Test",
            total_sheets=5,
            config_snapshot=config_data,
            config_path="/path/to/config.yaml",
        )
        await sqlite_backend.save(state)

        loaded = await sqlite_backend.load("config-test-job")

        assert loaded is not None
        assert loaded.config_snapshot == config_data
        assert loaded.config_path == "/path/to/config.yaml"

    async def test_config_snapshot_with_complex_nested_data(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test complex nested config structures survive SQLite round-trip."""
        config_data = {
            "name": "complex-job",
            "validations": [
                {"type": "file_exists", "path": "{{workspace}}/output.txt"},
                {"type": "content_contains", "path": "log.txt", "pattern": "SUCCESS"},
            ],
            "notifications": [
                {"type": "slack", "config": {"channel": "#builds", "enabled": True}},
            ],
            "retry": {"max_retries": 3, "delays": [1.0, 2.0, 4.0]},
        }

        state = CheckpointState(
            job_id="complex-config-job",
            job_name="Complex Config",
            total_sheets=3,
            config_snapshot=config_data,
        )
        await sqlite_backend.save(state)

        loaded = await sqlite_backend.load("complex-config-job")

        assert loaded is not None
        assert loaded.config_snapshot is not None
        assert len(loaded.config_snapshot["validations"]) == 2
        assert loaded.config_snapshot["notifications"][0]["config"]["enabled"] is True
        assert loaded.config_snapshot["retry"]["delays"] == [1.0, 2.0, 4.0]

    async def test_config_snapshot_none_is_allowed(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test that None config_snapshot is handled correctly."""
        state = CheckpointState(
            job_id="no-config-job",
            job_name="No Config",
            total_sheets=1,
            config_snapshot=None,
            config_path=None,
        )
        await sqlite_backend.save(state)

        loaded = await sqlite_backend.load("no-config-job")

        assert loaded is not None
        assert loaded.config_snapshot is None
        assert loaded.config_path is None


class TestSchemaMigration:
    """Test schema migration support."""

    async def test_schema_version_recorded(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test that schema version is recorded after migration."""
        import aiosqlite

        # Trigger initialization
        await sqlite_backend._ensure_initialized()

        async with aiosqlite.connect(sqlite_backend.db_path) as db:
            cursor = await db.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 2  # Updated to v2 for config_path column

    async def test_migration_is_idempotent(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test that running migrations multiple times is safe."""
        # Run initialization twice
        await sqlite_backend._ensure_initialized()
        sqlite_backend._initialized = False
        await sqlite_backend._ensure_initialized()

        # Should not raise and should still work
        state = CheckpointState(
            job_id="migration-test",
            job_name="Migration Test",
            total_sheets=1,
        )
        await sqlite_backend.save(state)

        loaded = await sqlite_backend.load("migration-test")
        assert loaded is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    async def test_empty_json_fields(
        self, sqlite_backend: SQLiteStateBackend, sample_state: CheckpointState
    ) -> None:
        """Test handling of empty lists and None values."""
        from mozart.core.checkpoint import SheetState

        sample_state.sheets[1] = SheetState(
            sheet_num=1,
            status=SheetStatus.PENDING,
            learned_patterns=[],
            passed_validations=[],
            failed_validations=[],
            validation_details=None,
            outcome_data=None,
        )
        await sqlite_backend.save(sample_state)

        loaded = await sqlite_backend.load(sample_state.job_id)
        assert loaded is not None
        batch = loaded.sheets[1]

        assert batch.learned_patterns == []
        assert batch.passed_validations == []
        assert batch.failed_validations == []
        assert batch.validation_details is None
        assert batch.outcome_data is None

    async def test_update_existing_job(
        self, sqlite_backend: SQLiteStateBackend, sample_state: CheckpointState
    ) -> None:
        """Test updating an existing job state."""
        await sqlite_backend.save(sample_state)

        # Update state
        sample_state.status = JobStatus.RUNNING
        sample_state.last_completed_sheet = 3
        await sqlite_backend.save(sample_state)

        loaded = await sqlite_backend.load(sample_state.job_id)
        assert loaded is not None
        assert loaded.status == JobStatus.RUNNING
        assert loaded.last_completed_sheet == 3

    async def test_special_characters_in_job_id(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test handling job IDs with special characters."""
        state = CheckpointState(
            job_id="job-with-special/chars:and@symbols",
            job_name="Special Job",
            total_sheets=1,
        )
        await sqlite_backend.save(state)

        loaded = await sqlite_backend.load("job-with-special/chars:and@symbols")
        assert loaded is not None
        assert loaded.job_id == "job-with-special/chars:and@symbols"

    async def test_concurrent_saves(
        self, sqlite_backend: SQLiteStateBackend
    ) -> None:
        """Test concurrent saves don't corrupt data."""
        import asyncio

        async def save_job(job_num: int) -> None:
            state = CheckpointState(
                job_id=f"concurrent-{job_num}",
                job_name=f"Concurrent Job {job_num}",
                total_sheets=5,
            )
            await sqlite_backend.save(state)

        # Save multiple jobs concurrently
        await asyncio.gather(*[save_job(i) for i in range(10)])

        # Verify all jobs saved correctly
        jobs = await sqlite_backend.list_jobs()
        assert len(jobs) == 10
