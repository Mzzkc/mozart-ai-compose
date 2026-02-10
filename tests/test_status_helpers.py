"""Tests for status command helper functions and CLI integration.

FIX-31: The status module (706 LOC) had its CLI paths tested in test_cli.py
but the internal helper functions (_collect_recent_errors, _infer_circuit_breaker_state,
_get_last_activity_time, _infer_error_type) were untested. This file provides
focused unit tests for those functions, plus additional CLI edge cases.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from mozart.cli import app
from mozart.cli.commands.status import (
    _collect_recent_errors,
    _get_last_activity_time,
    _infer_circuit_breaker_state,
    _infer_error_type,
)
from mozart.core.checkpoint import (
    CheckpointState,
    ErrorRecord,
    JobStatus,
    SheetState,
    SheetStatus,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_job(
    *,
    job_id: str = "status-test",
    total_sheets: int = 5,
    status: JobStatus = JobStatus.COMPLETED,
    sheets: dict[int, SheetState] | None = None,
) -> CheckpointState:
    """Build a minimal CheckpointState for status tests."""
    now = datetime.now(UTC)
    if sheets is None:
        sheets = {
            i: SheetState(
                sheet_num=i,
                status=SheetStatus.COMPLETED,
                attempt_count=1,
            )
            for i in range(1, total_sheets + 1)
        }
    return CheckpointState(
        job_id=job_id,
        job_name="Status Test Job",
        total_sheets=total_sheets,
        last_completed_sheet=max(sheets.keys()) if sheets else 0,
        status=status,
        created_at=now - timedelta(hours=1),
        updated_at=now,
        sheets=sheets,
    )


def _write_state(tmp_path: Path, state: CheckpointState) -> Path:
    fp = tmp_path / f"{state.job_id}.json"
    fp.write_text(json.dumps(state.model_dump(mode="json"), default=str))
    return tmp_path


# ---------------------------------------------------------------------------
# _infer_error_type tests
# ---------------------------------------------------------------------------


class TestInferErrorType:
    """Tests for the _infer_error_type() helper."""

    def test_none_category_returns_permanent(self) -> None:
        assert _infer_error_type(None) == "permanent"

    def test_rate_limit_category(self) -> None:
        assert _infer_error_type("rate_limit") == "rate_limit"
        assert _infer_error_type("RATE_LIMIT") == "rate_limit"
        assert _infer_error_type("Rate Limit Exceeded") == "rate_limit"

    def test_transient_categories(self) -> None:
        for category in ("transient", "timeout", "network", "signal"):
            assert _infer_error_type(category) == "transient"

    def test_unknown_category_returns_permanent(self) -> None:
        assert _infer_error_type("validation") == "permanent"
        assert _infer_error_type("E301") == "permanent"


# ---------------------------------------------------------------------------
# _collect_recent_errors tests
# ---------------------------------------------------------------------------


class TestCollectRecentErrors:
    """Tests for the _collect_recent_errors() helper."""

    def test_no_errors_returns_empty(self) -> None:
        job = _make_job()
        assert _collect_recent_errors(job) == []

    def test_collects_from_error_history(self) -> None:
        now = datetime.now(UTC)
        err = ErrorRecord(
            error_type="transient",
            error_code="E001",
            error_message="Timeout",
            attempt_number=1,
            timestamp=now,
        )
        sheets = {
            1: SheetState(
                sheet_num=1,
                status=SheetStatus.FAILED,
                attempt_count=1,
                error_history=[err],
            ),
        }
        job = _make_job(sheets=sheets, total_sheets=1, status=JobStatus.FAILED)
        result = _collect_recent_errors(job, limit=10)

        assert len(result) == 1
        assert result[0][0] == 1  # sheet_num
        assert result[0][1].error_code == "E001"

    def test_limit_respected(self) -> None:
        now = datetime.now(UTC)
        errors = [
            ErrorRecord(
                error_type="transient",
                error_code=f"E00{i}",
                error_message=f"Error {i}",
                attempt_number=i,
                timestamp=now - timedelta(minutes=5 - i),
            )
            for i in range(1, 6)
        ]
        sheets = {
            1: SheetState(
                sheet_num=1,
                status=SheetStatus.FAILED,
                attempt_count=5,
                error_history=errors,
            ),
        }
        job = _make_job(sheets=sheets, total_sheets=1, status=JobStatus.FAILED)
        result = _collect_recent_errors(job, limit=2)

        assert len(result) == 2

    def test_sorted_by_timestamp_descending(self) -> None:
        now = datetime.now(UTC)
        old_err = ErrorRecord(
            error_type="transient", error_code="E001",
            error_message="Old", attempt_number=1,
            timestamp=now - timedelta(hours=1),
        )
        new_err = ErrorRecord(
            error_type="transient", error_code="E002",
            error_message="New", attempt_number=2,
            timestamp=now,
        )
        sheets = {
            1: SheetState(
                sheet_num=1, status=SheetStatus.FAILED,
                attempt_count=2, error_history=[old_err, new_err],
            ),
        }
        job = _make_job(sheets=sheets, total_sheets=1, status=JobStatus.FAILED)
        result = _collect_recent_errors(job, limit=10)

        # Most recent first
        assert result[0][1].error_code == "E002"
        assert result[1][1].error_code == "E001"

    def test_synthetic_from_sheet_error_message(self) -> None:
        """When no error_history exists, synthetic ErrorRecord is created."""
        now = datetime.now(UTC)
        sheets = {
            1: SheetState(
                sheet_num=1,
                status=SheetStatus.FAILED,
                attempt_count=3,
                error_message="CLI not found",
                error_category="E201",
                completed_at=now,
            ),
        }
        job = _make_job(sheets=sheets, total_sheets=1, status=JobStatus.FAILED)
        result = _collect_recent_errors(job, limit=10)

        assert len(result) == 1
        assert result[0][1].error_message == "CLI not found"
        assert result[0][1].error_code == "E201"


# ---------------------------------------------------------------------------
# _get_last_activity_time tests
# ---------------------------------------------------------------------------


class TestGetLastActivityTime:
    """Tests for the _get_last_activity_time() helper."""

    def test_returns_updated_at_for_empty_job(self) -> None:
        """Even an empty job has updated_at, so it should return that."""
        job = _make_job(total_sheets=0, sheets={})
        job.last_completed_sheet = 0
        # updated_at is always set (non-optional), so result should be non-None
        result = _get_last_activity_time(job)
        assert result is not None
        assert result == job.updated_at

    def test_returns_updated_at(self) -> None:
        now = datetime.now(UTC)
        job = _make_job()
        job.updated_at = now
        result = _get_last_activity_time(job)
        assert result == now

    def test_prefers_most_recent_activity(self) -> None:
        now = datetime.now(UTC)
        sheet_activity = now + timedelta(minutes=5)
        sheets = {
            1: SheetState(
                sheet_num=1,
                status=SheetStatus.COMPLETED,
                attempt_count=1,
                last_activity_at=sheet_activity,
            ),
        }
        job = _make_job(sheets=sheets, total_sheets=1)
        job.updated_at = now
        result = _get_last_activity_time(job)
        assert result == sheet_activity


# ---------------------------------------------------------------------------
# _infer_circuit_breaker_state tests
# ---------------------------------------------------------------------------


class TestInferCircuitBreakerState:
    """Tests for the _infer_circuit_breaker_state() helper."""

    def test_no_sheets_returns_none(self) -> None:
        job = _make_job(total_sheets=0, sheets={})
        job.last_completed_sheet = 0
        assert _infer_circuit_breaker_state(job) is None

    def test_all_completed_returns_none(self) -> None:
        """No failures means no circuit breaker info to show."""
        job = _make_job(total_sheets=3)
        assert _infer_circuit_breaker_state(job) is None

    def test_single_failure_returns_closed(self) -> None:
        sheets = {
            1: SheetState(sheet_num=1, status=SheetStatus.COMPLETED, attempt_count=1),
            2: SheetState(sheet_num=2, status=SheetStatus.FAILED, attempt_count=3),
        }
        job = _make_job(sheets=sheets, total_sheets=2, status=JobStatus.FAILED)
        result = _infer_circuit_breaker_state(job)

        assert result is not None
        assert result["state"] == "closed"
        assert result["consecutive_failures"] == 1

    def test_many_failures_returns_open(self) -> None:
        """5+ consecutive failures from the end should be OPEN."""
        sheets = {
            i: SheetState(
                sheet_num=i,
                status=SheetStatus.FAILED,
                attempt_count=3,
            )
            for i in range(1, 7)
        }
        job = _make_job(sheets=sheets, total_sheets=6, status=JobStatus.FAILED)
        result = _infer_circuit_breaker_state(job)

        assert result is not None
        assert result["state"] == "open"
        assert result["consecutive_failures"] >= 5

    def test_mixed_then_failures(self) -> None:
        """Completed sheets followed by failures — count only consecutive from end."""
        sheets = {
            1: SheetState(sheet_num=1, status=SheetStatus.COMPLETED, attempt_count=1),
            2: SheetState(sheet_num=2, status=SheetStatus.COMPLETED, attempt_count=1),
            3: SheetState(sheet_num=3, status=SheetStatus.FAILED, attempt_count=3),
            4: SheetState(sheet_num=4, status=SheetStatus.FAILED, attempt_count=3),
        }
        job = _make_job(sheets=sheets, total_sheets=4, status=JobStatus.FAILED)
        result = _infer_circuit_breaker_state(job)

        assert result is not None
        assert result["state"] == "closed"
        assert result["consecutive_failures"] == 2

    def test_pending_sheets_ignored(self) -> None:
        """PENDING sheets don't count as failures or break the streak."""
        sheets = {
            1: SheetState(sheet_num=1, status=SheetStatus.COMPLETED, attempt_count=1),
            2: SheetState(sheet_num=2, status=SheetStatus.PENDING, attempt_count=0),
            3: SheetState(sheet_num=3, status=SheetStatus.FAILED, attempt_count=3),
        }
        job = _make_job(sheets=sheets, total_sheets=3, status=JobStatus.FAILED)
        result = _infer_circuit_breaker_state(job)

        assert result is not None
        # Sheet 3 failed, sheet 2 pending (skipped), sheet 1 completed (breaks)
        assert result["consecutive_failures"] >= 1


# ---------------------------------------------------------------------------
# Additional CLI integration tests
# ---------------------------------------------------------------------------


class TestStatusCommandEdgeCases:
    """Additional status CLI tests beyond those in test_cli.py."""

    def test_status_with_parallel_info(self, tmp_path: Path) -> None:
        """Test status displays parallel execution info."""
        state = CheckpointState(
            job_id="parallel-job",
            job_name="Parallel Test",
            total_sheets=10,
            last_completed_sheet=3,
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            parallel_enabled=True,
            parallel_max_concurrent=3,
            parallel_batches_executed=1,
            sheets_in_progress=[4, 5, 6],
            sheets={
                i: SheetState(sheet_num=i, status=SheetStatus.COMPLETED, attempt_count=1)
                for i in range(1, 4)
            },
        )
        _write_state(tmp_path, state)

        result = runner.invoke(
            app, ["status", "parallel-job", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Parallel" in result.stdout

    def test_status_with_circuit_breaker_inference(self, tmp_path: Path) -> None:
        """Status should show circuit breaker state when many sheets fail."""
        sheets = {
            i: SheetState(sheet_num=i, status=SheetStatus.FAILED, attempt_count=3)
            for i in range(1, 7)
        }
        state = CheckpointState(
            job_id="cb-job",
            job_name="CB Test",
            total_sheets=6,
            last_completed_sheet=0,
            status=JobStatus.FAILED,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            sheets=sheets,
        )
        _write_state(tmp_path, state)

        result = runner.invoke(
            app, ["status", "cb-job", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Circuit Breaker" in result.stdout
        assert "OPEN" in result.stdout

    def test_status_json_circuit_breaker(self, tmp_path: Path) -> None:
        """JSON output should include circuit_breaker field."""
        sheets = {
            i: SheetState(sheet_num=i, status=SheetStatus.FAILED, attempt_count=3)
            for i in range(1, 7)
        }
        state = CheckpointState(
            job_id="cb-json-job",
            job_name="CB JSON Test",
            total_sheets=6,
            last_completed_sheet=0,
            status=JobStatus.FAILED,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            sheets=sheets,
        )
        _write_state(tmp_path, state)

        result = runner.invoke(
            app, ["status", "cb-json-job", "--json", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["circuit_breaker"] is not None
        assert data["circuit_breaker"]["state"] == "open"

    def test_status_with_recent_errors(self, tmp_path: Path) -> None:
        """Status should show recent errors section."""
        now = datetime.now(UTC)
        err = ErrorRecord(
            error_type="permanent", error_code="E301",
            error_message="Validation failed", attempt_number=3,
            timestamp=now,
        )
        sheets = {
            1: SheetState(
                sheet_num=1, status=SheetStatus.FAILED,
                attempt_count=3, error_history=[err],
            ),
        }
        state = CheckpointState(
            job_id="err-job",
            job_name="Error Test",
            total_sheets=1,
            last_completed_sheet=0,
            status=JobStatus.FAILED,
            created_at=now,
            updated_at=now,
            sheets=sheets,
        )
        _write_state(tmp_path, state)

        result = runner.invoke(
            app, ["status", "err-job", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Error" in result.stdout or "error" in result.stdout.lower()
