"""Tests for Mozart CLI diagnose command and its internal helpers.

FIX-30: The diagnose module (912 LOC) was previously untested. This file
tests the pure-function report builder, the CLI entry points via CliRunner,
and edge cases like empty jobs, error history, and JSON output.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from mozart.cli import app
from mozart.cli.commands.diagnose import _build_diagnostic_report
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
    job_id: str = "diag-test",
    job_name: str = "Diagnose Test",
    total_sheets: int = 3,
    status: JobStatus = JobStatus.COMPLETED,
    sheets: dict[int, SheetState] | None = None,
    error_message: str | None = None,
) -> CheckpointState:
    """Build a minimal CheckpointState for diagnose tests."""
    now = datetime.now(UTC)
    if sheets is None:
        sheets = {
            i: SheetState(
                sheet_num=i,
                status=SheetStatus.COMPLETED,
                attempt_count=1,
                validation_passed=True,
                started_at=now - timedelta(minutes=total_sheets - i + 1),
                completed_at=now - timedelta(minutes=total_sheets - i),
                execution_duration_seconds=60.0,
            )
            for i in range(1, total_sheets + 1)
        }
    return CheckpointState(
        job_id=job_id,
        job_name=job_name,
        total_sheets=total_sheets,
        last_completed_sheet=max(sheets.keys()) if sheets else 0,
        status=status,
        created_at=now - timedelta(hours=1),
        started_at=now - timedelta(minutes=total_sheets + 1),
        updated_at=now,
        completed_at=now if status == JobStatus.COMPLETED else None,
        sheets=sheets,
        error_message=error_message,
    )


def _write_state(tmp_path: Path, state: CheckpointState) -> Path:
    """Write state to JSON file, return workspace path."""
    fp = tmp_path / f"{state.job_id}.json"
    fp.write_text(json.dumps(state.model_dump(mode="json"), default=str))
    return tmp_path


# ---------------------------------------------------------------------------
# _build_diagnostic_report unit tests
# ---------------------------------------------------------------------------


class TestBuildDiagnosticReport:
    """Tests for the pure _build_diagnostic_report() function."""

    def test_basic_fields(self) -> None:
        job = _make_job()
        report = _build_diagnostic_report(job)

        assert report["job_id"] == "diag-test"
        assert report["job_name"] == "Diagnose Test"
        assert report["status"] == "completed"
        assert "generated_at" in report

    def test_progress_section(self) -> None:
        job = _make_job(total_sheets=5)
        report = _build_diagnostic_report(job)

        progress = report["progress"]
        assert progress["total_sheets"] == 5
        assert progress["completed"] == 5
        assert progress["failed"] == 0
        assert progress["percent"] > 0

    def test_progress_with_failures(self) -> None:
        sheets = {
            1: SheetState(sheet_num=1, status=SheetStatus.COMPLETED, attempt_count=1),
            2: SheetState(sheet_num=2, status=SheetStatus.FAILED, attempt_count=3,
                          error_message="Validation failed"),
            3: SheetState(sheet_num=3, status=SheetStatus.COMPLETED, attempt_count=1),
        }
        job = _make_job(total_sheets=3, status=JobStatus.FAILED, sheets=sheets)
        report = _build_diagnostic_report(job)

        assert report["progress"]["completed"] == 2
        assert report["progress"]["failed"] == 1

    def test_timing_with_started_job(self) -> None:
        job = _make_job()
        report = _build_diagnostic_report(job)

        timing = report["timing"]
        assert timing["started_at"] is not None
        assert timing["completed_at"] is not None
        assert "duration_seconds" in timing
        assert timing["duration_seconds"] > 0

    def test_timing_without_start(self) -> None:
        job = _make_job()
        job.started_at = None
        report = _build_diagnostic_report(job)

        timing = report["timing"]
        assert timing["started_at"] is None
        assert "duration_seconds" not in timing

    def test_execution_timeline_order(self) -> None:
        job = _make_job(total_sheets=3)
        report = _build_diagnostic_report(job)

        timeline = report["execution_timeline"]
        assert len(timeline) == 3
        sheet_nums = [e["sheet_num"] for e in timeline]
        assert sheet_nums == [1, 2, 3]

    def test_execution_timeline_entry_fields(self) -> None:
        job = _make_job(total_sheets=1)
        report = _build_diagnostic_report(job)

        entry = report["execution_timeline"][0]
        assert "sheet_num" in entry
        assert "status" in entry
        assert "attempt_count" in entry
        assert "duration_seconds" in entry

    def test_error_collection_from_history(self) -> None:
        now = datetime.now(UTC)
        error = ErrorRecord(
            error_type="transient",
            error_code="E001",
            error_message="Timeout waiting for output",
            attempt_number=2,
            timestamp=now,
        )
        sheets = {
            1: SheetState(
                sheet_num=1,
                status=SheetStatus.FAILED,
                attempt_count=2,
                error_history=[error],
            ),
        }
        job = _make_job(total_sheets=1, status=JobStatus.FAILED, sheets=sheets)
        report = _build_diagnostic_report(job)

        assert report["error_count"] == 1
        err = report["errors"][0]
        assert err["error_type"] == "transient"
        assert err["error_code"] == "E001"
        assert err["error_message"] == "Timeout waiting for output"
        assert err["sheet_num"] == 1

    def test_error_collection_from_sheet_message(self) -> None:
        """When no error_history exists, errors are synthesized from sheet error_message."""
        sheets = {
            1: SheetState(
                sheet_num=1,
                status=SheetStatus.FAILED,
                attempt_count=3,
                error_message="CLI not found",
                error_category="E201",
                exit_code=127,
            ),
        }
        job = _make_job(total_sheets=1, status=JobStatus.FAILED, sheets=sheets)
        report = _build_diagnostic_report(job)

        assert report["error_count"] == 1
        err = report["errors"][0]
        assert err["error_message"] == "CLI not found"
        assert err["error_code"] == "E201"
        assert err["context"]["exit_code"] == 127

    def test_no_errors_when_clean(self) -> None:
        job = _make_job()
        report = _build_diagnostic_report(job)

        assert report["error_count"] == 0
        assert report["errors"] == []

    def test_preflight_warnings(self) -> None:
        sheets = {
            1: SheetState(
                sheet_num=1,
                status=SheetStatus.COMPLETED,
                attempt_count=1,
                preflight_warnings=["Long prompt (>50k tokens)", "No validations defined"],
            ),
        }
        job = _make_job(total_sheets=1, sheets=sheets)
        report = _build_diagnostic_report(job)

        assert len(report["preflight_warnings"]) == 2
        assert report["preflight_warnings"][0]["warning"] == "Long prompt (>50k tokens)"
        assert report["preflight_warnings"][0]["sheet_num"] == 1

    def test_prompt_metrics_and_token_stats(self) -> None:
        sheets = {
            1: SheetState(
                sheet_num=1,
                status=SheetStatus.COMPLETED,
                attempt_count=1,
                prompt_metrics={"estimated_tokens": 1000, "line_count": 50},
            ),
            2: SheetState(
                sheet_num=2,
                status=SheetStatus.COMPLETED,
                attempt_count=1,
                prompt_metrics={"estimated_tokens": 3000, "line_count": 150},
            ),
        }
        job = _make_job(total_sheets=2, sheets=sheets)
        report = _build_diagnostic_report(job)

        assert len(report["prompt_metrics"]) == 2
        assert "token_statistics" in report
        assert report["token_statistics"]["min"] == 1000
        assert report["token_statistics"]["max"] == 3000
        assert report["token_statistics"]["total"] == 4000

    def test_job_error_message(self) -> None:
        job = _make_job(error_message="Max retries exceeded globally")
        report = _build_diagnostic_report(job)

        assert report["job_error"] == "Max retries exceeded globally"

    def test_empty_job_no_sheets(self) -> None:
        job = _make_job(total_sheets=0, sheets={})
        job.last_completed_sheet = 0
        report = _build_diagnostic_report(job)

        assert report["progress"]["total_sheets"] == 0
        assert report["progress"]["percent"] == 0
        assert report["execution_timeline"] == []

    def test_execution_stats(self) -> None:
        job = _make_job()
        job.total_retry_count = 7
        job.rate_limit_waits = 3
        report = _build_diagnostic_report(job)

        assert report["execution_stats"]["total_retry_count"] == 7
        assert report["execution_stats"]["rate_limit_waits"] == 3


# ---------------------------------------------------------------------------
# CLI integration tests via CliRunner
# ---------------------------------------------------------------------------


class TestDiagnoseCommand:
    """Tests for `mozart diagnose` via CliRunner."""

    def test_diagnose_rich_output(self, tmp_path: Path) -> None:
        job = _make_job()
        _write_state(tmp_path, job)

        result = runner.invoke(
            app, ["diagnose", "diag-test", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Diagnose Test" in result.stdout or "diag-test" in result.stdout

    def test_diagnose_json_output(self, tmp_path: Path) -> None:
        job = _make_job()
        _write_state(tmp_path, job)

        result = runner.invoke(
            app, ["diagnose", "diag-test", "--json", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["job_id"] == "diag-test"
        assert data["status"] == "completed"
        assert "progress" in data
        assert "execution_timeline" in data

    def test_diagnose_job_not_found(self, tmp_path: Path) -> None:
        workspace = tmp_path / "empty"
        workspace.mkdir()

        result = runner.invoke(
            app, ["diagnose", "no-such-job", "--workspace", str(workspace)]
        )
        assert result.exit_code == 1
        assert "Job not found" in result.stdout

    def test_diagnose_nonexistent_workspace(self, tmp_path: Path) -> None:
        fake = tmp_path / "nope"

        result = runner.invoke(
            app, ["diagnose", "any-job", "--workspace", str(fake)]
        )
        assert result.exit_code == 1
        assert "Workspace not found" in result.stdout

    def test_diagnose_with_errors(self, tmp_path: Path) -> None:
        """Diagnose should display errors section when job has failures."""
        now = datetime.now(UTC)
        error = ErrorRecord(
            error_type="permanent",
            error_code="E301",
            error_message="Validation: file not found",
            attempt_number=3,
            timestamp=now,
        )
        sheets = {
            1: SheetState(
                sheet_num=1,
                status=SheetStatus.FAILED,
                attempt_count=3,
                error_history=[error],
            ),
        }
        job = _make_job(total_sheets=1, status=JobStatus.FAILED, sheets=sheets)
        _write_state(tmp_path, job)

        result = runner.invoke(
            app, ["diagnose", "diag-test", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        # Should show errors section
        assert "Error" in result.stdout or "error" in result.stdout.lower()

    def test_diagnose_json_with_errors(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        error = ErrorRecord(
            error_type="rate_limit",
            error_code="E102",
            error_message="Rate limited",
            attempt_number=1,
            timestamp=now,
        )
        sheets = {
            1: SheetState(
                sheet_num=1,
                status=SheetStatus.FAILED,
                attempt_count=2,
                error_history=[error],
            ),
        }
        job = _make_job(total_sheets=1, status=JobStatus.FAILED, sheets=sheets)
        _write_state(tmp_path, job)

        result = runner.invoke(
            app, ["diagnose", "diag-test", "--json", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["error_count"] == 1
        assert data["errors"][0]["error_type"] == "rate_limit"


class TestErrorsCommand:
    """Tests for `mozart errors` via CliRunner."""

    def test_errors_no_errors(self, tmp_path: Path) -> None:
        job = _make_job()
        _write_state(tmp_path, job)

        result = runner.invoke(
            app, ["errors", "diag-test", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "No errors found" in result.stdout

    def test_errors_with_history(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        err1 = ErrorRecord(
            error_type="transient", error_code="E001",
            error_message="Timeout", attempt_number=1, timestamp=now,
        )
        err2 = ErrorRecord(
            error_type="permanent", error_code="E301",
            error_message="File not found", attempt_number=2, timestamp=now,
        )
        sheets = {
            1: SheetState(
                sheet_num=1, status=SheetStatus.FAILED,
                attempt_count=2, error_history=[err1, err2],
            ),
        }
        job = _make_job(total_sheets=1, status=JobStatus.FAILED, sheets=sheets)
        _write_state(tmp_path, job)

        result = runner.invoke(
            app, ["errors", "diag-test", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Timeout" in result.stdout or "E001" in result.stdout

    def test_errors_json_output(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        err = ErrorRecord(
            error_type="transient", error_code="E001",
            error_message="Timeout", attempt_number=1, timestamp=now,
        )
        sheets = {
            1: SheetState(
                sheet_num=1, status=SheetStatus.FAILED,
                attempt_count=1, error_history=[err],
            ),
        }
        job = _make_job(total_sheets=1, status=JobStatus.FAILED, sheets=sheets)
        _write_state(tmp_path, job)

        result = runner.invoke(
            app, ["errors", "diag-test", "--json", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["total_errors"] == 1
        assert data["errors"][0]["error_code"] == "E001"

    def test_errors_filter_by_sheet(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        err1 = ErrorRecord(
            error_type="transient", error_code="E001",
            error_message="Timeout on sheet 1", attempt_number=1, timestamp=now,
        )
        err2 = ErrorRecord(
            error_type="permanent", error_code="E301",
            error_message="Error on sheet 2", attempt_number=1, timestamp=now,
        )
        sheets = {
            1: SheetState(sheet_num=1, status=SheetStatus.FAILED, attempt_count=1, error_history=[err1]),
            2: SheetState(sheet_num=2, status=SheetStatus.FAILED, attempt_count=1, error_history=[err2]),
        }
        job = _make_job(total_sheets=2, status=JobStatus.FAILED, sheets=sheets)
        _write_state(tmp_path, job)

        result = runner.invoke(
            app, ["errors", "diag-test", "--sheet", "1", "--json", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["total_errors"] == 1
        assert data["errors"][0]["error_message"] == "Timeout on sheet 1"

    def test_errors_filter_by_type(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        err1 = ErrorRecord(
            error_type="transient", error_code="E001",
            error_message="Timeout", attempt_number=1, timestamp=now,
        )
        err2 = ErrorRecord(
            error_type="rate_limit", error_code="E102",
            error_message="Rate limited", attempt_number=2, timestamp=now,
        )
        sheets = {
            1: SheetState(
                sheet_num=1, status=SheetStatus.FAILED,
                attempt_count=2, error_history=[err1, err2],
            ),
        }
        job = _make_job(total_sheets=1, status=JobStatus.FAILED, sheets=sheets)
        _write_state(tmp_path, job)

        result = runner.invoke(
            app, ["errors", "diag-test", "--type", "rate_limit", "--json", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["total_errors"] == 1
        assert data["errors"][0]["error_type"] == "rate_limit"

    def test_errors_synthetic_from_sheet_message(self, tmp_path: Path) -> None:
        """When no error_history, errors command synthesizes from sheet error_message."""
        sheets = {
            1: SheetState(
                sheet_num=1, status=SheetStatus.FAILED, attempt_count=2,
                error_message="CLI binary not found",
                error_category="E201",
            ),
        }
        job = _make_job(total_sheets=1, status=JobStatus.FAILED, sheets=sheets)
        _write_state(tmp_path, job)

        result = runner.invoke(
            app, ["errors", "diag-test", "--json", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["total_errors"] == 1
        assert "CLI binary not found" in data["errors"][0]["error_message"]

    def test_errors_job_not_found(self, tmp_path: Path) -> None:
        workspace = tmp_path / "empty"
        workspace.mkdir()

        result = runner.invoke(
            app, ["errors", "no-job", "--workspace", str(workspace)]
        )
        assert result.exit_code == 1
        assert "Job not found" in result.stdout


class TestLogsCommand:
    """Basic tests for `mozart logs` (limited scope — no running jobs)."""

    def test_logs_no_log_file(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["logs", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "No log files found" in result.stdout

    def test_logs_invalid_level(self, tmp_path: Path) -> None:
        # Create a dummy log file so it doesn't exit early on missing file
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "mozart.log"
        log_file.write_text('{"event": "test", "level": "INFO"}\n')

        result = runner.invoke(
            app, ["logs", "--workspace", str(tmp_path), "--level", "BOGUS"]
        )
        assert result.exit_code == 1
        assert "Invalid log level" in result.stdout
