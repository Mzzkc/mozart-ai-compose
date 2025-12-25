"""Tests for Mozart CLI commands."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from mozart.cli import app
from mozart.core.checkpoint import BatchState, BatchStatus, CheckpointState, JobStatus


runner = CliRunner()


class TestVersionCommand:
    """Tests for the --version flag."""

    def test_version_shows_version(self) -> None:
        """Test that --version prints version info."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "Mozart AI Compose" in result.stdout


class TestValidateCommand:
    """Tests for the validate command."""

    def test_validate_valid_config(self, sample_yaml_config: Path) -> None:
        """Test validation of a valid config file."""
        result = runner.invoke(app, ["validate", str(sample_yaml_config)])
        assert result.exit_code == 0
        assert "Valid configuration" in result.stdout

    def test_validate_nonexistent_file(self, tmp_path: Path) -> None:
        """Test validation of a nonexistent file."""
        fake_path = tmp_path / "nonexistent.yaml"
        result = runner.invoke(app, ["validate", str(fake_path)])
        assert result.exit_code != 0

    def test_validate_invalid_yaml(self, tmp_path: Path) -> None:
        """Test validation of invalid YAML content."""
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("name: test\nbatch:\n  size: -1")  # Invalid size
        result = runner.invoke(app, ["validate", str(bad_config)])
        assert result.exit_code == 1
        assert "Invalid configuration" in result.stdout


class TestListCommand:
    """Tests for the list command."""

    def test_list_empty_state(self, tmp_path: Path) -> None:
        """Test list command when no jobs exist."""
        workspace = tmp_path / "empty_workspace"
        workspace.mkdir()

        result = runner.invoke(app, ["list", "--workspace", str(workspace)])
        assert result.exit_code == 0
        assert "No jobs found" in result.stdout

    def test_list_with_json_state(self, tmp_path: Path) -> None:
        """Test list command with JSON state files."""
        # Create a mock JSON state file
        state = CheckpointState(
            job_id="test-job-1",
            job_name="Test Job 1",
            total_batches=5,
            last_completed_batch=3,
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        state_file = tmp_path / "test-job-1.json"
        state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        result = runner.invoke(app, ["list", "--workspace", str(tmp_path)])
        assert result.exit_code == 0
        assert "test-job-1" in result.stdout
        assert "running" in result.stdout.lower()
        assert "3/5" in result.stdout

    def test_list_with_multiple_jobs(self, tmp_path: Path) -> None:
        """Test list command with multiple job state files."""
        # Create multiple job state files
        for i, status in enumerate([JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.PENDING]):
            state = CheckpointState(
                job_id=f"job-{i}",
                job_name=f"Job {i}",
                total_batches=10,
                last_completed_batch=i * 3,
                status=status,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            state_file = tmp_path / f"job-{i}.json"
            state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        result = runner.invoke(app, ["list", "--workspace", str(tmp_path)])
        assert result.exit_code == 0
        assert "job-0" in result.stdout
        assert "job-1" in result.stdout
        assert "job-2" in result.stdout
        assert "Showing 3 job(s)" in result.stdout

    def test_list_filter_by_status(self, tmp_path: Path) -> None:
        """Test list command with status filter."""
        # Create jobs with different statuses
        for i, status in enumerate([JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.COMPLETED]):
            state = CheckpointState(
                job_id=f"job-{i}",
                job_name=f"Job {i}",
                total_batches=10,
                last_completed_batch=10 if status == JobStatus.COMPLETED else 5,
                status=status,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            state_file = tmp_path / f"job-{i}.json"
            state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        # Filter for completed jobs
        result = runner.invoke(
            app, ["list", "--workspace", str(tmp_path), "--status", "completed"]
        )
        assert result.exit_code == 0
        assert "job-0" in result.stdout
        assert "job-2" in result.stdout
        # job-1 is failed, should not appear
        assert "job-1" not in result.stdout
        assert "Showing 2 job(s)" in result.stdout

    def test_list_filter_by_invalid_status(self, tmp_path: Path) -> None:
        """Test list command with invalid status filter."""
        workspace = tmp_path / "ws"
        workspace.mkdir()

        result = runner.invoke(
            app, ["list", "--workspace", str(workspace), "--status", "bogus"]
        )
        assert result.exit_code == 1
        assert "Invalid status" in result.stdout

    def test_list_with_limit(self, tmp_path: Path) -> None:
        """Test list command respects --limit option."""
        # Create 5 job state files
        for i in range(5):
            state = CheckpointState(
                job_id=f"job-{i}",
                job_name=f"Job {i}",
                total_batches=10,
                last_completed_batch=i,
                status=JobStatus.COMPLETED,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            state_file = tmp_path / f"job-{i}.json"
            state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        # Request only 2
        result = runner.invoke(
            app, ["list", "--workspace", str(tmp_path), "--limit", "2"]
        )
        assert result.exit_code == 0
        assert "Showing 2 job(s)" in result.stdout

    def test_list_nonexistent_workspace(self, tmp_path: Path) -> None:
        """Test list command with nonexistent workspace."""
        fake_workspace = tmp_path / "does_not_exist"

        result = runner.invoke(app, ["list", "--workspace", str(fake_workspace)])
        assert result.exit_code == 1
        assert "Workspace not found" in result.stdout


class TestRunCommand:
    """Tests for the run command."""

    def test_run_dry_run(self, sample_yaml_config: Path) -> None:
        """Test run command in dry-run mode."""
        result = runner.invoke(app, ["run", str(sample_yaml_config), "--dry-run"])
        assert result.exit_code == 0
        assert "Dry run" in result.stdout
        assert "Batch Plan" in result.stdout

    def test_run_shows_config_panel(self, sample_yaml_config: Path) -> None:
        """Test run command displays configuration panel."""
        result = runner.invoke(app, ["run", str(sample_yaml_config), "--dry-run"])
        assert result.exit_code == 0
        assert "Job Configuration" in result.stdout
        assert "test-job" in result.stdout


class TestStatusCommand:
    """Tests for the status command."""

    def test_status_with_valid_job_id(self, tmp_path: Path) -> None:
        """Test status command with a valid job ID."""
        # Create a job state file
        state = CheckpointState(
            job_id="test-job-status",
            job_name="Test Job for Status",
            total_batches=5,
            last_completed_batch=3,
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            total_retry_count=2,
            rate_limit_waits=1,
            batches={
                1: BatchState(
                    batch_num=1,
                    status=BatchStatus.COMPLETED,
                    attempt_count=1,
                    validation_passed=True,
                ),
                2: BatchState(
                    batch_num=2,
                    status=BatchStatus.COMPLETED,
                    attempt_count=2,
                    validation_passed=True,
                ),
                3: BatchState(
                    batch_num=3,
                    status=BatchStatus.COMPLETED,
                    attempt_count=1,
                    validation_passed=True,
                ),
            },
        )

        state_file = tmp_path / "test-job-status.json"
        state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        result = runner.invoke(
            app, ["status", "test-job-status", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Test Job for Status" in result.stdout
        assert "test-job-status" in result.stdout
        assert "RUNNING" in result.stdout
        # Check progress is shown
        assert "3" in result.stdout and "5" in result.stdout

    def test_status_with_invalid_job_id(self, tmp_path: Path) -> None:
        """Test status command with an invalid job ID shows error."""
        workspace = tmp_path / "empty_ws"
        workspace.mkdir()

        result = runner.invoke(
            app, ["status", "nonexistent-job", "--workspace", str(workspace)]
        )
        assert result.exit_code == 1
        assert "Job not found" in result.stdout
        assert "nonexistent-job" in result.stdout

    def test_status_json_output_format(self, tmp_path: Path) -> None:
        """Test status --json outputs valid JSON."""
        state = CheckpointState(
            job_id="json-test-job",
            job_name="JSON Test Job",
            total_batches=10,
            last_completed_batch=5,
            status=JobStatus.COMPLETED,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            batches={
                1: BatchState(
                    batch_num=1,
                    status=BatchStatus.COMPLETED,
                    attempt_count=1,
                    validation_passed=True,
                ),
            },
        )

        state_file = tmp_path / "json-test-job.json"
        state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        result = runner.invoke(
            app, ["status", "json-test-job", "--json", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0

        # Parse the output as JSON to verify it's valid
        output_data = json.loads(result.stdout)
        assert output_data["job_id"] == "json-test-job"
        assert output_data["job_name"] == "JSON Test Job"
        assert output_data["status"] == "completed"
        assert output_data["progress"]["completed"] == 5
        assert output_data["progress"]["total"] == 10
        assert output_data["progress"]["percent"] == 50.0
        assert "1" in output_data["batches"]

    def test_status_json_output_for_missing_job(self, tmp_path: Path) -> None:
        """Test status --json outputs JSON error for missing job."""
        workspace = tmp_path / "empty_ws2"
        workspace.mkdir()

        result = runner.invoke(
            app, ["status", "missing-job", "--json", "--workspace", str(workspace)]
        )
        assert result.exit_code == 1

        # Parse the output as JSON to verify error format
        output_data = json.loads(result.stdout)
        assert "error" in output_data
        assert "missing-job" in output_data["error"]

    def test_status_shows_batch_details(self, tmp_path: Path) -> None:
        """Test status command shows batch details table."""
        state = CheckpointState(
            job_id="batch-details-job",
            job_name="Batch Details Test",
            total_batches=3,
            last_completed_batch=2,
            status=JobStatus.FAILED,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            error_message="Max retries exceeded",
            batches={
                1: BatchState(
                    batch_num=1,
                    status=BatchStatus.COMPLETED,
                    attempt_count=1,
                    validation_passed=True,
                ),
                2: BatchState(
                    batch_num=2,
                    status=BatchStatus.COMPLETED,
                    attempt_count=1,
                    validation_passed=True,
                ),
                3: BatchState(
                    batch_num=3,
                    status=BatchStatus.FAILED,
                    attempt_count=3,
                    validation_passed=False,
                    error_message="Validation failed: file not found",
                    error_category="validation",
                ),
            },
        )

        state_file = tmp_path / "batch-details-job.json"
        state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        result = runner.invoke(
            app, ["status", "batch-details-job", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Batch Details" in result.stdout
        assert "Max retries exceeded" in result.stdout
        # Check batch statuses are visible
        assert "completed" in result.stdout.lower()
        assert "failed" in result.stdout.lower()

    def test_status_nonexistent_workspace(self, tmp_path: Path) -> None:
        """Test status command with nonexistent workspace."""
        fake_workspace = tmp_path / "does_not_exist"

        result = runner.invoke(
            app, ["status", "some-job", "--workspace", str(fake_workspace)]
        )
        assert result.exit_code == 1
        assert "Workspace not found" in result.stdout


class TestResumeCommand:
    """Tests for the resume command."""

    def test_resume_job_not_found(self, tmp_path: Path) -> None:
        """Test resume shows error for non-existent job."""
        workspace = tmp_path / "empty_ws"
        workspace.mkdir()

        result = runner.invoke(
            app, ["resume", "nonexistent-job", "--workspace", str(workspace)]
        )
        assert result.exit_code == 1
        assert "Job not found" in result.stdout
        assert "nonexistent-job" in result.stdout

    def test_resume_completed_job_blocked(self, tmp_path: Path) -> None:
        """Test resume shows error for completed jobs without --force."""
        # Create a completed job state
        state = CheckpointState(
            job_id="completed-job",
            job_name="Completed Job",
            total_batches=5,
            last_completed_batch=5,
            status=JobStatus.COMPLETED,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            config_snapshot={
                "name": "completed-job",
                "batch": {"size": 5, "total_items": 25},
                "prompt": {"template": "Test"},
            },
        )

        state_file = tmp_path / "completed-job.json"
        state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        result = runner.invoke(
            app, ["resume", "completed-job", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "already completed" in result.stdout

    def test_resume_pending_job_blocked(self, tmp_path: Path) -> None:
        """Test resume shows error for pending (never started) jobs."""
        state = CheckpointState(
            job_id="pending-job",
            job_name="Pending Job",
            total_batches=5,
            last_completed_batch=0,
            status=JobStatus.PENDING,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        state_file = tmp_path / "pending-job.json"
        state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        result = runner.invoke(
            app, ["resume", "pending-job", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "not been started yet" in result.stdout

    def test_resume_paused_job_uses_config_snapshot(
        self, tmp_path: Path, sample_config_dict: dict
    ) -> None:
        """Test resume reconstructs config from config_snapshot."""
        # Create a paused job state with config_snapshot
        state = CheckpointState(
            job_id="paused-job",
            job_name="Paused Job",
            total_batches=5,
            last_completed_batch=2,
            status=JobStatus.PAUSED,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            config_snapshot=sample_config_dict,
        )

        state_file = tmp_path / "paused-job.json"
        state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        # Mock the runner at the module level where it's imported
        with patch("mozart.execution.runner.JobRunner") as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run = AsyncMock(return_value=CheckpointState(
                job_id="paused-job",
                job_name="Paused Job",
                total_batches=5,
                last_completed_batch=5,
                status=JobStatus.COMPLETED,
            ))
            mock_runner_cls.return_value = mock_runner

            # Also mock the backend constructors
            with patch("mozart.backends.claude_cli.ClaudeCliBackend") as mock_backend:
                mock_backend.from_config = AsyncMock(return_value=mock_backend)

                result = runner.invoke(
                    app, ["resume", "paused-job", "--workspace", str(tmp_path)]
                )

        # Should have started without errors (may fail later in mock)
        assert "Resume Job" in result.stdout or "Reconstructed config" in result.stdout

    def test_resume_failed_job_allowed(self, tmp_path: Path, sample_config_dict: dict) -> None:
        """Test resume is allowed for failed jobs."""
        state = CheckpointState(
            job_id="failed-job",
            job_name="Failed Job",
            total_batches=10,
            last_completed_batch=5,
            status=JobStatus.FAILED,
            error_message="Max retries exceeded",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            config_snapshot=sample_config_dict,
        )

        state_file = tmp_path / "failed-job.json"
        state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        # Mock the runner to avoid actual execution
        with patch("mozart.execution.runner.JobRunner") as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run = AsyncMock(return_value=CheckpointState(
                job_id="failed-job",
                job_name="Failed Job",
                total_batches=10,
                last_completed_batch=10,
                status=JobStatus.COMPLETED,
            ))
            mock_runner_cls.return_value = mock_runner

            with patch("mozart.backends.claude_cli.ClaudeCliBackend") as mock_backend:
                mock_backend.from_config = AsyncMock(return_value=mock_backend)

                result = runner.invoke(
                    app, ["resume", "failed-job", "--workspace", str(tmp_path)]
                )

        # Verify resume was attempted with correct resume point
        assert "Resume Job" in result.stdout
        assert "5/10" in result.stdout  # Progress shown

    def test_resume_missing_config(self, tmp_path: Path) -> None:
        """Test resume shows error when no config is available."""
        # Create a state without config_snapshot
        state = CheckpointState(
            job_id="no-config-job",
            job_name="No Config Job",
            total_batches=5,
            last_completed_batch=2,
            status=JobStatus.PAUSED,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            config_snapshot=None,
            config_path=None,
        )

        state_file = tmp_path / "no-config-job.json"
        state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        result = runner.invoke(
            app, ["resume", "no-config-job", "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "No config available" in result.stdout

    def test_resume_with_config_file(
        self, tmp_path: Path, sample_yaml_config: Path
    ) -> None:
        """Test resume with explicit --config file."""
        # Create a paused job without config_snapshot
        state = CheckpointState(
            job_id="test-job",  # Matches sample_config_dict name
            job_name="Test Job",
            total_batches=3,
            last_completed_batch=1,
            status=JobStatus.PAUSED,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            config_snapshot=None,
        )

        state_file = tmp_path / "test-job.json"
        state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        # Mock the runner
        with patch("mozart.execution.runner.JobRunner") as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run = AsyncMock(return_value=CheckpointState(
                job_id="test-job",
                job_name="Test Job",
                total_batches=3,
                last_completed_batch=3,
                status=JobStatus.COMPLETED,
            ))
            mock_runner_cls.return_value = mock_runner

            with patch("mozart.backends.claude_cli.ClaudeCliBackend") as mock_backend:
                mock_backend.from_config = AsyncMock(return_value=mock_backend)

                result = runner.invoke(
                    app, [
                        "resume", "test-job",
                        "--workspace", str(tmp_path),
                        "--config", str(sample_yaml_config),
                    ]
                )

        # Should have used the provided config
        assert "Using config from" in result.stdout

    def test_resume_force_completed(
        self, tmp_path: Path, sample_config_dict: dict
    ) -> None:
        """Test resume with --force allows rerunning completed jobs."""
        state = CheckpointState(
            job_id="force-job",
            job_name="Force Job",
            total_batches=5,
            last_completed_batch=5,
            status=JobStatus.COMPLETED,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            config_snapshot=sample_config_dict,
        )

        state_file = tmp_path / "force-job.json"
        state_file.write_text(json.dumps(state.model_dump(mode="json"), default=str))

        # Mock the runner
        with patch("mozart.execution.runner.JobRunner") as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run = AsyncMock(return_value=CheckpointState(
                job_id="force-job",
                job_name="Force Job",
                total_batches=5,
                last_completed_batch=5,
                status=JobStatus.COMPLETED,
            ))
            mock_runner_cls.return_value = mock_runner

            with patch("mozart.backends.claude_cli.ClaudeCliBackend") as mock_backend:
                mock_backend.from_config = AsyncMock(return_value=mock_backend)

                result = runner.invoke(
                    app, [
                        "resume", "force-job",
                        "--workspace", str(tmp_path),
                        "--force",
                    ]
                )

        # Should have proceeded with force
        assert "Force restarting" in result.stdout


class TestDashboardCommand:
    """Tests for the dashboard command."""

    def test_dashboard_starts_with_default_options(self, tmp_path: Path) -> None:
        """Test dashboard command starts server with default options (mocked)."""
        import sys
        mock_uvicorn = AsyncMock()
        mock_uvicorn.run = lambda *args, **kwargs: None

        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            result = runner.invoke(
                app, ["dashboard", "--workspace", str(tmp_path)]
            )

            assert result.exit_code == 0
            assert "Mozart Dashboard" in result.stdout
            assert "http://127.0.0.1:8000" in result.stdout
            assert "Docs:" in result.stdout

    def test_dashboard_custom_port(self, tmp_path: Path) -> None:
        """Test dashboard command with custom port."""
        import sys
        mock_uvicorn = AsyncMock()
        mock_uvicorn.run = lambda *args, **kwargs: None

        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            result = runner.invoke(
                app, ["dashboard", "--port", "3000", "--workspace", str(tmp_path)]
            )

            assert result.exit_code == 0
            assert "http://127.0.0.1:3000" in result.stdout

    def test_dashboard_custom_host(self, tmp_path: Path) -> None:
        """Test dashboard command with custom host."""
        import sys
        mock_uvicorn = AsyncMock()
        mock_uvicorn.run = lambda *args, **kwargs: None

        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            result = runner.invoke(
                app, ["dashboard", "--host", "0.0.0.0", "--workspace", str(tmp_path)]
            )

            assert result.exit_code == 0
            assert "http://0.0.0.0:8000" in result.stdout

    def test_dashboard_uses_sqlite_when_available(self, tmp_path: Path) -> None:
        """Test dashboard prefers SQLite backend when db exists."""
        import sys

        # Create a mock SQLite database file
        sqlite_path = tmp_path / ".mozart-state.db"
        sqlite_path.touch()

        mock_uvicorn = AsyncMock()
        mock_uvicorn.run = lambda *args, **kwargs: None

        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            result = runner.invoke(
                app, ["dashboard", "--workspace", str(tmp_path)]
            )

            assert result.exit_code == 0
            assert "SQLite state backend" in result.stdout

    def test_dashboard_falls_back_to_json_backend(self, tmp_path: Path) -> None:
        """Test dashboard falls back to JSON backend when no SQLite db."""
        import sys
        mock_uvicorn = AsyncMock()
        mock_uvicorn.run = lambda *args, **kwargs: None

        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            result = runner.invoke(
                app, ["dashboard", "--workspace", str(tmp_path)]
            )

            assert result.exit_code == 0
            assert "JSON state backend" in result.stdout

    def test_dashboard_shows_docs_url(self, tmp_path: Path) -> None:
        """Test dashboard shows Swagger docs URL in startup message."""
        import sys
        mock_uvicorn = AsyncMock()
        mock_uvicorn.run = lambda *args, **kwargs: None

        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            result = runner.invoke(
                app, ["dashboard", "--workspace", str(tmp_path)]
            )

            assert result.exit_code == 0
            assert "/docs" in result.stdout
            assert "/openapi.json" in result.stdout

    def test_dashboard_creates_app_with_correct_settings(self, tmp_path: Path) -> None:
        """Test dashboard creates app with correct title and workspace."""
        import sys
        mock_uvicorn = AsyncMock()
        mock_uvicorn.run = lambda *args, **kwargs: None

        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            result = runner.invoke(
                app, ["dashboard", "--workspace", str(tmp_path), "--port", "9000"]
            )

            assert result.exit_code == 0
            # Verify the startup panel shows correct info
            assert "Mozart Dashboard" in result.stdout
            assert "http://127.0.0.1:9000" in result.stdout
            assert "Starting Server" in result.stdout


class TestVerboseAndQuietFlags:
    """Tests for --verbose and --quiet global flags."""

    def test_verbose_flag_short(self) -> None:
        """Test -v flag is accepted."""
        result = runner.invoke(app, ["-v", "--version"])
        assert result.exit_code == 0

    def test_verbose_flag_long(self) -> None:
        """Test --verbose flag is accepted."""
        result = runner.invoke(app, ["--verbose", "--version"])
        assert result.exit_code == 0

    def test_quiet_flag_short(self) -> None:
        """Test -q flag is accepted."""
        result = runner.invoke(app, ["-q", "--version"])
        assert result.exit_code == 0

    def test_quiet_flag_long(self) -> None:
        """Test --quiet flag is accepted."""
        result = runner.invoke(app, ["--quiet", "--version"])
        assert result.exit_code == 0

    def test_version_flag_changed_to_capital_v(self) -> None:
        """Test --version uses -V (capital) shorthand."""
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert "Mozart AI Compose" in result.stdout


class TestRunCommandJsonOutput:
    """Tests for the --json flag on run command."""

    def test_run_dry_run_with_json_flag(self, sample_yaml_config: Path) -> None:
        """Test run --dry-run --json outputs valid JSON."""
        result = runner.invoke(
            app, ["run", str(sample_yaml_config), "--dry-run", "--json"]
        )
        assert result.exit_code == 0

        # Parse output as JSON
        output_data = json.loads(result.stdout)
        assert output_data["dry_run"] is True
        assert "job_name" in output_data
        assert "total_batches" in output_data

    def test_run_json_short_flag(self, sample_yaml_config: Path) -> None:
        """Test run --dry-run -j uses short flag."""
        result = runner.invoke(
            app, ["run", str(sample_yaml_config), "--dry-run", "-j"]
        )
        assert result.exit_code == 0

        # Should be valid JSON
        output_data = json.loads(result.stdout)
        assert output_data["dry_run"] is True

    def test_run_quiet_mode_hides_config_panel(self, sample_yaml_config: Path) -> None:
        """Test run with --quiet hides job configuration panel."""
        result = runner.invoke(
            app, ["-q", "run", str(sample_yaml_config), "--dry-run"]
        )
        assert result.exit_code == 0
        # In quiet mode, config panel is hidden
        assert "Job Configuration" not in result.stdout


class TestRunSummaryDisplay:
    """Tests for run summary display (indirect via CLI)."""

    def test_run_dry_run_shows_batch_plan(self, sample_yaml_config: Path) -> None:
        """Test dry-run still shows batch plan (not summary)."""
        result = runner.invoke(app, ["run", str(sample_yaml_config), "--dry-run"])
        assert result.exit_code == 0
        assert "Batch Plan" in result.stdout
        # Dry run shouldn't show summary (job wasn't run)
        assert "Run Summary" not in result.stdout
