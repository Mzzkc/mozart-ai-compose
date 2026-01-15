"""Tests for dashboard API routes."""
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from mozart.core.checkpoint import CheckpointState, JobStatus
from mozart.dashboard.app import create_app
from mozart.dashboard.services.job_control import JobStartResult, JobActionResult
from mozart.state.json_backend import JsonStateBackend


@pytest.fixture
def temp_state_dir():
    """Create temporary directory for state backend."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def app(temp_state_dir):
    """Create test app with temp state backend."""
    backend = JsonStateBackend(temp_state_dir)
    return create_app(state_backend=backend, cors_origins=["*"])


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_job_state():
    """Create sample job state."""
    return CheckpointState(
        job_id="test-job-123",
        job_name="Test Job",
        status=JobStatus.RUNNING,
        total_sheets=5,
        last_completed_sheet=2,
        current_sheet=3,
        worktree_path="/tmp/test-workspace",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_config_yaml():
    """Sample YAML config content."""
    return """
name: Test Job
workspace: ./test-workspace
sheet:
  total_sheets: 3
  template: |
    SHEET: {{sheet_num}}
    Test sheet content
backend:
  type: claude_cli
"""


class TestJobRoutes:
    """Test job control routes."""

    def test_start_job_with_config_content(self, client, sample_config_yaml):
        """Test starting job with inline config content."""
        with patch('mozart.dashboard.services.job_control.JobControlService.start_job') as mock_start:
            mock_start.return_value = JobStartResult(
                job_id="test-123",
                job_name="Test Job",
                status="running",
                workspace=Path("./test-workspace"),
                total_sheets=3,
                pid=12345
            )

            response = client.post("/api/jobs", json={
                "config_content": sample_config_yaml,
                "workspace": "./custom-workspace",
                "start_sheet": 1,
                "self_healing": True
            })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["job_id"] == "test-123"
        assert data["job_name"] == "Test Job"
        assert data["status"] == "running"
        assert data["workspace"] == "test-workspace"
        assert data["total_sheets"] == 3
        assert data["pid"] == 12345

        # Verify service was called correctly
        mock_start.assert_called_once()
        args, kwargs = mock_start.call_args
        assert kwargs["config_content"] == sample_config_yaml
        assert kwargs["workspace"] == Path("./custom-workspace")
        assert kwargs["start_sheet"] == 1
        assert kwargs["self_healing"] is True

    def test_start_job_with_config_path(self, client, temp_state_dir, sample_config_yaml):
        """Test starting job with config file path."""
        # Create config file
        config_file = temp_state_dir / "test-config.yaml"
        config_file.write_text(sample_config_yaml)

        with patch('mozart.dashboard.services.job_control.JobControlService.start_job') as mock_start:
            mock_start.return_value = JobStartResult(
                job_id="test-456",
                job_name="Test Job",
                status="running",
                workspace=Path("./test-workspace"),
                total_sheets=3,
                pid=54321
            )

            response = client.post("/api/jobs", json={
                "config_path": str(config_file)
            })

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-456"

        # Verify service was called with correct path
        mock_start.assert_called_once()
        args, kwargs = mock_start.call_args
        assert kwargs["config_path"] == Path(str(config_file))
        assert kwargs["config_content"] is None

    def test_start_job_validation_error(self, client):
        """Test validation error when no config provided."""
        response = client.post("/api/jobs", json={
            "workspace": "./test"
        })

        assert response.status_code == 400
        assert "Must provide either config_content or config_path" in response.json()["detail"]

    def test_start_job_both_configs_error(self, client, sample_config_yaml):
        """Test validation error when both configs provided."""
        response = client.post("/api/jobs", json={
            "config_content": sample_config_yaml,
            "config_path": "/some/path.yaml"
        })

        assert response.status_code == 400
        assert "Cannot provide both config_content and config_path" in response.json()["detail"]

    def test_start_job_file_not_found(self, client):
        """Test file not found error."""
        with patch('mozart.dashboard.services.job_control.JobControlService.start_job') as mock_start:
            mock_start.side_effect = FileNotFoundError("Config file not found: /nonexistent.yaml")

            response = client.post("/api/jobs", json={
                "config_path": "/nonexistent.yaml"
            })

        assert response.status_code == 404
        assert "Config file not found" in response.json()["detail"]

    def test_start_job_runtime_error(self, client, sample_config_yaml):
        """Test runtime error during job start."""
        with patch('mozart.dashboard.services.job_control.JobControlService.start_job') as mock_start:
            mock_start.side_effect = RuntimeError("Failed to start job: Permission denied")

            response = client.post("/api/jobs", json={
                "config_content": sample_config_yaml
            })

        assert response.status_code == 500
        assert "Failed to start job" in response.json()["detail"]

    def test_pause_job_success(self, client):
        """Test successful job pause."""
        with patch('mozart.dashboard.services.job_control.JobControlService.pause_job') as mock_pause:
            mock_pause.return_value = JobActionResult(
                success=True,
                job_id="test-123",
                status="paused",
                message="Job test-123 paused successfully"
            )

            response = client.post("/api/jobs/test-123/pause")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["job_id"] == "test-123"
        assert data["status"] == "paused"
        assert "paused successfully" in data["message"]

    def test_pause_job_not_found(self, client):
        """Test pausing non-existent job."""
        with patch('mozart.dashboard.services.job_control.JobControlService.pause_job') as mock_pause:
            mock_pause.return_value = JobActionResult(
                success=False,
                job_id="nonexistent",
                status="failed",
                message="Job not found: nonexistent"
            )

            response = client.post("/api/jobs/nonexistent/pause")

        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]

    def test_resume_job_success(self, client):
        """Test successful job resume."""
        with patch('mozart.dashboard.services.job_control.JobControlService.resume_job') as mock_resume:
            mock_resume.return_value = JobActionResult(
                success=True,
                job_id="test-123",
                status="running",
                message="Job test-123 resumed successfully"
            )

            response = client.post("/api/jobs/test-123/resume")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "running"

    def test_cancel_job_success(self, client):
        """Test successful job cancellation."""
        with patch('mozart.dashboard.services.job_control.JobControlService.cancel_job') as mock_cancel:
            mock_cancel.return_value = JobActionResult(
                success=True,
                job_id="test-123",
                status="cancelled",
                message="Job test-123 cancelled successfully"
            )

            response = client.post("/api/jobs/test-123/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "cancelled"

    def test_delete_job_success(self, client):
        """Test successful job deletion."""
        with patch('mozart.dashboard.services.job_control.JobControlService.delete_job') as mock_delete:
            mock_delete.return_value = True

            response = client.delete("/api/jobs/test-123")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["job_id"] == "test-123"
        assert "deleted successfully" in data["message"]

    def test_delete_job_not_found(self, client):
        """Test deleting non-existent job."""
        with patch('mozart.dashboard.services.job_control.JobControlService.delete_job') as mock_delete:

            mock_delete.return_value = False

            response = client.delete("/api/jobs/nonexistent")

        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]

    def test_delete_job_running(self, client, sample_job_state):
        """Test deleting running job (should fail)."""
        with patch('mozart.dashboard.services.job_control.JobControlService.delete_job') as mock_delete:

            mock_delete.return_value = False

            response = client.delete("/api/jobs/test-job-123")

        assert response.status_code == 409
        assert "Cannot delete running job" in response.json()["detail"]


class TestArtifactRoutes:
    """Test artifact/workspace file routes."""

    def test_list_artifacts_worktree_job(self, client, temp_state_dir):
        """Test listing artifacts for worktree-isolated job."""
        # Create test workspace with files
        workspace = temp_state_dir / "test-workspace"
        workspace.mkdir()
        (workspace / "file1.txt").write_text("content1")
        (workspace / "file2.md").write_text("# Markdown")
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "file3.py").write_text("print('hello')")

        # Mock job state with worktree path
        job_state = CheckpointState(
            job_id="test-123",
            job_name="Test Job",
            status=JobStatus.RUNNING,
            total_sheets=3,
            worktree_path=str(workspace),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=job_state)
            mock_get_backend.return_value = mock_backend

            response = client.get("/api/jobs/test-123/artifacts")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-123"
        assert data["workspace"] == str(workspace)
        assert data["total_files"] == 4  # 2 files + 1 dir + 1 file in subdir

        # Check file info
        files = {f["name"]: f for f in data["files"]}
        assert "file1.txt" in files
        assert files["file1.txt"]["type"] == "file"
        assert files["file1.txt"]["size"] == 8  # len("content1")
        assert "subdir" in files
        assert files["subdir"]["type"] == "directory"

    def test_list_artifacts_non_recursive(self, client, temp_state_dir):
        """Test non-recursive artifact listing."""
        workspace = temp_state_dir / "test-workspace"
        workspace.mkdir()
        (workspace / "file1.txt").write_text("content1")
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "hidden.txt").write_text("should not appear")

        job_state = CheckpointState(
            job_id="test-123",
            job_name="Test Job",
            status=JobStatus.RUNNING,
            total_sheets=3,
            worktree_path=str(workspace),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=job_state)
            mock_get_backend.return_value = mock_backend

            response = client.get("/api/jobs/test-123/artifacts?recursive=false")

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 2  # Only file1.txt and subdir
        file_names = [f["name"] for f in data["files"]]
        assert "file1.txt" in file_names
        assert "subdir" in file_names
        assert "hidden.txt" not in file_names

    def test_list_artifacts_job_not_found(self, client):
        """Test listing artifacts for non-existent job."""
        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=None)
            mock_get_backend.return_value = mock_backend

            response = client.get("/api/jobs/nonexistent/artifacts")

        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]

    def test_list_artifacts_no_worktree(self, client):
        """Test listing artifacts for job without worktree isolation."""
        job_state = CheckpointState(
            job_id="test-123",
            job_name="Test Job",
            status=JobStatus.RUNNING,
            total_sheets=3,
            worktree_path=None,  # No worktree isolation
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=job_state)
            mock_get_backend.return_value = mock_backend

            response = client.get("/api/jobs/test-123/artifacts")

        assert response.status_code == 404
        assert "No accessible workspace found" in response.json()["detail"]

    def test_get_artifact_text_file(self, client, temp_state_dir):
        """Test getting text file content."""
        workspace = temp_state_dir / "test-workspace"
        workspace.mkdir()
        test_file = workspace / "test.txt"
        test_content = "This is test content\nLine 2\nLine 3"
        test_file.write_text(test_content)

        job_state = CheckpointState(
            job_id="test-123",
            job_name="Test Job",
            status=JobStatus.RUNNING,
            total_sheets=3,
            worktree_path=str(workspace),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=job_state)
            mock_get_backend.return_value = mock_backend

            response = client.get("/api/jobs/test-123/artifacts/test.txt")

        assert response.status_code == 200
        assert response.text == test_content
        assert "text/plain" in response.headers["content-type"]

    def test_get_artifact_download_mode(self, client, temp_state_dir):
        """Test getting file in download mode."""
        workspace = temp_state_dir / "test-workspace"
        workspace.mkdir()
        test_file = workspace / "test.txt"
        test_file.write_text("test content")

        job_state = CheckpointState(
            job_id="test-123",
            job_name="Test Job",
            status=JobStatus.RUNNING,
            total_sheets=3,
            worktree_path=str(workspace),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=job_state)
            mock_get_backend.return_value = mock_backend

            response = client.get("/api/jobs/test-123/artifacts/test.txt?download=true")

        assert response.status_code == 200
        # Should have attachment header for download
        assert "attachment" in response.headers.get("content-disposition", "")

    def test_get_artifact_file_not_found(self, client, temp_state_dir):
        """Test getting non-existent file."""
        workspace = temp_state_dir / "test-workspace"
        workspace.mkdir()

        job_state = CheckpointState(
            job_id="test-123",
            job_name="Test Job",
            status=JobStatus.RUNNING,
            total_sheets=3,
            worktree_path=str(workspace),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=job_state)
            mock_get_backend.return_value = mock_backend

            response = client.get("/api/jobs/test-123/artifacts/nonexistent.txt")

        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]

    def test_get_artifact_directory_traversal_blocked(self, client, temp_state_dir):
        """Test that directory traversal is blocked."""
        workspace = temp_state_dir / "test-workspace"
        workspace.mkdir()

        job_state = CheckpointState(
            job_id="test-123",
            job_name="Test Job",
            status=JobStatus.RUNNING,
            total_sheets=3,
            worktree_path=str(workspace),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=job_state)
            mock_get_backend.return_value = mock_backend

            response = client.get("/api/jobs/test-123/artifacts/../../etc/passwd")

        assert response.status_code == 400
        assert "Invalid path" in response.json()["detail"]


class TestStreamRoutes:
    """Test SSE streaming routes."""

    def test_stream_job_status_job_not_found(self, client):
        """Test streaming status for non-existent job."""
        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=None)
            mock_get_backend.return_value = mock_backend

            response = client.get("/api/jobs/nonexistent/stream")

        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]

    def test_stream_job_status_invalid_poll_interval(self, client, sample_job_state):
        """Test streaming with invalid poll interval."""
        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=sample_job_state)
            mock_get_backend.return_value = mock_backend

            response = client.get("/api/jobs/test-123/stream?poll_interval=50")

        assert response.status_code == 400
        assert "Poll interval must be between" in response.json()["detail"]

    def test_stream_logs_invalid_tail_lines(self, client, sample_job_state):
        """Test log streaming with invalid tail_lines."""
        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=sample_job_state)
            mock_get_backend.return_value = mock_backend

            response = client.get("/api/jobs/test-123/logs?tail_lines=2000")

        assert response.status_code == 400
        assert "tail_lines must be between" in response.json()["detail"]

    def test_download_logs_success(self, client, temp_state_dir):
        """Test static log download."""
        workspace = temp_state_dir / "test-workspace"
        workspace.mkdir()
        log_file = workspace / "mozart.log"
        log_content = "LOG: Job started\nLOG: Sheet 1 completed\nLOG: Job finished"
        log_file.write_text(log_content)

        job_state = CheckpointState(
            job_id="test-123",
            job_name="Test Job",
            status=JobStatus.COMPLETED,
            total_sheets=1,
            worktree_path=str(workspace),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=job_state)
            mock_get_backend.return_value = mock_backend

            response = client.get("/api/jobs/test-123/logs/static")

        assert response.status_code == 200
        assert log_content in response.text
        assert "attachment" in response.headers["content-disposition"]
        assert "mozart-test-123-logs.txt" in response.headers["content-disposition"]

    def test_get_log_info_success(self, client, temp_state_dir):
        """Test getting log file info."""
        workspace = temp_state_dir / "test-workspace"
        workspace.mkdir()
        log_file = workspace / "mozart.log"
        log_content = "Line 1\nLine 2\nLine 3\n"
        log_file.write_text(log_content)

        job_state = CheckpointState(
            job_id="test-123",
            job_name="Test Job",
            status=JobStatus.COMPLETED,
            total_sheets=1,
            worktree_path=str(workspace),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=job_state)
            mock_get_backend.return_value = mock_backend

            response = client.get("/api/jobs/test-123/logs/info")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-123"
        assert data["log_file"] == "mozart.log"
        assert data["size_bytes"] == len(log_content.encode())
        assert data["lines"] == 3