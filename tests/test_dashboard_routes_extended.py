"""Extended route tests for additional coverage."""
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


class TestJobRoutesExtended:
    """Extended tests for job routes."""

    def test_start_job_request_validation_both_configs(self, client, sample_config_yaml):
        """Test validation when both config_content and config_path are provided."""
        response = client.post("/api/jobs", json={
            "config_content": sample_config_yaml,
            "config_path": "/some/path.yaml"
        })

        assert response.status_code == 400
        assert "Cannot provide both config_content and config_path" in response.json()["detail"]

    def test_start_job_yaml_parsing_error(self, client):
        """Test start job with malformed YAML."""
        malformed_yaml = """
name: Test Job
    sheet:  # Invalid indentation
  total_sheets: 3
"""

        with patch('mozart.dashboard.services.job_control.JobControlService.start_job') as mock_start:
            mock_start.side_effect = RuntimeError("Failed to start job: Invalid YAML")

            response = client.post("/api/jobs", json={
                "config_content": malformed_yaml
            })

        assert response.status_code == 500
        assert "Failed to start job" in response.json()["detail"]

    def test_start_job_with_all_optional_parameters(self, client, sample_config_yaml):
        """Test starting job with all optional parameters."""
        with patch('mozart.dashboard.services.job_control.JobControlService.start_job') as mock_start:
            mock_start.return_value = JobStartResult(
                job_id="test-456",
                job_name="Test Job",
                status="running",
                workspace=Path("./custom-workspace"),
                total_sheets=3,
                pid=98765
            )

            response = client.post("/api/jobs", json={
                "config_content": sample_config_yaml,
                "workspace": "./custom-workspace",
                "start_sheet": 2,
                "self_healing": True
            })

        assert response.status_code == 200
        data = response.json()
        assert data["workspace"] == "custom-workspace"
        assert data["pid"] == 98765

        # Verify all parameters were passed
        mock_start.assert_called_once()
        args, kwargs = mock_start.call_args
        assert kwargs["start_sheet"] == 2
        assert kwargs["self_healing"] is True

    def test_pause_job_already_paused(self, client):
        """Test pausing job that's already paused."""
        with patch('mozart.dashboard.services.job_control.JobControlService.pause_job') as mock_pause:
            mock_pause.return_value = JobActionResult(
                success=False,
                job_id="test-123",
                status="paused",
                message="Job is not running (status: paused)"
            )

            response = client.post("/api/jobs/test-123/pause")

        assert response.status_code == 409
        assert "Job is not running" in response.json()["detail"]

    def test_resume_job_not_found(self, client):
        """Test resuming non-existent job."""
        with patch('mozart.dashboard.services.job_control.JobControlService.resume_job') as mock_resume:
            mock_resume.return_value = JobActionResult(
                success=False,
                job_id="nonexistent",
                status="failed",
                message="Job not found: nonexistent"
            )

            response = client.post("/api/jobs/nonexistent/resume")

        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]

    def test_cancel_job_already_finished(self, client):
        """Test cancelling job that's already finished."""
        with patch('mozart.dashboard.services.job_control.JobControlService.cancel_job') as mock_cancel:
            mock_cancel.return_value = JobActionResult(
                success=False,
                job_id="test-123",
                status="completed",
                message="Job already finished (status: completed)"
            )

            response = client.post("/api/jobs/test-123/cancel")

        assert response.status_code == 409
        assert "Job already finished" in response.json()["detail"]

    def test_delete_job_currently_paused(self, client):
        """Test deleting paused job (should work)."""
        with patch('mozart.dashboard.services.job_control.JobControlService.delete_job') as mock_delete, \
             patch('mozart.dashboard.services.job_control.JobControlService._state_backend') as mock_backend:

            mock_delete.return_value = True

            # Mock paused job
            paused_job = CheckpointState(
                job_id="test-123",
                job_name="Test Job",
                status=JobStatus.PAUSED,
                total_sheets=3,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            mock_backend.load = AsyncMock(return_value=paused_job)

            response = client.delete("/api/jobs/test-123")

        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]


class TestArtifactRoutesExtended:
    """Extended tests for artifact routes."""

    def test_list_artifacts_with_filter_parameter(self, client, temp_state_dir):
        """Test listing artifacts with additional query parameters."""
        workspace = temp_state_dir / "test-workspace"
        workspace.mkdir()
        (workspace / "important.txt").write_text("important")
        (workspace / "temp.log").write_text("temporary")
        (workspace / "data.json").write_text('{"key": "value"}')

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

            # Test with recursive=true (default behavior)
            response = client.get("/api/jobs/test-123/artifacts?recursive=true")

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 3
        file_names = [f["name"] for f in data["files"]]
        assert "important.txt" in file_names
        assert "temp.log" in file_names
        assert "data.json" in file_names

    def test_get_artifact_binary_file(self, client, temp_state_dir):
        """Test getting binary file content."""
        workspace = temp_state_dir / "test-workspace"
        workspace.mkdir()
        binary_file = workspace / "test.bin"
        binary_content = b'\x00\x01\x02\x03\xFF'
        binary_file.write_bytes(binary_content)

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

            response = client.get("/api/jobs/test-123/artifacts/test.bin")

        assert response.status_code == 200
        assert response.content == binary_content
        assert "application/octet-stream" in response.headers["content-type"]

    def test_get_artifact_subdirectory_file(self, client, temp_state_dir):
        """Test getting file from subdirectory."""
        workspace = temp_state_dir / "test-workspace"
        workspace.mkdir()
        subdir = workspace / "logs"
        subdir.mkdir()
        log_file = subdir / "debug.log"
        log_content = "DEBUG: Application started\nINFO: Processing request"
        log_file.write_text(log_content)

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

            response = client.get("/api/jobs/test-123/artifacts/logs/debug.log")

        assert response.status_code == 200
        assert response.text == log_content

    def test_get_artifact_path_traversal_variations(self, client, temp_state_dir):
        """Test various path traversal attack patterns."""
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

            # Test different traversal patterns
            traversal_patterns = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "....//....//....//etc//passwd",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
            ]

            for pattern in traversal_patterns:
                response = client.get(f"/api/jobs/test-123/artifacts/{pattern}")
                assert response.status_code == 400
                assert "Invalid path" in response.json()["detail"]


class TestStreamRoutesExtended:
    """Extended tests for streaming routes."""

    def test_stream_job_status_edge_case_poll_intervals(self, client, temp_state_dir):
        """Test streaming with edge case poll intervals."""
        job_state = CheckpointState(
            job_id="test-123",
            job_name="Test Job",
            status=JobStatus.RUNNING,
            total_sheets=3,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=job_state)
            mock_get_backend.return_value = mock_backend

            # Test minimum valid poll interval
            response = client.get("/api/jobs/test-123/stream?poll_interval=100")
            assert response.status_code == 200

            # Test maximum valid poll interval
            response = client.get("/api/jobs/test-123/stream?poll_interval=60000")
            assert response.status_code == 200

            # Test just below minimum
            response = client.get("/api/jobs/test-123/stream?poll_interval=99")
            assert response.status_code == 400

            # Test just above maximum
            response = client.get("/api/jobs/test-123/stream?poll_interval=60001")
            assert response.status_code == 400

    def test_log_streaming_edge_case_tail_lines(self, client, temp_state_dir):
        """Test log streaming with edge case tail_lines values."""
        job_state = CheckpointState(
            job_id="test-123",
            job_name="Test Job",
            status=JobStatus.RUNNING,
            total_sheets=3,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch('mozart.dashboard.app.get_state_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.load = AsyncMock(return_value=job_state)
            mock_get_backend.return_value = mock_backend

            # Test minimum valid tail_lines
            response = client.get("/api/jobs/test-123/logs?tail_lines=1")
            assert response.status_code == 200

            # Test maximum valid tail_lines
            response = client.get("/api/jobs/test-123/logs?tail_lines=1000")
            assert response.status_code == 200

            # Test zero tail_lines (invalid)
            response = client.get("/api/jobs/test-123/logs?tail_lines=0")
            assert response.status_code == 400

            # Test negative tail_lines
            response = client.get("/api/jobs/test-123/logs?tail_lines=-5")
            assert response.status_code == 400

    def test_download_logs_no_log_file(self, client, temp_state_dir):
        """Test log download when no log file exists."""
        workspace = temp_state_dir / "test-workspace"
        workspace.mkdir()
        # No mozart.log file created

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

        assert response.status_code == 404
        assert "Log file not found" in response.json()["detail"]

    def test_get_log_info_no_log_file(self, client, temp_state_dir):
        """Test getting log info when no log file exists."""
        workspace = temp_state_dir / "test-workspace"
        workspace.mkdir()
        # No mozart.log file created

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

        assert response.status_code == 404
        assert "Log file not found" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__])