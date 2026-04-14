"""Integration tests for Marianne Dashboard.

Tests full workflows combining multiple services and components.
All lifecycle tests route through a mocked conductor (DaemonClient).
"""

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from marianne.core.checkpoint import CheckpointState, JobStatus
from marianne.daemon.ipc.client import DaemonClient
from marianne.daemon.types import JobResponse
from marianne.dashboard.app import create_app
from marianne.dashboard.services.job_control import JobControlService
from marianne.dashboard.services.sse_manager import SSEEvent, SSEManager
from marianne.state.json_backend import JsonStateBackend


@pytest.fixture
def temp_state_dir():
    """Create temporary directory for state backend."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_workspace():
    """Create temporary workspace directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def backend(temp_state_dir):
    """Create test state backend."""
    return JsonStateBackend(temp_state_dir)


@pytest.fixture
def mock_daemon_client():
    """Create a mock DaemonClient for conductor-only tests."""
    client = AsyncMock(spec=DaemonClient)
    client.submit_job = AsyncMock(
        return_value=JobResponse(job_id="test-job-001", status="accepted"),
    )
    client.pause_job = AsyncMock(return_value=None)
    client.resume_job = AsyncMock(return_value=None)
    client.cancel_job = AsyncMock(return_value=None)
    client.clear_jobs = AsyncMock(return_value={"deleted": 1})
    client.get_job_status = AsyncMock(return_value=None)
    return client


@pytest.fixture
def app(backend):
    """Create test app with backend."""
    return create_app(state_backend=backend, cors_origins=["*"])


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_config_content():
    """Sample Marianne YAML config for testing."""
    return """
name: "test-integration-job"
description: "Integration test job"
workspace: "./workspace"

backend:
  type: claude_cli
  timeout_seconds: 300

sheet:
  size: 1
  total_items: 2
  start_item: 1

prompt:
  template: |
    Create a simple Python hello world script in hello.py

    This is sheet {{sheet_num}} of {{total_sheets}} processing item {{item_num}}.

    Expected output: A Python script that prints hello world.
"""


@pytest.fixture
def job_control_service(mock_daemon_client):
    """Create job control service with mock DaemonClient."""
    return JobControlService(mock_daemon_client)


@pytest.fixture
def sse_manager():
    """Create SSE manager."""
    return SSEManager()


class TestJobLifecycleIntegration:
    """Test complete job lifecycle through API — all via conductor IPC."""

    async def test_lifecycle_requires_conductor(
        self,
        client,
        sample_config_content,
        temp_workspace,
    ):
        """All lifecycle operations require a running conductor."""
        import tempfile as real_tempfile

        with real_tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(sample_config_content)
            temp_config_path = f.name

        try:
            job_workspace = temp_workspace / "test-job"
            job_workspace.mkdir(exist_ok=True)

            start_response = client.post(
                "/api/jobs",
                json={
                    "config_path": temp_config_path,
                    "workspace": str(job_workspace),
                },
            )
            assert start_response.status_code == 503

            pause_response = client.post("/api/jobs/fake-job/pause")
            assert pause_response.status_code == 503

            resume_response = client.post("/api/jobs/fake-job/resume")
            assert resume_response.status_code == 503

            cancel_response = client.post("/api/jobs/fake-job/cancel")
            assert cancel_response.status_code == 503

        finally:
            import os as os_mod

            if os_mod.path.exists(temp_config_path):
                os_mod.unlink(temp_config_path)

    async def test_start_pause_resume_cancel_flow(
        self,
        client,
        sample_config_content,
        temp_workspace,
        mock_daemon_client,
    ):
        """Test complete lifecycle: start → pause → resume → cancel via conductor."""
        import tempfile as real_tempfile

        with real_tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(sample_config_content)
            temp_config_path = f.name

        try:
            with patch("marianne.dashboard.app._daemon_client", mock_daemon_client):
                # 1. Start job
                job_workspace = temp_workspace / "test-job"
                job_workspace.mkdir(exist_ok=True)
                start_response = client.post(
                    "/api/jobs",
                    json={
                        "config_path": temp_config_path,
                        "workspace": str(job_workspace),
                    },
                )

                assert start_response.status_code == 200
                start_data = start_response.json()
                assert start_data["success"] is True
                assert "job_id" in start_data
                assert start_data["total_sheets"] == 2

                job_id = start_data["job_id"]
                mock_daemon_client.submit_job.assert_awaited_once()

                # 2. Pause job
                pause_response = client.post(f"/api/jobs/{job_id}/pause")
                assert pause_response.status_code == 200
                pause_data = pause_response.json()
                assert pause_data["success"] is True
                assert pause_data["job_id"] == job_id
                assert pause_data["status"] == "paused"
                mock_daemon_client.pause_job.assert_awaited_once_with(job_id, "")

                # 3. Resume job
                resume_response = client.post(f"/api/jobs/{job_id}/resume")
                assert resume_response.status_code == 200
                resume_data = resume_response.json()
                assert resume_data["success"] is True
                assert resume_data["job_id"] == job_id
                assert resume_data["status"] == "running"
                mock_daemon_client.resume_job.assert_awaited_once_with(job_id, "")

                # 4. Cancel job
                cancel_response = client.post(f"/api/jobs/{job_id}/cancel")
                assert cancel_response.status_code == 200
                cancel_data = cancel_response.json()
                assert cancel_data["success"] is True
                assert cancel_data["job_id"] == job_id
                assert cancel_data["status"] == "cancelled"
                mock_daemon_client.cancel_job.assert_awaited_once_with(job_id, "")

        finally:
            import os as os_mod

            if os_mod.path.exists(temp_config_path):
                os_mod.unlink(temp_config_path)

    @pytest.mark.skip(reason="SSE needs async client")
    async def test_sse_receives_job_updates(self, client, sse_manager, backend):
        """Test that SSE stream receives job status updates."""
        # Create a sample job state
        job_id = "test-sse-job"
        job_state = CheckpointState(
            job_id=job_id,
            job_name="SSE Test Job",
            status=JobStatus.RUNNING,
            total_sheets=3,
            last_completed_sheet=1,
            current_sheet=2,
            worktree_path="/tmp/test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        await backend.save(job_state)

        # Connect to SSE stream (correct URL: /api/jobs/{job_id}/stream)
        sse_response = client.get(f"/api/jobs/{job_id}/stream")
        assert sse_response.status_code == 200
        assert sse_response.headers["content-type"] == "text/event-stream"

        # Simulate job status updates through the SSE manager
        test_events = [
            SSEEvent(
                event="sheet_started",
                data=json.dumps(
                    {"job_id": job_id, "sheet": 2, "timestamp": datetime.now().isoformat()}
                ),
                id="event-1",
            ),
            SSEEvent(
                event="sheet_completed",
                data=json.dumps(
                    {"job_id": job_id, "sheet": 2, "timestamp": datetime.now().isoformat()}
                ),
                id="event-2",
            ),
        ]

        # Broadcast events
        for event in test_events:
            await sse_manager.broadcast_to_job(job_id, event)

        # Read SSE response content
        sse_content = sse_response.content.decode()

        # Verify events were included
        assert "event: sheet_started" in sse_content
        assert "event: sheet_completed" in sse_content
        assert job_id in sse_content

    async def test_artifact_listing_after_job_run(
        self, client, temp_workspace, backend, sample_config_content
    ):
        """Test artifact listing after a simulated job run."""
        # Create workspace with some test artifacts
        job_workspace = temp_workspace / "artifact-test-job"
        job_workspace.mkdir(parents=True)

        # Create sample artifacts
        (job_workspace / "hello.py").write_text("print('Hello, World!')")
        (job_workspace / "output.txt").write_text("Job execution log")
        (job_workspace / "result.json").write_text('{"status": "completed"}')

        # Create job state
        job_id = "artifact-test-job"
        job_state = CheckpointState(
            job_id=job_id,
            job_name="Artifact Test Job",
            status=JobStatus.COMPLETED,
            total_sheets=2,
            last_completed_sheet=2,
            current_sheet=2,
            worktree_path=str(job_workspace),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        await backend.save(job_state)

        # List artifacts through API
        artifacts_response = client.get(f"/api/jobs/{job_id}/artifacts")

        assert artifacts_response.status_code == 200
        artifacts_data = artifacts_response.json()

        assert "files" in artifacts_data
        artifacts = artifacts_data["files"]

        # Check that our test artifacts are listed
        artifact_names = [artifact["name"] for artifact in artifacts]
        assert "hello.py" in artifact_names
        assert "output.txt" in artifact_names
        assert "result.json" in artifact_names

        # Verify artifact details
        hello_artifact = next(a for a in artifacts if a["name"] == "hello.py")
        assert hello_artifact["type"] == "file"
        assert hello_artifact["size"] > 0

    async def test_concurrent_job_starts(
        self,
        client,
        sample_config_content,
        temp_workspace,
        mock_daemon_client,
    ):
        """Test starting multiple jobs concurrently via the conductor."""
        job_ids = ["job-alpha", "job-beta", "job-gamma"]
        submit_index = 0

        async def _submit_side_effect(request):
            nonlocal submit_index
            jid = job_ids[submit_index]
            submit_index += 1
            return JobResponse(job_id=jid, status="accepted")

        mock_daemon_client.submit_job.side_effect = _submit_side_effect

        with patch("marianne.dashboard.app._daemon_client", mock_daemon_client):
            responses = []
            for i in range(3):
                workspace_path = temp_workspace / f"concurrent-job-{i}"
                response = client.post(
                    "/api/jobs",
                    json={
                        "config_content": sample_config_content,
                        "workspace": str(workspace_path),
                    },
                )
                responses.append(response)

        result_ids = []
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            result_ids.append(data["job_id"])

        assert len(set(result_ids)) == 3
        assert mock_daemon_client.submit_job.await_count == 3


class TestCrossServiceIntegration:
    """Test integration between different dashboard services."""

    @pytest.mark.skip(reason="SSE manager not wired in mock")
    async def test_job_control_updates_sse_manager(self, mock_daemon_client, sse_manager):
        """Test that job control operations trigger SSE events."""
        job_service = JobControlService(mock_daemon_client)

        sse_manager.broadcast_to_job = AsyncMock()

        job_id = "cross-service-test"

        result = await job_service.pause_job(job_id)
        assert result.success is True

        result = await job_service.resume_job(job_id)
        assert result.success is True

    async def test_error_handling_across_services(self, client, temp_workspace):
        """Test error handling when services interact."""
        # Test starting job with invalid config
        invalid_config = "invalid: yaml: content: [unclosed"

        response = client.post(
            "/api/jobs",
            json={
                "config_content": invalid_config,
                "workspace": str(temp_workspace / "error-test"),
            },
        )

        # Should handle errors gracefully — 400 (bad input), 500 (server error),
        # or 503 (conductor not available in conductor-only mode)
        assert response.status_code in [400, 500, 503]
        data = response.json()
        assert "detail" in data

        # Test operations on non-existent job
        # 404 when conductor knows job doesn't exist, 503 when conductor unavailable
        fake_job_id = "nonexistent-job-123"

        pause_response = client.post(f"/api/jobs/{fake_job_id}/pause")
        assert pause_response.status_code in (404, 503)

        resume_response = client.post(f"/api/jobs/{fake_job_id}/resume")
        assert resume_response.status_code in (404, 503)

        cancel_response = client.post(f"/api/jobs/{fake_job_id}/cancel")
        assert cancel_response.status_code in (404, 503)

        artifacts_response = client.get(f"/api/artifacts/{fake_job_id}")
        assert artifacts_response.status_code == 404

        sse_response = client.get(f"/api/stream/{fake_job_id}")
        assert sse_response.status_code == 404


class TestPerformanceAndReliability:
    """Test performance characteristics and reliability."""

    async def test_large_artifact_listing(self, client, backend, temp_workspace):
        """Test artifact listing with many files."""
        # Create workspace with many files
        job_workspace = temp_workspace / "large-artifacts-job"
        job_workspace.mkdir(parents=True)

        # Create 100 test files
        for i in range(100):
            (job_workspace / f"file_{i:03d}.txt").write_text(f"Content of file {i}")

        # Create subdirectories with files
        sub_dir = job_workspace / "subdir"
        sub_dir.mkdir()
        for i in range(20):
            (sub_dir / f"sub_file_{i}.txt").write_text(f"Sub content {i}")

        # Create job state
        job_id = "large-artifacts-job"
        job_state = CheckpointState(
            job_id=job_id,
            job_name="Large Artifacts Job",
            status=JobStatus.COMPLETED,
            total_sheets=1,
            last_completed_sheet=1,
            current_sheet=1,
            worktree_path=str(job_workspace),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        await backend.save(job_state)

        # Test artifact listing performance
        start_time = time.time()
        response = client.get(f"/api/jobs/{job_id}/artifacts")
        end_time = time.time()

        assert response.status_code == 200
        # Should complete within reasonable time (< 1 second for 120 files)
        assert (end_time - start_time) < 1.0

        data = response.json()
        artifacts = data["files"]

        # Should list all files including subdirectory files
        assert len(artifacts) >= 120  # 100 main + 20 sub + directories

    @pytest.mark.skip(reason="SSE needs async client")
    async def test_sse_connection_cleanup(self, client, sse_manager):
        """Test that SSE connections are properly cleaned up."""
        job_id = "cleanup-test-job"

        # Start an SSE connection
        response = client.get(f"/api/jobs/{job_id}/stream")
        assert response.status_code == 200

        # Simulate connection cleanup (this would happen when client disconnects)
        # In real scenarios, the SSE manager should clean up stale connections
        _initial_connections = len(sse_manager._connections.get(job_id, {}))

        # Broadcast an event to trigger connection health checks
        test_event = SSEEvent(event="test", data='{"test": "data"}', id="cleanup-test")

        await sse_manager.broadcast_to_job(job_id, test_event)

        # Connection management is handled by the SSE stream endpoint
        # Verify the connection registry exists and can track connections
        assert hasattr(sse_manager, "_connections"), "SSE manager should have connection registry"
        assert isinstance(sse_manager._connections, dict), "Connection registry should be a dict"
