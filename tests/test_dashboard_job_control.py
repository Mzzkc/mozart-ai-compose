"""Tests for JobControlService — conductor-only proxy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marianne.daemon.exceptions import DaemonNotRunningError
from marianne.daemon.types import JobResponse
from marianne.dashboard.services.job_control import (
    JobActionResult,
    JobControlService,
    JobStartResult,
    ProcessHealth,
)


@pytest.fixture
def mock_daemon_client() -> MagicMock:
    """Create a mock DaemonClient."""
    client = MagicMock()
    client.submit_job = AsyncMock()
    client.pause_job = AsyncMock()
    client.resume_job = AsyncMock()
    client.cancel_job = AsyncMock()
    client.clear_jobs = AsyncMock()
    client.get_job_status = AsyncMock()
    return client


@pytest.fixture
def job_control_service(mock_daemon_client: MagicMock) -> JobControlService:
    """Fixture for JobControlService backed by mock DaemonClient."""
    return JobControlService(mock_daemon_client)


@pytest.fixture
def sample_yaml_config() -> str:
    return """
name: "test-job"
description: "Test job for unit tests"
workspace: "./test-workspace"
sheet:
  size: 10
  total_items: 20
prompt:
  template: "Process item {{item}}"
"""


@pytest.fixture
def sample_config_file(tmp_path: Path, sample_yaml_config: str) -> Path:
    config_file = tmp_path / "test-config.yaml"
    config_file.write_text(sample_yaml_config)
    return config_file


class TestJobControlServiceInit:
    def test_requires_daemon_client(self) -> None:
        with pytest.raises(ValueError, match="DaemonClient is required"):
            JobControlService(None)  # type: ignore[arg-type]

    def test_accepts_daemon_client(self, mock_daemon_client: MagicMock) -> None:
        service = JobControlService(mock_daemon_client)
        assert service._client is mock_daemon_client


class TestStartJob:
    @pytest.mark.asyncio
    async def test_start_job_success(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
        sample_config_file: Path,
    ) -> None:
        mock_daemon_client.submit_job.return_value = JobResponse(
            job_id="job-abc",
            status="accepted",
        )

        result = await job_control_service.start_job(config_path=sample_config_file)

        assert isinstance(result, JobStartResult)
        assert result.job_id == "job-abc"
        assert result.job_name == "test-job"
        assert result.status == "accepted"
        assert result.via_daemon is True

        mock_daemon_client.submit_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_job_with_config_content(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
        sample_yaml_config: str,
    ) -> None:
        mock_daemon_client.submit_job.return_value = JobResponse(
            job_id="job-xyz",
            status="accepted",
        )

        result = await job_control_service.start_job(
            config_content=sample_yaml_config,
            workspace=Path("./custom-workspace"),
        )

        assert isinstance(result, JobStartResult)
        assert result.job_id == "job-xyz"
        assert result.job_name == "test-job"

    @pytest.mark.asyncio
    async def test_start_job_no_config_raises_error(
        self,
        job_control_service: JobControlService,
    ) -> None:
        with pytest.raises(ValueError, match="Must provide either"):
            await job_control_service.start_job()

    @pytest.mark.asyncio
    async def test_start_job_nonexistent_file_raises_error(
        self,
        job_control_service: JobControlService,
    ) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            await job_control_service.start_job(config_path=Path("/nonexistent/config.yaml"))

    @pytest.mark.asyncio
    async def test_start_job_traversal_raises_error(
        self,
        job_control_service: JobControlService,
    ) -> None:
        with pytest.raises(ValueError, match="traversal"):
            await job_control_service.start_job(config_path=Path("../../../etc/config.yaml"))

    @pytest.mark.asyncio
    async def test_start_job_conductor_not_running(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
        sample_config_file: Path,
    ) -> None:
        mock_daemon_client.submit_job.side_effect = DaemonNotRunningError("not running")

        with pytest.raises(RuntimeError, match="Conductor not running"):
            await job_control_service.start_job(config_path=sample_config_file)


class TestPauseJob:
    @pytest.mark.asyncio
    async def test_pause_job_success(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
    ) -> None:
        result = await job_control_service.pause_job("job-123")

        assert isinstance(result, JobActionResult)
        assert result.success is True
        assert result.job_id == "job-123"
        assert result.status == "paused"
        assert result.via_daemon is True
        mock_daemon_client.pause_job.assert_called_once_with("job-123", "")

    @pytest.mark.asyncio
    async def test_pause_job_conductor_not_running(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
    ) -> None:
        mock_daemon_client.pause_job.side_effect = DaemonNotRunningError("down")

        with pytest.raises(RuntimeError, match="Conductor not running"):
            await job_control_service.pause_job("job-123")


class TestResumeJob:
    @pytest.mark.asyncio
    async def test_resume_job_success(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
    ) -> None:
        result = await job_control_service.resume_job("job-123")

        assert result.success is True
        assert result.job_id == "job-123"
        assert result.status == "running"
        assert result.via_daemon is True
        mock_daemon_client.resume_job.assert_called_once_with("job-123", "")

    @pytest.mark.asyncio
    async def test_resume_job_conductor_not_running(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
    ) -> None:
        mock_daemon_client.resume_job.side_effect = DaemonNotRunningError("down")

        with pytest.raises(RuntimeError, match="Conductor not running"):
            await job_control_service.resume_job("job-123")


class TestCancelJob:
    @pytest.mark.asyncio
    async def test_cancel_job_success(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
    ) -> None:
        result = await job_control_service.cancel_job("job-123")

        assert result.success is True
        assert result.job_id == "job-123"
        assert result.status == "cancelled"
        assert result.via_daemon is True
        mock_daemon_client.cancel_job.assert_called_once_with("job-123", "")

    @pytest.mark.asyncio
    async def test_cancel_job_conductor_not_running(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
    ) -> None:
        mock_daemon_client.cancel_job.side_effect = DaemonNotRunningError("down")

        with pytest.raises(RuntimeError, match="Conductor not running"):
            await job_control_service.cancel_job("job-123")


class TestDeleteJob:
    @pytest.mark.asyncio
    async def test_delete_job_success(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
    ) -> None:
        mock_daemon_client.clear_jobs.return_value = {"deleted": 1}

        result = await job_control_service.delete_job("job-123")

        assert result is True
        mock_daemon_client.clear_jobs.assert_called_once_with(job_ids=["job-123"])

    @pytest.mark.asyncio
    async def test_delete_job_not_found(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
    ) -> None:
        mock_daemon_client.clear_jobs.return_value = {"deleted": 0}

        result = await job_control_service.delete_job("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_job_conductor_not_running(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
    ) -> None:
        mock_daemon_client.clear_jobs.side_effect = DaemonNotRunningError("down")

        with pytest.raises(RuntimeError, match="Conductor not running"):
            await job_control_service.delete_job("job-123")


class TestVerifyProcessHealth:
    @pytest.mark.asyncio
    async def test_running_job_health(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
    ) -> None:
        from datetime import UTC, datetime

        mock_daemon_client.get_job_status.return_value = {
            "job_id": "job-123",
            "job_name": "test",
            "total_sheets": 2,
            "status": "running",
            "pid": 12345,
            "started_at": datetime.now(UTC).isoformat(),
            "sheets": {},
        }

        health = await job_control_service.verify_process_health("job-123")

        assert isinstance(health, ProcessHealth)
        assert health.pid == 12345
        assert health.is_alive is True
        assert health.process_exists is True

    @pytest.mark.asyncio
    async def test_completed_job_health(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
    ) -> None:
        mock_daemon_client.get_job_status.return_value = {
            "job_id": "job-123",
            "job_name": "test",
            "total_sheets": 2,
            "status": "completed",
            "sheets": {},
        }

        health = await job_control_service.verify_process_health("job-123")

        assert health.is_alive is False
        assert health.process_exists is False

    @pytest.mark.asyncio
    async def test_conductor_not_running(
        self,
        job_control_service: JobControlService,
        mock_daemon_client: MagicMock,
    ) -> None:
        mock_daemon_client.get_job_status.side_effect = DaemonNotRunningError("down")

        health = await job_control_service.verify_process_health("job-123")

        assert health.pid is None
        assert health.is_alive is False
        assert health.process_exists is False


if __name__ == "__main__":
    pytest.main([__file__])
