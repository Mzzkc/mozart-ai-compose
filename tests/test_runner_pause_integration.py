"""Tests for runner pause/resume integration with conductor-only JobControlService.

Tests cover:
- JobControlService forwards pause/resume/cancel to DaemonClient
- DaemonNotRunningError is translated to RuntimeError
- Edge cases: multiple operations, unexpected errors from conductor
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from marianne.daemon.exceptions import DaemonNotRunningError
from marianne.dashboard.services.job_control import (
    JobActionResult,
    JobControlService,
)


def _make_daemon_client() -> MagicMock:
    mock = MagicMock()
    mock.pause_job = AsyncMock(return_value={"status": "paused"})
    mock.resume_job = AsyncMock(return_value={"status": "running"})
    mock.cancel_job = AsyncMock(return_value={"status": "cancelled"})
    mock.clear_jobs = AsyncMock(return_value={"deleted": 1})
    mock.get_job_status = AsyncMock(
        return_value={
            "job_id": "job-1",
            "job_name": "test",
            "status": "running",
            "pid": None,
        }
    )
    return mock


class TestJobControlServicePauseResume:
    """Test pause and resume through conductor proxy."""

    async def test_pause_forwards_to_daemon(self):
        mock = _make_daemon_client()
        service = JobControlService(mock)

        result = await service.pause_job("job-1")

        mock.pause_job.assert_awaited_once_with("job-1", "")
        assert isinstance(result, JobActionResult)
        assert result.success
        assert result.job_id == "job-1"

    async def test_resume_forwards_to_daemon(self):
        mock = _make_daemon_client()
        service = JobControlService(mock)

        result = await service.resume_job("job-1")

        mock.resume_job.assert_awaited_once_with("job-1", "")
        assert result.success
        assert result.status == "running"

    async def test_pause_conductor_not_running(self):
        mock = _make_daemon_client()
        mock.pause_job.side_effect = DaemonNotRunningError()
        service = JobControlService(mock)

        with pytest.raises(RuntimeError, match="Conductor not running"):
            await service.pause_job("job-1")

    async def test_resume_conductor_not_running(self):
        mock = _make_daemon_client()
        mock.resume_job.side_effect = DaemonNotRunningError()
        service = JobControlService(mock)

        with pytest.raises(RuntimeError, match="Conductor not running"):
            await service.resume_job("job-1")


class TestJobControlServiceCancel:
    """Test cancel through conductor proxy."""

    async def test_cancel_forwards_to_daemon(self):
        mock = _make_daemon_client()
        service = JobControlService(mock)

        result = await service.cancel_job("job-1")

        mock.cancel_job.assert_awaited_once_with("job-1", "")
        assert result.success
        assert result.status == "cancelled"

    async def test_cancel_conductor_not_running(self):
        mock = _make_daemon_client()
        mock.cancel_job.side_effect = DaemonNotRunningError()
        service = JobControlService(mock)

        with pytest.raises(RuntimeError, match="Conductor not running"):
            await service.cancel_job("job-1")


class TestJobControlServiceDelete:
    """Test delete through conductor proxy."""

    async def test_delete_forwards_to_daemon(self):
        mock = _make_daemon_client()
        service = JobControlService(mock)

        deleted = await service.delete_job("job-1")

        mock.clear_jobs.assert_awaited_once_with(job_ids=["job-1"])
        assert deleted is True

    async def test_delete_conductor_not_running(self):
        mock = _make_daemon_client()
        mock.clear_jobs.side_effect = DaemonNotRunningError()
        service = JobControlService(mock)

        with pytest.raises(RuntimeError, match="Conductor not running"):
            await service.delete_job("job-1")

    async def test_delete_not_found(self):
        mock = _make_daemon_client()
        mock.clear_jobs.return_value = {"deleted": 0}
        service = JobControlService(mock)

        deleted = await service.delete_job("ghost")
        assert deleted is False


class TestJobControlServicePauseResumeIntegration:
    """Integration-style multi-step flows."""

    async def test_pause_resume_cycle(self):
        mock = _make_daemon_client()
        service = JobControlService(mock)

        r1 = await service.pause_job("job-flow")
        assert r1.success

        r2 = await service.resume_job("job-flow")
        assert r2.success

        assert mock.pause_job.await_count == 1
        assert mock.resume_job.await_count == 1

    async def test_pause_cancel_cycle(self):
        mock = _make_daemon_client()
        service = JobControlService(mock)

        r1 = await service.pause_job("job-pc")
        assert r1.success

        r2 = await service.cancel_job("job-pc")
        assert r2.success

    async def test_unexpected_exception_propagates(self):
        mock = _make_daemon_client()
        mock.pause_job.side_effect = OSError("socket broken")
        service = JobControlService(mock)

        with pytest.raises(OSError, match="socket broken"):
            await service.pause_job("job-err")

    async def test_different_job_ids_are_independent(self):
        mock = _make_daemon_client()
        service = JobControlService(mock)

        r1 = await service.pause_job("job-a")
        r2 = await service.pause_job("job-b")

        assert r1.success
        assert r2.success
        assert r1.job_id == "job-a"
        assert r2.job_id == "job-b"

        mock.pause_job.assert_any_await("job-a", "")
        mock.pause_job.assert_any_await("job-b", "")
