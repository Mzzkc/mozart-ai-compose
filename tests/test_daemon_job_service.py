"""Tests for mozart.daemon.job_service module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mozart.core.checkpoint import CheckpointState, JobStatus
from mozart.daemon.exceptions import JobSubmissionError
from mozart.daemon.job_service import JobService
from mozart.daemon.output import NullOutput


# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def null_output() -> NullOutput:
    """Create a NullOutput for test isolation."""
    return NullOutput()


@pytest.fixture
def job_service(null_output: NullOutput) -> JobService:
    """Create a JobService with NullOutput."""
    return JobService(output=null_output)


@pytest.fixture
def sample_job_config() -> MagicMock:
    """Create a mock JobConfig for testing."""
    config = MagicMock()
    config.name = "test-job"
    config.workspace = Path("/tmp/test-workspace")
    config.state_backend = "json"
    config.sheet.total_sheets = 5
    config.backend.type = "claude_cli"
    config.learning.enabled = False
    config.notifications = None
    config.grounding.enabled = False
    config.workspace_lifecycle.archive_on_fresh = False
    return config


# ─── Instantiation ────────────────────────────────────────────────────────


class TestJobServiceInstantiation:
    """Tests for JobService construction."""

    def test_default_output_is_null(self):
        """Test JobService uses NullOutput when none provided."""
        service = JobService()
        assert isinstance(service._output, NullOutput)

    def test_custom_output(self, null_output: NullOutput):
        """Test JobService accepts custom output."""
        service = JobService(output=null_output)
        assert service._output is null_output

    def test_learning_store_none_by_default(self):
        """Test global_learning_store defaults to None."""
        service = JobService()
        assert service._learning_store is None

    def test_custom_learning_store(self):
        """Test JobService accepts custom learning store."""
        mock_store = MagicMock()
        service = JobService(global_learning_store=mock_store)
        assert service._learning_store is mock_store


# ─── start_job ────────────────────────────────────────────────────────────


class TestStartJob:
    """Tests for JobService.start_job()."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_pending_summary(
        self, job_service: JobService, sample_job_config: MagicMock
    ):
        """Test dry_run=True returns summary without executing."""
        sample_job_config.workspace = Path("/tmp/test-dry-run")

        with patch.object(Path, "mkdir"):
            summary = await job_service.start_job(
                sample_job_config,
                dry_run=True,
            )

        assert summary.job_id == "test-job"
        assert summary.final_status == JobStatus.PENDING
        assert summary.total_sheets == 5

    @pytest.mark.asyncio
    async def test_start_job_creates_workspace(
        self, job_service: JobService, sample_job_config: MagicMock
    ):
        """Test start_job creates workspace directory."""
        mock_mkdir = MagicMock()
        sample_job_config.workspace = MagicMock(spec=Path)
        sample_job_config.workspace.mkdir = mock_mkdir

        with patch.object(Path, "mkdir"):
            await job_service.start_job(sample_job_config, dry_run=True)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


# ─── pause_job ────────────────────────────────────────────────────────────


class TestPauseJob:
    """Tests for JobService.pause_job()."""

    @pytest.mark.asyncio
    async def test_pause_creates_signal_file(
        self, job_service: JobService, tmp_path: Path
    ):
        """Test pause_job creates a signal file in the workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create a mock state that's RUNNING
        state = CheckpointState(
            job_id="test-job",
            job_name="test-job",
            total_sheets=5,
            status=JobStatus.RUNNING,
        )

        with patch.object(
            JobService, "_find_job_state", new_callable=AsyncMock
        ) as mock_find:
            mock_find.return_value = (state, MagicMock())
            result = await job_service.pause_job("test-job", workspace)

        assert result is True
        signal_file = workspace / ".mozart-pause-test-job"
        assert signal_file.exists()

    @pytest.mark.asyncio
    async def test_pause_non_running_job_raises(
        self, job_service: JobService, tmp_path: Path
    ):
        """Test pause_job raises when job is not RUNNING."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        state = CheckpointState(
            job_id="test-job",
            job_name="test-job",
            total_sheets=5,
            status=JobStatus.COMPLETED,
        )

        with patch.object(
            JobService, "_find_job_state", new_callable=AsyncMock
        ) as mock_find:
            mock_find.return_value = (state, MagicMock())

            with pytest.raises(JobSubmissionError, match="not running"):
                await job_service.pause_job("test-job", workspace)

    @pytest.mark.asyncio
    async def test_pause_paused_job_raises(
        self, job_service: JobService, tmp_path: Path
    ):
        """Test pause_job raises when job is already PAUSED."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        state = CheckpointState(
            job_id="test-job",
            job_name="test-job",
            total_sheets=5,
            status=JobStatus.PAUSED,
        )

        with patch.object(
            JobService, "_find_job_state", new_callable=AsyncMock
        ) as mock_find:
            mock_find.return_value = (state, MagicMock())

            with pytest.raises(JobSubmissionError, match="not running"):
                await job_service.pause_job("test-job", workspace)


# ─── get_status ───────────────────────────────────────────────────────────


class TestGetStatus:
    """Tests for JobService.get_status()."""

    @pytest.mark.asyncio
    async def test_returns_state_when_found(
        self, job_service: JobService, tmp_path: Path
    ):
        """Test get_status returns CheckpointState when job exists."""
        state = CheckpointState(
            job_id="test-job",
            job_name="test-job",
            total_sheets=5,
            status=JobStatus.RUNNING,
        )

        mock_backend = AsyncMock()
        mock_backend.load = AsyncMock(return_value=state)

        with patch.object(
            JobService, "_create_state_backend", return_value=mock_backend
        ):
            result = await job_service.get_status("test-job", tmp_path)

        assert result is not None
        assert result.job_id == "test-job"
        assert result.status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(
        self, job_service: JobService, tmp_path: Path
    ):
        """Test get_status returns None when job doesn't exist."""
        mock_backend = AsyncMock()
        mock_backend.load = AsyncMock(return_value=None)

        with patch.object(
            JobService, "_create_state_backend", return_value=mock_backend
        ):
            result = await job_service.get_status("nonexistent", tmp_path)

        assert result is None


# ─── _create_backend ──────────────────────────────────────────────────────


class TestCreateBackend:
    """Tests for JobService._create_backend()."""

    def test_claude_cli_backend(self, job_service: JobService):
        """Test _create_backend creates ClaudeCliBackend for claude_cli."""
        mock_config = MagicMock()
        mock_config.backend.type = "claude_cli"

        with patch(
            "mozart.backends.claude_cli.ClaudeCliBackend.from_config"
        ) as mock_from_config:
            mock_from_config.return_value = MagicMock()
            job_service._create_backend(mock_config)
            mock_from_config.assert_called_once_with(mock_config.backend)

    def test_anthropic_api_backend(self, job_service: JobService):
        """Test _create_backend creates AnthropicApiBackend for anthropic_api."""
        mock_config = MagicMock()
        mock_config.backend.type = "anthropic_api"

        with patch(
            "mozart.backends.anthropic_api.AnthropicApiBackend.from_config"
        ) as mock_from_config:
            mock_from_config.return_value = MagicMock()
            job_service._create_backend(mock_config)
            mock_from_config.assert_called_once_with(mock_config.backend)

    def test_recursive_light_backend(self, job_service: JobService):
        """Test _create_backend creates RecursiveLightBackend for recursive_light."""
        mock_config = MagicMock()
        mock_config.backend.type = "recursive_light"

        with patch(
            "mozart.backends.recursive_light.RecursiveLightBackend.from_config"
        ) as mock_from_config:
            mock_from_config.return_value = MagicMock()
            job_service._create_backend(mock_config)
            mock_from_config.assert_called_once_with(mock_config.backend)

    def test_unknown_backend_falls_back_to_claude_cli(
        self, job_service: JobService
    ):
        """Test _create_backend defaults to ClaudeCliBackend for unknown types."""
        mock_config = MagicMock()
        mock_config.backend.type = "unknown_type"

        with patch(
            "mozart.backends.claude_cli.ClaudeCliBackend.from_config"
        ) as mock_from_config:
            mock_from_config.return_value = MagicMock()
            job_service._create_backend(mock_config)
            mock_from_config.assert_called_once_with(mock_config.backend)


# ─── _create_state_backend ────────────────────────────────────────────────


class TestCreateStateBackend:
    """Tests for JobService._create_state_backend()."""

    def test_json_backend_default(self, job_service: JobService, tmp_path: Path):
        """Test _create_state_backend creates JsonStateBackend by default."""
        backend = job_service._create_state_backend(tmp_path)
        from mozart.state import JsonStateBackend

        assert isinstance(backend, JsonStateBackend)

    def test_json_backend_explicit(self, job_service: JobService, tmp_path: Path):
        """Test _create_state_backend creates JsonStateBackend for 'json'."""
        backend = job_service._create_state_backend(tmp_path, "json")
        from mozart.state import JsonStateBackend

        assert isinstance(backend, JsonStateBackend)

    def test_sqlite_backend(self, job_service: JobService, tmp_path: Path):
        """Test _create_state_backend creates SQLiteStateBackend for 'sqlite'."""
        backend = job_service._create_state_backend(tmp_path, "sqlite")
        from mozart.state import SQLiteStateBackend

        assert isinstance(backend, SQLiteStateBackend)


# ─── _find_job_state ──────────────────────────────────────────────────────


class TestFindJobState:
    """Tests for JobService._find_job_state()."""

    @pytest.mark.asyncio
    async def test_workspace_not_found_raises(
        self, job_service: JobService, tmp_path: Path
    ):
        """Test _find_job_state raises when workspace doesn't exist."""
        nonexistent = tmp_path / "does-not-exist"

        with pytest.raises(JobSubmissionError, match="Workspace not found"):
            await job_service._find_job_state("test-job", nonexistent)

    @pytest.mark.asyncio
    async def test_job_not_found_raises(
        self, job_service: JobService, tmp_path: Path
    ):
        """Test _find_job_state raises when job doesn't exist in any backend."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with pytest.raises(JobSubmissionError, match="not found in workspace"):
            await job_service._find_job_state("nonexistent-job", workspace)

    @pytest.mark.asyncio
    async def test_finds_job_in_json_backend(
        self, job_service: JobService, tmp_path: Path
    ):
        """Test _find_job_state finds job via JsonStateBackend."""
        from mozart.state import JsonStateBackend

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Save a state via the JSON backend directly
        backend = JsonStateBackend(workspace)
        state = CheckpointState(
            job_id="test-job",
            job_name="test-job",
            total_sheets=5,
            status=JobStatus.RUNNING,
        )
        await backend.save(state)

        found_state, found_backend = await job_service._find_job_state(
            "test-job", workspace
        )
        assert found_state.job_id == "test-job"
        assert found_state.status == JobStatus.RUNNING


# ─── resume_job validation ────────────────────────────────────────────────


class TestResumeJobValidation:
    """Tests for resume_job edge cases and validation."""

    @pytest.mark.asyncio
    async def test_resume_completed_job_raises(
        self, job_service: JobService, tmp_path: Path
    ):
        """Test resume_job raises for completed jobs."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        state = CheckpointState(
            job_id="test-job",
            job_name="test-job",
            total_sheets=5,
            status=JobStatus.COMPLETED,
        )

        with patch.object(
            JobService, "_find_job_state", new_callable=AsyncMock
        ) as mock_find:
            mock_find.return_value = (state, MagicMock())

            with pytest.raises(JobSubmissionError, match="already completed"):
                await job_service.resume_job("test-job", workspace)

    @pytest.mark.asyncio
    async def test_resume_pending_job_raises(
        self, job_service: JobService, tmp_path: Path
    ):
        """Test resume_job raises for pending jobs."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        state = CheckpointState(
            job_id="test-job",
            job_name="test-job",
            total_sheets=5,
            status=JobStatus.PENDING,
        )

        with patch.object(
            JobService, "_find_job_state", new_callable=AsyncMock
        ) as mock_find:
            mock_find.return_value = (state, MagicMock())

            with pytest.raises(JobSubmissionError, match="not been started"):
                await job_service.resume_job("test-job", workspace)


# ─── _reconstruct_config ─────────────────────────────────────────────────


class TestReconstructConfig:
    """Tests for JobService._reconstruct_config()."""

    def test_explicit_config_takes_priority(self, job_service: JobService):
        """Test explicit config is returned directly (priority 1)."""
        mock_config = MagicMock()
        mock_state = MagicMock()

        result = job_service._reconstruct_config(mock_state, config=mock_config)
        assert result is mock_config

    def test_no_config_source_raises(self, job_service: JobService):
        """Test raises when no config source available."""
        mock_state = MagicMock()
        mock_state.config_snapshot = None
        mock_state.config_path = None

        with pytest.raises(JobSubmissionError, match="no config available"):
            job_service._reconstruct_config(mock_state)

    def test_reload_config_with_no_path_raises(self, job_service: JobService):
        """Test reload_config raises when no config path available."""
        mock_state = MagicMock()
        mock_state.config_path = None

        with pytest.raises(JobSubmissionError, match="no valid config path"):
            job_service._reconstruct_config(
                mock_state, reload_config=True
            )

    def test_config_snapshot_used_when_available(
        self, job_service: JobService, sample_job_config: MagicMock
    ):
        """Test config_snapshot is used as priority 3."""
        from mozart.core.config import JobConfig

        # Create a real minimal config dict
        snapshot = {
            "name": "test-job",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 5, "total_items": 10},
            "prompt": {"template": "{{ sheet_num }}"},
        }

        mock_state = MagicMock()
        mock_state.config_snapshot = snapshot
        mock_state.config_path = None

        result = job_service._reconstruct_config(mock_state)
        assert isinstance(result, JobConfig)
        assert result.name == "test-job"
