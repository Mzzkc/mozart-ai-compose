"""Tests for #98/#131: no_reload flag threading through IPC resume pipeline.

The --no-reload CLI flag was silently ignored when routing through the
conductor because it was never transmitted through the IPC pipeline.
These tests verify the flag is threaded end-to-end.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mozart.core.checkpoint import CheckpointState, JobStatus
from mozart.daemon.manager import DaemonJobStatus, JobManager, JobMeta


# ---------------------------------------------------------------------------
# 1. CLI: params dict includes no_reload when flag is set
# ---------------------------------------------------------------------------


class TestCliResumeNoReloadParam:
    """CLI must include no_reload in params sent to try_daemon_route."""

    @pytest.mark.asyncio
    async def test_no_reload_true_included_in_params(self) -> None:
        """When --no-reload is set, params dict contains no_reload=True."""
        captured_params: list[dict[str, Any]] = []

        async def fake_daemon_route(
            method: str, params: dict[str, Any], **kwargs: Any,
        ) -> tuple[bool, Any]:
            captured_params.append(params)
            return True, {
                "job_id": "test-job",
                "status": "accepted",
                "message": "ok",
            }

        with (
            patch(
                "mozart.daemon.detect.try_daemon_route",
                side_effect=fake_daemon_route,
            ),
            patch("mozart.cli.commands.resume.configure_global_logging"),
        ):
            from mozart.cli.commands.resume import _resume_job

            await _resume_job(
                job_id="test-job",
                config_file=None,
                workspace=None,
                force=False,
                no_reload=True,
            )

        assert len(captured_params) == 1
        assert captured_params[0].get("no_reload") is True

    @pytest.mark.asyncio
    async def test_no_reload_false_by_default(self) -> None:
        """When --no-reload is not set, params dict has no_reload=False."""
        captured_params: list[dict[str, Any]] = []

        async def fake_daemon_route(
            method: str, params: dict[str, Any], **kwargs: Any,
        ) -> tuple[bool, Any]:
            captured_params.append(params)
            return True, {
                "job_id": "test-job",
                "status": "accepted",
                "message": "ok",
            }

        with (
            patch(
                "mozart.daemon.detect.try_daemon_route",
                side_effect=fake_daemon_route,
            ),
            patch("mozart.cli.commands.resume.configure_global_logging"),
        ):
            from mozart.cli.commands.resume import _resume_job

            await _resume_job(
                job_id="test-job",
                config_file=None,
                workspace=None,
                force=False,
                no_reload=False,
            )

        assert len(captured_params) == 1
        assert captured_params[0].get("no_reload", False) is False


# ---------------------------------------------------------------------------
# 2. Manager: resume_job accepts no_reload and stores it for the task
# ---------------------------------------------------------------------------


class TestManagerResumeNoReload:
    """manager.resume_job must accept no_reload and forward it."""

    @pytest.fixture
    def manager(self) -> JobManager:
        """Minimal manager with FAILED job in meta."""
        mgr = MagicMock(spec=JobManager)
        mgr._job_meta = {
            "test-job": JobMeta(
                job_id="test-job",
                config_path=Path("/old.yaml"),
                workspace=Path("/tmp/ws"),
                status=DaemonJobStatus.FAILED,
            ),
        }
        mgr._jobs = {}
        mgr._pause_events = {}
        mgr._baton_adapter = None
        mgr._live_states = {}
        # Use real resume_job method
        mgr.resume_job = JobManager.resume_job.__get__(mgr, JobManager)
        mgr._on_task_done = MagicMock(spec=lambda job_id, task: None)
        return mgr

    @pytest.mark.asyncio
    async def test_no_reload_true_forwarded_to_task(self, manager: JobManager) -> None:
        """no_reload=True must reach _resume_job_task."""
        captured_args: list[dict[str, Any]] = []

        async def fake_task(
            job_id: str, workspace: Path, no_reload: bool = False,
        ) -> None:
            captured_args.append({"no_reload": no_reload})

        manager._resume_job_task = fake_task  # type: ignore[attr-defined]

        response = await manager.resume_job("test-job", no_reload=True)
        assert response.status == "accepted"

        # Let the task start
        await asyncio.sleep(0.05)

        # Cleanup
        task = manager._jobs.get("test-job")
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        assert captured_args == [{"no_reload": True}]

    @pytest.mark.asyncio
    async def test_no_reload_default_is_false(self, manager: JobManager) -> None:
        """When no_reload is omitted, _resume_job_task gets False."""
        captured_args: list[dict[str, Any]] = []

        async def fake_task(
            job_id: str, workspace: Path, no_reload: bool = False,
        ) -> None:
            captured_args.append({"no_reload": no_reload})

        manager._resume_job_task = fake_task  # type: ignore[attr-defined]

        response = await manager.resume_job("test-job")
        assert response.status == "accepted"

        await asyncio.sleep(0.05)

        task = manager._jobs.get("test-job")
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        assert captured_args == [{"no_reload": False}]


# ---------------------------------------------------------------------------
# 3. Service: _reconstruct_config respects no_reload (already has test,
#    but this verifies resume_job passes it through)
# ---------------------------------------------------------------------------


class TestServiceResumeNoReload:
    """service.resume_job must pass no_reload to _reconstruct_config."""

    def test_no_reload_true_uses_snapshot(self, tmp_path: Path) -> None:
        """When no_reload=True, snapshot is used even if config file exists."""
        from mozart.daemon.job_service import JobService

        config_file = tmp_path / "score.yaml"
        config_file.write_text(
            "name: disk-version\n"
            "backend:\n  type: claude_cli\n"
            "sheet:\n  size: 3\n  total_items: 9\n"
            "prompt:\n  template: 'Test {{ sheet_num }}'\n"
        )

        state = CheckpointState(
            job_id="test-job",
            job_name="test-job",
            total_sheets=3,
            status=JobStatus.PAUSED,
            config_path=str(config_file),
            config_snapshot={
                "name": "snapshot-version",
                "backend": {"type": "claude_cli"},
                "sheet": {"size": 3, "total_items": 9},
                "prompt": {"template": "Test {{ sheet_num }}"},
            },
        )

        service = JobService()

        result, was_reloaded = service._reconstruct_config(state, no_reload=True)
        assert result.name == "snapshot-version"
        assert was_reloaded is False

    def test_no_reload_false_reloads_from_disk(self, tmp_path: Path) -> None:
        """When no_reload=False (default), config is reloaded from disk."""
        from mozart.daemon.job_service import JobService

        config_file = tmp_path / "score.yaml"
        config_file.write_text(
            "name: disk-version\n"
            "backend:\n  type: claude_cli\n"
            "sheet:\n  size: 3\n  total_items: 9\n"
            "prompt:\n  template: 'Test {{ sheet_num }}'\n"
        )

        state = CheckpointState(
            job_id="test-job",
            job_name="test-job",
            total_sheets=3,
            status=JobStatus.PAUSED,
            config_path=str(config_file),
            config_snapshot={
                "name": "snapshot-version",
                "backend": {"type": "claude_cli"},
                "sheet": {"size": 3, "total_items": 9},
                "prompt": {"template": "Test {{ sheet_num }}"},
            },
        )

        service = JobService()

        result, was_reloaded = service._reconstruct_config(state)
        assert result.name == "disk-version"
        assert was_reloaded is True


# ---------------------------------------------------------------------------
# 4. Regression: #96 cost_limit_reached reset on config reload
# ---------------------------------------------------------------------------


class TestCostLimitResetOnReload:
    """Regression test for #96: cost_limit_reached must be reset when
    cost_limits config changes during resume with config reload."""

    def test_cost_limit_reached_reset_when_cost_limits_change(self) -> None:
        """cost_limit_reached must be reset when cost_limits section changes."""
        from mozart.core.config import JobConfig
        from mozart.execution.reconciliation import reconcile_config

        old_snapshot = {
            "name": "test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 3, "total_items": 9},
            "prompt": {"template": "Test {{ sheet_num }}"},
            "cost_limits": {
                "enabled": True,
                "max_cost_per_sheet": 5.0,
            },
        }

        state = CheckpointState(
            job_id="test-job",
            job_name="test-job",
            total_sheets=3,
            config_snapshot=old_snapshot,
            cost_limit_reached=True,
            total_estimated_cost=15.0,
            total_input_tokens=100000,
            total_output_tokens=50000,
        )

        new_config = JobConfig.model_validate({
            "name": "test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 3, "total_items": 9},
            "prompt": {"template": "Test {{ sheet_num }}"},
            "cost_limits": {
                "enabled": False,
            },
        })

        report = reconcile_config(state, new_config)

        assert report.has_changes, "cost_limits change should be detected"
        assert "cost_limits" in report.sections_changed
        assert state.cost_limit_reached is False, \
            "cost_limit_reached must be reset when cost_limits change"
        assert state.total_estimated_cost == 0.0, \
            "total_estimated_cost must be reset when cost_limits change"

    def test_cost_limit_not_reset_when_cost_limits_unchanged(self) -> None:
        """cost_limit_reached should NOT be reset when cost_limits don't change."""
        from mozart.core.config import JobConfig
        from mozart.execution.reconciliation import reconcile_config

        # Use a full model dump as snapshot so all defaults match
        base_config = JobConfig.model_validate({
            "name": "test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 3, "total_items": 9},
            "prompt": {"template": "Test {{ sheet_num }}"},
            "cost_limits": {
                "enabled": True,
                "max_cost_per_sheet": 5.0,
            },
        })
        full_snapshot = base_config.model_dump(mode="json")

        state = CheckpointState(
            job_id="test-job",
            job_name="test-job",
            total_sheets=3,
            config_snapshot=full_snapshot,
            cost_limit_reached=True,
            total_estimated_cost=15.0,
        )

        # Same config — no change
        new_config = JobConfig.model_validate(full_snapshot)

        report = reconcile_config(state, new_config)

        assert "cost_limits" not in report.sections_changed, \
            "cost_limits should not be in changed sections when unchanged"
        assert state.cost_limit_reached is True, \
            "cost_limit_reached should remain True when cost_limits unchanged"


# ---------------------------------------------------------------------------
# 5. Baton resume path: no_reload forwarded (#98 baton fix)
# ---------------------------------------------------------------------------


class TestBatonResumeNoReload:
    """_resume_via_baton must accept and respect no_reload parameter."""

    def test_resume_via_baton_accepts_no_reload(self) -> None:
        """_resume_via_baton must have no_reload parameter."""
        import inspect

        from mozart.daemon.manager import JobManager

        sig = inspect.signature(JobManager._resume_via_baton)
        assert "no_reload" in sig.parameters, (
            "_resume_via_baton must accept no_reload (fix for #98)"
        )

    def test_resume_job_task_forwards_no_reload_to_baton(self) -> None:
        """_resume_job_task routes no_reload to _resume_via_baton."""
        import ast

        source_file = Path(__file__).parent.parent / "src" / "mozart" / "daemon" / "manager.py"
        source = source_file.read_text()

        # The call to _resume_via_baton inside _resume_job_task must include
        # no_reload as a keyword argument
        assert "no_reload=no_reload" in source, (
            "_resume_job_task must forward no_reload to _resume_via_baton"
        )
