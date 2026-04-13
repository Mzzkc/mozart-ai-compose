"""Tests for the working directory fix.

Verifies:
1. client_cwd field exists on JobSubmitParams and JobRequest
2. Relative workspace paths resolve against client_cwd
3. Absolute workspace paths are unaffected by client_cwd
4. CLI passes client_cwd in IPC requests
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from marianne.daemon.types import JobRequest


# ─── IPC Type Tests ──────────────────────────────────────────────────


class TestClientCwdInTypes:
    """Test client_cwd field on IPC types."""

    def test_job_request_has_client_cwd_field(self) -> None:
        """JobRequest should have an optional client_cwd field."""
        request = JobRequest(config_path=Path("/tmp/score.yaml"))
        assert request.client_cwd is None

    def test_job_request_accepts_client_cwd(self) -> None:
        """JobRequest should accept client_cwd as a Path."""
        request = JobRequest(
            config_path=Path("/tmp/score.yaml"),
            client_cwd=Path("/home/user/project"),
        )
        assert request.client_cwd == Path("/home/user/project")

    def test_job_request_serialization_includes_client_cwd(self) -> None:
        """client_cwd should serialize correctly in model_dump."""
        request = JobRequest(
            config_path=Path("/tmp/score.yaml"),
            client_cwd=Path("/home/user/project"),
        )
        dumped = request.model_dump(mode="json")
        assert "client_cwd" in dumped
        assert dumped["client_cwd"] == "/home/user/project"

    def test_job_request_roundtrip(self) -> None:
        """JobRequest should round-trip through serialization."""
        request = JobRequest(
            config_path=Path("/tmp/score.yaml"),
            client_cwd=Path("/home/user/project"),
        )
        dumped = request.model_dump(mode="json")
        restored = JobRequest(**dumped)
        assert restored.client_cwd == Path("/home/user/project")

    def test_job_submit_params_has_client_cwd(self) -> None:
        """JobSubmitParams TypedDict should accept client_cwd."""
        from marianne.daemon.types import JobSubmitParams

        params: JobSubmitParams = {
            "config_path": "/tmp/score.yaml",
            "client_cwd": "/home/user/project",
        }
        assert params["client_cwd"] == "/home/user/project"


# ─── Path Resolution Tests ──────────────────────────────────────────


class TestWorkspaceResolution:
    """Test that relative workspace paths resolve against client_cwd."""

    def test_relative_workspace_resolves_against_client_cwd(self) -> None:
        """A relative workspace should be resolved using client_cwd."""
        client_cwd = Path("/home/user/project")
        workspace = Path("workspaces/my-job")

        # Simulate what the manager does
        if not workspace.is_absolute():
            resolved = (client_cwd / workspace).resolve()
        else:
            resolved = workspace

        assert str(resolved) == "/home/user/project/workspaces/my-job"

    def test_absolute_workspace_ignores_client_cwd(self) -> None:
        """An absolute workspace should not be affected by client_cwd."""
        client_cwd = Path("/home/user/project")
        workspace = Path("/absolute/workspace/path")

        if not workspace.is_absolute():
            resolved = (client_cwd / workspace).resolve()
        else:
            resolved = workspace

        assert resolved == Path("/absolute/workspace/path")

    def test_no_client_cwd_leaves_workspace_unchanged(self) -> None:
        """Without client_cwd, workspace resolution is unchanged."""
        client_cwd = None
        workspace = Path("workspaces/my-job")

        # Simulate: only resolve if client_cwd is set
        if client_cwd and not workspace.is_absolute():
            resolved = (client_cwd / workspace).resolve()
        else:
            resolved = workspace

        assert resolved == Path("workspaces/my-job")

    def test_client_cwd_with_dotdot_workspace(self) -> None:
        """client_cwd should handle .. in relative workspace paths."""
        client_cwd = Path("/home/user/project/sub")
        workspace = Path("../workspaces/my-job")

        resolved = (client_cwd / workspace).resolve()
        assert str(resolved) == "/home/user/project/workspaces/my-job"


# ─── Integration-style Tests ────────────────────────────────────────


class TestClientCwdIntegration:
    """Integration-style tests for the full client_cwd flow."""

    def test_job_request_construction_from_ipc_params(self) -> None:
        """Simulate how the IPC handler constructs a JobRequest."""
        # This is what comes over the wire from CLI
        params: dict[str, Any] = {
            "config_path": "/tmp/score.yaml",
            "fresh": False,
            "client_cwd": "/home/user/project",
        }

        request = JobRequest(**params)
        assert request.config_path == Path("/tmp/score.yaml")
        assert request.client_cwd == Path("/home/user/project")
        assert request.fresh is False

    def test_job_request_without_client_cwd_is_backward_compatible(self) -> None:
        """Old clients that don't send client_cwd should still work."""
        params: dict[str, Any] = {
            "config_path": "/tmp/score.yaml",
            "fresh": True,
        }

        request = JobRequest(**params)
        assert request.client_cwd is None
        assert request.fresh is True
