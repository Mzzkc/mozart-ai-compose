"""TDD tests for F-501: Conductor clone start command missing.

F-501: Users cannot start a clone conductor because `--conductor-clone` is
a global flag that goes BEFORE the command (`mzt --conductor-clone= start`),
but users expect command-specific syntax (`mzt start --conductor-clone`).

The fix makes BOTH syntaxes work:
1. Global: `mzt --conductor-clone=test start` (already works)
2. Command-specific: `mzt start --conductor-clone=test` (needs implementation)

TDD: These tests are written RED first.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer.testing

from marianne.cli import app


class TestConductorCloneCommandOptions:
    """F-501: start/stop/restart commands need --conductor-clone option."""

    def test_start_with_command_level_clone_flag(self, tmp_path: Path) -> None:
        """start --conductor-clone=test should work (command-level flag)."""
        runner = typer.testing.CliRunner()

        with (
            patch("marianne.daemon.process.start_conductor") as mock_start,
            patch("marianne.daemon.clone.set_clone_name") as mock_set_clone,
        ):
            # Command-level flag syntax: mzt start --conductor-clone=test
            result = runner.invoke(app, ["start", "--conductor-clone=test"])

            # Should succeed
            assert result.exit_code == 0, f"Expected success, got: {result.stdout}"

            # Should call set_clone_name with 'test'
            mock_set_clone.assert_called_once_with("test")

            # start_conductor should be called with clone_name='test'
            mock_start.assert_called_once()
            call_kwargs = mock_start.call_args.kwargs
            assert call_kwargs.get("clone_name") == "test"

    def test_start_with_default_clone_flag(self, tmp_path: Path) -> None:
        """start --conductor-clone= (equals, no value) should use default clone."""
        runner = typer.testing.CliRunner()

        with (
            patch("marianne.daemon.process.start_conductor") as mock_start,
            patch("marianne.daemon.clone.set_clone_name") as mock_set_clone,
        ):
            # Command-level flag with default: mzt start --conductor-clone=
            # (equals sign with empty value)
            result = runner.invoke(app, ["start", "--conductor-clone="])

            assert result.exit_code == 0, f"Expected success, got: {result.stdout}"

            # Should call set_clone_name with empty string (default clone)
            mock_set_clone.assert_called_once_with("")

            mock_start.assert_called_once()
            call_kwargs = mock_start.call_args.kwargs
            assert call_kwargs.get("clone_name") == ""

    def test_stop_with_command_level_clone_flag(self, tmp_path: Path) -> None:
        """stop --conductor-clone=test should work."""
        runner = typer.testing.CliRunner()

        mock_pid_file = tmp_path / "test.pid"
        mock_pid_file.write_text("12345")

        with (
            patch("marianne.daemon.process.stop_conductor") as mock_stop,
            patch("marianne.daemon.clone.set_clone_name") as mock_set_clone,
            patch("marianne.daemon.clone.resolve_clone_paths") as mock_resolve,
        ):
            mock_paths = MagicMock()
            mock_paths.pid_file = mock_pid_file
            mock_paths.socket = tmp_path / "test.sock"
            mock_resolve.return_value = mock_paths

            result = runner.invoke(app, ["stop", "--conductor-clone=test"])

            assert result.exit_code == 0, f"Expected success, got: {result.stdout}"
            mock_set_clone.assert_called_once_with("test")
            mock_stop.assert_called_once()

    def test_restart_with_command_level_clone_flag(self, tmp_path: Path) -> None:
        """restart --conductor-clone=test should work."""
        runner = typer.testing.CliRunner()

        with (
            patch("marianne.daemon.process.start_conductor") as mock_start,
            patch("marianne.daemon.process.stop_conductor") as mock_stop,
            patch("marianne.daemon.process.wait_for_conductor_exit", return_value=True),
            patch("marianne.daemon.clone.set_clone_name") as mock_set_clone,
        ):
            result = runner.invoke(app, ["restart", "--conductor-clone=test"])

            assert result.exit_code == 0, f"Expected success, got: {result.stdout}"
            mock_set_clone.assert_called_once_with("test")
            mock_stop.assert_called_once()
            mock_start.assert_called_once()

    def test_global_flag_still_works(self, tmp_path: Path) -> None:
        """mzt --conductor-clone=test start should still work (backward compat)."""
        runner = typer.testing.CliRunner()

        with (
            patch("marianne.daemon.process.start_conductor") as mock_start,
        ):
            # Global flag syntax: mzt --conductor-clone=test start
            result = runner.invoke(app, ["--conductor-clone=test", "start"])

            assert result.exit_code == 0, f"Expected success, got: {result.stdout}"
            mock_start.assert_called_once()
            call_kwargs = mock_start.call_args.kwargs
            assert call_kwargs.get("clone_name") == "test"

    def test_command_flag_overrides_global(self, tmp_path: Path) -> None:
        """Command-level --conductor-clone should override global if both given."""
        runner = typer.testing.CliRunner()

        with (
            patch("marianne.daemon.process.start_conductor") as mock_start,
            patch("marianne.daemon.clone.set_clone_name") as mock_set_clone,
        ):
            # Both global and command-level: command should win
            result = runner.invoke(
                app, ["--conductor-clone=global", "start", "--conductor-clone=local"]
            )

            assert result.exit_code == 0, f"Expected success, got: {result.stdout}"

            # set_clone_name called twice: once for global, once for command
            # The command-level one is what matters
            assert mock_set_clone.call_count == 2
            # Last call should be 'local'
            assert mock_set_clone.call_args_list[-1][0] == ("local",)

            mock_start.assert_called_once()
            call_kwargs = mock_start.call_args.kwargs
            assert call_kwargs.get("clone_name") == "local"


class TestCloneDocumentationCorrectness:
    """F-501: Documentation shows wrong syntax - must be fixed."""

    def test_clone_module_documents_correct_global_syntax(self) -> None:
        """clone.py module docstring must show correct global flag syntax."""
        from marianne.daemon import clone

        doc = clone.__doc__
        assert doc is not None, "clone.py must have module docstring"

        # Should NOT show incorrect syntax like "mzt start --conductor-clone"
        # Should show correct global syntax like "mzt --conductor-clone= start"

        # Check for correct examples
        assert "mzt --conductor-clone" in doc or "mzt start --conductor-clone" in doc, (
            "clone.py should document conductor-clone usage"
        )

        # If it shows "mzt start --conductor-clone", that's ALSO valid now
        # (because we're adding command-level support).
        # If it shows "mzt --conductor-clone= start", that's the global syntax.
        # Both should work after F-501 fix.
