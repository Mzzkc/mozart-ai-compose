"""Smoke tests for CLI entry points and module imports.

These tests verify that:
1. All public modules can be imported without errors
2. All CLI commands respond to --help without crashing
3. CLI commands fail gracefully with missing/invalid arguments

No mocking, no fixtures, no complex setup — just basic sanity checks.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mozart.cli import app

runner = CliRunner()


# =============================================================================
# Module import smoke tests
# =============================================================================

# Every top-level package under mozart should be importable.
# Catching circular imports, missing deps, and syntax errors.
IMPORTABLE_MODULES = [
    # Core
    "mozart",
    "mozart.core",
    "mozart.core.checkpoint",
    "mozart.core.constants",
    "mozart.core.fan_out",
    "mozart.core.logging",
    # Config
    "mozart.core.config",
    "mozart.core.config.backend",
    "mozart.core.config.execution",
    "mozart.core.config.job",
    "mozart.core.config.learning",
    "mozart.core.config.orchestration",
    "mozart.core.config.workspace",
    # Errors
    "mozart.core.errors",
    "mozart.core.errors.classifier",
    "mozart.core.errors.codes",
    "mozart.core.errors.models",
    "mozart.core.errors.parsers",
    "mozart.core.errors.signals",
    # CLI
    "mozart.cli",
    "mozart.cli.helpers",
    "mozart.cli.output",
    "mozart.cli.commands",
    "mozart.cli.commands.run",
    "mozart.cli.commands.resume",
    "mozart.cli.commands.pause",
    "mozart.cli.commands.status",
    "mozart.cli.commands.validate",
    "mozart.cli.commands.recover",
    "mozart.cli.commands.diagnose",
    "mozart.cli.commands.dashboard",
    "mozart.cli.commands.config_cmd",
    # Backends
    "mozart.backends",
    "mozart.backends.base",
    "mozart.backends.claude_cli",
    # State
    "mozart.state",
    "mozart.state.base",
    "mozart.state.json_backend",
    "mozart.state.memory",
    "mozart.state.sqlite_backend",
    # Execution
    "mozart.execution",
    "mozart.execution.hooks",
    "mozart.execution.runner",
    "mozart.execution.runner.base",
    "mozart.execution.runner.sheet",
    "mozart.execution.runner.lifecycle",
    "mozart.execution.runner.recovery",
    "mozart.execution.runner.cost",
    "mozart.execution.runner.models",
    "mozart.execution.validation",
    "mozart.execution.validation.engine",
    # Learning
    "mozart.learning",
    "mozart.learning.patterns",
    "mozart.learning.migration",
    "mozart.learning.store",
    "mozart.learning.store.base",
    "mozart.learning.store.models",
    # Validation
    "mozart.validation",
    "mozart.validation.base",
    "mozart.validation.runner",
    "mozart.validation.reporter",
    "mozart.validation.checks",
    "mozart.validation.checks.jinja",
    "mozart.validation.checks.paths",
    "mozart.validation.checks.config",
    # Daemon
    "mozart.daemon",
    "mozart.daemon.config",
    "mozart.daemon.detect",
    "mozart.daemon.exceptions",
    "mozart.daemon.health",
    "mozart.daemon.backpressure",
    "mozart.daemon.ipc",
    "mozart.daemon.ipc.protocol",
    "mozart.daemon.ipc.errors",
    # MCP
    "mozart.mcp",
    "mozart.mcp.resources",
    "mozart.mcp.tools",
    # Notifications
    "mozart.notifications",
    "mozart.notifications.base",
    "mozart.notifications.factory",
    # Prompts
    "mozart.prompts",
    "mozart.prompts.templating",
    # Utils
    "mozart.utils",
    "mozart.utils.time",
    # Workspace
    "mozart.workspace",
    "mozart.workspace.lifecycle",
]


class TestModuleImports:
    """Verify all public modules import without errors."""

    @pytest.mark.parametrize("module_name", IMPORTABLE_MODULES)
    def test_module_imports(self, module_name: str) -> None:
        """Each module should import cleanly without side effects."""
        mod = importlib.import_module(module_name)
        assert mod is not None


# =============================================================================
# CLI --help smoke tests (commands missing from existing test suite)
# =============================================================================


class TestCoreCommandsSmoke:
    """Smoke tests for core CLI commands missing help tests."""

    def test_run_help(self) -> None:
        """Test run --help exits cleanly and shows usage."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout.lower()

    def test_resume_help(self) -> None:
        """Test resume --help exits cleanly and shows usage."""
        result = runner.invoke(app, ["resume", "--help"])
        assert result.exit_code == 0
        assert "resume" in result.stdout.lower()

    def test_status_help(self) -> None:
        """Test status --help exits cleanly and shows usage."""
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "status" in result.stdout.lower()

    def test_list_help(self) -> None:
        """Test list --help exits cleanly and shows usage."""
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0

    def test_validate_help(self) -> None:
        """Test validate --help exits cleanly and shows usage."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.stdout.lower()

    def test_errors_help(self) -> None:
        """Test errors --help exits cleanly and shows usage."""
        result = runner.invoke(app, ["errors", "--help"])
        assert result.exit_code == 0

    def test_history_help(self) -> None:
        """Test history --help exits cleanly and shows usage."""
        result = runner.invoke(app, ["history", "--help"])
        assert result.exit_code == 0

    def test_recover_help(self) -> None:
        """Test recover --help exits cleanly (hidden command)."""
        result = runner.invoke(app, ["recover", "--help"])
        assert result.exit_code == 0

    def test_mcp_help(self) -> None:
        """Test mcp --help exits cleanly and shows usage."""
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0

    def test_modify_help(self) -> None:
        """Test modify --help exits cleanly and shows usage."""
        result = runner.invoke(app, ["modify", "--help"])
        assert result.exit_code == 0
        assert "modify" in result.stdout.lower()


class TestConfigCommandsSmoke:
    """Smoke tests for config subcommands."""

    def test_config_help(self) -> None:
        """Test config --help exits cleanly."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "config" in result.stdout.lower()

    def test_config_show_help(self) -> None:
        """Test config show --help exits cleanly."""
        result = runner.invoke(app, ["config", "show", "--help"])
        assert result.exit_code == 0

    def test_config_set_help(self) -> None:
        """Test config set --help exits cleanly."""
        result = runner.invoke(app, ["config", "set", "--help"])
        assert result.exit_code == 0

    def test_config_path_help(self) -> None:
        """Test config path --help exits cleanly."""
        result = runner.invoke(app, ["config", "path", "--help"])
        assert result.exit_code == 0

    def test_config_init_help(self) -> None:
        """Test config init --help exits cleanly."""
        result = runner.invoke(app, ["config", "init", "--help"])
        assert result.exit_code == 0


# =============================================================================
# CLI error handling smoke tests
# =============================================================================


class TestCommandErrorHandling:
    """Verify commands fail gracefully with missing/invalid arguments."""

    def test_run_no_args(self) -> None:
        """Test run without config file exits with error."""
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0

    def test_validate_no_args(self) -> None:
        """Test validate without config file exits with error."""
        result = runner.invoke(app, ["validate"])
        assert result.exit_code != 0

    def test_resume_no_args(self) -> None:
        """Test resume without job_id exits with error."""
        result = runner.invoke(app, ["resume"])
        assert result.exit_code != 0

    def test_status_nonexistent_job(self, tmp_path: Path) -> None:
        """Test status with nonexistent job returns error."""
        workspace = tmp_path / "empty_ws"
        workspace.mkdir()
        result = runner.invoke(
            app, ["status", "nonexistent-job", "--workspace", str(workspace)]
        )
        assert result.exit_code != 0

    def test_errors_no_args(self) -> None:
        """Test errors without job_id exits with error."""
        result = runner.invoke(app, ["errors"])
        assert result.exit_code != 0

    def test_history_no_args(self) -> None:
        """Test history without job_id exits with error."""
        result = runner.invoke(app, ["history"])
        assert result.exit_code != 0

    def test_recover_no_args(self) -> None:
        """Test recover without job_id exits with error."""
        result = runner.invoke(app, ["recover"])
        assert result.exit_code != 0

    def test_pause_no_args(self) -> None:
        """Test pause without job_id exits with error."""
        result = runner.invoke(app, ["pause"])
        assert result.exit_code != 0

    def test_unknown_command(self) -> None:
        """Test invoking a nonexistent command exits with error."""
        result = runner.invoke(app, ["nonexistent-command-xyz"])
        assert result.exit_code != 0


# =============================================================================
# Global flags smoke tests
# =============================================================================


class TestGlobalFlags:
    """Verify global flags work with various commands."""

    def test_verbose_flag(self) -> None:
        """Test --verbose flag doesn't crash."""
        result = runner.invoke(app, ["--verbose", "validate", "--help"])
        assert result.exit_code == 0

    def test_quiet_flag(self) -> None:
        """Test --quiet flag doesn't crash."""
        result = runner.invoke(app, ["--quiet", "validate", "--help"])
        assert result.exit_code == 0

    def test_version_flag(self) -> None:
        """Test --version flag shows version and exits."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "Mozart AI Compose" in result.stdout


# =============================================================================
# LogFollower unit tests
# =============================================================================


class TestLogFollower:
    """Unit tests for the extracted LogFollower class."""

    def test_parse_line_json(self, tmp_path: Path) -> None:
        """JSON lines are parsed into dicts."""
        from mozart.cli.commands.diagnose import LogFollower
        follower = LogFollower(log_path=tmp_path / "test.log")
        entry = follower.parse_line('{"event": "started", "level": "INFO"}')
        assert entry is not None
        assert entry["event"] == "started"
        assert entry["level"] == "INFO"

    def test_parse_line_plain_text(self, tmp_path: Path) -> None:
        """Non-JSON lines are wrapped in a dict with _raw flag."""
        from mozart.cli.commands.diagnose import LogFollower
        follower = LogFollower(log_path=tmp_path / "test.log")
        entry = follower.parse_line("plain text log line")
        assert entry is not None
        assert entry["_raw"] is True
        assert entry["event"] == "plain text log line"

    def test_parse_line_empty(self, tmp_path: Path) -> None:
        """Empty lines return None."""
        from mozart.cli.commands.diagnose import LogFollower
        follower = LogFollower(log_path=tmp_path / "test.log")
        assert follower.parse_line("") is None
        assert follower.parse_line("   ") is None

    def test_should_include_no_filters(self, tmp_path: Path) -> None:
        """With no filters, all entries pass."""
        from mozart.cli.commands.diagnose import LogFollower
        follower = LogFollower(log_path=tmp_path / "test.log")
        assert follower.should_include({"event": "x", "level": "DEBUG"}) is True
        assert follower.should_include({"event": "x", "level": "ERROR"}) is True

    def test_should_include_level_filter(self, tmp_path: Path) -> None:
        """Level filter excludes entries below threshold."""
        from mozart.cli.commands.diagnose import LogFollower
        follower = LogFollower(log_path=tmp_path / "test.log", min_level=2)  # WARNING
        assert follower.should_include({"event": "x", "level": "DEBUG"}) is False
        assert follower.should_include({"event": "x", "level": "INFO"}) is False
        assert follower.should_include({"event": "x", "level": "WARNING"}) is True
        assert follower.should_include({"event": "x", "level": "ERROR"}) is True

    def test_should_include_job_filter(self, tmp_path: Path) -> None:
        """Job ID filter excludes non-matching entries."""
        from mozart.cli.commands.diagnose import LogFollower
        follower = LogFollower(log_path=tmp_path / "test.log", job_id="my-job")
        assert follower.should_include({"event": "x", "job_id": "my-job"}) is True
        assert follower.should_include({"event": "x", "job_id": "other"}) is False
        assert follower.should_include({"event": "x"}) is False

    def test_format_entry_json_mode(self, tmp_path: Path) -> None:
        """JSON output mode returns raw JSON string."""
        from mozart.cli.commands.diagnose import LogFollower
        follower = LogFollower(log_path=tmp_path / "test.log", json_output=True)
        import json
        entry = {"event": "test", "level": "INFO"}
        result = follower.format_entry(entry)
        assert json.loads(result) == entry

    def test_format_entry_raw_line(self, tmp_path: Path) -> None:
        """Raw lines are returned as-is."""
        from mozart.cli.commands.diagnose import LogFollower
        follower = LogFollower(log_path=tmp_path / "test.log")
        result = follower.format_entry({"event": "plain text", "_raw": True})
        assert result == "plain text"

    def test_format_entry_structured(self, tmp_path: Path) -> None:
        """Structured entries include level, component, and event."""
        from mozart.cli.commands.diagnose import LogFollower
        follower = LogFollower(log_path=tmp_path / "test.log")
        entry = {
            "timestamp": "2026-01-01T10:30:00+00:00",
            "level": "WARNING",
            "event": "rate_limit_detected",
            "component": "runner",
            "job_id": "test-job",
            "sheet_num": 3,
        }
        result = follower.format_entry(entry)
        assert "WARNING" in result
        assert "runner" in result
        assert "rate_limit_detected" in result
        assert "sheet:3" in result

    def test_read_lines_from_file(self, tmp_path: Path) -> None:
        """Reading lines from a real file works."""
        from mozart.cli.commands.diagnose import LogFollower
        log_file = tmp_path / "test.log"
        log_file.write_text("line1\nline2\nline3\n")
        follower = LogFollower(log_path=log_file)
        lines = follower.read_lines()
        assert len(lines) == 3

    def test_read_lines_with_limit(self, tmp_path: Path) -> None:
        """Line limit returns only the last N lines."""
        from mozart.cli.commands.diagnose import LogFollower
        log_file = tmp_path / "test.log"
        log_file.write_text("line1\nline2\nline3\nline4\nline5\n")
        follower = LogFollower(log_path=log_file)
        lines = follower.read_lines(num_lines=2)
        assert len(lines) == 2
        assert "line4" in lines[0]
        assert "line5" in lines[1]

    def test_read_lines_nonexistent_file(self, tmp_path: Path) -> None:
        """Reading from nonexistent file returns empty list."""
        from mozart.cli.commands.diagnose import LogFollower
        follower = LogFollower(log_path=tmp_path / "missing.log")
        lines = follower.read_lines()
        assert lines == []
