"""Tests for mozart.cli.commands.config_cmd module.

Covers the `check` subcommand for config validation and the `show`
subcommand for live/disk config display.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from mozart.cli.commands.config_cmd import config_app

runner = CliRunner()


class TestConfigCheck:
    """Tests for `mozart config check` subcommand."""

    def test_valid_config_exits_0(self, tmp_path: Path) -> None:
        """Valid YAML config passes validation with exit code 0."""
        cfg = tmp_path / "valid.yaml"
        cfg.write_text(yaml.dump({"max_concurrent_jobs": 3}))

        result = runner.invoke(config_app, ["check", "--config", str(cfg)])
        assert result.exit_code == 0
        assert "Valid config" in result.output

    def test_invalid_config_exits_1(self, tmp_path: Path) -> None:
        """Invalid config (out-of-range value) exits with code 1."""
        cfg = tmp_path / "invalid.yaml"
        cfg.write_text(yaml.dump({"max_concurrent_jobs": -5}))

        result = runner.invoke(config_app, ["check", "--config", str(cfg)])
        assert result.exit_code == 1
        assert "Invalid config" in result.output

    def test_missing_file_exits_1(self, tmp_path: Path) -> None:
        """Non-existent config file exits with code 1."""
        cfg = tmp_path / "nonexistent.yaml"

        result = runner.invoke(config_app, ["check", "--config", str(cfg)])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_empty_config_is_valid(self, tmp_path: Path) -> None:
        """An empty YAML file (all defaults) is valid."""
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("")

        result = runner.invoke(config_app, ["check", "--config", str(cfg)])
        assert result.exit_code == 0

    def test_invalid_type_exits_1(self, tmp_path: Path) -> None:
        """Wrong type for a field (string where int expected) exits 1."""
        cfg = tmp_path / "badtype.yaml"
        cfg.write_text(yaml.dump({"max_concurrent_jobs": "not-a-number"}))

        result = runner.invoke(config_app, ["check", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_nested_invalid_config_exits_1(self, tmp_path: Path) -> None:
        """Invalid nested config (bad resource limit) exits 1."""
        cfg = tmp_path / "nested_invalid.yaml"
        cfg.write_text(yaml.dump({
            "resource_limits": {"max_memory_mb": 100},  # below 512 minimum
        }))

        result = runner.invoke(config_app, ["check", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_valid_nested_config_exits_0(self, tmp_path: Path) -> None:
        """Valid nested config passes."""
        cfg = tmp_path / "nested_valid.yaml"
        cfg.write_text(yaml.dump({
            "socket": {"path": "/tmp/custom.sock", "backlog": 10},
            "max_concurrent_jobs": 8,
            "resource_limits": {"max_memory_mb": 2048},
        }))

        result = runner.invoke(config_app, ["check", "--config", str(cfg)])
        assert result.exit_code == 0


class TestConfigShow:
    """Tests for `mozart config show` subcommand with live IPC support."""

    def test_show_live_config_from_conductor(self) -> None:
        """show displays live config when conductor is running."""
        from mozart.daemon.config import DaemonConfig

        live_config = DaemonConfig(max_concurrent_jobs=12)
        live_dict = live_config.model_dump(mode="json")

        with patch(
            "mozart.cli.commands.config_cmd._try_live_config",
            return_value=live_dict,
        ):
            result = runner.invoke(config_app, ["show"])

        assert result.exit_code == 0
        assert "live" in result.output
        assert "12" in result.output

    def test_show_fallback_to_disk_when_conductor_offline(
        self, tmp_path: Path,
    ) -> None:
        """show falls back to disk config when conductor is not running."""
        cfg = tmp_path / "daemon.yaml"
        cfg.write_text(yaml.dump({"max_concurrent_jobs": 7}))

        with patch(
            "mozart.cli.commands.config_cmd._try_live_config",
            return_value=None,
        ):
            result = runner.invoke(
                config_app, ["show", "--config", str(cfg)],
            )

        assert result.exit_code == 0
        assert "live" not in result.output
        assert "7" in result.output

    def test_show_defaults_when_no_conductor_no_file(self) -> None:
        """show displays defaults when conductor is offline and no config file exists."""
        with patch(
            "mozart.cli.commands.config_cmd._try_live_config",
            return_value=None,
        ):
            result = runner.invoke(
                config_app,
                ["show", "--config", "/tmp/nonexistent-mozart-test.yaml"],
            )

        assert result.exit_code == 0
        assert "defaults" in result.output


class TestFlattenModel:
    """Tests for _flatten_model helper used in config display."""

    def test_flatten_model_produces_dotted_keys(self) -> None:
        """Nested dicts are flattened to dot-notation keys."""
        from mozart.cli.commands.config_cmd import _flatten_model

        data = {"socket": {"path": "/tmp/test.sock", "backlog": 5}, "max_concurrent_jobs": 3}
        flat = _flatten_model(data)

        assert "socket.path" in flat
        assert "socket.backlog" in flat
        assert "max_concurrent_jobs" in flat
        # No "." prefix on top-level keys
        assert flat["max_concurrent_jobs"] == 3
        assert flat["socket.path"] == "/tmp/test.sock"

    def test_flatten_model_no_double_dot(self) -> None:
        """Ensure no '.socket.path' (leading dot) in flattened output."""
        from mozart.cli.commands.config_cmd import _flatten_model

        data = {"a": {"b": {"c": 1}}}
        flat = _flatten_model(data)

        assert "a.b.c" in flat
        # No key should start with a dot
        for key in flat:
            assert not key.startswith("."), f"Key '{key}' has leading dot"


class TestCheckSubcommandRegistered:
    """Verify the check subcommand is registered in config_app."""

    def test_check_in_commands(self) -> None:
        """The 'check' command is registered in config_app."""
        cmds = [c.name for c in config_app.registered_commands]
        assert "check" in cmds
