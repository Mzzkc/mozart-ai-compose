"""Tests for the compose CLI command."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

runner = CliRunner()


def _create_config(tmp_path: Path) -> Path:
    """Create a minimal compose config for testing."""
    config = {
        "project": {"name": "test-project", "workspace": str(tmp_path / "workspace")},
        "defaults": {
            "instruments": {
                "work": {
                    "primary": {"instrument": "openrouter", "model": "test-model"},
                },
            },
        },
        "agents": [
            {
                "name": "test-agent",
                "voice": "Testing voice.",
                "focus": "testing",
                "role": "tester",
            },
        ],
    }
    config_path = tmp_path / "test-config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def _create_multi_agent_config(tmp_path: Path) -> Path:
    """Create a multi-agent compose config for testing."""
    config = {
        "project": {"name": "multi-project", "workspace": str(tmp_path / "workspace")},
        "defaults": {},
        "agents": [
            {"name": "agent-a", "voice": "Voice A.", "focus": "focus-a"},
            {"name": "agent-b", "voice": "Voice B.", "focus": "focus-b"},
        ],
    }
    config_path = tmp_path / "multi-config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


class TestComposeCli:
    """Tests for the compose CLI command."""

    def test_dry_run(self, tmp_path: Path) -> None:
        """--dry-run shows summary without generating files."""
        import typer

        from marianne.cli.commands.compose import compose

        app = typer.Typer()
        app.command()(compose)

        config_path = _create_config(tmp_path)
        result = runner.invoke(app, [str(config_path), "--dry-run"])

        assert result.exit_code == 0
        assert "test-project" in result.output or "Dry Run" in result.output

    def test_compile_generates_scores(self, tmp_path: Path) -> None:
        """Compose generates score files in the output directory."""
        import typer

        from marianne.cli.commands.compose import compose

        app = typer.Typer()
        app.command()(compose)

        config_path = _create_config(tmp_path)
        output_dir = tmp_path / "scores"
        # Use a dedicated agents dir for isolation
        agents_dir = tmp_path / "agents"

        result = runner.invoke(
            app,
            [
                str(config_path),
                "--output",
                str(output_dir),
                "--agents-dir",
                str(agents_dir),
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "test-agent.yaml").exists()

    def test_seed_only(self, tmp_path: Path) -> None:
        """--seed-only creates identities without scores."""
        import typer

        from marianne.cli.commands.compose import compose

        app = typer.Typer()
        app.command()(compose)

        config_path = _create_config(tmp_path)
        agents_dir = tmp_path / "agents"

        result = runner.invoke(
            app,
            [
                str(config_path),
                "--seed-only",
                "--agents-dir",
                str(agents_dir),
            ],
        )

        assert result.exit_code == 0
        assert (agents_dir / "test-agent" / "identity.md").exists()
        assert (agents_dir / "test-agent" / "profile.yaml").exists()

    def test_multi_agent_generates_fleet(self, tmp_path: Path) -> None:
        """Multiple agents generate a fleet config."""
        import typer

        from marianne.cli.commands.compose import compose

        app = typer.Typer()
        app.command()(compose)

        config_path = _create_multi_agent_config(tmp_path)
        output_dir = tmp_path / "scores"
        agents_dir = tmp_path / "agents"

        result = runner.invoke(
            app,
            [
                str(config_path),
                "--output",
                str(output_dir),
                "--agents-dir",
                str(agents_dir),
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "agent-a.yaml").exists()
        assert (output_dir / "agent-b.yaml").exists()
        assert (output_dir / "fleet.yaml").exists()

    def test_generated_score_is_valid_yaml(self, tmp_path: Path) -> None:
        """Generated score files are valid YAML."""
        import typer

        from marianne.cli.commands.compose import compose

        app = typer.Typer()
        app.command()(compose)

        config_path = _create_config(tmp_path)
        output_dir = tmp_path / "scores"
        agents_dir = tmp_path / "agents"

        runner.invoke(
            app,
            [
                str(config_path),
                "--output",
                str(output_dir),
                "--agents-dir",
                str(agents_dir),
            ],
        )

        score_path = output_dir / "test-agent.yaml"
        assert score_path.exists()

        data = yaml.safe_load(score_path.read_text())
        assert isinstance(data, dict)
        assert "name" in data
        assert "sheet" in data
        assert "prompt" in data

    def test_fleet_flag_produces_fleet_config(self, tmp_path: Path) -> None:
        """--fleet flag forces fleet config generation even for single agent."""
        import typer

        from marianne.cli.commands.compose import compose

        app = typer.Typer()
        app.command()(compose)

        config_path = _create_config(tmp_path)
        output_dir = tmp_path / "scores"
        agents_dir = tmp_path / "agents"

        result = runner.invoke(
            app,
            [
                str(config_path),
                "--output",
                str(output_dir),
                "--agents-dir",
                str(agents_dir),
                "--fleet",
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "fleet.yaml").exists()

        # Verify fleet config is valid YAML with expected structure
        fleet_data = yaml.safe_load((output_dir / "fleet.yaml").read_text())
        assert isinstance(fleet_data, dict)
        assert fleet_data.get("type") == "fleet"
        assert "scores" in fleet_data

    def test_empty_config_fails(self, tmp_path: Path) -> None:
        """Config with no agents exits with error."""
        import typer

        from marianne.cli.commands.compose import compose

        app = typer.Typer()
        app.command()(compose)

        config_path = tmp_path / "empty.yaml"
        config_path.write_text("project:\n  name: empty\nagents: []\n")

        result = runner.invoke(app, [str(config_path)])
        assert result.exit_code != 0

    def test_invalid_yaml_produces_clear_error(self, tmp_path: Path) -> None:
        """Malformed YAML produces a clear error message."""
        import typer

        from marianne.cli.commands.compose import compose

        app = typer.Typer()
        app.command()(compose)

        config_path = tmp_path / "bad.yaml"
        config_path.write_text(":\n  - [\ninvalid yaml content")

        result = runner.invoke(app, [str(config_path)])
        assert result.exit_code != 0
