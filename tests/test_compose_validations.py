"""Tests for the validation generator module."""

from __future__ import annotations

from marianne.compose.validations import ValidationGenerator


def _make_agent_def(name: str = "canyon") -> dict[str, object]:
    return {
        "name": name,
        "voice": "Structure persists.",
        "focus": "architecture",
    }


def _make_defaults() -> dict[str, object]:
    return {
        "validations": [
            {
                "command": "pytest tests/ -x",
                "description": "Tests pass",
                "timeout_seconds": 300,
            },
        ],
    }


class TestValidationGenerator:
    """Tests for ValidationGenerator."""

    def test_generates_recon_validation(self) -> None:
        """Generates a file_exists check for the recon report."""
        gen = ValidationGenerator()
        result = gen.generate(_make_agent_def(), {})

        recon_vals = [v for v in result if "recon" in v.get("description", "").lower()]
        assert len(recon_vals) >= 1
        assert recon_vals[0]["type"] == "file_exists"
        assert "stage == 1" in recon_vals[0]["condition"]

    def test_generates_plan_validation(self) -> None:
        """Generates a file_exists check for the plan document."""
        gen = ValidationGenerator()
        result = gen.generate(_make_agent_def(), {})

        plan_vals = [v for v in result if "plan" in v.get("description", "").lower()]
        assert len(plan_vals) >= 1
        assert plan_vals[0]["type"] == "file_exists"
        assert "stage == 2" in plan_vals[0]["condition"]

    def test_generates_aar_validations(self) -> None:
        """Generates content_contains checks for AAR SUSTAIN/IMPROVE."""
        gen = ValidationGenerator()
        result = gen.generate(_make_agent_def(), {})

        aar_vals = [v for v in result if "AAR" in v.get("description", "")]
        assert len(aar_vals) >= 2

        patterns = [v.get("pattern", "") for v in aar_vals]
        assert "SUSTAIN:" in patterns
        assert "IMPROVE:" in patterns

    def test_includes_user_validations(self) -> None:
        """User-defined validations from defaults are included."""
        gen = ValidationGenerator()
        result = gen.generate(_make_agent_def(), _make_defaults())

        test_vals = [v for v in result if "Tests pass" in v.get("description", "")]
        assert len(test_vals) == 1
        assert test_vals[0]["type"] == "command_succeeds"
        assert "pytest" in test_vals[0]["command"]
        assert "stage == 3" in test_vals[0]["condition"]

    def test_generates_inspection_validation(self) -> None:
        """Generates file_exists check for inspection report."""
        gen = ValidationGenerator()
        result = gen.generate(_make_agent_def(), {})

        inspect_vals = [v for v in result if "inspection" in v.get("description", "").lower()]
        assert len(inspect_vals) >= 1

    def test_agent_name_in_paths(self) -> None:
        """Agent name is correctly embedded in file paths."""
        gen = ValidationGenerator()
        result = gen.generate(_make_agent_def("forge"), {})

        for v in result:
            if "path" in v:
                if "cycle-state" in v["path"]:
                    assert "forge" in v["path"]

    def test_maturity_and_budget_with_dirs(self) -> None:
        """Maturity and token budget checks generated when dirs provided."""
        gen = ValidationGenerator()
        result = gen.generate(
            _make_agent_def(),
            {},
            agents_dir="/test/agents",
            instruments_dir="/test/instruments",
        )

        maturity_vals = [v for v in result if "maturity" in v.get("description", "").lower()]
        assert len(maturity_vals) >= 1

        budget_vals = [v for v in result if "budget" in v.get("description", "").lower()]
        assert len(budget_vals) >= 1

    def test_generate_structural_recon(self) -> None:
        """generate_structural produces recon validation."""
        gen = ValidationGenerator()
        result = gen.generate_structural("canyon", "recon")

        assert len(result) >= 1
        assert result[0]["type"] == "file_exists"
        assert "recon" in result[0]["path"]

    def test_generate_structural_aar(self) -> None:
        """generate_structural produces AAR validations."""
        gen = ValidationGenerator()
        result = gen.generate_structural("canyon", "aar")

        assert len(result) >= 2
        patterns = [v.get("pattern", "") for v in result]
        assert "SUSTAIN:" in patterns
        assert "IMPROVE:" in patterns

    def test_generate_structural_unknown_phase(self) -> None:
        """generate_structural returns empty for unknown phase."""
        gen = ValidationGenerator()
        result = gen.generate_structural("canyon", "nonexistent")

        assert result == []

    def test_custom_agent_validations(self) -> None:
        """Agent-level custom validations are appended."""
        gen = ValidationGenerator()
        agent = dict(_make_agent_def())
        agent["validations"] = [
            {"type": "file_exists", "path": "/custom/path.md", "description": "Custom check"},
        ]
        result = gen.generate(agent, {})

        custom = [v for v in result if v.get("description") == "Custom check"]
        assert len(custom) == 1

    def test_workspace_uses_format_syntax(self) -> None:
        """Paths use {workspace} format syntax, not Jinja2 {{workspace}}."""
        gen = ValidationGenerator()
        result = gen.generate(_make_agent_def(), {})

        for v in result:
            path = v.get("path", "")
            if path:
                assert "{workspace}" in path, f"Missing {{workspace}} in path: {path}"
                assert "{{workspace}}" not in path, (
                    f"Path uses Jinja2 double braces instead of format syntax: {path}"
                )

    def test_workspace_format_in_commands(self) -> None:
        """Commands use {workspace} format syntax, not Jinja2 {{workspace}}."""
        gen = ValidationGenerator()
        result = gen.generate(
            _make_agent_def(),
            {},
            agents_dir="/test/agents",
            instruments_dir="/test/instruments",
        )

        for v in result:
            cmd = v.get("command", "")
            if "{workspace}" in cmd or "{{workspace}}" in cmd:
                assert "{{workspace}}" not in cmd, f"Command uses Jinja2 double braces: {cmd}"

    def test_workspace_format_in_structural(self) -> None:
        """generate_structural uses {workspace} format syntax."""
        gen = ValidationGenerator()
        for phase in ("recon", "plan", "inspect", "aar"):
            result = gen.generate_structural("test-agent", phase)
            for v in result:
                path = v.get("path", "")
                if path:
                    assert "{workspace}" in path
                    assert "{{workspace}}" not in path

    def test_cli_instrument_validations_are_command_succeeds(self) -> None:
        """CLI instrument sheets (temperature, maturity, budget) use command_succeeds."""
        gen = ValidationGenerator()
        result = gen.generate(
            _make_agent_def(),
            {},
            agents_dir="/test/agents",
            instruments_dir="/test/instruments",
        )

        # Temperature check (stage 4), maturity (stage 11), budget (stage 12)
        cli_stages = {"stage == 4", "stage == 11", "stage == 12"}
        cli_vals = [v for v in result if v.get("condition", "") in cli_stages]

        assert len(cli_vals) >= 3, (
            f"Expected at least 3 CLI instrument validations, got {len(cli_vals)}"
        )
        for v in cli_vals:
            assert v["type"] == "command_succeeds", (
                f"CLI validation at {v['condition']} should be command_succeeds, got {v['type']}"
            )

    def test_temperature_check_validation(self) -> None:
        """Temperature check validation generated when dirs provided."""
        gen = ValidationGenerator()
        result = gen.generate(
            _make_agent_def(),
            {},
            agents_dir="/test/agents",
            instruments_dir="/test/instruments",
        )

        temp_vals = [v for v in result if "temperature" in v.get("description", "").lower()]
        assert len(temp_vals) >= 1
        assert temp_vals[0]["type"] == "command_succeeds"
        assert "stage == 4" in temp_vals[0]["condition"]
        assert "temperature-check.sh" in temp_vals[0]["command"]

    def test_no_temperature_check_without_dirs(self) -> None:
        """Temperature check is NOT generated when dirs are missing."""
        gen = ValidationGenerator()
        result = gen.generate(_make_agent_def(), {})

        temp_vals = [v for v in result if "temperature" in v.get("description", "").lower()]
        assert len(temp_vals) == 0

    def test_coverage_validations_on_inspect(self) -> None:
        """Coverage validations from defaults applied to inspect sheet (stage 7)."""
        gen = ValidationGenerator()
        defaults: dict[str, object] = {
            "coverage_validations": [
                {
                    "command": "pytest --cov=src tests/",
                    "description": "Coverage check",
                    "timeout_seconds": 300,
                },
            ],
        }
        result = gen.generate(_make_agent_def(), defaults)

        cov_vals = [v for v in result if "Coverage check" in v.get("description", "")]
        assert len(cov_vals) == 1
        assert cov_vals[0]["type"] == "command_succeeds"
        assert "stage == 7" in cov_vals[0]["condition"]
        assert "pytest --cov" in cov_vals[0]["command"]

    def test_correct_validations_per_phase_type(self) -> None:
        """Each phase type gets appropriate validations."""
        gen = ValidationGenerator()
        defaults: dict[str, object] = {
            "validations": [{"command": "make test", "description": "TDD"}],
            "coverage_validations": [{"command": "make cov", "description": "Cov"}],
        }
        result = gen.generate(
            _make_agent_def(),
            defaults,
            agents_dir="/a",
            instruments_dir="/i",
        )

        # Group validations by condition stage
        by_stage: dict[str, list[dict[str, object]]] = {}
        for v in result:
            cond = v.get("condition", "")
            by_stage.setdefault(cond, []).append(v)

        # Recon (stage 1): file_exists
        assert any(v["type"] == "file_exists" for v in by_stage.get("stage == 1", []))
        # Plan (stage 2): file_exists
        assert any(v["type"] == "file_exists" for v in by_stage.get("stage == 2", []))
        # Work (stage 3): command_succeeds (TDD)
        assert any(v["type"] == "command_succeeds" for v in by_stage.get("stage == 3", []))
        # Temperature check (stage 4): command_succeeds
        assert any(v["type"] == "command_succeeds" for v in by_stage.get("stage == 4", []))
        # Inspect (stage 7): file_exists + command_succeeds (coverage)
        stage_7 = by_stage.get("stage == 7", [])
        assert any(v["type"] == "file_exists" for v in stage_7)
        assert any(v["type"] == "command_succeeds" for v in stage_7)
        # AAR (stage 8): content_contains
        assert any(v["type"] == "content_contains" for v in by_stage.get("stage == 8", []))
        # Maturity (stage 11): command_succeeds
        assert any(v["type"] == "command_succeeds" for v in by_stage.get("stage == 11", []))
        # Budget (stage 12): command_succeeds
        assert any(v["type"] == "command_succeeds" for v in by_stage.get("stage == 12", []))
