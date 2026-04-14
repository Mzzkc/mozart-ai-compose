"""Compilation pipeline — top-level orchestrator for the composition compiler.

Takes a semantic config YAML and produces complete Mozart scores for each
agent, plus identity directories, fleet configs, and shared technique modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from marianne.compose.fleet import FleetGenerator
from marianne.compose.identity import IdentitySeeder
from marianne.compose.instruments import InstrumentResolver
from marianne.compose.patterns import PatternExpander
from marianne.compose.sheets import SheetComposer
from marianne.compose.techniques import TechniqueWirer
from marianne.compose.validations import ValidationGenerator

_logger = logging.getLogger(__name__)


class CompilationPipeline:
    """Top-level compilation pipeline.

    Coordinates all compiler modules to produce complete Mozart scores
    from a semantic agent configuration.

    Usage::

        pipeline = CompilationPipeline()
        scores = pipeline.compile("config.yaml")
        # Returns: list of score file paths + identity directories created

        # Or programmatically:
        pipeline.compile_agent(agent_def, defaults, output_dir)
        pipeline.seed_identity(agent_def, agents_dir)
    """

    def __init__(
        self,
        *,
        agents_dir: Path | None = None,
        techniques_dir: Path | None = None,
        templates_dir: Path | None = None,
    ) -> None:
        self.identity_seeder = IdentitySeeder(agents_dir)
        self.sheet_composer = SheetComposer(templates_dir)
        self.technique_wirer = TechniqueWirer(techniques_dir)
        self.instrument_resolver = InstrumentResolver()
        self.validation_generator = ValidationGenerator()
        self.pattern_expander = PatternExpander()
        self.fleet_generator = FleetGenerator()

    def compile(
        self,
        config_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> list[Path]:
        """Compile a config file into Mozart scores.

        Args:
            config_path: Path to the semantic agent config YAML.
            output_dir: Output directory for generated scores. Defaults
                to ``scores/`` next to the config file.

        Returns:
            List of generated score file paths.
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        if output_dir is None:
            output_dir = config_path.parent / "scores"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        return self.compile_config(config, output_dir)

    def compile_config(
        self,
        config: dict[str, Any],
        output_dir: Path,
    ) -> list[Path]:
        """Compile a config dict into Mozart scores.

        Args:
            config: Parsed compiler config dict.
            output_dir: Directory to write generated scores.

        Returns:
            List of generated score file paths.
        """
        agents = config.get("agents", [])
        if not agents:
            raise ValueError("Config must contain at least one agent")

        defaults = config.get("defaults", {})
        project = config.get("project", {})
        workspace = project.get("workspace", str(output_dir / "workspace"))

        score_paths: list[Path] = []

        for agent_def in agents:
            path = self.compile_agent(
                agent_def, defaults, output_dir, workspace=workspace
            )
            score_paths.append(path)

        # Generate fleet config if multiple agents
        if len(agents) > 1:
            fleet_path = output_dir / "fleet.yaml"
            self.fleet_generator.write(config, output_dir, fleet_path)
            score_paths.append(fleet_path)
            _logger.info("Fleet config written: %s", fleet_path)

        _logger.info(
            "Compiled %d agent scores to %s", len(agents), output_dir
        )
        return score_paths

    def compile_agent(
        self,
        agent_def: dict[str, Any],
        defaults: dict[str, Any],
        output_dir: Path,
        *,
        workspace: str = "",
    ) -> Path:
        """Compile a single agent definition into a Mozart score.

        Args:
            agent_def: Agent definition dict.
            defaults: Global defaults from compiler config.
            output_dir: Directory to write the score.
            workspace: Workspace path for the agent.

        Returns:
            Path to the generated score file.
        """
        name = agent_def["name"]
        agents_dir = self.identity_seeder.agents_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Seed identity
        self.identity_seeder.seed(agent_def)

        # 2. Compose sheet structure
        sheet_config = self.sheet_composer.compose(
            agent_def, defaults, agents_dir=agents_dir
        )

        # 3. Wire techniques
        technique_result = self.technique_wirer.wire(
            agent_def, defaults, workspace=workspace
        )

        # Merge technique cadenzas into sheet cadenzas
        for sheet_num, tech_cadenzas in technique_result["cadenzas"].items():
            if sheet_num not in sheet_config.get("cadenzas", {}):
                sheet_config.setdefault("cadenzas", {})[sheet_num] = []
            sheet_config["cadenzas"][sheet_num].extend(tech_cadenzas)

        # 4. Resolve instruments
        instrument_result = self.instrument_resolver.resolve(agent_def, defaults)

        # Merge instrument assignments into sheet config
        if instrument_result.get("per_sheet_instruments"):
            sheet_config["per_sheet_instruments"] = instrument_result["per_sheet_instruments"]
        if instrument_result.get("per_sheet_instrument_config"):
            sheet_config["per_sheet_instrument_config"] = instrument_result[
                "per_sheet_instrument_config"
            ]
        if instrument_result.get("per_sheet_fallbacks"):
            sheet_config["per_sheet_fallbacks"] = instrument_result["per_sheet_fallbacks"]

        # 5. Generate validations
        validations = self.validation_generator.generate(
            agent_def, defaults, agents_dir=str(agents_dir)
        )

        # 6. Expand patterns
        pattern_names = agent_def.get("patterns", [])
        if pattern_names:
            self.pattern_expander.expand(pattern_names, agent_def)

        # 7. Build prompt config
        prompt_config = self._build_prompt(agent_def, defaults)

        # 8. Assemble score
        score = self._assemble_score(
            name=name,
            workspace=workspace or str(output_dir / "workspace"),
            sheet_config=sheet_config,
            prompt_config=prompt_config,
            instrument_result=instrument_result,
            validations=validations,
            defaults=defaults,
        )

        # Write score
        score_path = output_dir / f"{name}.yaml"
        with open(score_path, "w") as f:
            yaml.dump(score, f, default_flow_style=False, sort_keys=False, width=120)

        # Write agent card sidecar if A2A skills declared
        agent_card = technique_result.get("agent_card")
        if agent_card:
            card_path = output_dir / f"{name}.agent-card.yaml"
            with open(card_path, "w") as f:
                yaml.dump(agent_card, f, default_flow_style=False, sort_keys=False)

        _logger.info("Score written: %s", score_path)
        return score_path

    def seed_identity(
        self,
        agent_def: dict[str, Any],
        agents_dir: Path | None = None,
    ) -> Path:
        """Seed identity for a single agent.

        Convenience method that delegates to IdentitySeeder.
        """
        if agents_dir:
            seeder = IdentitySeeder(agents_dir)
        else:
            seeder = self.identity_seeder
        return seeder.seed(agent_def)

    def resolve_instruments(
        self,
        agent_def: dict[str, Any],
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve instruments for a single agent.

        Convenience method that delegates to InstrumentResolver.
        """
        return self.instrument_resolver.resolve(agent_def, defaults)

    def _build_prompt(
        self,
        agent_def: dict[str, Any],
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the prompt configuration for a score."""
        name = agent_def["name"]
        stakes = agent_def.get("meditation", defaults.get("stakes", ""))
        thinking_method = defaults.get("thinking_method", "")

        prompt: dict[str, Any] = {
            "variables": {
                "agent_name": name,
                "role": agent_def.get("role", "builder"),
                "focus": agent_def.get("focus", ""),
                "voice": agent_def.get("voice", ""),
                "agent_identity_dir": str(self.identity_seeder.agents_dir / name),
                "workspace": "{{workspace}}",
            },
        }

        if stakes:
            prompt["stakes"] = stakes
        if thinking_method:
            prompt["thinking_method"] = thinking_method

        return prompt

    def _assemble_score(
        self,
        *,
        name: str,
        workspace: str,
        sheet_config: dict[str, Any],
        prompt_config: dict[str, Any],
        instrument_result: dict[str, Any],
        validations: list[dict[str, Any]],
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the complete score dict."""
        concert_config = defaults.get("concert", {})
        max_depth = min(concert_config.get("max_chain_depth", 100), 100)
        pause_before = defaults.get("pause_before_chain", False)

        score: dict[str, Any] = {
            "name": name,
            "workspace": workspace,
            "backend": instrument_result.get("backend", {
                "type": "claude_cli",
                "skip_permissions": True,
                "timeout_seconds": 3600,
            }),
            "instrument_fallbacks": instrument_result.get("instrument_fallbacks", []),
            "sheet": sheet_config,
            "prompt": prompt_config,
            "retry": {
                "max_retries": 3,
                "base_delay_seconds": 30,
                "max_completion_attempts": 3,
                "completion_threshold_percent": 50,
            },
            "rate_limit": {
                "wait_minutes": 60,
                "max_waits": 24,
            },
            "stale_detection": {
                "enabled": True,
                "idle_timeout_seconds": 3600,
            },
            "parallel": {
                "enabled": True,
                "max_concurrent": 3,
            },
            "concert": {
                "enabled": True,
                "max_chain_depth": max_depth,
            },
            "validations": validations,
        }

        # Agent card for A2A — stored as sidecar metadata, not in the score
        # itself (JobConfig uses extra="forbid"). The conductor reads the
        # sidecar file on job start to register the agent card.
        # Stored in technique_result for callers that need it directly.

        # Self-chaining via on_success
        score_path_str = f"{{{{workspace}}}}/../{name}.yaml"
        score["on_success"] = [
            {
                "type": "run_job",
                "job_path": score_path_str,
                "detached": True,
                "fresh": True,
                "pause_before_chain": pause_before,
            }
        ]

        return score
