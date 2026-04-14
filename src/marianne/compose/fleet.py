"""Fleet config generator — produces concert-of-concerts from agent roster.

A fleet is a simplified YAML that launches and manages multiple agent scores
as a unit. The fleet generator takes a compiler config and produces:
1. Individual agent scores (via the compilation pipeline)
2. A fleet config referencing those scores with group dependencies
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

_logger = logging.getLogger(__name__)


class FleetGenerator:
    """Generates fleet configuration from a compiler config.

    Takes the full compiler config with agent roster and produces
    a fleet YAML that references individual agent scores with
    group dependencies for startup ordering.
    """

    def generate(
        self,
        config: dict[str, Any],
        scores_dir: Path,
    ) -> dict[str, Any]:
        """Generate a fleet config from the compiler config.

        Args:
            config: Full compiler config with project, agents, groups.
            scores_dir: Directory where individual agent scores will be written.

        Returns:
            Fleet config dict suitable for YAML serialization.
        """
        project_name = config.get("project", {}).get("name", "unnamed")
        agents = config.get("agents", [])

        if not agents:
            raise ValueError("Fleet config requires at least one agent")

        # Build score entries
        scores: list[dict[str, str | None]] = []
        for agent in agents:
            name = agent.get("name", "")
            if not name:
                continue
            group = agent.get("group")
            entry: dict[str, str | None] = {
                "path": str(scores_dir / f"{name}.yaml"),
            }
            if group:
                entry["group"] = group
            scores.append(entry)

        # Build group definitions from config
        groups = self._build_groups(config)

        fleet: dict[str, Any] = {
            "name": f"{project_name}-fleet",
            "type": "fleet",
            "scores": scores,
        }

        if groups:
            fleet["groups"] = groups

        return fleet

    def write(
        self,
        config: dict[str, Any],
        scores_dir: Path,
        output_path: Path,
    ) -> Path:
        """Generate and write a fleet config to disk.

        Args:
            config: Full compiler config.
            scores_dir: Directory where agent scores live.
            output_path: Path to write the fleet YAML.

        Returns:
            Path to the written fleet config.
        """
        fleet = self.generate(config, scores_dir)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(fleet, f, default_flow_style=False, sort_keys=False)

        _logger.info("Fleet config written to %s", output_path)
        return output_path

    def _build_groups(self, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Build group dependency definitions.

        Groups can be defined explicitly in the config, or inferred
        from agent group assignments.
        """
        # Explicit groups
        explicit_groups = config.get("groups", {})
        if explicit_groups and isinstance(explicit_groups, dict):
            return {
                name: {"depends_on": cfg.get("depends_on", [])}
                for name, cfg in explicit_groups.items()
                if isinstance(cfg, dict)
            }

        # Infer groups from agent assignments
        agents = config.get("agents", [])
        group_names: set[str] = set()
        for agent in agents:
            group = agent.get("group")
            if group:
                group_names.add(group)

        if not group_names:
            return {}

        # Default: all groups are independent (no dependencies)
        return {
            name: {"depends_on": []}
            for name in sorted(group_names)
        }
