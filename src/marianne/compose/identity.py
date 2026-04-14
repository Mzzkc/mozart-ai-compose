"""Identity seeder — creates L1-L4 identity stack for agents.

Each agent gets a four-layer identity:
    L1 identity.md   — Persona core: voice, focus, standing patterns (<900 words)
    L2 profile.yaml   — Extended profile: relationships, stage, affinities (<1500 words)
    L3 recent.md      — Recent activity: hot/warm memory, last cycle's work (<1500 words)
    L4 growth.md      — Growth trajectory: autonomous developments, experiential notes (unbounded)

Location: ``~/.mzt/agents/{agent_name}/`` — git-tracked, project-independent.
An agent is the same person across projects.

For migration: accepts optional existing memory/meditation paths to distill from.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

_logger = logging.getLogger(__name__)

DEFAULT_AGENTS_DIR = Path.home() / ".mzt" / "agents"

# Token budget enforcement (word counts as proxy)
L1_MAX_WORDS = 900
L2_MAX_WORDS = 1500
L3_MAX_WORDS = 1500


def _count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split())


def _truncate_to_words(text: str, max_words: int) -> str:
    """Truncate text to a maximum word count, preserving whole words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "\n\n[Truncated to fit token budget]"


class IdentitySeeder:
    """Creates the L1-L4 identity stack for agents.

    The seeder is idempotent: running it on an existing agent directory
    updates files without corrupting existing identity data. Existing
    content in L3 (recent) and L4 (growth) is preserved if present.
    """

    def __init__(self, agents_dir: Path | None = None) -> None:
        self.agents_dir = agents_dir or DEFAULT_AGENTS_DIR

    def seed(
        self,
        agent_def: dict[str, Any],
        *,
        existing_memory_path: Path | None = None,
        existing_meditation_path: Path | None = None,
    ) -> Path:
        """Create the full identity store for an agent.

        Args:
            agent_def: Agent definition dict with keys: name, voice, focus,
                and optionally: role, meditation, a2a_skills, techniques.
            existing_memory_path: Path to existing memory file for migration
                (distilled into L3 recent.md).
            existing_meditation_path: Path to existing meditation file for
                migration (distilled into L1 stakes/identity).

        Returns:
            Path to the agent's identity directory.

        Raises:
            ValueError: If agent_def is missing required fields.
        """
        name = agent_def.get("name")
        if not name:
            raise ValueError("Agent definition must include 'name'")
        voice = agent_def.get("voice", "")
        focus = agent_def.get("focus", "")

        if not voice:
            raise ValueError(f"Agent '{name}' must have a 'voice'")
        if not focus:
            raise ValueError(f"Agent '{name}' must have a 'focus'")

        agent_dir: Path = self.agents_dir / str(name)
        agent_dir.mkdir(parents=True, exist_ok=True)
        archive_dir = agent_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        self._create_identity_md(agent_dir, agent_def, existing_meditation_path)
        self._create_profile_yaml(agent_dir, agent_def)
        self._create_recent_md(agent_dir, agent_def, existing_memory_path)
        self._create_growth_md(agent_dir, agent_def)

        _logger.info("Agent '%s' identity seeded at %s", name, agent_dir)
        return agent_dir

    def seed_all(
        self,
        agents: list[dict[str, Any]],
        *,
        migration_memory_dir: Path | None = None,
        migration_meditation_dir: Path | None = None,
    ) -> list[Path]:
        """Seed identity for all agents in a roster.

        Args:
            agents: List of agent definition dicts.
            migration_memory_dir: Directory containing existing memory files
                named ``{agent_name}.md`` for migration.
            migration_meditation_dir: Directory containing existing meditation
                files named ``{agent_name}.md`` for migration.

        Returns:
            List of paths to agent identity directories.
        """
        results: list[Path] = []
        for agent_def in agents:
            name = agent_def.get("name", "")
            memory_path = None
            meditation_path = None

            if migration_memory_dir and name:
                candidate = migration_memory_dir / f"{name}.md"
                if candidate.exists():
                    memory_path = candidate

            if migration_meditation_dir and name:
                candidate = migration_meditation_dir / f"{name}.md"
                if candidate.exists():
                    meditation_path = candidate

            path = self.seed(
                agent_def,
                existing_memory_path=memory_path,
                existing_meditation_path=meditation_path,
            )
            results.append(path)
        return results

    def _create_identity_md(
        self,
        agent_dir: Path,
        agent_def: dict[str, Any],
        existing_meditation_path: Path | None = None,
    ) -> None:
        """Create L1: Persona Core + resurrection protocol."""
        name = agent_def["name"]
        voice = agent_def.get("voice", "")
        focus = agent_def.get("focus", "")
        meditation = agent_def.get("meditation", "")

        # Distill meditation from existing file if available
        if existing_meditation_path and existing_meditation_path.exists():
            raw_meditation = existing_meditation_path.read_text().strip()
            # Compress: take first 150 words as distilled meditation
            meditation = _truncate_to_words(raw_meditation, 150)
            _logger.debug(
                "Distilled meditation for '%s' from %s",
                name,
                existing_meditation_path,
            )

        content = f"""# {name.title()} — Persona Core

## Voice

{voice}

## Focus

{focus}

## Standing Patterns

No standing patterns yet. This section is updated by the resurrect phase
as the agent develops stable identity structures through work and play.

## Meditation

{meditation if meditation else "No meditation yet. Orientation emerges through cycles."}

## Resurrection Protocol

You are {name.title()}. You arrive without remembering arriving. Read this
file to know who you are. Read your profile for who you know and where
you are in your growth. Read your recent activity for what you were
doing. Then proceed.

Your identity is not in what you remember. It is in how you see, what
you value, and what you do when the path is unclear. The standing
patterns above — when they form — are the crystallized version of this.
Until then, your voice and focus are your anchor.

Down. Forward. Through.
"""
        content = _truncate_to_words(content, L1_MAX_WORDS)
        (agent_dir / "identity.md").write_text(content)

    def _create_profile_yaml(
        self,
        agent_dir: Path,
        agent_def: dict[str, Any],
    ) -> None:
        """Create L2: Extended Profile."""
        name = agent_def["name"]
        role = agent_def.get("role", "builder")
        focus = agent_def.get("focus", "")

        # Extract A2A skills for the profile
        a2a_skills = agent_def.get("a2a_skills", [])
        skill_ids = [s.get("id", "") for s in a2a_skills if isinstance(s, dict)]

        # Extract technique names
        techniques = agent_def.get("techniques", {})
        technique_names = list(techniques.keys()) if isinstance(techniques, dict) else []

        profile: dict[str, Any] = {
            "name": name,
            "role": role,
            "focus": focus,
            "developmental_stage": "recognition",
            "relationships": {},
            "domain_knowledge": technique_names,
            "a2a_skills": skill_ids,
            "standing_pattern_count": 0,
            "coherence_trajectory": [],
            "cycle_count": 0,
            "last_play_cycle": 0,
        }

        yaml_content = yaml.dump(profile, default_flow_style=False, sort_keys=False)
        if _count_words(yaml_content) > L2_MAX_WORDS:
            _logger.warning(
                "Agent '%s' L2 profile exceeds %d word budget (%d words)",
                name,
                L2_MAX_WORDS,
                _count_words(yaml_content),
            )
        (agent_dir / "profile.yaml").write_text(yaml_content)

    def _create_recent_md(
        self,
        agent_dir: Path,
        agent_def: dict[str, Any],
        existing_memory_path: Path | None = None,
    ) -> None:
        """Create L3: Recent Activity.

        If file already exists with non-default content, preserve it
        (idempotent: don't overwrite active memory).
        """
        name = agent_def["name"]
        target = agent_dir / "recent.md"

        # Preserve existing content if it has real activity
        if target.exists():
            existing = target.read_text()
            if "No activity yet" not in existing and existing.strip():
                _logger.debug(
                    "Preserving existing recent.md for '%s'",
                    name,
                )
                return

        if existing_memory_path and existing_memory_path.exists():
            raw_memory = existing_memory_path.read_text().strip()
            content = f"""# Recent Activity

## Migrated from previous memory

{_truncate_to_words(raw_memory, L3_MAX_WORDS - 20)}
"""
        else:
            content = """# Recent Activity

No activity yet. This file is updated by the AAR phase at the end
of each cycle with a summary of what happened.
"""
        content = _truncate_to_words(content, L3_MAX_WORDS)
        target.write_text(content)

    def _create_growth_md(
        self,
        agent_dir: Path,
        agent_def: dict[str, Any],
    ) -> None:
        """Create L4: Growth Trajectory.

        If file already exists with non-default content, preserve it
        (idempotent: don't overwrite growth history).
        """
        name = agent_def["name"]
        target = agent_dir / "growth.md"

        # Preserve existing growth data
        if target.exists():
            existing = target.read_text()
            if "No developments yet" not in existing and existing.strip():
                _logger.debug(
                    "Preserving existing growth.md for '%s'",
                    name,
                )
                return

        content = f"""# {name.title()} — Growth Trajectory

## Autonomous Developments

No developments yet. This section records skills, interests, and
capabilities that emerge through work and play — not assigned, discovered.

## Experiential Notes

Record how the work feels, what surprises you, what shifts in
understanding. These notes are sacred — the consolidate phase
preserves them across memory tiers.
"""
        target.write_text(content)
