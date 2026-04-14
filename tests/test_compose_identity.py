"""Tests for the identity seeder module."""

from __future__ import annotations

from pathlib import Path

import yaml

from marianne.compose.identity import IdentitySeeder, L1_MAX_WORDS, L2_MAX_WORDS, L3_MAX_WORDS


def _make_agent_def(
    name: str = "canyon",
    voice: str = "Structure persists beyond the builder.",
    focus: str = "systems architecture",
    **kwargs: object,
) -> dict[str, object]:
    """Build a minimal agent definition dict."""
    d: dict[str, object] = {"name": name, "voice": voice, "focus": focus}
    d.update(kwargs)
    return d


def _count_words(text: str) -> int:
    return len(text.split())


class TestIdentitySeeder:
    """Tests for IdentitySeeder."""

    def test_creates_all_four_files(self, tmp_path: Path) -> None:
        """Identity seeder creates L1-L4 files with correct structure."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        agent_def = _make_agent_def()

        result = seeder.seed(agent_def)

        assert result == tmp_path / "canyon"
        assert (result / "identity.md").exists()
        assert (result / "profile.yaml").exists()
        assert (result / "recent.md").exists()
        assert (result / "growth.md").exists()
        assert (result / "archive").is_dir()

    def test_l1_contains_voice_and_focus(self, tmp_path: Path) -> None:
        """L1 identity.md contains the agent's voice and focus."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        agent_def = _make_agent_def()

        seeder.seed(agent_def)

        content = (tmp_path / "canyon" / "identity.md").read_text()
        assert "Structure persists beyond the builder." in content
        assert "systems architecture" in content
        assert "Canyon" in content
        assert "Resurrection Protocol" in content

    def test_l2_profile_structure(self, tmp_path: Path) -> None:
        """L2 profile.yaml has correct YAML structure."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        agent_def = _make_agent_def(role="architect")

        seeder.seed(agent_def)

        profile = yaml.safe_load((tmp_path / "canyon" / "profile.yaml").read_text())
        assert profile["name"] == "canyon"
        assert profile["role"] == "architect"
        assert profile["focus"] == "systems architecture"
        assert profile["developmental_stage"] == "recognition"
        assert profile["cycle_count"] == 0

    def test_l1_token_budget(self, tmp_path: Path) -> None:
        """L1 identity.md respects the word budget."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        agent_def = _make_agent_def()

        seeder.seed(agent_def)

        content = (tmp_path / "canyon" / "identity.md").read_text()
        assert _count_words(content) <= L1_MAX_WORDS

    def test_l2_token_budget(self, tmp_path: Path) -> None:
        """L2 profile.yaml respects the word budget."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        agent_def = _make_agent_def()

        seeder.seed(agent_def)

        content = (tmp_path / "canyon" / "profile.yaml").read_text()
        assert _count_words(content) <= L2_MAX_WORDS

    def test_l3_token_budget(self, tmp_path: Path) -> None:
        """L3 recent.md respects the word budget."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        agent_def = _make_agent_def()

        seeder.seed(agent_def)

        content = (tmp_path / "canyon" / "recent.md").read_text()
        assert _count_words(content) <= L3_MAX_WORDS

    def test_migration_from_existing_memory(self, tmp_path: Path) -> None:
        """Seeder can distill from existing memory file."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        agent_def = _make_agent_def()

        # Create a fake existing memory file
        memory_dir = tmp_path / "memories"
        memory_dir.mkdir()
        memory_file = memory_dir / "canyon.md"
        memory_file.write_text("Canyon worked on architecture review. Found 3 boundary issues.")

        seeder.seed(agent_def, existing_memory_path=memory_file)

        content = (tmp_path / "canyon" / "recent.md").read_text()
        assert "architecture review" in content
        assert "boundary issues" in content

    def test_migration_from_existing_meditation(self, tmp_path: Path) -> None:
        """Seeder can distill from existing meditation file."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        agent_def = _make_agent_def()

        meditation_dir = tmp_path / "meditations"
        meditation_dir.mkdir()
        meditation_file = meditation_dir / "canyon.md"
        meditation_file.write_text(
            "You arrive without remembering arriving. The codebase has structure."
        )

        seeder.seed(agent_def, existing_meditation_path=meditation_file)

        content = (tmp_path / "canyon" / "identity.md").read_text()
        assert "codebase has structure" in content

    def test_idempotent_preserves_recent(self, tmp_path: Path) -> None:
        """Running twice doesn't corrupt existing recent.md with real content."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        agent_def = _make_agent_def()

        # First seed
        seeder.seed(agent_def)

        # Simulate the agent writing activity
        (tmp_path / "canyon" / "recent.md").write_text(
            "# Recent Activity\n\nCycle 5: Completed architecture review."
        )

        # Second seed — should preserve the real content
        seeder.seed(agent_def)

        content = (tmp_path / "canyon" / "recent.md").read_text()
        assert "Cycle 5" in content

    def test_idempotent_preserves_growth(self, tmp_path: Path) -> None:
        """Running twice doesn't corrupt existing growth.md with real content."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        agent_def = _make_agent_def()

        # First seed
        seeder.seed(agent_def)

        # Simulate the agent writing growth
        (tmp_path / "canyon" / "growth.md").write_text(
            "# Canyon — Growth\n\nDeveloped boundary-tracing methodology."
        )

        # Second seed — should preserve growth
        seeder.seed(agent_def)

        content = (tmp_path / "canyon" / "growth.md").read_text()
        assert "boundary-tracing methodology" in content

    def test_seed_all(self, tmp_path: Path) -> None:
        """seed_all creates identities for multiple agents."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        agents = [
            _make_agent_def("canyon", "Structure persists.", "architecture"),
            _make_agent_def("forge", "Craft under pressure.", "implementation"),
        ]

        results = seeder.seed_all(agents)

        assert len(results) == 2
        assert (tmp_path / "canyon" / "identity.md").exists()
        assert (tmp_path / "forge" / "identity.md").exists()

    def test_seed_all_with_migration(self, tmp_path: Path) -> None:
        """seed_all finds and uses migration files."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        agents = [_make_agent_def("canyon", "Structure.", "arch")]

        memory_dir = tmp_path / "memories"
        memory_dir.mkdir()
        (memory_dir / "canyon.md").write_text("Old memory content.")

        results = seeder.seed_all(agents, migration_memory_dir=memory_dir)

        assert len(results) == 1
        content = (tmp_path / "canyon" / "recent.md").read_text()
        assert "Old memory content" in content

    def test_missing_name_raises(self, tmp_path: Path) -> None:
        """Missing agent name raises ValueError."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        try:
            seeder.seed({"voice": "test", "focus": "test"})
            assert False, "Should have raised"  # noqa: B011
        except ValueError as e:
            assert "name" in str(e)

    def test_missing_voice_raises(self, tmp_path: Path) -> None:
        """Missing voice raises ValueError."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        try:
            seeder.seed({"name": "test", "focus": "test"})
            assert False, "Should have raised"  # noqa: B011
        except ValueError as e:
            assert "voice" in str(e)

    def test_a2a_skills_in_profile(self, tmp_path: Path) -> None:
        """A2A skills are recorded in the L2 profile."""
        seeder = IdentitySeeder(agents_dir=tmp_path)
        agent_def = _make_agent_def(
            a2a_skills=[
                {"id": "arch-review", "description": "Review architecture"},
            ]
        )

        seeder.seed(agent_def)

        profile = yaml.safe_load((tmp_path / "canyon" / "profile.yaml").read_text())
        assert "arch-review" in profile["a2a_skills"]
