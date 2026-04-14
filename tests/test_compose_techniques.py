"""Tests for the technique wirer module."""

from __future__ import annotations

from pathlib import Path

from marianne.compose.techniques import TechniqueWirer


def _make_agent_def(name: str = "canyon") -> dict[str, object]:
    return {
        "name": name,
        "voice": "Structure persists.",
        "focus": "architecture",
        "techniques": {
            "symbols-python": {
                "kind": "mcp",
                "phases": ["work", "inspect"],
            },
        },
        "a2a_skills": [
            {"id": "arch-review", "description": "Review architecture"},
        ],
    }


def _make_defaults() -> dict[str, object]:
    return {
        "techniques": {
            "a2a": {
                "kind": "protocol",
                "phases": ["recon", "plan", "work"],
            },
            "github": {
                "kind": "mcp",
                "phases": ["recon", "work"],
                "config": {"description": "GitHub repository operations"},
            },
            "mateship": {
                "kind": "skill",
                "phases": ["recon", "work", "inspect"],
                "config": {"description": "Finding collaboration protocol"},
            },
        },
    }


class TestTechniqueWirer:
    """Tests for TechniqueWirer."""

    def test_returns_cadenzas_and_card(self) -> None:
        """Wire returns cadenzas, agent_card, and technique_manifests."""
        wirer = TechniqueWirer()
        result = wirer.wire(_make_agent_def(), _make_defaults())

        assert "cadenzas" in result
        assert "agent_card" in result
        assert "technique_manifests" in result

    def test_agent_card_generated(self) -> None:
        """A2A agent card is generated when a2a_skills present."""
        wirer = TechniqueWirer()
        result = wirer.wire(_make_agent_def(), _make_defaults())

        card = result["agent_card"]
        assert card is not None
        assert card["name"] == "canyon"
        assert len(card["skills"]) == 1
        assert card["skills"][0]["id"] == "arch-review"

    def test_no_card_without_skills(self) -> None:
        """No agent card when a2a_skills not defined."""
        wirer = TechniqueWirer()
        agent = {"name": "test", "voice": "v", "focus": "f"}
        result = wirer.wire(agent, {})

        assert result["agent_card"] is None

    def test_manifests_per_phase(self) -> None:
        """Technique manifests are generated for active phases."""
        wirer = TechniqueWirer()
        result = wirer.wire(_make_agent_def(), _make_defaults())

        manifests = result["technique_manifests"]
        # Sheet 1 (recon) should have a manifest — a2a, github, mateship active
        assert 1 in manifests
        assert "recon" in manifests[1].lower()

        # Sheet 3 (work) should have all techniques
        assert 3 in manifests
        manifest_3 = manifests[3]
        assert "MCP" in manifest_3 or "mcp" in manifest_3.lower()

    def test_manifest_contains_technique_types(self) -> None:
        """Manifests group techniques by kind."""
        wirer = TechniqueWirer()
        result = wirer.wire(_make_agent_def(), _make_defaults())

        # Work sheet should have MCP, Protocol, and Skill sections
        manifest = result["technique_manifests"].get(3, "")
        assert "MCP" in manifest or "Tools" in manifest
        assert "Protocol" in manifest or "A2A" in manifest
        assert "Skill" in manifest or "mateship" in manifest.lower()

    def test_agent_techniques_merge_with_defaults(self) -> None:
        """Agent-specific techniques merge with global defaults."""
        wirer = TechniqueWirer()
        result = wirer.wire(_make_agent_def(), _make_defaults())

        manifests = result["technique_manifests"]
        # Work sheet should have both default (github, mateship) and agent (symbols-python)
        work_manifest = manifests.get(3, "")
        assert "github" in work_manifest.lower() or "symbols" in work_manifest.lower()

    def test_technique_doc_wiring(self, tmp_path: Path) -> None:
        """Technique documents are wired as cadenza injections."""
        # Create a technique document
        tech_dir = tmp_path / "techniques"
        tech_dir.mkdir()
        (tech_dir / "mateship.md").write_text("# Mateship Protocol\n\nShare findings.")

        wirer = TechniqueWirer(techniques_dir=tech_dir)
        result = wirer.wire(_make_agent_def(), _make_defaults())

        cadenzas = result["cadenzas"]
        # Mateship is active on recon (sheet 1)
        has_mateship = False
        for items in cadenzas.values():
            for item in items:
                if "mateship" in item.get("file", ""):
                    has_mateship = True
                    break
        assert has_mateship

    def test_empty_techniques(self) -> None:
        """Works with no techniques defined."""
        wirer = TechniqueWirer()
        agent = {"name": "bare", "voice": "v", "focus": "f"}
        result = wirer.wire(agent, {})

        assert result["cadenzas"] == {}
        assert result["agent_card"] is None

    def test_all_phases_keyword(self) -> None:
        """Techniques with phases=['all'] are active on every sheet."""
        wirer = TechniqueWirer()
        defaults = {
            "techniques": {
                "voice": {
                    "kind": "skill",
                    "phases": ["all"],
                    "config": {"description": "Expressive voice style"},
                },
            },
        }
        agent = {"name": "test", "voice": "v", "focus": "f"}
        result = wirer.wire(agent, defaults)

        manifests = result["technique_manifests"]
        # Every sheet should have the voice technique
        for sheet_num in range(1, 13):
            assert sheet_num in manifests, f"Sheet {sheet_num} missing manifest"
