"""Tests for the pattern expander module."""

from __future__ import annotations

from pathlib import Path

import pytest

from marianne.compose.patterns import (
    BUILTIN_PATTERNS,
    PatternExpander,
    PatternStage,
)


def _make_agent_def(name: str = "canyon") -> dict[str, object]:
    return {
        "name": name,
        "voice": "Structure persists.",
        "focus": "architecture",
    }


# ---------------------------------------------------------------------------
# Rosetta corpus location (may not exist in CI)
# ---------------------------------------------------------------------------
ROSETTA_CORPUS_DIR = Path(__file__).parent.parent / "scores" / "rosetta-corpus" / "patterns"


class TestPatternExpander:
    """Tests for PatternExpander prompt-extension mode."""

    def test_has_builtin_patterns(self) -> None:
        """Built-in patterns are loaded by default."""
        expander = PatternExpander()
        assert len(expander.patterns) >= 5
        assert "cathedral-construction" in expander.patterns
        assert "composting-cascade" in expander.patterns
        assert "soil-maturity-index" in expander.patterns
        assert "boundary-trace" in expander.patterns
        assert "forge-cycle" in expander.patterns

    def test_has_agent_cycle_patterns(self) -> None:
        """Agent cycle patterns are loaded as builtins."""
        expander = PatternExpander()
        assert "fan-out-synthesis" in expander.patterns
        assert "the-tool-chain" in expander.patterns
        assert "reconnaissance-pull" in expander.patterns

    def test_expand_single_pattern(self) -> None:
        """Expanding a single pattern returns prompt extensions."""
        expander = PatternExpander()
        result = expander.expand(["cathedral-construction"], _make_agent_def())

        assert "prompt_extensions" in result
        assert "applied_patterns" in result
        assert "cathedral-construction" in result["applied_patterns"]
        assert len(result["prompt_extensions"]) > 0

    def test_expand_multiple_patterns(self) -> None:
        """Multiple patterns merge their extensions."""
        expander = PatternExpander()
        result = expander.expand(["cathedral-construction", "boundary-trace"], _make_agent_def())

        assert len(result["applied_patterns"]) == 2
        # Both patterns contribute to inspect phase
        extensions = result["prompt_extensions"]
        assert "inspect" in extensions

    def test_unknown_pattern_raises(self) -> None:
        """Unknown patterns raise ValueError with a clear message."""
        expander = PatternExpander()
        with pytest.raises(ValueError, match="nonexistent-pattern"):
            expander.expand(["nonexistent-pattern"], _make_agent_def())

    def test_custom_patterns(self) -> None:
        """Custom patterns can be registered."""
        custom = {
            "my-pattern": {
                "description": "Test pattern",
                "phases": ["test"],
                "sheet_modifiers": {
                    "work": {"prompt_extension": "Do the custom thing."},
                },
            },
        }
        expander = PatternExpander(custom_patterns=custom)

        assert "my-pattern" in expander.patterns
        result = expander.expand(["my-pattern"], _make_agent_def())
        assert "my-pattern" in result["applied_patterns"]
        assert "work" in result["prompt_extensions"]

    def test_list_patterns(self) -> None:
        """list_patterns returns all available patterns."""
        expander = PatternExpander()
        patterns = expander.list_patterns()

        assert len(patterns) >= 8  # 5 original builtins + 3 agent cycle
        names = [p["name"] for p in patterns]
        assert "cathedral-construction" in names
        assert "fan-out-synthesis" in names

        for p in patterns:
            assert "name" in p
            assert "description" in p

    def test_get_pattern(self) -> None:
        """get_pattern returns pattern dict or None."""
        expander = PatternExpander()

        pattern = expander.get_pattern("cathedral-construction")
        assert pattern is not None
        assert "description" in pattern
        assert "sheet_modifiers" in pattern

        assert expander.get_pattern("nonexistent") is None

    def test_builtin_patterns_have_descriptions(self) -> None:
        """All built-in patterns have descriptions."""
        for name, pattern in BUILTIN_PATTERNS.items():
            assert "description" in pattern, f"Pattern '{name}' missing description"
            assert pattern["description"], f"Pattern '{name}' has empty description"

    def test_builtin_patterns_have_sheet_modifiers(self) -> None:
        """All built-in patterns have sheet_modifiers."""
        for name, pattern in BUILTIN_PATTERNS.items():
            assert "sheet_modifiers" in pattern, f"Pattern '{name}' missing modifiers"
            assert pattern["sheet_modifiers"], f"Pattern '{name}' has empty modifiers"

    def test_empty_pattern_list(self) -> None:
        """Empty pattern list returns empty results."""
        expander = PatternExpander()
        result = expander.expand([], _make_agent_def())

        assert result["applied_patterns"] == []
        assert result["prompt_extensions"] == {}


class TestPatternStageExpansion:
    """Tests for expand_stages — stage-level pattern expansion."""

    def test_recon_pull_expands_to_3_stages(self) -> None:
        """reconnaissance-pull produces 3 stages: recon, plan, execute."""
        expander = PatternExpander()
        stages = expander.expand_stages("reconnaissance-pull")

        assert len(stages) == 3
        assert stages[0].name == "recon"
        assert stages[1].name == "plan"
        assert stages[2].name == "execute"

    def test_fan_out_synthesis_expands_to_3_stages(self) -> None:
        """fan-out-synthesis produces 3 stages: prepare, analyze, synthesize."""
        expander = PatternExpander()
        stages = expander.expand_stages("fan-out-synthesis")

        assert len(stages) == 3
        assert stages[0].name == "prepare"
        assert stages[1].name == "analyze"
        assert stages[2].name == "synthesize"

    def test_tool_chain_expands_to_5_stages(self) -> None:
        """the-tool-chain produces 5 stages: plan, fetch, clean, analyze, interpret."""
        expander = PatternExpander()
        stages = expander.expand_stages("the-tool-chain")

        assert len(stages) == 5
        assert stages[0].name == "plan"
        assert stages[1].name == "fetch"
        assert stages[2].name == "clean"
        assert stages[3].name == "analyze"
        assert stages[4].name == "interpret"

    def test_cathedral_construction_expands_to_3_stages(self) -> None:
        """cathedral-construction produces 3 stages."""
        expander = PatternExpander()
        stages = expander.expand_stages("cathedral-construction")

        assert len(stages) == 3
        assert stages[0].name == "plan-iteration"
        assert stages[1].name == "build"
        assert stages[2].name == "inspect"

    def test_unknown_pattern_raises_in_expand_stages(self) -> None:
        """expand_stages raises ValueError for unknown patterns."""
        expander = PatternExpander()
        with pytest.raises(ValueError, match="nonexistent"):
            expander.expand_stages("nonexistent")

    def test_stages_have_purpose(self) -> None:
        """Each expanded stage has a non-empty purpose."""
        expander = PatternExpander()
        stages = expander.expand_stages("reconnaissance-pull")

        for stage in stages:
            assert stage.purpose, f"Stage '{stage.name}' missing purpose"

    def test_stages_have_instrument_guidance(self) -> None:
        """Each expanded stage has instrument guidance."""
        expander = PatternExpander()
        stages = expander.expand_stages("the-tool-chain")

        for stage in stages:
            assert stage.instrument_guidance, f"Stage '{stage.name}' missing instrument_guidance"

    def test_fan_out_stage_has_parameterized_sheets(self) -> None:
        """Fan-out stages report sheets as a fan_out string."""
        expander = PatternExpander()
        stages = expander.expand_stages("fan-out-synthesis")

        analyze_stage = stages[1]
        assert analyze_stage.name == "analyze"
        assert analyze_stage.sheets == "fan_out(6)"

    def test_parameters_customize_fan_out(self) -> None:
        """Pattern parameters can customize fan-out width."""
        expander = PatternExpander()
        stages = expander.expand_stages("fan-out-synthesis", params={"fan_out": {"analyze": 3}})

        analyze_stage = stages[1]
        assert analyze_stage.sheets == "fan_out(3)"

    def test_parameters_default_without_override(self) -> None:
        """Calling with no params uses pattern defaults."""
        expander = PatternExpander()
        stages = expander.expand_stages("fan-out-synthesis")
        # Default fan_out is 6 for analyze
        assert stages[1].sheets == "fan_out(6)"

    def test_stages_have_fallback_friendly(self) -> None:
        """Each stage reports fallback_friendly status."""
        expander = PatternExpander()
        stages = expander.expand_stages("the-tool-chain")

        # plan is not fallback-friendly, fetch/clean/analyze are
        assert stages[0].fallback_friendly is False
        assert stages[1].fallback_friendly is True

    def test_stages_have_artifacts(self) -> None:
        """Stages carry artifact declarations."""
        expander = PatternExpander()
        stages = expander.expand_stages("reconnaissance-pull")

        recon = stages[0]
        assert "recon-report.md" in recon.artifacts

    def test_pattern_stage_is_frozen(self) -> None:
        """PatternStage instances are immutable."""
        stage = PatternStage(
            name="test",
            sheets=1,
            purpose="test purpose",
            instrument_guidance="any",
            fallback_friendly=True,
            artifacts=("file.md",),
        )
        with pytest.raises(AttributeError):
            stage.name = "changed"  # type: ignore[misc]


class TestExpandStagesWithValidations:
    """Tests for expand_stages_with_validations — stage + validation shapes."""

    def test_returns_stages_and_validations(self) -> None:
        """Result has both stages and validations keys."""
        expander = PatternExpander()
        result = expander.expand_stages_with_validations("reconnaissance-pull")

        assert "stages" in result
        assert "validations" in result
        assert len(result["stages"]) == 3

    def test_artifact_generates_file_exists_validation(self) -> None:
        """Stages with artifacts produce file_exists validation shapes."""
        expander = PatternExpander()
        result = expander.expand_stages_with_validations("reconnaissance-pull")

        validations = result["validations"]
        file_exists_vals = [v for v in validations if v["type"] == "file_exists"]
        assert len(file_exists_vals) > 0
        # recon-report.md artifact should produce a validation
        paths = [v["path"] for v in file_exists_vals]
        assert any("recon-report.md" in p for p in paths)

    def test_fan_out_artifact_uses_instance_template(self) -> None:
        """Fan-out artifacts keep template variables."""
        expander = PatternExpander()
        result = expander.expand_stages_with_validations("fan-out-synthesis")

        validations = result["validations"]
        paths = [v.get("path", "") for v in validations]
        # analyze stage artifact includes {{ instance_id }}
        assert any("instance_id" in p or "scope.md" in p for p in paths)


class TestRosettaCorpusLoading:
    """Tests for loading patterns from Rosetta corpus markdown files."""

    @pytest.fixture()
    def corpus_dir(self) -> Path:
        if not ROSETTA_CORPUS_DIR.exists():
            pytest.skip("Rosetta corpus not available")
        return ROSETTA_CORPUS_DIR

    def test_load_corpus_produces_patterns(self, corpus_dir: Path) -> None:
        """Loading Rosetta corpus yields patterns."""
        expander = PatternExpander()
        loaded = expander.load_rosetta_corpus(corpus_dir)
        assert len(loaded) > 0

    def test_loaded_patterns_have_stages(self, corpus_dir: Path) -> None:
        """Patterns from corpus include stage definitions."""
        expander = PatternExpander()
        loaded = expander.load_rosetta_corpus(corpus_dir)

        with_stages = [name for name, p in loaded.items() if p.get("stages")]
        assert len(with_stages) > 0

    def test_loaded_patterns_merge_into_expander(self, corpus_dir: Path) -> None:
        """After loading, corpus patterns are available via get_pattern."""
        expander = PatternExpander()
        loaded = expander.load_rosetta_corpus(corpus_dir)

        if loaded:
            first_name = next(iter(loaded))
            assert expander.get_pattern(first_name) is not None

    def test_loaded_patterns_have_descriptions(self, corpus_dir: Path) -> None:
        """Corpus patterns have problem descriptions."""
        expander = PatternExpander()
        loaded = expander.load_rosetta_corpus(corpus_dir)

        for name, pattern in loaded.items():
            assert pattern.get("description"), f"Corpus pattern '{name}' missing description"
