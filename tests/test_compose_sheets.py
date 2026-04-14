"""Tests for the sheet composer module."""

from __future__ import annotations

from pathlib import Path

from marianne.compose.sheets import (
    CLI_SHEETS,
    PHASE_MAP,
    SHEET_DESCRIPTIONS,
    SHEET_PHASE,
    SHEETS_PER_CYCLE,
    SheetComposer,
)


def _make_agent_def(name: str = "canyon") -> dict[str, object]:
    return {
        "name": name,
        "voice": "Structure persists.",
        "focus": "architecture",
    }


def _make_defaults() -> dict[str, object]:
    return {
        "play_routing": {
            "memory_bloat_threshold": 3000,
            "stagnation_cycles": 3,
            "min_cycles_between_play": 5,
        },
    }


class TestSheetComposer:
    """Tests for SheetComposer."""

    def test_produces_12_sheets(self) -> None:
        """Sheet composer produces a 12-sheet cycle."""
        composer = SheetComposer()
        result = composer.compose(_make_agent_def(), _make_defaults())

        assert result["total_items"] == 12
        assert result["size"] == 1

    def test_has_fan_out(self) -> None:
        """Sheet config includes fan-out for phases 2 and 3."""
        composer = SheetComposer()
        result = composer.compose(_make_agent_def(), _make_defaults())

        fan_out = result["fan_out"]
        assert 5 in fan_out  # Phase 2 starts at sheet 5
        assert fan_out[5] == 3
        assert 8 in fan_out  # Phase 3 starts at sheet 8
        assert fan_out[8] == 3

    def test_has_dependencies(self) -> None:
        """Sheet config includes proper dependency DAG."""
        composer = SheetComposer()
        result = composer.compose(_make_agent_def(), _make_defaults())

        deps = result["dependencies"]
        # plan depends on recon
        assert deps[2] == [1]
        # work depends on plan
        assert deps[3] == [2]
        # phase 2 fan-out depends on temperature check
        assert deps[5] == [4]
        assert deps[6] == [4]
        assert deps[7] == [4]
        # phase 3 depends on all of phase 2
        assert set(deps[8]) == {5, 6, 7}
        # resurrect depends on maturity check
        assert deps[12] == [11]

    def test_has_prelude_with_identity(self) -> None:
        """Prelude includes the agent's L1 identity."""
        composer = SheetComposer()
        result = composer.compose(
            _make_agent_def(), _make_defaults(),
            agents_dir=Path("/test/agents"),
        )

        prelude = result["prelude"]
        assert any("identity.md" in p.get("file", "") for p in prelude)

    def test_has_cadenzas_per_phase(self) -> None:
        """Cadenzas are defined for relevant phases."""
        composer = SheetComposer()
        result = composer.compose(_make_agent_def(), _make_defaults())

        cadenzas = result["cadenzas"]
        # Recon should have profile + recent
        assert 1 in cadenzas
        assert len(cadenzas[1]) >= 2
        # Resurrect should have full identity load
        assert 12 in cadenzas
        assert len(cadenzas[12]) >= 3

    def test_descriptions_for_all_sheets(self) -> None:
        """All 12 sheets have descriptions."""
        composer = SheetComposer()
        result = composer.compose(_make_agent_def(), _make_defaults())

        descriptions = result["descriptions"]
        for i in range(1, SHEETS_PER_CYCLE + 1):
            assert i in descriptions, f"Sheet {i} missing description"

    def test_skip_when_for_play(self) -> None:
        """Play sheet has skip_when_command gating."""
        composer = SheetComposer()
        result = composer.compose(_make_agent_def(), _make_defaults())

        skip = result.get("skip_when_command", {})
        assert 6 in skip  # Play sheet is gated

    def test_play_gate_only_affects_play(self) -> None:
        """Temperature check gates only Play, not Integration or Inspect."""
        composer = SheetComposer()
        result = composer.compose(_make_agent_def(), _make_defaults())

        skip = result.get("skip_when_command", {})
        # Play (sheet 6) is gated
        assert 6 in skip
        # Integration (sheet 5) and Inspect (sheet 7) are NOT gated
        assert 5 not in skip
        assert 7 not in skip

    def test_phase_map_coverage(self) -> None:
        """Phase map covers all 12 sheets."""
        all_sheets: set[int] = set()
        for sheets in PHASE_MAP.values():
            all_sheets.update(sheets)
        assert all_sheets == set(range(1, 13))

    def test_sheet_phase_reverse_map(self) -> None:
        """Sheet-to-phase reverse map covers all sheets."""
        for i in range(1, 13):
            assert i in SHEET_PHASE

    def test_cli_sheets(self) -> None:
        """CLI sheets are correctly identified."""
        assert 4 in CLI_SHEETS   # Temperature check
        assert 11 in CLI_SHEETS  # Maturity check
        assert 3 not in CLI_SHEETS  # Work is not CLI

    def test_sheet_descriptions_constant(self) -> None:
        """SHEET_DESCRIPTIONS covers all 12 sheets."""
        assert len(SHEET_DESCRIPTIONS) == 12
        for i in range(1, 13):
            assert i in SHEET_DESCRIPTIONS

    def test_get_phase_for_sheet(self) -> None:
        """get_phase_for_sheet returns correct phase names."""
        composer = SheetComposer()
        assert composer.get_phase_for_sheet(1) == "recon"
        assert composer.get_phase_for_sheet(3) == "work"
        assert composer.get_phase_for_sheet(6) == "play"
        assert composer.get_phase_for_sheet(12) == "resurrect"

    def test_is_cli_sheet(self) -> None:
        """is_cli_sheet correctly identifies CLI instrument sheets."""
        composer = SheetComposer()
        assert composer.is_cli_sheet(4)
        assert composer.is_cli_sheet(11)
        assert not composer.is_cli_sheet(3)
        assert not composer.is_cli_sheet(12)

    def test_no_play_routing_still_works(self) -> None:
        """Composer works without play routing config."""
        composer = SheetComposer()
        result = composer.compose(_make_agent_def(), {})

        assert result["total_items"] == 12
        # No skip_when_command when no play routing
        assert result.get("skip_when_command", {}) == {}
