"""Sheet composer — produces the sheet structure for agent scores.

Generates a 12-sheet cycle structure with parallel fan-out phases:

    Phase 1   (sequential):   Recon -> Plan -> Work          (sheets 1-3)
    Phase 1.5 (CLI):          Temperature check (gates Play)  (sheet 4)
    Phase 2   (fan-out of 3): Integration || Play || Inspect  (sheets 5-7)
    Phase 3   (fan-out of 3): AAR || Consolidate || Reflect   (sheets 8-10)
    Phase 3.5 (CLI):          Maturity check                  (sheet 11)
    Phase 4   (sequential):   Resurrect                       (sheet 12)

Total: 12 sheets per cycle.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

SHEETS_PER_CYCLE = 12

# Phase name to sheet number(s) mapping
PHASE_MAP: dict[str, list[int]] = {
    "recon": [1],
    "plan": [2],
    "work": [3],
    "temperature_check": [4],
    "integration": [5],
    "play": [6],
    "inspect": [7],
    "aar": [8],
    "consolidate": [9],
    "reflect": [10],
    "maturity_check": [11],
    "resurrect": [12],
}

# Reverse: sheet number to phase name
SHEET_PHASE: dict[int, str] = {}
for _phase, _sheets in PHASE_MAP.items():
    for _s in _sheets:
        SHEET_PHASE[_s] = _phase

# CLI instrument sheets (not LLM calls)
CLI_SHEETS = {4, 11}

# Fan-out configuration
FANOUT_PHASE_2 = {5: "integration", 6: "play", 7: "inspect"}
FANOUT_PHASE_3 = {8: "aar", 9: "consolidate", 10: "reflect"}

# Sheet descriptions
SHEET_DESCRIPTIONS: dict[int, str] = {
    1: "Recon — survey project state",
    2: "Plan — write cycle plan",
    3: "Work — execute against plan",
    4: "Temperature check — gate play phase",
    5: "Integration — merge and coordinate",
    6: "Play — explore and experiment",
    7: "Inspect — quality and security review",
    8: "AAR — after-action review",
    9: "Consolidate — memory management",
    10: "Reflect — growth and insight",
    11: "Maturity check — developmental measurement",
    12: "Resurrect — identity refresh for next cycle",
}


class SheetComposer:
    """Composes the sheet structure for an agent score.

    Takes agent definitions and default config, produces the sheet section
    of a Mozart score YAML including fan-out, dependencies, cadenzas,
    and instrument assignments.
    """

    def __init__(self, templates_dir: Path | None = None) -> None:
        self.templates_dir = templates_dir

    def compose(
        self,
        agent_def: dict[str, Any],
        defaults: dict[str, Any],
        *,
        agents_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Compose the sheet configuration for an agent score.

        Args:
            agent_def: Agent definition dict.
            defaults: Global defaults from the compiler config.
            agents_dir: Path to agents identity directory.

        Returns:
            Dict representing the ``sheet:`` section of a Mozart score.
        """
        name = agent_def["name"]
        agents_dir_path = agents_dir or Path.home() / ".mzt" / "agents"
        identity_dir = str(agents_dir_path / name)

        sheet_config: dict[str, Any] = {
            "size": 1,
            "total_items": SHEETS_PER_CYCLE,
            "descriptions": dict(SHEET_DESCRIPTIONS),
            "prelude": self._build_prelude(identity_dir, defaults),
            "cadenzas": self._build_cadenzas(identity_dir, defaults),
            "fan_out": {
                # Phase 2: 3 parallel instances (integration, play, inspect)
                5: 3,
                # Phase 3: 3 parallel instances (aar, consolidate, reflect)
                8: 3,
            },
            "dependencies": self._build_dependencies(),
        }

        # Add skip_when_command for play gating
        skip_when = self._build_skip_when(agent_def, defaults, identity_dir)
        if skip_when:
            sheet_config["skip_when_command"] = skip_when

        return sheet_config

    def _build_prelude(
        self,
        identity_dir: str,
        defaults: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Build prelude injections (loaded for ALL sheets)."""
        prelude: list[dict[str, str]] = []

        # Agent L1 identity is always in prelude
        prelude.append({"file": f"{identity_dir}/identity.md", "as": "context"})

        # Add any global prelude files from defaults
        for item in defaults.get("prelude", []):
            if isinstance(item, dict):
                prelude.append(item)

        return prelude

    def _build_cadenzas(
        self,
        identity_dir: str,
        defaults: dict[str, Any],
    ) -> dict[int, list[dict[str, str]]]:
        """Build per-sheet cadenza injections."""
        profile = {"file": f"{identity_dir}/profile.yaml", "as": "context"}
        recent = {"file": f"{identity_dir}/recent.md", "as": "context"}
        growth = {"file": f"{identity_dir}/growth.md", "as": "context"}

        cadenzas: dict[int, list[dict[str, str]]] = {
            # Recon: profile + recent for orientation
            1: [profile, recent],
            # Plan: recent context
            2: [recent],
            # Work: recent for continuity
            3: [recent],
            # Play: growth for creative exploration
            6: [growth],
            # Inspect: profile for domain awareness
            7: [profile, recent],
            # AAR: recent for review
            8: [recent],
            # Consolidate: profile + recent for memory management
            9: [profile, recent],
            # Reflect: profile + growth for insight
            10: [profile, growth],
            # Resurrect: full identity load
            12: [profile, recent, growth],
        }

        # Add shared cadenza directories
        cadenza_config = defaults.get("cadenzas", {})
        active_dirs = cadenza_config.get("active", [])
        for item in active_dirs:
            if isinstance(item, dict):
                phases = item.get("phases", [])
                for sheet_num in range(1, SHEETS_PER_CYCLE + 1):
                    phase = SHEET_PHASE.get(sheet_num, "")
                    if phase in phases or "all" in phases:
                        if sheet_num not in cadenzas:
                            cadenzas[sheet_num] = []
                        cadenzas[sheet_num].append({
                            "directory": item["directory"],
                            "as": item.get("as", "context"),
                        })

        return cadenzas

    def _build_dependencies(self) -> dict[int, list[int]]:
        """Build sheet dependency DAG.

        Sequential phases have linear dependencies.
        Fan-out phases depend on the previous sequential sheet.
        Post-fan-out phases depend on all fan-out instances.
        """
        return {
            2: [1],       # plan depends on recon
            3: [2],       # work depends on plan
            4: [3],       # temperature check depends on work
            5: [4],       # phase 2 fan-out depends on temperature check
            6: [4],
            7: [4],
            8: [5, 6, 7],  # phase 3 fan-out depends on all of phase 2
            9: [5, 6, 7],
            10: [5, 6, 7],
            11: [8, 9, 10],  # maturity check depends on all of phase 3
            12: [11],        # resurrect depends on maturity check
        }

    def _build_skip_when(
        self,
        agent_def: dict[str, Any],
        defaults: dict[str, Any],
        identity_dir: str,
    ) -> dict[int, dict[str, Any]]:
        """Build skip_when_command for conditional sheet execution.

        The temperature check (sheet 4) gates the play sheet (sheet 6).
        """
        play_routing = defaults.get("play_routing", {})
        if not play_routing:
            return {}

        name = agent_def["name"]
        memory_bloat = play_routing.get("memory_bloat_threshold", 3000)
        stagnation = play_routing.get("stagnation_cycles", 3)
        min_cycles = play_routing.get("min_cycles_between_play", 5)

        # Play is gated: skip play (sheet 6) if temperature check says work
        temp_cmd = (
            f"AGENT_DIR={identity_dir} AGENT_NAME={name} "
            f"MEMORY_BLOAT_THRESHOLD={memory_bloat} "
            f"STAGNATION_CYCLES={stagnation} "
            f"MIN_CYCLES_BETWEEN_PLAY={min_cycles} "
            f"test -f {{{{workspace}}}}/cycle-state/temperature-play"
        )

        return {
            6: {
                "command": temp_cmd,
                "description": "Skip play if temperature check says work",
                "timeout_seconds": 30,
            },
        }

    def get_phase_for_sheet(self, sheet_num: int) -> str:
        """Return the phase name for a given sheet number."""
        return SHEET_PHASE.get(sheet_num, "unknown")

    def get_sheets_for_phase(self, phase: str) -> list[int]:
        """Return sheet numbers for a given phase name."""
        return PHASE_MAP.get(phase, [])

    def is_cli_sheet(self, sheet_num: int) -> bool:
        """Return True if the sheet is a CLI instrument (not an LLM call)."""
        return sheet_num in CLI_SHEETS
