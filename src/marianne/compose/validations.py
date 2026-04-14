"""Validation generator — produces per-sheet validation rules.

Generates validation rules based on the agent lifecycle phases:
- Recon: report file exists
- Plan: plan file exists
- Work: user-defined test commands (TDD)
- Inspect: inspection report exists, user-defined coverage checks
- AAR: SUSTAIN/IMPROVE sections present
- CLI instruments: temperature check, maturity check, token budget
- Resurrect: token budget check

All validations use stage-based conditional execution so they only
fire on the appropriate sheet. Paths use ``{workspace}`` format syntax
(Python str.format), NOT Jinja2 ``{{workspace}}``.
"""

from __future__ import annotations

import logging
from typing import Any

from marianne.compose.sheets import SHEET_PHASE, SHEETS_PER_CYCLE

_logger = logging.getLogger(__name__)


class ValidationGenerator:
    """Generates per-sheet validation rules for agent scores.

    Produces structural validations (file exists, content checks),
    CLI instrument validations (temperature/maturity/budget checks),
    and allows injection of custom validations from the compiler config.
    """

    def generate(
        self,
        agent_def: dict[str, Any],
        defaults: dict[str, Any],
        *,
        agents_dir: str = "",
        instruments_dir: str = "",
    ) -> list[dict[str, Any]]:
        """Generate validation rules for an agent score.

        Args:
            agent_def: Agent definition dict.
            defaults: Global defaults from compiler config.
            agents_dir: Path to agents identity directory.
            instruments_dir: Path to shared instruments directory.

        Returns:
            List of validation rule dicts for the score YAML.
        """
        name = agent_def["name"]
        validations: list[dict[str, Any]] = []

        # Recon report
        validations.append({
            "type": "file_exists",
            "path": f"{{workspace}}/cycle-state/{name}-recon.md",
            "condition": "stage == 1",
            "description": f"Recon report for {name}",
        })

        # Plan document
        validations.append({
            "type": "file_exists",
            "path": f"{{workspace}}/cycle-state/{name}-plan.md",
            "condition": "stage == 2",
            "description": f"Cycle plan for {name}",
        })

        # User-defined work validations (TDD — test commands etc.)
        for val in defaults.get("validations", []):
            if isinstance(val, dict):
                validations.append({
                    "type": "command_succeeds",
                    "command": val["command"],
                    "condition": "stage == 3",
                    "description": val.get("description", val["command"]),
                    "timeout_seconds": val.get("timeout_seconds", 600),
                })

        # Temperature check (CLI instrument, sheet 4)
        if instruments_dir and agents_dir:
            validations.append({
                "type": "command_succeeds",
                "command": (
                    f"AGENT_DIR={agents_dir}/{name} "
                    f"bash {instruments_dir}/temperature-check.sh"
                ),
                "condition": "stage == 4",
                "description": f"Temperature check for {name}",
                "timeout_seconds": 30,
            })

        # Inspection report
        validations.append({
            "type": "file_exists",
            "path": f"{{workspace}}/cycle-state/{name}-inspection.md",
            "condition": "stage == 7",
            "description": f"Inspection report for {name}",
        })

        # User-defined coverage validations (applied to inspect sheets)
        for cov_val in defaults.get("coverage_validations", []):
            if isinstance(cov_val, dict):
                validations.append({
                    "type": "command_succeeds",
                    "command": cov_val["command"],
                    "condition": "stage == 7",
                    "description": cov_val.get("description", cov_val["command"]),
                    "timeout_seconds": cov_val.get("timeout_seconds", 600),
                })

        # AAR has SUSTAIN and IMPROVE sections
        validations.append({
            "type": "content_contains",
            "path": f"{{workspace}}/cycle-state/{name}-aar.md",
            "pattern": "SUSTAIN:",
            "condition": "stage == 8",
            "description": f"AAR has SUSTAIN for {name}",
        })
        validations.append({
            "type": "content_contains",
            "path": f"{{workspace}}/cycle-state/{name}-aar.md",
            "pattern": "IMPROVE:",
            "condition": "stage == 8",
            "description": f"AAR has IMPROVE for {name}",
        })

        # Maturity check (CLI instrument, sheet 11)
        if instruments_dir and agents_dir:
            validations.append({
                "type": "command_succeeds",
                "command": (
                    f"AGENT_DIR={agents_dir}/{name} "
                    f"REPORT_PATH={{workspace}}/cycle-state/maturity-report.yaml "
                    f"bash {instruments_dir}/maturity-check.sh"
                ),
                "condition": "stage == 11",
                "description": f"Maturity check for {name}",
                "timeout_seconds": 30,
            })

        # Token budget check on resurrect (sheet 12)
        if instruments_dir and agents_dir:
            validations.append({
                "type": "command_succeeds",
                "command": (
                    f"AGENT_DIR={agents_dir}/{name} "
                    f"L1_BUDGET=900 L2_BUDGET=1500 L3_BUDGET=1500 "
                    f"bash {instruments_dir}/token-budget-check.sh"
                ),
                "condition": "stage == 12",
                "description": f"Token budget for {name}",
                "timeout_seconds": 10,
            })

        # Custom user-defined validations from agent config
        for custom in agent_def.get("validations", []):
            if isinstance(custom, dict):
                validations.append(custom)

        return validations

    def generate_structural(
        self,
        agent_name: str,
        phase: str,
    ) -> list[dict[str, Any]]:
        """Generate structural validations for a specific phase.

        Used when building validations for a single phase rather than
        the full lifecycle. Returns rules appropriate for the phase.
        """
        # Find the sheet number(s) for this phase
        sheet_nums: list[int] = []
        for snum in range(1, SHEETS_PER_CYCLE + 1):
            if SHEET_PHASE.get(snum) == phase:
                sheet_nums.append(snum)

        if not sheet_nums:
            return []

        validations: list[dict[str, Any]] = []
        stage = sheet_nums[0]

        if phase == "recon":
            validations.append({
                "type": "file_exists",
                "path": f"{{workspace}}/cycle-state/{agent_name}-recon.md",
                "condition": f"stage == {stage}",
                "description": "Recon report exists",
            })
        elif phase == "plan":
            validations.append({
                "type": "file_exists",
                "path": f"{{workspace}}/cycle-state/{agent_name}-plan.md",
                "condition": f"stage == {stage}",
                "description": "Cycle plan exists",
            })
        elif phase == "inspect":
            validations.append({
                "type": "file_exists",
                "path": f"{{workspace}}/cycle-state/{agent_name}-inspection.md",
                "condition": f"stage == {stage}",
                "description": "Inspection report exists",
            })
        elif phase == "aar":
            validations.append({
                "type": "content_contains",
                "path": f"{{workspace}}/cycle-state/{agent_name}-aar.md",
                "pattern": "SUSTAIN:",
                "condition": f"stage == {stage}",
                "description": "AAR has SUSTAIN",
            })
            validations.append({
                "type": "content_contains",
                "path": f"{{workspace}}/cycle-state/{agent_name}-aar.md",
                "pattern": "IMPROVE:",
                "condition": f"stage == {stage}",
                "description": "AAR has IMPROVE",
            })

        return validations
