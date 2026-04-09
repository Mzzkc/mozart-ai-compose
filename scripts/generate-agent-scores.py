#!/usr/bin/env python3
"""Generate one self-chaining Mozart score per agent.

Usage:
    python scripts/generate-agent-scores.py config.yaml -o scores/my-project/
    python scripts/generate-agent-scores.py config.yaml --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).parent
TEMPLATES_DIR = SCRIPT_DIR / "templates"
INSTRUMENTS_DIR = SCRIPT_DIR / "instruments"
SHEETS_PER_CYCLE = 13
DEFAULT_AGENTS_DIR = Path.home() / ".mzt" / "agents"


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    for field in ["name", "workspace", "agents"]:
        if field not in config:
            print(f"Error: missing required field '{field}'", file=sys.stderr)
            sys.exit(1)
    if not config["agents"]:
        print("Error: at least one agent required", file=sys.stderr)
        sys.exit(1)
    config.setdefault("spec_dir", "")
    config.setdefault("playspace", "")
    config.setdefault("validations", [])
    config.setdefault("pre_commit_commands", [])
    config.setdefault("backend", {"type": "claude_cli", "skip_permissions": True, "timeout_seconds": 3600})
    config.setdefault("prelude", [])
    config.setdefault("play_routing", {"memory_bloat_threshold": 3000, "stagnation_cycles": 3, "min_cycles_between_play": 5})
    config.setdefault("concert", {"max_chain_depth": 1000})
    config.setdefault("agents_dir", str(DEFAULT_AGENTS_DIR))
    config.setdefault("instrument_fallbacks", ["goose"])
    config.setdefault("instruments", {})
    return config


def _build_cadenzas(agent_identity_dir: str, config: dict[str, Any]) -> dict[int, list[dict[str, str]]]:
    cadenzas: dict[int, list[dict[str, str]]] = {}
    profile = {"file": f"{agent_identity_dir}/profile.yaml", "as": "context"}
    recent = {"file": f"{agent_identity_dir}/recent.md", "as": "context"}
    cadenzas[1] = [profile, recent]
    cadenzas[2] = [recent]
    if config.get("spec_dir"):
        spec_dir = Path(config["spec_dir"])
        if spec_dir.exists():
            for spec_file in sorted(spec_dir.glob("*")):
                if spec_file.is_file():
                    cadenzas[2].append({"file": str(spec_file), "as": "context"})
    cadenzas[7] = [recent]
    cadenzas[9] = [recent]
    cadenzas[10] = [profile, recent]
    cadenzas[11] = [profile]
    cadenzas[13] = [profile, recent]
    return cadenzas


def _build_validations(config: dict[str, Any], agent: dict[str, Any], instruments_dir: str) -> list[dict[str, Any]]:
    name = agent["name"]
    agents_dir = config["agents_dir"]
    v: list[dict[str, Any]] = []
    v.append({"type": "file_exists", "path": f"{{{{workspace}}}}/cycle-state/{name}-recon.md", "condition": "stage == 1", "description": "Recon report"})
    v.append({"type": "file_exists", "path": f"{{{{workspace}}}}/cycle-state/{name}-plan.md", "condition": "stage == 2", "description": "Cycle plan"})
    for val in config.get("validations", []):
        v.append({"type": "command_succeeds", "command": val["command"], "condition": "stage == 3", "description": val.get("description", val["command"]), "timeout_seconds": val.get("timeout_seconds", 600)})
    v.append({"type": "file_exists", "path": f"{{{{workspace}}}}/cycle-state/{name}-inspection.md", "condition": "stage == 8", "description": "Inspection report"})
    v.append({"type": "content_contains", "path": f"{{{{workspace}}}}/cycle-state/{name}-aar.md", "pattern": "SUSTAIN:", "condition": "stage == 9", "description": "AAR has SUSTAIN"})
    v.append({"type": "content_contains", "path": f"{{{{workspace}}}}/cycle-state/{name}-aar.md", "pattern": "IMPROVE:", "condition": "stage == 9", "description": "AAR has IMPROVE"})
    v.append({"type": "command_succeeds", "command": f"AGENT_DIR={agents_dir}/{name} REPORT_PATH={{{{workspace}}}}/cycle-state/maturity-report.yaml bash {instruments_dir}/maturity-check.sh", "condition": "stage == 12", "description": "Maturity check", "timeout_seconds": 30})
    v.append({"type": "command_succeeds", "command": f"AGENT_DIR={agents_dir}/{name} L1_BUDGET=900 L2_BUDGET=1500 L3_BUDGET=1500 bash {instruments_dir}/token-budget-check.sh", "condition": "stage == 13", "description": "Token budget", "timeout_seconds": 10})
    return v


def _inline_templates(templates_source: Path) -> str:
    parts = []
    template_map = {1: "01-recon.j2", 2: "02-plan.j2", 3: "03-work.j2", 5: "05-play.j2", 7: "07-integration.j2", 8: "08-inspect.j2", 9: "09-aar.j2", 10: "10-consolidate.j2", 11: "11-reflect.j2", 13: "13-resurrect.j2"}
    for sheet_num, filename in sorted(template_map.items()):
        source_path = templates_source / filename
        if source_path.exists():
            content = source_path.read_text()
            parts.append(f"{{% if stage == {sheet_num} %}}")
            parts.append(content)
            parts.append("{% endif %}")
    for cli_sheet in [4, 6, 12]:
        parts.append("{{% if stage == " + str(cli_sheet) + " %}}CLI instrument sheet — no prompt needed.{% endif %}")
    return "\n".join(parts)


def build_score(config: dict[str, Any], agent: dict[str, Any], output_dir: Path, inlined_template: str) -> dict[str, Any]:
    name = agent["name"]
    agents_dir = config["agents_dir"]
    agent_identity_dir = f"{agents_dir}/{name}"
    shared_dir = str(output_dir / "shared")
    instruments_dir = f"{shared_dir}/instruments"
    score_path = str(output_dir / f"{name}.yaml")

    prelude = list(config.get("prelude", []))
    prelude.append({"file": f"{agent_identity_dir}/identity.md", "as": "context"})

    skip_when: dict[int, dict[str, Any]] = {}
    play_env = config.get("play_routing", {})
    temp_cmd = f"WORKSPACE={{workspace}} AGENT_DIR={agents_dir}/{name} AGENT_NAME={name} MEMORY_BLOAT_THRESHOLD={play_env.get('memory_bloat_threshold', 3000)} STAGNATION_CYCLES={play_env.get('stagnation_cycles', 3)} MIN_CYCLES_BETWEEN_PLAY={play_env.get('min_cycles_between_play', 5)} bash {instruments_dir}/temperature-check.sh; test $? -eq 1"
    for s in [5, 6, 7]:
        skip_when[s] = {"command": temp_cmd, "description": f"Skip play sheet {s} if temperature says work", "timeout_seconds": 30}

    instruments_cfg = config.get("instruments", {})
    expensive_model = instruments_cfg.get("expensive_model")
    standard_model = instruments_cfg.get("standard_model")
    per_sheet_config: dict[int, dict[str, str]] = {}
    if expensive_model:
        for s in [3, 5]:
            per_sheet_config[s] = {"model": expensive_model}
    if standard_model:
        for s in [1, 2, 7, 8, 9, 10, 11, 13]:
            per_sheet_config[s] = {"model": standard_model}

    cadenzas = _build_cadenzas(agent_identity_dir, config)
    validations = _build_validations(config, agent, instruments_dir)

    score: dict[str, Any] = {
        "name": f"{config['name']}-{name}",
        "workspace": config["workspace"],
        "backend": config["backend"],
        "instrument_fallbacks": config.get("instrument_fallbacks", []),
        "sheet": {
            "size": 1,
            "total_items": SHEETS_PER_CYCLE,
            "prelude": prelude,
            "cadenzas": cadenzas,
            "skip_when_command": skip_when,
        },
        "retry": {"max_retries": 3, "base_delay_seconds": 30, "max_completion_attempts": 3, "completion_threshold_percent": 50},
        "rate_limit": {"wait_minutes": 60, "max_waits": 24},
        "stale_detection": {"enabled": True, "idle_timeout_seconds": 3600},
        "concert": {"enabled": True, "max_chain_depth": config["concert"]["max_chain_depth"]},
        "on_success": [{"type": "run_job", "job_path": score_path, "detached": True, "fresh": True}],
        "prompt": {
            "template": inlined_template,
            "variables": {
                "agent_name": name,
                "role": agent.get("role", "builder"),
                "focus": agent.get("focus", ""),
                "voice": agent.get("voice", ""),
                "agent_identity_dir": agent_identity_dir,
                "playspace": config.get("playspace", ""),
                "validations": config.get("validations", []),
                "pre_commit_commands": config.get("pre_commit_commands", []),
            },
        },
        "validations": validations,
    }
    if per_sheet_config:
        score["sheet"]["per_sheet_instrument_config"] = per_sheet_config
    return score


def copy_shared_assets(output_dir: Path) -> None:
    shared = output_dir / "shared"
    templates_out = shared / "templates"
    templates_out.mkdir(parents=True, exist_ok=True)
    if TEMPLATES_DIR.exists():
        for f in TEMPLATES_DIR.glob("*.j2"):
            shutil.copy2(f, templates_out / f.name)
    instruments_out = shared / "instruments"
    instruments_out.mkdir(parents=True, exist_ok=True)
    if INSTRUMENTS_DIR.exists():
        for f in INSTRUMENTS_DIR.glob("*.sh"):
            dest = instruments_out / f.name
            shutil.copy2(f, dest)
            dest.chmod(0o755)
    # Seed composer notes
    seed = shared / "composer-notes-seed.yaml"
    if not seed.exists():
        seed.write_text("notes: []\n")


def print_stats(config: dict[str, Any]) -> None:
    print(f"Agent Score Generator — Dry Run")
    print(f"{'=' * 50}")
    print(f"Project:    {config['name']}")
    print(f"Workspace:  {config['workspace']}")
    print(f"Agents: {len(config['agents'])}")
    for agent in config["agents"]:
        print(f"  - {agent['name']} ({agent.get('role', 'builder')}): {agent.get('focus', '')}")
    print(f"Sheets per cycle: {SHEETS_PER_CYCLE}")
    print(f"Instrument fallbacks: {config.get('instrument_fallbacks', [])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one self-chaining Mozart score per agent.")
    parser.add_argument("config", help="Path to generator config YAML")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.dry_run:
        print_stats(config)
        return
    if not args.output:
        print("Error: --output required", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_shared_assets(output_dir)

    inlined = _inline_templates(output_dir / "shared" / "templates")

    for agent in config["agents"]:
        score = build_score(config, agent, output_dir, inlined)
        score_path = output_dir / f"{agent['name']}.yaml"
        with open(score_path, "w") as f:
            yaml.dump(score, f, default_flow_style=False, sort_keys=False, width=120)
        print(f"Score written: {score_path}")

    print(f"\n{len(config['agents'])} scores generated in {output_dir}")


if __name__ == "__main__":
    main()
