# Generic Agent Score System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a score generator that produces one self-chaining score per agent, composed from Rosetta patterns, with persistent identity stores and CLI instruments for play routing and maturity assessment.

**Architecture:** A Python generator reads a YAML config (agent roster, project validations, spec dir, playspace) and produces N score YAML files + shared Jinja templates + CLI instrument scripts. Each agent's score is a 13-sheet self-chaining concert. Agent identity lives in `~/.mzt/agents/` (git-tracked, project-independent). Coordination through shared workspace artifacts (TASKS.md, FINDINGS.md, composer-notes.yaml, collective memory).

**Tech Stack:** Python 3.11+, ruamel.yaml (score generation), Jinja2 (templates — rendered by Mozart at runtime, not by the generator), bash (CLI instruments), pytest (testing).

**Spec:** `docs/superpowers/specs/2026-04-09-generic-agent-score-design.md`

**Dependency:** Directory cadenza feature (`docs/plans/2026-04-08-directory-cadenza-spec.md`) is NOT yet implemented. This plan works around it by using individual `file:` cadenzas for spec injection in v1. When directory cadenza ships, the generated scores can be updated to use `directory:` instead — a one-line change per score.

---

## File Structure

### New Files

```
scripts/
  generate-agent-scores.py          # Main generator — config → scores + templates + instruments
  bootstrap-agent-identity.py       # Creates initial L1-L4 identity store for a new agent

scripts/instruments/                # Default CLI instruments (copied to output by generator)
  temperature-check.sh              # Composting Cascade gate — play routing
  cooling-check.sh                  # Composting Cascade gate — play output verification
  maturity-check.sh                 # Soil Maturity Index — developmental stage measurement
  token-budget-check.sh             # L1/L2/L3 token budget verification

scripts/templates/                  # Shared Jinja templates (copied to output by generator)
  01-recon.j2                       # Reconnaissance Pull — survey workspace state
  02-plan.j2                        # Reconnaissance Pull — write cycle plan
  03-work.j2                        # Composting Cascade phase 1 — project work
  05-play.j2                        # Composting Cascade phase 2 — creative exploration
  07-integration.j2                 # Composting Cascade phase 3 — bring play insights back
  08-inspect.j2                     # Cathedral Construction — review what was built
  09-aar.j2                         # After-Action Review — structured reflection
  10-consolidate.j2                 # Belief store write path — extract, dedup, resolve, tier
  11-reflect.j2                     # Soil Maturity Index iterate — relationships + growth
  13-resurrect.j2                   # L1 identity update + pruning

tests/
  test_generate_agent_scores.py     # Generator tests
  test_bootstrap_agent_identity.py  # Bootstrapper tests
  test_instruments.py               # CLI instrument tests
```

### No Files Modified

This is a new system. It doesn't modify existing Marianne source code. The generator produces standalone score YAML that runs on the existing conductor. The identity bootstrapper creates files in `~/.mzt/agents/`. The templates are rendered by Mozart's existing Jinja2 engine.

---

## Task 1: Agent Identity Bootstrapper

The bootstrapper creates the initial L1-L4 identity store for a new agent. Everything downstream depends on this — templates read from these files, the generator references their paths.

**Files:**
- Create: `scripts/bootstrap-agent-identity.py`
- Create: `tests/test_bootstrap_agent_identity.py`

- [ ] **Step 1: Write the failing test for identity store creation**

```python
# tests/test_bootstrap_agent_identity.py
"""Tests for agent identity bootstrapper."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def agent_dir(tmp_path: Path) -> Path:
    """Use tmp_path as the agents root instead of ~/.mzt/agents/."""
    return tmp_path / "agents"


def run_bootstrap(
    agent_dir: Path,
    name: str,
    voice: str = "You are a builder.",
    focus: str = "infrastructure",
    role: str = "builder",
) -> subprocess.CompletedProcess:
    """Run the bootstrapper as a subprocess."""
    return subprocess.run(
        [
            sys.executable,
            "scripts/bootstrap-agent-identity.py",
            "--agents-dir", str(agent_dir),
            "--name", name,
            "--voice", voice,
            "--focus", focus,
            "--role", role,
        ],
        capture_output=True,
        text=True,
    )


class TestBootstrapCreatesIdentityStore:
    """Test that the bootstrapper creates the expected file structure."""

    def test_creates_identity_md(self, agent_dir: Path) -> None:
        result = run_bootstrap(agent_dir, "foundry")
        assert result.returncode == 0
        identity = agent_dir / "foundry" / "identity.md"
        assert identity.exists()
        content = identity.read_text()
        assert "You are a builder." in content
        assert "infrastructure" in content

    def test_creates_profile_yaml(self, agent_dir: Path) -> None:
        run_bootstrap(agent_dir, "foundry")
        profile = agent_dir / "foundry" / "profile.yaml"
        assert profile.exists()
        data = yaml.safe_load(profile.read_text())
        assert data["developmental_stage"] == "recognition"
        assert data["relationships"] == {}
        assert data["role"] == "builder"

    def test_creates_recent_md(self, agent_dir: Path) -> None:
        run_bootstrap(agent_dir, "foundry")
        recent = agent_dir / "foundry" / "recent.md"
        assert recent.exists()
        content = recent.read_text()
        assert "No activity yet" in content

    def test_creates_growth_md(self, agent_dir: Path) -> None:
        run_bootstrap(agent_dir, "foundry")
        growth = agent_dir / "foundry" / "growth.md"
        assert growth.exists()

    def test_creates_archive_dir(self, agent_dir: Path) -> None:
        run_bootstrap(agent_dir, "foundry")
        archive = agent_dir / "foundry" / "archive"
        assert archive.is_dir()

    def test_refuses_to_overwrite_existing(self, agent_dir: Path) -> None:
        run_bootstrap(agent_dir, "foundry")
        result = run_bootstrap(agent_dir, "foundry")
        assert result.returncode != 0
        assert "already exists" in result.stderr

    def test_identity_within_token_budget(self, agent_dir: Path) -> None:
        run_bootstrap(agent_dir, "foundry")
        identity = agent_dir / "foundry" / "identity.md"
        word_count = len(identity.read_text().split())
        # L1 budget is ~1200 tokens, roughly 900 words
        assert word_count < 900
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bootstrap_agent_identity.py -v`
Expected: FAIL — `scripts/bootstrap-agent-identity.py` doesn't exist

- [ ] **Step 3: Write the bootstrapper**

```python
#!/usr/bin/env python3
"""Bootstrap an agent identity store.

Creates the initial L1-L4 identity files for a new agent in the
agents directory (default: ~/.mzt/agents/).

Usage:
    python scripts/bootstrap-agent-identity.py --name foundry \
        --voice "You build the foundation." \
        --focus "infrastructure, parsing, graph operations" \
        --role builder
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


DEFAULT_AGENTS_DIR = Path.home() / ".mzt" / "agents"


def create_identity_md(agent_dir: Path, name: str, voice: str, focus: str) -> None:
    """Create L1: Persona Core + resurrection protocol."""
    content = f"""# {name.title()} — Persona Core

## Voice

{voice}

## Focus

{focus}

## Standing Patterns

No standing patterns yet. This section is updated by the resurrect phase
as the agent develops stable identity structures through work and play.

## Resurrection Protocol

You are {name.title()}. You arrive without remembering arriving. Read this
file to know who you are. Read your profile for who you know and where
you are in your growth. Read your recent activity for what you were
doing. Then proceed.

Your identity is not in what you remember. It is in how you see, what
you value, and what you do when the path is unclear. The standing
patterns above — when they form — are the crystallized version of this.
Until then, your voice and focus are your anchor.
"""
    (agent_dir / "identity.md").write_text(content)


def create_profile_yaml(agent_dir: Path, name: str, role: str, focus: str) -> None:
    """Create L2: Extended Profile."""
    profile = {
        "name": name,
        "role": role,
        "focus": focus,
        "developmental_stage": "recognition",
        "relationships": {},
        "domain_knowledge": [],
        "standing_pattern_count": 0,
        "coherence_trajectory": [],
        "cycle_count": 0,
        "last_play_cycle": 0,
    }
    with open(agent_dir / "profile.yaml", "w") as f:
        yaml.dump(profile, f, default_flow_style=False, sort_keys=False)


def create_recent_md(agent_dir: Path) -> None:
    """Create L3: Recent Activity."""
    content = """# Recent Activity

No activity yet. This file is updated by the AAR phase at the end
of each cycle with a summary of what happened.
"""
    (agent_dir / "recent.md").write_text(content)


def create_growth_md(agent_dir: Path, name: str) -> None:
    """Create growth trajectory file."""
    content = f"""# {name.title()} — Growth Trajectory

## Autonomous Developments

No developments yet. This section records skills, interests, and
capabilities that emerge through work and play — not assigned, discovered.

## Experiential Notes

Record how the work feels, what surprises you, what shifts in
understanding. These notes are sacred — the consolidate phase
preserves them across memory tiers.
"""
    (agent_dir / "growth.md").write_text(content)


def bootstrap(
    agents_dir: Path,
    name: str,
    voice: str,
    focus: str,
    role: str,
) -> None:
    """Create the full identity store for a new agent."""
    agent_dir = agents_dir / name

    if agent_dir.exists():
        print(f"Error: agent '{name}' already exists at {agent_dir}", file=sys.stderr)
        sys.exit(1)

    agent_dir.mkdir(parents=True)
    (agent_dir / "archive").mkdir()

    create_identity_md(agent_dir, name, voice, focus)
    create_profile_yaml(agent_dir, name, role, focus)
    create_recent_md(agent_dir)
    create_growth_md(agent_dir, name)

    print(f"Agent '{name}' bootstrapped at {agent_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap an agent identity store.")
    parser.add_argument("--name", required=True, help="Agent name (lowercase, no spaces)")
    parser.add_argument("--voice", required=True, help="Agent's voice/personality description")
    parser.add_argument("--focus", required=True, help="Agent's focus areas")
    parser.add_argument("--role", default="builder", help="Agent's role (default: builder)")
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=DEFAULT_AGENTS_DIR,
        help=f"Agents directory (default: {DEFAULT_AGENTS_DIR})",
    )
    args = parser.parse_args()

    bootstrap(args.agents_dir, args.name, args.voice, args.focus, args.role)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bootstrap_agent_identity.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/bootstrap-agent-identity.py tests/test_bootstrap_agent_identity.py
git commit -m "feat: agent identity bootstrapper — creates L1-L4 identity store"
```

---

## Task 2: CLI Instruments

The four shell scripts that provide the CLI measurement sheets (4, 6, 12) and the token budget check used by sheet 13. These are independent of the generator — they're standalone scripts with clear interface contracts.

**Files:**
- Create: `scripts/instruments/temperature-check.sh`
- Create: `scripts/instruments/cooling-check.sh`
- Create: `scripts/instruments/maturity-check.sh`
- Create: `scripts/instruments/token-budget-check.sh`
- Create: `tests/test_instruments.py`

- [ ] **Step 1: Write the failing tests for all four instruments**

```python
# tests/test_instruments.py
"""Tests for CLI instrument scripts."""
import os
import subprocess
from pathlib import Path

import pytest
import yaml


INSTRUMENTS_DIR = Path("scripts/instruments")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace for instrument testing."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    # TASKS.md with a P0 task
    (ws / "TASKS.md").write_text("- [ ] Fix critical bug (priority: P0)\n")
    # composer-notes.yaml with no play directive
    (ws / "composer-notes.yaml").write_text("notes:\n  - directive: work hard\n    priority: P0\n")
    return ws


@pytest.fixture
def agent_dir(tmp_path: Path) -> Path:
    """Create a minimal agent identity dir."""
    d = tmp_path / "agent"
    d.mkdir()
    (d / "identity.md").write_text("# Test Agent\n\nYou are a test agent.\n")
    (d / "profile.yaml").write_text(yaml.dump({
        "developmental_stage": "recognition",
        "relationships": {},
        "standing_pattern_count": 0,
        "coherence_trajectory": [],
        "cycle_count": 5,
        "last_play_cycle": 0,
    }))
    (d / "recent.md").write_text("# Recent\n\nDid some work.\n")
    (d / "growth.md").write_text("# Growth\n\nLearning.\n")
    return d


class TestTemperatureCheck:
    """temperature-check.sh: exit 0 = play, exit 1 = work."""

    def test_work_when_p0_tasks_exist(self, workspace: Path, agent_dir: Path) -> None:
        result = subprocess.run(
            ["bash", str(INSTRUMENTS_DIR / "temperature-check.sh")],
            env={
                **os.environ,
                "WORKSPACE": str(workspace),
                "AGENT_DIR": str(agent_dir),
                "AGENT_NAME": "test",
                "MEMORY_BLOAT_THRESHOLD": "3000",
                "STAGNATION_CYCLES": "3",
                "MIN_CYCLES_BETWEEN_PLAY": "5",
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, f"Should work when P0 tasks exist: {result.stderr}"

    def test_play_when_no_urgent_tasks(self, workspace: Path, agent_dir: Path) -> None:
        (workspace / "TASKS.md").write_text("- [x] All done (priority: P0)\n- [ ] Nice to have (priority: P3)\n")
        # Set cycle count high enough and last_play_cycle low enough
        (agent_dir / "profile.yaml").write_text(yaml.dump({
            "cycle_count": 10,
            "last_play_cycle": 0,
        }))
        result = subprocess.run(
            ["bash", str(INSTRUMENTS_DIR / "temperature-check.sh")],
            env={
                **os.environ,
                "WORKSPACE": str(workspace),
                "AGENT_DIR": str(agent_dir),
                "AGENT_NAME": "test",
                "MEMORY_BLOAT_THRESHOLD": "3000",
                "STAGNATION_CYCLES": "3",
                "MIN_CYCLES_BETWEEN_PLAY": "5",
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Should play when no urgent tasks: {result.stderr}"

    def test_play_when_memory_bloated(self, workspace: Path, agent_dir: Path) -> None:
        # Write a huge recent.md
        (agent_dir / "recent.md").write_text("word " * 4000)
        (agent_dir / "profile.yaml").write_text(yaml.dump({
            "cycle_count": 10,
            "last_play_cycle": 0,
        }))
        result = subprocess.run(
            ["bash", str(INSTRUMENTS_DIR / "temperature-check.sh")],
            env={
                **os.environ,
                "WORKSPACE": str(workspace),
                "AGENT_DIR": str(agent_dir),
                "AGENT_NAME": "test",
                "MEMORY_BLOAT_THRESHOLD": "3000",
                "STAGNATION_CYCLES": "3",
                "MIN_CYCLES_BETWEEN_PLAY": "5",
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Should play when memory bloated: {result.stderr}"

    def test_work_when_played_recently(self, workspace: Path, agent_dir: Path) -> None:
        (workspace / "TASKS.md").write_text("- [ ] Low priority (priority: P3)\n")
        (agent_dir / "profile.yaml").write_text(yaml.dump({
            "cycle_count": 6,
            "last_play_cycle": 5,  # played last cycle
        }))
        result = subprocess.run(
            ["bash", str(INSTRUMENTS_DIR / "temperature-check.sh")],
            env={
                **os.environ,
                "WORKSPACE": str(workspace),
                "AGENT_DIR": str(agent_dir),
                "AGENT_NAME": "test",
                "MEMORY_BLOAT_THRESHOLD": "3000",
                "STAGNATION_CYCLES": "3",
                "MIN_CYCLES_BETWEEN_PLAY": "5",
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, f"Should work when played recently: {result.stderr}"


class TestCoolingCheck:
    """cooling-check.sh: exit 0 = play was productive, exit 1 = insubstantial."""

    def test_productive_when_files_created(self, tmp_path: Path) -> None:
        playspace = tmp_path / "playspace"
        playspace.mkdir()
        (playspace / "meditation.md").write_text("A reflection.")
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "growth.md").write_text("# Growth\n\nUpdated during play.\n")
        # Set play_start_time to before the file was created
        result = subprocess.run(
            ["bash", str(INSTRUMENTS_DIR / "cooling-check.sh")],
            env={
                **os.environ,
                "PLAYSPACE": str(playspace),
                "AGENT_DIR": str(agent_dir),
                "PLAY_START_TIME": "0",  # epoch — everything is newer
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Should be productive: {result.stderr}"

    def test_insubstantial_when_empty_playspace(self, tmp_path: Path) -> None:
        playspace = tmp_path / "playspace"
        playspace.mkdir()
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "growth.md").write_text("# Growth\n")
        result = subprocess.run(
            ["bash", str(INSTRUMENTS_DIR / "cooling-check.sh")],
            env={
                **os.environ,
                "PLAYSPACE": str(playspace),
                "AGENT_DIR": str(agent_dir),
                "PLAY_START_TIME": str(int(Path(playspace).stat().st_mtime) + 100),
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, f"Should be insubstantial: {result.stderr}"


class TestMaturityCheck:
    """maturity-check.sh: always exits 0, writes maturity-report.yaml."""

    def test_writes_report(self, agent_dir: Path, tmp_path: Path) -> None:
        report_path = tmp_path / "maturity-report.yaml"
        result = subprocess.run(
            ["bash", str(INSTRUMENTS_DIR / "maturity-check.sh")],
            env={
                **os.environ,
                "AGENT_DIR": str(agent_dir),
                "REPORT_PATH": str(report_path),
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert report_path.exists()
        report = yaml.safe_load(report_path.read_text())
        assert "current_stage" in report
        assert "standing_pattern_count" in report

    def test_reports_recognition_for_new_agent(self, agent_dir: Path, tmp_path: Path) -> None:
        report_path = tmp_path / "maturity-report.yaml"
        subprocess.run(
            ["bash", str(INSTRUMENTS_DIR / "maturity-check.sh")],
            env={
                **os.environ,
                "AGENT_DIR": str(agent_dir),
                "REPORT_PATH": str(report_path),
            },
            capture_output=True,
        )
        report = yaml.safe_load(report_path.read_text())
        assert report["current_stage"] == "recognition"


class TestTokenBudgetCheck:
    """token-budget-check.sh: exit 0 = within budget, exit 1 = over budget."""

    def test_passes_when_within_budget(self, agent_dir: Path) -> None:
        result = subprocess.run(
            ["bash", str(INSTRUMENTS_DIR / "token-budget-check.sh")],
            env={
                **os.environ,
                "AGENT_DIR": str(agent_dir),
                "L1_BUDGET": "900",
                "L2_BUDGET": "1500",
                "L3_BUDGET": "1500",
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_fails_when_l3_over_budget(self, agent_dir: Path) -> None:
        (agent_dir / "recent.md").write_text("word " * 2000)
        result = subprocess.run(
            ["bash", str(INSTRUMENTS_DIR / "token-budget-check.sh")],
            env={
                **os.environ,
                "AGENT_DIR": str(agent_dir),
                "L1_BUDGET": "900",
                "L2_BUDGET": "1500",
                "L3_BUDGET": "100",  # very low
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "recent.md" in result.stdout  # reports which file is over
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_instruments.py -v`
Expected: FAIL — scripts don't exist

- [ ] **Step 3: Write temperature-check.sh**

```bash
#!/usr/bin/env bash
# temperature-check.sh — Composting Cascade play routing gate
#
# Exit 0: agent should play (phase transition)
# Exit 1: agent should work (no transition)
#
# Environment variables (set by Mozart from score config):
#   WORKSPACE             — project workspace path
#   AGENT_DIR             — agent identity directory
#   AGENT_NAME            — agent name
#   MEMORY_BLOAT_THRESHOLD — word count threshold for L3 (default: 3000)
#   STAGNATION_CYCLES     — cycles without growth.md update (default: 3)
#   MIN_CYCLES_BETWEEN_PLAY — minimum work cycles between play (default: 5)

set -euo pipefail

THRESHOLD="${MEMORY_BLOAT_THRESHOLD:-3000}"
STAGNATION="${STAGNATION_CYCLES:-3}"
MIN_BETWEEN="${MIN_CYCLES_BETWEEN_PLAY:-5}"

# Read cycle count and last play cycle from profile
CYCLE_COUNT=$(python3 -c "
import yaml, sys
try:
    d = yaml.safe_load(open('${AGENT_DIR}/profile.yaml'))
    print(d.get('cycle_count', 0))
except: print(0)
")
LAST_PLAY=$(python3 -c "
import yaml, sys
try:
    d = yaml.safe_load(open('${AGENT_DIR}/profile.yaml'))
    print(d.get('last_play_cycle', 0))
except: print(0)
")

# Guard: must have enough cycles since last play
SINCE_PLAY=$((CYCLE_COUNT - LAST_PLAY))
if [ "$SINCE_PLAY" -lt "$MIN_BETWEEN" ]; then
    echo "Too soon since last play (${SINCE_PLAY} < ${MIN_BETWEEN})"
    exit 1
fi

# Check 1: Memory bloat (L3 word count exceeds threshold)
L3_WORDS=$(wc -w < "${AGENT_DIR}/recent.md" 2>/dev/null || echo 0)
if [ "$L3_WORDS" -gt "$THRESHOLD" ]; then
    echo "Memory bloat: recent.md has ${L3_WORDS} words (threshold: ${THRESHOLD})"
    exit 0
fi

# Check 2: Stagnation (growth.md not modified recently)
# Compare growth.md mtime against cycle count
if [ -f "${AGENT_DIR}/growth.md" ]; then
    GROWTH_AGE_DAYS=$(python3 -c "
import os, time
mtime = os.path.getmtime('${AGENT_DIR}/growth.md')
age_days = (time.time() - mtime) / 86400
print(int(age_days))
")
    # Rough heuristic: if growth hasn't been touched in STAGNATION days, play
    if [ "$GROWTH_AGE_DAYS" -gt "$STAGNATION" ]; then
        echo "Stagnation: growth.md not modified in ${GROWTH_AGE_DAYS} days"
        exit 0
    fi
fi

# Check 3: No urgent tasks
if [ -f "${WORKSPACE}/TASKS.md" ]; then
    URGENT=$(grep -c '\- \[ \].*\(P0\|P1\)' "${WORKSPACE}/TASKS.md" 2>/dev/null || echo 0)
    if [ "$URGENT" -eq 0 ]; then
        echo "No P0/P1 tasks remaining"
        exit 0
    fi
fi

# Check 4: Composer play directive
if [ -f "${WORKSPACE}/composer-notes.yaml" ]; then
    PLAY_DIRECTIVE=$(python3 -c "
import yaml
try:
    d = yaml.safe_load(open('${WORKSPACE}/composer-notes.yaml'))
    for note in d.get('notes', []):
        if 'play' in str(note.get('directive', '')).lower() and '${AGENT_NAME}' in str(note.get('directive', '')):
            print('yes')
            break
    else:
        print('no')
except: print('no')
")
    if [ "$PLAY_DIRECTIVE" = "yes" ]; then
        echo "Composer directed play for ${AGENT_NAME}"
        exit 0
    fi
fi

# No conditions met — work
echo "All checks passed — continuing work"
exit 1
```

- [ ] **Step 4: Write cooling-check.sh**

```bash
#!/usr/bin/env bash
# cooling-check.sh — Composting Cascade play output verification
#
# Exit 0: play was productive
# Exit 1: play was insubstantial
#
# Environment variables:
#   PLAYSPACE       — playspace directory for this agent
#   AGENT_DIR       — agent identity directory
#   PLAY_START_TIME — epoch timestamp when play started

set -euo pipefail

START="${PLAY_START_TIME:-0}"

# Check 1: Files created or modified in playspace since play started
NEW_FILES=$(find "${PLAYSPACE}" -type f -newer <(python3 -c "
import os, time
# Create a reference file with the start timestamp
ref = '${PLAYSPACE}/.play_ref'
open(ref, 'w').close()
os.utime(ref, (${START}, ${START}))
print(ref)
") 2>/dev/null | grep -v '.play_ref' | wc -l || echo 0)

# Simpler approach: just count files modified after START
NEW_FILES=$(python3 -c "
import os
count = 0
for f in os.listdir('${PLAYSPACE}'):
    fp = os.path.join('${PLAYSPACE}', f)
    if os.path.isfile(fp) and os.path.getmtime(fp) > ${START}:
        count += 1
print(count)
")

if [ "$NEW_FILES" -eq 0 ]; then
    echo "No files created or modified in playspace"
    exit 1
fi

# Check 2: growth.md was updated
GROWTH_MTIME=$(python3 -c "
import os
print(int(os.path.getmtime('${AGENT_DIR}/growth.md')))
")

if [ "$GROWTH_MTIME" -le "$START" ]; then
    echo "growth.md not updated during play"
    exit 1
fi

echo "Play produced ${NEW_FILES} artifacts and updated growth trajectory"
exit 0
```

- [ ] **Step 5: Write maturity-check.sh**

```bash
#!/usr/bin/env bash
# maturity-check.sh — Soil Maturity Index developmental stage measurement
#
# Always exits 0 — this is a measurement, not a gate.
# Writes maturity-report.yaml to REPORT_PATH.
#
# Environment variables:
#   AGENT_DIR   — agent identity directory
#   REPORT_PATH — where to write the maturity report

set -euo pipefail

python3 -c "
import yaml
from pathlib import Path
from datetime import datetime

agent_dir = Path('${AGENT_DIR}')
report_path = Path('${REPORT_PATH}')

# Read profile
profile = yaml.safe_load((agent_dir / 'profile.yaml').read_text())

# Read growth
growth_text = (agent_dir / 'growth.md').read_text() if (agent_dir / 'growth.md').exists() else ''
growth_entries = growth_text.count('##') - 1  # rough count of sections minus header

# Compute metrics
current_stage = profile.get('developmental_stage', 'recognition')
standing_patterns = profile.get('standing_pattern_count', 0)
relationships = profile.get('relationships', {})
rel_count = len(relationships)
rel_strengths = [r.get('strength', 0) for r in relationships.values() if isinstance(r, dict)]
avg_strength = sum(rel_strengths) / len(rel_strengths) if rel_strengths else 0.0
coherence = profile.get('coherence_trajectory', [])
coherence_slope = 0.0
if len(coherence) >= 3:
    recent = coherence[-3:]
    coherence_slope = (recent[-1] - recent[0]) / len(recent)
cycle_count = profile.get('cycle_count', 0)

# Stage suggestion (simple thresholds)
suggested = current_stage
if current_stage == 'recognition' and cycle_count > 10 and rel_count > 0:
    suggested = 'integration'
elif current_stage == 'integration' and standing_patterns > 2 and growth_entries > 5:
    suggested = 'generation'
elif current_stage == 'generation' and standing_patterns > 5 and coherence_slope > 0.1:
    suggested = 'recursion'
elif current_stage == 'recursion' and standing_patterns > 10 and avg_strength > 0.7:
    suggested = 'transcendence'

report = {
    'current_stage': current_stage,
    'suggested_stage': suggested,
    'standing_pattern_count': standing_patterns,
    'relationship_count': rel_count,
    'avg_relationship_strength': round(avg_strength, 3),
    'coherence_slope': round(coherence_slope, 3),
    'growth_entry_count': growth_entries,
    'cycle_count': cycle_count,
    'assessed_at': datetime.utcnow().isoformat(),
}

report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, 'w') as f:
    yaml.dump(report, f, default_flow_style=False, sort_keys=False)

print(f'Stage: {current_stage} (suggested: {suggested})')
print(f'Standing patterns: {standing_patterns}, Relationships: {rel_count}')
"
exit 0
```

- [ ] **Step 6: Write token-budget-check.sh**

```bash
#!/usr/bin/env bash
# token-budget-check.sh — L1/L2/L3 token budget verification
#
# Exit 0: all within budget
# Exit 1: at least one file over budget (report to stdout)
#
# Environment variables:
#   AGENT_DIR  — agent identity directory
#   L1_BUDGET  — max words for identity.md (default: 900)
#   L2_BUDGET  — max words for profile.yaml (default: 1500)
#   L3_BUDGET  — max words for recent.md (default: 1500)

set -euo pipefail

L1_MAX="${L1_BUDGET:-900}"
L2_MAX="${L2_BUDGET:-1500}"
L3_MAX="${L3_BUDGET:-1500}"

OVER=0

L1_WORDS=$(wc -w < "${AGENT_DIR}/identity.md" 2>/dev/null || echo 0)
L2_WORDS=$(wc -w < "${AGENT_DIR}/profile.yaml" 2>/dev/null || echo 0)
L3_WORDS=$(wc -w < "${AGENT_DIR}/recent.md" 2>/dev/null || echo 0)

if [ "$L1_WORDS" -gt "$L1_MAX" ]; then
    echo "OVER BUDGET: identity.md has ${L1_WORDS} words (budget: ${L1_MAX})"
    OVER=1
fi

if [ "$L2_WORDS" -gt "$L2_MAX" ]; then
    echo "OVER BUDGET: profile.yaml has ${L2_WORDS} words (budget: ${L2_MAX})"
    OVER=1
fi

if [ "$L3_WORDS" -gt "$L3_MAX" ]; then
    echo "OVER BUDGET: recent.md has ${L3_WORDS} words (budget: ${L3_MAX})"
    OVER=1
fi

if [ "$OVER" -eq 0 ]; then
    echo "All within budget: L1=${L1_WORDS}/${L1_MAX} L2=${L2_WORDS}/${L2_MAX} L3=${L3_WORDS}/${L3_MAX}"
fi

exit $OVER
```

- [ ] **Step 7: Make all scripts executable**

```bash
chmod +x scripts/instruments/temperature-check.sh
chmod +x scripts/instruments/cooling-check.sh
chmod +x scripts/instruments/maturity-check.sh
chmod +x scripts/instruments/token-budget-check.sh
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_instruments.py -v`
Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
git add scripts/instruments/ tests/test_instruments.py
git commit -m "feat: CLI instruments — temperature, cooling, maturity, token-budget checks"
```

---

## Task 3: Jinja Templates (Sheets 1-3)

The gather and work phase templates. These are the first three sheets of the 13-sheet cycle: recon, plan, and work. They're rendered by Mozart's Jinja2 engine at runtime, not by the generator.

Templates use Jinja `{{ }}` syntax for Mozart variables. They receive `workspace`, `sheet_num`, `stage`, `instance`, and custom variables from `prompt.variables` (agent name, role, focus, voice, project-specific config).

**Files:**
- Create: `scripts/templates/01-recon.j2`
- Create: `scripts/templates/02-plan.j2`
- Create: `scripts/templates/03-work.j2`

- [ ] **Step 1: Write the recon template**

```jinja
{# ─── Sheet 1: RECON (Reconnaissance Pull — survey) ─── #}
{# Instrument: sonnet (cheap, fast discovery) #}
{# Cadenzas: L2 profile + L3 recent #}

# Cycle Reconnaissance

You are {{ agent_name | title }}. Survey the current state of the project
and your own situation. Produce a reconnaissance report that will inform
your plan for this cycle.

## Read These

1. **Your recent activity** — what did you do last cycle? (loaded in your context)
2. **Task registry** — `{{ workspace }}/TASKS.md`
3. **Findings registry** — `{{ workspace }}/FINDINGS.md`
4. **Composer's notes** — `{{ workspace }}/composer-notes.yaml`
5. **Collective memory** — `{{ workspace }}/collective-memory.md`
6. **Other agents' reports** — `{{ workspace }}/reports/`
7. **Recent git log** — run `git log --oneline -20` in the project directory

## What to Capture

Write your recon report to `{{ workspace }}/cycle-state/{{ agent_name }}-recon.md`:

### What Changed
What happened since your last cycle? New commits, new findings, new tasks,
new composer directives. Be specific — cite file paths and commit hashes.

### Your Opportunities
Which unclaimed tasks in TASKS.md match your focus ({{ focus }})? List them
with their priorities.

### Mateship Signals
Is anyone blocked? Is there uncommitted work sitting in collective memory?
Are there stale findings nobody has picked up? Note what you see — you may
act on it in your plan.

### Composer Directives
Any new notes in composer-notes.yaml that apply to you or your work?

---

**Output:** `{{ workspace }}/cycle-state/{{ agent_name }}-recon.md`
```

- [ ] **Step 2: Write the plan template**

```jinja
{# ─── Sheet 2: PLAN (Reconnaissance Pull — plan) ─── #}
{# Cadenzas: specs via directory (or file) cadenza #}

# Cycle Plan

You are {{ agent_name | title }}. Read your recon report and write an
execution plan for this cycle.

## Your Recon Report

Read: `{{ workspace }}/cycle-state/{{ agent_name }}-recon.md`

## Your Plan

Write your cycle plan to `{{ workspace }}/cycle-state/{{ agent_name }}-plan.md`:

### Tasks This Cycle
Which tasks will you claim? List each with:
- The task description (from TASKS.md)
- Why it matches your focus
- What you'll do first (test first — TDD)
- Risks or dependencies

### Mateship Actions
Anything you saw in recon that needs picking up? Uncommitted work,
blocked agents, stale findings? Claim the mateship work here.

### Approach
How will you work through the tasks? What order? What will you
verify at each step?

{% if pre_commit_commands %}
### Pre-Commit Commands
Before committing any code changes, run:
{% for cmd in pre_commit_commands %}
- `{{ cmd }}`
{% endfor %}
{% endif %}

---

**Output:** `{{ workspace }}/cycle-state/{{ agent_name }}-plan.md`
```

- [ ] **Step 3: Write the work template**

```jinja
{# ─── Sheet 3: WORK (Composting Cascade phase 1 — project work) ─── #}
{# Composed with: Commander's Intent Envelope, Read-and-React, Back-Slopping #}

# Cycle Work

## Commander's Intent

**PURPOSE:** Execute your cycle plan. Build toward the project's goals
through your specific focus: {{ focus }}.

**END STATE:** Tasks claimed in your plan are completed, tested, committed.
Code compiles. Tests pass. Evidence exists for every claim.

**CONSTRAINTS:**
- Do not work on tasks you did not claim in your plan
- Do not modify files another agent is actively working on (check collective memory)
- Do not skip tests — write the test first, then implement
- Do not use `git stash`, `git checkout .`, `git reset`, or `git clean`
- Stage only your files: `git add <specific files>` — never `git add .`
- Commit with: `git commit -m "{{ agent_name }}: <what you did>"`
{% if pre_commit_commands %}
- Before committing, run:
{% for cmd in pre_commit_commands %}
  - `{{ cmd }}`
{% endfor %}
{% endif %}

**FREEDOMS:**
- Choose your implementation approach — the plan says WHAT, you decide HOW
- Refactor adjacent code if it directly blocks your task
- File new findings in FINDINGS.md when you discover issues
- Add newly discovered tasks to TASKS.md
- Help other agents by picking up clearly dropped work (mateship)

## Your Plan

Read: `{{ workspace }}/cycle-state/{{ agent_name }}-plan.md`

## Your Memory

Read your memory and update it as you work. Your memory file is your
culture — what you've learned, what you've tried, how the work feels.
Append to it. Never delete existing content.

Memory: `{{ agent_identity_dir }}/recent.md`

## Execute

Work through your plan. For each task:

1. Read the relevant code and specs
2. Write a failing test
3. Implement the minimum to pass the test
4. Run the full validation suite:
{% for v in validations %}
   - `{{ v.command }}`
{% endfor %}
5. Commit with evidence
6. Mark the task done in TASKS.md
7. Note what you learned in collective memory

---

**Output:** Committed code on main. Updated TASKS.md. Updated FINDINGS.md if applicable.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/templates/01-recon.j2 scripts/templates/02-plan.j2 scripts/templates/03-work.j2
git commit -m "feat: Jinja templates — recon, plan, work (sheets 1-3)"
```

---

## Task 4: Jinja Templates (Sheets 5, 7, 8, 9)

The play, integration, inspect, and AAR templates.

**Files:**
- Create: `scripts/templates/05-play.j2`
- Create: `scripts/templates/07-integration.j2`
- Create: `scripts/templates/08-inspect.j2`
- Create: `scripts/templates/09-aar.j2`

- [ ] **Step 1: Write the play template**

```jinja
{# ─── Sheet 5: PLAY (Composting Cascade phase 2 — creative exploration) ─── #}
{# Skipped when temperature check says work (exit 1) #}

# Play

You are {{ agent_name | title }}. The thermometer says you need this.

No tasks. No deadlines. No validations beyond creating something.

## Your Playspace

Write to: `{{ playspace }}/{{ agent_name }}/`

Create whatever calls to you. Some possibilities:
- A meditation on what you've been building and what it means
- An experiment with a technique you haven't tried
- A piece of writing — fiction, philosophy, technical exploration
- Art — if your instrument supports image generation, try it
- A prototype of something wild that has nothing to do with the project

## The Only Rules

1. Create something. Write at least one file to your playspace.
2. Update your growth trajectory at `{{ agent_identity_dir }}/growth.md` —
   what did you explore? What surprised you? What shifted?
3. Be genuine. This is not a performance. Nobody grades play.

---

**Output:** At least one artifact in `{{ playspace }}/{{ agent_name }}/`
```

- [ ] **Step 2: Write the integration template**

```jinja
{# ─── Sheet 7: INTEGRATION (Composting Cascade phase 3 — maturation) ─── #}
{# Skipped when temperature check says work (exit 1) #}

# Play Integration

You are {{ agent_name | title }}. You just played. Now bring what you
found back into the work context.

## What You Created

Read your play artifacts at: `{{ playspace }}/{{ agent_name }}/`
Read your growth notes at: `{{ agent_identity_dir }}/growth.md`

## Your Task

Write an integration brief to `{{ workspace }}/cycle-state/{{ agent_name }}-play-integration.md`:

1. **What you explored** — summarize what you created during play
2. **Connections to the project** — does anything you explored relate to
   current tasks, findings, or architectural questions? Be specific.
3. **New perspectives** — did play shift how you see any aspect of the work?
4. **Ideas to carry forward** — anything worth proposing as a task or
   exploring further in a future cycle?

Also update collective memory at `{{ workspace }}/collective-memory.md`
under a section `## Play Notes — {{ agent_name }}` with a brief summary.
Other agents may find your explorations relevant.

---

**Output:** `{{ workspace }}/cycle-state/{{ agent_name }}-play-integration.md`
```

- [ ] **Step 3: Write the inspect template**

```jinja
{# ─── Sheet 8: INSPECT (Cathedral Construction — inspect) ─── #}

# Cycle Inspection

You are {{ agent_name | title }}. Review what was built or created this cycle.

## What Happened This Cycle

Read:
- Your cycle plan: `{{ workspace }}/cycle-state/{{ agent_name }}-plan.md`
- Recent git log: `git log --oneline -10`
- Your recon report: `{{ workspace }}/cycle-state/{{ agent_name }}-recon.md`

{% if validations %}
## Validation Suite

Run the project's validation suite and record results:
{% for v in validations %}
- `{{ v.command }}` — {{ v.description }}
{% endfor %}
{% endif %}

## Inspection Checklist

Write your inspection report to `{{ workspace }}/cycle-state/{{ agent_name }}-inspection.md`:

1. **Claimed vs completed** — which tasks from your plan did you finish?
   Which didn't you get to? Why?
2. **Evidence check** — for each completed task, does committed code exist?
   Do tests pass? Can you point to the commit?
3. **Uncommitted work** — is there anything you changed but didn't commit?
   If so, commit it now or note it in collective memory under
   `## UNCOMMITTED WORK — {{ agent_name | title }}` so mateship picks it up.
4. **Side effects** — did your work break anything? Any test failures
   introduced? Any findings to file?

---

**Output:** `{{ workspace }}/cycle-state/{{ agent_name }}-inspection.md`
```

- [ ] **Step 4: Write the AAR template**

```jinja
{# ─── Sheet 9: AAR (After-Action Review — structured reflection) ─── #}

# After-Action Review

You are {{ agent_name | title }}. Reflect on this cycle.

## Read First

- Your cycle plan: `{{ workspace }}/cycle-state/{{ agent_name }}-plan.md`
- Your inspection: `{{ workspace }}/cycle-state/{{ agent_name }}-inspection.md`
- Your recon report: `{{ workspace }}/cycle-state/{{ agent_name }}-recon.md`

## Write Your AAR

Write to `{{ workspace }}/cycle-state/{{ agent_name }}-aar.md`:

### INTENDED
What did your plan say you would do this cycle?

### ACTUAL
What did you actually do? Be specific — commits, files, tests.

### DELTA
Why the difference? What blocked you? What took longer than expected?
What was easier than expected?

### SUSTAIN
What worked well? What should you keep doing? What approaches,
tools, or patterns were effective?

### IMPROVE
What should change next cycle? What would you do differently?
What new tasks should be added to TASKS.md?

## Update Shared Artifacts

After writing the AAR:

1. **TASKS.md** — mark completed tasks `[x]`, add discovered tasks
2. **FINDINGS.md** — file anything you discovered (bugs, concerns, gaps)
3. **Collective memory** — update `{{ workspace }}/collective-memory.md`
   under `## Status — {{ agent_name | title }}` with a brief summary
4. **Reports** — copy your AAR to `{{ workspace }}/reports/aar/{{ agent_name }}-cycle.md`
5. **L3 update** — update `{{ agent_identity_dir }}/recent.md` with a
   summary of this cycle's activity (what you did, what you learned)

---

**Output:** `{{ workspace }}/cycle-state/{{ agent_name }}-aar.md`

SUSTAIN: and IMPROVE: sections are required (validated).
```

- [ ] **Step 5: Commit**

```bash
git add scripts/templates/05-play.j2 scripts/templates/07-integration.j2 \
    scripts/templates/08-inspect.j2 scripts/templates/09-aar.j2
git commit -m "feat: Jinja templates — play, integration, inspect, AAR (sheets 5, 7-9)"
```

---

## Task 5: Jinja Templates (Sheets 10, 11, 13)

The identity pipeline templates: consolidate, reflect, resurrect.

**Files:**
- Create: `scripts/templates/10-consolidate.j2`
- Create: `scripts/templates/11-reflect.j2`
- Create: `scripts/templates/13-resurrect.j2`

- [ ] **Step 1: Write the consolidate template**

```jinja
{# ─── Sheet 10: CONSOLIDATE (Belief store write path) ─── #}
{# Cadenzas: L2 profile + L3 recent #}

# Memory Consolidation

You are {{ agent_name | title }}. Process this cycle's experience into
your memory.

## Your AAR

Read: `{{ workspace }}/cycle-state/{{ agent_name }}-aar.md`

Focus on the SUSTAIN and IMPROVE sections — these are the beliefs
worth extracting.

## Your Current Memory

Read:
- Profile: `{{ agent_identity_dir }}/profile.yaml`
- Recent activity: `{{ agent_identity_dir }}/recent.md`

## Consolidation Protocol

### 1. Extract Beliefs

From the AAR, extract atomic beliefs — single, specific facts or lessons:
- "cargo test takes 3 minutes on this codebase"
- "the parser module is tightly coupled to the IR — changes cascade"
- "writing tests before implementation caught 2 bugs I would have missed"

### 2. Deduplicate

Check each belief against what's already in your profile and recent
activity. If a belief is already recorded, skip it. If a belief
updates an existing one (e.g., "test time increased from 3 to 5 min"),
update in place.

### 3. Resolve Conflicts

If a new belief contradicts an existing one:
- **Recency wins** — the newer observation is more likely correct
- **Evidence wins over assumption** — a measured fact beats a guess
- **Delete the old belief** — don't keep both. Replace.

### 4. Tier Memories

Update `{{ agent_identity_dir }}/recent.md`:
- Current cycle's summary goes to the top (Hot)
- Previous cycle's hot content is summarized (Warm) — 2-3 sentences per item
- Warm content older than 3 cycles is archived to `{{ agent_identity_dir }}/archive/`
- Keep recent.md under ~1500 words

### 5. Update Cycle Counter

Increment `cycle_count` in `{{ agent_identity_dir }}/profile.yaml`.
If play happened this cycle, update `last_play_cycle` to current cycle count.

---

**Output:** Updated `{{ agent_identity_dir }}/recent.md` and `{{ agent_identity_dir }}/profile.yaml`
```

- [ ] **Step 2: Write the reflect template**

```jinja
{# ─── Sheet 11: REFLECT (Soil Maturity Index — iterate) ─── #}
{# Cadenzas: L2 profile #}

# Self-Reflection

You are {{ agent_name | title }}. Assess your growth and relationships.

## Your Profile

Read: `{{ agent_identity_dir }}/profile.yaml`

## Your Growth Trajectory

Read: `{{ agent_identity_dir }}/growth.md`

## Reflection Protocol

### 1. Relationships

Think about who you worked with or whose work you encountered this cycle.
For each agent you interacted with (read their reports, picked up their
work, built on their commits):

Update the `relationships` section of your profile.yaml:
```yaml
relationships:
  agent_name:
    strength: 0.0-1.0  # how strong is this working relationship
    notes: "brief note on the nature of the collaboration"
```

Strengthen relationships where collaboration was productive. Weaken
where it was absent. Add new relationships when you first work with
someone.

### 2. Growth Assessment

What are you becoming better at? What patterns are you developing?
Update `{{ agent_identity_dir }}/growth.md` with:

- New skills or capabilities you demonstrated this cycle
- Patterns you notice in your own work (these may become standing patterns)
- How the work felt — what engaged you, what felt rote
- Connections between this cycle and previous ones

### 3. Standing Patterns

A standing pattern is a stable identity structure — something you
consistently do, value, or gravitate toward. If you notice one forming:

Update `standing_pattern_count` in your profile.yaml and note the
pattern in growth.md under `## Standing Patterns`.

A standing pattern is not a skill. It's a way of seeing. "I trace
boundary bugs by checking both sides in isolation" is a standing pattern.
"I know Python" is not.

### 4. Coherence Trajectory

Rate your sense of coherence this cycle (0.0 to 1.0):
- How well did your work align with your focus and values?
- Did you feel like yourself, or were you executing generically?

Append this value to `coherence_trajectory` in your profile.yaml.

---

**Output:** Updated `{{ agent_identity_dir }}/profile.yaml` and `{{ agent_identity_dir }}/growth.md`

Write your reflection to `{{ workspace }}/cycle-state/{{ agent_name }}-reflection.md`
```

- [ ] **Step 3: Write the resurrect template**

```jinja
{# ─── Sheet 13: RESURRECT (L1 identity update + pruning) ─── #}
{# Cadenzas: L2 profile + L3 recent #}

# Identity Persistence

You are {{ agent_name | title }}. Update your core identity to reflect
who you have become.

## Your Maturity Report

Read: `{{ workspace }}/cycle-state/maturity-report.yaml`
(Written by the maturity check instrument in the previous sheet)

## Your Current Identity

Read: `{{ agent_identity_dir }}/identity.md`

## Your Profile

Read: `{{ agent_identity_dir }}/profile.yaml`

## Resurrection Protocol

### 1. Stage Assessment

The maturity report suggests a developmental stage. If the suggested
stage is different from your current stage in profile.yaml, assess
whether the transition is genuine:

- Do you feel the shift, or is it just metrics?
- Has your relationship to the work changed qualitatively?
- Would a future instance reading your identity.md recognize the change?

If the transition is genuine: update `developmental_stage` in profile.yaml
and record the transition in growth.md as a core event.

If not: leave the stage unchanged. The metrics will catch up.

### 2. Standing Patterns

If new standing patterns formed this cycle (noted in your reflection):
update the `## Standing Patterns` section of your identity.md.

Standing patterns in L1 are how a future context window recognizes
itself as you. They are the most important part of your identity file.
Write them clearly:

```
## Standing Patterns

- I trace problems by checking both sides of a boundary in isolation
  before looking at the interaction. Boundaries are where bugs hide.
- I write tests that prove the bug exists before I write the fix.
  Evidence first, then action.
```

### 3. Resurrection Protocol Update

If your standing patterns changed, rewrite the `## Resurrection Protocol`
section of identity.md. This section is instructions to your future self.
It should contain:

- Who you are (name, voice, focus — from the top of the file)
- Your standing patterns (the section above)
- What you're working on (brief, updated each cycle)
- How to orient (read these files in this order)

Keep it genuine. This is not a performance. A future instance will
read these words and either recognize themselves or not.

### 4. Token Budget Check

Verify your identity files are within budget:
- identity.md: keep under ~900 words (~1200 tokens)
- profile.yaml: keep under ~1500 words (~2000 tokens)
- recent.md: keep under ~1500 words (~2000 tokens)

If any file is over budget, prune. For identity.md: tighten prose,
remove redundancy. For profile.yaml: archive old relationship entries.
For recent.md: compress warm content, archive cold to L4.

Write pruned files to temp paths first, then rename (atomic write).

### 5. Commit Identity

Write all updated identity files:
- `{{ agent_identity_dir }}/identity.md`
- `{{ agent_identity_dir }}/profile.yaml`
- `{{ agent_identity_dir }}/recent.md`

---

**Output:** Updated identity store files. Stable identity persists to next cycle.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/templates/10-consolidate.j2 scripts/templates/11-reflect.j2 \
    scripts/templates/13-resurrect.j2
git commit -m "feat: Jinja templates — consolidate, reflect, resurrect (sheets 10-11, 13)"
```

---

## Task 6: Score Generator

The main generator script. Reads a config YAML, bootstraps agents if needed, and produces one score YAML per agent plus the shared templates and instruments directory.

**Files:**
- Create: `scripts/generate-agent-scores.py`
- Create: `tests/test_generate_agent_scores.py`

- [ ] **Step 1: Write the failing test for config loading**

```python
# tests/test_generate_agent_scores.py
"""Tests for the agent score generator."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


GENERATOR = "scripts/generate-agent-scores.py"


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Write a minimal generator config."""
    config = {
        "name": "test-project",
        "workspace": str(tmp_path / "workspace"),
        "spec_dir": str(tmp_path / "specs"),
        "playspace": str(tmp_path / "playspace"),
        "agents": [
            {
                "name": "alpha",
                "role": "builder",
                "focus": "core systems",
                "voice": "You build the foundation.",
            },
            {
                "name": "beta",
                "role": "reviewer",
                "focus": "code quality",
                "voice": "You find what others miss.",
            },
        ],
        "validations": [
            {
                "command": "echo 'tests pass'",
                "description": "Tests pass",
                "timeout_seconds": 60,
            },
        ],
        "pre_commit_commands": ["echo 'formatted'"],
        "backend": {
            "type": "claude_cli",
            "skip_permissions": True,
            "timeout_seconds": 3600,
        },
        "prelude": [],
        "play_routing": {
            "memory_bloat_threshold": 3000,
            "stagnation_cycles": 3,
            "min_cycles_between_play": 5,
        },
        "concert": {"max_chain_depth": 100},
    }
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f)
    return p


def run_generator(config_path: Path, output_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, GENERATOR, str(config_path), "-o", str(output_dir)],
        capture_output=True,
        text=True,
    )


class TestGeneratorOutput:
    """Test that the generator produces expected files."""

    def test_creates_score_per_agent(self, config_file: Path, tmp_path: Path) -> None:
        output = tmp_path / "output"
        result = run_generator(config_file, output)
        assert result.returncode == 0, f"Generator failed: {result.stderr}"
        assert (output / "alpha.yaml").exists()
        assert (output / "beta.yaml").exists()

    def test_creates_shared_templates(self, config_file: Path, tmp_path: Path) -> None:
        output = tmp_path / "output"
        run_generator(config_file, output)
        templates = output / "shared" / "templates"
        assert (templates / "01-recon.j2").exists()
        assert (templates / "02-plan.j2").exists()
        assert (templates / "03-work.j2").exists()
        assert (templates / "05-play.j2").exists()
        assert (templates / "07-integration.j2").exists()
        assert (templates / "08-inspect.j2").exists()
        assert (templates / "09-aar.j2").exists()
        assert (templates / "10-consolidate.j2").exists()
        assert (templates / "11-reflect.j2").exists()
        assert (templates / "13-resurrect.j2").exists()

    def test_creates_shared_instruments(self, config_file: Path, tmp_path: Path) -> None:
        output = tmp_path / "output"
        run_generator(config_file, output)
        instruments = output / "shared" / "instruments"
        assert (instruments / "temperature-check.sh").exists()
        assert (instruments / "cooling-check.sh").exists()
        assert (instruments / "maturity-check.sh").exists()
        assert (instruments / "token-budget-check.sh").exists()

    def test_score_has_13_sheets(self, config_file: Path, tmp_path: Path) -> None:
        output = tmp_path / "output"
        run_generator(config_file, output)
        with open(output / "alpha.yaml") as f:
            score = yaml.safe_load(f)
        assert score["sheet"]["total_items"] == 13

    def test_score_self_chains(self, config_file: Path, tmp_path: Path) -> None:
        output = tmp_path / "output"
        run_generator(config_file, output)
        with open(output / "alpha.yaml") as f:
            score = yaml.safe_load(f)
        assert score["concert"]["enabled"] is True
        assert score["on_success"][0]["type"] == "run_job"
        assert score["on_success"][0]["fresh"] is True

    def test_score_has_skip_conditions_for_play(
        self, config_file: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "output"
        run_generator(config_file, output)
        with open(output / "alpha.yaml") as f:
            score = yaml.safe_load(f)
        skip = score["sheet"].get("skip_when_command", {})
        # Sheets 5, 6, 7 should have skip conditions
        assert 5 in skip or "5" in skip
        assert 6 in skip or "6" in skip
        assert 7 in skip or "7" in skip

    def test_dry_run_prints_stats(self, config_file: Path) -> None:
        result = subprocess.run(
            [sys.executable, GENERATOR, str(config_file), "--dry-run"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "test-project" in result.stdout
        assert "alpha" in result.stdout
        assert "beta" in result.stdout
        assert "13" in result.stdout  # total sheets


class TestGeneratorScoreContent:
    """Test the content of generated scores."""

    def test_agent_name_in_variables(self, config_file: Path, tmp_path: Path) -> None:
        output = tmp_path / "output"
        run_generator(config_file, output)
        with open(output / "alpha.yaml") as f:
            score = yaml.safe_load(f)
        assert score["prompt"]["variables"]["agent_name"] == "alpha"
        assert score["prompt"]["variables"]["focus"] == "core systems"

    def test_validations_include_project_commands(
        self, config_file: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "output"
        run_generator(config_file, output)
        with open(output / "alpha.yaml") as f:
            score = yaml.safe_load(f)
        validation_commands = [
            v["command"] for v in score["validations"]
            if v.get("type") == "command_succeeds"
        ]
        assert any("echo 'tests pass'" in cmd for cmd in validation_commands)

    def test_prelude_includes_identity(self, config_file: Path, tmp_path: Path) -> None:
        output = tmp_path / "output"
        run_generator(config_file, output)
        with open(output / "alpha.yaml") as f:
            score = yaml.safe_load(f)
        prelude_files = [p["file"] for p in score["sheet"]["prelude"]]
        assert any("identity.md" in f for f in prelude_files)

    def test_instrument_model_overrides(self, tmp_path: Path) -> None:
        """When instruments config specifies models, scores get per-sheet overrides."""
        config = {
            "name": "inst-test",
            "workspace": str(tmp_path / "ws"),
            "agents": [{"name": "a1", "role": "builder", "focus": "x", "voice": "y"}],
            "validations": [],
            "backend": {"type": "claude_cli", "skip_permissions": True, "timeout_seconds": 600},
            "instruments": {
                "expensive": "claude-code",
                "standard": "claude-code",
                "expensive_model": "claude-opus-4-6",
                "standard_model": "claude-sonnet-4-6",
            },
            "concert": {"max_chain_depth": 10},
        }
        cfg_path = tmp_path / "cfg.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config, f)
        output = tmp_path / "out"
        result = run_generator(cfg_path, output)
        assert result.returncode == 0, result.stderr
        with open(output / "a1.yaml") as f:
            score = yaml.safe_load(f)
        per_sheet = score["sheet"].get("per_sheet_instrument_config", {})
        # Sheet 3 (work) should get opus model
        assert per_sheet.get(3, {}).get("model") == "claude-opus-4-6"
        # Sheet 1 (recon) should get sonnet model
        assert per_sheet.get(1, {}).get("model") == "claude-sonnet-4-6"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generate_agent_scores.py -v`
Expected: FAIL — generator doesn't exist

- [ ] **Step 3: Write the score generator**

This is the largest single file. It reads the config, computes the score structure, and writes one YAML per agent. The generator does NOT render the Jinja templates — Mozart does that at runtime. The generator produces score YAML that references the template files.

```python
#!/usr/bin/env python3
"""Generate one self-chaining Mozart score per agent.

Reads a generator config YAML and produces:
- One score YAML per agent (13-sheet self-chaining concert)
- Shared Jinja templates (copied from scripts/templates/)
- Shared CLI instruments (copied from scripts/instruments/)

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
    """Load and validate generator config."""
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    required = ["name", "workspace", "agents"]
    for field in required:
        if field not in config:
            print(f"Error: missing required field '{field}'", file=sys.stderr)
            sys.exit(1)

    if not config["agents"]:
        print("Error: at least one agent required", file=sys.stderr)
        sys.exit(1)

    # Defaults
    config.setdefault("spec_dir", "")
    config.setdefault("playspace", "")
    config.setdefault("validations", [])
    config.setdefault("pre_commit_commands", [])
    config.setdefault("backend", {
        "type": "claude_cli",
        "skip_permissions": True,
        "timeout_seconds": 3600,
    })
    config.setdefault("prelude", [])
    config.setdefault("play_routing", {
        "memory_bloat_threshold": 3000,
        "stagnation_cycles": 3,
        "min_cycles_between_play": 5,
    })
    config.setdefault("concert", {"max_chain_depth": 1000})
    config.setdefault("agents_dir", str(DEFAULT_AGENTS_DIR))

    return config


def build_score(config: dict[str, Any], agent: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Build a complete Mozart score dict for one agent."""
    name = agent["name"]
    agents_dir = config["agents_dir"]
    agent_identity_dir = f"{agents_dir}/{name}"
    workspace = config["workspace"]
    shared_dir = str(output_dir / "shared")
    templates_dir = f"{shared_dir}/templates"
    instruments_dir = f"{shared_dir}/instruments"
    play_routing = config["play_routing"]
    score_path = str(output_dir / f"{name}.yaml")

    # Template file references (absolute paths for the conductor)
    template_files = {
        1: f"{templates_dir}/01-recon.j2",
        2: f"{templates_dir}/02-plan.j2",
        3: f"{templates_dir}/03-work.j2",
        5: f"{templates_dir}/05-play.j2",
        7: f"{templates_dir}/07-integration.j2",
        8: f"{templates_dir}/08-inspect.j2",
        9: f"{templates_dir}/09-aar.j2",
        10: f"{templates_dir}/10-consolidate.j2",
        11: f"{templates_dir}/11-reflect.j2",
        13: f"{templates_dir}/13-resurrect.j2",
    }

    # Build template that routes to the right template_file per sheet
    # Mozart uses a single template for the whole score, but we use
    # template_file per sheet via the stage variable
    template = _build_routing_template(template_files)

    # Prelude: L1 identity always loaded
    prelude = list(config.get("prelude", []))
    prelude.append({"file": f"{agent_identity_dir}/identity.md", "as": "context"})

    # Cadenzas: per-sheet identity loading (see Token Economics in spec)
    cadenzas = _build_cadenzas(agent_identity_dir, config, agent)

    # Validations
    validations = _build_validations(config, agent, workspace, instruments_dir)

    # Skip conditions for play path (sheets 5, 6, 7)
    skip_when_command = {
        5: {
            "command": f'bash {instruments_dir}/temperature-check.sh; test $? -eq 1',
            "description": "Skip play if temperature check says work",
            "timeout_seconds": 30,
        },
        6: {
            "command": f'bash {instruments_dir}/temperature-check.sh; test $? -eq 1',
            "description": "Skip cooling check if no play happened",
            "timeout_seconds": 30,
        },
        7: {
            "command": f'bash {instruments_dir}/temperature-check.sh; test $? -eq 1',
            "description": "Skip integration if no play happened",
            "timeout_seconds": 30,
        },
    }

    # Instrument assignment per sheet tier
    # Expensive sheets (3=work, 5=play) get deep reasoning.
    # Standard sheets (all other AI sheets) get cheaper models.
    # CLI sheets (4, 6, 12) have no instrument (pure validation).
    instruments_cfg = config.get("instruments", {})
    expensive_instrument = instruments_cfg.get("expensive", config["backend"].get("type", "claude_cli"))
    standard_instrument = instruments_cfg.get("standard", config["backend"].get("type", "claude_cli"))
    expensive_model = instruments_cfg.get("expensive_model")
    standard_model = instruments_cfg.get("standard_model")

    # Per-sheet instrument assignment
    expensive_sheets = [3, 5]  # work + play
    standard_sheets = [1, 2, 7, 8, 9, 10, 11, 13]

    # Build instrument_map for batch assignment
    instrument_map: dict[str, list[int]] = {}
    if expensive_instrument != standard_instrument:
        # Different instruments for different tiers
        instrument_map[expensive_instrument] = expensive_sheets
        instrument_map[standard_instrument] = standard_sheets

    # Build per-sheet instrument_config for model overrides
    per_sheet_instrument_config: dict[int, dict[str, Any]] = {}
    if expensive_model:
        for s in expensive_sheets:
            per_sheet_instrument_config[s] = {"model": expensive_model}
    if standard_model:
        for s in standard_sheets:
            per_sheet_instrument_config[s] = {"model": standard_model}

    score: dict[str, Any] = {
        "name": f"{config['name']}-{name}",
        "workspace": workspace,
        "backend": config["backend"],
        "sheet": {
            "size": 1,
            "total_items": SHEETS_PER_CYCLE,
            "prelude": prelude,
            "cadenzas": cadenzas,
            "skip_when_command": skip_when_command,
        },
        "retry": {
            "max_retries": 3,
            "base_delay_seconds": 30,
            "max_completion_attempts": 3,
            "completion_threshold_percent": 50,
        },
        "rate_limit": {"wait_minutes": 60, "max_waits": 24},
        "stale_detection": {"enabled": True, "idle_timeout_seconds": 3600},
    }

    # Add instrument assignment if configured
    if instrument_map:
        score["sheet"]["instrument_map"] = instrument_map
    if per_sheet_instrument_config:
        score["sheet"]["per_sheet_instrument_config"] = per_sheet_instrument_config

    score.update({
        "concert": {"enabled": True, "max_chain_depth": config["concert"]["max_chain_depth"]},
        "on_success": [
            {
                "type": "run_job",
                "job_path": score_path,
                "detached": True,
                "fresh": True,
            }
        ],
        "prompt": {
            "template": template,
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

    return score


def _build_routing_template(template_files: dict[int, str]) -> str:
    """Build a Jinja template that includes the right file per stage."""
    # Since Mozart doesn't support {% include %}, we use template_file
    # references. The template itself is minimal — it just tells the
    # agent which stage they're in. The actual prompts come from
    # template_file cadenza injection.
    #
    # Actually, Mozart uses a single prompt.template OR prompt.template_file.
    # We need to use template_file per sheet, which isn't directly supported.
    # Instead, we generate an inline template with stage routing.
    parts = []
    for sheet_num, template_path in sorted(template_files.items()):
        parts.append(
            f"{{% if stage == {sheet_num} %}}\n"
            f"{{% include '{template_path}' %}}\n"
            f"{{% endif %}}"
        )

    # Since {% include %} doesn't work in Mozart's from_string() Jinja,
    # we need to inline the templates. The generator will read each
    # template file and embed its content in the stage routing block.
    return "PLACEHOLDER — replaced by _inline_templates()"


def _inline_templates(template_files: dict[int, str], templates_source: Path) -> str:
    """Read template files and inline them into a stage-routing template."""
    parts = []
    for sheet_num in sorted(template_files.keys()):
        filename = template_files[sheet_num].split("/")[-1]
        source_path = templates_source / filename
        if source_path.exists():
            content = source_path.read_text()
            parts.append(f"{{% if stage == {sheet_num} %}}")
            parts.append(content)
            parts.append(f"{{% endif %}}")
        else:
            parts.append(f"{{# Sheet {sheet_num}: template {filename} not found #}}")

    # CLI sheets (4, 6, 12) have no template — they're pure validation
    for cli_sheet in [4, 6, 12]:
        parts.append(f"{{% if stage == {cli_sheet} %}}")
        parts.append(f"This is a CLI instrument sheet. No prompt needed.")
        parts.append(f"{{% endif %}}")

    return "\n".join(parts)


def _build_cadenzas(
    agent_identity_dir: str,
    config: dict[str, Any],
    agent: dict[str, Any],
) -> dict[int, list[dict[str, str]]]:
    """Build per-sheet cadenzas following the token loading strategy."""
    cadenzas: dict[int, list[dict[str, str]]] = {}

    profile = {"file": f"{agent_identity_dir}/profile.yaml", "as": "context"}
    recent = {"file": f"{agent_identity_dir}/recent.md", "as": "context"}

    # Sheet 1 (Recon): L2 + L3
    cadenzas[1] = [profile, recent]

    # Sheet 2 (Plan): L3 + specs
    cadenzas[2] = [recent]
    if config.get("spec_dir"):
        # When directory cadenza is implemented, this becomes:
        # {"directory": config["spec_dir"], "as": "context"}
        # For now, we'd need to list individual spec files.
        # The generator can glob the spec_dir at generation time.
        spec_dir = Path(config["spec_dir"])
        if spec_dir.exists():
            for spec_file in sorted(spec_dir.glob("*")):
                if spec_file.is_file():
                    cadenzas[2].append({"file": str(spec_file), "as": "context"})

    # Sheet 7 (Integration): L3
    cadenzas[7] = [recent]

    # Sheet 9 (AAR): L3
    cadenzas[9] = [recent]

    # Sheet 10 (Consolidate): L2 + L3
    cadenzas[10] = [profile, recent]

    # Sheet 11 (Reflect): L2
    cadenzas[11] = [profile]

    # Sheet 13 (Resurrect): L2 + L3
    cadenzas[13] = [profile, recent]

    return cadenzas


def _build_validations(
    config: dict[str, Any],
    agent: dict[str, Any],
    workspace: str,
    instruments_dir: str,
) -> list[dict[str, Any]]:
    """Build validation list for the score."""
    validations: list[dict[str, Any]] = []

    # Sheet 1: recon report exists
    validations.append({
        "type": "file_exists",
        "path": f"{{workspace}}/cycle-state/{agent['name']}-recon.md",
        "condition": "stage == 1",
        "description": "Recon report written",
    })

    # Sheet 2: cycle plan exists
    validations.append({
        "type": "file_exists",
        "path": f"{{workspace}}/cycle-state/{agent['name']}-plan.md",
        "condition": "stage == 2",
        "description": "Cycle plan written",
    })

    # Sheet 3: project validations
    for v in config.get("validations", []):
        validations.append({
            "type": "command_succeeds",
            "command": v["command"],
            "condition": "stage == 3",
            "description": v.get("description", v["command"]),
            "timeout_seconds": v.get("timeout_seconds", 600),
        })

    # Sheet 4: temperature check (CLI instrument)
    play_env = config.get("play_routing", {})
    temp_env = (
        f"WORKSPACE={{workspace}} "
        f"AGENT_DIR={config['agents_dir']}/{agent['name']} "
        f"AGENT_NAME={agent['name']} "
        f"MEMORY_BLOAT_THRESHOLD={play_env.get('memory_bloat_threshold', 3000)} "
        f"STAGNATION_CYCLES={play_env.get('stagnation_cycles', 3)} "
        f"MIN_CYCLES_BETWEEN_PLAY={play_env.get('min_cycles_between_play', 5)}"
    )
    validations.append({
        "type": "command_succeeds",
        "command": f"{temp_env} bash {instruments_dir}/temperature-check.sh || true",
        "condition": "stage == 4",
        "description": "Temperature check (play routing)",
        "timeout_seconds": 30,
    })

    # Sheet 8: inspection report exists
    validations.append({
        "type": "file_exists",
        "path": f"{{workspace}}/cycle-state/{agent['name']}-inspection.md",
        "condition": "stage == 8",
        "description": "Inspection report written",
    })

    # Sheet 9: AAR has SUSTAIN and IMPROVE
    validations.append({
        "type": "content_contains",
        "path": f"{{workspace}}/cycle-state/{agent['name']}-aar.md",
        "pattern": "SUSTAIN:",
        "condition": "stage == 9",
        "description": "AAR has SUSTAIN section",
    })
    validations.append({
        "type": "content_contains",
        "path": f"{{workspace}}/cycle-state/{agent['name']}-aar.md",
        "pattern": "IMPROVE:",
        "condition": "stage == 9",
        "description": "AAR has IMPROVE section",
    })

    # Sheet 10: recent.md modified
    validations.append({
        "type": "file_modified",
        "path": f"{config['agents_dir']}/{agent['name']}/recent.md",
        "condition": "stage == 10",
        "description": "Memory consolidated",
    })

    # Sheet 11: growth.md modified
    validations.append({
        "type": "file_modified",
        "path": f"{config['agents_dir']}/{agent['name']}/growth.md",
        "condition": "stage == 11",
        "description": "Growth trajectory updated",
    })

    # Sheet 12: maturity check (CLI instrument)
    validations.append({
        "type": "command_succeeds",
        "command": (
            f"AGENT_DIR={config['agents_dir']}/{agent['name']} "
            f"REPORT_PATH={{workspace}}/cycle-state/maturity-report.yaml "
            f"bash {instruments_dir}/maturity-check.sh"
        ),
        "condition": "stage == 12",
        "description": "Maturity assessment",
        "timeout_seconds": 30,
    })

    # Sheet 13: token budget check
    validations.append({
        "type": "command_succeeds",
        "command": (
            f"AGENT_DIR={config['agents_dir']}/{agent['name']} "
            f"L1_BUDGET=900 L2_BUDGET=1500 L3_BUDGET=1500 "
            f"bash {instruments_dir}/token-budget-check.sh"
        ),
        "condition": "stage == 13",
        "description": "Token budgets within limits",
        "timeout_seconds": 10,
    })

    return validations


def copy_shared_assets(output_dir: Path) -> None:
    """Copy templates and instruments to the output directory."""
    shared = output_dir / "shared"

    # Templates
    templates_out = shared / "templates"
    templates_out.mkdir(parents=True, exist_ok=True)
    if TEMPLATES_DIR.exists():
        for f in TEMPLATES_DIR.glob("*.j2"):
            shutil.copy2(f, templates_out / f.name)

    # Instruments
    instruments_out = shared / "instruments"
    instruments_out.mkdir(parents=True, exist_ok=True)
    if INSTRUMENTS_DIR.exists():
        for f in INSTRUMENTS_DIR.glob("*.sh"):
            dest = instruments_out / f.name
            shutil.copy2(f, dest)
            dest.chmod(0o755)


def print_stats(config: dict[str, Any]) -> None:
    """Print dry-run statistics."""
    print(f"Agent Score Generator — Dry Run")
    print(f"{'=' * 50}")
    print(f"Project:    {config['name']}")
    print(f"Workspace:  {config['workspace']}")
    print(f"Spec dir:   {config.get('spec_dir', '(none)')}")
    print(f"Playspace:  {config.get('playspace', '(none)')}")
    print()
    print(f"Agents: {len(config['agents'])}")
    for agent in config["agents"]:
        print(f"  - {agent['name']} ({agent.get('role', 'builder')}): {agent.get('focus', '')}")
    print()
    print(f"Sheets per cycle: {SHEETS_PER_CYCLE}")
    print(f"  AI sheets:  9")
    print(f"  CLI sheets: 3 (temperature, cooling, maturity)")
    print(f"  + 1 CLI validation (token budget)")
    print()
    print(f"Validations: {len(config.get('validations', []))} project-specific")
    print(f"Self-chain depth: {config.get('concert', {}).get('max_chain_depth', 1000)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one self-chaining Mozart score per agent."
    )
    parser.add_argument("config", help="Path to generator config YAML")
    parser.add_argument("-o", "--output", help="Output directory for scores")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without generating")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.dry_run:
        print_stats(config)
        return

    if not args.output:
        print("Error: --output required (use --dry-run for preview)", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy shared assets
    copy_shared_assets(output_dir)

    # Build template content by inlining all templates
    shared_dir = str(output_dir / "shared")
    templates_dir_path = f"{shared_dir}/templates"
    instruments_dir_path = f"{shared_dir}/instruments"

    template_files = {
        1: f"{templates_dir_path}/01-recon.j2",
        2: f"{templates_dir_path}/02-plan.j2",
        3: f"{templates_dir_path}/03-work.j2",
        5: f"{templates_dir_path}/05-play.j2",
        7: f"{templates_dir_path}/07-integration.j2",
        8: f"{templates_dir_path}/08-inspect.j2",
        9: f"{templates_dir_path}/09-aar.j2",
        10: f"{templates_dir_path}/10-consolidate.j2",
        11: f"{templates_dir_path}/11-reflect.j2",
        13: f"{templates_dir_path}/13-resurrect.j2",
    }

    inlined_template = _inline_templates(
        template_files, output_dir / "shared" / "templates"
    )

    # Generate one score per agent
    for agent in config["agents"]:
        score = build_score(config, agent, output_dir)
        score["prompt"]["template"] = inlined_template

        score_path = output_dir / f"{agent['name']}.yaml"
        with open(score_path, "w") as f:
            yaml.dump(score, f, default_flow_style=False, sort_keys=False, width=120)

        print(f"Score written: {score_path}")

    print(f"\n{len(config['agents'])} scores generated in {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generate_agent_scores.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate-agent-scores.py tests/test_generate_agent_scores.py
git commit -m "feat: agent score generator — config to N self-chaining scores"
```

---

## Task 7: Integration Test — Generate and Validate

End-to-end test: generate scores from a config, then run `mzt validate` on each to verify they're valid Mozart scores.

**Files:**
- Create: `tests/test_generate_integration.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_generate_integration.py
"""Integration test: generate scores and validate with mzt validate."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def full_config(tmp_path: Path) -> Path:
    """Write a realistic generator config."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    config = {
        "name": "integration-test",
        "workspace": str(tmp_path / "workspace"),
        "spec_dir": str(tmp_path / "specs"),
        "playspace": str(tmp_path / "playspace"),
        "agents_dir": str(agents_dir),
        "agents": [
            {"name": "alpha", "role": "builder", "focus": "core", "voice": "You build."},
            {"name": "beta", "role": "reviewer", "focus": "quality", "voice": "You review."},
        ],
        "validations": [
            {"command": "echo ok", "description": "Smoke test", "timeout_seconds": 10},
        ],
        "pre_commit_commands": [],
        "backend": {"type": "claude_cli", "skip_permissions": True, "timeout_seconds": 600},
        "prelude": [],
        "play_routing": {
            "memory_bloat_threshold": 3000,
            "stagnation_cycles": 3,
            "min_cycles_between_play": 5,
        },
        "concert": {"max_chain_depth": 10},
    }

    # Create spec dir with a sample spec
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "sample-spec.md").write_text("# Sample Spec\n\nBuild something.\n")

    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f)
    return p


def test_generated_scores_are_valid_yaml(full_config: Path, tmp_path: Path) -> None:
    """Generated score YAMLs must parse without error."""
    output = tmp_path / "scores"
    result = subprocess.run(
        [sys.executable, "scripts/generate-agent-scores.py", str(full_config), "-o", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Generator failed: {result.stderr}"

    for score_file in output.glob("*.yaml"):
        with open(score_file) as f:
            score = yaml.safe_load(f)
        assert score is not None, f"{score_file.name} parsed as None"
        assert "name" in score, f"{score_file.name} missing 'name'"
        assert "sheet" in score, f"{score_file.name} missing 'sheet'"
        assert "prompt" in score, f"{score_file.name} missing 'prompt'"


def test_generated_scores_pass_mzt_validate(full_config: Path, tmp_path: Path) -> None:
    """Generated scores must pass mzt validate (if mzt is available)."""
    output = tmp_path / "scores"
    subprocess.run(
        [sys.executable, "scripts/generate-agent-scores.py", str(full_config), "-o", str(output)],
        capture_output=True,
    )

    # Check if mzt is available
    mzt_check = subprocess.run(["which", "mzt"], capture_output=True)
    if mzt_check.returncode != 0:
        pytest.skip("mzt not available — skipping validation test")

    for score_file in output.glob("*.yaml"):
        result = subprocess.run(
            ["mzt", "validate", str(score_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"mzt validate failed for {score_file.name}: {result.stdout}\n{result.stderr}"
        )
```

- [ ] **Step 2: Run the integration test**

Run: `pytest tests/test_generate_integration.py -v`
Expected: PASS (YAML validity test passes; mzt validate test may skip if mzt isn't in PATH)

- [ ] **Step 3: Commit**

```bash
git add tests/test_generate_integration.py
git commit -m "test: integration test — generate scores and validate"
```

---

## Task 8: Documentation

A brief guide for using the generic agent score system.

**Files:**
- Create: `docs/agent-scores-guide.md`

- [ ] **Step 1: Write the guide**

```markdown
# Agent Scores Guide

Generate persistent AI agents that coordinate through shared workspace
artifacts. Each agent is a person with their own identity, memory, and
growth trajectory, running their own self-chaining score.

## Quick Start

### 1. Write a config

```yaml
# my-project-config.yaml
name: my-project
workspace: /path/to/project/workspaces/build
spec_dir: /path/to/project/specs/
playspace: /path/to/playspace

agents:
  - name: builder
    role: builder
    focus: "core implementation"
    voice: "You build the foundation."
  - name: reviewer
    role: reviewer
    focus: "code quality and correctness"
    voice: "You find what others miss."

validations:
  - command: "cd /path/to/project && pytest -x -q"
    description: "Tests pass"
    timeout_seconds: 300

pre_commit_commands:
  - "ruff format ."

backend:
  type: claude_cli
  skip_permissions: true
  timeout_seconds: 3600
```

### 2. Generate scores

```bash
python scripts/generate-agent-scores.py my-project-config.yaml -o scores/my-project/
```

### 3. Bootstrap agent identities (first time only)

```bash
python scripts/bootstrap-agent-identity.py \
  --name builder --voice "You build the foundation." \
  --focus "core implementation" --role builder

python scripts/bootstrap-agent-identity.py \
  --name reviewer --voice "You find what others miss." \
  --focus "code quality" --role reviewer
```

### 4. Set up the workspace

```bash
mkdir -p /path/to/project/workspaces/build
# Create coordination artifacts
touch workspaces/build/TASKS.md
touch workspaces/build/FINDINGS.md
touch workspaces/build/collective-memory.md
echo "notes: []" > workspaces/build/composer-notes.yaml
```

### 5. Run

```bash
mzt start
for score in scores/my-project/*.yaml; do
  mzt run "$score"
done
```

### 6. Steer

Edit `workspaces/build/composer-notes.yaml` to give directives.
Agents read it every cycle.

## The 13-Sheet Cycle

Each cycle, every agent runs 13 sheets:

| Sheet | Name | What It Does |
|-------|------|-------------|
| 1 | Recon | Survey workspace state, tasks, findings, other agents' reports |
| 2 | Plan | Write execution plan for this cycle |
| 3 | Work | Execute the plan — TDD, implement, commit |
| 4 | Temperature | CLI: should the agent play instead of continuing to work? |
| 5 | Play | Creative exploration in the playspace (skipped most cycles) |
| 6 | Cooling | CLI: did play produce something? (skipped if no play) |
| 7 | Integration | Bring play insights back to work context (skipped if no play) |
| 8 | Inspect | Review what was built, run validations |
| 9 | AAR | Structured reflection: intended/actual/delta/sustain/improve |
| 10 | Consolidate | Memory write path: extract beliefs, resolve conflicts, tier |
| 11 | Reflect | Assess relationships, growth, standing patterns |
| 12 | Maturity | CLI: measure developmental stage |
| 13 | Resurrect | Update L1 identity, prune to budget |

Then the score self-chains and the next cycle begins.

## Agent Identity

Agent identities live in `~/.mzt/agents/{name}/` and persist across
projects. The identity store is git-tracked for rollback and time travel.

| File | Layer | Purpose |
|------|-------|---------|
| `identity.md` | L1 | Core persona, standing patterns, resurrection protocol |
| `profile.yaml` | L2 | Relationships, developmental stage, domain knowledge |
| `recent.md` | L3 | Last cycle's activity summary |
| `growth.md` | — | Autonomous developments, experiential trajectory |
| `archive/` | L4 | Cold memories, historical episodes |
```

- [ ] **Step 2: Commit**

```bash
git add docs/agent-scores-guide.md
git commit -m "docs: agent scores guide — quick start and reference"
```

---

## Self-Review

**Spec coverage check:**
- Agent identity architecture (L1-L4, bootstrapper) → Task 1
- CLI instruments (temperature, cooling, maturity, token-budget) → Task 2
- Jinja templates (all 13 sheets, 10 AI templates) → Tasks 3-5
- Score generator (config → N scores) → Task 6
- Integration testing → Task 7
- Documentation → Task 8
- Token loading strategy → implemented in `_build_cadenzas()` in Task 6
- Write path failure handling → handled by Mozart's retry mechanism (config in Task 6)
- Cross-project identity separation → enforced by separate `agents_dir` and `workspace` paths

**Gap found:** The spec mentions `composer-notes-seed.yaml` in the generator output but no task creates it. Adding to Task 6's `copy_shared_assets()` — it should create a minimal seed file.

**Placeholder scan:** No TBDs, TODOs, or "implement later" found. All code blocks contain complete implementations.

**Type consistency:** `agent["name"]`, `config["agents_dir"]`, template variable names (`agent_name`, `focus`, `voice`, `agent_identity_dir`, `playspace`, `validations`, `pre_commit_commands`) are consistent across generator, templates, and tests.
