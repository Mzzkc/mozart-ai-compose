# Evolution Concert Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the self-evolution loop: `mozart learning export` CLI command, vision prelude document, and the `evolution-concert.yaml` score.

**Architecture:** Two new CLI commands (`learning-export`, `learning-record-evolution`) as thin wrappers around existing learning store queries. One prelude markdown file for context shaping. One YAML score file with 10 stages, 6 fan-out stages, TDF-aligned analysis, and two review/fix cycles.

**Tech Stack:** Typer CLI, SQLite (via GlobalLearningStore), YAML (Mozart score format), Jinja2 templates.

**Design doc:** `docs/plans/2026-02-24-evolution-concert-design.md`

---

## Post-Expansion Sheet Numbering Reference

The score has 10 logical stages. After fan-out expansion:

```
Sheet 1  → Stage 1  (Extract)
Sheet 2  → Stage 2, instance 1 (COMP)
Sheet 3  → Stage 2, instance 2 (SCI)
Sheet 4  → Stage 2, instance 3 (CULT)
Sheet 5  → Stage 2, instance 4 (EXP)
Sheet 6  → Stage 2, instance 5 (META)
Sheet 7  → Stage 3  (Synthesize)
Sheet 8  → Stage 4, instance 1 (Implement candidate 1)
Sheet 9  → Stage 4, instance 2 (Implement candidate 2)
Sheet 10 → Stage 4, instance 3 (Implement candidate 3)
Sheet 11 → Stage 4, instance 4 (Implement candidate 4)
Sheet 12 → Stage 4, instance 5 (Implement candidate 5)
Sheet 13 → Stage 5, instance 1 (Review R1 candidate 1)
Sheet 14 → Stage 5, instance 2 (Review R1 candidate 2)
Sheet 15 → Stage 5, instance 3 (Review R1 candidate 3)
Sheet 16 → Stage 5, instance 4 (Review R1 candidate 4)
Sheet 17 → Stage 5, instance 5 (Review R1 candidate 5)
Sheet 18 → Stage 6, instance 1 (Fix R1 candidate 1)
Sheet 19 → Stage 6, instance 2 (Fix R1 candidate 2)
Sheet 20 → Stage 6, instance 3 (Fix R1 candidate 3)
Sheet 21 → Stage 6, instance 4 (Fix R1 candidate 4)
Sheet 22 → Stage 6, instance 5 (Fix R1 candidate 5)
Sheet 23 → Stage 7, instance 1 (Review R2 candidate 1)
Sheet 24 → Stage 7, instance 2 (Review R2 candidate 2)
Sheet 25 → Stage 7, instance 3 (Review R2 candidate 3)
Sheet 26 → Stage 7, instance 4 (Review R2 candidate 4)
Sheet 27 → Stage 7, instance 5 (Review R2 candidate 5)
Sheet 28 → Stage 8, instance 1 (Fix R2 candidate 1)
Sheet 29 → Stage 8, instance 2 (Fix R2 candidate 2)
Sheet 30 → Stage 8, instance 3 (Fix R2 candidate 3)
Sheet 31 → Stage 8, instance 4 (Fix R2 candidate 4)
Sheet 32 → Stage 8, instance 5 (Fix R2 candidate 5)
Sheet 33 → Stage 9  (Integrate)
Sheet 34 → Stage 10 (Commit + Log)
```

Total: 34 sheets. Dependencies expand via N→N instance-match for stages 5-8.

---

### Task 1: `learning-export` command — test

**Files:**
- Create: `tests/test_cli_learning_export.py`

**Step 1: Write the test file**

Tests mock the learning store and verify files are written with correct structure.

```python
"""Tests for mozart learning-export CLI command."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from mozart.cli import app

runner = CliRunner()

_GS_PATCH = "mozart.learning.global_store.get_global_store"


def _make_pattern(
    *,
    pattern_id: str = "p1",
    pattern_type: str = "SEMANTIC_INSIGHT",
    pattern_name: str = "test_pattern",
    description: str = "Test insight description",
    effectiveness_score: float = 0.75,
    trust_score: float = 0.8,
    occurrence_count: int = 5,
    category: str = "root_cause",
    tags: list[str] | None = None,
) -> MagicMock:
    p = MagicMock()
    p.id = pattern_id
    p.pattern_type = pattern_type
    p.pattern_name = pattern_name
    p.description = description
    p.effectiveness_score = effectiveness_score
    p.trust_score = trust_score
    p.occurrence_count = occurrence_count
    p.variance = 0.1
    p.first_seen = "2026-02-01T00:00:00"
    p.last_seen = "2026-02-20T00:00:00"
    p.quarantine_status = "VALIDATED"
    p.context_tags = tags or [category]
    p.led_to_success_count = 4
    p.led_to_failure_count = 1
    return p


def _make_drift_metrics(
    *,
    pattern_id: str = "p1",
    pattern_name: str = "drifting_pattern",
    effectiveness_before: float = 0.8,
    effectiveness_after: float = 0.5,
    drift_magnitude: float = 0.3,
    drift_direction: str = "negative",
) -> MagicMock:
    m = MagicMock()
    m.pattern_id = pattern_id
    m.pattern_name = pattern_name
    m.effectiveness_before = effectiveness_before
    m.effectiveness_after = effectiveness_after
    m.drift_magnitude = drift_magnitude
    m.drift_direction = drift_direction
    m.grounding_confidence_avg = 0.7
    m.applications_analyzed = 10
    return m


def _make_trajectory_entry(
    *,
    cycle: int = 25,
    evolutions_completed: int = 2,
    evolutions_deferred: int = 1,
    issue_classes: list[str] | None = None,
    implementation_loc: int = 150,
    test_loc: int = 200,
) -> MagicMock:
    e = MagicMock()
    e.cycle = cycle
    e.recorded_at = "2026-02-20T00:00:00"
    e.evolutions_completed = evolutions_completed
    e.evolutions_deferred = evolutions_deferred
    e.issue_classes = issue_classes or ["infrastructure_activation"]
    e.cv_avg = 0.78
    e.implementation_loc = implementation_loc
    e.test_loc = test_loc
    e.loc_accuracy = 0.95
    e.research_candidates_resolved = 1
    e.research_candidates_created = 0
    e.notes = "Test cycle"
    return e


class TestLearningExport:
    """Tests for learning-export command."""

    @patch(_GS_PATCH)
    def test_creates_output_directory(
        self, mock_get_store: MagicMock, tmp_path: Path
    ) -> None:
        mock_store = MagicMock()
        mock_store.get_patterns.return_value = [_make_pattern()]
        mock_store.get_drifting_patterns.return_value = []
        mock_store.get_epistemic_drifting_patterns.return_value = []
        mock_store.calculate_pattern_entropy.return_value = MagicMock(
            shannon_entropy=2.5,
            diversity_index=0.7,
            unique_pattern_count=10,
            effective_pattern_count=8,
            dominant_pattern_share=0.15,
        )
        mock_store.get_entropy_alerts.return_value = []
        mock_store.get_trajectory.return_value = []
        mock_store.get_execution_stats.return_value = {}
        mock_get_store.return_value = mock_store

        out_dir = tmp_path / "export"
        result = runner.invoke(
            app, ["learning-export", "--output-dir", str(out_dir)]
        )
        assert result.exit_code == 0, result.stdout
        assert out_dir.exists()

    @patch(_GS_PATCH)
    def test_writes_semantic_insights_file(
        self, mock_get_store: MagicMock, tmp_path: Path
    ) -> None:
        mock_store = MagicMock()
        mock_store.get_patterns.return_value = [
            _make_pattern(pattern_id="p1", category="root_cause"),
            _make_pattern(pattern_id="p2", category="knowledge"),
        ]
        mock_store.get_drifting_patterns.return_value = []
        mock_store.get_epistemic_drifting_patterns.return_value = []
        mock_store.calculate_pattern_entropy.return_value = MagicMock(
            shannon_entropy=2.5, diversity_index=0.7,
            unique_pattern_count=10, effective_pattern_count=8,
            dominant_pattern_share=0.15,
        )
        mock_store.get_entropy_alerts.return_value = []
        mock_store.get_trajectory.return_value = []
        mock_store.get_execution_stats.return_value = {}
        mock_get_store.return_value = mock_store

        out_dir = tmp_path / "export"
        result = runner.invoke(
            app, ["learning-export", "--output-dir", str(out_dir)]
        )
        assert result.exit_code == 0, result.stdout

        insights_file = out_dir / "semantic-insights.md"
        assert insights_file.exists()
        content = insights_file.read_text()
        assert "SEMANTIC_INSIGHT" in content or "Semantic Insights" in content
        assert "test_pattern" in content

    @patch(_GS_PATCH)
    def test_writes_drift_report(
        self, mock_get_store: MagicMock, tmp_path: Path
    ) -> None:
        mock_store = MagicMock()
        mock_store.get_patterns.return_value = []
        mock_store.get_drifting_patterns.return_value = [
            _make_drift_metrics()
        ]
        mock_store.get_epistemic_drifting_patterns.return_value = []
        mock_store.calculate_pattern_entropy.return_value = MagicMock(
            shannon_entropy=2.5, diversity_index=0.7,
            unique_pattern_count=10, effective_pattern_count=8,
            dominant_pattern_share=0.15,
        )
        mock_store.get_entropy_alerts.return_value = []
        mock_store.get_trajectory.return_value = []
        mock_store.get_execution_stats.return_value = {}
        mock_get_store.return_value = mock_store

        out_dir = tmp_path / "export"
        result = runner.invoke(
            app, ["learning-export", "--output-dir", str(out_dir)]
        )
        assert result.exit_code == 0, result.stdout
        assert (out_dir / "drift-report.md").exists()

    @patch(_GS_PATCH)
    def test_writes_all_six_files(
        self, mock_get_store: MagicMock, tmp_path: Path
    ) -> None:
        mock_store = MagicMock()
        mock_store.get_patterns.return_value = []
        mock_store.get_drifting_patterns.return_value = []
        mock_store.get_epistemic_drifting_patterns.return_value = []
        mock_store.calculate_pattern_entropy.return_value = MagicMock(
            shannon_entropy=2.5, diversity_index=0.7,
            unique_pattern_count=10, effective_pattern_count=8,
            dominant_pattern_share=0.15,
        )
        mock_store.get_entropy_alerts.return_value = []
        mock_store.get_trajectory.return_value = []
        mock_store.get_execution_stats.return_value = {}
        mock_get_store.return_value = mock_store

        out_dir = tmp_path / "export"
        result = runner.invoke(
            app, ["learning-export", "--output-dir", str(out_dir)]
        )
        assert result.exit_code == 0, result.stdout

        expected_files = [
            "semantic-insights.md",
            "drift-report.md",
            "entropy-state.md",
            "pattern-health.md",
            "evolution-history.md",
            "error-landscape.md",
        ]
        for fname in expected_files:
            assert (out_dir / fname).exists(), f"Missing: {fname}"

    @patch(_GS_PATCH)
    def test_evolution_history_includes_trajectory(
        self, mock_get_store: MagicMock, tmp_path: Path
    ) -> None:
        mock_store = MagicMock()
        mock_store.get_patterns.return_value = []
        mock_store.get_drifting_patterns.return_value = []
        mock_store.get_epistemic_drifting_patterns.return_value = []
        mock_store.calculate_pattern_entropy.return_value = MagicMock(
            shannon_entropy=2.5, diversity_index=0.7,
            unique_pattern_count=10, effective_pattern_count=8,
            dominant_pattern_share=0.15,
        )
        mock_store.get_entropy_alerts.return_value = []
        mock_store.get_trajectory.return_value = [
            _make_trajectory_entry(cycle=25),
            _make_trajectory_entry(cycle=24, evolutions_completed=3),
        ]
        mock_store.get_execution_stats.return_value = {}
        mock_get_store.return_value = mock_store

        out_dir = tmp_path / "export"
        result = runner.invoke(
            app, ["learning-export", "--output-dir", str(out_dir)]
        )
        assert result.exit_code == 0, result.stdout

        content = (out_dir / "evolution-history.md").read_text()
        assert "25" in content  # cycle number
        assert "24" in content

    @patch(_GS_PATCH)
    def test_json_format_writes_json_files(
        self, mock_get_store: MagicMock, tmp_path: Path
    ) -> None:
        mock_store = MagicMock()
        mock_store.get_patterns.return_value = [_make_pattern()]
        mock_store.get_drifting_patterns.return_value = []
        mock_store.get_epistemic_drifting_patterns.return_value = []
        mock_store.calculate_pattern_entropy.return_value = MagicMock(
            shannon_entropy=2.5, diversity_index=0.7,
            unique_pattern_count=10, effective_pattern_count=8,
            dominant_pattern_share=0.15,
        )
        mock_store.get_entropy_alerts.return_value = []
        mock_store.get_trajectory.return_value = []
        mock_store.get_execution_stats.return_value = {}
        mock_get_store.return_value = mock_store

        out_dir = tmp_path / "export"
        result = runner.invoke(
            app,
            ["learning-export", "--output-dir", str(out_dir), "--format", "json"],
        )
        assert result.exit_code == 0, result.stdout
        assert (out_dir / "semantic-insights.json").exists()
        # Verify valid JSON
        data = json.loads((out_dir / "semantic-insights.json").read_text())
        assert isinstance(data, (list, dict))


class TestLearningRecordEvolution:
    """Tests for learning-record-evolution command."""

    @patch(_GS_PATCH)
    def test_records_evolution_entry(
        self, mock_get_store: MagicMock
    ) -> None:
        mock_store = MagicMock()
        mock_store.record_evolution_trajectory.return_value = "uuid-123"
        mock_get_store.return_value = mock_store

        result = runner.invoke(
            app,
            [
                "learning-record-evolution",
                "--cycle", "26",
                "--evolutions-completed", "2",
                "--issue-classes", "infrastructure_activation,testing_depth",
                "--implementation-loc", "150",
                "--test-loc", "200",
            ],
        )
        assert result.exit_code == 0, result.stdout
        mock_store.record_evolution_trajectory.assert_called_once()

    @patch(_GS_PATCH)
    def test_rejects_missing_required_fields(
        self, mock_get_store: MagicMock
    ) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        result = runner.invoke(
            app, ["learning-record-evolution", "--cycle", "26"]
        )
        # Should fail due to missing required options
        assert result.exit_code != 0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/emzi/Projects/mozart-ai-compose && python -m pytest tests/test_cli_learning_export.py -v --tb=short 2>&1 | head -30`
Expected: FAIL — `learning-export` and `learning-record-evolution` commands don't exist yet.

---

### Task 2: `learning-export` command — implementation

**Files:**
- Create: `src/mozart/cli/commands/learning/_export.py`
- Modify: `src/mozart/cli/commands/learning/__init__.py`
- Modify: `src/mozart/cli/__init__.py` (register commands)

**Step 1: Write the export module**

```python
"""Learning data export and evolution recording commands.

Commands:
- learning-export: Export learning store data to workspace files
- learning-record-evolution: Record an evolution cycle in the trajectory table
"""
from __future__ import annotations

import json as json_lib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from ...output import console


def _write_file(path: Path, content: str) -> None:
    """Write content to file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _format_markdown_insights(patterns: list) -> str:
    """Format SEMANTIC_INSIGHT patterns as structured markdown."""
    lines = ["# Semantic Insights\n"]

    if not patterns:
        lines.append("No semantic insights found in the learning store.\n")
        return "\n".join(lines)

    # Group by tags (category)
    categories: dict[str, list] = {}
    for p in patterns:
        tags = getattr(p, "context_tags", None) or []
        cat = tags[0] if tags else "uncategorized"
        categories.setdefault(cat, []).append(p)

    for cat, cat_patterns in sorted(categories.items()):
        lines.append(f"## {cat.replace('_', ' ').title()}\n")
        for p in cat_patterns:
            lines.append(f"### {p.pattern_name}")
            lines.append(f"- **ID:** `{p.id}`")
            lines.append(
                f"- **Effectiveness:** {p.effectiveness_score:.1%}"
                if p.effectiveness_score
                else "- **Effectiveness:** N/A"
            )
            lines.append(
                f"- **Trust:** {p.trust_score:.2f}"
                if p.trust_score is not None
                else "- **Trust:** N/A"
            )
            lines.append(f"- **Occurrences:** {p.occurrence_count}")
            lines.append(
                f"- **Success/Failure:** "
                f"{getattr(p, 'led_to_success_count', 0)}/"
                f"{getattr(p, 'led_to_failure_count', 0)}"
            )
            lines.append(
                f"- **Quarantine:** "
                f"{getattr(p, 'quarantine_status', 'unknown')}"
            )
            if p.description:
                lines.append(f"\n{p.description}\n")
            lines.append("")

    return "\n".join(lines)


def _format_markdown_drift(effectiveness_drift: list, epistemic_drift: list) -> str:
    """Format drift data as structured markdown."""
    lines = ["# Drift Report\n"]

    lines.append("## Effectiveness Drift\n")
    if not effectiveness_drift:
        lines.append("No patterns with significant effectiveness drift.\n")
    else:
        for m in effectiveness_drift:
            lines.append(f"### {m.pattern_name}")
            lines.append(f"- **Pattern ID:** `{m.pattern_id}`")
            lines.append(f"- **Before:** {m.effectiveness_before:.1%}")
            lines.append(f"- **After:** {m.effectiveness_after:.1%}")
            lines.append(f"- **Drift:** {m.drift_magnitude:.1%} ({m.drift_direction})")
            lines.append(
                f"- **Grounding:** {m.grounding_confidence_avg:.1%}"
            )
            lines.append(f"- **Applications analyzed:** {m.applications_analyzed}")
            lines.append("")

    lines.append("## Epistemic Drift\n")
    if not epistemic_drift:
        lines.append("No patterns with significant epistemic drift.\n")
    else:
        for m in epistemic_drift:
            lines.append(f"### {m.pattern_name}")
            lines.append(f"- **Pattern ID:** `{m.pattern_id}`")
            lines.append(f"- **Confidence before:** {m.confidence_before:.1%}")
            lines.append(f"- **Confidence after:** {m.confidence_after:.1%}")
            lines.append(f"- **Change:** {m.belief_change:+.1%} ({m.drift_direction})")
            lines.append(f"- **Entropy:** {m.belief_entropy:.3f}")
            lines.append("")

    return "\n".join(lines)


def _format_markdown_entropy(entropy_metrics, alerts: list) -> str:
    """Format entropy state as structured markdown."""
    lines = ["# Entropy State\n"]

    lines.append("## Current Metrics\n")
    lines.append(f"- **Shannon entropy:** {entropy_metrics.shannon_entropy:.3f}")
    lines.append(f"- **Diversity index:** {entropy_metrics.diversity_index:.3f}")
    lines.append(f"- **Unique patterns:** {entropy_metrics.unique_pattern_count}")
    lines.append(f"- **Effective patterns:** {entropy_metrics.effective_pattern_count}")
    lines.append(f"- **Dominant pattern share:** {entropy_metrics.dominant_pattern_share:.1%}")
    lines.append("")

    lines.append("## Recent Alerts\n")
    if not alerts:
        lines.append("No entropy alerts.\n")
    else:
        for a in alerts:
            lines.append(f"- {a}")
    lines.append("")

    return "\n".join(lines)


def _format_markdown_health(patterns: list) -> str:
    """Format pattern health data as structured markdown."""
    lines = ["# Pattern Health\n"]

    quarantined = [p for p in patterns if getattr(p, "quarantine_status", "") == "QUARANTINED"]
    low_trust = [p for p in patterns if (p.trust_score or 1.0) < 0.3]
    high_variance = [p for p in patterns if (p.variance or 0) > 0.5]
    zero_apps = [p for p in patterns if p.occurrence_count == 0]

    lines.append(f"## Quarantined Patterns ({len(quarantined)})\n")
    for p in quarantined:
        lines.append(f"- **{p.pattern_name}** (`{p.id}`): {p.description or 'no description'}")
    if not quarantined:
        lines.append("None.\n")
    lines.append("")

    lines.append(f"## Low Trust Patterns ({len(low_trust)})\n")
    for p in low_trust:
        lines.append(
            f"- **{p.pattern_name}** (`{p.id}`): "
            f"trust={p.trust_score:.2f}, eff={p.effectiveness_score:.1%}"
        )
    if not low_trust:
        lines.append("None.\n")
    lines.append("")

    lines.append(f"## High Variance Patterns ({len(high_variance)})\n")
    for p in high_variance:
        lines.append(
            f"- **{p.pattern_name}** (`{p.id}`): variance={p.variance:.3f}"
        )
    if not high_variance:
        lines.append("None.\n")
    lines.append("")

    lines.append(f"## Zero-Application Patterns ({len(zero_apps)})\n")
    for p in zero_apps:
        lines.append(f"- **{p.pattern_name}** (`{p.id}`): {p.description or 'no description'}")
    if not zero_apps:
        lines.append("None.\n")

    return "\n".join(lines)


def _format_markdown_evolution(entries: list) -> str:
    """Format evolution trajectory as structured markdown."""
    lines = ["# Evolution History\n"]

    if not entries:
        lines.append("No evolution trajectory entries found.\n")
        return "\n".join(lines)

    for e in entries:
        lines.append(f"## Cycle {e.cycle}")
        lines.append(f"- **Date:** {e.recorded_at}")
        lines.append(f"- **Evolutions completed:** {e.evolutions_completed}")
        lines.append(f"- **Evolutions deferred:** {e.evolutions_deferred}")
        lines.append(f"- **Issue classes:** {', '.join(e.issue_classes)}")
        lines.append(f"- **CV avg:** {e.cv_avg:.3f}")
        lines.append(f"- **Implementation LOC:** {e.implementation_loc}")
        lines.append(f"- **Test LOC:** {e.test_loc}")
        lines.append(f"- **LOC accuracy:** {e.loc_accuracy:.1%}")
        if e.notes:
            lines.append(f"- **Notes:** {e.notes}")
        lines.append("")

    return "\n".join(lines)


def _format_markdown_errors(stats: dict) -> str:
    """Format error landscape as structured markdown."""
    lines = ["# Error Landscape\n"]

    total_exec = stats.get("total_executions", 0)
    lines.append(f"## Execution Summary\n")
    lines.append(f"- **Total executions:** {total_exec}")
    success_rate = stats.get("success_without_retry_rate", 0) * 100
    lines.append(f"- **First-attempt success rate:** {success_rate:.1f}%")
    lines.append(f"- **Total error recoveries:** {stats.get('total_error_recoveries', 0)}")
    recovery_rate = stats.get("error_recovery_success_rate", 0) * 100
    lines.append(f"- **Recovery success rate:** {recovery_rate:.1f}%")
    lines.append("")

    return "\n".join(lines)


def learning_export(
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Directory to write export files"),
    ] = "./learning-export",
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: markdown or json"),
    ] = "markdown",
    since: Annotated[
        int,
        typer.Option("--since", "-s", help="Export data from last N days"),
    ] = 30,
) -> None:
    """Export learning store data to workspace files.

    Writes structured files for consumption by evolution scores:
    semantic-insights, drift-report, entropy-state, pattern-health,
    evolution-history, error-landscape.

    Examples:
        mozart learning-export --output-dir ./workspace/learning
        mozart learning-export --format json --since 60
    """
    from mozart.learning.global_store import get_global_store

    store = get_global_store()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ext = "json" if fmt == "json" else "md"

    # 1. Semantic insights
    insights = store.get_patterns(
        pattern_type="SEMANTIC_INSIGHT",
        min_priority=0.0,
        limit=200,
    )
    if fmt == "json":
        data = [
            {
                "id": p.id,
                "name": p.pattern_name,
                "type": p.pattern_type,
                "description": p.description,
                "effectiveness": p.effectiveness_score,
                "trust": p.trust_score,
                "occurrences": p.occurrence_count,
                "tags": getattr(p, "context_tags", []),
                "quarantine_status": getattr(p, "quarantine_status", None),
            }
            for p in insights
        ]
        _write_file(out / f"semantic-insights.{ext}", json_lib.dumps(data, indent=2))
    else:
        _write_file(out / f"semantic-insights.{ext}", _format_markdown_insights(insights))

    # 2. Drift report
    eff_drift = store.get_drifting_patterns(drift_threshold=0.15, limit=30)
    epi_drift = store.get_epistemic_drifting_patterns(drift_threshold=0.1, limit=30)
    if fmt == "json":
        data = {
            "effectiveness_drift": [
                {
                    "pattern_id": m.pattern_id,
                    "name": m.pattern_name,
                    "before": m.effectiveness_before,
                    "after": m.effectiveness_after,
                    "magnitude": m.drift_magnitude,
                    "direction": m.drift_direction,
                }
                for m in eff_drift
            ],
            "epistemic_drift": [
                {
                    "pattern_id": m.pattern_id,
                    "name": m.pattern_name,
                    "confidence_before": m.confidence_before,
                    "confidence_after": m.confidence_after,
                    "change": m.belief_change,
                    "direction": m.drift_direction,
                }
                for m in epi_drift
            ],
        }
        _write_file(out / f"drift-report.{ext}", json_lib.dumps(data, indent=2))
    else:
        _write_file(out / f"drift-report.{ext}", _format_markdown_drift(eff_drift, epi_drift))

    # 3. Entropy state
    entropy = store.calculate_pattern_entropy()
    alerts = store.get_entropy_alerts()
    if fmt == "json":
        data = {
            "shannon_entropy": entropy.shannon_entropy,
            "diversity_index": entropy.diversity_index,
            "unique_patterns": entropy.unique_pattern_count,
            "effective_patterns": entropy.effective_pattern_count,
            "dominant_share": entropy.dominant_pattern_share,
            "alerts": [str(a) for a in alerts],
        }
        _write_file(out / f"entropy-state.{ext}", json_lib.dumps(data, indent=2))
    else:
        _write_file(out / f"entropy-state.{ext}", _format_markdown_entropy(entropy, alerts))

    # 4. Pattern health
    all_patterns = store.get_patterns(min_priority=0.0, limit=500, exclude_quarantined=False)
    if fmt == "json":
        data = {
            "quarantined": [
                {"id": p.id, "name": p.pattern_name}
                for p in all_patterns
                if getattr(p, "quarantine_status", "") == "QUARANTINED"
            ],
            "low_trust": [
                {"id": p.id, "name": p.pattern_name, "trust": p.trust_score}
                for p in all_patterns
                if (p.trust_score or 1.0) < 0.3
            ],
        }
        _write_file(out / f"pattern-health.{ext}", json_lib.dumps(data, indent=2))
    else:
        _write_file(out / f"pattern-health.{ext}", _format_markdown_health(all_patterns))

    # 5. Evolution history
    trajectory = store.get_trajectory(limit=5)
    if fmt == "json":
        data = [
            {
                "cycle": e.cycle,
                "date": str(e.recorded_at),
                "completed": e.evolutions_completed,
                "deferred": e.evolutions_deferred,
                "issues": e.issue_classes,
                "impl_loc": e.implementation_loc,
                "test_loc": e.test_loc,
            }
            for e in trajectory
        ]
        _write_file(out / f"evolution-history.{ext}", json_lib.dumps(data, indent=2))
    else:
        _write_file(out / f"evolution-history.{ext}", _format_markdown_evolution(trajectory))

    # 6. Error landscape
    exec_stats = store.get_execution_stats()
    if fmt == "json":
        _write_file(out / f"error-landscape.{ext}", json_lib.dumps(exec_stats, indent=2))
    else:
        _write_file(out / f"error-landscape.{ext}", _format_markdown_errors(exec_stats))

    console.print(f"[green]Exported learning data to {out}/[/green]")
    console.print(f"  Semantic insights: {len(insights)} patterns")
    console.print(f"  Drift alerts: {len(eff_drift)} effectiveness, {len(epi_drift)} epistemic")
    console.print(f"  Patterns total: {len(all_patterns)}")
    console.print(f"  Evolution history: {len(trajectory)} cycles")


def learning_record_evolution(
    cycle: Annotated[int, typer.Option("--cycle", help="Evolution cycle number")] = ...,
    evolutions_completed: Annotated[
        int, typer.Option("--evolutions-completed", help="Number of evolutions completed")
    ] = ...,
    issue_classes: Annotated[
        str,
        typer.Option("--issue-classes", help="Comma-separated issue classes"),
    ] = ...,
    implementation_loc: Annotated[
        int, typer.Option("--implementation-loc", help="Lines of implementation code")
    ] = ...,
    test_loc: Annotated[
        int, typer.Option("--test-loc", help="Lines of test code")
    ] = ...,
    evolutions_deferred: Annotated[
        int, typer.Option("--evolutions-deferred", help="Number deferred")
    ] = 0,
    cv_avg: Annotated[
        float, typer.Option("--cv-avg", help="Average consciousness volume")
    ] = 0.0,
    loc_accuracy: Annotated[
        float, typer.Option("--loc-accuracy", help="LOC estimation accuracy (0-2)")
    ] = 1.0,
    notes: Annotated[
        str, typer.Option("--notes", help="Optional notes")
    ] = "",
) -> None:
    """Record an evolution cycle in the trajectory table.

    Examples:
        mozart learning-record-evolution --cycle 26 \\
            --evolutions-completed 2 \\
            --issue-classes "infrastructure_activation,testing_depth" \\
            --implementation-loc 150 --test-loc 200
    """
    from mozart.learning.global_store import get_global_store

    store = get_global_store()

    classes = [c.strip() for c in issue_classes.split(",") if c.strip()]

    entry_id = store.record_evolution_trajectory(
        cycle=cycle,
        evolutions_completed=evolutions_completed,
        evolutions_deferred=evolutions_deferred,
        issue_classes=classes,
        cv_avg=cv_avg,
        implementation_loc=implementation_loc,
        test_loc=test_loc,
        loc_accuracy=loc_accuracy,
        notes=notes or None,
    )

    console.print(f"[green]Recorded evolution cycle {cycle}[/green]")
    console.print(f"  Entry ID: {entry_id}")
    console.print(f"  Evolutions completed: {evolutions_completed}")
    console.print(f"  Issue classes: {', '.join(classes)}")
```

**Step 2: Register in `__init__.py`**

Add to `src/mozart/cli/commands/learning/__init__.py`:
```python
from mozart.cli.commands.learning._export import learning_export, learning_record_evolution
```

And add to `__all__`:
```python
"learning_export",
"learning_record_evolution",
```

**Step 3: Register in CLI app**

In `src/mozart/cli/__init__.py`, find where learning commands are registered with `app.command()` and add:
```python
app.command(name="learning-export")(learning_export)
app.command(name="learning-record-evolution")(learning_record_evolution)
```

**Step 4: Run tests**

Run: `cd /home/emzi/Projects/mozart-ai-compose && python -m pytest tests/test_cli_learning_export.py -v --tb=short`
Expected: All tests PASS.

**Step 5: Run existing tests to catch regressions**

Run: `cd /home/emzi/Projects/mozart-ai-compose && python -m pytest tests/test_cli.py tests/test_cli_learning.py -x -q --tb=short`
Expected: PASS (no regressions).

**Step 6: Type check and lint**

Run: `cd /home/emzi/Projects/mozart-ai-compose && python -m mypy src/mozart/cli/commands/learning/_export.py && python -m ruff check src/mozart/cli/commands/learning/_export.py`
Expected: Clean.

**Step 7: Commit**

```bash
git add src/mozart/cli/commands/learning/_export.py src/mozart/cli/commands/learning/__init__.py src/mozart/cli/__init__.py tests/test_cli_learning_export.py
git commit -m "feat(cli): add learning-export and learning-record-evolution commands

Thin CLI wrappers around existing learning store queries.
learning-export writes 6 structured files (markdown or JSON).
learning-record-evolution appends to evolution_trajectory table."
```

---

### Task 3: Vision prelude document

**Files:**
- Create: `scores/evolution-prelude.md`

**Step 1: Write the prelude**

This is a context document injected into every sheet of the evolution score. It shapes how agents think about Mozart. Write based on CLAUDE.md, the TDF skill, and the project's architectural principles.

Content should cover:
1. Mozart's purpose and direction
2. Interface consciousness (TDF-derived)
3. Quality bar
4. Evolution principles
5. Codebase reference (key files, patterns, conventions)

See design doc section "Context Shaping > Vision Prelude" for the content specification.

**Step 2: Commit**

```bash
git add scores/evolution-prelude.md
git commit -m "docs: add evolution vision prelude for context shaping

Injected into every sheet of the evolution concert score.
Encodes Mozart's purpose, TDF interface consciousness,
quality bar, and codebase conventions."
```

---

### Task 4: Evolution concert score

**Files:**
- Create: `scores/evolution-concert.yaml`

**Step 1: Write the score**

Assemble the full YAML score from the design doc. Key details:

- 10 logical stages, fan-out on stages 2,4,5,6,7,8 (each to 5)
- `total_items: 10` with `size: 1`
- Dependencies: `2:[1], 3:[2], 4:[3], 5:[4], 6:[5], 7:[6], 8:[7], 9:[4,5,6,7,8], 10:[9]`
- `skip_when_command` using post-expansion sheet numbers (see reference table above):
  - Sheets 8-12 (stage 4): skip if candidate N not in plan
  - Sheets 13-17 (stage 5): skip if impl report N not written
  - Sheets 18-22 (stage 6): skip if review 1 verdict is CLEAN
  - Sheets 23-27 (stage 7): skip if fix 1 not written (review was clean)
  - Sheets 28-32 (stage 8): skip if review 2 verdict is CLEAN
- Prelude: `scores/evolution-prelude.md` as skill
- Full prompt template with `{% if stage == N %}` Jinja conditionals
- All validations from design doc
- Concert config with `max_chain_depth: 10`
- `on_success` self-chain with `fresh: true`

**Step 2: Validate**

Run: `cd /home/emzi/Projects/mozart-ai-compose && mozart validate scores/evolution-concert.yaml`
Expected: Valid (warnings OK, no errors).

**Step 3: Dry run**

Run: `cd /home/emzi/Projects/mozart-ai-compose && mozart run scores/evolution-concert.yaml --dry-run 2>&1 | head -50`
Expected: Shows 34 sheets, correct fan-out expansion, rendered prompts for each stage.

**Step 4: Commit**

```bash
git add scores/evolution-concert.yaml
git commit -m "feat: add evolution-concert score — self-improving Mozart

10-stage self-chaining score with TDF-aligned analysis,
fan-out implementation, 2 rounds of review/fix, and
grounded validations tied to synthesis intent."
```

---

### Task 5: Integration verification

**Step 1: Full test suite**

Run: `cd /home/emzi/Projects/mozart-ai-compose && python -m pytest tests/ -x -q --tb=short`
Expected: All tests pass.

**Step 2: Type check**

Run: `cd /home/emzi/Projects/mozart-ai-compose && python -m mypy src/`
Expected: Clean.

**Step 3: End-to-end smoke test of export**

Run: `cd /home/emzi/Projects/mozart-ai-compose && mozart learning-export --output-dir /tmp/evolution-test-export && ls -la /tmp/evolution-test-export/`
Expected: 6 markdown files written.

**Step 4: Validate score one more time**

Run: `cd /home/emzi/Projects/mozart-ai-compose && mozart validate scores/evolution-concert.yaml --json`
Expected: Valid, no errors.
