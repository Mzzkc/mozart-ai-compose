"""Learning and pattern management commands for Mozart CLI.

This module implements commands for monitoring and managing the learning system:
- `patterns-why`: Analyze WHY patterns succeed with metacognitive insights
- `patterns-entropy`: Monitor pattern population diversity using Shannon entropy
- `patterns-budget`: Display exploration budget status and history
- `learning-stats`: View global learning statistics
- `learning-insights`: Show actionable insights from learning data
- `learning-drift`: Detect patterns with effectiveness drift
- `learning-epistemic-drift`: Detect patterns with epistemic drift
- `learning-activity`: View recent learning activity and pattern applications
- `entropy-status`: Display entropy response status and history

These commands provide visibility into Mozart's autonomous learning capabilities,
which include pattern recognition, effectiveness tracking, and drift detection.

★ Insight ─────────────────────────────────────
1. **Pattern learning as emergent behavior**: Mozart extracts patterns from execution
   outcomes automatically. The learning system doesn't require explicit training—it
   observes what succeeds and what fails, building an internal model of execution
   dynamics. This is similar to reinforcement learning but with explicit pattern
   reification.

2. **Shannon entropy for model collapse detection**: The patterns-entropy command uses
   information-theoretic metrics to detect when the pattern population becomes
   unhealthy. Low entropy means one pattern dominates (collapse risk), while high
   entropy indicates healthy diversity. This is proactive health monitoring.

3. **Epistemic vs effectiveness drift**: These are complementary signals. Effectiveness
   drift tracks outcome changes (lagging indicator), while epistemic drift tracks
   confidence changes (leading indicator). Monitoring both provides early warning of
   pattern degradation before it impacts execution success.
─────────────────────────────────────────────────
"""

from __future__ import annotations

import json as json_lib
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.panel import Panel
from rich.table import Table

from ..output import console

if TYPE_CHECKING:
    pass


# =============================================================================
# patterns-why command
# =============================================================================


def patterns_why(
    pattern_id: str = typer.Argument(
        None,
        help="Pattern ID to analyze (first 10 chars from 'patterns' command). "
        "If omitted, shows all patterns with captured success factors.",
    ),
    min_observations: int = typer.Option(
        1,
        "--min-obs",
        "-m",
        help="Minimum success factor observations required",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        help="Maximum number of patterns to display",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON for machine parsing",
    ),
) -> None:
    """Analyze WHY patterns succeed with metacognitive insights.

    Shows success factors - the context conditions that contribute to
    pattern effectiveness. This helps understand CAUSALITY behind patterns,
    not just correlation.

    v22 Evolution: Metacognitive Pattern Reflection

    Success factors include:
    - Validation types present when pattern succeeded
    - Error categories in the execution context
    - Prior sheet status and timing factors
    - Grounding confidence levels

    Examples:
        mozart patterns-why              # Show all patterns with WHY analysis
        mozart patterns-why abc123       # Analyze specific pattern
        mozart patterns-why --min-obs 3  # Only patterns with 3+ observations
        mozart patterns-why --json       # JSON output for scripting
    """
    from mozart.learning.global_store import get_global_store

    store = get_global_store()

    if pattern_id:
        # Analyze specific pattern
        patterns_list = store.get_patterns(min_priority=0.0, limit=1000)
        matching = [p for p in patterns_list if p.id.startswith(pattern_id)]

        if not matching:
            console.print(f"[red]No pattern found with ID starting with '{pattern_id}'[/red]")
            raise typer.Exit(1)

        if len(matching) > 1:
            console.print(f"[yellow]Multiple patterns match '{pattern_id}':[/yellow]")
            for p in matching[:5]:
                console.print(f"  [cyan]{p.id}[/cyan] - {p.pattern_name}")
            console.print(
                "[dim]Please provide more characters to uniquely identify the pattern.[/dim]"
            )
            raise typer.Exit(1)

        pattern = matching[0]
        analysis = store.analyze_pattern_why(pattern.id)

        if json_output:
            console.print(json_lib.dumps(analysis, indent=2, default=str))
            return

        # Human-readable WHY analysis
        console.print(Panel(
            f"[bold]{analysis.get('pattern_name', 'Unknown')}[/bold]\n"
            f"[dim]Type: {analysis.get('pattern_type', 'unknown')}[/dim]",
            title="WHY Analysis",
            border_style="magenta",
        ))

        if not analysis.get("has_factors"):
            console.print("\n[yellow]No success factors captured yet.[/yellow]")
            console.print(
                "[dim]Apply this pattern to successful executions "
                "to capture WHY it works.[/dim]"
            )
            return

        # Factors summary
        console.print("\n[bold]Factors Summary[/bold]")
        console.print(f"  {analysis.get('factors_summary', 'No summary')}")

        # Key conditions
        key_conditions = analysis.get("key_conditions", [])
        if key_conditions:
            console.print("\n[bold]Key Conditions[/bold]")
            for cond in key_conditions:
                console.print(f"  • {cond}")

        # Metrics
        console.print("\n[bold]Metrics[/bold]")
        table = Table(show_header=False, box=None)
        table.add_column("Field", style="dim", width=20)
        table.add_column("Value", style="bold")

        table.add_row("Observations", str(analysis.get("observation_count", 0)))
        table.add_row("Success Rate", f"{analysis.get('success_rate', 0):.1%}")
        table.add_row("Confidence", f"{analysis.get('confidence', 0):.2f}")
        table.add_row("Trust Score", f"{analysis.get('trust_score', 0):.2f}")
        table.add_row("Effectiveness", f"{analysis.get('effectiveness_score', 0):.2f}")

        console.print(table)

        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            console.print("\n[bold]Recommendations[/bold]")
            for rec in recommendations:
                console.print(f"  → {rec}")
    else:
        # Show all patterns with WHY analysis
        results = store.get_patterns_with_why(
            min_observations=min_observations,
            limit=limit,
        )

        if json_output:
            output = [
                {"pattern_id": p.id, "pattern_name": p.pattern_name, "analysis": a}
                for p, a in results
            ]
            console.print(json_lib.dumps(output, indent=2, default=str))
            return

        if not results:
            console.print(
                "[yellow]No patterns with captured success factors found.[/yellow]"
            )
            console.print(
                "[dim]Success factors are captured when patterns lead to "
                "successful executions.[/dim]"
            )
            return

        console.print(
            f"[bold]Patterns with WHY Analysis[/bold] "
            f"[dim]({len(results)} patterns with ≥{min_observations} observations)[/dim]\n"
        )

        table = Table(show_header=True, box=None)
        table.add_column("Pattern", style="cyan", width=30)
        table.add_column("Obs", justify="right", width=5)
        table.add_column("Success%", justify="right", width=8)
        table.add_column("Confidence", justify="right", width=10)
        table.add_column("Key Insight", width=40)

        for pattern, analysis in results:
            # Get first key condition as insight
            key_conditions = analysis.get("key_conditions", [])
            insight = key_conditions[0] if key_conditions else analysis.get("factors_summary", "")
            if len(insight) > 38:
                insight = insight[:35] + "..."

            obs = analysis.get("observation_count", 0)
            success_rate = analysis.get("success_rate", 0)
            confidence = analysis.get("confidence", 0)

            name = pattern.pattern_name
            display_name = f"{name[:28]}..." if len(name) > 30 else name
            table.add_row(
                display_name,
                str(obs),
                f"{success_rate:.0%}",
                f"{confidence:.2f}",
                insight,
            )

        console.print(table)
        console.print(
            "\n[dim]Use 'mozart patterns-why <id>' for detailed analysis.[/dim]"
        )


# =============================================================================
# learning-stats command
# =============================================================================


def learning_stats(
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON for machine parsing",
    ),
) -> None:
    """View global learning statistics.

    Shows summary statistics about the global learning store including
    execution counts, pattern counts, and effectiveness metrics.

    Examples:
        mozart learning-stats         # Human-readable summary
        mozart learning-stats --json  # JSON output for scripting
    """
    from mozart.learning.global_store import get_global_store
    from mozart.learning.migration import check_migration_status

    store = get_global_store()
    stats = store.get_execution_stats()
    migration = check_migration_status(store)

    if json_output:
        output = {
            "executions": {
                "total": stats.get("total_executions", 0),
                "first_attempt_success_rate": round(
                    stats.get("first_attempt_success_rate", 0) * 100, 1
                ),
            },
            "patterns": {
                "total": stats.get("total_patterns", 0),
                "avg_effectiveness": round(stats.get("avg_pattern_effectiveness", 0), 3),
            },
            "workspaces": {
                "unique": stats.get("unique_workspaces", 0),
            },
            "error_recoveries": {
                "total": stats.get("total_error_recoveries", 0),
                "success_rate": round(stats.get("error_recovery_success_rate", 0) * 100, 1),
            },
            "migration_needed": migration.get("needs_migration", False),
        }
        console.print(json_lib.dumps(output, indent=2))
        return

    # Human-readable output
    console.print("[bold]Global Learning Statistics[/bold]\n")

    # Execution stats
    console.print("[bold cyan]Executions[/bold cyan]")
    console.print(f"  Total recorded: [green]{stats.get('total_executions', 0)}[/green]")
    success_rate = stats.get("first_attempt_success_rate", 0) * 100
    color = 'green' if success_rate > 70 else 'yellow'
    console.print(f"  First-attempt success: [{color}]{success_rate:.1f}%[/]")

    # Pattern stats with data source breakdown
    console.print("\n[bold cyan]Patterns[/bold cyan]")
    console.print(f"  Total learned: [yellow]{stats.get('total_patterns', 0)}[/yellow]")
    avg_eff = stats.get("avg_pattern_effectiveness", 0)
    eff_color = 'green' if avg_eff > 0.6 else 'yellow'
    console.print(f"  Avg effectiveness: [{eff_color}]{avg_eff:.2f}[/]")

    # Count patterns by type for data source visibility
    all_patterns = store.get_patterns(limit=1000)
    output_patterns = sum(1 for p in all_patterns if p.pattern_type == "output_pattern")
    error_code_patterns = sum(1 for p in all_patterns if "error_code" in (p.pattern_name or ""))
    semantic_patterns = sum(1 for p in all_patterns if p.pattern_type == "semantic_failure")

    console.print("\n[bold cyan]Data Sources[/bold cyan]")
    console.print(f"  Output patterns extracted: [cyan]{output_patterns}[/cyan]")
    console.print(f"  Error code patterns: [cyan]{error_code_patterns}[/cyan]")
    console.print(f"  Semantic failure patterns: [cyan]{semantic_patterns}[/cyan]")

    # Workspace coverage
    console.print("\n[bold cyan]Workspaces[/bold cyan]")
    console.print(f"  Unique workspaces: [cyan]{stats.get('unique_workspaces', 0)}[/cyan]")

    # Error recovery stats
    console.print("\n[bold cyan]Error Recovery Learning[/bold cyan]")
    console.print(f"  Recoveries recorded: {stats.get('total_error_recoveries', 0)}")
    recovery_rate = stats.get("error_recovery_success_rate", 0) * 100
    rec_color = 'green' if recovery_rate > 70 else 'yellow'
    console.print(f"  Recovery success rate: [{rec_color}]{recovery_rate:.1f}%[/]")

    # Migration status
    if migration.get("needs_migration"):
        console.print(
            "\n[yellow]⚠ Migration needed:[/yellow] Run 'mozart aggregate-patterns' "
            "to import workspace-local outcomes"
        )


# =============================================================================
# learning-insights command
# =============================================================================


def learning_insights(
    limit: Annotated[int, typer.Option(help="Max patterns to show")] = 10,
    pattern_type: Annotated[str | None, typer.Option(help="Filter by type")] = None,
) -> None:
    """Show actionable insights from learning data.

    Displays patterns extracted from execution history including:
    - Output patterns (from stdout/stderr analysis)
    - Error code patterns (aggregated error statistics)
    - Success predictors (factors that correlate with success)

    Examples:
        mozart learning-insights
        mozart learning-insights --pattern-type output_pattern
        mozart learning-insights --limit 20
    """
    from mozart.learning.global_store import GlobalLearningStore

    console.print("[bold]Learning Insights[/bold]")
    console.print()

    store = GlobalLearningStore()
    patterns = store.get_patterns(
        pattern_type=pattern_type,
        limit=limit,
    )

    if not patterns:
        console.print("[dim]No patterns learned yet. Run some jobs![/dim]")
        return

    # Display table of patterns
    table = Table(title="Learned Patterns")
    table.add_column("Type", style="cyan")
    table.add_column("Description")
    table.add_column("Freq", justify="right")
    table.add_column("Effectiveness", justify="right")

    for p in patterns:
        desc_str = p.description or ""
        desc = desc_str[:45] + "..." if len(desc_str) > 45 else desc_str
        table.add_row(
            p.pattern_type,
            desc or "[no description]",
            str(p.occurrence_count),
            f"{p.effectiveness_score:.0%}" if p.effectiveness_score else "-",
        )

    console.print(table)


# =============================================================================
# learning-drift command
# =============================================================================


def learning_drift(
    threshold: float = typer.Option(
        0.2,
        "--threshold",
        "-t",
        help="Drift threshold (0.0-1.0) to flag patterns",
    ),
    window: int = typer.Option(
        5,
        "--window",
        "-w",
        help="Window size for drift comparison",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Maximum number of patterns to show",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON for machine parsing",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        "-s",
        help="Show only summary statistics",
    ),
) -> None:
    """Detect patterns with effectiveness drift.

    v12 Evolution: Goal Drift Detection - monitors pattern health by
    comparing recent effectiveness to historical effectiveness. Patterns
    that were once effective but are now declining may need investigation.

    Drift is calculated by comparing the pattern's effectiveness in its
    last N applications vs the previous N applications. A positive drift
    means improving, negative means declining.

    Examples:
        mozart learning-drift                # Show drifting patterns
        mozart learning-drift -t 0.15        # Lower threshold (more sensitive)
        mozart learning-drift -w 10          # Larger comparison window
        mozart learning-drift --summary      # Just show summary stats
        mozart learning-drift --json         # JSON output for scripting
    """
    from mozart.learning.global_store import get_global_store

    store = get_global_store()

    if summary:
        # Summary mode - just show aggregate stats
        drift_summary = store.get_pattern_drift_summary()

        if json_output:
            console.print(json_lib.dumps(drift_summary, indent=2))
            return

        console.print("[bold]Pattern Drift Summary[/bold]\n")
        console.print(f"  Total patterns: {drift_summary['total_patterns']}")
        console.print(f"  Patterns analyzed: {drift_summary['patterns_analyzed']}")
        drifting = drift_summary["patterns_drifting"]
        color = "red" if drifting > 0 else "green"
        console.print(f"  Patterns drifting: [{color}]{drifting}[/{color}]")
        console.print(
            f"  Avg drift magnitude: {drift_summary['avg_drift_magnitude']:.3f}"
        )
        if drift_summary["most_drifted"]:
            console.print(
                f"  Most drifted pattern: {drift_summary['most_drifted']}"
            )
        return

    # Full drift report
    drifting_patterns = store.get_drifting_patterns(
        drift_threshold=threshold,
        window_size=window,
        limit=limit,
    )

    if json_output:
        output = {
            "threshold": threshold,
            "window_size": window,
            "patterns": [
                {
                    "pattern_id": m.pattern_id,
                    "pattern_name": m.pattern_name,
                    "effectiveness_before": round(m.effectiveness_before, 3),
                    "effectiveness_after": round(m.effectiveness_after, 3),
                    "drift_magnitude": round(m.drift_magnitude, 3),
                    "drift_direction": m.drift_direction,
                    "grounding_confidence_avg": round(m.grounding_confidence_avg, 3),
                    "applications_analyzed": m.applications_analyzed,
                }
                for m in drifting_patterns
            ],
        }
        console.print(json_lib.dumps(output, indent=2))
        return

    # Human-readable table output
    console.print(f"[bold]Patterns with Drift > {threshold:.0%}[/bold]")
    console.print(f"[dim]Window size: {window} applications per period[/dim]\n")

    if not drifting_patterns:
        console.print("[green]✓ No patterns exceeding drift threshold[/green]")
        console.print("[dim]All patterns are stable or improving[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Pattern", style="cyan")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Drift", justify="right")
    table.add_column("Direction", justify="center")
    table.add_column("Grounding", justify="right")

    for m in drifting_patterns:
        # Color-code drift direction
        if m.drift_direction == "negative":
            dir_style = "[red]↓ declining[/red]"
        elif m.drift_direction == "positive":
            dir_style = "[green]↑ improving[/green]"
        else:
            dir_style = "[dim]→ stable[/dim]"

        # Color-code drift magnitude
        drift_color = "red" if m.drift_magnitude > 0.3 else "yellow"

        table.add_row(
            m.pattern_name[:30],  # Truncate long names
            f"{m.effectiveness_before:.1%}",
            f"{m.effectiveness_after:.1%}",
            f"[{drift_color}]{m.drift_magnitude:.1%}[/{drift_color}]",
            dir_style,
            f"{m.grounding_confidence_avg:.1%}",
        )

    console.print(table)

    # Add warning for declining patterns
    declining = [m for m in drifting_patterns if m.drift_direction == "negative"]
    if declining:
        console.print(
            f"\n[yellow]⚠ {len(declining)} pattern(s) showing declining effectiveness[/yellow]"
        )
        console.print("[dim]Consider reviewing these patterns for deprecation[/dim]")


# =============================================================================
# learning-epistemic-drift command
# =============================================================================


def learning_epistemic_drift(
    threshold: float = typer.Option(
        0.15,
        "--threshold",
        "-t",
        help="Epistemic drift threshold (0.0-1.0) to flag patterns",
    ),
    window: int = typer.Option(
        5,
        "--window",
        "-w",
        help="Window size for drift comparison",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Maximum number of patterns to show",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON for machine parsing",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        "-s",
        help="Show only summary statistics",
    ),
) -> None:
    """Detect patterns with epistemic drift (belief/confidence changes).

    v21 Evolution: Epistemic Drift Detection - monitors how confidence in
    patterns changes over time, complementing effectiveness drift. While
    effectiveness drift tracks outcome changes, epistemic drift tracks
    belief evolution.

    This enables detection of belief degradation BEFORE effectiveness
    actually declines - a leading indicator of pattern health.

    Epistemic drift is calculated by comparing average grounding confidence
    in recent applications vs older applications. High entropy (variance in
    confidence) amplifies the drift signal.

    Examples:
        mozart learning-epistemic-drift            # Show patterns with belief drift
        mozart learning-epistemic-drift -t 0.1    # Lower threshold (more sensitive)
        mozart learning-epistemic-drift -w 10     # Larger comparison window
        mozart learning-epistemic-drift --summary # Just show summary stats
        mozart learning-epistemic-drift --json    # JSON output for scripting
    """
    from mozart.learning.global_store import get_global_store

    store = get_global_store()

    if summary:
        # Summary mode - just show aggregate stats
        drift_summary = store.get_epistemic_drift_summary()

        if json_output:
            console.print(json_lib.dumps(drift_summary, indent=2))
            return

        console.print("[bold]Epistemic Drift Summary[/bold]\n")
        console.print(f"  Total patterns: {drift_summary['total_patterns']}")
        console.print(f"  Patterns analyzed: {drift_summary['patterns_analyzed']}")
        drifting = drift_summary["patterns_with_epistemic_drift"]
        color = "red" if drifting > 0 else "green"
        console.print(f"  Patterns with drift: [{color}]{drifting}[/{color}]")
        console.print(
            f"  Avg belief change: {drift_summary['avg_belief_change']:.3f}"
        )
        console.print(
            f"  Avg belief entropy: {drift_summary['avg_belief_entropy']:.3f}"
        )
        if drift_summary["most_unstable"]:
            console.print(
                f"  Most unstable pattern: {drift_summary['most_unstable']}"
            )
        return

    # Full epistemic drift report
    drifting_patterns = store.get_epistemic_drifting_patterns(
        drift_threshold=threshold,
        window_size=window,
        limit=limit,
    )

    if json_output:
        output = {
            "threshold": threshold,
            "window_size": window,
            "patterns": [
                {
                    "pattern_id": m.pattern_id,
                    "pattern_name": m.pattern_name,
                    "confidence_before": round(m.confidence_before, 3),
                    "confidence_after": round(m.confidence_after, 3),
                    "belief_change": round(m.belief_change, 3),
                    "belief_entropy": round(m.belief_entropy, 3),
                    "drift_direction": m.drift_direction,
                    "applications_analyzed": m.applications_analyzed,
                }
                for m in drifting_patterns
            ],
        }
        console.print(json_lib.dumps(output, indent=2))
        return

    # Human-readable table output
    console.print(f"[bold]Patterns with Epistemic Drift > {threshold:.0%}[/bold]")
    console.print(f"[dim]Window size: {window} applications per period[/dim]\n")

    if not drifting_patterns:
        console.print("[green]✓ No patterns exceeding epistemic drift threshold[/green]")
        console.print("[dim]All pattern beliefs are stable[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Pattern", style="cyan")
    table.add_column("Conf Before", justify="right")
    table.add_column("Conf After", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Direction", justify="center")
    table.add_column("Entropy", justify="right")

    for m in drifting_patterns:
        # Color-code drift direction
        if m.drift_direction == "weakening":
            dir_style = "[red]↓ weakening[/red]"
        elif m.drift_direction == "strengthening":
            dir_style = "[green]↑ strengthening[/green]"
        else:
            dir_style = "[dim]→ stable[/dim]"

        # Color-code belief change magnitude
        change_color = "red" if abs(m.belief_change) > 0.2 else "yellow"

        # Color-code entropy (high entropy = concerning)
        entropy_color = "red" if m.belief_entropy > 0.3 else "dim"

        table.add_row(
            m.pattern_name[:30],  # Truncate long names
            f"{m.confidence_before:.1%}",
            f"{m.confidence_after:.1%}",
            f"[{change_color}]{m.belief_change:+.1%}[/{change_color}]",
            dir_style,
            f"[{entropy_color}]{m.belief_entropy:.2f}[/{entropy_color}]",
        )

    console.print(table)

    # Add warning for weakening patterns
    weakening = [m for m in drifting_patterns if m.drift_direction == "weakening"]
    if weakening:
        console.print(
            f"\n[yellow]⚠ {len(weakening)} pattern(s) showing weakening confidence[/yellow]"
        )
        console.print(
            "[dim]These patterns may need investigation "
            "before effectiveness declines[/dim]"
        )

    # Add warning for high entropy patterns
    high_entropy = [m for m in drifting_patterns if m.belief_entropy > 0.3]
    if high_entropy:
        console.print(
            f"\n[yellow]⚠ {len(high_entropy)} pattern(s) with high belief entropy[/yellow]"
        )
        console.print("[dim]Inconsistent confidence suggests unstable pattern application[/dim]")


# =============================================================================
# patterns-entropy command
# =============================================================================


def patterns_entropy(
    alert_threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        help="Diversity index below this triggers alert (0.0-1.0)",
    ),
    history: bool = typer.Option(
        False,
        "--history",
        "-H",
        help="Show entropy history over time",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Number of history records to show",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON for machine parsing",
    ),
    record: bool = typer.Option(
        False,
        "--record",
        "-r",
        help="Record current entropy to history",
    ),
) -> None:
    """Monitor pattern population diversity using Shannon entropy.

    v21 Evolution: Pattern Entropy Monitoring - detects model collapse risk
    by tracking the diversity of pattern usage across the learning system.

    Shannon entropy measures how evenly patterns are used:
    - High entropy (H → max): Healthy diversity, many patterns contribute
    - Low entropy (H → 0): Single pattern dominates (collapse risk)

    The diversity_index normalizes entropy to 0-1 range for easy alerting.

    Examples:
        mozart patterns-entropy               # Show current entropy metrics
        mozart patterns-entropy --threshold 0.3  # Alert on low diversity
        mozart patterns-entropy --history     # View entropy trend over time
        mozart patterns-entropy --record      # Record snapshot for trend analysis
        mozart patterns-entropy --json        # JSON output for scripting
    """
    from mozart.learning.global_store import get_global_store

    store = get_global_store()

    if history:
        # Show entropy history
        history_records = store.get_pattern_entropy_history(limit=limit)  # type: ignore[attr-defined]

        if json_output:
            hist_output: list[dict[str, Any]] = []
            for h in history_records:
                hist_output.append({
                    "calculated_at": h.calculated_at.isoformat(),
                    "shannon_entropy": round(h.shannon_entropy, 4),
                    "max_possible_entropy": round(h.max_possible_entropy, 4),
                    "diversity_index": round(h.diversity_index, 4),
                    "unique_pattern_count": h.unique_pattern_count,
                    "effective_pattern_count": h.effective_pattern_count,
                    "total_applications": h.total_applications,
                    "dominant_pattern_share": round(h.dominant_pattern_share, 4),
                    "threshold_exceeded": h.threshold_exceeded,
                })
            console.print(json_lib.dumps(hist_output, indent=2))
            return

        if not history_records:
            console.print("[dim]No entropy history found.[/dim]")
            console.print("\n[dim]Hint: Use --record to start tracking entropy over time.[/dim]")
            return

        # Display history table
        table = Table(title="Pattern Entropy History")
        table.add_column("Time", style="dim", width=20)
        table.add_column("Shannon H", justify="right", width=10)
        table.add_column("Diversity", justify="right", width=10)
        table.add_column("Unique", justify="right", width=8)
        table.add_column("Effective", justify="right", width=10)
        table.add_column("Applications", justify="right", width=12)
        table.add_column("Dominant %", justify="right", width=10)

        for h in history_records:
            # Color diversity based on threshold
            div_color = "green" if h.diversity_index >= alert_threshold else "red"
            div_str = f"[{div_color}]{h.diversity_index:.3f}[/{div_color}]"

            # Color dominant share (high = concerning)
            if h.dominant_pattern_share > 0.5:
                dom_color = "red"
            elif h.dominant_pattern_share > 0.3:
                dom_color = "yellow"
            else:
                dom_color = "green"
            dom_str = f"[{dom_color}]{h.dominant_pattern_share:.1%}[/{dom_color}]"

            table.add_row(
                h.calculated_at.strftime("%Y-%m-%d %H:%M"),
                f"{h.shannon_entropy:.3f}",
                div_str,
                str(h.unique_pattern_count),
                str(h.effective_pattern_count),
                str(h.total_applications),
                dom_str,
            )

        console.print(table)
        console.print(f"\n[dim]Showing {len(history_records)} record(s)[/dim]")
        return

    # Calculate current entropy
    metrics = store.calculate_pattern_entropy()  # type: ignore[attr-defined]

    # Set threshold_exceeded based on config threshold
    metrics.threshold_exceeded = metrics.diversity_index < alert_threshold

    if record:
        # Record to history
        record_id = store.record_pattern_entropy(metrics)  # type: ignore[attr-defined]
        console.print(f"[green]Recorded entropy snapshot: {record_id[:10]}[/green]\n")

    if json_output:
        output = {
            "calculated_at": metrics.calculated_at.isoformat(),
            "shannon_entropy": round(metrics.shannon_entropy, 4),
            "max_possible_entropy": round(metrics.max_possible_entropy, 4),
            "diversity_index": round(metrics.diversity_index, 4),
            "unique_pattern_count": metrics.unique_pattern_count,
            "effective_pattern_count": metrics.effective_pattern_count,
            "total_applications": metrics.total_applications,
            "dominant_pattern_share": round(metrics.dominant_pattern_share, 4),
            "threshold_exceeded": metrics.threshold_exceeded,
            "alert_threshold": alert_threshold,
        }
        console.print(json_lib.dumps(output, indent=2))
        return

    # Display current metrics
    console.print("[bold]Pattern Population Entropy[/bold]\n")

    # Main metrics
    if metrics.total_applications == 0:
        console.print("[dim]No pattern applications yet.[/dim]")
        console.print("\n[dim]Hint: Run jobs with learning enabled to build patterns.[/dim]")
        return

    # Shannon entropy display
    console.print(f"  Shannon Entropy (H): [cyan]{metrics.shannon_entropy:.4f}[/cyan] bits")
    console.print(f"  Max Possible (H_max): [dim]{metrics.max_possible_entropy:.4f}[/dim] bits")

    # Diversity index with color
    div_color = "green" if metrics.diversity_index >= alert_threshold else "red"
    console.print(
        f"  Diversity Index: [{div_color}]{metrics.diversity_index:.4f}"
        f"[/{div_color}] (threshold: {alert_threshold})"
    )

    console.print("")

    # Pattern counts
    console.print(f"  Unique Patterns: [yellow]{metrics.unique_pattern_count}[/yellow]")
    console.print(
        f"  Effective Patterns: [yellow]{metrics.effective_pattern_count}"
        "[/yellow] (with ≥1 application)"
    )
    console.print(f"  Total Applications: [yellow]{metrics.total_applications}[/yellow]")

    # Dominant pattern warning
    if metrics.dominant_pattern_share > 0.5:
        dom_color = "red"
    elif metrics.dominant_pattern_share > 0.3:
        dom_color = "yellow"
    else:
        dom_color = "green"
    console.print(
        f"  Dominant Pattern Share: [{dom_color}]"
        f"{metrics.dominant_pattern_share:.1%}[/{dom_color}]"
    )

    # Alerts
    if metrics.threshold_exceeded:
        console.print("\n[red bold]⚠ LOW DIVERSITY ALERT[/red bold]")
        console.print("[red]Pattern population shows low diversity - model collapse risk![/red]")
        console.print(
            "[dim]Consider reviewing dominant patterns "
            "and encouraging exploration.[/dim]"
        )
    elif metrics.dominant_pattern_share > 0.5:
        console.print("\n[yellow]⚠ Single pattern holds >50% of applications[/yellow]")
        console.print("[dim]Monitor for further concentration.[/dim]")
    else:
        console.print("\n[green]✓ Healthy pattern diversity[/green]")


# =============================================================================
# patterns-budget command
# =============================================================================


def patterns_budget(
    job: str = typer.Option(
        None,
        "--job",
        "-j",
        help="Filter by specific job hash",
    ),
    history: bool = typer.Option(
        False,
        "--history",
        "-H",
        help="Show budget adjustment history",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Number of history records to show",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON for machine parsing",
    ),
) -> None:
    """Display exploration budget status and history.

    v23 Evolution: Exploration Budget Maintenance - monitors the dynamic
    exploration budget that prevents convergence to zero.

    The budget adjusts based on pattern entropy:
    - Low entropy → budget increases (boost) to inject diversity
    - Healthy entropy → budget decays toward floor
    - Budget never drops below floor (default 5%)

    Examples:
        mozart patterns-budget               # Show current budget status
        mozart patterns-budget --history     # View budget adjustment history
        mozart patterns-budget --job abc123  # Filter by specific job
        mozart patterns-budget --json        # JSON output for scripting
    """
    from mozart.learning.global_store import get_global_store

    store = get_global_store()

    if history:
        # Show budget history
        history_records = store.get_exploration_budget_history(
            job_hash=job, limit=limit
        )

        if json_output:
            budget_hist_output: list[dict[str, object]] = []
            for h in history_records:
                budget_hist_output.append({
                    "id": h.id,
                    "job_hash": h.job_hash,
                    "recorded_at": h.recorded_at.isoformat(),
                    "budget_value": round(h.budget_value, 4),
                    "entropy_at_time": round(h.entropy_at_time, 4) if h.entropy_at_time else None,
                    "adjustment_type": h.adjustment_type,
                    "adjustment_reason": h.adjustment_reason,
                })
            console.print(json_lib.dumps(budget_hist_output, indent=2))
            return

        if not history_records:
            console.print("[dim]No budget history found.[/dim]")
            console.print(
                "\n[dim]Hint: Enable exploration_budget in learning config "
                "to start tracking.[/dim]"
            )
            return

        # Display history table
        table = Table(title="Exploration Budget History")
        table.add_column("Time", style="dim", width=16)
        table.add_column("Budget", justify="right", width=8)
        table.add_column("Entropy", justify="right", width=8)
        table.add_column("Type", width=12)
        table.add_column("Reason", width=35)

        for h in history_records:
            # Color budget based on value
            if h.budget_value <= 0.10:
                budget_color = "yellow"
            elif h.budget_value >= 0.30:
                budget_color = "cyan"
            else:
                budget_color = "green"
            budget_str = f"[{budget_color}]{h.budget_value:.1%}[/{budget_color}]"

            # Color entropy if available
            if h.entropy_at_time is not None:
                ent_color = "red" if h.entropy_at_time < 0.3 else "green"
                ent_str = f"[{ent_color}]{h.entropy_at_time:.3f}[/{ent_color}]"
            else:
                ent_str = "[dim]—[/dim]"

            # Color adjustment type
            type_colors = {
                "initial": "blue",
                "boost": "green",
                "decay": "dim",
                "floor_enforced": "yellow",
                "ceiling_enforced": "yellow",
            }
            type_color = type_colors.get(h.adjustment_type, "white")
            type_str = f"[{type_color}]{h.adjustment_type}[/{type_color}]"

            table.add_row(
                h.recorded_at.strftime("%m-%d %H:%M:%S"),
                budget_str,
                ent_str,
                type_str,
                h.adjustment_reason or "",
            )

        console.print(table)
        console.print(f"\n[dim]Showing {len(history_records)} record(s)[/dim]")
        return

    # Get current budget and statistics
    current = store.get_exploration_budget(job_hash=job)
    stats = store.get_exploration_budget_statistics(job_hash=job)

    if json_output:
        entropy_val = None
        if current and current.entropy_at_time:
            entropy_val = round(current.entropy_at_time, 4)
        budget_output: dict[str, dict[str, Any]] = {
            "current": {
                "budget_value": round(current.budget_value, 4) if current else None,
                "entropy_at_time": entropy_val,
                "adjustment_type": current.adjustment_type if current else None,
                "recorded_at": current.recorded_at.isoformat() if current else None,
            },
            "statistics": {
                "avg_budget": round(stats["avg_budget"], 4),
                "min_budget": round(stats["min_budget"], 4),
                "max_budget": round(stats["max_budget"], 4),
                "total_adjustments": stats["total_adjustments"],
                "floor_enforcements": stats["floor_enforcements"],
                "boost_count": stats["boost_count"],
                "decay_count": stats["decay_count"],
            },
        }
        console.print(json_lib.dumps(budget_output, indent=2))
        return

    # Display current status
    console.print("[bold]Exploration Budget Status[/bold]\n")

    if current is None:
        console.print("[dim]No budget records found.[/dim]")
        console.print("\n[dim]Hint: Enable exploration_budget in learning config:[/dim]")
        console.print("[dim]  learning:[/dim]")
        console.print("[dim]    exploration_budget:[/dim]")
        console.print("[dim]      enabled: true[/dim]")
        return

    # Current budget with color
    if current.budget_value <= 0.10:
        budget_color = "yellow"
        budget_status = "Low (near floor)"
    elif current.budget_value >= 0.30:
        budget_color = "cyan"
        budget_status = "High (exploring)"
    else:
        budget_color = "green"
        budget_status = "Normal"

    console.print(
        f"  Current Budget: [{budget_color}]{current.budget_value:.1%}"
        f"[/{budget_color}] ({budget_status})"
    )

    if current.entropy_at_time is not None:
        ent_color = "red" if current.entropy_at_time < 0.3 else "green"
        console.print(
            f"  Entropy at Last Check: [{ent_color}]"
            f"{current.entropy_at_time:.4f}[/{ent_color}]"
        )

    console.print(f"  Last Adjustment: [dim]{current.adjustment_type}[/dim]")
    console.print(f"  Last Updated: [dim]{current.recorded_at.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")

    console.print("")

    # Statistics
    if stats["total_adjustments"] > 0:
        console.print("[bold]Statistics[/bold]")
        console.print(f"  Average Budget: [cyan]{stats['avg_budget']:.1%}[/cyan]")
        console.print(f"  Range: [dim]{stats['min_budget']:.1%} - {stats['max_budget']:.1%}[/dim]")
        console.print(f"  Total Adjustments: [yellow]{stats['total_adjustments']}[/yellow]")
        console.print(
            f"    Boosts: [green]{stats['boost_count']}[/green] | "
            f"Decays: [dim]{stats['decay_count']}[/dim] | "
            f"Floor Enforced: [yellow]{stats['floor_enforcements']}[/yellow]"
        )

    # Interpretation
    console.print("")
    if stats["floor_enforcements"] > stats["total_adjustments"] * 0.3:
        console.print("[yellow]⚠ Budget frequently hitting floor[/yellow]")
        console.print("[dim]Consider lowering entropy threshold or increasing boost amount.[/dim]")
    elif stats["boost_count"] > stats["decay_count"] * 2:
        console.print("[cyan]ℹ Frequent boosts - entropy may be consistently low[/cyan]")
        console.print("[dim]This may indicate pattern concentration issues.[/dim]")
    elif stats["total_adjustments"] > 0:
        console.print("[green]✓ Budget adjusting normally[/green]")


# =============================================================================
# entropy-status command
# =============================================================================


def entropy_status(
    job: str = typer.Option(
        None,
        "--job",
        "-j",
        help="Filter by specific job hash",
    ),
    history: bool = typer.Option(
        False,
        "--history",
        "-H",
        help="Show entropy response history",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Number of history records to show",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON for machine parsing",
    ),
    check: bool = typer.Option(
        False,
        "--check",
        "-c",
        help="Check if entropy response is needed (dry-run)",
    ),
) -> None:
    """Display entropy response status and history.

    v23 Evolution: Automatic Entropy Response - monitors the automatic
    diversity injection system that responds to low entropy conditions.

    When pattern entropy drops below threshold, the system automatically:
    - Boosts the exploration budget to encourage diversity
    - Revisits quarantined patterns for potential revalidation

    Examples:
        mozart entropy-status               # Show current entropy response status
        mozart entropy-status --history     # View response history
        mozart entropy-status --check       # Check if response is needed now
        mozart entropy-status --json        # JSON output for scripting
    """
    from mozart.learning.global_store import get_global_store

    store = get_global_store()

    if check:
        # Check if response is needed
        needs_response, entropy, reason = store.check_entropy_response_needed(
            job_hash=job or "default"
        )

        if json_output:
            output = {
                "needs_response": needs_response,
                "current_entropy": round(entropy, 4) if entropy else None,
                "reason": reason,
            }
            console.print(json_lib.dumps(output, indent=2))
            return

        console.print("[bold]Entropy Response Check[/bold]\n")
        if needs_response:
            console.print("[red bold]✓ Response NEEDED[/red bold]")
            console.print(f"  Current Entropy: [red]{entropy:.4f}[/red]")
            console.print(f"  Reason: {reason}")
            console.print(
                "\n[dim]Enable entropy_response in learning config "
                "to trigger automatically.[/dim]"
            )
        else:
            console.print("[green]✗ No response needed[/green]")
            if entropy is not None:
                console.print(f"  Current Entropy: [green]{entropy:.4f}[/green]")
            console.print(f"  Reason: {reason}")
        return

    if history:
        # Show response history
        history_records = store.get_entropy_response_history(
            job_hash=job, limit=limit
        )

        if json_output:
            resp_hist_output: list[dict[str, Any]] = []
            for h in history_records:
                resp_hist_output.append({
                    "id": h.id,
                    "job_hash": h.job_hash,
                    "recorded_at": h.recorded_at.isoformat(),
                    "entropy_at_trigger": round(h.entropy_at_trigger, 4),
                    "threshold_used": round(h.threshold_used, 4),
                    "actions_taken": h.actions_taken,
                    "budget_boosted": h.budget_boosted,
                    "quarantine_revisits": h.quarantine_revisits,
                })
            console.print(json_lib.dumps(resp_hist_output, indent=2))
            return

        if not history_records:
            console.print("[dim]No entropy response history found.[/dim]")
            console.print(
                "\n[dim]Hint: Enable entropy_response in learning config "
                "to start tracking.[/dim]"
            )
            return

        # Display history table
        table = Table(title="Entropy Response History")
        table.add_column("Time", style="dim", width=16)
        table.add_column("Entropy", justify="right", width=8)
        table.add_column("Threshold", justify="right", width=9)
        table.add_column("Budget+", justify="center", width=8)
        table.add_column("Revisits", justify="right", width=8)
        table.add_column("Actions", width=25)

        for h in history_records:
            budget_str = "[green]Yes[/green]" if h.budget_boosted else "[dim]No[/dim]"
            if h.quarantine_revisits > 0:
                revisit_str = f"[cyan]{h.quarantine_revisits}[/cyan]"
            else:
                revisit_str = "[dim]0[/dim]"
            actions_str = ", ".join(h.actions_taken) if h.actions_taken else "[dim]none[/dim]"

            table.add_row(
                h.recorded_at.strftime("%m-%d %H:%M:%S"),
                f"[red]{h.entropy_at_trigger:.3f}[/red]",
                f"{h.threshold_used:.3f}",
                budget_str,
                revisit_str,
                actions_str,
            )

        console.print(table)
        console.print(f"\n[dim]Showing {len(history_records)} record(s)[/dim]")
        return

    # Get current statistics
    stats = store.get_entropy_response_statistics(job_hash=job)
    last = store.get_last_entropy_response(job_hash=job)

    if json_output:
        last_resp: dict[str, Any] | None = None
        if last:
            last_resp = {
                "entropy_at_trigger": round(last.entropy_at_trigger, 4),
                "threshold_used": round(last.threshold_used, 4),
                "actions_taken": last.actions_taken,
                "recorded_at": last.recorded_at.isoformat(),
            }
        status_output: dict[str, Any] = {
            "statistics": {
                "total_responses": stats["total_responses"],
                "avg_entropy_at_trigger": round(stats["avg_entropy_at_trigger"], 4),
                "budget_boosts": stats["budget_boosts"],
                "quarantine_revisits": stats["quarantine_revisits"],
                "last_response": stats["last_response"],
            },
            "last_response": last_resp,
        }
        console.print(json_lib.dumps(status_output, indent=2))
        return

    # Display current status
    console.print("[bold]Entropy Response Status[/bold]\n")

    if stats["total_responses"] == 0:
        console.print("[dim]No entropy responses recorded yet.[/dim]")
        console.print("\n[dim]Hint: Enable entropy_response in learning config:[/dim]")
        console.print("[dim]  learning:[/dim]")
        console.print("[dim]    entropy_response:[/dim]")
        console.print("[dim]      enabled: true[/dim]")
        return

    # Statistics
    console.print(f"  Total Responses: [yellow]{stats['total_responses']}[/yellow]")
    console.print(f"  Avg Trigger Entropy: [red]{stats['avg_entropy_at_trigger']:.4f}[/red]")
    console.print(f"  Budget Boosts: [green]{stats['budget_boosts']}[/green]")
    console.print(f"  Quarantine Revisits: [cyan]{stats['quarantine_revisits']}[/cyan]")

    console.print("")

    # Last response details
    if last:
        console.print("[bold]Last Response[/bold]")
        console.print(f"  Time: [dim]{last.recorded_at.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
        console.print(f"  Entropy at Trigger: [red]{last.entropy_at_trigger:.4f}[/red]")
        console.print(f"  Threshold: [dim]{last.threshold_used:.4f}[/dim]")
        actions = ', '.join(last.actions_taken) if last.actions_taken else '[dim]none[/dim]'
        console.print(f"  Actions: {actions}")

    # Interpretation
    console.print("")
    if stats["total_responses"] > 10:
        console.print("[yellow]⚠ Many responses triggered[/yellow]")
        console.print("[dim]Pattern diversity may be consistently low. Review patterns.[/dim]")
    elif stats["quarantine_revisits"] > stats["total_responses"]:
        console.print("[cyan]ℹ Active quarantine revisiting[/cyan]")
        console.print("[dim]Previously problematic patterns are being reconsidered.[/dim]")
    else:
        console.print("[green]✓ Entropy response system active[/green]")


# =============================================================================
# learning-activity command
# =============================================================================


def learning_activity(
    hours: int = typer.Option(
        24,
        "--hours",
        "-h",
        help="Show activity from the last N hours",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON for machine parsing",
    ),
) -> None:
    """View recent learning activity and pattern applications.

    Learning Activation: Shows what patterns have been applied, their
    effectiveness, and provides insight into how the learning system
    is improving execution outcomes.

    Examples:
        mozart learning-activity           # Last 24 hours of activity
        mozart learning-activity -h 48     # Last 48 hours
        mozart learning-activity --json    # JSON output
    """
    from mozart.learning.global_store import get_global_store

    store = get_global_store()
    stats = store.get_execution_stats()

    # Get optimal execution window analysis
    window = store.get_optimal_execution_window()

    # Get recent similar executions for activity display
    cutoff = datetime.now() - timedelta(hours=hours)
    recent_executions = store.get_similar_executions(limit=20)
    recent_count = sum(
        1 for e in recent_executions
        if e.completed_at and e.completed_at > cutoff
    )

    if json_output:
        output = {
            "period_hours": hours,
            "recent_executions": recent_count,
            "first_attempt_success_rate": round(
                stats.get("first_attempt_success_rate", 0) * 100, 1
            ),
            "patterns_active": stats.get("total_patterns", 0),
            "optimal_hours": window.get("optimal_hours", []),
            "avoid_hours": window.get("avoid_hours", []),
            "scheduling_confidence": round(window.get("confidence", 0), 2),
        }
        console.print(json_lib.dumps(output, indent=2))
        return

    # Human-readable output
    console.print(f"[bold]Learning Activity (last {hours} hours)[/bold]\n")

    # Recent activity
    console.print("[bold cyan]Recent Executions[/bold cyan]")
    console.print(f"  Executions in period: [green]{recent_count}[/green]")
    success_rate = stats.get("first_attempt_success_rate", 0) * 100
    console.print(
        f"  First-attempt success: "
        f"[{'green' if success_rate > 70 else 'yellow'}]{success_rate:.1f}%[/]"
    )

    # Pattern application info
    console.print("\n[bold cyan]Pattern Application[/bold cyan]")
    pattern_count = stats.get("total_patterns", 0)
    if pattern_count > 0:
        console.print(f"  Active patterns: [yellow]{pattern_count}[/yellow]")
        avg_eff = stats.get("avg_pattern_effectiveness", 0)
        console.print(
            f"  Avg effectiveness: "
            f"[{'green' if avg_eff > 0.6 else 'yellow'}]{avg_eff:.2f}[/]"
        )
    else:
        console.print("  [dim]No patterns learned yet[/dim]")

    # Time-aware scheduling insights
    console.print("\n[bold cyan]Optimal Execution Windows[/bold cyan]")
    if window.get("confidence", 0) > 0.3:
        optimal = window.get("optimal_hours", [])
        avoid = window.get("avoid_hours", [])

        if optimal:
            optimal_str = ", ".join(f"{h:02d}:00" for h in optimal)
            console.print(f"  [green]✓ Best hours:[/green] {optimal_str}")
        if avoid:
            avoid_str = ", ".join(f"{h:02d}:00" for h in avoid)
            console.print(f"  [red]✗ Avoid hours:[/red] {avoid_str}")

        console.print(
            f"  Confidence: [cyan]{window.get('confidence', 0):.0%}[/cyan] "
            f"(based on {window.get('sample_count', 0)} samples)"
        )
    else:
        console.print("  [dim]Insufficient data for scheduling recommendations[/dim]")

    # Learning status summary
    console.print("\n[bold cyan]Learning Status[/bold cyan]")
    total_executions = stats.get("total_executions", 0)
    if total_executions >= 50:
        console.print("  [green]✓ Learning system is well-trained[/green]")
    elif total_executions >= 10:
        console.print("  [yellow]○ Learning system is gathering data[/yellow]")
    else:
        console.print("  [dim]○ Learning system is in early training[/dim]")


# =============================================================================
# patterns (list) command
# =============================================================================


def patterns_list(
    global_patterns: bool = typer.Option(
        True,
        "--global/--local",
        "-g/-l",
        help="Show global patterns (default) or local workspace patterns",
    ),
    min_priority: float = typer.Option(
        0.0,
        "--min-priority",
        "-p",
        help="Minimum priority score to display (0.0-1.0)",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Maximum number of patterns to display",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON for machine parsing",
    ),
    quarantined: bool = typer.Option(
        False,
        "--quarantined",
        "-q",
        help="Show only quarantined patterns",
    ),
    high_trust: bool = typer.Option(
        False,
        "--high-trust",
        help="Show only patterns with trust >= 0.7",
    ),
    low_trust: bool = typer.Option(
        False,
        "--low-trust",
        help="Show only patterns with trust <= 0.3",
    ),
) -> None:
    """View global learning patterns.

    Displays patterns learned from job executions across all workspaces.
    These patterns inform retry strategies, wait times, and validation.

    v19: Added quarantine status and trust score display and filtering.

    Examples:
        mozart patterns-list                  # Show global patterns
        mozart patterns-list --min-priority 0.5  # Only high-priority patterns
        mozart patterns-list --json           # JSON output for scripting
        mozart patterns-list --local          # Local workspace patterns only
        mozart patterns-list --quarantined    # Show quarantined patterns
        mozart patterns-list --high-trust     # Show trusted patterns only
    """
    from mozart.learning.global_store import QuarantineStatus, get_global_store

    store = get_global_store()

    # Build filter args for v19 quarantine/trust filtering
    filter_kwargs: dict[str, Any] = {
        "min_priority": min_priority,
        "limit": limit,
    }
    if quarantined:
        filter_kwargs["quarantine_status"] = QuarantineStatus.QUARANTINED
    if high_trust:
        filter_kwargs["min_trust"] = 0.7
    if low_trust:
        filter_kwargs["max_trust"] = 0.3

    if json_output:
        pattern_list = store.get_patterns(**filter_kwargs)
        output = []
        for p in pattern_list:
            output.append({
                "id": p.id,
                "pattern_type": p.pattern_type,
                "pattern_name": p.pattern_name,
                "description": p.description,
                "occurrence_count": p.occurrence_count,
                "effectiveness_score": round(p.effectiveness_score, 3),
                "priority_score": round(p.priority_score, 3),
                "context_tags": list(p.context_tags) if p.context_tags else [],
                # v19: Add quarantine and trust fields
                "quarantine_status": p.quarantine_status.value,
                "trust_score": round(p.trust_score, 3),
            })
        console.print(json_lib.dumps(output, indent=2))
        return

    # Get patterns from global store
    pattern_list = store.get_patterns(**filter_kwargs)

    if not pattern_list:
        console.print("[dim]No patterns found in global learning store.[/dim]")
        console.print(
            "\n[dim]Hint: Patterns are learned from job executions. "
            "Run jobs with learning enabled to build patterns.[/dim]"
        )
        return

    # Display patterns table - v19: added Status and Trust columns, v22: added Auto indicator
    table = Table(title="Global Learning Patterns")
    table.add_column("ID", style="cyan", no_wrap=True, width=10)
    table.add_column("Type", style="yellow", width=12)
    table.add_column("Name", style="bold", width=20)
    table.add_column("Status", width=10)  # v19: Quarantine status
    table.add_column("Trust", justify="right", width=6)  # v19: Trust score
    table.add_column("Auto", justify="center", width=4)  # v22: Auto-apply eligible
    table.add_column("Count", justify="right", width=5)
    table.add_column("Effect", justify="right", width=6)
    table.add_column("Prior", justify="right", width=5)

    for p in pattern_list:
        # v22: Check if pattern is auto-apply eligible
        # Default thresholds: trust >= 0.85, status = validated
        is_auto_eligible = (
            p.trust_score >= 0.85
            and p.quarantine_status.value == "validated"
        )
        auto_str = "[green]⚡[/green]" if is_auto_eligible else ""

        # Format effectiveness with color
        eff = p.effectiveness_score
        eff_color = "green" if eff > 0.7 else "yellow" if eff > 0.4 else "red"
        eff_str = f"[{eff_color}]{eff:.2f}[/{eff_color}]"

        # Format priority with color
        pri = p.priority_score
        pri_color = "green" if pri > 0.7 else "yellow" if pri > 0.4 else "dim"
        pri_str = f"[{pri_color}]{pri:.2f}[/{pri_color}]"

        # v19: Format quarantine status with color
        status = p.quarantine_status.value
        status_colors = {
            "pending": "dim",
            "quarantined": "red",
            "validated": "green",
            "retired": "dim italic",
        }
        status_color = status_colors.get(status, "dim")
        status_str = f"[{status_color}]{status}[/{status_color}]"

        # v19: Format trust score with color
        trust = p.trust_score
        trust_color = "green" if trust >= 0.7 else "yellow" if trust >= 0.4 else "red"
        trust_str = f"[{trust_color}]{trust:.2f}[/{trust_color}]"

        table.add_row(
            p.id[:10],
            p.pattern_type[:12],
            p.pattern_name[:20],
            status_str,
            trust_str,
            auto_str,
            str(p.occurrence_count),
            eff_str,
            pri_str,
        )

    console.print(table)
    console.print(f"\n[dim]Showing {len(pattern_list)} pattern(s)[/dim]")


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "patterns_why",
    "patterns_list",
    "learning_stats",
    "learning_insights",
    "learning_drift",
    "learning_epistemic_drift",
    "patterns_entropy",
    "patterns_budget",
    "entropy_status",
    "learning_activity",
]
