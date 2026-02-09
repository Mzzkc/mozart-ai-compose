"""Entropy monitoring and response commands.

Commands:
- patterns-entropy: Monitor pattern population diversity using Shannon entropy
- entropy-status: Display entropy response status and history
"""

from __future__ import annotations

import json as json_lib
from typing import Any

import typer
from rich.table import Table

from ...output import console


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

    Shannon entropy measures how evenly patterns are used:
    - High entropy (H -> max): Healthy diversity, many patterns contribute
    - Low entropy (H -> 0): Single pattern dominates (collapse risk)

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

        table = Table(title="Pattern Entropy History")
        table.add_column("Time", style="dim", width=20)
        table.add_column("Shannon H", justify="right", width=10)
        table.add_column("Diversity", justify="right", width=10)
        table.add_column("Unique", justify="right", width=8)
        table.add_column("Effective", justify="right", width=10)
        table.add_column("Applications", justify="right", width=12)
        table.add_column("Dominant %", justify="right", width=10)

        for h in history_records:
            div_color = "green" if h.diversity_index >= alert_threshold else "red"
            div_str = f"[{div_color}]{h.diversity_index:.3f}[/{div_color}]"

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

    metrics = store.calculate_pattern_entropy()  # type: ignore[attr-defined]
    metrics.threshold_exceeded = metrics.diversity_index < alert_threshold

    if record:
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

    console.print("[bold]Pattern Population Entropy[/bold]\n")

    if metrics.total_applications == 0:
        console.print("[dim]No pattern applications yet.[/dim]")
        console.print("\n[dim]Hint: Run jobs with learning enabled to build patterns.[/dim]")
        return

    console.print(f"  Shannon Entropy (H): [cyan]{metrics.shannon_entropy:.4f}[/cyan] bits")
    console.print(f"  Max Possible (H_max): [dim]{metrics.max_possible_entropy:.4f}[/dim] bits")

    div_color = "green" if metrics.diversity_index >= alert_threshold else "red"
    console.print(
        f"  Diversity Index: [{div_color}]{metrics.diversity_index:.4f}"
        f"[/{div_color}] (threshold: {alert_threshold})"
    )

    console.print("")

    console.print(f"  Unique Patterns: [yellow]{metrics.unique_pattern_count}[/yellow]")
    console.print(
        f"  Effective Patterns: [yellow]{metrics.effective_pattern_count}"
        "[/yellow] (with ≥1 application)"
    )
    console.print(f"  Total Applications: [yellow]{metrics.total_applications}[/yellow]")

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

    console.print("[bold]Entropy Response Status[/bold]\n")

    if stats["total_responses"] == 0:
        console.print("[dim]No entropy responses recorded yet.[/dim]")
        console.print("\n[dim]Hint: Enable entropy_response in learning config:[/dim]")
        console.print("[dim]  learning:[/dim]")
        console.print("[dim]    entropy_response:[/dim]")
        console.print("[dim]      enabled: true[/dim]")
        return

    console.print(f"  Total Responses: [yellow]{stats['total_responses']}[/yellow]")
    console.print(f"  Avg Trigger Entropy: [red]{stats['avg_entropy_at_trigger']:.4f}[/red]")
    console.print(f"  Budget Boosts: [green]{stats['budget_boosts']}[/green]")
    console.print(f"  Quarantine Revisits: [cyan]{stats['quarantine_revisits']}[/cyan]")

    console.print("")

    if last:
        console.print("[bold]Last Response[/bold]")
        console.print(f"  Time: [dim]{last.recorded_at.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
        console.print(f"  Entropy at Trigger: [red]{last.entropy_at_trigger:.4f}[/red]")
        console.print(f"  Threshold: [dim]{last.threshold_used:.4f}[/dim]")
        actions = ', '.join(last.actions_taken) if last.actions_taken else '[dim]none[/dim]'
        console.print(f"  Actions: {actions}")

    console.print("")
    if stats["total_responses"] > 10:
        console.print("[yellow]⚠ Many responses triggered[/yellow]")
        console.print("[dim]Pattern diversity may be consistently low. Review patterns.[/dim]")
    elif stats["quarantine_revisits"] > stats["total_responses"]:
        console.print("[cyan]ℹ Active quarantine revisiting[/cyan]")
        console.print("[dim]Previously problematic patterns are being reconsidered.[/dim]")
    else:
        console.print("[green]✓ Entropy response system active[/green]")
