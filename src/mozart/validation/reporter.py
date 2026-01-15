"""Validation output formatting and reporting.

Provides formatted output for validation results, supporting both
terminal display with Rich and JSON output for tooling.
"""

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mozart.validation.base import ValidationIssue, ValidationSeverity


class ValidationReporter:
    """Formats and outputs validation results.

    Supports multiple output formats:
    - Terminal output with colors (default)
    - JSON for machine parsing
    - Plain text for logs
    """

    SEVERITY_COLORS = {
        ValidationSeverity.ERROR: "red",
        ValidationSeverity.WARNING: "yellow",
        ValidationSeverity.INFO: "blue",
    }

    SEVERITY_ICONS = {
        ValidationSeverity.ERROR: "✗",
        ValidationSeverity.WARNING: "!",
        ValidationSeverity.INFO: "i",
    }

    def __init__(self, console: Console | None = None):
        """Initialize reporter.

        Args:
            console: Rich Console for output. Creates one if not provided.
        """
        self.console = console or Console()

    def report_terminal(
        self,
        issues: list[ValidationIssue],
        config_name: str,
        show_passed: bool = True,
    ) -> None:
        """Output validation results to terminal with formatting.

        Args:
            issues: List of validation issues
            config_name: Name of the config being validated
            show_passed: Whether to show passed checks summary
        """
        self.console.print()

        # Count by severity
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        warnings = [i for i in issues if i.severity == ValidationSeverity.WARNING]
        infos = [i for i in issues if i.severity == ValidationSeverity.INFO]

        # Print header based on results
        if not issues:
            self.console.print(
                Panel(
                    f"[green]✓ Configuration valid:[/green] {config_name}",
                    border_style="green",
                )
            )
            return

        # Print issues by severity
        if errors:
            self._print_section("ERRORS (must fix before running)", errors, "red")

        if warnings:
            self._print_section("WARNINGS (may cause issues)", warnings, "yellow")

        if infos:
            self._print_section("INFO (consider reviewing)", infos, "blue")

        # Print summary
        self.console.print()
        summary_parts = []
        if errors:
            summary_parts.append(f"[red]{len(errors)} error{'s' if len(errors) > 1 else ''} (must fix)[/red]")
        if warnings:
            summary_parts.append(f"[yellow]{len(warnings)} warning{'s' if len(warnings) > 1 else ''} (should fix)[/yellow]")
        if infos:
            summary_parts.append(f"[blue]{len(infos)} info note{'s' if len(infos) > 1 else ''}[/blue]")

        self.console.print(f"Summary: {', '.join(summary_parts)}")

        # Final status
        if errors:
            self.console.print("\n[bold red]Validation: FAILED[/bold red]")
        else:
            self.console.print("\n[bold green]Validation: PASSED[/bold green] (with warnings)")

    def _print_section(
        self,
        title: str,
        issues: list[ValidationIssue],
        color: str,
    ) -> None:
        """Print a section of issues."""
        self.console.print(f"\n[{color} bold]{title}:[/{color} bold]")

        for issue in issues:
            icon = self.SEVERITY_ICONS.get(issue.severity, "?")
            self.console.print(f"  [{color}]{icon}[/{color}] {issue.format_short()}")

            if issue.context:
                # Show context with indicator
                context_display = issue.context
                if len(context_display) > 70:
                    context_display = context_display[:67] + "..."
                self.console.print(f"         [dim]{context_display}[/dim]")

            if issue.suggestion:
                self.console.print(f"         [cyan]Suggestion:[/cyan] {issue.suggestion}")

    def report_json(self, issues: list[ValidationIssue]) -> str:
        """Output validation results as JSON.

        Args:
            issues: List of validation issues

        Returns:
            JSON string representation
        """
        result: dict[str, Any] = {
            "valid": not any(i.severity == ValidationSeverity.ERROR for i in issues),
            "error_count": sum(1 for i in issues if i.severity == ValidationSeverity.ERROR),
            "warning_count": sum(1 for i in issues if i.severity == ValidationSeverity.WARNING),
            "info_count": sum(1 for i in issues if i.severity == ValidationSeverity.INFO),
            "issues": [self._issue_to_dict(i) for i in issues],
        }
        return json.dumps(result, indent=2)

    def _issue_to_dict(self, issue: ValidationIssue) -> dict[str, Any]:
        """Convert a ValidationIssue to a dictionary."""
        result: dict[str, Any] = {
            "check_id": issue.check_id,
            "severity": issue.severity.value,
            "message": issue.message,
        }

        if issue.line is not None:
            result["line"] = issue.line
        if issue.column is not None:
            result["column"] = issue.column
        if issue.context:
            result["context"] = issue.context
        if issue.suggestion:
            result["suggestion"] = issue.suggestion
        if issue.auto_fixable:
            result["auto_fixable"] = issue.auto_fixable
        if issue.metadata:
            result["metadata"] = issue.metadata

        return result

    def format_plain(self, issues: list[ValidationIssue]) -> str:
        """Format issues as plain text for logs.

        Args:
            issues: List of validation issues

        Returns:
            Plain text representation
        """
        if not issues:
            return "Validation passed: no issues found"

        lines = []
        for issue in issues:
            severity = issue.severity.value.upper()
            line_info = f" (line {issue.line})" if issue.line else ""
            lines.append(f"[{severity}] {issue.check_id}{line_info}: {issue.message}")

            if issue.suggestion:
                lines.append(f"    Suggestion: {issue.suggestion}")

        # Summary
        errors = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        warnings = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
        infos = sum(1 for i in issues if i.severity == ValidationSeverity.INFO)

        lines.append("")
        lines.append(f"Total: {errors} errors, {warnings} warnings, {infos} info")
        lines.append(f"Validation: {'FAILED' if errors else 'PASSED'}")

        return "\n".join(lines)
