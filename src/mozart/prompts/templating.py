"""Prompt templating for Mozart jobs.

Handles building sheet prompts from templates and generating
auto-completion prompts for partial sheet recovery.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2

from mozart.core.config import PromptConfig, ValidationRule

if TYPE_CHECKING:
    from mozart.execution.validation import HistoricalFailure, ValidationResult


@dataclass
class SheetContext:
    """Context for building a sheet prompt."""

    sheet_num: int
    total_sheets: int
    start_item: int
    end_item: int
    workspace: Path

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for template rendering."""
        return {
            "sheet_num": self.sheet_num,
            "total_sheets": self.total_sheets,
            "start_item": self.start_item,
            "end_item": self.end_item,
            "workspace": str(self.workspace),
        }


@dataclass
class CompletionContext:
    """Context for generating completion prompts.

    Uses ValidationResult objects (not just ValidationRule) to ensure
    that file paths are properly expanded with actual values like workspace
    and sheet_num, rather than showing template placeholders.
    """

    sheet_num: int
    total_sheets: int
    passed_validations: list["ValidationResult"]  # Changed from ValidationRule
    failed_validations: list["ValidationResult"]  # Changed from ValidationRule
    completion_attempt: int
    max_completion_attempts: int
    original_prompt: str
    workspace: Path


class PromptBuilder:
    """Builds prompts including completion prompts for partial recovery.

    Handles Jinja2 template rendering and auto-generation of completion
    prompts when a sheet partially completes.
    """

    def __init__(
        self,
        config: PromptConfig,
        jinja_env: jinja2.Environment | None = None,
    ) -> None:
        """Initialize prompt builder.

        Args:
            config: Prompt configuration from job config.
            jinja_env: Optional custom Jinja2 environment.
        """
        self.config = config
        self.env = jinja_env or jinja2.Environment(
            undefined=jinja2.StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )

    def build_sheet_context(
        self,
        sheet_num: int,
        total_sheets: int,
        sheet_size: int,
        total_items: int,
        start_item: int,
        workspace: Path,
    ) -> SheetContext:
        """Build sheet context from job parameters.

        Args:
            sheet_num: Current sheet number (1-indexed).
            total_sheets: Total number of sheets.
            sheet_size: Items per sheet.
            total_items: Total items to process.
            start_item: First item number (1-indexed).
            workspace: Workspace directory.

        Returns:
            SheetContext with calculated item range.
        """
        sheet_start = (sheet_num - 1) * sheet_size + start_item
        sheet_end = min(sheet_start + sheet_size - 1, total_items)

        return SheetContext(
            sheet_num=sheet_num,
            total_sheets=total_sheets,
            start_item=sheet_start,
            end_item=sheet_end,
            workspace=workspace,
        )

    def build_sheet_prompt(
        self,
        context: SheetContext,
        patterns: list[str] | None = None,
        validation_rules: list[ValidationRule] | None = None,
        failure_history: list["HistoricalFailure"] | None = None,
    ) -> str:
        """Build the standard sheet prompt from config.

        Args:
            context: Sheet context with item range and workspace.
            patterns: Optional list of learned pattern descriptions to inject.
            validation_rules: Optional list of validation rules to inject as
                requirements. These are the rules that will be checked after
                sheet execution - injecting them helps Claude understand
                exactly what success looks like.
            failure_history: Optional list of historical failures from previous
                sheets. Injected to help Claude learn from past mistakes and
                avoid repeating the same errors (Evolution v6).

        Returns:
            Rendered prompt string.
        """
        template_context = context.to_dict()

        # Merge config variables
        template_context.update(self.config.variables)

        # Add stakes and thinking method
        template_context["stakes"] = self.config.stakes or ""
        template_context["thinking_method"] = self.config.thinking_method or ""

        if self.config.template:
            template = self.env.from_string(self.config.template)
            prompt = template.render(**template_context)
        elif self.config.template_file and self.config.template_file.exists():
            template_content = self.config.template_file.read_text()
            template = self.env.from_string(template_content)
            prompt = template.render(**template_context)
        else:
            prompt = self._build_default_prompt(context)

        # Inject failure history first (lessons learned from past)
        if failure_history:
            history_section = self._format_historical_failures(failure_history)
            prompt = f"{prompt}\n\n{history_section}"

        # Inject learned patterns if available
        if patterns:
            pattern_section = self._format_patterns_section(patterns)
            prompt = f"{prompt}\n\n{pattern_section}"

        # Inject validation requirements if available
        if validation_rules:
            validation_section = self._format_validation_requirements(
                validation_rules, template_context
            )
            prompt = f"{prompt}\n\n{validation_section}"

        return prompt

    def _format_historical_failures(
        self, failures: list["HistoricalFailure"]
    ) -> str:
        """Format historical validation failures for prompt injection.

        Creates a "lessons learned" section from past validation failures
        to help Claude avoid repeating the same mistakes.

        Args:
            failures: List of HistoricalFailure objects from previous sheets.

        Returns:
            Formatted markdown section for prompt injection.
        """
        if not failures:
            return ""

        lines = ["## Lessons From Previous Sheets", ""]
        lines.append(
            "Previous sheets encountered these validation failures. "
            "Avoid repeating these mistakes:"
        )
        lines.append("")

        for i, failure in enumerate(failures[:5], 1):
            # Build failure summary
            category_tag = (
                f"[{failure.failure_category.upper()}]"
                if failure.failure_category
                else "[FAILED]"
            )

            lines.append(f"{i}. **Sheet {failure.sheet_num}** {category_tag}")
            lines.append(f"   - {failure.description}")

            if failure.failure_reason:
                lines.append(f"   - Issue: {failure.failure_reason}")

            if failure.suggested_fix:
                lines.append(f"   - Fix: {failure.suggested_fix}")

            lines.append("")

        lines.append(
            "**Learn from these failures** - ensure similar issues don't recur."
        )
        lines.append("")

        return "\n".join(lines)

    def _format_patterns_section(self, patterns: list[str]) -> str:
        """Format learned patterns as a prompt section.

        Args:
            patterns: List of pattern description strings.

        Returns:
            Formatted markdown section for prompt injection.
        """
        if not patterns:
            return ""

        lines = ["## Learned Patterns", ""]
        lines.append("Based on previous executions, here are relevant insights:")
        lines.append("")

        for i, pattern in enumerate(patterns[:5], 1):
            lines.append(f"{i}. {pattern}")

        lines.append("")
        lines.append("Consider these patterns when executing this sheet.")
        lines.append("")

        return "\n".join(lines)

    def _format_validation_requirements(
        self,
        rules: list[ValidationRule],
        template_context: dict[str, Any],
    ) -> str:
        """Format validation rules as success requirements for prompt injection.

        Converts validation rules into clear requirements that help Claude
        understand exactly what success looks like. Expands template variables
        in paths/patterns to show actual expected values.

        Args:
            rules: List of validation rules to format.
            template_context: Template variables for path expansion.

        Returns:
            Formatted markdown section for prompt injection.
        """
        if not rules:
            return ""

        lines = ["---", "## Success Requirements (Validated Automatically)", ""]
        lines.append(
            "Your output will be validated against these requirements. "
            "All must pass for success:"
        )
        lines.append("")

        for i, rule in enumerate(rules, 1):
            # Expand template variables in paths and patterns
            expanded_path = self._expand_template(
                rule.path or "", template_context
            )
            expanded_pattern = self._expand_template(
                rule.pattern or "", template_context
            )

            desc = rule.description or f"Requirement {i}"

            if rule.type == "file_exists":
                lines.append(f"{i}. **{desc}**")
                lines.append(f"   - Create file: `{expanded_path}`")

            elif rule.type == "file_modified":
                lines.append(f"{i}. **{desc}**")
                lines.append(f"   - Modify file: `{expanded_path}` (must update it during execution)")

            elif rule.type == "content_contains":
                lines.append(f"{i}. **{desc}**")
                lines.append(f"   - File: `{expanded_path}`")
                lines.append(f"   - Must contain exactly: `{expanded_pattern}`")

            elif rule.type == "content_regex":
                lines.append(f"{i}. **{desc}**")
                lines.append(f"   - File: `{expanded_path}`")
                lines.append(f"   - Must match pattern: `{expanded_pattern}`")

            elif rule.type == "command_succeeds":
                command = rule.command or ""
                display_cmd = command[:60] + "..." if len(command) > 60 else command
                lines.append(f"{i}. **{desc}**")
                lines.append(f"   - Command must succeed: `{display_cmd}`")

            lines.append("")

        lines.append(
            "**Important**: These validations run automatically. Ensure your output "
            "matches exactly - partial matches or variations may fail validation."
        )

        return "\n".join(lines)

    def _expand_template(self, value: str, context: dict[str, Any]) -> str:
        """Expand template variables in a string.

        Args:
            value: String potentially containing {workspace}, {sheet_num}, etc.
            context: Template variables to substitute.

        Returns:
            Expanded string with variables replaced.
        """
        if not value:
            return value

        try:
            # Handle simple {variable} format
            result = value
            for key, val in context.items():
                placeholder = f"{{{key}}}"
                if placeholder in result:
                    result = result.replace(placeholder, str(val))
            return result
        except Exception:
            # If expansion fails, return original
            return value

    def _build_default_prompt(self, context: SheetContext) -> str:
        """Build a simple default prompt when no template is provided.

        Args:
            context: Sheet context.

        Returns:
            Simple prompt string.
        """
        prompt = (
            f"Processing sheet {context.sheet_num} of {context.total_sheets} "
            f"(items {context.start_item}-{context.end_item})"
        )

        if self.config.stakes:
            prompt += f"\n\n{self.config.stakes}"

        if self.config.thinking_method:
            prompt += f"\n\n{self.config.thinking_method}"

        return prompt

    def build_completion_prompt(
        self,
        ctx: CompletionContext,
        semantic_hints: list[str] | None = None,
    ) -> str:
        """Generate auto-completion prompt for partial failures.

        This prompt tells the agent:
        1. What validations already passed (don't redo)
        2. What validations failed (focus on these)
        3. Semantic hints for focused recovery
        4. Clear instruction to complete only missing items

        Args:
            ctx: Completion context with passed/failed validations.
            semantic_hints: Optional list of actionable hints from semantic
                validation analysis. These are suggested fixes derived from
                the failure_category and suggested_fix fields of failed
                ValidationResults.

        Returns:
            Completion prompt string.
        """
        passed_section = self._format_passed_validations(ctx.passed_validations)
        failed_section = self._format_failed_validations(ctx.failed_validations)
        hints_section = self._format_semantic_hints(semantic_hints)

        # Truncate original prompt if very long
        original_context = ctx.original_prompt
        if len(original_context) > 3000:
            truncation_msg = "\n\n[... original prompt truncated for brevity ...]"
            original_context = original_context[:3000] + truncation_msg

        completion_prompt = f"""## COMPLETION MODE - Sheet {ctx.sheet_num}

This is completion attempt {ctx.completion_attempt} of {ctx.max_completion_attempts}.

A previous execution of this sheet partially completed. Your job is to \
finish ONLY the incomplete items.

### ALREADY COMPLETED (DO NOT REDO)
The following outputs were successfully created and validated:
{passed_section}

These files exist and are valid. DO NOT recreate or modify them unless absolutely necessary.

### INCOMPLETE ITEMS (FOCUS HERE)
The following validations failed and need to be completed:
{failed_section}
{hints_section}
### INSTRUCTIONS
1. Review what already exists to understand the context
2. Complete ONLY the missing items listed above
3. Do not duplicate work that was already done
4. Ensure all validation requirements are met before finishing

### ORIGINAL TASK CONTEXT
{original_context}

---
Focus on completing the missing items. Do not start over from scratch."""

        return completion_prompt.strip()

    def _format_semantic_hints(self, hints: list[str] | None) -> str:
        """Format semantic hints for injection into completion prompt.

        Args:
            hints: List of actionable hints from semantic validation analysis.

        Returns:
            Formatted string with hints section, or empty string if no hints.
        """
        if not hints:
            return ""

        lines = [
            "",
            "### SUGGESTED FIXES (from validation analysis)",
        ]
        for i, hint in enumerate(hints, 1):
            lines.append(f"  {i}. {hint}")
        lines.append("")

        return "\n".join(lines)

    def _format_passed_validations(
        self, results: list["ValidationResult"]
    ) -> str:
        """Format passed validations for the completion prompt.

        Args:
            results: List of validation results that passed.
                     Uses results (not rules) to get expanded paths.

        Returns:
            Formatted string for prompt.
        """
        if not results:
            return "  (none)"

        lines = []
        for result in results:
            rule = result.rule
            # Use expanded path from result, fallback to rule description
            expanded_path = result.expected_value or result.actual_value
            desc = rule.description or expanded_path or rule.pattern or "Unnamed validation"

            if rule.type == "file_exists":
                lines.append(f"  - [CREATED] {desc}")
                if expanded_path:
                    lines.append(f"    File: {expanded_path}")
            elif rule.type == "file_modified":
                lines.append(f"  - [UPDATED] {desc}")
                if expanded_path:
                    # Extract just the path from "mtime>..." format
                    file_path = result.actual_value
                    if file_path and file_path.startswith("mtime="):
                        file_path = result.expected_value  # Fallback
                    lines.append(f"    File: {expanded_path}")
            elif rule.type in ("content_contains", "content_regex"):
                lines.append(f"  - [VERIFIED] {desc}")
                if expanded_path:
                    lines.append(f"    File: {expanded_path}")

        return "\n".join(lines)

    def _format_failed_validations(
        self, results: list["ValidationResult"]
    ) -> str:
        """Format failed validations for the completion prompt.

        Provides actionable information about what needs to be done.
        Uses ValidationResult objects to get expanded file paths and
        semantic failure reasons.

        Args:
            results: List of validation results that failed.

        Returns:
            Formatted string for prompt.
        """
        if not results:
            return "  (none)"

        lines = []
        for i, result in enumerate(results, 1):
            rule = result.rule
            desc = rule.description or "Unnamed validation"

            # Use semantic failure information if available (Priority 2 evolution)
            if result.failure_category and result.failure_reason:
                lines.append(f"  {i}. [{result.failure_category.upper()}] {desc}")
                lines.append(f"     Why: {result.failure_reason}")
                if result.suggested_fix:
                    lines.append(f"     Fix: {result.suggested_fix}")
            else:
                # Fallback to legacy formatting for backwards compatibility
                expanded_path = result.expected_value or result.actual_value

                if rule.type == "file_exists":
                    lines.append(f"  {i}. [MISSING] {desc}")
                    if expanded_path:
                        lines.append(f"     Expected file: {expanded_path}")
                        lines.append("     Action: Create this file with the required content")

                elif rule.type == "file_modified":
                    lines.append(f"  {i}. [NOT UPDATED] {desc}")
                    # For file_modified, error_message contains the actual path
                    actual_path = None
                    if result.error_message and ":" in result.error_message:
                        actual_path = result.error_message.split(": ", 1)[-1]
                    display_path = actual_path or expanded_path or rule.path
                    if display_path:
                        lines.append(f"     File needs modification: {display_path}")
                        lines.append(
                            "     Action: You MUST append/write new content to this file."
                        )

                elif rule.type == "content_contains":
                    lines.append(f"  {i}. [CONTENT MISSING] {desc}")
                    if rule.pattern:
                        lines.append(f"     Required text: {rule.pattern}")
                    if expanded_path:
                        lines.append(f"     In file: {expanded_path}")
                    lines.append("     Action: Add the required content to the file")

                elif rule.type == "content_regex":
                    lines.append(f"  {i}. [PATTERN NOT MATCHED] {desc}")
                    if rule.pattern:
                        lines.append(f"     Required pattern: {rule.pattern}")
                    if expanded_path:
                        lines.append(f"     In file: {expanded_path}")
                    lines.append("     Action: Ensure file content matches the pattern")

                elif rule.type == "command_succeeds":
                    lines.append(f"  {i}. [COMMAND FAILED] {desc}")
                    if result.error_message:
                        # Truncate for readability
                        err_summary = result.error_message[:200]
                        if len(result.error_message) > 200:
                            err_summary += "..."
                        lines.append(f"     Error: {err_summary}")
                    lines.append("     Action: Fix the command errors")

            lines.append("")  # Blank line between items

        return "\n".join(lines).rstrip()


def build_sheet_prompt_simple(
    config: PromptConfig,
    sheet_num: int,
    total_sheets: int,
    sheet_size: int,
    total_items: int,
    start_item: int,
    workspace: Path,
) -> str:
    """Convenience function to build a sheet prompt.

    This provides a simpler interface for cases where you don't need
    to reuse the PromptBuilder.

    Args:
        config: Prompt configuration.
        sheet_num: Current sheet number.
        total_sheets: Total number of sheets.
        sheet_size: Items per sheet.
        total_items: Total items.
        start_item: First item number.
        workspace: Workspace directory.

    Returns:
        Rendered prompt string.
    """
    builder = PromptBuilder(config)
    context = builder.build_sheet_context(
        sheet_num=sheet_num,
        total_sheets=total_sheets,
        sheet_size=sheet_size,
        total_items=total_items,
        start_item=start_item,
        workspace=workspace,
    )
    return builder.build_sheet_prompt(context)
