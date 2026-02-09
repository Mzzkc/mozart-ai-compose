"""Validation framework for sheet outputs.

Executes validation rules against sheet outputs and tracks results
for partial completion recovery.

This module provides:
- ValidationResult: Result of a single validation check
- SheetValidationResult: Aggregate result for a sheet
- ValidationEngine: Runs validation rules against outputs
- FailureHistoryStore: Queries past validation failures for history-aware prompts
"""

import asyncio
import logging
import re
import shlex
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mozart.core.checkpoint import ValidationDetailDict
from mozart.core.config import ValidationRule
from mozart.core.constants import VALIDATION_COMMAND_TIMEOUT_SECONDS, VALIDATION_OUTPUT_TRUNCATE_CHARS
from mozart.utils.time import utc_now

_logger = logging.getLogger("mozart.execution.validation")

if TYPE_CHECKING:
    from mozart.core.checkpoint import CheckpointState


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    rule: ValidationRule
    passed: bool
    actual_value: str | None = None
    expected_value: str | None = None
    error_message: str | None = None
    checked_at: datetime = field(default_factory=utc_now)
    check_duration_ms: float = 0.0
    # Learning metadata (Phase 1: Learning Foundation)
    confidence: float = 1.0
    """Confidence in this validation result (0.0-1.0). Default 1.0 = fully confident."""
    confidence_factors: dict[str, float] = field(default_factory=dict)
    """Factors affecting confidence, e.g., {'file_age': 0.9, 'pattern_specificity': 0.8}."""
    # Semantic validation (Priority 2: Semantic Validation with Why)
    failure_reason: str | None = None
    """Semantic explanation of why validation failed.
    Example: 'File was not created during execution'
    Example: 'Content missing required pattern: class MyComponent'"""
    failure_category: str | None = None
    """Category of failure: 'missing', 'malformed', 'incomplete', 'stale', 'error'."""
    suggested_fix: str | None = None
    """Hint for how to fix the issue.
    Example: 'Ensure the file is created in workspace/output/'"""

    def to_dict(self) -> ValidationDetailDict:
        """Convert to serializable dictionary."""
        # type: ignore needed because mypy/pyright cannot infer a dict literal
        # as matching a total=False TypedDict — all keys verified via schema
        return {
            "rule_type": self.rule.type,
            "description": self.rule.description,
            "path": self.rule.path,
            "pattern": self.rule.pattern,
            "passed": self.passed,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
            "error_message": self.error_message,
            "checked_at": self.checked_at.isoformat(),
            "check_duration_ms": self.check_duration_ms,
            "confidence": self.confidence,
            "confidence_factors": self.confidence_factors,
            "failure_reason": self.failure_reason,
            "failure_category": self.failure_category,
            "suggested_fix": self.suggested_fix,
        }  # type: ignore[return-value]

    def format_failure_summary(self) -> str:
        """Format failure information for prompt injection.

        Returns:
            Formatted string with category, reason, and fix hint.
            Empty string if validation passed.
        """
        if self.passed:
            return ""

        parts: list[str] = []
        if self.failure_category:
            parts.append(f"[{self.failure_category.upper()}]")
        if self.failure_reason:
            parts.append(self.failure_reason)
        if self.suggested_fix:
            parts.append(f"Fix: {self.suggested_fix}")

        return " ".join(parts)


@dataclass
class SheetValidationResult:
    """Aggregate result of all validations for a sheet."""

    sheet_num: int
    results: list[ValidationResult]
    rules_checked: int = 0

    @property
    def all_passed(self) -> bool:
        """Check if all validations passed.

        Returns True only when at least one rule was checked and all passed.
        Returns True for empty results (no applicable rules) for backward
        compatibility — callers should check rules_checked if they need
        to distinguish "nothing checked" from "all passed."
        """
        return all(r.passed for r in self.results)

    @property
    def passed_count(self) -> int:
        """Count of passed validations."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        """Count of failed validations (excluding skipped)."""
        return sum(
            1 for r in self.results
            if not r.passed and r.failure_category != "skipped"
        )

    @property
    def skipped_count(self) -> int:
        """Count of skipped validations (due to staged fail-fast)."""
        return sum(
            1 for r in self.results
            if r.failure_category == "skipped"
        )

    @property
    def executed_count(self) -> int:
        """Count of validations that actually executed (not skipped)."""
        return len(self.results) - self.skipped_count

    @property
    def pass_percentage(self) -> float:
        """Percentage of validations that passed (including skipped as failed).

        For staged validation decisions, use executed_pass_percentage instead.
        """
        if not self.results:
            return 100.0  # No validations = all passed
        return (self.passed_count / len(self.results)) * 100

    @property
    def executed_pass_percentage(self) -> float:
        """Percentage of EXECUTED validations that passed.

        Excludes skipped validations from the calculation. This is more
        appropriate for completion mode decisions with staged validations,
        since skipped validations weren't given a chance to run.
        """
        executed = self.executed_count
        if executed == 0:
            return 100.0  # All skipped = nothing to judge
        return (self.passed_count / executed) * 100

    @property
    def majority_passed(self) -> bool:
        """Returns True if >50% of validations passed."""
        return self.pass_percentage > 50.0

    @property
    def aggregate_confidence(self) -> float:
        """Calculate weighted aggregate confidence across all validation results.

        Weighting strategy:
        - Passed validations contribute their full confidence
        - Failed validations contribute their confidence with a penalty
        - Empty results return 1.0 (no evidence = assume confident)

        Returns:
            Weighted average confidence (0.0-1.0).
        """
        if not self.results:
            return 1.0

        # Weight passed validations higher (pass=1.0, fail=0.5)
        # This reflects that successful validations are more informative
        total_weight = 0.0
        weighted_sum = 0.0

        for result in self.results:
            weight = 1.0 if result.passed else 0.5
            weighted_sum += result.confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 1.0

    def get_passed_rules(self) -> list[ValidationRule]:
        """Get rules that passed."""
        return [r.rule for r in self.results if r.passed]

    def get_failed_rules(self) -> list[ValidationRule]:
        """Get rules that failed."""
        return [r.rule for r in self.results if not r.passed]

    def get_passed_results(self) -> list[ValidationResult]:
        """Get results that passed."""
        return [r for r in self.results if r.passed]

    def get_failed_results(self) -> list[ValidationResult]:
        """Get results that failed."""
        return [r for r in self.results if not r.passed]

    def to_dict_list(self) -> list[ValidationDetailDict]:
        """Convert all results to serializable list."""
        return [r.to_dict() for r in self.results]

    def get_semantic_summary(self) -> dict[str, Any]:
        """Aggregate semantic information from failed validations.

        Returns a summary of failure categories and their counts,
        useful for making retry strategy decisions.

        Returns:
            Dictionary with:
            - category_counts: dict mapping category to count
            - dominant_category: the most common failure category
            - has_semantic_info: whether any result has semantic fields
        """
        category_counts: dict[str, int] = {}
        has_semantic_info = False

        for result in self.results:
            if not result.passed and result.failure_category:
                has_semantic_info = True
                category = result.failure_category
                category_counts[category] = category_counts.get(category, 0) + 1

        # Find dominant category
        dominant_category: str | None = None
        if category_counts:
            dominant_category = max(category_counts, key=lambda k: category_counts[k])

        return {
            "category_counts": category_counts,
            "dominant_category": dominant_category,
            "has_semantic_info": has_semantic_info,
            "total_failures": self.failed_count,
        }

    def get_actionable_hints(self, limit: int = 3) -> list[str]:
        """Extract actionable hints from failed validations.

        Collects suggested_fix values from failed ValidationResults,
        useful for injecting into completion prompts.

        Args:
            limit: Maximum number of hints to return. Defaults to 3
                   to avoid overwhelming the completion prompt.

        Returns:
            List of suggested_fix strings, deduplicated and limited.
        """
        hints: list[str] = []
        seen: set[str] = set()

        for result in self.results:
            if not result.passed and result.suggested_fix:
                # Truncate long hints for prompt brevity
                hint = result.suggested_fix
                if len(hint) > 100:
                    hint = hint[:97] + "..."

                if hint not in seen:
                    seen.add(hint)
                    hints.append(hint)

                if len(hints) >= limit:
                    break

        return hints


class FileModificationTracker:
    """Tracks file mtimes before sheet execution for file_modified checks.

    The file_modified validation checks if a file was updated during sheet
    execution. This tracker snapshots mtimes before execution so we can
    compare after execution completes.
    """

    def __init__(self) -> None:
        self._mtimes: dict[str, float] = {}

    def snapshot(self, paths: list[Path]) -> None:
        """Capture mtimes of files before sheet execution.

        Args:
            paths: List of file paths to track. Non-existent files are
                   recorded with mtime=0.0 so we can detect creation.
        """
        for path in paths:
            path_str = str(path.resolve())
            if path.exists():
                self._mtimes[path_str] = path.stat().st_mtime
            else:
                self._mtimes[path_str] = 0.0  # File doesn't exist yet

    def was_modified(self, path: Path) -> bool:
        """Check if file was modified (or created) after snapshot.

        Args:
            path: File path to check.

        Returns:
            True if file's mtime is newer than snapshot, or if file
            was created after snapshot (didn't exist before).
        """
        resolved = path.resolve()
        try:
            current_mtime = resolved.stat().st_mtime
        except (OSError, ValueError):
            return False  # File doesn't exist or path is invalid
        original_mtime = self._mtimes.get(str(resolved), 0.0)
        return current_mtime > original_mtime

    def get_original_mtime(self, path: Path) -> float | None:
        """Get the original mtime from snapshot."""
        path_str = str(path.resolve())
        return self._mtimes.get(path_str)

    def clear(self) -> None:
        """Clear all tracked mtimes."""
        self._mtimes.clear()


class ValidationEngine:
    """Executes validation rules against sheet outputs.

    Handles path template expansion and dispatches to type-specific
    validation methods.
    """

    def __init__(self, workspace: Path, sheet_context: dict[str, Any]) -> None:
        """Initialize validation engine.

        Args:
            workspace: Base workspace directory.
            sheet_context: Context dict with sheet_num, start_item, end_item, etc.
        """
        self.workspace = workspace.resolve()
        self.sheet_context = sheet_context
        self._mtime_tracker = FileModificationTracker()

    def expand_path(self, path_template: str) -> Path:
        """Expand path template with sheet context variables.

        Supports: {sheet_num}, {workspace}, {start_item}, {end_item}

        Args:
            path_template: Path with {variable} placeholders.

        Returns:
            Expanded Path object.

        Raises:
            ValueError: If expanded path resolves outside the workspace directory.
        """
        # Build context, ensuring workspace is set correctly
        context = dict(self.sheet_context)
        context["workspace"] = str(self.workspace)

        expanded = path_template.format(**context)
        resolved = Path(expanded).resolve()

        if not resolved.is_relative_to(self.workspace):
            _logger.warning(
                "path_traversal_blocked",
                template=path_template,
                resolved=str(resolved),
                workspace=str(self.workspace),
            )
            raise ValueError(
                f"Path '{expanded}' resolves to '{resolved}' which is outside "
                f"workspace '{self.workspace}'"
            )

        return Path(expanded)

    def snapshot_mtime_files(self, rules: list[ValidationRule]) -> None:
        """Snapshot mtimes for all file_modified rules before sheet execution.

        Call this BEFORE running the sheet so we can detect modifications.

        Args:
            rules: List of validation rules to scan for file_modified types.
        """
        paths = [
            self.expand_path(r.path)
            for r in rules
            if r.type == "file_modified" and r.path
        ]
        self._mtime_tracker.snapshot(paths)

    def _check_condition(self, condition: str | None) -> bool:
        """Check if a validation condition is satisfied.

        Supports simple expressions like:
        - "sheet_num >= 6"
        - "sheet_num == 3"
        - "sheet_num <= 5"
        - "sheet_num > 2"
        - "sheet_num < 8"

        Also supports compound conditions with 'and':
        - "sheet_num >= 2 and sheet_num < 12"

        Args:
            condition: Condition expression string, or None to always match.

        Returns:
            True if condition is satisfied or None, False otherwise.
        """
        if condition is None:
            return True

        condition = condition.strip()

        # Handle compound conditions with 'and'
        if " and " in condition:
            parts = condition.split(" and ")
            return all(self._check_single_condition(p.strip()) for p in parts)

        return self._check_single_condition(condition)

    def _check_single_condition(self, condition: str) -> bool:
        """Check a single comparison condition.

        Args:
            condition: Single condition like "sheet_num >= 6".

        Returns:
            True if condition is satisfied, False otherwise.
        """
        sheet_num = self.sheet_context.get("sheet_num", 0)

        # Try to match pattern: variable operator value
        match = re.match(r"(\w+)\s*(>=|<=|==|!=|>|<)\s*(\d+)", condition)
        if not match:
            # Unknown condition format - default to True (don't skip)
            return True

        var_name, operator, value_str = match.groups()
        value = int(value_str)

        # Get the variable value
        var_value: int
        if var_name == "sheet_num":
            var_value = sheet_num
        else:
            # Unknown variable - check if it's in context
            ctx_value = self.sheet_context.get(var_name)
            if ctx_value is None or not isinstance(ctx_value, int):
                return True  # Unknown or non-int variable - don't skip
            var_value = ctx_value

        # Evaluate the condition
        if operator == ">=":
            return bool(var_value >= value)
        elif operator == "<=":
            return bool(var_value <= value)
        elif operator == "==":
            return bool(var_value == value)
        elif operator == "!=":
            return bool(var_value != value)
        elif operator == ">":
            return bool(var_value > value)
        elif operator == "<":
            return bool(var_value < value)
        else:
            return True  # Unknown operator - don't skip

    def _filter_applicable_rules(
        self, rules: list[ValidationRule]
    ) -> list[ValidationRule]:
        """Filter rules to only those whose conditions are satisfied.

        Args:
            rules: List of all validation rules.

        Returns:
            List of rules that apply to the current sheet.
        """
        return [r for r in rules if self._check_condition(r.condition)]

    def get_applicable_rules(
        self, rules: list[ValidationRule]
    ) -> list[ValidationRule]:
        """Get rules that apply to the current sheet context.

        This public wrapper exists so that ``PromptBuilder`` can query which
        rules apply *before* execution (to inject validation requirements
        into prompts), while ``_filter_applicable_rules`` remains a private
        implementation detail of the validation engine.

        Args:
            rules: List of all validation rules.

        Returns:
            List of rules that apply to the current sheet (condition satisfied).
        """
        return self._filter_applicable_rules(rules)

    async def run_validations(self, rules: list[ValidationRule]) -> SheetValidationResult:
        """Execute all validation rules and return aggregate result.

        This method runs all validations without staging. For staged execution
        (fail-fast on stage failure), use run_staged_validations() instead.

        Rules with conditions that don't match the current sheet context are
        automatically skipped (e.g., condition="sheet_num >= 6" on sheet 1).

        Args:
            rules: List of validation rules to execute.

        Returns:
            SheetValidationResult with all individual results.
        """
        # Filter to only applicable rules based on conditions
        applicable_rules = self._filter_applicable_rules(rules)
        results: list[ValidationResult] = []

        for rule in applicable_rules:
            result = await self._run_single_validation(rule)
            results.append(result)

        return SheetValidationResult(
            sheet_num=self.sheet_context.get("sheet_num", 0),
            results=results,
            rules_checked=len(applicable_rules),
        )

    async def run_staged_validations(
        self, rules: list[ValidationRule]
    ) -> tuple[SheetValidationResult, int | None]:
        """Execute validations in stage order with fail-fast behavior.

        Validations are grouped by stage and run in ascending order.
        If any validation in a stage fails, higher stages are skipped.

        Rules with conditions that don't match the current sheet context are
        automatically skipped (e.g., condition="sheet_num >= 6" on sheet 1).

        Args:
            rules: List of validation rules to execute.

        Returns:
            Tuple of (SheetValidationResult, failed_stage or None).
            failed_stage is the stage number that failed, or None if all passed.
        """
        # Filter to only applicable rules based on conditions
        applicable_rules = self._filter_applicable_rules(rules)

        if not applicable_rules:
            return SheetValidationResult(
                sheet_num=self.sheet_context.get("sheet_num", 0),
                results=[],
                rules_checked=0,
            ), None

        # Group rules by stage
        stages: dict[int, list[ValidationRule]] = {}
        for rule in applicable_rules:
            stage = rule.stage
            if stage not in stages:
                stages[stage] = []
            stages[stage].append(rule)

        # Run stages in order
        all_results: list[ValidationResult] = []
        failed_stage: int | None = None

        for stage_num in sorted(stages.keys()):
            stage_rules = stages[stage_num]
            stage_passed = True

            for rule in stage_rules:
                result = await self._run_single_validation(rule)
                all_results.append(result)
                if not result.passed:
                    stage_passed = False

            if not stage_passed:
                failed_stage = stage_num
                self._mark_remaining_stages_skipped(
                    stages, stage_num, all_results
                )
                break

        return SheetValidationResult(
            sheet_num=self.sheet_context.get("sheet_num", 0),
            results=all_results,
            rules_checked=len(applicable_rules),
        ), failed_stage

    @staticmethod
    def _mark_remaining_stages_skipped(
        stages: dict[int, list[ValidationRule]],
        failed_stage: int,
        results: list[ValidationResult],
    ) -> None:
        """Mark all rules in stages after the failed stage as skipped.

        Args:
            stages: Mapping of stage number to rules in that stage.
            failed_stage: The stage number that failed.
            results: List to append skipped ValidationResults to.
        """
        for remaining_stage in sorted(stages.keys()):
            if remaining_stage > failed_stage:
                for rule in stages[remaining_stage]:
                    results.append(
                        ValidationResult(
                            rule=rule,
                            passed=False,
                            error_message=f"Skipped: Stage {failed_stage} failed",
                            failure_reason=f"Skipped due to failure in stage {failed_stage}",
                            failure_category="skipped",
                            confidence=0.0,
                        )
                    )

    # Validation types that benefit from retry logic (filesystem race conditions)
    _RETRYABLE_VALIDATION_TYPES = frozenset({
        "file_exists",
        "file_modified",
        "content_contains",
        "content_regex",
        "command_succeeds",
    })

    async def _run_single_validation(self, rule: ValidationRule) -> ValidationResult:
        """Execute a single validation rule with optional retry logic.

        For file-based validations, retries help handle filesystem race conditions
        where the sheet creates/modifies files that are immediately validated.

        Args:
            rule: The validation rule to execute.

        Returns:
            ValidationResult with pass/fail status and details.
        """
        start = time.monotonic()

        # Determine retry behavior
        should_retry = (
            rule.type in self._RETRYABLE_VALIDATION_TYPES
            and rule.retry_count > 0
        )
        max_attempts = rule.retry_count + 1 if should_retry else 1
        delay_seconds = rule.retry_delay_ms / 1000.0

        last_result: ValidationResult | None = None

        for attempt in range(max_attempts):
            try:
                if rule.type == "file_exists":
                    result = self._check_file_exists(rule)
                elif rule.type == "file_modified":
                    result = self._check_file_modified(rule)
                elif rule.type == "content_contains":
                    result = self._check_content_contains(rule)
                elif rule.type == "content_regex":
                    result = self._check_content_regex(rule)
                elif rule.type == "command_succeeds":
                    result = self._check_command_succeeds(rule)
                else:
                    result = ValidationResult(
                        rule=rule,
                        passed=False,
                        error_message=f"Unknown validation type: {rule.type}",
                    )

                # If passed, return immediately
                if result.passed:
                    result.check_duration_ms = (time.monotonic() - start) * 1000
                    return result

                last_result = result

                # If more attempts remaining, wait and retry (non-blocking)
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay_seconds)

            except Exception as e:
                last_result = ValidationResult(
                    rule=rule,
                    passed=False,
                    expected_value=rule.path or rule.pattern,
                    error_message=f"Validation error: {e}",
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay_seconds)

        # Return the last failed result
        if last_result:
            last_result.check_duration_ms = (time.monotonic() - start) * 1000
            return last_result

        # Fallback (shouldn't reach here)
        return ValidationResult(
            rule=rule,
            passed=False,
            error_message="Validation failed after all attempts",
            check_duration_ms=(time.monotonic() - start) * 1000,
        )

    def _check_file_exists(self, rule: ValidationRule) -> ValidationResult:
        """Check if a file exists.

        Args:
            rule: Validation rule with path template.

        Returns:
            ValidationResult indicating if file exists.
        """
        if not rule.path:
            return ValidationResult(
                rule=rule,
                passed=False,
                error_message="file_exists rule requires 'path' field",
                failure_reason="Validation rule is missing required 'path' field",
                failure_category="error",
                suggested_fix="Add 'path' field to the validation rule configuration",
            )

        path = self.expand_path(rule.path)
        exists = path.exists() and path.is_file()

        if exists:
            return ValidationResult(
                rule=rule,
                passed=True,
                actual_value=str(path),
                expected_value=str(path),
            )

        # Use relative path for cleaner error messages
        display_path = path.name if len(str(path)) > 50 else str(path)

        return ValidationResult(
            rule=rule,
            passed=False,
            actual_value=None,
            expected_value=str(path),
            error_message=f"File not found: {path}",
            failure_reason=f"File '{display_path}' does not exist",
            failure_category="missing",
            suggested_fix=f"Create file at: {path}",
        )

    def _check_file_modified(self, rule: ValidationRule) -> ValidationResult:
        """Check if a file was modified after sheet started.

        Requires snapshot_mtime_files() to be called before sheet execution.

        Args:
            rule: Validation rule with path template.

        Returns:
            ValidationResult indicating if file was modified.
        """
        if not rule.path:
            return ValidationResult(
                rule=rule,
                passed=False,
                error_message="file_modified rule requires 'path' field",
                failure_reason="Validation rule is missing required 'path' field",
                failure_category="error",
                suggested_fix="Add 'path' field to the validation rule configuration",
            )

        path = self.expand_path(rule.path)
        display_path = path.name if len(str(path)) > 50 else str(path)

        if not path.exists():
            return ValidationResult(
                rule=rule,
                passed=False,
                actual_value=None,
                expected_value=str(path),
                error_message=f"File does not exist: {path}",
                failure_reason=f"File '{display_path}' does not exist (cannot check modification)",
                failure_category="missing",
                suggested_fix="Create the file first, then modify it",
            )

        was_modified = self._mtime_tracker.was_modified(path)
        original_mtime = self._mtime_tracker.get_original_mtime(path)

        if was_modified:
            return ValidationResult(
                rule=rule,
                passed=True,
                actual_value=f"mtime={path.stat().st_mtime:.6f}",
                expected_value=f"mtime>{original_mtime:.6f}" if original_mtime else "modified",
            )

        return ValidationResult(
            rule=rule,
            passed=False,
            actual_value=f"mtime={path.stat().st_mtime:.6f}",
            expected_value=f"mtime>{original_mtime:.6f}" if original_mtime else "modified",
            error_message=f"File not modified: {path}",
            failure_reason=f"File '{display_path}' was not modified during execution",
            failure_category="stale",
            suggested_fix="Verify the task updates this file with new content",
        )

    def _check_content_contains(self, rule: ValidationRule) -> ValidationResult:
        """Check if file contains expected content.

        Args:
            rule: Validation rule with path and pattern.

        Returns:
            ValidationResult indicating if content was found.
        """
        if not rule.path:
            return ValidationResult(
                rule=rule,
                passed=False,
                error_message="content_contains rule requires 'path' field",
                failure_reason="Validation rule is missing required 'path' field",
                failure_category="error",
                suggested_fix="Add 'path' field to the validation rule configuration",
            )
        if not rule.pattern:
            return ValidationResult(
                rule=rule,
                passed=False,
                error_message="content_contains rule requires 'pattern' field",
                failure_reason="Validation rule is missing required 'pattern' field",
                failure_category="error",
                suggested_fix="Add 'pattern' field to the validation rule configuration",
            )

        path = self.expand_path(rule.path)
        display_path = path.name if len(str(path)) > 50 else str(path)

        if not path.exists():
            return ValidationResult(
                rule=rule,
                passed=False,
                actual_value=None,
                expected_value=rule.pattern,
                error_message=f"File not found: {path}",
                failure_reason=f"File '{display_path}' does not exist (cannot check content)",
                failure_category="missing",
                suggested_fix=f"Create file '{display_path}' containing '{rule.pattern}'",
            )

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            warnings.warn(
                f"File has encoding issues, using replacement chars: {path}",
                UnicodeWarning,
                stacklevel=2,
            )
            content = path.read_text(encoding="utf-8", errors="replace")

        contains = rule.pattern in content

        if contains:
            return ValidationResult(
                rule=rule,
                passed=True,
                actual_value=f"contains={contains}",
                expected_value=rule.pattern,
            )

        # Truncate pattern for display if too long
        display_pattern = (
            rule.pattern[:50] + "..." if len(rule.pattern) > 50 else rule.pattern
        )

        return ValidationResult(
            rule=rule,
            passed=False,
            actual_value=f"contains={contains}",
            expected_value=rule.pattern,
            error_message=f"Pattern not found in {path}: {rule.pattern}",
            failure_reason=f"File '{display_path}' missing expected content: '{display_pattern}'",
            failure_category="incomplete",
            suggested_fix=f"Add exactly '{rule.pattern}' to the file (this exact text is validated)",
        )

    def _check_content_regex(self, rule: ValidationRule) -> ValidationResult:
        """Check if file content matches regex pattern.

        Args:
            rule: Validation rule with path and regex pattern.

        Returns:
            ValidationResult indicating if regex matched.
        """
        if not rule.path:
            return ValidationResult(
                rule=rule,
                passed=False,
                error_message="content_regex rule requires 'path' field",
                failure_reason="Validation rule is missing required 'path' field",
                failure_category="error",
                suggested_fix="Add 'path' field to the validation rule configuration",
            )
        if not rule.pattern:
            return ValidationResult(
                rule=rule,
                passed=False,
                error_message="content_regex rule requires 'pattern' field",
                failure_reason="Validation rule is missing required 'pattern' field",
                failure_category="error",
                suggested_fix="Add 'pattern' field to the validation rule configuration",
            )

        path = self.expand_path(rule.path)
        display_path = path.name if len(str(path)) > 50 else str(path)

        if not path.exists():
            return ValidationResult(
                rule=rule,
                passed=False,
                actual_value=None,
                expected_value=rule.pattern,
                error_message=f"File not found: {path}",
                failure_reason=f"File '{display_path}' does not exist (cannot check content)",
                failure_category="missing",
                suggested_fix="Create the file with content matching the pattern",
            )

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            warnings.warn(
                f"File has encoding issues, using replacement chars: {path}",
                UnicodeWarning,
                stacklevel=2,
            )
            content = path.read_text(encoding="utf-8", errors="replace")

        try:
            match = re.search(rule.pattern, content)
        except re.error as e:
            return ValidationResult(
                rule=rule,
                passed=False,
                error_message=f"Invalid regex pattern: {e}",
                failure_reason=f"Regex pattern is invalid: {e}",
                failure_category="error",
                suggested_fix="Fix the regex pattern syntax in the validation rule",
            )

        if match:
            return ValidationResult(
                rule=rule,
                passed=True,
                actual_value=match.group(0),
                expected_value=rule.pattern,
            )

        # Truncate pattern for display if too long
        display_pattern = (
            rule.pattern[:50] + "..." if len(rule.pattern) > 50 else rule.pattern
        )

        return ValidationResult(
            rule=rule,
            passed=False,
            actual_value=None,
            expected_value=rule.pattern,
            error_message=f"Regex not matched in {path}: {rule.pattern}",
            failure_reason=(
                f"File '{display_path}' doesn't match pattern: {display_pattern}"
            ),
            failure_category="malformed",
            suggested_fix="Check the file format matches expectations",
        )

    def _check_command_succeeds(self, rule: ValidationRule) -> ValidationResult:
        """Check if a shell command succeeds (exit code 0).

        This is useful for running quality control tools like:
        - pytest: `pytest tests/ -q`
        - mypy: `mypy src/`
        - ruff: `ruff check src/`

        Security Note:
            Commands are executed via ``["/bin/sh", "-c", command]`` (explicit
            shell invocation, not ``shell=True``). This enables shell features
            like pipes and redirects while being deterministic about which
            shell is used.

            **Trust Model:**
            Commands come from job configuration files (YAML), which are
            treated as trusted code — authored by the job creator who has
            full control over the execution environment. This is equivalent
            to a Makefile or CI pipeline definition: the config author IS
            the execution authority.

            **CI/CD and Multi-User Warning:**
            In CI/CD environments or shared repositories, validation commands
            run with the orchestrator's privileges. Treat job YAML files with
            the same security scrutiny as ``Makefile``, ``.github/workflows/``,
            or ``Jenkinsfile`` — they are executable code. Code review for
            command_succeeds rules should verify that commands don't:
            - Exfiltrate secrets (``curl`` with env vars)
            - Modify system state outside the workspace
            - Escalate privileges (``sudo``, ``chmod 777``)

            **Mitigations in place:**
            1. Config files are authored locally, not from untrusted input
            2. Working directory is constrained to the job workspace
            3. Commands have a 5-minute timeout to prevent resource exhaustion
            4. Context values are shell-quoted via ``shlex.quote()``
            5. Commands with high-risk patterns are logged at warning level

            **When NOT to use command_succeeds:**
            - Never interpolate untrusted data into commands
            - Never allow users to provide arbitrary commands via UI/API
            - If you need to run commands with variable data, use
              ValidationRule.working_directory for path customization

        Args:
            rule: Validation rule with command to execute.

        Returns:
            ValidationResult indicating if command succeeded.
        """
        if not rule.command:
            return ValidationResult(
                rule=rule,
                passed=False,
                error_message="command_succeeds rule requires 'command' field",
                failure_reason="Validation rule is missing required 'command' field",
                failure_category="error",
                suggested_fix="Add 'command' field to the validation rule configuration",
            )

        # Determine working directory
        cwd = (
            self.expand_path(rule.working_directory)
            if rule.working_directory
            else self.workspace
        )

        # Expand {workspace}, {sheet_num}, etc. in command strings,
        # matching the same variable expansion used in expand_path().
        # Use str.replace() instead of str.format() to avoid conflicts
        # with shell variable syntax like ${VAR} and ${VAR:-default}.
        # Values are shell-quoted to prevent injection via context values.
        #
        # Trust model: Mozart {placeholder} values are shell-quoted (safe).
        # Shell ${VAR} syntax is intentionally preserved and expanded by
        # /bin/sh — the config file author is the trust boundary for shell
        # variable usage. This is by design: validation commands may
        # legitimately reference environment variables like ${PATH}.
        context = dict(self.sheet_context)
        context["workspace"] = str(self.workspace)
        expanded_command = rule.command
        for key, value in context.items():
            expanded_command = expanded_command.replace(
                "{" + key + "}", shlex.quote(str(value))
            )

        # Get a display-friendly command summary
        display_command = (
            expanded_command[:50] + "..."
            if len(expanded_command) > 50
            else expanded_command
        )

        # Mitigation: log warning for commands with high-risk patterns.
        # This doesn't block execution (the config author is the trust boundary)
        # but creates an audit trail for security review.
        _HIGH_RISK_PATTERNS = (
            "sudo ", "chmod 777", "rm -rf /", "curl ", "wget ",
            "eval ", "> /etc/", "| sh", "| bash",
        )
        cmd_lower = expanded_command.lower()
        for pattern in _HIGH_RISK_PATTERNS:
            if pattern in cmd_lower:
                _logger.warning(
                    "Validation command contains high-risk pattern '%s': %s",
                    pattern.strip(),
                    display_command,
                )
                break

        try:
            # Use explicit ["/bin/sh", "-c", command] instead of shell=True
            # to prevent argument injection. shell=True passes to the system
            # shell which may vary; explicit invocation is deterministic.
            result = subprocess.run(
                ["/bin/sh", "-c", expanded_command],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=VALIDATION_COMMAND_TIMEOUT_SECONDS,
            )

            success = result.returncode == 0

            # Build output summary (truncate if very long)
            output = result.stdout + result.stderr
            if len(output) > VALIDATION_OUTPUT_TRUNCATE_CHARS:
                output_summary = output[:VALIDATION_OUTPUT_TRUNCATE_CHARS] + f"\n... ({len(output)} chars total)"
            else:
                output_summary = output

            if success:
                return ValidationResult(
                    rule=rule,
                    passed=True,
                    actual_value=f"exit_code={result.returncode}",
                    expected_value="exit_code=0",
                    confidence=1.0,
                    confidence_factors={"exit_code": 1.0},
                )

            # Extract first line of error for concise reason
            first_error_line = output.strip().split("\n")[0] if output.strip() else ""
            if len(first_error_line) > 80:
                first_error_line = first_error_line[:80] + "..."

            return ValidationResult(
                rule=rule,
                passed=False,
                actual_value=f"exit_code={result.returncode}",
                expected_value="exit_code=0",
                error_message=f"Command failed: {output_summary}",
                failure_reason=(
                    f"Command failed (exit {result.returncode}): {first_error_line}"
                ),
                failure_category="error",
                suggested_fix="Review command output for error details",
                confidence=0.8,  # Slightly lower confidence on failure
                confidence_factors={"exit_code": 0.5},
            )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                rule=rule,
                passed=False,
                expected_value="exit_code=0",
                error_message=f"Command timed out after {VALIDATION_COMMAND_TIMEOUT_SECONDS} seconds",
                failure_reason=f"Command '{display_command}' timed out after {VALIDATION_COMMAND_TIMEOUT_SECONDS} seconds",
                failure_category="error",
                suggested_fix="Increase timeout or optimize the command",
            )
        except Exception as e:
            return ValidationResult(
                rule=rule,
                passed=False,
                expected_value="exit_code=0",
                error_message=f"Command execution error: {e}",
                failure_reason=f"Command execution failed: {e}",
                failure_category="error",
                suggested_fix="Check command syntax and permissions",
            )


# =============================================================================
# Failure History Store (Evolution v6: History-Aware Prompt Generation)
# =============================================================================


@dataclass
class HistoricalFailure:
    """A single historical validation failure for prompt injection.

    Captures the essential information from a past validation failure
    to help Claude avoid repeating the same mistakes.

    Attributes:
        sheet_num: Sheet number where the failure occurred.
        rule_type: Type of validation rule (file_exists, content_contains, etc.).
        description: Human-readable description of the validation.
        failure_reason: Why the validation failed.
        failure_category: Category of failure (missing, malformed, etc.).
        suggested_fix: Hint for how to fix the issue.
    """

    sheet_num: int
    rule_type: str
    description: str
    failure_reason: str | None = None
    failure_category: str | None = None
    suggested_fix: str | None = None


class FailureHistoryStore:
    """Queries past validation failures from checkpoint state.

    This class enables history-aware prompt generation by extracting
    validation failures from previous sheets and finding similar failures
    that might inform the current sheet's execution.

    The similarity matching uses rule-based heuristics:
    - Same validation rule type (e.g., both file_exists)
    - Same failure category (e.g., both "missing")
    - Recent failures are prioritized (more relevant to current context)

    Example:
        ```python
        store = FailureHistoryStore(state)
        failures = store.query_similar_failures(
            current_sheet=5,
            rule_types=["file_exists", "content_contains"],
            limit=3,
        )
        # Inject failures into prompt
        prompt = builder.build_sheet_prompt(
            context=ctx,
            failure_history=failures,
        )
        ```
    """

    def __init__(self, state: "CheckpointState") -> None:
        """Initialize failure history store.

        Args:
            state: Current checkpoint state with sheet validation details.
        """
        self._state = state

    def query_similar_failures(
        self,
        current_sheet: int,
        rule_types: list[str] | None = None,
        failure_categories: list[str] | None = None,
        limit: int = 3,
    ) -> list[HistoricalFailure]:
        """Query past validation failures similar to expected patterns.

        Searches completed and failed sheets for validation failures that
        match the specified criteria. Returns the most recent matching
        failures first.

        Args:
            current_sheet: Current sheet number (excludes this sheet from results).
            rule_types: Filter by validation rule types (e.g., ["file_exists"]).
                        If None, matches all rule types.
            failure_categories: Filter by failure categories (e.g., ["missing"]).
                               If None, matches all categories.
            limit: Maximum number of failures to return.

        Returns:
            List of HistoricalFailure objects, most recent first.
        """
        failures: list[HistoricalFailure] = []

        # Iterate through sheets in reverse order (most recent first)
        for sheet_num in sorted(self._state.sheets.keys(), reverse=True):
            if sheet_num >= current_sheet:
                # Skip current and future sheets
                continue

            # Use defensive .get() pattern for safety
            sheet = self._state.sheets.get(sheet_num)
            if not sheet or not sheet.validation_details:
                continue

            validation_details = sheet.validation_details

            # Extract failures from validation_details
            for detail in validation_details:
                # Skip passed validations
                if detail.get("passed", False):
                    continue

                rule_type = detail.get("rule_type", "")
                failure_category = detail.get("failure_category")

                # Apply filters
                if rule_types and rule_type not in rule_types:
                    continue
                if failure_categories and failure_category not in failure_categories:
                    continue

                failure = HistoricalFailure(
                    sheet_num=sheet_num,
                    rule_type=rule_type,
                    description=detail.get("description", ""),
                    failure_reason=detail.get("failure_reason"),
                    failure_category=failure_category,
                    suggested_fix=detail.get("suggested_fix"),
                )
                failures.append(failure)

                if len(failures) >= limit:
                    return failures

        return failures

    def query_recent_failures(
        self,
        current_sheet: int,
        lookback_sheets: int = 3,
        limit: int = 3,
    ) -> list[HistoricalFailure]:
        """Query recent validation failures from nearby sheets.

        A simpler query that just returns recent failures without
        type/category filtering. Useful for general history awareness.

        Args:
            current_sheet: Current sheet number.
            lookback_sheets: How many previous sheets to check.
            limit: Maximum number of failures to return.

        Returns:
            List of HistoricalFailure objects from recent sheets.
        """
        failures: list[HistoricalFailure] = []

        # Check sheets immediately before current
        for offset in range(1, lookback_sheets + 1):
            sheet_num = current_sheet - offset
            if sheet_num <= 0:
                break

            sheet = self._state.sheets.get(sheet_num)
            if not sheet or not sheet.validation_details:
                continue

            for detail in sheet.validation_details:
                if detail.get("passed", False):
                    continue

                failure = HistoricalFailure(
                    sheet_num=sheet_num,
                    rule_type=detail.get("rule_type", ""),
                    description=detail.get("description", ""),
                    failure_reason=detail.get("failure_reason"),
                    failure_category=detail.get("failure_category"),
                    suggested_fix=detail.get("suggested_fix"),
                )
                failures.append(failure)

                if len(failures) >= limit:
                    return failures

        return failures

    def has_failures(self, current_sheet: int) -> bool:
        """Check if there are any historical failures to query.

        Args:
            current_sheet: Current sheet number.

        Returns:
            True if there are failures from previous sheets.
        """
        for sheet_num in self._state.sheets:
            if sheet_num >= current_sheet:
                continue

            sheet = self._state.sheets[sheet_num]
            if not sheet.validation_details:
                continue

            for detail in sheet.validation_details:
                if not detail.get("passed", False):
                    return True

        return False


# =============================================================================
# Cross-Sheet Semantic Validation (v20 evolution)
# =============================================================================


@dataclass
class KeyVariable:
    """A key-value pair extracted from sheet output.

    Attributes:
        key: The variable name/identifier.
        value: The variable value (as string).
        source_line: The line where this variable was found.
        line_number: Line number in the output (1-indexed).
    """

    key: str
    value: str
    source_line: str = ""
    line_number: int = 0


@dataclass
class SemanticInconsistency:
    """Represents a semantic inconsistency between sheets.

    Attributes:
        key: The key that has inconsistent values.
        sheet_a: First sheet number in comparison.
        value_a: Value from sheet A.
        sheet_b: Second sheet number in comparison.
        value_b: Value from sheet B.
        severity: Severity of inconsistency (warning, error).
    """

    key: str
    sheet_a: int
    value_a: str
    sheet_b: int
    value_b: str
    severity: str = "warning"

    def format_message(self) -> str:
        """Format as human-readable message."""
        return (
            f"Key '{self.key}' has inconsistent values: "
            f"sheet {self.sheet_a}='{self.value_a}' vs "
            f"sheet {self.sheet_b}='{self.value_b}'"
        )


@dataclass
class SemanticConsistencyResult:
    """Result of cross-sheet semantic consistency check.

    Attributes:
        sheets_compared: Sheets that were compared.
        inconsistencies: List of detected inconsistencies.
        keys_checked: Total number of keys checked.
        checked_at: When the check was performed.
    """

    sheets_compared: list[int] = field(default_factory=list)
    inconsistencies: list[SemanticInconsistency] = field(default_factory=list)
    keys_checked: int = 0
    checked_at: datetime = field(default_factory=utc_now)

    @property
    def is_consistent(self) -> bool:
        """True if no inconsistencies were found."""
        return len(self.inconsistencies) == 0

    @property
    def error_count(self) -> int:
        """Count of error-severity inconsistencies."""
        return sum(1 for i in self.inconsistencies if i.severity == "error")

    @property
    def warning_count(self) -> int:
        """Count of warning-severity inconsistencies."""
        return sum(1 for i in self.inconsistencies if i.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "sheets_compared": self.sheets_compared,
            "inconsistencies": [
                {
                    "key": i.key,
                    "sheet_a": i.sheet_a,
                    "value_a": i.value_a,
                    "sheet_b": i.sheet_b,
                    "value_b": i.value_b,
                    "severity": i.severity,
                }
                for i in self.inconsistencies
            ],
            "keys_checked": self.keys_checked,
            "checked_at": self.checked_at.isoformat(),
            "is_consistent": self.is_consistent,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


class KeyVariableExtractor:
    """Extracts key-value pairs from sheet output content.

    Supports multiple formats:
    - KEY: VALUE (colon-separated)
    - KEY=VALUE (equals-separated)
    - KEY = VALUE (equals with spaces)

    Keys must be uppercase or snake_case identifiers.
    Values are trimmed of whitespace.

    Example:
        ```python
        extractor = KeyVariableExtractor()
        variables = extractor.extract("STATUS: complete\\nCOUNT=42")
        # Returns [KeyVariable(key="STATUS", value="complete", ...),
        #          KeyVariable(key="COUNT", value="42", ...)]
        ```
    """

    # Combined pattern for KEY: VALUE or KEY=VALUE (single pass over content)
    # Uses named groups to capture separator for debugging if needed
    _KEY_VALUE_PATTERN = re.compile(
        r"^([A-Z][A-Z0-9_]*)\s*(?::|=)\s*(.+)$",
        re.MULTILINE
    )

    def __init__(
        self,
        key_filter: list[str] | None = None,
        case_sensitive: bool = False,
    ) -> None:
        """Initialize extractor.

        Args:
            key_filter: If provided, only extract keys matching these names.
                       If None, extract all matching keys.
            case_sensitive: Whether key matching is case-sensitive.
        """
        self.key_filter = key_filter
        self.case_sensitive = case_sensitive

    def extract(self, content: str) -> list[KeyVariable]:
        """Extract key-value pairs from content.

        Args:
            content: Text content to search for key-value pairs.

        Returns:
            List of extracted KeyVariable objects.
        """
        if not content:
            return []

        variables: list[KeyVariable] = []
        seen_keys: set[str] = set()

        # Build line number mapping
        lines = content.split("\n")
        line_map: dict[str, int] = {}
        for i, line in enumerate(lines, 1):
            line_map[line] = i

        # Extract key-value pairs (single pass using combined pattern)
        for match in self._KEY_VALUE_PATTERN.finditer(content):
            key = match.group(1)
            value = match.group(2).strip()
            source_line = match.group(0)

            if self._should_include(key) and key not in seen_keys:
                variables.append(KeyVariable(
                    key=key,
                    value=value,
                    source_line=source_line,
                    line_number=line_map.get(source_line, 0),
                ))
                seen_keys.add(key)

        return variables

    def _should_include(self, key: str) -> bool:
        """Check if key should be included based on filter."""
        if self.key_filter is None:
            return True

        if self.case_sensitive:
            return key in self.key_filter
        else:
            key_lower = key.lower()
            return any(k.lower() == key_lower for k in self.key_filter)


class SemanticConsistencyChecker:
    """Checks semantic consistency between sequential sheet outputs.

    Compares key-value pairs extracted from sheet outputs to detect
    when the same key has different values across sheets.

    Example:
        ```python
        checker = SemanticConsistencyChecker()
        outputs = {
            1: "STATUS: running\\nVERSION: 1.0",
            2: "STATUS: complete\\nVERSION: 1.0",  # STATUS changed
        }
        result = checker.check_consistency(outputs)
        if not result.is_consistent:
            for inc in result.inconsistencies:
                print(inc.format_message())
        ```
    """

    def __init__(
        self,
        extractor: KeyVariableExtractor | None = None,
        strict_mode: bool = False,
    ) -> None:
        """Initialize checker.

        Args:
            extractor: Custom key variable extractor. Uses default if None.
            strict_mode: If True, all inconsistencies are errors.
                        If False, inconsistencies are warnings.
        """
        self.extractor = extractor or KeyVariableExtractor()
        self.strict_mode = strict_mode

    def check_consistency(
        self,
        sheet_outputs: dict[int, str],
        sequential_only: bool = True,
    ) -> SemanticConsistencyResult:
        """Check semantic consistency across sheet outputs.

        Args:
            sheet_outputs: Map of sheet_num -> output content.
            sequential_only: If True, only compare sequential sheets (N vs N+1).
                           If False, compare all sheet pairs.

        Returns:
            SemanticConsistencyResult with any inconsistencies found.
        """
        result = SemanticConsistencyResult(
            sheets_compared=sorted(sheet_outputs.keys()),
        )

        if len(sheet_outputs) < 2:
            return result

        # Extract variables from each sheet
        sheet_variables: dict[int, dict[str, KeyVariable]] = {}
        for sheet_num, content in sheet_outputs.items():
            variables = self.extractor.extract(content)
            sheet_variables[sheet_num] = {v.key: v for v in variables}

        # Count total unique keys
        all_keys: set[str] = set()
        for vars_dict in sheet_variables.values():
            all_keys.update(vars_dict.keys())
        result.keys_checked = len(all_keys)

        # Compare sheets
        sheets_sorted = sorted(sheet_outputs.keys())

        if sequential_only:
            # Compare sequential pairs only
            for i in range(len(sheets_sorted) - 1):
                sheet_a = sheets_sorted[i]
                sheet_b = sheets_sorted[i + 1]
                self._compare_sheets(
                    sheet_a, sheet_variables[sheet_a],
                    sheet_b, sheet_variables[sheet_b],
                    result,
                )
        else:
            # Hash-group approach: for each key, group sheets by value.
            # Only cross-group pairs (different values) are inconsistencies.
            # This is O(n*k + g^2*m) where k = keys, g = groups, m = group sizes,
            # which is much faster than O(n^2*k) pairwise when most sheets agree.
            from collections import defaultdict

            for key in all_keys:
                # Group sheets by their (lowered) value for this key
                value_groups: dict[str, list[int]] = defaultdict(list)
                for sheet_num in sheets_sorted:
                    var = sheet_variables[sheet_num].get(key)
                    if var is not None:
                        value_groups[var.value.lower()].append(sheet_num)

                # If all sheets have the same value (or only one has the key), skip
                if len(value_groups) <= 1:
                    continue

                # Report all cross-group pairs as inconsistencies
                group_list = list(value_groups.values())
                for gi in range(len(group_list)):
                    for gj in range(gi + 1, len(group_list)):
                        for sheet_a in group_list[gi]:
                            for sheet_b in group_list[gj]:
                                var_a = sheet_variables[sheet_a][key]
                                var_b = sheet_variables[sheet_b][key]
                                result.inconsistencies.append(SemanticInconsistency(
                                    key=key,
                                    sheet_a=sheet_a,
                                    value_a=var_a.value,
                                    sheet_b=sheet_b,
                                    value_b=var_b.value,
                                    severity="error" if self.strict_mode else "warning",
                                ))

        return result

    def _compare_sheets(
        self,
        sheet_a: int,
        vars_a: dict[str, KeyVariable],
        sheet_b: int,
        vars_b: dict[str, KeyVariable],
        result: SemanticConsistencyResult,
    ) -> None:
        """Compare variables between two sheets."""
        # Find common keys
        common_keys = set(vars_a.keys()) & set(vars_b.keys())

        for key in common_keys:
            var_a = vars_a[key]
            var_b = vars_b[key]

            # Compare values (case-insensitive for flexibility)
            if var_a.value.lower() != var_b.value.lower():
                result.inconsistencies.append(SemanticInconsistency(
                    key=key,
                    sheet_a=sheet_a,
                    value_a=var_a.value,
                    sheet_b=sheet_b,
                    value_b=var_b.value,
                    severity="error" if self.strict_mode else "warning",
                ))
