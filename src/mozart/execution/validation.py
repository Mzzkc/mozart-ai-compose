"""Validation framework for sheet outputs.

Executes validation rules against sheet outputs and tracks results
for partial completion recovery.
"""

import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mozart.core.config import ValidationRule


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    rule: ValidationRule
    passed: bool
    actual_value: str | None = None
    expected_value: str | None = None
    error_message: str | None = None
    checked_at: datetime = field(default_factory=_utc_now)
    check_duration_ms: float = 0.0
    # Learning metadata (Phase 1: Learning Foundation)
    confidence: float = 1.0
    """Confidence in this validation result (0.0-1.0). Default 1.0 = fully confident."""
    confidence_factors: dict[str, float] = field(default_factory=dict)
    """Factors affecting confidence, e.g., {'file_age': 0.9, 'pattern_specificity': 0.8}."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
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
        }


@dataclass
class SheetValidationResult:
    """Aggregate result of all validations for a sheet."""

    sheet_num: int
    results: list[ValidationResult]

    @property
    def all_passed(self) -> bool:
        """Check if all validations passed."""
        return all(r.passed for r in self.results)

    @property
    def passed_count(self) -> int:
        """Count of passed validations."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        """Count of failed validations."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_percentage(self) -> float:
        """Percentage of validations that passed."""
        if not self.results:
            return 100.0  # No validations = all passed
        return (self.passed_count / len(self.results)) * 100

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

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Convert all results to serializable list."""
        return [r.to_dict() for r in self.results]


class FileModificationTracker:
    """Tracks file mtimes before sheet execution for file_modified checks.

    The file_modified validation checks if a file was updated during batch
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
        path_str = str(path.resolve())
        if not path.exists():
            return False  # File doesn't exist, can't be modified

        current_mtime = path.stat().st_mtime
        original_mtime = self._mtimes.get(path_str, 0.0)
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
        self.workspace = workspace
        self.sheet_context = sheet_context
        self._mtime_tracker = FileModificationTracker()

    def expand_path(self, path_template: str) -> Path:
        """Expand path template with batch context variables.

        Supports: {sheet_num}, {workspace}, {start_item}, {end_item}

        Args:
            path_template: Path with {variable} placeholders.

        Returns:
            Expanded Path object.
        """
        # Build context, ensuring workspace is set correctly
        context = dict(self.sheet_context)
        context["workspace"] = str(self.workspace)

        expanded = path_template.format(**context)
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

    def run_validations(self, rules: list[ValidationRule]) -> SheetValidationResult:
        """Execute all validation rules and return aggregate result.

        Args:
            rules: List of validation rules to execute.

        Returns:
            SheetValidationResult with all individual results.
        """
        results: list[ValidationResult] = []

        for rule in rules:
            result = self._run_single_validation(rule)
            results.append(result)

        return SheetValidationResult(
            sheet_num=self.sheet_context.get("sheet_num", 0),
            results=results,
        )

    def _run_single_validation(self, rule: ValidationRule) -> ValidationResult:
        """Execute a single validation rule.

        Args:
            rule: The validation rule to execute.

        Returns:
            ValidationResult with pass/fail status and details.
        """
        start = time.monotonic()

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

            result.check_duration_ms = (time.monotonic() - start) * 1000
            return result

        except Exception as e:
            return ValidationResult(
                rule=rule,
                passed=False,
                expected_value=rule.path or rule.pattern,
                error_message=f"Validation error: {e}",
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
            )

        path = self.expand_path(rule.path)
        exists = path.exists() and path.is_file()

        return ValidationResult(
            rule=rule,
            passed=exists,
            actual_value=str(path) if exists else None,
            expected_value=str(path),
            error_message=None if exists else f"File not found: {path}",
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
            )

        path = self.expand_path(rule.path)

        if not path.exists():
            return ValidationResult(
                rule=rule,
                passed=False,
                actual_value=None,
                expected_value=str(path),
                error_message=f"File does not exist: {path}",
            )

        was_modified = self._mtime_tracker.was_modified(path)
        original_mtime = self._mtime_tracker.get_original_mtime(path)

        return ValidationResult(
            rule=rule,
            passed=was_modified,
            actual_value=f"mtime={path.stat().st_mtime:.6f}",
            expected_value=f"mtime>{original_mtime:.6f}" if original_mtime else "modified",
            error_message=None if was_modified else f"File not modified: {path}",
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
            )
        if not rule.pattern:
            return ValidationResult(
                rule=rule,
                passed=False,
                error_message="content_contains rule requires 'pattern' field",
            )

        path = self.expand_path(rule.path)

        if not path.exists():
            return ValidationResult(
                rule=rule,
                passed=False,
                actual_value=None,
                expected_value=rule.pattern,
                error_message=f"File not found: {path}",
            )

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="replace")

        contains = rule.pattern in content

        return ValidationResult(
            rule=rule,
            passed=contains,
            actual_value=f"contains={contains}",
            expected_value=rule.pattern,
            error_message=None if contains else f"Pattern not found in {path}: {rule.pattern}",
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
            )
        if not rule.pattern:
            return ValidationResult(
                rule=rule,
                passed=False,
                error_message="content_regex rule requires 'pattern' field",
            )

        path = self.expand_path(rule.path)

        if not path.exists():
            return ValidationResult(
                rule=rule,
                passed=False,
                actual_value=None,
                expected_value=rule.pattern,
                error_message=f"File not found: {path}",
            )

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="replace")

        try:
            match = re.search(rule.pattern, content)
        except re.error as e:
            return ValidationResult(
                rule=rule,
                passed=False,
                error_message=f"Invalid regex pattern: {e}",
            )

        return ValidationResult(
            rule=rule,
            passed=match is not None,
            actual_value=match.group(0) if match else None,
            expected_value=rule.pattern,
            error_message=None if match else f"Regex not matched in {path}: {rule.pattern}",
        )

    def _check_command_succeeds(self, rule: ValidationRule) -> ValidationResult:
        """Check if a shell command succeeds (exit code 0).

        This is useful for running quality control tools like:
        - pytest: `pytest tests/ -q`
        - mypy: `mypy src/`
        - ruff: `ruff check src/`

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
            )

        # Determine working directory
        cwd = (
            self.expand_path(rule.working_directory)
            if rule.working_directory
            else self.workspace
        )

        try:
            result = subprocess.run(
                rule.command,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            success = result.returncode == 0

            # Build output summary (truncate if very long)
            output = result.stdout + result.stderr
            if len(output) > 500:
                output_summary = output[:500] + f"\n... ({len(output)} chars total)"
            else:
                output_summary = output

            return ValidationResult(
                rule=rule,
                passed=success,
                actual_value=f"exit_code={result.returncode}",
                expected_value="exit_code=0",
                error_message=None if success else f"Command failed: {output_summary}",
                confidence=1.0 if success else 0.8,  # Slightly lower confidence on failure
                confidence_factors={
                    "exit_code": 1.0 if success else 0.5,
                },
            )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                rule=rule,
                passed=False,
                expected_value="exit_code=0",
                error_message="Command timed out after 300 seconds",
            )
        except Exception as e:
            return ValidationResult(
                rule=rule,
                passed=False,
                expected_value="exit_code=0",
                error_message=f"Command execution error: {e}",
            )
