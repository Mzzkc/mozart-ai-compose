"""Enhanced validation module for Mozart job configurations.

This module provides comprehensive validation beyond Pydantic schema validation,
including Jinja template syntax checking, path existence validation, regex
pattern compilation, and cross-reference checks.

The validation system is designed for two use cases:
1. Pre-execution validation via `mozart validate` command
2. Self-healing diagnosis to identify fixable configuration issues

Example usage:
    from mozart.validation import ValidationRunner, create_default_checks

    runner = ValidationRunner(create_default_checks())
    issues = runner.validate(config, config_path, raw_yaml)

    for issue in issues:
        print(f"[{issue.severity}] {issue.check_id}: {issue.message}")
"""

from mozart.validation.base import (
    ValidationCheck,
    ValidationIssue,
    ValidationSeverity,
)
from mozart.validation.reporter import ValidationReporter
from mozart.validation.runner import ValidationRunner, create_default_checks

__all__ = [
    "ValidationCheck",
    "ValidationIssue",
    "ValidationSeverity",
    "ValidationRunner",
    "ValidationReporter",
    "create_default_checks",
]
