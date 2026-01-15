"""Base types and protocols for the validation system.

Defines the core abstractions:
- ValidationSeverity: Error/Warning/Info classification
- ValidationIssue: A single issue found during validation
- ValidationCheck: Protocol for implementing validation checks
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from mozart.core.config import JobConfig


class ValidationSeverity(str, Enum):
    """Severity level for validation issues.

    - ERROR: Must fix before execution - job will fail
    - WARNING: Should fix - may cause runtime failures
    - INFO: Informational - consider reviewing
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue found in the configuration.

    Attributes:
        check_id: Unique identifier for the check (e.g., V001, V101)
        severity: ERROR, WARNING, or INFO
        message: Human-readable description of the issue
        line: Line number in config file (if applicable)
        column: Column number (if applicable)
        context: Surrounding text for context
        suggestion: How to fix the issue
        auto_fixable: Whether --self-healing can automatically fix this
    """

    check_id: str
    severity: ValidationSeverity
    message: str
    line: int | None = None
    column: int | None = None
    context: str | None = None
    suggestion: str | None = None
    auto_fixable: bool = False
    # Additional metadata for healing system
    metadata: dict[str, str] = field(default_factory=dict)

    def format_short(self) -> str:
        """Format as a single-line summary."""
        loc = f"Line {self.line}: " if self.line else ""
        return f"[{self.check_id}] {loc}{self.message}"

    def format_full(self) -> str:
        """Format with full details including suggestion."""
        lines = [self.format_short()]
        if self.context:
            lines.append(f"         Context: {self.context}")
        if self.suggestion:
            lines.append(f"         Suggestion: {self.suggestion}")
        return "\n".join(lines)


class ValidationCheck(Protocol):
    """Protocol for configuration validation checks.

    Each check examines a specific aspect of the configuration and
    returns a list of ValidationIssue objects (empty if check passes).

    Checks should be:
    - Idempotent (safe to run multiple times)
    - Side-effect free (don't modify anything)
    - Fast (validation should complete quickly)
    """

    @property
    def check_id(self) -> str:
        """Unique identifier for this check (e.g., V001)."""
        ...

    @property
    def severity(self) -> ValidationSeverity:
        """Default severity for issues from this check."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this check does."""
        ...

    def check(
        self,
        config: "JobConfig",
        config_path: Path,
        raw_yaml: str,
    ) -> list[ValidationIssue]:
        """Run the validation check.

        Args:
            config: Parsed JobConfig object (already validated by Pydantic)
            config_path: Path to the config file (for resolving relative paths)
            raw_yaml: Raw YAML text (for line number extraction)

        Returns:
            List of ValidationIssue objects (empty if check passes)
        """
        ...
