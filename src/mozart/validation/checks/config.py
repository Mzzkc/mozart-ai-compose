"""Configuration structure validation checks.

Validates configuration values like regex patterns, timeout ranges,
and validation rule completeness.
"""

import re
from pathlib import Path

from mozart.core.config import JobConfig
from mozart.validation.base import ValidationIssue, ValidationSeverity


class RegexPatternCheck:
    """Check that regex patterns in validations compile (V007).

    Invalid regex patterns will cause runtime errors during validation.
    """

    @property
    def check_id(self) -> str:
        return "V007"

    @property
    def severity(self) -> ValidationSeverity:
        return ValidationSeverity.ERROR

    @property
    def description(self) -> str:
        return "Validates regex patterns compile correctly"

    def check(
        self,
        config: JobConfig,
        config_path: Path,
        raw_yaml: str,
    ) -> list[ValidationIssue]:
        """Check regex patterns in validations and rate limit detection."""
        issues: list[ValidationIssue] = []

        # Check validation rule patterns
        for i, validation in enumerate(config.validations):
            if validation.type == "content_regex" and validation.pattern:
                issue = self._check_pattern(
                    validation.pattern,
                    f"validation[{i}].pattern",
                    self._find_pattern_line(raw_yaml, validation.pattern),
                )
                if issue:
                    issues.append(issue)

        # Check rate limit detection patterns
        for i, pattern in enumerate(config.rate_limit.detection_patterns):
            issue = self._check_pattern(
                pattern,
                f"rate_limit.detection_patterns[{i}]",
                None,
            )
            if issue:
                issues.append(issue)

        return issues

    def _check_pattern(
        self,
        pattern: str,
        location: str,
        line: int | None,
    ) -> ValidationIssue | None:
        """Check if a single regex pattern compiles."""
        try:
            re.compile(pattern)
            return None
        except re.error as e:
            # Extract position from error if available
            pos_info = ""
            if hasattr(e, "pos") and e.pos is not None:
                pos_info = f" at position {e.pos}"

            return ValidationIssue(
                check_id=self.check_id,
                severity=self.severity,
                message=f"Invalid regex pattern in {location}: {e.msg}{pos_info}",
                line=line,
                context=pattern[:60] + "..." if len(pattern) > 60 else pattern,
                suggestion=self._suggest_regex_fix(str(e.msg), pattern),
                metadata={
                    "pattern": pattern,
                    "location": location,
                    "error": str(e),
                },
            )

    def _suggest_regex_fix(self, error_msg: str, pattern: str) -> str:
        """Suggest fixes for common regex errors."""
        error_lower = error_msg.lower()

        if "nothing to repeat" in error_lower:
            return "Escape special characters like *, +, ? with backslash: \\* \\+ \\?"

        if "unbalanced parenthesis" in error_lower or "missing )" in error_lower:
            return "Check parentheses are balanced or escape them: \\( \\)"

        if "unterminated character class" in error_lower:
            return "Close the character class with ] or escape the opening [: \\["

        if "bad escape" in error_lower:
            return "Invalid escape sequence - use raw string r'pattern' or double backslash"

        # Check for common mistake: unescaped special chars
        special_chars = [".", "*", "+", "?", "[", "]", "(", ")", "{", "}", "^", "$", "|"]
        for char in special_chars:
            if char in pattern and f"\\{char}" not in pattern:
                return f"Consider escaping '{char}' with backslash if literal match intended"

        return "Review regex syntax - see Python re module documentation"

    def _find_pattern_line(self, yaml_str: str, pattern: str) -> int | None:
        """Find the line number of a pattern in the YAML."""
        # Escape regex special chars for searching
        search_pattern = re.escape(pattern[:30])
        for i, line in enumerate(yaml_str.split("\n"), 1):
            if search_pattern[:20] in line:
                return i
        return None


class ValidationTypeCheck:
    """Check that validation rules have required fields (V008).

    Ensures validations have the fields needed for their type.
    """

    @property
    def check_id(self) -> str:
        return "V008"

    @property
    def severity(self) -> ValidationSeverity:
        return ValidationSeverity.ERROR

    @property
    def description(self) -> str:
        return "Checks validation rules have required fields"

    def check(
        self,
        config: JobConfig,
        config_path: Path,
        raw_yaml: str,
    ) -> list[ValidationIssue]:
        """Check validation rules have required fields."""
        issues: list[ValidationIssue] = []

        required_fields = {
            "file_exists": ["path"],
            "file_modified": ["path"],
            "content_contains": ["path", "pattern"],
            "content_regex": ["path", "pattern"],
            "command_succeeds": ["command"],
        }

        for i, validation in enumerate(config.validations):
            required = required_fields.get(validation.type, [])
            missing = []

            for field in required:
                value = getattr(validation, field, None)
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing.append(field)

            if missing:
                issues.append(
                    ValidationIssue(
                        check_id=self.check_id,
                        severity=self.severity,
                        message=f"Validation rule {i + 1} ({validation.type}) missing required fields: {', '.join(missing)}",
                        suggestion=f"Add {', '.join(missing)} to the validation rule",
                        metadata={
                            "validation_index": str(i),
                            "validation_type": validation.type,
                            "missing_fields": ",".join(missing),
                        },
                    )
                )

        return issues


class TimeoutRangeCheck:
    """Check timeout values are reasonable (V103/V104).

    Warns about very short timeouts (may cause failures) or
    very long timeouts (may waste resources).
    """

    # Thresholds in seconds
    MIN_REASONABLE_TIMEOUT = 60  # 1 minute
    MAX_REASONABLE_TIMEOUT = 7200  # 2 hours

    @property
    def check_id(self) -> str:
        return "V103"

    @property
    def severity(self) -> ValidationSeverity:
        return ValidationSeverity.WARNING

    @property
    def description(self) -> str:
        return "Checks timeout values are reasonable"

    def check(
        self,
        config: JobConfig,
        config_path: Path,
        raw_yaml: str,
    ) -> list[ValidationIssue]:
        """Check timeout values."""
        issues: list[ValidationIssue] = []

        timeout = config.backend.timeout_seconds

        if timeout < self.MIN_REASONABLE_TIMEOUT:
            issues.append(
                ValidationIssue(
                    check_id="V103",
                    severity=ValidationSeverity.WARNING,
                    message=f"Very short timeout ({timeout}s) may cause premature failures",
                    line=self._find_line_in_yaml(raw_yaml, "timeout_seconds:"),
                    suggestion=f"Consider timeout_seconds >= {self.MIN_REASONABLE_TIMEOUT} for Claude CLI operations",
                    metadata={
                        "timeout": str(timeout),
                        "threshold": str(self.MIN_REASONABLE_TIMEOUT),
                    },
                )
            )

        if timeout > self.MAX_REASONABLE_TIMEOUT:
            issues.append(
                ValidationIssue(
                    check_id="V104",
                    severity=ValidationSeverity.INFO,
                    message=f"Very long timeout ({timeout}s = {timeout / 3600:.1f}h) - consider if this is necessary",
                    line=self._find_line_in_yaml(raw_yaml, "timeout_seconds:"),
                    suggestion="Long timeouts can tie up resources; consider breaking into smaller tasks",
                    metadata={
                        "timeout": str(timeout),
                        "threshold": str(self.MAX_REASONABLE_TIMEOUT),
                    },
                )
            )

        return issues

    def _find_line_in_yaml(self, yaml_str: str, marker: str) -> int | None:
        """Find the line number of a marker in the YAML."""
        for i, line in enumerate(yaml_str.split("\n"), 1):
            if marker in line:
                return i
        return None


class EmptyPatternCheck:
    """Check for empty patterns in validations (V106)."""

    @property
    def check_id(self) -> str:
        return "V106"

    @property
    def severity(self) -> ValidationSeverity:
        return ValidationSeverity.WARNING

    @property
    def description(self) -> str:
        return "Checks for empty patterns in content validations"

    def check(
        self,
        config: JobConfig,
        config_path: Path,
        raw_yaml: str,
    ) -> list[ValidationIssue]:
        """Check for empty patterns."""
        issues: list[ValidationIssue] = []

        for i, validation in enumerate(config.validations):
            if validation.type in ("content_contains", "content_regex"):
                if validation.pattern is not None and validation.pattern.strip() == "":
                    issues.append(
                        ValidationIssue(
                            check_id=self.check_id,
                            severity=self.severity,
                            message=f"Empty pattern in validation rule {i + 1} will match any content",
                            suggestion="Add a meaningful pattern or remove this validation",
                            metadata={
                                "validation_index": str(i),
                            },
                        )
                    )

        return issues
