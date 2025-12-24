"""Error classification and handling.

Categorizes errors to determine appropriate retry behavior.
"""

import re
from dataclasses import dataclass
from enum import Enum


class ErrorCategory(str, Enum):
    """Categories of errors with different retry behaviors."""

    RATE_LIMIT = "rate_limit"
    """Retriable with long wait - API/service is rate limiting."""

    TRANSIENT = "transient"
    """Retriable with backoff - temporary network/service issues."""

    VALIDATION = "validation"
    """Retriable - Claude ran but didn't produce expected output."""

    AUTH = "auth"
    """Fatal - authentication/authorization failure, needs user intervention."""

    NETWORK = "network"
    """Retriable with backoff - network connectivity issues."""

    TIMEOUT = "timeout"
    """Retriable - operation timed out."""

    FATAL = "fatal"
    """Non-retriable - stop job immediately."""


@dataclass
class ClassifiedError:
    """An error with its classification and metadata."""

    category: ErrorCategory
    message: str
    original_error: Exception | None = None
    exit_code: int | None = None
    retriable: bool = True
    suggested_wait_seconds: float | None = None

    @property
    def is_rate_limit(self) -> bool:
        return self.category == ErrorCategory.RATE_LIMIT

    @property
    def should_retry(self) -> bool:
        return self.retriable and self.category not in (ErrorCategory.AUTH, ErrorCategory.FATAL)


class ErrorClassifier:
    """Classifies errors based on patterns and exit codes.

    Pattern matching follows the approach from run-batch-review.sh
    which checks output for rate limit indicators.
    """

    def __init__(
        self,
        rate_limit_patterns: list[str] | None = None,
        auth_patterns: list[str] | None = None,
        network_patterns: list[str] | None = None,
    ):
        """Initialize classifier with detection patterns.

        Args:
            rate_limit_patterns: Regex patterns indicating rate limiting
            auth_patterns: Regex patterns indicating auth failures
            network_patterns: Regex patterns indicating network issues
        """
        self.rate_limit_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (rate_limit_patterns or [
                r"rate.?limit",
                r"usage.?limit",
                r"quota",
                r"too many requests",
                r"429",
                r"capacity",
                r"try again later",
                r"overloaded",
            ])
        ]

        self.auth_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (auth_patterns or [
                r"unauthorized",
                r"authentication",
                r"invalid.?api.?key",
                r"permission.?denied",
                r"access.?denied",
                r"401",
                r"403",
            ])
        ]

        self.network_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (network_patterns or [
                r"connection.?refused",
                r"connection.?reset",
                r"connection.?timeout",
                r"network.?unreachable",
                r"dns.?resolution",
                r"ssl.?error",
                r"ECONNREFUSED",
                r"ETIMEDOUT",
            ])
        ]

    def classify(
        self,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        exception: Exception | None = None,
    ) -> ClassifiedError:
        """Classify an error based on output and exit code.

        Args:
            stdout: Standard output from the command
            stderr: Standard error from the command
            exit_code: Process exit code (0 = success)
            exception: Optional exception that was raised

        Returns:
            ClassifiedError with category and metadata
        """
        combined = f"{stdout}\n{stderr}"
        if exception:
            combined += f"\n{str(exception)}"

        # Check for rate limiting first (most common retriable)
        if self._matches_any(combined, self.rate_limit_patterns):
            return ClassifiedError(
                category=ErrorCategory.RATE_LIMIT,
                message="Rate limit detected",
                original_error=exception,
                exit_code=exit_code,
                retriable=True,
                suggested_wait_seconds=3600.0,  # 1 hour default
            )

        # Check for auth failures (fatal)
        if self._matches_any(combined, self.auth_patterns):
            return ClassifiedError(
                category=ErrorCategory.AUTH,
                message="Authentication or authorization failure",
                original_error=exception,
                exit_code=exit_code,
                retriable=False,
            )

        # Check for network issues (retriable with backoff)
        if self._matches_any(combined, self.network_patterns):
            return ClassifiedError(
                category=ErrorCategory.NETWORK,
                message="Network connectivity issue",
                original_error=exception,
                exit_code=exit_code,
                retriable=True,
                suggested_wait_seconds=30.0,
            )

        # Check exit code
        if exit_code == 0:
            # Command succeeded but might have validation issues
            return ClassifiedError(
                category=ErrorCategory.VALIDATION,
                message="Command succeeded but output validation needed",
                exit_code=exit_code,
                retriable=True,
            )

        if exit_code is not None:
            # Non-zero exit codes
            if exit_code in (1, 2):
                # Common transient errors
                return ClassifiedError(
                    category=ErrorCategory.TRANSIENT,
                    message=f"Command failed with exit code {exit_code}",
                    original_error=exception,
                    exit_code=exit_code,
                    retriable=True,
                    suggested_wait_seconds=10.0,
                )

            if exit_code == 124:
                # Timeout (from `timeout` command)
                return ClassifiedError(
                    category=ErrorCategory.TIMEOUT,
                    message="Command timed out",
                    exit_code=exit_code,
                    retriable=True,
                    suggested_wait_seconds=60.0,
                )

        # Default to fatal for unknown errors
        return ClassifiedError(
            category=ErrorCategory.FATAL,
            message=f"Unknown error (exit_code={exit_code})",
            original_error=exception,
            exit_code=exit_code,
            retriable=False,
        )

    def _matches_any(self, text: str, patterns: list[re.Pattern[str]]) -> bool:
        """Check if text matches any of the patterns."""
        return any(p.search(text) for p in patterns)

    @classmethod
    def from_config(cls, rate_limit_patterns: list[str]) -> "ErrorClassifier":
        """Create classifier from config patterns."""
        return cls(rate_limit_patterns=rate_limit_patterns)
