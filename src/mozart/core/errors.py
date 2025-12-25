"""Error classification and handling.

Categorizes errors to determine appropriate retry behavior.
"""

import re
import signal
from dataclasses import dataclass
from enum import Enum
from typing import Literal

# Type alias for exit reasons (matches backend)
ExitReason = Literal["completed", "timeout", "killed", "error"]


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

    SIGNAL = "signal"
    """Process killed by signal - may be retriable depending on signal."""

    FATAL = "fatal"
    """Non-retriable - stop job immediately."""


# Signals that indicate the process should be retried
RETRIABLE_SIGNALS: set[int] = {
    signal.SIGTERM,  # Graceful termination request
    signal.SIGHUP,   # Terminal hangup
    signal.SIGPIPE,  # Broken pipe (network issues)
}

# Signals that indicate a fatal error (crash, out of memory, etc.)
FATAL_SIGNALS: set[int] = {
    signal.SIGSEGV,  # Segmentation fault
    signal.SIGBUS,   # Bus error
    signal.SIGABRT,  # Abort signal
    signal.SIGFPE,   # Floating point exception
    signal.SIGILL,   # Illegal instruction
}


def get_signal_name(sig_num: int) -> str:
    """Get human-readable signal name."""
    signal_names: dict[int, str] = {
        signal.SIGTERM: "SIGTERM",
        signal.SIGKILL: "SIGKILL",
        signal.SIGINT: "SIGINT",
        signal.SIGSEGV: "SIGSEGV",
        signal.SIGABRT: "SIGABRT",
        signal.SIGBUS: "SIGBUS",
        signal.SIGFPE: "SIGFPE",
        signal.SIGHUP: "SIGHUP",
        signal.SIGPIPE: "SIGPIPE",
    }
    return signal_names.get(sig_num, f"signal {sig_num}")


@dataclass
class ClassifiedError:
    """An error with its classification and metadata."""

    category: ErrorCategory
    message: str
    original_error: Exception | None = None
    exit_code: int | None = None
    exit_signal: int | None = None
    exit_reason: ExitReason | None = None
    retriable: bool = True
    suggested_wait_seconds: float | None = None

    @property
    def is_rate_limit(self) -> bool:
        return self.category == ErrorCategory.RATE_LIMIT

    @property
    def is_signal_kill(self) -> bool:
        """True if process was killed by a signal."""
        return self.exit_signal is not None

    @property
    def signal_name(self) -> str | None:
        """Human-readable signal name if killed by signal."""
        if self.exit_signal is None:
            return None
        return get_signal_name(self.exit_signal)

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
        exit_signal: int | None = None,
        exit_reason: ExitReason | None = None,
        exception: Exception | None = None,
    ) -> ClassifiedError:
        """Classify an error based on output, exit code, and signal.

        Args:
            stdout: Standard output from the command
            stderr: Standard error from the command
            exit_code: Process exit code (0 = success), None if killed by signal
            exit_signal: Signal number if killed by signal
            exit_reason: Why execution ended (completed, timeout, killed, error)
            exception: Optional exception that was raised

        Returns:
            ClassifiedError with category and metadata
        """
        combined = f"{stdout}\n{stderr}"
        if exception:
            combined += f"\n{str(exception)}"

        # Handle signal-based exits first (new in Task 3)
        if exit_signal is not None:
            return self._classify_signal(
                exit_signal=exit_signal,
                exit_reason=exit_reason,
                exception=exception,
                stdout=stdout,
                stderr=stderr,
            )

        # Handle timeout exit reason (even without signal)
        if exit_reason == "timeout":
            return ClassifiedError(
                category=ErrorCategory.TIMEOUT,
                message="Command timed out",
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=True,
                suggested_wait_seconds=60.0,
            )

        # Check for rate limiting first (most common retriable)
        if self._matches_any(combined, self.rate_limit_patterns):
            return ClassifiedError(
                category=ErrorCategory.RATE_LIMIT,
                message="Rate limit detected",
                original_error=exception,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
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
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=False,
            )

        # Check for network issues (retriable with backoff)
        if self._matches_any(combined, self.network_patterns):
            return ClassifiedError(
                category=ErrorCategory.NETWORK,
                message="Network connectivity issue",
                original_error=exception,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
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
                exit_signal=None,
                exit_reason=exit_reason,
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
                    exit_signal=None,
                    exit_reason=exit_reason,
                    retriable=True,
                    suggested_wait_seconds=10.0,
                )

            if exit_code == 124:
                # Timeout (from `timeout` command - legacy support)
                return ClassifiedError(
                    category=ErrorCategory.TIMEOUT,
                    message="Command timed out",
                    exit_code=exit_code,
                    exit_signal=None,
                    exit_reason=exit_reason,
                    retriable=True,
                    suggested_wait_seconds=60.0,
                )

        # Default to fatal for unknown errors
        return ClassifiedError(
            category=ErrorCategory.FATAL,
            message=f"Unknown error (exit_code={exit_code})",
            original_error=exception,
            exit_code=exit_code,
            exit_signal=None,
            exit_reason=exit_reason,
            retriable=False,
        )

    def _classify_signal(
        self,
        exit_signal: int,
        exit_reason: ExitReason | None,
        exception: Exception | None,
        stdout: str,
        stderr: str,
    ) -> ClassifiedError:
        """Classify an error when process was killed by a signal.

        Different signals have different retry behaviors:
        - SIGTERM, SIGHUP, SIGPIPE: Likely retriable (graceful termination, hangup)
        - SIGKILL: Could be OOM killer or timeout - check exit_reason
        - SIGSEGV, SIGBUS, SIGABRT: Fatal crashes, not retriable
        """
        signal_name = get_signal_name(exit_signal)

        # Timeout-induced kills are retriable
        if exit_reason == "timeout":
            return ClassifiedError(
                category=ErrorCategory.TIMEOUT,
                message=f"Process killed by {signal_name} due to timeout",
                original_error=exception,
                exit_code=None,
                exit_signal=exit_signal,
                exit_reason=exit_reason,
                retriable=True,
                suggested_wait_seconds=60.0,
            )

        # Fatal signals indicate program crashes
        if exit_signal in FATAL_SIGNALS:
            return ClassifiedError(
                category=ErrorCategory.FATAL,
                message=f"Process crashed with {signal_name}",
                original_error=exception,
                exit_code=None,
                exit_signal=exit_signal,
                exit_reason=exit_reason,
                retriable=False,
            )

        # SIGKILL without timeout context is suspicious - could be OOM
        if exit_signal == signal.SIGKILL:
            # Check for OOM indicators in output
            # Be careful: "killed" alone is too generic, look for OOM-specific patterns
            combined = f"{stdout}\n{stderr}".lower()
            oom_indicators = [
                "oom",
                "out of memory",
                "oom-killer",
                "cannot allocate memory",
                "memory cgroup",
            ]
            if any(indicator in combined for indicator in oom_indicators):
                return ClassifiedError(
                    category=ErrorCategory.FATAL,
                    message=f"Process killed by {signal_name} (possible OOM)",
                    original_error=exception,
                    exit_code=None,
                    exit_signal=exit_signal,
                    exit_reason=exit_reason,
                    retriable=False,  # OOM will likely recur
                )
            # SIGKILL from unknown source - try once more
            return ClassifiedError(
                category=ErrorCategory.SIGNAL,
                message=f"Process killed by {signal_name}",
                original_error=exception,
                exit_code=None,
                exit_signal=exit_signal,
                exit_reason=exit_reason,
                retriable=True,
                suggested_wait_seconds=30.0,
            )

        # Retriable signals
        if exit_signal in RETRIABLE_SIGNALS:
            return ClassifiedError(
                category=ErrorCategory.SIGNAL,
                message=f"Process terminated by {signal_name}",
                original_error=exception,
                exit_code=None,
                exit_signal=exit_signal,
                exit_reason=exit_reason,
                retriable=True,
                suggested_wait_seconds=10.0,
            )

        # SIGINT - user interrupt, not retriable
        if exit_signal == signal.SIGINT:
            return ClassifiedError(
                category=ErrorCategory.FATAL,
                message=f"Process interrupted by {signal_name}",
                original_error=exception,
                exit_code=None,
                exit_signal=exit_signal,
                exit_reason=exit_reason,
                retriable=False,
            )

        # Unknown signal - be conservative, allow one retry
        return ClassifiedError(
            category=ErrorCategory.SIGNAL,
            message=f"Process killed by {signal_name}",
            original_error=exception,
            exit_code=None,
            exit_signal=exit_signal,
            exit_reason=exit_reason,
            retriable=True,
            suggested_wait_seconds=30.0,
        )

    def _matches_any(self, text: str, patterns: list[re.Pattern[str]]) -> bool:
        """Check if text matches any of the patterns."""
        return any(p.search(text) for p in patterns)

    @classmethod
    def from_config(cls, rate_limit_patterns: list[str]) -> "ErrorClassifier":
        """Create classifier from config patterns."""
        return cls(rate_limit_patterns=rate_limit_patterns)
