"""Error classification and handling.

Categorizes errors to determine appropriate retry behavior.
"""

import re
import signal
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from mozart.core.logging import get_logger

# Module-level logger for error classification
_logger = get_logger("errors")

# Type alias for exit reasons (matches backend)
ExitReason = Literal["completed", "timeout", "killed", "error"]


class ErrorCode(str, Enum):
    """Structured error codes for comprehensive error classification.

    Error codes are organized by category using numeric prefixes:
    - E0xx: Execution errors (timeouts, crashes, kills)
    - E1xx: Rate limit / capacity errors
    - E2xx: Validation errors
    - E3xx: Configuration errors
    - E4xx: State errors
    - E5xx: Backend errors
    - E6xx: Preflight errors

    Error codes are stable identifiers that can be used for:
    - Programmatic error handling and routing
    - Log aggregation and alerting
    - Documentation and troubleshooting guides
    - Metrics and observability dashboards
    """

    # E0xx: Execution errors
    EXECUTION_TIMEOUT = "E001"
    """Command execution exceeded timeout limit."""

    EXECUTION_KILLED = "E002"
    """Process was killed by a signal (external termination)."""

    EXECUTION_CRASHED = "E003"
    """Process crashed (segfault, bus error, abort, etc.)."""

    EXECUTION_INTERRUPTED = "E004"
    """Process was interrupted by user (SIGINT/Ctrl+C)."""

    EXECUTION_OOM = "E005"
    """Process was killed due to out of memory condition."""

    EXECUTION_UNKNOWN = "E009"
    """Unknown execution error with non-zero exit code."""

    # E1xx: Rate limit / capacity
    RATE_LIMIT_API = "E101"
    """API rate limit exceeded (429, quota, throttling)."""

    RATE_LIMIT_CLI = "E102"
    """CLI-level rate limiting detected."""

    CAPACITY_EXCEEDED = "E103"
    """Service capacity exceeded (overloaded, try again later)."""

    # E2xx: Validation errors
    VALIDATION_FILE_MISSING = "E201"
    """Expected output file does not exist."""

    VALIDATION_CONTENT_MISMATCH = "E202"
    """Output content does not match expected pattern."""

    VALIDATION_COMMAND_FAILED = "E203"
    """Validation command returned non-zero exit code."""

    VALIDATION_TIMEOUT = "E204"
    """Validation check timed out."""

    VALIDATION_GENERIC = "E209"
    """Generic validation failure (output validation needed)."""

    # E3xx: Configuration errors
    CONFIG_INVALID = "E301"
    """Configuration file is malformed or invalid."""

    CONFIG_MISSING_FIELD = "E302"
    """Required configuration field is missing."""

    CONFIG_PATH_NOT_FOUND = "E303"
    """Configuration file path does not exist."""

    CONFIG_PARSE_ERROR = "E304"
    """Failed to parse configuration file (YAML/JSON syntax error)."""

    # E4xx: State errors
    STATE_CORRUPTION = "E401"
    """Checkpoint state file is corrupted or inconsistent."""

    STATE_LOAD_FAILED = "E402"
    """Failed to load checkpoint state from storage."""

    STATE_SAVE_FAILED = "E403"
    """Failed to save checkpoint state to storage."""

    STATE_VERSION_MISMATCH = "E404"
    """Checkpoint state version is incompatible."""

    # E5xx: Backend errors
    BACKEND_CONNECTION = "E501"
    """Failed to connect to backend service."""

    BACKEND_AUTH = "E502"
    """Backend authentication or authorization failed."""

    BACKEND_RESPONSE = "E503"
    """Invalid or unexpected response from backend."""

    BACKEND_TIMEOUT = "E504"
    """Backend request timed out."""

    BACKEND_NOT_FOUND = "E505"
    """Backend executable or service not found."""

    # E6xx: Preflight errors
    PREFLIGHT_PATH_MISSING = "E601"
    """Required path does not exist (working_dir, referenced file)."""

    PREFLIGHT_PROMPT_TOO_LARGE = "E602"
    """Prompt exceeds recommended token limit."""

    PREFLIGHT_WORKING_DIR_INVALID = "E603"
    """Working directory is not accessible or not a directory."""

    PREFLIGHT_VALIDATION_SETUP = "E604"
    """Validation target path or pattern is invalid."""

    # E9xx: Transient/network errors
    NETWORK_CONNECTION_FAILED = "E901"
    """Network connection failed (refused, reset, unreachable)."""

    NETWORK_DNS_ERROR = "E902"
    """DNS resolution failed."""

    NETWORK_SSL_ERROR = "E903"
    """SSL/TLS handshake or certificate error."""

    NETWORK_TIMEOUT = "E904"
    """Network operation timed out."""

    # Fallback
    UNKNOWN = "E999"
    """Unclassified error - requires investigation."""

    @property
    def category(self) -> str:
        """Get the category prefix (first digit) of this error code.

        Returns:
            Category string like "execution", "rate_limit", "validation", etc.
        """
        category_map = {
            "0": "execution",
            "1": "rate_limit",
            "2": "validation",
            "3": "configuration",
            "4": "state",
            "5": "backend",
            "6": "preflight",
            "9": "transient",
        }
        # Error code format: "E<digit><digit><digit>"
        if len(self.value) >= 2:
            first_digit = self.value[1]
            return category_map.get(first_digit, "unknown")
        return "unknown"

    @property
    def is_retriable(self) -> bool:
        """Check if this error code is generally retriable.

        Returns:
            True if errors with this code are typically retriable.
        """
        # Non-retriable codes
        non_retriable = {
            ErrorCode.EXECUTION_CRASHED,
            ErrorCode.EXECUTION_INTERRUPTED,
            ErrorCode.EXECUTION_OOM,
            ErrorCode.BACKEND_AUTH,
            ErrorCode.CONFIG_INVALID,
            ErrorCode.CONFIG_MISSING_FIELD,
            ErrorCode.CONFIG_PARSE_ERROR,
            ErrorCode.STATE_CORRUPTION,
            ErrorCode.STATE_VERSION_MISMATCH,
        }
        return self not in non_retriable


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
    """An error with its classification and metadata.

    ClassifiedError combines high-level category (for retry logic) with
    specific error codes (for diagnostics and logging). The error_code
    provides stable identifiers for programmatic handling while category
    determines retry behavior.
    """

    category: ErrorCategory
    message: str
    error_code: ErrorCode = field(default=ErrorCode.UNKNOWN)
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

    @property
    def code(self) -> str:
        """Get the error code string value (e.g., 'E001')."""
        return self.error_code.value


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
                r"ECONNREFUSED",
                r"ETIMEDOUT",
            ])
        ]

        # More specific network sub-patterns for error code differentiation
        self.dns_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"dns.?resolution",
                r"name.?resolution",
                r"getaddrinfo",
                r"could not resolve",
                r"ENOTFOUND",
            ]
        ]

        self.ssl_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"ssl.?error",
                r"tls.?error",
                r"certificate",
                r"SSL_ERROR",
                r"handshake.?failed",
            ]
        ]

        # Capacity patterns (subset of rate limit for specific code)
        self.capacity_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"capacity",
                r"try again later",
                r"overloaded",
                r"service.?unavailable",
            ]
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
            ClassifiedError with category, error_code, and metadata
        """
        combined = f"{stdout}\n{stderr}"
        if exception:
            combined += f"\n{str(exception)}"

        # Handle signal-based exits first (new in Task 3)
        if exit_signal is not None:
            result = self._classify_signal(
                exit_signal=exit_signal,
                exit_reason=exit_reason,
                exception=exception,
                stdout=stdout,
                stderr=stderr,
            )
            _logger.warning(
                "error_classified",
                category=result.category.value,
                error_code=result.error_code.value,
                exit_signal=exit_signal,
                exit_reason=exit_reason,
                retriable=result.retriable,
                suggested_wait=result.suggested_wait_seconds,
                message=result.message,
            )
            return result

        # Handle timeout exit reason (even without signal)
        if exit_reason == "timeout":
            result = ClassifiedError(
                category=ErrorCategory.TIMEOUT,
                message="Command timed out",
                error_code=ErrorCode.EXECUTION_TIMEOUT,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=True,
                suggested_wait_seconds=60.0,
            )
            _logger.warning(
                "error_classified",
                category=result.category.value,
                error_code=result.error_code.value,
                exit_code=exit_code,
                exit_reason=exit_reason,
                retriable=result.retriable,
                message=result.message,
            )
            return result

        # Check for rate limiting first (most common retriable)
        if self._matches_any(combined, self.rate_limit_patterns):
            # Differentiate between capacity vs rate limit
            if self._matches_any(combined, self.capacity_patterns):
                error_code = ErrorCode.CAPACITY_EXCEEDED
            else:
                error_code = ErrorCode.RATE_LIMIT_API
            result = ClassifiedError(
                category=ErrorCategory.RATE_LIMIT,
                message="Rate limit detected",
                error_code=error_code,
                original_error=exception,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=True,
                suggested_wait_seconds=3600.0,  # 1 hour default
            )
            _logger.warning(
                "error_classified",
                category=result.category.value,
                error_code=result.error_code.value,
                exit_code=exit_code,
                retriable=result.retriable,
                suggested_wait=result.suggested_wait_seconds,
                message=result.message,
            )
            return result

        # Check for auth failures (fatal)
        if self._matches_any(combined, self.auth_patterns):
            result = ClassifiedError(
                category=ErrorCategory.AUTH,
                message="Authentication or authorization failure",
                error_code=ErrorCode.BACKEND_AUTH,
                original_error=exception,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=False,
            )
            _logger.warning(
                "error_classified",
                category=result.category.value,
                error_code=result.error_code.value,
                exit_code=exit_code,
                retriable=result.retriable,
                message=result.message,
            )
            return result

        # Check for network issues (retriable with backoff)
        # Check specific sub-types first for more precise error codes
        if self._matches_any(combined, self.dns_patterns):
            result = ClassifiedError(
                category=ErrorCategory.NETWORK,
                message="DNS resolution failed",
                error_code=ErrorCode.NETWORK_DNS_ERROR,
                original_error=exception,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=True,
                suggested_wait_seconds=30.0,
            )
            _logger.warning(
                "error_classified",
                category=result.category.value,
                error_code=result.error_code.value,
                exit_code=exit_code,
                retriable=result.retriable,
                suggested_wait=result.suggested_wait_seconds,
                message=result.message,
            )
            return result

        if self._matches_any(combined, self.ssl_patterns):
            result = ClassifiedError(
                category=ErrorCategory.NETWORK,
                message="SSL/TLS error",
                error_code=ErrorCode.NETWORK_SSL_ERROR,
                original_error=exception,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=True,
                suggested_wait_seconds=30.0,
            )
            _logger.warning(
                "error_classified",
                category=result.category.value,
                error_code=result.error_code.value,
                exit_code=exit_code,
                retriable=result.retriable,
                suggested_wait=result.suggested_wait_seconds,
                message=result.message,
            )
            return result

        if self._matches_any(combined, self.network_patterns):
            result = ClassifiedError(
                category=ErrorCategory.NETWORK,
                message="Network connectivity issue",
                error_code=ErrorCode.NETWORK_CONNECTION_FAILED,
                original_error=exception,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=True,
                suggested_wait_seconds=30.0,
            )
            _logger.warning(
                "error_classified",
                category=result.category.value,
                error_code=result.error_code.value,
                exit_code=exit_code,
                retriable=result.retriable,
                suggested_wait=result.suggested_wait_seconds,
                message=result.message,
            )
            return result

        # Check exit code
        if exit_code == 0:
            # Command succeeded but might have validation issues
            # Note: This is not an error, just needs validation - no warning log
            return ClassifiedError(
                category=ErrorCategory.VALIDATION,
                message="Command succeeded but output validation needed",
                error_code=ErrorCode.VALIDATION_GENERIC,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=True,
            )

        if exit_code is not None:
            # Non-zero exit codes
            if exit_code in (1, 2):
                # Common transient errors
                result = ClassifiedError(
                    category=ErrorCategory.TRANSIENT,
                    message=f"Command failed with exit code {exit_code}",
                    error_code=ErrorCode.EXECUTION_UNKNOWN,
                    original_error=exception,
                    exit_code=exit_code,
                    exit_signal=None,
                    exit_reason=exit_reason,
                    retriable=True,
                    suggested_wait_seconds=10.0,
                )
                _logger.warning(
                    "error_classified",
                    category=result.category.value,
                    error_code=result.error_code.value,
                    exit_code=exit_code,
                    retriable=result.retriable,
                    suggested_wait=result.suggested_wait_seconds,
                    message=result.message,
                )
                return result

            if exit_code == 124:
                # Timeout (from `timeout` command - legacy support)
                result = ClassifiedError(
                    category=ErrorCategory.TIMEOUT,
                    message="Command timed out",
                    error_code=ErrorCode.EXECUTION_TIMEOUT,
                    exit_code=exit_code,
                    exit_signal=None,
                    exit_reason=exit_reason,
                    retriable=True,
                    suggested_wait_seconds=60.0,
                )
                _logger.warning(
                    "error_classified",
                    category=result.category.value,
                    error_code=result.error_code.value,
                    exit_code=exit_code,
                    retriable=result.retriable,
                    suggested_wait=result.suggested_wait_seconds,
                    message=result.message,
                )
                return result

            if exit_code == 127:
                # Command not found
                result = ClassifiedError(
                    category=ErrorCategory.FATAL,
                    message=f"Command not found (exit_code={exit_code})",
                    error_code=ErrorCode.BACKEND_NOT_FOUND,
                    original_error=exception,
                    exit_code=exit_code,
                    exit_signal=None,
                    exit_reason=exit_reason,
                    retriable=False,
                )
                _logger.warning(
                    "error_classified",
                    category=result.category.value,
                    error_code=result.error_code.value,
                    exit_code=exit_code,
                    retriable=result.retriable,
                    message=result.message,
                )
                return result

        # Default to unknown for unclassified errors
        result = ClassifiedError(
            category=ErrorCategory.FATAL,
            message=f"Unknown error (exit_code={exit_code})",
            error_code=ErrorCode.UNKNOWN,
            original_error=exception,
            exit_code=exit_code,
            exit_signal=None,
            exit_reason=exit_reason,
            retriable=False,
        )
        _logger.warning(
            "error_classified",
            category=result.category.value,
            error_code=result.error_code.value,
            exit_code=exit_code,
            retriable=result.retriable,
            message=result.message,
        )
        return result

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

        Returns ClassifiedError with appropriate error_code from the E0xx execution range.
        """
        signal_name = get_signal_name(exit_signal)

        # Timeout-induced kills are retriable
        if exit_reason == "timeout":
            return ClassifiedError(
                category=ErrorCategory.TIMEOUT,
                message=f"Process killed by {signal_name} due to timeout",
                error_code=ErrorCode.EXECUTION_TIMEOUT,
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
                error_code=ErrorCode.EXECUTION_CRASHED,
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
                    error_code=ErrorCode.EXECUTION_OOM,
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
                error_code=ErrorCode.EXECUTION_KILLED,
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
                error_code=ErrorCode.EXECUTION_KILLED,
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
                error_code=ErrorCode.EXECUTION_INTERRUPTED,
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
            error_code=ErrorCode.EXECUTION_KILLED,
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
