"""Error classification and handling.

Categorizes errors to determine appropriate retry behavior.

This module provides:
- ErrorCode: Structured error codes (E0xx-E9xx) for machine-readable classification
- ErrorCategory: High-level categories for retry behavior
- ClassifiedError: Single error with classification and metadata
- ClassificationResult: Multi-error result with root cause identification
- ErrorChain: Error chain from symptom to root cause
- ParsedCliError: Structured error from CLI JSON output
- ErrorInfo: Machine-readable error metadata (Google AIP-193 inspired)

The classification algorithm:
1. Parse structured JSON errors[] from CLI output
2. Classify each error independently (no short-circuiting)
3. Select root cause using priority scoring
4. Return all errors with primary/secondary designation
"""

from __future__ import annotations

import json
import re
import signal
import warnings
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Literal, NamedTuple

from mozart.core.logging import get_logger

if TYPE_CHECKING:
    from typing import Any

# Module-level logger for error classification
_logger = get_logger("errors")

# Type alias for exit reasons (matches backend)
ExitReason = Literal["completed", "timeout", "killed", "error"]


# =============================================================================
# Retry Delay Constants
# =============================================================================


class RetryDelays:
    """Constants for retry delay durations.

    Centralizes magic numbers for retry timing to ensure consistency
    across the codebase and make timing decisions discoverable.

    These values represent standard delays for different error scenarios.
    Actual delays may be adjusted dynamically based on error context,
    parsed reset times, or learning from previous attempts.
    """

    # Rate limit delays
    API_RATE_LIMIT: float = 3600.0  # 1 hour for API rate limits
    CLI_RATE_LIMIT: float = 900.0  # 15 minutes for CLI rate limits
    CAPACITY_OVERLOAD: float = 300.0  # 5 minutes for service capacity issues

    # Token/quota delays
    QUOTA_EXHAUSTED_MIN: float = 300.0  # Minimum 5 minutes for quota resets
    QUOTA_EXHAUSTED_DEFAULT: float = 3600.0  # Default 1 hour if no reset time

    # Transient error delays
    NETWORK_ERROR: float = 30.0  # Network connectivity issues
    SERVICE_ERROR: float = 60.0  # Service unavailable / 5xx errors
    TIMEOUT_ERROR: float = 60.0  # Execution timeout recovery

    # Short delays for quick retries
    TRANSIENT_SHORT: float = 5.0  # Brief transient errors
    TRANSIENT_MEDIUM: float = 10.0  # Medium transient errors
    TRANSIENT_LONG: float = 30.0  # Longer transient errors

    # No delay (immediate retry or non-retriable)
    NONE: float = 0.0


# =============================================================================
# Severity Levels
# =============================================================================


class Severity(IntEnum):
    """Severity levels for error classification.

    Lower numeric value = higher severity. This allows comparisons like
    `if severity <= Severity.ERROR` to check for serious issues.

    Assignments:
    - CRITICAL: Job cannot continue, requires immediate attention
      (E003 crash, E005 OOM, E401 corruption, E502 auth, E505 binary not found)
    - ERROR: Operation failed, may be retriable (most error codes)
    - WARNING: Degraded operation, job may continue (E103 capacity, E204 validation timeout)
    - INFO: Informational, no action required (reserved for future diagnostic codes)
    """

    CRITICAL = 1
    ERROR = 2
    WARNING = 3
    INFO = 4


# =============================================================================
# Parsed CLI Error (from JSON output)
# =============================================================================


@dataclass
class ParsedCliError:
    """A single error extracted from CLI JSON output.

    Claude CLI returns structured JSON with an `errors[]` array:
    ```json
    {
      "result": "...",
      "errors": [
        {"type": "system", "message": "Rate limit exceeded"},
        {"type": "user", "message": "spawn claude ENOENT"}
      ],
      "cost_usd": 0.05
    }
    ```

    This dataclass represents one item from that array.

    Attributes:
        error_type: Error type from CLI: "system", "user", "tool".
        message: Human-readable error message.
        tool_name: For tool errors, the name of the failed tool.
        metadata: Additional structured metadata from the error.
    """

    error_type: str
    """Error type from CLI: "system", "user", "tool"."""

    message: str
    """Human-readable error message."""

    tool_name: str | None = None
    """For tool errors, the name of the failed tool."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional structured metadata."""


# =============================================================================
# Error Info (Google AIP-193 inspired)
# =============================================================================


@dataclass
class ErrorInfo:
    """Machine-readable error identification (Google AIP-193 inspired).

    Provides structured metadata for programmatic error handling.

    Example:
    ```python
    error_info = ErrorInfo(
        reason="BINARY_NOT_FOUND",
        domain="mozart.backend.claude_cli",
        metadata={
            "expected_binary": "claude",
            "search_path": "/usr/bin:/usr/local/bin",
            "suggestion": "Ensure claude CLI is installed and in PATH",
        }
    )
    ```

    Attributes:
        reason: UPPER_SNAKE_CASE identifier for the specific error reason.
        domain: Service/component identifier (e.g., "mozart.backend.claude_cli").
        metadata: Dynamic contextual information as key-value pairs.
    """

    reason: str
    """UPPER_SNAKE_CASE identifier for the specific error reason.
    Example: "RATE_LIMIT_EXCEEDED", "BINARY_NOT_FOUND"
    """

    domain: str
    """Service/component identifier.
    Example: "mozart.backend.claude_cli", "mozart.execution"
    """

    metadata: dict[str, str] = field(default_factory=dict)
    """Dynamic contextual information.
    Example: {"binary_path": "/usr/bin/claude", "exit_code": "127"}
    """


# =============================================================================
# Retry Behavior
# =============================================================================


class RetryBehavior(NamedTuple):
    """Precise retry behavior recommendation for a specific error code.

    Unlike ErrorCategory which provides broad retry guidelines, RetryBehavior
    encodes error-code-specific knowledge about optimal retry strategies.

    Attributes:
        delay_seconds: Recommended delay before retrying (0 = no delay).
        is_retriable: Whether this error is generally retriable.
        reason: Human-readable explanation for the retry behavior.
    """

    delay_seconds: float
    is_retriable: bool
    reason: str


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

    QUOTA_EXHAUSTED = "E104"
    """Token/usage quota exhausted - wait until reset time."""

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

    CONFIG_MCP_ERROR = "E305"
    """MCP server/plugin configuration error (missing env vars, invalid config)."""

    CONFIG_CLI_MODE_ERROR = "E306"
    """Claude CLI mode mismatch (e.g., streaming mode incompatible with operation)."""

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

    def get_retry_behavior(self) -> RetryBehavior:
        """Get precise retry behavior for this error code.

        Returns error-code-specific delay and retry recommendations.
        This provides finer-grained control than ErrorCategory alone.

        Returns:
            RetryBehavior with delay, retriability, and reason.
        """
        # Error-code-specific retry behaviors
        # Organized by category for maintainability
        behaviors: dict[ErrorCode, RetryBehavior] = {
            # E0xx: Execution errors
            ErrorCode.EXECUTION_TIMEOUT: RetryBehavior(
                delay_seconds=60.0,
                is_retriable=True,
                reason="Timeout - allow system to recover before retry",
            ),
            ErrorCode.EXECUTION_KILLED: RetryBehavior(
                delay_seconds=30.0,
                is_retriable=True,
                reason="Process killed externally - may be transient",
            ),
            ErrorCode.EXECUTION_CRASHED: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="Process crashed - likely a bug, not transient",
            ),
            ErrorCode.EXECUTION_INTERRUPTED: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="User interrupted - respect user intent",
            ),
            ErrorCode.EXECUTION_OOM: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="Out of memory - will likely recur",
            ),
            ErrorCode.EXECUTION_UNKNOWN: RetryBehavior(
                delay_seconds=10.0,
                is_retriable=True,
                reason="Unknown execution error - attempt retry with backoff",
            ),
            # E1xx: Rate limit / capacity
            ErrorCode.RATE_LIMIT_API: RetryBehavior(
                delay_seconds=3600.0,  # 1 hour for API rate limits
                is_retriable=True,
                reason="API rate limit - wait for quota reset",
            ),
            ErrorCode.RATE_LIMIT_CLI: RetryBehavior(
                delay_seconds=900.0,  # 15 minutes for CLI rate limits
                is_retriable=True,
                reason="CLI rate limit - shorter wait than API",
            ),
            ErrorCode.CAPACITY_EXCEEDED: RetryBehavior(
                delay_seconds=300.0,  # 5 minutes for capacity
                is_retriable=True,
                reason="Service overloaded - wait for capacity",
            ),
            ErrorCode.QUOTA_EXHAUSTED: RetryBehavior(
                delay_seconds=0.0,  # Dynamic - parsed from reset time in message
                is_retriable=True,
                reason="Token quota exhausted - wait until reset time",
            ),
            # E2xx: Validation errors
            ErrorCode.VALIDATION_FILE_MISSING: RetryBehavior(
                delay_seconds=5.0,
                is_retriable=True,
                reason="File not created - prompt may need adjustment",
            ),
            ErrorCode.VALIDATION_CONTENT_MISMATCH: RetryBehavior(
                delay_seconds=5.0,
                is_retriable=True,
                reason="Content doesn't match - retry with same prompt",
            ),
            ErrorCode.VALIDATION_COMMAND_FAILED: RetryBehavior(
                delay_seconds=10.0,
                is_retriable=True,
                reason="Validation command failed - may be transient",
            ),
            ErrorCode.VALIDATION_TIMEOUT: RetryBehavior(
                delay_seconds=30.0,
                is_retriable=True,
                reason="Validation timed out - allow more time",
            ),
            ErrorCode.VALIDATION_GENERIC: RetryBehavior(
                delay_seconds=5.0,
                is_retriable=True,
                reason="Validation needed - retry with validation",
            ),
            # E3xx: Configuration errors - generally not retriable
            ErrorCode.CONFIG_INVALID: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="Invalid config - requires user fix",
            ),
            ErrorCode.CONFIG_MISSING_FIELD: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="Missing config field - requires user fix",
            ),
            ErrorCode.CONFIG_PATH_NOT_FOUND: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="Config file not found - requires user fix",
            ),
            ErrorCode.CONFIG_PARSE_ERROR: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="Config parse error - requires user fix",
            ),
            # E4xx: State errors
            ErrorCode.STATE_CORRUPTION: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="Corrupted state - requires manual recovery",
            ),
            ErrorCode.STATE_LOAD_FAILED: RetryBehavior(
                delay_seconds=5.0,
                is_retriable=True,
                reason="State load failed - may be transient I/O",
            ),
            ErrorCode.STATE_SAVE_FAILED: RetryBehavior(
                delay_seconds=5.0,
                is_retriable=True,
                reason="State save failed - may be transient I/O",
            ),
            ErrorCode.STATE_VERSION_MISMATCH: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="State version mismatch - requires migration",
            ),
            # E5xx: Backend errors
            ErrorCode.BACKEND_CONNECTION: RetryBehavior(
                delay_seconds=30.0,
                is_retriable=True,
                reason="Backend connection failed - may recover",
            ),
            ErrorCode.BACKEND_AUTH: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="Auth failed - requires credential fix",
            ),
            ErrorCode.BACKEND_RESPONSE: RetryBehavior(
                delay_seconds=15.0,
                is_retriable=True,
                reason="Invalid backend response - may be transient",
            ),
            ErrorCode.BACKEND_TIMEOUT: RetryBehavior(
                delay_seconds=60.0,
                is_retriable=True,
                reason="Backend timeout - allow recovery time",
            ),
            ErrorCode.BACKEND_NOT_FOUND: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="Backend not found - requires installation",
            ),
            # E6xx: Preflight errors
            ErrorCode.PREFLIGHT_PATH_MISSING: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="Required path missing - requires user fix",
            ),
            ErrorCode.PREFLIGHT_PROMPT_TOO_LARGE: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="Prompt too large - requires user fix",
            ),
            ErrorCode.PREFLIGHT_WORKING_DIR_INVALID: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="Invalid working dir - requires user fix",
            ),
            ErrorCode.PREFLIGHT_VALIDATION_SETUP: RetryBehavior(
                delay_seconds=0.0,
                is_retriable=False,
                reason="Invalid validation setup - requires user fix",
            ),
            # E9xx: Transient/network errors
            ErrorCode.NETWORK_CONNECTION_FAILED: RetryBehavior(
                delay_seconds=30.0,
                is_retriable=True,
                reason="Network connection failed - may recover",
            ),
            ErrorCode.NETWORK_DNS_ERROR: RetryBehavior(
                delay_seconds=30.0,
                is_retriable=True,
                reason="DNS error - may be transient",
            ),
            ErrorCode.NETWORK_SSL_ERROR: RetryBehavior(
                delay_seconds=30.0,
                is_retriable=True,
                reason="SSL error - may be transient",
            ),
            ErrorCode.NETWORK_TIMEOUT: RetryBehavior(
                delay_seconds=60.0,
                is_retriable=True,
                reason="Network timeout - allow recovery time",
            ),
            # Fallback
            ErrorCode.UNKNOWN: RetryBehavior(
                delay_seconds=30.0,
                is_retriable=True,
                reason="Unknown error - attempt retry with backoff",
            ),
        }

        # Return specific behavior or a sensible default
        return behaviors.get(
            self,
            RetryBehavior(
                delay_seconds=30.0,
                is_retriable=self.is_retriable,
                reason=f"Default behavior for {self.value}",
            ),
        )

    def get_severity(self) -> Severity:
        """Get the severity level for this error code.

        Severity assignments:
        - CRITICAL: Fatal errors requiring immediate attention
        - ERROR: Most error codes (default)
        - WARNING: Degraded but potentially temporary conditions
        - INFO: Reserved for future diagnostic codes

        Returns:
            Severity level for this error code.
        """
        # Critical errors - job cannot continue
        critical_codes = {
            ErrorCode.EXECUTION_CRASHED,
            ErrorCode.EXECUTION_OOM,
            ErrorCode.STATE_CORRUPTION,
            ErrorCode.BACKEND_AUTH,
            ErrorCode.BACKEND_NOT_FOUND,
        }
        if self in critical_codes:
            return Severity.CRITICAL

        # Warning level - degraded but potentially temporary
        warning_codes = {
            ErrorCode.CAPACITY_EXCEEDED,
            ErrorCode.VALIDATION_TIMEOUT,
        }
        if self in warning_codes:
            return Severity.WARNING

        # Default to ERROR for most codes
        return Severity.ERROR


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

    CONFIGURATION = "configuration"
    """Non-retriable - configuration error needs user intervention (e.g., MCP setup)."""


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
    error_info: ErrorInfo | None = None
    """Optional structured metadata for this error."""

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

    @property
    def severity(self) -> Severity:
        """Get the severity level for this error."""
        return self.error_code.get_severity()


# =============================================================================
# Error Chain (for multi-error scenarios)
# =============================================================================


@dataclass
class ErrorChain:
    """Represents a chain of errors from symptom to root cause.

    When multiple errors occur, this class helps identify the actual
    root cause vs symptoms. For example, if ENOENT and rate limit both
    appear, ENOENT is likely the root cause (missing binary prevents
    any operation).

    Attributes:
        errors: All errors in order of detection (first = earliest).
        root_cause: The error identified as the most fundamental cause.
        symptoms: Errors that are likely consequences of the root cause.
        confidence: 0.0-1.0 confidence in root cause identification.
    """

    errors: list[ClassifiedError]
    """All errors in order of detection (first = earliest)."""

    root_cause: ClassifiedError
    """The error identified as the most fundamental cause."""

    symptoms: list[ClassifiedError] = field(default_factory=list)
    """Errors that are likely consequences of the root cause."""

    confidence: float = 1.0
    """0.0-1.0 confidence in root cause identification."""


# =============================================================================
# Classification Result (new multi-error result type)
# =============================================================================


@dataclass
class ClassificationResult:
    """Complete classification result with root cause and context.

    This is the new result type from the classifier, providing access
    to all detected errors while maintaining backward compatibility
    through the `primary` attribute.

    Example:
    ```python
    # New code - returns ClassificationResult
    classification = classifier.classify(stdout, stderr, exit_code)
    result = classification.primary  # Backward compatible

    # Access all errors
    for error in classification.all_errors:
        log.info(f"Error: {error.error_code.value} - {error.message}")
    ```

    Attributes:
        primary: The identified root cause error.
        secondary: Secondary/symptom errors for debugging.
        raw_errors: Original parsed errors from CLI JSON.
        confidence: 0.0-1.0 confidence in root cause identification.
        classification_method: How classification was done.
    """

    primary: ClassifiedError
    """The identified root cause error."""

    secondary: list[ClassifiedError] = field(default_factory=list)
    """Secondary/symptom errors for debugging."""

    raw_errors: list[ParsedCliError] = field(default_factory=list)
    """Original parsed errors from CLI JSON."""

    confidence: float = 1.0
    """0.0-1.0 confidence in root cause identification."""

    classification_method: str = "structured"
    """How classification was done: "structured", "exit_code", "regex_fallback"."""

    @property
    def all_errors(self) -> list[ClassifiedError]:
        """All errors including primary and secondary."""
        return [self.primary] + self.secondary

    @property
    def error_codes(self) -> list[str]:
        """All error codes for logging/metrics."""
        return [e.error_code.value for e in self.all_errors]

    @property
    def category(self) -> ErrorCategory:
        """Category of the primary error (backward compatibility)."""
        return self.primary.category

    @property
    def message(self) -> str:
        """Message of the primary error (backward compatibility)."""
        return self.primary.message

    @property
    def error_code(self) -> ErrorCode:
        """Error code of the primary error (backward compatibility)."""
        return self.primary.error_code

    @property
    def retriable(self) -> bool:
        """Whether the primary error is retriable (backward compatibility)."""
        return self.primary.retriable

    @property
    def should_retry(self) -> bool:
        """Whether to retry based on primary error (backward compatibility)."""
        return self.primary.should_retry

    def to_error_chain(self) -> ErrorChain:
        """Convert to ErrorChain for detailed analysis."""
        return ErrorChain(
            errors=self.all_errors,
            root_cause=self.primary,
            symptoms=self.secondary,
            confidence=self.confidence,
        )


# =============================================================================
# Root Cause Priority (for selecting root cause from multiple errors)
# =============================================================================


# Priority scores for root cause selection.
# Lower score = more likely to be root cause.
# Tier 1 (10-19): Environment issues - prevent execution entirely
# Tier 2 (20-29): Configuration issues - bad config causes cascading failures
# Tier 3 (30-39): Authentication - auth failures cause downstream errors
# Tier 4 (40-49): Network/Connection - network issues cause service errors
# Tier 5 (50-59): Service issues - specific service problems
# Tier 6 (60-69): Execution issues - runtime problems
# Tier 7 (70-79): State issues - checkpoint problems
# Tier 8 (80-89): Validation/Output - usually symptoms, not causes
# Tier 9 (90+): Unknown/Generic
ROOT_CAUSE_PRIORITY: dict[ErrorCode, int] = {
    # Tier 1: Environment issues (priority 10-19)
    ErrorCode.BACKEND_NOT_FOUND: 10,
    ErrorCode.PREFLIGHT_PATH_MISSING: 11,
    ErrorCode.PREFLIGHT_WORKING_DIR_INVALID: 12,
    ErrorCode.CONFIG_PATH_NOT_FOUND: 13,

    # Tier 2: Configuration issues (priority 20-29)
    ErrorCode.CONFIG_INVALID: 20,
    ErrorCode.CONFIG_MISSING_FIELD: 21,
    ErrorCode.CONFIG_PARSE_ERROR: 22,
    ErrorCode.CONFIG_MCP_ERROR: 23,
    ErrorCode.CONFIG_CLI_MODE_ERROR: 24,

    # Tier 3: Authentication (priority 30-39)
    ErrorCode.BACKEND_AUTH: 30,

    # Tier 4: Network/Connection (priority 40-49)
    ErrorCode.NETWORK_CONNECTION_FAILED: 40,
    ErrorCode.NETWORK_DNS_ERROR: 41,
    ErrorCode.NETWORK_SSL_ERROR: 42,
    ErrorCode.BACKEND_CONNECTION: 43,

    # Tier 5: Service Issues (priority 50-59)
    ErrorCode.RATE_LIMIT_API: 50,
    ErrorCode.RATE_LIMIT_CLI: 51,
    ErrorCode.CAPACITY_EXCEEDED: 52,
    ErrorCode.QUOTA_EXHAUSTED: 53,
    ErrorCode.BACKEND_TIMEOUT: 54,
    ErrorCode.NETWORK_TIMEOUT: 55,

    # Tier 6: Execution Issues (priority 60-69)
    ErrorCode.EXECUTION_TIMEOUT: 60,
    ErrorCode.EXECUTION_OOM: 61,
    ErrorCode.EXECUTION_CRASHED: 62,
    ErrorCode.EXECUTION_KILLED: 63,
    ErrorCode.EXECUTION_INTERRUPTED: 64,

    # Tier 7: State Issues (priority 70-79)
    ErrorCode.STATE_CORRUPTION: 70,
    ErrorCode.STATE_VERSION_MISMATCH: 71,
    ErrorCode.STATE_LOAD_FAILED: 72,
    ErrorCode.STATE_SAVE_FAILED: 73,

    # Tier 8: Validation/Output Issues (priority 80-89)
    ErrorCode.VALIDATION_FILE_MISSING: 80,
    ErrorCode.VALIDATION_CONTENT_MISMATCH: 81,
    ErrorCode.VALIDATION_COMMAND_FAILED: 82,
    ErrorCode.VALIDATION_TIMEOUT: 83,
    ErrorCode.VALIDATION_GENERIC: 84,
    ErrorCode.PREFLIGHT_PROMPT_TOO_LARGE: 85,
    ErrorCode.PREFLIGHT_VALIDATION_SETUP: 86,

    # Tier 9: Unknown/Generic (priority 90+)
    ErrorCode.EXECUTION_UNKNOWN: 90,
    ErrorCode.BACKEND_RESPONSE: 91,
    ErrorCode.UNKNOWN: 99,
}


# =============================================================================
# JSON Parsing Utilities
# =============================================================================


def try_parse_json_errors(output: str, stderr: str = "") -> list[ParsedCliError]:
    """Extract errors[] array from JSON output.

    Claude CLI returns structured JSON with an `errors[]` array:
    ```json
    {
      "result": "...",
      "errors": [
        {"type": "system", "message": "Rate limit exceeded"},
        {"type": "user", "message": "spawn claude ENOENT"}
      ],
      "cost_usd": 0.05
    }
    ```

    This function parses that structure, handling:
    - Non-JSON preamble (CLI startup messages)
    - Multiple JSON objects (takes first valid one with errors[])
    - JSON in stderr (some error modes write there)
    - Truncated JSON (tries to recover)

    Args:
        output: Raw stdout from Claude CLI execution.
        stderr: Optional stderr output (some errors appear here).

    Returns:
        List of ParsedCliError objects, or empty list if parsing fails.
    """
    errors: list[ParsedCliError] = []

    # Try both stdout and stderr - errors can appear in either
    for text in [output, stderr]:
        if not text:
            continue

        found_errors = _extract_json_errors_from_text(text)
        if found_errors:
            errors.extend(found_errors)

    # Deduplicate by message (same error might appear in both streams)
    seen_messages: set[str] = set()
    unique_errors: list[ParsedCliError] = []
    for error in errors:
        if error.message not in seen_messages:
            seen_messages.add(error.message)
            unique_errors.append(error)

    return unique_errors


def _extract_json_errors_from_text(text: str) -> list[ParsedCliError]:
    """Extract errors from a single text stream.

    Handles multiple JSON objects and partial parsing.

    Args:
        text: Text that may contain JSON with errors[] array.

    Returns:
        List of ParsedCliError objects found.
    """
    errors: list[ParsedCliError] = []

    # Find all potential JSON object starts
    idx = 0
    while idx < len(text):
        json_start = text.find("{", idx)
        if json_start == -1:
            break

        # Try to find matching closing brace with bracket counting
        depth = 0
        json_end = json_start
        in_string = False
        escape_next = False

        for i in range(json_start, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

        if depth != 0:
            # Incomplete JSON, try parsing anyway (might work for simple cases)
            json_end = len(text)

        try:
            json_str = text[json_start:json_end]
            data = json.loads(json_str)

            if "errors" in data and isinstance(data["errors"], list):
                for item in data["errors"]:
                    if not isinstance(item, dict):
                        continue
                    error = ParsedCliError(
                        error_type=item.get("type", "unknown"),
                        message=item.get("message", ""),
                        tool_name=item.get("tool_name"),
                        metadata=item.get("metadata", {}),
                    )
                    errors.append(error)

                # Found valid errors, return them
                if errors:
                    return errors

        except (json.JSONDecodeError, TypeError, KeyError):
            pass

        # Move past this JSON object to find next potential one
        idx = json_end if json_end > json_start else json_start + 1

    return errors


def classify_single_json_error(
    parsed_error: ParsedCliError,
    exit_code: int | None = None,
    exit_reason: ExitReason | None = None,
) -> ClassifiedError:
    """Classify a single error from the JSON errors[] array.

    This function uses type-based classification first, then falls back to
    message pattern matching. The error type from CLI ("system", "user", "tool")
    guides initial classification.

    Args:
        parsed_error: A ParsedCliError extracted from CLI JSON output.
        exit_code: Optional exit code for context.
        exit_reason: Optional exit reason for context.

    Returns:
        ClassifiedError with appropriate category and error code.
    """
    message = parsed_error.message.lower()
    error_type = parsed_error.error_type.lower()

    # === Type-based classification ===

    if error_type == "system":
        # System errors are usually API/service level
        # Check rate limit patterns
        rate_limit_indicators = [
            "rate limit", "rate_limit", "quota", "too many requests",
            "429", "hit your limit", "limit exceeded", "daily limit",
        ]
        if any(indicator in message for indicator in rate_limit_indicators):
            # Differentiate capacity vs rate limit
            capacity_indicators = ["capacity", "overloaded", "try again later", "unavailable"]
            if any(indicator in message for indicator in capacity_indicators):
                return ClassifiedError(
                    category=ErrorCategory.RATE_LIMIT,
                    message=parsed_error.message,
                    error_code=ErrorCode.CAPACITY_EXCEEDED,
                    exit_code=exit_code,
                    exit_reason=exit_reason,
                    retriable=True,
                    suggested_wait_seconds=300.0,
                )
            return ClassifiedError(
                category=ErrorCategory.RATE_LIMIT,
                message=parsed_error.message,
                error_code=ErrorCode.RATE_LIMIT_API,
                exit_code=exit_code,
                exit_reason=exit_reason,
                retriable=True,
                suggested_wait_seconds=3600.0,
            )

        # Check auth patterns
        auth_indicators = ["unauthorized", "authentication", "invalid api key", "401", "403"]
        if any(indicator in message for indicator in auth_indicators):
            return ClassifiedError(
                category=ErrorCategory.AUTH,
                message=parsed_error.message,
                error_code=ErrorCode.BACKEND_AUTH,
                exit_code=exit_code,
                exit_reason=exit_reason,
                retriable=False,
            )

    elif error_type == "user":
        # User errors are usually environment/config issues
        # ENOENT is critical - often the root cause
        # Common patterns: "ENOENT", "spawn claude ENOENT", "command not found"
        if "enoent" in message or "command not found" in message:
            return ClassifiedError(
                category=ErrorCategory.CONFIGURATION,
                message=parsed_error.message,
                error_code=ErrorCode.BACKEND_NOT_FOUND,
                exit_code=exit_code,
                exit_reason=exit_reason,
                retriable=True,  # Might recover after reinstall
                suggested_wait_seconds=30.0,
                error_info=ErrorInfo(
                    reason="BINARY_NOT_FOUND",
                    domain="mozart.backend.claude_cli",
                    metadata={"original_message": parsed_error.message},
                ),
            )

        if "permission denied" in message or "access denied" in message:
            return ClassifiedError(
                category=ErrorCategory.AUTH,
                message=parsed_error.message,
                error_code=ErrorCode.BACKEND_AUTH,
                exit_code=exit_code,
                exit_reason=exit_reason,
                retriable=False,
            )

        if "no such file" in message or "not found" in message:
            return ClassifiedError(
                category=ErrorCategory.CONFIGURATION,
                message=parsed_error.message,
                error_code=ErrorCode.CONFIG_PATH_NOT_FOUND,
                exit_code=exit_code,
                exit_reason=exit_reason,
                retriable=False,
            )

    elif error_type == "tool":
        # Tool errors need message analysis
        if "mcp" in message or "server" in message:
            return ClassifiedError(
                category=ErrorCategory.CONFIGURATION,
                message=parsed_error.message,
                error_code=ErrorCode.CONFIG_MCP_ERROR,
                exit_code=exit_code,
                exit_reason=exit_reason,
                retriable=False,
            )

        # Tool execution failures are often validation issues
        return ClassifiedError(
            category=ErrorCategory.VALIDATION,
            message=parsed_error.message,
            error_code=ErrorCode.VALIDATION_COMMAND_FAILED,
            exit_code=exit_code,
            exit_reason=exit_reason,
            retriable=True,
            suggested_wait_seconds=10.0,
        )

    # === Message pattern fallback ===

    # Network errors
    network_indicators = [
        "connection refused", "connection reset", "econnrefused",
        "etimedout", "network unreachable",
    ]
    if any(indicator in message for indicator in network_indicators):
        return ClassifiedError(
            category=ErrorCategory.NETWORK,
            message=parsed_error.message,
            error_code=ErrorCode.NETWORK_CONNECTION_FAILED,
            exit_code=exit_code,
            exit_reason=exit_reason,
            retriable=True,
            suggested_wait_seconds=30.0,
        )

    # DNS errors
    dns_indicators = ["dns", "getaddrinfo", "enotfound", "resolve"]
    if any(indicator in message for indicator in dns_indicators):
        return ClassifiedError(
            category=ErrorCategory.NETWORK,
            message=parsed_error.message,
            error_code=ErrorCode.NETWORK_DNS_ERROR,
            exit_code=exit_code,
            exit_reason=exit_reason,
            retriable=True,
            suggested_wait_seconds=30.0,
        )

    # SSL/TLS errors
    ssl_indicators = ["ssl", "tls", "certificate", "handshake"]
    if any(indicator in message for indicator in ssl_indicators):
        return ClassifiedError(
            category=ErrorCategory.NETWORK,
            message=parsed_error.message,
            error_code=ErrorCode.NETWORK_SSL_ERROR,
            exit_code=exit_code,
            exit_reason=exit_reason,
            retriable=True,
            suggested_wait_seconds=30.0,
        )

    # Timeout patterns
    timeout_indicators = ["timeout", "timed out"]
    if any(indicator in message for indicator in timeout_indicators):
        return ClassifiedError(
            category=ErrorCategory.TIMEOUT,
            message=parsed_error.message,
            error_code=ErrorCode.EXECUTION_TIMEOUT,
            exit_code=exit_code,
            exit_reason=exit_reason,
            retriable=True,
            suggested_wait_seconds=60.0,
        )

    # Default: unknown error with the original message
    return ClassifiedError(
        category=ErrorCategory.TRANSIENT,
        message=parsed_error.message,
        error_code=ErrorCode.UNKNOWN,
        exit_code=exit_code,
        exit_reason=exit_reason,
        retriable=True,
        suggested_wait_seconds=30.0,
    )


def select_root_cause(errors: list[ClassifiedError]) -> tuple[ClassifiedError, list[ClassifiedError], float]:
    """Select the most likely root cause from multiple errors.

    Uses priority-based scoring where lower score = more fundamental cause.
    Applies context modifiers for specific error combinations that commonly
    mask root causes.

    Known masking patterns:
    - ENOENT masks everything (missing binary causes cascading failures)
    - Auth errors mask rate limits (can't hit rate limit if auth fails)
    - Network errors mask service errors (can't reach service to get errors)
    - Config errors mask execution errors (bad config causes execution failure)
    - Timeout masks completion (timed out = never got to complete)

    Args:
        errors: List of classified errors to analyze.

    Returns:
        Tuple of (root_cause, symptoms, confidence).
        - root_cause: The most fundamental error that likely caused others
        - symptoms: Other errors that are likely consequences
        - confidence: 0.0-1.0 confidence in root cause identification
          (higher when there's a clear priority gap)
    """
    if not errors:
        # Return an unknown error as fallback
        unknown = ClassifiedError(
            category=ErrorCategory.FATAL,
            message="No errors provided",
            error_code=ErrorCode.UNKNOWN,
            retriable=False,
        )
        return (unknown, [], 0.0)

    if len(errors) == 1:
        return (errors[0], [], 1.0)

    # Calculate modified priorities using index-based lookup
    # (ClassifiedError is a mutable dataclass and not hashable)
    error_codes_present = {e.error_code for e in errors}
    priorities: list[int] = []

    for error in errors:
        priority = ROOT_CAUSE_PRIORITY.get(error.error_code, 99)

        # === Priority Modifiers for Common Masking Patterns ===

        # ENOENT (missing binary) masks everything - it's almost always root cause
        if error.error_code == ErrorCode.BACKEND_NOT_FOUND:
            if any(e.error_code != ErrorCode.BACKEND_NOT_FOUND for e in errors):
                priority -= 10  # Strong boost - ENOENT is very fundamental

        # Config path not found is similar - can't run without config
        if error.error_code == ErrorCode.CONFIG_PATH_NOT_FOUND:
            priority -= 5

        # Auth errors mask rate limits (can't be rate limited if auth fails)
        if error.error_code == ErrorCode.BACKEND_AUTH:
            if ErrorCode.RATE_LIMIT_API in error_codes_present or ErrorCode.RATE_LIMIT_CLI in error_codes_present:
                priority -= 5

        # Network errors mask service errors
        if error.error_code in (
            ErrorCode.NETWORK_CONNECTION_FAILED,
            ErrorCode.NETWORK_DNS_ERROR,
            ErrorCode.NETWORK_SSL_ERROR,
        ):
            if ErrorCode.BACKEND_TIMEOUT in error_codes_present or ErrorCode.RATE_LIMIT_API in error_codes_present:
                priority -= 3

        # MCP config errors mask tool execution errors
        if error.error_code == ErrorCode.CONFIG_MCP_ERROR:
            if ErrorCode.VALIDATION_COMMAND_FAILED in error_codes_present:
                priority -= 3

        # CLI mode errors (streaming vs JSON) are config issues that mask execution
        if error.error_code == ErrorCode.CONFIG_CLI_MODE_ERROR:
            if any(e.error_code.category == "execution" for e in errors):
                priority -= 3

        # Timeout is a symptom when paired with rate limits (waited too long)
        if error.error_code == ErrorCode.EXECUTION_TIMEOUT:
            if ErrorCode.RATE_LIMIT_API in error_codes_present:
                priority += 5  # Demote timeout - rate limit is root cause

        priorities.append(priority)

    # Find minimum priority (root cause)
    min_idx = min(range(len(errors)), key=lambda i: priorities[i])
    root_cause = errors[min_idx]
    root_priority = priorities[min_idx]

    # Build symptoms list (all errors except root cause)
    symptoms = [errors[i] for i in range(len(errors)) if i != min_idx]
    symptom_priorities = [priorities[i] for i in range(len(errors)) if i != min_idx]

    # Calculate confidence based on priority gap
    # Higher gap = clearer root cause = more confidence
    if symptom_priorities:
        next_priority = min(symptom_priorities)
        gap = next_priority - root_priority

        # Base confidence starts at 0.5 for multiple errors
        # Each priority tier gap adds 5% confidence
        confidence = min(0.5 + (gap * 0.05), 1.0)

        # Boost confidence for known high-signal root causes
        if root_cause.error_code in (
            ErrorCode.BACKEND_NOT_FOUND,  # ENOENT is almost always correct
            ErrorCode.BACKEND_AUTH,  # Auth failures are clear
            ErrorCode.CONFIG_PATH_NOT_FOUND,  # Missing config is clear
        ):
            confidence = min(confidence + 0.15, 1.0)

        # Lower confidence when all errors are in same tier (ambiguous)
        if gap == 0:
            confidence = 0.4  # Significant ambiguity
    else:
        confidence = 1.0

    return (root_cause, symptoms, confidence)


# =============================================================================
# Deprecation Helpers
# =============================================================================


def _emit_deprecation_warning(old_name: str, new_name: str) -> None:
    """Issue a deprecation warning for old API usage.

    Used internally to warn callers about deprecated API methods.

    Args:
        old_name: Name of the deprecated method/function.
        new_name: Name of the replacement method/function.
    """
    warnings.warn(
        f"{old_name} is deprecated, use {new_name} instead",
        DeprecationWarning,
        stacklevel=3,
    )


# =============================================================================
# Error Classifier
# =============================================================================


class ErrorClassifier:
    """Classifies errors based on patterns and exit codes.

    Pattern matching follows the approach from run-sheet-review.sh
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
                r"hit.{0,10}limit",  # "You've hit your limit"
                r"limit.{0,10}resets?",  # "limit · resets 9pm"
                r"daily.{0,10}limit",  # "daily limit reached"
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

        # Token quota exhaustion patterns (Claude Max token budgets)
        # These indicate daily/hourly token quotas are exhausted with a reset time
        self.quota_exhaustion_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"token.{0,10}exhausted",  # "tokens exhausted"
                r"token.{0,10}budget.{0,10}(used|exhausted|depleted)",
                r"usage.{0,10}(will\s+)?reset.{0,10}(at|in)",  # "usage will reset at 9pm"
                r"resets?.{0,10}\d+\s*[ap]m",  # "resets 9pm" or "reset at 9 pm"
                r"resets?.{0,10}in\s+\d+\s*(hour|minute|min|hr)",  # "resets in 3 hours"
                r"daily.{0,10}(token|usage).{0,10}limit",
                r"hourly.{0,10}(token|usage).{0,10}limit",
                r"(used|consumed).{0,10}all.{0,10}(token|credit)",
                r"no.{0,10}(token|credit).{0,10}(left|remaining)",
                r"token.{0,10}allowance.{0,10}(used|exhausted)",
                r"recharge.{0,10}(at|in)",  # "recharge at midnight"
            ]
        ]

        # Regex to extract reset time from messages
        # Captures patterns like "9pm", "9 pm", "21:00", "in 3 hours"
        self.reset_time_patterns = [
            re.compile(r"resets?\s+(?:at\s+)?(\d{1,2})\s*([ap]m)", re.IGNORECASE),  # "resets at 9pm"
            re.compile(r"resets?\s+(?:at\s+)?(\d{1,2}):(\d{2})", re.IGNORECASE),  # "resets at 21:00"
            re.compile(r"resets?\s+in\s+(\d+)\s*(hour|hr|minute|min)s?", re.IGNORECASE),  # "resets in 3 hours"
            re.compile(r"reset.{0,20}(\d{1,2})\s*([ap]m)", re.IGNORECASE),  # "reset ... 9pm"
        ]

        # MCP/Plugin configuration errors (non-retriable)
        self.mcp_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"MCP server error",
                r"mcp-config-invalid",
                r"Missing environment variables:",
                r"Plugin MCP server",
                r"MCP server .+ invalid",
            ]
        ]

        # CLI mode mismatch errors (non-retriable configuration issue)
        self.cli_mode_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"only prompt commands are supported in streaming mode",
                r"streaming mode.*not supported",
                r"output format.*not compatible",
            ]
        ]

        # Missing binary/file errors (ENOENT) - check BEFORE streaming mode
        # These are often the REAL cause when CLI reports misleading errors
        self.enoent_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"ENOENT",
                r"spawn .+ ENOENT",
                r"no such file or directory",
                r"command not found",
                r"not found in PATH",
            ]
        ]

    def parse_reset_time(self, text: str) -> float | None:
        """Parse reset time from message and return seconds until reset.

        Supports patterns like:
        - "resets at 9pm" -> seconds until 9pm (or next day if past)
        - "resets at 21:00" -> seconds until 21:00
        - "resets in 3 hours" -> 3 * 3600 seconds
        - "resets in 30 minutes" -> 30 * 60 seconds

        Args:
            text: Error message that may contain reset time info.

        Returns:
            Seconds until reset, or None if no reset time found.
            Returns minimum of 300 seconds (5 min) to avoid immediate retries.
        """
        from datetime import datetime, timedelta

        for pattern in self.reset_time_patterns:
            match = pattern.search(text)
            if not match:
                continue

            groups = match.groups()

            # Pattern: "resets in X hours/minutes"
            if len(groups) == 2 and groups[1] and groups[1].lower() in ("hour", "hr", "minute", "min"):
                amount = int(groups[0])
                unit = groups[1].lower()
                if unit in ("hour", "hr"):
                    seconds = amount * 3600
                else:  # minute, min
                    seconds = amount * 60
                return max(seconds, 300.0)  # At least 5 minutes

            # Pattern: "resets at X:XX" (24-hour time)
            if len(groups) == 2 and groups[1] and groups[1].isdigit():
                hour = int(groups[0])
                minute = int(groups[1])
                now = datetime.now()
                reset_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if reset_time <= now:
                    reset_time += timedelta(days=1)  # Next day
                seconds = (reset_time - now).total_seconds()
                return max(seconds, 300.0)

            # Pattern: "resets at Xpm/Xam"
            if len(groups) == 2 and groups[1] and groups[1].lower() in ("am", "pm"):
                hour = int(groups[0])
                meridiem = groups[1].lower()
                if meridiem == "pm" and hour != 12:
                    hour += 12
                elif meridiem == "am" and hour == 12:
                    hour = 0
                now = datetime.now()
                reset_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                if reset_time <= now:
                    reset_time += timedelta(days=1)  # Next day
                seconds = (reset_time - now).total_seconds()
                return max(seconds, 300.0)

        # No pattern matched, return default wait
        return None

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

        # Check for token quota exhaustion FIRST (more specific than rate limit)
        # This is when Claude Max says "tokens exhausted, resets at 9pm"
        if self._matches_any(combined, self.quota_exhaustion_patterns):
            # Try to parse reset time from message
            wait_seconds = self.parse_reset_time(combined)
            if wait_seconds is None:
                wait_seconds = 3600.0  # Default 1 hour if no time found

            result = ClassifiedError(
                category=ErrorCategory.RATE_LIMIT,  # Same category, different handling
                message="Token quota exhausted - waiting until reset",
                error_code=ErrorCode.QUOTA_EXHAUSTED,
                original_error=exception,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=True,
                suggested_wait_seconds=wait_seconds,
            )
            _logger.warning(
                "quota_exhausted",
                component="errors",
                error_code=result.error_code.value,
                exit_code=exit_code,
                retriable=result.retriable,
                suggested_wait_seconds=wait_seconds,
                message=result.message,
            )
            return result

        # Check for rate limiting (most common retriable)
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

        # Check for ENOENT (missing binary/file) FIRST
        # This is often the REAL cause when CLI reports misleading errors like "streaming mode"
        if self._matches_any(combined, self.enoent_patterns):
            result = ClassifiedError(
                category=ErrorCategory.CONFIGURATION,
                message="Missing file or binary (ENOENT) - CLI dependency may be missing or being updated",
                error_code=ErrorCode.BACKEND_NOT_FOUND,
                original_error=exception,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=True,  # Retriable - the file might appear after reinstall/update
                suggested_wait_seconds=30,
            )
            _logger.error(
                "error_classified",
                category=result.category.value,
                error_code=result.error_code.value,
                exit_code=exit_code,
                retriable=result.retriable,
                message=result.message,
                hint="A required file or binary is missing. Check Claude CLI installation.",
            )
            return result

        # Check for CLI mode mismatch (must be before auth check)
        # "streaming mode" error can look like auth failure but is config issue
        if self._matches_any(combined, self.cli_mode_patterns):
            result = ClassifiedError(
                category=ErrorCategory.CONFIGURATION,
                message="CLI mode mismatch - streaming mode incompatible with operation",
                error_code=ErrorCode.CONFIG_CLI_MODE_ERROR,
                original_error=exception,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=False,
            )
            _logger.error(
                "error_classified",
                category=result.category.value,
                error_code=result.error_code.value,
                exit_code=exit_code,
                retriable=result.retriable,
                message=result.message,
                hint="Mozart now defaults to JSON output format. This error should not recur.",
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

        # Check for MCP/Plugin configuration errors (non-retriable)
        if self._matches_any(combined, self.mcp_patterns):
            result = ClassifiedError(
                category=ErrorCategory.CONFIGURATION,
                message="MCP server configuration error - check environment variables",
                error_code=ErrorCode.CONFIG_MCP_ERROR,
                original_error=exception,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=False,
            )
            _logger.error(
                "error_classified",
                category=result.category.value,
                error_code=result.error_code.value,
                exit_code=exit_code,
                retriable=result.retriable,
                message=result.message,
                hint="MCP plugins may need environment variables. Check your shell environment or disable the plugin.",
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

    def classify_execution(
        self,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        exit_signal: int | None = None,
        exit_reason: ExitReason | None = None,
        exception: Exception | None = None,
    ) -> ClassificationResult:
        """Classify execution errors using structured JSON parsing with fallback.

        This is the new multi-error classification method that:
        1. Parses structured JSON errors[] from CLI output (if present)
        2. Classifies each error independently (no short-circuiting)
        3. Analyzes exit code and signal for additional context
        4. Selects root cause using priority-based scoring
        5. Returns all errors with primary/secondary designation

        This method returns ClassificationResult which provides access to
        all detected errors while maintaining backward compatibility through
        the `primary` attribute.

        Args:
            stdout: Standard output from the command (may contain JSON).
            stderr: Standard error from the command.
            exit_code: Process exit code (0 = success), None if killed by signal.
            exit_signal: Signal number if killed by signal.
            exit_reason: Why execution ended (completed, timeout, killed, error).
            exception: Optional exception that was raised.

        Returns:
            ClassificationResult with primary error, secondary errors, and metadata.

        Example:
            ```python
            result = classifier.classify_execution(stdout, stderr, exit_code)

            # Access primary (root cause) error
            if result.primary.category == ErrorCategory.RATE_LIMIT:
                wait_time = result.primary.suggested_wait_seconds

            # Access all errors for debugging
            for error in result.all_errors:
                logger.info(f"{error.error_code.value}: {error.message}")
            ```
        """
        all_errors: list[ClassifiedError] = []
        raw_errors: list[ParsedCliError] = []
        classification_method = "structured"

        # === PHASE 1: Parse Structured JSON ===
        # Pass both stdout and stderr - errors can appear in either stream
        json_errors = try_parse_json_errors(stdout, stderr)
        raw_errors = json_errors

        if json_errors:
            for parsed_error in json_errors:
                classified = classify_single_json_error(
                    parsed_error,
                    exit_code=exit_code,
                    exit_reason=exit_reason,
                )
                all_errors.append(classified)

        # === PHASE 2: Exit Code / Signal Analysis ===
        if exit_signal is not None:
            signal_error = self._classify_signal(
                exit_signal=exit_signal,
                exit_reason=exit_reason,
                exception=exception,
                stdout=stdout,
                stderr=stderr,
            )
            # Only add if not duplicating an existing error code
            if not any(e.error_code == signal_error.error_code for e in all_errors):
                all_errors.append(signal_error)
                if not json_errors:
                    classification_method = "exit_code"

        elif exit_reason == "timeout":
            timeout_error = ClassifiedError(
                category=ErrorCategory.TIMEOUT,
                message="Command timed out",
                error_code=ErrorCode.EXECUTION_TIMEOUT,
                exit_code=exit_code,
                exit_signal=None,
                exit_reason=exit_reason,
                retriable=True,
                suggested_wait_seconds=60.0,
            )
            if not any(e.error_code == ErrorCode.EXECUTION_TIMEOUT for e in all_errors):
                all_errors.append(timeout_error)
                if not json_errors:
                    classification_method = "exit_code"

        # === PHASE 3: Exception Analysis ===
        if exception is not None:
            exc_str = str(exception).lower()
            # Try to classify based on exception message
            if "timeout" in exc_str:
                exc_error = ClassifiedError(
                    category=ErrorCategory.TIMEOUT,
                    message=str(exception),
                    error_code=ErrorCode.EXECUTION_TIMEOUT,
                    original_error=exception,
                    exit_code=exit_code,
                    exit_reason=exit_reason,
                    retriable=True,
                    suggested_wait_seconds=60.0,
                )
            elif "connection" in exc_str or "network" in exc_str:
                exc_error = ClassifiedError(
                    category=ErrorCategory.NETWORK,
                    message=str(exception),
                    error_code=ErrorCode.NETWORK_CONNECTION_FAILED,
                    original_error=exception,
                    exit_code=exit_code,
                    exit_reason=exit_reason,
                    retriable=True,
                    suggested_wait_seconds=30.0,
                )
            else:
                exc_error = ClassifiedError(
                    category=ErrorCategory.TRANSIENT,
                    message=str(exception),
                    error_code=ErrorCode.UNKNOWN,
                    original_error=exception,
                    exit_code=exit_code,
                    exit_reason=exit_reason,
                    retriable=True,
                    suggested_wait_seconds=30.0,
                )
            # Only add if we don't have the same error code already
            if not any(e.error_code == exc_error.error_code for e in all_errors):
                all_errors.append(exc_error)

        # === PHASE 4: Regex Fallback (only if no structured errors) ===
        if not all_errors:
            classification_method = "regex_fallback"
            fallback_error = self.classify(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                exit_signal=exit_signal,
                exit_reason=exit_reason,
                exception=exception,
            )
            all_errors.append(fallback_error)

        # === PHASE 5: Root Cause Selection ===
        root_cause, symptoms, confidence = select_root_cause(all_errors)

        # Log the classification result
        _logger.info(
            "execution_classified",
            method=classification_method,
            primary_code=root_cause.error_code.value,
            error_count=len(all_errors),
            confidence=confidence,
            all_codes=[e.error_code.value for e in all_errors],
        )

        return ClassificationResult(
            primary=root_cause,
            secondary=symptoms,
            raw_errors=raw_errors,
            confidence=confidence,
            classification_method=classification_method,
        )

    @classmethod
    def from_config(cls, rate_limit_patterns: list[str]) -> "ErrorClassifier":
        """Create classifier from config patterns."""
        return cls(rate_limit_patterns=rate_limit_patterns)
