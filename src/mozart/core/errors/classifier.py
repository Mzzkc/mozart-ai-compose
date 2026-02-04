"""ErrorClassifier implementation for pattern-based error classification.

This is the main classifier class that analyzes stdout, stderr, exit codes,
and signals to produce ClassifiedError instances with appropriate retry behavior.

This module provides:
- _emit_deprecation_warning(): Helper for deprecated API warnings
- ErrorClassifier: Main classifier class with pattern matching and JSON parsing
"""

from __future__ import annotations

import re
import signal
import warnings
from mozart.core.logging import get_logger

from .codes import ErrorCategory, ErrorCode, ExitReason
from .models import ClassificationResult, ClassifiedError, ParsedCliError
from .parsers import classify_single_json_error, select_root_cause, try_parse_json_errors
from .signals import FATAL_SIGNALS, RETRIABLE_SIGNALS, get_signal_name

# Module-level logger for error classification
_logger = get_logger("errors")


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
