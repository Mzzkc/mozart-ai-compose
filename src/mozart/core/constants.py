"""Global constants for Mozart.

Centralizes magic numbers used throughout the codebase,
making them discoverable, consistent, and easy to modify.
"""

# =============================================================================
# Text Truncation Limits (characters)
# =============================================================================

TRUNCATE_STDOUT_TAIL_CHARS = 500
"""Default truncation limit for stdout/stderr tails in error display and state."""

TRUNCATE_PROMPT_PREVIEW_CHARS = 500
"""Maximum characters shown for prompt previews in escalation context."""

TRUNCATE_ERROR_MESSAGE_CHARS = 200
"""Maximum characters for error message summaries."""

# =============================================================================
# Healing / Diagnostic Context Limits
# =============================================================================

HEALING_CONTEXT_TAIL_CHARS = 10000
"""Maximum stdout/stderr characters captured for self-healing diagnostic context."""

# =============================================================================
# Process Execution Defaults
# =============================================================================

PROCESS_DEFAULT_TIMEOUT_SECONDS = 300.0
"""Default timeout for subprocess execution (5 minutes)."""

# =============================================================================
# Duration Formatting Constants
# =============================================================================

SECONDS_PER_MINUTE = 60
"""Seconds in one minute, for duration formatting."""

SECONDS_PER_HOUR = 3600
"""Seconds in one hour, for duration formatting."""

# =============================================================================
# Dashboard / Rate Limiting
# =============================================================================

RATE_LIMIT_REQUESTS_PER_MINUTE = 60
"""Maximum requests per minute for API rate limiting."""

RATE_LIMIT_REQUESTS_PER_HOUR = 1000
"""Maximum requests per hour for API rate limiting."""

RATE_LIMIT_BURST_LIMIT = 10
"""Maximum burst requests in a short window."""

SSE_QUEUE_TIMEOUT_SECONDS = 30.0
"""Timeout for SSE event queue reads."""

LOG_STREAM_MAX_TAIL_LINES = 1000
"""Maximum lines returned by log streaming endpoint."""
