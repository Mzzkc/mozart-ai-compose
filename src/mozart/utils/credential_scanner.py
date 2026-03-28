"""Credential pattern scanner for agent output redaction.

Scans stdout_tail and stderr_tail for API key patterns before storage.
This prevents credentials from propagating to CheckpointState, learning
store, dashboard, diagnostics, and MCP resources.

Addresses F-003: No output scanning for credential patterns.

Key patterns detected:
- Anthropic API keys: sk-ant-api* (30+ chars after prefix)
- OpenAI API keys: sk-proj-*, sk-[a-zA-Z0-9]{20,} (not short sk-* strings)
- Google API keys: AIzaSy* (35+ chars)
- AWS access keys: AKIA* (20 chars total)
- Bearer tokens: Authorization: Bearer <token>

The scanner is deliberately conservative — better to miss a non-credential
than to redact legitimate output. False positives degrade output quality;
false negatives are caught by log review.
"""

from __future__ import annotations

import re
from typing import Any

# Compiled patterns for credential detection.
# Each tuple: (compiled regex, replacement label, description)
_CREDENTIAL_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # Anthropic API keys: sk-ant-api03-<base64ish, 30+ chars>
    (
        re.compile(r"sk-ant-api\S{10,}"),
        "[REDACTED_ANTHROPIC_KEY]",
        "Anthropic API key",
    ),
    # OpenAI API keys: sk-proj-<alnum, 20+ chars> or sk-<alnum, 40+ chars>
    # Must be long enough to avoid false positives on short "sk-" prefixes
    (
        re.compile(r"sk-proj-[a-zA-Z0-9_-]{20,}"),
        "[REDACTED_OPENAI_KEY]",
        "OpenAI project API key",
    ),
    (
        re.compile(r"sk-[a-zA-Z0-9]{40,}"),
        "[REDACTED_OPENAI_KEY]",
        "OpenAI API key",
    ),
    # Google API keys: AIzaSy<30+ chars of base64ish>
    (
        re.compile(r"AIzaSy[a-zA-Z0-9_-]{28,}"),
        "[REDACTED_GOOGLE_KEY]",
        "Google API key",
    ),
    # AWS access keys: AKIA<16 alphanumeric chars>
    (
        re.compile(r"AKIA[A-Z0-9]{16}"),
        "[REDACTED_AWS_KEY]",
        "AWS access key ID",
    ),
    # Bearer tokens in Authorization headers
    (
        re.compile(r"(?<=Bearer\s)[a-zA-Z0-9._-]{20,}"),
        "[REDACTED_BEARER_TOKEN]",
        "Bearer token",
    ),
]


def redact_credentials(text: Any) -> Any:
    """Redact credential patterns from text.

    Scans the input for known API key patterns and replaces them with
    descriptive placeholders. Designed to be called on stdout_tail and
    stderr_tail before storage.

    Args:
        text: String to scan. None is passed through unchanged.

    Returns:
        The text with credential patterns replaced by [REDACTED_*] labels.
        None if input was None.
    """
    if text is None:
        return None
    if not isinstance(text, str) or not text:
        return text

    result = text
    for pattern, replacement, _desc in _CREDENTIAL_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def scan_for_credentials(text: str) -> list[str]:
    """Detect credential patterns without redacting.

    Returns a list of human-readable descriptions of detected credential
    types. Useful for logging warnings when credentials are found.

    Args:
        text: String to scan.

    Returns:
        List of credential type descriptions found (empty if clean).
    """
    if not text:
        return []

    found: list[str] = []
    for pattern, _replacement, description in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            found.append(description)
    return found
