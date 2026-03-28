"""Tests for mozart.utils.credential_scanner — output credential redaction.

TDD: Tests written BEFORE implementation. The scanner detects and redacts
API key patterns in agent output before storage in CheckpointState,
learning store, dashboard, and diagnostics.

Addresses F-003: stdout_tail/stderr_tail stored in 6+ locations without
scanning for credential patterns.
"""

from __future__ import annotations

import pytest


class TestRedactCredentials:
    """Tests for the redact_credentials utility function."""

    def test_no_credentials_unchanged(self) -> None:
        """Text without credentials passes through unchanged."""
        from mozart.utils.credential_scanner import redact_credentials

        text = "Hello, this is normal output with no secrets."
        assert redact_credentials(text) == text

    def test_anthropic_api_key_redacted(self) -> None:
        """Anthropic API keys (sk-ant-...) are redacted."""
        from mozart.utils.credential_scanner import redact_credentials

        text = "Using key sk-ant-api03-abcdefghijklmnop1234567890 for auth"
        result = redact_credentials(text)
        assert "sk-ant-api03-abcdefghijklmnop1234567890" not in result
        assert "[REDACTED_ANTHROPIC_KEY]" in result
        assert "Using key" in result  # surrounding text preserved
        assert "for auth" in result

    def test_openai_api_key_redacted(self) -> None:
        """OpenAI API keys (sk-...) are redacted."""
        from mozart.utils.credential_scanner import redact_credentials

        text = "export OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012mno345"
        result = redact_credentials(text)
        assert "sk-proj-abc123" not in result
        assert "[REDACTED_OPENAI_KEY]" in result

    def test_google_api_key_redacted(self) -> None:
        """Google API keys (AIza...) are redacted."""
        from mozart.utils.credential_scanner import redact_credentials

        text = "GOOGLE_API_KEY=AIzaSyA1B2C3D4E5F6G7H8I9J0KlMnOpQrSt"
        result = redact_credentials(text)
        assert "AIzaSyA1B2C3" not in result
        assert "[REDACTED_GOOGLE_KEY]" in result

    def test_aws_access_key_redacted(self) -> None:
        """AWS access keys (AKIA...) are redacted."""
        from mozart.utils.credential_scanner import redact_credentials

        text = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"
        result = redact_credentials(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "[REDACTED_AWS_KEY]" in result

    def test_generic_bearer_token_redacted(self) -> None:
        """Bearer tokens in Authorization headers are redacted."""
        from mozart.utils.credential_scanner import redact_credentials

        text = 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWI'
        result = redact_credentials(text)
        assert "eyJhbGciOiJ" not in result
        assert "[REDACTED_BEARER_TOKEN]" in result

    def test_multiple_credentials_all_redacted(self) -> None:
        """Multiple different credential types in one text are all caught."""
        from mozart.utils.credential_scanner import redact_credentials

        text = (
            "Keys: sk-ant-api03-abcdefghijk1234567890 and AKIAIOSFODNN7EXAMPLE "
            "and AIzaSyAbcdefghijklmnopqrstuvwxyz12"
        )
        result = redact_credentials(text)
        assert "sk-ant-api03-abcdefghijk" not in result
        assert "AKIAIOSFODNN7" not in result
        assert "AIzaSyAbcdef" not in result

    def test_none_input_returns_none(self) -> None:
        """None input returns None (for optional stdout_tail)."""
        from mozart.utils.credential_scanner import redact_credentials

        assert redact_credentials(None) is None  # type: ignore[arg-type]

    def test_empty_string_returns_empty(self) -> None:
        """Empty string returns empty string."""
        from mozart.utils.credential_scanner import redact_credentials

        assert redact_credentials("") == ""

    @pytest.mark.adversarial
    def test_partial_key_not_false_positive(self) -> None:
        """Short strings that look like key prefixes but aren't full keys."""
        from mozart.utils.credential_scanner import redact_credentials

        # "sk-" alone is too short to be a key
        text = "Use sk-something for the field"
        result = redact_credentials(text)
        # Short sk- prefixes without sufficient length shouldn't be redacted
        assert "sk-something" in result

    @pytest.mark.adversarial
    def test_base64_in_code_not_false_positive(self) -> None:
        """Base64 in code that isn't a bearer token shouldn't be redacted."""
        from mozart.utils.credential_scanner import redact_credentials

        text = 'data = base64.b64encode(b"hello")'
        result = redact_credentials(text)
        # Normal base64 operations shouldn't trigger
        assert result == text

    @pytest.mark.adversarial
    def test_large_text_performance(self) -> None:
        """Scanner handles large outputs (100KB) without excessive time."""
        from mozart.utils.credential_scanner import redact_credentials

        text = "Normal output line\n" * 5000  # ~100KB
        text += "Hidden key: sk-ant-api03-secret123456789012345678\n"
        text += "More output\n" * 5000
        result = redact_credentials(text)
        assert "secret12345" not in result


class TestScanForCredentials:
    """Tests for scan_for_credentials — detection without redaction."""

    def test_detects_anthropic_key(self) -> None:
        """Detects Anthropic API key presence."""
        from mozart.utils.credential_scanner import scan_for_credentials

        text = "Key: sk-ant-api03-abcdefghijk1234567890123456"
        found = scan_for_credentials(text)
        assert len(found) > 0
        assert any("anthropic" in f.lower() for f in found)

    def test_clean_text_empty_result(self) -> None:
        """Clean text returns empty list."""
        from mozart.utils.credential_scanner import scan_for_credentials

        found = scan_for_credentials("No credentials here")
        assert found == []
