"""Tests for Mozart execution backends.

Tests cover:
- AnthropicApiBackend: API client, error handling, rate limit detection
- ClaudeCliBackend: Basic structure (CLI tests need integration tests)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import anthropic
from mozart.backends.anthropic_api import AnthropicApiBackend
from mozart.backends.claude_cli import ClaudeCliBackend
from mozart.core.config import BackendConfig


# ============================================================================
# AnthropicApiBackend Tests
# ============================================================================


class TestAnthropicApiBackendInit:
    """Test AnthropicApiBackend initialization."""

    def test_init_defaults(self) -> None:
        """Test default initialization values."""
        backend = AnthropicApiBackend()
        assert backend.model == "claude-sonnet-4-20250514"
        assert backend.api_key_env == "ANTHROPIC_API_KEY"
        assert backend.max_tokens == 8192
        assert backend.temperature == 0.7
        assert backend.timeout_seconds == 300.0
        assert backend.name == "anthropic-api"

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        backend = AnthropicApiBackend(
            model="claude-opus-4-20250514",
            api_key_env="MY_API_KEY",
            max_tokens=4096,
            temperature=0.5,
            timeout_seconds=60.0,
        )
        assert backend.model == "claude-opus-4-20250514"
        assert backend.api_key_env == "MY_API_KEY"
        assert backend.max_tokens == 4096
        assert backend.temperature == 0.5
        assert backend.timeout_seconds == 60.0

    def test_from_config(self) -> None:
        """Test creating backend from config."""
        config = BackendConfig(
            type="anthropic_api",
            model="claude-opus-4-20250514",
            api_key_env="CUSTOM_KEY",
            max_tokens=2048,
            temperature=0.3,
        )
        backend = AnthropicApiBackend.from_config(config)
        assert backend.model == "claude-opus-4-20250514"
        assert backend.api_key_env == "CUSTOM_KEY"
        assert backend.max_tokens == 2048
        assert backend.temperature == 0.3


class TestAnthropicApiBackendExecute:
    """Test AnthropicApiBackend execute method."""

    @pytest.fixture
    def backend_with_key(self, monkeypatch: pytest.MonkeyPatch) -> AnthropicApiBackend:
        """Create backend with API key set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        return AnthropicApiBackend()

    @pytest.mark.asyncio
    async def test_execute_success(self, backend_with_key: AnthropicApiBackend) -> None:
        """Test successful API execution."""
        # Mock the response
        mock_content_block = MagicMock()
        mock_content_block.text = "Hello, world!"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_response = MagicMock()
        mock_response.content = [mock_content_block]
        mock_response.usage = mock_usage

        # Mock the client
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.object(backend_with_key, "_get_client", return_value=mock_client):
            result = await backend_with_key.execute("Say hello")

        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "Hello, world!"
        assert result.stderr == ""
        assert result.tokens_used == 15
        assert result.model == "claude-sonnet-4-20250514"
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_execute_rate_limit_error(
        self, backend_with_key: AnthropicApiBackend
    ) -> None:
        """Test rate limit error handling."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.RateLimitError(
                message="Rate limit exceeded",
                response=mock_response,
                body=None,
            )
        )

        with patch.object(backend_with_key, "_get_client", return_value=mock_client):
            result = await backend_with_key.execute("Test prompt")

        assert result.success is False
        assert result.exit_code == 429
        assert result.rate_limited is True
        assert result.error_type == "rate_limit"
        assert "rate limit" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_authentication_error(
        self, backend_with_key: AnthropicApiBackend
    ) -> None:
        """Test authentication error handling."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.AuthenticationError(
                message="Invalid API key",
                response=mock_response,
                body=None,
            )
        )

        with patch.object(backend_with_key, "_get_client", return_value=mock_client):
            result = await backend_with_key.execute("Test prompt")

        assert result.success is False
        assert result.exit_code == 401
        assert result.error_type == "authentication"
        assert "authentication" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_bad_request_error(
        self, backend_with_key: AnthropicApiBackend
    ) -> None:
        """Test bad request error handling."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.BadRequestError(
                message="Invalid request",
                response=mock_response,
                body=None,
            )
        )

        with patch.object(backend_with_key, "_get_client", return_value=mock_client):
            result = await backend_with_key.execute("Test prompt")

        assert result.success is False
        assert result.exit_code == 400
        assert result.error_type == "bad_request"

    @pytest.mark.asyncio
    async def test_execute_timeout_error(
        self, backend_with_key: AnthropicApiBackend
    ) -> None:
        """Test timeout error handling."""
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.APITimeoutError(request=MagicMock())
        )

        with patch.object(backend_with_key, "_get_client", return_value=mock_client):
            result = await backend_with_key.execute("Test prompt")

        assert result.success is False
        assert result.exit_code == 408
        assert result.error_type == "timeout"

    @pytest.mark.asyncio
    async def test_execute_connection_error(
        self, backend_with_key: AnthropicApiBackend
    ) -> None:
        """Test connection error handling."""
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.APIConnectionError(request=MagicMock())
        )

        with patch.object(backend_with_key, "_get_client", return_value=mock_client):
            result = await backend_with_key.execute("Test prompt")

        assert result.success is False
        assert result.exit_code == 503
        assert result.error_type == "connection"

    @pytest.mark.asyncio
    async def test_execute_no_api_key(self) -> None:
        """Test error when API key is not set."""
        backend = AnthropicApiBackend(api_key_env="NONEXISTENT_KEY")
        result = await backend.execute("Test prompt")

        assert result.success is False
        assert result.exit_code == 1
        assert result.error_type == "configuration"
        assert "NONEXISTENT_KEY" in result.error_message


class TestAnthropicApiBackendRateLimitDetection:
    """Test rate limit detection patterns."""

    @pytest.fixture
    def backend(self) -> AnthropicApiBackend:
        """Create backend for testing."""
        return AnthropicApiBackend()

    def test_detect_rate_limit_patterns(self, backend: AnthropicApiBackend) -> None:
        """Test various rate limit patterns are detected."""
        patterns = [
            "rate limit exceeded",
            "Rate-Limit: exceeded",
            "usage limit reached",
            "quota exceeded",
            "too many requests",
            "error 429",
            "capacity exceeded",
            "try again later",
        ]
        for pattern in patterns:
            assert backend._detect_rate_limit(pattern) is True, f"Failed for: {pattern}"

    def test_no_rate_limit_detected(self, backend: AnthropicApiBackend) -> None:
        """Test that normal messages don't trigger rate limit detection."""
        normal_messages = [
            "Success",
            "Hello, world!",
            "Error: invalid input",
            "Connection failed",
        ]
        for message in normal_messages:
            assert backend._detect_rate_limit(message) is False, f"Failed for: {message}"


class TestAnthropicApiBackendHealthCheck:
    """Test AnthropicApiBackend health check."""

    @pytest.mark.asyncio
    async def test_health_check_no_api_key(self) -> None:
        """Test health check fails without API key."""
        backend = AnthropicApiBackend(api_key_env="NONEXISTENT_KEY")
        result = await backend.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test successful health check."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        backend = AnthropicApiBackend()

        mock_content_block = MagicMock()
        mock_content_block.text = "ok"

        mock_response = MagicMock()
        mock_response.content = [mock_content_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.object(backend, "_get_client", return_value=mock_client):
            result = await backend.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test health check failure on API error."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        backend = AnthropicApiBackend()

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))

        with patch.object(backend, "_get_client", return_value=mock_client):
            result = await backend.health_check()

        assert result is False


# ============================================================================
# ClaudeCliBackend Tests
# ============================================================================


class TestClaudeCliBackendInit:
    """Test ClaudeCliBackend initialization."""

    def test_init_defaults(self) -> None:
        """Test default initialization values."""
        backend = ClaudeCliBackend()
        assert backend.skip_permissions is True
        assert backend.output_format is None
        assert backend.working_directory is None
        assert backend.timeout_seconds == 1800.0
        assert backend.name == "claude-cli"

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        from pathlib import Path

        backend = ClaudeCliBackend(
            skip_permissions=False,
            output_format="json",
            working_directory=Path("/tmp"),
            timeout_seconds=60.0,
        )
        assert backend.skip_permissions is False
        assert backend.output_format == "json"
        assert backend.working_directory == Path("/tmp")
        assert backend.timeout_seconds == 60.0

    def test_from_config(self) -> None:
        """Test creating backend from config."""
        from pathlib import Path

        config = BackendConfig(
            type="claude_cli",
            skip_permissions=False,
            output_format="json",
            working_directory=Path("/tmp"),
            timeout_seconds=120.0,
        )
        backend = ClaudeCliBackend.from_config(config)
        assert backend.skip_permissions is False
        assert backend.output_format == "json"
        assert backend.working_directory == Path("/tmp")
        assert backend.timeout_seconds == 120.0


class TestClaudeCliBackendRateLimitDetection:
    """Test ClaudeCliBackend rate limit detection."""

    @pytest.fixture
    def backend(self) -> ClaudeCliBackend:
        """Create backend for testing."""
        return ClaudeCliBackend()

    def test_detect_rate_limit_patterns(self, backend: ClaudeCliBackend) -> None:
        """Test various rate limit patterns are detected."""
        patterns = [
            "rate limit exceeded",
            "usage limit reached",
            "quota exceeded",
            "too many requests",
            "429",
            "capacity exceeded",
            "try again later",
        ]
        for pattern in patterns:
            assert (
                backend._detect_rate_limit(pattern, "") is True
            ), f"Failed for: {pattern}"

    def test_no_rate_limit_detected(self, backend: ClaudeCliBackend) -> None:
        """Test that normal messages don't trigger rate limit detection."""
        normal_messages = [
            "Success",
            "Hello, world!",
            "Error: invalid input",
        ]
        for message in normal_messages:
            assert (
                backend._detect_rate_limit(message, "") is False
            ), f"Failed for: {message}"
