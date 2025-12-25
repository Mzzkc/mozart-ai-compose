"""Anthropic API backend using the official SDK.

Direct API access for Claude models without needing the CLI installed.
Provides rate limit detection, token tracking, and graceful error handling.
"""

import os
import re
import time

import anthropic

from mozart.backends.base import Backend, ExecutionResult
from mozart.core.config import BackendConfig


class AnthropicApiBackend(Backend):
    """Run prompts directly via the Anthropic API.

    Uses the official anthropic SDK for direct API access.
    Supports all Claude models available through the API.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        timeout_seconds: float = 300.0,  # 5 minute default for API
    ):
        """Initialize API backend.

        Args:
            model: Model ID to use (e.g., claude-sonnet-4-20250514)
            api_key_env: Environment variable containing API key
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature (0.0-1.0)
            timeout_seconds: Maximum time for API request
        """
        self.model = model
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

        # Get API key from environment
        self._api_key = os.environ.get(api_key_env)

        # Create async client (lazily initialized in execute)
        self._client: anthropic.AsyncAnthropic | None = None

    @classmethod
    def from_config(cls, config: BackendConfig) -> "AnthropicApiBackend":
        """Create backend from configuration."""
        return cls(
            model=config.model,
            api_key_env=config.api_key_env,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
        )

    @property
    def name(self) -> str:
        return "anthropic-api"

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Get or create the async Anthropic client."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    f"API key not found in environment variable: {self.api_key_env}"
                )
            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key,
                timeout=self.timeout_seconds,
            )
        return self._client

    async def execute(self, prompt: str) -> ExecutionResult:
        """Execute a prompt via the Anthropic API.

        Args:
            prompt: The prompt to send to Claude

        Returns:
            ExecutionResult with API response and metadata
        """
        start_time = time.monotonic()

        try:
            client = self._get_client()

            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            duration = time.monotonic() - start_time

            # Extract response text
            content_blocks = response.content
            response_text = ""
            for block in content_blocks:
                if hasattr(block, "text"):
                    response_text += block.text

            # Calculate tokens used
            tokens_used = (
                response.usage.input_tokens + response.usage.output_tokens
                if response.usage
                else None
            )

            return ExecutionResult(
                success=True,
                exit_code=0,
                stdout=response_text,
                stderr="",
                duration_seconds=duration,
                model=self.model,
                tokens_used=tokens_used,
            )

        except anthropic.RateLimitError as e:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                exit_code=429,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                rate_limited=True,
                error_type="rate_limit",
                error_message=f"Rate limited: {e}",
                model=self.model,
            )

        except anthropic.AuthenticationError as e:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                exit_code=401,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                error_type="authentication",
                error_message=f"Authentication failed: {e}",
                model=self.model,
            )

        except anthropic.BadRequestError as e:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                exit_code=400,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                error_type="bad_request",
                error_message=f"Bad request: {e}",
                model=self.model,
            )

        except anthropic.APITimeoutError as e:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                exit_code=408,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                error_type="timeout",
                error_message=f"API timeout after {self.timeout_seconds}s: {e}",
                model=self.model,
            )

        except anthropic.APIConnectionError as e:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                exit_code=503,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                error_type="connection",
                error_message=f"Connection error: {e}",
                model=self.model,
            )

        except anthropic.APIStatusError as e:
            duration = time.monotonic() - start_time
            # Check if this is a rate limit error by status code or message
            rate_limited = self._detect_rate_limit(str(e))
            return ExecutionResult(
                success=False,
                exit_code=e.status_code if hasattr(e, "status_code") else 500,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                rate_limited=rate_limited,
                error_type="rate_limit" if rate_limited else "api_error",
                error_message=str(e),
                model=self.model,
            )

        except RuntimeError as e:
            # API key not found
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                error_type="configuration",
                error_message=str(e),
                model=self.model,
            )

        except Exception as e:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                error_type="exception",
                error_message=f"Unexpected error: {e}",
                model=self.model,
            )

    def _detect_rate_limit(self, message: str) -> bool:
        """Check output for rate limit indicators."""
        patterns = [
            r"rate.?limit",
            r"usage.?limit",
            r"quota",
            r"too many requests",
            r"429",
            r"capacity",
            r"try again later",
        ]
        return any(re.search(p, message.lower(), re.IGNORECASE) for p in patterns)

    async def health_check(self) -> bool:
        """Check if the API is available and authenticated.

        Uses a minimal prompt to verify connectivity.
        """
        if not self._api_key:
            return False

        try:
            client = self._get_client()
            # Minimal prompt to verify API access
            response = await client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Reply with only: ok"}],
            )
            # Check we got a response
            return len(response.content) > 0
        except Exception:
            return False

    async def close(self) -> None:
        """Close the async client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
