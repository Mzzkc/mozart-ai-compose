"""Recursive Light backend using HTTP API.

Connects Mozart to Recursive Light Framework for TDF-aligned
judgment and confidence scoring via HTTP API bridge.

Phase 3: Language Bridge implementation.
"""

import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from mozart.backends.base import Backend, ExecutionResult
from mozart.core.logging import get_logger

# Module-level logger for Recursive Light backend
_logger = get_logger("backend.recursive_light")


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


class RecursiveLightBackend(Backend):
    """Execute prompts via Recursive Light HTTP API.

    Uses httpx.AsyncClient to communicate with the Recursive Light
    server for TDF-aligned processing with confidence scoring,
    domain activations, and boundary state tracking.

    The RL server provides dual-LLM processing:
    - LLM #1 (unconscious): Confidence assessment and domain activation
    - LLM #2 (conscious): Response generation with accumulated wisdom

    Attributes:
        rl_endpoint: Base URL for the Recursive Light API.
        user_id: Unique identifier for this Mozart instance.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        rl_endpoint: str = "http://localhost:8080",
        user_id: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize Recursive Light backend.

        Args:
            rl_endpoint: Base URL for the Recursive Light API server.
                Defaults to localhost:8080 for local development.
            user_id: Unique identifier for this Mozart instance.
                Generates a UUID if not provided.
            timeout: Request timeout in seconds. Defaults to 30.0.
        """
        self.rl_endpoint = rl_endpoint.rstrip("/")
        self.user_id = user_id or str(uuid.uuid4())
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        return "recursive-light"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client.

        Lazy initialization to avoid creating client before event loop.

        Returns:
            Configured httpx.AsyncClient instance.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.rl_endpoint,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Content-Type": "application/json",
                    "X-Mozart-User-ID": self.user_id,
                },
            )
        return self._client

    async def execute(self, prompt: str) -> ExecutionResult:
        """Execute a prompt through Recursive Light API.

        Sends the prompt to RL's /api/process endpoint and parses
        the response for text output plus RL-specific metadata
        (confidence, domain activations, boundary states, quality).

        Args:
            prompt: The prompt to send to Recursive Light.

        Returns:
            ExecutionResult with output text and RL metadata populated.
            On connection errors, returns a failed result with graceful
            error handling (not raising exceptions).
        """
        start_time = time.monotonic()
        started_at = _utc_now()

        # Log HTTP request details at DEBUG level
        _logger.debug(
            "http_request",
            endpoint=f"{self.rl_endpoint}/api/process",
            user_id=self.user_id,
            timeout=self.timeout,
            prompt_length=len(prompt),
        )

        try:
            client = await self._get_client()

            # Build request payload
            payload = {
                "user_id": self.user_id,
                "message": prompt,
            }

            # POST to RL process endpoint
            response = await client.post("/api/process", json=payload)
            duration = time.monotonic() - start_time

            if response.status_code != 200:
                _logger.error(
                    "api_error_response",
                    duration_seconds=duration,
                    status_code=response.status_code,
                    response_text=response.text[:500] if response.text else None,
                )
                return ExecutionResult(
                    success=False,
                    exit_code=response.status_code,
                    stdout="",
                    stderr=f"RL API error: {response.status_code} - {response.text}",
                    duration_seconds=duration,
                    started_at=started_at,
                    error_type="api_error",
                    error_message=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            # Parse response JSON
            data = response.json()

            # Extract RL-specific metadata with graceful fallbacks
            result = self._parse_rl_response(data, duration, started_at)

            # Log successful response with confidence scores at INFO level
            _logger.info(
                "http_response",
                duration_seconds=duration,
                status_code=response.status_code,
                confidence_score=result.confidence_score,
                response_length=len(result.stdout) if result.stdout else 0,
                has_domain_activations=result.domain_activations is not None,
                has_boundary_states=result.boundary_states is not None,
            )

            return result

        except httpx.ConnectError as e:
            duration = time.monotonic() - start_time
            _logger.warning(
                "connection_error",
                duration_seconds=duration,
                endpoint=self.rl_endpoint,
                error_message=str(e),
            )
            return ExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=f"Connection error: {e}",
                duration_seconds=duration,
                started_at=started_at,
                error_type="connection_error",
                error_message=f"Failed to connect to RL at {self.rl_endpoint}: {e}",
            )

        except httpx.TimeoutException as e:
            duration = time.monotonic() - start_time
            _logger.warning(
                "request_timeout",
                duration_seconds=duration,
                timeout_seconds=self.timeout,
                endpoint=self.rl_endpoint,
            )
            return ExecutionResult(
                success=False,
                exit_code=124,  # Timeout exit code
                stdout="",
                stderr=f"Request timed out: {e}",
                duration_seconds=duration,
                started_at=started_at,
                error_type="timeout",
                error_message=f"Timed out after {self.timeout}s",
            )

        except httpx.HTTPStatusError as e:
            duration = time.monotonic() - start_time
            is_rate_limited = e.response.status_code == 429
            if is_rate_limited:
                _logger.warning(
                    "rate_limit_error",
                    duration_seconds=duration,
                    status_code=e.response.status_code,
                    endpoint=self.rl_endpoint,
                )
            else:
                _logger.error(
                    "http_status_error",
                    duration_seconds=duration,
                    status_code=e.response.status_code,
                    endpoint=self.rl_endpoint,
                    error_message=str(e),
                )
            return ExecutionResult(
                success=False,
                exit_code=e.response.status_code,
                stdout="",
                stderr=f"HTTP error: {e}",
                duration_seconds=duration,
                started_at=started_at,
                error_type="http_error",
                error_message=str(e),
                rate_limited=is_rate_limited,
            )

        except Exception as e:
            duration = time.monotonic() - start_time
            _logger.exception(
                "unexpected_error",
                duration_seconds=duration,
                endpoint=self.rl_endpoint,
                error_message=str(e),
            )
            return ExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                started_at=started_at,
                error_type="exception",
                error_message=f"Unexpected error: {e}",
            )

    def _parse_rl_response(
        self,
        data: dict[str, Any],
        duration: float,
        started_at: datetime,
    ) -> ExecutionResult:
        """Parse Recursive Light API response into ExecutionResult.

        Extracts response text and RL-specific metadata fields with
        graceful handling of missing or malformed data.

        Args:
            data: JSON response from RL API.
            duration: Execution duration in seconds.
            started_at: When execution started.

        Returns:
            ExecutionResult with RL metadata populated.
        """
        # Extract response text (primary output)
        response_text = data.get("response", "")
        if not isinstance(response_text, str):
            response_text = str(response_text)

        # Extract confidence score (0.0-1.0)
        confidence = data.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
                if not 0.0 <= confidence <= 1.0:
                    confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = None

        # Extract domain activations
        domain_activations = self._extract_domain_activations(data)

        # Extract boundary states
        boundary_states = self._extract_boundary_states(data)

        # Extract quality conditions
        quality_conditions = self._extract_quality_conditions(data)

        return ExecutionResult(
            success=True,
            exit_code=0,
            stdout=response_text,
            stderr="",
            duration_seconds=duration,
            started_at=started_at,
            # RL-specific metadata
            confidence_score=confidence,
            domain_activations=domain_activations,
            boundary_states=boundary_states,
            quality_conditions=quality_conditions,
        )

    def _extract_domain_activations(
        self, data: dict[str, Any]
    ) -> dict[str, float] | None:
        """Extract domain activation levels from RL response.

        Looks for domain activations in various possible response structures:
        - data["domains"] = {"COMP": 0.8, "SCI": 0.7, ...}
        - data["domain_activations"] = {...}
        - data["activations"] = {...}

        Args:
            data: JSON response from RL API.

        Returns:
            Dict mapping domain names to activation levels, or None.
        """
        for key in ("domains", "domain_activations", "activations"):
            if key in data and isinstance(data[key], dict):
                result: dict[str, float] = {}
                for domain, value in data[key].items():
                    try:
                        result[str(domain)] = float(value)
                    except (TypeError, ValueError):
                        continue
                if result:
                    return result
        return None

    def _extract_boundary_states(
        self, data: dict[str, Any]
    ) -> dict[str, dict[str, Any]] | None:
        """Extract boundary states from RL response.

        Looks for boundary states in various possible response structures:
        - data["boundaries"] = {"COMP↔SCI": {"permeability": 0.8, ...}}
        - data["boundary_states"] = {...}

        Args:
            data: JSON response from RL API.

        Returns:
            Dict mapping boundary names to state dicts, or None.
        """
        for key in ("boundaries", "boundary_states"):
            if key in data and isinstance(data[key], dict):
                result: dict[str, dict[str, Any]] = {}
                for boundary, state in data[key].items():
                    if isinstance(state, dict):
                        result[str(boundary)] = dict(state)
                if result:
                    return result
        return None

    def _extract_quality_conditions(
        self, data: dict[str, Any]
    ) -> dict[str, float] | None:
        """Extract quality condition assessments from RL response.

        Looks for quality conditions in various possible structures:
        - data["quality"] = {"coherence": 0.9, "relevance": 0.85, ...}
        - data["quality_conditions"] = {...}
        - data["conditions"] = {...}

        Args:
            data: JSON response from RL API.

        Returns:
            Dict mapping condition names to values, or None.
        """
        for key in ("quality", "quality_conditions", "conditions"):
            if key in data and isinstance(data[key], dict):
                result: dict[str, float] = {}
                for condition, value in data[key].items():
                    try:
                        result[str(condition)] = float(value)
                    except (TypeError, ValueError):
                        continue
                if result:
                    return result
        return None

    async def health_check(self) -> bool:
        """Check if Recursive Light server is available and responding.

        Attempts to reach the RL health endpoint (or root) to verify
        connectivity before starting a job.

        Returns:
            True if RL server is healthy and responding, False otherwise.
        """
        try:
            client = await self._get_client()

            # Try health endpoint first, then fall back to root
            for endpoint in ("/health", "/api/health", "/"):
                try:
                    response = await client.get(endpoint)
                    if response.status_code == 200:
                        return True
                except httpx.HTTPStatusError:
                    continue

            return False

        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client connection.

        Should be called when done using the backend to clean up resources.
        """
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "RecursiveLightBackend":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - closes client."""
        await self.close()
