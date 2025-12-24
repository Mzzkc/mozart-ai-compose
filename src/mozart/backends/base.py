"""Abstract base for Claude execution backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExecutionResult:
    """Result of executing a prompt through a backend.

    Captures all relevant output and metadata for validation and debugging.
    """

    success: bool
    """Whether the execution completed without error (exit code 0)."""

    exit_code: int
    """Process exit code or HTTP status code."""

    stdout: str
    """Standard output from the command."""

    stderr: str
    """Standard error from the command."""

    duration_seconds: float
    """Execution duration in seconds."""

    started_at: datetime = field(default_factory=datetime.utcnow)
    """When execution started."""

    # Error classification hints
    rate_limited: bool = False
    """Whether rate limiting was detected."""

    error_type: str | None = None
    """Classified error type if failed."""

    error_message: str | None = None
    """Human-readable error message."""

    # Metadata
    model: str | None = None
    """Model used for execution."""

    tokens_used: int | None = None
    """Tokens consumed (API backend only)."""

    # Recursive Light metadata (Phase 3 Language Bridge)
    confidence_score: float | None = None
    """RL confidence score (0.0-1.0) from dual-LLM processing."""

    domain_activations: dict[str, float] | None = None
    """RL domain activation levels (COMP, SCI, CULT, EXP)."""

    boundary_states: dict[str, dict[str, Any]] | None = None
    """RL boundary states between domains (permeability, resonance)."""

    quality_conditions: dict[str, float] | None = None
    """RL quality condition assessments (coherence, relevance, etc.)."""

    @property
    def output(self) -> str:
        """Combined stdout and stderr."""
        return f"{self.stdout}\n{self.stderr}".strip()


class Backend(ABC):
    """Abstract base class for Claude execution backends.

    Backends handle the actual execution of prompts through Claude,
    whether via CLI subprocess or direct API calls.
    """

    @abstractmethod
    async def execute(self, prompt: str) -> ExecutionResult:
        """Execute a prompt and return the result.

        Args:
            prompt: The prompt to send to Claude

        Returns:
            ExecutionResult with output and metadata
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is available and working.

        Used to verify connectivity before starting a job,
        and to check if rate limits have lifted.

        Returns:
            True if backend is ready, False otherwise
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...
