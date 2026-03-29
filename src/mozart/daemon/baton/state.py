"""Baton state model — the conductor's execution memory.

These are the data structures the baton uses to track what's happening
during a performance: which sheets are in what state, which instruments
are healthy, how many attempts have been made, what the cost budget
looks like. This is the conductor's working memory — separate from
the persistent CheckpointState, which is the sheet's record of outcomes.

The baton state persists to SQLite (in ``~/.mozart/mozart-state.db``)
for restart recovery. All models support ``to_dict()``/``from_dict()``
serialization for this purpose.

Key types:

- ``BatonSheetStatus`` — extended status enum (pending → ready → dispatched
  → running → completed/failed/skipped, plus waiting and fermata)
- ``SheetExecutionState`` — per-sheet tracking (attempts, costs, status)
- ``InstrumentState`` — per-instrument tracking (rate limits, circuit breaker)
- ``BatonJobState`` — per-job tracking (sheets, cost, pause state)
- ``AttemptContext`` — what the conductor tells the musician for each attempt
- ``CircuitBreakerState`` — three-state circuit breaker (closed/open/half-open)

See: ``docs/plans/2026-03-26-baton-design.md`` — Persistent State section
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mozart.daemon.baton.events import SheetAttemptResult

# =============================================================================
# Enums
# =============================================================================


class BatonSheetStatus(str, Enum):
    """Extended sheet status for the baton's internal tracking.

    The baton needs more states than the existing ``SheetStatus`` enum
    (pending/in_progress/completed/failed/skipped) to express its
    scheduling decisions:

    - ``READY`` — dependencies met, eligible for dispatch
    - ``DISPATCHED`` — sent to a musician, not yet running
    - ``WAITING`` — blocked on a rate-limited instrument
    - ``RETRY_SCHEDULED`` — failed, awaiting retry timer
    - ``FERMATA`` — paused for human/AI escalation decision
    - ``CANCELLED`` — cancelled by user or timeout
    """

    PENDING = "pending"
    READY = "ready"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    WAITING = "waiting"
    RETRY_SCHEDULED = "retry_scheduled"
    FERMATA = "fermata"

    @property
    def is_terminal(self) -> bool:
        """Whether this status represents a final state (no more transitions)."""
        return self in _TERMINAL_BATON_STATUSES


# Frozenset for O(1) terminal status checks — used by BatonSheetStatus.is_terminal
# and by core.py for event handler guards.
_TERMINAL_BATON_STATUSES = frozenset({
    BatonSheetStatus.COMPLETED,
    BatonSheetStatus.FAILED,
    BatonSheetStatus.SKIPPED,
    BatonSheetStatus.CANCELLED,
})

# Statuses that satisfy downstream dependencies
_SATISFIED_BATON_STATUSES = frozenset({
    BatonSheetStatus.COMPLETED,
    BatonSheetStatus.SKIPPED,
})

# Statuses eligible for dispatch
_DISPATCHABLE_BATON_STATUSES = frozenset({
    BatonSheetStatus.PENDING,
    BatonSheetStatus.READY,
})


class AttemptMode(str, Enum):
    """The mode a sheet attempt runs in.

    - ``NORMAL`` — standard execution (first try or retry)
    - ``COMPLETION`` — partial validation passed, trying to complete
    - ``HEALING`` — self-healing after retry exhaustion
    """

    NORMAL = "normal"
    COMPLETION = "completion"
    HEALING = "healing"


class CircuitBreakerState(str, Enum):
    """Three-state circuit breaker for per-instrument health tracking.

    - ``CLOSED`` — healthy, accepting requests
    - ``OPEN`` — unhealthy, rejecting requests, waiting for recovery timer
    - ``HALF_OPEN`` — probing, allowing one request to test recovery
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# =============================================================================
# AttemptContext — what the conductor tells the musician
# =============================================================================


@dataclass
class AttemptContext:
    """Context provided by the conductor to the musician for a single attempt.

    Each dispatch carries this context so the musician knows:
    - Which attempt this is (1 = first try, 2+ = retry)
    - What mode to operate in (normal/completion/healing)
    - Any extra context for non-normal modes
    - Learned patterns from the learning store (instrument-scoped)
    - Previous attempt results (for failure history injection)
    """

    attempt_number: int
    """1-based attempt number. First try = 1, first retry = 2, etc."""

    mode: AttemptMode
    """The execution mode for this attempt."""

    completion_prompt_suffix: str | None = None
    """For completion mode: appended to the prompt to fix partial failures."""

    healing_context: dict[str, Any] | None = None
    """For healing mode: diagnostic context from self-healing analysis."""

    previous_results: list[SheetAttemptResult] | None = None
    """Previous attempt results for failure history injection into prompts."""

    learned_patterns: list[str] | None = None
    """Patterns from the learning store, scoped to this instrument."""


# =============================================================================
# SheetExecutionState — per-sheet tracking
# =============================================================================


@dataclass
class SheetExecutionState:
    """The conductor's per-sheet tracking during a performance.

    This is the baton's working memory for each sheet. It tracks attempt
    counts, status, cost, and timing. The ``attempt_results`` list holds
    the full ``SheetAttemptResult`` for each attempt, enabling the
    conductor to make informed retry/completion/healing decisions.

    Rate-limited attempts do NOT increment ``normal_attempts`` — rate
    limits are tempo changes, not failures.
    """

    sheet_num: int
    """The sheet number within the job (1-based)."""

    instrument_name: str
    """The instrument assigned to this sheet."""

    status: BatonSheetStatus = BatonSheetStatus.PENDING
    """Current scheduling status."""

    normal_attempts: int = 0
    """Number of normal execution attempts (retries). Rate-limited attempts excluded."""

    completion_attempts: int = 0
    """Number of completion mode attempts."""

    healing_attempts: int = 0
    """Number of self-healing attempts."""

    max_retries: int = 3
    """Maximum normal retry attempts (from RetryConfig)."""

    max_completion: int = 5
    """Maximum completion mode attempts (from RetryConfig)."""

    attempt_results: list[SheetAttemptResult] = field(default_factory=list)
    """Full results from every attempt (including rate-limited ones)."""

    next_retry_at: float | None = None
    """Monotonic time when the next retry is scheduled (or None)."""

    total_cost_usd: float = 0.0
    """Cumulative cost across all attempts for this sheet."""

    total_duration_seconds: float = 0.0
    """Cumulative execution duration across all attempts."""

    def record_attempt(self, result: SheetAttemptResult) -> None:
        """Record an attempt result and update tracking state.

        Only non-successful, non-rate-limited attempts increment
        ``normal_attempts``. Successes and rate-limited attempts are
        recorded in ``attempt_results`` for history but don't consume
        retry budget. This matches the baton design spec: retry budget
        tracks failures, not total attempts.
        """
        self.attempt_results.append(result)
        self.total_cost_usd += result.cost_usd
        self.total_duration_seconds += result.duration_seconds

        # Only count failed, non-rate-limited attempts toward retry budget.
        # Successful attempts don't consume retries. Rate-limited attempts
        # are tempo changes, not failures.
        if not result.rate_limited and not result.execution_success:
            self.normal_attempts += 1

    @property
    def can_retry(self) -> bool:
        """Whether this sheet has remaining retry budget."""
        return self.normal_attempts < self.max_retries

    @property
    def can_complete(self) -> bool:
        """Whether this sheet has remaining completion mode budget."""
        return self.completion_attempts < self.max_completion

    @property
    def is_exhausted(self) -> bool:
        """Whether both retry and completion budgets are exhausted."""
        return not self.can_retry and not self.can_complete

    @property
    def total_attempts(self) -> int:
        """Total attempts across all modes."""
        return self.normal_attempts + self.completion_attempts + self.healing_attempts

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for SQLite persistence."""
        return {
            "sheet_num": self.sheet_num,
            "instrument_name": self.instrument_name,
            "status": self.status.value,
            "normal_attempts": self.normal_attempts,
            "completion_attempts": self.completion_attempts,
            "healing_attempts": self.healing_attempts,
            "max_retries": self.max_retries,
            "max_completion": self.max_completion,
            "total_cost_usd": self.total_cost_usd,
            "total_duration_seconds": self.total_duration_seconds,
            "next_retry_at": self.next_retry_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SheetExecutionState:
        """Restore from a serialized dict."""
        state = cls(
            sheet_num=data["sheet_num"],
            instrument_name=data["instrument_name"],
            max_retries=data.get("max_retries", 3),
            max_completion=data.get("max_completion", 5),
        )
        state.status = BatonSheetStatus(data.get("status", "pending"))
        state.normal_attempts = data.get("normal_attempts", 0)
        state.completion_attempts = data.get("completion_attempts", 0)
        state.healing_attempts = data.get("healing_attempts", 0)
        state.total_cost_usd = data.get("total_cost_usd", 0.0)
        state.total_duration_seconds = data.get("total_duration_seconds", 0.0)
        state.next_retry_at = data.get("next_retry_at")
        return state


# =============================================================================
# InstrumentState — per-instrument tracking
# =============================================================================


@dataclass
class InstrumentState:
    """Per-instrument state tracking for rate limits, circuit breakers, and concurrency.

    The baton tracks each instrument's health independently. Rate limits
    on claude-code don't affect gemini-cli. Circuit breaker thresholds
    are per-instrument. Concurrency limits come from the InstrumentProfile.
    """

    name: str
    """Instrument name (matches InstrumentProfile.name)."""

    max_concurrent: int
    """Maximum concurrent sheets on this instrument (from InstrumentProfile)."""

    running_count: int = 0
    """Number of currently running sheets on this instrument."""

    rate_limited: bool = False
    """Whether this instrument is currently rate-limited."""

    rate_limit_expires_at: float | None = None
    """Monotonic time when the rate limit is expected to clear."""

    circuit_breaker: CircuitBreakerState = CircuitBreakerState.CLOSED
    """Current circuit breaker state."""

    consecutive_failures: int = 0
    """Consecutive failures across all jobs for this instrument."""

    circuit_breaker_threshold: int = 5
    """Number of consecutive failures to trip the circuit breaker."""

    circuit_breaker_recovery_at: float | None = None
    """Monotonic time for circuit breaker recovery check."""

    @property
    def is_available(self) -> bool:
        """Whether this instrument can accept new sheets.

        Available when not rate-limited and circuit breaker is not open.
        Half-open allows one probe request through.
        """
        if self.rate_limited:
            return False
        return self.circuit_breaker != CircuitBreakerState.OPEN

    @property
    def at_capacity(self) -> bool:
        """Whether all concurrent slots are in use."""
        return self.running_count >= self.max_concurrent

    def record_success(self) -> None:
        """Record a successful execution on this instrument.

        Resets consecutive failures. If circuit breaker is half-open,
        closes it (the probe succeeded).
        """
        self.consecutive_failures = 0
        if self.circuit_breaker == CircuitBreakerState.HALF_OPEN:
            self.circuit_breaker = CircuitBreakerState.CLOSED

    def record_failure(self) -> None:
        """Record a failed execution on this instrument.

        Increments consecutive failures. If threshold reached, opens
        the circuit breaker. If already half-open, reopens it.
        """
        self.consecutive_failures += 1

        if self.circuit_breaker == CircuitBreakerState.HALF_OPEN:
            # Probe failed — back to open
            self.circuit_breaker = CircuitBreakerState.OPEN
        elif self.consecutive_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker = CircuitBreakerState.OPEN

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for SQLite persistence."""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "rate_limited": self.rate_limited,
            "rate_limit_expires_at": self.rate_limit_expires_at,
            "circuit_breaker": self.circuit_breaker.value,
            "consecutive_failures": self.consecutive_failures,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_recovery_at": self.circuit_breaker_recovery_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstrumentState:
        """Restore from a serialized dict."""
        state = cls(
            name=data["name"],
            max_concurrent=data.get("max_concurrent", 4),
        )
        state.rate_limited = data.get("rate_limited", False)
        state.rate_limit_expires_at = data.get("rate_limit_expires_at")
        state.circuit_breaker = CircuitBreakerState(
            data.get("circuit_breaker", "closed")
        )
        state.consecutive_failures = data.get("consecutive_failures", 0)
        state.circuit_breaker_threshold = data.get("circuit_breaker_threshold", 5)
        state.circuit_breaker_recovery_at = data.get("circuit_breaker_recovery_at")
        return state


# =============================================================================
# BatonJobState — per-job tracking
# =============================================================================


@dataclass
class BatonJobState:
    """The baton's per-job tracking during a performance.

    Contains all sheet states for a job, plus job-level flags
    (paused, pacing, cost tracking).
    """

    job_id: str
    """The unique job identifier."""

    total_sheets: int
    """Total number of sheets in this job."""

    paused: bool = False
    """Whether dispatching is paused for this job."""

    pacing_active: bool = False
    """Whether inter-sheet pacing delay is currently active."""

    sheets: dict[int, SheetExecutionState] = field(default_factory=dict)
    """Map of sheet_num → SheetExecutionState."""

    def register_sheet(self, sheet: SheetExecutionState) -> None:
        """Register a sheet's execution state with this job."""
        self.sheets[sheet.sheet_num] = sheet

    def get_sheet(self, sheet_num: int) -> SheetExecutionState | None:
        """Get a sheet's execution state, or None if not registered."""
        return self.sheets.get(sheet_num)

    @property
    def total_cost_usd(self) -> float:
        """Total cost across all sheets in this job."""
        return sum(s.total_cost_usd for s in self.sheets.values())

    @property
    def completed_count(self) -> int:
        """Number of sheets in COMPLETED status."""
        return sum(
            1 for s in self.sheets.values()
            if s.status == BatonSheetStatus.COMPLETED
        )

    @property
    def terminal_count(self) -> int:
        """Number of sheets in a terminal status (completed, failed, skipped)."""
        return sum(1 for s in self.sheets.values() if s.status.is_terminal)

    @property
    def is_complete(self) -> bool:
        """Whether all registered sheets have reached a terminal status."""
        if not self.sheets:
            return False
        return all(s.status.is_terminal for s in self.sheets.values())

    @property
    def running_sheets(self) -> list[SheetExecutionState]:
        """Sheets currently in RUNNING status."""
        return [
            s for s in self.sheets.values()
            if s.status == BatonSheetStatus.RUNNING
        ]

    @property
    def has_any_failed(self) -> bool:
        """Whether any sheet has reached FAILED status."""
        return any(
            s.status == BatonSheetStatus.FAILED for s in self.sheets.values()
        )
