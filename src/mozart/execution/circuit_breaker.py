"""Circuit breaker pattern for resilient execution.

Implements the circuit breaker pattern to prevent cascading failures
by temporarily blocking calls after repeated failures.

The circuit breaker has three states:
- CLOSED: Normal operation, requests flow through
- OPEN: Blocking requests after too many failures
- HALF_OPEN: Testing if the service has recovered

State transitions:
- CLOSED -> OPEN: When failure_count >= failure_threshold
- OPEN -> HALF_OPEN: After recovery_timeout has elapsed
- HALF_OPEN -> CLOSED: On success in half-open state
- HALF_OPEN -> OPEN: On failure in half-open state

Example usage:
    from mozart.execution.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=300.0)

    if breaker.can_execute():
        try:
            result = await backend.execute(prompt)
            if result.success:
                breaker.record_success()
            else:
                breaker.record_failure()
        except Exception:
            breaker.record_failure()
            raise
    else:
        # Circuit is open - wait or use fallback
        wait_time = breaker.time_until_retry()
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any

from mozart.core.logging import get_logger

# Module-level logger for circuit breaker events
_logger = get_logger("circuit_breaker")


class CircuitState(str, Enum):
    """State of the circuit breaker.

    The circuit breaker transitions between these states based on
    success/failure patterns:

    - CLOSED: Normal operation. Failures are tracked but requests are allowed.
    - OPEN: Blocking mode. Requests are rejected until recovery_timeout elapses.
    - HALF_OPEN: Testing mode. A single request is allowed to test recovery.
    """

    CLOSED = "closed"
    """Normal operation - requests are allowed and failures are tracked."""

    OPEN = "open"
    """Blocking calls - requests are rejected, waiting for recovery timeout."""

    HALF_OPEN = "half_open"
    """Testing recovery - one request is allowed to test if service recovered."""


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring.

    Provides visibility into the circuit breaker's behavior for
    observability and debugging, including cost tracking.
    """

    total_successes: int = 0
    """Total number of successful operations recorded."""

    total_failures: int = 0
    """Total number of failed operations recorded."""

    times_opened: int = 0
    """Number of times the circuit has transitioned to OPEN state."""

    times_half_opened: int = 0
    """Number of times the circuit has transitioned to HALF_OPEN state."""

    times_closed: int = 0
    """Number of times the circuit has transitioned to CLOSED from another state."""

    last_failure_at: float | None = None
    """Timestamp of the most recent failure (monotonic time)."""

    last_state_change_at: float | None = None
    """Timestamp of the most recent state transition."""

    consecutive_failures: int = 0
    """Current count of consecutive failures (resets on success)."""

    # Cost tracking (v4 evolution: Cost Circuit Breaker)
    total_input_tokens: int = 0
    """Total input tokens consumed across all executions."""

    total_output_tokens: int = 0
    """Total output tokens consumed across all executions."""

    total_estimated_cost: float = 0.0
    """Total estimated cost in USD across all executions."""

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for logging/serialization.

        Returns:
            Dictionary representation of all statistics.
        """
        return {
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "times_opened": self.times_opened,
            "times_half_opened": self.times_half_opened,
            "times_closed": self.times_closed,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_at": self.last_failure_at,
            "last_state_change_at": self.last_state_change_at,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_estimated_cost": self.total_estimated_cost,
        }


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures.

    The circuit breaker monitors execution success/failure and automatically
    blocks further requests when a failure threshold is exceeded. This prevents
    overwhelming a failing service and gives it time to recover.

    Thread-safe: All state modifications are protected by a lock.

    Attributes:
        failure_threshold: Number of consecutive failures before opening circuit.
        recovery_timeout: Seconds to wait before testing recovery (OPEN -> HALF_OPEN).
        state: Current circuit state.
        stats: Statistics about circuit breaker behavior.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 300.0,
        name: str = "default",
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening
                the circuit. Default is 5.
            recovery_timeout: Seconds to wait in OPEN state before transitioning
                to HALF_OPEN to test recovery. Default is 300 (5 minutes).
            name: Name for this circuit breaker (used in logging).
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")

        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._name = name

        # State (protected by lock)
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._stats = CircuitBreakerStats()

        # Thread safety
        self._lock = Lock()

        _logger.debug(
            "circuit_breaker.initialized",
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

    @property
    def failure_threshold(self) -> int:
        """Number of consecutive failures before opening circuit."""
        return self._failure_threshold

    @property
    def recovery_timeout(self) -> float:
        """Seconds to wait before testing recovery."""
        return self._recovery_timeout

    @property
    def name(self) -> str:
        """Name of this circuit breaker."""
        return self._name

    def get_state(self) -> CircuitState:
        """Get the current circuit state.

        This method handles automatic state transitions:
        - If OPEN and recovery_timeout has elapsed, transitions to HALF_OPEN.

        Returns:
            Current CircuitState.
        """
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def _maybe_transition_to_half_open(self) -> None:
        """Check if we should transition from OPEN to HALF_OPEN.

        Should be called while holding the lock.
        """
        if self._state != CircuitState.OPEN:
            return

        if self._last_failure_time is None:
            return

        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= self._recovery_timeout:
            self._set_state(CircuitState.HALF_OPEN)
            _logger.info(
                "circuit_breaker.state_changed",
                name=self._name,
                from_state=CircuitState.OPEN.value,
                to_state=CircuitState.HALF_OPEN.value,
                reason="recovery_timeout_elapsed",
                elapsed_seconds=round(elapsed, 2),
            )

    def _set_state(self, new_state: CircuitState) -> None:
        """Set the circuit state and update statistics.

        Should be called while holding the lock.

        Args:
            new_state: The state to transition to.
        """
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        now = time.monotonic()
        self._stats.last_state_change_at = now

        # Update transition counters
        if new_state == CircuitState.OPEN:
            self._stats.times_opened += 1
        elif new_state == CircuitState.HALF_OPEN:
            self._stats.times_half_opened += 1
        elif new_state == CircuitState.CLOSED and old_state != CircuitState.CLOSED:
            self._stats.times_closed += 1

    def can_execute(self) -> bool:
        """Check if a request can be executed.

        Returns True if:
        - Circuit is CLOSED (normal operation)
        - Circuit is HALF_OPEN (testing recovery)
        - Circuit is OPEN but recovery_timeout has elapsed (transitions to HALF_OPEN)

        Returns False if:
        - Circuit is OPEN and recovery_timeout hasn't elapsed

        Returns:
            True if the request should be allowed, False if it should be blocked.
        """
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        """Record a successful operation.

        Effects by state:
        - CLOSED: Resets consecutive failure count
        - HALF_OPEN: Transitions to CLOSED (recovery confirmed)
        - OPEN: No effect (shouldn't happen - request blocked)
        """
        with self._lock:
            self._stats.total_successes += 1
            self._stats.consecutive_failures = 0
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                # Recovery confirmed - close the circuit
                _logger.info(
                    "circuit_breaker.state_changed",
                    name=self._name,
                    from_state=CircuitState.HALF_OPEN.value,
                    to_state=CircuitState.CLOSED.value,
                    reason="recovery_confirmed",
                )
                self._set_state(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                _logger.debug(
                    "circuit_breaker.success_recorded",
                    name=self._name,
                    state=self._state.value,
                )

    def record_failure(self) -> None:
        """Record a failed operation.

        Effects by state:
        - CLOSED: Increments failure count, may transition to OPEN
        - HALF_OPEN: Transitions to OPEN (recovery failed)
        - OPEN: No effect (shouldn't happen - request blocked)
        """
        with self._lock:
            now = time.monotonic()
            self._stats.total_failures += 1
            self._stats.consecutive_failures += 1
            self._stats.last_failure_at = now
            self._failure_count += 1
            self._last_failure_time = now

            if self._state == CircuitState.HALF_OPEN:
                # Recovery test failed - reopen the circuit
                _logger.info(
                    "circuit_breaker.state_changed",
                    name=self._name,
                    from_state=CircuitState.HALF_OPEN.value,
                    to_state=CircuitState.OPEN.value,
                    reason="recovery_test_failed",
                )
                self._set_state(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    # Threshold exceeded - open the circuit
                    _logger.info(
                        "circuit_breaker.state_changed",
                        name=self._name,
                        from_state=CircuitState.CLOSED.value,
                        to_state=CircuitState.OPEN.value,
                        reason="failure_threshold_exceeded",
                        failure_count=self._failure_count,
                        failure_threshold=self._failure_threshold,
                    )
                    self._set_state(CircuitState.OPEN)
                else:
                    _logger.debug(
                        "circuit_breaker.failure_recorded",
                        name=self._name,
                        state=self._state.value,
                        failure_count=self._failure_count,
                        failure_threshold=self._failure_threshold,
                    )

    def time_until_retry(self) -> float | None:
        """Get time remaining until retry is allowed.

        Returns:
            Seconds until the circuit transitions to HALF_OPEN, or None if
            the circuit is not OPEN.
        """
        with self._lock:
            if self._state != CircuitState.OPEN:
                return None

            if self._last_failure_time is None:
                return None

            elapsed = time.monotonic() - self._last_failure_time
            remaining = self._recovery_timeout - elapsed
            return max(0.0, remaining)

    def record_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float,
    ) -> None:
        """Record token usage and estimated cost from an execution.

        Updates running totals for cost tracking. Call this after each
        successful or failed execution that consumed tokens.

        Args:
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens consumed.
            estimated_cost: Estimated cost in USD for this execution.
        """
        with self._lock:
            self._stats.total_input_tokens += input_tokens
            self._stats.total_output_tokens += output_tokens
            self._stats.total_estimated_cost += estimated_cost

            _logger.debug(
                "circuit_breaker.cost_recorded",
                name=self._name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=round(estimated_cost, 6),
                total_input_tokens=self._stats.total_input_tokens,
                total_output_tokens=self._stats.total_output_tokens,
                total_estimated_cost=round(self._stats.total_estimated_cost, 4),
            )

    def check_cost_threshold(self, max_cost: float) -> bool:
        """Check if total estimated cost exceeds a threshold.

        Args:
            max_cost: Maximum allowed cost in USD.

        Returns:
            True if threshold is exceeded (should stop), False otherwise.
        """
        with self._lock:
            exceeded = self._stats.total_estimated_cost > max_cost
            if exceeded:
                _logger.warning(
                    "circuit_breaker.cost_threshold_exceeded",
                    name=self._name,
                    total_estimated_cost=round(self._stats.total_estimated_cost, 4),
                    max_cost=max_cost,
                )
            return exceeded

    def get_cost_summary(self) -> dict[str, float]:
        """Get cost summary for reporting.

        Returns:
            Dictionary with input_tokens, output_tokens, and estimated_cost.
        """
        with self._lock:
            return {
                "input_tokens": self._stats.total_input_tokens,
                "output_tokens": self._stats.total_output_tokens,
                "estimated_cost_usd": round(self._stats.total_estimated_cost, 4),
            }

    def get_stats(self) -> CircuitBreakerStats:
        """Get current statistics.

        Returns:
            Copy of current CircuitBreakerStats.
        """
        with self._lock:
            # Return a copy to prevent external modification
            return CircuitBreakerStats(
                total_successes=self._stats.total_successes,
                total_failures=self._stats.total_failures,
                times_opened=self._stats.times_opened,
                times_half_opened=self._stats.times_half_opened,
                times_closed=self._stats.times_closed,
                last_failure_at=self._stats.last_failure_at,
                last_state_change_at=self._stats.last_state_change_at,
                consecutive_failures=self._stats.consecutive_failures,
                total_input_tokens=self._stats.total_input_tokens,
                total_output_tokens=self._stats.total_output_tokens,
                total_estimated_cost=self._stats.total_estimated_cost,
            )

    def reset(self) -> None:
        """Reset the circuit breaker to initial state.

        This resets:
        - State to CLOSED
        - Failure counts to 0
        - Last failure time to None

        Statistics are NOT reset (use get_stats() to view history).
        """
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._stats.consecutive_failures = 0

            if old_state != CircuitState.CLOSED:
                self._stats.times_closed += 1
                self._stats.last_state_change_at = time.monotonic()
                _logger.info(
                    "circuit_breaker.reset",
                    name=self._name,
                    from_state=old_state.value,
                )

    def force_open(self) -> None:
        """Force the circuit to OPEN state.

        Useful for manual intervention or testing.
        """
        with self._lock:
            if self._state != CircuitState.OPEN:
                old_state = self._state
                self._set_state(CircuitState.OPEN)
                self._last_failure_time = time.monotonic()
                _logger.info(
                    "circuit_breaker.force_opened",
                    name=self._name,
                    from_state=old_state.value,
                )

    def force_close(self) -> None:
        """Force the circuit to CLOSED state.

        Useful for manual intervention or testing. Also resets failure counts.
        """
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._stats.consecutive_failures = 0

            if old_state != CircuitState.CLOSED:
                self._stats.times_closed += 1
                self._stats.last_state_change_at = time.monotonic()
                _logger.info(
                    "circuit_breaker.force_closed",
                    name=self._name,
                    from_state=old_state.value,
                )

    def __repr__(self) -> str:
        """Get string representation of circuit breaker."""
        return (
            f"CircuitBreaker(name={self._name!r}, state={self._state.value}, "
            f"failures={self._failure_count}/{self._failure_threshold})"
        )


# Re-export for convenience
__all__ = [
    "CircuitBreaker",
    "CircuitBreakerStats",
    "CircuitState",
]
