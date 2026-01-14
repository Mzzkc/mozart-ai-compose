"""Adaptive retry strategy with intelligent pattern detection.

Analyzes error history to make smart retry decisions, detecting patterns like:
- Rapid consecutive failures → longer exponential backoff
- Same error code repeated → different strategy (may be persistent issue)
- Rate limits → use rate limit delay from error classification
- Transient errors → standard retry with jitter

Example usage:
    from mozart.execution.retry_strategy import (
        AdaptiveRetryStrategy, ErrorRecord, RetryRecommendation
    )

    strategy = AdaptiveRetryStrategy()

    # Record errors as they occur
    error_history: list[ErrorRecord] = []
    error_history.append(ErrorRecord.from_classified_error(classified_error))

    # Get retry recommendation
    recommendation = strategy.analyze(error_history)
    if recommendation.should_retry:
        await asyncio.sleep(recommendation.delay_seconds)
        # Retry the operation
    else:
        # Give up or escalate
        logger.error(f"Not retrying: {recommendation.reason}")
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

from mozart.core.errors import (
    ClassificationResult,
    ClassifiedError,
    ErrorCategory,
    ErrorCode,
    RetryBehavior,
)
from mozart.core.logging import get_logger

if TYPE_CHECKING:
    from mozart.learning.global_store import GlobalLearningStore


# =============================================================================
# Delay Learning Types (Evolution: Dynamic Retry Delay Learning)
# =============================================================================


@dataclass
class DelayOutcome:
    """Record of a delay used and its outcome for learning.

    Captures the relationship between delay duration and subsequent success/failure,
    enabling the system to learn optimal delays for each error type.

    Attributes:
        error_code: The ErrorCode that triggered the retry.
        delay_seconds: The delay that was actually used before retrying.
        succeeded_after: Whether the retry succeeded after this delay.
        timestamp: When this delay was recorded.
    """

    error_code: ErrorCode
    delay_seconds: float
    succeeded_after: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class DelayHistory:
    """Tracks historical delay outcomes for learning optimal delays.

    Maintains a record of (error_code, delay, success) tuples to enable
    the system to learn which delays work best for each error type.

    Thread-safe: Uses a threading.Lock to protect all mutable operations.
    Pruning maintains chronological order by sorting retained outcomes
    by timestamp after grouping by error code.
    """

    def __init__(self, max_history: int = 100) -> None:
        """Initialize delay history.

        Args:
            max_history: Maximum number of outcomes to retain per error code.
        """
        self._history: list[DelayOutcome] = []
        self._max_history = max_history
        self._lock = threading.Lock()

    def record(self, outcome: DelayOutcome) -> None:
        """Record a delay outcome.

        Thread-safe: Uses lock to protect append and pruning operations.

        Args:
            outcome: The delay outcome to record.

        Raises:
            ValueError: If outcome is None or outcome.delay_seconds is negative.
        """
        if outcome is None:
            raise ValueError("outcome cannot be None")
        if outcome.delay_seconds < 0:
            raise ValueError(f"delay_seconds must be >= 0, got {outcome.delay_seconds}")

        with self._lock:
            self._history.append(outcome)

            # Prune old history if needed (keep most recent per error code)
            if len(self._history) > self._max_history * 10:
                # Keep last N for each error code
                from collections import defaultdict

                by_code: defaultdict[ErrorCode, list[DelayOutcome]] = defaultdict(list)
                for o in self._history:
                    by_code[o.error_code].append(o)

                self._history = []
                for outcomes in by_code.values():
                    self._history.extend(outcomes[-self._max_history :])

                # Restore chronological order after grouping by error code
                self._history.sort(key=lambda o: o.timestamp)

    def query_for_error_code(self, code: ErrorCode) -> list[DelayOutcome]:
        """Query outcomes for a specific error code.

        Args:
            code: The error code to query.

        Returns:
            List of DelayOutcome for this error code.
        """
        return [o for o in self._history if o.error_code == code]

    def get_average_successful_delay(self, code: ErrorCode) -> float | None:
        """Get average delay that led to success for an error code.

        Args:
            code: The error code to query.

        Returns:
            Average successful delay in seconds, or None if no successful samples.
        """
        successful = [o for o in self._history if o.error_code == code and o.succeeded_after]
        if not successful:
            return None
        return sum(o.delay_seconds for o in successful) / len(successful)

    def get_sample_count(self, code: ErrorCode) -> int:
        """Get number of samples for an error code.

        Args:
            code: The error code to query.

        Returns:
            Number of delay outcomes recorded for this code.
        """
        return len([o for o in self._history if o.error_code == code])

# Module-level logger
_logger = get_logger("retry_strategy")


class RetryPattern(str, Enum):
    """Detected error patterns that influence retry strategy.

    Each pattern triggers a different retry behavior to maximize
    the chance of recovery while minimizing wasted attempts.
    """

    NONE = "none"
    """No clear pattern detected - use default retry behavior."""

    RAPID_FAILURES = "rapid_failures"
    """Multiple failures in quick succession - needs longer cooldown."""

    REPEATED_ERROR_CODE = "repeated_error_code"
    """Same error code appearing repeatedly - may be persistent issue."""

    RATE_LIMITED = "rate_limited"
    """Rate limiting detected - use rate limit wait time."""

    CASCADING_FAILURES = "cascading_failures"
    """Errors are getting worse/different - system may be degrading."""

    INTERMITTENT = "intermittent"
    """Errors are spread out with successes in between - normal transient."""

    RECOVERY_IN_PROGRESS = "recovery_in_progress"
    """Recent success after failures - system may be recovering."""


@dataclass
class ErrorRecord:
    """Record of a single error occurrence for pattern analysis.

    Captures all relevant information about an error to enable
    intelligent pattern detection across multiple errors.

    Attributes:
        timestamp: When the error occurred (UTC).
        error_code: Structured error code (e.g., E001, E101).
        category: High-level error category (rate_limit, transient, etc.).
        message: Human-readable error description.
        exit_code: Process exit code if applicable.
        exit_signal: Signal number if killed by signal.
        retriable: Whether this specific error is retriable.
        suggested_wait: Classifier's suggested wait time in seconds.
        sheet_num: Sheet number where error occurred.
        attempt_num: Which attempt number this was (1-indexed).
        monotonic_time: Monotonic timestamp for precise timing calculations.
        root_cause_confidence: Confidence in root cause identification (0.0-1.0).
        secondary_error_count: Number of secondary errors detected.
    """

    timestamp: datetime
    error_code: ErrorCode
    category: ErrorCategory
    message: str
    exit_code: int | None = None
    exit_signal: int | None = None
    retriable: bool = True
    suggested_wait: float | None = None
    sheet_num: int | None = None
    attempt_num: int = 1
    monotonic_time: float = field(default_factory=time.monotonic)
    root_cause_confidence: float | None = None
    secondary_error_count: int = 0

    @classmethod
    def from_classified_error(
        cls,
        error: ClassifiedError,
        sheet_num: int | None = None,
        attempt_num: int = 1,
    ) -> ErrorRecord:
        """Create an ErrorRecord from a ClassifiedError.

        This is the primary factory method for creating ErrorRecords
        in the retry flow.

        Args:
            error: ClassifiedError from the error classifier.
            sheet_num: Optional sheet number for context.
            attempt_num: Which retry attempt this represents.

        Returns:
            ErrorRecord populated from the classified error.
        """
        return cls(
            timestamp=datetime.now(UTC),
            error_code=error.error_code,
            category=error.category,
            message=error.message,
            exit_code=error.exit_code,
            exit_signal=error.exit_signal,
            retriable=error.retriable,
            suggested_wait=error.suggested_wait_seconds,
            sheet_num=sheet_num,
            attempt_num=attempt_num,
        )

    @classmethod
    def from_classification_result(
        cls,
        result: ClassificationResult,
        sheet_num: int | None = None,
        attempt_num: int = 1,
    ) -> ErrorRecord:
        """Create an ErrorRecord from a ClassificationResult.

        This factory method captures root cause information from the multi-error
        classification, including confidence in root cause identification and
        the count of secondary errors. This enables the retry strategy to
        consider root cause confidence when making retry decisions.

        Args:
            result: ClassificationResult from classify_execution().
            sheet_num: Optional sheet number for context.
            attempt_num: Which retry attempt this represents.

        Returns:
            ErrorRecord with root cause confidence and secondary error count.

        Raises:
            ValueError: If confidence is not in valid range [0.0, 1.0].
        """
        # Validate confidence is in valid range (defensive check)
        if not 0.0 <= result.confidence <= 1.0:
            raise ValueError(
                f"root_cause_confidence must be 0.0-1.0, got {result.confidence}"
            )

        primary = result.primary
        return cls(
            timestamp=datetime.now(UTC),
            error_code=primary.error_code,
            category=primary.category,
            message=primary.message,
            exit_code=primary.exit_code,
            exit_signal=primary.exit_signal,
            retriable=primary.retriable,
            suggested_wait=primary.suggested_wait_seconds,
            sheet_num=sheet_num,
            attempt_num=attempt_num,
            root_cause_confidence=result.confidence,
            secondary_error_count=len(result.secondary),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for logging/serialization.

        Returns:
            Dictionary representation with all fields.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "error_code": self.error_code.value,
            "category": self.category.value,
            "message": self.message,
            "exit_code": self.exit_code,
            "exit_signal": self.exit_signal,
            "retriable": self.retriable,
            "suggested_wait": self.suggested_wait,
            "sheet_num": self.sheet_num,
            "attempt_num": self.attempt_num,
            "root_cause_confidence": (
                round(self.root_cause_confidence, 3)
                if self.root_cause_confidence is not None
                else None
            ),
            "secondary_error_count": self.secondary_error_count,
        }


@dataclass
class RetryRecommendation:
    """Recommendation from the adaptive retry strategy.

    Encapsulates the decision of whether to retry, how long to wait,
    and the reasoning behind the decision for observability.

    Attributes:
        should_retry: Whether a retry should be attempted.
        delay_seconds: Recommended delay before retrying.
        reason: Human-readable explanation of the decision.
        confidence: Confidence in this recommendation (0.0-1.0).
        detected_pattern: The pattern that influenced this decision.
        strategy_used: Name of the strategy/heuristic that was applied.
        root_cause_confidence: Confidence in root cause identification (0.0-1.0, None if N/A).
    """

    should_retry: bool
    delay_seconds: float
    reason: str
    confidence: float
    detected_pattern: RetryPattern = RetryPattern.NONE
    strategy_used: str = "default"
    root_cause_confidence: float | None = None

    def __post_init__(self) -> None:
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")
        if self.delay_seconds < 0:
            raise ValueError(f"delay_seconds must be >= 0, got {self.delay_seconds}")

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for logging/serialization.

        Returns:
            Dictionary representation with all fields.
        """
        return {
            "should_retry": self.should_retry,
            "delay_seconds": round(self.delay_seconds, 2),
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "detected_pattern": self.detected_pattern.value,
            "strategy_used": self.strategy_used,
            "root_cause_confidence": (
                round(self.root_cause_confidence, 3)
                if self.root_cause_confidence is not None
                else None
            ),
        }


@dataclass
class RetryStrategyConfig:
    """Configuration for the adaptive retry strategy.

    All timing values are in seconds. Thresholds are tuned for typical
    Claude CLI execution patterns.

    Attributes:
        base_delay: Starting delay for exponential backoff.
        max_delay: Maximum delay cap.
        exponential_base: Multiplier for exponential backoff.
        rapid_failure_window: Window (seconds) to detect rapid failures.
        rapid_failure_threshold: Number of failures in window to trigger.
        rapid_failure_multiplier: Extra delay multiplier for rapid failures.
        repeated_error_threshold: Same error code count before flagging.
        repeated_error_strategy_change_threshold: Count before strategy change.
        min_confidence: Minimum confidence for retry recommendation.
        jitter_factor: Random jitter to add (0.0-1.0 of delay).
    """

    base_delay: float = 10.0
    max_delay: float = 3600.0  # 1 hour
    exponential_base: float = 2.0
    rapid_failure_window: float = 60.0  # 1 minute
    rapid_failure_threshold: int = 3
    rapid_failure_multiplier: float = 2.0
    repeated_error_threshold: int = 2
    repeated_error_strategy_change_threshold: int = 3
    min_confidence: float = 0.3
    jitter_factor: float = 0.25  # 25% jitter

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.exponential_base <= 1:
            raise ValueError("exponential_base must be > 1")
        if self.rapid_failure_window <= 0:
            raise ValueError("rapid_failure_window must be positive")
        if self.rapid_failure_threshold < 1:
            raise ValueError("rapid_failure_threshold must be >= 1")


class AdaptiveRetryStrategy:
    """Intelligent retry strategy that analyzes error patterns.

    The strategy examines error history to detect patterns and make
    informed retry decisions. Key features:

    1. **Rapid Failure Detection**: If multiple errors occur in a short
       window, applies longer backoff to avoid overwhelming the system.

    2. **Repeated Error Detection**: If the same error code appears
       repeatedly, may recommend different strategies or lower confidence.

    3. **Rate Limit Handling**: Uses suggested wait times from rate limit
       errors, with additional buffer.

    4. **Cascading Failure Detection**: If errors are getting different/worse,
       may recommend stopping to prevent further damage.

    5. **Recovery Detection**: If recent attempts succeeded after failures,
       uses shorter delays to capitalize on recovery.

    6. **Delay Learning with Circuit Breaker**: When a DelayHistory is provided,
       the strategy learns optimal delays from past outcomes. A circuit breaker
       protects against bad learned delays by reverting to static delays after
       3 consecutive failures.

    Circuit Breaker State Design:
        The circuit breaker state (_learned_delay_failures, _use_learned_delay)
        is intentionally ephemeral and NOT persisted. This is a deliberate design
        choice with the following trade-offs:

        Benefits:
        - After restart, the system gets a "fresh start" to try learned delays
        - Avoids persisting potentially stale circuit breaker state
        - Simple implementation without additional state management

        Trade-offs:
        - After restart, may retry with a previously-failed learned delay once
        - Circuit breaker will re-trigger after 3 failures if the learned delay
          is still problematic

        The DelayHistory itself CAN be persisted (it's just delay outcomes), but
        the circuit breaker resets on each AdaptiveRetryStrategy instantiation.
        Use reset_circuit_breaker() to manually reset circuit breaker state for
        a specific error code during runtime.

    Thread-safe: No mutable state; all analysis is based on input history.

    Example:
        strategy = AdaptiveRetryStrategy()

        # Analyze error history
        recommendation = strategy.analyze(error_history)

        # Log the decision
        logger.info(
            "retry_decision",
            should_retry=recommendation.should_retry,
            delay=recommendation.delay_seconds,
            pattern=recommendation.detected_pattern.value,
            reason=recommendation.reason,
        )
    """

    def __init__(
        self,
        config: RetryStrategyConfig | None = None,
        delay_history: DelayHistory | None = None,
        global_learning_store: GlobalLearningStore | None = None,
    ) -> None:
        """Initialize the adaptive retry strategy.

        Args:
            config: Optional configuration. Uses defaults if not provided.
            delay_history: Optional delay history for learning. If not provided,
                learning features are disabled (purely static delays).
            global_learning_store: Optional global learning store for cross-workspace
                learned delays (Evolution #3: Learned Wait Time Injection).
                If provided, blend_historical_delay() will query global store
                for cross-workspace learned delays when in-memory history is
                insufficient.
        """
        self.config = config or RetryStrategyConfig()
        self._delay_history = delay_history
        self._global_store = global_learning_store

        # Circuit breaker state: track consecutive failures when using learned delays.
        # If > 3 failures with learned delay, revert to static for that error code.
        #
        # NOTE: This state is intentionally ephemeral (not persisted). See class
        # docstring "Circuit Breaker State Design" section for rationale.
        self._learned_delay_failures: dict[ErrorCode, int] = {}
        self._use_learned_delay: dict[ErrorCode, bool] = {}

    def analyze(
        self,
        error_history: list[ErrorRecord],
        max_retries: int | None = None,
    ) -> RetryRecommendation:
        """Analyze error history and recommend retry behavior.

        This is the main entry point for the adaptive retry strategy.
        It examines the error history to detect patterns and returns
        a recommendation with reasoning.

        Args:
            error_history: List of ErrorRecords in chronological order.
            max_retries: Optional maximum retries to consider (for confidence).

        Returns:
            RetryRecommendation with decision, delay, and reasoning.
        """
        if not error_history:
            # No errors - this shouldn't happen, but handle gracefully
            return RetryRecommendation(
                should_retry=True,
                delay_seconds=self.config.base_delay,
                reason="No error history - using default retry",
                confidence=0.5,
                detected_pattern=RetryPattern.NONE,
                strategy_used="default",
            )

        # Get the most recent error
        latest_error = error_history[-1]
        attempt_count = len(error_history)

        # Check for non-retriable error first
        if not latest_error.retriable:
            return self._recommend_no_retry(
                latest_error,
                "Error is not retriable",
                confidence=0.95,
                pattern=RetryPattern.NONE,
            )

        # Detect patterns in the error history
        pattern = self._detect_pattern(error_history)

        # Get recommendation based on pattern
        recommendation = self._recommend_for_pattern(
            pattern=pattern,
            error_history=error_history,
            latest_error=latest_error,
            attempt_count=attempt_count,
            max_retries=max_retries,
        )

        # Propagate root cause confidence from latest error to recommendation
        recommendation.root_cause_confidence = latest_error.root_cause_confidence

        # Log the decision including root cause confidence
        _logger.info(
            "retry_strategy.decision",
            should_retry=recommendation.should_retry,
            delay_seconds=round(recommendation.delay_seconds, 2),
            confidence=round(recommendation.confidence, 3),
            detected_pattern=pattern.value,
            strategy_used=recommendation.strategy_used,
            attempt_count=attempt_count,
            latest_error_code=latest_error.error_code.value,
            reason=recommendation.reason,
            root_cause_confidence=(
                round(latest_error.root_cause_confidence, 3)
                if latest_error.root_cause_confidence is not None
                else None
            ),
            secondary_error_count=latest_error.secondary_error_count,
        )

        return recommendation

    def _detect_pattern(self, error_history: list[ErrorRecord]) -> RetryPattern:
        """Detect patterns in the error history.

        Examines the sequence of errors to identify patterns that
        should influence retry behavior.

        Args:
            error_history: List of ErrorRecords in chronological order.

        Returns:
            The most significant RetryPattern detected.
        """
        if not error_history:
            return RetryPattern.NONE

        latest = error_history[-1]

        # Check for rate limiting (highest priority)
        if latest.category == ErrorCategory.RATE_LIMIT:
            return RetryPattern.RATE_LIMITED

        # Check for rapid failures (failures close together)
        if self._has_rapid_failures(error_history):
            return RetryPattern.RAPID_FAILURES

        # Check for repeated same error code
        if self._has_repeated_error_code(error_history):
            return RetryPattern.REPEATED_ERROR_CODE

        # Check for cascading (different error codes appearing)
        if self._has_cascading_errors(error_history):
            return RetryPattern.CASCADING_FAILURES

        # Check for intermittent pattern (would need success info)
        # For now, treat as no clear pattern
        return RetryPattern.NONE

    def _has_rapid_failures(self, error_history: list[ErrorRecord]) -> bool:
        """Check if failures are happening in rapid succession.

        Args:
            error_history: Error history to analyze.

        Returns:
            True if rapid failure pattern is detected.
        """
        if len(error_history) < self.config.rapid_failure_threshold:
            return False

        # Look at the most recent N errors
        recent = error_history[-self.config.rapid_failure_threshold:]

        # Check if all occurred within the window
        now = time.monotonic()
        oldest_in_recent = recent[0].monotonic_time
        time_span = now - oldest_in_recent

        return time_span <= self.config.rapid_failure_window

    def _has_repeated_error_code(self, error_history: list[ErrorRecord]) -> bool:
        """Check if the same error code is repeating.

        Args:
            error_history: Error history to analyze.

        Returns:
            True if repeated error code pattern is detected.
        """
        if len(error_history) < self.config.repeated_error_threshold:
            return False

        # Count occurrences of the latest error code
        latest_code = error_history[-1].error_code
        recent_count = sum(
            1 for e in error_history[-5:]  # Look at last 5
            if e.error_code == latest_code
        )

        return recent_count >= self.config.repeated_error_threshold

    def _has_cascading_errors(self, error_history: list[ErrorRecord]) -> bool:
        """Check if errors are cascading (different types appearing).

        A cascading pattern suggests the system is degrading in
        different ways, which is concerning.

        Args:
            error_history: Error history to analyze.

        Returns:
            True if cascading error pattern is detected.
        """
        if len(error_history) < 3:
            return False

        # Count unique error codes in recent history
        recent_codes = {e.error_code for e in error_history[-5:]}

        # If we have 3+ different error codes recently, that's cascading
        return len(recent_codes) >= 3

    def _recommend_for_pattern(
        self,
        pattern: RetryPattern,
        error_history: list[ErrorRecord],
        latest_error: ErrorRecord,
        attempt_count: int,
        max_retries: int | None,
    ) -> RetryRecommendation:
        """Generate recommendation based on detected pattern.

        Args:
            pattern: The detected error pattern.
            error_history: Full error history.
            latest_error: Most recent error.
            attempt_count: Number of attempts so far.
            max_retries: Maximum retries allowed (if known).

        Returns:
            RetryRecommendation tailored to the pattern.
        """
        if pattern == RetryPattern.RATE_LIMITED:
            return self._recommend_rate_limit_retry(latest_error, attempt_count)

        if pattern == RetryPattern.RAPID_FAILURES:
            return self._recommend_rapid_failure_retry(
                error_history, attempt_count, max_retries
            )

        if pattern == RetryPattern.REPEATED_ERROR_CODE:
            return self._recommend_repeated_error_retry(
                error_history, latest_error, attempt_count, max_retries
            )

        if pattern == RetryPattern.CASCADING_FAILURES:
            return self._recommend_cascading_retry(
                error_history, attempt_count, max_retries
            )

        # Default: standard exponential backoff
        return self._recommend_standard_retry(latest_error, attempt_count, max_retries)

    def _recommend_rate_limit_retry(
        self,
        error: ErrorRecord,
        attempt_count: int,
    ) -> RetryRecommendation:
        """Recommend retry for rate limit errors.

        Uses the suggested wait from error classification, with a buffer.

        Args:
            error: The rate limit error.
            attempt_count: Current attempt count.

        Returns:
            RetryRecommendation with rate limit delay.
        """
        # Use suggested wait if available, otherwise default to 1 hour
        base_wait = error.suggested_wait or 3600.0

        # Add a small buffer (10%) to avoid hitting limit again immediately
        delay = min(base_wait * 1.1, self.config.max_delay)

        return RetryRecommendation(
            should_retry=True,
            delay_seconds=delay,
            reason=f"Rate limit detected - waiting {delay:.0f}s (suggested: {base_wait:.0f}s)",
            confidence=0.85,  # High confidence in rate limit handling
            detected_pattern=RetryPattern.RATE_LIMITED,
            strategy_used="rate_limit_wait",
        )

    def _recommend_rapid_failure_retry(
        self,
        error_history: list[ErrorRecord],
        attempt_count: int,
        max_retries: int | None,
    ) -> RetryRecommendation:
        """Recommend retry for rapid consecutive failures.

        Applies extra multiplier to give system time to recover.

        Args:
            error_history: Full error history.
            attempt_count: Current attempt count.
            max_retries: Maximum retries allowed.

        Returns:
            RetryRecommendation with extended backoff.
        """
        # Calculate base exponential delay
        base_delay = self.config.base_delay * (
            self.config.exponential_base ** (attempt_count - 1)
        )

        # Apply rapid failure multiplier
        delay = min(
            base_delay * self.config.rapid_failure_multiplier,
            self.config.max_delay,
        )

        # Apply jitter
        delay = self._apply_jitter(delay)

        # Lower confidence when seeing rapid failures
        confidence = self._calculate_confidence(attempt_count, max_retries)
        confidence *= 0.8  # Reduce confidence for rapid failures

        return RetryRecommendation(
            should_retry=True,
            delay_seconds=delay,
            reason=(
                f"Rapid consecutive failures detected ({len(error_history)} in "
                f"{self.config.rapid_failure_window}s) - applying extended backoff"
            ),
            confidence=max(confidence, self.config.min_confidence),
            detected_pattern=RetryPattern.RAPID_FAILURES,
            strategy_used="rapid_failure_backoff",
        )

    def _recommend_repeated_error_retry(
        self,
        error_history: list[ErrorRecord],
        latest_error: ErrorRecord,
        attempt_count: int,
        max_retries: int | None,
    ) -> RetryRecommendation:
        """Recommend retry for repeated same error code.

        When the same error repeats, it may indicate a persistent issue.
        Lower confidence and suggest longer delays.

        Args:
            error_history: Full error history.
            latest_error: Most recent error.
            attempt_count: Current attempt count.
            max_retries: Maximum retries allowed.

        Returns:
            RetryRecommendation with adjusted strategy.
        """
        # Count how many times this error has appeared
        same_error_count = sum(
            1 for e in error_history
            if e.error_code == latest_error.error_code
        )

        # If we've seen it too many times, recommend not retrying
        if same_error_count >= self.config.repeated_error_strategy_change_threshold:
            return RetryRecommendation(
                should_retry=False,
                delay_seconds=0,
                reason=(
                    f"Error {latest_error.error_code.value} has occurred "
                    f"{same_error_count} times - likely persistent issue"
                ),
                confidence=0.75,
                detected_pattern=RetryPattern.REPEATED_ERROR_CODE,
                strategy_used="repeated_error_abort",
            )

        # Otherwise, retry with increased delay
        base_delay = self.config.base_delay * (
            self.config.exponential_base ** attempt_count
        )
        delay = min(base_delay, self.config.max_delay)
        delay = self._apply_jitter(delay)

        # Reduced confidence for repeated errors
        confidence = self._calculate_confidence(attempt_count, max_retries)
        confidence *= 0.7  # Significantly reduce confidence

        return RetryRecommendation(
            should_retry=True,
            delay_seconds=delay,
            reason=(
                f"Error {latest_error.error_code.value} repeated {same_error_count} times "
                f"- retrying with extended delay"
            ),
            confidence=max(confidence, self.config.min_confidence),
            detected_pattern=RetryPattern.REPEATED_ERROR_CODE,
            strategy_used="repeated_error_retry",
        )

    def _recommend_cascading_retry(
        self,
        error_history: list[ErrorRecord],
        attempt_count: int,
        max_retries: int | None,
    ) -> RetryRecommendation:
        """Recommend retry for cascading (varied) errors.

        Cascading errors suggest system instability. Be more cautious.

        Args:
            error_history: Full error history.
            attempt_count: Current attempt count.
            max_retries: Maximum retries allowed.

        Returns:
            RetryRecommendation with cautious approach.
        """
        # Get unique error codes
        unique_codes = {e.error_code for e in error_history[-5:]}

        # If too many different errors, recommend stopping
        if len(unique_codes) >= 4:
            return RetryRecommendation(
                should_retry=False,
                delay_seconds=0,
                reason=(
                    f"Cascading failures with {len(unique_codes)} different error types "
                    f"- system appears unstable"
                ),
                confidence=0.7,
                detected_pattern=RetryPattern.CASCADING_FAILURES,
                strategy_used="cascading_abort",
            )

        # Otherwise, use longer delay
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt_count),
            self.config.max_delay,
        )
        delay = self._apply_jitter(delay)

        confidence = self._calculate_confidence(attempt_count, max_retries)
        confidence *= 0.6  # Low confidence for cascading errors

        return RetryRecommendation(
            should_retry=True,
            delay_seconds=delay,
            reason=(
                f"Cascading failures with {len(unique_codes)} error types "
                f"- retrying cautiously"
            ),
            confidence=max(confidence, self.config.min_confidence),
            detected_pattern=RetryPattern.CASCADING_FAILURES,
            strategy_used="cascading_retry",
        )

    def blend_historical_delay(
        self,
        error_code: ErrorCode,
        static_delay: float,
    ) -> tuple[float, str]:
        """Blend learned delay with static delay for an error code.

        Uses historical delay outcomes to compute an optimal learned delay,
        then blends it with the static (ErrorCode-based) delay based on
        sample size. Respects circuit breaker state.

        Evolution #3: Learned Wait Time Injection - now also queries the
        global learning store for cross-workspace learned delays when
        in-memory history is insufficient. This enables new jobs to benefit
        from delays learned by previous jobs across all workspaces.

        Priority order:
        1. In-memory delay history (if sufficient samples)
        2. Global learning store (cross-workspace learned delays)
        3. Static delay (fallback)

        Args:
            error_code: The error code to get delay for.
            static_delay: The static delay from ErrorCode.get_retry_behavior().

        Returns:
            Tuple of (blended_delay, strategy_name).
            - If no history or circuit breaker triggered: (static_delay, "static")
            - If history available: (blended_delay, "learned_blend")
            - If global store has data: (delay, "global_learned" or "global_learned_blend")
        """
        # Circuit breaker check: if we've reverted to static for this code, use static
        if not self._use_learned_delay.get(error_code, True):
            return static_delay, "static_circuit_breaker"

        # First, try in-memory delay history (highest priority - job-specific learning)
        if self._delay_history is not None:
            learned_delay = self._delay_history.get_average_successful_delay(error_code)
            if learned_delay is not None:
                # Calculate blend weight based on sample count
                sample_count = self._delay_history.get_sample_count(error_code)
                weight = min(sample_count / 10.0, 1.0)

                # Blend: weight * learned + (1 - weight) * static
                blended = weight * learned_delay + (1 - weight) * static_delay
                return blended, "learned_blend"
            else:
                # History exists but no successful samples yet -> bootstrap phase
                # Fall through to check global store, but if global store also
                # has no data, return "static_bootstrap" instead of "static"
                pass

        # Second, try global learning store (Evolution #3: cross-workspace learning)
        if self._global_store is not None:
            try:
                global_delay, confidence, strategy = (
                    self._global_store.get_learned_wait_time_with_fallback(
                        error_code=error_code.value,
                        static_delay=static_delay,
                        min_samples=3,
                        min_confidence=0.7,
                    )
                )
                # Only use global store result if it's not just a static fallback
                if strategy != "static_fallback":
                    _logger.debug(
                        "retry_strategy.global_learned_delay",
                        error_code=error_code.value,
                        delay=round(global_delay, 2),
                        confidence=round(confidence, 3),
                        strategy=strategy,
                    )
                    return global_delay, strategy
            except Exception as e:
                # Global store query failure shouldn't block retry
                _logger.warning(
                    "retry_strategy.global_store_error",
                    error_code=error_code.value,
                    error=str(e),
                )

        # Fallback: distinguish between "no history at all" vs "history but no samples"
        if self._delay_history is not None:
            # History exists but no successful samples -> bootstrap phase
            return static_delay, "static_bootstrap"
        else:
            # No history at all -> static
            return static_delay, "static"

    def record_delay_outcome(
        self,
        error_code: ErrorCode,
        delay_used: float,
        succeeded: bool,
    ) -> None:
        """Record the outcome of a retry delay for learning.

        Should be called after each retry attempt to update the delay history.
        Also updates circuit breaker state.

        Args:
            error_code: The error code that was being retried.
            delay_used: The delay in seconds that was used.
            succeeded: Whether the retry succeeded after this delay.
        """
        if self._delay_history is None:
            return

        # Record the outcome
        outcome = DelayOutcome(
            error_code=error_code,
            delay_seconds=delay_used,
            succeeded_after=succeeded,
        )
        self._delay_history.record(outcome)

        # Update circuit breaker state
        if self._use_learned_delay.get(error_code, True):
            # We were using learned delay
            if succeeded:
                # Success: reset failure count
                self._learned_delay_failures[error_code] = 0
            else:
                # Failure: increment and check threshold
                failures = self._learned_delay_failures.get(error_code, 0) + 1
                self._learned_delay_failures[error_code] = failures

                if failures > 3:
                    # Circuit breaker triggers: revert to static
                    self._use_learned_delay[error_code] = False
                    _logger.warning(
                        "circuit_breaker.triggered",
                        error_code=error_code.value,
                        consecutive_failures=failures,
                        message="Reverting to static delay for this error code",
                    )

    def reset_circuit_breaker(self, error_code: ErrorCode) -> None:
        """Reset circuit breaker for an error code, re-enabling learned delays.

        Call this method when you want to give learned delays another chance
        after the circuit breaker has tripped. Common scenarios:

        - After manual intervention that fixed the underlying issue
        - After a cooling-off period with successful static delays
        - At the start of a new batch/job where conditions may have changed

        Note: The circuit breaker state is ephemeral (not persisted), so it
        automatically resets when a new AdaptiveRetryStrategy is instantiated.
        This method is for resetting during runtime without reinstantiation.

        Args:
            error_code: The error code to reset circuit breaker for.

        Example:
            # After manual fix, give learned delays another chance
            strategy.reset_circuit_breaker(ErrorCode.E101)
        """
        self._use_learned_delay[error_code] = True
        self._learned_delay_failures[error_code] = 0
        _logger.info(
            "circuit_breaker.reset",
            error_code=error_code.value,
            message="Circuit breaker reset, learned delays re-enabled",
        )

    def _recommend_standard_retry(
        self,
        error: ErrorRecord,
        attempt_count: int,
        max_retries: int | None,
    ) -> RetryRecommendation:
        """Recommend standard retry using ErrorCode-specific behavior.

        Uses ErrorCode.get_retry_behavior() for precise delay recommendations
        rather than generic exponential backoff. Falls back to exponential
        backoff if no error code available.

        Args:
            error: The latest error.
            attempt_count: Current attempt count.
            max_retries: Maximum retries allowed.

        Returns:
            RetryRecommendation with ErrorCode-specific or standard backoff.
        """
        # Get ErrorCode-specific retry behavior
        retry_behavior = error.error_code.get_retry_behavior()

        # Check if ErrorCode says this is not retriable
        if not retry_behavior.is_retriable:
            return RetryRecommendation(
                should_retry=False,
                delay_seconds=0,
                reason=f"ErrorCode {error.error_code.value} not retriable: {retry_behavior.reason}",
                confidence=0.90,
                detected_pattern=RetryPattern.NONE,
                strategy_used="error_code_not_retriable",
            )

        # Determine delay: ErrorCode-specific > suggested_wait > exponential backoff
        if retry_behavior.delay_seconds > 0:
            # Use ErrorCode-specific delay as base, scale with attempt count
            base_delay = retry_behavior.delay_seconds
            # Mild exponential increase for repeated errors (1.5x per attempt)
            # Final delay is capped by config.max_delay below for consistency
            static_delay = base_delay * (1.5 ** (attempt_count - 1))

            # Try to blend with learned delay (Evolution: Dynamic Retry Delay Learning)
            delay, blend_strategy = self.blend_historical_delay(
                error.error_code, static_delay
            )
            if blend_strategy == "learned_blend":
                strategy_used = "learned_delay"
                reason_detail = f"{retry_behavior.reason} (learned blend)"
            else:
                strategy_used = "error_code_specific"
                reason_detail = retry_behavior.reason
        elif error.suggested_wait:
            # Fall back to classifier's suggested wait
            delay = error.suggested_wait
            strategy_used = "suggested_wait"
            reason_detail = "using classifier's suggested wait"
        else:
            # Fall back to exponential backoff
            delay = self.config.base_delay * (
                self.config.exponential_base ** (attempt_count - 1)
            )
            strategy_used = "exponential_backoff"
            reason_detail = "using exponential backoff"

        delay = min(delay, self.config.max_delay)
        delay = self._apply_jitter(delay)
        confidence = self._calculate_confidence(attempt_count, max_retries)

        return RetryRecommendation(
            should_retry=True,
            delay_seconds=delay,
            reason=f"Retry (attempt {attempt_count}): {reason_detail}",
            confidence=confidence,
            detected_pattern=RetryPattern.NONE,
            strategy_used=strategy_used,
        )

    def _recommend_no_retry(
        self,
        error: ErrorRecord,
        reason: str,
        confidence: float,
        pattern: RetryPattern,
    ) -> RetryRecommendation:
        """Create a no-retry recommendation.

        Args:
            error: The error that caused no-retry decision.
            reason: Human-readable reason.
            confidence: Confidence in this decision.
            pattern: Pattern that was detected.

        Returns:
            RetryRecommendation with should_retry=False.
        """
        return RetryRecommendation(
            should_retry=False,
            delay_seconds=0,
            reason=f"{reason}: {error.message}",
            confidence=confidence,
            detected_pattern=pattern,
            strategy_used="no_retry",
        )

    def _calculate_confidence(
        self,
        attempt_count: int,
        max_retries: int | None,
    ) -> float:
        """Calculate confidence based on retry progress.

        Confidence decreases as we approach max retries.

        Args:
            attempt_count: Current attempt number.
            max_retries: Maximum retries allowed (if known).

        Returns:
            Confidence value between 0.3 and 0.95.
        """
        if max_retries is None:
            # Without max_retries, use a decay based on attempt count
            # Confidence starts high and decreases logarithmically
            import math
            confidence = 0.95 / (1 + math.log1p(attempt_count - 1) * 0.3)
        else:
            # With max_retries, scale linearly
            remaining_ratio = 1 - (attempt_count / max_retries)
            confidence = 0.95 * max(remaining_ratio, 0.1)

        return max(confidence, self.config.min_confidence)

    def _apply_jitter(self, delay: float) -> float:
        """Apply random jitter to delay.

        Jitter helps prevent thundering herd problems when multiple
        clients retry simultaneously.

        Args:
            delay: Base delay in seconds.

        Returns:
            Delay with jitter applied.
        """
        import random

        # Add 0-jitter_factor of the delay as random jitter
        jitter = delay * self.config.jitter_factor * random.random()
        return delay + jitter


# Re-export for convenience
__all__ = [
    "AdaptiveRetryStrategy",
    "DelayHistory",
    "DelayOutcome",
    "ErrorRecord",
    "RetryBehavior",
    "RetryPattern",
    "RetryRecommendation",
    "RetryStrategyConfig",
]
