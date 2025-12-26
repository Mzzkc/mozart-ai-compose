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

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

from mozart.core.errors import ClassifiedError, ErrorCategory, ErrorCode
from mozart.core.logging import get_logger

if TYPE_CHECKING:
    pass

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
    """

    should_retry: bool
    delay_seconds: float
    reason: str
    confidence: float
    detected_pattern: RetryPattern = RetryPattern.NONE
    strategy_used: str = "default"

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

    def __init__(self, config: RetryStrategyConfig | None = None) -> None:
        """Initialize the adaptive retry strategy.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or RetryStrategyConfig()

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

        # Log the decision
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

    def _recommend_standard_retry(
        self,
        error: ErrorRecord,
        attempt_count: int,
        max_retries: int | None,
    ) -> RetryRecommendation:
        """Recommend standard exponential backoff retry.

        This is the default strategy when no specific pattern is detected.

        Args:
            error: The latest error.
            attempt_count: Current attempt count.
            max_retries: Maximum retries allowed.

        Returns:
            RetryRecommendation with standard backoff.
        """
        # Use suggested wait if available, otherwise exponential backoff
        if error.suggested_wait:
            delay = min(error.suggested_wait, self.config.max_delay)
        else:
            delay = self.config.base_delay * (
                self.config.exponential_base ** (attempt_count - 1)
            )
            delay = min(delay, self.config.max_delay)

        delay = self._apply_jitter(delay)
        confidence = self._calculate_confidence(attempt_count, max_retries)

        return RetryRecommendation(
            should_retry=True,
            delay_seconds=delay,
            reason=f"Standard retry with exponential backoff (attempt {attempt_count})",
            confidence=confidence,
            detected_pattern=RetryPattern.NONE,
            strategy_used="exponential_backoff",
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
    "ErrorRecord",
    "RetryPattern",
    "RetryRecommendation",
    "RetryStrategyConfig",
]
