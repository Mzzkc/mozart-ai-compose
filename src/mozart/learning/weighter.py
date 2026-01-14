"""Pattern weighting and priority calculation.

This module implements the pattern weighting system designed in Movement III:
- Combined recency + effectiveness weighting (CV 0.80)
- 10% monthly decay without confirmation
- Effectiveness threshold enforcement (0.3)
- Uncertainty classification (epistemic vs aleatoric)

The weighter calculates priority scores for patterns, determining which
patterns are most relevant for application in future executions.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class WeightingConfig:
    """Configuration for pattern weighting calculation.

    Attributes:
        decay_rate_per_month: Fraction of priority lost per month (default 0.1 = 10%).
        effectiveness_threshold: Below this, patterns are deprecated (default 0.3).
        epistemic_threshold: Variance threshold for learnable patterns (default 0.4).
        min_applications_for_effectiveness: Minimum applications before using
            actual effectiveness rate (default 3).
        frequency_normalization_base: Log base for frequency factor (default 100).
    """

    decay_rate_per_month: float = 0.1
    effectiveness_threshold: float = 0.3
    epistemic_threshold: float = 0.4
    min_applications_for_effectiveness: int = 3
    frequency_normalization_base: int = 100


class WeightablePattern(Protocol):
    """Protocol for patterns that can be weighted.

    Any pattern class that has these attributes can be weighted.
    """

    @property
    def occurrence_count(self) -> int: ...
    @property
    def led_to_success_count(self) -> int: ...
    @property
    def led_to_failure_count(self) -> int: ...
    @property
    def last_confirmed(self) -> datetime: ...
    @property
    def variance(self) -> float: ...


class PatternWeighter:
    """Calculates priority scores for patterns.

    Implements the combined recency + effectiveness weighting algorithm
    as specified in the Movement III design document.

    The priority formula is:
        priority = (
            effectiveness_score
            × recency_factor
            × frequency_factor
            × (1 - variance)
        )

    Where:
        - effectiveness_score = successes / (successes + failures + 1)
        - recency_factor = (1 - decay_rate) ^ months_since_last_confirmed
        - frequency_factor = min(1.0, log(occurrence_count + 1) / log(100))
        - variance = std_dev(pattern_application_outcomes)
    """

    def __init__(self, config: WeightingConfig | None = None) -> None:
        """Initialize the pattern weighter.

        Args:
            config: Weighting configuration. Uses defaults if None.
        """
        self.config = config or WeightingConfig()

    def calculate_priority(
        self,
        occurrence_count: int,
        led_to_success_count: int,
        led_to_failure_count: int,
        last_confirmed: datetime,
        variance: float = 0.0,
        now: datetime | None = None,
    ) -> float:
        """Calculate the priority score for a pattern.

        Args:
            occurrence_count: How many times this pattern was observed.
            led_to_success_count: Times pattern led to success when applied.
            led_to_failure_count: Times pattern led to failure when applied.
            last_confirmed: When this pattern was last confirmed effective.
            variance: Standard deviation of pattern application outcomes.
            now: Current time for recency calculation (defaults to now).

        Returns:
            Priority score from 0.0 to 1.0.
        """
        now = now or datetime.now()

        # Calculate effectiveness score
        effectiveness = self.calculate_effectiveness(
            led_to_success_count,
            led_to_failure_count,
        )

        # Calculate recency factor with exponential decay
        recency = self.calculate_recency_factor(last_confirmed, now)

        # Calculate frequency factor
        frequency = self.calculate_frequency_factor(occurrence_count)

        # Calculate variance penalty
        variance_factor = 1.0 - min(1.0, variance)

        # Combined priority score
        priority = effectiveness * recency * frequency * variance_factor

        return max(0.0, min(1.0, priority))

    def calculate_effectiveness(
        self,
        success_count: int,
        failure_count: int,
    ) -> float:
        """Calculate effectiveness score from success/failure counts.

        Uses Laplace smoothing (add-one) to handle cold start:
        effectiveness = (successes + 0.5) / (successes + failures + 1)

        This gives a slightly optimistic prior (0.5) for patterns with
        no application history.

        Args:
            success_count: Number of successful outcomes after application.
            failure_count: Number of failed outcomes after application.

        Returns:
            Effectiveness score from 0.0 to 1.0.
        """
        total = success_count + failure_count
        if total < self.config.min_applications_for_effectiveness:
            # Not enough data, return neutral prior
            return 0.5

        # Laplace smoothing
        return (success_count + 0.5) / (total + 1)

    def calculate_recency_factor(
        self,
        last_confirmed: datetime,
        now: datetime | None = None,
    ) -> float:
        """Calculate recency factor with exponential decay.

        Formula: recency = (1 - decay_rate) ^ months_since_last_confirmed

        With default 10% decay, a pattern loses:
        - 10% after 1 month
        - 19% after 2 months
        - 27% after 3 months
        - 35% after 4 months
        - etc.

        Args:
            last_confirmed: When the pattern was last confirmed effective.
            now: Current time (defaults to now).

        Returns:
            Recency factor from 0.0 to 1.0.
        """
        now = now or datetime.now()
        delta = now - last_confirmed
        months = delta.days / 30.0

        # Exponential decay
        decay_base = 1.0 - self.config.decay_rate_per_month
        recency: float = decay_base ** months

        return float(max(0.0, min(1.0, recency)))

    def calculate_frequency_factor(self, occurrence_count: int) -> float:
        """Calculate frequency factor from occurrence count.

        Formula: frequency = min(1.0, log(count + 1) / log(100))

        This gives diminishing returns for high counts:
        - 1 occurrence: ~0.15
        - 10 occurrences: ~0.52
        - 50 occurrences: ~0.85
        - 100+ occurrences: 1.0

        Args:
            occurrence_count: How many times the pattern was observed.

        Returns:
            Frequency factor from 0.0 to 1.0.
        """
        if occurrence_count <= 0:
            return 0.0

        base = self.config.frequency_normalization_base
        log_count = math.log(occurrence_count + 1)
        log_base = math.log(base)

        return min(1.0, log_count / log_base)

    def is_deprecated(
        self,
        led_to_success_count: int,
        led_to_failure_count: int,
    ) -> bool:
        """Check if a pattern should be deprecated due to low effectiveness.

        A pattern is deprecated if:
        1. It has enough application data (>= min_applications)
        2. Its effectiveness is below the threshold

        Args:
            led_to_success_count: Times pattern led to success.
            led_to_failure_count: Times pattern led to failure.

        Returns:
            True if pattern should be deprecated.
        """
        total = led_to_success_count + led_to_failure_count
        if total < self.config.min_applications_for_effectiveness:
            # Not enough data to deprecate
            return False

        effectiveness = self.calculate_effectiveness(
            led_to_success_count,
            led_to_failure_count,
        )

        return effectiveness < self.config.effectiveness_threshold

    def classify_uncertainty(self, variance: float) -> str:
        """Classify the uncertainty type of a pattern.

        Epistemic uncertainty (variance < threshold): The pattern behavior
        can be learned with more data. Keep tracking and applying.

        Aleatoric uncertainty (variance >= threshold): The pattern has
        inherently unpredictable outcomes. Deprioritize but don't remove.

        Args:
            variance: The variance of pattern application outcomes.

        Returns:
            'epistemic' or 'aleatoric'
        """
        if variance < self.config.epistemic_threshold:
            return "epistemic"
        return "aleatoric"

    def calculate_variance(self, outcomes: list[bool]) -> float:
        """Calculate variance from a list of boolean outcomes.

        Used to track consistency of pattern applications.

        Args:
            outcomes: List of True (success) / False (failure) outcomes.

        Returns:
            Variance from 0.0 (all same) to 0.25 (50/50 split).
        """
        if len(outcomes) < 2:
            return 0.0

        mean = sum(1 if o else 0 for o in outcomes) / len(outcomes)
        # Need parentheses around the subtraction to get correct variance
        variance = sum(((1 if o else 0) - mean) ** 2 for o in outcomes) / len(outcomes)

        return variance

    def update_pattern_priority(
        self,
        pattern: WeightablePattern,
        now: datetime | None = None,
    ) -> float:
        """Calculate updated priority for a pattern object.

        Convenience method that extracts fields from a pattern object.

        Args:
            pattern: A pattern implementing WeightablePattern protocol.
            now: Current time for recency calculation.

        Returns:
            Updated priority score.
        """
        return self.calculate_priority(
            occurrence_count=pattern.occurrence_count,
            led_to_success_count=pattern.led_to_success_count,
            led_to_failure_count=pattern.led_to_failure_count,
            last_confirmed=pattern.last_confirmed,
            variance=pattern.variance,
            now=now,
        )


def calculate_priority(
    occurrence_count: int,
    led_to_success_count: int,
    led_to_failure_count: int,
    last_confirmed: datetime,
    variance: float = 0.0,
    config: WeightingConfig | None = None,
) -> float:
    """Convenience function to calculate priority without creating a weighter.

    Args:
        occurrence_count: How many times this pattern was observed.
        led_to_success_count: Times pattern led to success when applied.
        led_to_failure_count: Times pattern led to failure when applied.
        last_confirmed: When this pattern was last confirmed effective.
        variance: Standard deviation of pattern application outcomes.
        config: Optional weighting configuration.

    Returns:
        Priority score from 0.0 to 1.0.
    """
    weighter = PatternWeighter(config)
    return weighter.calculate_priority(
        occurrence_count=occurrence_count,
        led_to_success_count=led_to_success_count,
        led_to_failure_count=led_to_failure_count,
        last_confirmed=last_confirmed,
        variance=variance,
    )
