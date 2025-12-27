"""Pattern detection and application for learning from outcomes.

This module implements the "Close the Learning Loop" evolution:
- PatternDetector: Analyzes outcomes to detect recurring patterns
- PatternMatcher: Matches patterns to current execution context
- PatternApplicator: Generates prompt modifications from patterns

The pattern system enables Mozart to learn from past executions
and apply that learning to improve future sheet executions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from mozart.learning.outcomes import SheetOutcome


class PatternType(Enum):
    """Types of patterns that can be detected from outcomes."""

    VALIDATION_FAILURE = "validation_failure"
    """Recurring validation failure pattern (e.g., file not created)."""

    RETRY_SUCCESS = "retry_success"
    """Pattern where retry succeeds after specific failure."""

    COMPLETION_MODE = "completion_mode"
    """Pattern where completion mode is effective."""

    FIRST_ATTEMPT_SUCCESS = "first_attempt_success"
    """Pattern of successful first attempts (positive pattern)."""

    HIGH_CONFIDENCE = "high_confidence"
    """Pattern with high validation confidence."""

    LOW_CONFIDENCE = "low_confidence"
    """Pattern with low validation confidence (needs attention)."""

    SEMANTIC_FAILURE = "semantic_failure"
    """Pattern detected from semantic failure_reason/failure_category analysis.

    These patterns are extracted from the failure_reason and failure_category
    fields in ValidationResult, providing deeper insight into WHY failures occur.
    Examples:
    - 'stale' category appearing frequently (files not modified)
    - 'file not created' reason appearing across multiple sheets
    """


@dataclass
class DetectedPattern:
    """A pattern detected from historical outcomes.

    Patterns are learned behaviors that can inform future executions.
    They include both positive patterns (what works) and negative
    patterns (what to avoid).
    """

    pattern_type: PatternType
    description: str
    """Human-readable description of what this pattern represents."""

    frequency: int = 1
    """How often this pattern has been observed."""

    success_rate: float = 0.0
    """Rate at which this pattern leads to success (0.0-1.0)."""

    last_seen: datetime = field(default_factory=datetime.now)
    """When this pattern was last observed."""

    context_tags: list[str] = field(default_factory=list)
    """Tags for matching: job types, validation types, error categories."""

    evidence: list[str] = field(default_factory=list)
    """Sheet IDs that contributed to detecting this pattern."""

    confidence: float = 0.5
    """Confidence in this pattern (0.0-1.0). Higher = more reliable."""

    def to_prompt_guidance(self) -> str:
        """Format this pattern as guidance for prompts.

        Returns:
            A concise string suitable for injection into prompts.
        """
        if self.pattern_type == PatternType.VALIDATION_FAILURE:
            return f"⚠️ Common issue: {self.description} (seen {self.frequency}x)"
        elif self.pattern_type == PatternType.RETRY_SUCCESS:
            return f"✓ Tip: {self.description} (works {self.success_rate:.0%} of the time)"
        elif self.pattern_type == PatternType.COMPLETION_MODE:
            return f"📝 Partial completion: {self.description}"
        elif self.pattern_type == PatternType.FIRST_ATTEMPT_SUCCESS:
            return f"✓ Best practice: {self.description}"
        elif self.pattern_type == PatternType.LOW_CONFIDENCE:
            return f"⚠️ Needs attention: {self.description}"
        elif self.pattern_type == PatternType.SEMANTIC_FAILURE:
            return f"🔍 Semantic insight: {self.description} (seen {self.frequency}x)"
        else:
            return self.description


class PatternDetectorProtocol(Protocol):
    """Protocol for pattern detection implementations."""

    def detect_all(self) -> list[DetectedPattern]:
        """Detect all patterns from outcomes."""
        ...


class PatternMatcherProtocol(Protocol):
    """Protocol for pattern matching implementations."""

    def match(self, context: dict[str, Any]) -> list[DetectedPattern]:
        """Match patterns to a given context."""
        ...


class PatternDetector:
    """Detects patterns from historical sheet outcomes.

    Analyzes a collection of SheetOutcome objects to identify
    recurring patterns that can inform future executions.
    """

    def __init__(self, outcomes: list["SheetOutcome"]) -> None:
        """Initialize the pattern detector.

        Args:
            outcomes: List of historical sheet outcomes to analyze.
        """
        self.outcomes = outcomes

    def detect_all(self) -> list[DetectedPattern]:
        """Detect all pattern types from outcomes.

        Returns:
            List of detected patterns sorted by confidence.
        """
        patterns: list[DetectedPattern] = []

        # Detect various pattern types
        patterns.extend(self._detect_validation_patterns())
        patterns.extend(self._detect_retry_patterns())
        patterns.extend(self._detect_completion_patterns())
        patterns.extend(self._detect_success_patterns())
        patterns.extend(self._detect_confidence_patterns())
        patterns.extend(self._detect_semantic_patterns())

        # Sort by confidence (highest first)
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        return patterns

    def _detect_validation_patterns(self) -> list[DetectedPattern]:
        """Detect recurring validation failure patterns.

        Looks for validation types that fail frequently.

        Returns:
            List of validation failure patterns.
        """
        patterns: list[DetectedPattern] = []

        # Group validation failures by type
        failure_counts: dict[str, int] = {}
        failure_evidence: dict[str, list[str]] = {}

        for outcome in self.outcomes:
            if outcome.validation_pass_rate < 1.0:
                for vr in outcome.validation_results:
                    if not vr.get("passed", True):
                        vtype = vr.get("rule_type", "unknown")
                        failure_counts[vtype] = failure_counts.get(vtype, 0) + 1
                        if vtype not in failure_evidence:
                            failure_evidence[vtype] = []
                        failure_evidence[vtype].append(outcome.sheet_id)

        # Create patterns for recurring failures (seen >= 2 times)
        for vtype, count in failure_counts.items():
            if count >= 2:
                patterns.append(
                    DetectedPattern(
                        pattern_type=PatternType.VALIDATION_FAILURE,
                        description=f"'{vtype}' validation tends to fail",
                        frequency=count,
                        success_rate=0.0,
                        context_tags=[f"validation:{vtype}"],
                        evidence=failure_evidence.get(vtype, [])[:5],
                        confidence=min(0.9, 0.5 + (count * 0.1)),
                    )
                )

        return patterns

    def _detect_retry_patterns(self) -> list[DetectedPattern]:
        """Detect patterns where retries are successful.

        Identifies conditions under which retrying leads to success.

        Returns:
            List of retry success patterns.
        """
        patterns: list[DetectedPattern] = []

        # Find outcomes with retries that eventually succeeded
        retry_successes = [
            o for o in self.outcomes
            if o.retry_count > 0 and o.validation_pass_rate == 1.0
        ]

        if len(retry_successes) >= 2:
            # Calculate average retries needed
            avg_retries = sum(o.retry_count for o in retry_successes) / len(retry_successes)
            patterns.append(
                DetectedPattern(
                    pattern_type=PatternType.RETRY_SUCCESS,
                    description=f"Retrying works: avg {avg_retries:.1f} retries lead to success",
                    frequency=len(retry_successes),
                    success_rate=1.0,
                    context_tags=["retry:effective"],
                    evidence=[o.sheet_id for o in retry_successes[:5]],
                    confidence=min(0.8, 0.5 + (len(retry_successes) * 0.1)),
                )
            )

        return patterns

    def _detect_completion_patterns(self) -> list[DetectedPattern]:
        """Detect patterns related to completion mode usage.

        Identifies when completion mode is effective.

        Returns:
            List of completion mode patterns.
        """
        patterns: list[DetectedPattern] = []

        # Find outcomes where completion mode was used
        completion_used = [o for o in self.outcomes if o.completion_mode_used]

        if len(completion_used) >= 2:
            success_in_completion = [
                o for o in completion_used
                if o.validation_pass_rate == 1.0
            ]
            success_rate = (
                len(success_in_completion) / len(completion_used)
                if completion_used else 0.0
            )

            patterns.append(
                DetectedPattern(
                    pattern_type=PatternType.COMPLETION_MODE,
                    description=f"Completion mode used {len(completion_used)}x "
                               f"({success_rate:.0%} success rate)",
                    frequency=len(completion_used),
                    success_rate=success_rate,
                    context_tags=["completion:used"],
                    evidence=[o.sheet_id for o in completion_used[:5]],
                    confidence=min(0.7, 0.4 + (len(completion_used) * 0.1)),
                )
            )

        return patterns

    def _detect_success_patterns(self) -> list[DetectedPattern]:
        """Detect positive patterns from successful first attempts.

        Identifies what leads to first-attempt success.

        Returns:
            List of first-attempt success patterns.
        """
        patterns: list[DetectedPattern] = []

        # Find first-attempt successes
        first_attempt_successes = [
            o for o in self.outcomes if o.first_attempt_success
        ]

        if len(first_attempt_successes) >= 3:
            success_rate = len(first_attempt_successes) / len(self.outcomes)
            patterns.append(
                DetectedPattern(
                    pattern_type=PatternType.FIRST_ATTEMPT_SUCCESS,
                    description=f"First-attempt success rate: {success_rate:.0%}",
                    frequency=len(first_attempt_successes),
                    success_rate=success_rate,
                    context_tags=["success:first_attempt"],
                    evidence=[o.sheet_id for o in first_attempt_successes[:5]],
                    confidence=min(0.9, 0.6 + (success_rate * 0.3)),
                )
            )

        return patterns

    def _detect_confidence_patterns(self) -> list[DetectedPattern]:
        """Detect patterns related to validation confidence.

        Identifies low-confidence validations that need attention.

        Returns:
            List of confidence-related patterns.
        """
        patterns: list[DetectedPattern] = []

        # Analyze confidence factors from validation results
        low_confidence_validations: list[dict[str, Any]] = []

        for outcome in self.outcomes:
            for vr in outcome.validation_results:
                confidence = vr.get("confidence", 1.0)
                if confidence < 0.7:
                    low_confidence_validations.append(vr)

        if len(low_confidence_validations) >= 2:
            patterns.append(
                DetectedPattern(
                    pattern_type=PatternType.LOW_CONFIDENCE,
                    description=f"{len(low_confidence_validations)} validations "
                               "have low confidence - review needed",
                    frequency=len(low_confidence_validations),
                    success_rate=0.0,
                    context_tags=["confidence:low"],
                    confidence=0.6,
                )
            )

        return patterns

    def _detect_semantic_patterns(self) -> list[DetectedPattern]:
        """Detect patterns from semantic failure_reason and failure_category.

        Analyzes the semantic fields in ValidationResult and SheetOutcome
        to identify recurring failure patterns with more context about WHY
        failures occur, not just THAT they occur.

        This enables learning like:
        - "stale" failures appear 80% of the time for Sheet X
        - "file not created" reason is most common
        - Certain fix suggestions are effective

        Returns:
            List of semantic failure patterns.
        """
        patterns: list[DetectedPattern] = []

        # Aggregate failure categories across all outcomes
        category_counts: dict[str, int] = {}
        category_evidence: dict[str, list[str]] = {}

        # Aggregate semantic patterns (failure_reason keywords)
        reason_counts: dict[str, int] = {}
        reason_evidence: dict[str, list[str]] = {}

        # Aggregate fix suggestions
        fix_counts: dict[str, int] = {}

        for outcome in self.outcomes:
            # Use the pre-aggregated fields if available
            if outcome.failure_category_counts:
                for cat, count in outcome.failure_category_counts.items():
                    category_counts[cat] = category_counts.get(cat, 0) + count
                    if cat not in category_evidence:
                        category_evidence[cat] = []
                    category_evidence[cat].append(outcome.sheet_id)

            if outcome.semantic_patterns:
                for pattern in outcome.semantic_patterns:
                    reason_counts[pattern] = reason_counts.get(pattern, 0) + 1
                    if pattern not in reason_evidence:
                        reason_evidence[pattern] = []
                    reason_evidence[pattern].append(outcome.sheet_id)

            if outcome.fix_suggestions:
                for fix in outcome.fix_suggestions:
                    fix_counts[fix] = fix_counts.get(fix, 0) + 1

            # Also analyze raw validation_results for backward compatibility
            for vr in outcome.validation_results:
                if vr.get("passed", True):
                    continue

                # Extract failure_category
                cat_value = vr.get("failure_category")
                if cat_value and isinstance(cat_value, str):
                    category_counts[cat_value] = category_counts.get(cat_value, 0) + 1
                    if cat_value not in category_evidence:
                        category_evidence[cat_value] = []
                    if outcome.sheet_id not in category_evidence[cat_value]:
                        category_evidence[cat_value].append(outcome.sheet_id)

                # Extract normalized keywords from failure_reason
                reason = vr.get("failure_reason")
                if reason:
                    # Simple keyword extraction: lowercase, common phrases
                    normalized = self._normalize_failure_reason(reason)
                    if normalized:
                        reason_counts[normalized] = reason_counts.get(normalized, 0) + 1
                        if normalized not in reason_evidence:
                            reason_evidence[normalized] = []
                        if outcome.sheet_id not in reason_evidence[normalized]:
                            reason_evidence[normalized].append(outcome.sheet_id)

                # Track fix suggestions
                fix_value = vr.get("suggested_fix")
                if fix_value and isinstance(fix_value, str):
                    fix_counts[fix_value] = fix_counts.get(fix_value, 0) + 1

        # Create patterns for recurring failure categories (seen >= 2 times)
        for cat, count in category_counts.items():
            if count >= 2:
                patterns.append(
                    DetectedPattern(
                        pattern_type=PatternType.SEMANTIC_FAILURE,
                        description=f"'{cat}' failures are common ({count}x)",
                        frequency=count,
                        success_rate=0.0,
                        context_tags=[f"failure_category:{cat}"],
                        evidence=category_evidence.get(cat, [])[:5],
                        confidence=min(0.85, 0.5 + (count * 0.1)),
                    )
                )

        # Create patterns for recurring failure reasons (seen >= 2 times)
        for reason, count in reason_counts.items():
            if count >= 2 and reason:
                patterns.append(
                    DetectedPattern(
                        pattern_type=PatternType.SEMANTIC_FAILURE,
                        description=f"Recurring issue: '{reason}' ({count}x)",
                        frequency=count,
                        success_rate=0.0,
                        context_tags=[f"failure_reason:{reason}"],
                        evidence=reason_evidence.get(reason, [])[:5],
                        confidence=min(0.80, 0.45 + (count * 0.1)),
                    )
                )

        # Create pattern for effective fix suggestions (seen >= 3 times)
        for fix, count in fix_counts.items():
            if count >= 3:
                patterns.append(
                    DetectedPattern(
                        pattern_type=PatternType.SEMANTIC_FAILURE,
                        description=f"Suggested fix: '{fix}' (recommended {count}x)",
                        frequency=count,
                        success_rate=0.5,  # Suggestions, not proven
                        context_tags=["fix_suggestion"],
                        confidence=min(0.70, 0.4 + (count * 0.05)),
                    )
                )

        return patterns

    def _normalize_failure_reason(self, reason: str) -> str:
        """Normalize a failure_reason string to a canonical form for aggregation.

        Extracts key phrases and normalizes to lowercase for comparison.

        Args:
            reason: The raw failure_reason string.

        Returns:
            Normalized string suitable for aggregation, or empty if not useful.
        """
        if not reason:
            return ""

        # Lowercase and trim
        normalized = reason.lower().strip()

        # Common patterns to extract
        patterns = [
            "file not created",
            "file not modified",
            "pattern not found",
            "content empty",
            "content missing",
            "command failed",
            "timeout",
            "permission denied",
            "not found",
            "syntax error",
            "import error",
            "type error",
        ]

        for pattern in patterns:
            if pattern in normalized:
                return pattern

        # If no common pattern matches, return first 50 chars as-is
        # This prevents aggregating very specific error messages
        if len(normalized) > 50:
            return ""  # Too specific to aggregate

        return normalized

    @staticmethod
    def calculate_success_rate(outcomes: list["SheetOutcome"]) -> float:
        """Calculate overall success rate from outcomes.

        Args:
            outcomes: List of sheet outcomes.

        Returns:
            Success rate as a float (0.0-1.0).
        """
        if not outcomes:
            return 0.0

        successful = sum(1 for o in outcomes if o.validation_pass_rate == 1.0)
        return successful / len(outcomes)


class PatternMatcher:
    """Matches detected patterns to execution context.

    Given a set of detected patterns and a current execution context,
    finds patterns that are relevant to the current situation.
    """

    def __init__(self, patterns: list[DetectedPattern]) -> None:
        """Initialize the pattern matcher.

        Args:
            patterns: List of detected patterns to match against.
        """
        self.patterns = patterns

    def match(
        self,
        context: dict[str, Any],
        limit: int = 5,
    ) -> list[DetectedPattern]:
        """Find patterns relevant to the given context.

        Args:
            context: Context dict with job_id, sheet_num, validation_types, etc.
            limit: Maximum number of patterns to return.

        Returns:
            List of matching patterns sorted by relevance.
        """
        scored_patterns: list[tuple[float, DetectedPattern]] = []

        for pattern in self.patterns:
            score = self._score_relevance(pattern, context)
            if score > 0:
                scored_patterns.append((score, pattern))

        # Sort by score (highest first) and return top N
        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored_patterns[:limit]]

    def _score_relevance(
        self,
        pattern: DetectedPattern,
        context: dict[str, Any],
    ) -> float:
        """Score how relevant a pattern is to the context.

        Args:
            pattern: The pattern to score.
            context: Current execution context.

        Returns:
            Relevance score (0.0-1.0). Higher = more relevant.
        """
        score = 0.0

        # Base score from pattern confidence
        score += pattern.confidence * 0.3

        # Score from frequency (more frequent = more relevant)
        frequency_score = min(1.0, pattern.frequency / 10.0)
        score += frequency_score * 0.2

        # Score from context tag matching
        context_tags = context.get("tags", [])
        if isinstance(context_tags, list):
            matching_tags = len(set(pattern.context_tags) & set(context_tags))
            if matching_tags > 0:
                score += min(1.0, matching_tags * 0.2) * 0.3

        # Score from recency (more recent = more relevant)
        # Exponential decay: 50% weight loss per week
        age_days = (datetime.now() - pattern.last_seen).days
        recency_score = 0.5 ** (age_days / 7)
        score += recency_score * 0.2

        return float(min(1.0, score))


class PatternApplicator:
    """Applies patterns to modify prompts for better execution.

    Takes matched patterns and generates prompt modifications
    that incorporate learned insights.
    """

    def __init__(self, patterns: list[DetectedPattern]) -> None:
        """Initialize the pattern applicator.

        Args:
            patterns: List of patterns to apply.
        """
        self.patterns = patterns

    def generate_prompt_section(self) -> str:
        """Generate a prompt section from patterns.

        Returns:
            Formatted markdown section for prompt injection.
        """
        if not self.patterns:
            return ""

        lines = ["## Learned Patterns", ""]
        lines.append(
            "Based on previous executions, here are relevant insights:"
        )
        lines.append("")

        for i, pattern in enumerate(self.patterns[:5], 1):
            guidance = pattern.to_prompt_guidance()
            lines.append(f"{i}. {guidance}")

        lines.append("")
        lines.append("Consider these patterns when executing this sheet.")
        lines.append("")

        return "\n".join(lines)

    def get_pattern_descriptions(self) -> list[str]:
        """Get pattern descriptions as a list of strings.

        Returns:
            List of pattern guidance strings.
        """
        return [p.to_prompt_guidance() for p in self.patterns[:5]]
