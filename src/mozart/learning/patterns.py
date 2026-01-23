"""Pattern detection and application for learning from outcomes.

This module implements the "Close the Learning Loop" evolution:
- PatternDetector: Analyzes outcomes to detect recurring patterns
- PatternMatcher: Matches patterns to current execution context
- PatternApplicator: Generates prompt modifications from patterns

The pattern system enables Mozart to learn from past executions
and apply that learning to improve future sheet executions.
"""

import re
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

    OUTPUT_PATTERN = "output_pattern"
    """Pattern extracted from stdout/stderr output during execution.

    These patterns are detected by analyzing the raw output text for common
    error signatures, stack traces, and failure indicators. Useful for learning
    from execution-level failures that may not be captured by validation.
    """


@dataclass
class ExtractedPattern:
    """A pattern extracted from stdout/stderr output analysis.

    Represents a specific error or failure pattern found in execution output,
    with context about where it appeared and confidence in the detection.
    """

    pattern_name: str
    """Canonical name for this pattern (e.g., 'rate_limit', 'import_error')."""

    matched_text: str
    """The actual text that matched the pattern."""

    line_number: int
    """Line number in the output where pattern was found."""

    context_before: list[str] = field(default_factory=list)
    """Lines of context before the match (up to 2 lines)."""

    context_after: list[str] = field(default_factory=list)
    """Lines of context after the match (up to 2 lines)."""

    confidence: float = 0.8
    """Confidence in this pattern detection (0.0-1.0)."""

    source: str = "stdout"
    """Source of the pattern: 'stdout' or 'stderr'."""


@dataclass
class DetectedPattern:
    """A pattern detected from historical outcomes.

    Patterns are learned behaviors that can inform future executions.
    They include both positive patterns (what works) and negative
    patterns (what to avoid).

    v19 Evolution: Extended with optional quarantine_status and trust_score
    fields for integration with Pattern Quarantine & Trust Scoring features.
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

    # Effectiveness tracking fields (Evolution: Pattern Effectiveness)
    applications: int = 0
    """Number of times this pattern was applied (included in prompts)."""

    successes_after_application: int = 0
    """Number of first_attempt_success outcomes when this pattern was applied."""

    # v19: Quarantine & Trust fields (optional, from global store)
    quarantine_status: str | None = None
    """Quarantine status from global store: pending, quarantined, validated, retired."""

    trust_score: float | None = None
    """Trust score (0.0-1.0) from global store. None if not from global store."""

    @property
    def effectiveness_rate(self) -> float:
        """Compute effectiveness rate from applications and successes.

        Returns:
            Effectiveness rate (0.0-1.0). Returns 0.4 (slightly below neutral)
            when applications < 3 to prefer proven patterns over unproven ones.
            This prevents unproven patterns from being treated equally with
            patterns that have demonstrated moderate (50%) success.
        """
        if self.applications < 3:
            return 0.4  # Slightly below neutral to prefer proven patterns
        return self.successes_after_application / self.applications

    @property
    def effectiveness_weight(self) -> float:
        """Compute weight for blending effectiveness into relevance scoring.

        Uses gradual ramp-up: full weight only after 5 applications.
        This prevents new patterns from being over-weighted.

        Returns:
            Weight (0.0-1.0) based on sample size.
        """
        return min(self.applications / 5.0, 1.0)

    @property
    def is_quarantined(self) -> bool:
        """Check if pattern is in quarantine status.

        v19 Evolution: Used for quarantine-aware scoring.
        """
        return self.quarantine_status == "quarantined"

    @property
    def is_validated(self) -> bool:
        """Check if pattern is in validated status.

        v19 Evolution: Used for trust-aware scoring.
        """
        return self.quarantine_status == "validated"

    def to_prompt_guidance(self) -> str:
        """Format this pattern as guidance for prompts.

        v19 Evolution: Now includes quarantine/trust context when available.

        Returns:
            A concise string suitable for injection into prompts.
        """
        # v19: Add trust indicator if available
        trust_indicator = ""
        if self.trust_score is not None:
            if self.trust_score >= 0.7:
                trust_indicator = " [High trust]"
            elif self.trust_score <= 0.3:
                trust_indicator = " [Low trust]"
            else:
                trust_indicator = f" [Trust: {self.trust_score:.0%}]"

        # v19: Add quarantine warning if applicable
        if self.is_quarantined:
            return f"⚠️ [QUARANTINED] {self.description}{trust_indicator}"

        if self.pattern_type == PatternType.VALIDATION_FAILURE:
            return f"⚠️ Common issue: {self.description} (seen {self.frequency}x){trust_indicator}"
        elif self.pattern_type == PatternType.RETRY_SUCCESS:
            rate = f"works {self.success_rate:.0%} of the time"
            return f"✓ Tip: {self.description} ({rate}){trust_indicator}"
        elif self.pattern_type == PatternType.COMPLETION_MODE:
            return f"📝 Partial completion: {self.description}{trust_indicator}"
        elif self.pattern_type == PatternType.FIRST_ATTEMPT_SUCCESS:
            return f"✓ Best practice: {self.description}{trust_indicator}"
        elif self.pattern_type == PatternType.LOW_CONFIDENCE:
            return f"⚠️ Needs attention: {self.description}{trust_indicator}"
        elif self.pattern_type == PatternType.SEMANTIC_FAILURE:
            seen = f"seen {self.frequency}x"
            return f"🔍 Semantic insight: {self.description} ({seen}){trust_indicator}"
        else:
            return f"{self.description}{trust_indicator}"


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
        patterns.extend(self._detect_error_code_patterns())

        # Calculate effectiveness for each pattern from outcomes
        self._calculate_effectiveness(patterns)

        # Sort by confidence (highest first)
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        return patterns

    def _calculate_effectiveness(self, patterns: list[DetectedPattern]) -> None:
        """Calculate effectiveness metrics for patterns based on outcomes.

        For each pattern, counts how many times it was applied (via patterns_applied)
        and how many of those applications resulted in first_attempt_success.

        This creates the feedback loop: patterns that lead to success get
        higher effectiveness_rate, which then influences relevance scoring.

        Args:
            patterns: List of patterns to update with effectiveness data.
        """
        # Build a lookup from pattern description to pattern object
        # (patterns are matched by their prompt guidance string)
        pattern_lookup: dict[str, DetectedPattern] = {}
        for pattern in patterns:
            guidance = pattern.to_prompt_guidance()
            pattern_lookup[guidance] = pattern

        # Scan outcomes for patterns_applied
        for outcome in self.outcomes:
            for applied_desc in outcome.patterns_applied:
                if applied_desc in pattern_lookup:
                    pattern = pattern_lookup[applied_desc]
                    pattern.applications += 1
                    if outcome.first_attempt_success:
                        pattern.successes_after_application += 1

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

    def _detect_error_code_patterns(self) -> list[DetectedPattern]:
        """Detect patterns from error codes in outcomes.

        Analyzes error_code and error_history fields in SheetOutcome to identify
        recurring execution errors that can inform retry strategies and error
        handling improvements.

        Error codes are categorized (E0xx, E1xx, etc.) and aggregated across
        outcomes to identify recurring issues.

        Returns:
            List of error code patterns.
        """
        patterns: list[DetectedPattern] = []

        # Aggregate error codes across outcomes
        error_counts: dict[str, int] = {}
        error_evidence: dict[str, list[str]] = {}
        error_categories: dict[str, int] = {}

        for outcome in self.outcomes:
            # Extract error codes from error_history if available
            error_history = getattr(outcome, "error_history", None)
            if error_history and isinstance(error_history, list):
                for error_entry in error_history:
                    if isinstance(error_entry, dict):
                        error_code = error_entry.get("error_code")
                        if error_code:
                            error_counts[error_code] = (
                                error_counts.get(error_code, 0) + 1
                            )
                            if error_code not in error_evidence:
                                error_evidence[error_code] = []
                            if outcome.sheet_id not in error_evidence[error_code]:
                                error_evidence[error_code].append(outcome.sheet_id)

                            # Track by category (first 2 chars after E)
                            if error_code.startswith("E") and len(error_code) >= 3:
                                category = error_code[:3] + "x"  # E00x, E10x, etc.
                                error_categories[category] = (
                                    error_categories.get(category, 0) + 1
                                )

            # Also check validation_results for error_code fields
            for vr in outcome.validation_results:
                error_code = vr.get("error_code")
                if error_code:
                    error_counts[error_code] = error_counts.get(error_code, 0) + 1
                    if error_code not in error_evidence:
                        error_evidence[error_code] = []
                    if outcome.sheet_id not in error_evidence[error_code]:
                        error_evidence[error_code].append(outcome.sheet_id)

        # Create patterns for recurring error codes (seen >= 2 times)
        for code, count in error_counts.items():
            if count >= 2:
                patterns.append(
                    DetectedPattern(
                        pattern_type=PatternType.VALIDATION_FAILURE,
                        description=f"Error code '{code}' occurs frequently ({count}x)",
                        frequency=count,
                        success_rate=0.0,
                        context_tags=[f"error_code:{code}"],
                        evidence=error_evidence.get(code, [])[:5],
                        confidence=min(0.85, 0.5 + (count * 0.1)),
                    )
                )

        # Create patterns for error categories (aggregated)
        for category, count in error_categories.items():
            if count >= 3:
                patterns.append(
                    DetectedPattern(
                        pattern_type=PatternType.VALIDATION_FAILURE,
                        description=f"Error category '{category}' is common ({count}x)",
                        frequency=count,
                        success_rate=0.0,
                        context_tags=[f"error_category:{category}"],
                        confidence=min(0.80, 0.45 + (count * 0.08)),
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

        # Common patterns to extract - ordered by specificity (more specific first)
        patterns = [
            "file not created",
            "file not modified",
            "pattern not found",
            "content empty",
            "content missing",
            "command failed",
            "timeout",
            "permission denied",
            "rate limit",
            "connection refused",
            "connection failed",
            "connection",
            "not found",
            "syntax error",
            "import error",
            "type error",
            "authentication",
            "authorization",
            "access denied",
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

        v19 Evolution: Now includes quarantine penalty and trust bonus/penalty.
        - Quarantined patterns get -0.3 score penalty
        - Validated patterns get +0.1 bonus
        - High trust (>0.7) patterns get +0.1 to +0.2 bonus
        - Low trust (<0.3) patterns get -0.1 to -0.2 penalty

        Args:
            pattern: The pattern to score.
            context: Current execution context.

        Returns:
            Relevance score (0.0-1.0). Higher = more relevant.
        """
        score = 0.0

        # Weights normalized to sum to ~1.0:
        # confidence=0.25, frequency=0.20, context_tags=0.25,
        # recency=0.15, effectiveness=0.15 (max)
        # v19: quarantine/trust adjustments are additive modifiers

        # Base score from pattern confidence
        score += pattern.confidence * 0.25

        # Score from frequency (more frequent = more relevant)
        frequency_score = min(1.0, pattern.frequency / 10.0)
        score += frequency_score * 0.20

        # Score from context tag matching
        context_tags = context.get("tags", [])
        if isinstance(context_tags, list):
            matching_tags = len(set(pattern.context_tags) & set(context_tags))
            if matching_tags > 0:
                score += min(1.0, matching_tags * 0.2) * 0.25

        # Score from recency (more recent = more relevant)
        # Exponential decay: 50% weight loss per week
        age_days = (datetime.now() - pattern.last_seen).days
        recency_score = 0.5 ** (age_days / 7)
        score += recency_score * 0.15

        # Score from effectiveness (Evolution: Pattern Effectiveness)
        # Blend effectiveness into score, weighted by sample size
        # effectiveness_weight = min(applications / 5, 1.0)
        # effectiveness_rate = successes / applications (or 0.4 if < 3 samples)
        effectiveness_boost = (
            pattern.effectiveness_rate * pattern.effectiveness_weight * 0.15
        )
        score += effectiveness_boost

        # v19: Quarantine status adjustments
        if pattern.is_quarantined:
            score -= 0.3  # Significant penalty for quarantined patterns
        elif pattern.is_validated:
            score += 0.1  # Bonus for validated patterns

        # v19: Trust score adjustments
        if pattern.trust_score is not None:
            if pattern.trust_score >= 0.7:
                # High trust bonus: +0.1 to +0.2 based on trust level
                trust_bonus = 0.1 + (pattern.trust_score - 0.7) * 0.33
                score += trust_bonus
            elif pattern.trust_score <= 0.3:
                # Low trust penalty: -0.1 to -0.2 based on trust level
                trust_penalty = 0.1 + (0.3 - pattern.trust_score) * 0.33
                score -= trust_penalty

        return float(max(0.0, min(1.0, score)))


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


class OutputPatternExtractor:
    """Extracts patterns from stdout/stderr output for learning.

    Analyzes execution output to detect common failure patterns, error signatures,
    and other indicators that can inform future executions. This enables Mozart
    to learn from the raw output of failed executions, not just validation results.

    The extractor uses a dictionary of regex patterns to identify common error
    types like rate limits, import errors, permission denied, etc.
    """

    # Common failure patterns to detect in output
    # Each key is a pattern name, value is a tuple of (regex, confidence)
    FAILURE_PATTERNS: dict[str, tuple[str, float]] = {
        # Rate limiting and API errors
        "rate_limit": (
            r"(?i)(rate\s*limit|too\s*many\s*requests|429|quota\s*exceeded)",
            0.95,
        ),
        "api_error": (
            r"(?i)(api\s*error|service\s*unavailable|503|500\s*internal)",
            0.90,
        ),
        "timeout": (
            r"(?i)(timeout|timed?\s*out|connection\s*timed?\s*out|deadline\s*exceeded)",
            0.90,
        ),
        # Python-specific errors
        "import_error": (
            r"(?i)(ImportError|ModuleNotFoundError|No\s*module\s*named)",
            0.95,
        ),
        "syntax_error": (
            r"(?i)(SyntaxError|IndentationError|TabError)",
            0.95,
        ),
        "type_error": (
            r"(?i)(TypeError:\s*[^\n]+)",
            0.90,
        ),
        "attribute_error": (
            r"(?i)(AttributeError:\s*[^\n]+)",
            0.90,
        ),
        "key_error": (
            r"(?i)(KeyError:\s*[^\n]+)",
            0.85,
        ),
        "value_error": (
            r"(?i)(ValueError:\s*[^\n]+)",
            0.85,
        ),
        "name_error": (
            r"(?i)(NameError:\s*name\s*'[^']+'\s*is\s*not\s*defined)",
            0.95,
        ),
        # File system errors
        "permission_denied": (
            r"(?i)(permission\s*denied|access\s*denied|EACCES)",
            0.95,
        ),
        "file_not_found": (
            r"(?i)(FileNotFoundError|No\s*such\s*file|ENOENT)",
            0.95,
        ),
        "disk_full": (
            r"(?i)(no\s*space\s*left|disk\s*full|ENOSPC)",
            0.95,
        ),
        # Network errors
        "connection_refused": (
            r"(?i)(connection\s*refused|ECONNREFUSED)",
            0.90,
        ),
        "connection_reset": (
            r"(?i)(connection\s*reset|ECONNRESET)",
            0.85,
        ),
        "dns_error": (
            r"(?i)(could\s*not\s*resolve\s*host|dns\s*error|NXDOMAIN)",
            0.90,
        ),
        # Authentication errors
        "auth_failure": (
            r"(?i)(authentication\s*fail|invalid\s*credential|unauthorized|401)",
            0.90,
        ),
        "token_expired": (
            r"(?i)(token\s*expired|session\s*expired|jwt\s*expired)",
            0.90,
        ),
        # Generic errors with stack traces
        "traceback": (
            r"Traceback\s*\(most\s*recent\s*call\s*last\)",
            0.75,
        ),
        "assertion_error": (
            r"(?i)(AssertionError:\s*[^\n]*)",
            0.85,
        ),
        # Memory errors
        "out_of_memory": (
            r"(?i)(out\s*of\s*memory|MemoryError|OOM|killed\s*by\s*kernel)",
            0.95,
        ),
    }

    def __init__(self) -> None:
        """Initialize the output pattern extractor.

        Compiles all regex patterns for efficient matching.
        """
        self._compiled_patterns: dict[str, tuple[re.Pattern[str], float]] = {}
        for name, (pattern, confidence) in self.FAILURE_PATTERNS.items():
            self._compiled_patterns[name] = (re.compile(pattern), confidence)

    def extract_from_output(
        self,
        output: str,
        source: str = "stdout",
    ) -> list[ExtractedPattern]:
        """Extract patterns from execution output.

        Scans the output text for known failure patterns and returns
        a list of ExtractedPattern objects with context.

        Args:
            output: The stdout or stderr text to analyze.
            source: Source identifier ('stdout' or 'stderr').

        Returns:
            List of extracted patterns found in the output.
        """
        if not output or not output.strip():
            return []

        patterns: list[ExtractedPattern] = []
        lines = output.splitlines()

        for name, (compiled_pattern, confidence) in self._compiled_patterns.items():
            for match in compiled_pattern.finditer(output):
                # Calculate line number
                line_num = output[:match.start()].count('\n') + 1

                # Get context lines
                context_before, context_after = self._get_line_context(
                    lines, line_num - 1  # Convert to 0-indexed
                )

                patterns.append(
                    ExtractedPattern(
                        pattern_name=name,
                        matched_text=match.group(0),
                        line_number=line_num,
                        context_before=context_before,
                        context_after=context_after,
                        confidence=confidence,
                        source=source,
                    )
                )

        # Sort by line number to maintain order of occurrence
        patterns.sort(key=lambda p: p.line_number)

        # Deduplicate patterns with same name at same line
        seen: set[tuple[str, int]] = set()
        deduped: list[ExtractedPattern] = []
        for p in patterns:
            key = (p.pattern_name, p.line_number)
            if key not in seen:
                seen.add(key)
                deduped.append(p)

        return deduped

    def _get_line_context(
        self,
        lines: list[str],
        line_index: int,
        context_size: int = 2,
    ) -> tuple[list[str], list[str]]:
        """Get context lines before and after a given line.

        Args:
            lines: All lines of the output.
            line_index: Index of the line (0-based).
            context_size: Number of context lines to include.

        Returns:
            Tuple of (lines_before, lines_after).
        """
        start = max(0, line_index - context_size)
        end = min(len(lines), line_index + context_size + 1)

        before = lines[start:line_index] if line_index > 0 else []
        after = lines[line_index + 1:end] if line_index < len(lines) - 1 else []

        return before, after

    def get_pattern_summary(
        self,
        patterns: list[ExtractedPattern],
    ) -> dict[str, int]:
        """Get a summary count of pattern types found.

        Args:
            patterns: List of extracted patterns.

        Returns:
            Dict mapping pattern_name to occurrence count.
        """
        summary: dict[str, int] = {}
        for p in patterns:
            summary[p.pattern_name] = summary.get(p.pattern_name, 0) + 1
        return summary
