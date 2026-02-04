"""Pattern management mixin for JobRunner.

Contains methods for querying, applying, and recording feedback on
patterns from both the local outcome store and the global learning store.
This module implements the Learning Activation pattern that bridges
stored knowledge with prompt injection.

Architecture:
    PatternsMixin is mixed with other mixins to compose the full JobRunner.
    It expects the following attributes from base.py:

    Required attributes:
        - config: JobConfig
        - _logger: MozartLogger
        - _global_learning_store: GlobalLearningStore | None
        - outcome_store: OutcomeStore | None
        - _exploration_pattern_ids: list[str]
        - _exploitation_pattern_ids: list[str]
        - console: Console

    Provides methods:
        - _query_relevant_patterns(): Query patterns for injection
        - _record_pattern_feedback(): Record pattern application outcomes
        - _assess_failure_risk(): Assess failure risk from historical data

    Note: _aggregate_to_global_store() is defined in LifecycleMixin since
    it's called during job finalization, not per-sheet execution.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mozart.learning.global_store import GlobalLearningStore

from mozart.core.config import JobConfig
from mozart.core.logging import MozartLogger
from mozart.learning.outcomes import OutcomeStore


class PatternsMixin:
    """Mixin providing pattern management for JobRunner.

    This mixin implements the Learning Activation pattern, bridging:
    - Global pattern store with prompt injection
    - Pattern application with feedback recording
    - Historical data with failure risk assessment

    Pattern Application Flow:
        1. _query_relevant_patterns() queries global store
        2. Patterns are injected into sheet prompts
        3. After execution, _record_pattern_feedback() records outcomes
        4. On job completion, _aggregate_to_global_store() aggregates all

    Epsilon-Greedy Exploration:
        The pattern application uses epsilon-greedy exploration to balance
        exploitation of proven patterns with exploration of untested ones.
        This addresses the cold-start problem where new patterns never get
        applied because they have low initial scores.

    Key Attributes (from base.py):
        _exploration_pattern_ids: Patterns selected via exploration
        _exploitation_pattern_ids: Patterns selected via exploitation
    """

    # Type hints for attributes provided by base.py
    config: JobConfig
    _logger: MozartLogger
    _global_learning_store: "GlobalLearningStore | None"
    outcome_store: OutcomeStore | None
    _exploration_pattern_ids: list[str]
    _exploitation_pattern_ids: list[str]

    def _query_relevant_patterns(
        self,
        job_id: str,
        sheet_num: int,
        context_tags: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Query relevant patterns from the global learning store.

        Learning Activation: This method bridges the global learning store's
        pattern knowledge with the prompt injection system. It queries patterns
        that are relevant to the current execution context and formats them
        for injection into sheet prompts.

        Pattern Application v4: Implements epsilon-greedy exploration mode.
        When random() < exploration_rate, also queries lower-priority patterns
        (down to exploration_min_priority) to collect effectiveness data.
        This breaks the cold-start problem where patterns with low effectiveness
        scores never get applied and thus never improve.

        Args:
            job_id: Current job identifier for similar job matching.
            sheet_num: Current sheet number.
            context_tags: Optional tags for context-based filtering.

        Returns:
            Tuple of (pattern_descriptions, pattern_ids).
            - pattern_descriptions: Human-readable strings for prompt injection
            - pattern_ids: IDs for tracking pattern application outcomes
        """
        # Reset exploration tracking for this query
        self._exploration_pattern_ids = []
        self._exploitation_pattern_ids = []

        if self._global_learning_store is None:
            return [], []

        try:
            # Build context tags from execution context for filtering
            # This enables selective pattern retrieval based on current job type
            query_context_tags = context_tags or []
            if not query_context_tags:
                # Auto-generate tags from job context if not provided
                query_context_tags = [
                    f"sheet:{sheet_num}",
                ]
                # Add job name as tag for similar job matching
                if job_id:
                    query_context_tags.append(f"job:{job_id}")

            # Determine exploration vs exploitation mode
            exploration_rate = self.config.learning.exploration_rate
            exploration_min_priority = self.config.learning.exploration_min_priority
            is_exploration_mode = random.random() < exploration_rate

            # Standard exploitation threshold (proven patterns)
            exploitation_threshold = 0.3

            if is_exploration_mode:
                # EXPLORATION MODE: Include lower-priority patterns
                # Use the lower threshold to find untested/low-confidence patterns
                min_priority = exploration_min_priority
                self._logger.info(
                    "patterns.exploration_mode_triggered",
                    job_id=job_id,
                    sheet_num=sheet_num,
                    exploration_rate=exploration_rate,
                    min_priority=min_priority,
                    reason="collecting_effectiveness_data",
                )
            else:
                # EXPLOITATION MODE: Only proven high-priority patterns
                min_priority = exploitation_threshold

            # v22: Query auto-apply patterns first if enabled
            auto_apply_patterns: list[Any] = []
            auto_apply_config = self.config.learning.auto_apply
            if (
                auto_apply_config
                and auto_apply_config.enabled
                and self._global_learning_store
            ):
                auto_apply_patterns = self._global_learning_store.get_patterns_for_auto_apply(
                    trust_threshold=auto_apply_config.trust_threshold,
                    require_validated=auto_apply_config.require_validated_status,
                    limit=auto_apply_config.max_patterns_per_sheet,
                    context_tags=query_context_tags if query_context_tags else None,
                )
                if auto_apply_patterns and auto_apply_config.log_applications:
                    self._logger.info(
                        "patterns.auto_apply_selected",
                        job_id=job_id,
                        sheet_num=sheet_num,
                        auto_apply_count=len(auto_apply_patterns),
                        pattern_ids=[p.id for p in auto_apply_patterns],
                        trust_threshold=auto_apply_config.trust_threshold,
                    )

            # Query patterns from global store with context filtering
            patterns = self._global_learning_store.get_patterns(
                min_priority=min_priority,
                limit=5,
                context_tags=query_context_tags if query_context_tags else None,
            )

            # Merge auto-apply patterns with regular patterns, avoiding duplicates
            if auto_apply_patterns:
                auto_apply_ids = {p.id for p in auto_apply_patterns}
                # Filter out already-selected auto-apply patterns from regular query
                patterns = [p for p in patterns if p.id not in auto_apply_ids]
                # Auto-apply patterns come first (highest trust)
                patterns = auto_apply_patterns + patterns

            # If no patterns match with context filtering, fall back to unfiltered
            if not patterns:
                self._logger.debug(
                    "patterns.query_global_fallback",
                    job_id=job_id,
                    sheet_num=sheet_num,
                    context_tags=query_context_tags,
                    reason="no_patterns_matched_context",
                    exploration_mode=is_exploration_mode,
                )
                patterns = self._global_learning_store.get_patterns(
                    min_priority=min_priority,
                    limit=5,
                )

            if not patterns:
                return [], []

            # Format patterns for prompt injection
            descriptions: list[str] = []
            pattern_ids: list[str] = []

            # Track auto-applied pattern IDs for indicators
            auto_apply_ids = {p.id for p in auto_apply_patterns} if auto_apply_patterns else set()

            for pattern in patterns:
                # v22: Check if this is an auto-applied pattern
                is_auto_applied = pattern.id in auto_apply_ids

                # Categorize as exploration vs exploitation for tracking
                if is_auto_applied:
                    # Auto-applied patterns are a special category
                    self._exploitation_pattern_ids.append(pattern.id)
                    mode_indicator = "⚡"  # Auto-apply indicator
                elif pattern.priority_score < exploitation_threshold:
                    # This pattern would not have been selected without exploration
                    self._exploration_pattern_ids.append(pattern.id)
                    mode_indicator = "🔍"  # Exploration indicator
                else:
                    self._exploitation_pattern_ids.append(pattern.id)
                    mode_indicator = ""

                # Build description based on pattern type and effectiveness
                if pattern.effectiveness_score > 0.7:
                    effectiveness_indicator = "✓"
                elif pattern.effectiveness_score > 0.4:
                    effectiveness_indicator = "○"
                else:
                    effectiveness_indicator = "⚠"

                # v22: Include trust score for auto-applied patterns
                trust_info = f", trust={pattern.trust_score:.0%}" if is_auto_applied else ""

                # Format: [indicator] description (occurrence count)
                # Include mode indicator for exploration/auto-apply patterns
                desc = (
                    f"{mode_indicator}{effectiveness_indicator} "
                    f"{pattern.description or pattern.pattern_name} "
                    f"(seen {pattern.occurrence_count}x, "
                    f"{pattern.effectiveness_score:.0%} effective{trust_info})"
                )
                descriptions.append(desc)
                pattern_ids.append(pattern.id)

            self._logger.debug(
                "patterns.query_global",
                job_id=job_id,
                sheet_num=sheet_num,
                patterns_found=len(descriptions),
                pattern_ids=pattern_ids,
                context_tags_used=query_context_tags,
                exploration_mode=is_exploration_mode,
                exploration_patterns=len(self._exploration_pattern_ids),
                exploitation_patterns=len(self._exploitation_pattern_ids),
                auto_apply_patterns=len(auto_apply_ids),
            )

            return descriptions, pattern_ids

        except Exception as e:
            # Pattern query failure shouldn't block execution
            self._logger.warning(
                "patterns.query_global_failed",
                job_id=job_id,
                sheet_num=sheet_num,
                error=str(e),
            )
            return [], []

    def _assess_failure_risk(
        self,
        job_id: str,
        sheet_num: int,
    ) -> dict[str, Any]:
        """Assess failure risk based on historical execution data.

        Learning Activation: Analyzes past executions to assess the risk
        of failure for the current sheet, enabling proactive adjustments
        to retry strategy and confidence thresholds.

        Args:
            job_id: Current job identifier.
            sheet_num: Current sheet number.

        Returns:
            Dict with risk assessment:
            - risk_level: "low", "medium", "high", or "unknown"
            - confidence: Confidence in assessment (0.0-1.0)
            - factors: List of contributing factors
            - recommended_adjustments: Suggested parameter changes
        """
        if self._global_learning_store is None:
            return {
                "risk_level": "unknown",
                "confidence": 0.0,
                "factors": ["no global store available"],
                "recommended_adjustments": [],
            }

        try:
            stats = self._global_learning_store.get_execution_stats()
            first_attempt_rate = stats.get("first_attempt_success_rate", 0.0)
            total_executions = stats.get("total_executions", 0)

            # Assess risk based on historical success rate
            if total_executions < 10:
                risk_level = "unknown"
                confidence = 0.2
                factors = [f"insufficient data ({total_executions} executions)"]
            elif first_attempt_rate > 0.7:
                risk_level = "low"
                confidence = min(0.9, total_executions / 100)
                factors = [f"high first-attempt success rate ({first_attempt_rate:.0%})"]
            elif first_attempt_rate > 0.4:
                risk_level = "medium"
                confidence = min(0.8, total_executions / 100)
                factors = [f"moderate first-attempt success rate ({first_attempt_rate:.0%})"]
            else:
                risk_level = "high"
                confidence = min(0.9, total_executions / 100)
                factors = [f"low first-attempt success rate ({first_attempt_rate:.0%})"]

            # Check for active rate limits
            is_limited, wait_time = self._global_learning_store.is_rate_limited()
            if is_limited:
                risk_level = "high"
                factors.append(f"active rate limit (expires in {wait_time:.0f}s)")

            # Build recommendations based on risk level
            recommendations: list[str] = []
            if risk_level == "high":
                recommendations.append("consider increasing retry delays")
                recommendations.append("enable completion mode aggressively")
            elif risk_level == "medium":
                recommendations.append("monitor validation confidence closely")

            return {
                "risk_level": risk_level,
                "confidence": confidence,
                "factors": factors,
                "recommended_adjustments": recommendations,
            }

        except Exception as e:
            self._logger.warning(
                "risk_assessment.failed",
                job_id=job_id,
                sheet_num=sheet_num,
                error=str(e),
            )
            return {
                "risk_level": "unknown",
                "confidence": 0.0,
                "factors": [f"assessment failed: {e}"],
                "recommended_adjustments": [],
            }

    async def _record_pattern_feedback(
        self,
        pattern_ids: list[str],
        validation_passed: bool,
        first_attempt_success: bool,
        sheet_num: int,
        grounding_confidence: float | None = None,
        validation_types: list[str] | None = None,
        error_categories: list[str] | None = None,
        prior_sheet_status: str | None = None,
        retry_iteration: int = 0,
        escalation_was_pending: bool = False,
    ) -> None:
        """Record pattern application feedback to global learning store.

        This closes the pattern feedback loop by recording whether patterns
        that were applied to a sheet execution led to a successful outcome.
        Distinguishes between exploration and exploitation patterns for
        differential effectiveness calculation.

        v12 Evolution: Grounding→Pattern Feedback - now passes grounding_confidence
        to the global learning store, enabling grounding-weighted effectiveness.

        v22 Evolution: Metacognitive Pattern Reflection - now captures success
        factors (context conditions) when patterns succeed, enabling WHY analysis.

        Args:
            pattern_ids: List of pattern IDs that were applied.
            validation_passed: Whether the sheet validations passed.
            first_attempt_success: Whether the sheet succeeded on first attempt.
            sheet_num: Sheet number for logging context.
            grounding_confidence: Grounding confidence (0.0-1.0) from external validation.
                                 None if grounding hooks were not executed.
            validation_types: Validation types active (file, regex, artifact, etc.)
            error_categories: Error categories present in execution.
            prior_sheet_status: Status of prior sheet (completed, failed, skipped).
            retry_iteration: Which retry attempt this is (0 = first attempt).
            escalation_was_pending: Whether escalation was pending.
        """
        if self._global_learning_store is None or not pattern_ids:
            return

        # Determine outcome improvement:
        # - outcome_improved=True if validation passed AND first_attempt
        # - outcome_improved=False if validation failed (patterns didn't help)
        outcome_improved = validation_passed and first_attempt_success

        for pattern_id in pattern_ids:
            try:
                # Determine application mode based on tracking
                # Exploration patterns were selected via epsilon-greedy below threshold
                if pattern_id in self._exploration_pattern_ids:
                    application_mode = "exploration"
                else:
                    application_mode = "exploitation"

                self._global_learning_store.record_pattern_application(
                    pattern_id=pattern_id,
                    execution_id=f"sheet_{sheet_num}",
                    outcome_improved=outcome_improved,
                    retry_count_before=0,  # We don't track this per-pattern
                    retry_count_after=0 if first_attempt_success else 1,
                    application_mode=application_mode,
                    validation_passed=validation_passed,
                    grounding_confidence=grounding_confidence,
                )

                # v22 Evolution: Update success factors when pattern succeeds
                # Only capture factors when the pattern led to success
                if outcome_improved:
                    self._global_learning_store.update_success_factors(
                        pattern_id=pattern_id,
                        validation_types=validation_types,
                        error_categories=error_categories,
                        prior_sheet_status=prior_sheet_status,
                        retry_iteration=retry_iteration,
                        escalation_was_pending=escalation_was_pending,
                        grounding_confidence=grounding_confidence,
                    )
                    self._logger.debug(
                        "learning.success_factors_updated",
                        pattern_id=pattern_id,
                        sheet_num=sheet_num,
                        validation_types=validation_types,
                        prior_sheet_status=prior_sheet_status,
                    )

                self._logger.debug(
                    "learning.pattern_feedback_recorded",
                    pattern_id=pattern_id,
                    sheet_num=sheet_num,
                    outcome_improved=outcome_improved,
                    validation_passed=validation_passed,
                    first_attempt_success=first_attempt_success,
                    application_mode=application_mode,
                    grounding_confidence=grounding_confidence,
                )
            except Exception as e:
                # Pattern feedback recording should not block execution
                self._logger.warning(
                    "learning.pattern_feedback_failed",
                    pattern_id=pattern_id,
                    sheet_num=sheet_num,
                    error=str(e),
                )

