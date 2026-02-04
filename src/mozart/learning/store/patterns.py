"""Pattern-related mixin for GlobalLearningStore.

This module provides the PatternMixin class that handles all pattern-related
operations including:
- Recording and updating patterns
- Pattern application tracking
- Effectiveness and priority calculations
- Quarantine lifecycle management
- Trust scoring
- Success factor analysis (metacognitive reflection)
- Pattern discovery broadcasting

Extracted from global_store.py as part of the modularization effort.
"""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mozart.learning.store.models import (
    PatternDiscoveryEvent,
    PatternRecord,
    QuarantineStatus,
    SuccessFactors,
)

if TYPE_CHECKING:
    pass  # Type checking imports can be added here if needed

# Import logger from base module
from mozart.learning.store.base import _logger


class PatternMixin:
    """Mixin providing pattern-related methods for GlobalLearningStore.

    This mixin requires that the composed class provides:
    - _get_connection(): Context manager yielding sqlite3.Connection
    - _logger: Logger instance for logging

    Pattern Management Methods:
    - record_pattern: Create or update a pattern
    - record_pattern_application: Record pattern usage and update effectiveness
    - update_pattern_effectiveness: Manually recalculate effectiveness
    - recalculate_all_pattern_priorities: Batch priority recalculation
    - get_patterns: Query patterns with filtering
    - get_pattern_by_id: Get a single pattern

    Quarantine Lifecycle Methods:
    - quarantine_pattern: Move to QUARANTINED status
    - validate_pattern: Move to VALIDATED status
    - retire_pattern: Move to RETIRED status
    - get_quarantined_patterns: Get patterns needing review
    - get_pattern_provenance: Get pattern origin information

    Trust Scoring Methods:
    - calculate_trust_score: Calculate and update trust score
    - update_trust_score: Incremental trust updates
    - recalculate_all_trust_scores: Batch trust recalculation
    - get_high_trust_patterns: Get trusted patterns
    - get_low_trust_patterns: Get untrusted patterns
    - get_patterns_for_auto_apply: Get patterns eligible for autonomous use

    Success Factors Methods (v22 Metacognitive Reflection):
    - update_success_factors: Record WHY a pattern succeeded
    - get_success_factors: Get a pattern's success factors
    - analyze_pattern_why: Generate human-readable WHY analysis
    - get_patterns_with_why: Get patterns with their analysis

    Pattern Discovery Methods (v14 Broadcasting):
    - record_pattern_discovery: Broadcast a new pattern to other jobs
    - check_recent_pattern_discoveries: Poll for patterns from other jobs
    - cleanup_expired_pattern_discoveries: Maintenance cleanup
    - get_active_pattern_discoveries: Get all unexpired discoveries

    Pattern Retirement Methods:
    - retire_drifting_patterns: Auto-retire patterns with negative drift
    - get_retired_patterns: Get deprecated patterns
    """

    # Type hints for attributes provided by the composed class
    _logger: Any
    _get_connection: Any

    def record_pattern(
        self,
        pattern_type: str,
        pattern_name: str,
        description: str | None = None,
        context_tags: list[str] | None = None,
        suggested_action: str | None = None,
        provenance_job_hash: str | None = None,
        provenance_sheet_num: int | None = None,
    ) -> str:
        """Record or update a pattern in the global store.

        If a pattern with the same type and name exists, increments its count.
        Otherwise, creates a new pattern.

        v19 Evolution: Now accepts provenance parameters to track the origin of patterns.

        Args:
            pattern_type: The type of pattern (e.g., 'validation_failure').
            pattern_name: A unique name for this pattern.
            description: Human-readable description.
            context_tags: Tags for matching context.
            suggested_action: Recommended action for this pattern.
            provenance_job_hash: Hash of the job that discovered this pattern.
            provenance_sheet_num: Sheet number where pattern was discovered.

        Returns:
            The pattern ID.
        """
        now = datetime.now().isoformat()
        pattern_id = hashlib.sha256(
            f"{pattern_type}:{pattern_name}".encode()
        ).hexdigest()[:16]

        with self._get_connection() as conn:
            # Check if pattern exists
            cursor = conn.execute(
                "SELECT id, occurrence_count FROM patterns WHERE id = ?",
                (pattern_id,),
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing pattern
                conn.execute(
                    """
                    UPDATE patterns SET
                        occurrence_count = occurrence_count + 1,
                        last_seen = ?,
                        description = COALESCE(?, description),
                        suggested_action = COALESCE(?, suggested_action),
                        context_tags = ?
                    WHERE id = ?
                    """,
                    (
                        now,
                        description,
                        suggested_action,
                        json.dumps(context_tags or []),
                        pattern_id,
                    ),
                )
            else:
                # Insert new pattern with provenance and default quarantine status
                conn.execute(
                    """
                    INSERT INTO patterns (
                        id, pattern_type, pattern_name, description,
                        occurrence_count, first_seen, last_seen, last_confirmed,
                        led_to_success_count, led_to_failure_count,
                        effectiveness_score, variance, suggested_action,
                        context_tags, priority_score,
                        quarantine_status, provenance_job_hash, provenance_sheet_num,
                        trust_score, trust_calculation_date
                    ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, 0, 0, 0.5, 0.0, ?, ?, 0.5,
                              ?, ?, ?, 0.5, ?)
                    """,
                    (
                        pattern_id,
                        pattern_type,
                        pattern_name,
                        description,
                        now,
                        now,
                        now,
                        suggested_action,
                        json.dumps(context_tags or []),
                        QuarantineStatus.PENDING.value,
                        provenance_job_hash,
                        provenance_sheet_num,
                        now,
                    ),
                )

        return pattern_id

    def record_pattern_application(
        self,
        pattern_id: str,
        execution_id: str,
        outcome_improved: bool,
        retry_count_before: int = 0,
        retry_count_after: int = 0,
        application_mode: str = "exploitation",
        validation_passed: bool | None = None,
        grounding_confidence: float | None = None,
    ) -> str:
        """Record that a pattern was applied to an execution.

        This creates the feedback loop for effectiveness tracking. After recording
        the application, automatically updates effectiveness_score and priority_score
        for the pattern.

        v12 Evolution: Grounding→Pattern Feedback - grounding_confidence is now
        stored and used to weight effectiveness calculations. Patterns applied
        during high-grounding executions carry more weight.

        Args:
            pattern_id: The pattern that was applied.
            execution_id: The execution it was applied to.
            outcome_improved: Whether the outcome was better than baseline.
            retry_count_before: Retry count before pattern applied.
            retry_count_after: Retry count after pattern applied.
            application_mode: 'exploration' or 'exploitation' - how the pattern was selected.
            validation_passed: Whether validation passed on first attempt.
            grounding_confidence: Grounding confidence (0.0-1.0) from external validation.
                                 None if grounding hooks were not executed.

        Returns:
            The application record ID.
        """
        # TODO(v25): Add validation_passed column to pattern_applications table
        # This parameter is accepted for API compatibility but not yet stored.
        # Future: Store and use for validation-aware effectiveness calculation,
        # differentiating patterns that help pass validation vs those that just
        # reduce retry count.
        _ = validation_passed  # Suppress unused variable warning
        app_id = str(uuid.uuid4())
        now = datetime.now()
        now_iso = now.isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO pattern_applications (
                    id, pattern_id, execution_id, applied_at,
                    outcome_improved, retry_count_before, retry_count_after,
                    grounding_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    app_id,
                    pattern_id,
                    execution_id,
                    now_iso,
                    outcome_improved,
                    retry_count_before,
                    retry_count_after,
                    grounding_confidence,
                ),
            )

            # Update pattern counts
            if outcome_improved:
                conn.execute(
                    """
                    UPDATE patterns SET
                        led_to_success_count = led_to_success_count + 1,
                        last_confirmed = ?
                    WHERE id = ?
                    """,
                    (now_iso, pattern_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE patterns SET
                        led_to_failure_count = led_to_failure_count + 1
                    WHERE id = ?
                    """,
                    (pattern_id,),
                )

            # Fetch updated pattern data for effectiveness/priority recalculation
            cursor = conn.execute(
                """
                SELECT led_to_success_count, led_to_failure_count, last_confirmed,
                       occurrence_count, variance
                FROM patterns WHERE id = ?
                """,
                (pattern_id,),
            )
            row = cursor.fetchone()

            if row:
                # Calculate new effectiveness score
                new_effectiveness = self._calculate_effectiveness_v2(
                    pattern_id=pattern_id,
                    led_to_success_count=row["led_to_success_count"],
                    led_to_failure_count=row["led_to_failure_count"],
                    last_confirmed=datetime.fromisoformat(row["last_confirmed"]),
                    now=now,
                    conn=conn,
                )

                # Calculate new priority score
                new_priority = self._calculate_priority_score(
                    effectiveness=new_effectiveness,
                    occurrence_count=row["occurrence_count"],
                    variance=row["variance"],
                )

                # Update both scores in the database
                conn.execute(
                    """
                    UPDATE patterns SET
                        effectiveness_score = ?,
                        priority_score = ?
                    WHERE id = ?
                    """,
                    (new_effectiveness, new_priority, pattern_id),
                )

                _logger.debug(
                    f"Updated pattern {pattern_id}: effectiveness={new_effectiveness:.3f}, "
                    f"priority={new_priority:.3f}, mode={application_mode}"
                )

        return app_id

    def _calculate_effectiveness_v2(
        self,
        pattern_id: str,
        led_to_success_count: int,
        led_to_failure_count: int,
        last_confirmed: datetime,
        now: datetime,
        conn: sqlite3.Connection,
        min_applications: int = 3,
    ) -> float:
        """Calculate effectiveness score using Bayesian moving average with decay.

        This is the v2 formula that combines historical and recent performance
        with recency weighting. Patterns that haven't proven themselves recently
        will see their effectiveness decay.

        v12 Evolution: Grounding→Pattern Feedback - effectiveness is now weighted
        by grounding confidence. High-grounding executions carry more trust weight.

        Formula:
            base_effectiveness = (α × recent + (1-α) × historical) × recency_decay
            effectiveness = base_effectiveness × grounding_weight

        Where:
        - α = 0.7 (weight recent outcomes more heavily)
        - recent = success rate in last 5 applications
        - historical = all-time success rate with Laplace smoothing
        - recency_decay = 0.9^(days_since_last_success / 30)
        - grounding_weight = 0.7 + 0.3 × avg_grounding (when grounding data exists)

        Args:
            pattern_id: The pattern being calculated.
            led_to_success_count: Total successes.
            led_to_failure_count: Total failures.
            last_confirmed: When pattern last led to success.
            now: Current timestamp.
            conn: Database connection for recent applications query.
            min_applications: Minimum applications before trusting data.

        Returns:
            Effectiveness score between 0.0 and 1.0.
        """
        total = led_to_success_count + led_to_failure_count

        # Cold start: return optimistic prior to encourage exploration
        if total < min_applications:
            return 0.55

        # Historical effectiveness with Laplace smoothing (prevents 0/1 extremes)
        historical = (led_to_success_count + 0.5) / (total + 1)

        # Query recent applications for recency weighting and grounding confidence
        cursor = conn.execute(
            """
            SELECT outcome_improved, grounding_confidence
            FROM pattern_applications
            WHERE pattern_id = ?
            ORDER BY applied_at DESC
            LIMIT 5
            """,
            (pattern_id,),
        )
        recent_apps = cursor.fetchall()

        if recent_apps:
            recent_successes = sum(1 for app in recent_apps if app["outcome_improved"])
            recent = recent_successes / len(recent_apps)

            # v12: Calculate average grounding confidence from recent applications
            grounding_values = [
                app["grounding_confidence"]
                for app in recent_apps
                if app["grounding_confidence"] is not None
            ]
            if grounding_values:
                avg_grounding = sum(grounding_values) / len(grounding_values)
            else:
                # No grounding data - use neutral multiplier (1.0)
                avg_grounding = 1.0
        else:
            recent = historical
            avg_grounding = 1.0

        # Combine with recency weighting: α = 0.7 prioritizes recent outcomes
        alpha = 0.7
        combined = alpha * recent + (1 - alpha) * historical

        # Apply recency penalty: patterns decay if not recently confirmed
        days_since = (now - last_confirmed).days
        decay = 0.9 ** (days_since / 30.0)

        base_effectiveness = combined * decay

        # v12: Apply grounding weight
        # Formula: grounding_weight = 0.7 + 0.3 × avg_grounding
        # This means:
        # - avg_grounding = 0.0 → weight = 0.70 (30% penalty for ungrounded)
        # - avg_grounding = 0.5 → weight = 0.85 (15% penalty for low grounding)
        # - avg_grounding = 1.0 → weight = 1.00 (full trust for high grounding)
        # When avg_grounding = 1.0 (no grounding data), weight = 1.0 (backward compatible)
        grounding_weight = 0.7 + 0.3 * avg_grounding

        return base_effectiveness * grounding_weight

    def _calculate_priority_score(
        self,
        effectiveness: float,
        occurrence_count: int,
        variance: float,
    ) -> float:
        """Calculate priority score from effectiveness and other factors.

        Priority determines which patterns get selected first during
        exploitation mode. Higher priority = more likely to be applied.

        Formula:
            priority = effectiveness × frequency_factor × (1 - variance)

        Where:
        - frequency_factor = min(1.0, log10(occurrence_count + 1) / 2)
          This gives a boost to frequently-seen patterns (up to 100 occurrences)
        - variance penalizes inconsistent patterns

        Args:
            effectiveness: Calculated effectiveness score (0.0-1.0).
            occurrence_count: How many times pattern has been seen.
            variance: Measure of outcome inconsistency (0.0-1.0).

        Returns:
            Priority score between 0.0 and 1.0.
        """
        import math

        # Frequency factor: log scale to avoid over-weighting very common patterns
        # log10(1) = 0, log10(10) = 1, log10(100) = 2
        # Divided by 2 so patterns need ~100 occurrences to reach max frequency boost
        frequency_factor = min(1.0, math.log10(occurrence_count + 1) / 2.0)

        # Variance penalty: inconsistent patterns should have lower priority
        variance_penalty = 1 - variance

        # Combine factors
        priority = effectiveness * frequency_factor * variance_penalty

        # Ensure bounds
        return max(0.0, min(1.0, priority))

    def update_pattern_effectiveness(
        self,
        pattern_id: str,
    ) -> float | None:
        """Manually recalculate and update a pattern's effectiveness.

        This can be called independently from record_pattern_application()
        for batch updates or manual recalculation.

        Args:
            pattern_id: The pattern to update.

        Returns:
            New effectiveness score, or None if pattern not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT led_to_success_count, led_to_failure_count, last_confirmed,
                       occurrence_count, variance
                FROM patterns WHERE id = ?
                """,
                (pattern_id,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            now = datetime.now()
            new_effectiveness = self._calculate_effectiveness_v2(
                pattern_id=pattern_id,
                led_to_success_count=row["led_to_success_count"],
                led_to_failure_count=row["led_to_failure_count"],
                last_confirmed=datetime.fromisoformat(row["last_confirmed"]),
                now=now,
                conn=conn,
            )

            new_priority = self._calculate_priority_score(
                effectiveness=new_effectiveness,
                occurrence_count=row["occurrence_count"],
                variance=row["variance"],
            )

            conn.execute(
                """
                UPDATE patterns SET
                    effectiveness_score = ?,
                    priority_score = ?
                WHERE id = ?
                """,
                (new_effectiveness, new_priority, pattern_id),
            )

            _logger.debug(
                f"Manual update pattern {pattern_id}: "
                f"effectiveness={new_effectiveness:.3f}, priority={new_priority:.3f}"
            )
            return new_effectiveness

    def recalculate_all_pattern_priorities(self) -> int:
        """Recalculate priorities for all patterns.

        Use for batch maintenance, e.g., daily recalculation to apply
        recency decay to patterns that haven't been used recently.

        Returns:
            Number of patterns updated.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM patterns"
            )
            pattern_ids = [row["id"] for row in cursor.fetchall()]

        updated = 0
        for pattern_id in pattern_ids:
            result = self.update_pattern_effectiveness(pattern_id)
            if result is not None:
                updated += 1

        _logger.info(f"Recalculated priorities for {updated} patterns")
        return updated

    def get_patterns(
        self,
        pattern_type: str | None = None,
        min_priority: float = 0.3,
        limit: int = 20,
        context_tags: list[str] | None = None,
        quarantine_status: QuarantineStatus | None = None,
        exclude_quarantined: bool = False,
        min_trust: float | None = None,
        max_trust: float | None = None,
    ) -> list[PatternRecord]:
        """Get patterns from the global store.

        v19 Evolution: Extended with quarantine and trust filtering options.

        Args:
            pattern_type: Optional filter by pattern type.
            min_priority: Minimum priority score to include.
            limit: Maximum number of patterns to return.
            context_tags: Optional list of tags for context-based filtering.
                         Patterns match if ANY of their tags match ANY query tag.
                         If None or empty, no tag filtering is applied.
            quarantine_status: Filter by specific quarantine status.
            exclude_quarantined: If True, exclude QUARANTINED patterns.
            min_trust: Filter patterns with trust_score >= this value.
            max_trust: Filter patterns with trust_score <= this value.

        Returns:
            List of PatternRecord objects sorted by priority.
        """
        with self._get_connection() as conn:
            # Build query dynamically based on filters
            where_clauses = ["priority_score >= ?"]
            params: list[Any] = [min_priority]

            if pattern_type:
                where_clauses.append("pattern_type = ?")
                params.append(pattern_type)

            # v19: Quarantine status filtering
            if quarantine_status is not None:
                where_clauses.append("quarantine_status = ?")
                params.append(quarantine_status.value)
            elif exclude_quarantined:
                where_clauses.append("quarantine_status != ?")
                params.append(QuarantineStatus.QUARANTINED.value)

            # v19: Trust score filtering
            if min_trust is not None:
                where_clauses.append("trust_score >= ?")
                params.append(min_trust)
            if max_trust is not None:
                where_clauses.append("trust_score <= ?")
                params.append(max_trust)

            # Context tag filtering: match if ANY pattern tag matches ANY query tag
            # Uses json_each() to iterate over the JSON array stored in context_tags
            # Note: Explicitly check for non-empty list to handle [] vs None correctly
            # Safe from SQL injection: placeholders only contain "?" characters,
            # actual tag values are bound via params.extend()
            if context_tags is not None and len(context_tags) > 0:
                tag_placeholders = ", ".join("?" for _ in context_tags)
                where_clauses.append(
                    f"""EXISTS (
                        SELECT 1 FROM json_each(context_tags)
                        WHERE json_each.value IN ({tag_placeholders})
                    )"""
                )
                params.extend(context_tags)

            params.append(limit)
            query = f"""
                SELECT * FROM patterns
                WHERE {" AND ".join(where_clauses)}
                ORDER BY priority_score DESC
                LIMIT ?
            """
            cursor = conn.execute(query, tuple(params))

            records = []
            for row in cursor.fetchall():
                records.append(self._row_to_pattern_record(row))

            return records

    def _row_to_discovery_event(self, row: sqlite3.Row) -> PatternDiscoveryEvent:
        """Convert a database row to a PatternDiscoveryEvent.

        v14: Helper method to centralize PatternDiscoveryEvent construction.

        Args:
            row: Database row from pattern_discovery_events table.

        Returns:
            PatternDiscoveryEvent instance with all fields populated.
        """
        return PatternDiscoveryEvent(
            id=row["id"],
            pattern_id=row["pattern_id"],
            pattern_name=row["pattern_name"],
            pattern_type=row["pattern_type"],
            source_job_hash=row["source_job_hash"],
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            effectiveness_score=row["effectiveness_score"],
            context_tags=json.loads(row["context_tags"] or "[]"),
        )

    def _row_to_pattern_record(self, row: sqlite3.Row) -> PatternRecord:
        """Convert a database row to a PatternRecord.

        v19: Helper method to centralize PatternRecord construction with new fields.

        Args:
            row: Database row from patterns table.

        Returns:
            PatternRecord instance with all fields populated.
        """
        return PatternRecord(
            id=row["id"],
            pattern_type=row["pattern_type"],
            pattern_name=row["pattern_name"],
            description=row["description"],
            occurrence_count=row["occurrence_count"],
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            last_confirmed=datetime.fromisoformat(row["last_confirmed"]),
            led_to_success_count=row["led_to_success_count"],
            led_to_failure_count=row["led_to_failure_count"],
            effectiveness_score=row["effectiveness_score"],
            variance=row["variance"],
            suggested_action=row["suggested_action"],
            context_tags=json.loads(row["context_tags"] or "[]"),
            priority_score=row["priority_score"],
            # v19 fields with safe defaults for backward compatibility
            quarantine_status=QuarantineStatus(row["quarantine_status"])
            if row["quarantine_status"]
            else QuarantineStatus.PENDING,
            provenance_job_hash=row["provenance_job_hash"],
            provenance_sheet_num=row["provenance_sheet_num"],
            quarantined_at=datetime.fromisoformat(row["quarantined_at"])
            if row["quarantined_at"]
            else None,
            validated_at=datetime.fromisoformat(row["validated_at"])
            if row["validated_at"]
            else None,
            quarantine_reason=row["quarantine_reason"],
            trust_score=row["trust_score"] if row["trust_score"] is not None else 0.5,
            trust_calculation_date=datetime.fromisoformat(row["trust_calculation_date"])
            if row["trust_calculation_date"]
            else None,
            # v22 fields for metacognitive pattern reflection
            success_factors=SuccessFactors.from_dict(json.loads(row["success_factors"]))
            if row["success_factors"]
            else None,
            success_factors_updated_at=datetime.fromisoformat(row["success_factors_updated_at"])
            if row["success_factors_updated_at"]
            else None,
        )

    # =========================================================================
    # v19: Pattern Quarantine & Provenance Methods
    # =========================================================================

    def quarantine_pattern(
        self,
        pattern_id: str,
        reason: str | None = None,
    ) -> bool:
        """Move a pattern to QUARANTINED status.

        Quarantined patterns are excluded from automatic application but
        retained for investigation and historical reference.

        Args:
            pattern_id: The pattern ID to quarantine.
            reason: Optional reason for quarantine.

        Returns:
            True if pattern was quarantined, False if pattern not found.
        """
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM patterns WHERE id = ?",
                (pattern_id,),
            )
            if not cursor.fetchone():
                _logger.warning(f"Pattern {pattern_id} not found for quarantine")
                return False

            conn.execute(
                """
                UPDATE patterns SET
                    quarantine_status = ?,
                    quarantined_at = ?,
                    quarantine_reason = ?
                WHERE id = ?
                """,
                (
                    QuarantineStatus.QUARANTINED.value,
                    now,
                    reason,
                    pattern_id,
                ),
            )

        _logger.info(f"Quarantined pattern {pattern_id}: {reason or 'no reason given'}")
        return True

    def validate_pattern(self, pattern_id: str) -> bool:
        """Move a pattern to VALIDATED status.

        Validated patterns are trusted for autonomous application and
        receive a trust bonus in relevance scoring.

        Args:
            pattern_id: The pattern ID to validate.

        Returns:
            True if pattern was validated, False if pattern not found.
        """
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM patterns WHERE id = ?",
                (pattern_id,),
            )
            if not cursor.fetchone():
                _logger.warning(f"Pattern {pattern_id} not found for validation")
                return False

            conn.execute(
                """
                UPDATE patterns SET
                    quarantine_status = ?,
                    validated_at = ?,
                    quarantine_reason = NULL
                WHERE id = ?
                """,
                (
                    QuarantineStatus.VALIDATED.value,
                    now,
                    pattern_id,
                ),
            )

        _logger.info(f"Validated pattern {pattern_id}")
        return True

    def retire_pattern(self, pattern_id: str) -> bool:
        """Move a pattern to RETIRED status.

        Retired patterns are no longer in active use but retained for
        historical reference and trend analysis.

        Args:
            pattern_id: The pattern ID to retire.

        Returns:
            True if pattern was retired, False if pattern not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM patterns WHERE id = ?",
                (pattern_id,),
            )
            if not cursor.fetchone():
                _logger.warning(f"Pattern {pattern_id} not found for retirement")
                return False

            conn.execute(
                """
                UPDATE patterns SET
                    quarantine_status = ?
                WHERE id = ?
                """,
                (
                    QuarantineStatus.RETIRED.value,
                    pattern_id,
                ),
            )

        _logger.info(f"Retired pattern {pattern_id}")
        return True

    def get_quarantined_patterns(self, limit: int = 50) -> list[PatternRecord]:
        """Get all patterns currently in QUARANTINED status.

        Args:
            limit: Maximum number of patterns to return.

        Returns:
            List of quarantined PatternRecord objects.
        """
        return self.get_patterns(
            quarantine_status=QuarantineStatus.QUARANTINED,
            min_priority=0.0,  # Include all regardless of priority
            limit=limit,
        )

    def get_pattern_by_id(self, pattern_id: str) -> PatternRecord | None:
        """Get a single pattern by its ID.

        Args:
            pattern_id: The pattern ID to retrieve.

        Returns:
            PatternRecord if found, None otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM patterns WHERE id = ?",
                (pattern_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_pattern_record(row)
            return None

    def get_pattern_provenance(self, pattern_id: str) -> dict[str, Any] | None:
        """Get provenance information for a pattern.

        Returns details about the pattern's origin and lifecycle.

        Args:
            pattern_id: The pattern ID to query.

        Returns:
            Dict with provenance info, or None if pattern not found.
        """
        pattern = self.get_pattern_by_id(pattern_id)
        if not pattern:
            return None

        return {
            "pattern_id": pattern.id,
            "pattern_name": pattern.pattern_name,
            "quarantine_status": pattern.quarantine_status.value,
            "first_seen": pattern.first_seen.isoformat(),
            "last_seen": pattern.last_seen.isoformat(),
            "last_confirmed": pattern.last_confirmed.isoformat(),
            "provenance_job_hash": pattern.provenance_job_hash,
            "provenance_sheet_num": pattern.provenance_sheet_num,
            "quarantined_at": pattern.quarantined_at.isoformat()
            if pattern.quarantined_at
            else None,
            "validated_at": pattern.validated_at.isoformat()
            if pattern.validated_at
            else None,
            "quarantine_reason": pattern.quarantine_reason,
            "trust_score": pattern.trust_score,
            "trust_calculation_date": pattern.trust_calculation_date.isoformat()
            if pattern.trust_calculation_date
            else None,
        }

    # =========================================================================
    # v19: Pattern Trust Scoring Methods
    # =========================================================================

    def calculate_trust_score(self, pattern_id: str) -> float | None:
        """Calculate and update trust score for a pattern.

        Trust score formula:
            trust = 0.5 + (success_rate × 0.3) - (failure_rate × 0.4) + (age_factor × 0.2)

        Where:
        - success_rate = led_to_success / occurrence_count (capped at 1.0)
        - failure_rate = led_to_failure / occurrence_count (capped at 1.0)
        - age_factor = decay based on last_confirmed timestamp

        Quarantined patterns get a -0.2 penalty.
        Validated patterns get a +0.1 bonus.

        Args:
            pattern_id: The pattern ID to calculate trust for.

        Returns:
            New trust score (0.0-1.0), or None if pattern not found.
        """
        pattern = self.get_pattern_by_id(pattern_id)
        if not pattern:
            return None

        # Calculate base metrics
        total = pattern.occurrence_count
        if total == 0:
            total = 1  # Avoid division by zero

        success_rate = min(1.0, pattern.led_to_success_count / total)
        failure_rate = min(1.0, pattern.led_to_failure_count / total)

        # Age factor using recency decay
        now = datetime.now()
        days_since_confirmed = (now - pattern.last_confirmed).days
        age_factor = 0.9 ** (days_since_confirmed / 30.0)  # 10% decay per month

        # Base trust calculation
        trust = 0.5 + (success_rate * 0.3) - (failure_rate * 0.4) + (age_factor * 0.2)

        # Apply quarantine penalty
        if pattern.quarantine_status == QuarantineStatus.QUARANTINED:
            trust -= 0.2

        # Apply validation bonus
        if pattern.quarantine_status == QuarantineStatus.VALIDATED:
            trust += 0.1

        # Clamp to valid range
        trust = max(0.0, min(1.0, trust))

        # Update in database
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE patterns SET
                    trust_score = ?,
                    trust_calculation_date = ?
                WHERE id = ?
                """,
                (trust, now.isoformat(), pattern_id),
            )

        _logger.debug(f"Calculated trust score for {pattern_id}: {trust:.3f}")
        return trust

    def update_trust_score(self, pattern_id: str, delta: float) -> float | None:
        """Update trust score by a delta amount.

        Useful for incremental trust updates based on recent outcomes.

        Args:
            pattern_id: The pattern ID to update.
            delta: Amount to add to trust score (can be negative).

        Returns:
            New trust score after update, or None if pattern not found.
        """
        pattern = self.get_pattern_by_id(pattern_id)
        if not pattern:
            return None

        new_trust = max(0.0, min(1.0, pattern.trust_score + delta))

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE patterns SET
                    trust_score = ?,
                    trust_calculation_date = ?
                WHERE id = ?
                """,
                (new_trust, datetime.now().isoformat(), pattern_id),
            )

        _logger.debug(
            f"Updated trust score for {pattern_id}: {pattern.trust_score:.3f} → {new_trust:.3f}"
        )
        return new_trust

    def get_high_trust_patterns(
        self,
        threshold: float = 0.7,
        limit: int = 50,
    ) -> list[PatternRecord]:
        """Get patterns with high trust scores.

        Args:
            threshold: Minimum trust score to include.
            limit: Maximum number of patterns to return.

        Returns:
            List of high-trust PatternRecord objects.
        """
        return self.get_patterns(
            min_priority=0.0,
            min_trust=threshold,
            limit=limit,
        )

    def get_low_trust_patterns(
        self,
        threshold: float = 0.3,
        limit: int = 50,
    ) -> list[PatternRecord]:
        """Get patterns with low trust scores.

        Args:
            threshold: Maximum trust score to include.
            limit: Maximum number of patterns to return.

        Returns:
            List of low-trust PatternRecord objects.
        """
        return self.get_patterns(
            min_priority=0.0,
            max_trust=threshold,
            limit=limit,
        )

    def get_patterns_for_auto_apply(
        self,
        trust_threshold: float = 0.85,
        require_validated: bool = True,
        limit: int = 3,
        context_tags: list[str] | None = None,
    ) -> list[PatternRecord]:
        """Get patterns eligible for autonomous application.

        v22 Evolution: Trust-Aware Autonomous Application - returns patterns
        that meet the criteria for being applied without human confirmation.

        Criteria for auto-apply eligibility:
        1. Trust score >= trust_threshold (default 0.85)
        2. Quarantine status == VALIDATED (if require_validated=True)
        3. Pattern is not retired

        Args:
            trust_threshold: Minimum trust score for auto-apply (default 0.85).
            require_validated: Require VALIDATED quarantine status.
            limit: Maximum patterns to return.
            context_tags: Optional context tags to filter by relevance.

        Returns:
            List of PatternRecord objects eligible for auto-apply,
            ordered by trust_score descending.
        """
        with self._get_connection() as conn:
            # Build query with auto-apply criteria
            query = """
                SELECT * FROM patterns
                WHERE trust_score >= ?
                AND quarantine_status != 'retired'
            """
            params: list[Any] = [trust_threshold]

            if require_validated:
                query += " AND quarantine_status = 'validated'"

            query += " ORDER BY trust_score DESC, priority_score DESC"
            query += " LIMIT ?"
            params.append(limit * 2)  # Fetch extra for post-filter

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        patterns = [self._row_to_pattern_record(row) for row in rows]

        # Filter by context tags if specified
        if context_tags:
            tags_set = set(context_tags)
            patterns = [
                p for p in patterns
                if tags_set.intersection(set(p.context_tags))
            ]

        # Apply limit after filtering
        patterns = patterns[:limit]

        _logger.debug(
            f"Found {len(patterns)} patterns eligible for auto-apply "
            f"(threshold={trust_threshold}, validated={require_validated})"
        )
        return patterns

    def recalculate_all_trust_scores(self) -> int:
        """Recalculate trust scores for all patterns.

        Use for batch maintenance or after significant events.

        Returns:
            Number of patterns updated.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT id FROM patterns")
            pattern_ids = [row["id"] for row in cursor.fetchall()]

        updated = 0
        for pattern_id in pattern_ids:
            result = self.calculate_trust_score(pattern_id)
            if result is not None:
                updated += 1

        _logger.info(f"Recalculated trust scores for {updated} patterns")
        return updated

    # =========================================================================
    # v22: Metacognitive Pattern Reflection Methods
    # =========================================================================

    def update_success_factors(
        self,
        pattern_id: str,
        validation_types: list[str] | None = None,
        error_categories: list[str] | None = None,
        prior_sheet_status: str | None = None,
        retry_iteration: int = 0,
        escalation_was_pending: bool = False,
        grounding_confidence: float | None = None,
    ) -> SuccessFactors | None:
        """Update success factors for a pattern based on a successful application.

        This captures the WHY behind pattern success - the context conditions
        that were present when the pattern worked. Factors are aggregated over
        multiple successful applications to build a reliable understanding.

        v22 Evolution: Metacognitive Pattern Reflection - enables better pattern
        selection by understanding causality, not just correlation.

        Args:
            pattern_id: The pattern that succeeded.
            validation_types: Validation types active (file, regex, artifact, etc.)
            error_categories: Error categories present (rate_limit, auth, etc.)
            prior_sheet_status: Status of prior sheet (completed, failed, skipped)
            retry_iteration: Which retry this success occurred on (0 = first)
            escalation_was_pending: Whether escalation was pending
            grounding_confidence: Grounding confidence if external validation present

        Returns:
            Updated SuccessFactors, or None if pattern not found.
        """
        pattern = self.get_pattern_by_id(pattern_id)
        if not pattern:
            return None

        now = datetime.now()
        time_bucket = SuccessFactors.get_time_bucket(now.hour)

        # Get existing factors or create new ones
        if pattern.success_factors:
            factors = pattern.success_factors
            # Update with new observation
            factors.occurrence_count += 1

            # Merge validation types
            if validation_types:
                existing = set(factors.validation_types)
                existing.update(validation_types)
                factors.validation_types = sorted(existing)

            # Merge error categories
            if error_categories:
                existing_errors = set(factors.error_categories)
                existing_errors.update(error_categories)
                factors.error_categories = sorted(existing_errors)

            # Update with latest context (most recent takes precedence for non-aggregated fields)
            if prior_sheet_status:
                factors.prior_sheet_status = prior_sheet_status
            factors.time_of_day_bucket = time_bucket
            factors.retry_iteration = retry_iteration
            factors.escalation_was_pending = escalation_was_pending
            if grounding_confidence is not None:
                factors.grounding_confidence = grounding_confidence

            # Recalculate success rate based on pattern's overall success
            total = pattern.led_to_success_count + pattern.led_to_failure_count
            if total > 0:
                factors.success_rate = pattern.led_to_success_count / total
        else:
            # Create new factors
            factors = SuccessFactors(
                validation_types=validation_types or [],
                error_categories=error_categories or [],
                prior_sheet_status=prior_sheet_status,
                time_of_day_bucket=time_bucket,
                retry_iteration=retry_iteration,
                escalation_was_pending=escalation_was_pending,
                grounding_confidence=grounding_confidence,
                occurrence_count=1,
                success_rate=1.0,  # First observation is a success
            )

        # Persist to database
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE patterns SET
                    success_factors = ?,
                    success_factors_updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(factors.to_dict()), now.isoformat(), pattern_id),
            )

        _logger.debug(
            f"Updated success factors for {pattern_id}: "
            f"{factors.occurrence_count} observations, "
            f"success_rate={factors.success_rate:.2f}"
        )
        return factors

    def get_success_factors(self, pattern_id: str) -> SuccessFactors | None:
        """Get the success factors for a pattern.

        Args:
            pattern_id: The pattern ID to get factors for.

        Returns:
            SuccessFactors if the pattern has captured factors, None otherwise.
        """
        pattern = self.get_pattern_by_id(pattern_id)
        if not pattern:
            return None
        return pattern.success_factors

    def analyze_pattern_why(self, pattern_id: str) -> dict[str, Any]:
        """Analyze WHY a pattern succeeds with structured explanation.

        This produces a human-readable analysis of the success factors,
        suitable for display in CLI or logging.

        Args:
            pattern_id: The pattern to analyze.

        Returns:
            Dictionary with analysis results:
            - pattern_name: Name of the pattern
            - has_factors: Whether success factors have been captured
            - factors_summary: High-level summary
            - key_conditions: Most significant context conditions
            - confidence: How confident we are in the WHY analysis
            - recommendations: Suggestions for pattern application
        """
        pattern = self.get_pattern_by_id(pattern_id)
        if not pattern:
            return {"error": f"Pattern {pattern_id} not found"}

        result: dict[str, Any] = {
            "pattern_name": pattern.pattern_name,
            "pattern_type": pattern.pattern_type,
            "has_factors": pattern.success_factors is not None,
            "trust_score": pattern.trust_score,
            "effectiveness_score": pattern.effectiveness_score,
        }

        if not pattern.success_factors:
            result["factors_summary"] = "No success factors captured yet"
            result["key_conditions"] = []
            result["confidence"] = 0.0
            result["recommendations"] = [
                "Apply this pattern more times to capture success factors"
            ]
            return result

        factors = pattern.success_factors

        # Build factors summary
        summaries = []
        if factors.validation_types:
            summaries.append(f"validation types: {', '.join(factors.validation_types)}")
        if factors.error_categories:
            summaries.append(f"error categories: {', '.join(factors.error_categories)}")
        if factors.time_of_day_bucket:
            summaries.append(f"typically succeeds in: {factors.time_of_day_bucket}")
        if factors.prior_sheet_status:
            summaries.append(f"prior sheet was: {factors.prior_sheet_status}")

        result["factors_summary"] = "; ".join(summaries) if summaries else "Context captured"

        # Identify key conditions
        key_conditions = []
        if factors.grounding_confidence and factors.grounding_confidence > 0.7:
            key_conditions.append(
                f"High grounding confidence ({factors.grounding_confidence:.2f})"
            )
        if factors.retry_iteration == 0:
            key_conditions.append("Succeeds on first attempt")
        elif factors.retry_iteration > 0:
            key_conditions.append(f"Often succeeds after {factors.retry_iteration} retries")
        if factors.validation_types:
            key_conditions.append(f"Works with {len(factors.validation_types)} validation types")
        if not factors.escalation_was_pending:
            key_conditions.append("Succeeds without escalation")

        result["key_conditions"] = key_conditions

        # Calculate confidence based on observation count and success rate
        observation_confidence = min(1.0, factors.occurrence_count / 10)  # Full confidence at 10 obs
        result["confidence"] = observation_confidence * factors.success_rate

        # Generate recommendations
        recommendations = []
        if factors.occurrence_count < 5:
            recommendations.append("Need more observations for reliable analysis")
        if factors.success_rate > 0.8:
            recommendations.append("High confidence pattern - consider for auto-apply")
        if factors.success_rate < 0.5:
            recommendations.append("Low success rate - review pattern relevance")
        if factors.time_of_day_bucket:
            recommendations.append(f"Best applied during {factors.time_of_day_bucket}")

        result["recommendations"] = recommendations
        result["observation_count"] = factors.occurrence_count
        result["success_rate"] = factors.success_rate

        return result

    def get_patterns_with_why(
        self,
        min_observations: int = 1,
        limit: int = 20,
    ) -> list[tuple[PatternRecord, dict[str, Any]]]:
        """Get patterns with their WHY analysis.

        Useful for displaying patterns with their success factors in CLI.

        Args:
            min_observations: Minimum success factor observations required.
            limit: Maximum number of patterns to return.

        Returns:
            List of (PatternRecord, analysis_dict) tuples.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM patterns
                WHERE success_factors IS NOT NULL
                ORDER BY priority_score DESC, trust_score DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        results = []
        for row in rows:
            pattern = self._row_to_pattern_record(row)
            if (
                pattern.success_factors
                and pattern.success_factors.occurrence_count >= min_observations
            ):
                analysis = self.analyze_pattern_why(pattern.id)
                results.append((pattern, analysis))

        return results

    # =========================================================================
    # Evolution v14: Real-time Pattern Broadcasting
    # =========================================================================

    def record_pattern_discovery(
        self,
        pattern_id: str,
        pattern_name: str,
        pattern_type: str,
        job_id: str,
        effectiveness_score: float = 1.0,
        context_tags: list[str] | None = None,
        ttl_seconds: float = 300.0,
    ) -> str:
        """Record a pattern discovery for cross-job broadcasting.

        When a job discovers a new pattern, it broadcasts the discovery so
        other concurrent jobs can benefit immediately rather than waiting
        for aggregation.

        v14 Evolution: Real-time Pattern Broadcasting - enables immediate
        pattern sharing across concurrent jobs.

        Args:
            pattern_id: ID of the discovered pattern.
            pattern_name: Human-readable name of the pattern.
            pattern_type: Type of pattern (validation_failure, etc.).
            job_id: ID of the job that discovered the pattern.
            effectiveness_score: Initial effectiveness score (0.0-1.0).
            context_tags: Optional context tags for pattern matching.
            ttl_seconds: Time-to-live in seconds (default 5 minutes).

        Returns:
            The discovery event record ID.
        """
        from datetime import timedelta

        from mozart.learning.store.base import GlobalLearningStoreBase

        record_id = str(uuid.uuid4())
        now = datetime.now()
        job_hash = GlobalLearningStoreBase.hash_job(job_id)
        expires_at = now + timedelta(seconds=ttl_seconds)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO pattern_discovery_events (
                    id, pattern_id, pattern_name, pattern_type,
                    source_job_hash, recorded_at, expires_at,
                    effectiveness_score, context_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    pattern_id,
                    pattern_name,
                    pattern_type,
                    job_hash,
                    now.isoformat(),
                    expires_at.isoformat(),
                    effectiveness_score,
                    json.dumps(context_tags or []),
                ),
            )

        _logger.info(
            f"Broadcast pattern discovery '{pattern_name}' (type: {pattern_type}), "
            f"expires in {ttl_seconds:.0f}s"
        )
        return record_id

    def check_recent_pattern_discoveries(
        self,
        exclude_job_id: str | None = None,
        pattern_type: str | None = None,
        min_effectiveness: float = 0.0,
        limit: int = 20,
    ) -> list[PatternDiscoveryEvent]:
        """Check for recent pattern discoveries from other jobs.

        Jobs can poll this to discover patterns found by concurrent jobs,
        enabling immediate adoption of successful patterns.

        v14 Evolution: Real-time Pattern Broadcasting - enables jobs to
        receive pattern broadcasts from other concurrent jobs.

        Args:
            exclude_job_id: Optional job ID to exclude (typically self).
            pattern_type: Optional filter by pattern type.
            min_effectiveness: Minimum effectiveness score to include.
            limit: Maximum number of discoveries to return.

        Returns:
            List of PatternDiscoveryEvent objects from other jobs.
        """
        from mozart.learning.store.base import GlobalLearningStoreBase

        now = datetime.now()
        exclude_hash = GlobalLearningStoreBase.hash_job(exclude_job_id) if exclude_job_id else None

        with self._get_connection() as conn:
            # Build query based on filters
            query = """
                SELECT * FROM pattern_discovery_events
                WHERE expires_at > ?
                AND effectiveness_score >= ?
            """
            params: list[str | float] = [now.isoformat(), min_effectiveness]

            if exclude_hash is not None:
                query += " AND source_job_hash != ?"
                params.append(exclude_hash)

            if pattern_type is not None:
                query += " AND pattern_type = ?"
                params.append(pattern_type)

            query += " ORDER BY recorded_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [self._row_to_discovery_event(row) for row in cursor.fetchall()]

    def cleanup_expired_pattern_discoveries(self) -> int:
        """Remove expired pattern discovery events.

        Should be called periodically to prevent the pattern_discovery_events
        table from growing unbounded.

        v14 Evolution: Real-time Pattern Broadcasting - maintains table health.

        Returns:
            Number of expired events removed.
        """
        now = datetime.now()

        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM pattern_discovery_events WHERE expires_at <= ?",
                (now.isoformat(),),
            )
            deleted = cursor.rowcount

        if deleted > 0:
            _logger.debug(f"Cleaned up {deleted} expired pattern discovery events")

        return deleted

    def get_active_pattern_discoveries(
        self,
        pattern_type: str | None = None,
    ) -> list[PatternDiscoveryEvent]:
        """Get all active (unexpired) pattern discovery events.

        Args:
            pattern_type: Optional filter by pattern type.

        Returns:
            List of PatternDiscoveryEvent objects that haven't expired yet.
        """
        now = datetime.now()

        with self._get_connection() as conn:
            if pattern_type:
                cursor = conn.execute(
                    """
                    SELECT * FROM pattern_discovery_events
                    WHERE expires_at > ? AND pattern_type = ?
                    ORDER BY recorded_at DESC
                    """,
                    (now.isoformat(), pattern_type),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM pattern_discovery_events
                    WHERE expires_at > ?
                    ORDER BY recorded_at DESC
                    """,
                    (now.isoformat(),),
                )

            return [self._row_to_discovery_event(row) for row in cursor.fetchall()]

# Export public API
__all__ = [
    "PatternMixin",
]
