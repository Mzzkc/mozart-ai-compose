"""Pattern CRUD and effectiveness mixin for GlobalLearningStore.

Provides methods for creating, updating, and managing pattern effectiveness:
- record_pattern: Create or update a pattern
- record_pattern_application: Record pattern usage and update effectiveness
- _calculate_effectiveness_v2: Bayesian moving average with decay
- _calculate_priority_score: Priority from effectiveness and frequency
- update_pattern_effectiveness: Manual effectiveness recalculation
- recalculate_all_pattern_priorities: Batch priority recalculation
"""

import hashlib
import json
import math
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime

from mozart.core.logging import MozartLogger
from mozart.learning.store.base import _logger
from mozart.learning.store.models import QuarantineStatus


class PatternCrudMixin:
    """Mixin providing pattern CRUD and effectiveness methods.

    This mixin requires that the composed class provides:
    - _get_connection(): Context manager yielding sqlite3.Connection
    - update_pattern_effectiveness(): For batch recalculation (self-referential)
    """

    _logger: MozartLogger
    _get_connection: Callable[[], AbstractContextManager[sqlite3.Connection]]
    batch_connection: Callable[[], AbstractContextManager[sqlite3.Connection]]

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
            cursor = conn.execute(
                "SELECT id, occurrence_count FROM patterns WHERE id = ?",
                (pattern_id,),
            )
            existing = cursor.fetchone()

            if existing:
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

        Creates the feedback loop for effectiveness tracking. After recording
        the application, automatically updates effectiveness_score and priority_score.

        Args:
            pattern_id: The pattern that was applied.
            execution_id: The execution it was applied to.
            outcome_improved: Whether the outcome was better than baseline.
            retry_count_before: Retry count before pattern applied.
            retry_count_after: Retry count after pattern applied.
            application_mode: 'exploration' or 'exploitation'.
            validation_passed: Whether validation passed on first attempt.
            grounding_confidence: Grounding confidence (0.0-1.0).

        Returns:
            The application record ID.
        """
        _ = validation_passed  # Accepted for API compatibility, not yet stored
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

        Formula:
            base_effectiveness = (alpha * recent + (1-alpha) * historical) * recency_decay
            effectiveness = base_effectiveness * grounding_weight

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

        if total < min_applications:
            return 0.55

        historical = (led_to_success_count + 0.5) / (total + 1)

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

            grounding_values = [
                app["grounding_confidence"]
                for app in recent_apps
                if app["grounding_confidence"] is not None
            ]
            if grounding_values:
                avg_grounding = sum(grounding_values) / len(grounding_values)
            else:
                avg_grounding = 1.0
        else:
            recent = historical
            avg_grounding = 1.0

        alpha = 0.7
        combined = alpha * recent + (1 - alpha) * historical

        days_since = (now - last_confirmed).days
        decay = 0.9 ** (days_since / 30.0)

        base_effectiveness = combined * decay

        grounding_weight = 0.7 + 0.3 * avg_grounding

        return base_effectiveness * grounding_weight

    def _calculate_priority_score(
        self,
        effectiveness: float,
        occurrence_count: int,
        variance: float,
    ) -> float:
        """Calculate priority score from effectiveness and other factors.

        Formula:
            priority = effectiveness * frequency_factor * (1 - variance)

        Args:
            effectiveness: Calculated effectiveness score (0.0-1.0).
            occurrence_count: How many times pattern has been seen.
            variance: Measure of outcome inconsistency (0.0-1.0).

        Returns:
            Priority score between 0.0 and 1.0.
        """
        frequency_factor = min(1.0, math.log10(occurrence_count + 1) / 2.0)
        variance_penalty = 1 - variance
        priority = effectiveness * frequency_factor * variance_penalty
        return max(0.0, min(1.0, priority))

    def update_pattern_effectiveness(
        self,
        pattern_id: str,
    ) -> float | None:
        """Manually recalculate and update a pattern's effectiveness.

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

        Uses batch_connection() to reuse a single SQLite connection across
        all pattern updates, avoiding N+1 connection overhead.

        Returns:
            Number of patterns updated.
        """
        with self.batch_connection():
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
