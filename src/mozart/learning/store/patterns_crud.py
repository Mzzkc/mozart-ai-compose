"""Pattern CRUD and effectiveness mixin for GlobalLearningStore.

Provides methods for creating, updating, and managing pattern effectiveness:
- record_pattern: Create or update a pattern
- record_pattern_application: Record pattern usage and update effectiveness
- _calculate_effectiveness: Bayesian moving average with decay
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
from dataclasses import dataclass
from datetime import datetime

from mozart.core.logging import MozartLogger
from mozart.learning.store.base import _logger
from mozart.learning.store.models import QuarantineStatus


@dataclass(frozen=True)
class PatternProvenance:
    """Provenance information for where a pattern was discovered.

    Groups the job_hash and sheet_num that always travel together
    when recording pattern origins.
    """

    job_hash: str | None = None
    sheet_num: int | None = None


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
        provenance: PatternProvenance | None = None,
        # Deprecated individual params — use `provenance` instead
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
            provenance: Grouped provenance info (job_hash + sheet_num).
            provenance_job_hash: Deprecated — use provenance instead.
            provenance_sheet_num: Deprecated — use provenance instead.

        Returns:
            The pattern ID.
        """
        # Resolve provenance: prefer grouped param, fall back to individual params
        if provenance is not None:
            job_hash = provenance.job_hash
            sheet_num = provenance.sheet_num
        else:
            job_hash = provenance_job_hash
            sheet_num = provenance_sheet_num

        now = datetime.now().isoformat()
        # Normalize pattern_name for consistent deduplication:
        # lowercase and collapse whitespace so "File Not Created" and
        # "file not created" hash to the same ID.
        normalized_name = " ".join(pattern_name.lower().split())
        pattern_id = hashlib.sha256(
            f"{pattern_type}:{normalized_name}".encode()
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
                        job_hash,
                        sheet_num,
                        now,
                    ),
                )

        return pattern_id

    def record_pattern_application(
        self,
        pattern_id: str,
        execution_id: str,
        pattern_led_to_success: bool,
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
            pattern_led_to_success: Whether applying this pattern led to
                execution success (validation passed on first attempt).
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
            # Guard: verify pattern exists before recording application
            exists = conn.execute(
                "SELECT 1 FROM patterns WHERE id = ?", (pattern_id,)
            ).fetchone()
            if not exists:
                _logger.warning(
                    "pattern_application_skipped",
                    pattern_id=pattern_id,
                    reason="pattern_not_found",
                )
                return app_id

            conn.execute(
                """
                INSERT INTO pattern_applications (
                    id, pattern_id, execution_id, applied_at,
                    pattern_led_to_success, retry_count_before, retry_count_after,
                    grounding_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    app_id,
                    pattern_id,
                    execution_id,
                    now_iso,
                    pattern_led_to_success,
                    retry_count_before,
                    retry_count_after,
                    grounding_confidence,
                ),
            )

            if pattern_led_to_success:
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
                last_confirmed_raw = row["last_confirmed"]
                last_confirmed = (
                    datetime.fromisoformat(last_confirmed_raw)
                    if last_confirmed_raw
                    else now
                )
                new_effectiveness = self._calculate_effectiveness(
                    pattern_id=pattern_id,
                    led_to_success_count=row["led_to_success_count"],
                    led_to_failure_count=row["led_to_failure_count"],
                    last_confirmed=last_confirmed,
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

    @staticmethod
    def _fetch_recent_applications(
        conn: sqlite3.Connection,
        pattern_id: str,
        limit: int = 5,
    ) -> tuple[float | None, float]:
        """Fetch recent application statistics from the database.

        Separates data fetching from the effectiveness formula so each
        concern can be tested and understood independently.

        Args:
            conn: Database connection.
            pattern_id: Pattern to query.
            limit: Number of recent applications to consider.

        Returns:
            Tuple of (recent_success_rate, avg_grounding_confidence).
            recent_success_rate is None if no recent applications exist.
        """
        cursor = conn.execute(
            """
            SELECT pattern_led_to_success, grounding_confidence
            FROM pattern_applications
            WHERE pattern_id = ?
            ORDER BY applied_at DESC
            LIMIT ?
            """,
            (pattern_id, limit),
        )
        recent_apps = cursor.fetchall()

        if not recent_apps:
            return None, 1.0

        recent_successes = sum(1 for app in recent_apps if app["pattern_led_to_success"])
        recent_rate = recent_successes / len(recent_apps)

        grounding_values = [
            app["grounding_confidence"]
            for app in recent_apps
            if app["grounding_confidence"] is not None
        ]
        avg_grounding = (
            sum(grounding_values) / len(grounding_values)
            if grounding_values
            else 1.0
        )

        return recent_rate, avg_grounding

    @staticmethod
    def _bayesian_effectiveness(
        historical: float,
        recent: float,
        days_since_confirmed: int,
        avg_grounding: float,
        alpha: float = 0.7,
    ) -> float:
        """Pure math: Bayesian moving average with recency decay and grounding weight.

        Args:
            historical: Historical success rate (Laplace-smoothed).
            recent: Recent success rate from last N applications.
            days_since_confirmed: Days since pattern last led to success.
            avg_grounding: Average grounding confidence (0.0-1.0).
            alpha: Weight for recent vs historical (0.0-1.0).

        Returns:
            Effectiveness score between 0.0 and 1.0.
        """
        combined: float = alpha * recent + (1 - alpha) * historical
        decay: float = 0.9 ** (days_since_confirmed / 30.0)
        grounding_weight: float = 0.7 + 0.3 * avg_grounding
        return combined * decay * grounding_weight

    def _calculate_effectiveness(
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

        Orchestrates data prep and math helpers to produce the final score.

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

        recent_rate, avg_grounding = self._fetch_recent_applications(conn, pattern_id)
        recent = recent_rate if recent_rate is not None else historical

        days_since = (now - last_confirmed).days

        return self._bayesian_effectiveness(
            historical=historical,
            recent=recent,
            days_since_confirmed=days_since,
            avg_grounding=avg_grounding,
        )

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
            last_confirmed_raw = row["last_confirmed"]
            last_confirmed = (
                datetime.fromisoformat(last_confirmed_raw)
                if last_confirmed_raw
                else now
            )
            new_effectiveness = self._calculate_effectiveness(
                pattern_id=pattern_id,
                led_to_success_count=row["led_to_success_count"],
                led_to_failure_count=row["led_to_failure_count"],
                last_confirmed=last_confirmed,
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

        _logger.info("priorities_recalculated", count=updated)
        return updated
