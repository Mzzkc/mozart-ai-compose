"""Pattern query and lookup mixin for GlobalLearningStore.

Provides methods for querying, filtering, and retrieving patterns:
- get_patterns: Query with filtering by type, priority, tags, quarantine, trust
- get_pattern_by_id: Single pattern lookup
- get_pattern_provenance: Provenance information retrieval
- _row_to_pattern_record: Database row to PatternRecord conversion
- _row_to_discovery_event: Database row to PatternDiscoveryEvent conversion

This is the "hub" mixin — other pattern sub-mixins depend on its query methods.
"""

import json
import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Any

from mozart.core.logging import MozartLogger
from mozart.learning.store.base import WhereBuilder
from mozart.learning.store.models import (
    PatternDiscoveryEvent,
    PatternRecord,
    QuarantineStatus,
    SuccessFactors,
)


class PatternQueryMixin:
    """Mixin providing pattern query methods for GlobalLearningStore.

    This mixin requires that the composed class provides:
    - _get_connection(): Context manager yielding sqlite3.Connection
    """

    _logger: MozartLogger
    _get_connection: Callable[[], AbstractContextManager[sqlite3.Connection]]

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
            wb = WhereBuilder()
            wb.add("priority_score >= ?", min_priority)

            if pattern_type:
                wb.add("pattern_type = ?", pattern_type)

            # v19: Quarantine status filtering
            if quarantine_status is not None:
                wb.add("quarantine_status = ?", quarantine_status.value)
            elif exclude_quarantined:
                wb.add("quarantine_status != ?", QuarantineStatus.QUARANTINED.value)

            # v19: Trust score filtering
            if min_trust is not None:
                wb.add("trust_score >= ?", min_trust)
            if max_trust is not None:
                wb.add("trust_score <= ?", max_trust)

            # Context tag filtering: match if ANY pattern tag matches ANY query tag
            # Uses json_each() to iterate over the JSON array stored in context_tags
            if context_tags is not None and len(context_tags) > 0:
                tag_placeholders = ", ".join("?" for _ in context_tags)
                wb.add(
                    f"""EXISTS (
                        SELECT 1 FROM json_each(context_tags)
                        WHERE json_each.value IN ({tag_placeholders})
                    )""",
                    *context_tags,
                )

            where_sql, params = wb.build()
            cursor = conn.execute(
                f"""
                SELECT * FROM patterns
                WHERE {where_sql}
                ORDER BY priority_score DESC
                LIMIT ?
                """,
                (*params, limit),
            )

            records = []
            for row in cursor.fetchall():
                records.append(self._row_to_pattern_record(row))

            return records

    def _row_to_discovery_event(self, row: sqlite3.Row) -> PatternDiscoveryEvent:
        """Convert a database row to a PatternDiscoveryEvent.

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
