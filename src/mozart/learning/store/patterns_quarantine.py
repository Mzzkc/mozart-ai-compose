"""Pattern quarantine lifecycle mixin for GlobalLearningStore.

Provides quarantine lifecycle management methods:
- quarantine_pattern: Move to QUARANTINED status
- validate_pattern: Move to VALIDATED status
- retire_pattern: Move to RETIRED status
- get_quarantined_patterns: Get patterns needing review
"""

import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime

from mozart.core.logging import MozartLogger
from mozart.learning.store.base import _logger
from mozart.learning.store.models import PatternRecord, QuarantineStatus


class PatternQuarantineMixin:
    """Mixin providing pattern quarantine lifecycle methods.

    This mixin requires that the composed class provides:
    - _get_connection(): Context manager yielding sqlite3.Connection
    - get_patterns(): For querying quarantined patterns (from PatternQueryMixin)
    """

    _logger: MozartLogger
    _get_connection: Callable[[], AbstractContextManager[sqlite3.Connection]]

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
            min_priority=0.0,
            limit=limit,
        )
