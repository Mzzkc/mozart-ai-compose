"""Tests for learning store priority formula fix (#101) and FK constraint fix (#129).

Covers:
- TEST-LS-001 through TEST-LS-006: Priority formula frequency floor
- TEST-FK-001 through TEST-FK-006: Soft delete and FK integrity

These tests are TDD — written BEFORE the implementation to specify
the exact expected behavior.
"""

import math
import sqlite3
import threading
from pathlib import Path

import pytest

from mozart.learning.store import GlobalLearningStore
from mozart.learning.store.patterns_crud import PatternCrudMixin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_db(tmp_path: Path) -> Path:
    """Return a temporary database path for an isolated learning store."""
    return tmp_path / "test-learning.db"


@pytest.fixture()
def store(temp_db: Path) -> GlobalLearningStore:
    """Create a fresh GlobalLearningStore with a temporary database."""
    return GlobalLearningStore(db_path=temp_db)


def _insert_pattern_raw(
    store: GlobalLearningStore,
    pattern_id: str,
    effectiveness: float = 0.5,
    occurrence_count: int = 1,
    variance: float = 0.0,
    pattern_type: str = "TEST_PATTERN",
    pattern_name: str = "test pattern",
    conn: sqlite3.Connection | None = None,
) -> None:
    """Insert a pattern with exact field values, bypassing record_pattern's defaults.

    If conn is provided, uses it directly (for batch inserts). Otherwise
    opens a new connection.
    """
    from datetime import datetime

    now = datetime.now().isoformat()
    priority = PatternCrudMixin._calculate_priority_score(
        PatternCrudMixin,  # type: ignore[arg-type]
        effectiveness=effectiveness,
        occurrence_count=occurrence_count,
        variance=variance,
    )

    def _do_insert(c: sqlite3.Connection) -> None:
        c.execute(
            """
            INSERT INTO patterns (
                id, pattern_type, pattern_name, description,
                occurrence_count, first_seen, last_seen, last_confirmed,
                led_to_success_count, led_to_failure_count,
                effectiveness_score, variance, suggested_action,
                context_tags, priority_score,
                quarantine_status, trust_score, trust_calculation_date,
                active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, NULL, '[]', ?,
                      'pending', 0.5, ?, 1)
            """,
            (
                pattern_id,
                pattern_type,
                pattern_name,
                f"Test pattern {pattern_id}",
                occurrence_count,
                now,
                now,
                now,
                effectiveness,
                variance,
                priority,
                now,
            ),
        )

    if conn is not None:
        _do_insert(conn)
    else:
        with store._get_connection() as c:
            _do_insert(c)


# ---------------------------------------------------------------------------
# TEST-LS-001 through TEST-LS-006: Priority Formula
# ---------------------------------------------------------------------------


class TestPriorityFormula:
    """Tests for the priority formula fix (issue #101)."""

    def test_ls_001_new_pattern_visible_at_default_threshold(
        self, store: GlobalLearningStore,
    ) -> None:
        """TEST-LS-001: Pattern with eff=0.5, occ=1, var=0.0 appears
        in get_patterns() with default min_priority=0.3."""
        _insert_pattern_raw(
            store,
            pattern_id="ls001",
            effectiveness=0.5,
            occurrence_count=1,
            variance=0.0,
        )
        results = store.get_patterns()
        ids = [p.id for p in results]
        assert "ls001" in ids, (
            "Single-occurrence pattern must be visible at default threshold"
        )
        match = next(p for p in results if p.id == "ls001")
        assert match.priority_score >= 0.3

    def test_ls_002_frequency_factor_has_floor(self) -> None:
        """TEST-LS-002: Frequency factor for occurrence_count=1 is >= 0.5,
        and resulting priority for eff=0.5, var=0.0 is >= 0.25."""
        mixin = PatternCrudMixin()
        priority = mixin._calculate_priority_score(
            effectiveness=0.5, occurrence_count=1, variance=0.0,
        )

        # The raw frequency factor is log10(2)/2 ≈ 0.15, but the floor
        # raises it to >= 0.5 (actually 0.6).
        raw_freq = min(1.0, math.log10(2) / 2.0)
        effective_freq = max(PatternCrudMixin.FREQUENCY_FACTOR_FLOOR, raw_freq)
        assert effective_freq >= 0.5, "Frequency factor floor must be >= 0.5"
        assert priority >= 0.25, (
            f"Priority for eff=0.5 occ=1 var=0.0 must be >= 0.25, got {priority}"
        )

    def test_ls_003_semantic_insights_remain_queryable(
        self, store: GlobalLearningStore,
    ) -> None:
        """TEST-LS-003: 100 SEMANTIC_INSIGHT patterns with occ=1 are mostly
        returned by get_patterns(pattern_type=..., min_priority=0.3)."""
        with store._get_connection() as conn:
            for i in range(100):
                _insert_pattern_raw(
                    store,
                    pattern_id=f"si_{i:03d}",
                    effectiveness=0.5,
                    occurrence_count=1,
                    variance=0.0,
                    pattern_type="SEMANTIC_INSIGHT",
                    pattern_name=f"insight {i}",
                    conn=conn,
                )
            for i in range(10):
                _insert_pattern_raw(
                    store,
                    pattern_id=f"ep_{i:03d}",
                    effectiveness=0.7,
                    occurrence_count=50,
                    variance=0.1,
                    pattern_type="EXECUTION_PATTERN",
                    pattern_name=f"exec pattern {i}",
                    conn=conn,
                )

        results = store.get_patterns(
            pattern_type="SEMANTIC_INSIGHT", min_priority=0.3, limit=200,
        )
        assert len(results) >= 80, (
            f"At least 80 of 100 semantic insights must be returned, got {len(results)}"
        )
        # Verify ordering
        priorities = [p.priority_score for p in results]
        assert priorities == sorted(priorities, reverse=True)

    def test_ls_004_recalculation_preserves_visibility(
        self, store: GlobalLearningStore,
    ) -> None:
        """TEST-LS-004: After recalculate_all_pattern_priorities(), a
        zero-application pattern stays visible."""
        pid = store.record_pattern(
            pattern_type="RECALC_TEST",
            pattern_name="recalc test pattern",
            description="Should survive recalculation",
        )

        # Verify initially visible
        results = store.get_patterns(min_priority=0.3, limit=100)
        ids = [p.id for p in results]
        assert pid in ids, "Pattern must be visible before recalculation"

        # Recalculate all priorities
        store.recalculate_all_pattern_priorities()

        # Must still be visible
        results = store.get_patterns(min_priority=0.3, limit=100)
        ids = [p.id for p in results]
        assert pid in ids, "Pattern must remain visible after recalculation"
        match = next(p for p in results if p.id == pid)
        assert match.priority_score >= 0.3

    def test_ls_005_high_variance_still_penalized(
        self, store: GlobalLearningStore,
    ) -> None:
        """TEST-LS-005: High-variance pattern (eff=0.8, occ=50, var=0.9)
        has priority < 0.15 and does NOT appear at min_priority=0.3."""
        _insert_pattern_raw(
            store,
            pattern_id="hv001",
            effectiveness=0.8,
            occurrence_count=50,
            variance=0.9,
        )
        # Check raw priority
        mixin = PatternCrudMixin()
        priority = mixin._calculate_priority_score(
            effectiveness=0.8, occurrence_count=50, variance=0.9,
        )
        assert priority < 0.15, (
            f"High-variance priority must be < 0.15, got {priority}"
        )

        # Must not appear at default threshold
        results = store.get_patterns(min_priority=0.3, limit=100)
        ids = [p.id for p in results]
        assert "hv001" not in ids

    def test_ls_006_healthy_priority_distribution(
        self, store: GlobalLearningStore,
    ) -> None:
        """TEST-LS-006: A realistic pattern population has a healthy
        priority spread — at least 500 of 1000 patterns have priority >= 0.3,
        and high-variance group mostly < 0.3."""
        import random

        random.seed(42)  # Deterministic

        with store._get_connection() as conn:
            # Group 1: 200 patterns, occ=1, eff=0.5
            for i in range(200):
                _insert_pattern_raw(
                    store,
                    pattern_id=f"g1_{i:03d}",
                    effectiveness=0.5,
                    occurrence_count=1,
                    variance=0.0,
                    pattern_name=f"group1 {i}",
                    conn=conn,
                )

            # Group 2: 300 patterns, occ=2-5, eff=0.3-0.7
            for i in range(300):
                _insert_pattern_raw(
                    store,
                    pattern_id=f"g2_{i:03d}",
                    effectiveness=0.3 + random.random() * 0.4,
                    occurrence_count=random.randint(2, 5),
                    variance=random.random() * 0.2,
                    pattern_name=f"group2 {i}",
                    conn=conn,
                )

            # Group 3: 300 patterns, occ=10-50, eff=0.5-0.9
            for i in range(300):
                _insert_pattern_raw(
                    store,
                    pattern_id=f"g3_{i:03d}",
                    effectiveness=0.5 + random.random() * 0.4,
                    occurrence_count=random.randint(10, 50),
                    variance=random.random() * 0.2,
                    pattern_name=f"group3 {i}",
                    conn=conn,
                )

            # Group 4: 200 high-variance, occ=1, eff=0.5, var=0.8
            for i in range(200):
                _insert_pattern_raw(
                    store,
                    pattern_id=f"g4_{i:03d}",
                    effectiveness=0.5,
                    occurrence_count=1,
                    variance=0.8,
                    pattern_name=f"group4 {i}",
                    conn=conn,
                )

        # Query all patterns
        all_patterns = store.get_patterns(min_priority=0.0, limit=1000)
        above_threshold = [p for p in all_patterns if p.priority_score >= 0.3]
        assert len(above_threshold) >= 500, (
            f"At least 500 patterns must have priority >= 0.3, got {len(above_threshold)}"
        )

        # High-variance group should mostly be below threshold
        g4_patterns = [p for p in all_patterns if p.id.startswith("g4_")]
        g4_above = [p for p in g4_patterns if p.priority_score >= 0.3]
        assert len(g4_above) < len(g4_patterns) * 0.2, (
            f"High-variance group should mostly be below 0.3 threshold, "
            f"got {len(g4_above)}/{len(g4_patterns)} above"
        )


# ---------------------------------------------------------------------------
# TEST-FK-001 through TEST-FK-006: FK Constraint and Soft Delete
# ---------------------------------------------------------------------------


class TestFKConstraints:
    """Tests for FK constraint handling and soft delete (issue #129)."""

    def test_fk_001_feedback_succeeds_when_pattern_exists(
        self, store: GlobalLearningStore,
    ) -> None:
        """TEST-FK-001: Recording pattern application succeeds for
        existing patterns."""
        pid = store.record_pattern(
            pattern_type="FK_TEST",
            pattern_name="fk test pattern",
        )

        app_id = store.record_pattern_application(
            pattern_id=pid,
            execution_id="fk_test_exec_001",
            pattern_led_to_success=True,
        )
        assert app_id is not None

        # Verify the application was recorded
        pattern = store.get_pattern_by_id(pid)
        assert pattern is not None
        assert pattern.led_to_success_count == 1

        # Verify no warnings logged (the happy path)
        with store._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM pattern_applications WHERE pattern_id = ?",
                (pid,),
            )
            assert cursor.fetchone()["cnt"] == 1

    def test_fk_002_feedback_for_deleted_pattern_does_not_raise(
        self, store: GlobalLearningStore, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """TEST-FK-002: Feedback for a non-existent pattern does not raise,
        and logs a structured warning."""
        app_id = store.record_pattern_application(
            pattern_id="nonexistent_pattern_id",
            execution_id="exec_id_123",
            pattern_led_to_success=True,
        )

        assert app_id is not None  # Returns an ID even on skip

        # Verify warning was logged (structlog outputs to stdout)
        captured = capsys.readouterr()
        assert "pattern_application_skipped" in captured.out
        assert "pattern_not_found" in captured.out

    def test_fk_003_concurrent_deletion_and_feedback(
        self, store: GlobalLearningStore,
    ) -> None:
        """TEST-FK-003: Concurrent pattern soft-delete and feedback recording
        does not raise IntegrityError."""
        pid = store.record_pattern(
            pattern_type="RACE_TEST",
            pattern_name="race test pattern",
        )

        errors: list[Exception] = []
        barrier = threading.Barrier(2, timeout=5)

        def delete_pattern() -> None:
            try:
                barrier.wait()
                store.soft_delete_pattern(pid)
            except Exception as e:
                errors.append(e)

        def record_feedback() -> None:
            try:
                barrier.wait()
                store.record_pattern_application(
                    pattern_id=pid,
                    execution_id="race_exec_001",
                    pattern_led_to_success=True,
                )
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=delete_pattern)
        t2 = threading.Thread(target=record_feedback)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # No IntegrityError should propagate
        integrity_errors = [
            e for e in errors if isinstance(e, sqlite3.IntegrityError)
        ]
        assert len(integrity_errors) == 0, (
            f"IntegrityError must not propagate: {integrity_errors}"
        )

    def test_fk_004_soft_delete_preserves_fk_integrity(
        self, store: GlobalLearningStore,
    ) -> None:
        """TEST-FK-004: Soft-deleted pattern's applications remain queryable,
        new applications succeed, and get_patterns excludes by default."""
        pid = store.record_pattern(
            pattern_type="SOFT_DEL_TEST",
            pattern_name="soft delete test",
        )

        # Record 5 applications
        for i in range(5):
            store.record_pattern_application(
                pattern_id=pid,
                execution_id=f"sd_exec_{i}",
                pattern_led_to_success=True,
            )

        # Soft delete
        assert store.soft_delete_pattern(pid) is True

        # Default get_patterns must NOT return it
        results = store.get_patterns(min_priority=0.0, limit=100)
        ids = [p.id for p in results]
        assert pid not in ids, "Soft-deleted pattern must not appear in default query"

        # With include_inactive=True, it must appear
        results_all = store.get_patterns(
            min_priority=0.0, limit=100, include_inactive=True,
        )
        ids_all = [p.id for p in results_all]
        assert pid in ids_all, "Soft-deleted pattern must appear with include_inactive=True"

        # New applications referencing the soft-deleted pattern must succeed
        # (the row still exists, so no FK violation)
        app_id = store.record_pattern_application(
            pattern_id=pid,
            execution_id="sd_exec_new",
            pattern_led_to_success=True,
        )
        assert app_id is not None

        # Existing applications must be intact
        with store._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM pattern_applications WHERE pattern_id = ?",
                (pid,),
            )
            count = cursor.fetchone()["cnt"]
        assert count >= 5, f"Existing applications must remain, got {count}"

    def test_fk_005_update_success_factors_handles_missing_pattern(
        self, store: GlobalLearningStore,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """TEST-FK-005: update_success_factors for a missing pattern returns
        None, does not crash, and logs a warning."""
        result = store.update_success_factors(
            pattern_id="gone999",
            validation_types=["file_exists"],
        )

        assert result is None

        # Verify warning was logged (structlog outputs to stdout)
        captured = capsys.readouterr()
        assert "update_success_factors_skipped" in captured.out
        assert "pattern_not_found" in captured.out

    def test_fk_006_bulk_feedback_after_pruning(
        self, store: GlobalLearningStore,
    ) -> None:
        """TEST-FK-006: After soft-deleting 25 of 50 patterns, feedback
        for the remaining 25 succeeds without errors."""
        pattern_ids = []
        for i in range(50):
            pid = store.record_pattern(
                pattern_type="BULK_TEST",
                pattern_name=f"bulk pattern {i}",
            )
            pattern_ids.append(pid)
            # Record 10 applications each
            for j in range(10):
                store.record_pattern_application(
                    pattern_id=pid,
                    execution_id=f"bulk_exec_{i}_{j}",
                    pattern_led_to_success=True,
                )

        # Soft-delete 25 patterns
        pruned = pattern_ids[:25]
        kept = pattern_ids[25:]
        for pid in pruned:
            store.soft_delete_pattern(pid)

        # New feedback for the remaining 25 must all succeed
        errors: list[Exception] = []
        for pid in kept:
            try:
                store.record_pattern_application(
                    pattern_id=pid,
                    execution_id=f"bulk_post_prune_{pid}",
                    pattern_led_to_success=True,
                )
            except Exception as e:
                errors.append(e)

        assert len(errors) == 0, f"All 25 feedbacks must succeed: {errors}"

        # Historical applications for pruned patterns must not cause errors
        with store._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM pattern_applications"
            )
            total = cursor.fetchone()["cnt"]
        # 50 patterns × 10 apps + 25 kept × 1 new app = 525
        assert total >= 525, (
            f"Total applications must be consistent, got {total}"
        )
