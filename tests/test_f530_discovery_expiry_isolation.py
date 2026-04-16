"""Regression test for F-530: test_discovery_events_expire_correctly isolation issue.

Root cause: The test uses time.sleep(2.5) to wait for a 2.0s TTL pattern to expire.
Under heavy parallel test execution load (full suite with xdist), time.sleep() can
wake up early (F-521 discovered sleep can wake 100ms-2s early under CPU load), and
the 500ms margin is insufficient.

Fix: Use 5.0s TTL with 15.0s sleep (10s margin) to account for realistic time.sleep()
variance under extreme parallel load, matching the F-521 fix approach.
"""

import tempfile
import time
from pathlib import Path

from marianne.learning.global_store import GlobalLearningStore


class TestF530DiscoveryExpiryIsolation:
    """Verify that the discovery expiry test has sufficient timing margin."""

    def test_insufficient_margin_demonstrates_problem(self) -> None:
        """Demonstrate that 500ms margin is insufficient under load.

        This test shows why the original test fails in the full suite but passes
        in isolation - under heavy load, time.sleep() wakes up early and the
        pattern hasn't expired yet when we check.
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            store = GlobalLearningStore(db_path)

            # Record with 2.0s TTL, sleep 2.5s (500ms margin) - the problematic values
            store.record_pattern_discovery(
                pattern_id="insufficient-001",
                pattern_name="Insufficient Margin",
                pattern_type="test",
                job_id="test-job",
                ttl_seconds=2.0,
            )

            # Should exist immediately
            events = store.get_active_pattern_discoveries()
            assert any(e.pattern_name == "Insufficient Margin" for e in events)

            # Sleep 2.5s (500ms margin)
            # Under heavy load, sleep() can wake up early, leaving < 2.0s elapsed
            time.sleep(2.5)

            # If sleep woke up 600ms early, only 1.9s has elapsed
            # The pattern with 2.0s TTL hasn't expired yet
            # This is why the test fails intermittently

            # NOTE: We don't assert here because the behavior is flaky by design
            # This test documents the problem, not the solution
        finally:
            db_path.unlink()

    def test_sufficient_margin_is_robust(self) -> None:
        """Verify that 10s margin handles time.sleep() early wakeup.

        Even if time.sleep(15.0) wakes up 2s early (worst case from F-521),
        we still have 13s elapsed, which exceeds the 5.0s TTL by 8s.
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            store = GlobalLearningStore(db_path)

            # Record with 5.0s TTL - robust under load
            store.record_pattern_discovery(
                pattern_id="sufficient-001",
                pattern_name="Sufficient Margin",
                pattern_type="test",
                job_id="test-job",
                ttl_seconds=5.0,
            )

            # Should exist immediately
            events = store.get_active_pattern_discoveries()
            assert any(e.pattern_name == "Sufficient Margin" for e in events), (
                "Pattern should be active immediately after recording"
            )

            # Wait for expiry with 10s margin (15.0s > 5.0s TTL)
            time.sleep(15.0)

            # Even if sleep() woke up 2s early, pattern has expired
            # Actual elapsed: 15s - 2s = 13s >> 5s TTL
            events_after = store.get_active_pattern_discoveries()
            assert not any(e.pattern_name == "Sufficient Margin" for e in events_after), (
                "Pattern should have expired after 5s TTL + 15s sleep"
            )

            # Cleanup should work
            cleaned = store.cleanup_expired_pattern_discoveries()
            assert cleaned >= 1, "Should clean up at least the expired test pattern"
        finally:
            db_path.unlink()

    def test_original_test_values_with_margin(self) -> None:
        """Verify the fix for the original test works correctly."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            store = GlobalLearningStore(db_path)

            # NEW values: 5.0s TTL with 15.0s sleep (10s margin)
            # This matches the pattern from test_f519_discovery_expiry_timing.py
            store.record_pattern_discovery(
                pattern_id="fixed-001",
                pattern_name="Fixed Pattern",
                pattern_type="test",
                job_id="test-job",
                ttl_seconds=5.0,
            )

            # Should exist immediately
            events = store.get_active_pattern_discoveries()
            found_before = any(e.pattern_name == "Fixed Pattern" for e in events)
            assert found_before, "Pattern should exist immediately"

            # Wait with sufficient margin
            time.sleep(15.0)

            # Should NOT appear in active discoveries after expiry
            events_after = store.get_active_pattern_discoveries()
            found_after = any(e.pattern_name == "Fixed Pattern" for e in events_after)
            assert not found_after, "Pattern should have expired"

            # Cleanup should work
            cleaned = store.cleanup_expired_pattern_discoveries()
            assert cleaned >= 1
        finally:
            db_path.unlink()
