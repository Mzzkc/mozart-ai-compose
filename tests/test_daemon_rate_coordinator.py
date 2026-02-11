"""Tests for mozart.daemon.rate_coordinator module.

Covers RateLimitCoordinator: reporting, querying via the RateLimitChecker
protocol, waiting, sync helpers, event pruning, and integration with
the GlobalSheetScheduler.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from mozart.daemon.rate_coordinator import RateLimitCoordinator, RateLimitEvent


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def coordinator() -> RateLimitCoordinator:
    """Fresh coordinator instance."""
    return RateLimitCoordinator()


# ─── Basic Reporting ───────────────────────────────────────────────────


class TestReporting:
    """Tests for report_rate_limit."""

    @pytest.mark.asyncio
    async def test_report_creates_active_limit(
        self, coordinator: RateLimitCoordinator,
    ):
        """Reporting a limit makes the backend rate-limited."""
        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=30.0,
            job_id="job-a",
            sheet_num=1,
        )

        is_limited, remaining = await coordinator.is_rate_limited("claude_cli")
        assert is_limited is True
        assert remaining > 0
        assert remaining <= 30.0

    @pytest.mark.asyncio
    async def test_report_extends_existing_limit(
        self, coordinator: RateLimitCoordinator,
    ):
        """A longer wait replaces a shorter existing limit."""
        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=10.0,
            job_id="job-a",
            sheet_num=1,
        )
        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=60.0,
            job_id="job-b",
            sheet_num=2,
        )

        is_limited, remaining = await coordinator.is_rate_limited("claude_cli")
        assert is_limited is True
        # Should reflect the longer limit, not the shorter one
        assert remaining > 10.0

    @pytest.mark.asyncio
    async def test_report_doesnt_shorten_limit(
        self, coordinator: RateLimitCoordinator,
    ):
        """A shorter wait does NOT reduce an existing longer limit."""
        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=60.0,
            job_id="job-a",
            sheet_num=1,
        )
        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=5.0,
            job_id="job-b",
            sheet_num=2,
        )

        is_limited, remaining = await coordinator.is_rate_limited("claude_cli")
        assert is_limited is True
        assert remaining > 5.0  # Still waiting for the longer one

    @pytest.mark.asyncio
    async def test_different_backends_independent(
        self, coordinator: RateLimitCoordinator,
    ):
        """Limits on one backend don't affect others."""
        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=30.0,
            job_id="job-a",
            sheet_num=1,
        )

        is_limited, remaining = await coordinator.is_rate_limited("openai")
        assert is_limited is False
        assert remaining == 0.0

    @pytest.mark.asyncio
    async def test_report_stores_event(
        self, coordinator: RateLimitCoordinator,
    ):
        """Reported events are stored and accessible."""
        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=30.0,
            job_id="job-a",
            sheet_num=3,
        )

        events = coordinator.recent_events
        assert len(events) == 1
        assert events[0].backend_type == "claude_cli"
        assert events[0].job_id == "job-a"
        assert events[0].sheet_num == 3
        assert events[0].suggested_wait_seconds == 30.0


# ─── Protocol Compliance ──────────────────────────────────────────────


class TestProtocolCompliance:
    """Tests that is_rate_limited satisfies the RateLimitChecker protocol."""

    @pytest.mark.asyncio
    async def test_returns_tuple_when_limited(
        self, coordinator: RateLimitCoordinator,
    ):
        """Returns (True, float) when limited."""
        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=30.0,
            job_id="job-a",
            sheet_num=1,
        )

        result = await coordinator.is_rate_limited("claude_cli")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is True
        assert isinstance(result[1], float)

    @pytest.mark.asyncio
    async def test_returns_tuple_when_not_limited(
        self, coordinator: RateLimitCoordinator,
    ):
        """Returns (False, 0.0) when not limited."""
        result = await coordinator.is_rate_limited("claude_cli")
        assert result == (False, 0.0)

    @pytest.mark.asyncio
    async def test_accepts_model_parameter(
        self, coordinator: RateLimitCoordinator,
    ):
        """The model parameter is accepted (protocol compat)."""
        result = await coordinator.is_rate_limited(
            "claude_cli", model="claude-sonnet-4-5-20250929",
        )
        assert result == (False, 0.0)


# ─── Waiting ──────────────────────────────────────────────────────────


class TestWaiting:
    """Tests for wait_if_limited."""

    @pytest.mark.asyncio
    async def test_returns_zero_when_not_limited(
        self, coordinator: RateLimitCoordinator,
    ):
        """No wait when backend is clear."""
        waited = await coordinator.wait_if_limited("claude_cli")
        assert waited == 0.0

    @pytest.mark.asyncio
    async def test_waits_for_limit_to_clear(
        self, coordinator: RateLimitCoordinator,
    ):
        """Waits approximately the right duration."""
        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=0.1,  # 100ms — fast for tests
            job_id="job-a",
            sheet_num=1,
        )

        start = time.monotonic()
        waited = await coordinator.wait_if_limited("claude_cli")
        elapsed = time.monotonic() - start

        assert waited > 0
        assert elapsed >= 0.05  # At least half the wait
        assert elapsed < 1.0  # Sanity bound


# ─── Sync Helpers ─────────────────────────────────────────────────────


class TestSyncHelpers:
    """Tests for sync helpers and properties."""

    @pytest.mark.asyncio
    async def test_is_limited_sync(
        self, coordinator: RateLimitCoordinator,
    ):
        """Sync check matches async check."""
        assert coordinator.is_limited_sync("claude_cli") is False

        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=30.0,
            job_id="job-a",
            sheet_num=1,
        )

        assert coordinator.is_limited_sync("claude_cli") is True
        assert coordinator.is_limited_sync("openai") is False

    @pytest.mark.asyncio
    async def test_active_limits_property(
        self, coordinator: RateLimitCoordinator,
    ):
        """active_limits returns only currently active limits."""
        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=30.0,
            job_id="job-a",
            sheet_num=1,
        )
        await coordinator.report_rate_limit(
            backend_type="openai",
            wait_seconds=60.0,
            job_id="job-b",
            sheet_num=2,
        )

        limits = coordinator.active_limits
        assert "claude_cli" in limits
        assert "openai" in limits
        assert limits["claude_cli"] > 0
        assert limits["openai"] > 0
        assert limits["openai"] > limits["claude_cli"]

    @pytest.mark.asyncio
    async def test_active_limits_excludes_expired(
        self, coordinator: RateLimitCoordinator,
    ):
        """Expired limits are not included in active_limits."""
        # Report a limit that expires almost immediately
        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=0.01,
            job_id="job-a",
            sheet_num=1,
        )
        await asyncio.sleep(0.02)  # Let it expire

        limits = coordinator.active_limits
        assert "claude_cli" not in limits

    @pytest.mark.asyncio
    async def test_recent_events_ordered_newest_first(
        self, coordinator: RateLimitCoordinator,
    ):
        """recent_events returns newest events first."""
        for i in range(3):
            await coordinator.report_rate_limit(
                backend_type="claude_cli",
                wait_seconds=float(i + 1),
                job_id=f"job-{i}",
                sheet_num=i + 1,
            )

        events = coordinator.recent_events
        assert len(events) == 3
        # Most recent first
        assert events[0].job_id == "job-2"
        assert events[-1].job_id == "job-0"


# ─── Event Pruning ────────────────────────────────────────────────────


class TestEventPruning:
    """Tests for event list housekeeping."""

    @pytest.mark.asyncio
    async def test_old_events_pruned_on_report(
        self, coordinator: RateLimitCoordinator,
    ):
        """Events older than 1 hour are pruned when new events arrive."""
        # Manually inject an old event
        old_event = RateLimitEvent(
            backend_type="claude_cli",
            detected_at=time.monotonic() - 7200,  # 2 hours ago
            suggested_wait_seconds=30.0,
            job_id="old-job",
            sheet_num=1,
        )
        coordinator._events.append(old_event)
        assert len(coordinator._events) == 1

        # Report a new event — old one should be pruned
        await coordinator.report_rate_limit(
            backend_type="openai",
            wait_seconds=10.0,
            job_id="new-job",
            sheet_num=1,
        )

        assert len(coordinator._events) == 1
        assert coordinator._events[0].job_id == "new-job"


# ─── Scheduler Integration ────────────────────────────────────────────


class TestSchedulerIntegration:
    """Test that RateLimitCoordinator works with GlobalSheetScheduler."""

    @pytest.mark.asyncio
    async def test_coordinator_as_rate_limiter(self):
        """Coordinator satisfies RateLimitChecker protocol in scheduler."""
        from mozart.daemon.config import DaemonConfig
        from mozart.daemon.scheduler import GlobalSheetScheduler, SheetInfo

        config = DaemonConfig(max_concurrent_sheets=5, max_concurrent_jobs=5)
        scheduler = GlobalSheetScheduler(config)
        coordinator = RateLimitCoordinator()

        # Wire up
        scheduler.set_rate_limiter(coordinator)

        # Register sheets with different backends
        sheets = [
            SheetInfo(job_id="job-a", sheet_num=1, backend_type="claude_cli"),
            SheetInfo(job_id="job-a", sheet_num=2, backend_type="openai"),
        ]
        await scheduler.register_job("job-a", sheets)

        # Rate-limit claude_cli
        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=60.0,
            job_id="job-a",
            sheet_num=1,
        )

        # Only the openai sheet should dispatch
        entry = await scheduler.next_sheet()
        assert entry is not None
        assert entry.info.backend_type == "openai"

        # claude_cli sheet stays in queue
        assert scheduler.queued_count == 1

    @pytest.mark.asyncio
    async def test_coordinator_clears_after_expiry(self):
        """After limit expires, scheduler can dispatch that backend."""
        from mozart.daemon.config import DaemonConfig
        from mozart.daemon.scheduler import GlobalSheetScheduler, SheetInfo

        config = DaemonConfig(max_concurrent_sheets=5, max_concurrent_jobs=5)
        scheduler = GlobalSheetScheduler(config)
        coordinator = RateLimitCoordinator()
        scheduler.set_rate_limiter(coordinator)

        sheets = [
            SheetInfo(job_id="job-a", sheet_num=1, backend_type="claude_cli"),
        ]
        await scheduler.register_job("job-a", sheets)

        # Rate-limit for a very short time
        await coordinator.report_rate_limit(
            backend_type="claude_cli",
            wait_seconds=0.01,
            job_id="job-a",
            sheet_num=1,
        )

        # Wait for limit to expire
        await asyncio.sleep(0.02)

        # Now it should dispatch
        entry = await scheduler.next_sheet()
        assert entry is not None
        assert entry.info.backend_type == "claude_cli"
