"""Tests for mozart.daemon.scheduler module.

Covers GlobalSheetScheduler: priority ordering, fair-share enforcement,
DAG-aware enqueue/dequeue, concurrency limits, rate-limited backend
skipping, backpressure integration, and stats reporting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mozart.daemon.config import DaemonConfig
from mozart.daemon.scheduler import (
    GlobalSheetScheduler,
    SheetInfo,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def daemon_config(tmp_path: Path) -> DaemonConfig:
    """DaemonConfig with low concurrency for testing."""
    return DaemonConfig(
        max_concurrent_sheets=3,
        max_concurrent_jobs=5,
        pid_file=tmp_path / "test.pid",
    )


@pytest.fixture
def scheduler(daemon_config: DaemonConfig) -> GlobalSheetScheduler:
    """Fresh scheduler instance."""
    return GlobalSheetScheduler(daemon_config)


def _sheet(
    job_id: str = "job-a",
    sheet_num: int = 1,
    job_priority: int = 5,
    dag_depth: int = 0,
    retries: int = 0,
    backend: str = "claude_cli",
) -> SheetInfo:
    """Helper to create a SheetInfo with sensible defaults."""
    return SheetInfo(
        job_id=job_id,
        sheet_num=sheet_num,
        job_priority=job_priority,
        dag_depth=dag_depth,
        retries_so_far=retries,
        backend_type=backend,
    )


# ─── Basic Operations ─────────────────────────────────────────────────


class TestRegisterAndDeregister:
    """Tests for register_job / deregister_job."""

    @pytest.mark.asyncio
    async def test_register_enqueues_independent_sheets(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Sheets with no dependencies are enqueued immediately."""
        sheets = [_sheet(sheet_num=i) for i in range(1, 4)]
        await scheduler.register_job("job-a", sheets)

        assert scheduler.queued_count == 3
        assert scheduler.active_count == 0

    @pytest.mark.asyncio
    async def test_register_respects_dag_deps(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Only DAG-ready sheets are enqueued on registration."""
        sheets = [_sheet(sheet_num=i) for i in range(1, 4)]
        deps = {2: {1}, 3: {2}}  # 1 → 2 → 3

        await scheduler.register_job("job-a", sheets, dependencies=deps)
        # Only sheet 1 has no deps
        assert scheduler.queued_count == 1

    @pytest.mark.asyncio
    async def test_deregister_removes_all_pending(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Deregistering a job removes all its queued sheets."""
        sheets = [_sheet(sheet_num=i) for i in range(1, 6)]
        await scheduler.register_job("job-a", sheets)
        assert scheduler.queued_count == 5

        await scheduler.deregister_job("job-a")
        assert scheduler.queued_count == 0

    @pytest.mark.asyncio
    async def test_deregister_doesnt_affect_other_jobs(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Deregistering job-a leaves job-b's sheets intact."""
        await scheduler.register_job("job-a", [_sheet("job-a", i) for i in range(1, 4)])
        await scheduler.register_job("job-b", [_sheet("job-b", i) for i in range(1, 3)])
        assert scheduler.queued_count == 5

        await scheduler.deregister_job("job-a")
        assert scheduler.queued_count == 2


# ─── Priority Ordering ────────────────────────────────────────────────


class TestPriorityOrdering:
    """Tests for priority-based sheet dispatch."""

    @pytest.mark.asyncio
    async def test_higher_priority_dispatched_first(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Sheets with lower job_priority number (= more urgent) dispatch first."""
        # Priority 1 (urgent) and priority 10 (low)
        sheets = [
            _sheet("job-a", sheet_num=1, job_priority=10),
            _sheet("job-a", sheet_num=2, job_priority=1),
        ]
        await scheduler.register_job("job-a", sheets)

        entry = await scheduler.next_sheet()
        assert entry is not None
        # Sheet 2 has priority=1 → lower score → dispatched first
        assert entry.info.sheet_num == 2

    @pytest.mark.asyncio
    async def test_retried_sheets_get_priority_boost(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Sheets with more retries get a priority boost (sunk cost)."""
        sheets = [
            _sheet("job-a", sheet_num=1, job_priority=5, retries=0),
            _sheet("job-a", sheet_num=2, job_priority=5, retries=5),
        ]
        await scheduler.register_job("job-a", sheets)

        entry = await scheduler.next_sheet()
        assert entry is not None
        # Sheet 2 has 5 retries → -25 priority reduction → goes first
        assert entry.info.sheet_num == 2

    @pytest.mark.asyncio
    async def test_shallow_dag_depth_preferred(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Shallower DAG depth sheets dispatch before deeper ones."""
        sheets = [
            _sheet("job-a", sheet_num=1, dag_depth=5),
            _sheet("job-a", sheet_num=2, dag_depth=0),
        ]
        await scheduler.register_job("job-a", sheets)

        entry = await scheduler.next_sheet()
        assert entry is not None
        assert entry.info.sheet_num == 2  # depth 0 preferred


# ─── Concurrency Limits ───────────────────────────────────────────────


class TestConcurrencyLimits:
    """Tests for max_concurrent_sheets enforcement."""

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Cannot dispatch more sheets than max_concurrent_sheets."""
        sheets = [_sheet(sheet_num=i) for i in range(1, 10)]
        await scheduler.register_job("job-a", sheets)

        dispatched = []
        for _ in range(10):
            entry = await scheduler.next_sheet()
            if entry is None:
                break
            dispatched.append(entry)

        assert len(dispatched) == 3  # max_concurrent_sheets = 3
        assert scheduler.active_count == 3

    @pytest.mark.asyncio
    async def test_completing_sheet_frees_slot(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Marking a sheet complete allows the next one to dispatch."""
        sheets = [_sheet(sheet_num=i) for i in range(1, 5)]
        await scheduler.register_job("job-a", sheets)

        # Fill up all 3 slots
        entries = []
        for _ in range(3):
            e = await scheduler.next_sheet()
            assert e is not None
            entries.append(e)

        # All slots full
        assert await scheduler.next_sheet() is None

        # Complete one
        await scheduler.mark_complete("job-a", entries[0].info.sheet_num, success=True)
        assert scheduler.active_count == 2

        # Now one more can dispatch
        entry = await scheduler.next_sheet()
        assert entry is not None
        assert scheduler.active_count == 3


# ─── DAG Dependencies ─────────────────────────────────────────────────


class TestDAGDependencies:
    """Tests for DAG-aware scheduling."""

    @pytest.mark.asyncio
    async def test_dependent_sheet_enqueued_after_dep_completes(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Completing sheet 1 makes sheet 2 (which depends on 1) available."""
        sheets = [_sheet(sheet_num=1), _sheet(sheet_num=2)]
        deps = {2: {1}}

        await scheduler.register_job("job-a", sheets, dependencies=deps)
        assert scheduler.queued_count == 1  # Only sheet 1

        # Dispatch and complete sheet 1
        entry = await scheduler.next_sheet()
        assert entry is not None
        assert entry.info.sheet_num == 1
        await scheduler.mark_complete("job-a", 1, success=True)

        # Sheet 2 should now be queued
        assert scheduler.queued_count == 1
        entry2 = await scheduler.next_sheet()
        assert entry2 is not None
        assert entry2.info.sheet_num == 2

    @pytest.mark.asyncio
    async def test_diamond_dag(self, scheduler: GlobalSheetScheduler):
        """Diamond dependency: 1 → {2, 3} → 4."""
        sheets = [_sheet(sheet_num=i) for i in range(1, 5)]
        deps = {2: {1}, 3: {1}, 4: {2, 3}}

        await scheduler.register_job("job-a", sheets, dependencies=deps)
        assert scheduler.queued_count == 1  # Only sheet 1

        # Complete sheet 1 → sheets 2, 3 become ready
        e1 = await scheduler.next_sheet()
        assert e1 is not None
        await scheduler.mark_complete("job-a", 1, success=True)
        assert scheduler.queued_count == 2  # Sheets 2 and 3

        # Complete sheets 2 and 3
        e2 = await scheduler.next_sheet()
        e3 = await scheduler.next_sheet()
        assert e2 is not None and e3 is not None
        await scheduler.mark_complete("job-a", e2.info.sheet_num, success=True)

        # Sheet 4 still waiting for the other dep
        assert scheduler.queued_count == 0  # Not yet ready (needs both 2 AND 3)

        await scheduler.mark_complete("job-a", e3.info.sheet_num, success=True)
        # Now sheet 4 should be ready
        assert scheduler.queued_count == 1
        e4 = await scheduler.next_sheet()
        assert e4 is not None
        assert e4.info.sheet_num == 4


# ─── Fair Share Scheduling ─────────────────────────────────────────────


class TestFairShare:
    """Tests for per-job fair-share scheduling."""

    @pytest.mark.asyncio
    async def test_hard_cap_prevents_single_job_hogging(
        self, daemon_config: DaemonConfig,
    ):
        """A single job can't take all slots when another job is registered."""
        # max_concurrent_sheets=3, two jobs.
        # fair_share = 3/2 = 1.5 → hard_cap = 1.5 * 2 = 3
        # With just one active job, fair_share = 3/1 = 3, hard_cap = 6
        # After second job registers and dispatches, fair share recalculates
        scheduler = GlobalSheetScheduler(daemon_config)

        # Register two jobs
        await scheduler.register_job(
            "job-a",
            [_sheet("job-a", i) for i in range(1, 6)],
        )
        await scheduler.register_job(
            "job-b",
            [_sheet("job-b", i, job_priority=1) for i in range(1, 4)],
        )

        # Dispatch all 3 slots — job-b has higher priority (priority=1)
        dispatched_jobs = []
        for _ in range(3):
            e = await scheduler.next_sheet()
            if e is not None:
                dispatched_jobs.append(e.info.job_id)

        assert len(dispatched_jobs) == 3
        # Job-b (priority=1) should get preference
        assert "job-b" in dispatched_jobs

    @pytest.mark.asyncio
    async def test_fair_share_interleaves_initial_dispatch(
        self, daemon_config: DaemonConfig,
    ):
        """With two equal-priority jobs, initial dispatch interleaves.

        Fair-share penalty increases priority score (= lower urgency)
        for jobs that already have running sheets, causing the scheduler
        to alternate between jobs.
        """
        config = daemon_config.model_copy(update={"max_concurrent_sheets": 4})
        scheduler = GlobalSheetScheduler(config)

        await scheduler.register_job(
            "job-a",
            [_sheet("job-a", i, job_priority=5) for i in range(1, 10)],
        )
        await scheduler.register_job(
            "job-b",
            [_sheet("job-b", i, job_priority=5) for i in range(1, 10)],
        )

        # Dispatch all 4 slots
        dispatched = {"job-a": 0, "job-b": 0}
        for _ in range(4):
            e = await scheduler.next_sheet()
            assert e is not None
            dispatched[e.info.job_id] += 1

        # Fair-share penalty should distribute slots across both jobs
        assert dispatched["job-a"] >= 1
        assert dispatched["job-b"] >= 1
        # With 4 slots and 2 jobs, expect roughly 2 each
        assert dispatched["job-a"] + dispatched["job-b"] == 4

    @pytest.mark.asyncio
    async def test_starved_job_catches_up_after_completion(
        self, daemon_config: DaemonConfig,
    ):
        """When a job with many running sheets completes some, the
        starved job gets the freed slots.
        """
        config = daemon_config.model_copy(update={"max_concurrent_sheets": 4})
        scheduler = GlobalSheetScheduler(config)

        await scheduler.register_job(
            "job-a",
            [_sheet("job-a", i, job_priority=5) for i in range(1, 10)],
        )
        await scheduler.register_job(
            "job-b",
            [_sheet("job-b", i, job_priority=5) for i in range(1, 10)],
        )

        # Fill all 4 slots — due to fair-share, should be 2+2
        entries = []
        for _ in range(4):
            e = await scheduler.next_sheet()
            assert e is not None
            entries.append(e)

        # Find which entries belong to which job
        job_a_entries = [e for e in entries if e.info.job_id == "job-a"]
        job_b_entries = [e for e in entries if e.info.job_id == "job-b"]

        # Complete all of job-b's running sheets
        for e in job_b_entries:
            await scheduler.mark_complete("job-b", e.info.sheet_num, True)

        # Now dispatch freed slots — job-b should get them back
        # (job-a has 2 running at fair-share, job-b has 0)
        new_entries = []
        for _ in range(len(job_b_entries)):
            e = await scheduler.next_sheet()
            if e is not None:
                new_entries.append(e)

        new_jobs = [e.info.job_id for e in new_entries]
        assert "job-b" in new_jobs


# ─── Rate Limit Integration ───────────────────────────────────────────


class TestRateLimitIntegration:
    """Tests for rate-limiter integration via protocol."""

    @pytest.mark.asyncio
    async def test_rate_limited_sheets_skipped(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Sheets using a rate-limited backend are skipped."""

        class MockRateLimiter:
            async def is_rate_limited(
                self, backend_type: str, model: str | None = None,
            ) -> tuple[bool, float]:
                return (backend_type == "claude_cli", 30.0)

        scheduler.set_rate_limiter(MockRateLimiter())

        sheets = [
            _sheet("job-a", 1, backend="claude_cli"),
            _sheet("job-a", 2, backend="openai"),
        ]
        await scheduler.register_job("job-a", sheets)

        # Only the openai sheet should dispatch
        entry = await scheduler.next_sheet()
        assert entry is not None
        assert entry.info.backend_type == "openai"

        # The claude_cli sheet should be skipped (put back in queue)
        assert scheduler.queued_count == 1

    @pytest.mark.asyncio
    async def test_rate_limited_skip_preserves_scheduler_state(
        self, scheduler: GlobalSheetScheduler,
    ):
        """When all queued sheets are rate-limited, scheduler state is unchanged.

        Verifies that skipping rate-limited sheets does not corrupt
        active_count, queued_count, or per-job tracking.
        """

        class AllLimitedRateLimiter:
            async def is_rate_limited(
                self, backend_type: str, model: str | None = None,
            ) -> tuple[bool, float]:
                return (True, 60.0)

        scheduler.set_rate_limiter(AllLimitedRateLimiter())

        sheets = [_sheet("job-a", i, backend="claude_cli") for i in range(1, 4)]
        await scheduler.register_job("job-a", sheets)

        # Capture state before dispatch attempt
        queued_before = scheduler.queued_count
        active_before = scheduler.active_count

        # All sheets are rate-limited — nothing should dispatch
        entry = await scheduler.next_sheet()
        assert entry is None

        # State must be unchanged
        assert scheduler.queued_count == queued_before
        assert scheduler.active_count == active_before
        assert scheduler.active_count == 0

        # Stats should also reflect the unchanged state
        stats = await scheduler.get_stats()
        assert stats.queued == 3
        assert stats.active == 0
        assert stats.per_job_queued == {"job-a": 3}
        assert stats.per_job_running == {"job-a": 0}

    @pytest.mark.asyncio
    async def test_no_rate_limiter_dispatches_normally(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Without a rate limiter, all sheets dispatch normally."""
        sheets = [_sheet(sheet_num=i) for i in range(1, 4)]
        await scheduler.register_job("job-a", sheets)

        dispatched = 0
        for _ in range(3):
            e = await scheduler.next_sheet()
            if e is not None:
                dispatched += 1

        assert dispatched == 3


# ─── Backpressure Integration ─────────────────────────────────────────


class TestBackpressureIntegration:
    """Tests for backpressure controller integration."""

    @pytest.mark.asyncio
    async def test_backpressure_blocks_dispatch(
        self, scheduler: GlobalSheetScheduler,
    ):
        """When backpressure says no, next_sheet returns None."""

        class MockBackpressure:
            async def can_start_sheet(self) -> tuple[bool, float]:
                return (False, 0.0)

        scheduler.set_backpressure(MockBackpressure())

        sheets = [_sheet(sheet_num=i) for i in range(1, 4)]
        await scheduler.register_job("job-a", sheets)

        # Backpressure blocks all dispatch
        assert await scheduler.next_sheet() is None
        assert scheduler.queued_count == 3  # Still queued

    @pytest.mark.asyncio
    async def test_backpressure_soft_delay(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Soft backpressure adds a delay but still allows dispatch."""

        class MockSoftBackpressure:
            async def can_start_sheet(self) -> tuple[bool, float]:
                return (True, 0.001)  # 1ms delay

        scheduler.set_backpressure(MockSoftBackpressure())

        sheets = [_sheet(sheet_num=1)]
        await scheduler.register_job("job-a", sheets)

        entry = await scheduler.next_sheet()
        assert entry is not None
        assert entry.info.sheet_num == 1


# ─── Stats ─────────────────────────────────────────────────────────────


class TestStats:
    """Tests for scheduler statistics."""

    @pytest.mark.asyncio
    async def test_stats_reflect_state(self, scheduler: GlobalSheetScheduler):
        """Stats accurately report queued and active counts."""
        await scheduler.register_job(
            "job-a", [_sheet("job-a", i) for i in range(1, 5)],
        )

        stats = await scheduler.get_stats()
        assert stats.queued == 4
        assert stats.active == 0
        assert stats.max_concurrent == 3
        assert stats.per_job_queued == {"job-a": 4}
        assert stats.per_job_running == {"job-a": 0}

        # Dispatch 2
        await scheduler.next_sheet()
        await scheduler.next_sheet()

        stats = await scheduler.get_stats()
        assert stats.active == 2
        assert stats.queued == 2
        assert stats.per_job_running == {"job-a": 2}

    @pytest.mark.asyncio
    async def test_stats_multiple_jobs(self, scheduler: GlobalSheetScheduler):
        """Stats track per-job counts correctly across multiple jobs."""
        await scheduler.register_job(
            "job-a", [_sheet("job-a", i) for i in range(1, 3)],
        )
        await scheduler.register_job(
            "job-b", [_sheet("job-b", i) for i in range(1, 4)],
        )

        stats = await scheduler.get_stats()
        assert stats.queued == 5
        assert stats.per_job_queued.get("job-a") == 2
        assert stats.per_job_queued.get("job-b") == 3


# ─── Edge Cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_next_sheet_on_empty_queue(
        self, scheduler: GlobalSheetScheduler,
    ):
        """next_sheet returns None on an empty queue."""
        assert await scheduler.next_sheet() is None

    @pytest.mark.asyncio
    async def test_mark_complete_unknown_job(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Marking complete for an unknown job doesn't raise."""
        # Should be a no-op, not crash
        await scheduler.mark_complete("nonexistent", 1, success=True)
        assert scheduler.active_count == 0

    @pytest.mark.asyncio
    async def test_deregister_unknown_job(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Deregistering an unknown job is a no-op."""
        await scheduler.deregister_job("nonexistent")
        assert scheduler.queued_count == 0

    @pytest.mark.asyncio
    async def test_register_empty_sheet_list(
        self, scheduler: GlobalSheetScheduler,
    ):
        """Registering a job with no sheets is valid."""
        await scheduler.register_job("empty-job", [])
        assert scheduler.queued_count == 0

    @pytest.mark.asyncio
    async def test_properties_match_stats(
        self, scheduler: GlobalSheetScheduler,
    ):
        """active_count and queued_count properties match stats."""
        await scheduler.register_job(
            "job-a", [_sheet(sheet_num=i) for i in range(1, 4)],
        )
        await scheduler.next_sheet()

        stats = await scheduler.get_stats()
        assert scheduler.active_count == stats.active
        assert scheduler.queued_count == stats.queued
