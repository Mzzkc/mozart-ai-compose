"""TDD tests for F-211: Baton checkpoint sync for ALL status-changing events.

The baton's _sync_sheet_status handled only specific event types, missing
others that change sheet status. The fix uses state-diff syncing: after
each event, compare baton sheet statuses against a cache of last-synced
checkpoint values. Sync any that changed. Event-type agnostic, future-proof.

Only tests transitions that produce VISIBLE checkpoint status changes
(different strings in the _BATON_TO_CHECKPOINT mapping).
"""

from __future__ import annotations

from unittest.mock import ANY, MagicMock

from marianne.daemon.baton.adapter import BatonAdapter
from marianne.daemon.baton.events import (
    BatonEvent,
    CancelJob,
    EscalationNeeded,
    EscalationResolved,
    EscalationTimeout,
    JobTimeout,
    RateLimitExpired,
    RateLimitHit,
    RetryDue,
    SheetAttemptResult,
    SheetSkipped,
    ShutdownRequested,
)
from marianne.daemon.baton.state import BatonSheetStatus, SheetExecutionState


# =============================================================================
# Helpers
# =============================================================================


def _make_sheet(
    num: int,
    instrument: str = "claude-cli",
    max_retries: int = 3,
    status: BatonSheetStatus = BatonSheetStatus.PENDING,
) -> SheetExecutionState:
    """Create a SheetExecutionState for testing."""
    state = SheetExecutionState(
        sheet_num=num,
        instrument_name=instrument,
        max_retries=max_retries,
    )
    state.status = status
    return state


def _make_adapter(
    sync_callback: MagicMock | None = None,
) -> BatonAdapter:
    """Create a BatonAdapter with a mock sync callback."""
    cb = sync_callback or MagicMock()
    return BatonAdapter(state_sync_callback=cb)


def _register_job(
    adapter: BatonAdapter,
    job_id: str = "test-job",
    num_sheets: int = 3,
    instrument: str = "claude-cli",
) -> dict[int, SheetExecutionState]:
    """Register a job with N PENDING sheets."""
    sheets = {i: _make_sheet(i, instrument=instrument) for i in range(1, num_sheets + 1)}
    deps: dict[int, list[int]] = {i: [] for i in range(1, num_sheets + 1)}
    adapter._baton.register_job(job_id=job_id, sheets=sheets, dependencies=deps)
    return sheets


def _register_job_with_statuses(
    adapter: BatonAdapter,
    job_id: str,
    sheet_statuses: dict[int, BatonSheetStatus],
    instrument: str = "claude-cli",
) -> dict[int, SheetExecutionState]:
    """Register a job with sheets in specific statuses."""
    sheets = {
        num: _make_sheet(num, instrument=instrument, status=status)
        for num, status in sheet_statuses.items()
    }
    deps: dict[int, list[int]] = {i: [] for i in sheet_statuses}
    adapter._baton.register_job(job_id=job_id, sheets=sheets, dependencies=deps)
    return sheets


async def _handle(adapter: BatonAdapter, event: BatonEvent) -> None:
    """Handle an event through the baton, then run sync.

    Mimics the main loop: pre-capture → handle → sync.
    """
    pre_capture = adapter._capture_pre_event_state(event)
    await adapter._baton.handle_event(event)
    adapter._sync_sheet_status(event, pre_capture=pre_capture)


# =============================================================================
# Existing behavior (must not regress)
# =============================================================================


class TestExistingBehavior:
    """Verify existing sync for SheetAttemptResult/SheetSkipped still works."""

    async def test_sheet_attempt_result_syncs(self) -> None:
        cb = MagicMock()
        adapter = _make_adapter(cb)
        _register_job(adapter)

        await _handle(adapter, SheetAttemptResult(
            job_id="test-job", sheet_num=1, instrument_name="claude-cli",
            attempt=1, execution_success=True, validations_passed=1,
            validations_total=1, validation_pass_rate=100.0,
        ))

        cb.assert_any_call("test-job", 1, "completed", ANY)

    async def test_sheet_skipped_syncs(self) -> None:
        cb = MagicMock()
        adapter = _make_adapter(cb)
        _register_job(adapter)

        await _handle(adapter, SheetSkipped(
            job_id="test-job", sheet_num=1, reason="skip_when",
        ))

        cb.assert_any_call("test-job", 1, "skipped", ANY)

    async def test_escalation_resolved_syncs(self) -> None:
        cb = MagicMock()
        adapter = _make_adapter(cb)
        _register_job(adapter)
        await _handle(adapter, EscalationNeeded(
            job_id="test-job", sheet_num=1, reason="test",
        ))
        cb.reset_mock()

        await _handle(adapter, EscalationResolved(
            job_id="test-job", sheet_num=1, decision="retry",
        ))

        cb.assert_any_call("test-job", 1, "pending", ANY)

    async def test_escalation_timeout_syncs(self) -> None:
        cb = MagicMock()
        adapter = _make_adapter(cb)
        _register_job(adapter)
        await _handle(adapter, EscalationNeeded(
            job_id="test-job", sheet_num=1, reason="test",
        ))
        cb.reset_mock()

        await _handle(adapter, EscalationTimeout(
            job_id="test-job", sheet_num=1,
        ))

        cb.assert_any_call("test-job", 1, "failed", ANY)

    async def test_cancel_job_syncs_all(self) -> None:
        cb = MagicMock()
        adapter = _make_adapter(cb)
        _register_job(adapter, num_sheets=3)
        cb.reset_mock()

        await _handle(adapter, CancelJob(job_id="test-job"))

        for sn in [1, 2, 3]:
            cb.assert_any_call("test-job", sn, "cancelled", ANY)

    async def test_shutdown_non_graceful_syncs(self) -> None:
        cb = MagicMock()
        adapter = _make_adapter(cb)
        _register_job(adapter, job_id="j1", num_sheets=2)
        _register_job(adapter, job_id="j2", num_sheets=2)
        cb.reset_mock()

        await _handle(adapter, ShutdownRequested(graceful=False))

        for jid in ["j1", "j2"]:
            for sn in [1, 2]:
                cb.assert_any_call(jid, sn, "cancelled", ANY)


# =============================================================================
# NEW: Events that were NOT synced before — checkpoint-visible changes
# =============================================================================


class TestF211EscalationNeededSync:
    """EscalationNeeded: PENDING("pending") → FERMATA("in_progress").

    This is a checkpoint-visible change. Without sync, on restart the
    checkpoint shows "pending" while the baton intended FERMATA, causing
    the sheet to be re-dispatched instead of awaiting escalation.
    """

    async def test_escalation_needed_syncs_fermata(self) -> None:
        cb = MagicMock()
        adapter = _make_adapter(cb)
        _register_job(adapter)
        cb.reset_mock()

        await _handle(adapter, EscalationNeeded(
            job_id="test-job", sheet_num=1, reason="test",
        ))

        # FERMATA → "fermata" is a checkpoint-visible change from "pending"
        cb.assert_any_call("test-job", 1, "fermata", ANY)


class TestF211JobTimeoutSync:
    """JobTimeout: all PENDING("pending") → CANCELLED("cancelled").

    This is a checkpoint-visible change. Without sync, on restart
    timed-out sheets would be re-dispatched.
    """

    async def test_job_timeout_syncs_all(self) -> None:
        cb = MagicMock()
        adapter = _make_adapter(cb)
        _register_job(adapter, num_sheets=3)
        cb.reset_mock()

        await _handle(adapter, JobTimeout(job_id="test-job"))

        for sn in [1, 2, 3]:
            cb.assert_any_call("test-job", sn, "cancelled", ANY)

    async def test_job_timeout_preserves_completed(self) -> None:
        """Completed sheets keep their status after job timeout."""
        cb = MagicMock()
        adapter = _make_adapter(cb)
        _register_job(adapter, num_sheets=3)

        # Complete sheet 1
        await _handle(adapter, SheetAttemptResult(
            job_id="test-job", sheet_num=1, instrument_name="claude-cli",
            attempt=1, execution_success=True, validations_passed=1,
            validations_total=1, validation_pass_rate=100.0,
        ))
        cb.reset_mock()

        await _handle(adapter, JobTimeout(job_id="test-job"))

        # Only sheets 2 and 3 should get the new "cancelled" sync
        cb.assert_any_call("test-job", 2, "cancelled", ANY)
        cb.assert_any_call("test-job", 3, "cancelled", ANY)
        # Sheet 1 shouldn't re-sync (already "completed", unchanged)
        sheet1_calls = [c for c in cb.call_args_list if c[0][1] == 1]
        assert len(sheet1_calls) == 0


class TestF211RateLimitSync:
    """RateLimitExpired: WAITING("in_progress") → PENDING("pending").

    RateLimitHit changes DISPATCHED→WAITING (both "in_progress" in checkpoint,
    so no visible change). But RateLimitExpired changes WAITING→PENDING, which
    IS a visible checkpoint change.
    """

    async def test_rate_limit_expired_syncs_pending(self) -> None:
        """After rate limit clears, WAITING sheets sync back to pending."""
        cb = MagicMock()
        adapter = _make_adapter(cb)
        # Register with sheets already WAITING (simulates post-rate-limit state)
        _register_job_with_statuses(adapter, "test-job", {
            1: BatonSheetStatus.WAITING,
            2: BatonSheetStatus.WAITING,
        }, instrument="claude-cli")
        # Seed the sync cache: sheets are currently WAITING="waiting"
        adapter._synced_status[("test-job", 1)] = "waiting"
        adapter._synced_status[("test-job", 2)] = "waiting"

        # Also need to mark the instrument as rate-limited so expired clears it
        adapter._baton._instruments["claude-cli"].rate_limited = True

        await _handle(adapter, RateLimitExpired(instrument="claude-cli"))

        # WAITING→PENDING is "in_progress"→"pending" — visible change
        cb.assert_any_call("test-job", 1, "pending", ANY)
        cb.assert_any_call("test-job", 2, "pending", ANY)


class TestF211RetryDueSync:
    """RetryDue: RETRY_SCHEDULED("retry_scheduled") → PENDING("pending").

    With 1:1 mapping, these are different checkpoint statuses, so the
    transition IS a checkpoint-visible change. The sync fires for the
    PENDING status after RetryDue.
    """

    async def test_retry_due_syncs_pending(self) -> None:
        """RETRY_SCHEDULED → PENDING: visible change, sync fires."""
        cb = MagicMock()
        adapter = _make_adapter(cb)
        # Register with sheet 1 in RETRY_SCHEDULED
        _register_job_with_statuses(adapter, "test-job", {
            1: BatonSheetStatus.RETRY_SCHEDULED,
        })
        # Seed sync cache with current status
        adapter._synced_status[("test-job", 1)] = "retry_scheduled"

        await _handle(adapter, RetryDue(job_id="test-job", sheet_num=1))

        # RETRY_SCHEDULED→PENDING is "retry_scheduled"→"pending" — visible change
        cb.assert_any_call("test-job", 1, "pending", ANY)


# =============================================================================
# State-diff behavior
# =============================================================================


class TestStateDiffIdempotent:
    """State-diff approach only syncs CHANGED checkpoint statuses."""

    async def test_no_duplicate_sync(self) -> None:
        """Unchanged sheets don't get re-synced."""
        cb = MagicMock()
        adapter = _make_adapter(cb)
        _register_job(adapter)

        # Complete sheet 1
        await _handle(adapter, SheetAttemptResult(
            job_id="test-job", sheet_num=1, instrument_name="claude-cli",
            attempt=1, execution_success=True, validations_passed=1,
            validations_total=1, validation_pass_rate=100.0,
        ))
        cb.reset_mock()

        # Complete sheet 2 — should NOT re-sync sheet 1
        await _handle(adapter, SheetAttemptResult(
            job_id="test-job", sheet_num=2, instrument_name="claude-cli",
            attempt=1, execution_success=True, validations_passed=1,
            validations_total=1, validation_pass_rate=100.0,
        ))

        sheet1_calls = [c for c in cb.call_args_list if c[0][1] == 1]
        assert len(sheet1_calls) == 0

    async def test_escalation_flow_syncs_each_transition(self) -> None:
        """Full escalation flow: each visible transition syncs once."""
        cb = MagicMock()
        adapter = _make_adapter(cb)
        _register_job(adapter, num_sheets=1)

        # PENDING → FERMATA ("pending" → "fermata")
        await _handle(adapter, EscalationNeeded(
            job_id="test-job", sheet_num=1, reason="test",
        ))
        cb.assert_any_call("test-job", 1, "fermata", ANY)
        cb.reset_mock()

        # FERMATA → FAILED ("fermata" → "failed") via timeout
        await _handle(adapter, EscalationTimeout(
            job_id="test-job", sheet_num=1,
        ))
        cb.assert_any_call("test-job", 1, "failed", ANY)
        cb.reset_mock()

        # No more changes — sync should be noop
        await _handle(adapter, EscalationTimeout(
            job_id="test-job", sheet_num=1,
        ))
        sheet1_calls = [c for c in cb.call_args_list if c[0][1] == 1]
        assert len(sheet1_calls) == 0


class TestStateDiffCancelPreserves:
    """CancelJob preserves completed sheets and syncs the rest."""

    async def test_cancel_with_mixed_states(self) -> None:
        cb = MagicMock()
        adapter = _make_adapter(cb)
        _register_job(adapter, num_sheets=3)

        # Complete sheet 1
        await _handle(adapter, SheetAttemptResult(
            job_id="test-job", sheet_num=1, instrument_name="claude-cli",
            attempt=1, execution_success=True, validations_passed=1,
            validations_total=1, validation_pass_rate=100.0,
        ))
        cb.reset_mock()

        await _handle(adapter, CancelJob(job_id="test-job"))

        # Only non-terminal sheets synced (CANCELLED→"cancelled")
        cb.assert_any_call("test-job", 2, "cancelled", ANY)
        cb.assert_any_call("test-job", 3, "cancelled", ANY)
        sheet1_calls = [c for c in cb.call_args_list if c[0][1] == 1]
        assert len(sheet1_calls) == 0


class TestStateDiffNoCallback:
    """No callback set — sync is a safe noop."""

    async def test_no_callback_no_error(self) -> None:
        adapter = BatonAdapter()
        _register_job(adapter)
        await _handle(adapter, CancelJob(job_id="test-job"))


class TestStateDiffCallbackFailure:
    """Callback failures are logged, not raised."""

    async def test_callback_exception_handled(self) -> None:
        cb = MagicMock(side_effect=RuntimeError("DB write failed"))
        adapter = _make_adapter(cb)
        _register_job(adapter)
        await _handle(adapter, CancelJob(job_id="test-job"))
