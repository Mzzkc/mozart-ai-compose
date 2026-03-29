"""Baton core — event inbox, main loop, and sheet registry.

The baton core is the event-driven execution heart of the conductor.
It processes events from five sources (musicians, timers, external
commands, observer, internal dispatch), maintains per-sheet execution
state, resolves ready sheets, and coordinates dispatch.

The main loop is simple by design::

    while not shutting_down:
        event = await inbox.get()
        handle(event)        # updates state
        dispatch_ready()     # dispatches sheets if state changed
        persist()            # save if dirty

Everything else — retry logic, rate limit management, cost enforcement —
is done inside event handlers, not in separate subsystems.

See: ``docs/plans/2026-03-26-baton-design.md`` for the full architecture.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mozart.daemon.baton.dispatch import DispatchConfig

from mozart.daemon.baton.events import (
    BatonEvent,
    CancelJob,
    ConfigReloaded,
    CronTick,
    DispatchRetry,
    EscalationNeeded,
    EscalationResolved,
    EscalationTimeout,
    JobTimeout,
    PacingComplete,
    PauseJob,
    ProcessExited,
    RateLimitExpired,
    RateLimitHit,
    ResourceAnomaly,
    ResumeJob,
    RetryDue,
    SheetAttemptResult,
    SheetSkipped,
    ShutdownRequested,
    StaleCheck,
)
from mozart.daemon.baton.state import (
    _DISPATCHABLE_BATON_STATUSES,
    _SATISFIED_BATON_STATUSES,
    _TERMINAL_BATON_STATUSES,
    BatonSheetStatus,
    CircuitBreakerState,
    InstrumentState,
    SheetExecutionState,
)

_logger = logging.getLogger(__name__)


@dataclass
class _JobRecord:
    """Internal tracking for a registered job."""

    job_id: str
    sheets: dict[int, SheetExecutionState]
    dependencies: dict[int, list[int]]
    paused: bool = False
    user_paused: bool = False  # Tracks user-initiated pause (PauseJob)
    created_at: float = field(default_factory=time.time)


class BatonCore:
    """The baton's event-driven execution core.

    Manages the event inbox, processes events, tracks sheet state
    across all jobs, resolves ready sheets, and coordinates dispatch.

    The baton does NOT own backend execution — it decides WHEN to
    dispatch, not HOW. Sheet execution is delegated to the musician
    (via dispatch callbacks registered by the conductor).

    Usage::

        baton = BatonCore()
        baton.register_job("j1", sheets, deps)

        # Run the main loop (blocks until shutdown)
        await baton.run()

        # Or process events manually (for testing)
        await baton.handle_event(some_event)
    """

    _DEFAULT_INSTRUMENT_CONCURRENCY: int = 4
    """Default max concurrent sheets per instrument when not specified."""

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[BatonEvent] = asyncio.Queue()
        self._jobs: dict[str, _JobRecord] = {}
        self._instruments: dict[str, InstrumentState] = {}
        self._job_cost_limits: dict[str, float] = {}
        self._shutting_down = False
        self._running = False
        self._state_dirty = False

    @property
    def inbox(self) -> asyncio.Queue[BatonEvent]:
        """The event inbox — put events here for the baton to process."""
        return self._inbox

    @property
    def is_running(self) -> bool:
        """Whether the main loop is currently running."""
        return self._running

    @property
    def job_count(self) -> int:
        """Number of registered jobs."""
        return len(self._jobs)

    @property
    def running_sheet_count(self) -> int:
        """Number of sheets currently in 'dispatched' status."""
        return sum(
            1
            for job in self._jobs.values()
            for sheet in job.sheets.values()
            if sheet.status == BatonSheetStatus.DISPATCHED
        )

    # =========================================================================
    # Instrument Registry
    # =========================================================================

    def register_instrument(
        self, name: str, *, max_concurrent: int = 4
    ) -> InstrumentState:
        """Register an instrument for tracking.

        If already registered, returns the existing state (idempotent).

        Args:
            name: Instrument name (matches InstrumentProfile.name).
            max_concurrent: Maximum concurrent sheets on this instrument.

        Returns:
            The InstrumentState for the instrument.
        """
        if name in self._instruments:
            return self._instruments[name]

        state = InstrumentState(name=name, max_concurrent=max_concurrent)
        self._instruments[name] = state
        _logger.debug(
            "baton.instrument_registered",
            extra={"instrument": name, "max_concurrent": max_concurrent},
        )
        return state

    def get_instrument_state(self, name: str) -> InstrumentState | None:
        """Get the tracking state for a specific instrument."""
        return self._instruments.get(name)

    def build_dispatch_config(
        self, *, max_concurrent_sheets: int = 10
    ) -> DispatchConfig:
        """Build a DispatchConfig from the current instrument state.

        This bridges the gap between the baton's instrument tracking
        and the dispatch logic's configuration needs. Called before
        each dispatch cycle.

        Args:
            max_concurrent_sheets: Global concurrency ceiling.

        Returns:
            DispatchConfig with rate-limited instruments, open circuit
            breakers, and per-instrument concurrency limits derived
            from the current InstrumentState.
        """
        # Deferred import to break circular dependency
        # (dispatch.py imports BatonCore at runtime)
        from mozart.daemon.baton.dispatch import DispatchConfig  # noqa: N814

        rate_limited: set[str] = set()
        open_breakers: set[str] = set()
        concurrency: dict[str, int] = {}

        for name, inst in self._instruments.items():
            if inst.rate_limited:
                rate_limited.add(name)
            if inst.circuit_breaker == CircuitBreakerState.OPEN:
                open_breakers.add(name)
            concurrency[name] = inst.max_concurrent

        return DispatchConfig(
            max_concurrent_sheets=max_concurrent_sheets,
            instrument_concurrency=concurrency,
            rate_limited_instruments=rate_limited,
            open_circuit_breakers=open_breakers,
        )

    def set_job_cost_limit(self, job_id: str, max_cost_usd: float) -> None:
        """Set a per-job cost limit. The baton pauses the job when exceeded.

        Args:
            job_id: The job to set the limit for.
            max_cost_usd: Maximum total cost in USD.
        """
        self._job_cost_limits[job_id] = max_cost_usd

    def get_rate_limited_instruments(self) -> set[str]:
        """Get the set of currently rate-limited instrument names.

        Used by dispatch logic to skip rate-limited instruments.
        """
        return {
            name for name, inst in self._instruments.items()
            if inst.rate_limited
        }

    def get_open_circuit_breakers(self) -> set[str]:
        """Get the set of instruments with open circuit breakers.

        Used by dispatch logic to skip unhealthy instruments.
        """
        return {
            name for name, inst in self._instruments.items()
            if inst.circuit_breaker == CircuitBreakerState.OPEN
        }

    def _auto_register_instruments(
        self, sheets: dict[int, SheetExecutionState]
    ) -> None:
        """Auto-register instruments for any sheets using untracked instruments."""
        for sheet in sheets.values():
            if sheet.instrument_name not in self._instruments:
                self.register_instrument(
                    sheet.instrument_name,
                    max_concurrent=self._DEFAULT_INSTRUMENT_CONCURRENCY,
                )

    def _update_instrument_on_success(self, instrument_name: str) -> None:
        """Record a successful execution on an instrument."""
        inst = self._instruments.get(instrument_name)
        if inst is not None:
            inst.record_success()

    def _update_instrument_on_failure(self, instrument_name: str) -> None:
        """Record a failed execution on an instrument."""
        inst = self._instruments.get(instrument_name)
        if inst is not None:
            inst.record_failure()

    def _check_job_cost_limit(self, job_id: str) -> None:
        """Check if a job has exceeded its cost limit and pause if so."""
        limit = self._job_cost_limits.get(job_id)
        if limit is None:
            return

        job = self._jobs.get(job_id)
        if job is None:
            return

        total_cost = sum(s.total_cost_usd for s in job.sheets.values())
        if total_cost > limit:
            job.paused = True
            self._state_dirty = True
            _logger.warning(
                "baton.job.cost_limit_exceeded",
                extra={
                    "job_id": job_id,
                    "total_cost": total_cost,
                    "limit": limit,
                },
            )

    # =========================================================================
    # Sheet Registry
    # =========================================================================

    def register_job(
        self,
        job_id: str,
        sheets: dict[int, SheetExecutionState],
        dependencies: dict[int, list[int]],
    ) -> None:
        """Register a job's sheets with the baton for scheduling.

        Args:
            job_id: Unique job identifier.
            sheets: Map of sheet_num → SheetExecutionState.
            dependencies: Map of sheet_num → list of dependency sheet_nums.
                Sheets not in this map have no dependencies.
        """
        if job_id in self._jobs:
            _logger.warning(
                "baton.register_job.duplicate",
                extra={"job_id": job_id},
            )
            return

        # Auto-register any instruments used by the job's sheets
        self._auto_register_instruments(sheets)

        self._jobs[job_id] = _JobRecord(
            job_id=job_id,
            sheets=sheets,
            dependencies=dependencies,
        )
        self._state_dirty = True

        _logger.info(
            "baton.job_registered",
            extra={
                "job_id": job_id,
                "sheet_count": len(sheets),
                "dependency_count": len(dependencies),
            },
        )

    def deregister_job(self, job_id: str) -> None:
        """Remove a job from the baton's tracking."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._state_dirty = True
            _logger.info("baton.job_deregistered", extra={"job_id": job_id})

    def get_sheet_state(
        self, job_id: str, sheet_num: int
    ) -> SheetExecutionState | None:
        """Get the scheduling state for a specific sheet."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return job.sheets.get(sheet_num)

    def is_job_paused(self, job_id: str) -> bool:
        """Check if a job's dispatch is paused."""
        job = self._jobs.get(job_id)
        return job.paused if job is not None else False

    def is_job_complete(self, job_id: str) -> bool:
        """Check if all sheets in a job are in terminal state."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        return all(
            sheet.status in _TERMINAL_BATON_STATUSES
            for sheet in job.sheets.values()
        )

    # =========================================================================
    # Ready Sheet Resolution
    # =========================================================================

    def get_ready_sheets(self, job_id: str) -> list[SheetExecutionState]:
        """Find sheets that are ready to dispatch.

        A sheet is ready when:
        1. Status is 'pending' or 'ready'
        2. All dependencies are satisfied (completed or skipped)
        3. The job is not paused
        """
        job = self._jobs.get(job_id)
        if job is None or job.paused:
            return []

        ready: list[SheetExecutionState] = []
        for sheet_num, sheet in job.sheets.items():
            if sheet.status not in _DISPATCHABLE_BATON_STATUSES:
                continue

            # Check dependencies
            deps = job.dependencies.get(sheet_num, [])
            deps_satisfied = all(
                self._is_dependency_satisfied(job, dep) for dep in deps
            )
            if deps_satisfied:
                ready.append(sheet)

        return ready

    def _is_dependency_satisfied(self, job: _JobRecord, dep_num: int) -> bool:
        """Check if a dependency sheet is in a satisfied state."""
        dep_sheet = job.sheets.get(dep_num)
        if dep_sheet is None:
            # Missing dependency — treat as satisfied (defensive)
            _logger.warning(
                "baton.missing_dependency",
                extra={"job_id": job.job_id, "dep_num": dep_num},
            )
            return True
        return dep_sheet.status in _SATISFIED_BATON_STATUSES

    # =========================================================================
    # Event Handling
    # =========================================================================

    async def handle_event(self, event: BatonEvent) -> None:
        """Process a single event. Updates state but does NOT dispatch.

        This is the core decision-making method. Each event type has
        a specific handler that updates sheet state.

        Per the baton spec: handler exceptions are logged, not re-raised.
        The baton continues processing subsequent events.
        """
        try:
            match event:
                # === Musician events ===
                case SheetAttemptResult():
                    self._handle_attempt_result(event)

                case SheetSkipped():
                    self._handle_sheet_skipped(event)

                # === Rate limit events ===
                case RateLimitHit():
                    self._handle_rate_limit_hit(event)

                case RateLimitExpired():
                    self._handle_rate_limit_expired(event)

                # === Timer events ===
                case RetryDue():
                    self._handle_retry_due(event)

                case StaleCheck():
                    pass  # TODO: implement stale detection

                case CronTick():
                    pass  # TODO: implement cron scheduling

                case JobTimeout():
                    self._handle_job_timeout(event)

                case PacingComplete():
                    pass  # TODO: implement pacing

                # === Escalation events ===
                case EscalationNeeded():
                    self._handle_escalation_needed(event)

                case EscalationResolved():
                    self._handle_escalation_resolved(event)

                case EscalationTimeout():
                    self._handle_escalation_timeout(event)

                # === External command events ===
                case PauseJob():
                    self._handle_pause_job(event)

                case ResumeJob():
                    self._handle_resume_job(event)

                case CancelJob():
                    self._handle_cancel_job(event)

                case ConfigReloaded():
                    pass  # TODO: implement config reload

                case ShutdownRequested():
                    self._handle_shutdown(event)

                # === Observer events ===
                case ProcessExited():
                    self._handle_process_exited(event)

                case ResourceAnomaly():
                    pass  # TODO: implement backpressure

                # === Internal events ===
                case DispatchRetry():
                    pass  # Dispatch retry — _dispatch_ready handles this

                case _:
                    _logger.warning(
                        "baton.unknown_event",
                        extra={"event_type": type(event).__name__},
                    )

        except Exception:
            _logger.error(
                "baton.event_handler_failed",
                extra={"event_type": type(event).__name__},
                exc_info=True,
            )

    # =========================================================================
    # Event Handlers — private
    # =========================================================================

    def _handle_attempt_result(self, event: SheetAttemptResult) -> None:
        """Process a musician's execution report."""
        job = self._jobs.get(event.job_id)
        if job is None:
            _logger.warning(
                "baton.attempt_result.unknown_job",
                extra={"job_id": event.job_id, "sheet_num": event.sheet_num},
            )
            return

        sheet = job.sheets.get(event.sheet_num)
        if sheet is None:
            _logger.warning(
                "baton.attempt_result.unknown_sheet",
                extra={"job_id": event.job_id, "sheet_num": event.sheet_num},
            )
            return

        # Terminal guard: once a sheet reaches a terminal state, no event
        # can change it. Late-arriving results (e.g., from cancelled tasks)
        # are safely ignored.
        if sheet.status in _TERMINAL_BATON_STATUSES:
            _logger.debug(
                "baton.attempt_result.terminal_noop",
                extra={
                    "job_id": event.job_id,
                    "sheet_num": event.sheet_num,
                    "status": sheet.status.value,
                },
            )
            return

        # Record the attempt (tracks cost, duration, and increments
        # normal_attempts for non-rate-limited results).
        sheet.record_attempt(event)

        if event.rate_limited:
            # Rate limit — NOT a failure. record_attempt() already
            # skipped normal_attempts increment for rate-limited results.
            sheet.status = BatonSheetStatus.WAITING
            self._state_dirty = True
            _logger.info(
                "baton.sheet.rate_limited",
                extra={
                    "job_id": event.job_id,
                    "sheet_num": event.sheet_num,
                    "instrument": event.instrument_name,
                },
            )
            return

        # F-018 guard: when execution succeeds with no validations,
        # treat as 100% pass rate regardless of the default value.
        # A musician that reports execution_success=True with
        # validations_total=0 should not trigger unnecessary retries.
        effective_pass_rate = event.validation_pass_rate
        if (
            event.execution_success
            and event.validations_total == 0
            and effective_pass_rate < 100.0
        ):
            effective_pass_rate = 100.0

        if event.execution_success and effective_pass_rate >= 100.0:
            # Perfect execution — mark complete
            sheet.status = BatonSheetStatus.COMPLETED
            self._update_instrument_on_success(event.instrument_name)
            self._state_dirty = True
            _logger.info(
                "baton.sheet.completed",
                extra={
                    "job_id": event.job_id,
                    "sheet_num": event.sheet_num,
                    "attempt": event.attempt,
                    "cost_usd": event.cost_usd,
                },
            )
            # Cost enforcement: even on success, check if job cost exceeded
            self._check_job_cost_limit(event.job_id)
            return

        if event.execution_success and effective_pass_rate > 0:
            # Partial validation pass — completion mode
            # The musician succeeded in execution but some validations failed.
            # Try re-dispatching with "finish your work" context.
            self._update_instrument_on_success(event.instrument_name)
            sheet.completion_attempts += 1
            if sheet.can_complete:
                sheet.status = BatonSheetStatus.PENDING
                self._state_dirty = True
                _logger.info(
                    "baton.sheet.completion_mode",
                    extra={
                        "job_id": event.job_id,
                        "sheet_num": event.sheet_num,
                        "pass_rate": effective_pass_rate,
                        "completion_attempt": sheet.completion_attempts,
                        "max_completion": sheet.max_completion,
                    },
                )
            else:
                # Completion budget exhausted
                sheet.status = BatonSheetStatus.FAILED
                self._state_dirty = True
                _logger.warning(
                    "baton.sheet.completion_exhausted",
                    extra={
                        "job_id": event.job_id,
                        "sheet_num": event.sheet_num,
                        "completion_attempts": sheet.completion_attempts,
                    },
                )
                self._propagate_failure_to_dependents(
                    event.job_id, event.sheet_num
                )
            self._check_job_cost_limit(event.job_id)
            return

        if not event.execution_success:
            # Execution failed — update instrument failure tracking
            self._update_instrument_on_failure(event.instrument_name)

            if event.error_classification == "AUTH_FAILURE":
                sheet.status = BatonSheetStatus.FAILED
                self._state_dirty = True
                _logger.error(
                    "baton.sheet.auth_failure",
                    extra={
                        "job_id": event.job_id,
                        "sheet_num": event.sheet_num,
                    },
                )
                self._propagate_failure_to_dependents(
                    event.job_id, event.sheet_num
                )
                self._check_job_cost_limit(event.job_id)
                return

        # Validation failure (pass_rate == 0) or execution failure
        # Check if retries exhausted (record_attempt already incremented)
        if not sheet.can_retry:
            sheet.status = BatonSheetStatus.FAILED
            self._state_dirty = True
            _logger.warning(
                "baton.sheet.retries_exhausted",
                extra={
                    "job_id": event.job_id,
                    "sheet_num": event.sheet_num,
                    "attempts": sheet.normal_attempts,
                },
            )
            self._propagate_failure_to_dependents(
                event.job_id, event.sheet_num
            )
        else:
            # Schedule retry
            sheet.status = BatonSheetStatus.RETRY_SCHEDULED
            self._state_dirty = True
            _logger.info(
                "baton.sheet.retry_scheduled",
                extra={
                    "job_id": event.job_id,
                    "sheet_num": event.sheet_num,
                    "attempt": sheet.normal_attempts,
                    "max_retries": sheet.max_retries,
                },
            )

        self._check_job_cost_limit(event.job_id)

    def _handle_sheet_skipped(self, event: SheetSkipped) -> None:
        """Mark a sheet as skipped.

        Terminal guard: completed, failed, skipped, or cancelled sheets
        cannot be re-marked as skipped by a late event.
        """
        job = self._jobs.get(event.job_id)
        if job is None:
            return
        sheet = job.sheets.get(event.sheet_num)
        if sheet is None:
            return
        if sheet.status in _TERMINAL_BATON_STATUSES:
            _logger.debug(
                "baton.sheet_skipped.terminal_noop",
                extra={
                    "job_id": event.job_id,
                    "sheet_num": event.sheet_num,
                    "status": sheet.status.value,
                },
            )
            return
        sheet.status = BatonSheetStatus.SKIPPED
        self._state_dirty = True
        _logger.info(
            "baton.sheet.skipped",
            extra={
                "job_id": event.job_id,
                "sheet_num": event.sheet_num,
                "reason": event.reason,
            },
        )

    def _handle_rate_limit_hit(self, event: RateLimitHit) -> None:
        """Mark instrument as rate-limited. NOT a sheet failure.

        Updates both the per-sheet status (to WAITING for dispatched/running
        sheets) and the per-instrument state (rate_limited=True). This ensures
        the dispatch logic knows to skip this instrument for all jobs.

        Only transition sheets that are currently dispatched or running.
        Pending/ready sheets haven't been sent to an instrument yet;
        terminal sheets must never regress.
        """
        # Update instrument-level state
        inst = self._instruments.get(event.instrument)
        if inst is not None:
            inst.rate_limited = True
            inst.rate_limit_expires_at = time.monotonic() + event.wait_seconds

        # Update sheet-level state — ALL dispatched/running sheets on this
        # instrument across ALL jobs move to waiting, not just the one that
        # triggered the rate limit. Rate limits are per-instrument, not per-sheet.
        for job in self._jobs.values():
            for sheet in job.sheets.values():
                if (
                    sheet.instrument_name == event.instrument
                    and sheet.status in (
                        BatonSheetStatus.DISPATCHED,
                        BatonSheetStatus.RUNNING,
                    )
                ):
                    sheet.status = BatonSheetStatus.WAITING
        self._state_dirty = True

    def _handle_rate_limit_expired(self, event: RateLimitExpired) -> None:
        """Rate limit cleared — unmark instrument and move waiting sheets back to pending.

        Clears the instrument-level rate limit flag so the dispatch logic
        will resume dispatching sheets to this instrument.
        """
        # Clear instrument-level rate limit
        inst = self._instruments.get(event.instrument)
        if inst is not None:
            inst.rate_limited = False
            inst.rate_limit_expires_at = None

        # Move waiting sheets back to pending
        for job in self._jobs.values():
            for sheet in job.sheets.values():
                if (
                    sheet.status == BatonSheetStatus.WAITING
                    and sheet.instrument_name == event.instrument
                ):
                    sheet.status = BatonSheetStatus.PENDING
        self._state_dirty = True

    def _handle_retry_due(self, event: RetryDue) -> None:
        """Timer fired — sheet is ready for retry."""
        job = self._jobs.get(event.job_id)
        if job is None:
            return
        sheet = job.sheets.get(event.sheet_num)
        if sheet is not None and sheet.status == BatonSheetStatus.RETRY_SCHEDULED:
            sheet.status = BatonSheetStatus.PENDING
            self._state_dirty = True

    def _handle_job_timeout(self, event: JobTimeout) -> None:
        """Job wall-clock timeout — cancel remaining sheets."""
        job = self._jobs.get(event.job_id)
        if job is None:
            return
        for sheet in job.sheets.values():
            if sheet.status not in _TERMINAL_BATON_STATUSES:
                sheet.status = BatonSheetStatus.CANCELLED
        self._state_dirty = True
        _logger.warning(
            "baton.job.timeout",
            extra={"job_id": event.job_id},
        )

    def _handle_escalation_needed(self, event: EscalationNeeded) -> None:
        """Enter fermata — pause job, await composer decision.

        Terminal sheets are not affected — a completed/failed/skipped sheet
        cannot enter fermata. The job is still paused to await the composer's
        decision about the escalation context.
        """
        job = self._jobs.get(event.job_id)
        if job is None:
            return
        sheet = job.sheets.get(event.sheet_num)
        if sheet is not None and sheet.status not in _TERMINAL_BATON_STATUSES:
            sheet.status = BatonSheetStatus.FERMATA
        job.paused = True
        self._state_dirty = True

    def _handle_escalation_resolved(self, event: EscalationResolved) -> None:
        """Composer made a decision on a fermata.

        Only unpauses the job if the user didn't also pause it.
        A user-initiated pause (PauseJob) is independent of escalation.
        """
        job = self._jobs.get(event.job_id)
        if job is None:
            return
        sheet = job.sheets.get(event.sheet_num)
        if sheet is not None and sheet.status == BatonSheetStatus.FERMATA:
            # Apply decision
            if event.decision == "retry":
                sheet.status = BatonSheetStatus.PENDING
            elif event.decision == "skip":
                sheet.status = BatonSheetStatus.SKIPPED
            elif event.decision == "accept":
                sheet.status = BatonSheetStatus.COMPLETED
            else:
                sheet.status = BatonSheetStatus.FAILED
                self._propagate_failure_to_dependents(
                    event.job_id, event.sheet_num
                )
        # Only unpause if the user hasn't independently paused the job
        if not job.user_paused:
            job.paused = False
        self._state_dirty = True

    def _handle_escalation_timeout(self, event: EscalationTimeout) -> None:
        """No escalation response — default to safe action (fail sheet).

        Only unpauses if the user didn't also pause the job.
        """
        job = self._jobs.get(event.job_id)
        if job is None:
            return
        sheet = job.sheets.get(event.sheet_num)
        if sheet is not None and sheet.status == BatonSheetStatus.FERMATA:
            sheet.status = BatonSheetStatus.FAILED
            self._propagate_failure_to_dependents(
                event.job_id, event.sheet_num
            )
        # Only unpause if the user hasn't independently paused the job
        if not job.user_paused:
            job.paused = False
        self._state_dirty = True

    def _handle_pause_job(self, event: PauseJob) -> None:
        """Pause dispatching for a job (user-initiated)."""
        job = self._jobs.get(event.job_id)
        if job is not None:
            job.paused = True
            job.user_paused = True
            self._state_dirty = True

    def _handle_resume_job(self, event: ResumeJob) -> None:
        """Resume dispatching for a paused job (user-initiated)."""
        job = self._jobs.get(event.job_id)
        if job is not None:
            job.paused = False
            job.user_paused = False
            self._state_dirty = True

    def _handle_cancel_job(self, event: CancelJob) -> None:
        """Cancel all sheets and deregister the job."""
        job = self._jobs.get(event.job_id)
        if job is not None:
            for sheet in job.sheets.values():
                if sheet.status not in _TERMINAL_BATON_STATUSES:
                    sheet.status = BatonSheetStatus.CANCELLED
            self.deregister_job(event.job_id)
            _logger.info(
                "baton.job.cancelled",
                extra={"job_id": event.job_id},
            )

    def _handle_shutdown(self, event: ShutdownRequested) -> None:
        """Begin shutdown."""
        self._shutting_down = True
        if not event.graceful:
            # Cancel all non-terminal sheets
            for job in self._jobs.values():
                for sheet in job.sheets.values():
                    if sheet.status not in _TERMINAL_BATON_STATUSES:
                        sheet.status = BatonSheetStatus.CANCELLED
        _logger.info(
            "baton.shutdown",
            extra={"graceful": event.graceful},
        )

    def _handle_process_exited(self, event: ProcessExited) -> None:
        """Backend process died — mark sheet as crashed if running."""
        job = self._jobs.get(event.job_id)
        if job is None:
            return
        sheet = job.sheets.get(event.sheet_num)
        if sheet is not None and sheet.status == BatonSheetStatus.DISPATCHED:
            sheet.normal_attempts += 1
            if not sheet.can_retry:
                sheet.status = BatonSheetStatus.FAILED
                self._propagate_failure_to_dependents(
                    event.job_id, event.sheet_num
                )
            else:
                sheet.status = BatonSheetStatus.RETRY_SCHEDULED
            self._state_dirty = True

    # =========================================================================
    # Dependency Failure Propagation
    # =========================================================================

    def _propagate_failure_to_dependents(
        self, job_id: str, failed_sheet_num: int
    ) -> None:
        """Mark all transitive dependents of a failed sheet as failed.

        When a sheet fails (retries exhausted, auth failure, etc.), its
        dependent sheets can never be satisfied — they'd sit in pending
        forever, creating zombie jobs. This method cascades the failure
        to all downstream sheets using iterative BFS.

        Only non-terminal sheets are affected. Completed, skipped, and
        already-failed sheets are left unchanged.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return

        # Build a reverse dependency map: sheet_num → list of sheets
        # that depend on it
        dependents: dict[int, list[int]] = {}
        for sheet_num, deps in job.dependencies.items():
            for dep in deps:
                if dep not in dependents:
                    dependents[dep] = []
                dependents[dep].append(sheet_num)

        # BFS from the failed sheet to all transitive dependents
        queue = list(dependents.get(failed_sheet_num, []))
        visited: set[int] = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            sheet = job.sheets.get(current)
            if sheet is None:
                continue

            # Only fail non-terminal sheets
            if sheet.status not in _TERMINAL_BATON_STATUSES:
                sheet.status = BatonSheetStatus.FAILED
                _logger.info(
                    "baton.sheet.dependency_failed",
                    extra={
                        "job_id": job_id,
                        "sheet_num": current,
                        "failed_dependency": failed_sheet_num,
                    },
                )

            # Continue propagation to this sheet's dependents
            for downstream in dependents.get(current, []):
                if downstream not in visited:
                    queue.append(downstream)

        if visited:
            self._state_dirty = True

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def run(self) -> None:
        """The baton's main event loop.

        Processes events from the inbox, updates state, and dispatches
        ready sheets. Runs until a ShutdownRequested event is received.
        """
        self._running = True
        _logger.info("baton.started")

        try:
            while not self._shutting_down:
                event = await self._inbox.get()
                await self.handle_event(event)
                # dispatch_ready() would be called here when wired to
                # conductor
        except asyncio.CancelledError:
            _logger.info("baton.cancelled")
            raise
        finally:
            self._running = False
            _logger.info("baton.stopped")

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_diagnostics(self, job_id: str) -> dict[str, Any] | None:
        """Get diagnostic information for a job.

        Returns a dict with sheet counts by status, instrument state,
        and other debugging information. Returns None if job not found.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None

        status_counts: dict[str, int] = {}
        instruments_used: set[str] = set()
        for sheet in job.sheets.values():
            status_key = sheet.status.value
            status_counts[status_key] = status_counts.get(status_key, 0) + 1
            instruments_used.add(sheet.instrument_name)

        return {
            "job_id": job_id,
            "paused": job.paused,
            "sheets": {
                "total": len(job.sheets),
                **status_counts,
            },
            "instruments_used": sorted(instruments_used),
        }
