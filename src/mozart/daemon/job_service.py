"""Core job execution service — decoupled from CLI.

Extracted from CLI commands to enable both CLI and daemon usage.
The CLI becomes a thin wrapper around this service.

The service encapsulates the run/resume/pause/status lifecycle
without any dependency on Rich, Typer, or CLI-level globals.
All user-facing output goes through OutputProtocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from mozart.core.checkpoint import CheckpointState, JobStatus
from mozart.core.logging import get_logger
from mozart.daemon.exceptions import JobSubmissionError
from mozart.daemon.output import NullOutput, OutputProtocol

if TYPE_CHECKING:
    from mozart.backends.base import Backend
    from mozart.core.config import JobConfig
    from mozart.execution.grounding import GroundingEngine
    from mozart.execution.runner import JobRunner
    from mozart.execution.runner.models import RunSummary
    from mozart.learning.global_store import GlobalLearningStore
    from mozart.learning.outcomes import OutcomeStore
    from mozart.notifications.base import NotificationManager
    from mozart.state.base import StateBackend

_logger = get_logger("daemon.job_service")


class _JobComponents(TypedDict):
    """Typed container for execution components created by _setup_components."""

    backend: Backend
    outcome_store: OutcomeStore | None
    global_learning_store: GlobalLearningStore | None
    notification_manager: NotificationManager | None
    escalation_handler: None
    grounding_engine: GroundingEngine | None


class JobService:
    """Core job execution service.

    Encapsulates the logic from CLI run/resume/pause commands into
    a reusable service that can be called from CLI, daemon, dashboard,
    or MCP server.

    All user-facing output goes through the OutputProtocol abstraction,
    allowing different frontends (Rich console, structlog, SSE, null)
    to receive execution events without code changes.
    """

    def __init__(
        self,
        *,
        output: OutputProtocol | None = None,
        global_learning_store: GlobalLearningStore | None = None,
    ) -> None:
        self._output = output or NullOutput()
        self._learning_store = global_learning_store

    # ─── Job Lifecycle ───────────────────────────────────────────────────

    async def start_job(
        self,
        config: JobConfig,
        *,
        fresh: bool = False,
        self_healing: bool = False,
        self_healing_auto_confirm: bool = False,
        dry_run: bool = False,
    ) -> RunSummary:
        """Start a job from config.

        Mirrors the logic in cli/commands/run.py::_run_job():
        1. Ensure workspace exists
        2. Handle --fresh (delete state, archive workspace)
        3. Setup backends and components via _setup_components()
        4. Create RunnerContext and JobRunner
        5. Call runner.run()

        Args:
            config: Validated job configuration.
            fresh: Delete existing state before running.
            self_healing: Enable automatic diagnosis and remediation.
            self_healing_auto_confirm: Auto-confirm suggested fixes.
            dry_run: If True, validate and return without executing.

        Returns:
            RunSummary with execution statistics.

        Raises:
            JobSubmissionError: If config is invalid or workspace setup fails.
            DaemonError: For unexpected daemon-level failures.
        """
        job_id = config.name

        self._output.job_event(job_id, "starting", {
            "total_sheets": config.sheet.total_sheets,
            "fresh": fresh,
        })

        # Ensure workspace exists
        config.workspace.mkdir(parents=True, exist_ok=True)

        # Create state backend
        state_backend = self._create_state_backend(config.workspace, config.state_backend)

        # Handle --fresh: delete existing state
        if fresh:
            was_deleted = await state_backend.delete(config.name)
            if was_deleted:
                self._output.log("info", "Deleted existing state for fresh start",
                                 job_id=job_id)

            # Archive workspace files if configured
            if config.workspace_lifecycle.archive_on_fresh:
                from mozart.workspace.lifecycle import WorkspaceArchiver

                archiver = WorkspaceArchiver(config.workspace, config.workspace_lifecycle)
                archive_path = archiver.archive()
                if archive_path:
                    self._output.log("info", "Archived workspace",
                                     job_id=job_id, archive=archive_path.name)

        if dry_run:
            # Return a minimal summary for dry runs
            from mozart.execution.runner.models import RunSummary

            return RunSummary(
                job_id=job_id,
                job_name=config.name,
                total_sheets=config.sheet.total_sheets,
                final_status=JobStatus.PENDING,
            )

        # Setup all execution components (backend, learning, notifications, etc.)
        components = self._setup_components(config)

        runner = self._create_runner(
            config, components, state_backend,
            job_id=job_id,
            self_healing=self_healing,
            self_healing_auto_confirm=self_healing_auto_confirm,
        )

        return await self._execute_runner(
            runner=runner,
            job_id=job_id,
            job_name=config.name,
            total_sheets=config.sheet.total_sheets,
            notification_manager=components["notification_manager"],
        )

    async def resume_job(
        self,
        job_id: str,
        workspace: Path,
        *,
        config: JobConfig | None = None,
        reload_config: bool = False,
        config_path: Path | None = None,
        self_healing: bool = False,
        self_healing_auto_confirm: bool = False,
    ) -> RunSummary:
        """Resume a paused or failed job.

        Mirrors the logic in cli/commands/resume.py::_resume_job():
        1. Find and validate job state
        2. Reconstruct config (from snapshot, provided config, or reload)
        3. Calculate resume point
        4. Setup backends and components
        5. Create runner and call runner.run(start_sheet=...)

        Args:
            job_id: Job identifier to resume.
            workspace: Workspace directory containing job state.
            config: Optional explicit JobConfig (overrides snapshot).
            reload_config: Reload config from original stored path.
            config_path: Path to config file for reload.
            self_healing: Enable automatic diagnosis and remediation.
            self_healing_auto_confirm: Auto-confirm suggested fixes.

        Returns:
            RunSummary with execution statistics.

        Raises:
            JobSubmissionError: If job state not found or not resumable.
            DaemonError: For unexpected failures.
        """
        # Phase 1: Find job state
        found_state, found_backend = await self._find_job_state(job_id, workspace)

        # Validate resumable state
        resumable_statuses = {JobStatus.PAUSED, JobStatus.FAILED, JobStatus.RUNNING}
        if found_state.status not in resumable_statuses:
            if found_state.status == JobStatus.COMPLETED:
                raise JobSubmissionError(
                    f"Job '{job_id}' is already completed. "
                    "Cannot resume a completed job without force flag."
                )
            elif found_state.status == JobStatus.PENDING:
                raise JobSubmissionError(
                    f"Job '{job_id}' has not been started yet. Use start_job() instead."
                )

        # Phase 2: Reconstruct config
        resolved_config = self._reconstruct_config(
            found_state, config=config, reload_config=reload_config,
            config_path=config_path,
        )

        # Update config snapshot if new config was provided
        if config is not None or reload_config:
            found_state.config_snapshot = resolved_config.model_dump(mode="json")

        # Calculate resume point
        resume_sheet = found_state.last_completed_sheet + 1
        if resume_sheet > found_state.total_sheets:
            resume_sheet = found_state.total_sheets

        self._output.job_event(job_id, "resuming", {
            "resume_sheet": resume_sheet,
            "total_sheets": found_state.total_sheets,
            "previous_status": found_state.status.value,
        })

        # Reset job status to RUNNING
        found_state.status = JobStatus.RUNNING
        found_state.error_message = None
        await found_backend.save(found_state)

        # Phase 3: Setup components and run
        components = self._setup_components(resolved_config)

        runner = self._create_runner(
            resolved_config, components, found_backend,
            job_id=job_id,
            self_healing=self_healing,
            self_healing_auto_confirm=self_healing_auto_confirm,
        )

        remaining = found_state.total_sheets - found_state.last_completed_sheet
        stored_config_path = (
            str(config_path) if config_path else found_state.config_path
        )

        return await self._execute_runner(
            runner=runner,
            job_id=job_id,
            job_name=resolved_config.name,
            total_sheets=resolved_config.sheet.total_sheets,
            notification_manager=components["notification_manager"],
            notify_total_sheets=remaining,
            start_sheet=resume_sheet,
            config_path=stored_config_path,
        )

    async def pause_job(self, job_id: str, workspace: Path) -> bool:
        """Pause a running job via signal file.

        Mirrors the logic in cli/commands/pause.py::_pause_job():
        Creates a pause signal file that the runner polls at sheet boundaries.

        Args:
            job_id: Job identifier to pause.
            workspace: Workspace directory containing job state.

        Returns:
            True if pause signal was created successfully.

        Raises:
            JobSubmissionError: If job not found or not in a pausable state.
        """
        found_state, _ = await self._find_job_state(job_id, workspace)

        if found_state.status != JobStatus.RUNNING:
            raise JobSubmissionError(
                f"Job '{job_id}' is {found_state.status.value}, not running. "
                "Only running jobs can be paused."
            )

        # Create pause signal file
        signal_file = workspace / f".mozart-pause-{job_id}"
        signal_file.touch()

        self._output.job_event(job_id, "pause_signal_sent", {
            "signal_file": str(signal_file),
        })

        return True

    async def get_status(
        self,
        job_id: str,
        workspace: Path,
    ) -> CheckpointState | None:
        """Get job status from state backend.

        Args:
            job_id: Job identifier.
            workspace: Workspace directory containing job state.

        Returns:
            CheckpointState if found, None if job doesn't exist.
        """
        state_backend = self._create_state_backend(workspace)
        return await state_backend.load(job_id)

    # ─── Internal Helpers ────────────────────────────────────────────────

    def _create_runner(
        self,
        config: JobConfig,
        components: _JobComponents,
        state_backend: StateBackend,
        *,
        job_id: str,
        self_healing: bool = False,
        self_healing_auto_confirm: bool = False,
    ) -> JobRunner:
        """Create a configured JobRunner from components."""
        from mozart.execution.runner import JobRunner as JR
        from mozart.execution.runner import RunnerContext

        def _progress_callback(completed: int, total: int, eta_seconds: float | None) -> None:
            self._output.progress(job_id, completed, total, eta_seconds)

        runner_context = RunnerContext(
            outcome_store=components["outcome_store"],
            escalation_handler=None,
            progress_callback=_progress_callback,
            global_learning_store=components["global_learning_store"] or self._learning_store,
            grounding_engine=components["grounding_engine"],
            self_healing_enabled=self_healing,
            self_healing_auto_confirm=self_healing_auto_confirm,
        )

        return JR(
            config=config,
            backend=components["backend"],
            state_backend=state_backend,
            context=runner_context,
        )

    @staticmethod
    def _get_or_create_summary(
        runner: JobRunner,
        job_id: str,
        job_name: str,
        total_sheets: int,
        status: JobStatus,
    ) -> RunSummary:
        """Get summary from runner or create a minimal fallback."""
        from mozart.execution.runner.models import RunSummary

        summary = runner.get_summary()
        if summary:
            summary.final_status = status
            return summary
        return RunSummary(
            job_id=job_id,
            job_name=job_name,
            total_sheets=total_sheets,
            final_status=status,
        )

    async def _execute_runner(
        self,
        runner: JobRunner,
        job_id: str,
        job_name: str,
        total_sheets: int,
        notification_manager: NotificationManager | None,
        notify_total_sheets: int | None = None,
        start_sheet: int | None = None,
        config_path: str | None = None,
    ) -> RunSummary:
        """Execute a runner with unified error handling.

        Handles GracefulShutdownError (→ PAUSED), FatalError (→ FAILED),
        and notification lifecycle (start, complete/fail, close).
        """
        from mozart.execution.runner import FatalError, GracefulShutdownError

        try:
            if notification_manager:
                await notification_manager.notify_job_start(
                    job_id=job_id,
                    job_name=job_name,
                    total_sheets=notify_total_sheets or total_sheets,
                )

            run_kwargs: dict[str, Any] = {}
            if start_sheet is not None:
                run_kwargs["start_sheet"] = start_sheet
            if config_path is not None:
                run_kwargs["config_path"] = config_path

            state, summary = await runner.run(**run_kwargs)

            if notification_manager:
                if state.status == JobStatus.COMPLETED:
                    await notification_manager.notify_job_complete(
                        job_id=job_id,
                        job_name=job_name,
                        success_count=summary.completed_sheets,
                        failure_count=summary.failed_sheets,
                        duration_seconds=summary.total_duration_seconds,
                    )
                elif state.status == JobStatus.FAILED:
                    await notification_manager.notify_job_failed(
                        job_id=job_id,
                        job_name=job_name,
                        error_message=f"Job failed with status: {state.status.value}",
                        sheet_num=state.current_sheet,
                    )

            self._output.job_event(job_id, "completed", {
                "status": state.status.value,
                "completed_sheets": summary.completed_sheets,
            })

            return summary

        except GracefulShutdownError:
            self._output.job_event(job_id, "paused")
            return self._get_or_create_summary(
                runner, job_id, job_name, total_sheets, JobStatus.PAUSED,
            )

        except FatalError as e:
            self._output.log("error", f"Fatal error: {e}", job_id=job_id)

            if notification_manager:
                try:
                    await notification_manager.notify_job_failed(
                        job_id=job_id,
                        job_name=job_name,
                        error_message=str(e),
                    )
                except Exception:
                    _logger.warning("notification_failed_during_error_handling", exc_info=True)

            return self._get_or_create_summary(
                runner, job_id, job_name, total_sheets, JobStatus.FAILED,
            )

        finally:
            if notification_manager:
                try:
                    await notification_manager.close()
                except Exception:
                    _logger.warning("notification_cleanup_failed", exc_info=True)

    def _create_backend(self, config: JobConfig) -> Backend:
        """Create execution backend from config.

        Extracted from cli/commands/_shared.py::create_backend().
        Supports: claude_cli, anthropic_api, recursive_light.
        """
        from mozart.backends.anthropic_api import AnthropicApiBackend
        from mozart.backends.claude_cli import ClaudeCliBackend
        from mozart.backends.recursive_light import RecursiveLightBackend

        if config.backend.type == "recursive_light":
            return RecursiveLightBackend.from_config(config.backend)
        elif config.backend.type == "anthropic_api":
            return AnthropicApiBackend.from_config(config.backend)
        else:
            return ClaudeCliBackend.from_config(config.backend)

    def _create_state_backend(
        self,
        workspace: Path,
        backend_type: str = "json",
    ) -> StateBackend:
        """Create state persistence backend.

        Extracted from cli/helpers.py::create_state_backend_from_config().
        """
        from mozart.state import JsonStateBackend, SQLiteStateBackend

        if backend_type == "sqlite":
            return SQLiteStateBackend(workspace / ".mozart-state.db")
        else:
            return JsonStateBackend(workspace)

    def _setup_components(self, config: JobConfig) -> _JobComponents:
        """Setup all execution components for a job.

        Mirrors cli/commands/_shared.py::setup_all() but without
        Rich console dependencies or CLI-level verbosity checks.
        """
        backend = self._create_backend(config)

        # Learning setup (mirrors _shared.py::setup_learning)
        outcome_store = None
        global_learning_store = None
        if config.learning.enabled:
            from mozart.learning.outcomes import JsonOutcomeStore

            outcome_store_path = config.get_outcome_store_path()
            if config.learning.outcome_store_type == "json":
                outcome_store = JsonOutcomeStore(outcome_store_path)

            # Prefer the injected store (from daemon LearningHub) over
            # the module-level singleton.  This avoids opening a second
            # SQLite connection when the daemon already owns one.
            if self._learning_store is not None:
                global_learning_store = self._learning_store
            else:
                from mozart.learning.global_store import get_global_store

                global_learning_store = get_global_store()

        # Notification setup (mirrors _shared.py::setup_notifications)
        notification_manager = None
        if config.notifications:
            from mozart.notifications import NotificationManager
            from mozart.notifications.factory import create_notifiers_from_config

            notifiers = create_notifiers_from_config(config.notifications)
            if notifiers:
                notification_manager = NotificationManager(notifiers)

        # Grounding setup (mirrors _shared.py::setup_grounding)
        grounding_engine = None
        if config.grounding.enabled:
            from mozart.execution.grounding import GroundingEngine, create_hook_from_config

            grounding_engine = GroundingEngine(hooks=[], config=config.grounding)
            for hook_config in config.grounding.hooks:
                try:
                    hook = create_hook_from_config(hook_config)
                    grounding_engine.add_hook(hook)
                except ValueError as e:
                    _logger.warning("failed_to_create_hook", error=str(e))

        return {
            "backend": backend,
            "outcome_store": outcome_store,
            "global_learning_store": global_learning_store,
            "notification_manager": notification_manager,
            "escalation_handler": None,  # Interactive-only
            "grounding_engine": grounding_engine,
        }

    async def _find_job_state(
        self,
        job_id: str,
        workspace: Path,
    ) -> tuple[CheckpointState, StateBackend]:
        """Find and return job state from workspace.

        Mirrors cli/helpers.py::find_job_state() and require_job_state()
        but raises DaemonError instead of calling typer.Exit().

        Returns:
            Tuple of (CheckpointState, StateBackend).

        Raises:
            JobSubmissionError: If workspace doesn't exist or job not found.
        """
        from mozart.state import JsonStateBackend, SQLiteStateBackend

        if not workspace.exists():
            raise JobSubmissionError(f"Workspace not found: {workspace}")

        backends: list[StateBackend] = []

        # Check for SQLite first (preferred for concurrent access)
        sqlite_path = workspace / ".mozart-state.db"
        if sqlite_path.exists():
            backends.append(SQLiteStateBackend(sqlite_path))

        # JSON backend as fallback
        backends.append(JsonStateBackend(workspace))

        for backend in backends:
            try:
                state = await backend.load(job_id)
                if state is not None:
                    return state, backend
            except Exception as e:
                _logger.warning("error_querying_backend", job_id=job_id, error=str(e))
                continue

        raise JobSubmissionError(
            f"Job '{job_id}' not found in workspace: {workspace}"
        )

    def _reconstruct_config(
        self,
        state: CheckpointState,
        *,
        config: JobConfig | None = None,
        reload_config: bool = False,
        config_path: Path | None = None,
    ) -> JobConfig:
        """Reconstruct JobConfig for resume using priority fallback.

        Mirrors cli/commands/resume.py::_reconstruct_config() but raises
        exceptions instead of calling typer.Exit().

        Priority order:
        1. Provided config object (explicit override)
        2. reload_config from config_path or stored path
        3. Cached config_snapshot in state
        4. Stored config_path as last resort

        Returns:
            Reconstructed JobConfig.

        Raises:
            JobSubmissionError: If no config source is available.
        """
        from mozart.core.config import JobConfig as JC

        # Priority 1: Explicit config
        if config is not None:
            return config

        # Priority 2: Reload from file
        if reload_config:
            path = config_path or (Path(state.config_path) if state.config_path else None)
            if path and path.exists():
                try:
                    return JC.from_yaml(path)
                except Exception as e:
                    raise JobSubmissionError(f"Error reloading config from {path}: {e}") from e
            raise JobSubmissionError(
                "Cannot reload config: no valid config path available."
            )

        # Priority 3: Config snapshot from state
        if state.config_snapshot:
            try:
                return JC.model_validate(state.config_snapshot)
            except Exception as e:
                raise JobSubmissionError(
                    f"Error reconstructing config from snapshot: {e}"
                ) from e

        # Priority 4: Stored config_path
        if state.config_path:
            path = Path(state.config_path)
            if path.exists():
                try:
                    return JC.from_yaml(path)
                except Exception as e:
                    raise JobSubmissionError(
                        f"Error loading config from stored path {path}: {e}"
                    ) from e

        raise JobSubmissionError(
            "Cannot resume: no config available. "
            "Provide a config object or ensure state has a config_snapshot."
        )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "JobService",
]
