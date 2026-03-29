"""Baton musician — single-attempt sheet execution.

The musician is the bridge between the baton's dispatch decision and the
backend's execution. It plays ONE attempt, runs validations, records
outcomes, and reports back via SheetAttemptResult into the baton's inbox.

The musician NEVER retries. It NEVER decides completion mode, escalation,
or rate limit recovery. It reports; the conductor (baton) decides.

The 6-step flow:
    1. Build prompt (render template + inject context)
    2. Configure backend (timeout, working directory)
    3. Play: ``backend.execute(prompt, timeout_seconds=sheet.timeout)``
    4. Listen: run validations on output (if execution succeeded)
    5. Record: capture output with credential redaction
    6. Report: put ``SheetAttemptResult`` into baton inbox

Design decisions:
    - The musician is a free function, not a class. No state to manage.
    - Exceptions from the backend are caught and reported, never re-raised.
      The baton must never crash because a backend threw.
    - Credential redaction happens before the result is put in the inbox.
      No unscanned output ever reaches the baton.
    - F-018 contract: when execution succeeds with no validations,
      ``validation_pass_rate`` is set to 100.0. The default 0.0 would
      cause unnecessary retries.

See: ``docs/plans/2026-03-26-baton-design.md`` — Single-Attempt Musician
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from mozart.backends.base import Backend, ExecutionResult
from mozart.core.constants import TRUNCATE_STDOUT_TAIL_CHARS
from mozart.core.sheet import Sheet
from mozart.daemon.baton.events import SheetAttemptResult
from mozart.daemon.baton.state import AttemptContext
from mozart.utils.credential_scanner import redact_credentials

_logger = logging.getLogger(__name__)


async def sheet_task(
    *,
    job_id: str,
    sheet: Sheet,
    backend: Backend,
    attempt_context: AttemptContext,
    inbox: asyncio.Queue[SheetAttemptResult],
) -> None:
    """Execute a single sheet attempt and report the result.

    This is the musician's entire job. Play once, report in full detail.
    The conductor (baton) decides what happens next.

    Args:
        job_id: The job this sheet belongs to.
        sheet: The sheet to execute (prompt, validations, timeout, etc.).
        backend: The backend to execute through.
        attempt_context: Context from the conductor (attempt number, mode, etc.).
        inbox: The baton's event inbox to report results to.

    Never raises — all exceptions are caught and reported via the inbox.
    """
    start_time = time.monotonic()

    try:
        # Step 1: Build prompt
        prompt = _build_prompt(sheet, attempt_context)

        # Step 2-3: Execute through backend
        exec_result = await _execute(backend, prompt, sheet.timeout_seconds)

        # Step 4: Run validations (only if execution succeeded)
        val_passed, val_total, val_rate, val_details = await _validate(
            sheet, exec_result
        )

        # Step 5: Record output with credential redaction
        stdout_tail, stderr_tail = _capture_output(exec_result)

        # Step 6: Classify errors
        error_class, error_msg = _classify_error(exec_result)

        # Build and report result
        duration = exec_result.duration_seconds
        result = SheetAttemptResult(
            job_id=job_id,
            sheet_num=sheet.num,
            instrument_name=sheet.instrument_name,
            attempt=attempt_context.attempt_number,
            execution_success=exec_result.success,
            exit_code=exec_result.exit_code,
            duration_seconds=duration,
            validations_passed=val_passed,
            validations_total=val_total,
            validation_pass_rate=val_rate,
            validation_details=val_details,
            error_classification=error_class,
            error_message=error_msg,
            rate_limited=exec_result.rate_limited,
            cost_usd=_estimate_cost(exec_result),
            input_tokens=exec_result.input_tokens or 0,
            output_tokens=exec_result.output_tokens or 0,
            model_used=exec_result.model,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )

    except Exception as exc:
        # Step 6 (exception path): Never crash the baton
        duration = time.monotonic() - start_time
        error_msg = f"{type(exc).__name__}: {exc}"
        _logger.error(
            "musician.sheet_task.exception",
            extra={
                "job_id": job_id,
                "sheet_num": sheet.num,
                "error": error_msg,
            },
            exc_info=True,
        )

        result = SheetAttemptResult(
            job_id=job_id,
            sheet_num=sheet.num,
            instrument_name=sheet.instrument_name,
            attempt=attempt_context.attempt_number,
            execution_success=False,
            exit_code=None,
            duration_seconds=duration,
            error_classification="TRANSIENT",
            error_message=error_msg,
            rate_limited=False,
        )

    # Always report — the baton must know what happened
    await inbox.put(result)
    _logger.info(
        "musician.sheet_task.reported",
        extra={
            "job_id": job_id,
            "sheet_num": sheet.num,
            "success": result.execution_success,
            "pass_rate": result.validation_pass_rate,
            "duration": result.duration_seconds,
        },
    )


# =========================================================================
# Internal helpers
# =========================================================================


def _build_prompt(sheet: Sheet, context: AttemptContext) -> str:
    """Build the execution prompt from the sheet template.

    For now, uses the raw prompt template. Full Jinja2 rendering with
    cross-sheet context will be added when the baton wires into the
    conductor (step 28) and has access to the rendering pipeline.

    For completion mode, appends the completion suffix to guide the
    musician toward fixing partial failures.
    """
    prompt = sheet.prompt_template or ""

    # Completion mode: append suffix to help fix partial failures
    if context.completion_prompt_suffix:
        prompt = f"{prompt}\n\n{context.completion_prompt_suffix}"

    return prompt


async def _execute(
    backend: Backend,
    prompt: str,
    timeout_seconds: float,
) -> ExecutionResult:
    """Execute the prompt through the backend.

    This is a thin wrapper that passes the timeout and returns the result.
    All exception handling is done by the caller (sheet_task).
    """
    return await backend.execute(prompt, timeout_seconds=timeout_seconds)


async def _validate(
    sheet: Sheet,
    exec_result: ExecutionResult,
) -> tuple[int, int, float, dict[str, Any] | None]:
    """Run validations on the execution output.

    Returns:
        (passed_count, total_count, pass_rate, details_dict)

    F-018 contract: when execution succeeds with no validations,
    pass_rate is 100.0. The default 0.0 would cause unnecessary retries.
    """
    # Don't run validations if execution failed
    if not exec_result.success:
        return 0, 0, 0.0, None

    # No validation rules → 100% pass rate (F-018 contract)
    if not sheet.validations:
        return 0, 0, 100.0, None

    # Run validations through the validation engine
    try:
        from mozart.execution.validation.engine import ValidationEngine

        engine = ValidationEngine(
            workspace=sheet.workspace,
            sheet_context={"sheet_num": sheet.num},
        )
        result = await engine.run_validations(sheet.validations)

        passed = result.passed_count
        total = len(result.results)
        rate = result.pass_percentage
        details = {
            "passed": passed,
            "failed": result.failed_count,
            "skipped": result.skipped_count,
            "pass_percentage": rate,
        }
        return passed, total, rate, details

    except Exception as exc:
        _logger.warning(
            "musician.validation.error",
            extra={
                "sheet_num": sheet.num,
                "error": str(exc),
            },
        )
        # Validation engine failure — report as 0% to trigger retry
        return 0, 0, 0.0, {"error": str(exc)}


def _capture_output(exec_result: ExecutionResult) -> tuple[str, str]:
    """Capture and sanitize output for the result.

    Truncates to TRUNCATE_STDOUT_TAIL_CHARS and redacts credentials.
    """
    stdout = exec_result.stdout or ""
    stderr = exec_result.stderr or ""

    # Truncate
    if len(stdout) > TRUNCATE_STDOUT_TAIL_CHARS:
        stdout = stdout[-TRUNCATE_STDOUT_TAIL_CHARS:]
    if len(stderr) > TRUNCATE_STDOUT_TAIL_CHARS:
        stderr = stderr[-TRUNCATE_STDOUT_TAIL_CHARS:]

    # Credential redaction
    stdout = redact_credentials(stdout) or ""
    stderr = redact_credentials(stderr) or ""

    return stdout, stderr


def _classify_error(
    exec_result: ExecutionResult,
) -> tuple[str | None, str | None]:
    """Classify execution errors for the baton's decision tree.

    Returns:
        (classification, message) — both None for successful executions.

    Classifications:
        AUTH_FAILURE — API key invalid, permission denied
        TRANSIENT — timeout, signal kill, exit_code=None
        EXECUTION_ERROR — other failures
    """
    if exec_result.success:
        return None, None

    if exec_result.rate_limited:
        return None, None  # Rate limits are NOT errors

    # exit_code=None → process killed by signal → TRANSIENT
    if exec_result.exit_code is None:
        return "TRANSIENT", exec_result.error_message or "Process killed by signal"

    # Check for auth failures in error output
    error_text = (exec_result.stderr or "").lower()
    auth_patterns = [
        "authentication",
        "unauthorized",
        "api key",
        "permission denied",
        "invalid_api_key",
        "401",
        "403",
    ]
    if any(pattern in error_text for pattern in auth_patterns):
        return "AUTH_FAILURE", exec_result.error_message or "Authentication failure"

    # Default: EXECUTION_ERROR
    return (
        "EXECUTION_ERROR",
        exec_result.error_message or f"Exit code {exec_result.exit_code}",
    )


def _estimate_cost(exec_result: ExecutionResult) -> float:
    """Estimate cost from token counts.

    Uses conservative estimates for common models. Real cost tracking
    will come from InstrumentProfile.ModelCapacity in a future iteration.
    """
    input_tokens = exec_result.input_tokens or 0
    output_tokens = exec_result.output_tokens or 0

    # Conservative estimate: $3/1M input, $15/1M output (Claude Sonnet pricing)
    # This is a placeholder — real pricing comes from InstrumentProfile
    cost = (input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)
    return cost
