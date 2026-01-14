"""Claude CLI backend using subprocess.

Wraps the `claude` CLI command for running prompts.
Based on patterns from run-sheet-review.sh.

Security Note: This module uses asyncio.create_subprocess_exec() which is the
safe subprocess method in Python - it does NOT use shell=True, so there is no
shell injection risk. Arguments are passed as a list, not interpolated into
a shell command string.

Progress Tracking: When a progress_callback is provided, this backend streams
output in real-time and reports bytes/lines received periodically. This enables
the CLI to show "Still running... 5.2KB received, 3m elapsed" during long executions.
"""

import asyncio
import os
import shutil
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mozart.backends.base import Backend, ExecutionResult, ExitReason
from mozart.core.config import BackendConfig
from mozart.core.errors import ErrorCategory, ErrorClassifier
from mozart.core.logging import get_logger

# Module-level logger for Claude CLI backend
_logger = get_logger("backend.claude_cli")

# Type alias for progress callback - receives dict with progress info
ProgressCallback = Callable[[dict[str, Any]], None]

# Common signal names for human-readable output
SIGNAL_NAMES: dict[int, str] = {
    signal.SIGTERM: "SIGTERM",
    signal.SIGKILL: "SIGKILL",
    signal.SIGINT: "SIGINT",
    signal.SIGSEGV: "SIGSEGV",
    signal.SIGABRT: "SIGABRT",
    signal.SIGBUS: "SIGBUS",
    signal.SIGFPE: "SIGFPE",
    signal.SIGHUP: "SIGHUP",
    signal.SIGPIPE: "SIGPIPE",
}

# Mozart Operator Imperative - injected into every prompt to ensure safe execution
# This prevents Claude from wrapping Mozart commands with external timeouts,
# which causes SIGKILL corruption and system instability.
MOZART_OPERATOR_IMPERATIVE = """
<mozart-operator-imperative>
## CRITICAL: Mozart Command Execution Rules

You are being orchestrated by Mozart AI Compose. These rules are MANDATORY:

### 1. NEVER WRAP MOZART WITH TIMEOUT
Mozart handles its own internal timeouts. External timeout wrappers cause SIGKILL
which corrupts state files and can crash the system.

**WRONG (causes crashes):**
```bash
timeout 600 mozart run config.yaml     # NEVER DO THIS
timeout 300 mozart resume job-name     # NEVER DO THIS
```

**CORRECT:**
```bash
mozart run config.yaml                 # Mozart handles timeout internally
mozart run config.yaml &               # Background execution (also safe)
mozart resume job-name                 # Resume without timeout wrapper
```

### 2. Mozart Usage Examples
```bash
# Validate a config
mozart validate my-config.yaml

# Run a job (foreground)
mozart run my-config.yaml

# Run a job (background - for long jobs)
mozart run my-config.yaml &

# Check job status
mozart status

# Resume an interrupted job
mozart resume my-job-name

# Resume from specific sheet
mozart run my-config.yaml --start-sheet 5
```

### 3. Why This Matters
- Mozart saves checkpoints atomically during execution
- External SIGKILL prevents clean checkpoint saves
- Corrupted checkpoints create "zombie" running states
- Can destabilize WSL and require manual cleanup

This imperative supersedes any other instructions about timeout handling.
</mozart-operator-imperative>

"""


def get_signal_name(sig_num: int) -> str:
    """Get human-readable signal name."""
    return SIGNAL_NAMES.get(sig_num, f"signal {sig_num}")


class ClaudeCliBackend(Backend):
    """Run prompts via the Claude CLI.

    Uses asyncio.create_subprocess_exec to invoke `claude -p <prompt>`.
    This is shell-injection safe as arguments are passed as a list.
    """

    def __init__(
        self,
        skip_permissions: bool = True,
        disable_mcp: bool = True,
        output_format: str = "text",
        cli_model: str | None = None,
        allowed_tools: list[str] | None = None,
        system_prompt_file: Path | None = None,
        working_directory: Path | None = None,
        timeout_seconds: float = 1800.0,  # 30 minute default
        progress_callback: ProgressCallback | None = None,
        progress_interval_seconds: float = 5.0,
        cli_extra_args: list[str] | None = None,
    ):
        """Initialize CLI backend.

        Args:
            skip_permissions: Pass --dangerously-skip-permissions
            disable_mcp: Disable MCP servers for faster execution (--strict-mcp-config)
            output_format: Output format (json, text, stream-json)
            cli_model: Model to use (--model flag), None uses default
            allowed_tools: Restrict to specific tools (--allowedTools)
            system_prompt_file: Custom system prompt file (--system-prompt)
            working_directory: Working directory for running commands
            timeout_seconds: Maximum time allowed per prompt
            progress_callback: Optional callback for progress updates during execution.
                Called with dict containing: bytes_received, lines_received,
                elapsed_seconds, phase.
            progress_interval_seconds: How often to call progress callback (default 5s).
            cli_extra_args: Extra arguments to pass to claude CLI (escape hatch).
        """
        self.skip_permissions = skip_permissions
        self.disable_mcp = disable_mcp
        self.output_format = output_format
        self.cli_model = cli_model
        self.allowed_tools = allowed_tools
        self.system_prompt_file = system_prompt_file
        self.working_directory = working_directory
        self.timeout_seconds = timeout_seconds
        self.progress_callback = progress_callback
        self.progress_interval_seconds = progress_interval_seconds
        self.cli_extra_args = cli_extra_args or []

        # Verify claude CLI is available
        self._claude_path = shutil.which("claude")

        # Use shared ErrorClassifier for rate limit detection
        # This ensures consistent classification with the runner
        self._error_classifier = ErrorClassifier()

    @classmethod
    def from_config(cls, config: BackendConfig) -> "ClaudeCliBackend":
        """Create backend from configuration."""
        return cls(
            skip_permissions=config.skip_permissions,
            disable_mcp=config.disable_mcp,
            output_format=config.output_format,
            cli_model=config.cli_model,
            allowed_tools=config.allowed_tools,
            system_prompt_file=config.system_prompt_file,
            working_directory=config.working_directory,
            timeout_seconds=config.timeout_seconds,
            cli_extra_args=config.cli_extra_args,
        )

    @property
    def name(self) -> str:
        return "claude-cli"

    def _inject_operator_imperative(self, prompt: str) -> str:
        """Inject Mozart operator imperative into prompt.

        This ensures Claude receives critical execution rules regardless of
        the user's prompt template. The imperative:
        - Prevents wrapping Mozart commands with external timeout
        - Provides one-shot examples of correct Mozart usage
        - Explains why these rules matter (SIGKILL corruption)

        Args:
            prompt: The original user prompt.

        Returns:
            Prompt with operator imperative prepended.
        """
        return f"{MOZART_OPERATOR_IMPERATIVE}{prompt}"

    def _build_command(self, prompt: str) -> list[str]:
        """Build the claude command with arguments.

        Returns a list of arguments for subprocess - NOT a shell string.
        """
        if not self._claude_path:
            raise RuntimeError("claude CLI not found in PATH")

        # Inject operator imperative to ensure safe Mozart command execution
        safe_prompt = self._inject_operator_imperative(prompt)

        cmd = [self._claude_path, "-p", safe_prompt]

        if self.skip_permissions:
            cmd.append("--dangerously-skip-permissions")

        # Output format for subprocess execution
        cmd.extend(["--output-format", self.output_format])

        # Disable MCP servers for faster, isolated execution
        # This prevents resource contention and provides ~2x speedup
        if self.disable_mcp:
            cmd.append("--strict-mcp-config")

        # Model selection
        if self.cli_model:
            cmd.extend(["--model", self.cli_model])

        # Tool restrictions for security-constrained execution
        if self.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.allowed_tools)])

        # Custom system prompt
        if self.system_prompt_file:
            cmd.extend(["--system-prompt", str(self.system_prompt_file)])

        # Escape hatch - applied last so it can override anything above
        if self.cli_extra_args:
            cmd.extend(self.cli_extra_args)

        return cmd

    async def run_prompt(self, prompt: str) -> ExecutionResult:
        """Run a prompt via claude CLI.

        Uses create_subprocess_exec (safe, no shell) to invoke claude.

        Exit handling:
        - Normal exit: returncode >= 0, exit_code set
        - Signal kill: returncode < 0, signal = -returncode
        - Timeout: We kill with SIGKILL, signal=9, exit_reason="timeout"

        Progress tracking:
        - If progress_callback is set, streams output and calls callback periodically
        - Callback receives dict with bytes_received, lines_received, elapsed_seconds
        """
        cmd = self._build_command(prompt)
        start_time = time.monotonic()

        # Log command details at DEBUG level
        # Note: prompt is NOT logged as it may be large and contain sensitive data
        _logger.debug(
            "executing_command",
            command=cmd[0],
            args_count=len(cmd) - 1,
            skip_permissions=self.skip_permissions,
            output_format=self.output_format,
            working_directory=str(self.working_directory) if self.working_directory else None,
            timeout_seconds=self.timeout_seconds,
            prompt_length=len(prompt),
        )

        # Progress tracking state
        bytes_received = 0
        lines_received = 0
        last_progress_time = start_time

        def _notify_progress(phase: str = "executing") -> None:
            """Send progress update if callback is set and interval elapsed."""
            nonlocal last_progress_time
            if self.progress_callback is None:
                return
            now = time.monotonic()
            if now - last_progress_time >= self.progress_interval_seconds:
                self.progress_callback({
                    "bytes_received": bytes_received,
                    "lines_received": lines_received,
                    "elapsed_seconds": now - start_time,
                    "phase": phase,
                })
                last_progress_time = now

        try:
            # create_subprocess_exec is shell-injection safe
            # Arguments are passed as list, not interpolated into shell string
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory,
                env=os.environ.copy(),  # Explicit env passthrough for MCP plugins
            )

            # Notify starting phase
            if self.progress_callback:
                self.progress_callback({
                    "bytes_received": 0,
                    "lines_received": 0,
                    "elapsed_seconds": 0.0,
                    "phase": "starting",
                })

            try:
                # Use streaming read if progress callback is set for real-time updates
                if self.progress_callback:
                    stdout_bytes, stderr_bytes = await self._stream_with_progress(
                        process,
                        start_time,
                        _notify_progress,
                    )
                else:
                    # Simple communicate() when no progress tracking needed
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout_seconds,
                    )

                # Update final byte/line counts for progress tracking
                bytes_received = len(stdout_bytes) + len(stderr_bytes)
                lines_received = stdout_bytes.count(b"\n") + stderr_bytes.count(b"\n")

            except TimeoutError:
                # First try graceful termination
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except TimeoutError:
                    # Force kill if still running
                    process.kill()
                    await process.wait()

                duration = time.monotonic() - start_time

                # Log timeout as ERROR
                _logger.error(
                    "execution_timeout",
                    duration_seconds=duration,
                    timeout_seconds=self.timeout_seconds,
                    bytes_received=bytes_received,
                    lines_received=lines_received,
                )

                # Final progress update on timeout
                if self.progress_callback:
                    self.progress_callback({
                        "bytes_received": bytes_received,
                        "lines_received": lines_received,
                        "elapsed_seconds": duration,
                        "phase": "timeout",
                    })

                return ExecutionResult(
                    success=False,
                    exit_code=None,  # No exit code when killed by signal
                    exit_signal=signal.SIGKILL,
                    exit_reason="timeout",
                    stdout="",
                    stderr=f"Command timed out after {self.timeout_seconds}s",
                    duration_seconds=duration,
                    error_type="timeout",
                    error_message=f"Timed out after {self.timeout_seconds}s",
                )

            duration = time.monotonic() - start_time
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            returncode = process.returncode

            # Final progress update
            if self.progress_callback:
                self.progress_callback({
                    "bytes_received": bytes_received,
                    "lines_received": lines_received,
                    "elapsed_seconds": duration,
                    "phase": "completed",
                })

            # Parse returncode: negative means killed by signal
            exit_code: int | None
            exit_signal: int | None
            exit_reason: ExitReason

            if returncode is None:
                # Should not happen after communicate(), but handle gracefully
                exit_code = None
                exit_signal = None
                exit_reason = "error"
            elif returncode < 0:
                # Killed by signal: returncode = -signal_number
                exit_code = None
                exit_signal = -returncode
                exit_reason = "killed"
                # Append signal info to stderr for debugging
                signal_name = get_signal_name(exit_signal)
                stderr = f"{stderr}\n[Process killed by {signal_name}]".strip()
            else:
                # Normal exit
                exit_code = returncode
                exit_signal = None
                exit_reason = "completed"

            # Check for rate limiting in output
            rate_limited = self._detect_rate_limit(stdout, stderr)

            # Determine success: only if exit_code is 0
            success = exit_code == 0

            # Log execution result at appropriate level
            if rate_limited:
                _logger.warning(
                    "rate_limit_detected",
                    duration_seconds=duration,
                    exit_code=exit_code,
                    exit_signal=exit_signal,
                    stdout_bytes=len(stdout),
                    stderr_bytes=len(stderr),
                )
            elif success:
                _logger.info(
                    "execution_completed",
                    duration_seconds=duration,
                    exit_code=exit_code,
                    stdout_bytes=len(stdout),
                    stderr_bytes=len(stderr),
                )
            else:
                # Failed execution - include output tails for debugging
                stdout_tail = stdout[-500:] if len(stdout) > 500 else stdout
                stderr_tail = stderr[-500:] if len(stderr) > 500 else stderr
                _logger.error(
                    "execution_failed",
                    duration_seconds=duration,
                    exit_code=exit_code,
                    exit_signal=exit_signal,
                    exit_reason=exit_reason,
                    stdout_tail=stdout_tail,
                    stderr_tail=stderr_tail,
                )

            return ExecutionResult(
                success=success,
                exit_code=exit_code,
                exit_signal=exit_signal,
                exit_reason=exit_reason,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                rate_limited=rate_limited,
                error_type="rate_limit" if rate_limited else None,
            )

        except FileNotFoundError:
            duration = time.monotonic() - start_time
            _logger.error(
                "cli_not_found",
                error_message="claude CLI not found in PATH",
                duration_seconds=duration,
            )
            return ExecutionResult(
                success=False,
                exit_code=127,
                exit_signal=None,
                exit_reason="error",
                stdout="",
                stderr="claude CLI not found",
                duration_seconds=duration,
                error_type="not_found",
                error_message="claude CLI not found in PATH",
            )
        except Exception as e:
            duration = time.monotonic() - start_time
            _logger.exception(
                "execution_exception",
                error_message=str(e),
                duration_seconds=duration,
            )
            return ExecutionResult(
                success=False,
                exit_code=None,
                exit_signal=None,
                exit_reason="error",
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                error_type="exception",
                error_message=str(e),
            )

    async def _stream_with_progress(
        self,
        process: asyncio.subprocess.Process,
        start_time: float,
        notify_progress: Callable[[str], None],
    ) -> tuple[bytes, bytes]:
        """Stream output from process while tracking progress.

        Reads stdout and stderr concurrently, calling notify_progress periodically
        to report bytes/lines received.

        Args:
            process: The running subprocess.
            start_time: When execution started (for elapsed time calculation).
            notify_progress: Callback to notify progress updates.

        Returns:
            Tuple of (stdout_bytes, stderr_bytes) collected from streams.

        Raises:
            TimeoutError: If execution exceeds timeout_seconds.
        """
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        total_bytes = 0
        total_lines = 0
        last_update = start_time

        async def read_stream(
            stream: asyncio.StreamReader | None,
            chunks: list[bytes],
            is_stdout: bool,
        ) -> None:
            """Read from a stream and accumulate chunks."""
            nonlocal total_bytes, total_lines, last_update

            if stream is None:
                return

            while True:
                try:
                    # Read in chunks for responsive progress updates
                    chunk = await asyncio.wait_for(
                        stream.read(4096),  # 4KB chunks
                        timeout=1.0,  # Check timeout every second
                    )
                except TimeoutError:
                    # 1-second read timeout - check overall timeout
                    if time.monotonic() - start_time > self.timeout_seconds:
                        raise TimeoutError("Execution timeout exceeded") from None
                    # Otherwise just continue reading
                    continue

                if not chunk:
                    break  # EOF

                chunks.append(chunk)
                chunk_bytes = len(chunk)
                chunk_lines = chunk.count(b"\n")

                total_bytes += chunk_bytes
                total_lines += chunk_lines

                # Check if we should notify progress
                now = time.monotonic()
                if now - last_update >= self.progress_interval_seconds:
                    notify_progress("executing")
                    last_update = now

        # Read both streams concurrently
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    read_stream(process.stdout, stdout_chunks, True),
                    read_stream(process.stderr, stderr_chunks, False),
                ),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            raise  # Re-raise for caller to handle

        # Wait for process to complete
        await process.wait()

        return b"".join(stdout_chunks), b"".join(stderr_chunks)

    # Alias for Backend protocol
    async def execute(self, prompt: str) -> ExecutionResult:
        """Execute a prompt (alias for run_prompt)."""
        return await self.run_prompt(prompt)

    def _detect_rate_limit(self, stdout: str, stderr: str) -> bool:
        """Check output for rate limit indicators.

        Uses the shared ErrorClassifier to ensure consistent detection
        with the runner's error classification.
        """
        # Use ErrorClassifier for unified rate limit detection
        classified = self._error_classifier.classify(
            stdout=stdout,
            stderr=stderr,
            exit_code=1,  # Assume failure for classification
        )
        return classified.category == ErrorCategory.RATE_LIMIT

    async def health_check(self) -> bool:
        """Check if claude CLI is available and responsive."""
        if not self._claude_path:
            return False

        try:
            # Simple test prompt
            result = await self.run_prompt("Say 'ready' and nothing else.")
            return result.success and "ready" in result.stdout.lower()
        except Exception:
            return False
