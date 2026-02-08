"""Generic subprocess manager for CLI backends.

Handles subprocess lifecycle: spawn, stream, timeout, cleanup.
Can be composed into any CLI-based backend (Claude, Gemini, GPT, local LLMs).

Security Note: Uses asyncio.create_subprocess_exec() which is shell-injection
safe - arguments are passed as a list, not interpolated into a shell command.

Example - Creating a new CLI backend:

    from mozart.backends.base import Backend, ExecutionResult
    from mozart.backends.process_manager import ProcessManager

    class GeminiCliBackend(Backend):
        def __init__(self, timeout_seconds: float = 300.0):
            self._process_mgr = ProcessManager(
                timeout_seconds=timeout_seconds,
                kill_children_on_exit=True,  # Kill any spawned servers
            )

        async def execute(self, prompt: str) -> ExecutionResult:
            cmd = ["gemini-cli", "--prompt", prompt]
            result = await self._process_mgr.run(cmd)

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout.decode("utf-8", errors="replace"),
                stderr=result.stderr.decode("utf-8", errors="replace"),
                duration_seconds=result.duration_seconds,
                exit_code=result.returncode,
                exit_signal=result.exit_signal,
            )

        async def health_check(self) -> bool:
            result = await self._process_mgr.run(["gemini-cli", "--version"])
            return result.returncode == 0

        @property
        def name(self) -> str:
            return "gemini-cli"
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mozart.core.logging import get_logger

_logger = get_logger("process_manager")

# Timeout constants
GRACEFUL_TERMINATION_TIMEOUT: float = 5.0  # Seconds to wait for graceful termination
PROCESS_EXIT_TIMEOUT: float = 5.0  # Seconds to wait for process exit after streams close


@dataclass
class ProcessResult:
    """Result of running a subprocess."""

    stdout: bytes
    stderr: bytes
    returncode: int | None
    exit_signal: int | None
    duration_seconds: float
    timed_out: bool = False
    killed_by_exception: bool = False


ProgressCallback = Callable[[dict[str, Any]], None]


class ProcessManager:
    """Manages subprocess lifecycle with proper cleanup.

    Features:
    - Process group isolation (start_new_session=True)
    - Timeout handling with graceful then forced termination
    - Progress streaming with callbacks
    - Proper cleanup on exceptions to prevent leaks
    - Workaround for long-running child processes (e.g., MCP servers)

    Usage:
        mgr = ProcessManager(timeout_seconds=300)
        result = await mgr.run(["my-cli", "--flag", "arg"], cwd="/path")
    """

    def __init__(
        self,
        timeout_seconds: float = 300.0,
        progress_callback: ProgressCallback | None = None,
        progress_interval_seconds: float = 1.0,
        kill_children_on_exit: bool = True,
    ):
        """Initialize process manager.

        Args:
            timeout_seconds: Max execution time before killing process.
            progress_callback: Optional callback for progress updates.
            progress_interval_seconds: How often to call progress callback.
            kill_children_on_exit: If True, kill entire process group on exit.
                Enable this for CLIs that spawn long-running children (like MCP servers).
        """
        self.timeout_seconds = timeout_seconds
        self.progress_callback = progress_callback
        self.progress_interval_seconds = progress_interval_seconds
        self.kill_children_on_exit = kill_children_on_exit

    async def run(
        self,
        cmd: list[str],
        cwd: Path | str | None = None,
        env: dict[str, str] | None = None,
    ) -> ProcessResult:
        """Run a subprocess with proper lifecycle management.

        Args:
            cmd: Command and arguments as list (shell-injection safe).
            cwd: Working directory for the process.
            env: Environment variables (defaults to current env).

        Returns:
            ProcessResult with stdout, stderr, and exit info.
        """
        start_time = time.monotonic()
        process: asyncio.subprocess.Process | None = None

        # Use current environment if not specified
        if env is None:
            env = os.environ.copy()

        _logger.debug(
            "process.starting",
            command=cmd[0],
            args_count=len(cmd) - 1,
            cwd=str(cwd) if cwd else None,
            timeout_seconds=self.timeout_seconds,
        )

        try:
            # start_new_session creates a new process group for clean cleanup
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
                start_new_session=True,
            )

            # Read output with progress tracking
            try:
                if self.progress_callback:
                    stdout, stderr = await self._stream_with_progress(
                        process, start_time
                    )
                else:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout_seconds,
                    )

            except TimeoutError:
                # Handle timeout with graceful then forced termination
                await self._terminate_process(process)
                duration = time.monotonic() - start_time

                _logger.warning(
                    "process.timeout",
                    pid=process.pid,
                    timeout_seconds=self.timeout_seconds,
                    duration_seconds=duration,
                )

                return ProcessResult(
                    stdout=b"",
                    stderr=f"Process timed out after {self.timeout_seconds}s".encode(),
                    returncode=None,
                    exit_signal=signal.SIGKILL,
                    duration_seconds=duration,
                    timed_out=True,
                )

            # Normal completion
            duration = time.monotonic() - start_time
            returncode = process.returncode

            # Parse returncode: negative means killed by signal
            exit_signal = None
            if returncode is not None and returncode < 0:
                exit_signal = -returncode
                returncode = None

            _logger.debug(
                "process.completed",
                pid=process.pid,
                returncode=returncode,
                exit_signal=exit_signal,
                duration_seconds=duration,
                stdout_bytes=len(stdout),
                stderr_bytes=len(stderr),
            )

            return ProcessResult(
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                exit_signal=exit_signal,
                duration_seconds=duration,
            )

        except Exception as e:
            # CRITICAL: Kill orphaned process to prevent leaks
            duration = time.monotonic() - start_time

            if process is not None and process.returncode is None:
                _logger.warning(
                    "process.killing_orphan",
                    pid=process.pid,
                    reason="exception",
                    error=str(e),
                )
                await self._kill_process_group(process)

            _logger.exception(
                "process.exception",
                error=str(e),
                duration_seconds=duration,
            )

            return ProcessResult(
                stdout=b"",
                stderr=str(e).encode(),
                returncode=None,
                exit_signal=None,
                duration_seconds=duration,
                killed_by_exception=True,
            )

    async def _stream_with_progress(
        self,
        process: asyncio.subprocess.Process,
        start_time: float,
    ) -> tuple[bytes, bytes]:
        """Stream output while reporting progress.

        Also handles the case where the process doesn't exit after
        streams close (e.g., waiting on child processes like MCP servers).
        """
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        last_progress_time = start_time

        async def read_stream(
            stream: asyncio.StreamReader | None,
            chunks: list[bytes],
        ) -> None:
            if stream is None:
                return
            while True:
                chunk = await stream.read(8192)
                if not chunk:
                    break
                chunks.append(chunk)

                # Progress callback
                nonlocal last_progress_time
                now = time.monotonic()
                if now - last_progress_time >= self.progress_interval_seconds:
                    if self.progress_callback:
                        total_bytes = sum(len(c) for c in stdout_chunks) + sum(
                            len(c) for c in stderr_chunks
                        )
                        self.progress_callback({
                            "bytes_received": total_bytes,
                            "elapsed_seconds": now - start_time,
                            "phase": "streaming",
                        })
                    last_progress_time = now

        # Read both streams concurrently
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    read_stream(process.stdout, stdout_chunks),
                    read_stream(process.stderr, stderr_chunks),
                ),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            raise  # Re-raise for caller to handle

        # Wait for process to exit, with defensive cleanup for hung processes
        try:
            await asyncio.wait_for(process.wait(), timeout=PROCESS_EXIT_TIMEOUT)
        except TimeoutError:
            # Process didn't exit - likely waiting on child processes
            _logger.warning(
                "process.exit_timeout",
                pid=process.pid,
                message="Process did not exit after streams closed",
                timeout_seconds=PROCESS_EXIT_TIMEOUT,
            )

            if self.kill_children_on_exit:
                await self._kill_process_group(process)

        return b"".join(stdout_chunks), b"".join(stderr_chunks)

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        """Gracefully terminate, then force kill if needed."""
        try:
            process.terminate()
            await asyncio.wait_for(
                process.wait(), timeout=GRACEFUL_TERMINATION_TIMEOUT
            )
        except TimeoutError:
            await self._kill_process_group(process)

    async def _kill_process_group(self, process: asyncio.subprocess.Process) -> None:
        """Kill the entire process group (process + all children)."""
        pid = process.pid
        if pid is None:
            return

        # Kill entire process group
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass  # Process group may not exist

        # Ensure main process is dead
        try:
            process.kill()
        except ProcessLookupError:
            pass

        try:
            await process.wait()
        except Exception:
            pass
