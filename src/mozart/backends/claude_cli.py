"""Claude CLI backend using subprocess.

Wraps the `claude` CLI command for running prompts.
Based on patterns from run-batch-review.sh.

Security Note: This module uses asyncio.create_subprocess_exec() which is the
safe subprocess method in Python - it does NOT use shell=True, so there is no
shell injection risk. Arguments are passed as a list, not interpolated into
a shell command string.
"""

import asyncio
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from mozart.backends.base import Backend, ExecutionResult
from mozart.core.config import BackendConfig


class ClaudeCliBackend(Backend):
    """Run prompts via the Claude CLI.

    Uses asyncio.create_subprocess_exec to invoke `claude -p <prompt>`.
    This is shell-injection safe as arguments are passed as a list.
    """

    def __init__(
        self,
        skip_permissions: bool = True,
        output_format: Optional[str] = None,
        working_directory: Optional[Path] = None,
        timeout_seconds: float = 1800.0,  # 30 minute default
    ):
        """Initialize CLI backend.

        Args:
            skip_permissions: Pass --dangerously-skip-permissions
            output_format: Output format (json, text, stream-json)
            working_directory: Working directory for running commands
            timeout_seconds: Maximum time allowed per prompt
        """
        self.skip_permissions = skip_permissions
        self.output_format = output_format
        self.working_directory = working_directory
        self.timeout_seconds = timeout_seconds

        # Verify claude CLI is available
        self._claude_path = shutil.which("claude")

    @classmethod
    def from_config(cls, config: BackendConfig) -> "ClaudeCliBackend":
        """Create backend from configuration."""
        return cls(
            skip_permissions=config.skip_permissions,
            output_format=config.output_format,
            working_directory=config.working_directory,
            timeout_seconds=config.timeout_seconds,
        )

    @property
    def name(self) -> str:
        return "claude-cli"

    def _build_command(self, prompt: str) -> list[str]:
        """Build the claude command with arguments.

        Returns a list of arguments for subprocess - NOT a shell string.
        """
        if not self._claude_path:
            raise RuntimeError("claude CLI not found in PATH")

        cmd = [self._claude_path, "-p", prompt]

        if self.skip_permissions:
            cmd.append("--dangerously-skip-permissions")

        if self.output_format:
            cmd.extend(["--output-format", self.output_format])

        return cmd

    async def run_prompt(self, prompt: str) -> ExecutionResult:
        """Run a prompt via claude CLI.

        Uses create_subprocess_exec (safe, no shell) to invoke claude.
        """
        cmd = self._build_command(prompt)
        start_time = time.monotonic()

        try:
            # create_subprocess_exec is shell-injection safe
            # Arguments are passed as list, not interpolated into shell string
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                duration = time.monotonic() - start_time
                return ExecutionResult(
                    success=False,
                    exit_code=124,  # Timeout exit code
                    stdout="",
                    stderr="Command timed out",
                    duration_seconds=duration,
                    error_type="timeout",
                    error_message=f"Timed out after {self.timeout_seconds}s",
                )

            duration = time.monotonic() - start_time
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = process.returncode or 0

            # Check for rate limiting in output
            rate_limited = self._detect_rate_limit(stdout, stderr)

            return ExecutionResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                rate_limited=rate_limited,
                error_type="rate_limit" if rate_limited else None,
            )

        except FileNotFoundError:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                exit_code=127,
                stdout="",
                stderr="claude CLI not found",
                duration_seconds=duration,
                error_type="not_found",
                error_message="claude CLI not found in PATH",
            )
        except Exception as e:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                error_type="exception",
                error_message=str(e),
            )

    # Alias for Backend protocol
    async def execute(self, prompt: str) -> ExecutionResult:
        """Execute a prompt (alias for run_prompt)."""
        return await self.run_prompt(prompt)

    def _detect_rate_limit(self, stdout: str, stderr: str) -> bool:
        """Check output for rate limit indicators."""
        patterns = [
            r"rate.?limit",
            r"usage.?limit",
            r"quota",
            r"too many requests",
            r"429",
            r"capacity",
            r"try again later",
        ]
        combined = f"{stdout}\n{stderr}".lower()
        return any(re.search(p, combined, re.IGNORECASE) for p in patterns)

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
