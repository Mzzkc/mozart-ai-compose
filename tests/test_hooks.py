"""Tests for post-success hook execution.

Covers:
- HookResult dataclass
- ConcertContext tracking
- HookExecutor template expansion
- Hook type dispatch
- Basic hook execution flow
- Detached child process survival (Popen vs asyncio transport)
"""

import os
import signal
import subprocess as _subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from mozart.core.config import JobConfig, PostSuccessHookConfig
from mozart.execution.hooks import ConcertContext, HookExecutor, HookResult


class TestHookResult:
    """Tests for HookResult dataclass."""

    def test_basic_result(self) -> None:
        """HookResult should store hook execution metadata."""
        result = HookResult(
            hook_type="run_command",
            description="Test hook",
            success=True,
            exit_code=0,
            duration_seconds=1.5,
        )

        assert result.hook_type == "run_command"
        assert result.description == "Test hook"
        assert result.success is True
        assert result.exit_code == 0
        assert result.duration_seconds == 1.5

    def test_failed_result(self) -> None:
        """HookResult should store failure information."""
        result = HookResult(
            hook_type="run_script",
            description="Failing hook",
            success=False,
            exit_code=1,
            error_message="Script returned non-zero",
        )

        assert result.success is False
        assert result.error_message == "Script returned non-zero"

    def test_chained_job_result(self) -> None:
        """HookResult should store chained job info for run_job hooks."""
        result = HookResult(
            hook_type="run_job",
            description="Chain to next job",
            success=True,
            chained_job_path=Path("/config/next.yaml"),
            chained_job_workspace=Path("/workspace"),
        )

        assert result.chained_job_path == Path("/config/next.yaml")
        assert result.chained_job_workspace == Path("/workspace")


class TestConcertContext:
    """Tests for ConcertContext tracking."""

    def test_default_values(self) -> None:
        """ConcertContext should have sensible defaults."""
        ctx = ConcertContext(concert_id="test-concert")

        assert ctx.concert_id == "test-concert"
        assert ctx.chain_depth == 0
        assert ctx.parent_job_id is None
        assert ctx.total_jobs_run == 0
        assert ctx.jobs_in_chain == []

    def test_chain_tracking(self) -> None:
        """ConcertContext should track job chain."""
        ctx = ConcertContext(
            concert_id="concert-1",
            chain_depth=2,
            parent_job_id="job-1",
            total_jobs_run=3,
            jobs_in_chain=["job-0", "job-1", "job-2"],
        )

        assert ctx.chain_depth == 2
        assert len(ctx.jobs_in_chain) == 3
        assert "job-1" in ctx.jobs_in_chain


class TestHookExecutor:
    """Tests for HookExecutor class."""

    @pytest.fixture
    def minimal_config(self) -> JobConfig:
        """Create minimal JobConfig for testing."""
        return JobConfig.model_validate({
            "name": "test-job",
            "description": "Test job",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 10, "total_items": 10},
            "prompt": {"template": "Test prompt"},
        })

    @pytest.fixture
    def config_with_hooks(self) -> JobConfig:
        """Create JobConfig with hooks configured."""
        return JobConfig.model_validate({
            "name": "hook-test-job",
            "description": "Job with hooks",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 10, "total_items": 10},
            "prompt": {"template": "Test"},
            "on_success": [
                {
                    "type": "run_command",
                    "command": "echo 'Success!'",
                    "description": "Echo success",
                },
            ],
        })

    def test_expand_hook_variables(self, minimal_config: JobConfig) -> None:
        """_expand_hook_variables should substitute workspace and job variables."""
        executor = HookExecutor(
            config=minimal_config,
            workspace=Path("/test/workspace"),
        )

        result = executor._expand_hook_variables("{workspace}/output/{job_id}.json")

        assert result == "/test/workspace/output/test-job.json"

    def test_expand_hook_variables_sheet_count(self, minimal_config: JobConfig) -> None:
        """_expand_hook_variables should substitute sheet_count."""
        executor = HookExecutor(
            config=minimal_config,
            workspace=Path("/work"),
        )

        result = executor._expand_hook_variables("Processed {sheet_count} sheets")

        assert result == "Processed 1 sheets"

    @pytest.mark.asyncio
    async def test_no_hooks_returns_empty(self, minimal_config: JobConfig) -> None:
        """execute_hooks should return empty list when no hooks configured."""
        executor = HookExecutor(
            config=minimal_config,
            workspace=Path("/test"),
        )

        result = await executor.execute_hooks()

        assert result == []

    @pytest.mark.asyncio
    async def test_run_command_hook(self, config_with_hooks: JobConfig) -> None:
        """run_command hooks should execute shell commands."""
        executor = HookExecutor(
            config=config_with_hooks,
            workspace=Path("/tmp"),
        )

        results = await executor.execute_hooks()

        # Assuming 'echo' is available and succeeds
        assert len(results) == 1
        assert results[0].hook_type == "run_command"
        assert results[0].success is True
        assert results[0].exit_code == 0

    @pytest.mark.asyncio
    async def test_hook_failure_tracking(self) -> None:
        """Failed hooks should have error information."""
        config = JobConfig.model_validate({
            "name": "fail-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "on_success": [
                {
                    "type": "run_command",
                    "command": "exit 42",  # Intentional failure
                    "description": "Failing command",
                },
            ],
        })

        executor = HookExecutor(config=config, workspace=Path("/tmp"))
        results = await executor.execute_hooks()

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].exit_code == 42

    @pytest.mark.asyncio
    async def test_run_job_missing_path(self) -> None:
        """run_job hook should fail gracefully for missing job_path."""
        config = JobConfig.model_validate({
            "name": "run-job-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "on_success": [
                {
                    "type": "run_job",
                    "job_path": "/nonexistent/job.yaml",
                    "description": "Chain to missing job",
                },
            ],
        })

        executor = HookExecutor(config=config, workspace=Path("/tmp"))
        results = await executor.execute_hooks()

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error_message is not None
        assert "not found" in results[0].error_message.lower()

    @pytest.mark.asyncio
    async def test_unknown_hook_type(self) -> None:
        """Unknown hook types should fail gracefully."""
        config = JobConfig.model_validate({
            "name": "unknown-hook-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
        })
        # Manually add a hook and modify its type attribute after creation
        hook = PostSuccessHookConfig(
            type="run_command",  # Valid type for construction
            command="echo test",  # Required for run_command type
            description="Unknown hook",
        )
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(hook, "type", "unsupported_type")
        config.on_success = [hook]

        executor = HookExecutor(config=config, workspace=Path("/tmp"))
        results = await executor.execute_hooks()

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error_message is not None
        assert "Unknown hook type" in results[0].error_message

    def test_get_next_job_to_chain_none(self, minimal_config: JobConfig) -> None:
        """get_next_job_to_chain should return None when no chained jobs."""
        executor = HookExecutor(config=minimal_config, workspace=Path("/tmp"))

        result = executor.get_next_job_to_chain()

        assert result is None

    def test_get_next_job_to_chain_success(self, minimal_config: JobConfig) -> None:
        """get_next_job_to_chain should return successful run_job path."""
        executor = HookExecutor(config=minimal_config, workspace=Path("/tmp"))

        # Simulate a successful run_job hook
        executor.hook_results = [
            HookResult(
                hook_type="run_job",
                description="Chain job",
                success=True,
                chained_job_path=Path("/config/next.yaml"),
                chained_job_workspace=Path("/work"),
            ),
        ]

        result = executor.get_next_job_to_chain()

        assert result is not None
        job_path, workspace = result
        assert job_path == Path("/config/next.yaml")
        assert workspace == Path("/work")


class TestConcertLimits:
    """Tests for concert chain depth limiting."""

    @pytest.mark.asyncio
    async def test_chain_depth_limit_enforced(self, tmp_path: Path) -> None:
        """run_job should fail when chain depth limit reached."""
        # Create a real job config file so we pass the existence check
        job_config = tmp_path / "next-job.yaml"
        job_config.write_text("name: next-job\nbackend:\n  type: claude_cli\n")

        config = JobConfig.model_validate({
            "name": "depth-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "concert": {
                "enabled": True,
                "max_chain_depth": 2,
            },
            "on_success": [
                {
                    "type": "run_job",
                    "job_path": str(job_config),
                    "description": "Chained job",
                },
            ],
        })

        # Create context at depth limit
        concert_ctx = ConcertContext(
            concert_id="test",
            chain_depth=2,  # Already at limit
        )

        executor = HookExecutor(
            config=config,
            workspace=Path("/tmp"),
            concert_context=concert_ctx,
        )

        results = await executor.execute_hooks()

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error_message is not None
        assert "depth limit" in results[0].error_message.lower()


class TestHookErrorPaths:
    """Tests for hook error paths: timeout, abort-on-failure, detached spawning, exceptions."""

    @pytest.mark.asyncio
    async def test_abort_on_failure_stops_remaining_hooks(self) -> None:
        """When on_failure='abort', remaining hooks should not execute."""
        config = JobConfig.model_validate({
            "name": "abort-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "on_success": [
                {
                    "type": "run_command",
                    "command": "exit 1",
                    "description": "Failing hook",
                    "on_failure": "abort",
                },
                {
                    "type": "run_command",
                    "command": "echo 'should not run'",
                    "description": "Should be skipped",
                },
            ],
        })

        executor = HookExecutor(config=config, workspace=Path("/tmp"))
        results = await executor.execute_hooks()

        # Only the first hook should have run
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].exit_code == 1

    @pytest.mark.asyncio
    async def test_continue_on_failure_runs_remaining_hooks(self) -> None:
        """When on_failure='continue' (default), remaining hooks should execute."""
        config = JobConfig.model_validate({
            "name": "continue-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "on_success": [
                {
                    "type": "run_command",
                    "command": "exit 1",
                    "description": "Failing hook",
                    "on_failure": "continue",
                },
                {
                    "type": "run_command",
                    "command": "echo 'still runs'",
                    "description": "Should still run",
                },
            ],
        })

        executor = HookExecutor(config=config, workspace=Path("/tmp"))
        results = await executor.execute_hooks()

        assert len(results) == 2
        assert results[0].success is False
        assert results[1].success is True

    @pytest.mark.asyncio
    async def test_concert_abort_on_hook_failure(self) -> None:
        """abort_concert_on_hook_failure should stop remaining hooks."""
        config = JobConfig.model_validate({
            "name": "concert-abort-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "concert": {
                "enabled": True,
                "abort_concert_on_hook_failure": True,
            },
            "on_success": [
                {
                    "type": "run_command",
                    "command": "exit 1",
                    "description": "Failing hook",
                },
                {
                    "type": "run_command",
                    "command": "echo 'should not run'",
                    "description": "Should be skipped",
                },
            ],
        })

        executor = HookExecutor(config=config, workspace=Path("/tmp"))
        results = await executor.execute_hooks()

        # Only the first hook should have run due to concert abort
        assert len(results) == 1
        assert results[0].success is False

    @pytest.mark.asyncio
    async def test_command_timeout_returns_failure(self) -> None:
        """Hook exceeding timeout should return failure with timeout message.

        Uses a subprocess that creates its own process group so kill() works
        properly (shell subprocesses don't propagate SIGKILL to children).
        """
        config = JobConfig.model_validate({
            "name": "timeout-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "on_success": [
                {
                    "type": "run_command",
                    # Use exec to replace the shell process so kill() reaches it
                    "command": "exec sleep 60",
                    "description": "Slow hook",
                    "timeout_seconds": 0.5,
                },
            ],
        })

        executor = HookExecutor(config=config, workspace=Path("/tmp"))
        results = await executor.execute_hooks()

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error_message is not None
        assert "timeout" in results[0].error_message.lower()

    @pytest.mark.asyncio
    async def test_script_timeout_returns_failure(self) -> None:
        """run_script hook exceeding timeout should return failure.

        run_script uses subprocess_exec (no shell), so kill() targets
        the process directly.
        """
        import sys

        config = JobConfig.model_validate({
            "name": "script-timeout-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "on_success": [
                {
                    "type": "run_script",
                    # Use python -c for a direct process (no shell children)
                    "command": f"{sys.executable} -c 'import time; time.sleep(60)'",
                    "description": "Slow script",
                    "timeout_seconds": 0.5,
                },
            ],
        })

        executor = HookExecutor(config=config, workspace=Path("/tmp"))
        results = await executor.execute_hooks()

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error_message is not None
        assert "timeout" in results[0].error_message.lower()

    def test_run_command_missing_command(self) -> None:
        """run_command with no command should be rejected at construction."""
        with pytest.raises(ValidationError, match="requires 'command' field"):
            PostSuccessHookConfig(
                type="run_command",
                description="Missing command",
            )

    def test_run_script_missing_command(self) -> None:
        """run_script with no command should be rejected at construction."""
        with pytest.raises(ValidationError, match="requires 'command' field"):
            PostSuccessHookConfig(
                type="run_script",
                description="Missing script",
            )

    def test_run_job_no_job_path(self) -> None:
        """run_job with no job_path should be rejected at construction."""
        with pytest.raises(ValidationError, match="requires 'job_path' field"):
            PostSuccessHookConfig(
                type="run_job",
                description="Missing path",
            )

    @pytest.mark.asyncio
    async def test_hook_exception_captured_as_failure(self) -> None:
        """Exceptions during hook execution should be captured, not raised."""
        config = JobConfig.model_validate({
            "name": "exception-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "on_success": [
                {
                    "type": "run_script",
                    "command": "",  # Will cause shlex.split to succeed but empty args
                    "description": "Exception hook",
                },
            ],
        })

        executor = HookExecutor(config=config, workspace=Path("/tmp"))
        results = await executor.execute_hooks()

        # Should produce a result (possibly failure), not raise
        assert len(results) == 1
        assert results[0].success is False

    @pytest.mark.asyncio
    async def test_exception_with_abort_stops_remaining(self) -> None:
        """Exception in hook with on_failure='abort' should stop remaining hooks."""
        config = JobConfig.model_validate({
            "name": "exc-abort-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "on_success": [
                {
                    "type": "run_script",
                    "command": "/nonexistent/binary/xyzabc123",
                    "description": "Will throw FileNotFoundError",
                    "on_failure": "abort",
                },
                {
                    "type": "run_command",
                    "command": "echo 'should not run'",
                    "description": "Should be skipped",
                },
            ],
        })

        executor = HookExecutor(config=config, workspace=Path("/tmp"))
        results = await executor.execute_hooks()

        # First hook fails with exception, abort prevents second hook
        assert len(results) == 1
        assert results[0].success is False

    @pytest.mark.asyncio
    async def test_chain_depth_below_limit_proceeds(self, tmp_path: Path) -> None:
        """run_job should proceed past the depth check when below limit.

        Rather than actually spawning mozart (which may not be in PATH or may
        take too long), we test that the depth check logic is correct by
        directly calling _execute_run_job and verifying the error is NOT
        about depth limit.
        """
        job_config = tmp_path / "next-job.yaml"
        job_config.write_text("name: next-job\nbackend:\n  type: claude_cli\n")

        config = JobConfig.model_validate({
            "name": "depth-ok-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "concert": {
                "enabled": True,
                "max_chain_depth": 5,
                "cooldown_between_jobs_seconds": 0,
            },
        })

        hook = PostSuccessHookConfig(
            type="run_job",
            job_path=job_config,
            description="Chained job",
            timeout_seconds=2.0,  # Short timeout to avoid hanging
        )

        concert_ctx = ConcertContext(
            concert_id="test",
            chain_depth=1,  # Below limit of 5
        )

        executor = HookExecutor(
            config=config,
            workspace=Path("/tmp"),
            concert_context=concert_ctx,
        )

        result = await executor._execute_run_job(hook)

        # Should NOT fail due to depth limit (may fail for other reasons
        # like mozart not being in PATH, but NOT depth limit)
        if not result.success and result.error_message:
            assert "depth limit" not in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_hook_duration_tracked(self) -> None:
        """Hook execution should track duration_seconds."""
        config = JobConfig.model_validate({
            "name": "duration-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "on_success": [
                {
                    "type": "run_command",
                    "command": "echo done",
                    "description": "Quick hook",
                },
            ],
        })

        executor = HookExecutor(config=config, workspace=Path("/tmp"))
        results = await executor.execute_hooks()

        assert len(results) == 1
        assert results[0].duration_seconds >= 0.0


class TestDetachedChildSurvival:
    """Tests that detached hook children survive parent event loop closure.

    Verifies the fix for the asyncio BaseSubprocessTransport SIGKILL bug:
    asyncio.create_subprocess_exec wraps children in a transport whose
    __del__ kills the child PID when the event loop closes. Using
    subprocess.Popen instead avoids this because Popen.__del__ only warns.
    """

    @pytest.mark.asyncio
    async def test_detached_child_survives_parent(self, tmp_path: Path) -> None:
        """A detached hook child should still be alive after the hook returns.

        This verifies subprocess.Popen is used (not asyncio.create_subprocess_exec),
        since the asyncio transport would SIGKILL the child on event loop close.
        """
        # Create a real job config so _execute_run_job passes existence check
        job_config = tmp_path / "sleep-job.yaml"
        job_config.write_text(
            "name: sleep-job\n"
            "backend:\n  type: claude_cli\n"
            "sheet:\n  size: 1\n  total_items: 1\n"
            "prompt:\n  template: Test\n"
        )

        config = JobConfig.model_validate({
            "name": "detach-survival-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "concert": {
                "enabled": True,
                "cooldown_between_jobs_seconds": 0,
            },
            "on_success": [
                {
                    "type": "run_job",
                    "job_path": str(job_config),
                    "description": "Detached chain test",
                    "detached": True,
                },
            ],
        })

        # Patch Popen to spawn a real `sleep 30` instead of `mozart run ...`
        # (mozart may not be in PATH in test environments). We capture the
        # real Popen object to check liveness afterward.
        spawned_pids: list[int] = []
        original_popen = _subprocess.Popen

        def capturing_popen(cmd, **kwargs):
            # Replace the mozart command with sleep
            proc = original_popen(["sleep", "30"], **kwargs)
            spawned_pids.append(proc.pid)
            return proc

        executor = HookExecutor(config=config, workspace=tmp_path)

        try:
            with patch("mozart.execution.hooks._subprocess.Popen", side_effect=capturing_popen):
                results = await executor.execute_hooks()

            # Hook should report success
            assert len(results) == 1
            assert results[0].success is True
            assert len(spawned_pids) == 1

            pid = spawned_pids[0]

            # The child should still be alive after the hook returned
            # (os.kill with signal 0 checks existence without sending a signal)
            os.kill(pid, 0)  # Should NOT raise ProcessLookupError

        finally:
            # Clean up: kill any spawned children
            for pid in spawned_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                except (ProcessLookupError, ChildProcessError):
                    pass

    @pytest.mark.asyncio
    async def test_detached_liveness_check_catches_dead_child(self, tmp_path: Path) -> None:
        """If the detached child dies immediately, the hook should report failure."""
        job_config = tmp_path / "fail-job.yaml"
        job_config.write_text(
            "name: fail-job\n"
            "backend:\n  type: claude_cli\n"
            "sheet:\n  size: 1\n  total_items: 1\n"
            "prompt:\n  template: Test\n"
        )

        config = JobConfig.model_validate({
            "name": "detach-dead-test",
            "description": "Test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "Test"},
            "concert": {
                "enabled": True,
                "cooldown_between_jobs_seconds": 0,
            },
            "on_success": [
                {
                    "type": "run_job",
                    "job_path": str(job_config),
                    "description": "Detached dead child test",
                    "detached": True,
                },
            ],
        })

        # Spawn a child that exits immediately (exit 0 completes before
        # the 200ms liveness check)
        original_popen = _subprocess.Popen

        def dying_popen(cmd, **kwargs):
            proc = original_popen(["true"], **kwargs)  # exits immediately
            return proc

        executor = HookExecutor(config=config, workspace=tmp_path)

        with patch("mozart.execution.hooks._subprocess.Popen", side_effect=dying_popen):
            results = await executor.execute_hooks()

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error_message is not None
        assert "exited immediately" in results[0].error_message
