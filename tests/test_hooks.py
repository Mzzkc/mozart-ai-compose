"""Tests for post-success hook execution.

Covers:
- HookResult dataclass
- ConcertContext tracking
- HookExecutor template expansion
- Hook type dispatch
- Basic hook execution flow
"""

import asyncio
from pathlib import Path

import pytest

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

    def test_expand_template(self, minimal_config: JobConfig) -> None:
        """_expand_template should substitute workspace and job variables."""
        executor = HookExecutor(
            config=minimal_config,
            workspace=Path("/test/workspace"),
        )

        result = executor._expand_template("{workspace}/output/{job_id}.json")

        assert "/test/workspace/output/test-job.json" == result

    def test_expand_template_sheet_count(self, minimal_config: JobConfig) -> None:
        """_expand_template should substitute sheet_count."""
        executor = HookExecutor(
            config=minimal_config,
            workspace=Path("/work"),
        )

        result = executor._expand_template("Processed {sheet_count} sheets")

        assert "Processed 1 sheets" == result

    def test_no_hooks_returns_empty(self, minimal_config: JobConfig) -> None:
        """execute_hooks should return empty list when no hooks configured."""
        executor = HookExecutor(
            config=minimal_config,
            workspace=Path("/test"),
        )

        result = asyncio.get_event_loop().run_until_complete(
            executor.execute_hooks()
        )

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
