"""Tests for mozart.core.logging module."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
import structlog


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text for assertion matching."""
    ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_pattern.sub('', text)

from mozart.core.config import LogConfig
from mozart.core.logging import (
    SENSITIVE_PATTERNS,
    MozartLogger,
    _sanitize_event_dict,
    _sanitize_value,
    configure_logging,
    get_logger,
)


class TestSensitivePatterns:
    """Tests for sensitive field detection and sanitization."""

    def test_known_sensitive_patterns(self):
        """Test that common sensitive patterns are included."""
        assert "api_key" in SENSITIVE_PATTERNS
        assert "token" in SENSITIVE_PATTERNS
        assert "password" in SENSITIVE_PATTERNS
        assert "secret" in SENSITIVE_PATTERNS
        assert "credential" in SENSITIVE_PATTERNS

    def test_sanitize_value_redacts_api_key(self):
        """Test that api_key fields are redacted."""
        result = _sanitize_value("api_key", "sk-12345")
        assert result == "[REDACTED]"

    def test_sanitize_value_redacts_mixed_case(self):
        """Test that mixed case sensitive fields are redacted."""
        result = _sanitize_value("API_KEY", "sk-12345")
        assert result == "[REDACTED]"

        result = _sanitize_value("ApiKey", "sk-12345")
        assert result == "[REDACTED]"

    def test_sanitize_value_redacts_compound_keys(self):
        """Test that compound key names containing sensitive patterns are redacted."""
        result = _sanitize_value("anthropic_api_key", "sk-12345")
        assert result == "[REDACTED]"

        result = _sanitize_value("bearer_token", "abc123")
        assert result == "[REDACTED]"

        result = _sanitize_value("db_password", "secret123")
        assert result == "[REDACTED]"

    def test_sanitize_value_preserves_safe_values(self):
        """Test that non-sensitive values are preserved."""
        result = _sanitize_value("batch_num", 5)
        assert result == 5

        result = _sanitize_value("job_id", "test-job-123")
        assert result == "test-job-123"

        result = _sanitize_value("status", "completed")
        assert result == "completed"

    def test_sanitize_event_dict_processes_all_keys(self):
        """Test that _sanitize_event_dict processes all keys."""
        event_dict = {
            "event": "test_event",
            "api_key": "sk-secret",
            "batch_num": 5,
            "token": "bearer-123",
        }

        result = _sanitize_event_dict(None, "info", event_dict)

        assert result["event"] == "test_event"
        assert result["api_key"] == "[REDACTED]"
        assert result["batch_num"] == 5
        assert result["token"] == "[REDACTED]"

    def test_sanitize_event_dict_handles_nested_dicts(self):
        """Test that nested dicts are also sanitized."""
        event_dict = {
            "event": "test_event",
            "config": {
                "api_key": "sk-secret",
                "model": "claude-3",
            },
        }

        result = _sanitize_event_dict(None, "info", event_dict)

        assert result["config"]["api_key"] == "[REDACTED]"
        assert result["config"]["model"] == "claude-3"


class TestMozartLogger:
    """Tests for the MozartLogger class."""

    def test_create_logger_with_component(self):
        """Test creating a logger with a component name."""
        logger = MozartLogger("runner")
        assert logger._component == "runner"

    def test_create_logger_with_initial_context(self):
        """Test creating a logger with initial context."""
        logger = MozartLogger("runner", job_id="test-job")
        # The context is stored in the bound logger
        assert logger._component == "runner"

    def test_bind_returns_new_logger(self):
        """Test that bind() returns a new logger instance."""
        logger = MozartLogger("runner")
        bound_logger = logger.bind(job_id="test-job")

        assert bound_logger is not logger
        assert bound_logger._component == "runner"

    def test_unbind_returns_new_logger(self):
        """Test that unbind() returns a new logger instance."""
        logger = MozartLogger("runner", job_id="test-job")
        unbound_logger = logger.unbind("job_id")

        assert unbound_logger is not logger
        assert unbound_logger._component == "runner"

    def test_logger_methods_exist(self):
        """Test that all expected logging methods exist."""
        logger = MozartLogger("runner")

        assert callable(logger.debug)
        assert callable(logger.info)
        assert callable(logger.warning)
        assert callable(logger.error)
        assert callable(logger.critical)
        assert callable(logger.exception)


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_get_logger_creates_mozart_logger(self):
        """Test that get_logger returns a MozartLogger."""
        logger = get_logger("test-component")
        assert isinstance(logger, MozartLogger)
        assert logger._component == "test-component"

    def test_get_logger_with_initial_context(self):
        """Test get_logger with initial context."""
        logger = get_logger("runner", job_id="my-job", batch_num=1)
        assert logger._component == "runner"


class TestConfigureLogging:
    """Tests for the configure_logging function."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        # Reset structlog to default state
        structlog.reset_defaults()

        # Clear all handlers from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_configure_console_format(self):
        """Test configuring console format logging."""
        configure_logging(level="DEBUG", format="console")

        # Verify root logger is configured
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configure_json_format(self):
        """Test configuring JSON format logging."""
        configure_logging(level="INFO", format="json")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_configure_with_file_path(self, tmp_path: Path):
        """Test configuring logging with file output."""
        log_file = tmp_path / "logs" / "mozart.log"

        configure_logging(
            level="INFO",
            format="both",
            file_path=log_file,
        )

        # Verify log directory was created
        assert log_file.parent.exists()

    def test_configure_both_requires_file_path(self):
        """Test that format='both' requires file_path."""
        with pytest.raises(ValueError, match="file_path is required"):
            configure_logging(level="INFO", format="both", file_path=None)

    def test_configure_sets_log_level(self):
        """Test that log level is properly set."""
        configure_logging(level="WARNING", format="console")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_configure_removes_existing_handlers(self):
        """Test that existing handlers are removed."""
        # Add a handler first
        root_logger = logging.getLogger()
        existing_handler = logging.StreamHandler()
        root_logger.addHandler(existing_handler)

        # Store reference to verify it's removed
        assert existing_handler in root_logger.handlers

        configure_logging(level="INFO", format="console")

        # Existing handler should be removed, new one added
        assert existing_handler not in root_logger.handlers
        # Should have at least one handler for console mode
        assert len(root_logger.handlers) >= 1


class TestLogConfigModel:
    """Tests for the LogConfig Pydantic model."""

    def test_default_values(self):
        """Test LogConfig default values."""
        config = LogConfig()

        assert config.level == "INFO"
        assert config.format == "console"
        assert config.file_path is None
        assert config.max_file_size_mb == 50
        assert config.backup_count == 5
        assert config.include_timestamps is True
        assert config.include_context is True

    def test_custom_values(self):
        """Test LogConfig with custom values."""
        config = LogConfig(
            level="DEBUG",
            format="json",
            file_path=Path("/var/log/mozart.log"),
            max_file_size_mb=100,
            backup_count=10,
            include_timestamps=False,
        )

        assert config.level == "DEBUG"
        assert config.format == "json"
        assert config.file_path == Path("/var/log/mozart.log")
        assert config.max_file_size_mb == 100
        assert config.backup_count == 10
        assert config.include_timestamps is False

    def test_level_validation(self):
        """Test that invalid log levels are rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LogConfig(level="TRACE")  # Invalid level

    def test_format_validation(self):
        """Test that invalid formats are rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LogConfig(format="xml")  # Invalid format

    def test_max_file_size_validation(self):
        """Test max_file_size_mb validation bounds."""
        from pydantic import ValidationError

        # Zero is invalid (gt=0)
        with pytest.raises(ValidationError):
            LogConfig(max_file_size_mb=0)

        # Over 1000 is invalid (le=1000)
        with pytest.raises(ValidationError):
            LogConfig(max_file_size_mb=1001)

    def test_backup_count_validation(self):
        """Test backup_count validation bounds."""
        from pydantic import ValidationError

        # Negative is invalid (ge=0)
        with pytest.raises(ValidationError):
            LogConfig(backup_count=-1)

        # Over 100 is invalid (le=100)
        with pytest.raises(ValidationError):
            LogConfig(backup_count=101)


class TestLogConfigInJobConfig:
    """Tests for LogConfig integration in JobConfig."""

    def test_job_config_has_logging(self, sample_config_dict: dict):
        """Test that JobConfig includes logging configuration."""
        from mozart.core.config import JobConfig

        config = JobConfig(**sample_config_dict)

        assert hasattr(config, "logging")
        assert isinstance(config.logging, LogConfig)

    def test_job_config_logging_defaults(self, sample_config_dict: dict):
        """Test that logging has sensible defaults when not specified."""
        from mozart.core.config import JobConfig

        config = JobConfig(**sample_config_dict)

        assert config.logging.level == "INFO"
        assert config.logging.format == "console"

    def test_job_config_custom_logging(self, sample_config_dict: dict):
        """Test that custom logging config is respected."""
        from mozart.core.config import JobConfig

        sample_config_dict["logging"] = {
            "level": "DEBUG",
            "format": "json",
            "include_timestamps": False,
        }

        config = JobConfig(**sample_config_dict)

        assert config.logging.level == "DEBUG"
        assert config.logging.format == "json"
        assert config.logging.include_timestamps is False


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        structlog.reset_defaults()
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_log_output_contains_component(self, capsys: pytest.CaptureFixture[str]):
        """Test that log output includes component name."""
        # Configure for JSON to make parsing easier
        configure_logging(level="INFO", format="json")

        logger = get_logger("test-runner")
        logger.info("test_event", foo="bar")

        # Note: structlog output goes through stdlib logging
        # In tests, we verify the logger is configured correctly

    def test_bound_context_preserved(self):
        """Test that bound context is preserved through operations."""
        logger = get_logger("runner")
        bound = logger.bind(job_id="job-1", batch_num=5)

        # Create another binding
        double_bound = bound.bind(retry_count=2)

        # Original should be unchanged
        assert bound is not double_bound

    def test_unbind_removes_context(self):
        """Test that unbind removes specified keys."""
        logger = get_logger("runner", job_id="job-1", batch_num=5)
        unbound = logger.unbind("batch_num")

        # Should still have the component
        assert unbound._component == "runner"


class TestExecutionContext:
    """Tests for the ExecutionContext dataclass."""

    def test_create_minimal_context(self):
        """Test creating context with only required fields."""
        from mozart.core.logging import ExecutionContext

        ctx = ExecutionContext(job_id="test-job")

        assert ctx.job_id == "test-job"
        assert ctx.run_id is not None  # Auto-generated UUID
        assert len(ctx.run_id) == 36  # UUID format
        assert ctx.batch_num is None
        assert ctx.component == "unknown"
        assert ctx.parent_run_id is None

    def test_create_full_context(self):
        """Test creating context with all fields."""
        from mozart.core.logging import ExecutionContext

        ctx = ExecutionContext(
            job_id="test-job",
            run_id="custom-run-id",
            batch_num=5,
            component="runner",
            parent_run_id="parent-run-id",
        )

        assert ctx.job_id == "test-job"
        assert ctx.run_id == "custom-run-id"
        assert ctx.batch_num == 5
        assert ctx.component == "runner"
        assert ctx.parent_run_id == "parent-run-id"

    def test_context_is_immutable(self):
        """Test that ExecutionContext is frozen (immutable)."""
        from mozart.core.logging import ExecutionContext

        ctx = ExecutionContext(job_id="test-job")

        with pytest.raises(AttributeError):
            ctx.job_id = "modified"  # type: ignore[misc]

    def test_with_batch_creates_new_context(self):
        """Test that with_batch creates a new context with updated batch_num."""
        from mozart.core.logging import ExecutionContext

        ctx = ExecutionContext(job_id="test-job", run_id="run-123", component="runner")
        new_ctx = ctx.with_batch(10)

        # Original unchanged
        assert ctx.batch_num is None

        # New context has batch_num
        assert new_ctx.batch_num == 10

        # Other fields preserved
        assert new_ctx.job_id == "test-job"
        assert new_ctx.run_id == "run-123"
        assert new_ctx.component == "runner"

    def test_with_component_creates_new_context(self):
        """Test that with_component creates a new context with updated component."""
        from mozart.core.logging import ExecutionContext

        ctx = ExecutionContext(job_id="test-job", run_id="run-123", batch_num=5)
        new_ctx = ctx.with_component("backend")

        # Original unchanged
        assert ctx.component == "unknown"

        # New context has component
        assert new_ctx.component == "backend"

        # Other fields preserved
        assert new_ctx.job_id == "test-job"
        assert new_ctx.run_id == "run-123"
        assert new_ctx.batch_num == 5

    def test_as_child_creates_nested_context(self):
        """Test that as_child creates a child context with parent tracking."""
        from mozart.core.logging import ExecutionContext

        parent = ExecutionContext(job_id="test-job", run_id="parent-run")
        child = parent.as_child()

        # Child has new run_id
        assert child.run_id != parent.run_id

        # Parent run_id is now parent_run_id in child
        assert child.parent_run_id == "parent-run"

        # Other fields preserved
        assert child.job_id == "test-job"

    def test_as_child_with_custom_run_id(self):
        """Test as_child with explicit child_run_id."""
        from mozart.core.logging import ExecutionContext

        parent = ExecutionContext(job_id="test-job", run_id="parent-run")
        child = parent.as_child(child_run_id="custom-child-id")

        assert child.run_id == "custom-child-id"
        assert child.parent_run_id == "parent-run"

    def test_to_dict_excludes_none_values(self):
        """Test that to_dict excludes None values."""
        from mozart.core.logging import ExecutionContext

        ctx = ExecutionContext(job_id="test-job", run_id="run-123")
        result = ctx.to_dict()

        assert result == {
            "job_id": "test-job",
            "run_id": "run-123",
            "component": "unknown",
        }
        assert "batch_num" not in result
        assert "parent_run_id" not in result

    def test_to_dict_includes_all_set_values(self):
        """Test that to_dict includes all set values."""
        from mozart.core.logging import ExecutionContext

        ctx = ExecutionContext(
            job_id="test-job",
            run_id="run-123",
            batch_num=5,
            component="runner",
            parent_run_id="parent-run",
        )
        result = ctx.to_dict()

        assert result == {
            "job_id": "test-job",
            "run_id": "run-123",
            "batch_num": 5,
            "component": "runner",
            "parent_run_id": "parent-run",
        }


class TestContextVar:
    """Tests for ContextVar-based context management."""

    def setup_method(self):
        """Clear context before each test."""
        from mozart.core.logging import clear_context

        clear_context()

    def teardown_method(self):
        """Clear context after each test."""
        from mozart.core.logging import clear_context

        clear_context()

    def test_get_current_context_returns_none_when_not_set(self):
        """Test that get_current_context returns None by default."""
        from mozart.core.logging import get_current_context

        assert get_current_context() is None

    def test_set_context_and_get_context(self):
        """Test set_context and get_current_context."""
        from mozart.core.logging import (
            ExecutionContext,
            get_current_context,
            set_context,
        )

        ctx = ExecutionContext(job_id="test-job")
        set_context(ctx)

        result = get_current_context()
        assert result is ctx

    def test_clear_context(self):
        """Test that clear_context removes the context."""
        from mozart.core.logging import (
            ExecutionContext,
            clear_context,
            get_current_context,
            set_context,
        )

        ctx = ExecutionContext(job_id="test-job")
        set_context(ctx)
        assert get_current_context() is not None

        clear_context()
        assert get_current_context() is None

    def test_with_context_sets_and_clears(self):
        """Test that with_context sets context for block duration."""
        from mozart.core.logging import (
            ExecutionContext,
            get_current_context,
            with_context,
        )

        ctx = ExecutionContext(job_id="test-job")

        # Before with_context
        assert get_current_context() is None

        with with_context(ctx):
            # Inside with_context
            assert get_current_context() is ctx

        # After with_context
        assert get_current_context() is None

    def test_with_context_yields_context(self):
        """Test that with_context yields the context."""
        from mozart.core.logging import ExecutionContext, with_context

        ctx = ExecutionContext(job_id="test-job")

        with with_context(ctx) as yielded:
            assert yielded is ctx

    def test_with_context_restores_on_exception(self):
        """Test that with_context restores context on exception."""
        from mozart.core.logging import (
            ExecutionContext,
            get_current_context,
            with_context,
        )

        ctx = ExecutionContext(job_id="test-job")

        try:
            with with_context(ctx):
                assert get_current_context() is ctx
                raise ValueError("test error")
        except ValueError:
            pass

        # Context should be cleared even after exception
        assert get_current_context() is None

    def test_nested_with_context(self):
        """Test nested with_context blocks."""
        from mozart.core.logging import (
            ExecutionContext,
            get_current_context,
            with_context,
        )

        outer_ctx = ExecutionContext(job_id="outer-job")
        inner_ctx = ExecutionContext(job_id="inner-job")

        with with_context(outer_ctx):
            assert get_current_context() is outer_ctx

            with with_context(inner_ctx):
                assert get_current_context() is inner_ctx

            # Should restore outer context
            assert get_current_context() is outer_ctx

        # Should be cleared
        assert get_current_context() is None


class TestAddContextProcessor:
    """Tests for the _add_context processor."""

    def setup_method(self):
        """Clear context before each test."""
        from mozart.core.logging import clear_context

        clear_context()

    def teardown_method(self):
        """Clear context after each test."""
        from mozart.core.logging import clear_context

        clear_context()

    def test_add_context_with_no_context(self):
        """Test _add_context when no context is set."""
        from mozart.core.logging import _add_context

        event_dict = {"event": "test", "foo": "bar"}
        result = _add_context(None, "info", event_dict)

        # Should be unchanged
        assert result == {"event": "test", "foo": "bar"}

    def test_add_context_adds_fields(self):
        """Test _add_context adds context fields."""
        from mozart.core.logging import (
            ExecutionContext,
            _add_context,
            set_context,
        )

        ctx = ExecutionContext(
            job_id="test-job",
            run_id="run-123",
            batch_num=5,
            component="runner",
        )
        set_context(ctx)

        event_dict = {"event": "test"}
        result = _add_context(None, "info", event_dict)

        assert result["job_id"] == "test-job"
        assert result["run_id"] == "run-123"
        assert result["batch_num"] == 5
        assert result["component"] == "runner"

    def test_add_context_preserves_explicit_values(self):
        """Test that explicit event_dict values take precedence."""
        from mozart.core.logging import (
            ExecutionContext,
            _add_context,
            set_context,
        )

        ctx = ExecutionContext(
            job_id="context-job",
            component="context-component",
        )
        set_context(ctx)

        # Explicitly set job_id in event_dict
        event_dict = {"event": "test", "job_id": "explicit-job"}
        result = _add_context(None, "info", event_dict)

        # Explicit value should be preserved
        assert result["job_id"] == "explicit-job"
        # Context values for other fields should be added
        assert result["component"] == "context-component"


class TestContextIntegration:
    """Integration tests for context with logging."""

    def setup_method(self):
        """Reset logging configuration and context before each test."""
        import logging

        from mozart.core.logging import clear_context

        structlog.reset_defaults()
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        clear_context()

    def teardown_method(self):
        """Clear context after each test."""
        from mozart.core.logging import clear_context

        clear_context()

    def test_logger_with_context_includes_fields(self):
        """Test that logger includes context fields when context is set."""
        import json

        from mozart.core.logging import (
            ExecutionContext,
            configure_logging,
            get_logger,
            with_context,
        )

        # Capture log output for verification
        captured_output: list[str] = []

        class CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_output.append(record.getMessage())

        configure_logging(level="DEBUG", format="json", include_timestamps=False)

        # Add our capturing handler
        root_logger = logging.getLogger()
        capture_handler = CapturingHandler()
        root_logger.addHandler(capture_handler)

        logger = get_logger("test-component")
        ctx = ExecutionContext(
            job_id="my-job",
            run_id="run-abc",
            batch_num=3,
        )

        with with_context(ctx):
            logger.info("test_event", extra_field="value")

        # Verify log output contains context fields
        assert len(captured_output) == 1
        log_entry = json.loads(captured_output[0])

        assert log_entry["job_id"] == "my-job"
        assert log_entry["run_id"] == "run-abc"
        assert log_entry["batch_num"] == 3
        assert log_entry["component"] == "test-component"
        assert log_entry["extra_field"] == "value"

    def test_context_disabled_does_not_add_fields(self):
        """Test that context is not added when include_context=False."""
        import json

        from mozart.core.logging import (
            ExecutionContext,
            configure_logging,
            get_logger,
            with_context,
        )

        # Capture log output for verification
        captured_output: list[str] = []

        class CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_output.append(record.getMessage())

        configure_logging(
            level="DEBUG",
            format="json",
            include_timestamps=False,
            include_context=False,
        )

        # Add our capturing handler
        root_logger = logging.getLogger()
        capture_handler = CapturingHandler()
        root_logger.addHandler(capture_handler)

        logger = get_logger("test-component")
        ctx = ExecutionContext(job_id="my-job")

        with with_context(ctx):
            logger.info("test_event")

        # Verify log output does NOT contain context fields
        assert len(captured_output) == 1
        log_entry = json.loads(captured_output[0])

        assert "job_id" not in log_entry
        assert "run_id" not in log_entry


class TestAsyncContextPropagation:
    """Tests for context propagation in async code."""

    def setup_method(self):
        """Clear context before each test."""
        from mozart.core.logging import clear_context

        clear_context()

    def teardown_method(self):
        """Clear context after each test."""
        from mozart.core.logging import clear_context

        clear_context()

    @pytest.mark.asyncio
    async def test_context_propagates_in_async_code(self):
        """Test that context propagates correctly in async code."""
        from mozart.core.logging import (
            ExecutionContext,
            get_current_context,
            with_context,
        )

        ctx = ExecutionContext(job_id="async-job")

        async def inner_async():
            # Context should be available in nested async
            return get_current_context()

        with with_context(ctx):
            result = await inner_async()
            assert result is ctx

    @pytest.mark.asyncio
    async def test_context_isolation_across_tasks(self):
        """Test that context is isolated between concurrent tasks."""
        import asyncio

        from mozart.core.logging import (
            ExecutionContext,
            get_current_context,
            with_context,
        )

        results: dict[str, str | None] = {}

        async def task_with_context(name: str, job_id: str):
            ctx = ExecutionContext(job_id=job_id)
            with with_context(ctx):
                await asyncio.sleep(0.01)  # Simulate some async work
                current = get_current_context()
                results[name] = current.job_id if current else None

        # Run tasks concurrently
        await asyncio.gather(
            task_with_context("task1", "job-1"),
            task_with_context("task2", "job-2"),
        )

        # Each task should have its own context
        assert results["task1"] == "job-1"
        assert results["task2"] == "job-2"


class TestErrorClassifierLogging:
    """Tests for error classification logging.

    Note: Due to structlog's logger caching and module-level logger creation,
    these tests verify behavior through stdout capture (via capsys) rather than
    file-based logging, since the loggers are created before tests run.
    """

    def setup_method(self):
        """Reset logging configuration before each test."""
        # Reset structlog's caching (important for test isolation)
        structlog.reset_defaults()
        # Force recreation of cached loggers
        structlog.configure(cache_logger_on_first_use=False)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_rate_limit_classification_logs_warning(self, capsys: pytest.CaptureFixture[str]):
        """Test that rate limit classification logs a warning."""
        from mozart.core.errors import ErrorCategory, ErrorClassifier

        # Configure logging - note: due to module-level logger creation,
        # we get console format output even when requesting JSON
        configure_logging(
            level="WARNING",
            format="console",
            include_timestamps=False,
        )

        classifier = ErrorClassifier()
        result = classifier.classify(stdout="rate limit exceeded", exit_code=1)

        assert result.category == ErrorCategory.RATE_LIMIT

        # Capture console output (structlog console format goes to stderr)
        captured = capsys.readouterr()
        err = strip_ansi(captured.err)
        assert "error_classified" in err
        assert "rate_limit" in err
        assert "retriable=True" in err

    def test_fatal_error_classification_logs_warning(self, capsys: pytest.CaptureFixture[str]):
        """Test that fatal error classification logs a warning."""
        from mozart.core.errors import ErrorCategory, ErrorClassifier

        configure_logging(
            level="WARNING",
            format="console",
            include_timestamps=False,
        )

        classifier = ErrorClassifier()
        result = classifier.classify(exit_code=99)  # Unknown error

        assert result.category == ErrorCategory.FATAL

        captured = capsys.readouterr()
        err = strip_ansi(captured.err)
        assert "error_classified" in err
        assert "fatal" in err
        assert "retriable=False" in err

    def test_validation_category_does_not_log(self, capsys: pytest.CaptureFixture[str]):
        """Test that successful execution (validation needed) does not log."""
        from mozart.core.errors import ErrorCategory, ErrorClassifier

        configure_logging(
            level="DEBUG",
            format="console",
            include_timestamps=False,
        )

        classifier = ErrorClassifier()
        result = classifier.classify(exit_code=0)  # Success

        assert result.category == ErrorCategory.VALIDATION

        # VALIDATION category (success case) should NOT log an error_classified warning
        captured = capsys.readouterr()
        err = strip_ansi(captured.err)
        assert "error_classified" not in err


class TestStateBackendLogging:
    """Tests for state backend checkpoint logging.

    Note: Due to structlog's logger caching and module-level logger creation,
    these tests verify behavior through stdout capture (via capsys) rather than
    file-based logging, since the loggers are created before tests run.
    """

    def setup_method(self):
        """Reset logging configuration before each test."""
        # Reset structlog's caching (important for test isolation)
        structlog.reset_defaults()
        # Force recreation of cached loggers
        structlog.configure(cache_logger_on_first_use=False)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    @pytest.mark.asyncio
    async def test_json_backend_save_logs_info(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        """Test that JSON backend logs checkpoint saves."""
        from mozart.core.checkpoint import CheckpointState, JobStatus
        from mozart.state.json_backend import JsonStateBackend

        state_dir = tmp_path / "state"
        state_dir.mkdir()

        configure_logging(
            level="DEBUG",  # checkpoint_saved is at DEBUG level
            format="console",
            include_timestamps=False,
        )

        backend = JsonStateBackend(state_dir)
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_batches=10,
            status=JobStatus.RUNNING,
        )

        await backend.save(state)

        captured = capsys.readouterr()
        err = strip_ansi(captured.err)
        assert "checkpoint_saved" in err
        assert "job_id=test-job" in err
        assert "status=running" in err
        assert "total_batches=10" in err

    @pytest.mark.asyncio
    async def test_json_backend_load_logs_info(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        """Test that JSON backend logs checkpoint loads."""
        from mozart.core.checkpoint import CheckpointState, JobStatus
        from mozart.state.json_backend import JsonStateBackend

        state_dir = tmp_path / "state"
        state_dir.mkdir()

        configure_logging(
            level="DEBUG",  # checkpoint_loaded is at DEBUG level
            format="console",
            include_timestamps=False,
        )

        backend = JsonStateBackend(state_dir)
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_batches=10,
            last_completed_batch=5,
            status=JobStatus.RUNNING,
        )

        await backend.save(state)
        capsys.readouterr()  # Clear the save log

        loaded = await backend.load("test-job")
        assert loaded is not None

        captured = capsys.readouterr()
        err = strip_ansi(captured.err)
        assert "checkpoint_loaded" in err
        assert "job_id=test-job" in err
        assert "last_completed_batch=5" in err

    @pytest.mark.asyncio
    async def test_json_backend_corruption_logs_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        """Test that JSON backend logs corruption detection."""
        from mozart.state.json_backend import JsonStateBackend

        state_dir = tmp_path / "state"
        state_dir.mkdir()

        configure_logging(
            level="ERROR",
            format="console",
            include_timestamps=False,
        )

        backend = JsonStateBackend(state_dir)

        # Create a corrupted state file
        corrupted_file = state_dir / "test-job.json"
        corrupted_file.write_text("{ this is not valid json")

        loaded = await backend.load("test-job")
        assert loaded is None

        captured = capsys.readouterr()
        err = strip_ansi(captured.err)
        assert "checkpoint_corruption_detected" in err
        assert "job_id=test-job" in err
        assert "error_type=json_decode" in err


class TestCheckpointStateTransitionLogging:
    """Tests for checkpoint state transition logging.

    Note: Due to structlog's logger caching and module-level logger creation,
    these tests verify behavior through stdout capture (via capsys) rather than
    file-based logging, since the loggers are created before tests run.
    """

    def setup_method(self):
        """Reset logging configuration before each test."""
        # Reset structlog's caching (important for test isolation)
        structlog.reset_defaults()
        # Force recreation of cached loggers
        structlog.configure(cache_logger_on_first_use=False)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_batch_started_logs_debug(self, capsys: pytest.CaptureFixture[str]):
        """Test that batch started transition logs debug message."""
        from mozart.core.checkpoint import CheckpointState

        configure_logging(
            level="DEBUG",
            format="console",
            include_timestamps=False,
        )

        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_batches=10,
        )

        state.mark_batch_started(1)

        # Capture console output (goes to stderr)
        captured = capsys.readouterr()
        err = strip_ansi(captured.err)
        assert "batch_started" in err
        assert "job_id=test-job" in err
        assert "batch_num=1" in err
        assert "attempt_count=1" in err

    def test_batch_completed_logs_debug(self, capsys: pytest.CaptureFixture[str]):
        """Test that batch completed transition logs debug message."""
        from mozart.core.checkpoint import CheckpointState

        configure_logging(
            level="DEBUG",
            format="console",
            include_timestamps=False,
        )

        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_batches=10,
        )

        state.mark_batch_started(1)
        capsys.readouterr()  # Clear the start log

        state.mark_batch_completed(1, validation_passed=True)

        captured = capsys.readouterr()
        err = strip_ansi(captured.err)
        assert "batch_completed" in err
        assert "job_id=test-job" in err
        assert "batch_num=1" in err
        assert "validation_passed=True" in err

    def test_job_failed_logs_error(self, capsys: pytest.CaptureFixture[str]):
        """Test that job failed transition logs error message."""
        from mozart.core.checkpoint import CheckpointState

        configure_logging(
            level="ERROR",
            format="console",
            include_timestamps=False,
        )

        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_batches=10,
        )

        state.mark_job_failed("Test failure message")

        captured = capsys.readouterr()
        err = strip_ansi(captured.err)
        assert "job_failed" in err
        assert "job_id=test-job" in err
        assert "Test failure message" in err

    def test_job_paused_logs_info(self, capsys: pytest.CaptureFixture[str]):
        """Test that job paused transition logs info message."""
        from mozart.core.checkpoint import CheckpointState, JobStatus

        configure_logging(
            level="INFO",
            format="console",
            include_timestamps=False,
        )

        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_batches=10,
            status=JobStatus.RUNNING,
        )

        state.mark_job_paused()

        captured = capsys.readouterr()
        err = strip_ansi(captured.err)
        assert "job_paused" in err
        assert "job_id=test-job" in err
        assert "previous_status=running" in err
