"""Tests for mozart.daemon.config module."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mozart.daemon.config import DaemonConfig, ResourceLimitConfig, SocketConfig


class TestResourceLimitConfig:
    """Tests for ResourceLimitConfig model."""

    def test_defaults(self):
        """Test default values are applied."""
        config = ResourceLimitConfig()
        assert config.max_memory_mb == 8192
        assert config.max_processes == 50
        assert config.max_api_calls_per_minute == 60

    def test_custom_values(self):
        """Test custom values override defaults."""
        config = ResourceLimitConfig(
            max_memory_mb=1024,
            max_processes=10,
            max_api_calls_per_minute=30,
        )
        assert config.max_memory_mb == 1024
        assert config.max_processes == 10
        assert config.max_api_calls_per_minute == 30

    def test_max_memory_mb_minimum(self):
        """Test max_memory_mb must be >= 512."""
        config = ResourceLimitConfig(max_memory_mb=512)
        assert config.max_memory_mb == 512

        with pytest.raises(ValidationError):
            ResourceLimitConfig(max_memory_mb=511)

    def test_max_memory_mb_below_minimum_rejected(self):
        """Test max_memory_mb below 512 is rejected."""
        with pytest.raises(ValidationError):
            ResourceLimitConfig(max_memory_mb=0)

    def test_max_processes_minimum(self):
        """Test max_processes must be >= 5."""
        config = ResourceLimitConfig(max_processes=5)
        assert config.max_processes == 5

        with pytest.raises(ValidationError):
            ResourceLimitConfig(max_processes=4)

    def test_max_processes_below_minimum_rejected(self):
        """Test max_processes below 5 is rejected."""
        with pytest.raises(ValidationError):
            ResourceLimitConfig(max_processes=0)

    def test_max_api_calls_minimum(self):
        """Test max_api_calls_per_minute must be >= 1."""
        config = ResourceLimitConfig(max_api_calls_per_minute=1)
        assert config.max_api_calls_per_minute == 1

        with pytest.raises(ValidationError):
            ResourceLimitConfig(max_api_calls_per_minute=0)

    def test_max_api_calls_negative_rejected(self):
        """Test negative max_api_calls_per_minute is rejected."""
        with pytest.raises(ValidationError):
            ResourceLimitConfig(max_api_calls_per_minute=-1)


class TestSocketConfig:
    """Tests for SocketConfig model."""

    def test_defaults(self):
        """Test default values are applied."""
        config = SocketConfig()
        assert config.path == Path("/tmp/mozart.sock")
        assert config.permissions == 0o660
        assert config.backlog == 5

    def test_custom_path(self):
        """Test custom socket path."""
        config = SocketConfig(path=Path("/run/user/1000/mozart.sock"))
        assert config.path == Path("/run/user/1000/mozart.sock")

    def test_custom_permissions(self):
        """Test custom socket permissions."""
        config = SocketConfig(permissions=0o600)
        assert config.permissions == 0o600

    def test_backlog_minimum(self):
        """Test backlog must be >= 1."""
        config = SocketConfig(backlog=1)
        assert config.backlog == 1

        with pytest.raises(ValidationError):
            SocketConfig(backlog=0)

    def test_backlog_negative_rejected(self):
        """Test negative backlog is rejected."""
        with pytest.raises(ValidationError):
            SocketConfig(backlog=-1)

    def test_custom_backlog(self):
        """Test custom backlog value."""
        config = SocketConfig(backlog=128)
        assert config.backlog == 128


class TestDaemonConfig:
    """Tests for DaemonConfig model."""

    def test_defaults(self):
        """Test default values are applied."""
        config = DaemonConfig()
        assert config.pid_file == Path("/tmp/mozart.pid")
        assert config.max_concurrent_jobs == 5
        assert config.max_concurrent_sheets == 10
        assert config.state_backend_type == "sqlite"
        assert config.state_db_path == Path("~/.mozart/daemon-state.db")
        assert config.log_level == "info"
        assert config.log_file is None

    def test_nested_socket_defaults(self):
        """Test socket config is initialized with defaults."""
        config = DaemonConfig()
        assert isinstance(config.socket, SocketConfig)
        assert config.socket.path == Path("/tmp/mozart.sock")

    def test_nested_resource_limits_defaults(self):
        """Test resource limits config is initialized with defaults."""
        config = DaemonConfig()
        assert isinstance(config.resource_limits, ResourceLimitConfig)
        assert config.resource_limits.max_memory_mb == 8192

    def test_max_concurrent_jobs_minimum(self):
        """Test max_concurrent_jobs must be >= 1."""
        config = DaemonConfig(max_concurrent_jobs=1)
        assert config.max_concurrent_jobs == 1

        with pytest.raises(ValidationError):
            DaemonConfig(max_concurrent_jobs=0)

    def test_max_concurrent_jobs_maximum(self):
        """Test max_concurrent_jobs must be <= 50."""
        config = DaemonConfig(max_concurrent_jobs=50)
        assert config.max_concurrent_jobs == 50

        with pytest.raises(ValidationError):
            DaemonConfig(max_concurrent_jobs=51)

    def test_max_concurrent_sheets_minimum(self):
        """Test max_concurrent_sheets must be >= 1."""
        config = DaemonConfig(max_concurrent_sheets=1)
        assert config.max_concurrent_sheets == 1

        with pytest.raises(ValidationError):
            DaemonConfig(max_concurrent_sheets=0)

    def test_max_concurrent_sheets_maximum(self):
        """Test max_concurrent_sheets must be <= 100."""
        config = DaemonConfig(max_concurrent_sheets=100)
        assert config.max_concurrent_sheets == 100

        with pytest.raises(ValidationError):
            DaemonConfig(max_concurrent_sheets=101)

    def test_state_backend_type_sqlite(self):
        """Test state_backend_type accepts sqlite (the only valid value)."""
        config = DaemonConfig(state_backend_type="sqlite")
        assert config.state_backend_type == "sqlite"

    def test_state_backend_type_json_rejected(self):
        """Test state_backend_type='json' is rejected (reserved, frozen to sqlite)."""
        with pytest.raises(ValidationError):
            DaemonConfig(state_backend_type="json")

    def test_state_backend_type_invalid_rejected(self):
        """Test invalid state_backend_type is rejected."""
        with pytest.raises(ValidationError):
            DaemonConfig(state_backend_type="redis")

    def test_custom_pid_file(self):
        """Test custom PID file path."""
        config = DaemonConfig(pid_file=Path("/run/mozart.pid"))
        assert config.pid_file == Path("/run/mozart.pid")

    def test_custom_state_db_path(self):
        """Test custom state DB path."""
        config = DaemonConfig(state_db_path=Path("/var/lib/mozart/state.db"))
        assert config.state_db_path == Path("/var/lib/mozart/state.db")

    def test_custom_log_level(self):
        """Test custom log level."""
        config = DaemonConfig(log_level="debug")
        assert config.log_level == "debug"

    def test_custom_log_file(self):
        """Test custom log file path."""
        config = DaemonConfig(log_file=Path("/var/log/mozart.log"))
        assert config.log_file == Path("/var/log/mozart.log")

    def test_full_custom_config(self):
        """Test fully customized daemon config."""
        config = DaemonConfig(
            socket=SocketConfig(
                path=Path("/run/user/1000/mozart.sock"),
                permissions=0o600,
                backlog=10,
            ),
            pid_file=Path("/run/mozart.pid"),
            max_concurrent_jobs=10,
            max_concurrent_sheets=20,
            resource_limits=ResourceLimitConfig(
                max_memory_mb=4096,
                max_processes=25,
                max_api_calls_per_minute=120,
            ),
            state_backend_type="sqlite",
            state_db_path=Path("/data/mozart.db"),
            log_level="debug",
            log_file=Path("/var/log/mozart.log"),
        )
        assert config.socket.path == Path("/run/user/1000/mozart.sock")
        assert config.socket.permissions == 0o600
        assert config.socket.backlog == 10
        assert config.pid_file == Path("/run/mozart.pid")
        assert config.max_concurrent_jobs == 10
        assert config.max_concurrent_sheets == 20
        assert config.resource_limits.max_memory_mb == 4096
        assert config.resource_limits.max_processes == 25
        assert config.resource_limits.max_api_calls_per_minute == 120
        assert config.state_backend_type == "sqlite"
        assert config.state_db_path == Path("/data/mozart.db")
        assert config.log_level == "debug"
        assert config.log_file == Path("/var/log/mozart.log")

    def test_serialization_roundtrip(self):
        """Test config survives model_dump -> model_validate roundtrip."""
        original = DaemonConfig(
            max_concurrent_jobs=8,
            log_level="warning",
            state_backend_type="sqlite",
        )
        dumped = original.model_dump()
        restored = DaemonConfig.model_validate(dumped)
        assert restored.max_concurrent_jobs == original.max_concurrent_jobs
        assert restored.log_level == original.log_level
        assert restored.state_backend_type == original.state_backend_type

    def test_nested_dict_construction(self):
        """Test constructing DaemonConfig from nested dicts (YAML-style)."""
        config = DaemonConfig.model_validate({
            "socket": {
                "path": "/tmp/custom.sock",
                "backlog": 10,
            },
            "max_concurrent_jobs": 3,
            "resource_limits": {
                "max_memory_mb": 2048,
            },
        })
        assert config.socket.path == Path("/tmp/custom.sock")
        assert config.socket.backlog == 10
        assert config.max_concurrent_jobs == 3
        assert config.resource_limits.max_memory_mb == 2048
        # Non-specified fields should get defaults
        assert config.resource_limits.max_processes == 50

    def test_config_file_field_accepts_path(self):
        """config_file can be set to a Path without triggering warnings."""
        config = DaemonConfig(config_file=Path("/tmp/test.yaml"))
        assert config.config_file == Path("/tmp/test.yaml")

    def test_config_file_default_is_none(self):
        """config_file defaults to None."""
        config = DaemonConfig()
        assert config.config_file is None


class TestLoadConfig:
    """Tests for _load_config() helper."""

    def test_load_from_yaml_sets_config_file(self, tmp_path: Path):
        """_load_config sets config_file to the resolved path of the loaded file."""
        from mozart.daemon.process import _load_config

        cfg_path = tmp_path / "daemon.yaml"
        cfg_path.write_text(yaml.dump({"max_concurrent_jobs": 3}))

        config = _load_config(cfg_path)
        assert config.config_file == cfg_path.resolve()
        assert config.max_concurrent_jobs == 3

    def test_load_with_none_returns_defaults(self):
        """_load_config(None) returns default config with config_file=None."""
        from mozart.daemon.process import _load_config

        config = _load_config(None)
        assert config.config_file is None
        assert config.max_concurrent_jobs == 5  # default

    def test_load_nonexistent_file_returns_defaults(self, tmp_path: Path):
        """_load_config with non-existent file returns defaults."""
        from mozart.daemon.process import _load_config

        config = _load_config(tmp_path / "nope.yaml")
        assert config.config_file is None


class TestJobManagerApplyConfig:
    """Tests for JobManager.apply_config() hot-reload method."""

    def test_apply_config_rebuilds_semaphore_on_change(self):
        """apply_config creates a new semaphore when max_concurrent_jobs changes."""
        from mozart.daemon.manager import JobManager

        config = DaemonConfig(max_concurrent_jobs=5)
        manager = JobManager(config)
        old_sem = manager._concurrency_semaphore

        new_config = DaemonConfig(max_concurrent_jobs=10)
        manager.apply_config(new_config)

        assert manager._concurrency_semaphore is not old_sem
        assert manager._config.max_concurrent_jobs == 10

    def test_apply_config_no_semaphore_change_when_unchanged(self):
        """apply_config keeps the same semaphore when max_concurrent_jobs is unchanged."""
        from mozart.daemon.manager import JobManager

        config = DaemonConfig(max_concurrent_jobs=5)
        manager = JobManager(config)
        old_sem = manager._concurrency_semaphore

        new_config = DaemonConfig(max_concurrent_jobs=5)
        manager.apply_config(new_config)

        assert manager._concurrency_semaphore is old_sem

    def test_apply_config_updates_config_reference(self):
        """apply_config replaces the _config reference."""
        from mozart.daemon.manager import JobManager

        old_config = DaemonConfig(job_timeout_seconds=3600.0)
        manager = JobManager(old_config)

        new_config = DaemonConfig(job_timeout_seconds=7200.0)
        manager.apply_config(new_config)

        assert manager._config is new_config
        assert manager._config.job_timeout_seconds == 7200.0

    def test_apply_config_noop_when_identical(self):
        """apply_config with identical config does not rebuild semaphore."""
        from mozart.daemon.manager import JobManager

        config = DaemonConfig()
        manager = JobManager(config)
        old_sem = manager._concurrency_semaphore

        identical_config = DaemonConfig()
        manager.apply_config(identical_config)

        assert manager._concurrency_semaphore is old_sem
        assert manager._config is identical_config


class TestResourceMonitorUpdateLimits:
    """Tests for ResourceMonitor.update_limits() hot-reload method."""

    def test_update_limits_replaces_config(self):
        """update_limits replaces the internal config reference."""
        from mozart.daemon.monitor import ResourceMonitor

        old_limits = ResourceLimitConfig(max_memory_mb=4096)
        monitor = ResourceMonitor(old_limits)

        new_limits = ResourceLimitConfig(max_memory_mb=8192)
        monitor.update_limits(new_limits)

        assert monitor._config is new_limits
        assert monitor._config.max_memory_mb == 8192

    def test_update_limits_preserves_other_state(self):
        """update_limits does not affect degraded/failure state."""
        from mozart.daemon.monitor import ResourceMonitor

        limits = ResourceLimitConfig()
        monitor = ResourceMonitor(limits)
        monitor._degraded = True
        monitor._consecutive_failures = 3

        new_limits = ResourceLimitConfig(max_memory_mb=2048)
        monitor.update_limits(new_limits)

        # Internal state untouched
        assert monitor._degraded is True
        assert monitor._consecutive_failures == 3
        assert monitor._config is new_limits
