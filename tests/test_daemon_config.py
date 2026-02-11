"""Tests for mozart.daemon.config module."""

from pathlib import Path

import pytest
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
        assert config.path == Path("/tmp/mozartd.sock")
        assert config.permissions == 0o660
        assert config.backlog == 5

    def test_custom_path(self):
        """Test custom socket path."""
        config = SocketConfig(path=Path("/run/user/1000/mozartd.sock"))
        assert config.path == Path("/run/user/1000/mozartd.sock")

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
        assert config.pid_file == Path("/tmp/mozartd.pid")
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
        assert config.socket.path == Path("/tmp/mozartd.sock")

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

    def test_state_backend_type_json(self):
        """Test state_backend_type accepts json."""
        config = DaemonConfig(state_backend_type="json")
        assert config.state_backend_type == "json"

    def test_state_backend_type_sqlite(self):
        """Test state_backend_type accepts sqlite."""
        config = DaemonConfig(state_backend_type="sqlite")
        assert config.state_backend_type == "sqlite"

    def test_state_backend_type_invalid_rejected(self):
        """Test invalid state_backend_type is rejected."""
        with pytest.raises(ValidationError):
            DaemonConfig(state_backend_type="redis")

    def test_custom_pid_file(self):
        """Test custom PID file path."""
        config = DaemonConfig(pid_file=Path("/run/mozartd.pid"))
        assert config.pid_file == Path("/run/mozartd.pid")

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
        config = DaemonConfig(log_file=Path("/var/log/mozartd.log"))
        assert config.log_file == Path("/var/log/mozartd.log")

    def test_full_custom_config(self):
        """Test fully customized daemon config."""
        config = DaemonConfig(
            socket=SocketConfig(
                path=Path("/run/user/1000/mozartd.sock"),
                permissions=0o600,
                backlog=10,
            ),
            pid_file=Path("/run/mozartd.pid"),
            max_concurrent_jobs=10,
            max_concurrent_sheets=20,
            resource_limits=ResourceLimitConfig(
                max_memory_mb=4096,
                max_processes=25,
                max_api_calls_per_minute=120,
            ),
            state_backend_type="json",
            state_db_path=Path("/data/mozart.db"),
            log_level="debug",
            log_file=Path("/var/log/mozartd.log"),
        )
        assert config.socket.path == Path("/run/user/1000/mozartd.sock")
        assert config.socket.permissions == 0o600
        assert config.socket.backlog == 10
        assert config.pid_file == Path("/run/mozartd.pid")
        assert config.max_concurrent_jobs == 10
        assert config.max_concurrent_sheets == 20
        assert config.resource_limits.max_memory_mb == 4096
        assert config.resource_limits.max_processes == 25
        assert config.resource_limits.max_api_calls_per_minute == 120
        assert config.state_backend_type == "json"
        assert config.state_db_path == Path("/data/mozart.db")
        assert config.log_level == "debug"
        assert config.log_file == Path("/var/log/mozartd.log")

    def test_serialization_roundtrip(self):
        """Test config survives model_dump -> model_validate roundtrip."""
        original = DaemonConfig(
            max_concurrent_jobs=8,
            log_level="warning",
            state_backend_type="json",
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
