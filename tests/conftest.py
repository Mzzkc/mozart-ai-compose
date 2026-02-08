"""Pytest fixtures for Mozart tests."""

import logging
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import structlog


@pytest.fixture(autouse=True)
def reset_logging_state() -> Generator[None, None, None]:
    """Reset logging state before and after each test.

    This ensures test isolation for logging configuration.
    """
    # Import helpers module directly for access to internal state
    from mozart.cli import helpers as cli_helpers

    # Store original state from the CliLoggingConfig dataclass
    cfg = cli_helpers._log_config
    original_configured = cfg.configured
    original_log_level = cfg.level
    original_log_file = cfg.file
    original_log_format = cfg.format

    # Reset state before test
    cfg.configured = False
    cfg.level = "INFO"
    cfg.file = None
    cfg.format = "console"

    # Reset structlog to default state
    structlog.reset_defaults()

    # Clear all handlers from root logger
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    for handler in original_handlers:
        root_logger.removeHandler(handler)

    yield

    # Restore state after test
    cfg.configured = original_configured
    cfg.level = original_log_level
    cfg.file = original_log_file
    cfg.format = original_log_format

    # Restore original handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    for handler in original_handlers:
        root_logger.addHandler(handler)


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def sample_config_dict() -> dict:
    """Return a sample job configuration dictionary."""
    return {
        "name": "test-job",
        "description": "Test job for unit tests",
        "backend": {
            "type": "claude_cli",
            "skip_permissions": True,
        },
        "sheet": {
            "size": 10,
            "total_items": 30,
        },
        "prompt": {
            "template": "Process sheet {{ sheet_num }} of {{ total_sheets }}.",
        },
        "retry": {
            "max_retries": 2,
        },
        "validations": [
            {
                "type": "file_exists",
                "path": "{workspace}/output-{sheet_num}.txt",
                "description": "Output file exists",
            },
        ],
    }


@pytest.fixture
def sample_yaml_config(tmp_path: Path, sample_config_dict: dict) -> Path:
    """Create a sample YAML config file."""
    import yaml

    config_path = tmp_path / "test-config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path
