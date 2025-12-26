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
    import mozart.cli as cli_module

    # Store original state
    original_configured = cli_module._logging_configured
    original_log_level = cli_module._log_level
    original_log_file = cli_module._log_file
    original_log_format = cli_module._log_format

    # Reset state before test
    cli_module._logging_configured = False
    cli_module._log_level = "INFO"
    cli_module._log_file = None
    cli_module._log_format = "console"

    # Reset structlog to default state
    structlog.reset_defaults()

    # Clear all handlers from root logger
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    for handler in original_handlers:
        root_logger.removeHandler(handler)

    yield

    # Restore state after test
    cli_module._logging_configured = original_configured
    cli_module._log_level = original_log_level
    cli_module._log_file = original_log_file
    cli_module._log_format = original_log_format

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
            "template": "Process batch {{ sheet_num }} of {{ total_sheets }}.",
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
