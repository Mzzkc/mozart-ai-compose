"""Tests for ObserverRecorder and ObserverConfig persistence fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mozart.daemon.config import ObserverConfig


class TestObserverConfigPersistence:
    """Verify new persistence fields on ObserverConfig."""

    def test_defaults(self) -> None:
        config = ObserverConfig()
        assert config.persist_events is True
        assert ".git/" in config.exclude_patterns
        assert "__pycache__/" in config.exclude_patterns
        assert "node_modules/" in config.exclude_patterns
        assert ".venv/" in config.exclude_patterns
        assert "*.pyc" in config.exclude_patterns
        assert config.coalesce_window_seconds == 2.0
        assert config.max_timeline_bytes == 10_485_760

    def test_disable_persistence(self) -> None:
        config = ObserverConfig(persist_events=False)
        assert config.persist_events is False

    def test_custom_exclude_patterns(self) -> None:
        config = ObserverConfig(exclude_patterns=[".git/", "build/"])
        assert config.exclude_patterns == [".git/", "build/"]

    def test_coalesce_window_minimum(self) -> None:
        config = ObserverConfig(coalesce_window_seconds=0.0)
        assert config.coalesce_window_seconds == 0.0

    def test_max_timeline_bytes_minimum(self) -> None:
        with pytest.raises(ValidationError):
            ObserverConfig(max_timeline_bytes=100)  # Below 4KB minimum
