"""Tests for ObserverRecorder and ObserverConfig persistence fields."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from mozart.daemon.config import ObserverConfig
from mozart.daemon.observer_recorder import ObserverRecorder


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


class TestPathExclusion:
    """Verify path exclusion filtering."""

    def _make_recorder(self, **config_kwargs: object) -> ObserverRecorder:
        config = ObserverConfig(**config_kwargs)
        bus = AsyncMock()
        bus.subscribe = lambda callback, event_filter=None: "sub-id"
        return ObserverRecorder(config=config, event_bus=bus)

    def test_default_excludes_git(self) -> None:
        recorder = self._make_recorder()
        assert recorder._should_exclude(".git/objects/abc123")

    def test_default_excludes_pycache(self) -> None:
        recorder = self._make_recorder()
        assert recorder._should_exclude("src/__pycache__/foo.cpython-312.pyc")

    def test_default_excludes_node_modules(self) -> None:
        recorder = self._make_recorder()
        assert recorder._should_exclude("node_modules/lodash/index.js")

    def test_default_excludes_venv(self) -> None:
        recorder = self._make_recorder()
        assert recorder._should_exclude(".venv/lib/python3.12/site.py")

    def test_default_excludes_pyc(self) -> None:
        recorder = self._make_recorder()
        assert recorder._should_exclude("src/foo.pyc")

    def test_allows_normal_paths(self) -> None:
        recorder = self._make_recorder()
        assert not recorder._should_exclude("src/main.py")
        assert not recorder._should_exclude("output-3.md")
        assert not recorder._should_exclude("tests/test_foo.py")

    def test_custom_patterns(self) -> None:
        recorder = self._make_recorder(exclude_patterns=["build/", "*.tmp"])
        assert recorder._should_exclude("build/output.js")
        assert recorder._should_exclude("data.tmp")
        assert not recorder._should_exclude(".git/HEAD")  # Not in custom list

    def test_empty_patterns_excludes_nothing(self) -> None:
        recorder = self._make_recorder(exclude_patterns=[])
        assert not recorder._should_exclude(".git/HEAD")
        assert not recorder._should_exclude("src/__pycache__/foo.pyc")
