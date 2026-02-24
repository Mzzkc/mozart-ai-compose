"""Tests for ObserverRecorder and ObserverConfig persistence fields."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from mozart.daemon.config import ObserverConfig
from mozart.daemon.observer_recorder import ObserverRecorder
from mozart.daemon.types import ObserverEvent


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


class TestJSONLPersistence:
    """Verify JSONL write, register/unregister lifecycle."""

    def _make_recorder(self, **config_kwargs: object) -> ObserverRecorder:
        config = ObserverConfig(**config_kwargs)
        bus = AsyncMock()
        bus.subscribe = lambda callback, event_filter=None: "sub-id"
        return ObserverRecorder(config=config, event_bus=bus)

    def test_register_creates_state(self, tmp_path: Path) -> None:
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        assert "job-1" in recorder._jobs

    def test_register_opens_jsonl(self, tmp_path: Path) -> None:
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        assert jsonl_path.exists()

    def test_unregister_closes_file(self, tmp_path: Path) -> None:
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        recorder.unregister_job("job-1")
        assert "job-1" not in recorder._jobs

    def test_write_event_produces_jsonl_line(self, tmp_path: Path) -> None:
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        event: ObserverEvent = {
            "job_id": "job-1",
            "sheet_num": 0,
            "event": "observer.file_created",
            "data": {"path": "output.md"},
            "timestamp": time.time(),
        }
        recorder._write_event("job-1", event)
        recorder.flush("job-1")

        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["event"] == "observer.file_created"

    def test_excluded_path_not_written(self, tmp_path: Path) -> None:
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        event: ObserverEvent = {
            "job_id": "job-1",
            "sheet_num": 0,
            "event": "observer.file_modified",
            "data": {"path": ".git/objects/abc123"},
            "timestamp": time.time(),
        }
        recorder._write_event("job-1", event)
        recorder.flush("job-1")

        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        content = jsonl_path.read_text().strip()
        assert content == ""

    def test_unregister_unknown_job_is_noop(self) -> None:
        recorder = self._make_recorder()
        recorder.unregister_job("nonexistent")  # Should not raise

    def test_disabled_persistence_skips_file(self, tmp_path: Path) -> None:
        recorder = self._make_recorder(persist_events=False)
        recorder.register_job("job-1", tmp_path)
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        assert not jsonl_path.exists()

    # --- Expert review edge case tests ---

    def test_register_twice_is_idempotent(self, tmp_path: Path) -> None:
        """Registering the same job_id twice should not create a second state."""
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        state_before = recorder._jobs["job-1"]
        recorder.register_job("job-1", tmp_path)
        assert recorder._jobs["job-1"] is state_before

    def test_flush_after_unregister_is_noop(self, tmp_path: Path) -> None:
        """Flush on a job that was already unregistered should not raise."""
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        recorder.unregister_job("job-1")
        recorder.flush("job-1")  # Should not raise

    def test_event_with_none_data_does_not_crash(self, tmp_path: Path) -> None:
        """An event with data=None should be written without error."""
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        event: ObserverEvent = {
            "job_id": "job-1",
            "sheet_num": 0,
            "event": "observer.process_spawned",
            "data": None,
            "timestamp": time.time(),
        }
        recorder._write_event("job-1", event)
        recorder.flush("job-1")

        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["data"] is None

    def test_event_for_unregistered_job_is_ignored(self) -> None:
        """Writing an event for a job that was never registered is a no-op."""
        recorder = self._make_recorder()
        event: ObserverEvent = {
            "job_id": "job-unknown",
            "sheet_num": 0,
            "event": "observer.file_created",
            "data": {"path": "foo.py"},
            "timestamp": time.time(),
        }
        recorder._write_event("job-unknown", event)  # Should not raise

    def test_file_write_failure_still_populates_ring_buffer(
        self, tmp_path: Path
    ) -> None:
        """Ring buffer MUST receive events even when file I/O fails."""
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        # Sabotage the file handle to simulate write failure
        state = recorder._jobs["job-1"]
        state.file_handle.close()
        state.file_handle = _BrokenWriter()

        event: ObserverEvent = {
            "job_id": "job-1",
            "sheet_num": 0,
            "event": "observer.file_created",
            "data": {"path": "output.md"},
            "timestamp": time.time(),
        }
        recorder._write_event("job-1", event)

        # Ring buffer should still have the event
        assert len(state.recent_events) == 1
        assert state.recent_events[0]["event"] == "observer.file_created"

    def test_excluded_event_not_in_ring_buffer(self, tmp_path: Path) -> None:
        """Excluded path events should not appear in the ring buffer either."""
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        event: ObserverEvent = {
            "job_id": "job-1",
            "sheet_num": 0,
            "event": "observer.file_modified",
            "data": {"path": ".git/HEAD"},
            "timestamp": time.time(),
        }
        recorder._write_event("job-1", event)
        state = recorder._jobs["job-1"]
        assert len(state.recent_events) == 0


class TestCoalescing:
    """Verify same-file modification coalescing."""

    def _make_recorder(self, **kw: object) -> ObserverRecorder:
        config = ObserverConfig(**kw)
        bus = AsyncMock()
        bus.subscribe = lambda callback, event_filter=None: "sub-id"
        return ObserverRecorder(config=config, event_bus=bus)

    def test_rapid_same_file_mods_coalesce(self, tmp_path: Path) -> None:
        """Multiple edits to same file within window produce one JSONL line."""
        recorder = self._make_recorder(coalesce_window_seconds=2.0)
        recorder.register_job("job-1", tmp_path)
        now = time.time()
        for i in range(5):
            event: ObserverEvent = {
                "job_id": "job-1",
                "sheet_num": 0,
                "event": "observer.file_modified",
                "data": {"path": "same-file.md"},
                "timestamp": now + i * 0.1,  # 100ms apart, within 2s window
            }
            recorder._handle_event(event)

        recorder.flush("job-1")
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = [ln for ln in jsonl_path.read_text().strip().split("\n") if ln]
        # Should produce 1 coalesced event, not 5
        assert len(lines) == 1

    def test_different_files_not_coalesced(self, tmp_path: Path) -> None:
        """Events for different files are never coalesced."""
        recorder = self._make_recorder(coalesce_window_seconds=2.0)
        recorder.register_job("job-1", tmp_path)
        now = time.time()
        for i in range(3):
            event: ObserverEvent = {
                "job_id": "job-1",
                "sheet_num": 0,
                "event": "observer.file_modified",
                "data": {"path": f"file-{i}.md"},
                "timestamp": now + i * 0.1,
            }
            recorder._handle_event(event)

        recorder.flush("job-1")
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = [ln for ln in jsonl_path.read_text().strip().split("\n") if ln]
        assert len(lines) == 3

    def test_process_events_not_coalesced(self, tmp_path: Path) -> None:
        """Process events always write immediately, never coalesced."""
        recorder = self._make_recorder(coalesce_window_seconds=2.0)
        recorder.register_job("job-1", tmp_path)
        now = time.time()
        for i in range(3):
            event: ObserverEvent = {
                "job_id": "job-1",
                "sheet_num": 0,
                "event": "observer.process_spawned",
                "data": {"pid": 1000 + i, "name": "pytest"},
                "timestamp": now + i * 0.1,
            }
            recorder._handle_event(event)

        recorder.flush("job-1")
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = [ln for ln in jsonl_path.read_text().strip().split("\n") if ln]
        assert len(lines) == 3

    def test_zero_coalesce_window_disables(self, tmp_path: Path) -> None:
        """Setting coalesce window to 0 disables coalescing entirely."""
        recorder = self._make_recorder(coalesce_window_seconds=0.0)
        recorder.register_job("job-1", tmp_path)
        now = time.time()
        for i in range(3):
            event: ObserverEvent = {
                "job_id": "job-1",
                "sheet_num": 0,
                "event": "observer.file_modified",
                "data": {"path": "same-file.md"},
                "timestamp": now + i * 0.1,
            }
            recorder._handle_event(event)

        recorder.flush("job-1")
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = [ln for ln in jsonl_path.read_text().strip().split("\n") if ln]
        assert len(lines) == 3  # No coalescing

    def test_coalescing_uses_event_timestamps(self, tmp_path: Path) -> None:
        """REVIEW FIX 4: Coalescing compares event['timestamp'], not wall clock.

        Events with fabricated timestamps that span beyond the window
        should NOT be coalesced, even if sent in rapid succession.
        """
        recorder = self._make_recorder(coalesce_window_seconds=1.0)
        recorder.register_job("job-1", tmp_path)

        # Event 1: ts=1000.0
        recorder._handle_event({
            "job_id": "job-1",
            "sheet_num": 0,
            "event": "observer.file_modified",
            "data": {"path": "same-file.md"},
            "timestamp": 1000.0,
        })
        # Event 2: ts=1005.0 — 5 seconds later, outside 1s window
        recorder._handle_event({
            "job_id": "job-1",
            "sheet_num": 0,
            "event": "observer.file_modified",
            "data": {"path": "same-file.md"},
            "timestamp": 1005.0,
        })

        recorder.flush("job-1")
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = [ln for ln in jsonl_path.read_text().strip().split("\n") if ln]
        # Both events should be written: first flushed when window expired,
        # second flushed by explicit flush()
        assert len(lines) == 2

    def test_coalesced_events_appear_in_ring_buffer(self, tmp_path: Path) -> None:
        """REVIEW FIX 3: Coalesced events MUST still enter ring buffer immediately."""
        recorder = self._make_recorder(coalesce_window_seconds=2.0)
        recorder.register_job("job-1", tmp_path)
        state = recorder._jobs["job-1"]
        now = time.time()

        for i in range(5):
            recorder._handle_event({
                "job_id": "job-1",
                "sheet_num": 0,
                "event": "observer.file_modified",
                "data": {"path": "active-file.md"},
                "timestamp": now + i * 0.1,
            })

        # Ring buffer should have ALL 5 events, even though JSONL will coalesce
        assert len(state.recent_events) == 5

    def test_file_created_not_coalesced(self, tmp_path: Path) -> None:
        """Only file_modified events are coalesced, not file_created."""
        recorder = self._make_recorder(coalesce_window_seconds=2.0)
        recorder.register_job("job-1", tmp_path)
        now = time.time()
        for i in range(3):
            recorder._handle_event({
                "job_id": "job-1",
                "sheet_num": 0,
                "event": "observer.file_created",
                "data": {"path": "same-file.md"},
                "timestamp": now + i * 0.1,
            })

        recorder.flush("job-1")
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = [ln for ln in jsonl_path.read_text().strip().split("\n") if ln]
        assert len(lines) == 3  # file_created always written


class TestSizeCap:
    """Verify JSONL size cap and truncation."""

    def _make_recorder(self, **kw: object) -> ObserverRecorder:
        config = ObserverConfig(**kw)
        bus = AsyncMock()
        bus.subscribe = lambda callback, event_filter=None: "sub-id"
        return ObserverRecorder(config=config, event_bus=bus)

    @pytest.mark.asyncio
    async def test_truncates_when_over_cap(self, tmp_path: Path) -> None:
        """REVIEW FIX 5: Use max_timeline_bytes=4096 (Pydantic ge=4096)."""
        import asyncio

        recorder = self._make_recorder(
            max_timeline_bytes=4096, coalesce_window_seconds=0.0,
        )
        recorder.register_job("job-1", tmp_path)

        # Write enough events to exceed 4KB
        for i in range(200):
            event: ObserverEvent = {
                "job_id": "job-1",
                "sheet_num": 0,
                "event": "observer.file_created",
                "data": {"path": f"file-{i:04d}.md"},
                "timestamp": time.time(),
            }
            recorder._handle_event(event)

        # REVIEW FIX 2: Truncation is async — give background task time to run
        await asyncio.sleep(0.1)
        recorder.flush("job-1")

        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        size = jsonl_path.stat().st_size
        assert size <= 4096

    @pytest.mark.asyncio
    async def test_surviving_lines_are_valid_json(self, tmp_path: Path) -> None:
        """REVIEW FIX 1+5: Every surviving line after truncation must be valid JSON.

        This catches the corrupt-first-line bug from naive midpoint truncation.
        """
        import asyncio

        recorder = self._make_recorder(
            max_timeline_bytes=4096, coalesce_window_seconds=0.0,
        )
        recorder.register_job("job-1", tmp_path)

        for i in range(200):
            event: ObserverEvent = {
                "job_id": "job-1",
                "sheet_num": 0,
                "event": "observer.file_created",
                "data": {"path": f"file-{i:04d}.md"},
                "timestamp": time.time(),
            }
            recorder._handle_event(event)

        # Let async truncation complete
        await asyncio.sleep(0.1)
        recorder.flush("job-1")

        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = [ln for ln in jsonl_path.read_text().strip().split("\n") if ln]
        assert len(lines) > 0, "Should have surviving events"
        # Every line must parse as valid JSON
        for i, line in enumerate(lines):
            try:
                parsed = json.loads(line)
                assert "event" in parsed, f"Line {i} missing 'event' key"
            except json.JSONDecodeError:
                pytest.fail(f"Line {i} is corrupt JSONL: {line!r}")


class _BrokenWriter:
    """A file-like object that raises OSError on write."""

    def write(self, _data: str) -> int:
        raise OSError("disk full")

    def flush(self) -> None:
        raise OSError("disk full")

    def close(self) -> None:
        pass
