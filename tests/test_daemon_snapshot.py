"""Tests for the SnapshotManager — Phase 4 completion snapshots."""

from __future__ import annotations

import time

import pytest

from mozart.daemon.snapshot import SnapshotManager


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def snapshot_base(tmp_path):
    """Temporary base directory for snapshots."""
    return tmp_path / "snapshots"


@pytest.fixture()
def workspace(tmp_path):
    """Temporary workspace with typical job artifacts."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "my-job.json").write_text('{"status": "completed"}')
    (ws / "mozart.log").write_text("INFO: job completed")
    return ws


@pytest.fixture()
def mgr(snapshot_base):
    """SnapshotManager pointed at temp base dir."""
    return SnapshotManager(base_dir=snapshot_base)


# ─── Capture tests ──────────────────────────────────────────────────


class TestCapture:
    """Tests for SnapshotManager.capture()."""

    def test_capture_creates_snapshot_directory(self, mgr, workspace, snapshot_base):
        result = mgr.capture("test-job", workspace)
        assert result is not None
        assert snapshot_base.exists()
        # Snapshot should be under base/job_id/timestamp/
        job_dir = snapshot_base / "test-job"
        assert job_dir.is_dir()
        ts_dirs = list(job_dir.iterdir())
        assert len(ts_dirs) == 1
        assert ts_dirs[0].is_dir()

    def test_capture_copies_json_files(self, mgr, workspace):
        result = mgr.capture("test-job", workspace)
        from pathlib import Path
        snap = Path(result)
        assert (snap / "my-job.json").exists()
        assert (snap / "my-job.json").read_text() == '{"status": "completed"}'

    def test_capture_copies_log_files(self, mgr, workspace):
        result = mgr.capture("test-job", workspace)
        from pathlib import Path
        snap = Path(result)
        assert (snap / "mozart.log").exists()
        assert (snap / "mozart.log").read_text() == "INFO: job completed"

    def test_capture_returns_snapshot_path_string(self, mgr, workspace):
        result = mgr.capture("test-job", workspace)
        assert isinstance(result, str)
        assert "test-job" in result

    def test_capture_missing_workspace_returns_none(self, mgr, tmp_path):
        result = mgr.capture("test-job", tmp_path / "nonexistent")
        assert result is None

    def test_capture_empty_workspace_returns_none(self, mgr, tmp_path):
        empty_ws = tmp_path / "empty"
        empty_ws.mkdir()
        result = mgr.capture("test-job", empty_ws)
        assert result is None

    def test_capture_multiple_snapshots_for_same_job(self, mgr, workspace, snapshot_base):
        path1 = mgr.capture("test-job", workspace)
        # Ensure different timestamp
        time.sleep(1.1)
        path2 = mgr.capture("test-job", workspace)
        assert path1 != path2
        job_dir = snapshot_base / "test-job"
        assert len(list(job_dir.iterdir())) == 2

    def test_capture_preserves_additional_json_files(self, mgr, workspace):
        (workspace / "extra.json").write_text('{"extra": true}')
        result = mgr.capture("test-job", workspace)
        from pathlib import Path
        snap = Path(result)
        assert (snap / "extra.json").exists()

    def test_capture_ignores_directories_in_workspace(self, mgr, workspace):
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "nested.json").write_text("{}")
        result = mgr.capture("test-job", workspace)
        from pathlib import Path
        snap = Path(result)
        # Only top-level files are captured
        assert not (snap / "subdir").exists()


# ─── Cleanup tests ──────────────────────────────────────────────────


class TestCleanup:
    """Tests for TTL-based snapshot cleanup."""

    def test_cleanup_removes_old_snapshots(self, mgr, snapshot_base):
        # Create a snapshot with an old timestamp
        old_ts = str(int(time.time()) - 8 * 24 * 3600)  # 8 days ago
        old_dir = snapshot_base / "test-job" / old_ts
        old_dir.mkdir(parents=True)
        (old_dir / "state.json").write_text("{}")

        removed = mgr.cleanup(max_age_hours=168)  # 1 week
        assert removed == 1
        assert not old_dir.exists()

    def test_cleanup_preserves_recent_snapshots(self, mgr, workspace, snapshot_base):
        # Capture a fresh snapshot
        mgr.capture("test-job", workspace)
        removed = mgr.cleanup(max_age_hours=168)
        assert removed == 0
        job_dir = snapshot_base / "test-job"
        assert len(list(job_dir.iterdir())) == 1

    def test_cleanup_removes_empty_job_dirs(self, mgr, snapshot_base):
        old_ts = str(int(time.time()) - 8 * 24 * 3600)
        old_dir = snapshot_base / "cleanup-job" / old_ts
        old_dir.mkdir(parents=True)
        (old_dir / "state.json").write_text("{}")

        mgr.cleanup(max_age_hours=168)
        # Job dir should be removed since it's now empty
        assert not (snapshot_base / "cleanup-job").exists()

    def test_cleanup_no_base_dir_returns_zero(self, tmp_path):
        mgr = SnapshotManager(base_dir=tmp_path / "nonexistent")
        assert mgr.cleanup() == 0

    def test_cleanup_mixed_old_and_new(self, mgr, snapshot_base):
        job_dir = snapshot_base / "test-job"

        # Old snapshot (8 days ago)
        old_ts = str(int(time.time()) - 8 * 24 * 3600)
        old_dir = job_dir / old_ts
        old_dir.mkdir(parents=True)
        (old_dir / "state.json").write_text("{}")

        # New snapshot (just now)
        new_ts = str(int(time.time()))
        new_dir = job_dir / new_ts
        new_dir.mkdir(parents=True)
        (new_dir / "state.json").write_text("{}")

        removed = mgr.cleanup(max_age_hours=168)
        assert removed == 1
        assert not old_dir.exists()
        assert new_dir.exists()


# ─── List tests ──────────────────────────────────────────────────────


class TestListSnapshots:
    """Tests for SnapshotManager.list_snapshots()."""

    def test_list_empty_returns_empty(self, mgr):
        result = mgr.list_snapshots("nonexistent-job")
        assert result == []

    def test_list_returns_snapshots_newest_first(self, mgr, workspace):
        path1 = mgr.capture("test-job", workspace)
        time.sleep(1.1)
        path2 = mgr.capture("test-job", workspace)

        snapshots = mgr.list_snapshots("test-job")
        assert len(snapshots) == 2
        # Newest first
        assert snapshots[0]["path"] == path2
        assert snapshots[1]["path"] == path1

    def test_list_snapshot_has_required_keys(self, mgr, workspace):
        mgr.capture("test-job", workspace)
        snapshots = mgr.list_snapshots("test-job")
        assert len(snapshots) == 1
        assert "timestamp" in snapshots[0]
        assert "path" in snapshots[0]
        # Timestamp should be a numeric string
        int(snapshots[0]["timestamp"])  # Should not raise


# ─── Edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests."""

    def test_base_dir_property(self, mgr, snapshot_base):
        assert mgr.base_dir == snapshot_base

    def test_default_base_dir(self):
        mgr = SnapshotManager()
        assert str(mgr.base_dir).endswith("snapshots")

    def test_capture_with_only_log_file(self, mgr, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "mozart.log").write_text("some logs")
        result = mgr.capture("log-only-job", ws)
        assert result is not None
        from pathlib import Path
        snap = Path(result)
        assert (snap / "mozart.log").exists()
