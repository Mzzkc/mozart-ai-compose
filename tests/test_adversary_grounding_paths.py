"""Adversarial tests for grounding path validation (T1.2).

Verifies that FileChecksumGroundingHook rejects path traversal attempts
and absolute paths that could read arbitrary files.
"""

from __future__ import annotations

import pytest

from marianne.execution.grounding import (
    FileChecksumGroundingHook,
    GroundingContext,
)


def _make_context(**overrides: object) -> GroundingContext:
    """Create a minimal GroundingContext for testing."""
    defaults = dict(
        job_id="test-job",
        sheet_num=0,
        prompt="test",
        output="test output",
    )
    defaults.update(overrides)
    return GroundingContext(**defaults)  # type: ignore[arg-type]


class TestChecksumPathValidation:
    """T1.2: FileChecksumGroundingHook path traversal guards."""

    @pytest.mark.asyncio
    async def test_rejects_absolute_path(self, tmp_path) -> None:
        """Absolute paths must not escape the allowed root."""
        hook = FileChecksumGroundingHook(
            expected_checksums={"/etc/passwd": "abc123"},
            allowed_root=tmp_path,
        )
        result = await hook.validate(_make_context())
        assert not result.passed
        assert any("escapes" in m for m in result.details.get("mismatches", []))

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self, tmp_path) -> None:
        """Paths with '..' must not escape the allowed root."""
        hook = FileChecksumGroundingHook(
            expected_checksums={"../../etc/shadow": "abc123"},
            allowed_root=tmp_path,
        )
        result = await hook.validate(_make_context())
        assert not result.passed
        assert any("escapes" in m for m in result.details.get("mismatches", []))

    @pytest.mark.asyncio
    async def test_rejects_nested_traversal(self, tmp_path) -> None:
        """Deeply nested traversal attempts are also caught."""
        hook = FileChecksumGroundingHook(
            expected_checksums={"foo/../../../etc/hosts": "abc123"},
            allowed_root=tmp_path,
        )
        result = await hook.validate(_make_context())
        assert not result.passed
        assert any("escapes" in m for m in result.details.get("mismatches", []))

    @pytest.mark.asyncio
    async def test_accepts_relative_path_within_root(self, tmp_path) -> None:
        """Normal relative paths within allowed_root should work."""
        test_file = tmp_path / "output.txt"
        test_file.write_text("hello world")

        import hashlib
        expected = hashlib.sha256(b"hello world").hexdigest()

        hook = FileChecksumGroundingHook(
            expected_checksums={"output.txt": expected},
            allowed_root=tmp_path,
        )
        # Change cwd to tmp_path so the file resolves correctly
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = await hook.validate(_make_context())
        finally:
            os.chdir(old_cwd)
        assert result.passed
        assert "1 file checksum(s) validated" in result.message

    @pytest.mark.asyncio
    async def test_no_allowed_root_allows_all_paths(self, tmp_path) -> None:
        """Without allowed_root, paths are not restricted (backward compat)."""
        hook = FileChecksumGroundingHook(
            expected_checksums={"/etc/hosts": "abc123"},
        )
        result = await hook.validate(_make_context())
        # Should NOT be a security rejection — just file-not-found
        mismatches = result.details.get("mismatches", [])
        security = [m for m in mismatches if "escapes" in m]
        assert len(security) == 0
