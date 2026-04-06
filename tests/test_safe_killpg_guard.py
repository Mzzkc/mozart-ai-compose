"""Regression tests for F-490: _safe_killpg guard in claude_cli backend.

F-490 root cause: ``os.killpg(os.getpgid(process.pid), signal.SIGKILL)`` in
claude_cli.py had no validation. If ``process.pid == 1`` (mock, stub, or
reaped-and-recycled PID), then ``os.getpgid(1) == 1``, and
``os.killpg(1, SIGKILL)`` compiles kernel-side to ``kill(-1, SIGKILL)`` —
"send SIGKILL to every process UID 1000 can signal except init." That kills
``systemd --user``, every bash shell in every WSL terminal, and any pytest
running in the same session. Exact signature of the user-reported
"exit code 00000009" WSL2 crashes.

These tests lock in the guard behavior. They deliberately do NOT run the
real ``os.killpg`` syscall for the unsafe cases — if the guard ever
regresses, a test run alone would kill the user's session.
"""

from __future__ import annotations

import os
import signal
from unittest.mock import patch

import pytest

from marianne.backends.claude_cli import _safe_killpg


class TestSafeKillpgGuardBlocks:
    """The guard MUST refuse these inputs and MUST NOT call os.killpg."""

    def test_pgid_one_is_blocked(self) -> None:
        """pgid=1 is the F-490 trigger — killpg(1, SIGKILL) == kill(-1, SIGKILL)."""
        with patch("marianne.backends.claude_cli.os.killpg") as mock_killpg:
            result = _safe_killpg(1, signal.SIGKILL, context="test")
        assert result is False, "guard must return False when blocking"
        mock_killpg.assert_not_called()

    def test_pgid_zero_is_blocked(self) -> None:
        """pgid=0 would signal the caller's own pgroup — also unsafe."""
        with patch("marianne.backends.claude_cli.os.killpg") as mock_killpg:
            result = _safe_killpg(0, signal.SIGKILL, context="test")
        assert result is False
        mock_killpg.assert_not_called()

    def test_pgid_negative_is_blocked(self) -> None:
        """Negative pgid is invalid / kernel-special; always refuse."""
        with patch("marianne.backends.claude_cli.os.killpg") as mock_killpg:
            result = _safe_killpg(-1, signal.SIGKILL, context="test")
        assert result is False
        mock_killpg.assert_not_called()

    def test_own_pgroup_is_blocked(self) -> None:
        """Signaling our own pgroup would kill pytest + whatever shares the group."""
        own_pgid = os.getpgid(0)
        with patch("marianne.backends.claude_cli.os.killpg") as mock_killpg:
            result = _safe_killpg(own_pgid, signal.SIGKILL, context="test")
        assert result is False
        mock_killpg.assert_not_called()

    def test_blocked_call_logs_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Guard must log at warning level so refusals are visible in conductor.log.

        Mozart uses structlog which writes to stdout/stderr directly in the
        default configuration, so capsys is the right capture tool.
        """
        with patch("marianne.backends.claude_cli.os.killpg"):
            _safe_killpg(1, signal.SIGKILL, context="unit_test")
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "killpg_guard_refused" in combined, (
            f"expected killpg_guard_refused in log output, got: {combined!r}"
        )
        assert "pgid_le_1" in combined, "guard must report the reason"
        assert "unit_test" in combined, "guard must report the caller context"


class TestSafeKillpgGuardAllows:
    """The guard MUST permit valid pgids that are not the caller's own."""

    def test_valid_other_pgid_is_allowed(self) -> None:
        """A pgid > 1 that is not our own should pass through to os.killpg."""
        own_pgid = os.getpgid(0)
        # Pick a pgid that is definitely not 1 and not ours.
        # 999999 is almost certainly not a live pgid, but the guard doesn't
        # check liveness — it only checks the blast-radius conditions.
        fake_pgid = 999999 if own_pgid != 999999 else 999998

        with patch("marianne.backends.claude_cli.os.killpg") as mock_killpg:
            result = _safe_killpg(fake_pgid, signal.SIGKILL, context="test")

        assert result is True, "guard must return True when allowing"
        mock_killpg.assert_called_once_with(fake_pgid, signal.SIGKILL)

    def test_guard_does_not_block_getpgid_failure(self) -> None:
        """If os.getpgid(0) raises, the own-pgroup check must not crash the guard.

        The guard should still allow the kill through (having only the
        pgid<=1 check to rely on), because the alternative is to silently
        swallow every cleanup call when getpgid is broken.
        """
        fake_pgid = 999999
        with patch("marianne.backends.claude_cli.os.getpgid",
                   side_effect=OSError("mocked")), \
                patch("marianne.backends.claude_cli.os.killpg") as mock_killpg:
            result = _safe_killpg(fake_pgid, signal.SIGKILL, context="test")

        assert result is True
        mock_killpg.assert_called_once_with(fake_pgid, signal.SIGKILL)


class TestSafeKillpgGuardSignalTypes:
    """Guard must apply to all signals, not just SIGKILL."""

    @pytest.mark.parametrize("sig", [
        signal.SIGTERM,
        signal.SIGKILL,
        signal.SIGINT,
        signal.SIGHUP,
    ])
    def test_all_signals_blocked_on_pgid_one(self, sig: signal.Signals) -> None:
        """pgid=1 must be blocked regardless of signal — any signal to pgid=1
        translates to kill(-1, sig) which affects every process in the session."""
        with patch("marianne.backends.claude_cli.os.killpg") as mock_killpg:
            result = _safe_killpg(1, sig, context="test")
        assert result is False
        mock_killpg.assert_not_called()
