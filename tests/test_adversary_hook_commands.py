"""Adversarial tests for hook command validation (T1.1).

Verifies that obviously destructive hook commands are rejected
before being passed to subprocess execution.
"""

from __future__ import annotations

import re

import pytest

from marianne.daemon.manager import JobManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager() -> JobManager:
    """Create a JobManager with minimal stubs (no daemon needed)."""
    return JobManager.__new__(JobManager)


# ---------------------------------------------------------------------------
# Destructive pattern tests
# ---------------------------------------------------------------------------

class TestHookCommandValidation:
    """T1.1: Hook command validation guards."""

    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /",
            "rm -fr /",
            "rm -r -f /home",
            "rm -rf --no-preserve-root /",
            "rm -Rf /var/lib",
            "mkfs.ext4 /dev/sda1",
            "dd if=/dev/zero of=/dev/sda",
            "chmod -R 777 /",
            "chmod -R 000 /etc",
        ],
    )
    def test_rejects_destructive_commands(self, command: str) -> None:
        mgr = _make_manager()
        with pytest.raises(ValueError, match="destructive pattern"):
            mgr._validate_hook_command(command, hook_type="run_command")

    def test_rejects_oversized_command(self) -> None:
        mgr = _make_manager()
        command = "echo " + "x" * 5000
        with pytest.raises(ValueError, match="exceeds 4096 chars"):
            mgr._validate_hook_command(command, hook_type="run_command")

    @pytest.mark.parametrize(
        "command",
        [
            "echo hello",
            "python3 -m pytest tests/",
            "rm -rf ./build/",           # relative path is fine
            "git log --oneline -20",
            "docker compose up -d",
            "make lint && make test",
        ],
    )
    def test_accepts_normal_commands(self, command: str) -> None:
        mgr = _make_manager()
        # Should not raise
        mgr._validate_hook_command(command, hook_type="run_command")

    def test_fork_bomb_pattern_rejected(self) -> None:
        """The classic :(){:|:&};: fork bomb pattern."""
        mgr = _make_manager()
        command = ":(){ :|:& };:"
        with pytest.raises(ValueError, match="destructive pattern"):
            mgr._validate_hook_command(command, hook_type="run_command")

    def test_rewrite_dev_sd_rejected(self) -> None:
        mgr = _make_manager()
        with pytest.raises(ValueError, match="destructive pattern"):
            mgr._validate_hook_command(
                "dd if=/dev/zero of=/dev/sdb bs=1M",
                hook_type="run_script",
            )
