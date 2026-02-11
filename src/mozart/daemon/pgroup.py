"""Process group management for orphan prevention.

Addresses issue #38: cascading crashes from orphaned MCP server processes.
When the daemon is the process group leader, all child processes (backends,
validators, MCP servers) belong to the same pgid, enabling clean group-wide
shutdown via os.killpg().

Lifecycle:
    1. setup() — create new process group (daemon becomes leader)
    2. cleanup_orphans() — periodic scan for leaked processes
    3. kill_all_children() — shutdown: signal entire group
    4. atexit handler — last-resort cleanup if shutdown is interrupted
"""

from __future__ import annotations

import atexit
import os
import signal

from mozart.core.logging import get_logger

_logger = get_logger("daemon.pgroup")


class ProcessGroupManager:
    """Manages the daemon's process group to prevent orphans.

    The daemon calls setup() early in its lifecycle to become the process
    group leader.  During shutdown, kill_all_children() sends SIGTERM to
    the entire group, ensuring no child process (including deeply nested
    MCP servers) survives the daemon.

    An atexit handler provides last-resort cleanup even if the normal
    shutdown path is skipped.
    """

    def __init__(self) -> None:
        self._original_pgid: int = os.getpgrp()
        self._is_leader: bool = False
        self._atexit_registered: bool = False

    @property
    def is_leader(self) -> bool:
        """Whether the daemon is the process group leader."""
        return self._is_leader

    @property
    def original_pgid(self) -> int:
        """Process group ID before setup() was called."""
        return self._original_pgid

    def setup(self) -> None:
        """Create a new process group with the daemon as leader.

        Must be called early in daemon startup, before spawning any
        child processes.  Idempotent — safe to call multiple times.
        """
        if self._is_leader:
            return

        try:
            os.setpgrp()
            self._is_leader = True
            _logger.info(
                "pgroup.setup_complete",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                original_pgid=self._original_pgid,
            )
        except OSError as exc:
            # May fail if already the group leader (e.g. after daemonize)
            if os.getpid() == os.getpgrp():
                self._is_leader = True
                _logger.debug(
                    "pgroup.already_leader",
                    pid=os.getpid(),
                )
            else:
                _logger.warning(
                    "pgroup.setup_failed",
                    error=str(exc),
                    pid=os.getpid(),
                )

        # Register last-resort atexit cleanup
        if self._is_leader and not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True

    def kill_all_children(self, sig: int = signal.SIGTERM) -> int:
        """Send signal to all processes in our group except ourselves.

        Args:
            sig: Signal number to send (default SIGTERM for graceful stop).

        Returns:
            The process group ID that was signaled, or 0 if no signal sent.
        """
        if not self._is_leader:
            _logger.debug("pgroup.not_leader_skip_kill")
            return 0

        pgid = os.getpgrp()
        my_pid = os.getpid()

        # Count children before signaling for logging
        child_count = self._count_group_members(pgid, exclude_pid=my_pid)

        if child_count == 0:
            _logger.debug("pgroup.no_children_to_signal")
            return pgid

        try:
            # Temporarily ignore the signal in our own process so killpg
            # doesn't terminate us along with the children.
            old_handler = signal.signal(sig, signal.SIG_IGN)
            try:
                os.killpg(pgid, sig)
            finally:
                signal.signal(sig, old_handler)

            _logger.info(
                "pgroup.signaled_children",
                signal=signal.Signals(sig).name,
                pgid=pgid,
                child_count=child_count,
            )
            return pgid
        except ProcessLookupError:
            _logger.debug("pgroup.no_processes_in_group", pgid=pgid)
            return 0
        except PermissionError:
            _logger.warning("pgroup.permission_denied", pgid=pgid)
            return 0

    def cleanup_orphans(self) -> list[int]:
        """Find and clean up orphaned child processes.

        Detects two categories:
        1. Zombie children — reaped via waitpid
        2. Orphaned MCP servers — processes whose parent has died
           (reparented to init/PID 1) that still match MCP patterns

        Returns:
            List of PIDs that were cleaned up.
        """
        orphans: list[int] = []

        # Strategy 1: psutil-based deep scan (preferred)
        try:
            import psutil

            current = psutil.Process(os.getpid())
            for child in current.children(recursive=True):
                try:
                    if not child.is_running():
                        continue

                    status = child.status()

                    # Reap zombies
                    if status == psutil.STATUS_ZOMBIE:
                        try:
                            os.waitpid(child.pid, os.WNOHANG)
                        except ChildProcessError:
                            pass
                        orphans.append(child.pid)
                        _logger.debug(
                            "pgroup.reaped_zombie",
                            pid=child.pid,
                        )
                        continue

                    # Detect orphaned MCP servers (reparented away from us).
                    # On classic Unix, reparented children get ppid=1 (init).
                    # On systemd user instances or WSL, the reparent target
                    # may be a user-level init with a different PID.  We check
                    # both ppid==1 AND ppid!=our_pid to catch both cases.
                    cmdline = " ".join(child.cmdline()).lower()
                    my_pid = os.getpid()
                    child_ppid = child.ppid()
                    is_orphan = child_ppid != my_pid and child_ppid != 0
                    if "mcp" in cmdline and is_orphan:
                        child.terminate()
                        orphans.append(child.pid)
                        _logger.warning(
                            "pgroup.killed_orphan_mcp",
                            pid=child.pid,
                            ppid=child_ppid,
                            cmdline_snippet=cmdline[:120],
                        )

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except ImportError:
            # Strategy 2: /proc fallback (Linux only)
            orphans.extend(self._cleanup_orphans_proc())

        if orphans:
            _logger.info(
                "pgroup.orphan_cleanup",
                cleaned_count=len(orphans),
                pids=orphans,
            )

        return orphans

    # ─── Internal ──────────────────────────────────────────────────────

    @staticmethod
    def _count_group_members(pgid: int, exclude_pid: int) -> int:
        """Count processes in the given process group (excluding one PID).

        Uses psutil if available, falls back to /proc on Linux.
        """
        try:
            import psutil

            count = 0
            for proc in psutil.process_iter(["pid", "pgid"]):
                try:
                    info = proc.info
                    if info["pgid"] == pgid and info["pid"] != exclude_pid:
                        count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return count
        except ImportError:
            pass

        # Fallback: /proc (Linux only)
        count = 0
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                if pid == exclude_pid:
                    continue
                try:
                    with open(f"/proc/{pid}/stat") as f:
                        parts = f.read().split()
                        # Field 5 (0-indexed 4) is pgid
                        if len(parts) > 4 and int(parts[4]) == pgid:
                            count += 1
                except (OSError, ValueError, IndexError):
                    continue
        except OSError:
            pass
        return count

    @staticmethod
    def _cleanup_orphans_proc() -> list[int]:
        """Fallback orphan cleanup using only /proc and os.waitpid."""
        orphans: list[int] = []
        # Reap any zombie children
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
                orphans.append(pid)
            except ChildProcessError:
                break
        return orphans

    def _atexit_cleanup(self) -> None:
        """Last-resort cleanup registered via atexit.

        Sends SIGTERM to all children in our process group.  This runs
        even if the normal shutdown path is skipped (e.g. unhandled
        exception in the event loop).
        """
        if not self._is_leader:
            return

        try:
            pgid = os.getpgrp()
            my_pid = os.getpid()
            child_count = self._count_group_members(pgid, exclude_pid=my_pid)

            if child_count > 0:
                # Use SIG_IGN dance to avoid killing ourselves
                old_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
                try:
                    os.killpg(pgid, signal.SIGTERM)
                finally:
                    signal.signal(signal.SIGTERM, old_handler)

                _logger.info(
                    "pgroup.atexit_cleanup",
                    pgid=pgid,
                    child_count=child_count,
                )
        except Exception:
            # atexit handlers must not raise
            pass


__all__ = ["ProcessGroupManager"]
