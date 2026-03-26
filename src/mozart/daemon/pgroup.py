"""Process group management for orphan prevention.

Addresses issue #38: cascading crashes from orphaned MCP server processes.
When the daemon is the process group leader, all child processes (backends,
validators, MCP servers) belong to the same pgid, enabling clean group-wide
shutdown via os.killpg().

Lifecycle:
    1. setup() — create new process group (daemon becomes leader)
    2. cleanup_orphans() — scan daemon's child tree for leaked processes
    3. reap_orphaned_backends() — system-wide scan for leaked backend children
    4. kill_all_children() — shutdown: signal entire group
    5. atexit handler — last-resort cleanup if shutdown is interrupted

The distinction between cleanup_orphans() and reap_orphaned_backends():
    cleanup_orphans() walks the daemon's own child tree via psutil.
    This misses processes that were reparented to init (PID 1) after
    their parent (e.g. Claude CLI) exited — which is exactly how
    MCP/LSP server leaks happen.  reap_orphaned_backends() scans
    ALL processes owned by the current user for known orphan patterns.
"""

from __future__ import annotations

import atexit
import os
import signal

from mozart.core.logging import get_logger

_logger = get_logger("daemon.pgroup")

# Patterns that identify orphaned backend child processes.
# These are processes spawned by Claude CLI (or similar backends) that
# should not outlive their parent.  Matched against the joined cmdline.
_ORPHAN_CMDLINE_PATTERNS: tuple[str, ...] = (
    "symbols run",           # Claude Code LSP MCP servers
    "tsserver",              # TypeScript language server
    "typingsinstaller",      # TypeScript typings installer
    "pyright-langserver",    # Pyright language server
    "rust-analyzer",         # Rust language server (via symbols)
    "typescript-language-server",  # TS language server (via symbols)
    "clangd",                # C/C++ language server (via symbols)
)


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
        """Find and clean up orphaned child processes in the daemon's tree.

        Detects two categories:
        1. Zombie children — reaped via waitpid
        2. Orphaned MCP servers — processes whose parent has died
           (reparented to init/PID 1) that still match MCP patterns

        Note: This only scans the daemon's own child tree.  For processes
        that escaped the tree entirely (reparented to init), use
        reap_orphaned_backends() which does a system-wide scan.

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

    def reap_orphaned_backends(self) -> list[int]:
        """System-wide scan for orphaned backend child processes.

        Scans ALL processes owned by the current user for known orphan
        patterns (LSP servers, MCP servers, etc.) that have been reparented
        to init (PID 1).  These are processes that escaped the daemon's
        process tree because their parent (Claude CLI) exited before they
        could be cleaned up.

        This is the safety net that catches leaks missed by cleanup_orphans().
        Called periodically by the daemon monitor and on daemon startup.

        Returns:
            List of PIDs that were killed.
        """
        killed: list[int] = []
        my_uid = os.getuid()
        my_pid = os.getpid()

        try:
            import psutil

            for proc in psutil.process_iter(["pid", "ppid", "cmdline", "uids"]):
                try:
                    info = proc.info
                    # Only our user's processes
                    uids = info.get("uids")
                    if uids is None or uids.real != my_uid:
                        continue
                    # Skip ourselves
                    if info["pid"] == my_pid:
                        continue
                    # Only orphans (reparented to init or similar)
                    ppid = info.get("ppid", -1)
                    if ppid not in (0, 1):
                        continue

                    cmdline_parts = info.get("cmdline")
                    if not cmdline_parts:
                        continue
                    cmdline = " ".join(cmdline_parts).lower()

                    # Match against known orphan patterns
                    for pattern in _ORPHAN_CMDLINE_PATTERNS:
                        if pattern in cmdline:
                            proc.kill()  # SIGKILL — these already ignored SIGTERM
                            killed.append(info["pid"])
                            _logger.warning(
                                "pgroup.reaped_orphaned_backend_child",
                                pid=info["pid"],
                                pattern=pattern,
                                cmdline_snippet=cmdline[:120],
                            )
                            break

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

        except ImportError:
            # /proc fallback for Linux without psutil
            killed.extend(self._reap_orphans_proc())

        if killed:
            _logger.info(
                "pgroup.orphaned_backend_children_reaped",
                count=len(killed),
                pids=killed,
            )

        return killed

    # ─── Internal ──────────────────────────────────────────────────────

    @staticmethod
    def _count_group_members(pgid: int, exclude_pid: int) -> int:
        """Count processes in the given process group (excluding one PID).

        Delegates to ``SystemProbe.count_group_members()``.
        Uses a lazy import to avoid import-time side effects that can
        interfere with pytest's exit cleanup.

        Returns 0 when the probe fails entirely (None from SystemProbe),
        which is fail-safe: "no members found → don't signal anyone".
        """
        from mozart.daemon.system_probe import SystemProbe

        return SystemProbe.count_group_members(pgid, exclude_pid=exclude_pid) or 0

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

    @staticmethod
    def _reap_orphans_proc() -> list[int]:
        """Fallback system-wide orphan scan using /proc (Linux only)."""
        killed: list[int] = []
        my_uid = os.getuid()
        my_pid = os.getpid()
        proc_path = "/proc"

        try:
            entries = os.listdir(proc_path)
        except OSError:
            return killed

        for entry in entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == my_pid:
                continue

            try:
                # Check ownership
                stat = os.stat(f"{proc_path}/{pid}")
                if stat.st_uid != my_uid:
                    continue

                # Check ppid
                with open(f"{proc_path}/{pid}/status") as f:
                    ppid = -1
                    for line in f:
                        if line.startswith("PPid:"):
                            ppid = int(line.split()[1])
                            break
                if ppid not in (0, 1):
                    continue

                # Check cmdline
                with open(f"{proc_path}/{pid}/cmdline") as f:
                    cmdline = f.read().replace("\0", " ").lower()

                for pattern in _ORPHAN_CMDLINE_PATTERNS:
                    if pattern in cmdline:
                        try:
                            os.kill(pid, signal.SIGKILL)
                            killed.append(pid)
                            _logger.warning(
                                "pgroup.reaped_orphaned_backend_child",
                                pid=pid,
                                pattern=pattern,
                                cmdline_snippet=cmdline[:120],
                            )
                        except (OSError, ProcessLookupError):
                            pass
                        break

            except (OSError, ValueError):
                continue

        return killed

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
            # atexit handlers must not raise — but log for post-mortem debugging
            _logger.warning("atexit_cleanup_failed", exc_info=True)


__all__ = ["ProcessGroupManager"]
