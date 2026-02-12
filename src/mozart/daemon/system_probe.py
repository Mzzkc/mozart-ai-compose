"""Consolidated system probes for the Mozart daemon.

Provides a single ``SystemProbe`` class that encapsulates the
"try psutil → fallback to /proc" pattern used across the daemon for:

- Memory usage (RSS)
- Child process counting
- Zombie detection and reaping
- Process group member counting

Extracted from ``monitor.py`` and ``pgroup.py`` (FIX-16) to eliminate
5× duplication of the psutil/proc fallback logic.  All methods are
static — callers import the class and call methods directly.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psutil

_logger = logging.getLogger("daemon.system_probe")


class SystemProbe:
    """Consolidated system resource probes.

    Each method tries psutil first, then falls back to /proc on Linux.
    Returns ``None`` when all probes fail — callers should treat that
    as a critical condition (fail-closed).
    """

    @staticmethod
    def get_memory_mb() -> float | None:
        """Get current process RSS memory in MB.

        Uses ``psutil.Process().memory_info().rss`` if available,
        falls back to reading ``VmRSS`` from ``/proc/self/status``.

        Returns:
            RSS in megabytes, or ``None`` when all probes fail.
        """
        try:
            import psutil

            proc: psutil.Process = psutil.Process()
            rss_bytes: int = proc.memory_info().rss
            return rss_bytes / (1024 * 1024)
        except ImportError:
            pass
        except Exception:
            _logger.debug("psutil_memory_probe_failed", exc_info=True)
        # Fallback: /proc/self/status (Linux only)
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # kB -> MB
        except (OSError, ValueError):
            pass
        _logger.debug("memory_probe_failed", exc_info=True)
        return None

    @staticmethod
    def get_child_count() -> int | None:
        """Count child processes of the current process (recursive).

        Uses ``psutil.Process().children(recursive=True)`` if available,
        falls back to scanning ``/proc/*/status`` for matching PPid.

        Returns:
            Number of child processes, or ``None`` when all probes fail.
        """
        try:
            import psutil

            current = psutil.Process()
            return len(current.children(recursive=True))
        except ImportError:
            pass
        except Exception:
            _logger.debug("psutil_child_count_probe_failed", exc_info=True)
        return SystemProbe._proc_child_count()

    @staticmethod
    def get_zombies() -> list[int]:
        """Detect zombie child processes.

        Uses psutil to iterate children and check status, falls back
        to ``os.waitpid(-1, WNOHANG)`` to reap any zombie children.

        Returns:
            List of zombie PIDs that were detected (not necessarily reaped).
        """
        zombies: list[int] = []
        try:
            import psutil

            current = psutil.Process()
            for child in current.children(recursive=True):
                try:
                    if child.status() == psutil.STATUS_ZOMBIE:
                        zombies.append(child.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return zombies
        except ImportError:
            pass
        except Exception:
            _logger.debug("zombie_probe_failed", exc_info=True)
            return zombies
        return SystemProbe._waitpid_reap_loop()

    @staticmethod
    def reap_zombies() -> list[int]:
        """Detect and reap zombie child processes.

        Similar to ``get_zombies()`` but also calls ``os.waitpid()``
        for each detected zombie (psutil path) to actually reap them.

        Returns:
            List of PIDs that were reaped.
        """
        reaped: list[int] = []
        try:
            import psutil

            current = psutil.Process()
            for child in current.children(recursive=True):
                try:
                    if child.status() == psutil.STATUS_ZOMBIE:
                        try:
                            os.waitpid(child.pid, os.WNOHANG)
                        except ChildProcessError:
                            pass
                        reaped.append(child.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return reaped
        except ImportError:
            pass
        except Exception:
            _logger.debug("zombie_reap_failed", exc_info=True)
            return reaped
        return SystemProbe._waitpid_reap_loop()

    @staticmethod
    def count_group_members(pgid: int, exclude_pid: int = 0) -> int:
        """Count processes in a process group, excluding one PID.

        Uses psutil ``Process.pgid`` per-process if available, falls
        back to reading ``/proc/*/stat`` field 5 (0-indexed 4).

        Note: ``psutil.process_iter(["pgid"])`` raises ``ValueError``
        because ``pgid`` is not a valid ``as_dict`` attribute.  We use
        per-process ``os.getpgid(proc.pid)`` instead.

        Args:
            pgid: Process group ID to count members of.
            exclude_pid: PID to exclude from count (typically self).

        Returns:
            Number of matching processes (0 if probes fail).
        """
        try:
            import psutil

            count = 0
            for proc in psutil.process_iter(["pid"]):
                try:
                    pid = proc.info["pid"]
                    if pid == exclude_pid:
                        continue
                    if os.getpgid(pid) == pgid:
                        count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    continue
            return count
        except ImportError:
            pass
        return SystemProbe._proc_group_count(pgid, exclude_pid)

    # ─── /proc fallback helpers ───────────────────────────────────

    @staticmethod
    def _proc_child_count() -> int:
        """Count child processes by scanning /proc/*/status for PPid.

        Linux-only fallback when psutil is unavailable.
        """
        my_pid = os.getpid()
        count = 0
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                ppid = SystemProbe._read_proc_ppid(entry)
                if ppid == my_pid:
                    count += 1
        except OSError:
            _logger.debug("child_process_probe_failed", exc_info=True)
        return count

    @staticmethod
    def _read_proc_ppid(pid_str: str) -> int | None:
        """Read the PPid field from /proc/{pid}/status."""
        try:
            with open(f"/proc/{pid_str}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        return int(line.split()[1])
        except (OSError, ValueError):
            pass
        return None

    @staticmethod
    def _waitpid_reap_loop() -> list[int]:
        """Reap zombie children via waitpid loop.

        Fallback when psutil is unavailable. Also detects + reaps.
        """
        reaped: list[int] = []
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
                if os.WIFSIGNALED(status) or os.WIFEXITED(status):
                    reaped.append(pid)
            except ChildProcessError:
                break
        return reaped

    @staticmethod
    def _proc_group_count(pgid: int, exclude_pid: int = 0) -> int:
        """Count process group members by scanning /proc/*/stat.

        Linux-only fallback when psutil is unavailable.
        """
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


__all__ = ["SystemProbe"]
