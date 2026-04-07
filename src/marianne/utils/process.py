"""Process safety utilities.

Shared guards for process group operations that prevent PID-recycle
or mock-object bugs from escalating into session-wide kills (F-490).
"""

from __future__ import annotations

import os

from marianne.core.logging import get_logger

_logger = get_logger("utils.process")


def safe_killpg(pgid: int, sig: int, *, context: str = "") -> bool:
    """Session-safe wrapper around os.killpg (F-490).

    Refuses when pgid would target init, the caller's own process group,
    or an invalid value. Prevents PID-recycle or mock-object bugs from
    translating into ``kill(-1, SIGKILL)`` that nukes the user session.

    In particular, ``os.killpg(1, sig)`` compiles to ``kill(-1, sig)``
    in the kernel, which sends the signal to every process the caller
    owns except init — killing systemd --user, every terminal, pytest,
    and the daemon.

    Guards:
    - ``pgid <= 1``: init (1), own pgroup in killpg(0) idiom (0), or invalid
    - ``pgid == os.getpgid(0)``: our own process group (would kill us plus
      whatever shell/pytest/terminal shares the group)

    Returns True if the signal was actually sent, False if blocked. Callers
    should treat False the same as a successful kill for cleanup purposes —
    the target is either unreachable or would have killed the caller.
    """
    if pgid <= 1:
        _logger.warning(
            "killpg_guard_refused",
            reason="pgid_le_1", pgid=pgid, signal=sig, context=context,
        )
        return False
    try:
        own_pgid = os.getpgid(0)
        if pgid == own_pgid:
            _logger.warning(
                "killpg_guard_refused",
                reason="own_pgroup", pgid=pgid, signal=sig, context=context,
            )
            return False
    except OSError:
        pass  # getpgid failed — fall through to killpg with validated pgid
    os.killpg(pgid, sig)
    return True
