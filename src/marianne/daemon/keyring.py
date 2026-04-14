"""API key keyring — rotation, cooldown tracking, key selection.

The conductor maintains a keyring of API keys per instrument. Keys are
NEVER stored in config files, score YAML, or anything in the git repo.
Keys live in $SECRETS_DIR/ and are referenced by path.

Key files are read at selection time. The key value is returned to the
caller and not cached — minimizing the time secrets live in memory.

Thread safety: all mutable state is protected by an asyncio Lock.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from marianne.core.logging import get_logger
from marianne.daemon.keyring_config import KeyringConfig

_logger = get_logger("daemon.keyring")


class _KeyState:
    """Per-key mutable state for cooldown tracking.

    Attributes:
        cooldown_until: Monotonic timestamp when cooldown expires.
            0.0 means the key is not rate-limited.
    """

    __slots__ = ("cooldown_until",)

    def __init__(self) -> None:
        self.cooldown_until: float = 0.0

    def is_available(self) -> bool:
        """Check if the key's cooldown has expired."""
        return time.monotonic() >= self.cooldown_until


class ApiKeyKeyring:
    """Manages API key selection with rotation and rate limit tracking.

    Reads KeyringConfig from daemon config. Loads keys from disk at
    selection time (path references, never caches values). Selects keys
    via configurable rotation policies.

    Usage::

        keyring = ApiKeyKeyring(config.keyring)

        if keyring.has_keys("openrouter"):
            key = await keyring.select_key("openrouter")
            # Use key for API request
            # If rate limited:
            keyring.report_rate_limit("openrouter", key_index=0, cooldown_seconds=30.0)
    """

    def __init__(self, config: KeyringConfig) -> None:
        self._config = config

        # Per-instrument, per-key state for cooldown tracking
        self._states: dict[str, list[_KeyState]] = {}
        for instrument_name, instr_keyring in config.instruments.items():
            self._states[instrument_name] = [
                _KeyState() for _ in instr_keyring.keys
            ]

        # Round-robin index per instrument
        self._rr_index: dict[str, int] = dict.fromkeys(config.instruments, 0)

        # Protect mutable state
        self._lock = asyncio.Lock()

    def has_keys(self, instrument_name: str) -> bool:
        """Check if keys are configured for an instrument."""
        return instrument_name in self._config.instruments

    async def select_key(self, instrument_name: str) -> str:
        """Select and load an API key for the given instrument.

        Reads the key file from disk. The key value is returned to the
        caller and not cached in the keyring.

        Args:
            instrument_name: Instrument to select a key for.

        Returns:
            The API key string.

        Raises:
            KeyError: If no keys are configured for the instrument.
            FileNotFoundError: If the key file doesn't exist.
            ValueError: If the key file is empty.
        """
        if instrument_name not in self._config.instruments:
            msg = (
                f"No keys configured for instrument '{instrument_name}'. "
                f"Configured instruments: {sorted(self._config.instruments.keys())}"
            )
            raise KeyError(msg)

        instr_keyring = self._config.instruments[instrument_name]
        rotation = instr_keyring.rotation

        async with self._lock:
            if rotation == "round-robin":
                key_index = self._select_round_robin(instrument_name, instr_keyring)
            else:
                key_index = self._select_least_recently_rate_limited(
                    instrument_name, instr_keyring,
                )

        key_entry = instr_keyring.keys[key_index]
        key_value = self._load_key_from_disk(key_entry.path)

        _logger.debug(
            "keyring.key_selected",
            extra={
                "instrument": instrument_name,
                "key_label": key_entry.label,
                "key_index": key_index,
                "rotation": rotation,
            },
        )

        return key_value

    def report_rate_limit(
        self,
        instrument_name: str,
        key_index: int,
        *,
        cooldown_seconds: float,
    ) -> None:
        """Report that a key hit a rate limit.

        Args:
            instrument_name: The instrument whose key was rate-limited.
            key_index: Index of the key in the instrument's key list.
            cooldown_seconds: How long to wait before retrying this key.

        Raises:
            KeyError: If the instrument is not configured.
            IndexError: If key_index is out of range.
        """
        if instrument_name not in self._states:
            msg = f"No keys configured for instrument '{instrument_name}'"
            raise KeyError(msg)

        states = self._states[instrument_name]
        if key_index < 0 or key_index >= len(states):
            msg = (
                f"Key index {key_index} out of range for instrument "
                f"'{instrument_name}' (has {len(states)} keys)"
            )
            raise IndexError(msg)

        states[key_index].cooldown_until = time.monotonic() + cooldown_seconds

        _logger.debug(
            "keyring.rate_limit_reported",
            extra={
                "instrument": instrument_name,
                "key_index": key_index,
                "cooldown_seconds": cooldown_seconds,
            },
        )

    def get_key_label(self, instrument_name: str, key_index: int) -> str:
        """Get the human-readable label for a key.

        Args:
            instrument_name: The instrument name.
            key_index: Index of the key.

        Returns:
            The label string.

        Raises:
            KeyError: If the instrument is not configured.
            IndexError: If key_index is out of range.
        """
        if instrument_name not in self._config.instruments:
            msg = f"No keys configured for instrument '{instrument_name}'"
            raise KeyError(msg)
        keys = self._config.instruments[instrument_name].keys
        if key_index < 0 or key_index >= len(keys):
            msg = f"Key index {key_index} out of range (has {len(keys)} keys)"
            raise IndexError(msg)
        return keys[key_index].label

    # -----------------------------------------------------------------
    # Internal — rotation strategies (called under lock)
    # -----------------------------------------------------------------

    def _select_least_recently_rate_limited(
        self,
        instrument_name: str,
        instr_keyring: object,  # noqa: ARG002 — kept for signature symmetry with _select_round_robin
    ) -> int:
        """Select the key with no cooldown, or the one whose cooldown expires soonest."""
        states = self._states[instrument_name]

        # First pass: find any key that is not rate-limited
        for i, state in enumerate(states):
            if state.is_available():
                return i

        # All keys are rate-limited — return the one whose cooldown expires soonest
        best_index = 0
        best_until = states[0].cooldown_until
        for i, state in enumerate(states[1:], start=1):
            if state.cooldown_until < best_until:
                best_until = state.cooldown_until
                best_index = i

        _logger.warning(
            "keyring.all_keys_rate_limited",
            extra={
                "instrument": instrument_name,
                "returning_key_index": best_index,
                "cooldown_expires_at": best_until,
            },
        )
        return best_index

    def _select_round_robin(
        self,
        instrument_name: str,
        instr_keyring: object,  # noqa: ARG002 — kept for signature symmetry
    ) -> int:
        """Select the next key in round-robin order, skipping rate-limited keys."""
        states = self._states[instrument_name]
        num_keys = len(states)
        start = self._rr_index[instrument_name]

        # Try each key in order from current position
        for offset in range(num_keys):
            idx = (start + offset) % num_keys
            if states[idx].is_available():
                # Advance past this key for next call
                self._rr_index[instrument_name] = (idx + 1) % num_keys
                return idx

        # All rate-limited — use current position anyway and advance
        idx = start % num_keys
        self._rr_index[instrument_name] = (idx + 1) % num_keys
        return idx

    # -----------------------------------------------------------------
    # Internal — key loading
    # -----------------------------------------------------------------

    @staticmethod
    def _load_key_from_disk(path_str: str) -> str:
        """Load a key value from a file on disk.

        Expands environment variables and ~ in the path. Strips whitespace
        from the key value.

        Args:
            path_str: Path to the key file (may contain $VAR or ~).

        Returns:
            The key value (stripped of whitespace).

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is empty after stripping.
        """
        expanded = os.path.expandvars(path_str)
        expanded = os.path.expanduser(expanded)
        key_path = Path(expanded)

        if not key_path.exists():
            msg = f"Key file not found: {key_path}"
            raise FileNotFoundError(msg)

        key_value = key_path.read_text(encoding="utf-8").strip()
        if not key_value:
            msg = f"Key file is empty: {key_path}"
            raise ValueError(msg)

        return key_value


# Convenience alias used by validation and external consumers.
Keyring = ApiKeyKeyring
