"""Daemon operational profiles.

Profiles are partial DaemonConfig overrides shipped as YAML files.
They configure how the conductor operates — concurrency, logging,
profiling intensity, resource limits — without touching score configs.

Built-in profiles:
  dev       — Debug logging, strace enabled, low concurrency
  intensive — Long timeouts, high resource limits
  minimal   — Profiler and learning disabled, low concurrency
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PROFILES_DIR = Path(__file__).parent

# Built-in profile names (no .yaml suffix)
BUILTIN_PROFILES: frozenset[str] = frozenset(
    p.stem for p in _PROFILES_DIR.glob("*.yaml")
)


def list_profiles() -> list[str]:
    """Return sorted list of built-in profile names."""
    return sorted(BUILTIN_PROFILES)


def get_profile(name: str) -> dict[str, Any]:
    """Load a built-in profile by name.

    Args:
        name: Profile name (e.g. "dev", "intensive", "minimal").

    Returns:
        Parsed YAML dict suitable for deep-merging into DaemonConfig data.

    Raises:
        FileNotFoundError: If the profile does not exist.
    """
    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(list_profiles())
        raise FileNotFoundError(
            f"Unknown daemon profile '{name}'. "
            f"Available profiles: {available}"
        )
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    - Dict values are merged recursively.
    - All other values in *override* replace those in *base*.
    - Keys in *base* not present in *override* are preserved.
    """
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


__all__ = [
    "BUILTIN_PROFILES",
    "deep_merge",
    "get_profile",
    "list_profiles",
]
