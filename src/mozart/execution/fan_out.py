"""Compatibility re-export: fan-out expansion moved to core.fan_out.

The fan-out module is pure data transformation with no execution-layer
dependencies, so it belongs in core/ (not execution/). This shim
preserves backward compatibility for existing imports.
"""

from mozart.core.fan_out import (  # noqa: F401
    FanOutExpansion,
    FanOutMetadata,
    expand_fan_out,
)

__all__ = ["FanOutExpansion", "FanOutMetadata", "expand_fan_out"]
