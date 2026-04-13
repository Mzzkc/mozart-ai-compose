"""API key keyring configuration models.

The conductor maintains a keyring of API keys per instrument. Keys are
NEVER stored in config files, score YAML, or anything in the git repo.
Keys live in $SECRETS_DIR/ and are referenced by path.

The keyring supports rotation policies:
- least-recently-rate-limited: pick the key that hasn't hit rate limits recently
- round-robin: rotate through keys in order

Key files are read at dispatch time by the conductor. The key values never
appear in logs, events, or state files.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class KeyEntry(BaseModel):
    """A single API key file reference.

    The path points to a file containing the key value. The conductor reads
    the file at dispatch time. The label is for human identification in
    logs and diagnostics (never the key value itself).
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description="Path to key file in $SECRETS_DIR/. "
        "Supports $SECRETS_DIR and ~ expansion at runtime.",
    )
    label: str = Field(
        description="Human-readable label for this key. "
        "Used in logs and diagnostics, never the actual key value.",
    )


class InstrumentKeyring(BaseModel):
    """Key management for a single instrument.

    Multiple keys enable rotation when one hits rate limits. The rotation
    policy determines which key is selected for each dispatch.
    """

    model_config = ConfigDict(extra="forbid")

    keys: list[KeyEntry] = Field(
        description="API key entries for this instrument",
    )
    rotation: str = Field(
        default="least-recently-rate-limited",
        description="Key rotation policy: "
        "'least-recently-rate-limited' (default) or 'round-robin'",
    )


class KeyringConfig(BaseModel):
    """Top-level keyring configuration for all instruments.

    Lives in the daemon config (conductor-level), not per-score.
    All scores running under the conductor share the keyring.

    Example YAML::

        keyring:
          instruments:
            openrouter:
              keys:
                - path: "$SECRETS_DIR/openrouter-primary.key"
                  label: "primary"
                - path: "$SECRETS_DIR/openrouter-secondary.key"
                  label: "secondary"
              rotation: least-recently-rate-limited
            anthropic:
              keys:
                - path: "$SECRETS_DIR/anthropic.key"
                  label: "main"
    """

    model_config = ConfigDict(extra="forbid")

    instruments: dict[str, InstrumentKeyring] = Field(
        default_factory=dict,
        description="Per-instrument key configurations. "
        "Keys are instrument names matching registered profiles.",
    )
