"""Tests for the API key keyring — rotation logic, cooldown, key selection.

The keyring reads key files from disk, tracks per-key rate limit cooldowns,
and selects keys via configurable rotation policies. Keys are never stored
in memory longer than needed — loaded on select, discarded after use.

TDD: tests written before implementation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from marianne.daemon.keyring import ApiKeyKeyring
from marianne.daemon.keyring_config import InstrumentKeyring, KeyEntry, KeyringConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def tmp_keys(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create three temporary key files."""
    k1 = tmp_path / "key-primary.key"
    k1.write_text("sk-primary-secret-value")
    k2 = tmp_path / "key-secondary.key"
    k2.write_text("sk-secondary-secret-value")
    k3 = tmp_path / "key-tertiary.key"
    k3.write_text("sk-tertiary-secret-value")
    return k1, k2, k3


@pytest.fixture()
def keyring_config(tmp_keys: tuple[Path, Path, Path]) -> KeyringConfig:
    """Create a KeyringConfig with three OpenRouter keys."""
    k1, k2, k3 = tmp_keys
    return KeyringConfig(
        instruments={
            "openrouter": InstrumentKeyring(
                keys=[
                    KeyEntry(path=str(k1), label="primary"),
                    KeyEntry(path=str(k2), label="secondary"),
                    KeyEntry(path=str(k3), label="tertiary"),
                ],
                rotation="least-recently-rate-limited",
            ),
        },
    )


@pytest.fixture()
def round_robin_config(tmp_keys: tuple[Path, Path, Path]) -> KeyringConfig:
    """KeyringConfig with round-robin rotation."""
    k1, k2, k3 = tmp_keys
    return KeyringConfig(
        instruments={
            "openrouter": InstrumentKeyring(
                keys=[
                    KeyEntry(path=str(k1), label="primary"),
                    KeyEntry(path=str(k2), label="secondary"),
                    KeyEntry(path=str(k3), label="tertiary"),
                ],
                rotation="round-robin",
            ),
        },
    )


# =============================================================================
# Construction
# =============================================================================


class TestConstruction:
    """ApiKeyKeyring can be created from a KeyringConfig."""

    def test_creates_from_config(self, keyring_config: KeyringConfig) -> None:
        keyring = ApiKeyKeyring(keyring_config)
        assert keyring is not None

    def test_empty_config(self) -> None:
        keyring = ApiKeyKeyring(KeyringConfig())
        assert keyring is not None

    def test_has_keys_for_configured_instruments(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)
        assert keyring.has_keys("openrouter")
        assert not keyring.has_keys("anthropic")


# =============================================================================
# Key Selection — least-recently-rate-limited
# =============================================================================


class TestLeastRecentlyRateLimited:
    """Default rotation: pick the key that hasn't been rate limited recently."""

    async def test_selects_first_key_initially(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)
        key = await keyring.select_key("openrouter")
        assert key == "sk-primary-secret-value"

    async def test_selects_next_key_after_rate_limit(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)

        # Mark primary as rate-limited
        keyring.report_rate_limit("openrouter", 0, cooldown_seconds=60.0)

        key = await keyring.select_key("openrouter")
        assert key == "sk-secondary-secret-value"

    async def test_skips_all_rate_limited_returns_least_recently_limited(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)

        # Mark all keys as rate-limited with different timestamps
        keyring.report_rate_limit("openrouter", 0, cooldown_seconds=60.0)
        keyring.report_rate_limit("openrouter", 1, cooldown_seconds=60.0)
        keyring.report_rate_limit("openrouter", 2, cooldown_seconds=60.0)

        # When all are limited, return the one whose cooldown expires soonest
        # (the first one limited, since it was limited earliest)
        key = await keyring.select_key("openrouter")
        assert key == "sk-primary-secret-value"

    async def test_expired_cooldown_key_is_available(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)

        # Rate limit with very short cooldown
        keyring.report_rate_limit("openrouter", 0, cooldown_seconds=0.0)

        # Should still be selectable (cooldown expired immediately)
        key = await keyring.select_key("openrouter")
        assert key == "sk-primary-secret-value"


# =============================================================================
# Key Selection — round-robin
# =============================================================================


class TestRoundRobin:
    """Round-robin rotation cycles through keys in order."""

    async def test_cycles_through_keys(
        self,
        round_robin_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(round_robin_config)

        k1 = await keyring.select_key("openrouter")
        k2 = await keyring.select_key("openrouter")
        k3 = await keyring.select_key("openrouter")
        k4 = await keyring.select_key("openrouter")

        assert k1 == "sk-primary-secret-value"
        assert k2 == "sk-secondary-secret-value"
        assert k3 == "sk-tertiary-secret-value"
        assert k4 == "sk-primary-secret-value"  # wraps around

    async def test_skips_rate_limited_in_round_robin(
        self,
        round_robin_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(round_robin_config)

        # Mark secondary as rate-limited
        keyring.report_rate_limit("openrouter", 1, cooldown_seconds=60.0)

        k1 = await keyring.select_key("openrouter")
        k2 = await keyring.select_key("openrouter")
        k3 = await keyring.select_key("openrouter")

        assert k1 == "sk-primary-secret-value"
        assert k2 == "sk-tertiary-secret-value"  # skipped secondary
        assert k3 == "sk-primary-secret-value"  # wraps


# =============================================================================
# Key Loading
# =============================================================================


class TestKeyLoading:
    """Keys are loaded from disk files."""

    async def test_reads_key_from_file(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)
        key = await keyring.select_key("openrouter")
        assert key == "sk-primary-secret-value"

    async def test_strips_whitespace_from_key_file(
        self,
        tmp_path: Path,
    ) -> None:
        key_file = tmp_path / "key-with-whitespace.key"
        key_file.write_text("  sk-padded-key  \n")

        config = KeyringConfig(
            instruments={
                "openrouter": InstrumentKeyring(
                    keys=[KeyEntry(path=str(key_file), label="padded")],
                ),
            },
        )
        keyring = ApiKeyKeyring(config)
        key = await keyring.select_key("openrouter")
        assert key == "sk-padded-key"

    async def test_missing_key_file_raises(self, tmp_path: Path) -> None:
        config = KeyringConfig(
            instruments={
                "openrouter": InstrumentKeyring(
                    keys=[KeyEntry(path=str(tmp_path / "nonexistent.key"), label="missing")],
                ),
            },
        )
        keyring = ApiKeyKeyring(config)
        with pytest.raises(FileNotFoundError):
            await keyring.select_key("openrouter")

    async def test_empty_key_file_raises(self, tmp_path: Path) -> None:
        key_file = tmp_path / "empty.key"
        key_file.write_text("")

        config = KeyringConfig(
            instruments={
                "openrouter": InstrumentKeyring(
                    keys=[KeyEntry(path=str(key_file), label="empty")],
                ),
            },
        )
        keyring = ApiKeyKeyring(config)
        with pytest.raises(ValueError, match="empty"):
            await keyring.select_key("openrouter")

    async def test_expands_env_vars_in_path(
        self,
        tmp_path: Path,
    ) -> None:
        key_file = tmp_path / "expanded.key"
        key_file.write_text("sk-expanded")

        config = KeyringConfig(
            instruments={
                "openrouter": InstrumentKeyring(
                    keys=[
                        KeyEntry(
                            path="$SECRETS_DIR/expanded.key",
                            label="env-expanded",
                        ),
                    ],
                ),
            },
        )
        keyring = ApiKeyKeyring(config)
        with patch.dict("os.environ", {"SECRETS_DIR": str(tmp_path)}):
            key = await keyring.select_key("openrouter")
        assert key == "sk-expanded"


# =============================================================================
# Error handling
# =============================================================================


class TestErrors:
    """Error cases for unknown instruments and empty keyrings."""

    async def test_select_key_unknown_instrument(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)
        with pytest.raises(KeyError, match="anthropic"):
            await keyring.select_key("anthropic")

    async def test_select_key_no_instruments(self) -> None:
        keyring = ApiKeyKeyring(KeyringConfig())
        with pytest.raises(KeyError, match="openrouter"):
            await keyring.select_key("openrouter")


# =============================================================================
# Thread safety (asyncio Lock)
# =============================================================================


class TestConcurrency:
    """Concurrent access to the keyring is safe."""

    async def test_concurrent_selects_are_safe(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)
        results = await asyncio.gather(*[keyring.select_key("openrouter") for _ in range(20)])
        assert all(r.startswith("sk-") for r in results)

    async def test_concurrent_rate_limit_reports_are_safe(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)

        # Concurrently report rate limits and select keys
        async def report_and_select(idx: int) -> str:
            keyring.report_rate_limit("openrouter", idx % 3, cooldown_seconds=0.1)
            return await keyring.select_key("openrouter")

        results = await asyncio.gather(*[report_and_select(i) for i in range(20)])
        assert all(r.startswith("sk-") for r in results)


# =============================================================================
# Cooldown tracking
# =============================================================================


class TestCooldownTracking:
    """Cooldown timestamps are tracked per-key."""

    def test_report_rate_limit_by_index(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)
        # Should not raise
        keyring.report_rate_limit("openrouter", 0, cooldown_seconds=30.0)

    def test_report_rate_limit_invalid_index(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)
        with pytest.raises(IndexError):
            keyring.report_rate_limit("openrouter", 99, cooldown_seconds=30.0)

    def test_report_rate_limit_unknown_instrument(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)
        with pytest.raises(KeyError):
            keyring.report_rate_limit("anthropic", 0, cooldown_seconds=30.0)

    async def test_cooldown_expiry_makes_key_available(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)

        # Rate limit primary with zero cooldown
        keyring.report_rate_limit("openrouter", 0, cooldown_seconds=0.0)

        # Primary should be available (cooldown already expired)
        key = await keyring.select_key("openrouter")
        assert key == "sk-primary-secret-value"

    def test_get_key_index_returns_label(
        self,
        keyring_config: KeyringConfig,
    ) -> None:
        keyring = ApiKeyKeyring(keyring_config)
        assert keyring.get_key_label("openrouter", 0) == "primary"
        assert keyring.get_key_label("openrouter", 1) == "secondary"
