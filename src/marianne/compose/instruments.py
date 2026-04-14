"""Instrument resolver — produces per-agent per-sheet instrument assignments.

Resolution order:
1. Start with defaults for each phase type (recon, plan, work, etc.)
2. Apply per-agent overrides
3. For each sheet, resolve the primary + full fallback chain
4. Emit per_sheet_instruments and per_sheet_instrument_config in the score YAML
5. Every sheet gets the full instrument catalog as its tail — no dead ends
"""

from __future__ import annotations

import logging
from typing import Any

from marianne.compose.sheets import SHEET_PHASE, SHEETS_PER_CYCLE

_logger = logging.getLogger(__name__)

# Map phase names to the instrument tier they should use from defaults
PHASE_TIER_MAP: dict[str, str] = {
    "recon": "recon",
    "plan": "plan",
    "work": "work",
    "temperature_check": "work",
    "integration": "work",
    "play": "play",
    "inspect": "inspect",
    "aar": "aar",
    "consolidate": "consolidate",
    "reflect": "reflect",
    "maturity_check": "work",
    "resurrect": "resurrect",
}


class InstrumentResolver:
    """Resolves per-sheet instrument assignments with deep fallback chains.

    Produces a matrix of primary instruments and fallback chains for every
    sheet in the cycle. Free-tier models are the defaults; paid models are
    power-ups available in the fallback chain.
    """

    def resolve(
        self,
        agent_def: dict[str, Any],
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve instrument assignments for an agent.

        Args:
            agent_def: Agent definition with optional instrument overrides.
            defaults: Global defaults with instrument definitions per tier.

        Returns:
            Dict with keys:
                ``backend``: dict — primary backend config
                ``instrument_fallbacks``: list[str] — score-level fallbacks
                ``per_sheet_instruments``: dict[int, str] — per-sheet primary
                ``per_sheet_instrument_config``: dict[int, dict] — per-sheet config
                ``per_sheet_fallbacks``: dict[int, list[str]] — per-sheet fallback chains
        """
        default_instruments = defaults.get("instruments", {})
        agent_instruments = agent_def.get("instruments", {})

        per_sheet_instruments: dict[int, str] = {}
        per_sheet_config: dict[int, dict[str, Any]] = {}
        per_sheet_fallbacks: dict[int, list[str]] = {}

        # Build the full instrument catalog for tail fallbacks
        all_instruments = self._collect_all_instruments(default_instruments)

        for sheet_num in range(1, SHEETS_PER_CYCLE + 1):
            phase = SHEET_PHASE.get(sheet_num, "work")
            tier = PHASE_TIER_MAP.get(phase, "work")

            # Resolve: agent override > default for this tier
            resolved = self._resolve_for_tier(
                tier, agent_instruments, default_instruments
            )

            if resolved:
                primary = resolved.get("primary", {})
                instrument_name = primary.get("instrument", "")
                if instrument_name:
                    per_sheet_instruments[sheet_num] = instrument_name

                model = primary.get("model", "")
                provider = primary.get("provider", "")
                timeout = primary.get("timeout_seconds", 0)
                config: dict[str, Any] = {}
                if model:
                    config["model"] = model
                if provider:
                    config["provider"] = provider
                if timeout:
                    config["timeout_seconds"] = timeout
                if config:
                    per_sheet_config[sheet_num] = config

                # Build fallback chain: explicit fallbacks + full catalog tail
                fallbacks = self._build_fallback_chain(
                    resolved.get("fallbacks", []),
                    all_instruments,
                    exclude=instrument_name,
                )
                if fallbacks:
                    per_sheet_fallbacks[sheet_num] = fallbacks

        # Determine score-level primary backend
        work_tier = self._resolve_for_tier(
            "work", agent_instruments, default_instruments
        )
        primary_backend = self._to_backend_config(work_tier)

        # Score-level fallbacks from the full catalog
        score_fallbacks = list(all_instruments)

        return {
            "backend": primary_backend,
            "instrument_fallbacks": score_fallbacks,
            "per_sheet_instruments": per_sheet_instruments,
            "per_sheet_instrument_config": per_sheet_config,
            "per_sheet_fallbacks": per_sheet_fallbacks,
        }

    def _resolve_for_tier(
        self,
        tier: str,
        agent_instruments: dict[str, Any],
        default_instruments: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve the instrument config for a tier.

        Agent overrides take precedence over defaults.
        """
        # Check agent-level override for this tier
        if tier in agent_instruments:
            agent_tier = agent_instruments[tier]
            if isinstance(agent_tier, dict):
                # Merge: agent primary overrides, inherit default fallbacks
                default_tier = default_instruments.get(tier, {})
                merged = dict(default_tier) if isinstance(default_tier, dict) else {}
                merged.update(agent_tier)
                # Inherit fallbacks from defaults if not specified
                if "fallbacks" not in agent_tier and isinstance(default_tier, dict):
                    merged["fallbacks"] = default_tier.get("fallbacks", [])
                return merged

        # Fall back to defaults
        default_tier = default_instruments.get(tier, {})
        if isinstance(default_tier, dict):
            return dict(default_tier)
        return {}

    def _build_fallback_chain(
        self,
        explicit_fallbacks: list[dict[str, Any]],
        all_instruments: list[str],
        exclude: str = "",
    ) -> list[str]:
        """Build a complete fallback chain.

        Starts with explicit fallbacks, then appends all instruments
        from the catalog that aren't already in the chain.
        """
        chain: list[str] = []
        seen: set[str] = set()

        if exclude:
            seen.add(exclude)

        # Add explicit fallbacks first
        for fb in explicit_fallbacks:
            if isinstance(fb, dict):
                name = fb.get("instrument", "")
            elif isinstance(fb, str):
                name = fb
            else:
                continue
            if name and name not in seen:
                chain.append(name)
                seen.add(name)

        # Append remaining catalog instruments as tail
        for name in all_instruments:
            if name not in seen:
                chain.append(name)
                seen.add(name)

        return chain

    def _collect_all_instruments(
        self,
        default_instruments: dict[str, Any],
    ) -> list[str]:
        """Collect all unique instrument names from defaults."""
        instruments: list[str] = []
        seen: set[str] = set()

        for tier_config in default_instruments.values():
            if not isinstance(tier_config, dict):
                continue
            # Primary
            primary = tier_config.get("primary", {})
            if isinstance(primary, dict):
                name = primary.get("instrument", "")
                if name and name not in seen:
                    instruments.append(name)
                    seen.add(name)
            # Fallbacks
            for fb in tier_config.get("fallbacks", []):
                if isinstance(fb, dict):
                    name = fb.get("instrument", "")
                elif isinstance(fb, str):
                    name = fb
                else:
                    continue
                if name and name not in seen:
                    instruments.append(name)
                    seen.add(name)

        return instruments

    def _to_backend_config(self, tier_config: dict[str, Any]) -> dict[str, Any]:
        """Convert a tier config's primary to a backend config dict."""
        primary = tier_config.get("primary", {})
        if not isinstance(primary, dict):
            return {"type": "claude_cli", "skip_permissions": True, "timeout_seconds": 3600}

        instrument = primary.get("instrument", "claude-code")
        model = primary.get("model", "")
        timeout = primary.get("timeout_seconds", 3600)

        # Map instrument names to backend types
        backend_type_map: dict[str, str] = {
            "claude-code": "claude_cli",
            "openrouter": "openrouter",
            "gemini-cli": "claude_cli",
            "opencode": "claude_cli",
            "goose": "claude_cli",
        }

        backend: dict[str, Any] = {
            "type": backend_type_map.get(instrument, "claude_cli"),
            "skip_permissions": True,
            "timeout_seconds": timeout,
        }
        if model:
            backend["model"] = model

        return backend
