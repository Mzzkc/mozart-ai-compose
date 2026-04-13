# Instrument Forge — Specification

**Date:** 2026-04-13
**Status:** Draft
**Purpose:** Build a system that dynamically discovers AI providers and tools, probes them for model metadata, and generates valid InstrumentProfile YAML files for Marianne.

---

## Problem

Instrument profiles are hand-written YAML with hardcoded model lists that go stale. Adding a new instrument requires manually researching a provider's API, writing YAML, and validating it against the schema. There is no way to discover what's available, verify what works, or keep things current. The landscape of AI providers, aggregators, CLI tools, local servers, and cloud platforms is massive and constantly changing. Marianne needs a system that handles this dynamically.

## Goal

An **Instrument Forge** — a modular Python system that:

1. **Discovers** providers across the full landscape: direct API providers, model aggregators, CLI coding agents, local/self-hosted servers, cloud platforms, and whatever emerges next
2. **Probes** each discovered provider to extract model metadata: IDs, context windows, output limits, pricing, capabilities, auth requirements
3. **Detects** installed CLI tools and infers their instrument configuration (flags, output formats, headless execution support)
4. **Generates** valid InstrumentProfile YAML from discovered data, validated against the real schema
5. **Integrates** as `mzt instruments sync` — re-runnable to refresh profiles as the landscape changes
6. **Surfaces** in `mzt doctor` (hints when profiles are missing or stale) and `mzt init` (suggests running sync after project scaffolding)

The forge must handle an unknown and growing number of providers. It cannot depend on a hardcoded list. It must work when some providers are unreachable (missing keys, down services, uninstalled tools) and succeed on whatever is available.

## Where Things Live

- **Forge code:** `src/marianne/instruments/forge/` — Protocol-based provider adapters, probers, profile generator, discovery engine
- **Generated profiles:** Written to `~/.marianne/instruments/` (org-level, default) or `.marianne/instruments/` (venue-level with flag) — uses the existing loader override semantics
- **Built-in profiles** at `src/marianne/instruments/builtins/` remain hand-curated and untouched by the forge
- **CLI integration:** `mzt instruments sync` subcommand on the existing `instruments` command group, hints in `doctor`, suggestion in `init`

## Existing Infrastructure

The system already has:
- `InstrumentProfile` schema at `src/marianne/core/config/instruments.py` (Pydantic v2, validated, covers CLI and HTTP instruments)
- `InstrumentProfileLoader` at `src/marianne/instruments/loader.py` — loads from builtins → `~/.marianne/instruments/` → `.marianne/instruments/`, later overrides earlier
- `InstrumentRegistry` at `src/marianne/instruments/registry.py` — name→profile dict, used by the conductor for instrument resolution
- `PluginCliBackend` at `src/marianne/execution/instruments/cli_backend.py` — executes any CLI instrument from a profile YAML
- `mzt instruments list|check` at `src/marianne/cli/commands/instruments.py`
- `mzt doctor` at `src/marianne/cli/commands/doctor.py` — checks binary availability
- `mzt init` at `src/marianne/cli/commands/init_cmd.py` — scaffolds projects

## Scope of Discovery

The forge must be capable of discovering providers across at minimum these domains, and designed to extend to new ones:

- **Direct API providers** — Anthropic, OpenAI, Mistral, Cohere, DeepSeek, xAI, AI21, Perplexity, Z.ai, and others with model listing endpoints
- **Model aggregators** — OpenRouter, Together AI, Fireworks AI, Groq, Replicate, Deepinfra, and others that surface hundreds of models through unified APIs
- **CLI coding agents** — Claude Code, Crush, OpenCode, Goose, Gemini CLI, Codex CLI, Aider, Cline, and others that support headless prompt execution
- **Local/self-hosted servers** — Ollama, LM Studio, vLLM, LocalAI, llama.cpp server, TGI, and others that expose local model APIs
- **Cloud platforms** — AWS Bedrock, Azure OpenAI, Google Cloud Vertex AI, and others with platform-specific auth and model access

This is not an exhaustive list. The forge's value comes from its ability to discover beyond what it was explicitly told about.

## Design Constraints

- Follow all project conventions in `.marianne/spec/conventions.yaml`: async, Pydantic v2, Protocol-based, defensive, logged
- Follow all project constraints in `.marianne/spec/constraints.yaml`: tests pass, mypy strict, ruff clean, backward compatible
- Generated profiles must validate against the real `InstrumentProfile` schema — import it, don't duplicate it
- Partial failure is expected — skip unreachable providers, log why, continue
- No new dependencies without justification (MN-005)
- The forge is a product feature for the world, not a local development tool — handle auth diversity (API keys, OAuth, IAM, keyless/free, subscriptions)

## What This Spec Does NOT Prescribe

- Internal module structure — the implementing agent should read the codebase and design what fits
- Specific provider adapter implementations — discover the landscape first, then design adapters
- Fan-out counts, stage counts, or score structure — the composer should analyze forces and derive structure
- Model lists or default selections — the forge discovers these dynamically
- Specific CLI flag mappings — the forge detects these from installed tools

## Success Criteria

1. `mzt instruments sync` runs, discovers providers, generates valid profiles
2. Generated profiles load through the existing `InstrumentProfileLoader` without modification
3. `mzt instruments list` shows both built-in and forge-generated profiles
4. `mzt doctor` suggests sync when appropriate
5. The forge can be re-run and updates profiles with current data
6. Tests cover the forge modules, including adversarial cases (malformed APIs, unreachable providers, broken CLI tools)
7. All quality gates pass: pytest, mypy --strict, ruff check
