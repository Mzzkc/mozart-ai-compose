# Provider Compliance & Terms of Service Guide

**Date:** 2026-03-26
**Status:** DRAFT — best-effort analysis, not legal advice
**Last verified:** 2026-03-26

---

## Overview

Marianne orchestrates AI agents across multiple providers. Each provider has Terms of Service that govern how their tools and APIs may be used. This document summarizes the compliance status for every instrument Marianne supports or plans to support.

**Marianne's architecture:** Users provide their own API keys and credentials (BYOK — Bring Your Own Key). Marianne does not proxy, resell, or share API access. Marianne adds substantial proprietary value: score decomposition, multi-stage orchestration, validation gates, checkpoint/resume, cost tracking, learning store, fan-out, dependency DAGs, concert chaining, self-healing. This is an orchestration system, not a thin wrapper.

---

## Provider-by-Provider Analysis

### Anthropic

**Claude Code CLI (`claude_cli` backend): SAFE**

Claude Code is Anthropic's official product. Programmatic invocation via `-p` flag is explicitly supported and documented. The Agent SDK documentation actively encourages building tools on top of Claude Code. Marianne's `ClaudeCliBackend` uses standard `asyncio.create_subprocess_exec` — the intended usage pattern.

**Anthropic API (`anthropic_api` backend): SAFE with caveats**

Marianne is not a thin wrapper. It provides substantial proprietary logic on top of the Messages API. Users authenticate with their own API keys (`ANTHROPIC_API_KEY`). This is the BYOK pattern that Anthropic endorses.

**What IS prohibited:**
- Using OAuth tokens from Claude Free/Pro/Max subscriptions programmatically (outside Claude Code / Claude.ai)
- Building a SaaS that is primarily a UI over `messages.create` with no proprietary logic
- Routing requests through subscription credentials on behalf of third-party users

**Marianne's obligation:** Document that `anthropic_api` requires the user's own API key from console.anthropic.com. Warn if credentials appear to be OAuth tokens rather than API keys.

### OpenAI

**Codex CLI: SAFE**

OpenAI actively promotes Codex CLI for orchestration. The Agents SDK is OpenAI's own orchestration framework. `codex exec --full-auto` is designed for unattended operation.

**OpenAI API: SAFE**

Usage policies focus on content restrictions (no malware, no CSAM, etc.), not integration patterns. No anti-wrapper or anti-orchestration provisions found.

### Google

**Gemini CLI: SAFE**

Open source (Apache 2.0). Designed for automation (`--yolo`, `--approval-mode auto_edit`).

**Gemini API: SAFE**

Standard API terms apply. Note: unpaid tier allows Google to use prompts for product improvement. Paid tier does not.

**Marianne's obligation:** Document the paid/unpaid data usage distinction so users can make informed choices for sensitive work.

### Amazon

**Amazon Q Developer CLI: SAFE**

Governed by AWS Service Terms. No restrictions on programmatic invocation found. IAM-based access control applies — organizational policies may restrict usage.

### Ollama (Local)

**No TOS concern.** Open source, runs locally. No API keys, no cloud provider terms. Model-specific licenses (Llama, Mistral, etc.) are the user's responsibility, not Marianne's.

### Goose (Block)

**No TOS concern.** Open source (Apache 2.0). BYOK model — users provide their own LLM API keys.

### Aider

**No TOS concern.** Open source (Apache 2.0). BYOK via litellm.

### Cline

**No TOS concern.** Open source (Apache 2.0). BYOK model.

### Amp (Sourcegraph)

**REQUIRES INVESTIGATION.** Sourcegraph product with its own authentication and billing. Not enough public TOS information to make a definitive assessment. Do not add as a supported instrument until TOS is verified.

---

## Summary Table

| Provider | CLI Instrument | API Instrument | Risk | Action |
|----------|---------------|----------------|------|--------|
| Anthropic | Claude Code: SAFE | Anthropic API: SAFE (BYOK) | Low | Document BYOK, warn on OAuth tokens |
| OpenAI | Codex: SAFE | API: SAFE | Minimal | None |
| Google | Gemini: SAFE | API: SAFE | Minimal | Document paid/unpaid data distinction |
| Amazon | Q CLI: SAFE | n/a | Low | Document AWS terms apply |
| Local | Ollama: SAFE | n/a | None | Document model licenses are user's responsibility |
| OSS Tools | Goose, Aider, Cline: SAFE | n/a | None | None |
| Amp | INVESTIGATE | n/a | Unknown | Research before adding |

---

## Required Disclosures

Marianne MUST provide the following in documentation and at first-run:

1. **API key ownership:** "Marianne uses YOUR API credentials. You are responsible for compliance with your provider's terms of service."

2. **Cost responsibility:** "Marianne orchestrates multiple AI calls per job. Monitor your usage and set cost_limits in your score to prevent unexpected charges."

3. **Anthropic OAuth warning:** "Do not use OAuth tokens from Claude Free/Pro/Max subscriptions with the anthropic_api instrument. Use API keys from console.anthropic.com."

4. **Gemini data usage:** "Google's unpaid Gemini tier may use your prompts to improve their products. Use the paid tier for sensitive work."

5. **Model licenses:** "When using Ollama with open-weight models, ensure you comply with each model's license terms (e.g., Llama Community License, Mistral license)."

---

*This document is a best-effort analysis based on publicly available terms as of 2026-03-26. It is not legal advice. Terms change. Verify current terms before production use.*
