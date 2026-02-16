# Mozart AI Compose Documentation

Orchestration tool for Claude AI sessions — define multi-stage workflows as YAML scores, and let Mozart handle execution, retries, rate limiting, validation, and cross-stage context.

## Suggested Reading Paths

**New Users:**
1. [Getting Started](getting-started.md) — Installation, daemon setup, first job
2. [Score Writing Guide](score-writing-guide.md) — How to author Mozart scores
3. [Examples Directory](../examples/) — 24 working score configurations

**Score Authors:**
1. [Score Writing Guide](score-writing-guide.md) — Comprehensive authoring guide
2. [Configuration Reference](configuration-reference.md) — Every config field documented
3. [Examples Directory](../examples/) — Real-world score patterns
4. [Mozart Score Playspace](https://github.com/Mzzkc/mozart-score-playspace) — Creative showcase with real output

**System Administrators:**
1. [Daemon Guide](daemon-guide.md) — Conductor setup, configuration, troubleshooting
2. [CLI Reference](cli-reference.md) — Complete command reference
3. [Limitations](limitations.md) — Known limitations and workarounds

## Documentation

### Getting Started

- [**Getting Started**](getting-started.md) — Installation, daemon setup, first job
- [**Score Writing Guide**](score-writing-guide.md) — Comprehensive guide to authoring Mozart scores

### Reference

- [**CLI Reference**](cli-reference.md) — All Mozart commands with examples and options
- [**Configuration Reference**](configuration-reference.md) — Complete YAML schema documentation
- [**Limitations**](limitations.md) — Known constraints and recommended workarounds

### System Guides

- [**Daemon Guide**](daemon-guide.md) — Conductor daemon setup, operation, and debugging
- [**MCP Integration**](MCP-INTEGRATION.md) — Mozart's MCP server for external tool integration
- [**Mozart Reference**](mozart-reference.md) — Architecture overview and internal concepts

### Learning & Internals

- [**Distributed Learning Architecture**](DISTRIBUTED-LEARNING-ARCHITECTURE.md) — Cross-job pattern learning system

### Research (Internal)

- [**Token Compression Strategies**](research/TOKEN-COMPRESSION-STRATEGIES.md) — Research into reducing token costs (not yet implemented)
- [**Opus Convergence Analysis**](research/OPUS-CONVERGENCE-ANALYSIS.md) — Meta-analysis comparing Mozart and RLF evolution patterns

### Examples

- [**Examples Directory**](../examples/) — 24 working Mozart score configurations
- [**Mozart Score Playspace**](https://github.com/Mzzkc/mozart-score-playspace) — Creative showcase: philosophy, worldbuilding, education, and more

## Repository

- [**GitHub Repository**](https://github.com/Mzzkc/mozart-ai-compose) — Source code, issues, and contributions
