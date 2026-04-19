# Capability Audit: Compiler, Techniques, MCP, Code Mode, A2A

**Date:** 2026-04-19
**Triggered by:** 32-agent roster concert failure (05-migration session). All 32 generated scores failed at launch. This audit traces what's actually built vs aspirationally built across the systems involved.
**Scope:** Empirical findings only. Citations mandatory. Suggestions section is forward-looking.

---

## Executive Summary

Three findings reshape the picture:

1. **The compiler silently drops most of what its input declares.** Roster scores were generated from a rich input config (`full-roster-config.yaml`, 959 lines, full technique declarations + deep instrument fallback chains + cadenza directives) but emitted `canyon.yaml` with none of it. Three specific defects, all in `compiler/src/marianne_compiler/`.

2. **Cloudflare Dynamic Workers pattern is fully designed but completely unwired.** `src/marianne/execution/interface_gen.py` implements typed Python stubs + MCP-proxy-over-Unix-socket implementations exactly as Cloudflare Dynamic Workers describes. Zero call sites. Free models emitting code today get bare-Python execution with no MCP reach. The promise of "free models punch above their weight via code mode" is unfulfilled in current code.

3. **PROPER A2A reach today is effectively Gemini CLI only.** Verified across 4 CLIs (claude-code, gemini-cli, codex, opencode/crush). Only gemini-cli ships native A2A support (client-side, since v0.5.0, 2025-09-08). Anthropic, OpenAI, sst/anomalyco, and Charm have not shipped native A2A in their flagship CLIs. Marianne's `@delegate name: task` routing is custom prompt-pattern matching, not the A2A spec — and is not interoperable with anything external.

Taken together: the roster failure is mostly compiler surgery (small, scoped); the larger capability gap (free models with real reach + interop with external A2A agents) needs wiring work in `src/marianne/`, not the compiler.

---

## 1. The Composition Compiler

### What it's supposed to do

Take a semantic agent config YAML, produce complete Mozart scores per agent — wiring in identity, sheet structure, technique cadenzas, instrument resolution, validations, and prompt templates.

### What it actually emits

Reference: `workspaces/composition-compiler-build/agent-concert/canyon.yaml` (generated from `full-roster-config.yaml`).

Comparison:

| Aspect | Input declared | Output emitted |
|---|---|---|
| Per-phase techniques (8 with `kind:` + `phases:`) | Full block lines 128-152 | None — no technique cadenzas wired |
| Cadenza directives (`shared/active`, `directives`, token_budget) | Full block lines 154-162 | None — only identity files |
| Per-phase instruments with deep fallback chains (7 fallbacks per phase) | Full block lines 36-126 | Squashed to `backend.type: claude_cli, model: google/gemini-2.5-flash` (broken combo) |
| Per-sheet fallbacks | Implied from input | `[claude-code]` only (1 entry, not 7) |
| Prompt template (per design spec §3.3) | N/A — compiler responsibility | Missing entirely |

So the migration's Stage 4 did its job — it produced exactly the rich config that was specified. The compiler ate it and emitted scores stripped of nearly all wiring.

### Three defects with citations

**Defect 1 — No prompt template emitted.**

`compiler/src/marianne_compiler/pipeline.py:246-272` (`_build_prompt`) emits `variables`, `stakes`, `thinking_method`, but never builds a `template:` field. The composition compiler design spec at `docs/specs/2026-04-13-composition-compiler-design.md` §3.3 specifies score output must include a 12-phase Jinja template that wires declared techniques per phase.

Result: agents have no per-phase prompt invocation. Stakes/thinking_method exist as variables but nothing references them.

**Defect 2 — Technique manifests dropped silently.**

`compiler/src/marianne_compiler/techniques.py:42-109` (`TechniqueWirer.wire`) builds two outputs:
- `cadenzas: dict[int, list[dict]]` — per-sheet cadenza file references for technique `.md` files
- `technique_manifests: dict[int, str]` — auto-generated per-phase markdown manifests listing what techniques are available this phase, bucketed by kind (MCP / Protocols / Skills)

`compiler/src/marianne_compiler/pipeline.py:163-166` only merges the `cadenzas` portion. The `technique_manifests` are never written, never injected, never referenced. The entire per-phase "what techniques do you have right now" output is silently discarded.

**Defect 3 — `techniques_dir` has no default.**

`compiler/src/marianne_compiler/techniques.py:33-40` accepts `techniques_dir: Path | None = None`. If unset, `_find_technique_doc` (line 178-197) returns None for every lookup. With no default and no override in the migration pipeline, every cadenza injection becomes a no-op.

Combined with Defect 2: even if the input declares 8 techniques, the compiler produces zero technique-related context for the agent.

**Bonus — `InstrumentResolver` mangles backend config.**

`compiler/src/marianne_compiler/instruments.py:226-253` (`_to_backend_config`) uses a `backend_type_map` that squashes `opencode` to `claude_cli` while carrying the OpenRouter model string through. Result: `{type: claude_cli, model: google/gemini-2.5-flash}` — a broken combination. The conductor's CLI backend would attempt to invoke `claude` with a model name it doesn't recognize.

Same file: `per_sheet_fallbacks` is built from a single fallback per phase, not the deep chain the input declared. Loses the carefully-designed degradation strategy.

**Bonus — Duplicate technique source of truth.**

`compiler/src/marianne_compiler/technique_modules/` and `plugins/marianne/techniques/` contain the same five files (coordination, identity, mateship, memory-protocol, voice). Two repos, one truth, future drift guaranteed. **Catalog issue.**

---

## 2. Techniques — How They Actually Work

### Empirical behavior

A technique declaration like:
```yaml
techniques:
  coordination:
    kind: skill
    phases: [recon, plan, integration]
```

Is processed by `TechniqueWirer.wire()` at `compiler/src/marianne_compiler/techniques.py:42-109` to produce two things per declared phase:

1. **A cadenza entry** that injects the matching `.md` file: `{"file": "<techniques_dir>/coordination.md", "as": "skill"}` (or `as: "tool"` for non-skill kinds).

2. **A manifest entry** appended to that phase's auto-generated technique manifest markdown — categorized into "MCP Tools", "Protocols", or "Skills" sections based on `kind:`.

That's it.

### What `kind:` actually does

Empirically, `kind:` controls only:
- The `as:` label on the cadenza (`"skill"` or `"tool"`)
- Which manifest section the technique lands in

It does **not** trigger any special behavior — no MCP server configuration, no A2A endpoint registration, no protocol handler setup. The docstrings claim those things; the code only labels and buckets.

### Match with composer's mental model

The composer's stated want: "techniques are injections, more or less, per sheet." That **is** what the wirer is trying to do. The kinds are decorative today.

If you want `kind:` to be load-bearing, that's a design shift (separate scope from compiler remediation):
- `kind: mcp` would touch instrument config (configure shared pool MCP server for that sheet)
- `kind: protocol` would register A2A endpoints
- `kind: skill` would stay as injection

---

## 3. Shared MCP Pool

### Status: Wired end-to-end for MCP-native CLIs

The mechanism, traced through citations:

1. `src/marianne/daemon/mcp_pool.py:80-160` — `McpPoolManager` starts long-lived MCP server subprocesses, each on a Unix socket
2. `src/marianne/daemon/mcp_pool.py:237-295` — `generate_mcp_config_file()` writes a JSON config telling the CLI: "your MCP servers are commands like `python -m marianne.daemon.mcp_proxy_shim /tmp/mzt/mcp/X.sock`"
3. `src/marianne/daemon/baton/techniques.py:147-195` — per-phase resolver calls the pool for the active MCP techniques
4. `src/marianne/execution/instruments/cli_backend.py:231-241` — `set_mcp_config()` accepts the path
5. `src/marianne/execution/instruments/cli_backend.py:338-340` — `--mcp-config <path>` injected into subprocess args if the instrument profile declares `mcp_config_flag`
6. `src/marianne/daemon/mcp_proxy_shim.py:1-28` — when the CLI launches the shim as a stdio MCP server, the shim forwards stdio↔Unix socket bidirectionally

### Per-CLI status

Confirmed `mcp_config_flag` in profiles at `src/marianne/instruments/builtins/`:
- `claude-code.yaml:129` — `--mcp-config` ✓
- `opencode.yaml` — supports MCP via `mcpServers` config block (no `mcp_config_flag` shown in the snippet I reviewed; verify)
- `gemini-cli.yaml` — supports MCP (verify flag name)
- `codex-cli.yaml` — supports MCP (verify flag name)

Profiles without `mcp_config_flag` fall back to `mcp_disable_args` (line 341-342) which is a safe no-op (e.g., `--strict-mcp-config + empty config` for claude-code).

### Replaces per-CLI-harness MCP spinup?

**Yes** — that's the architectural value. Instead of every CLI cold-starting its own MCP servers per run, the pool warms them once and exposes a thin stdio→socket shim per CLI invocation. Real, wired, end-to-end.

---

## 4. Code Mode + Interface Generation

### code_mode.py: working as a sandbox runner

`src/marianne/execution/code_mode.py:94-188` (`CodeModeExecutor`):
- `_extract_code_blocks` (in technique_router) finds markdown-fenced executable code (python/bash/js/ts) in agent output
- `execute()` writes the block to a temp file in the workspace
- `_wrap_with_sandbox` wraps in bwrap sandbox (network-isolated, workspace bind-mount)
- `_run_code` runs `python3 file.py` (or interpreter for the language)
- Returns `CodeExecutionResult` with stdout/stderr/exit code/duration
- `render_code_mode_error` formats failures as markdown for retry context injection

**This works.** It's a real, wired-end-to-end post-execution sandbox bridge.

### interface_gen.py: completely unwired

`src/marianne/execution/interface_gen.py:1-26` documents intent: implement Cloudflare Dynamic Workers pattern. Inject typed Python stubs (~500 tokens) into prompt. Load real implementations into the sandbox alongside agent code. Implementations proxy MCP `tools/call` over Unix sockets.

`src/marianne/execution/interface_gen.py:78-229` implements `InterfaceGenerator` with:
- `generate_stubs()` — class signatures with type hints for prompt injection (lines 102-122)
- `generate_implementation()` — async classes that call `_mcp_call(socket_path, tool_name, arguments)` (lines 124-169)
- `_PROXY_HELPER` (lines 256-316) — minimal async JSON-RPC over Unix socket, embedded in every generated implementation (no Marianne import dependency in sandbox)

**Verified zero call sites.** `grep InterfaceGenerator|generate_implementation|generate_stubs|interface_gen` across `src/marianne/` returns only references inside `interface_gen.py` itself. Nothing imports it. Nothing calls it.

### What this means for free models today

A free OpenRouter model emitting:
```python
result = await workspace.read_file("foo.py")
```
…would get `NameError: name 'workspace' is not defined` because:
1. No technique declarations are mapped to `TechniqueDeclaration` objects
2. No stubs are injected into the prompt (model didn't even know `workspace` exists)
3. No implementations are written into the sandbox to make `workspace` importable

So free-tier models in code mode currently have access only to: stdlib + workspace files (via direct filesystem reads). No MCP servers. No technique reach.

### Wiring needed

Approximately 100-200 LoC bridging four touch points:
1. **Build `TechniqueDeclaration` lists** from the score's active MCP techniques per phase (currently done partially in `daemon/baton/techniques.py`, but the output isn't fed to interface_gen)
2. **Generate stubs** and inject into the prompt template as the technique manifest (currently the prompt template doesn't even exist — see Compiler Defect 1)
3. **Generate implementations** and write to a known path in the sandbox (e.g., `mzt_bindings.py` in the workspace)
4. **Modify `code_mode.py`** to make that file importable from agent code (e.g., write alongside the temp file, or set `PYTHONPATH`)

All four pieces exist in isolation. They just don't talk.

---

## 5. A2A — What Marianne Has

### Marianne's "A2A" mechanism

Traced through:
- `src/marianne/daemon/a2a/registry.py:58-103` — in-memory `AgentCardRegistry` (not persisted across daemon restarts)
- `src/marianne/daemon/a2a/inbox.py:156-198` — `A2AInbox.submit_task` stores tasks atomically with checkpoint state
- `src/marianne/daemon/a2a/inbox.py:315-351` — `render_pending_context()` renders pending tasks as markdown for cadenza injection
- `src/marianne/daemon/technique_router.py:131-134` — regex matches `@delegate (\w+)\s*:\s*(.+)` in agent output
- `src/marianne/daemon/technique_router.py:282-295` — extracts and routes back to target inbox

**This is custom prompt-pattern routing.** The agent reads a markdown list of pending tasks in its prompt, emits `@delegate name: task` somewhere in its output, the router pattern-matches it, the inbox stores it. Receiving agent picks it up via cadenza injection on next sheet.

It works. It's internal-only. It's not interoperable with anything external.

### Is it standard A2A?

**No.** Real A2A (Google's spec, donated to Linux Foundation June 2025, [a2a-protocol.org](https://a2a-protocol.org/latest/)) requires:
- Agent cards published at `.well-known/agent.json`
- JSON-RPC over HTTP/SSE
- Structured task objects with state machines (submitted/working/input-required/completed/failed/canceled)
- Streaming updates via SSE
- Authentication

Marianne's `@delegate` shares only the name "A2A". Calling it A2A is overclaiming. Honest naming would be "internal agent delegation" or "in-process A2A-shaped routing."

---

## 6. PROPER A2A Per CLI (verified 2026-04-19)

Verified via release notes, repo source code, and issue tracker analysis. Citations at end.

### Claude Code (Anthropic)

**Native A2A support: NO** (high confidence)

CHANGELOG.md through v2.1.114 (2026-04-18) — zero mentions of A2A or agent-to-agent. Repo code search returns no matches. Anthropic's "subagents" are in-process Claude-spawned, not A2A peers.

Community wrappers exist: `ericabouaf/claude-a2a`, `jcwatson11/claude-a2a` (unofficial).

**MCP support:** Yes, extensive. Latest release: v2.1.114 (2026-04-18).

### Gemini CLI (Google)

**Native A2A support: YES (client-side only)** (high confidence)

- A2A landed in **v0.5.0 (2025-09-08)**
- HTTP auth for A2A added in **v0.33.0 (2026-03-11)**
- Docs: `agent_card_url` / `agent_card_json` config; agent definitions in `.gemini/agents/*.md`
- Source: `packages/core/src/agents/agentLoader.ts`, `packages/core/src/agents/a2a-errors.ts`

**Scope:** A2A *client*. Gemini CLI delegates to remote A2A agents but does NOT expose itself as an A2A server.

**MCP support:** Yes. Latest release: v0.38.2 (2026-04-17).

Docs: https://geminicli.com/docs/core/remote-agents/

### Codex CLI (OpenAI)

**Native A2A support: UNKNOWN, leaning PARTIAL/experimental** (medium confidence)

- Issue [openai/codex#11980](https://github.com/openai/codex/issues/11980) (open, 2026-02-17) titled "feat(a2a): add ACP session management and token usage tracking to A2A server" references `codex-rs/a2a` and `codex-rs/mcp-server`
- BUT: direct GitHub API listing of `codex-rs/` shows **no `a2a` directory** on `main`
- Code search for `a2a` returns zero hits in shipped code
- Official docs (developers.openai.com/codex/) describe orchestration via MCP + Agents SDK + subagents — never mention A2A

**Best read:** A2A code may exist on contributor branches; not visibly shipped on `main`, not documented as a user feature. Treat as not-shipped for planning.

**MCP support:** Yes (Codex exposes itself as an MCP server). Latest release: rust-v0.121.0 (2026-04-15).

### OpenCode (sst → anomalyco) and Crush (charmbracelet)

**Native A2A support: NO** (high confidence)

- `sst/opencode` transferred to `anomalyco/opencode`
- Issue [anomalyco/opencode#3023](https://github.com/anomalyco/opencode/issues/3023) requesting A2A: closed 2026-02-06
- PR [anomalyco/opencode#10452](https://github.com/anomalyco/opencode/pull/10452) "feat: add A2A": **NOT merged** — auto-closed 2026-03-31 by stale-bot
- **Crush** (Charm successor): community focus is on **ACP** (Zed's Agent Client Protocol), not A2A
- Third-party wrapper exists: `shashikanth-gs/a2a-opencode` (now `a2a-wrapper`)

**MCP support:** Yes (both). OpenCode latest: v1.4.11 (2026-04-18).

### Industry context

- A2A donated to Linux Foundation June 2025
- **Google stack ships native A2A** (Gemini CLI, ADK, Gemini Enterprise, Cloud Run, Elastic integrations)
- **Non-Google CLI vendors** (Anthropic, OpenAI, sst/anomalyco, Charm) **have not shipped native A2A** in flagship CLIs
- **Competing protocol:** Zed's **ACP** (Agent Client Protocol) — Crush, Codex (`cola-io/codex-acp`), and others integrating ACP instead of or alongside A2A

**Realistic near-term assessment:** "Use proper A2A" today means Gemini CLI + Google ADK agents as first-class peers. Everything else needs community wrappers or the MCP-wrapper path described below.

---

## 7. What Needs to Happen Next

### Tier 1 — Unblock the roster concert (smallest scope, highest leverage)

**1.1 Compiler remediation (3 defects).** Fix `compiler/src/marianne_compiler/`:
- `pipeline.py:_build_prompt` — emit `prompt.template` (12-phase Jinja per design §3.3)
- `pipeline.py` (~163-166) — stop dropping `technique_result["technique_manifests"]`; merge into per-phase prompt context
- `techniques.py:__init__` — sensible default for `techniques_dir`, override-able
- `instruments.py:_to_backend_config` — correct backend type per instrument; preserve full per-phase fallback chains

**1.2 Delete the duplicate technique source of truth.** Pick one location (compiler-internal vs plugin-side) and remove the other. Document the choice. **File catalog issue.**

**1.3 Make `kind:` load-bearing (composer decision 2026-04-19).** Sketch — needs detailed design before implementation:

All techniques are per-sheet. The `kind:` controls *what* gets done per declared sheet, not whether/where:

- `kind: skill` → cadenza injection of the technique's `.md` file (current behavior). The bedrock case.
- `kind: mcp` → register the named MCP server with the shared pool for the declared phases; `--mcp-config` for that sheet's CLI invocation includes only the active servers. The technique's `config:` block can carry server-spawn args (command, args, env). With Tier 2, also feeds `InterfaceGenerator` so free models get bindings.
- `kind: protocol` → register a protocol adapter (A2A, ACP, or future) for those phases. Spawns the adapter's MCP-wrapper server (see Tier 3 design) so any MCP-capable CLI gains protocol reach. The technique's `config:` block names which protocol (`a2a`, `acp`, ...).

**Open design questions for Tier 1.3:**
- Where does `kind: mcp`'s config block live? In the technique's own `.md` frontmatter? In a sibling `.yaml`? Keeping it with the `.md` keeps the technique self-contained.
- How does kind-specific behavior get routed in the wirer? A dispatcher pattern (`KindHandler` protocol with implementers per kind) keeps it extensible per the "infinitely adaptable" principle.
- What's the migration path for existing techniques? They all have `kind:` declared today (cosmetically); making it load-bearing means existing `.md` files need their config to make the kind functional.

**Recommended:** before coding Tier 1.3, write a short design note covering the dispatcher pattern + per-kind behavior contract. Otherwise risk of baking decisions in code that need rework.

### Tier 2 — Free agents punch above their weight (Cloudflare Dynamic Workers wiring)

The goal: a free OpenRouter model emitting `await workspace.read_file("foo.py")` actually reads the file via the shared MCP pool. Currently this would `NameError` because no bindings exist in the sandbox.

**2.1 Build `TechniqueDeclaration` lists from active MCP techniques (per phase).**

- File: `src/marianne/daemon/baton/techniques.py` (currently has `generate_mcp_config_file` at line 147-195 — extend or add sibling)
- Add: `build_technique_declarations(active_techniques) -> list[TechniqueDeclaration]`
- Each MCP-kind technique that resolves to a pool socket becomes one `TechniqueDeclaration`
- Tool list per declaration must come from MCP server introspection (`tools/list` JSON-RPC call against the pool socket — can be cached at pool-warm time)
- New: pool needs a `list_tools(server_name) -> list[MCPToolSpec]` method that introspects each warmed MCP server once on startup

**2.2 Generate stubs and inject into the prompt template.**

- File: `compiler/src/marianne_compiler/pipeline.py:_build_prompt` (the same defect-1 location)
- Per phase, call `InterfaceGenerator().generate_stubs(declarations_for_this_phase)` and inject as `{{ technique_stubs }}` in the template
- Template snippet must clearly say something like: "These classes are available in your sandbox. Import them and call methods directly. Each method is async."
- Acceptance: a generated score's prompt template, when rendered for a phase with active MCP techniques, contains typed Python class stubs in a code-fenced block

**2.3 Generate implementations and persist to sandbox.**

- File: new — `src/marianne/execution/code_mode_bindings.py` (or extend `code_mode.py`)
- Before each `CodeModeExecutor.execute()` call, write `mzt_bindings.py` to the workspace using `InterfaceGenerator().generate_implementation(declarations, socket_paths={...})`
- `socket_paths` must be the same Unix socket paths the MCP pool exposes (already known to `McpPoolManager`)
- The `mzt_bindings.py` file must be importable from the agent's code — either by writing it alongside the temp code file in the same workspace dir (already the cwd), or via `PYTHONPATH=<workspace>`

**2.4 Sandbox bind-mount adjustment.**

- File: `src/marianne/execution/sandbox.py:SandboxWrapper.build_command` (referenced from `code_mode.py:365-383`)
- Confirm the workspace bind-mount is read-write so `mzt_bindings.py` is visible to the sandboxed Python process
- Confirm the MCP pool's Unix sockets are bind-mounted into the sandbox (currently the sandbox is `network_isolated=True` — Unix sockets are filesystem objects, not network, but verify they're not blocked)
- If sockets aren't reachable from inside sandbox, two options: bind-mount the socket dir, OR run the proxy helper as an unsandboxed subprocess that the sandboxed code talks to via stdin/stdout

**2.5 Wire it all from `musician.py`.**

- File: `src/marianne/daemon/baton/musician.py:144-177` (where code blocks already get extracted and run)
- Before `_execute_code_blocks`, ensure `mzt_bindings.py` has been written for the active phase's techniques
- The `CodeModeExecutor` needs to be constructed with the active phase's `TechniqueDeclaration` list (or have a `set_bindings(declarations)` method)

**Acceptance criteria for Tier 2:**

- Free model emits a code block that imports and calls a binding (e.g., `from mzt_bindings import workspace; await workspace.read_file("foo")`)
- Sandbox executes the code; the binding successfully calls the MCP pool over Unix socket
- Result is captured and injected back into the sheet output
- Free model with code mode + MCP techniques can complete a task that requires real tool use (read files, write files, query a tool)
- ~100-200 LoC across the touch points above

**Estimated complexity:** medium. The hardest piece is sandbox/socket interaction (2.4) — depends on bwrap config specifics. The rest is glue code between existing components.

### Tier 3 — PROPER A2A and ACP reach (interop with external systems)

**Composer decision 2026-04-19:** ship BOTH A2A and ACP support. Generalization principle: design a generic `ProtocolAdapter` abstraction; A2A and ACP are concrete implementations. New protocols plug in the same way.

**3.0 Generic protocol adapter design.**

- File: new — `src/marianne/protocols/` package
- `ProtocolAdapter` Protocol with methods like `serve()`, `discover(target)`, `send_task(target, payload)`, `get_task(task_id)`, `subscribe(task_id)`
- Per-protocol implementers: `A2AAdapter`, `ACPAdapter`, future `XYZAdapter`
- Each adapter knows its own spec details (HTTP/SSE vs stdio, JSON-RPC method names, agent card format)
- Common surface so the MCP wrapper layer (3.2) can be protocol-agnostic

**3.1 Run protocol-compliant servers in the conductor.**

- A2A: HTTP/SSE server publishing agent cards at `.well-known/agent.json`, implementing `tasks/send`, `tasks/get`, `tasks/sendSubscribe` per the [A2A spec](https://a2a-protocol.org/latest/), task state machine
- ACP: stdio-based server per [Zed ACP spec](https://github.com/zed-industries/agent-client-protocol), session management, capability negotiation
- Both run as subsystems of the conductor (alongside the existing daemon machinery)
- Each adapter exposes its own server lifecycle (`serve()`)

**3.2 Wrap protocol client operations as MCP servers.**

- Generate one MCP server per protocol: `marianne-a2a-client`, `marianne-acp-client`
- Tools follow a common naming pattern: `<protocol>.discover`, `<protocol>.send_task`, `<protocol>.get_task`, `<protocol>.subscribe`, `<protocol>.list_known_agents`
- Implementation calls the corresponding `ProtocolAdapter` method
- Add to shared MCP pool the same way every other MCP server is added

**3.3 Expose via technique declarations.**

- `techniques.protocols-a2a: { kind: protocol, phases: [...], config: { protocol: a2a } }` — wires A2A-client MCP server for those phases
- `techniques.protocols-acp: { kind: protocol, phases: [...], config: { protocol: acp } }` — wires ACP-client MCP server
- Per Tier 1.3, `kind: protocol` becomes meaningful via this dispatch

**3.4 Marianne agents become *discoverable* by external A2A/ACP clients (composer flagged 2026-04-19 as active interest — relevant to ongoing Marianne agent work).**

- The conductor's A2A server (3.1) publishes agent cards for currently-running Marianne agents
- The conductor's ACP server accepts external client sessions targeting specific Marianne agents
- Bidirectional interop — Marianne is a peer in the broader agent ecosystem
- Implication: external tools (Zed, Google ADK, anything A2A-aware) can discover and invoke Marianne agents directly without going through the conductor CLI

After this tier:
- Marianne agents can discover and delegate to external A2A agents (Google ADK + anyone implementing A2A) and external ACP agents (Zed-aligned tools)
- External A2A/ACP agents can discover and delegate to Marianne
- All MCP-capable CLIs under Marianne — including free OpenRouter models post-Tier-2 — get protocol reach
- Adding a third protocol later: write one `ProtocolAdapter` implementer + add a `kind: protocol` config option. No core changes.

### Tier 4 — Honesty layer

**4.1 Resolve `@delegate` mechanism (composer flagged 2026-04-19: didn't recognize it).**

The `@delegate name: task` text routing in `src/marianne/daemon/technique_router.py:131-134` + `src/marianne/daemon/a2a/` is **inherited code** the composer doesn't remember authoring. It currently provides Marianne-internal agent delegation via prompt-pattern matching. Two paths:

**Path A — DECIDED 2026-04-19.** Once Tier 3 ships, `@delegate` is deleted. The internal Marianne-to-Marianne delegation goes through the same A2A adapter as external delegation, via a `LocalProtocolAdapter` that lives in-process (no HTTP) but speaks the same `ProtocolAdapter` interface. Single delegation primitive. Cleaner architecture. No two-mechanism complexity.

**4.2 Update spec corpus.** Whatever Tier 3 actually delivers should be documented in `.marianne/spec/architecture.yaml` so the system's own self-description matches reality. Includes:
- Generic `ProtocolAdapter` abstraction
- A2A and ACP as concrete adapters
- `kind: protocol` semantics (post-Tier-1.3)
- Removal of `@delegate` if Path A taken

### Tier 5 — Default shippable agents (composer noted 2026-04-19: mostly solved)

Existing default agents are most of the way there. Remaining work is **presentation and UX**, not fundamental scope. Out of scope for this audit.

External-integration techniques (Reddit, Medium, email, moltbook, marketing) tracked separately in `docs/technique-ideas.md`.

### Tier 6 — Catalog issues to file

- Duplicate technique `.md` files at two locations (Tier 1.2 fixes this; file the issue regardless for tracking)
- ACP vs A2A protocol decision for industry interop (link to Tier 3 design)
- `interface_gen.py` zero-call-sites (link to Tier 2 work)

---

## 8. Acceptance Criteria for Tier 1 (the immediate work)

When the compiler is fixed, recompiling `workspaces/composition-compiler-build/full-roster-config.yaml` must produce scores where (using canyon as reference):

- `prompt.template` exists and contains 12-phase Jinja with per-phase technique invocations
- Technique manifests are wired as cadenzas per declared phase (a2a on recon/plan/work/integration/inspect/aar; coordination on recon/plan/integration; etc.)
- `backend.type` matches the declared instrument (opencode is opencode, not claude_cli)
- `per_sheet_fallbacks` carries the full declared chain per phase (7 entries, not 1)
- `mypy src/`, `ruff check src/`, and existing compiler tests still pass

---

## References

### File:line citations

- `compiler/src/marianne_compiler/pipeline.py:163-166, 246-272`
- `compiler/src/marianne_compiler/techniques.py:33-40, 42-109, 178-197`
- `compiler/src/marianne_compiler/instruments.py:226-253`
- `src/marianne/execution/code_mode.py:94-188`
- `src/marianne/execution/interface_gen.py:78-229, 256-316`
- `src/marianne/daemon/mcp_pool.py:80-160, 237-295`
- `src/marianne/daemon/mcp_proxy_shim.py:1-28`
- `src/marianne/daemon/baton/techniques.py:147-195`
- `src/marianne/daemon/technique_router.py:131-134, 282-295`
- `src/marianne/daemon/a2a/registry.py:58-103`
- `src/marianne/daemon/a2a/inbox.py:156-198, 315-351`
- `src/marianne/execution/instruments/cli_backend.py:231-241, 338-340`
- `src/marianne/instruments/builtins/claude-code.yaml:129`

### External citations (CLI A2A verification)

- A2A protocol spec: https://a2a-protocol.org/latest/
- Gemini CLI Remote Subagents: https://geminicli.com/docs/core/remote-agents/
- Gemini CLI changelog: https://geminicli.com/docs/changelogs/
- Gemini CLI a2a-errors.ts: https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/agents/a2a-errors.ts
- Claude Code CHANGELOG: https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md
- Codex A2A issue: https://github.com/openai/codex/issues/11980
- OpenCode A2A request (closed): https://github.com/anomalyco/opencode/issues/3023
- OpenCode A2A PR (auto-closed unmerged): https://github.com/anomalyco/opencode/pull/10452
- Third-party OpenCode A2A wrapper: https://github.com/shashikanth-gs/a2a-opencode
- Codex ACP wrapper: https://github.com/cola-io/codex-acp
- Crush ACP request: https://github.com/charmbracelet/crush/issues/990

### Related Marianne docs

- Composition compiler design: `docs/specs/2026-04-13-composition-compiler-design.md`
- Coordination plan (Marianne-internal A2A): `plugins/marianne/techniques/coordination.md`
- Mateship protocol: `plugins/marianne/techniques/mateship.md`
- Technique ideas log (running list): `docs/technique-ideas.md`
- Migration that produced the failure: `scores-internal/composition-compiler/05-migration.yaml`
- Failed input (rich): `workspaces/composition-compiler-build/full-roster-config.yaml`
- Failed output (impoverished): `workspaces/composition-compiler-build/agent-concert/canyon.yaml`
