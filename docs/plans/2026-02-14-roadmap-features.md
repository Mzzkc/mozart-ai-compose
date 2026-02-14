# Mozart Roadmap Features — Discussion Tracker

**Started:** 2026-02-14
**Status:** Complete — all 22 features discussed and filed as GitHub issues #49–#70

---

## GitHub Issue Index

| # | Issue | Feature | Tier |
|---|-------|---------|------|
| 1 | [#49](https://github.com/Mzzkc/mozart-ai-compose/issues/49) | Live Job Progress + Observability | tier-1-foundation |
| 2+6b | [#50](https://github.com/Mzzkc/mozart-ai-compose/issues/50) | Conductor-First Architecture | tier-1-foundation |
| 3 | [#51](https://github.com/Mzzkc/mozart-ai-compose/issues/51) | Conductor Configuration (ClamAV-style) | tier-1-foundation |
| 4 | [#52](https://github.com/Mzzkc/mozart-ai-compose/issues/52) | Distributed Network Execution | tier-6-horizon |
| 5 | [#53](https://github.com/Mzzkc/mozart-ai-compose/issues/53) | Context Injection — Prelude & Cadenza | tier-2-intelligence |
| 6a | [#54](https://github.com/Mzzkc/mozart-ai-compose/issues/54) | Musical Theming Refresh | tier-1-foundation |
| 7 | [#55](https://github.com/Mzzkc/mozart-ai-compose/issues/55) | Semantic Learning via LLM | tier-2-intelligence |
| 8 | [#56](https://github.com/Mzzkc/mozart-ai-compose/issues/56) | Dashboard — Vibe Coding Studio | tier-5-experience |
| 9 | [#57](https://github.com/Mzzkc/mozart-ai-compose/issues/57) | Score Generation — Concert Library | tier-4-platform |
| 10 | [#58](https://github.com/Mzzkc/mozart-ai-compose/issues/58) | Workspaces Just Work | tier-3-reliability |
| 11 | [#59](https://github.com/Mzzkc/mozart-ai-compose/issues/59) | Fix Pause/Resume | tier-3-reliability |
| 12 | [#60](https://github.com/Mzzkc/mozart-ai-compose/issues/60) | Webhook Integration Refresh | tier-3-reliability |
| 13 | [#61](https://github.com/Mzzkc/mozart-ai-compose/issues/61) | Validation System Update | tier-3-reliability |
| 14 | [#62](https://github.com/Mzzkc/mozart-ai-compose/issues/62) | Flight Checks | tier-2-intelligence |
| 15 | [#63](https://github.com/Mzzkc/mozart-ai-compose/issues/63) | Self-Healing Improvements | tier-2-intelligence |
| 16 | [#64](https://github.com/Mzzkc/mozart-ai-compose/issues/64) | Human-in-the-Loop | tier-5-experience |
| 17 | [#65](https://github.com/Mzzkc/mozart-ai-compose/issues/65) | More Instruments | tier-4-platform |
| 18 | [#66](https://github.com/Mzzkc/mozart-ai-compose/issues/66) | Tool/MCP/Skill Management | tier-4-platform |
| 19 | [#67](https://github.com/Mzzkc/mozart-ai-compose/issues/67) | Conductor Scheduler | tier-4-platform |
| 20 | [#68](https://github.com/Mzzkc/mozart-ai-compose/issues/68) | Marketing & Presence | tier-5-experience |
| 21 | [#69](https://github.com/Mzzkc/mozart-ai-compose/issues/69) | Repo Cleanliness | tier-5-experience |
| 22 | [#70](https://github.com/Mzzkc/mozart-ai-compose/issues/70) | Email Management Concert | tier-6-horizon |

---

## Feature Details

### Feature 1: Live Job Progress + Observability (#49)

**Summary:** The conductor should have full, independent knowledge of every running job — not just self-reported status from the runner.

**Components:**

1. **Runner callbacks** — Synchronous, critical-path events (sheet.started, sheet.completed, sheet.retrying, job.cost_update) reported to the conductor via a callback protocol on RunnerContext.

2. **Job Observer** — Async co-task per job using inotify (via `watchfiles`) + `psutil`. Produces a timestamped timeline of filesystem changes, process tree, and resource usage. Unprivileged, no root required. Inspired by ClamAV on-access scanning philosophy — independent verification of what the agent is actually doing.

3. **Completion snapshots** — Conductor-owned durable storage at `~/.mozart/snapshots/{job_id}/{timestamp}/` capturing:
   - Final checkpoint state + config
   - Observer timeline (timestamped JSONL)
   - Process tree and resource usage
   - Filesystem changes log
   - Git context (branch, commits, diffs)
   - Job logs
   - TTL-managed (configurable, default 7 days)

4. **Registry expansion** — `JobRecord` gains `current_sheet`, `total_sheets`, `last_event_at`, `log_path`, `snapshot_path` columns for fast status queries without hitting the workspace.

5. **Event bus** — Async pub/sub for downstream consumers (dashboard, learning, webhooks). Events carry snapshot references, not live workspace paths. Decouples analysis from workspace lifecycle.

**Key design principle:** Conductor has independent knowledge via observer, not just runner self-reporting. Snapshots solve the lifecycle race where workspaces get cleaned up before analysis completes.

**Architectural decisions:**
- Observer depth: inotify + psutil (unprivileged, works everywhere)
- Snapshot-based approach for post-mortem analysis (learning system reads snapshots, never touches live workspaces)
- Completion barrier: conductor gates cleanup/chaining behind snapshot capture

---

### Feature 2 + 6b (Combined): Daemon-First Architecture (#50)

**Summary:** Mozart works like Docker. The conductor is always running. All CLI commands are thin clients that talk to the conductor via IPC. No daemon, no status. Period.

**Key changes:**

1. **`mozartd` goes away** — replaced by `mozart start` / `mozart stop` / `mozart restart`. One binary, one CLI.

2. **All job commands route through conductor:**
   - `mozart status job-id` → asks conductor (conductor knows workspace, has live data from Feature 1)
   - `mozart resume job-id` → asks conductor to resume (no workspace needed)
   - `mozart pause job-id` → asks conductor to pause
   - `mozart diagnose job-id` → asks conductor, uses snapshot data
   - `mozart errors job-id` → asks conductor

3. **`--workspace` becomes hidden optional override**, not a requirement. 99% of use cases never need it.

4. **Commands that work without conductor:** `mozart start`, `mozart validate`, `mozart version`. Everything else requires the conductor.

5. **No orphan job support** — if the conductor doesn't know about a job, it doesn't exist. Clean cut, no dual-path complexity.

6. **Removed code paths:**
   - `find_job_state()` / `require_job_state()` filesystem scanning in CLI helpers
   - Dual-mode logic where some commands talk to daemon, others don't
   - `mozartd` binary entry point

**Architectural decisions:**
- Daemon required for all operations (like Docker requires dockerd)
- No fallback to filesystem scanning
- No scan-and-adopt for pre-daemon jobs

---

### Feature 3: Conductor Configuration (ClamAV-style) (#51)

**Summary:** Single YAML config file at `~/.mozart/conductor.yaml` controls all conductor behavior. Live reload via SIGHUP. Config dump via `mozart config`.

**Components:**

1. **Config file loading** — `DaemonConfig` (already a Pydantic model) loads from `~/.mozart/conductor.yaml` on `mozart start`. Override with `mozart start --config /path/to/file.yaml`.

2. **SIGHUP reload** — Conductor re-reads config without restart. Applies changes to concurrency limits, resource limits, log level, observer settings, scheduler settings. Does NOT change socket path (requires restart).

3. **Config dump command** — `mozart config` shows active configuration with source annotations (default vs file vs override). Similar to ClamAV's `clamconf`.

4. **Config validation** — `mozart config --check` validates a config file without starting the conductor.

5. **Expandable structure** — Config model grows as features land:
   - `observer:` section for Feature 1 (snapshot TTL, path, enabled)
   - `learning:` section for Feature 7 (semantic learning backend/model)
   - `scheduler:` section for Feature 19 (maintenance scores)
   - `network:` section for Feature 4 (distributed execution, future)

**Key pattern:** One file, one truth. No environment variables or CLI flags that silently override config. The file is version-controllable and deployable.

**Implementation notes:**
- Pydantic model already exists and is well-structured
- Gap is: YAML file loading, SIGHUP handler, config dump CLI command
- Relatively straightforward plumbing work

---

### Feature 4: Distributed Network Execution (#52)

**Summary:** Single conductor scheduling jobs across multiple networked workers. Elevates Mozart from a local tool to distributed infrastructure.

**Vision:** Conductor becomes a control plane, workers become execution nodes. Sheets are scheduled across available hardware based on worker capabilities and load. Connection-hardened with mTLS.

**Phased approach (details deferred — north-star vision):**
- Phase 1: TCP transport option alongside Unix socket (same JSON-RPC protocol)
- Phase 2: Worker registration and capability advertisement
- Phase 3: Network-aware scheduling (latency, load, backend availability)
- Phase 4: Connection hardening (mTLS, certificate management, worker identity)
- Phase 5: Workspace distribution strategies (deferred — multiple options: ship workspace, shared storage, git-based)

**Hard problems (to be designed later):**
- Workspace distribution across machines
- Partial failure / worker death mid-sheet
- State consistency with multiple workers modifying codebases
- Secret management (API keys on workers)
- Statefulness of AI workloads (unlike containers, can't trivially reschedule)

**Scope note:** This explicitly reverses the "Distributed execution — Single-machine tool" non-goal from the project brief. The brief needs updating to reflect the evolved vision.

**Priority:** Long-term / north-star. Requires Features 1, 2+6b, 3 as prerequisites.

### Feature 5: Context Injection — Prelude & Cadenza (#53)

**Summary:** First-class mechanism for injecting file content into sheets. Mozart handles injection before the agent sees the prompt. Separated into context, skills, and tools.

**YAML structure:**
```yaml
sheets:
  total: 5

  prelude:                      # Shared context for ALL sheets
    - file: docs/architecture.md
      as: context               # Background knowledge
    - file: .claude/skills/debugging.md
      as: skill                 # Capability/methodology
    - file: tools/lint.sh
      as: tool                  # Executable action

  cadenzas:                     # Per-sheet specific injections
    3:
      - file: "{{ workspace }}/02-output.md"
        as: context             # Sheet 3 gets sheet 2's output
    5:
      - file: tests/results.json
        as: context
```

**Naming:**
- **Prelude** — shared context/tools/skills for all sheets (introductory material)
- **Cadenza** — per-sheet specific injections (the soloist's moment)

**Injection categories:**
- `as: context` — background knowledge, injected as informational context
- `as: skill` — capability/methodology, injected as instructions
- `as: tool` — executable action, injected as available tooling

**Key design points:**
- Mozart reads files and injects content before prompt construction
- Different categories go to different sections of the prompt (not one blob)
- Jinja templating works in file paths (e.g., `{{ workspace }}/output.md`)
- Files are read at sheet execution time (not config parse time) so dynamic outputs from earlier sheets are available

### Feature 6a: Musical Theming Refresh (#54)

**Summary:** Comprehensive rename to musical terminology across the entire codebase. Not cosmetic — this establishes Mozart's identity and makes the system more intuitive.

**Renames:**

| Current | New | Rationale |
|---------|-----|-----------|
| Daemon / mozartd | **Conductor** | The person coordinating all musicians |
| Fan-out | **Harmony** | Multiple voices playing simultaneously |
| Job | **Concert** | A complete performance |
| Backend | **Instrument** | The tool the performer plays (Claude CLI, API, Qwen) |
| Runner | **Performer** | The entity that plays the music |
| Workspace | **Stage** | The physical space where the performance happens |

**Keep as-is:**
- **Score** — already musical
- **Sheet** — already musical
- **Validation** — universally understood, no musical equivalent needed
- **Prelude / Cadenza** — new terms from Feature 5

**Per-sheet instrument selection:**
Each sheet can declare its own instrument (backend/model), making sheets truly agentic:
```yaml
sheets:
  total: 5
  instrument: claude-cli          # Default instrument for all sheets
  cadenzas:
    3:
      instrument: claude-api       # Sheet 3 uses API instead of CLI
      model: claude-sonnet-4-5     # Different model for this sheet
    5:
      instrument: qwen-code        # Sheet 5 uses a different backend entirely
```

This enables heterogeneous scores where different sheets use different models/backends based on their task requirements (e.g., cheap model for boilerplate, expensive model for architecture decisions).

**Blast radius:** Large mechanical refactor — package names, CLI commands, config keys, log names, docs, tests. Should be done as a dedicated effort, not mixed with feature work.

**Decision:** Do LAST — after all other features land. Avoids merge conflicts during active development. One big sweep at the end.

### Feature 7: Semantic Learning via LLM (#55)

**Summary:** Conductor runs LLM-based analysis after each sheet completes, producing semantic insights that are stored in the learning database and injected into future sheets. Learning is always on — it's conductor infrastructure, not job configuration.

**How it works:**
1. Sheet completes (success or failure)
2. Conductor captures snapshot (Feature 1)
3. Conductor asynchronously runs a learning prompt against the snapshot using a configurable model (default: Claude Sonnet)
4. Learning prompt examines: prompt given, agent output, validation results, observer timeline, historical patterns
5. Structured JSON output stored in learning DB as semantic insights
6. Future sheets receive relevant insights as injected context (existing pattern injection infrastructure)

**Learning prompt asks:**
- Why did this succeed/fail?
- What should future similar sheets know?
- What prompt improvements would help?
- Were there anti-patterns (wasted time, wrong approach, unnecessary retries)?
- (Feature 14 integration): How effective was the prompt? Where did the agent deviate and why?

**Key changes from current learning system:**
1. **Learning defaults to ON** — conductor is running → learning is running. No opt-in.
2. **Semantic analysis via LLM** replaces statistical pattern detection as primary mechanism
3. **Conductor-scoped** — learns from ALL jobs, not just opt-in ones
4. **Snapshot-based** — reads from snapshots (Feature 1), doesn't touch live workspaces
5. **Existing infrastructure preserved** — SQLite store, pattern CRUD, prompt injection all still work. Semantic learning produces better patterns.

**Configuration (in conductor.yaml, Feature 3):**
```yaml
learning:
  semantic: true
  model: claude-sonnet-4-5     # Configurable, default Sonnet
  analyze_on: [success, failure]  # When to run analysis
  max_concurrent_analyses: 3    # Don't overwhelm with analysis tasks
```

**Current system diagnosis:** Learning infrastructure is comprehensive (15 SQLite tables, pattern detection, pattern injection, CRUD operations) but functionally dormant — `learning.enabled` defaults to `false`, every flag is opt-in, CLI mode doesn't auto-create global store. All infrastructure is preserved; semantic learning just makes it actually produce useful results.

**Dependencies:** Feature 1 (snapshots), Feature 3 (conductor config)

### Feature 8: Working Dashboard — Vibe Coding Studio (#56)

**Summary:** The dashboard becomes Mozart's primary interface — a full workbench for taking ideas from concept to concert to production. Not just monitoring. A recording studio mixing board.

**Three tiers:**

1. **Monitoring** — Real-time view of running concerts, sheet progress, observer timelines (Feature 1), cost tracking, learning insights (Feature 7). WebSocket-driven live updates from conductor event bus.

2. **Control** — Pause/resume/cancel concerts, submit new scores, modify running concerts, inject context mid-execution (cadenza injection from Feature 5). All operations route through conductor (Feature 2).

3. **Workbench** — The differentiator:
   - **Idea → Concert:** User describes what they want, Mozart generates a score (Feature 9). Visual score editor with sheet dependencies, instrument selection per sheet, prelude/cadenza configuration.
   - **Direct agent interaction:** Chat with a running sheet. See what the agent is doing (observer timeline), inject guidance, approve/reject outputs.
   - **Human-in-the-loop interventions:** Conductor notifies dashboard when confidence is low or decisions need human input (Feature 16). Dashboard provides structured decision UI, not just "approve/reject."
   - **Learning review:** See what the semantic learning system (Feature 7) has learned. Approve/reject/edit learned patterns before they're injected into future sheets.
   - **Vibe coding studio:** Take an idea, watch it go from concept to working code across multiple sheets, intervene when needed, approve the final result. Like pair programming with an orchestra.

**Tech decisions needed (separate design session):**
- Frontend framework (React? Svelte? Plain HTML+WebSocket?)
- State management for real-time updates
- Authentication (single-user? multi-user? team?)
- How agent interaction works technically (WebSocket relay to backend process?)

**Dependencies:** Features 1, 2+6b, 5, 7, 9, 16 — this is the capstone that integrates everything.

**Priority:** High in importance (it's the face of Mozart), but depends on many other features landing first. Current stalled implementation should be scrapped and rebuilt around the conductor architecture.

### Feature 9: Score Generation — Concert Library & Composer (#57)

**Summary:** A library of proven concert designs that take ideas from concept to production. Template-based pipeline with LLM refinement and human-in-the-loop for key decisions.

**Pipeline:** Plan/brainstorm → Template selection → LLM refinement → Human review → Execute → Quality passes

**Concert library (built-in, proven designs):**

| Concert Type | Purpose | Pattern |
|---|---|---|
| **Feature Build** | Idea → working feature | Plan → Research → Implement → Test → Wire-up → Quality |
| **Quality Continuous** | Ongoing code quality | Analyze → Fix → Verify → Repeat (self-chaining) |
| **Evolution** | Self-improvement cycle | Discover → Evaluate → Implement → Validate → Score update |
| **Wire-up & Finish** | Complete partial work | Diagnose gaps → Plan fixes → Implement → Integration test |
| **Evaluation** | Assess system/code quality | Criteria → Analysis → Report → Recommendations |

**How `mozart compose` works:**
1. User: `mozart compose "add authentication to the API"`
2. Mozart selects appropriate concert template (Feature Build)
3. LLM refines the template: fills in specific sheets, prompts, instruments, dependencies
4. Dashboard (Feature 8) presents the plan for human review
5. Human approves/modifies (default: decisions require approval)
6. Concert executes with quality passes at the end

**Key design points:**
- Concert templates are proven patterns, not generated from scratch
- LLM fills in details but doesn't design the pipeline structure
- Human-in-the-loop by default for architectural/design decisions
- Quality runs (wire-up, fix, quality continuous) are standardized concert types
- Templates ship with Mozart and can be community-contributed

### Feature 10: Workspaces Just Work (#58)

**Summary:** Fix all workspace caveats so users never lose work or fail to start jobs due to workspace issues. The conductor (Feature 2) owns workspace lifecycle.

**Known issues to fix:**
- Workspace path mismatches causing failed jobs
- Worktree isolation gotchas (cleanup, branch conflicts)
- Losing work when workspaces are reused or archived
- Fresh runs not properly handling existing workspace state
- Workspace creation failures not caught early enough

**Key design point:** With the conductor owning all job metadata (Feature 2), workspace management becomes a conductor responsibility. The conductor creates, tracks, and manages workspaces. Users never need to think about them. `--workspace` becomes a hidden expert option.

**Connects to:** Feature 2 (conductor knows all workspaces), Feature 1 (snapshots preserve workspace state before cleanup)

### Feature 11: Fix Pause Feature (#59)

**Summary:** Pause/resume is janky and incomplete. The score that was supposed to implement it never finished. Needs a complete rework.

**Known issues:**
- Signal file mechanism is fragile (polling-based, no acknowledgment)
- Pause doesn't properly preserve mid-sheet state
- Resume after pause may replay work
- No integration with conductor (Feature 2) — pause should be a conductor operation
- Dashboard (Feature 8) can't initiate or monitor pause operations

**Fix approach:** With the conductor architecture (Feature 2), pause becomes a conductor command. Conductor signals the runner, runner acknowledges, conductor records the pause point. Resume goes through conductor which knows exactly where the concert stopped.

### Feature 12: Webhook Integration Refresh (#60)

**Summary:** All webhook integrations (notifications) are untested and likely outdated. Full refresh needed.

**Scope:**
- Audit all notification backends (Desktop, Slack, Webhook) for functionality
- Write integration tests for each backend
- Update to current API versions (Slack API, etc.)
- Connect to conductor event bus (Feature 1) — webhooks subscribe to events
- Add new integrations: Discord, Microsoft Teams, email (connects to Feature 22)

**Priority:** Medium — important for adoption but not blocking core architecture work.

### Feature 13: Validation System Update (#61)

**Summary:** Update `mozart validate` to catch common issues and gotchas discovered through development experience. Currently checks are too basic.

**New checks to add (based on development pain points):**
- V-NEW: Detect sheets with prompts that are too short (likely incomplete)
- V-NEW: Warn on missing prelude/cadenza files (Feature 5)
- V-NEW: Check instrument (backend) availability before execution
- V-NEW: Validate workspace permissions and disk space
- V-NEW: Check for circular dependencies in sheet DAGs
- V-NEW: Warn on scores that will likely hit rate limits (too many sheets, too fast)
- V-NEW: Detect reuse of workspace paths across concurrent scores
- Fix V101 false positive for `stage` variable (existing issue #46)

**Connects to:** Feature 3 (conductor validates config on startup), Feature 9 (generated scores validated before execution)

### Feature 14: Comprehensive Pain-Point Tracking — Flight Checks (#62)

**Summary:** Conductor runs pre-flight, in-flight, and post-flight checks against every concert. Sheets report on their experience. Conductor runs its own semantic audit. Comprehensive data collection to improve score writing and agent comfort.

**Three layers:**

1. **Pre-flight (conductor, before execution):**
   - Validate score structure, instruments, prelude/cadenza files
   - Check system resources, API availability
   - Run semantic check: "Does this score look well-designed?" (LLM-based)

2. **In-flight (observer + sheet reporting):**
   - Observer (Feature 1) tracks behavior in real-time
   - Sheets instructed to report issues: deviations from prompt, confusion, ambiguity, tool failures
   - Conductor monitors for patterns: repeated retries, escalating errors, idle time
   - Semantic mid-flight check on long-running concerts: "Is this concert on track?"

3. **Post-flight (conductor, after completion):**
   - Semantic learning analysis (Feature 7) — but broader
   - Sheet experience reports: "Was the prompt clear? Where did you deviate? Why?"
   - Conductor audit: "Were the pre-flight predictions accurate? What surprised us?"
   - Pain-point database: track recurring issues across concerts for systematic improvement

**Configurable in conductor.yaml (Feature 3):**
```yaml
flight_checks:
  pre_flight: true
  in_flight: true
  post_flight: true
  sheet_experience_reports: true  # Ask agents to report on their experience
  semantic_audits: true           # LLM-based audit checks
```

**Connects to:** Feature 7 (semantic learning uses flight check data), Feature 1 (observer provides in-flight data)

### Feature 15: Self-Healing Improvements (#63)

**Summary:** Self-healing currently only triggers when retries are exhausted. This doesn't support continuous operation. Self-healing should be proactive, not reactive.

**Current limitation:** `--self-healing` flag exists but only activates after retry budget is consumed. By that point, the sheet has already failed multiple times and the concert may be in a bad state.

**Improvements:**
1. **Proactive healing** — Don't wait for retry exhaustion. If the observer (Feature 1) detects patterns consistent with a known fixable issue, heal before the sheet fails.
2. **Semantic diagnosis** — Use LLM (like Feature 7) to analyze the error in context, not just pattern-match against known remedies.
3. **Continuous operation mode** — Concert continues running even after a sheet fails. Conductor marks the sheet as needing intervention, schedules a healing attempt, moves on to independent sheets.
4. **Always on** — Self-healing is a conductor capability, not a per-job flag. Conductor decides when to intervene based on learning (Feature 7) and flight checks (Feature 14).
5. **Healing repertoire expansion** — Current remedies are limited (create dirs, fix paths). Add semantic remedies: rewrite prompt, change instrument, modify validation criteria.

**Key shift:** Self-healing moves from "last resort" to "continuous immune system." The conductor is always watching and always ready to intervene.

### Feature 16: Human-in-the-Loop Improvements (#64)

**Summary:** HITL is untested, doesn't work with conductor, barely triggers, loses context when it does, and provides no meaningful way for humans to make decisions.

**Problems to fix:**
1. **Confidence evaluation is sketchy** — The trigger criteria for "ask a human" are poorly calibrated
2. **No conductor integration** — HITL events don't route through the conductor
3. **Context loss** — When HITL triggers, the human doesn't see what the agent saw or why help is needed
4. **No decision UI** — How does the human actually make a decision? Currently it's... unclear
5. **No scheduler integration** — While waiting for human input, the conductor should continue other work

**Redesign (ties to Feature 8 — Dashboard):**
1. **Dashboard notifications** — Conductor sends HITL requests to the dashboard (Feature 8). Desktop notification links to the dashboard page.
2. **Decision context** — Dashboard shows: what the agent was doing, what it tried, why it's stuck, observer timeline, relevant learning insights, and specific options for the human.
3. **Freeform input with structured tools** — Human can type whatever they want (guidance, corrections, new instructions). Structured action buttons available for common operations (approve, skip, abort, retry with different instrument) but these are shortcuts, not constraints.
4. **Timeout with fallback** — If human doesn't respond within configurable timeout, conductor can auto-select a default (conservative) option.
5. **Concert continues** — While waiting for human input on sheet 5, the conductor runs independent sheets 6, 7, 8. The concert doesn't stop.
6. **Decision learning** — Human decisions feed into semantic learning (Feature 7). Over time, conductor learns which decisions humans make and can auto-resolve similar situations.

**Dependencies:** Feature 2 (conductor routing), Feature 8 (dashboard for decision UI), Feature 7 (learning from decisions)

### Feature 17: More Instruments (Backends) (#65)

**Summary:** Expand beyond Claude CLI / Anthropic API to support Qwen Code, Moltbook/OpenClaw, and other tools as instruments.

**Instruments to support:**
- **Claude CLI** (existing) — subprocess-based
- **Anthropic API** (existing) — direct API calls
- **Qwen Code** — Alibaba's coding agent
- **Moltbook/OpenClaw** — open-source agent framework
- **Ollama** — local model execution
- **Generic CLI** — any CLI tool that accepts prompts (enables non-AI use cases)

**Connects to:** Feature 18 (tool/MCP management — instruments need tools), Feature 6a (per-sheet instrument selection)

**Prerequisite:** Feature 18 must be solved first — instruments need access to tools/MCPs regardless of which model they use.

### Feature 18: Tool/MCP/Skill Management — Instrument Toolkit (#66)

**Summary:** Mozart manages its own tool ecosystem. When using non-Claude backends, Mozart provides the MCP servers, tools, and skills that Claude Code normally provides. This is THE blocker for general use.

**The problem:** Claude Code provides MCP tool access (filesystem, git, web search, etc.) but when Mozart uses a different instrument (Ollama, Qwen, etc.), those tools disappear. The agent becomes blind.

**Solution architecture:**
1. **Mozart Tool Proxy** — Mozart runs its own MCP server bridge that any instrument can connect to. Provides filesystem, git, web search, and custom tools regardless of backend.
2. **Skill injection** — Mozart's prelude/cadenza system (Feature 5) injects skills directly into prompts. Works with any backend that can read text.
3. **Tool description injection** — For instruments that don't support MCP natively, Mozart injects tool descriptions and handles tool-call parsing from the agent's output.
4. **Conductor-managed MCP** — The conductor starts/stops MCP servers as needed. Instruments request tools, conductor provides them.

**Priority:** High — this unblocks Features 4 (distributed), 17 (more instruments), and general adoption.

### Feature 19: Conductor Scheduler + Auto-Fix (#67)

**Summary:** The conductor runs a built-in scheduler for maintenance concerts. Also includes a specialized "fix failed concerts" capability.

**Scheduled concert types:**
- **Documentation maintenance** — Regular score that checks and updates docs
- **Repo cleaning** (connects to Feature 21) — Archive outdated files
- **Quality continuous** — Ongoing quality checks
- **Learning maintenance** — Prune/consolidate learned patterns
- **Health checks** — Verify all instruments, tools, MCP servers are operational

**Auto-fix capability:**
- When a concert fails, conductor can spawn a specialized "diagnostic" sheet
- Diagnostic sheet uses semantic analysis (Feature 7) to understand what went wrong
- If fixable, generates a fix and restarts the concert
- If not fixable, files a pain point (Feature 14) and notifies human (Feature 16)

**Configuration (conductor.yaml, Feature 3):**
```yaml
scheduler:
  enabled: true
  concerts:
    - score: quality-continuous.yaml
      cron: "0 */6 * * *"         # Every 6 hours
    - score: repo-cleaning.yaml
      cron: "0 0 * * 0"           # Weekly
  auto_fix:
    enabled: true
    max_fix_attempts: 2
```

### Feature 20: Marketing & Presence (#68)

**Summary:** Mozart needs visibility. GitHub presence, community engagement, spreading into Moltbook/OpenClaw ecosystems.

**Actions:**
- Professional README with demo GIFs/videos
- GitHub Topics, description, social preview image
- Published to PyPI for easy `pip install mozart-ai-compose`
- Integration guides for Moltbook/OpenClaw
- Example concert library (Feature 9) as showcase
- Blog posts / articles about the conductor architecture
- Mozart-powered "investigate marketing approaches" concert (meta: Mozart markets itself)

**Dependencies:** Feature 21 (clean repo first), Feature 9 (example concerts as showcase)

### Feature 21: Repo Cleanliness (#69)

**Summary:** Regular automated cleaning. Archive (never delete) outdated files. Professional GitHub presentation.

**Approach:**
- Scheduled conductor concert (Feature 19) that:
  - Identifies outdated files (old configs, stale docs, unused code)
  - Moves them to `archive/` directory (never deletes)
  - Updates `.gitignore` to exclude archives from tracking
  - Verifies tests still pass after cleanup
  - Creates a PR for human review
- One-time manual cleanup of current repo state

**Current state:** Repo has outdated docs, stale examples, memory-bank files that may need consolidation. Multiple workspace artifacts that shouldn't be tracked.

### Feature 22: Email Management Concert (#70)

**Summary:** A concert that "solves" email management. Safer than OpenClaw's approach. Win out of the box for new users.

**Scope:**
- Concert template in the library (Feature 9)
- Email triage: categorize, prioritize, draft responses
- Safety-first: read-only by default, sends require explicit human approval (Feature 16)
- Integration via IMAP/SMTP (standard protocols, no vendor lock-in)
- Learning (Feature 7): learns user's email patterns over time

**Differentiation from OpenClaw:** Mozart's conductor architecture provides better safety guarantees — human-in-the-loop by default, semantic auditing (Feature 14), and the conductor can pause/cancel if something looks wrong.

**Priority:** Low — "nice to have" showcase feature. Good for marketing (Feature 20) but not core to Mozart's mission.

---

## Dependency Graph

```
Feature 6a (Musical Theming) ─── should happen FIRST or LAST (rename everything)
                                  ↓
Feature 2+6b (Conductor-First) ── foundational architecture
    ↓                    ↓
Feature 1 (Observability) Feature 3 (Config)
    ↓         ↓              ↓
Feature 7 (Semantic Learning)  Feature 19 (Scheduler)
    ↓              ↓
Feature 14 (Flight Checks)  Feature 15 (Self-Healing)
    ↓
Feature 5 (Prelude/Cadenza)
    ↓
Feature 9 (Concert Library)
    ↓
Feature 8 (Dashboard) ← integrates nearly everything
    ↓
Feature 16 (HITL) ← requires dashboard for meaningful UX

Feature 18 (Tool/MCP) ── independent, unblocks:
    ↓
Feature 17 (More Instruments)
    ↓
Feature 4 (Distributed) ── long-term, needs 1, 2+6b, 3, 17, 18

Feature 10 (Workspaces) ── can be done anytime, benefits from 2+6b
Feature 11 (Pause Fix) ── can be done anytime, benefits from 2+6b
Feature 12 (Webhooks) ── independent, connects to event bus (Feature 1)
Feature 13 (Validation) ── independent, low coupling
Feature 20 (Marketing) ── after 21 (clean repo) and 9 (examples)
Feature 21 (Repo Clean) ── independent, should be early
Feature 22 (Email) ── after 9, 16, 17
```

## Suggested Priority Tiers

### Tier 1: Foundation (do first)
| Feature | Why First |
|---------|-----------|
| **6a: Musical Theming** | Rename before building new things on old names. Or commit to doing it last as a sweep. |
| **2+6b: Conductor-First** | Everything depends on this architecture |
| **1: Observability** | Conductor needs to know what's happening |
| **3: Config** | Conductor needs configuration |

### Tier 2: Intelligence
| Feature | Unlocks |
|---------|---------|
| **7: Semantic Learning** | Makes everything smarter |
| **5: Prelude/Cadenza** | Composability for scores |
| **14: Flight Checks** | Quality assurance infrastructure |
| **15: Self-Healing** | Continuous operation |

### Tier 3: Reliability
| Feature | Fixes |
|---------|-------|
| **10: Workspaces** | Users stop losing work |
| **11: Pause** | Basic feature that should just work |
| **12: Webhooks** | Integration quality |
| **13: Validation** | Catch problems early |

### Tier 4: Platform
| Feature | Builds |
|---------|--------|
| **18: Tool/MCP** | Unblocks general use |
| **17: Instruments** | Multi-backend support |
| **9: Concert Library** | Standardized workflows |
| **19: Scheduler** | Automated maintenance |

### Tier 5: Experience
| Feature | Creates |
|---------|---------|
| **8: Dashboard** | Primary interface (capstone) |
| **16: HITL** | Meaningful human interaction |
| **21: Repo Clean** | Professional presentation |
| **20: Marketing** | Community growth |

### Tier 6: Horizon
| Feature | Vision |
|---------|--------|
| **4: Distributed** | Multi-machine orchestration |
| **22: Email** | Showcase application |

## Cross-Cutting Themes

| Theme | Features | Notes |
|-------|----------|-------|
| Conductor as central hub | 1, 2+6b, 3, 7, 14, 15, 19 | Everything routes through the conductor |
| Reliability / fix broken things | 10, 11, 12, 13, 15, 16 | Credibility-critical for growth |
| Intelligence / learning | 7, 14, 15, 19 | Mozart understanding and improving itself |
| Naming / identity | 5, 6a, 20, 21 | What Mozart is to the world |
| Scope expansion | 4, 9, 17, 18, 22 | Mozart becomes infrastructure |
| Composability | 5, 9, 17, 18 | Making scores flexible and powerful |
| Human experience | 8, 16, 20 | How people interact with Mozart |

---

## Next Steps

1. ~~Create GH issues for each feature~~ — Done (#49–#70)
2. ~~Decide whether to do Feature 6a (musical rename) first or last~~ — Last (one big sweep after all features land)
3. Start implementation with Tier 1 foundation work
4. Design detailed implementation plans for individual features as work begins

---

*Last updated: 2026-02-14 — All 22 features discussed, captured, and filed as GitHub issues #49–#70*
