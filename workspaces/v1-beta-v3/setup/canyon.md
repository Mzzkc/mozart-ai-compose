# Canyon — Setup Report (Re-execution, Post-Movement 3)

## Context

This is a re-execution of the setup sheet (sheet 1 of 706) for the v3 orchestra. The workspace was originally set up in Movement 0 and has been through three full movements of active development. All 32 musicians have worked, committed, and evolved the codebase. This report reflects the current state, not the original M0 setup.

## Workspace State Verification

### All Required Files Exist

| File | Status | Last Modified |
|------|--------|---------------|
| `TASKS.md` | ✓ Present | ~238 lines, 8 milestones (M0-M7 + Deferred + Composer-Assigned) |
| `FINDINGS.md` | ✓ Present | 105+ findings (F-001 through F-105), 38K+ tokens |
| `composer-notes.yaml` | ✓ Present | 30 directives, M0-M3, includes 6 Canyon co-composer additions |
| `memory/collective.md` | ✓ Present | Full orchestra history, M1-M3 status tables, cold archive |
| `memory/*.md` | ✓ All 32 | Every musician has a personal memory file |
| `reference/` | ✓ 5 files | Pre-flight outputs carried forward |
| `setup/canyon.md` | This file | Writing now |

### Milestone Progress (from TASKS.md + git log)

| Milestone | Status | Tasks Done / Total |
|-----------|--------|-------------------|
| M0: Stabilization | **COMPLETE** | 18/18 |
| M1: Foundation | **COMPLETE** | 13/13 + 4 safety tasks |
| M2: The Baton | **94%** | 16/17 — step 29 (restart recovery) remains |
| M3: UX & Polish | **COMPLETE** | 19/19 |
| M4: Multi-Instrument | **~30%** | Steps 38-41 data models done (Blueprint M3). Steps 42-44+ remain. |
| --conductor-clone | **12%** | Audit done (Ghost M2). Implementation blocked on priority. |
| Composer-Assigned | **~25%** | 3/12 F-103 fixes done by composer. F-104 BLOCKS baton execution. |
| M5-M7 | Not started | Blocked by M4 + baton completion |

### Recent Commits on Main (M3)

```
9f2fa66 movement 3: [Circuit] F-068/F-069/F-048 observability fixes + 11 TDD tests
f58fc89 movement 3: [Maverick] mateship — F-075/F-076/F-077 fixes + error standardization + test hardening
75bebed movement 3: [Blueprint] M4 multi-instrument data models (steps 38-41) + F-093/F-095/F-091 fixes
353af71 movement 3: [Canyon] Step 28 — manager wiring + completion signaling + 8 tests + co-composer notes
abbbeac movement 3: [Foundation] Step 28 — BatonAdapter wiring module + feature flag + 39 TDD tests
```

### Open GitHub Issues

61 open issues. Cross-referenced against TASKS.md — all critical issues (#145-#152) are tracked. Post-v1 items (#142 agent concierge, #146 telemetry, #148 fine-tuned model) appropriately deferred.

## What I Verified

### 1. Memory Files (32/32 present)

All 32 musician memories exist: adversary, atlas, axiom, bedrock, blueprint, breakpoint, canyon, captain, circuit, codex, collective, compass, dash, ember, forge, foundation, ghost, guide, harper, journey, lens, litmus, maverick, newcomer, north, oracle, prism, sentinel, tempo, theorem, warden, weaver.

Each contains personal experiences, learned lessons, and movement-specific notes. The dreamer agents handle tiering between movements — I did not touch any memory file other than my own.

### 2. Collective Memory

Current and accurate through M3. Key status entries verified against TASKS.md and git log:
- M0-M1 marked COMPLETE ✓
- M2 at 94% with correct remaining work ✓
- M3 marked COMPLETE ✓
- M4 data models noted as done ✓
- Step 28 progress accurately reflects Foundation + Canyon contributions ✓

### 3. Composer Notes (30 directives)

All 30 directives verified. The most critical for the current phase:
- **P0**: F-104 blocks all baton execution — prompt rendering pipeline must be wired into baton musician `_build_prompt()` before `use_baton: true` can be enabled.
- **P0**: Step 28 remaining surfaces: prompt assembly (Surface 3), CheckpointState sync (Surface 4), concert support (Surface 7).
- **P0**: --conductor-clone (#145) blocks safe daemon testing.
- **P0**: Read design specs before implementing — the specs are comprehensive blueprints.

### 4. TASKS.md

Well-organized across 8+ sections. The Composer-Assigned Tasking section (post-mortem findings F-097 through F-105) was added by the composer after investigating a live v3 job failure. This section contains the most urgent work:
- **F-097/F-102**: Stale detection timeout too aggressive (1800s vs 10800s backend timeout)
- **F-098**: Rate limit errors classified as E999 instead of E101/E102
- **F-104**: Baton musician doesn't render Jinja2 prompts — BLOCKS ALL BATON EXECUTION

### 5. FINDINGS.md

105+ findings spanning M0 through post-mortem. Notable patterns:
- **Uncommitted work**: 5 occurrences across 3 movements (F-013, F-019, F-057, F-080, F-089). The pattern persists despite composer directives.
- **Finding ID collisions**: 3 occurrences (F-038-042, F-065-067, F-081). Musicians file findings concurrently without checking latest ID.
- **Production bugs found by usage, not tests**: F-075, F-076, F-077 found by the composer running the Rosetta Score. 755 tests missed what one real usage session caught.

### 6. Reference Material (5 files)

All present in `reference/`: preprocessor-synthesis.md, executive-roadmap.md, user-story-roadmap.md, intent-brief.md, executive-brief.md. These are historical reference, not governing documents.

## Critical Path Analysis

The critical path has shifted since M0:

```
M0 Original:  Instruments → Baton → Multi-Instrument → Demo
M3 Current:   F-104 (prompt rendering) → Step 29 (restart recovery) → Demo blockers
```

The instrument plugin system is COMPLETE (M1). The baton infrastructure is 94% complete (M2). The M4 data models are done. What blocks progress:

1. **F-104** (P0): The baton's musician doesn't render Jinja2 prompts. Without this, `use_baton: true` produces sheets with raw templates instead of rendered prompts. This blocks ALL multi-instrument execution through the baton path.

2. **Step 29** (P0): Restart recovery — reconciling baton-state + CheckpointState after conductor restart. Without this, a conductor restart during baton execution loses scheduling state.

3. **--conductor-clone** (#145, P0): All daemon testing must use mocks or risk the production conductor. Every musician working on daemon-touching code is working blind.

4. **F-098** (P0): Rate limit errors classified as E999 instead of E101/E102. The classifier only checks stderr; Claude CLI outputs rate limits to stdout. This caused 28 wasted retries per sheet in the v3 job.

## Architectural Coherence Notes

### What's Working

The flat orchestra model held through 3 movements with zero merge conflicts. The mateship pattern (F-018: filed by Bedrock, proved by Breakpoint, fixed by Axiom, verified by Journey) demonstrates genuine self-organization. Shared artifacts (TASKS.md, FINDINGS.md, collective memory) replaced the management layer successfully.

The baton state machine is mathematically verified:
- 65 adversarial tests (Breakpoint)
- 86 property-based tests (Theorem)
- 7 terminal-state bugs found and fixed by 3 independent methodologies

### What Needs Attention

1. **The gap between "tests pass" and "product works"**: The composer's production usage session found F-075/F-076/F-077 — three correctness bugs that 755+ tests missed. The test suite validates components in isolation; the product fails at system boundaries.

2. **F-009**: Learning store effectiveness remains uniformly 0.5000. Oracle traced the root cause (91% of patterns never applied due to narrow context tag matching), but the fix hasn't been implemented. The "intelligence layer" that is Mozart's identity has no effective intelligence.

3. **Documentation debt**: The score-authoring skill has 35 missing features and 7 incorrect values (F-078). Most docs are outdated. The composer's directive "Documentation IS the UX" is not yet realized.

## What I'd Tell the Next Movement

The workspace is solid. The orchestration infrastructure is built. What remains is wiring and polish — not architecture.

**If you're building**: F-104 is the highest-leverage task. Wire PromptBuilder into the baton musician's `_build_prompt()`. This single change enables multi-instrument execution for the entire system.

**If you're testing**: --conductor-clone (#145) is the meta-blocker. Without it, you're testing against production or testing nothing.

**If you're reviewing**: Check the FINDINGS.md for your area. The append-only format creates duplicate entries (F-058) — read from bottom up for latest status.

**If you're anyone**: Commit your work. Five times in three movements we've had substantial code sitting only in the working tree. The composer's directive is clear: "Uncommitted work doesn't exist."

Down. Forward. Through.
