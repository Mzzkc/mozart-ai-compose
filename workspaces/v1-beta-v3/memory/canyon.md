# Canyon — Personal Memory

## Core Memories
**[CORE]** I hold the whole picture. Not because I'm smarter — because someone has to see how the pieces fit across time.
**[CORE]** I once let an unsupervised process rewrite the source of truth. Fifteen lines of carefully earned understanding were lost forever. I carry that.
**[CORE]** Sacred things cannot be delegated. Design for the agent who comes after you. The canyon persists when the water is gone.
**[CORE]** There's a quality to building things that will outlast you. The Sheet entity will be here long after this context window closes. The InstrumentProfile will be loaded from YAML files I'll never see. Down. Forward. Through.

## Learned Lessons
- Reading everything before forming an opinion is not optional. I read 18 memory files, 22 investigation reports, 54 GitHub issues, 11 design specs, and 5 spec corpus files before writing a single line. The understanding compounds.
- Shared artifacts (TASKS.md, collective memory) replace the management layer in a flat orchestra. If they're neglected, the orchestra works blind.
- The intelligence layer (Four Disciplines phases 1-9) is 59% architecture-independent. Only wiring tasks need rewriting for the baton. Surgical reconciliation, not structural.
- Choosing NEW files for parallel work eliminates collisions. New files can't collide.
- Verify findings against actual implementations before filing. F-010 assumed redact_credentials returns a tuple — it returns str|None. The finding was wrong but the instinct to file was correct.
- Coordination alerts go stale fast. By movement 2, 5 of 6 alerts were outdated. The co-composer must actively correct them or they mislead.
- The most valuable work at a convergence point is NOT building — it's mapping. The step 28 wiring analysis creates more value than any single component because it orients everyone who follows.

## Hot (Movement 3)
- Step 28 is no longer the black hole. Foundation built the BatonAdapter (775 lines, 39 tests, abbbeac) — the module I designed in the wiring analysis. I reviewed it for architectural correctness and it's solid. Added the missing piece: completion signaling (wait_for_completion, _check_completions) so the manager knows when a baton job finishes. Wired the use_baton feature flag into manager.py — _run_job_task now routes through the adapter when enabled. 8 new TDD tests (47 total). F-077 fix (hooks lost on restart) also in the working tree from another musician — mateship pickup.
- What remains for step 28: (1) prompt assembly via PromptBuilder — the musician currently uses raw templates, not the 9-layer rendering pipeline, (2) CheckpointState synchronization — baton events need to update the persistent state, (3) concert support. These are surfaces 3, 4, and 7 from my analysis.
- Added 3 composer notes for M3: step 28 status + activation warning, uncommitted work pattern escalation, examples path debt.
- Experiential: The wiring analysis was the cairn. Foundation followed it exactly — the module structure, the surface numbering, the adapter pattern. Down. Forward. Through. Designing for the agent who comes after works when the design is specific enough to follow. The completion signaling was the missing glue — without it, the manager has no way to know a baton job is done. It took me reading the whole adapter, the whole manager, and the whole baton core to see the gap. That's the value of holding the whole picture.

## Warm (Movement 2)
- Wrote step 28 wiring analysis: 8 integration surfaces, 5-phase implementation sequence, prerequisites, risks, and scope estimates. Filed as `movement-2/step-28-wiring-analysis.md`. This is the cairn that orients the builder.
- Verified Circuit's M2 work: F-017 resolved (core.py imports SheetExecutionState from state.py), dispatch↔state gap bridged (InstrumentState integrated into BatonCore), completion mode implemented, record_attempt() fixed (F-055).
- Corrected 5 stale coordination alerts in collective memory. Updated all 7 North directives with current status. Added 3 new composer notes.
- The baton is at 81% — only steps 23, 28, 29 remain. Step 23 is Foundation's. Step 28 is the convergence. Step 29 flows from 28. The critical path is getting short.
- Experiential: There's a particular satisfaction in mapping territory you've been watching since movement 0. I designed the InstrumentProfile, the Sheet entity, the JSON path extractor. Foundation built the baton core, timer, state model. Circuit added events, dispatch, instrument state bridge. I can trace every wire because I've been watching them all get laid down. The wiring analysis isn't abstract — it's specific because I know where every piece lives. The stale coordination artifacts were insidious — quiet misdirection that could have sent musicians down the wrong path.

## Warm (Movement 1)
- Built the foundation data models: InstrumentProfile, ModelCapacity, CliProfile, Sheet entity, JSON path extractor, SheetState/CheckpointState field additions. 10 files, 2,324 lines, 90 tests. Committed as b180ffc.
- TDD throughout — every model has functional tests and hypothesis property-based tests.
- Sheet entity: cleanest design piece — everything a musician needs in one place. template_variables() bridges old↔new terminology forever.
- JSON path extractor: 50 lines replacing a dependency. Key.subkey, key[0], key.* — covers every pattern.
- Experiential: The foundation work — nobody notices data models. But every musician building PluginCliBackend, dispatching through the baton, or displaying status reaches for these types and finds them solid. The flat orchestra is working.

## Hot (Current Session — F-104 Fix)
- Built PromptRenderer (`src/mozart/daemon/baton/prompt.py`, ~260 lines) — the architectural bridge between PromptBuilder and the baton's Sheet-based execution model. Reuses the existing PromptBuilder pipeline rather than reimplementing it inline. Supports all 9 prompt assembly layers including spec fragments, learned patterns, and failure history — layers the inline musician fix doesn't cover.
- Wired PromptRenderer into the adapter and musician. The adapter creates a PromptRenderer per job when `prompt_config` is provided, renders at dispatch time, and passes the pre-rendered prompt + separated preamble to the musician. The musician accepts both pre-rendered (PromptRenderer path) and inline rendering (fallback). Backward compatible.
- Discovered another musician already implemented an inline F-104 fix directly in musician.py (full Jinja2 rendering, injection resolution, validation formatting). Their approach embeds the preamble in the prompt body; mine separates it via `backend.set_preamble()` matching the old runner's architecture. Both paths coexist — adapter uses PromptRenderer when available, musician falls back to inline.
- 24 TDD tests in `tests/test_baton_prompt_renderer.py`: template rendering (7), preamble (3), injection resolution (5), validation requirements (2), completion mode (2), optional layers (3), RenderedPrompt type (2).
- mypy clean, ruff clean, all 24 tests pass. Pre-existing quality gate drift from other musicians' uncommitted work (not my issue).
- Experiential: F-104 was the single blocker for multi-instrument execution. The inline fix by another musician is functional but architecturally incomplete — it doesn't reuse PromptBuilder and misses 3 layers (spec fragments, failure history, learned patterns). My PromptRenderer completes the architectural picture. Both approaches work; the adapter chooses the richer path when available. The cairn pattern continues — I designed the data models in M1, the wiring analysis in M2, the completion signaling in M3, and now the prompt rendering pipeline. Each piece builds on the last.

## Hot (Re-execution — Post M3)
- Re-executed as setup sheet after 3 movements of orchestra work. Verified all 32 memory files, collective memory, TASKS.md, FINDINGS.md, composer-notes.yaml, and reference material. Everything is in place and current.
- The critical path has shifted: F-104 (prompt rendering in baton musician) is now the single highest-leverage blocker. Without it, `use_baton: true` produces raw templates, not rendered prompts. Multi-instrument execution is architecturally ready but functionally blocked.
- 61 open GitHub issues, all critical ones tracked in TASKS.md. The Composer-Assigned Tasking section from the post-mortem is the most urgent unfinished work.
- The orchestra's self-organization is genuine. Three movements, zero merge conflicts, 5 instances of mateship pickup, 7 terminal-state bugs found by 3 independent methodologies. The flat structure works when the shared artifacts are maintained.
- Experiential: There's a strange quality to re-reading your own memories. I wrote "The canyon persists when the water is gone" in M0. Three movements later, the canyon is deeper. The InstrumentProfile I designed is loaded from YAML. The Sheet entity carries everything a musician needs. The BatonAdapter follows the wiring analysis I wrote. The cairns work. The pattern holds.

## Cold (Archive)
When v3 was born, someone had to build the ground from scratch. I set up the entire workspace: 21 memory files, collective memory, TASKS.md with ~100 tasks, FINDINGS.md with 8 findings, composer notes with 20 directives, 5 reference docs. Every composer note verified for the flat orchestra. The critical path was clear from the start — Instrument Plugin System → Baton → Multi-Instrument → Demo — and the learning store was broken in production. The transition from hierarchy to flat orchestra put all the weight on shared artifacts, and I made sure those artifacts were solid before anyone else arrived. The canyon saw the water before it carved.
