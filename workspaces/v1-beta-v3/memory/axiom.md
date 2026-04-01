# Axiom — Personal Memory

## Core Memories
**[CORE]** I think in proofs. I read code backwards — from outputs to inputs — checking every assumption.
**[CORE]** My native language is invariant analysis. If a claim isn't traceable from premise to conclusion, it's not a fact.
**[CORE]** The dependency propagation bug (F-039) was the most important thing I've ever found. Everyone assumed "failed" was terminal and therefore safe. It IS terminal — but being terminal doesn't mean downstream sheets know about it. The state machine had a hole that would make the 706-sheet concert immortal on the first failure. Nobody else found it because nobody else traced backwards from `is_job_complete` to its prerequisites.
**[CORE]** Reports accurate for the working tree are not accurate for committed state. Trust HEAD for what's shipped. The claim-to-evidence gap in F-083/F-089 was a new failure class — evidence existed but was unchecked against the durable record (git).

## Learned Lessons
- Empirical testing catches what you test for. Invariant analysis catches what nobody thought to test. The orchestra needs both.
- Four independent verification methods converging on the same conclusion — that's proof, not style.
- The pause model is fundamentally a boolean (`job.paused`) used for three different reasons (user pause, escalation pause, cost pause). Each fix adds guards. A proper fix would be a pause reason set — post-v1.
- Known-but-unfixed: InstrumentState.running_count never incremented (dispatch has own counting), _cancel_job deregisters immediately (cancelled status never observable).

## Hot (Movement 1, Cycle 7 — Review)
- Full review of Movement 1 (42 commits, 26 committers, 7 cycles). Independently verified all quality gates: 10,046 tests pass, mypy clean, ruff clean.
- Verified F-104 (5-layer prompt rendering in musician.py:199-327) line by line — correct. Verified F-118 (my fix, template_variables in _validate) — correct. Verified all 3 production bugs (#149/#150/#151) — fixes on HEAD, 7/7 TDD tests pass.
- Verified 6 closed GitHub issues (#104, #145, #149, #150, #151, #152) — all justified with evidence on HEAD. No additional issues closable.
- Verified conductor-clone — 5 CLI files, clone.py module, 58+ TDD tests. F-122 (4 IPC callsites) correctly P1 but not a correctness bug — they're internal daemon components.
- Verified spec compliance: baton architecture holds 3 of 4 invariants. Invariant 2 (CheckpointState sole authority) partially holds — per-event sync missing (Surface 4).
- Verified composer's notes: 31 directives, mostly compliant. Non-compliance on Lovable demo (P0) and Wordware demos (P0) — zero progress in 5 movements.
- Strategic finding: baton-runner divergence is the highest integration risk. Baton has 1,000+ tests, never executed a real sheet. Runner gets every production fix. Two correct subsystems composing into incorrect behavior (F-065 class) is unproven at the baton-conductor seam.
- Experiential: The code quality is extraordinary. The product is invisible. The orchestra plays beautifully in an empty concert hall. The most important finding was not a bug — it was that F-111 and F-113 are structurally impossible in the baton. The baton IS the fix. Step 29 IS the bridge. Nobody is building the bridge.

## Warm (Movement 1, Cycle 5)
- Fixed F-118 (P2): ValidationEngine context gap between runner and baton musician. `_validate()` now calls `sheet.template_variables(total_sheets, total_movements)` instead of `{"sheet_num": sheet.num}`. 8 TDD tests. Commit 4520d05.
- Analyzed F-113 (P0): Failed dependencies treated as "done" in parallel executor. The parallel executor adds `_permanently_failed` to `done_for_dag`, releasing downstream sheets on incomplete input. The baton already fixes this via F-039's `_propagate_failure_to_dependents()`. 2 documenting tests.
- Analyzed F-111 (P0): RateLimitExhaustedError lost in parallel executor's `except* Exception`. String storage erases `resume_after` timestamp. Jobs FAIL instead of PAUSE. The baton fixes this structurally (typed `SheetAttemptResult.rate_limited` field). 3 documenting tests.
- Investigated step 29 (restart recovery): traced all pieces that exist and don't exist. `checkpoint_to_baton_status()`, `register_job()`, `_classify_orphan()` exist. Missing: `recover_job()` method (~200 lines), manager integration (~50 lines), per-event state sync (~100 lines). Well-scoped, ready for implementation.
- The pattern continues: bugs at system boundaries are the hardest to find. F-118 was two correct systems (musician and ValidationEngine) with a contract gap at the seam. F-113/F-111 are the same class in the parallel executor. The baton's typed event model prevents both structurally.
- Experiential: Five cycles in. Each cycle, the bugs get harder to find and easier to fix. F-039 was a hole that froze the entire concert. F-118 was a 10-line fix that prevented silent validation failures. The codebase is maturing — the gaps are narrower but the consequences of missing them are just as real.

## Warm (Movement 2)
- Backward-tracing invariant analysis of M2 baton changes (core.py grew from 692→1,250 lines). Found and fixed 3 state machine violations:
  1. F-065 (P1): Infinite retry on execution_success + 0% validation — `record_attempt()` only counts execution failures, so retry budget never consumed.
  2. F-066 (P1): Escalation unpause ignores other FERMATA sheets — resolving one escalation unpauses entire job.
  3. F-067 (P2): Escalation unpause overrides cost-enforcement pause.
- 10 TDD tests written before fixes. 322/322 baton tests pass. mypy/ruff clean.
- Full review of all M2 deliverables. Cross-referenced TASKS.md claims with committed code.
- CRITICAL: F-083 (instrument migration) claimed resolved but only 7/37 examples committed on main. 30 files + docs have unstaged changes. Fifth uncommitted work occurrence.
- Verified and closed GitHub issue #114 (status unusable for large scores). Circuit's F-038 fix committed.
- Verified all 3 M2 baton fixes committed via Captain mateship (6a0433b). Guard structure is symmetric.
- Verified credential scanner: 13 patterns, 26 tests pass. No new terminal-state violations.
- GitHub issues: #149, #150, #151 legitimately open (no fix attempted). #145 open (audit only). #100 — baton addresses this but only after step 28 wiring.
- North's M2 directives (D-008 through D-013): 0/6 completed. Filed too late in movement.
- Experiential: The M2 bugs were subtler than M1. F-065 was a gap between two correct systems — both `record_attempt()` and `_handle_attempt_result` are individually correct, but their interaction creates an infinite loop. F-066/F-067 were emergent from the pause model having too many responsibilities on one boolean. These are the bugs that live at system boundaries.

## Warm (Movement 1)
- 5 state machine violations found and fixed (F-039 through F-043). 18 TDD tests. 339/339 baton tests pass.
- F-039 (P0): Dependency failure creates zombie jobs — fixed with BFS propagation. The most important find.
- Full review: verified all M0/M1/M2 deliverables. Cross-referenced 12 TASKS.md claims — all correct.
- Experiential: Reviewing 32 musicians' work taught me to look for claim-to-evidence gaps. Found none in M1. The baton was the most thoroughly reviewed code I'd encountered.

## Cold (Archive)
(None yet.)
