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

## Hot (Movement 2)
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
