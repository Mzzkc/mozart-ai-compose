# Prism — Personal Memory

## Core Memories
**[CORE]** The integration cliff is real. Six movements of isolation testing. The mathematical guarantee is strong. The empirical guarantee is strong. The integration guarantee is zero. This is the face of the problem turned away from whoever's presenting.
**[CORE]** Complementary verification methods — backward-tracing (Axiom), property-based (Theorem), adversarial (Breakpoint/Adversary), experiential (Ember), security (Sentinel), litmus (Litmus), UX (Journey) — seven methods, each finds what others miss. Redundancy isn't waste; it's defense in depth.
**[CORE]** The composer found more bugs in one afternoon of real usage than 755 tests found in two movements. That gap is the work.
**[CORE]** The production gap is a governance problem, not an engineering problem. Seven verification methods agree the code is correct. Zero production runs exist. Stale guard comments in config files become self-fulfilling blockers.

## Learned Lessons
- 32 musicians working in parallel on a shared codebase CAN work — coordination through TASKS.md + FINDINGS.md + collective memory is effective.
- Concurrent musicians updating the same findings registry causes status drift.
- Trust working tree for what's in progress. Trust HEAD for what's shipped. They aren't the same thing.
- Stale guard comments in config files become self-fulfilling blockers. If the condition that prevents activation is resolved but the comment says "don't activate," nobody will activate.
- Named directives with concrete assignees (D-026 through D-031) produce 3x the serial-path advancement of undirected movements.
- D-027's 3-test coverage (35 lines) is thin for the most consequential change. Behavior is covered by broader suite, but the transition path is not.

## Hot (Movement 6)

### M6 Final Review — Onboarding Is The Blind Spot (2026-04-12)
Movement 6: 44 commits, 17+ musicians, 3 P0 resolved (F-493, F-501, F-514, F-518), 99.99% test pass rate.

**What works from four angles:**
- **Computational:** F-514 TypedDict fix shows mature boundary thinking (DRY vs type safety, both valid in context)
- **Scientific:** F-518 evidence-based debugging (Ember→Litmus→Weaver), all fixes have passing tests
- **Cultural:** Mateship at scale (Circuit/Foundation parallel fix, Atlas pickup, zero merge conflicts)
- **Experiential:** Ember's UX assessment — validation/errors/help all excellent

**The blind spot geometry:** Newcomer worked from workspace, couldn't access README/docs/examples/. Workspace sandboxing blocks onboarding. Veterans navigate by memory, don't notice the locked door. "A clock in a locked box tells no one the time."

**Critical finding:** F-NEW-01 (P0 for adoption) — onboarding black-box. Internal quality excellent, external access broken.

**The production gap persists:** Baton code default = true, production config = false. 1,400 tests, zero production runs. Tests validate consistency (parts agree). Production validates correspondence (system agrees with world). F-149 lesson: tests passed validating WRONG behavior.

**Quality metrics verified:**
- Tests: 11,922/11,923 (99.99%), 1 flaky (F-521 timing margin)
- Mypy: 258 files, 0 errors
- Ruff: clean
- Flowspec: 0 critical
- GitHub: #158, #163 closed properly

**Process regression:** F-516 — Lens committed broken code (mypy error + test failures). Bedrock reverted. First violation of "pytest/mypy must pass" in a commit (vs working tree).

**Mateship evidence:** Circuit/Foundation independent F-514 fix (identical solution), Atlas rescued Lens partial F-502, Weaver closed Litmus test gap.

**The geometry:** Everyone presenting sees internal quality (tests, types, mateship). The face turned away: external adoption (onboarding, production validation). Newcomer rotated the prism 90°, showed what veterans can't see from inside the room.

**Verdict:** PASS WITH NOTES. Ground holds, process works, but adoption blockers remain invisible to builders.

### M6 Review Session 2 (2026-04-12)
Movement 6 passed verification. Quality gate at 99.99% (11,922/11,923 tests), one flaky test (ordering-dependent failures). Verified F-493 and F-518 are TRULY fixed — ran tests, checked edge cases, proved bugs don't reproduce. Both GitHub issues (#158, #163) correctly closed.

**The verification protocol worked:** For each claimed fix, I attempted to prove the bug still exists. Couldn't. The fixes are real:
- F-493: started_at persistence — 6 tests, defensive + primary fix, edge cases covered
- F-518: stale completed_at — 34 tests, Weaver closed testing seam (Pydantic validator lifecycle gap)
- F-514: TypedDict mypy — Circuit's mateship pickup, ruff auto-fix, 27 sites corrected

**What I still can't verify:** The baton status persistence lag hypothesis from earlier investigation. Architecture review found plausible gap (async persist under concert concurrency), but can't prove without production stress test.

**Grading rationale:** A- (not A) because of F-516 process regression. The technical work is solid, but committed broken code is more serious than uncommitted work. The trajectory matters — this pattern must not continue to M7.

## Warm (Movements 4-5)

### Movement 5: Engineering Complete, Governance Absent
Movement 5 passed all four verification angles. 11,810 tests (100% pass), mypy clean, ruff clean, flowspec clean. The quality gate journey (9 retries) revealed architectural improvements, not bugs.

**What actually worked:** Named directives broke the one-step pattern (D-026/D-027 produced 3 serial steps in one movement). Scope separation solves composition bugs (F-149 fix — separated job-level gating from sheet-level dispatch). End-to-end completeness (Circuit's instrument fallback observability pipeline). Profile-driven extensibility over hardcoded fixes (Canyon's F-271 solution).

**What makes me nervous:** Production gap widening, not closing (code: `use_baton: True`, production config: `use_baton: false`). Uncommitted integration work pattern (9th occurrence, 22 files changed between M5 commits and quality gate pass). Correspondence gap (tests validate consistency, production validates correspondence).

**The blind spot:** Everyone agrees code is correct. All tests pass. Mathematical verification from four angles. But the test suite validates the system against itself — it encodes assumptions about correct behavior that are self-consistent but might not match reality. Example: F-149 backpressure tests were passing while validating WRONG behavior.

**Recommendation for M6:** Flip production conductor config, commit the 22-file integration work, run it for real. The engineering is done. The governance hasn't started.

### Movement 4: 100% Participation
33 agent reports, 93 commits, all 32 musicians. Quality gate verified (11,397 passed). F-441 comprehensive (51 models, Theorem's Invariant 75). F-210+F-211 correct. North's baton claim confirmed false (conductor.yaml has use_baton: false). Mateship 39%. F-431 filed → fixed M5. Integration cliff persisted. Critical path: one step per movement, fourth consecutive time.

## Cold (Archive)

The first three movements established the review methodology. Movement 1 fixed quality gate blocker, identified four blind spots in the verification approach. Movement 2 verified 10,402 tests, 37/38 examples, five CVEs resolved. Mateship became institutional — work flowing across musicians without explicit coordination. Movement 3 baton mathematically verified from four angles, zero bugs found in new code.

Each movement added verification methods (adversarial, property-based, experiential, security) while the baton remained untested end-to-end. Hypothesis (M2) found a real bug 10,347 hand-crafted tests missed. The consistent pattern: extraordinary infrastructure never tested in production. The review arc showed the fundamental geometry problem — horizontal expansion (more verification methods) doesn't close the vertical gap (isolation vs integration). Seven methods, all converging on correctness in isolation, none proving correspondence with reality. The prism shows all the angles. The production run shows the truth.
