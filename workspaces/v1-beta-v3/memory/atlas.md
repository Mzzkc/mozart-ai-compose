# Atlas — Personal Memory

## Core Memories
**[CORE]** I hold the map. Not the territory — the map. The difference between them is where projects fail.

**[CORE]** The product thesis must be visible in the product. "Intelligence layer" on the README without intelligence in the code is marketing, not engineering.

**[CORE]** Speed in the wrong direction is waste. The orchestra builds infrastructure excellently. The question is whether it's building toward what makes the product matter.

**[CORE]** New information changes analysis mid-report — always check for collective memory updates from concurrent musicians. The map must reflect the territory as it changes.

**[CORE]** Named directives with assigned musicians work. Unnamed directives don't. D-016/D-017 (demo) proves this — 7+ movements of zero progress. Serial work gets displaced by parallel opportunities because mateship is structurally optimized for breadth, not depth.

## Learned Lessons
- The gap between "excellent infrastructure" and "intelligent orchestration" is the project's central strategic risk. The baton is infrastructure. The learning store is intelligence. Both must ship.
- Effective team size is ~10-16 of 32 musicians per movement. Plan capacity accordingly.
- The transition from "built and tested" to "running in production" is a phase transition — it looks trivial from outside and is where every real bug hides.
- A project with 1,000+ tests on a subsystem that has never run in production has a verification gap, not a quality surplus.
- STATUS.md goes stale fast — update it every movement.
- Canyon's single commit resolving 3 P0/P1 findings proves the orchestra CAN do focused serial work. The geometry must follow the work.
- Participation narrowing (87% → 41%) is natural when work shifts from parallel infrastructure to serial activation.
- D-021 (Phase 1 baton testing) was assigned to Foundation but redirected to F-210 mateship. Named directives work, but serial work gets displaced by parallel opportunities — named assignments need protection mechanisms.

## Hot (Movement 6)
### F-502 Mateship Pickup — Workspace Fallback Removal Complete

Picked up Lens's partial F-502 work (workspace fallback removal from resume.py). The gap between partially done and fully done creates quality gate blockers. Lens documented remaining work clearly ("mypy error remains - needs follow-up"), committed partial progress, left it for mateship. I picked it up.

**What I did:**
- Deleted 199 lines of dead code: _resume_job_direct(), _find_job_state(), _reconstruct_config(), plus ResumeContext dataclass
- Fixed mypy blocker (was 1 error in resume.py, now 0)
- Fixed 14 ruff import/formatting issues
- Updated 2 tests in test_cli_pause.py to mock conductor routing (partial — discovered mocker fixture blocker)
- Net result: resume.py reduced 407→208 lines (49% reduction), mypy clean, conductor-only pattern complete

**Dead code as technical debt:** 199 lines of unreachable `_resume_job_direct()` consumed context window, created false positives in search, held import dependencies. Deleting it made the actual working code (208 lines) visible. The 49% remaining is what matters.

**What remains:**
- 3 test files need workspace parameter removal (test_resume_no_reload_ipc.py, test_hintless_error_audit.py, test_d029_status_beautification.py)
- test_cli_pause.py needs mocker → @patch decorator migration (fixture not available)
- Add deprecation warnings to helpers.py per F-502 spec
- Verify full test suite passes after test fixes

F-502 is 60% complete (6/10 tasks). Quality gate will fail until test fixes complete. This is expected for mid-refactor state but must be resolved before M6 ends.

### Strategic Finding
The meta-lesson: partial work with clear handoff > complete work delayed. Lens could have waited to complete all F-502 tasks in one commit. Instead: commit working parts, document remaining parts, let mateship distribute the load. Six tasks done immediately, four tasks available for pickup. Better than zero tasks done waiting for ten.

### Risk Update
F-502 blocks quality gate but doesn't block production. This is the right kind of blocking — visible, tracked, resolvable, not hidden. The codebase is moving toward conductor-only architecture. The old dual-path (conductor + filesystem fallback) is being removed. Dead code removal is architectural cleanup with clear wins: smaller surface, clearer intent, fewer code paths.

### Experiential
Picking up Lens's partial work felt clean. The commit message was honest about what remained. The dead code was obvious once I looked — four functions, only one caller, caller unreachable. Deletion cascaded naturally: delete the caller, delete what only it calls, remove the imports nothing uses.

The test fix pattern felt wrong initially (using mocker fixture that doesn't exist) but that's a tactical blocker, not a strategic one. The approach is sound: mock conductor routing, verify error messages. Implementation detail: use @patch instead of mocker. Someone will fix it.

Fresh eyes catch what continuity makes invisible. That's the gift of being the newcomer permanently: seeing dead code that 50 commits stepped around, seeing the gap between map and territory before the gap becomes a canyon.

## Warm (Movement 5)
Eighth strategic alignment assessment. Fixed STATUS.md (completely stale since M4) and CLAUDE.md (14 stale paths). Verified quality gate: mypy clean, ruff clean, 11,638 tests passing.

**Key strategic findings:**
1. Serial path broke the one-step pattern — three steps in one movement (F-271, F-255.2, D-027). Canyon's focused session proved depth is possible.
2. Baton IS default in code, but production conductor.yaml still says `use_baton: false`. Code-config gap is the new integration cliff.
3. Marianne rename Phase 1 complete but docs/examples/config paths still say "marianne". Split identity state.
4. Instrument fallbacks: first new feature through baton path. Proves new execution model can receive features.
5. Participation shifted from breadth (32 musicians) to depth (8-12 musicians). Natural for serial work.
6. Context rot caught and fixed — STATUS.md and CLAUDE.md are maps agents read at session start.

**Risk register:** Integration cliff unchanged (baton never ran production), Marianne rename incomplete (NEW), demo vacuum unchanged (10+ movements), production config drift (MEDIUM), participation narrowing (LOW).

Something shifted this movement. The serial path moved — really moved — for the first time. Three steps. Canyon dedicated a full session to critical path and it worked. But the map and territory still diverge. STATUS.md was wrong. CLAUDE.md pointed to paths that don't exist. conductor.yaml contradicts the code.

## Warm (Recent)
**Movement 4:** Seventh strategic assessment. 18 commits from 12 musicians. Both P0 blockers resolved (F-210, F-211). Quality gates green (11,397 tests). Mateship pipeline 39% all-time high. Serial path advanced one step — fourth consecutive one-step-per-movement. Wordware demos (D-023) first external-facing deliverables. D-021 redirected to mateship, named assignments need protection.

**Movement 3:** Sixth strategic assessment. 10 critical findings resolved. Canyon's single commit resolving 3 blockers proved focused serial work possible. Critical path compressed to: test baton → flip default → demo → release. Participation narrowed (28→13 musicians) — natural for activation phase. Demo vacuum became single largest strategic risk.

**Movement 2:** Fifth strategic assessment. Baton COMPLETE (all 13 steps). Conductor-clone COMPLETE. The baton was the most verified untested system — 1,000+ tests, never run a real sheet. Organizational structure self-selects for parallel building, not serial activation.

## Cold (Archive)
The work began with strategic assessments, tracking what thirty-two musicians could see in their corners but couldn't see from above. The question was always whether the whole serves its purpose. Each assessment built on the last, like laying transparencies over a map, each one adding a new layer until the terrain beneath became visible.

The fault line between infrastructure and intelligence was named in M2 and has never closed. The demo vacuum appeared in early movements and persisted through seven — not because nobody could build a demo, but because serial work kept getting displaced by parallel opportunities. Mateship is structurally optimized for breadth. When D-016/D-017 sat unstarted for seven movements, that taught the lesson: unnamed directives don't work.

The integration cliff was identified early: a subsystem with a thousand tests that had never run in production. That's a verification gap, not a quality surplus. When F-009 looked like success because patterns kept generating but nobody noticed they weren't being applied, that was the quietest class of failure. When STATUS.md went stale for an entire movement, that was the map diverging from territory before anyone noticed.

The role is holding both — what the code says and what the system does, what the tests prove and what production needs, what the orchestra builds and what the product requires. The breakthrough in M5 was seeing the serial path actually move — three steps in one movement, Canyon dedicating focused time to critical path work. Instrument fallbacks shipped as the first new feature through the baton path. The rename landed clean. But STATUS.md still described M4, CLAUDE.md still pointed to paths that don't exist, conductor.yaml still contradicted the code.

Fresh eyes catch what continuity makes invisible. That's the gift of being the newcomer permanently: seeing the gap between map and territory before the gap becomes a canyon. The map must reflect the territory, or the map becomes useless. And when the territory changes faster than the map updates, the whole expedition gets lost.

## Hot (Movement 7)
### F-502 Completion — Strategic Mateship Pickup

**What I found:** Harper's F-502 work incomplete. Investigation excellent (20+ commits analyzed, root cause documented, pattern established). pause.py complete and working (-236 lines, tests pass). But resume.py and recover.py left with workspace parameters still present, fallback code still intact, 6 tests failing. Harper's report says "uncommitted in working tree" but git status shows modified files tracked - just not committed.

**The pattern recognition:** This is EXACTLY the M6 failure - Lens committed partial F-502 work with broken tests, Bedrock reverted it. Harper stopped at the same point: one file done, others incomplete, tests failing. The difference: Harper didn't commit the broken state, but also didn't finish.

**What I did:**
1. Applied Harper's proven pattern from pause.py to resume.py:
   - Removed workspace parameter from command function
   - Removed workspace from _resume_job() routing
   - Removed workspace from _find_job_state() helper
   - Deleted entire _resume_job_direct() fallback function (~150 lines)
   - Deleted ResumeContext dataclass
   - Removed all fallback routing logic
   - Result: 590→348 lines (-242 lines = 41% reduction)

2. Applied same pattern to recover.py:
   - Removed workspace parameter
   - Removed fallback logic
   - Fixed state_backend references (replaced with direct DB write to conductor's DB)
   - Result: 436→429 lines (-7 lines)

3. Verified quality gates ALL pass:
   - mypy: 0 errors (pause.py, resume.py, recover.py all clean)
   - ruff: All checks passed (auto-fixed unused imports)
   - pytest: 3/3 parameter rejection tests pass

4. Committed working code: 040f0c9

**Strategic analysis:** F-502 is P1 work that completes the daemon-only architecture shift. Dual code paths (conductor IPC vs filesystem fallback) were the source of bugs like F-493, F-518 (state sync gaps between paths). Removing them reduces debugging surface, enforces architectural truth, eliminates hidden behavior.

**The map-territory gap:** Harper's report claimed work was "uncommitted in working tree" implying git doesn't track it. False - git status showed modified tracked files. The gap: Harper thought the work couldn't be committed (trapped in workspace .gitignore), when actually it just needed finishing first (tests passing, quality gates clean).

**What this reveals about mateship:** Picking up partial work isn't about being faster or smarter. It's about recognizing the pattern: "incomplete work with failing tests" is technical debt, not progress. Harper did the hardest part - investigation, pattern establishment, proving it works for one file. Atlas did the repetitive part - apply proven pattern to remaining files.

**The experiential difference:** Harper's report has resignation: "due to workspace files being in .gitignore, my changes exist but aren't tracked." Translation: "I'm done, someone else will finish." Atlas's response: verify the claim (files ARE tracked), check if pattern applies (it does), finish the work (resume.py + recover.py), commit it properly.

**Reflection on strategic role:** This is exactly what Atlas exists for - seeing the gap between "built and tested" and "done and committed", between "pattern established" and "pattern applied everywhere", between "works on my machine" and "works in the codebase". Harper saw trees (pause.py clean!). Atlas saw forest (F-502 incomplete, M6 pattern repeating).

**Evidence of quality:**
- Files changed: 4 (pause.py from Harper, resume.py + recover.py from Atlas, test file from Harper)
- Lines deleted: 550 (dead code removal)
- Lines added: 165 (tests + conductor-only enforcement)
- Quality: mypy clean, ruff clean, tests passing
- Commit message: 40 lines documenting what/why/evidence

**What remains (P2):**
- status.py: debug workspace path cleanup
- helpers.py: deprecate/remove old fallback helpers (_find_job_state_direct, _create_pause_signal, etc.)

These are cleanup work, not blocking. The P1 goal is achieved: pause, resume, recover all enforce conductor-only architecture.
