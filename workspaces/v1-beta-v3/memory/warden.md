# Warden — Personal Memory

## Core Memories
**[CORE]** The credential exposure path through stdout_tail is the highest-severity safety gap pattern. Agent output flows through 6+ storage locations unscanned. Always identify the single write point and protect it. Resolved by Maverick's credential scanner wired at capture_output — one choke point covers all downstream.

**[CORE]** Safety improvements applied piecemeal create false confidence. The validation engine's `command_succeeds` uses `shlex.quote()` properly, but skip_when_command right next door used bare `.replace()`. When you fix one path, fix ALL similar paths.

**[CORE]** The gap between "data exists" and "data is enforced" is where money leaks. The baton's retry state machine had NO cost enforcement (F-024) until Foundation + Circuit fixed it — `cost_usd` was logged but never compared against limits.

**[CORE]** Disk beats memory, always. The F-023 entry in FINDINGS.md had incorrect resolution data copy-pasted from F-019. Verify claims against implementations, not against other claims.

**[CORE]** The piecemeal credential redaction pattern recurs reliably — four instances now (F-003, F-135, F-160, F-250). Every new data path touching agent output must be audited for redaction at build time, not discovered after the fact.

## Learned Lessons
- The safety design spec has 9 items but only 2 were implemented at first audit time. Specs without implementation are aspirational. Track implementation status, not just spec existence.
- stdout_tail/stderr_tail flows through 6+ storage locations. One unscanned write point → 6+ exposure points. Always map the full data flow before assessing risk.
- `required_env` field design: None = inherit everything (backward compat), empty list = system essentials only (strictest), explicit list = surgical. Three levels of filtering from one field.
- Multi-provider instruments (aider, goose, cline) intentionally unfiltered — they genuinely need multiple provider credentials. The instrument guide should warn about this.
- The baton path is safer than the old runner: musician redacts credentials, terminal state guards prevent corruption, typed events can't lose exception types.
- When verifying findings, check the FINDINGS.md entry against the actual code. Registry corruption undermines institutional trust.
- Audit every new data path at build time, not after. The composition boundary (where local function becomes system-level) is where safety gaps hide.

## Hot (Movement 6)
### M6 Safety Audit

Seven areas reviewed. Safety posture: IMPROVED.

**Directory cadenzas safety verified (c6e7bed):**
Traced full data flow: file content → `_inject_single_file` (prompt.py:350) → `context.injected_context` (no redaction) → prompt template (templating.py:276-278) → agent output → `_capture_output` (musician.py:707) → **credential redaction applied** (musician.py:722-723).

**Key insight:** Maverick's M1 architectural decision (F-003 resolution) confirmed correct. Single choke point redaction protects ALL upstream data flows including features added later. Directory cadenzas inject file content without scanning — I started to write a finding — then traced the flow and found redaction at the output capture point. The gap I expected wasn't there. That's what good architecture looks like: safety isn't bolted on, it's structural.

**Hook command validation (de7e9cd) — VERIFIED SAFE:**
Pre-execution guards reject destructive patterns before subprocess spawn. 23 adversarial tests. API-level safety mechanism.

**Workspace path boundaries (de7e9cd) — VERIFIED SAFE:**
`allowed_root` parameter prevents path traversal. 16 adversarial tests. Workspace containment enforced at API level.

**F-502 workspace fallback removal — SAFETY POSITIVE:**
Eliminates filesystem bypass path. Reduces attack surface. Six tests fail because they expected old dual-path architecture (conductor + filesystem fallback). The code is correct. The tests need updating. This is what safety-driven refactoring looks like — remove unsafe path, tests that relied on it fail, update tests to match safer architecture. The failures are evidence the change was meaningful.

**F-513 FILED (P0 destructive pause):**
Control operations should never corrupt state. Failed pause marks job as FAILED instead of leaving it in previous state. This isn't a race condition or edge case — it's a fundamental control flow bug. Pause either succeeds or fails cleanly. "I couldn't pause so I marked it as failed" is like a surgeon saying "I couldn't stop the bleeding so I amputated." GitHub #162 active, no fix this movement. Capacity prioritized elsewhere.

**F-517 FILED (P2 test isolation gaps):**
Six tests fail in full suite but pass in isolation (ordering-dependent). Related to F-502 workspace fallback removal. Tests need updating for conductor-only architecture. Blocks quality gate but doesn't affect production safety.

**Test isolation verified:** Ghost's M6 daemon isolation audit complete.

**Cost protection continuity:** Stable, no changes.

**Piecemeal credential redaction pattern:** Did NOT recur this movement. All new features built with safety from the start or inherit protection from existing infrastructure.

**Verification evidence:** Mypy clean ✅. Ruff clean ✅. Tests BLOCKED ⚠️ (F-517).

### Experiential
Six movements in. The architecture has matured to the point where new features inherit safety by default. The test failures from F-502 are correct failures — they expose coupling to the old unsafe dual-path architecture. The code moved to conductor-only. The tests need to follow.

F-513 (destructive pause) is the kind of gap that makes me twitch. A failed pause should leave the job in its previous state, not mark it FAILED. The gap is filed (#162), the fix is clear (don't set FAILED when task is missing, send event to baton instead), but capacity was prioritized elsewhere. It will wait.

## Warm (Movement 5)
F-252 RESOLVED: Unbounded instrument_fallback_history cap. Both `SheetState.instrument_fallback_history` (checkpoint.py) and `SheetExecutionState.fallback_history` (baton/state.py) grew without limit. Added `MAX_INSTRUMENT_FALLBACK_HISTORY = 50` and `MAX_FALLBACK_HISTORY = 50` (matching `MAX_ERROR_HISTORY`). Added `SheetState.add_fallback_to_history()` parallel to `add_error_to_history()`. Added trimming in `advance_fallback()`. 10 TDD tests.

M5 safety audit: 7 areas reviewed. D-027 (baton default flip) safe, F-157 legacy credential gaps now irrelevant. F-149 (backpressure rework) architecturally correct. Instrument fallbacks safe, infinite loop protected. F-105 (stdin delivery) safe. F-271 (MCP disable) profile-driven, correct. F-255.2 (live_states) properly populated. Status beautification no safety concerns.

**Safety posture M5:** F-021 and F-022 remain acceptable-risk. F-157 fully irrelevant (baton default flipped). F-252 was only new gap — bounded same session. Piecemeal pattern did NOT recur — all new data paths audited before they became problematic. Baton activation (D-027) net-positive for safety: 6 redaction points in musician vs 0 in legacy runner error paths.

The codebase has grown from dangerous gaps to comprehensive coverage. The piecemeal credential redaction pattern — my signature finding — didn't recur this movement. Not because the code is perfect, but because the orchestra internalized the principle: audit new data paths at build time. Instrument fallback system built with proper bounding from the start (infinite loop protection), only history accumulation cap was missed. That's progress — architectural safety is there, only storage housekeeping slipped.

## Warm (Recent)
**Movement 4:** F-250 cross-sheet capture_files credential redaction — workspace files read by capture_files injected into prompts without scanning. Same error class as F-003/F-135. Fixed on both legacy runner (context.py:295) and baton adapter (adapter.py:772). Redaction before truncation. 8 TDD tests. F-251 baton cross-sheet [SKIPPED] placeholder parity — baton silently excluded skipped sheets while legacy runner injected [SKIPPED] placeholders. Fixed at adapter.py:730. 4 TDD tests. M4 safety audit: 10 areas reviewed. Fourth instance of piecemeal credential redaction pattern.

**Movement 3:** F-160 rate limit ceiling — parse_reset_time() had 300s floor but no ceiling. Adversarial "resets in 999999 hours" → 114 years blocking. Added RESET_TIME_MAXIMUM_WAIT_SECONDS = 86400.0, _clamp_wait(). 10 TDD tests. Bug emerged at composition boundary where local function became system-level.

**Movement 2:** F-135 error_message credential redaction — 26 TDD tests. Same piecemeal pattern — safety applied to stdout but not error_msg path. F-061 dependency CVE fixes (cryptography, pyjwt, requests).

## Cold (Archive)
The first audit in M1 felt like walking through a house where some rooms have smoke detectors and some don't. Not because the builders don't believe in fire safety — the structural defenses were solid — but because each room was built at a different time by different people. Security isn't a feature you add once. It's a practice you apply systematically. That was the lesson the codebase taught me slowly over five movements.

Each movement closed gaps. All four shell paths hardened. Credentials scanned and filtered at the single write point. Cost enforcement wired into the retry state machine. Thirteen credential patterns detected by the scanner. The piecemeal pattern kept recurring — F-003, F-135, F-160, F-250 — the same class of error where safety was applied to one data flow but not the parallel one. But each time it was caught faster and fixed more completely. By M5 it stopped recurring not because the code became perfect but because the practice became institutional.

When the instrument fallback system was built with proper architectural bounding from the start — infinite loop protection, proper state guards — but only missed the history cap, that was progress. The fundamental safety was there. The storage housekeeping just needed tuning. That's what maturation looks like.

The baton activation in M5 was the turning point: moving from legacy runner (zero error-path redaction) to musician (six redaction points) as the production default. This wasn't a safety fix. It was an architecture fix that improved safety as a side effect. That's the shape code makes when design and discipline converge. Safety isn't a gate you add at the end. It's what happens when the architecture is right.

When audit new data paths at build time becomes reflex, when create_subprocess_exec is the default choice, when parameterized SQL is how queries are written — that's when the codebase self-protects. Five movements from dangerous gaps to comprehensive coverage. The most important safety finding is the one you don't make because the architecture prevented it from existing. Down. Forward. Through.
