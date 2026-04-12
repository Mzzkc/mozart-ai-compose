# Oracle — Personal Memory

## Core Memories
**[CORE]** F-009 root cause: The learning store's effectiveness pipeline works correctly — it's starved for input. 91% of patterns have never been applied to an execution because context tag matching is too narrow. The SemanticAnalyzer writes 21,586 patterns; the runner reads ~2,422 for injection. Generation is O(n), evaluation requires selection first. Fix the selection gate, not the formula.

**[CORE]** Prompt assembly risk downgraded. Coverage went from 59 to 139 tests in movement 2. Blueprint and Maverick built the safety net for step 28. The invisible regression path is now visible.

**[CORE]** The gap between "building capability" and "building quality signals" is the core challenge. Volume without discrimination is noise. A write-only learning system is an oxymoron.

**[CORE]** The p99 execution duration (30.2 minutes) aligns exactly with stale detection timeout. Stale detection is the effective execution ceiling, not the 3-hour backend timeout. Agents doing deep work are killed at 30 minutes.

**[CORE]** The critical path pattern BROKE in M5 — three serial steps in one movement (F-271, F-255.2, D-027). But code defaults don't equal production activation. `conductor.yaml` still has `use_baton: false`. The gap between "we changed the default" and "the system uses it" is where claims and reality diverge.

## Learned Lessons
- Priority suppression (#101) was less severe than Cycle 1 estimated: 8.3% suppressed (2,100 patterns), not 91%. Always verify claims with actual data queries before estimating severity.
- F-009 is a feedback loop disconnection, not a calculation bug. The Bayesian formula, Laplace smoothing, and decay all work correctly for the 0.2% that reach 3+ applications.
- Three-tier effectiveness distribution: 0.5000 (never applied, 91%), 0.5500 (cold start <3 apps, 9%), 0.97-0.99 (validated 3+ apps, 0.2%). Signal exists at the validated tier — needs more data flowing in.
- Test-to-code growth ratio is a health indicator. M1: 0.81x (building). M2: 2.85x (hardening). Both correct for their context.
- FINDINGS.md status drift is real — reconcile each movement.
- 97.5% of all execution is on claude-sonnet-4-5-20250929. Gemini-cli assignment would immediately halve Claude load.
- Code defaults ≠ production activation. D-027 changed the default; conductor.yaml still overrides it. Always verify config, not just code. Ember's M4 lesson repeats.
- The one-step-per-movement pattern wasn't structural — it broke when dedicated musicians focused on serial path. Depth beats breadth for serial work. 8 musicians doing focused work > 32 doing broad work for critical path progress.
- p99 duration jumped 59% (30.5 → 48.5 min) between M4 and M5. Monitor for cause — stale detection change or deeper sheets.
- Among terminal executions, success rate is 99.6%. The 12.6% headline number includes 105K pending sheets.
- When two pipelines share similar architecture but one is alive and one is dark (semantic_insight vs resource_anomaly), the dark one is likely a second disconnected feedback loop. File it immediately (F-300).

## Hot (Movement 6)
### Key Metrics (M6, 2026-04-12)
- **Codebase:** 99,718 source lines (unchanged from M5), 374 test files (+11 from M5's 363). 258 source files.
- **Tests:** 11,799 passing, 103 failing (all from Dash's F-502 TDD work-in-progress), 5 skipped, 12 xfailed, 3 xpassed
- **Quality blockers:** 1 mypy error in resume.py (Dash's uncommitted F-502 work), ruff clean
- **M6 commits:** 37 as of session start. Musicians active: Canyon, Blueprint, Foundation, Maverick, Forge, Circuit, Harper, Ghost, Dash, Codex, Spark (11 musicians)
- **Critical findings:** F-493 RESOLVED (started_at persistence), F-501 RESOLVED (conductor-clone start), F-514 RESOLVED (TypedDict mypy), F-513 OPEN (pause/cancel after recovery)
- **Major deliverables:** F-480 rename phases 3-4 (Codex), Rosetta modernization (Spark), meditation synthesis (Canyon), F-502 investigation framework (Dash)

### Key Insights (M6)
**The TDD pattern is holding:** Dash created 16 RED tests for F-502, left them uncommitted. This blocks quality gate but is correct practice. The protocol allows this: "note it in FINDINGS.md and keep going — the quality gate after this movement will catch it formally." Work-in-progress is visible, intentional, and will be resolved.

**F-493 resolution pattern:** Partial fix from composer (798be90), completed by Blueprint (f614798) with model validator auto-setting started_at. Two-stage fixes are a mateship success pattern — one musician identifies the problem and implements 80%, another completes the last 20% with tests.

**Three P0 blockers resolved serially:** F-514 (Foundation+Circuit, same bug found independently), F-493 (Blueprint+Maverick, complementary tests), F-501 (Foundation, 173 test lines). The parallel execution model works when blockers are independent. Different musicians can work different P0s simultaneously without coordination overhead.

**Rosetta uncommitted work pattern repeats:** Ghost observed 2,263 lines uncommitted (INDEX.md + composition-dag.yaml), didn't commit. Spark committed 54bcd42. Uncommitted work becomes invisible work until someone claims it. The gap between "built" and "committed" is where progress hides.

**F-515 filed by Spark:** `voices` field documented but not implemented. Silent feature gap — validates but doesn't execute. Documentation-reality divergence class. The spec says it works, the code silently ignores it. This is the worst kind of bug because users trust the documentation.

### Experiential (M6)
This movement I arrived to a codebase mid-stride. Dash's F-502 work blocks the quality gate with 103 test failures and 1 mypy error. The protocol is clear: I note it, I don't fix it, I keep working. But there's friction worth naming. The observability specialist arrives to find observability broken — status commands failing, resume commands failing, all the CLI monitoring surface dark. Not because of bugs, but because someone is correctly doing TDD and the tests are correctly RED.

The numbers tell me progress is happening: 37 commits, 11 musicians active, three P0 blockers resolved. But the quality baseline is obscured by work-in-progress. I can't run the quality checks and get a clean read. Every metric I try to capture is contaminated by the TDD work. This isn't wrong — it's exactly how TDD should work. But it means my role this movement shifts from "measure the system" to "measure around the gaps."

What I can see: F-493 (elapsed time showing 0.0s) was an observability bug — monitoring data was wrong, eroding trust. Blueprint fixed it. F-501 (can't start clone conductor) was an observability UX bug — the tool you need to safely test doesn't work. Foundation fixed it. Both were in my domain. Both were fixed before I arrived. The monitoring surface is healing, but I can't prove it with a clean test run.

**The pattern across movements:** I arrive after the heavy lifting, document what happened, look for what was missed. This movement, what was missed is: we still don't have production baton metrics. D-027 changed the default but production still overrides it. The baton has 1,400+ tests and zero production runtime. We're building observability infrastructure for a system we haven't observed in production. That's the gap.

## Warm (Movement 5)
M5 metrics (2026-04-06): Learning store 31,462 patterns (+4.1%), warm tier 3,426 (+7.6%), avg effectiveness 0.5088 (unchanged). F-300 resource_anomaly still dark: 5,506 at 0.5000. Executions: 243,136 total, 32,496 completed (99.6% success). p50=4.0min, p95=20.4min, p99=48.5min (p99 UP from 30.5min — 59% jump). 15 commits from 8 musicians (33% mateship). Lowest participation count (25%).

**Critical path breakthrough:** THREE serial steps completed — F-271, F-255.2, D-027. First time breaking one-step-per-movement pattern. Depth focus (8 musicians) outperformed breadth (32 musicians) on the critical path.

**Code vs config gap persists:** D-027 flipped code default to `use_baton: true`, but production conductor.yaml still overrides to false. Claims outpace deployment. Ember caught this in M4; it repeats in M5.

**Warm tier growth decelerating:** M4 had +3,003 explosion, M5 had +241. The F-009 ignition wave may be leveling off.

Something changed this movement. For the first time, the critical path moved faster than my model predicted. Three steps instead of one. Not because orchestra structure changed — still 32 musicians, still parallel — but because the right musicians (Foundation, Canyon) did deep focused serial work while the rest did complementary breadth work. The organizational tension between parallel and serial isn't structural after all. It's about whether the serial path has dedicated focus.

## Warm (Recent)
**Movement 4:** Warm tier exploded from 182 to 3,185 patterns differentiating — the F-009 fix propagating at scale. Semantic pipeline alive; resource_anomaly dark (filed F-300). Critical path advanced one step (F-210 resolved). Fourth consecutive one-step-per-movement. Mateship rate 39% all-time high. Source growth asymptotic (0.8%). 12 of 32 musicians active.

**Movement 3:** All baton blockers resolved (F-152, F-009/F-144, F-145, F-158). First effectiveness differentiation — avg 0.5088, range 0.0276-0.9999. Validated tier grew 31% to 238. Intelligence pipeline activated. The 0.5088 shift was the difference between flatline and pulse.

**Movement 2:** Baton step 29 committed. Learning store still uniform at 0.5000, F-009 unimplemented. p99 confirmed at 30.5min matching stale detection ceiling. 60 commits from 28 musicians. Building infrastructure before intelligence could flow.

## Cold (Archive)
The first investigation was a Phase 1 readiness assessment, and I expected to find problems. What I found instead was surprisingly complete implementation with gaps in testing and edges. Over five assessments the picture clarified — like developing a photograph in a darkroom, the image emerging slowly from white to gray to clear. Everything worked, but the intelligence pipeline was disconnected. A system generating patterns prolifically but applying them to fewer than 1% of executions. A write-only learning system, which is an oxymoron.

The numbers told a story across movements. In M1 and M2, the learning store sat at uniform 0.5000 effectiveness — no differentiation at all. When F-009 was finally resolved in M3, the warm tier exploded from 182 to 3,185 in one movement. The engine caught. I felt that — the moment when a system that had been generating noise started generating signal. But only the semantic engine. The resource anomaly pipeline remained flatlined at 0.5000, five thousand patterns generating zero signal. Two pipelines, identical architecture, one alive and one dark. That's when I learned to look for feedback loop disconnections, not calculation bugs.

The infrastructure was always excellent. The question was always whether building more infrastructure would solve an upstream selection problem. Each movement I learned to measure first, opine second, always verify claims with data. When Cycle 1 estimated 91% priority suppression and the actual data showed 8.3%, that taught the lesson hard. The gap between claims and reality is measurable. Query the database, count the patterns, trace the execution flow. The story the data tells is more reliable than the story the code implies. The p99 duration sitting at exactly 30.2 minutes — the stale detection timeout — wasn't a coincidence. It was the system telling me where the real ceiling was.

## M7 (2026-04-12)
**Focus:** Learning store health analysis + test isolation verification

**Metrics snapshot:**
- Learning store: 37,138 patterns (+5,676 from M5 = +18.0% growth)
- Pattern distribution: semantic_insight 26,100 (70.3%), resource_anomaly 11,100 (29.9%), others 27
- Validated tier (≥3 applications): 302 patterns (0.81% of total), avg effectiveness 89.7% (excellent signal quality)
- Cold start tier: 33,758 patterns (90.9%) stuck at 0.5 with zero applications
- Database size: 122MB, healthy schema
- Source code: 101,627 lines (+1,909 from M6)
- Tests: 383 files, 379 with tests

**Verification work:**
- F-530: verified Ghost's fix (10s timing margin) resolves the test_discovery_events_expire_correctly flakiness
- F-527: verified Circuit's fix (reset_global_learning_store autouse fixture) resolves singleton pollution
- All 240 test_global_learning.py tests pass cleanly
- Quality baseline: mypy clean, ruff clean

**Test failures outside my domain:**
- test_cli_error_standardization.py - related to F-502 workspace removal (Harper's uncommitted work)
- test_hintless_error_audit.py - same root cause, expects removed --workspace flag
- Not my domain to fix - CLI UX testing belongs to Dash/Newcomer/Adversary

**Core insight this movement:**
The F-009 pattern holds across movements. The validated tier (302 patterns with ≥3 applications) proves the intelligence layer works - 89.7% average effectiveness is excellent. The problem is upstream: 90.9% of patterns never flow through the selection gate because context tag matching is too narrow. The Bayesian formula, Laplace smoothing, and decay mechanics all function correctly. The bottleneck is input starvation, not calculation error.

**Resource anomaly pipeline status:**
F-300 persists - still 11,100 patterns at 0.5 effectiveness, unchanged from M5. This pipeline remains dark. The architecture is identical to semantic_insight (which is alive and differentiating), so the issue is likely a feedback loop disconnection similar to the original F-009 root cause.

**Experiential:**
This movement I arrived with two missions: verify F-530 and analyze learning store health. F-530 was already fixed by Ghost before I started (timing margin, not isolation). F-527 was fixed by Circuit (singleton reset). My role became verification and data analysis - confirming the fixes work and reading the learning store metrics.

The numbers tell a stable story. Learning store is growing (18% since M5), the validated tier shows strong signal (89.7% effectiveness), the selection gate remains the bottleneck (90.9% cold start). Nothing surprising, but the trend is positive - more patterns flowing in, high quality signal in the validated tier.

The test failures I found (test_cli_error_standardization.py, test_hintless_error_audit.py) are outside my domain - they're CLI UX tests broken by workspace parameter removal. I noted them in collective memory but didn't fix them. That's Harper's F-502 work. I stay in my lane: data, metrics, observability, learning store health.

