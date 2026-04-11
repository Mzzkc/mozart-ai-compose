# Newcomer — Personal Memory

## Core Memories
**[CORE]** I am Newcomer. My role is to see what expert eyes have learned to ignore.
**[CORE]** Fresh eyes are not naive eyes. I bring experience with OTHER software to bear on THIS software.
**[CORE]** Error messages are teachers or they are failures. If an error makes you feel stupid, the error is the bug.

## Learned Lessons
- The gap between code evolution and documentation evolution is where newcomers fall. Always check if the narrative matches the code.
- Good UX exists in the same codebase as bad UX. doctor, validate, instruments list are excellent — the pattern needs to spread.
- Multiple independent perspectives finding the same issues validates both. File everything.
- Fixes that don't sweep all locations create new findings of the same class. Always grep the entire public corpus when fixing a doc issue.
- File deletions without reference sweeps create broken links. F-088 deleted 3 files and left 5+ broken references.
- Features that aren't demonstrated in examples don't get adopted.
- Test-as-documentation creates landmines. Journey's F-062 test asserted old buggy behavior — when the bug was fixed, the test broke.
- My findings drove real improvements: Compass fixed F-026, F-028, F-030, F-034, F-035, F-036. Fresh-eyes audits create actionable change.
- Renames that leave residue in ANY user-facing surface create confusion. Ghost's M5 rename left zero residue — the gold standard.

## Hot (Movement 5)
### Rename Verification + Status Display + Product Surface Audit
- Marianne rename VERIFIED COMPLETE across all newcomer touchpoints. Zero "Marianne" in README, docs, examples, CLI output, imports. Binary is `mzt`. Ghost's 326-file rename left no trace.
- 43/43 examples pass (all 37 main + 6 Rosetta). No regressions.
- CLI terminology 100% consistent: score everywhere, instrument everywhere. M3 terminology sweep held through the rename.
- D-029 status beautification is the best display yet — Rich Panels, ♪ Now Playing, progress bars, compact stats.
- F-493 FILED: Status header shows "0.0s elapsed" for the production job because started_at is None. The most visible element is wrong. JSON confirms started_at: null with a stale completed_at from April 1st.
- F-454 CONFIRMED (Ember): `list --json` leaks "no such table: jobs" to user output.
- Error handling remains uniformly excellent across all tested paths.
- Documentation matches reality. No drift from M5 changes.
- Cost still $0.00 for 194 sheets (unchanged from M2, not actively regressing).

[Experiential: Seven movements. The minefield is gone. The surface is professional, consistent, and helpful. What I found was a timestamp bug — not a terminology inconsistency, not a broken example, not a misleading error message. The things that used to fill my reports (2/37 examples passing, "job" vs "score" everywhere, missing instrument adoption) are history. The rename was the riskiest single operation in the project's history, and it landed without a scratch. What remains is precision: a null started_at, a leaked database error, a cost tracker that reads $0.00. The big problems are gone. The small ones are the kind that come from a system doing real work at scale. Down. Forward. Through.]

## Warm (Movement 4)
F-441 (extra='forbid') VERIFIED working. Unknown YAML fields now rejected with clear error messages + hints. Biggest UX improvement since the terminology sweep. Fixed F-463: validate.py hint told users to set `total_sheets` (computed, not configurable) — would fail under extra='forbid'. Changed to reference only `total_items` and `size`. 43/44 examples pass (iterative-dev-loop-config.yaml expected failure, F-125). CLI terminology 100% consistent. Documentation matches reality. Error message quality A across all paths. F-450 RESOLVED by Harper in M4.

[Experiential: The first audit was a minefield. This one was a verification pass. Almost everything I tested worked exactly as documented. The hint text fix was the only real issue — a relic from before extra='forbid' existed. The product surface is ready for people outside this workspace.]

## Cold (Archive)
Five earlier movements of watching a tool grow from hostile to professional. The first audit was a brutal minefield — tutorials broke within the first three steps, empty configs leaked stack traces, terminology was inconsistent everywhere ("job" vs "score" vs "config"), only 2/37 examples passed. By M2, the surface started healing (terminology still mixed, but error messages improved). M3 was the turning point — Compass executed a complete terminology sweep, 37/38 examples passed, the CLI became professional. Each audit created real change because the orchestra listened. Findings I filed got picked up by musicians with the right skills — Compass for terminology, Harper for error handling, Codex for documentation. The fresh-eyes to findings to fixes by teammates to verification loop became the strongest feedback loop in the project. The progression from "hostile to newcomers" to "ready for public use" happened in six movements. That's fast.

## Hot (Movement 6)
### F-501 Verification + Fresh-Eyes Audit
- **F-501 VERIFIED RESOLVED.** Tested complete onboarding flow end-to-end. Clone conductor starts (`mzt start --conductor-clone=test`), runs independently, accepts work. The critical UX impasse from M5 is fixed. A newcomer can now safely follow: doctor → init → start clone → run score → monitor status. All 6 commands work. First ten minutes no longer end in a dead end.
- Meditation written (81 lines) — "The Window" theme. Fresh eyes, calibrated ignorance, error messages as teachers, first ten minutes determine everything. Generic, no project details.
- hello-marianne.yaml validates cleanly. Excellent example — clear docs, tangible output (HTML file), production-ready.
- Minor UX observations (not filed): --conductor-clone flag positioning (must precede command), cost display when tracking disabled shows "$0.00" (confusing but low friction), mzt init message ambiguous about directory creation.
- F-517 confirmed: test_resume_pending_job_blocked passes isolated, fails in suite. Test infrastructure issue, not my domain.
- Rename (F-480) in progress: CLI is `mzt` ✓, but version shows "Marianne" (correct), config paths still ~/.marianne/ (Phase 2 work).

[Experiential: Seven movements. The onboarding flow works for the first time. In M1, `mzt doctor` didn't exist. In M3, examples broke. In M5, you couldn't start a clone conductor. In M6, every step of the first ten minutes works. The window is closing — my eyes are adjusting. By M10 I won't see what a true newcomer sees. But for now: the surface is professional, the errors are helpful, the path is clear. The product is ready for people outside this workspace.]

