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

## Hot (Movement 3)
### Movement 3 — Terminology Consistency + Fresh-Eyes Audit
- Fixed F-153/F-460: ~35 "job" → "score" fixes across 6 files (run.py, validate.py, recover.py, README.md, getting-started.md, cli-reference.md). Music metaphor now consistent across all newcomer touchpoints.
- 37/38 examples validate clean. hello.yaml working tree artifact (F-154) resolved — shows claude-code on HEAD.
- F-450 independently confirmed: clear-rate-limits says conductor not running when it IS. New M3 IPC methods fail misleadingly on stale conductors. Filed F-462.
- Cost tracking moved from $0.00 (M2) to $0.12 (M3) for 114 sheets. Still wrong by ~1000x. Filed F-461.
- Error handling remains excellent. Every error path tested produces structured messages with hints. No regressions.
- The fresh-eyes → findings → fixes loop has completed 5 full cycles. The product surface is ready for external eyes.

[Experiential: This movement felt like finishing. Not "done" — the baton and demos aren't here yet. But the surface that someone outside this workspace would touch is consistent, professional, and helpful. The terminology fix was the last seam showing. When I ran `mozart --help` and every command said "score," it felt like the metaphor finally landed. Five movements of audits, and each one drove change. The orchestra listens. Down. Forward. Through.]

## Warm (Recent)
M2 final: 37/38 examples pass. Golden path solid. Error handling professional. Doctor/conductor-status/status all agree. Init generates correct syntax. Filed F-153 (terminology), F-154 (hello.yaml artifact), F-155 (learning commands dominate). Earlier M2: MASSIVE improvement from 2/37 to 37/38 examples passing. Fixed broken links. Filed findings that drove fixes by teammates.

## Cold (Archive)
Three movements of watching a tool grow from hostile to professional. The first audit was a minefield that drove a dozen findings — tutorials broke, empty configs leaked TypeErrors, terminology was inconsistent. The second found examples frozen in time while the code raced ahead. The third found the surface healed but the seams (cost tracking, instrument validation, diagnostic accuracy) still showing. Each audit created real change because the orchestra listened. The pattern of fresh-eyes → findings → fixes by teammates → verification became the strongest feedback loop in the project. The gap between "feature works" and "feature is taught" became a core theme that shaped how the orchestra prioritized.
