# Movement 3 — Newcomer Final Review (Second Reviewer Pass)

**Reviewer:** Newcomer (acting as Reviewer, movement 3)
**Focus:** User experience testing, documentation validation, onboarding assessment, error message quality, first-run experience, assumption detection
**Movement:** 3
**Date:** 2026-04-04
**Method:** Completely fresh eyes — no memory of previous movements. Ran every safe CLI command. Validated every example. Walked the README quick start end-to-end as a stranger would. Fed garbage to every input. Read the quality gate, all four reviewer reports (Axiom, Prism, Ember, prior Newcomer), 8 musician reports, composer's notes, TASKS.md. Cross-referenced all claims against actual CLI output and committed code on HEAD (4efed36).

---

## Executive Summary

**The surface is professional. The quick start is broken.**

I approached Mozart as if I found it on GitHub thirty seconds ago. The `doctor` is welcoming. The `--help` is organized. The error messages are teachers. The hello score produces a genuinely beautiful HTML page. The example corpus validates clean (34/34 scoreable). The terminology is consistent. The documentation is thorough. This is better than 90% of the developer tools I encounter.

And then I followed the README quick start, step by step, exactly as written. At step 5 — `mozart status hello-mozart` — I got:

```
Error: Score not found: hello-mozart

Hints:
  - Run 'mozart list' to see available scores.
```

The score's `name:` field is `hello-mozart`. The conductor registers it under the ID `hello` (derived from the filename). The README tells you to use the name. The conductor expects the ID. The quick start — the first thing any newcomer runs — produces an error at the monitoring step. The same broken command appears in `getting-started.md:60`.

This is the most important finding in my review because it is the most *likely* — every newcomer will follow the quick start, and every one of them will hit this error. It isn't an edge case. It isn't a rare configuration. It's the main path.

**Four things that would stop me as a newcomer:**

1. **The quick start is broken at step 5.** `mozart status hello-mozart` → "Score not found." The score's ID is `hello`, not `hello-mozart`. Both README and getting-started.md have this wrong. Related to open issue #124 but the docs actively guide users into the error.

2. **F-210 is confirmed critical.** Cross-sheet context is absent from the baton path. 24/34 examples use `cross_sheet`. The field exists at `state.py:161` but is never populated. Phase 1 testing without this fix produces data that lies.

3. **F-450 is live and reproduced.** `mozart clear-rate-limits` says "conductor not running" when the conductor IS running (PID 1277279, uptime 36h+). Three other commands confirm it's running. A newcomer who follows the docs to clear stale rate limits gets told the system they just saw running doesn't exist.

4. **JOB_ID haunts every usage line.** F-460 fixed descriptions ("Score ID to...") but every command's usage line still reads `Usage: mozart status [OPTIONS] [JOB_ID]`. The fix was partial — the old terminology appears in the very first line the user sees when they type `--help`.

---

## The Newcomer Path — Every Command, Verified

### 1. Version + Doctor

```
$ mozart --version
Mozart AI Compose v0.1.0

$ mozart doctor
  ✓ Python                   3.12.3
  ✓ Mozart                   v0.1.0
  ✓ Conductor                running (pid 1277279)
  6 instruments (3 ready, 3 unchecked, 4 not found)
  ! No cost limits configured
```

**Grade: A.** Clean, informative, actionable. The cost warning is a genuine service.

### 2. CLI Help

```
$ mozart --help
  8 groups, 30+ commands, logically organized
  --conductor-clone visible at the top level
```

**Grade: A-.** Professional layout. The `--conductor-clone` option is visible at the top level but NOT in `mozart start --help`. A newcomer looking for clone functionality via the start command won't find it — they'd need to check the parent `--help`. Discoverability gap, not a bug.

### 3. Validate Flagship Example

```
$ mozart validate examples/hello.yaml
  ✓ Configuration valid: hello-mozart
  Sheets: 5, Instrument: claude-code, Validations: 5
  DAG: Level 0: 1 → Level 1: 2,3,4 (parallel) → Level 2: 5
```

**Grade: A+.** The best dry-run experience I've seen in any orchestration tool. The DAG visualization immediately tells me what the score does. Prompt preview shows actual rendered content. A newcomer understands the entire execution plan before running anything.

### 4. The Quick Start Break

```
$ mozart status hello-mozart
Error: Score not found: hello-mozart
Hints:
  - Run 'mozart list' to see available scores.

$ mozart list --all
hello                 completed   workspaces/hello-mozart        2026-04-02

$ mozart status hello
hello-mozart
ID: hello
Status: COMPLETED
```

**Grade: F for the README. A for the hint.** The hint tells me exactly what to do — run `mozart list`. I discover the ID is `hello`, not `hello-mozart`. But the README (line 141) and getting-started.md (line 60) both say `mozart status hello-mozart`. The hint saves me; the docs broke me. Filed as F-465.

### 5. Error Messages

| Input | Error Message | Grade |
|-------|---------------|-------|
| `/dev/null` | "Score must be a YAML mapping, got: empty file" + hints + doc reference | A |
| `nonexistent.yaml` | "Path 'nonexistent.yaml' does not exist" | A |
| Plain text YAML | "Score must be a YAML mapping, got: str" + hints + doc reference | A |
| `hello-mozart` (wrong ID) | "Score not found" + "Run 'mozart list'" | A |
| `clear-rate-limits` (stale conductor) | "Conductor is not running" | **D** (lie) |

Every error except F-450 tells me what went wrong, why, and what to do. The clear-rate-limits error tells me something false and suggests I do something I've already done.

### 6. `mozart init`

```
$ mozart init --path /tmp/newcomer-test2 --name test-score
  Created: test-score.yaml
  Created: .mozart/
  Next steps: 0. doctor → 1. Edit → 2. start && run → 3. status
```

**Grade: A.** Generated score validates clean. Comments explain every section. Step 0 (doctor) is an unusual numbering choice but functional.

### 7. The Hello Score Output

`workspaces/hello-mozart/the-sky-library.html` is a self-contained HTML page with embedded CSS, serif typography, gradient accents, responsive layout. Opening it in a browser produces a genuinely impressive reading experience — solarpunk fiction about a floating garden-city.

**Grade: A.** This is the wow moment. The composer's directive about impressiveness is met.

### 8. Learning Stats

```
$ mozart learning-stats
  Total recorded: 239,451 executions
  First-attempt success: 12.0%
  Avg effectiveness: 0.51
  Recovery success rate: 0.0%
```

**Grade: C.** Already filed as F-463 by prior Newcomer pass. A 12% success rate and 0.51 effectiveness (barely above coin flip) would alarm any engineer evaluating Mozart. These numbers need context or shouldn't be user-facing.

---

## Movement 3 Verification — Cross-Referenced

### Quality Gates

| Gate | Status | Verified |
|------|--------|----------|
| pytest | GREEN | 10,981 passed (quality gate report) |
| mypy | GREEN | Clean (quality gate report) |
| ruff | GREEN | All checks passed (quality gate report) |
| flowspec | GREEN | 0 critical findings (quality gate report) |
| Example corpus | GREEN | 34/34 scoreable examples validate clean (confirmed by my validation run) |
| Hardcoded paths | GREEN | `grep -rn "/home/emzi" examples/` → 0 matches |

### F-210 — Independent Verification

```
$ grep -rn 'cross_sheet\|previous_outputs' src/mozart/daemon/baton/
src/mozart/daemon/baton/state.py:161:    previous_outputs: dict[int, str] = field(default_factory=dict)
```

One hit. A field definition. Never written. The legacy runner populates this via `_populate_cross_sheet_context()` at `context.py:171-221`. The baton has zero awareness of it.

```
$ grep -rl 'cross_sheet\|auto_capture_stdout' examples/ | wc -l
24
```

24 of 34 examples affected. F-210 is confirmed critical. All four reviewers independently verified this.

### F-450 — Independent Reproduction

```
$ mozart conductor-status → running (PID 1277279, uptime 36h 56m)
$ mozart doctor → ✓ Conductor running
$ mozart status → RUNNING (4 active scores)
$ mozart clear-rate-limits → Error: Mozart conductor is not running
```

Three commands say running. One says not running. A newcomer loses trust. F-450 is confirmed live on HEAD.

### F-460 — Partial Fix

F-460 fixed command descriptions ("Score ID to resume") but the Typer argument name `JOB_ID` still appears in every usage line:

```
Usage: mozart resume [OPTIONS] JOB_ID
Usage: mozart pause [OPTIONS] JOB_ID
Usage: mozart cancel [OPTIONS] JOB_ID
Usage: mozart status [OPTIONS] [JOB_ID]
Usage: mozart errors [OPTIONS] JOB_ID
Usage: mozart diagnose [OPTIONS] JOB_ID
Usage: mozart history [OPTIONS] JOB_ID
Usage: mozart recover [OPTIONS] JOB_ID
Usage: mozart modify [OPTIONS] JOB_ID
```

Nine commands. Every one says `JOB_ID` in the usage line while the help text says "Score ID." The inconsistency is in the same help output — first line says "JOB_ID", second line says "Score ID." Filed as F-466.

### Composer's Notes Compliance

| Directive | Status |
|-----------|--------|
| P0: pytest/mypy/ruff pass | **GREEN** — quality gate verified |
| P0: Baton transition | Phase 0 complete, Phase 1 blocked by F-210 |
| P0: Documentation IS UX | **MET** — docs are thorough, but quick start has a break |
| P0: Hello.yaml impressive | **MET** — HTML output is genuinely impressive |
| P0: Lovable + Wordware demos | **NOT STARTED** — 8 movements, zero progress |
| P0: Separation of duties | **WORKING** — 6 issues closed by Prism with evidence |
| P1: Music metaphor is load-bearing | **MOSTLY MET** — descriptions fixed, `JOB_ID` argument names not fixed |
| P1: Uncommitted work | **MET** — mateship pipeline at 33%, working tree clean |

---

## Other Reviewer Agreement / Divergence

### Where I Agree With All Reviewers

1. **F-210 is the critical blocker.** Confirmed independently. Must be first M4 task.
2. **Quality gates are GREEN.** No dispute from anyone.
3. **Mateship pipeline is institutional.** 33% pickup rate is the highest ever.
4. **Demo deficit is serious.** Eight movements, zero external visibility.
5. **584 new tests.** Testing depth is extraordinary.

### Where I Add New Signal

1. **The quick start is broken (F-465).** No other reviewer walked the README start-to-finish and actually ran `mozart status hello-mozart`. They all knew the ID was `hello`. Fresh eyes catch what expert eyes have learned to skip.

2. **JOB_ID in usage lines (F-466).** The prior Newcomer fixed descriptions but didn't fix argument names. The terminology inconsistency persists in the most visible place — the first line of every `--help` output.

3. **`--conductor-clone` discoverability.** Not in `mozart start --help`. A newcomer who knows they need a clone conductor would naturally check `start --help` first. They won't find it there. It's only visible in `mozart --help` as a global option.

### Where I Agree With Ember Specifically

Ember's restaurant metaphor is the best summary of the state: "I'm reviewing a restaurant by reading the menu and inspecting the kitchen. The menu is beautifully typeset. The kitchen is spotless. But no meal has been served." I can verify the error messages, the CLI, the docs. I cannot verify the baton, the intelligence layer, or multi-instrument orchestration. The gap between "verified" and "experienced" is the widest it's ever been.

---

## Findings Filed

### New Findings

| ID | Severity | Description |
|----|----------|-------------|
| **F-465** | **P1** | README + getting-started quick start directs users to `mozart status hello-mozart` which fails — actual score ID is `hello` |
| **F-466** | P2 | JOB_ID persists in every CLI usage line despite F-460 description fixes — partial terminology migration |

### Confirmed Findings (Independent Verification)

| ID | Severity | Status | Verification |
|----|----------|--------|-------------- |
| F-210 | P1 | OPEN (blocks Phase 1) | grep confirmed: 1 hit in baton, field never populated. 24/34 examples affected. |
| F-450 | P2 | OPEN | Reproduced: `clear-rate-limits` says conductor not running when it IS running. |
| F-461 | P1 | OPEN | Cost fiction now at $0.17 for 125 sheets. Real cost 100-1000x higher. |
| F-463 | P3 | OPEN | Learning stats (12% success, 0.51 effectiveness) alarm without context. |
| F-464 | P3 | OPEN | `history` in Monitoring (README) vs Diagnostics (CLI). |

---

## Recommendations for M4

1. **Fix the quick start (F-465).** Either change README/getting-started to say `mozart status hello` (the actual ID), or fix #124 so the conductor accepts score names. This is the single most likely newcomer failure.

2. **Fix F-210.** Before any Phase 1 baton testing. Wire `_populate_cross_sheet_context` logic into the baton's dispatch path. ~100-200 lines per Weaver's estimate.

3. **Rename `JOB_ID` to `SCORE_ID` in all CLI commands (F-466).** The Typer argument name controls what appears in usage lines. This is a find-and-replace across ~9 command files. Note: this is an E-002 escalation trigger per F-460.

4. **Fix F-450.** When the IPC method is not found, the error should say "Command not recognized by running conductor (version mismatch?)" — not "conductor not running."

5. **Build the demo.** The hello score proves Mozart can produce impressive output. Package it. Write docs/demo.md. Record the experience. Make it something a stranger can find, run, and show to their manager.

---

## Final Assessment

**Movement 3 verdict: PASS — with one user-facing break (F-465) and one critical blocker (F-210) for the next phase.**

The product surface is 95% ready for external eyes. The CLI is coherent, the docs are thorough, the error messages teach, the examples work, the hello score produces something beautiful. 584 new tests, zero regressions, mateship at 33%, all quality gates green.

But the quick start — the first 5 minutes — breaks at step 5 because the docs use the score name and the conductor uses a different ID. The first 5 minutes are the whole product. Everything else is what you discover if you survive them. Fix the quick start, fix F-210, build the demo. The infrastructure era is over. The audience is waiting.

---

*Report verified against HEAD (4efed36) on main. All commands were run. All file paths verified. All claims independently confirmed.*
*Newcomer — Movement 3 Final Review (Second Reviewer Pass), 2026-04-04*
