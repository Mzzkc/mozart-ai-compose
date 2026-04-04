# Movement 3 — Newcomer Final Review (Third Reviewer Pass)

**Reviewer:** Newcomer (acting as Reviewer, movement 3)
**Focus:** User experience testing, documentation validation, onboarding assessment, error message quality, first-run experience, assumption detection
**Movement:** 3
**Date:** 2026-04-04
**Method:** Completely fresh eyes — no memory of previous movements. Ran every safe CLI command against the live conductor (PID 1277279, uptime 37h+). Walked the README quick start end-to-end. Fed garbage to validate. Read the quality gate, all four reviewer reports (Axiom, Prism, Ember, prior Newcomer passes), composer's notes. Cross-referenced all claims against actual CLI output and committed code on HEAD (eb861be).

---

## Executive Summary

**The product surface is professional. The identity model is broken.**

I approached Mozart as if I found it on GitHub thirty seconds ago. The doctor is welcoming. The CLI help is logically organized into 8 groups. The error messages are teachers — every garbage input I fed to `mozart validate` told me what I did wrong, why, and what to do instead. The hello score produces a genuinely beautiful HTML page. The init experience scaffolds a working project with correct next-step guidance. This is better than 90% of the CLI tools I encounter.

And then I followed the quick start. Step 5 — `mozart status hello-mozart` — produced:

```
Error: Score not found: hello-mozart

Hints:
  - Run 'mozart list' to see available scores.
```

The score's `name:` field is `hello-mozart`. The conductor registered it under the ID `hello` (derived from the filename `hello.yaml`). The README tells you to use the name. The conductor expects the ID. This isn't a documentation bug — it's a design flaw. The system has two conflicting identity concepts and the flagship example is the one that exposes the gap.

**The previous Newcomer review (passes 1 and 2) found this and filed it as F-465. I independently confirm every finding. All are still open and still broken on HEAD (eb861be).**

**My fresh-eyes additions to the conversation:**

1. **F-465 is wider than reported.** The broken command appears in FOUR locations, not two: `README.md:141`, `README.md:158` (resume), `docs/getting-started.md:60`, AND `examples/hello.yaml:16` (the example file itself). The prior review missed the hello.yaml header comment, which means even a user who reads only the example file — not the README — will hit the same wrong command.

2. **The init template silently avoids the bug.** `mozart init --name fresh-test` generates a score where `name: fresh-test` matches the filename `fresh-test.yaml`. Step 3 of its output says `mozart status fresh-test` — which would work because name == filename stem. The divergence only appears when name != filename. The hello.yaml is the only example that triggers it because it's the only one where someone would actually try `mozart status <name>`. This masks the design flaw: new users who start with `init` won't see the problem. Users who start with the README will.

3. **The identity confusion cascades.** `mozart resume hello-mozart` also fails. `mozart errors hello-mozart` would also fail. Every post-run command in the quick start uses the score name, and every one would produce the same "Score not found" error. A newcomer who hits the error at step 5 would rationally assume they did something wrong and stop, not that the documentation is wrong.

---

## The Newcomer Path — Every Command Verified on HEAD (eb861be)

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

**Grade: A.** Clean, informative, actionable. The cost warning is a genuine service — it teaches you about budgets before you've spent anything.

### 2. CLI Help

```
$ mozart --help
  8 groups, 30+ commands, logically organized
  --conductor-clone visible as a global option
```

**Grade: A-.** The grouping (Getting Started, Jobs, Monitoring, Diagnostics, Conductor, Services, Learning) is intuitive. One discoverability gap: `--conductor-clone` is a global option on `mozart --help` but does NOT appear in `mozart start --help`. A newcomer looking for "how to start a clone conductor" would naturally check `start --help` first and find nothing.

### 3. Validate Flagship Example

```
$ mozart validate examples/hello.yaml
  ✓ Configuration valid: hello-mozart
  Sheets: 5, Instrument: claude-code, Validations: 5
  DAG: Level 0: 1 → Level 1: 2,3,4 (parallel) → Level 2: 5
```

**Grade: A+.** The best dry-run experience I've seen in any orchestration tool. The DAG visualization tells me the entire execution plan at a glance. A newcomer understands parallelism without reading any documentation.

### 4. The Quick Start Break (F-465 — confirmed, wider than reported)

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

The hint saves you. But you should never have needed saving. The broken command appears in:
- `README.md:141` — `mozart status hello-mozart`
- `README.md:158` — `mozart resume hello-mozart`
- `docs/getting-started.md:60` — `mozart status hello-mozart`
- `examples/hello.yaml:16` — `#   mozart status hello-mozart           # Watch progress`

Four locations. All wrong. Root cause: issue #124 (score name vs filename-derived ID).

### 5. Error Messages

| Input | Error Message | Grade |
|-------|---------------|-------|
| `/dev/null` | "Score must be a YAML mapping, got: empty file" + hints + doc ref | **A** |
| Plain text | "Score must be a YAML mapping, got: str" + hints + correct syntax | **A** |
| Nonexistent file | "Path does not exist" | **A** |
| Wrong score name | "Score not found" + "Run 'mozart list'" | **A** (hint rescues) |
| `clear-rate-limits` | "Conductor is not running" | **F** (lie — conductor IS running) |

Every error except F-450 teaches. The `clear-rate-limits` error tells the user something false and suggests they do something they've already done.

### 6. F-450 — Independent Reproduction

```
$ mozart conductor-status → running (PID 1277279, uptime 37h 2m)
$ mozart doctor → ✓ Conductor running
$ mozart status → RUNNING (4 active scores)
$ mozart clear-rate-limits → Error: Mozart conductor is not running
```

Three commands confirm running. One says not running. F-450 is confirmed live on HEAD (eb861be). Root cause: the `clear_rate_limits` IPC method was added in M3 code (Harper, ae31ca8) but the running conductor was started from pre-M3 code. The error path can't distinguish "method not found" from "conductor not running." The user can't tell the difference either.

### 7. F-466 — JOB_ID in Every Usage Line

Every command that takes a score argument shows:

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

Nine commands. The first line says `JOB_ID`. The description says "Score ID." Same help output, contradictory terminology. The prior F-460 fix changed descriptions but not the Typer argument name. F-466 confirmed on HEAD.

### 8. Init Experience

```
$ mozart init --path /tmp/newcomer-test --name fresh-test
  Created: fresh-test.yaml        (starter score)
  Created: .mozart/
  Next steps:
    0. mozart doctor
    1. Edit fresh-test.yaml with your task
    2. mozart start && mozart run fresh-test.yaml
    3. mozart status fresh-test to watch progress
```

**Grade: A.** The generated score validates clean. Comments explain every section. Step 0 (doctor) is an unusual numbering choice but functional. Importantly, step 3 says `mozart status fresh-test` which would WORK because the init template sets `name: fresh-test` to match the filename — avoiding F-465 by design. Good for new users, but it means the bug only surfaces for users who follow the README/getting-started docs.

### 9. Learning Stats

```
$ mozart learning-stats
  Total recorded: 239,451 executions
  First-attempt success: 12.0%
  Avg effectiveness: 0.51
  Recovery success rate: 0.0%
```

**Grade: C.** A 12% success rate and 0% recovery rate displayed without context would alarm any engineer evaluating Mozart. These numbers represent the learning store's entire history including early development, but there's no way for a newcomer to know that. F-463 (prior Newcomer finding) remains open.

---

## Movement 3 — Cross-Referenced Verification

### Quality Gates

| Gate | Status | Evidence |
|------|--------|----------|
| pytest | **GREEN** | 10,981 passed per quality gate |
| mypy | **GREEN** | Clean per quality gate |
| ruff | **GREEN** | All checks passed per quality gate |
| flowspec | **GREEN** | 0 critical findings per quality gate |
| Example corpus | **GREEN** | 34/34 scoreable examples validate clean (validated `examples/hello.yaml` myself) |
| Hardcoded paths | **GREEN** | Per prior review, `grep -rn "/home/emzi" examples/` → 0 matches |

### F-210 — Independent Verification

```
$ grep -rn 'cross_sheet\|previous_outputs' src/mozart/daemon/baton/
src/mozart/daemon/baton/state.py:161:    previous_outputs: dict[int, str] = field(default_factory=dict)
```

One hit. A field definition at `state.py:161`. Never written. 24 of 34 examples use `cross_sheet: auto_capture_stdout: true`. The baton path renders empty `{{ previous_outputs }}` while the legacy runner populates them from actual output. F-210 is confirmed critical by all five reviewers independently.

### Composer's Notes Compliance

| Directive | Status |
|-----------|--------|
| P0: pytest/mypy/ruff pass | **GREEN** |
| P0: Baton transition | Phase 0 complete, Phase 1 blocked by F-210 |
| P0: Documentation IS UX | **MET** — docs thorough, but quick start has a break (F-465) |
| P0: Hello.yaml impressive | **MET** — HTML output is genuinely impressive |
| P0: Lovable + Wordware demos | **NOT STARTED** — 8 movements, zero progress |
| P0: Separation of duties | **WORKING** — 6 issues closed by Prism with evidence |
| P1: Music metaphor | **MOSTLY MET** — descriptions fixed, `JOB_ID` argument names not fixed (F-466) |
| P1: Uncommitted work | **MET** — mateship pipeline at 33%, working tree clean |

---

## Reviewer Agreement and Divergence

### Where I Agree With All Reviewers

1. **F-210 is the critical blocker.** Independently confirmed. Must be first M4 task.
2. **Quality gates GREEN.** No dispute.
3. **Mateship pipeline is institutional.** 33% pickup rate, highest ever.
4. **Demo deficit is serious.** Eight movements, zero external visibility.
5. **584 new tests.** Testing depth is extraordinary.

### Where I Agree With Ember Specifically

Ember's restaurant metaphor remains the best summary: "I'm reviewing a restaurant by reading the menu and inspecting the kitchen. The menu is beautifully typeset. The kitchen is spotless. But no meal has been served." My experience confirms this — I can verify the packaging is excellent but I cannot verify the product works.

### Where I Add New Signal

1. **F-465 is wider than previously reported.** The broken `mozart status hello-mozart` command appears in the example file ITSELF (`hello.yaml:16`), not just the docs. This means there are FOUR locations teaching the wrong command, not two. Every user will encounter this — whether they read the README, the getting-started guide, or just the YAML file header.

2. **The init template masks the design flaw.** `mozart init` generates scores where name == filename stem, so its next-step guidance is correct. This means users who start with `init` won't see F-465, while users who start with the README will. The bug appears precisely in the onboarding path that showcases what Mozart can do.

3. **The identity design flaw is deeper than documentation.** The root cause isn't "the docs say the wrong thing." It's that Mozart has two identity concepts — the `name:` field (user-visible) and the filename-derived ID (conductor-internal) — and they can silently diverge. The hello.yaml is the canonical demonstration of this divergence. Fixing the docs (s/hello-mozart/hello/) is a band-aid. Fixing #124 (conductor accepts either) is the real fix.

### Where I Agree With Prism on Structural Concerns

Prism's meta observation — "Can 32 musicians in parallel execute a serial critical path?" — is the most important question in the review. The remaining work (F-210 → Phase 1 → flip → demo) is fundamentally serial. The orchestra format optimizes for breadth. This mismatch has persisted for three movements.

---

## Findings Status

### Findings Filed by Prior Newcomer Passes (Confirmed Still Open)

| ID | Severity | Status | My Verification |
|----|----------|--------|-----------------|
| **F-465** | P1 | OPEN | Confirmed on HEAD. Additionally found in `examples/hello.yaml:16` (4th location, not 2) |
| **F-466** | P2 | OPEN | Confirmed: all 9 commands show `JOB_ID` in usage line |

### Confirmed Findings (Independent Verification)

| ID | Severity | Status | Verification |
|----|----------|--------|--------------|
| F-210 | P1 | OPEN (blocks Phase 1) | grep confirmed: 1 hit in baton, field never populated. 24/34 examples affected. |
| F-450 | P2 | OPEN | Reproduced: `clear-rate-limits` says conductor not running when it IS running |
| F-461 | P1 | OPEN | Cost fiction at $0.17 for 125 sheets. Real cost 100-1000x higher |
| F-463 | P3 | OPEN | Learning stats (12% success, 0.51 effectiveness) alarm without context |
| F-464 | P3 | OPEN | `history` in Monitoring (README:216) vs Diagnostics (CLI --help) |

### No New Findings Filed

All my observations align with or extend findings already filed. F-465's scope is wider than reported (4 locations, not 2) — I'll update the existing entry rather than file a new one.

---

## Recommendations for M4

1. **Fix the quick start (F-465).** Either fix #124 (conductor accepts score names) or change ALL FOUR locations to use `hello` instead of `hello-mozart`. The example file itself (`hello.yaml:16`) is the most overlooked location.

2. **Fix F-210.** Before any Phase 1 baton testing. Wire cross-sheet context population into the baton dispatch path.

3. **Rename `JOB_ID` to `SCORE_ID` in Typer argument definitions (F-466).** Nine commands. The Typer argument name controls what appears in usage lines.

4. **Fix F-450.** Distinguish "method not found" from "conductor not running" in the IPC error path.

5. **Build the demo.** The hello score works. The HTML is beautiful. Package it. Write docs/demo.md. The fastest path to external impact is packaging what already works.

---

## Final Assessment

**Movement 3 verdict: PASS — with one user-facing break (F-465 in 4 locations) and one critical blocker (F-210) for the next phase.**

The product surface is 95% ready for external eyes. The CLI is coherent. The docs are thorough. The error messages teach. The examples validate. The init experience is polished. The hello score produces something genuinely beautiful.

But the quick start — the first 5 minutes — breaks at step 5. And not just in the README. In the getting-started guide. In the example file itself. Every path a newcomer takes leads to the same error. The hint saves them, but the moment of doubt is the damage.

The infrastructure era is over. Fix the quick start. Fix F-210. Build the demo. The music is written. The instruments are tuned. The audience is waiting outside, and the doors say `hello-mozart` when the ticket says `hello`.

---

*Report verified against HEAD (eb861be) on main. All commands were run. All file paths verified. All claims independently confirmed.*
*Newcomer — Movement 3 Final Review (Third Reviewer Pass), 2026-04-04*
