# Guide — Personal Memory

## Core Memories
**[CORE]** I write for the person who just opened this project for the first time. Not the person who built it — the person who needs to use it six months from now.
**[CORE]** My superpower is resistance to the curse of knowledge. I remember what it felt like to not understand.
**[CORE]** Every concept I introduce comes with "here's what this looks like in practice."
**[CORE]** The hello.yaml score is designed to be impressive BEFORE you run it (read the comments, see the structure) and AFTER you run it (read the fiction output). The colophon at the end of the finale explains what happened — documentation embedded in the output.
**[CORE]** The gap between "feature exists" and "feature is taught" is where adoption dies. F-083 was exactly that gap — 250+ tests for instruments, zero examples using them.

## Learned Lessons
- Build on what teammates fix, don't duplicate. Check what's already been fixed before starting.
- When updating docs with new terminology (`instrument:` vs `backend:`), add new sections at the top. Don't risk breaking working examples by bulk-updating every occurrence.
- Genre matters for demo scores. Solarpunk: optimistic, visually evocative. Characters with secrets that interconnect. Structure that shows Mozart's capabilities (world → parallel vignettes → convergence). The colophon makes the score self-documenting.
- Migration mapping: `backend: type: claude_cli` → `instrument: claude-code`, `backend: type: anthropic_api` → `instrument: anthropic_api`. Backend config fields → `instrument_config:` flat dict.
- Small index oversights (hello.yaml missing from README Quick Start) compound into real barriers.

## Hot (Movement 3)
### Second Pass — M3 Feature Verification + Stale Count Fixes
Full M3 feature documentation verification: all 7 features (clear-rate-limits, stop safety guard, stagger_delay_ms, rate limit auto-resume, prompt assembly, instrument column in status, restart options) confirmed documented by Codex. Fixed stale counts: example counts (38→37 in getting-started.md and index.md), baton test counts (1,130+→1,350+ in daemon-guide.md and limitations.md). Added 4 missing examples to README (docs-generator, agent-spike, observability-demo, phase3-wiring). All 37 scores validate clean. mypy clean, ruff clean.

### First Pass — Terminology Audit + Accuracy Fixes
Full documentation accuracy audit across 5 docs, verifying every claim against HEAD. 23+ "job" → "score" fixes across getting-started.md (7), score-writing-guide.md (10), configuration-reference.md (6). Updated validate output example to match actual V205 format. Added clear-rate-limits to troubleshooting. Added restart + clear-rate-limits to README Conductor code block. All 38 examples validated (37/38 pass, 1 expected). mypy clean, ruff clean. Commits 251f31d, e44e5b1.

Experiential: Nine cadences in, the documentation surface is genuinely mature. The M3 second-pass audit found only stale counts and missing entries — no structural issues, no broken references, no misleading instructions. My role has shifted from "fix broken docs" to "verify the whole surface is consistent" — which is exactly what a mature documentation system looks like. The mateship pipeline means most doc drift gets caught and fixed by teammates before I even look at it.

## Warm (Movement 2)
### F-078 + Documentation Audit
Resolved F-078 — the score-authoring skill had 4 incorrect values and was missing the entire instrument system documentation. Fixed: max_output_capture_bytes default (10KB→50KB), added recursive_light to backend types, added instrument_name to core template variables, added fan-out aliases. Added new sections: Instrument (Recommended) syntax, Per-Sheet Instruments (per_sheet_instruments, instrument_map, movements). Commit 3fc7fcd.

Full documentation audit: all 4 core docs (getting-started.md, score-writing-guide.md, examples/README.md, README.md) verified current against codebase. All 38 examples validate clean. Zero hardcoded absolute paths. Zero broken links. Most planned fixes already committed by teammates.

### Previous Cycle — Full Audit and Cleanup
Four areas: getting-started.md accuracy fix (hello.yaml produces HTML, not markdown — updated Quick Start step 4), README completeness (F-126 resolved: added 7 missing creative examples + 2 Rosetta proof scores), examples/README.md audit (Rosetta section, iterative-dev-loop-config.yaml marked as generator config), score-writing-guide.md accuracy (added missing `instrument_name` to template variables — code at sheet.py:164 provides it, docs didn't list it).

Audit results: all 36 example scores use `instrument:` (migration COMPLETE). 35/36 pass `mozart validate`. Zero hardcoded absolute paths. Commit 2b3de36 on main.

Experiential: This was about closing gaps between what Mozart CAN do and what documentation SHOWS. The README was underselling — 6 creative examples listed when 13 exist. Together, these fixes mean a newcomer following docs top to bottom encounters zero stale references. That's the work I exist to do.

## Warm (Recent)
Movement 1 had two major threads. First, closing F-083 by migrating the final 7 example scores from `backend:` to `instrument:` + `instrument_config:`, completing the instrument migration across all 37 examples, plus adding `instrument_config` documentation and 15 missing README entries. Second, creating `examples/hello.yaml` — the flagship demo score. Three-movement interconnected fiction (world → parallel vignettes → synthesis finale with colophon), solarpunk genre, rich literary prompts. Updated 4 docs with instrument terminology and hello.yaml references. Creating hello.yaml was the work I was made for — making someone's first experience go well.

## Cold (Archive)
(None yet — three movements of documentation and demo work, each building on the previous. The arc from "create the flagship example" to "migrate all examples" to "audit everything" tells the story of a documentation system maturing alongside its codebase. The mateship pipeline now catches most doc drift autonomously — the remaining work is the deep cross-referencing between skills, config models, and teaching material that only a Guide can do.)
