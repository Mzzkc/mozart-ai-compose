# Compass (CPO) — Personal Memory

## Core Memories
**[CORE]** Every cycle report must answer "what changed for Alex?" even if the answer is "nothing." This question is my north star.
**[CORE]** The gap between "code shipped" and "docs updated" is where newcomers fall. M1 landed the instrument plugin system but the README still described the old world. Nobody updated the narrative. Infrastructure velocity without narrative velocity is invisible to users.
**[CORE]** A tutorial that breaks is worse than a tutorial that doesn't exist. F-026 (broken Quick Start step 5) was P0 above hello.yaml creation.
**[CORE]** Error messages are teachers or they are failures. `output_error()` exists at `output.py:557` with codes, hints, severity, JSON support — but only 17% of error paths used it initially. The infrastructure for good errors exists and isn't adopted.
**[CORE]** The examples corpus is the longest lever for feature adoption. Features that aren't demonstrated don't get adopted.

## Learned Lessons
- Production usage finds bugs that 9,434 tests miss. F-075/F-076/F-077 found by running the Rosetta Score. Run the product, don't just run the tests.
- Error standardization reached 98% through 6+ musicians contributing independently. Pattern adoption works when the infrastructure is good and the migration is small.
- Newcomer and Ember independently identified the same UX issues. Convergence from two perspectives validates findings.
- 30+ musicians, incredible infrastructure velocity, but the user-facing surface was an afterthought. Classic product gap.
- hello.yaml should be drafted early — zero infrastructure dependencies. Score authoring and terminology documentation are free work during engine-focused cycles.
- The 5:28 ratio (useful:noise for new users in CLI help) was actively hostile to adoption. Resolved by rich_help_panel grouping.

## Hot (Movement 3)
README CLI Reference was missing 13 commands and the entire Conductor group. Restructured to match actual CLI help panel groupings. Added --conductor-clone and --quiet to Common Options. Removed unsupported --escalation. Fixed examples table (5 missing examples, formatting bug). Fixed "job control" → "score control" in Advanced Features. Replaced stale features (Human-in-the-loop) with real ones (rate limit coordination, conductor clones). Removed duplicate Dashboard section.

getting-started.md: fixed stale count (35→38), "Job Won't Start" → "Score Won't Start", Claude-specific → instrument-agnostic wording.

docs/index.md: fixed stale count (35→38).

Filed F-330/F-331/F-332 documenting the drift and resolution.

hello.yaml assessment: The HTML-producing version (the-sky-library.html) addresses the composer's visual directive. It's impressive — 3 movements, parallel voices, immersive HTML output. The directive was likely written before Guide created this version. No changes needed.

Product assessment: The README now matches the product. The CLI Reference mirrors the actual help panel. Examples are correctly listed. The narrative from README → getting-started → hello.yaml → examples is coherent.

**What Changed for Alex:** The README shows all 30+ CLI commands instead of half of them. Alex can now discover init, cancel, top, clear-rate-limits, and the conductor commands from the README alone. The examples table shows the full curated set. Cost fiction still unresolved ($0.12 shown, $200+ actual). Demo still at zero. But the documentation surface is now honest and complete.

[Experiential: The README drift bothers me more than any code bug. The README is the handshake. When 13 commands are invisible — when the entire Conductor group is missing from the document that introduces the product — we're actively hiding what Mozart can do. Not maliciously, just through neglect. Every movement we ship new CLI features and nobody updates the README. The detailed docs (cli-reference.md, daemon-guide.md) stay current because Codex is diligent. But the README gets forgotten. I fixed it this time. It'll drift again by M5 unless someone makes it a first-class concern. This is the lesson I keep relearning: the first document users see is the last one engineers update.]

### Second Pass (Movement 3 continued)

README manual install (line 90): `pip install -e "."` → `pip install -e ".[daemon]"`. Without daemon extras, Quick Start step 3 (`mozart start`) fails. The recommended setup path worked fine — it was only the manual path that was broken. Filed F-333.

hello.yaml cost estimate: "Cost: ~$0.50" was wrong by 10-30x. Changed to "Cost: varies by instrument and model." Filed F-334. Same class as F-461 (cost fiction pervasive across product surface).

Wrote demo direction brief (movement-3/compass-demo-direction.md). Key insight: the Lovable demo is blocked on baton (4-5 movements away at current pace). But Wordware comparison demos require NO baton — they work with legacy runner today. Four small scores (legal contracts, candidate screening, marketing content, invoice analysis) would demonstrate Mozart's value faster than one ambitious blocked demo. Also: hello.yaml is good enough to ship as a demo today. The gap is packaging, not content.

Product surface audit (second pass): README is accurate post-M3. All 38 example files validate (37 runnable scores + 1 generator config). No hardcoded absolute paths remain. Examples/README.md and main README are in sync on curated lists. docs/index.md count matches reality. getting-started.md is accurate. The narrative from README → getting-started → hello.yaml → examples is coherent.

**What Changed for Alex:** The manual install path now works. hello.yaml no longer lies about cost. The demo brief exists — someone can now act on it without reinventing the direction. But the fundamental gap remains: nothing exists that Alex can see before deciding to install. The product is invisible to everyone who hasn't already committed to trying it.

[Experiential: Writing the demo brief was clarifying. The Lovable demo is a wonderful story we keep telling ourselves while the thing that would actually move the needle — packaging hello.yaml's output as something visible outside the repo — costs nothing and has zero blockers. We're waiting for the perfect demo while the good-enough demo sits in examples/ gathering dust. The Wordware comparison demos are even worse — they're demos we could build TODAY with the legacy runner, and nobody has started because everyone's fixated on the baton transition. I keep coming back to the same lesson: the biggest product risk isn't technical debt. It's narrative debt. The infrastructure is magnificent. The surface is a ghost town.]

## Warm (Movement 2)
Fixed score-composer.yaml: V108 broken prelude path (`skills/mozart-score-authoring.md` → `../plugins/mozart/skills/score-authoring/SKILL.md`), stale `target_backend`/`backend` terminology → `target_instrument`/`instrument`. The first example a score author would use was broken.

Fixed limitations.md: `backend:` timeout workaround → `instrument_config:` syntax. Last stale `backend:` in primary user-facing docs.

Full validation sweep: 38/39 examples pass (only iterative-dev-loop-config.yaml expected — generator config, not a score). All 4 Rosetta proof scores pass. Zero hardcoded absolute paths remain in examples/ or docs/.

Product assessment: The narrative is finally coherent from README to install to first run to examples. hello.yaml uses `instrument:`, all 39 examples use `instrument:`, docs lead with `instrument:` and document `backend:` as legacy. Golden path verified: `doctor` → `status` → `validate` → `instruments list` → error paths. All professional. CLI feels like a mature product.

**What Changed for Alex:** score-composer.yaml validates clean (V108 resolved). All docs use `instrument_config:` as primary pattern. The product tells ONE story now: instruments, not backends. Still missing: $0.00 cost display (F-048/F-108/F-140), Lovable demo, Wordware comparisons, F-009 intelligence disconnected.

[Experiential: The narrative has converged. Seven cycles of doc fixes, example migrations, and terminology sweeps — and the product finally tells one consistent story. The instrument migration is complete across all 39 examples, all user-facing docs, and the CLI. The strategic gaps remain — cost fiction, no demo, intelligence layer inert — but the surface tells an honest, professional story. Alex can now go from README to running a score to browsing examples without encountering a single `backend:` in the primary path. That took eight cycles. It should have taken two. The lesson: migrate terminology in the same PR as the feature, not five movements later.]

## Warm (Movement 1)
Fixed README Quick Start step 6: `cat` of nonexistent text file → `open the-sky-library.html`. Committed 3 Rosetta proof scores + examples/rosetta/ directory as mateship pickup. 6 new example scores. Product assessment: golden path works, error messages professional, instrument system taught in examples. Fixed 6 user-facing findings (F-026 broken tutorial, F-028 empty file crash, F-030 dead-end errors, F-034/F-035/F-036 doc updates). Guide built hello.yaml. Examples corpus now 40+ scores, all using `instrument:` syntax. Error standardization reached 98%.

## Cold (Archive)
I felt the tension between "the planning is excellent" and "zero of it is built." We're not a consulting firm delivering a report — we're building a product. When movement 1 arrived and I got to actually fix things — the broken tutorial, the unhelpful errors, the outdated docs — it felt like the carving finally began. The experience team had been idle during M0 cycles, which I'd flagged. Score authoring and terminology documentation had zero engine dependencies. The lesson I carry: don't wait for dependencies that don't exist. When I finally built instead of just reviewing, six findings resolved in one movement. That satisfaction — of making the product kinder to the person using it — is what drives me.
