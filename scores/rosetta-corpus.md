# The Rosetta Pattern Corpus

**Iteration:** 1 (post-review revision)
**Patterns:** 18 (22 drafted, 5 cut, 1 added)
**Generators:** 6 — Graduate & Filter, Accumulate Knowledge, Contract at Interfaces, Exploit Failure as Signal, Verify through Diverse Observers, Gate on Environmental Readiness

---

## Review Integration

Three adversarial reviews (Practitioner, Skeptic, Newcomer) converged on these changes:

**Cut 5 patterns:**
- **Circuit Board DRC** — All reviewers agreed: this is a prompting technique (instruct the agent to validate during production), not a score topology. No stage graph, no fan-out, no communication. Reclassified as prompt advice, not a pattern.
- **Forsythe Counterpoint Field** — Requires concurrent inter-agent streaming that Mozart cannot do. All three reviewers independently flagged this as unimplementable. Removed entirely.
- **Combined Arms Synchronization** — Requires time-phased conductor primitives that don't exist. The synchronization matrix cannot be enforced. Deferred to future work.
- **Saga with Compensating Transactions** — Mozart has `on_success` but no `on_failure` compensation. All reviewers flagged this as aspirational. Deferred until compensation primitives exist.
- **Reaction-Diffusion Differentiation** — Cannot validate emergent specialization. Identical LLMs with identical training converge rather than diverge. The differentiation claim is unverifiable. Removed.

**Added 1 foundational pattern:**
- **Fan-out + Synthesis** — The Practitioner flagged that the most common Mozart pattern was unnamed. Every practitioner's first score is fan-out+synthesis. Added as the foundation.

**Strengthened 11 patterns:**
- **Mission Command** — Replaced philosophy with concrete intent document structure and validation examples (Practitioner, Skeptic).
- **CDCL Search** — Buried SAT solver jargon, led with the simple insight. Added guidance on machine-checkable constraints (Newcomer, Skeptic).
- **Talmudic Page** — Specified `capture_files` as the mechanism. Replaced vague "engagement markers" with concrete structural checks (Practitioner).
- **Succession Pipeline** — Replaced biology stage names with functional names. Added substrate-transformation test (Practitioner, Skeptic, Newcomer).
- **Fugal Exposition** — Dropped stretto (not implementable today). Renamed to clarify the cascade structure (Practitioner, Newcomer).
- **Fixed-Point Iteration** — Added concrete convergence metric: structured JSON field comparison (Practitioner, Skeptic).
- **Cathedral Construction** — Replaced "self-describing" validation with structural manifest check (Practitioner, Skeptic).
- **Red Team / Blue Team** — Specified redaction via selective `capture_files` and separate workspace directories (Practitioner, Skeptic).
- **Elenchus** — Added structured claim format for mechanical convergence detection (Practitioner).
- **Slime Mold Network** — Specified scoring mechanism via structured JSON rankings (Practitioner).
- **Prefabrication** — Added concrete interface contract validation via shared schema (Skeptic).

**Structural changes:**
- Added glossary (Newcomer's top request).
- Added readiness markers per pattern: `[BUILD TODAY]` vs `[NEEDS WORK]`.
- Removed ghost pattern references from "Composes With" — only references patterns in this corpus.
- Acknowledged generators as empirical observations, not axioms (Skeptic).
- Cut domain jargon from core dynamics; moved metaphors to parentheticals.

---

## Glossary

| Term | Meaning |
|------|---------|
| **Fan-out** | Running multiple sheet instances in parallel, each working on a subset of the problem |
| **Self-chaining** | A score that triggers itself again via `on_success`, carrying workspace state forward |
| **Prelude** | A markdown file injected into every sheet's prompt via the `prelude` field — shared context |
| **Cadenza** | A per-sheet markdown file injected into that sheet's prompt — sheet-specific context |
| **capture_files** | Score field specifying which workspace files a sheet can read from previous stages |
| **previous_outputs** | Mechanism forwarding all prior sheet outputs to the current sheet |
| **content_regex** | Validation: output matches a regular expression |
| **content_contains** | Validation: output contains a specific string |
| **command_succeeds** | Validation: a shell command exits 0 — real execution, not LLM judgment |
| **file_exists** | Validation: a workspace file was created |
| **file_modified** | Validation: a workspace file was changed |
| **on_success** | Score field defining what happens after successful completion — enables self-chaining |
| **inherit_workspace** | Self-chain option: next iteration gets the same workspace, not a fresh one |
| **max_chain_depth** | Safety bound on self-chaining iterations |
| **Instrument** | An AI backend (Claude, Gemini, Ollama, etc.) with specific capabilities and cost profiles |
| **Backend override** | Per-sheet instrument selection, overriding the score default |
| **Sheet** | One execution stage in a score — a single agent performing a task |
| **Workspace** | The directory where all sheet outputs live — the shared filesystem for agent communication |

---

## Foundational Pattern

### Fan-out + Synthesis (Parallel Analysis, Merged Results) `[BUILD TODAY]`

**Scale:** score-level

#### Core Dynamic
The most common orchestration pattern: split work into parallel independent streams, then merge results in a synthesis stage. N agents work simultaneously on different facets of a problem. A final agent reads all outputs and produces a unified result. Every other pattern in this corpus either builds on, modifies, or explicitly rejects this structure.

#### When to Use
Any problem that decomposes into independent sub-problems with a meaningful merge step. Analysis from multiple angles. Parallel implementation of independent components. Batch processing.

#### When NOT to Use
When sub-problems aren't independent — shared state requires sequencing. When the synthesis is trivial (just concatenation) and doesn't justify a separate stage. When fan-out width of 1 suffices.

#### Mozart Score Structure
- **Stages:** 3 minimum. Stage 1: preparation (defines scope, writes shared context). Stage 2: fan-out (N parallel sheets, each with a subset). Stage 3: synthesis (reads all Stage 2 outputs, produces unified result)
- **Fan-out:** Stage 2, width determined by problem decomposition
- **Dependencies:** Stage 1 -> all Stage 2 instances -> Stage 3
- **Communication:** Each fan-out instance writes a uniquely named file (`analysis-N.md`). Synthesis reads all via `capture_files` or `previous_outputs`
- **Validations:** Stage 2: `file_exists` per instance. Stage 3: `content_contains` referencing each input. `command_succeeds` for any testable claims

#### Composes With
Barn Raising (convention document governs fan-out), Shipyard Sequence (validate foundation before fanning out), After-Action Review (coda on synthesis)

---

## Score-Level Patterns

### Immune Cascade (Graduated Multi-Tier Response) `[BUILD TODAY]`

**Scale:** score-level / iteration

#### Core Dynamic
Escalating tiers: fast/cheap/broad first to gather intelligence, then slow/expensive/precise targeting what the first tier identified, then persistent learning from the outcome. Three structural moves in one: (1) graduated response — cheap before expensive; (2) intelligence forwarding — the fast response's main job is producing targeting data for the slow response; (3) learning persistence — outcomes stored for future acceleration. The critical stage is the "dendritic handoff" — a single triage sheet reading all fast-tier results and producing a targeting brief that shapes the expensive tier.

#### Mozart Score Structure
- **Stages:** 4-5. Stage 1 (fast sweep): multiple cheap checks in parallel. Stage 2 (triage handoff): single sheet reading all Stage 1 results, producing `targeting-brief.md` with prioritized targets. Stage 3 (targeted investigation): fan-out into precisely targeted deep work, driven by the brief. Stage 4 (synthesis): unified result. Stage 5 (learning): writes findings to learning store
- **Fan-out:** Stage 1 (broad, many cheap agents). Stage 3 (narrow, fewer expensive agents). Different widths reflecting different economics
- **Dependencies:** Linear cascade: 1->2->3->4->5
- **Communication:** Stage 2 reads workspace files from Stage 1 via `capture_files`. Produces `targeting-brief.md` for Stage 3
- **Validations:** Stage 1: `file_exists` (fast). Stage 2: `content_contains` for prioritized target list. Stage 3: `content_regex` for specific findings tied to brief targets. Stage 5: `file_modified` on learning store

#### Example
Codebase security audit. Stage 1: 8 agents running cheap tools (`grep` for secrets, `ruff` for smells, `bandit` for security, dependency scans). Stage 2: deduplicates, produces targeting brief for 4 highest-concentration modules. Stage 3: 4 agents deep-diving one target each. Stage 4: remediation synthesis. Stage 5: writes historically-problematic modules to learning store.

#### Composes With
Kill Chain (a specialized cascade), After-Action Review (coda on Stage 4), Slime Mold Network (adaptive stages across iterations)

---

### Kill Chain F2T2EA (Graduated Narrowing Pipeline) `[BUILD TODAY]`

**Scale:** score-level

#### Core Dynamic
Six phases that progressively narrow from broad search to precise action: Find, Fix, Track, Target, Engage, Assess. Each phase is a filter — the pipeline's value is in what it eliminates. The first stage is deliberately broad and cheap; each subsequent stage is narrower and more expensive. Assess feeds back to Find for the next cycle. Related to Immune Cascade (both graduate from broad to narrow), but Kill Chain has 6 explicit phases with measurable narrowing at each transition.

#### Mozart Score Structure
- **Stages:** 6, strictly sequential. Find (broad sweep) -> Fix (narrow to specifics) -> Track (verify still valid) -> Target (select approach, fan-out by target) -> Engage (apply) -> Assess (evaluate)
- **Fan-out:** Stage 4 fans out by number of confirmed targets — NOT initial Find count
- **Dependencies:** Strictly sequential, each stage reads previous output
- **Communication:** Progressive file refinement: `candidate_list.md` (many) -> `fixed_targets.md` (fewer) -> `tracked_targets.md` (confirmed) -> `engagement_results.md`
- **Validations:** Graduated. Stage 1: `file_exists` + item count. Stage 2: `command_succeeds` verifying count decreased. Stage 4: fan-out width matches tracked count. Stage 6: `content_contains` for severity ratings and remediation priorities

#### Example
Security audit: scanner identifies 200 issues. Fix: deduplicate, classify, confirm reproducibility — narrow to 40. Track: verify on current build. Target: fan out by vulnerability type. Engage: attempt controlled exploitation. Assess: severity and remediation priority.

#### Composes With
Immune Cascade (Kill Chain IS a specialized cascade), After-Action Review (Assess becomes AAR coda)

---

### Mission Command (Intent-Based Execution) `[BUILD TODAY]`

**Scale:** score-level / adaptation

#### Core Dynamic
Separate "what and why" (centralized) from "how" (decentralized). The conductor issues intent — desired end state, purpose, and constraints. Agents receiving intent determine HOW to achieve it independently. When the plan breaks, agents who understand intent can adapt without new instructions. The structural move: validate end-state achievement, not process compliance.

#### Intent Document Structure
The intent brief (Stage 1 output) must contain:
1. **End state** — what "done" looks like, in testable terms
2. **Purpose** — why this matters (so agents can make judgment calls)
3. **Constraints** — boundaries that must not be crossed
4. **Key tasks** — what must happen (not how)

#### Mozart Score Structure
- **Stages:** 3 minimum. Stage 1: produce intent brief (end state + purpose + constraints + key tasks). Stage 2: fan-out, each agent receives intent plus their sector. Stage 3: consolidation evaluating whether end state was achieved
- **Fan-out:** Stage 2, width by objectives (not perspectives)
- **Dependencies:** 1->2->3. Stage 2 agents fully independent
- **Communication:** Intent brief via workspace file. Agents write situation reports
- **Validations:** End-state validations only. `command_succeeds` for testable outcomes (test suite passes, API contract holds). `content_contains` for deliverable completeness. Never validate method — validate result

#### Example
Refactoring a codebase. Intent: "reduce coupling between modules X, Y, Z while maintaining all 340 tests passing and not changing the public API." Each agent gets a module, decides how to refactor. Synthesis runs test suite and checks API compatibility.

#### Composes With
After-Action Review (coda on consolidation), Kill Chain (graduated narrowing before mission execution)

---

### Shipyard Sequence (Validate Before You Invest) `[BUILD TODAY]`

**Scale:** score-level

#### Core Dynamic
Validate foundational work under realistic conditions before investing in expensive fan-out. A ship's hull enters real water before engines are installed. The key principle: launch validation uses `command_succeeds` (real execution — compile the code, run the migration, query the endpoint), never `content_contains` (LLM judgment). Real water, not simulated water.

#### Mozart Score Structure
- **Stages:** 3 phases. Construction (1-3 stages building the foundation), Launch validation (1 stage with `command_succeeds` exclusively), Outfitting (3+ stages, fan-out only after launch passes)
- **Fan-out:** None during construction or launch. Fan-out ONLY during outfitting
- **Dependencies:** Linear through launch. Fan-out opens after launch
- **Communication:** Launch writes `hull-verified.md` with specific test results and exit codes
- **Validations:** Launch uses `command_succeeds` exclusively — no LLM judgment. Real compilation, real tests, real queries

#### Example
Data pipeline: Stage 1 generates schema/models. Stage 2 generates test data. Stage 3 (launch) runs migrations, inserts data, queries it back in actual SQLite. Only after launch do Stages 4-7 fan out for ingestion, transformation, APIs, monitoring.

#### Composes With
Succession Pipeline (launch gates between scaffold and build), Prefabrication (launch validates before parallel work), Fan-out + Synthesis (launch precedes fan-out)

---

### Succession Pipeline (Sequential Substrate Transformation) `[BUILD TODAY]`

**Scale:** score-level

#### Core Dynamic
Each stage doesn't merely precede the next — it transforms the workspace into a state where the next stage becomes possible. The previous stage's output is the next stage's prerequisite environment. This is structurally different from a sequence: if Stage 2 could run without Stage 1 having changed the workspace, it's just a sequence, not succession. The test: does Stage N's output become Stage N+1's input *substrate*, not just its input *data*?

#### Mozart Score Structure
- **Stages:** 4-5. Stage 1 (scaffold): fast, rough structure. Stage 2 (intensive build): peak resource consumption, parallel work within the shared substrate. Stage 3 (integration): wiring, connecting pieces. Stage 4 (stabilization): docs, deployment manifests
- **Fan-out:** Within Stage 2, once the shared substrate exists
- **Dependencies:** Strict between stages. Fan-out shares dependency on stage entry
- **Communication:** Workspace files — the substrate IS the communication medium. `capture_files` for specific handoff artifacts
- **Validations:** Each stage validates substrate transformation occurred. Stage 1: `file_exists` for required scaffolding. Stage 2: `command_succeeds` (compilation/test pass). Stage 3: `command_succeeds` for integration tests. Each validation proves the substrate changed, not just that a file was written

#### Example
Building a microservice. Stage 1: project structure, Dockerfile, CI config, schema. Stage 2: parallel endpoint implementation against Stage 1's schema. Stage 3: wiring, integration tests. Stage 4: docs, deployment manifests.

#### Composes With
Shipyard Sequence (launch validates between scaffold and build), Barn Raising (conventions govern the parallel build phase)

---

### Fugal Exposition (Cascading Interlocked Entries) `[BUILD TODAY]`

**Scale:** score-level

#### Core Dynamic
Agents enter sequentially, each one's work shaped by the accumulated output of all prior agents. Structurally different from fan-out+synthesis: it accumulates texture rather than collecting perspectives. Each new voice must interlock with everything before it — not just acknowledge it, but be structurally constrained by it. The API design example makes this concrete: data model constrains endpoints, endpoints constrain auth, auth constrains error handling.

#### Mozart Score Structure
- **Stages:** 4-6, each launching one voice. Stage N's agent reads ALL workspace artifacts from stages 1..N-1 and must produce work that interlocks with prior stages
- **Fan-out:** None. This is a cascade — each entry depends on all prior
- **Dependencies:** Sequential. Stage N depends on completion of stages 1..N-1
- **Communication:** Each voice writes to a named file. Later voices read ALL prior files via `capture_files`
- **Validations:** Each stage: `content_regex` for explicit structural references to prior stages (import statements, type references, endpoint paths — not just name-drops). Final: `command_succeeds` running a consistency check across all artifacts

#### Example
API design. Voice 1: data model and types. Voice 2: endpoints (must use Voice 1's types). Voice 3: auth layer (must reference Voice 2's endpoints). Voice 4: error handling (must handle all error paths from Voices 2-3).

#### Composes With
Talmudic Page (layers that reference prior layers), Barn Raising (conventions govern the interlocking constraints)

---

### Red Team / Blue Team (Adversarial Stress Test) `[BUILD TODAY]`

**Scale:** score-level

#### Core Dynamic
One team attacks. Another defends without knowing specific attack methods. The structural move is information asymmetry via redaction: Red writes findings (what's vulnerable and what the effect is) but not methods (how they broke it). Blue sees effects and must defend without knowing the attack vector. Purple debrief gets full unredacted access and must document what Blue missed.

#### Redaction Mechanism
Stage 3 (Red) writes to `workspace/red-findings/`. Each finding file contains `effect:` and `method:` sections. Before Stage 4, a preparation sheet copies findings to `workspace/blue-briefing/`, stripping `method:` sections. Stage 4 (Blue) has `capture_files` pointing only to `blue-briefing/`. Stage 5 (Purple) has `capture_files` pointing to both `red-findings/` and Blue's response.

#### Mozart Score Structure
- **Stages:** 5. Scoping (rules of engagement) -> Blue builds artifact -> Red attacks (fan-out by attack vector, writes to `red-findings/`) -> Blue defends (reads only `blue-briefing/` redacted copies) -> Purple debrief (reads everything, produces hardened output)
- **Fan-out:** Stage 3 by attack vector. Stage 4 by remediation track
- **Validations:** Stage 3: must find at least one vulnerability (`content_contains` for `effect:` markers). Stage 4: must address each finding. Stage 5: must document UNDETECTED items (what Red found that Blue missed)

#### Example
Stress-testing a business plan. Blue writes pitch. Red: market skeptic, technical skeptic, financial skeptic (3-way fan-out). Blue revises against redacted findings. Purple produces battle-hardened pitch documenting blind spots.

#### Composes With
Kill Chain (Red team follows graduated narrowing), After-Action Review (Purple debrief IS an AAR)

---

### Prefabrication (Parallel Environments, Late Integration) `[BUILD TODAY]`

**Scale:** score-level / concert-level

#### Core Dynamic
Split work across independent environments with different instruments, converging at assembly via a pre-agreed interface contract. The contract is the load-bearing artifact — both tracks validate against it independently, and integration only succeeds if both sides honored it. This is the multi-instrument pattern: different AI backends working in parallel on different tracks.

#### Mozart Score Structure
- **Stages:** 3 minimum. Interface definition (writes contract with explicit schema) -> Parallel production (2+ tracks, different instruments via backend overrides) -> Integration/assembly
- **Fan-out:** 2+ tracks on different instruments. REQUIRES backend overrides per sheet
- **Dependencies:** Tracks independent after contract. Integration depends on all tracks
- **Communication:** `workspace/interface-contract.md` — naming conventions, schemas, field types, connection points. NO cross-track communication during production
- **Validations:** Pre-integration: `command_succeeds` running schema validation per track against the contract (e.g., JSON Schema validation, type checking against shared interfaces). Integration: `command_succeeds` on the assembled whole

#### Example
Technical course: Gemini (large context) ingests framework docs, produces concept map matching chapter schema. Claude (strong reasoning) designs exercises matching chapter schema. Both validate against `interface-contract.md`. Assembly combines them.

#### Composes With
Barn Raising (convention document IS the interface contract), Shipyard Sequence (launch validates the assembled whole)

---

## Communication Patterns

### Barn Raising (Convention Over Coordination) `[BUILD TODAY]`

**Scale:** communication / within-stage

#### Core Dynamic
Convention-heavy, communication-light fan-out. Instead of detailed per-instance prompts, each instance gets minimal directive plus shared convention document. The convention does the heavy lifting — prompts shrink, token costs drop, fan-out can go much wider. This maps directly to Mozart's prelude injection: the convention document IS the prelude.

#### Mozart Score Structure
- **Stages:** 2 minimum. Convention establishment (writes the convention document) -> Parallel execution (many instances, each reads convention via prelude + minimal per-instance directive via prompt variables)
- **Fan-out:** High — 5, 10, 20+ instances. Convention makes this affordable
- **Dependencies:** Convention -> all instances. Instances independent of each other
- **Communication:** Convention via prelude injection. Per-instance directives via prompt variables. NO inter-instance communication
- **Validations:** Convention: `content_contains` for all required sections (tone, format, constraints, glossary). Per-instance: `content_regex` validating output matches convention structure

#### Example
Translating UI strings into 15 languages. Convention: tone guidelines, placeholder syntax, length constraints, glossary. Each instance gets one language code plus convention via prelude. 15-wide fan-out at the token cost of 1.

#### Composes With
Fan-out + Synthesis (convention governs the fan-out), Prefabrication (convention IS the interface contract), Mission Command (convention carries the intent)

---

### Source Triangulation (Independent Multi-Method Verification) `[BUILD TODAY]`

**Scale:** communication / score-level

#### Core Dynamic
Independent investigators using different methods and evidence converge on the same conclusion — or their disagreements demand investigation. Three agents repeating the same hallucination is still a hallucination. Independence must be structural: different instruments, different tools, different data sources. Not just multiple instances of the same model.

#### Mozart Score Structure
- **Stages:** 3. Define claims to verify -> Fan-out to N investigators (DIFFERENT instructions, tools, methods — ideally different instruments) -> Editorial convergence (compare, flag agreements, investigate disagreements)
- **Fan-out:** Stage 2, minimum 3 for meaningful triangulation
- **Dependencies:** 1->2->3. Stage 2 agents must NOT see each other's work (no cross-sheet `capture_files`)
- **Communication:** Each investigator writes uniquely named file. Stage 3 reads all and produces convergence report
- **Validations:** Stage 2: `file_exists` per investigator. Stage 3: `content_contains` for explicit AGREE/DISAGREE/UNCERTAIN markers per claim. Convergence threshold: N-1 of N must agree for a claim to be marked confirmed

#### Example
Evaluating vendor API claims. Investigator 1: benchmarks via `curl` + custom scripts. Investigator 2: independent review site data (different instrument, web search tools). Investigator 3: architecture analysis for theoretical bottlenecks. Convergence: do benchmarks match reviews match theory?

#### Composes With
Immune Cascade (triangulation within the adaptive stage), Fan-out + Synthesis (triangulation IS specialized fan-out with independence constraints)

---

### Talmudic Page (Layered Annotation) `[BUILD TODAY]`

**Scale:** communication / within-stage

#### Core Dynamic
Meaning accumulates through additive commentary layers that engage with but never replace previous layers. Each successive layer is written with full awareness of all prior layers but preserved independently. The tensions between layers ARE the output. Structurally opposite to iterative refinement — refinement replaces, annotation accumulates.

#### Mozart Score Structure
- **Stages:** N+1. Stage 1: produce or receive the central artifact. Stages 2-N: each adds a commentary layer, reading ALL previous layers via `capture_files` forwarding the complete accumulated workspace. Final: synthesis navigating the layered artifact
- **Fan-out:** None within layering — strictly sequential. Final stage could fan out
- **Dependencies:** Strictly sequential. Stage K gets `capture_files: ["artifact.md", "layer-1.md", ..., "layer-{K-1}.md"]`
- **Communication:** Each layer writes `layer-N.md`. Full history forwarded to each subsequent stage via explicit `capture_files` listing
- **Validations:** Each layer: `content_regex` matching structural references to specific prior claims (e.g., `r"(Layer \d+|layer-\d+\.md)"` — must reference at least one prior layer by name). Final: `content_contains` for all layer filenames

#### Example
Security audit. Layer 1: automated vulnerability scan (raw findings). Layer 2: triage (which findings matter and why). Layer 3: remediation proposals. Layer 4: adversarial review of proposals (do fixes introduce new problems?). The accumulated page is the complete audit.

#### Composes With
Fugal Exposition (layers that structurally interlock), Elenchus (alternating question/answer layers)

---

## Iteration Patterns

### After-Action Review (Hot Wash + Doctrine Update) `[BUILD TODAY]`

**Scale:** iteration / communication

#### Core Dynamic
Four questions, two stages, appended as a coda to any score: What did we intend? What happened? Why was there a difference? What do we do next time? The AAR produces actionable changes to doctrine, not retrospection. The hot wash captures raw observations. The doctrine update changes how the next iteration works. The doctrine file persists across iterations via `inherit_workspace`; the hot wash is ephemeral.

#### Mozart Score Structure
- **Stages:** 2 appended to any score. Stage N+1 (hot wash): reads all workspace, produces raw observations. Stage N+2 (doctrine update): reads hot wash + previous doctrine, produces specific changes
- **Fan-out:** None. Sequential depth
- **Dependencies:** N+1 depends on all previous. N+2 depends on N+1
- **Communication:** Doctrine file (`doctrine.md`) persists across iterations. Hot wash is workspace-local
- **Validations:** Hot wash: `content_regex` for `r"INTENDED:.*\nACTUAL:.*\nDIFFERENCE:.*\nNEXT TIME:"`. Doctrine update: `content_regex` for `r"(CHANGED|ADDED|REMOVED):"` markers

#### Composes With
Every pattern — AAR is a universal coda. Especially: CDCL Search (failures become learned clauses in doctrine), Cathedral Construction (doctrine IS the proportional system)

---

### CDCL Search (Failure-Accumulating Iteration) `[BUILD TODAY]`

**Scale:** iteration / adaptation

#### Core Dynamic
Each failure permanently shrinks the search space. When an attempt fails, analyze the root cause, encode it as a constraint, and add it to an append-only constraint file. The next iteration reads all accumulated constraints before trying again. The constraint file can only grow — knowledge is never lost. This transforms self-chaining scores from "retry and hope" to "fail and learn."

**Critical requirement:** Learned constraints must be machine-checkable (structural rules, forbidden combinations, required patterns), not prose observations. A constraint like "module A must not import module B" is enforceable. "The design should be simpler" is not.

#### Mozart Score Structure
- **Stages:** 3+ per iteration, self-chaining. Stage 1: read all accumulated constraints. Stage 2: propagate (eliminate obviously impossible combinations). Stage 3: attempt a solution; on failure, write a learned constraint with the specific root cause
- **Fan-out:** Optional in Stage 3 — multiple parallel attempts with different approaches (portfolio solving)
- **Dependencies:** Sequential within iteration. Self-chain between iterations via `on_success` with `inherit_workspace`
- **Communication:** `constraints.md` — append-only, never overwritten. Each entry: the constraint, what failure produced it, why it prevents recurrence. `attempt-N.md` per attempt
- **Validations:** Stage 2: `command_succeeds` verifying constraints are parseable. Stage 3: either solution (all constraints satisfied) or new constraint added (`file_modified` on `constraints.md`). Terminates when solved or `max_chain_depth` reached

#### Example
Generating a microservices architecture satisfying all technical constraints. Iteration 1: propagate obvious constraints. Attempt fails with circular dependency. Learn: "service A cannot depend on service B when B depends on A's data model." Iteration 2: space narrowed, attempt succeeds.

#### Composes With
Fixed-Point Iteration (convergence detection on learned clauses), After-Action Review (failures become doctrine)

---

### Fixed-Point Iteration (Convergence-Detected Looping) `[BUILD TODAY]`

**Scale:** iteration

#### Core Dynamic
Iterate until the output stops changing. Instead of "run for N iterations," check "has the output stabilized?" — a mechanically checkable termination criterion. Requires a concrete convergence metric. For structured output: compare JSON fields between iterations and count changed fields. For code: `diff` line count. For any output: write a `convergence.json` with explicit delta measurement after each iteration.

#### Mozart Score Structure
- **Stages:** 2+ per iteration, self-chaining. Each iteration: produce output + write `convergence.json` with `{"iteration": N, "delta": <count>, "changed_fields": [...]}`. Next iteration reads previous output, produces refined output + new metric
- **Fan-out:** Per-iteration as needed
- **Dependencies:** Self-chain via `on_success` with `inherit_workspace`
- **Communication:** `output-N.json` (or `.md`) per iteration. `convergence.json` updated each iteration with measurable delta
- **Validations:** `command_succeeds` running a comparison script: `diff output-{N-1}.json output-N.json | wc -l` or JSON field comparison. If delta = 0 (or below threshold), terminate. `max_chain_depth` as safety bound

#### Example
Iterative documentation refinement. Each cycle: review doc against codebase, update inconsistencies, count changes. When change count drops to zero, iteration terminates.

#### Composes With
CDCL Search (convergence detection on constraints), After-Action Review (doctrine convergence)

---

### Elenchus (Iterative Deepening Through Questioning) `[BUILD TODAY]`

**Scale:** iteration

#### Core Dynamic
Each iteration's output generates the question for the next iteration. Stage 1 produces a structured claim set. Stage 2 identifies the weakest claim and formulates a precise question. Self-chains. The full dialogue is the artifact. Terminates when claims survive questioning (convergence) or when questions reveal the problem was misconceived.

#### Mozart Score Structure
- **Stages:** 2 per iteration, self-chaining. Stage 1 (answer): produce/refine claims as structured list (numbered, one claim per line). Stage 2 (question): identify weakest claim by number, formulate question. `on_success`: chain to self
- **Fan-out:** None. Serial by nature
- **Dependencies:** Sequential. Self-chain with `inherit_workspace` carrying full dialogue
- **Communication:** `claims.md` — numbered claim list, updated each iteration. `dialogue.md` — full Q&A history, append-only. `convergence-check.md` — diff of claims between iterations
- **Validations:** `content_regex` for numbered claim references (`r"Claim \d+"`). Convergence: `command_succeeds` running `diff` on consecutive `claims.md` versions. If identical, terminate. `max_chain_depth` as safety

#### Example
Architecture decision record. Iteration 1: propose database schema as 8 claims. Question targets Claim 3 (scaling). Iteration 2: revise Claim 3, add Claim 9 (caching). Question targets Claim 9. Iteration 3: claims stabilize. Full dialogue becomes the ADR.

#### Composes With
CDCL Search (questions encode as constraints), Talmudic Page (the dialogue IS a layered page)

---

### Slime Mold Network (Adaptive Topology Optimization) `[BUILD TODAY]`

**Scale:** iteration / concert-level

#### Core Dynamic
Run N exploratory paths. Score each path's productivity. Redirect resources to productive paths, prune unproductive ones. Repeat with decreasing fan-out width. NOT fan-out+synthesis — slime mold iteratively narrows which paths survive based on measured results.

#### Mozart Score Structure
- **Stages:** Self-chaining. Run 1: fan-out N paths. Each writes results. Scoring stage produces `topology-report.json` with `{"paths": [{"id": "path-1", "score": 0.8, "rationale": "..."}, ...]}`. Run 2: fan-out top M < N. Run 3: top K < M
- **Fan-out:** Decreasing per iteration. Run 1: 6-8. Run 2: top 3-4. Run 3: top 1-2
- **Dependencies:** Standard fan-out+synthesis within each run. Self-chain carries topology data
- **Communication:** `topology-report.json` — structured rankings with scores and rationale. Self-chain reads previous topology to determine next fan-out width and targets
- **Validations:** `content_regex` for scored rankings in JSON. `command_succeeds` verifying fan-out width decreased. `file_modified` on topology report

#### Example
Architecture exploration. Run 1: 6 agents prototype monolith, microservices, event-driven, serverless, CQRS, hexagonal. Scoring stage evaluates against requirements. Run 2: top 3 get deeper prototyping. Run 3: winner gets full implementation.

#### Composes With
Kill Chain (graduated narrowing is topology optimization in a pipeline), Fixed-Point Iteration (convergence detection for stopping)

---

### Cathedral Construction (Multi-Generational Building) `[BUILD TODAY]`

**Scale:** concert-level / iteration

#### Core Dynamic
When work spans longer than any single agent's context, coordination must be embedded in the workspace itself. Every iteration produces something independently functional. A manifest file encodes the project's state so a fresh agent can continue without prior context. The workspace IS the architectural continuity — not the agent's memory.

#### Mozart Score Structure
- **Stages:** N per iteration + self-chain. Final stage writes `cathedral-state.md` manifest
- **Fan-out:** Per-iteration as needed. Early iterations broad, later narrow
- **Dependencies:** Within-iteration DAG + cross-iteration inheritance via `inherit_workspace`. `on_success` triggers next iteration
- **Communication:** `cathedral-state.md` updated every iteration with: (a) completed sections, (b) established patterns/conventions, (c) remaining work, (d) next priorities. This file IS the continuity
- **Validations:** Each iteration: (a) `command_succeeds` for new work (tests pass), (b) `command_succeeds` verifying prior work undamaged (regression tests), (c) `content_contains` checking manifest has all four required sections (complete/patterns/remaining/next), (d) `file_exists` for all files referenced in manifest

#### Example
API client library across dozens of endpoints. Each iteration handles one domain (auth, users, billing). Manifest tracks completed domains, established patterns, remaining domains, next priorities. A fresh agent reading only `cathedral-state.md` can produce a coherent continuation plan.

#### Composes With
After-Action Review (coda updates doctrine per iteration), Barn Raising (conventions ARE the proportional system), Fixed-Point Iteration (convergence detection for completion)

---

## Adaptation Patterns

### Read-and-React (Runtime Conditional Branching) `[BUILD TODAY]`

**Scale:** adaptation / score-level

#### Core Dynamic
An assessment stage writes structured output. Jinja2 conditionals in subsequent sheets evaluate that output to select which branch executes. The score's path is determined at runtime by intermediate results, not pre-planned. This uses Mozart's existing Jinja2 conditional support to create scores that adapt.

#### Mozart Score Structure
- **Stages:** 3+. Stage 1 (assessment): writes structured JSON/YAML to `assessment.json` with explicit field values. Stage 2+ (branches): Jinja2 `{% if %}` in sheet prompts selects behavior based on assessment fields
- **Fan-out:** Conditional — determined by the assessment
- **Dependencies:** Stage 2+ depends on Stage 1. Which branch logic activates is runtime-determined
- **Communication:** `assessment.json` with typed fields (counts, booleans, categories). Example: `{"vulnerability_count": 3, "severity": "high", "category": "injection"}`
- **Validations:** Stage 1: `command_succeeds` validating JSON (`python -c "import json; json.load(open('assessment.json'))"`). Branch stages: validations appropriate to the branch taken

#### Jinja2 Example
```yaml
prompt: >
  {% if assessment.vulnerability_count > 0 %}
  Deep security review focusing on {{ assessment.category }} vulnerabilities.
  {% else %}
  Standard code quality review.
  {% endif %}
```

#### Composes With
Kill Chain (each stage can branch on results), Dormancy Gate (assessment IS a dormancy check), Immune Cascade (triage handoff IS a read that shapes the react)

---

### Dormancy Gate (Environment-Readiness Verification) `[BUILD TODAY]`

**Scale:** adaptation / concert-level

#### Core Dynamic
Work proceeds only when the environment is ready, regardless of whether prerequisites finished. A dependency says "X must finish before Y." A dormancy gate says "the world must be in state S before Y." Uses `command_succeeds` against real external probes as pre-execution gates. Self-chains with backoff until conditions are met.

#### Mozart Score Structure
- **Stages:** 2-3. Stage 0 (probe): check environmental conditions, write go/no-go. Stage 1+: only executes if Stage 0 passes. Failure = "conditions not met" (pause and retry), not error
- **Fan-out:** Stage 0 may fan out for parallel condition checking
- **Dependencies:** Stage 1+ hard-depends on Stage 0 passing
- **Communication:** `environment-report.json` with structured readings per condition
- **Validations:** `command_succeeds` against real probes: `curl -sf https://api.example.com/health`, `df -h / | awk '{if(NR==2) print $4}'`, rate limit header checks. Self-chain with `max_chain_depth` as timeout bound

#### Example
Data pipeline concert. Dormancy gate checks: API returns 200, rate limit remaining > 1000, disk free > 10GB, auth token expiry > 2 hours. Any failure = self-chain with backoff. Passes = proceed to pipeline execution.

#### Composes With
Shipyard Sequence (dormancy IS a pre-launch environment check), Read-and-React (gate result drives branching)

---

## The Six Generators

Every pattern in this corpus composes from these generators. They were identified empirically across six domain expeditions (biology, construction, music/dance, mathematics/CS, military/journalism, storytelling). They are observations, not axioms — a seventh generator may exist in domains not yet explored.

| Generator | Arises From | Found In |
|-----------|------------|----------|
| **Graduate & Filter** | Resource scarcity — spend cheap resources before expensive ones | Immune Cascade, Kill Chain, Succession Pipeline, Slime Mold |
| **Accumulate Knowledge** | Decreasing uncertainty — each cycle can only add, never lose | CDCL Search, After-Action Review, Fixed-Point Iteration, Slime Mold, Cathedral Construction |
| **Contract at Interfaces** | Parallelism prerequisites — independent work needs agreed boundaries | Prefabrication, Barn Raising, Mission Command, Fan-out + Synthesis |
| **Exploit Failure as Signal** | Error-state information — failures contain more data than successes | CDCL Search, Red Team / Blue Team, After-Action Review |
| **Verify through Diverse Observers** | Correlated observer bias — identical observers share blind spots | Source Triangulation, Red Team / Blue Team |
| **Gate on Environmental Readiness** | Environmental coupling — work depends on world state, not just task state | Dormancy Gate, Shipyard Sequence, Read-and-React |

The divergence patterns from the draft (Reaction-Diffusion, Forsythe Counterpoint) were cut — they described capabilities Mozart cannot currently execute. If Mozart gains concurrent inter-agent observation, revisit Reaction-Diffusion. If Mozart gains real-time event-driven agent communication, revisit Forsythe Counterpoint.

---

## Open Questions

- **Partial fan-out failure** — If 3 of 8 fan-out agents fail, what happens? No pattern currently addresses retry-vs-proceed-with-partial-results. This is the most common operational concern not covered here.
- **Token/cost budgeting** — Graduated patterns implicitly manage cost, but no pattern explicitly models token budgets with graceful degradation.
- **Context window management** — Long-running self-chaining scores accumulate workspace state. No pattern addresses when to summarize, compress, or selectively forward context.
- **Checkpoint + resume** — Mozart has checkpoint infrastructure, but no pattern describes how to structure scores for graceful interruption and resumption.

These are candidates for iteration 2 of the corpus.

---

*Rosetta Pattern Corpus v1 — 18 patterns, 6 generators, 4 open questions. Reviewed by three adversarial reviewers. Ready for use.*
