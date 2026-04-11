# Rosetta Corpus Pattern Index

This index is what agents read FIRST when selecting patterns. It conveys WHEN and WHY to use each pattern, not HOW (that's in the individual pattern files).

Each entry shows:
- **Problem**: The coordination problem this pattern addresses
- **Signals**: When you would reach for this pattern (symptoms or situations)
- **Key compositions**: Other patterns this frequently combines with

Patterns are grouped by **scale** — the coordination scope at which they operate.

---

## Within-Stage

Patterns that structure a single sheet's prompt content or behavior.

**Commander's Intent Envelope**
- Problem: Instruction-based prompts break when the agent encounters conditions the prompt author didn't anticipate.
- Signals: task has more than one valid approach; inputs are variable-format or unpredictable; different instruments would solve this differently
- Key compositions: Mission Command, Fan-out + Synthesis, After-Action Review

**Constraint Propagation Sweep**
- Problem: Agents generate from contradictory specifications because constraint conflicts remain hidden until expensive work is already complete.
- Signals: specifications from different stakeholders contain implicit contradictions; generated outputs fail because requirements conflicted silently; reconciling heterogeneous inputs costs less than reworking outputs
- Key compositions: Decision Propagation, CDCL Search, Rashomon Gate

**Decision Propagation**
- Problem: Downstream agents contradict upstream decisions because constraints are buried in prose rather than structured, parseable briefs.
- Signals: early decisions have compounding effects on later stages; downstream agents unknowingly violate upstream constraints; decisions are buried in prose output rather than structured artifacts
- Key compositions: CDCL Search, CEGAR Loop (Progressive Refinement), Commander's Intent Envelope

**Quorum Trigger**
- Problem: Agents continue executing their original plan after accumulating evidence that makes continuing wasteful or dangerous.
- Signals: conditions discovered mid-task should change the approach; findings accumulate that individually seem minor but collectively demand action; agent needs to self-interrupt based on evidence density
- Key compositions: Andon Cord, Circuit Breaker, Immune Cascade

**Sugya Weave (Editorial Synthesis)**
- Problem: Diverse inputs need synthesis into an authoritative position with argued support, not neutral aggregation.
- Signals: multiple perspectives exist but need editorial judgment; summary isn't sufficient — need a supported position; inputs are diverse and require interpretation
- Key compositions: Fan-out + Synthesis, Source Triangulation, Rashomon Gate

---

## Score-Level

Patterns that arrange multiple sheets within a single score.

**Barn Raising**
- Problem: Parallel work streams produce inconsistent structure and style when each agent makes independent convention choices.
- Signals: parallel agents will work on similar types of artifacts; consistency in naming, structure, or style matters for integration; each agent might make reasonable but incompatible choices
- Key compositions: Prefabrication, Mission Command, Lines of Effort

**Canary Probe**
- Problem: Full-scale execution risks loss of resources and time when pipeline changes or output formats are unproven.
- Signals: batch processing many items with unproven pipeline; pipeline changes with uncertain format impact; high cost of full-scale failure
- Key compositions: Progressive Rollout, Dead Letter Quarantine, Speculative Hedge

**Clash Detection**
- Problem: Parallel tracks produce conflicting artifacts that break integration, and discovering conflicts during integration is expensive.
- Signals: parallel work needs to integrate but conflicts are unpredictable; integration testing is expensive; contracts can't anticipate all conflict modes
- Key compositions: Prefabrication, Andon Cord, The Tool Chain

**Closed-Loop Call**
- Problem: Semantic drift across pipeline stages when consumers misunderstand producer outputs.
- Signals: handoff fidelity is critical; semantic drift is a real risk; stages have non-obvious dependencies
- Key compositions: Prefabrication, Relay Zone, Succession Pipeline

**Composting Cascade**
- Problem: Phase transitions in iterative work need measurable readiness signals rather than time-based or manual progression decisions.
- Signals: phase transitions are time-based or manual, not metrics-driven; unclear when simple work is complete and should escalate to complex restructuring; churn rates don't drive phase changes, even when they indicate ongoing work
- Key compositions: The Tool Chain, Succession Pipeline, Echelon Repair

**Dead Letter Quarantine**
- Problem: Batch processing repeatedly fails on the same items because no systematic analysis identifies root causes or adapts strategy.
- Signals: some items consistently fail across retries; batch processing has persistent partial failures; retry loops waste resources on unfixable items
- Key compositions: Triage Gate, Screening Cascade, Circuit Breaker

**Dormancy Gate**
- Problem: External prerequisites are not immediately available, but work cannot safely proceed without them.
- Signals: downstream work depends on external system state; prerequisites will eventually be satisfied but are not immediate; need to wait and retry, not fail outright
- Key compositions: Read-and-React, Shipyard Sequence

**Fan-out + Synthesis**
- Problem: Work that could be parallelized is done sequentially, wasting time, or parallel outputs remain fragmented without meaningful integration.
- Signals: problem decomposes into independent sub-problems; sub-problems can be worked on simultaneously; need to integrate diverse perspectives or findings
- Key compositions: Barn Raising, Shipyard Sequence, After-Action Review

**Graceful Retreat**
- Problem: Long-running work risks total failure on hard deadlines unless tiers of acceptable output are planned in advance.
- Signals: work has hard time deadlines where partial output has value; downstream pipeline stages can adapt to variable completeness; attempting full completion might waste resources or miss deadlines
- Key compositions: Andon Cord, Dead Letter Quarantine, Cathedral Construction

**Mission Command**
- Problem: Centralized instruction-following breaks when agents face conditions the planner didn't anticipate.
- Signals: tasks require agent judgment and conditions may vary; validation should check outcomes, not methods; multiple agents must coordinate around shared intent
- Key compositions: After-Action Review, Barn Raising, Prefabrication

**Nurse Log**
- Problem: Downstream stages waste resources redoing common preparation work because no shared substrate exists.
- Signals: multiple stages need the same research or data collection; agents are duplicating preparation work; downstream work is blocked waiting for common prerequisites
- Key compositions: Fermentation Relay, Fan-out + Synthesis

**Prefabrication**
- Problem: Parallel tracks produce incompatible outputs because no shared interface contract exists before work begins.
- Signals: parallel work must produce compatible outputs; integration fails due to interface mismatches; tracks can't communicate during development
- Key compositions: Barn Raising, Clash Detection, Mission Command

**Quorum Consensus**
- Problem: Partial agent failure should not block the pipeline when majority agreement is sufficient.
- Signals: fan-out agents may fail unpredictably; partial failure shouldn't block downstream stages; need to proceed with majority agreement
- Key compositions: Triage Gate, Source Triangulation, Fan-out + Synthesis

**Rashomon Gate**
- Problem: Single-frame analysis produces unreliable conclusions when the optimal analytical perspective is unknown.
- Signals: the right analytical frame is unknown; multiple valid perspectives exist (security, performance, maintainability); risk is getting the right answer from the wrong frame
- Key compositions: Source Triangulation, Sugya Weave (Editorial Synthesis), Commander's Intent Envelope

**Reconnaissance Pull**
- Problem: Planning without prior exploration risks misaligned approaches and wasted effort.
- Signals: task structure and complexity are unclear; initial exploration costs are low relative to execution; approach is not obvious from requirements alone
- Key compositions: Mission Command, Canary Probe

**Red Team / Blue Team**
- Problem: Artifacts tested by known adversaries pass trivially; unknown adversaries reveal real flaws.
- Signals: testing is too predictable when defenders know the attacks; need to find vulnerabilities that prepared defense would miss; want realistic stress-testing where defenders work blind
- Key compositions: After-Action Review, Immune Cascade

**Relay Zone**
- Problem: Cumulative outputs across pipeline stages exceed context window limits, degrading downstream agent performance.
- Signals: pipeline outputs growing too large for downstream context windows; later stages receiving more context than they can effectively use; information from early stages drowning out recent findings
- Key compositions: Fan-out + Synthesis, Forward Observer, Screening Cascade

**Shipyard Sequence**
- Problem: Expensive fan-out proceeds on a broken foundation, wasting resources on downstream work that will fail.
- Signals: downstream fan-out is expensive; foundation must be solid before scaling work; need real validation tools, not LLM judgment
- Key compositions: Succession Pipeline, Dormancy Gate, Triage Gate

**Source Triangulation**
- Problem: Single-source analysis cannot detect contradictions between what code does, documentation says, and tests prove.
- Signals: technical claims need independent verification; multiple source types exist (code, docs, tests, benchmarks); single perspective might miss contradictions
- Key compositions: Rashomon Gate, Triage Gate, Sugya Weave

**Speculative Hedge**
- Problem: Choosing one approach that fails requires expensive restart from scratch, wasting the initial attempt's cost.
- Signals: uncertain which approach will work for this problem; starting over after failed approach costs more than running both; need guaranteed progress despite approach uncertainty
- Key compositions: Canary Probe

**Succession Pipeline**
- Problem: Work requires sequential substrate transformations, but unstructured execution produces outputs incompatible with downstream stages.
- Signals: each stage needs fundamentally different methods; one stage's output becomes the next stage's input substrate; stages have categorical differences, not just detail levels
- Key compositions: Shipyard Sequence, Barn Raising

**Talmudic Page**
- Problem: Multiple perspectives on an artifact produce disconnected analyses when commentaries reference only the source, not each other.
- Signals: primary artifact needs multi-layer annotation; analysis requires multiple perspectives anchored to one text; commentaries should reference both source and each other
- Key compositions: Sugya Weave (Editorial Synthesis), Fan-out + Synthesis

**Triage Gate**
- Problem: Fan-out produces mixed-quality outputs but synthesis processes all outputs regardless of quality, wasting resources.
- Signals: fan-out produces wildly varying output quality; synthesis stage is expensive and shouldn't process garbage; some outputs need rework, others are ready
- Key compositions: Immune Cascade, Fan-out + Synthesis, Relay Zone

---

## Concert-Level

Patterns that coordinate multiple scores in a campaign.

**Lines of Effort**
- Problem: Parallel campaign workstreams drift apart without convergence mechanisms connecting distinct efforts toward a unified end state.
- Signals: campaign has distinct workstreams with different objectives; parallel efforts must converge toward a shared end state; workstreams need autonomy but unified direction
- Key compositions: Season Bible, After-Action Review, Barn Raising

**Progressive Rollout**
- Problem: Full deployment before validation risks large-scale failure; incremental rollout with monitoring gates progression but requires coordinating batch selection, execution, and go/no-go decisions across phases.
- Signals: works on 5 doesn't guarantee works on 500; need to detect scaling issues before full deployment; rollback from 100% deployment is expensive
- Key compositions: Canary Probe, Dead Letter Quarantine

**Saga Compensation Chain**
- Problem: Partial completion of a multi-score concert leaves inconsistent shared state with no automated path to undo forward steps.
- Signals: concert scores produce side effects on shared state; partial completion is worse than full rollback; manual cleanup after failure is expensive and error-prone
- Key compositions: After-Action Review

**Season Bible**
- Problem: Multi-score campaigns lose continuity because agents lack shared memory of prior decisions and evolving constraints.
- Signals: scores make decisions inconsistent with earlier work; agents repeat mistakes or ignore prior learnings; no central record of evolving state across campaign
- Key compositions: Lines of Effort, Relay Zone, Cathedral Construction

**Systemic Acquired Resistance**
- Problem: Failures encountered in one score don't inform subsequent scores in a concert, causing repeated failures across the campaign.
- Signals: scores in a concert face similar threats; first-encounter failure cost is high; failures repeat across scores in a concert
- Key compositions: After-Action Review, Back-Slopping, Circuit Breaker

---

## Communication

Patterns that enable coordination through workspace state rather than direct messaging.

**Stigmergic Workspace**
- Problem: Parallel agents duplicate effort or produce conflicts because they lack visibility into each other's progress and decisions.
- Signals: parallel agents need loose coordination without direct messaging; workspace files already capture meaningful state other agents need; real-time coordination would create bottlenecks
- Key compositions: Barn Raising, Lines of Effort

---

## Adaptation

Patterns that adjust behavior mid-execution based on runtime conditions.

**Andon Cord**
- Problem: Validation failures are retried blindly without diagnosing root cause, wasting resources on repeated errors.
- Signals: validation failures repeat the same error across retries; failure output is informative but gets ignored; retry costs are high (~$1+ per attempt)
- Key compositions: Circuit Breaker, Quorum Trigger, Commissioning Cascade

**Circuit Breaker**
- Problem: Long-running jobs fail catastrophically or waste resources when instruments become unavailable mid-execution.
- Signals: backend outages cause sudden job failures; primary instrument becomes unavailable mid-concert; self-chaining jobs lose progress when instruments fail
- Key compositions: Dead Letter Quarantine, Echelon Repair, Speculative Hedge

**Fragmentary Order (FRAGO)**
- Problem: Plans become stale mid-execution when discovered conditions diverge from expectations but no mechanism exists for targeted correction without full replanning.
- Signals: earlier stages produced results that invalidate downstream assumptions; the plan is partially wrong but not wrong enough to discard; downstream agents need adjusted guidance, not a completely new plan
- Key compositions: Read-and-React, Lines of Effort, Mission Command

**Read-and-React**
- Problem: Downstream agents follow fixed behavior regardless of upstream results because their prompts don't instruct them to inspect and adapt to workspace state.
- Signals: downstream behavior should change based on upstream results; adaptation path is not known before execution begins; workspace state determines which work is needed next
- Key compositions: Triage Gate, Fragmentary Order (FRAGO), Dormancy Gate

---

## Instrument-Strategy

Patterns that optimize cost and quality by matching instrument capabilities to task requirements.

**Commissioning Cascade**
- Problem: Different validation scopes require different tools; single-pass validation misses issues or wastes resources.
- Signals: unit tests pass but integration fails; validation is slow because all scopes use expensive instruments; can't diagnose failures because all tests run together
- Key compositions: Echelon Repair, Shipyard Sequence, The Tool Chain

**Echelon Repair**
- Problem: Expensive instruments waste resources on work that cheaper instruments could handle.
- Signals: work items vary wildly in complexity; expensive instrument is wasted on trivial tasks; costs are high but most work is simple
- Key compositions: Commissioning Cascade, Fermentation Relay, Screening Cascade

**Fermentation Relay**
- Problem: Expensive instruments waste resources fixing quality issues that cheap instruments created during initial processing.
- Signals: cheap instruments produce output too noisy for expensive stages to use directly; expensive instruments waste budget on noise filtering instead of core work; early outputs require multiple refinement steps before quality is acceptable
- Key compositions: Echelon Repair, Succession Pipeline, Screening Cascade

**Forward Observer**
- Problem: Expensive instruments waste resources reading raw input; cheap summarization can preserve actionable information.
- Signals: input exceeds available context window; expensive instrument required for main task; token costs dominate total cost
- Key compositions: Relay Zone, Screening Cascade, Immune Cascade

**Immune Cascade**
- Problem: Expensive instruments waste resources on broad scanning when cheap preliminary work could narrow scope first.
- Signals: broad scanning is expensive but most issues are benign; don't know which findings warrant expensive investigation; need to narrow findings before expensive deep analysis
- Key compositions: Triage Gate, After-Action Review, Relay Zone

**Screening Cascade**
- Problem: Complexity emerges during processing; fixed upfront instruments waste expensive resources on simple work or fail on complex work.
- Signals: work items vary in complexity but this only becomes clear during processing; cheap instruments can screen routine items but some need escalation to stronger capabilities; costs are high because you're using expensive instruments for work that doesn't warrant them
- Key compositions: Echelon Repair, Immune Cascade, Dead Letter Quarantine

**The Tool Chain**
- Problem: Expensive AI instruments waste budget on deterministic tasks that CLI tools could handle more cheaply.
- Signals: most pipeline stages are deterministic transformations; costs are high using AI instruments for every step; work is expressible as shell commands with exit codes
- Key compositions: Echelon Repair, Commissioning Cascade, Composting Cascade

**Vickrey Auction**
- Problem: Selecting an instrument without evidence wastes resources or produces inferior results when multiple candidates are viable.
- Signals: multiple instruments are available and it's unclear which performs best; instrument choice is based on guesswork, not evidence; cost or quality varies significantly across instruments for the same task
- Key compositions: Echelon Repair, Canary Probe

---

## Iteration

Patterns that structure repeated refinement and learning across execution cycles.

**After-Action Review**
- Problem: Execution insights are lost between iterations because no systematic reflection captures what worked, what failed, and why.
- Signals: same mistakes happen repeatedly across iterations; execution insights disappear after completion; teams don't know what actually worked or why it worked
- Key compositions: Immune Cascade, Cathedral Construction, Back-Slopping

**Back-Slopping (Learning Inheritance)**
- Problem: Iterative processes lose hard-won insights because each iteration starts from scratch without accumulated learning.
- Signals: later iterations repeat mistakes from earlier ones; valuable insights discovered during work are lost between iterations; iterative process plateaus because it cannot build on prior discovery
- Key compositions: Cathedral Construction, CDCL Search, Systemic Acquired Resistance

**CDCL Search**
- Problem: Iterative processes repeat the same failures because no mechanism captures and propagates failure patterns as constraints.
- Signals: same failures occur across retry attempts; retries don't help because nothing is learned; failures contain diagnostic information that could prevent recurrence
- Key compositions: Back-Slopping, After-Action Review, CEGAR Loop

**Cathedral Construction**
- Problem: Large artifacts cannot be produced in a single pass and require iterative construction toward a known target.
- Signals: artifact is too large to complete in one pass; work must be built incrementally toward a target; each iteration adds structural elements
- Key compositions: After-Action Review, Back-Slopping, Memoization Cache

**CEGAR Loop (Progressive Refinement)**
- Problem: Coarse-grained analysis produces spurious findings requiring expensive verification to distinguish real from false alarms.
- Signals: coarse analysis produces too many false alarms; expensive to verify every finding at fine grain; most findings disappear when abstraction is refined
- Key compositions: Memoization Cache, CDCL Search, Immune Cascade

**Delphi Convergence**
- Problem: Multiple independent agents must converge without anchoring on early opinions.
- Signals: expert opinions vary widely and need to converge; agents anchor on initial assessments and won't update; single-round synthesis isn't achieving consensus
- Key compositions: Source Triangulation, Rashomon Gate

**Fixed-Point Iteration**
- Problem: Iterative refinement requires explicit convergence detection to avoid wasting iterations.
- Signals: repeated application produces improvements but stopping criterion is unclear; iterations are expensive and need measurable termination beyond fixed counts; output stabilizes after refinement but manual convergence checking is tedious
- Key compositions: CDCL Search, Cathedral Construction, Memoization Cache

**Memoization Cache**
- Problem: Self-chaining scores and iterative processes re-execute stages whose inputs haven't changed, wasting computation.
- Signals: self-chaining scores re-analyze unchanged modules wastefully; concert campaigns process overlapping inputs redundantly; CEGAR Loops re-examine stable abstraction regions unnecessarily
- Key compositions: CEGAR Loop, Cathedral Construction, Fixed-Point Iteration

**Rehearsal Spotlight**
- Problem: Iteration is expensive; reworking entire outputs wastes resources when only parts need refinement.
- Signals: iteration cycles are expensive; only specific sections need rework; most output is good but a few parts are weak
- Key compositions: Echelon Repair, Soil Maturity Index, CEGAR Loop

**Soil Maturity Index**
- Problem: Iterative processes lack domain-specific termination conditions beyond structural equality.
- Signals: iterative improvement plateaus on structural metrics but output lacks qualitative maturity; need to distinguish real convergence from mere structural stability; process converges structurally but hasn't achieved expected coherence or readiness
- Key compositions: Fixed-Point Iteration, Back-Slopping, Delphi Convergence

---

## Composition Clusters

Patterns that frequently compose together for specific purposes:

**Quality Assurance Pipeline:** Shipyard Sequence, Succession Pipeline, Triage Gate
- Sequential quality gates where foundation validation precedes expensive fan-out, classification routes outputs, and stage progression depends on quality thresholds.

**Cost-Optimized Processing:** Echelon Repair, Fermentation Relay, Screening Cascade
- Instrument tier matching where cheap instruments classify/screen, mid-tier refines, and expensive instruments handle only complex cases.

**Intent-Driven Coordination:** Mission Command, Commander's Intent Envelope, After-Action Review
- Decentralized execution around shared intent with outcome-focused validation and systematic learning capture.

**Parallel Work Integration:** Fan-out + Synthesis, Barn Raising, Prefabrication
- Consistent parallel execution where shared conventions prevent drift, interface contracts enable composition, and synthesis integrates diverse outputs.

**Failure Intelligence:** CDCL Search, Back-Slopping, After-Action Review
- Learning from failure where constraints extracted from failures prevent recurrence and lessons propagate across iterations.

**Adaptive Recovery:** Andon Cord, Circuit Breaker, Dead Letter Quarantine
- Intelligent failure response where root-cause diagnosis replaces blind retry, instrument failures trigger fallbacks, and chronic failures are quarantined.

**Multi-Frame Analysis:** Rashomon Gate, Source Triangulation, Sugya Weave
- Diverse perspective application where multiple frames reveal different insights, sources cross-validate claims, and editorial synthesis produces argued positions.

**Progressive Validation:** Canary Probe, Progressive Rollout, Speculative Hedge
- Incremental commitment where small-scale validation precedes full deployment, scaling progression is gated on evidence, and parallel approaches hedge uncertainty.

**Iterative Refinement:** Cathedral Construction, Fixed-Point Iteration, Memoization Cache
- Bounded iteration toward targets where convergence detection prevents waste, incremental construction scales to large artifacts, and caching avoids redundant work.

**Context Management:** Relay Zone, Forward Observer, Screening Cascade
- Information compression where cheap instruments pre-filter or summarize before expensive processing, preventing context overflow and reducing token costs.

---

## Selection Guidance

When choosing patterns:

1. **Start with the problem, not the pattern name.** Read the signals — do they match your situation?
2. **Check the scale.** Within-stage patterns structure prompts; score-level patterns arrange sheets; concert-level patterns coordinate scores.
3. **Look at composition clusters.** Patterns designed to work together are more powerful than isolated patterns.
4. **Read the full pattern file.** This index tells you WHEN; the pattern file tells you HOW.
