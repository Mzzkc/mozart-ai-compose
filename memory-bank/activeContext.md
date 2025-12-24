# Mozart AI Compose - Active Context

**Last Updated:** 2025-12-24
**Current Phase:** Phase 1 (Learning Foundation) - IN PROGRESS
**Status:** AGI Evolution Plan approved, implementation started

---

## CRITICAL: Resume From Here

**Plan File:** `/home/emzi/.claude/plans/dapper-dancing-noodle.md`

**Implementation Progress:**
- [x] Phase 1.1: Extended BatchState with learning metadata (DONE)
- [ ] Phase 1.2: Create learning module with OutcomeStore
- [ ] Phase 1.3: Add confidence scoring to ValidationResult
- [ ] Phase 1.4: Integrate outcome recording in runner
- [ ] Phase 2: Confidence-based execution
- [ ] Phase 3: HTTP bridge to Recursive Light
- [ ] Phase 4: Judgment integration

**File Modified This Session:**
- `/home/emzi/Projects/mozart-ai-compose/src/mozart/core/checkpoint.py` - Added learning fields to BatchState

---

## Session Summary (2025-12-24)

### Issues Fixed
1. **Batch 9 retry loop** - Changed `file_modified` validation to `content_contains` in Naurva config
2. **Skill file bloat** - Created compressed skills (71% smaller):
   - `commit-review-coordination.compressed.md` (886→147 lines)
   - `multi-agent-coordination.compressed.md` (214→86 lines)
3. **WSL crash investigation** - Identified as Arrow Lake firmware issue (Intel Core Ultra 9 275HX)

### Major Planning Completed
**Vision:** Mozart + Recursive Light = AGI architecture
- Mozart = execution layer (hands)
- Recursive Light = judgment layer (mind)
- Together: accumulated wisdom, genuine agency

**TDF Analysis:** Full 4-domain + 6 pairings + 4 triplets analysis on "replace software engineer" goal
- Conclusion: Leverage multiplier (10x individual) not literal replacement
- Gap: judgment, taste, novel problem solving (Recursive Light fills this)

### AGI Evolution Plan (MVP: Phases 1-4)

**Phase 1: Learning Foundation** (Mozart-only)
- Extend BatchState with outcome_data, confidence_score, learned_patterns
- Create OutcomeStore for recording/querying outcomes
- Add confidence scoring to ValidationResult

**Phase 2: Confidence-Based Execution** (Mozart-only)
- Adaptive retry strategy based on confidence
- Escalation protocol for low-confidence situations

**Phase 3: Language Bridge** (HTTP API)
- RecursiveLightBackend in Mozart
- Mozart endpoints in Recursive Light (/api/mozart/process)

**Phase 4: Judgment Integration**
- JudgmentQuery/JudgmentResponse protocol
- RL provides judgment on proceed/retry/escalate

---

## Files Changed This Session

### Naurva Config
- `/home/emzi/Projects/Naurva/mozart-batch-review.yaml`
  - Changed `file_modified` → `content_contains` for tracking files
  - Updated skill references to compressed versions
  - Reduced timeout 45min → 30min

### Skills (New Compressed Versions)
- `/home/emzi/.claude/skills/commit-review-coordination.compressed.md`
- `/home/emzi/.claude/skills/multi-agent-coordination.compressed.md`

### Mozart Core
- `/home/emzi/Projects/mozart-ai-compose/src/mozart/core/checkpoint.py`
  - Added to BatchState: outcome_data, confidence_score, learned_patterns, similar_outcomes_count, first_attempt_success, outcome_category

### Naurva State
- `/home/emzi/Projects/Naurva/coordination-workspace/naurva-commit-review.json`
  - Reset batch 9 from stuck → completed
  - Next batch: 10

---

## Next Session: Continue Phase 1

### Immediate Tasks
1. Create `/home/emzi/Projects/mozart-ai-compose/src/mozart/learning/__init__.py`
2. Create `/home/emzi/Projects/mozart-ai-compose/src/mozart/learning/outcomes.py` with:
   - BatchOutcome dataclass
   - OutcomeStore protocol
   - JsonOutcomeStore implementation
3. Extend ValidationResult with confidence scoring
4. Integrate outcome recording in JobRunner

### Key Files to Read
1. This file (activeContext.md)
2. Plan: `/home/emzi/.claude/plans/dapper-dancing-noodle.md`
3. Checkpoint: `/home/emzi/Projects/mozart-ai-compose/src/mozart/core/checkpoint.py` (modified)
4. Validation: `/home/emzi/Projects/mozart-ai-compose/src/mozart/execution/validation.py`
5. Runner: `/home/emzi/Projects/mozart-ai-compose/src/mozart/execution/runner.py`

---

## Integration Context

### Recursive Light (for Phase 3-4)
- Location: `/home/emzi/Projects/recursive-light/`
- VifApi: Main entry point with first_pass(), second_pass()
- Dual-LLM: LLM #1 (unconscious) for confidence, LLM #2 (conscious) for response
- CAM: Collective Associative Memory for cross-session wisdom
- Key types: Llm1Output, BoundaryState, QualityConditions, Insight, Hyperedge

### Bridge Decision
- HTTP API (not PyO3 FFI) for language boundary
- Mozart calls RL via httpx
- RL exposes /api/mozart/process and /api/mozart/judgment endpoints

---

## Technical Notes

### WSL Stability (Arrow Lake)
- Intel Core Ultra 9 275HX has firmware bugs affecting WSL2
- Exit code 0x00000001 during long I/O operations
- Mitigations: BIOS update, reduced timeout, WSL kernel update

### Validation Semantic Fix
- `file_modified` checks MECHANISM (mtime changed)
- `content_contains` checks INTENT (batch content exists)
- For cumulative tracking files, use content_contains with batch markers

---

*Session ended due to context limits. Plan is saved and ready for continuation.*
