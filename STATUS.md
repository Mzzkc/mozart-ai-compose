# Mozart AI Compose - Status

**Overall:** AGI Evolution Phase 1 IN PROGRESS
**Vision:** Mozart + Recursive Light = AGI Architecture

---

## Quick Reference

| Component | Status | Notes |
|-----------|--------|-------|
| Core Config | ✅ Done | 8 Pydantic models |
| State Models | ✅ Done | CheckpointState, BatchState (extended with learning fields) |
| Error Classification | ✅ Done | Pattern-based classifier |
| JSON State Backend | ✅ Done | Atomic saves, list/load/save |
| Claude CLI Backend | ✅ Done | Async subprocess, rate limit detection |
| CLI | ✅ Done | 6 commands, Rich output |
| Validation Framework | ✅ Done | 4 validation types, mtime tracking |
| Runner Loop | ✅ Done | Partial completion recovery |
| **Learning Foundation** | 🔄 Phase 1 | BatchState extended, OutcomeStore next |
| **Confidence Execution** | ⏳ Phase 2 | Adaptive retry, escalation |
| **RL Bridge** | ⏳ Phase 3 | HTTP API to Recursive Light |
| **Judgment Integration** | ⏳ Phase 4 | RL provides judgment |

---

## AGI Evolution Roadmap

**Plan File:** `/home/emzi/.claude/plans/dapper-dancing-noodle.md`

### Phase 1: Learning Foundation 🔄 IN PROGRESS
- [x] Extend BatchState with learning metadata (outcome_data, confidence_score, etc.)
- [ ] Create learning module (`src/mozart/learning/`)
- [ ] Add OutcomeStore for recording/querying outcomes
- [ ] Add confidence scoring to ValidationResult
- [ ] Integrate outcome recording in JobRunner

### Phase 2: Confidence-Based Execution ⏳
- [ ] Adaptive retry strategy based on confidence
- [ ] Escalation protocol (EscalationHandler, EscalationContext)
- [ ] Config extension for learning settings

### Phase 3: Language Bridge (HTTP API) ⏳
- [ ] RecursiveLightBackend in Mozart
- [ ] Extended ExecutionResult with RL metadata
- [ ] Mozart endpoints in Recursive Light (/api/mozart/process)

### Phase 4: Judgment Integration ⏳
- [ ] JudgmentQuery/JudgmentResponse protocol
- [ ] JudgmentClient implementation
- [ ] Judgment-aware runner
- [ ] RL judgment endpoint (/api/mozart/judgment)

### Future (Post-MVP)
- Phase 5: Memory Integration (Mozart outcomes → RL CAM)
- Phase 6: Adaptive Prompting (wisdom-enhanced prompts)
- Phase 7: Agent Emergence (RL agent uses Mozart as tool)

---

## Architecture Vision

```
┌─────────────────────────────────────────────────────────────┐
│                    RECURSIVE LIGHT (Rust)                    │
│  Memory Tiers | CAM | Dual-LLM | Judgment | Wisdom          │
│                         ↑ HTTP API ↓                         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      MOZART (Python)                         │
│  Batch Orchestration | Validation | Learning | Execution    │
└─────────────────────────────────────────────────────────────┘
```

- **Mozart** = Execution layer (hands) - reliable batch orchestration
- **Recursive Light** = Judgment layer (mind) - wisdom, confidence, learning
- **Together** = AGI architecture with accumulated wisdom and genuine agency

---

## Session 2025-12-24 Summary

### Fixes Applied
1. Batch 9 retry loop → Changed `file_modified` to `content_contains`
2. Skill bloat → Created compressed skills (71% smaller)
3. WSL crashes → Identified as Arrow Lake firmware issue

### Files Modified
- `src/mozart/core/checkpoint.py` - Added learning fields to BatchState
- `/home/emzi/Projects/Naurva/mozart-batch-review.yaml` - Fixed validation, compressed skills
- `/home/emzi/.claude/skills/*.compressed.md` - New compressed skill files

---

## Working Commands

```bash
cd ~/Projects/mozart-ai-compose
source .venv/bin/activate

mozart --help              # Show all commands
mozart validate <yaml>     # Validate config file
mozart run <yaml> --dry-run  # Show batch plan without executing
mozart run <yaml>          # Execute job
```

---

## Key Files for Next Session

1. `memory-bank/activeContext.md` - Full session context
2. `/home/emzi/.claude/plans/dapper-dancing-noodle.md` - Implementation plan
3. `src/mozart/core/checkpoint.py` - Modified (learning fields added)
4. `src/mozart/execution/validation.py` - Next to modify (confidence)
5. `src/mozart/execution/runner.py` - Integration target

---

*Last Updated: 2025-12-24*
