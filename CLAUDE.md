# Mozart AI Compose - Claude Instructions

Project-specific instructions for AI assistants working on this codebase.

---

## Session Protocol

**Session Start:**
1. Read `STATUS.md` (root) - Quick status overview
2. Read `memory-bank/activeContext.md` - Current state and focus
3. Read `memory-bank/projectbrief.md` if unfamiliar with project purpose
4. Check `memory-bank/progress.md` for what's been done

**Session End:**
1. Update `memory-bank/activeContext.md` with session results
2. Update `STATUS.md` if phase/component status changed
3. Add entry to `memory-bank/progress.md` if significant work
4. Create session summary in `memory-bank/sessions/` if >1hr complex work

---

## Code Patterns

**Async Throughout:** All I/O operations use async. CLI backend uses `asyncio.create_subprocess_exec`.

**Pydantic v2:** All config and state models use Pydantic BaseModel with validation.

**Protocol-Based:** Backends and state storage use Protocol classes for swappability.

**Type Hints:** All functions have type hints. Use `mypy` for checking.

---

## Key Files

| Purpose | File |
|---------|------|
| CLI entry | `src/mozart/cli.py` |
| Config models | `src/mozart/core/config.py` |
| State models | `src/mozart/core/checkpoint.py` |
| Error handling | `src/mozart/core/errors.py` |
| Claude backend | `src/mozart/backends/claude_cli.py` |
| State storage | `src/mozart/state/json_backend.py` |
| Example config | `examples/batch-review.yaml` |

---

## Testing

```bash
cd ~/Projects/mozart-ai-compose
source .venv/bin/activate

# Validate config
mozart validate examples/batch-review.yaml

# Dry run
mozart run examples/batch-review.yaml --dry-run

# Type check
mypy src/

# Lint
ruff check src/
```

---

## Development

```bash
# Install editable with dev deps
pip install -e ".[dev]"

# Run tests
pytest

# Watch for changes (future)
pytest --watch
```

---

## Patterns from Naurva Scripts

This project generalizes patterns from `/home/emzi/Projects/Naurva/run-batch-review.sh`:

1. **State checkpoint** - JSON file with last_completed_batch, status
2. **Fallback state inference** - Check artifact files if checkpoint missing
3. **Rate limit detection** - Pattern matching on output
4. **Separate retry vs wait counters** - Rate limits don't consume retries
5. **Validation before marking complete** - File existence + mtime checks
6. **Atomic state saves** - Write temp file, then rename

---

## Reference Skills

When working on this project, relevant skills:
- `/home/emzi/.claude/skills/session-startup-protocol.md`
- `/home/emzi/.claude/skills/session-shutdown-protocol.md`
- `/home/emzi/.claude/skills/wolf-prevention-patterns.md`

---

*Created: 2025-12-18*
