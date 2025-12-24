# Mozart AI Compose - Project Brief

**Purpose:** Generic orchestration tool for running multiple Claude AI sessions with configurable prompts, state management, and observability.

**Origin:** Generalization of `run-batch-review.sh` from Naurva project into reusable Python framework.

---

## Core Problem

Running Claude for batch operations (code review, data processing, content generation) requires:
- Resumable execution (crash recovery, rate limit handling)
- Progress tracking and validation
- Configurable prompts with templating
- Observability (what's running, what failed, history)

Manual bash scripts work but don't scale. Need declarative YAML configs + web dashboard.

---

## Goals

1. **Declarative Jobs** - YAML config files define batch jobs
2. **Resumable** - Checkpoint-based state; restart picks up where left off
3. **Multi-Backend** - Claude CLI (subprocess) or Anthropic API (direct)
4. **Validation** - Verify expected outputs created after each batch
5. **Smart Retry** - Error classification (rate limit vs transient vs fatal)
6. **Dashboard** - Web UI for monitoring, history, retry controls

---

## Non-Goals (Out of Scope)

- Multi-model orchestration (OpenAI, Gemini, etc.) - Claude-focused
- Complex DAG workflows - Simple sequential batches only
- Distributed execution - Single-machine tool
- Real-time streaming UI - Batch-oriented, not chat

---

## Architecture

```
mozart/
├── core/           # Config, checkpoint, error classification
├── backends/       # Claude CLI + Anthropic API
├── execution/      # Runner loop, validation, retry
├── state/          # JSON + SQLite backends
├── prompts/        # Jinja2 templating
├── notifications/  # Desktop, Slack, webhook
└── dashboard/      # FastAPI + web UI
```

---

## Key Patterns (From Naurva Scripts)

**State Management:**
- Checkpoint file tracks last_completed_batch, status, timestamps
- Fallback: infer state from artifact files if checkpoint missing
- Atomic saves via temp file + rename

**Error Handling:**
- Pattern matching for rate limits (429, quota, capacity)
- Separate retry budget vs rate limit waits
- Error classification → appropriate retry behavior

**Validation:**
- File existence checks for required outputs
- Modification time (mtime) checks for tracking files
- All validations must pass before batch marked complete

**Prompt Templating:**
- Jinja2 with batch_num, total_batches, start_item, end_item
- Stakes injection for motivation
- Thinking method injection for cognitive approach

---

## Success Criteria

1. Can run Naurva batch review via YAML config (parity with bash script)
2. Resume works seamlessly after crash/kill
3. Dashboard shows job status, progress, history
4. Validate command catches config errors before run
5. Notifications work (at minimum desktop)

---

## Technology Choices

- **Python 3.11+** - Modern features, type hints
- **Pydantic v2** - Config validation, JSON schema
- **Typer** - CLI with Rich output
- **FastAPI** - Dashboard backend
- **SQLite** - Job history storage
- **Jinja2** - Prompt templating

---

*Created: 2025-12-18*
*Last Updated: 2025-12-18*
