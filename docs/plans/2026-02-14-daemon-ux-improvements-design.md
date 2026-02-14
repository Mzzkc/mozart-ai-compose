# Daemon UX Improvements Design

Three targeted fixes to make daemon-mode Mozart usable from inside Claude Code sessions and improve job management ergonomics.

## 1. Strip CLAUDECODE env var from subprocess

**Problem:** Claude Code sets `CLAUDECODE` in its environment. When Mozart's CLI backend spawns `claude` as a subprocess, Claude Code refuses to start ("cannot be launched inside another Claude Code session"). This blocks all Mozart jobs launched from Claude Code terminals.

**Fix:** In `claude_cli.py`, strip `CLAUDECODE` from the env dict passed to `create_subprocess_exec`. Mozart's subprocess is an independent session, not a nested one.

**Files:** `src/mozart/backends/claude_cli.py` (~1 line change at line 561)

## 2. `mozart list` queries daemon for running jobs

**Problem:** `mozart list` only scans workspace directories for state files. It knows nothing about jobs the daemon is actively managing. You can submit a job and have no way to see it's running without knowing the exact job ID and workspace.

**Fix:** When daemon is available, `_list_jobs()` queries it via `DaemonClient.list_jobs()` first, then merges with file-based results. Daemon jobs that are running/queued appear even if no state file exists yet. File-based jobs fill in completed/failed history.

**Files:**
- `src/mozart/cli/commands/status.py` — update `_list_jobs()` to query daemon
- No daemon-side changes needed (RPC `job.list` already exists end-to-end)

## 3. Human-friendly job IDs

**Problem:** Job IDs are `{name}-{8-char-hex}` (e.g., `quality-continuous-34f65607`). The hex suffix is cumbersome to type and meaningless to humans.

**Fix:** Use the config file stem as the job ID directly (e.g., `quality-continuous`). If a job with that name already exists in the daemon's in-memory `_job_meta`, append an incrementing suffix: `quality-continuous-2`, `quality-continuous-3`, etc. Only active daemon jobs are checked (not historical workspace files).

**Files:** `src/mozart/daemon/manager.py` — change `submit_job()` ID generation (~5 lines)
