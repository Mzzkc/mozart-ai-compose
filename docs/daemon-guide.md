# Daemon Guide

Mozart uses a daemon process (`mozartd`) to manage job execution, similar to how Docker requires `dockerd`. The daemon centralizes resource management, maintains a persistent job registry, and provides health monitoring with backpressure.

## Why a Daemon?

Running jobs through a long-lived daemon process provides several advantages over direct CLI execution:

- **Centralized resource management.** The daemon monitors memory and child process counts, applying backpressure when the system is under load. Jobs submitted during high pressure are rejected rather than allowed to destabilize the system.
- **Persistent job registry.** A SQLite-backed registry tracks all submitted jobs across daemon restarts. Jobs left running when a daemon crashes are automatically marked as failed on the next startup.
- **Cross-job learning.** A single `GlobalLearningStore` instance is shared across all concurrent jobs. Pattern discoveries in one job are immediately available to others — no cross-process SQLite locking contention.
- **Health monitoring.** Liveness and readiness probes are available over the IPC socket, enabling external monitoring tools to check daemon health.

## Quick Start

```bash
# 1. Start the daemon
mozartd start --foreground    # Foreground (logs to console)
mozartd start                 # Background (double-fork, logs to file)

# 2. Run a job (routed through daemon)
mozart run my-score.yaml

# 3. Check daemon status
mozartd status

# 4. Stop the daemon
mozartd stop
```

## The Daemon Requirement

`mozart run` **requires a running daemon**. If no daemon is detected, the command exits with an error:

```
Error: Mozart daemon is not running.
Start it with: mozartd start
```

**Exceptions that work without a daemon:**

| Command | Daemon Required? | Why |
|---------|-----------------|-----|
| `mozart run config.yaml` | Yes | Job execution goes through daemon |
| `mozart run --dry-run config.yaml` | No | Only validates and displays the job plan |
| `mozart validate config.yaml` | No | Static config validation, no execution |

### How Auto-Detection Works

When `mozart run` is invoked:

1. The CLI calls `is_daemon_available()` from `mozart.daemon.detect`
2. This resolves the socket path (default: `/tmp/mozartd.sock`)
3. A `DaemonClient` attempts to open a Unix socket connection
4. If the connection succeeds, the job is submitted via JSON-RPC
5. If the connection fails (socket missing, refused, timeout), the CLI reports the daemon is not running

The detection is **fail-safe**: any exception during detection returns `False`, ensuring daemon bugs never break the CLI. The CLI simply falls through to the "daemon not running" error message.

## Architecture

The daemon is composed of several layers:

```
┌──────────────────────────────────────────────────┐
│                   DaemonProcess                   │
│  Lifecycle owner: boot, signal handling, cleanup  │
├──────────────────────────────────────────────────┤
│                    DaemonServer                   │
│  Unix socket + JSON-RPC 2.0 (NDJSON protocol)    │
├────────────────────┬─────────────────────────────┤
│   RequestHandler   │      HealthChecker          │
│  Method dispatch   │  Liveness + readiness       │
├────────────────────┴─────────────────────────────┤
│                    JobManager                     │
│  Task tracking, concurrency semaphore, shutdown   │
├──────────────┬───────────────┬───────────────────┤
│  JobRegistry │  JobService   │  BackpressureCtrl │
│  SQLite DB   │  Run/resume   │  Memory-based     │
│  persistence │  pause/status │  load management  │
├──────────────┴───────────────┴───────────────────┤
│              ResourceMonitor + LearningHub        │
│  Periodic checks   │  Cross-job pattern sharing   │
└──────────────────────────────────────────────────┘
```

### Process Model

- **DaemonProcess** — Owns the lifecycle. In background mode, performs a classic double-fork to detach from the terminal. Installs signal handlers for SIGTERM and SIGINT.
- **DaemonServer** — Listens on a Unix domain socket. Each client connection is handled as an asyncio task. Messages are newline-delimited JSON (NDJSON), each containing a JSON-RPC 2.0 request or response.
- **JobManager** — Tracks jobs as `asyncio.Task` instances. Uses a `Semaphore` to enforce the `max_concurrent_jobs` limit. Jobs exceeding `job_timeout_seconds` are cancelled.
- **JobService** — Decoupled execution engine (no CLI dependencies). Handles the full run/resume/pause/status lifecycle for individual jobs.
- **JobRegistry** — SQLite-backed persistent storage. Survives daemon restarts. On startup, orphaned jobs (status `queued` or `running` from a previous daemon) are marked as `failed`.

### IPC Protocol

The daemon uses **JSON-RPC 2.0** over a **Unix domain socket** with newline-delimited JSON framing:

```json
{"jsonrpc":"2.0","method":"job.submit","params":{"config_path":"/path/to/config.yaml"},"id":1}
```

Registered RPC methods:

| Method | Description |
|--------|-------------|
| `job.submit` | Submit a new job for execution |
| `job.status` | Get status of a specific job |
| `job.pause` | Pause a running job |
| `job.resume` | Resume a paused job |
| `job.cancel` | Cancel a running job |
| `job.list` | List all known jobs |
| `daemon.status` | Get daemon status summary (PID, uptime, memory, version) |
| `daemon.shutdown` | Initiate daemon shutdown |
| `daemon.health` | Liveness probe |
| `daemon.ready` | Readiness probe |

### Backpressure

The `BackpressureController` assesses system pressure based on memory usage (as a percentage of `max_memory_mb`) and child process count:

| Pressure Level | Memory Threshold | Effect |
|---------------|-----------------|--------|
| NONE | < 50% | Normal operation |
| LOW | 50–70% | 2s delay between sheet dispatches |
| MEDIUM | 70–85% | 10s delay between sheet dispatches |
| HIGH | > 85% or active rate limit | New job submissions rejected |
| CRITICAL | > 95% or monitor degraded | Emergency: lowest-priority jobs may be cancelled |

When `should_accept_job()` returns `False` (HIGH or CRITICAL pressure), the JobManager rejects new submissions with "System under high pressure — try again later."

## Configuration

The daemon is configured via a YAML file passed to `mozartd start --config <path>`. Without a config file, all defaults are used.

### DaemonConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `socket.path` | `Path` | `/tmp/mozartd.sock` | Unix domain socket path |
| `socket.permissions` | `int` | `0o660` | Socket file permissions (octal) |
| `socket.backlog` | `int` | `5` | Max pending connections |
| `pid_file` | `Path` | `/tmp/mozartd.pid` | PID file path |
| `max_concurrent_jobs` | `int` | `5` | Max simultaneous jobs (1–50) |
| `job_timeout_seconds` | `float` | `21600.0` | Max wall-clock time per job (6 hours) |
| `shutdown_timeout_seconds` | `float` | `300.0` | Max wait for graceful shutdown (5 min) |
| `monitor_interval_seconds` | `float` | `15.0` | Resource check interval |
| `max_job_history` | `int` | `1000` | Terminal jobs kept in memory before eviction |
| `log_level` | `str` | `"info"` | One of: `debug`, `info`, `warning`, `error` |
| `log_file` | `Path \| None` | `None` | Log file path (`None` = stderr only) |

### Resource Limits (nested under `resource_limits`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_memory_mb` | `int` | `8192` | Max RSS memory before backpressure triggers |
| `max_processes` | `int` | `50` | Max child processes |
| `max_api_calls_per_minute` | `int` | `60` | **Not yet enforced** — reserved for future rate limiting |

### Reserved Fields (No Effect Currently)

These fields are accepted but produce warnings if set to non-default values:

| Field | Default | Status |
|-------|---------|--------|
| `max_concurrent_sheets` | `10` | Reserved for Phase 3 scheduler |
| `state_backend_type` | `"sqlite"` | Reserved for persistent daemon state |
| `state_db_path` | `~/.mozart/daemon-state.db` | Reserved for persistent daemon state |
| `config_file` | `None` | Reserved for SIGHUP config reload |

### Example Config File

```yaml
socket:
  path: /tmp/mozartd.sock
  permissions: 0o660

pid_file: /tmp/mozartd.pid
max_concurrent_jobs: 3
job_timeout_seconds: 43200  # 12 hours for long jobs
log_level: debug
log_file: ~/.mozart/mozartd.log

resource_limits:
  max_memory_mb: 4096
  max_processes: 30
```

## Monitoring

### `mozartd status`

The primary monitoring command. Queries the daemon via IPC and displays:

```
mozartd is running (PID 12345)
  Uptime: 2h 15m 30s
  [+] Readiness: ready
  Running jobs: 2
  Memory: 1024.5 MB
  Child processes: 8
  Accepting work: True
  Version: 0.1.0
```

Readiness shows `[+] ready` when the daemon is accepting jobs, or `[-] not_ready` when under pressure, shutting down, or experiencing elevated failure rates.

### Daemon Logs

In foreground mode, logs go to stderr in console format. In background mode, logs are structured JSON. If `log_file` is configured, logs are written there:

```bash
# View daemon logs (if log_file is set)
tail -f ~/.mozart/mozartd.log
```

### Programmatic Health Probes

The `daemon.health` and `daemon.ready` IPC methods are available for integration with external monitoring. They are used internally by `mozartd status`:

- **Liveness** (`daemon.health`): Returns OK if the daemon can execute the handler — minimal cost, no resource checks.
- **Readiness** (`daemon.ready`): Returns `ready` when memory is within limits, failure rate is normal, notifications are functional, and the daemon is not shutting down.

## What's Built But NOT Yet Wired (Phase 3)

Two significant components are **fully built and tested** but **not yet integrated** into the execution path:

### GlobalSheetScheduler

Located in `daemon/scheduler.py`. Provides cross-job sheet scheduling with:

- Priority min-heap across all active jobs
- Per-job fair-share scheduling
- DAG dependency awareness (sheet ordering within a job)
- Rate-limit-aware dispatch (skips backends with active rate limits)
- Backpressure integration (delays dispatch under load)

**Current state:** Jobs run monolithically via `JobService.start_job()` — all sheets execute sequentially within a single asyncio task. When the scheduler is wired in, the manager will decompose jobs into individual sheets and use `next_sheet()` / `mark_complete()` for per-sheet dispatch.

### RateLimitCoordinator

Located in `daemon/rate_coordinator.py`. Shares rate limit state across all concurrent jobs:

- When any job hits a rate limit, all jobs using that backend back off
- In-memory coordination (no cross-process locking)
- Clamps maximum wait to 1 hour to prevent misparsed responses from blocking indefinitely

**Current state:** The write path is active — `JobManager._on_rate_limit` feeds rate limit events from job runners into the coordinator via `RunnerContext.rate_limit_callback`. The read path (`is_rate_limited()`) is consumed by the GlobalSheetScheduler, which is not yet driving execution.

## Migration from Pre-Daemon Usage

If you previously ran Mozart without a daemon:

**Before (direct execution, no longer supported for `mozart run`):**
```bash
mozart run my-score.yaml
```

**After (daemon required):**
```bash
# One-time: start the daemon
mozartd start

# Then run jobs as before
mozart run my-score.yaml

# The daemon stays running — no need to restart per job
```

### setup.sh

The `setup.sh` script includes a `--daemon` flag that installs daemon dependencies (psutil for resource monitoring):

```bash
./setup.sh --daemon          # Install with daemon support
./setup.sh --dev --daemon    # Dev + daemon
```

This installs the `daemon` extras group, which includes `psutil` for the `ResourceMonitor` and `SystemProbe` components.

## Troubleshooting

### "Mozart daemon is not running"

The daemon is not reachable at the default socket path (`/tmp/mozartd.sock`).

```bash
# Start the daemon
mozartd start

# Verify it's running
mozartd status
```

### "mozartd is already running"

A daemon instance is already active. Check with:

```bash
mozartd status
```

If the daemon is unresponsive but the PID file exists, it may have crashed:

```bash
# Stop cleans up stale PID files
mozartd stop
mozartd start
```

### Stale PID File

If `mozartd stop` reports "mozartd is not running" but a PID file exists, the file is automatically cleaned up. If it persists:

```bash
rm /tmp/mozartd.pid
mozartd start
```

### Permission Errors on Socket

The socket is created with `0o660` permissions by default (owner + group read/write). If another user needs access, adjust `socket.permissions` in the daemon config. The daemon also rejects symlinked socket paths as a security measure.

### "--escalation incompatible with daemon"

The `--escalation` flag requires interactive console prompts which are not available in daemon mode. Escalation is not currently supported when running through the daemon.

### Orphan Recovery on Restart

When the daemon starts, it checks the persistent registry for jobs that were `queued` or `running` during the previous daemon session. These orphaned jobs are automatically marked as `failed` with the error message "Daemon restarted while job was active." Check for recovered orphans in the daemon logs:

```
manager.orphans_recovered count=2
```

### Advisory Lock Conflicts

The daemon uses `fcntl.flock()` on the PID file to prevent concurrent starts. If you see "PID file locked" errors, another `mozartd start` is in progress. Wait a moment and retry.

### Force Stop

If the daemon is unresponsive to graceful shutdown:

```bash
mozartd stop --force    # Sends SIGKILL instead of SIGTERM
```

This kills the process immediately without waiting for running jobs to complete. Use as a last resort — running jobs will not checkpoint cleanly.
