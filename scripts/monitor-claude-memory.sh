#!/usr/bin/env bash
#
# monitor-claude-memory.sh — Track Claude Code memory usage over time.
#
# Logs RSS, VmHWM, thread count, and child process count every N seconds
# so when a crash happens, we have the trajectory leading up to it.
#
# Usage:
#   ./scripts/monitor-claude-memory.sh              # auto-detect Claude PID, log to stdout
#   ./scripts/monitor-claude-memory.sh > memory.log # log to file
#   ./scripts/monitor-claude-memory.sh 10           # custom interval (default: 30s)
#   ./scripts/monitor-claude-memory.sh 5 1234       # custom interval + explicit PID
#
# The script also monitors for orphaned MCP server processes (children of
# PID 1 or the WSL init relay) and reports their count and total RSS.

set -euo pipefail

INTERVAL=${1:-30}
EXPLICIT_PID=${2:-}

find_claude_pid() {
    # Find the Claude Code process for the current terminal
    local pid
    pid=$(ps aux | grep 'claude$' | grep -v grep | head -1 | awk '{print $2}')
    if [ -z "$pid" ]; then
        echo "ERROR: No Claude Code process found" >&2
        exit 1
    fi
    echo "$pid"
}

count_orphaned_mcp() {
    # Count node/clangd processes whose parent is PID 1 or a WSL init relay
    # These are leaked from crashed Claude sessions
    local count rss
    count=$(ps -eo pid,ppid,comm | awk '
        $2 <= 2 && ($3 == "node" || $3 == "clangd.main" || $3 ~ /npm/) {n++}
        END {print n+0}
    ')
    rss=$(ps -eo ppid,rss,comm | awk '
        $1 <= 2 && ($3 == "node" || $3 == "clangd.main" || $3 ~ /npm/) {s+=$2}
        END {printf "%.0f", s/1024}
    ')
    echo "${count}:${rss}"
}

PID=${EXPLICIT_PID:-$(find_claude_pid)}

if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: PID $PID is not running" >&2
    exit 1
fi

echo "# Claude Code Memory Monitor"
echo "# PID: $PID"
echo "# Interval: ${INTERVAL}s"
echo "# Started: $(date -Iseconds)"
echo "#"
echo "# Columns: timestamp, rss_mb, hwm_mb, vsize_mb, threads, children, orphan_count, orphan_rss_mb, total_system_used_mb"
echo "#"

while kill -0 "$PID" 2>/dev/null; do
    TS=$(date +%H:%M:%S)

    # Read /proc directly for accuracy
    RSS=$(awk '/VmRSS/ {printf "%.0f", $2/1024}' /proc/"$PID"/status 2>/dev/null || echo "?")
    HWM=$(awk '/VmHWM/ {printf "%.0f", $2/1024}' /proc/"$PID"/status 2>/dev/null || echo "?")
    VSZ=$(awk '/VmSize/ {printf "%.0f", $2/1024}' /proc/"$PID"/status 2>/dev/null || echo "?")
    THR=$(awk '/Threads/ {print $2}' /proc/"$PID"/status 2>/dev/null || echo "?")
    CHILDREN=$(ps --ppid "$PID" --no-headers 2>/dev/null | wc -l)
    ORPHAN_DATA=$(count_orphaned_mcp)
    ORPHAN_COUNT=${ORPHAN_DATA%%:*}
    ORPHAN_RSS=${ORPHAN_DATA##*:}
    SYS_USED=$(free -m | awk '/Mem:/ {print $3}')

    echo "${TS}  rss=${RSS}MB  hwm=${HWM}MB  vsz=${VSZ}MB  thr=${THR}  children=${CHILDREN}  orphans=${ORPHAN_COUNT}(${ORPHAN_RSS}MB)  sys=${SYS_USED}MB"

    sleep "$INTERVAL"
done

echo "# Process $PID exited at $(date -Iseconds)"
echo "# Final system memory: $(free -m | awk '/Mem:/ {printf "%dMB used / %dMB total", $3, $2}')"
