#!/usr/bin/env bash
# install-mozartd.sh — Install Mozart daemon (mozartd) as a systemd service.
#
# Usage:
#   ./scripts/install-mozartd.sh            # user service (no root needed)
#   sudo ./scripts/install-mozartd.sh --system  # system-wide service
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INSTALL_MODE="user"

# ─── Argument parsing ──────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --system)
            INSTALL_MODE="system"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--system]"
            echo ""
            echo "  --system   Install as system service (requires root)"
            echo "  (default)  Install as user service (~/.config/systemd/user/)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ─── Helpers ───────────────────────────────────────────────────────────

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# ─── Pre-flight checks ────────────────────────────────────────────────

if [[ "$INSTALL_MODE" == "system" ]] && [[ $EUID -ne 0 ]]; then
    error "System service installation requires root. Use sudo or --help."
fi

if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install Python 3.11+ first."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 11 ]]; then
    error "Python 3.11+ required (found $PYTHON_VERSION)"
fi

# ─── Step 1: Install Python package with daemon extras ─────────────────

info "Installing mozart-ai-compose with daemon extras..."
pip install -e "${PROJECT_ROOT}[daemon]" || {
    warn "pip install failed; trying with --user flag..."
    pip install --user -e "${PROJECT_ROOT}[daemon]"
}

# Verify mozartd is on PATH
if ! command -v mozartd &>/dev/null; then
    warn "mozartd not found on PATH after install."
    warn "You may need to add ~/.local/bin to your PATH."
fi

# ─── Step 2: Create config directory ──────────────────────────────────

MOZART_DIR="$HOME/.mozart"
info "Creating config directory: $MOZART_DIR"
mkdir -p "$MOZART_DIR"

# ─── Step 3: Write default config ────────────────────────────────────

CONFIG_FILE="$MOZART_DIR/mozartd.yaml"
if [[ -f "$CONFIG_FILE" ]]; then
    info "Config already exists: $CONFIG_FILE (skipping)"
else
    info "Writing default config: $CONFIG_FILE"
    cat > "$CONFIG_FILE" << 'YAML'
# Mozart Daemon Configuration
# See: mozartd start --help

# Unix socket for CLI <-> daemon communication
socket:
  path: /tmp/mozartd.sock
  permissions: 0o660

# PID file location
pid_file: /tmp/mozartd.pid

# Concurrency
max_concurrent_jobs: 5
max_concurrent_sheets: 10

# Resource limits
resource_limits:
  max_memory_mb: 8192
  max_processes: 50
  max_api_calls_per_minute: 60

# State persistence
state_backend_type: sqlite
state_db_path: ~/.mozart/daemon-state.db

# Logging
log_level: info
log_file: ~/.mozart/mozartd.log

# Graceful shutdown timeout (seconds)
shutdown_timeout_seconds: 300

# Resource monitor check interval (seconds)
monitor_interval_seconds: 15
YAML
fi

# ─── Step 4: Install systemd service ─────────────────────────────────

MOZARTD_PATH=$(command -v mozartd 2>/dev/null || echo "/usr/local/bin/mozartd")

if [[ "$INSTALL_MODE" == "system" ]]; then
    # System service
    SERVICE_DIR="/etc/systemd/system"
    info "Installing system service to $SERVICE_DIR/mozartd@.service"
    cp "$SCRIPT_DIR/mozartd.service" "$SERVICE_DIR/mozartd@.service"

    # Update ExecStart with actual path
    sed -i "s|/usr/local/bin/mozartd|${MOZARTD_PATH}|g" "$SERVICE_DIR/mozartd@.service"

    systemctl daemon-reload
    info "System service installed. Enable with:"
    info "  sudo systemctl enable mozartd@\$USER"
    info "  sudo systemctl start mozartd@\$USER"
else
    # User service
    USER_SERVICE_DIR="$HOME/.config/systemd/user"
    mkdir -p "$USER_SERVICE_DIR"
    info "Installing user service to $USER_SERVICE_DIR/mozartd.service"

    # Adapt the service file for user mode
    sed \
        -e "s|/usr/local/bin/mozartd|${MOZARTD_PATH}|g" \
        -e '/^User=%i$/d' \
        -e '/^Group=%i$/d' \
        -e 's|^ProtectHome=read-only$|# ProtectHome not applicable for user services|' \
        -e "s|^ExecStart=.*|ExecStart=${MOZARTD_PATH} start --foreground --config ${CONFIG_FILE}|" \
        "$SCRIPT_DIR/mozartd.service" > "$USER_SERVICE_DIR/mozartd.service"

    systemctl --user daemon-reload
    info "User service installed."
fi

# ─── Step 5: Enable the service ──────────────────────────────────────

if [[ "$INSTALL_MODE" == "user" ]]; then
    info "Enabling user service..."
    systemctl --user enable mozartd.service

    # Enable linger so the daemon survives logout
    if command -v loginctl &>/dev/null; then
        loginctl enable-linger "$USER" 2>/dev/null || true
    fi

    info ""
    info "Installation complete! Commands:"
    info "  systemctl --user start mozartd    # Start daemon"
    info "  systemctl --user status mozartd   # Check status"
    info "  systemctl --user stop mozartd     # Stop daemon"
    info "  journalctl --user -u mozartd -f   # View logs"
    info ""
    info "Config: $CONFIG_FILE"
fi
