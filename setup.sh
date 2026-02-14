#!/usr/bin/env bash
#
# Mozart AI Compose - Setup Script
# Automated installation and environment setup
#
# Usage:
#   ./setup.sh           # Standard installation
#   ./setup.sh --dev     # Include development dependencies
#   ./setup.sh --daemon  # Include daemon support (mozartd)
#   ./setup.sh --docs    # Include documentation site tools
#   ./setup.sh --help    # Show usage information
#
set -euo pipefail

# Configuration
PYTHON_MIN_VERSION="3.11"
VENV_DIR=".venv"

# Colors (disabled if not a terminal)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Help message
show_help() {
    cat << EOF
Mozart AI Compose - Setup Script

Usage: ./setup.sh [OPTIONS]

Options:
    --dev       Include development dependencies (pytest, mypy, ruff)
    --daemon    Include daemon support (psutil for mozartd)
    --docs      Include documentation site tools (mkdocs-material)
    --no-venv   Skip virtual environment creation (use current environment)
    --clean     Remove existing virtual environment before setup
    --help      Show this help message

Examples:
    ./setup.sh              # Standard installation
    ./setup.sh --dev        # Development setup with test tools
    ./setup.sh --daemon     # Install with daemon support
    ./setup.sh --docs       # Install with documentation tools
    ./setup.sh --dev --daemon --docs  # Everything
    ./setup.sh --clean      # Fresh install (removes existing venv)
    ./setup.sh --no-venv    # Install to current Python environment

Prerequisites:
    - Python ${PYTHON_MIN_VERSION}+
    - Claude CLI (optional, required for claude_cli backend)
    - git (required for worktree isolation)

EOF
}

# Parse arguments
DEV_MODE=false
DAEMON_MODE=false
DOCS_MODE=false
USE_VENV=true
CLEAN_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --daemon)
            DAEMON_MODE=true
            shift
            ;;
        --docs)
            DOCS_MODE=true
            shift
            ;;
        --no-venv)
            USE_VENV=false
            shift
            ;;
        --clean)
            CLEAN_INSTALL=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Header
echo ""
echo "=================================="
echo "  Mozart AI Compose Setup"
echo "=================================="
echo ""

# Step 1: Check Python version
log_info "Checking Python version..."

# Find Python executable
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &> /dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        min_major=$(echo "$PYTHON_MIN_VERSION" | cut -d. -f1)
        min_minor=$(echo "$PYTHON_MIN_VERSION" | cut -d. -f2)

        if [[ "$major" -gt "$min_major" ]] || { [[ "$major" -eq "$min_major" ]] && [[ "$minor" -ge "$min_minor" ]]; }; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    log_error "Python ${PYTHON_MIN_VERSION}+ is required but not found"
    log_info "Please install Python ${PYTHON_MIN_VERSION} or later"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1)
log_success "Found $PYTHON_VERSION"

# Step 2: Virtual environment setup
if [[ "$USE_VENV" == true ]]; then
    if [[ "$CLEAN_INSTALL" == true ]] && [[ -d "$VENV_DIR" ]]; then
        log_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
        log_success "Removed $VENV_DIR"
    fi

    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating virtual environment..."
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        log_success "Created virtual environment at $VENV_DIR"
    else
        log_info "Using existing virtual environment at $VENV_DIR"
    fi

    # Activate virtual environment
    log_info "Activating virtual environment..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    log_success "Virtual environment activated"

    # Update pip
    log_info "Upgrading pip..."
    pip install --quiet --upgrade pip
    log_success "Pip upgraded"
else
    log_warn "Skipping virtual environment (--no-venv specified)"
fi

# Step 3: Install Mozart
log_info "Installing Mozart AI Compose..."

# Build extras list based on flags
EXTRAS=""
if [[ "$DEV_MODE" == true ]]; then EXTRAS="${EXTRAS:+$EXTRAS,}dev"; fi
if [[ "$DAEMON_MODE" == true ]]; then EXTRAS="${EXTRAS:+$EXTRAS,}daemon"; fi
if [[ "$DOCS_MODE" == true ]]; then EXTRAS="${EXTRAS:+$EXTRAS,}docs"; fi

if [[ -n "$EXTRAS" ]]; then
    pip install --quiet -e ".[$EXTRAS]"
    log_success "Installed Mozart with extras: $EXTRAS"
else
    pip install --quiet -e "."
    log_success "Installed Mozart"
fi

# Step 4: Verify installation
log_info "Verifying installation..."

if command -v mozart &> /dev/null; then
    MOZART_VERSION=$(mozart --version 2>&1 || echo "unknown")
    log_success "Mozart CLI available: $MOZART_VERSION"
else
    log_error "Mozart CLI not found in PATH"
    log_info "You may need to activate the virtual environment: source $VENV_DIR/bin/activate"
    exit 1
fi

# Step 5: Check Claude CLI (optional)
log_info "Checking for Claude CLI..."

if command -v claude &> /dev/null; then
    CLAUDE_VERSION=$(claude --version 2>&1 | head -1 || echo "unknown")
    log_success "Claude CLI found: $CLAUDE_VERSION"
else
    log_warn "Claude CLI not found"
    log_info "Install Claude CLI for the claude_cli backend: https://docs.anthropic.com/claude-code"
    log_info "Mozart can still use the anthropic_api backend without Claude CLI"
fi

# Step 6: Verify daemon installation (if requested)
if [[ "$DAEMON_MODE" == true ]]; then
    log_info "Verifying daemon installation..."
    if command -v mozartd &> /dev/null; then
        log_success "Mozart daemon (mozartd) available"
    else
        log_error "mozartd not found in PATH after installation"
        log_info "You may need to activate the virtual environment: source $VENV_DIR/bin/activate"
    fi
fi

# Step 7: Verify docs installation (if requested)
if [[ "$DOCS_MODE" == true ]]; then
    log_info "Verifying documentation tools..."
    if command -v mkdocs &> /dev/null; then
        log_success "mkdocs available"
        log_info "Building documentation site to verify..."
        if mkdocs build --quiet 2>/dev/null; then
            log_success "Documentation site built successfully"
        else
            log_warn "Documentation site build failed (may need mkdocs.yml configuration)"
        fi
    else
        log_error "mkdocs not found in PATH after installation"
        log_info "You may need to activate the virtual environment: source $VENV_DIR/bin/activate"
    fi
fi

# Step 8: Validate example configuration
log_info "Validating example configuration..."

if [[ -f "examples/simple-sheet.yaml" ]]; then
    if mozart validate examples/simple-sheet.yaml &> /dev/null; then
        log_success "Example configuration valid"
    else
        log_warn "Example validation failed (this may be expected without Claude CLI)"
    fi
else
    log_warn "Example configuration not found"
fi

# Summary
echo ""
echo "=================================="
echo "  Setup Complete"
echo "=================================="
echo ""

if [[ "$USE_VENV" == true ]]; then
    echo "To activate the virtual environment:"
    echo "  source $VENV_DIR/bin/activate"
    echo ""
fi

echo "To verify Mozart is working:"
echo "  mozart --version"
echo "  mozart validate examples/simple-sheet.yaml"
echo ""

if [[ "$DAEMON_MODE" == true ]]; then
    echo "To run your first job (daemon required):"
    echo "  mozartd start              # Start the daemon"
    echo "  mozartd status             # Verify daemon is running"
    echo "  mozart run examples/simple-sheet.yaml --dry-run"
    echo ""
    echo "Daemon commands:"
    echo "  mozartd stop               # Stop the daemon"
    echo ""
else
    echo "To run your first job:"
    echo "  mozart run examples/simple-sheet.yaml --dry-run"
    echo ""
    echo "Note: mozart run requires a running daemon."
    echo "  Install with --daemon flag for mozartd support."
    echo ""
fi

if [[ "$DOCS_MODE" == true ]]; then
    echo "Documentation site:"
    echo "  mkdocs serve               # Browse at http://localhost:8000"
    echo "  mkdocs build               # Build static site to site/"
    echo ""
fi

if [[ "$DEV_MODE" == true ]]; then
    echo "Development tools installed:"
    echo "  pytest tests/              # Run tests"
    echo "  mypy src/                  # Type checking"
    echo "  ruff check src/            # Linting"
    echo ""
fi

log_success "Mozart AI Compose is ready to use"
