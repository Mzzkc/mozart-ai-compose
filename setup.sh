#!/usr/bin/env bash
#
# Mozart AI Compose - Setup Script
# Automated installation and environment setup
#
# Usage:
#   ./setup.sh           # Standard installation
#   ./setup.sh --dev     # Include development dependencies
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
    --no-venv   Skip virtual environment creation (use current environment)
    --clean     Remove existing virtual environment before setup
    --help      Show this help message

Examples:
    ./setup.sh              # Standard installation
    ./setup.sh --dev        # Development setup with test tools
    ./setup.sh --clean      # Fresh install (removes existing venv)
    ./setup.sh --no-venv    # Install to current Python environment

Prerequisites:
    - Python ${PYTHON_MIN_VERSION}+
    - Claude CLI (optional, required for claude_cli backend)

EOF
}

# Parse arguments
DEV_MODE=false
USE_VENV=true
CLEAN_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            DEV_MODE=true
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

if [[ "$DEV_MODE" == true ]]; then
    pip install --quiet -e ".[dev]"
    log_success "Installed Mozart with development dependencies"
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

# Step 6: Validate example configuration
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

echo "To run your first job:"
echo "  mozart run examples/simple-sheet.yaml --dry-run"
echo ""

if [[ "$DEV_MODE" == true ]]; then
    echo "Development tools installed:"
    echo "  pytest tests/              # Run tests"
    echo "  mypy src/                  # Type checking"
    echo "  ruff check src/            # Linting"
    echo ""
fi

log_success "Mozart AI Compose is ready to use"
