#!/usr/bin/env bash

# Local Development Install Script for Mistral Vibe Fork
# This script installs the current repo locally for prototyping/testing
# It handles reinstallation by uninstalling any existing mistral-vibe first

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

function info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

function success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

function warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function check_uv() {
    if ! command -v uv &> /dev/null; then
        error "uv is not installed. Please install uv first:"
        error "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    info "uv version: $(uv --version)"
}

function uninstall_existing() {
    info "Checking for existing mistral-vibe installations..."
    
    # Check if installed via uv tool
    if uv tool list 2>/dev/null | grep -q "^mistral-vibe"; then
        warning "Found existing mistral-vibe installation via uv tool"
        info "Uninstalling existing mistral-vibe..."
        uv tool uninstall mistral-vibe
        success "Uninstalled existing mistral-vibe"
    else
        info "No existing mistral-vibe installation found via uv tool"
    fi
    
    # Also check if vibe command exists but might be installed differently
    if command -v vibe &> /dev/null; then
        warning "'vibe' command still exists in PATH at: $(command -v vibe)"
        warning "You may have another installation (pip, pipx, etc.)"
        warning "This script will proceed, but you may need to manually remove the other installation"
    fi
}

function install_local() {
    info "Installing mistral-vibe from local repository..."
    info "Repository root: $REPO_ROOT"
    
    cd "$REPO_ROOT"
    
    # Install in editable mode for development/prototyping
    # This allows changes to the code to be reflected immediately
    uv tool install --editable "$REPO_ROOT"
    
    success "Local installation completed!"
}

function verify_installation() {
    info "Verifying installation..."
    
    if ! command -v vibe &> /dev/null; then
        error "'vibe' command not found after installation"
        error "Please check your PATH and uv tool installation"
        exit 1
    fi
    
    if ! command -v vibe-acp &> /dev/null; then
        error "'vibe-acp' command not found after installation"
        exit 1
    fi
    
    # Show which vibe is being used
    local vibe_path
    vibe_path=$(command -v vibe)
    success "vibe installed at: $vibe_path"
    
    # Show version (this might fail if there are dependency issues, so don't fail the script)
    if vibe --version 2>/dev/null; then
        success "Version check passed"
    else
        warning "Could not get version info, but installation appears successful"
    fi
}

function main() {
    echo
    echo "██████████████████░░"
    echo "██████████████████░░"
    echo "████  ██████  ████░░"
    echo "████    ██    ████░░"
    echo "████          ████░░"
    echo "████  ██  ██  ████░░"
    echo "██      ██      ██░░"
    echo "██████████████████░░"
    echo "██████████████████░░"
    echo
    echo "Local Development Install for Mistral Vibe Fork"
    echo "Repository: $REPO_ROOT"
    echo
    
    check_uv
    uninstall_existing
    install_local
    verify_installation
    
    echo
    success "Installation completed successfully!"
    echo
    echo "This is a LOCAL development install from:"
    echo "  $REPO_ROOT"
    echo
    echo "Commands available:"
    echo "  vibe      - Start the CLI agent"
    echo "  vibe-acp  - Start the ACP server"
    echo
    echo "To uninstall:"
    echo "  uv tool uninstall mistral-vibe"
    echo
    echo "To reinstall after code changes:"
    echo "  scripts/install_local.sh"
    echo
}

main
