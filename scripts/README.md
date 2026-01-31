# Project Management Scripts

This directory contains scripts that support project versioning and deployment workflows.

## Local Development Install

### `install_local.sh`

Installs the current repository locally for development/prototyping. This is useful when working on a fork of mistral-vibe.

**Features:**
- Detects and uninstalls any existing `mistral-vibe` installation (prevents conflicts)
- Installs from the local repo in editable mode (changes to code are reflected immediately)
- Verifies the installation was successful

**Usage:**

```bash
# Install/reinstall from local repo
./scripts/install_local.sh
```

**After installation:**
- `vibe` - Start the CLI agent
- `vibe-acp` - Start the ACP server

**Uninstall:**
```bash
uv tool uninstall mistral-vibe
```

## Versioning

### Usage

```bash
# Bump major version (1.0.0 -> 2.0.0)
uv run scripts/bump_version.py major

# Bump minor version (1.0.0 -> 1.1.0)
uv run scripts/bump_version.py minor

# Bump patch/micro version (1.0.0 -> 1.0.1)
uv run scripts/bump_version.py micro
# or
uv run scripts/bump_version.py patch
```
