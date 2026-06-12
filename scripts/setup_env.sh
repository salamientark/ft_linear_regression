#!/usr/bin/env bash

# Check uv existence
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

# Install deps
uv sync

# Source virtual environment
source .venv/bin/activate

# Launch a new shell with virtual environment activated
exec "$SHELL"
