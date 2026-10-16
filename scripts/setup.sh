#!/usr/bin/env bash
# Create a virtualenv and install the project. Wheels exist for all dependencies
# on Linux/macOS, so no system packages are needed anymore.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.11
uv pip install -e ".[dev]"

[ -f .env ] || cp .env.example .env
echo "Setup complete. Edit .env, then: source .venv/bin/activate && emma --help"
