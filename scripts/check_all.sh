#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

uv run ruff check thesisev tests
uv run ruff format --check thesisev tests
uv run ty check
uv run pytest tests -q
