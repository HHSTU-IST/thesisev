#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

# `tests/` is gitignored (see .gitignore), so a fresh clone carries no suite.
# Fall back to linting the package alone: ruff treats a missing path as a hard
# E902 / IO failure, so the directory is probed instead of assumed.
targets=(thesisev)
if [[ -d tests ]]; then
  targets+=(tests)
else
  echo "note: tests/ is absent in this checkout (gitignored); skipping it"
fi

uv run ruff check "${targets[@]}"
uv run ruff format --check "${targets[@]}"
uv run ty check

if [[ -d tests ]]; then
  uv run pytest tests -q
else
  echo "note: pytest skipped, no tests/ directory to run"
fi
