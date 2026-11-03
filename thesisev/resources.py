"""Helpers for loading packaged JSON resources."""

from __future__ import annotations

import copy
import json
from functools import cache
from pathlib import Path
from typing import Any

from thesisev.paths import config_dir


@cache
def _load_json_at_path(resource_path: Path) -> Any:
    """Parse and cache a JSON file keyed by its *resolved* path.

    Keying on the resolved path (not the bare filename) means a change to the
    active project root re-reads the correct file rather than serving stale
    content from the cache.
    """

    with resource_path.open(encoding="utf-8") as file:
        return json.load(file)


def load_json_resource(filename: str) -> Any:
    """Load a JSON resource from the package config directory.

    The config directory is resolved lazily (not frozen at import) so the
    lookup follows the active project root at call time rather than the
    working directory captured during import.

    The parsed payload is cached to avoid re-reading the same bundled config on
    every evaluation, but a deep copy is returned each time: some consumers
    (e.g. ``normalize_format_spec_payload`` hands a dict payload back by
    reference) may mutate it, and those edits must never leak into the cache.
    """

    return copy.deepcopy(_load_json_at_path(config_dir() / filename))
