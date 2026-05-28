"""Helpers for loading packaged JSON resources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).with_name("data")


def load_json_resource(filename: str) -> Any:
    """Load a JSON resource from the package data directory."""

    resource_path = DATA_DIR / filename
    with resource_path.open(encoding="utf-8") as file:
        return json.load(file)
