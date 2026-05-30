"""Helpers for loading packaged JSON resources."""

from __future__ import annotations

import json
from typing import Any

from thesisev.paths import config_dir

CONFIG_DIR = config_dir()


def load_json_resource(filename: str) -> Any:
    """Load a JSON resource from the package config directory."""

    resource_path = CONFIG_DIR / filename
    with resource_path.open(encoding="utf-8") as file:
        return json.load(file)
