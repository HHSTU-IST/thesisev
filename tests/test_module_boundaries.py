"""Regression tests for framework-free module boundaries (P1).

Two invariants that the earlier refactor established and that can silently
regress when someone re-adds a web import to a shared module:

* preset resolution lives in ``thesisev.service`` (no ASGI machinery) and the
  HTTP layer just forwards to it;
* ``import thesisev.cli`` must not drag FastAPI / uvicorn into the interpreter.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import thesisev
from thesisev.service import resolve_preset_files

REPO_ROOT = Path(thesisev.__file__).resolve().parent.parent


def test_resolve_preset_files_known_preset() -> None:
    rubric, fmt = resolve_preset_files("thesis_tech")
    assert rubric == "score_thesis_tech.json"
    assert fmt == "score_thesis_tech_f.json"


def test_resolve_preset_files_unknown_preset_raises() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        resolve_preset_files("does_not_exist")


def test_preset_registry_is_single_source_of_truth() -> None:
    """The rubric<->format mapping is derived from one registry, not duplicated.

    ``PRESET_CONFIGS`` is the sole declaration; ``FORMAT_RUBRIC_BY_SCORE_RUBRIC``
    and the default rubric are computed from it, ``is_builtin_preset_rubric``
    reads the same set, and the scoring/api layers must expose the *same* mapping
    object so no transport code can drift from the registry.
    """

    from thesisev import api, scoring
    from thesisev.rubric_utils import (
        DEFAULT_THESIS_TECH_RUBRIC,
        FORMAT_RUBRIC_BY_SCORE_RUBRIC,
        PRESET_CONFIGS,
        is_builtin_preset_rubric,
    )

    expected = {
        config["rubric"]: config["format"]
        for config in PRESET_CONFIGS.values()
        if config.get("format")
    }
    assert expected == FORMAT_RUBRIC_BY_SCORE_RUBRIC
    assert PRESET_CONFIGS["thesis_tech"]["rubric"] == DEFAULT_THESIS_TECH_RUBRIC

    for config in PRESET_CONFIGS.values():
        assert is_builtin_preset_rubric(config["rubric"]) is True
    assert is_builtin_preset_rubric("custom_rubric.json") is False

    # Same object identity across layers -> a registry edit is instantly visible.
    assert scoring.FORMAT_RUBRIC_BY_SCORE_RUBRIC is FORMAT_RUBRIC_BY_SCORE_RUBRIC
    assert api.FORMAT_RUBRIC_BY_SCORE_RUBRIC is FORMAT_RUBRIC_BY_SCORE_RUBRIC


def test_importing_cli_does_not_pull_fastapi() -> None:
    """Run in a subprocess so sibling tests importing fastapi can't mask this.

    ``sys.modules`` is process-global, so an in-process check would be
    order-dependent (test_api_jobs imports FastAPI). A fresh interpreter is the
    only reliable way to assert the CLI import graph stays framework-free.
    """

    code = (
        "import sys\n"
        "import thesisev.cli\n"
        "offending = [m for m in ('fastapi', 'uvicorn', 'starlette')\n"
        "             if m in sys.modules]\n"
        "sys.exit('cli pulled in: ' + repr(offending) if offending else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
