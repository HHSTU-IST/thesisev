"""Regression tests for API helpers (rubric summaries + history locking)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from thesisev import api
from thesisev.api import (
    coerce_history_entries,
    load_builtin_rubric_summary,
    read_history_unlocked,
)
from thesisev.resources import load_json_resource
from thesisev.rubric_utils import normalize_rubric_payload
from thesisev.scoring import append_builtin_format_rubric_item, normalize_rubric_items


def test_load_json_resource_is_mutation_safe():
    """Cached payloads must hand out copies so caller edits cannot leak."""

    baseline = copy.deepcopy(load_json_resource("score_thesis_tech.json"))
    poisoned = load_json_resource("score_thesis_tech.json")
    if isinstance(poisoned, dict):
        poisoned["__injected__"] = "leak"
    elif isinstance(poisoned, list):
        poisoned.append("__injected__")

    reloaded = load_json_resource("score_thesis_tech.json")
    assert reloaded == baseline
    assert "__injected__" not in (reloaded if isinstance(reloaded, dict) else [])
    assert load_json_resource("score_thesis_tech.json") == baseline


def test_thesis_summary_total_matches_scoring_engine():
    """The UI-visible rubric summary must total the same as scoring input."""
    payload = normalize_rubric_payload(load_json_resource("score_thesis_tech.json"))
    items = append_builtin_format_rubric_item(
        normalize_rubric_items(payload), rubric_filename="score_thesis_tech.json"
    )
    engine_total = round(sum(item.max_score for item in items), 4)

    summary = load_builtin_rubric_summary("score_thesis_tech.json")
    assert summary["total_score"] == engine_total
    assert any(item["criterion"] == "格式规范" for item in summary["items"])


def test_thesis_summary_has_stable_keys():
    summary = load_builtin_rubric_summary("score_thesis_tech.json")
    keys = [item.get("key") for item in summary["items"]]
    assert "translation" in keys
    assert "format" in keys


# --------------------------------------------------------------------------
# History boundary validation (P2-C): history.json is untrusted persisted data
# --------------------------------------------------------------------------


def test_coerce_history_entries_keeps_only_dicts():
    assert coerce_history_entries("not a list") == []
    assert coerce_history_entries({"a": 1}) == []
    assert coerce_history_entries(None) == []
    assert coerce_history_entries([]) == []
    mixed = [{"id": "1"}, "junk", None, 42, {"id": "2"}]
    assert coerce_history_entries(mixed) == [{"id": "1"}, {"id": "2"}]


def test_read_history_missing_file_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(api, "HISTORY_PATH", tmp_path / "missing.json")
    assert read_history_unlocked() == []


def test_read_history_malformed_json_degrades_to_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A truncated / corrupt file must not raise on the /history endpoint."""

    history = tmp_path / "history.json"
    history.write_text("{ this is not valid json", encoding="utf-8")
    monkeypatch.setattr(api, "HISTORY_PATH", history)
    assert read_history_unlocked() == []


def test_read_history_non_list_payload_degrades_to_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """append_history.insert needs a list; a stray object must not survive."""

    history = tmp_path / "history.json"
    history.write_text('{"oops": "not a list"}', encoding="utf-8")
    monkeypatch.setattr(api, "HISTORY_PATH", history)
    assert read_history_unlocked() == []


def test_read_history_drops_malformed_entries_but_keeps_valid_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    history = tmp_path / "history.json"
    history.write_text(
        '[{"id": "a"}, "corrupt", {"id": "b"}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "HISTORY_PATH", history)
    assert read_history_unlocked() == [{"id": "a"}, {"id": "b"}]
