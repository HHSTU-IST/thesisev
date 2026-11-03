"""Regression tests for rubric key decoupling (方案 A1)."""

from __future__ import annotations

from thesisev.resources import load_json_resource
from thesisev.rubric_utils import (
    FORMAT_RUBRIC_KEY,
    KNOWN_RUBRIC_KEYS,
    RubricItem,
    build_criterion,
    infer_rubric_key,
    merge_rubric_items,
    normalize_rubric_items,
    normalize_rubric_payload,
    parse_rubric_item,
)


def test_parse_rubric_item_carries_key():
    """parse_rubric_item must preserve a configured stable key."""
    item = parse_rubric_item(
        criterion="英文摘要", value={"key": "translation", "score": 5}
    )
    assert item["key"] == "translation"
    assert item["criterion"] == "英文摘要"


def test_normalize_rubric_items_infers_key_from_name():
    """Items without an explicit key fall back to the stable key map."""
    items = normalize_rubric_items(
        [
            {
                "criterion": "选题及工作量",
                "standard": [],
                "score": 10,
                "evaluation": "local",
            }
        ]
    )
    assert items[0].key == "topic_workload"


def test_alias_maps_to_same_key():
    """译文 and 英文摘要 must dispatch to the same translation scorer."""
    assert KNOWN_RUBRIC_KEYS["译文"] == "translation"
    assert KNOWN_RUBRIC_KEYS["英文摘要"] == "translation"
    assert infer_rubric_key("英文摘要") == "translation"


def test_normalize_rubric_items_preserves_configured_order_and_keys():
    """The six thesis items resolve to stable keys in configured order."""
    payload = normalize_rubric_payload(load_json_resource("score_thesis_tech.json"))
    items = normalize_rubric_items(payload)
    keys = [item.key for item in items]
    assert keys == [
        "topic_workload",
        "research_argument",
        "experiment_analysis",
        "writing_quality",
        "translation",
        "innovation",
    ]
    assert all(item.max_score > 0 for item in items)


def test_merge_rubric_items_overlays_by_key_not_name():
    """Uploaded rubric scores override defaults even when labels drift."""
    defaults = [
        RubricItem(
            name="译文",
            standards=[],
            evaluation="llm",
            max_score=5.0,
            key="translation",
        ),
        RubricItem(
            name="论文质量",
            standards=[],
            evaluation="llm",
            max_score=20.0,
            key="writing_quality",
        ),
    ]
    uploaded = [
        RubricItem(
            name="英文摘要",  # renamed label, same key
            standards=["翻译准确"],
            evaluation="local",
            max_score=5.0,
            key="translation",
        )
    ]
    merged = merge_rubric_items(defaults, uploaded)
    assert len(merged) == 2
    assert merged[0].key == "translation"
    assert merged[0].name == "英文摘要"  # overlay applied
    assert merged[1].name == "论文质量"  # untouched
    assert merged[1].key == "writing_quality"


def test_build_criterion_clamps_score_to_max():
    """A local scorer returning a score above max must be clamped."""
    item = RubricItem(
        name="创新", standards=[], evaluation="local", max_score=10.0, key="innovation"
    )
    criterion = build_criterion(
        key="innovation",
        rubric_item=item,
        score=15.0,
        evidence=["x"],
        deductions=[],
        suggestions=[],
    )
    assert criterion.score == 10.0
    assert criterion.deductions == []  # full score needs no fabricated deduction


def test_build_criterion_adds_visible_deduction_when_below_full():
    """Non-full scores always expose why points were lost."""
    item = RubricItem(
        name="创新", standards=[], evaluation="local", max_score=10.0, key="innovation"
    )
    criterion = build_criterion(
        key="innovation",
        rubric_item=item,
        score=6.0,
        evidence=["x"],
        deductions=[],
        suggestions=[],
    )
    assert criterion.score == 6.0
    assert any("扣 4 分" in text for text in criterion.deductions)


def test_format_rubric_key_is_stable():
    assert FORMAT_RUBRIC_KEY == "format"


def test_parse_score_value_rejects_non_finite():
    """P0 regression: NaN/Infinity must raise, not poison the score."""

    import pytest

    from thesisev.rubric_utils import parse_score_value

    assert parse_score_value(3.5) == 3.5
    assert parse_score_value(2) == 2.0
    with pytest.raises(ValueError):
        parse_score_value(float("nan"))
    with pytest.raises(ValueError):
        parse_score_value(float("inf"))
    with pytest.raises(ValueError):
        parse_score_value(float("-inf"))
    with pytest.raises(TypeError):
        parse_score_value(True)
