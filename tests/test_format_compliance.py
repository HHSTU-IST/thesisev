"""Regression tests for global paragraph compliance rate (方案 A3)."""

from __future__ import annotations

from typing import Any

from tests.conftest import build_docx_paragraph
from thesisev.models import ThesisDocument
from thesisev.resources import load_json_resource
from thesisev.scoring_format import (
    evaluate_docx_expected_rule,
    extract_format_rules,
    normalize_format_spec_payload,
)


def _load_body_rule() -> dict[str, Any]:
    """Return the body_font rule from the built-in format spec."""
    spec = normalize_format_spec_payload(load_json_resource("score_thesis_tech_f.json"))
    rules = extract_format_rules(spec)
    return next(rule for rule in rules if rule["id"] == "body_font")


def _doc_with(
    paragraphs: list[dict[str, Any]], *, total_words: int = 4000
) -> ThesisDocument:
    """Wrap docx snapshot paragraphs into a ThesisDocument."""
    return ThesisDocument(
        title="t",
        source_path="t.docx",
        source_type="docx",
        raw_text="",
        cleaned_text="x" * total_words,
        front_matter="",
        abstract="",
        sections=[],
        root_sections=[],
        paragraphs=[],
        sentences=[],
        total_word_count=total_words,
        format_snapshot={
            "paragraphs": paragraphs,
            "tables": [],
            "sections": [],
            "word_count": total_words,
            "section_count": 5,
        },
    )


def test_fully_compliant_paragraphs_pass():
    rule = _load_body_rule()
    paragraphs = [
        build_docx_paragraph(f"正文段落{i} 实验数据分析内容") for i in range(10)
    ]
    verdict, evidence = evaluate_docx_expected_rule(_doc_with(paragraphs), rule)
    assert verdict is True
    assert "全部合规" in evidence


def test_two_violations_out_of_twelve_fail():
    rule = _load_body_rule()
    paragraphs = [
        build_docx_paragraph(f"正文段落{i} 实验数据分析内容") for i in range(10)
    ]
    paragraphs.append(build_docx_paragraph("黑体违规段", font="黑体"))
    paragraphs.append(build_docx_paragraph("字号违规段", size=10.5))
    verdict, evidence = evaluate_docx_expected_rule(_doc_with(paragraphs), rule)
    assert verdict is False
    assert "合规" in evidence
    assert "反例" in evidence


def test_single_violation_in_twelve_fails_under_95_percent():
    """10/12 compliant is 83% < 95% threshold so the rule must fail."""
    rule = _load_body_rule()
    paragraphs = [
        build_docx_paragraph(f"正文段落{i} 实验数据分析内容") for i in range(9)
    ]
    paragraphs.append(build_docx_paragraph("黑体违规段", font="黑体"))
    paragraphs.append(build_docx_paragraph("普通段一"))
    paragraphs.append(build_docx_paragraph("普通段二"))
    verdict, _ = evaluate_docx_expected_rule(_doc_with(paragraphs), rule)
    assert verdict is False


def test_single_violation_in_twenty_passes_at_exactly_95_percent():
    rule = _load_body_rule()
    paragraphs = [
        build_docx_paragraph(f"正文段落{i} 实验数据分析内容") for i in range(20)
    ]
    paragraphs[3] = build_docx_paragraph("第二十段中唯一黑体", font="黑体")
    verdict, _ = evaluate_docx_expected_rule(_doc_with(paragraphs), rule)
    assert verdict is True


def test_rule_reporting_preserves_expected_spec_text():
    """The scoring layer appends the full expectation for human review."""
    from thesisev.scoring_format import score_format_rules

    rule = _load_body_rule()
    paragraphs = [
        build_docx_paragraph(f"正文段落{i} 实验数据分析内容") for i in range(12)
    ]
    summary = score_format_rules(
        document=_doc_with(paragraphs),
        format_issues=[],
        rules=[rule],
        format_requirements=None,
    )
    joined_evidence = "；".join(summary["evidence"])
    # Every scored rule must state what it expects, even when compliant.
    assert "宋体" in joined_evidence


def test_unusable_expected_items_return_none_not_silent_pass():
    """P1 regression: empty / non-dict expected must degrade to manual review.

    Previously an empty expected list fell through and was treated as satisfied,
    silently awarding the points. The guard now yields a ``None`` verdict.
    """
    rule = {
        "id": "orphan",
        "label": "孤立规则",
        "check": {"expected": ["not-a-dict", 42]},
    }
    verdict, evidence = evaluate_docx_expected_rule(_doc_with([]), rule)
    assert verdict is None
    assert "需人工核对" in evidence


def test_non_list_expected_returns_false():
    """P1 regression: a malformed (non-list) expected payload is a hard failure."""
    rule = {"id": "broken", "label": "坏规则", "check": {"expected": "oops"}}
    verdict, evidence = evaluate_docx_expected_rule(_doc_with([]), rule)
    assert verdict is False
    assert "格式无效" in evidence
