"""Regression tests for scoring pipeline dispatch (方案 A2 / A1 self-check)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import build_docx_paragraph, build_thesis_document
from thesisev.models import ThesisDocument
from thesisev.rubric_utils import normalize_rubric_items
from thesisev.scoring import (
    calculate_score_report,
    score_item_locally,
    validate_rubric_local_scorability,
)


def _docx_document(
    paragraphs: list[dict[str, Any]], *, total_words: int = 3800
) -> ThesisDocument:
    return ThesisDocument(
        title="基于边缘计算的物联网系统",
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
            "sections": [{"page_width": 11906}],
            "word_count": total_words,
            "section_count": 6,
        },
    )


def _heading(text: str, style: str, size: float) -> dict[str, Any]:
    return build_docx_paragraph(text, style=style, font="黑体", size=size, indent=None)


def _compliant_docx() -> ThesisDocument:
    paragraphs = [
        _heading("1 系统总体设计", "Heading 1", 16.0),
        _heading("1.1 需求分析", "Heading 2", 14.0),
    ]
    paragraphs += [
        build_docx_paragraph(f"正文第{i}段 边缘计算数据处理分析内容") for i in range(12)
    ]
    return _docx_document(paragraphs)


def test_thesis_rubric_includes_format_item():
    """resolve_rubric_items appends the built-in format item for thesis presets."""
    report = calculate_score_report(
        document=_compliant_docx(),
        topic_analysis={"document_ratio": 0.6, "topic_keywords": ["边缘计算"]},
        format_issues=[],
        writing_issues=[],
        content_context={},
        keywords=["边缘计算"],
        technology_details=[],
        model_config=None,
    )
    keys = [criterion.key for criterion in report.criteria]
    assert "format" in keys
    format_criterion = next(
        criterion for criterion in report.criteria if criterion.key == "format"
    )
    assert format_criterion.evaluation == "local_program"
    assert format_criterion.max_score == 15.0


def test_format_violations_lower_thesis_total_score():
    """A docx with font violations must score lower on format than a clean one."""
    clean = calculate_score_report(
        document=_compliant_docx(),
        topic_analysis={"document_ratio": 0.6, "topic_keywords": ["边缘计算"]},
        format_issues=[],
        writing_issues=[],
        content_context={},
        keywords=["边缘计算"],
        technology_details=[],
        model_config=None,
    )
    bad_doc = _docx_document(
        [
            _heading("1 系统总体设计", "Heading 1", 16.0),
            _heading("1.1 需求分析", "Heading 2", 14.0),
            *[
                build_docx_paragraph(f"正文第{i}段 边缘计算数据处理分析内容")
                for i in range(9)
            ],
            build_docx_paragraph("黑体违规", font="黑体"),
            build_docx_paragraph("黑体违规二", font="黑体"),
            build_docx_paragraph("字号违规", size=10.5),
        ]
    )
    dirty = calculate_score_report(
        document=bad_doc,
        topic_analysis={"document_ratio": 0.6, "topic_keywords": ["边缘计算"]},
        format_issues=[],
        writing_issues=[],
        content_context={},
        keywords=["边缘计算"],
        technology_details=[],
        model_config=None,
    )
    clean_format = next(c for c in clean.criteria if c.key == "format")
    dirty_format = next(c for c in dirty.criteria if c.key == "format")
    assert clean_format.score > dirty_format.score
    assert dirty_format.deductions  # visible reasons must exist


def test_unknown_local_key_raises_instead_of_silent_zero():
    """validate_rubric_local_scorability must reject unmapped local items."""
    items = normalize_rubric_items(
        [
            {
                "criterion": "未知项",
                "key": "bogus_key",
                "standard": [],
                "score": 5,
                "evaluation": "local",
            }
        ]
    )
    with pytest.raises(ValueError, match="no local scorer"):
        validate_rubric_local_scorability(
            items, rubric_filename="score_thesis_tech.json"
        )


def test_score_item_locally_dispatch_by_key():
    """score_item_locally routes each thesis key to its scorer."""
    document = build_thesis_document()
    kwargs: dict[str, Any] = {
        "document": document,
        "topic_analysis": {"document_ratio": 0.6, "topic_keywords": ["边缘计算"]},
        "format_issues": [],
        "writing_issues": [],
        "keywords": ["边缘计算", "目标检测"],
        "technology_details": [],
        "format_requirements": None,
        "rubric_filename": "score_thesis_tech.json",
    }
    from thesisev.rubric_utils import RubricItem

    expectations = {
        "topic_workload": "topic_workload",
        "translation": "translation",
    }
    for key, expected in expectations.items():
        item = RubricItem(
            name="测试项", standards=[], evaluation="local", max_score=10.0, key=key
        )
        criterion = score_item_locally(rubric_item=item, **kwargs)
        assert criterion.key == expected


def test_determine_score_source_treats_local_program_as_local():
    """P0 regression: fully-local runs (incl. format engine) must report local."""

    from thesisev.rubric_utils import ScoreCriterion
    from thesisev.scoring import determine_score_source

    def crit(evaluation: str) -> ScoreCriterion:
        return ScoreCriterion(
            key="k", name="n", score=1.0, max_score=2.0, evaluation=evaluation
        )

    # The bug case: a rubric item + the deterministic format engine, no LLM.
    assert determine_score_source([crit("local"), crit("local_program")]) == "local"
    assert determine_score_source([crit("llm_fallback_local")]) == "local"
    # All remote stays remote; the empty default is local.
    assert determine_score_source([crit("llm"), crit("llm")]) == "llm"
    assert determine_score_source([crit("")]) == "local"
    # Genuine mixing of a live LLM score with a local one is still mixed.
    assert determine_score_source([crit("llm"), crit("local")]) == "mixed"
