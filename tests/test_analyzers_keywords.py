"""extract_keywords must reuse caller-supplied topic keywords.

The evaluation pipeline computes topic keywords once inside
``annotate_topic_relevance``; recomputing them inside ``extract_keywords`` was
a redundant, relatively expensive tiktoken pass. These tests lock in that the
optional ``topic_keywords`` argument short-circuits the recompute while the
default (standalone) path still derives them.
"""

from __future__ import annotations

import pytest

from thesisev import analyzers
from thesisev.analyzers import extract_keywords
from thesisev.models import ThesisDocument


def _minimal_document() -> ThesisDocument:
    return ThesisDocument(
        title="示例论文",
        source_path="smoke.md",
        source_type="md",
        raw_text="正文内容",
        cleaned_text="正文内容",
        front_matter="",
        abstract="",
        sections=[],
        root_sections=[],
        paragraphs=[],
        sentences=[],
        total_word_count=1000,
    )


def test_extract_keywords_skips_recompute_when_topic_keywords_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_document: ThesisDocument) -> list[str]:
        msg = "extract_topic_keywords must not run when topic_keywords is passed"
        raise AssertionError(msg)

    monkeypatch.setattr(analyzers, "extract_topic_keywords", _boom)
    keywords = extract_keywords(
        _minimal_document(), topic_keywords=["深度学习", "边缘计算"]
    )
    assert "深度学习" in keywords
    assert "边缘计算" in keywords


def test_extract_keywords_derives_topic_keywords_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _spy(_document: ThesisDocument) -> list[str]:
        calls.append("called")
        return [" spies "]

    monkeypatch.setattr(analyzers, "extract_topic_keywords", _spy)
    extract_keywords(_minimal_document())
    assert calls == ["called"]
