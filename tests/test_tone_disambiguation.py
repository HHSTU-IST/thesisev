"""B3: 语气（口语化）词典消歧 — 降低上下文误报。"""

from __future__ import annotations

from thesisev.analyzers import detect_colloquial_issues
from thesisev.models import Paragraph, Section, Sentence, ThesisDocument


def _doc_with(*sentences: str) -> ThesisDocument:
    paragraph = Paragraph(
        index=0,
        text="".join(sentences),
        sentences=[Sentence(index=i, text=text) for i, text in enumerate(sentences)],
        word_count=sum(len(text) for text in sentences),
    )
    section = Section(
        identifier="1",
        level=1,
        title="正文",
        heading="正文",
        numbering="1",
        content="".join(sentences),
        paragraphs=[paragraph],
        sentences=list(sentences),
        word_count=sum(len(text) for text in sentences),
    )
    return ThesisDocument(
        title="语气样例",
        source_path="tone.md",
        source_type="md",
        raw_text="",
        cleaned_text="",
        front_matter="",
        abstract="",
        sections=[section],
        root_sections=[section],
        paragraphs=[paragraph],
        sentences=list(sentences),
        total_word_count=2000,
    )


def _rules(doc: ThesisDocument) -> list[str]:
    return [issue.rule_id for issue in detect_colloquial_issues(doc)]


def test_plain_colloquial_still_detected() -> None:
    assert "colloquial_actually" in _rules(_doc_with("其实，这种方法仍有改进空间。"))


def test_actually_as_fixed_compound_is_skipped() -> None:
    doc = _doc_with("该问题的其实质在于数据分布不均衡。", "其实际开销取决于模型结构。")
    assert "colloquial_actually" not in _rules(doc)


def test_single_char_very_is_never_reported() -> None:
    doc = _doc_with("该方法具有很强的实时性，适用于边缘设备。")
    assert "colloquial_very" not in _rules(doc)


def test_especially_in_particular_is_skipped() -> None:
    doc = _doc_with("特别是在资源受限场景下，轻量化设计尤为重要。")
    assert "colloquial_special" not in _rules(doc)


def test_then_before_operation_is_skipped() -> None:
    doc = _doc_with("然后将结果写入缓存，最后统一上报。")
    assert "colloquial_then" not in _rules(doc)
