"""B3: 确定性逻辑审查 — 证据链与数据一致性。"""

from __future__ import annotations

from thesisev.analyzers import detect_issue_groups
from thesisev.logic_review import detect_logic_issues
from thesisev.models import Paragraph, Section, Sentence, ThesisDocument


def _section(
    identifier: str,
    title: str,
    sentences: list[str],
    *,
    level: int = 1,
) -> Section:
    paragraphs = [
        Paragraph(
            index=0,
            text="".join(sentences),
            sentences=[
                Sentence(index=i, text=text) for i, text in enumerate(sentences)
            ],
            word_count=sum(len(text) for text in sentences),
        )
    ]
    return Section(
        identifier=identifier,
        level=level,
        title=title,
        heading=title,
        numbering=identifier,
        content="".join(sentences),
        paragraphs=paragraphs,
        sentences=sentences,
        word_count=sum(len(text) for text in sentences),
    )


def _document(sections: list[Section]) -> ThesisDocument:
    return ThesisDocument(
        title="逻辑样例",
        source_path="logic.md",
        source_type="md",
        raw_text="",
        cleaned_text="",
        front_matter="",
        abstract="",
        sections=sections,
        root_sections=[section for section in sections if section.level == 1],
        paragraphs=[],
        sentences=[],
        total_word_count=3000,
    )


def test_conclusion_without_evidence_is_flagged() -> None:
    doc = _document(
        [
            _section("1", "引言", ["本文提出一种改进算法。"]),
            _section("2", "相关技术", ["边缘计算与物联网是研究热点。"]),
            _section(
                "3",
                "结论",
                ["综上，本文方法具有较好的应用价值。", "未来可进一步优化。"],
            ),
        ]
    )
    issues = detect_logic_issues(doc)
    rules = [issue.rule_id for issue in issues]
    assert "conclusion_without_evidence" in rules
    issue = next(
        item for item in issues if item.rule_id == "conclusion_without_evidence"
    )
    assert issue.category == "逻辑问题"
    assert issue.section_title == "结论"


def test_conclusion_with_evidence_is_not_flagged() -> None:
    doc = _document(
        [
            _section("1", "引言", ["本文提出一种改进算法。"]),
            _section(
                "2", "实验与结果分析", ["实验表明算法有效。", "测试集准确率达 95%。"]
            ),
            _section("3", "结论", ["综上，方法具有较好的应用价值。"]),
        ]
    )
    rules = [issue.rule_id for issue in detect_logic_issues(doc)]
    assert "conclusion_without_evidence" not in rules


def test_conflicting_metric_values_across_sections() -> None:
    doc = _document(
        [
            _section("1", "引言", ["现有方法识别准确率约为 85%，仍有提升空间。"]),
            _section("2", "方案设计", ["本文方法识别准确率为 96%，识别精度显著提高。"]),
            _section("3", "实验结果", ["测试表明识别准确率达 97%，算法耗时约 12ms。"]),
            _section("4", "结论", ["综上，本文方法有效。"]),
        ]
    )
    rules = [issue.rule_id for issue in detect_logic_issues(doc)]
    assert "conflicting_metric_value" in rules


def test_consistent_metric_values_not_flagged() -> None:
    doc = _document(
        [
            _section("1", "引言", ["边缘计算目标检测是研究热点。"]),
            _section("2", "方案", ["本文方法识别准确率为 96%，识别速度也更快。"]),
            _section("3", "实验结果", ["最终识别准确率为 96%。", "与预期一致。"]),
            _section("4", "结论", ["综上，方法具有应用价值。"]),
        ]
    )
    rules = [issue.rule_id for issue in detect_logic_issues(doc)]
    assert "conflicting_metric_value" not in rules


def test_short_document_skips_logic_checks() -> None:
    doc = _document(
        [
            _section("1", "结论", ["摘要性短文档。"]),
        ]
    )
    assert detect_logic_issues(doc) == []


def test_issue_groups_expose_logic_issues() -> None:
    doc = _document(
        [
            _section("1", "引言", ["提出一种方法。"]),
            _section("2", "相关工作", ["概述已有研究。"]),
            _section("3", "结论", ["综上，方法有应用价值。"]),
        ]
    )
    groups = detect_issue_groups(doc)
    assert any(issue.category == "逻辑问题" for issue in groups.all_issues)
