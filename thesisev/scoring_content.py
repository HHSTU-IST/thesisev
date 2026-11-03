"""Local content scoring heuristics for thesis rubrics."""

from __future__ import annotations

import re
from typing import Any

from thesisev.models import Issue, TechnologyStackItem, ThesisDocument
from thesisev.rubric_utils import RubricItem, ScoreCriterion, build_criterion


def score_topic_workload(
    document: ThesisDocument,
    topic_analysis: dict[str, Any],
    technology_details: list[TechnologyStackItem],
    rubric_item: RubricItem,
) -> ScoreCriterion:
    """Score topic value, workload, and overall completeness."""

    max_score = rubric_item.max_score
    score = 0.0
    evidence: list[str] = [
        f"篇幅 {document.total_word_count}",
        f"章节数 {len(document.sections)}",
        f"主题相关占比 {float(topic_analysis.get('document_ratio', 0)) * 100:.1f}%",
    ]
    deductions: list[str] = []
    suggestions: list[str] = []

    score += max_score * ratio_from_thresholds(
        document.total_word_count, ((5000, 0.3), (3000, 0.22), (1500, 0.14))
    )
    score += max_score * ratio_from_thresholds(
        len(document.sections), ((5, 0.25), (3, 0.18), (2, 0.1))
    )
    score += max_score * ratio_from_thresholds(
        float(topic_analysis.get("document_ratio", 0)),
        ((0.6, 0.25), (0.45, 0.18), (0.3, 0.1)),
    )
    score += max_score * ratio_from_thresholds(
        len(technology_details), ((3, 0.2), (2, 0.16), (1, 0.1))
    )

    if document.total_word_count < 3000:
        deductions.append("正文规模偏小，工作量支撑不足")
        suggestions.append("补充需求分析、系统设计或实验验证内容")
    if len(document.sections) < 3:
        deductions.append("章节结构偏少，综合训练过程呈现不足")
    if float(topic_analysis.get("document_ratio", 0)) < 0.45:
        deductions.append("主题相关内容占比偏低")
        suggestions.append("压缩泛化背景描述，增加与课题直接相关的分析")
    if len(technology_details) < 2:
        deductions.append("可识别技术覆盖较少")

    return build_criterion(
        key="topic_workload",
        rubric_item=rubric_item,
        score=score,
        evidence=evidence,
        deductions=deductions,
        suggestions=suggestions,
    )


def score_research_argument(
    document: ThesisDocument, keywords: list[str], rubric_item: RubricItem
) -> ScoreCriterion:
    """Score literature use and argumentation signals."""

    text = document.cleaned_text
    max_score = rubric_item.max_score
    citation_count = count_citations(text)
    reference_section = has_any(text, ("参考文献", "文献综述", "相关研究"))
    argument_count = count_terms(
        text, ("因此", "表明", "综上", "说明", "可见", "依据", "分析")
    )
    keyword_hits = sum(1 for keyword in keywords if keyword and keyword in text)
    score = 0.0
    score += max_score * (0.28 if reference_section else 0.0)
    score += max_score * ratio_from_thresholds(
        citation_count, ((8, 0.28), (4, 0.2), (1, 0.1))
    )
    score += max_score * ratio_from_thresholds(
        argument_count, ((12, 0.24), (6, 0.17), (2, 0.08))
    )
    score += max_score * ratio_from_thresholds(
        keyword_hits, ((3, 0.2), (2, 0.14), (1, 0.08))
    )

    evidence = [
        f"引用标记 {citation_count} 处",
        f"论证表达 {argument_count} 处",
        f"关键词覆盖 {keyword_hits}/{len(keywords)}",
    ]
    deductions: list[str] = []
    suggestions: list[str] = []
    if not reference_section:
        deductions.append("未识别到参考文献、文献综述或相关研究部分")
        suggestions.append("补充参考文献列表，并在正文中建立引用对应关系")
    if citation_count < 4:
        deductions.append("引用数量偏少或引用标记不明显")
    if argument_count < 6:
        deductions.append("分析、综合和归纳类论证表达不足")

    return build_criterion(
        key="research_argument",
        rubric_item=rubric_item,
        score=score,
        evidence=evidence,
        deductions=deductions,
        suggestions=suggestions,
    )


def score_translation(
    document: ThesisDocument, rubric_item: RubricItem
) -> ScoreCriterion:
    """Score Chinese-English abstract translation completeness."""

    text = document.raw_text
    max_score = rubric_item.max_score
    has_chinese_abstract = has_any(text, ("摘要", "中文摘要"))
    has_english_abstract = bool(re.search(r"\bAbstract\b", text, re.IGNORECASE))
    english_words = len(re.findall(r"\b[A-Za-z][A-Za-z\-]{2,}\b", text))
    english_sentences = len(re.findall(r"[A-Za-z][^.!?]{12,}[.!?]", text))
    score = 0.0
    score += max_score * (0.28 if has_chinese_abstract else 0.0)
    score += max_score * (0.32 if has_english_abstract else 0.0)
    score += max_score * ratio_from_thresholds(
        english_words, ((120, 0.22), (60, 0.16), (25, 0.08))
    )
    score += max_score * ratio_from_thresholds(
        english_sentences, ((5, 0.18), (3, 0.12), (1, 0.06))
    )

    evidence = [
        f"中文摘要 {'已识别' if has_chinese_abstract else '未识别'}",
        f"英文摘要 {'已识别' if has_english_abstract else '未识别'}",
        f"英文词数约 {english_words}",
    ]
    deductions: list[str] = []
    suggestions: list[str] = []
    if not has_english_abstract:
        deductions.append("未识别到英文摘要 Abstract")
        suggestions.append("补充英文摘要，并覆盖研究目标、方法、结果和关键词")
    if english_words < 60:
        deductions.append("英文摘要篇幅偏短，译文完整性不足")

    return build_criterion(
        key="translation",
        rubric_item=rubric_item,
        score=score,
        evidence=evidence,
        deductions=deductions,
        suggestions=suggestions,
    )


def score_experiment_analysis(
    document: ThesisDocument,
    technology_details: list[TechnologyStackItem],
    rubric_item: RubricItem,
) -> ScoreCriterion:
    """Score design, data, analysis, feasibility, and benefit signals."""

    text = document.cleaned_text
    max_score = rubric_item.max_score
    scheme_terms = count_terms(text, ("方案", "设计", "架构", "流程", "方法", "实现"))
    data_terms = count_terms(
        text, ("数据", "采集", "计算", "处理", "结果", "验证", "测试")
    )
    analysis_terms = count_terms(text, ("分析", "论证", "对比", "评估", "推导", "可靠"))
    feasibility_terms = count_terms(
        text, ("可行", "工艺", "成本", "社会", "经济", "效益")
    )
    figure_table_count = count_terms(text, ("图", "表"))
    score = 0.0
    score += max_score * ratio_from_thresholds(
        scheme_terms, ((10, 0.24), (5, 0.17), (2, 0.08))
    )
    score += max_score * ratio_from_thresholds(
        data_terms, ((10, 0.24), (5, 0.17), (2, 0.08))
    )
    score += max_score * ratio_from_thresholds(
        analysis_terms, ((8, 0.22), (4, 0.15), (1, 0.06))
    )
    score += max_score * ratio_from_thresholds(
        feasibility_terms, ((4, 0.14), (2, 0.09), (1, 0.05))
    )
    score += max_score * ratio_from_thresholds(
        len(technology_details), ((3, 0.1), (2, 0.07), (1, 0.04))
    )
    score += max_score * ratio_from_thresholds(
        figure_table_count, ((8, 0.06), (3, 0.04), (1, 0.02))
    )

    evidence = [
        f"方案/设计相关表达 {scheme_terms} 处",
        f"数据/测试相关表达 {data_terms} 处",
        f"分析/论证相关表达 {analysis_terms} 处",
        f"可行性/效益相关表达 {feasibility_terms} 处",
    ]
    deductions: list[str] = []
    suggestions: list[str] = []
    if scheme_terms < 5:
        deductions.append("实验或设计方案描述不够充分")
    if data_terms < 5:
        deductions.append("数据采集、计算、处理或测试结果呈现不足")
    if feasibility_terms < 2:
        deductions.append("社会、经济效益或可行性分析较弱")
        suggestions.append("增加可行性、成本收益或应用场景分析")

    return build_criterion(
        key="experiment_analysis",
        rubric_item=rubric_item,
        score=score,
        evidence=evidence,
        deductions=deductions,
        suggestions=suggestions,
    )


def score_writing_quality(
    document: ThesisDocument, writing_issues: list[Issue], rubric_item: RubricItem
) -> ScoreCriterion:
    """Score writing quality from locally detected writing issues."""

    max_score = rubric_item.max_score
    severity_weights = {"low": 0.35, "medium": 0.65, "high": 1.0}
    issue_counts: dict[tuple[str, str], int] = {}
    issue_examples: dict[tuple[str, str], Issue] = {}
    for issue in writing_issues:
        key = (issue.category, issue.rule_id)
        issue_counts[key] = issue_counts.get(key, 0) + 1
        issue_examples.setdefault(key, issue)

    deduction = 0.0
    for key, count in issue_counts.items():
        issue = issue_examples[key]
        issue_deduction = severity_weights.get(issue.severity, 0.8)
        deduction += issue_deduction * count
    if len(document.sections) <= 1:
        deduction += 1.2
    if not document.abstract:
        deduction += 0.8
    score = max_score - min(max_score * 0.8, deduction)

    evidence = [
        f"书面表达问题 {len(writing_issues)} 项",
        f"章节数 {len(document.sections)}",
    ]
    deductions: list[str] = []
    suggestions: list[str] = []
    if writing_issues:
        deductions.append("存在口语化表达问题")
        suggestions.append("统一技术论文书面表达")
    if len(document.sections) <= 1:
        deductions.append("章节结构不稳定，影响条理性判断")
    if not document.abstract:
        deductions.append("未识别到摘要")
    return build_criterion(
        key="writing_quality",
        rubric_item=rubric_item,
        score=score,
        evidence=evidence,
        deductions=deductions,
        suggestions=suggestions,
    )


def score_innovation(
    document: ThesisDocument,
    technology_details: list[TechnologyStackItem],
    rubric_item: RubricItem,
) -> ScoreCriterion:
    """Score innovation and application-value signals."""

    text = document.cleaned_text
    max_score = rubric_item.max_score
    innovation_terms = count_terms(
        text, ("创新", "提出", "改进", "优化", "新型", "首次", "贡献")
    )
    value_terms = count_terms(
        text, ("应用价值", "推广", "实践", "落地", "价值", "意义", "展望")
    )
    conclusion_terms = count_terms(text, ("结论", "总结", "展望"))
    score = 0.0
    score += max_score * ratio_from_thresholds(
        innovation_terms, ((5, 0.45), (3, 0.32), (1, 0.16))
    )
    score += max_score * ratio_from_thresholds(
        value_terms, ((4, 0.3), (2, 0.2), (1, 0.1))
    )
    score += max_score * ratio_from_thresholds(conclusion_terms, ((2, 0.15), (1, 0.08)))
    score += max_score * ratio_from_thresholds(
        len(technology_details), ((3, 0.1), (2, 0.07), (1, 0.04))
    )

    evidence = [
        f"创新相关表达 {innovation_terms} 处",
        f"应用价值相关表达 {value_terms} 处",
        f"结论/展望相关表达 {conclusion_terms} 处",
    ]
    deductions: list[str] = []
    suggestions: list[str] = []
    if innovation_terms < 3:
        deductions.append("创新点表述不够明确")
        suggestions.append("用单独段落说明改进点、差异化设计和预期价值")
    if value_terms < 2:
        deductions.append("应用价值或推广意义支撑不足")

    return build_criterion(
        key="innovation",
        rubric_item=rubric_item,
        score=score,
        evidence=evidence,
        deductions=deductions,
        suggestions=suggestions,
    )


def ratio_from_thresholds(
    value: float, thresholds: tuple[tuple[float, float], ...]
) -> float:
    """Return the first ratio whose threshold is met."""

    for threshold, ratio in thresholds:
        if value >= threshold:
            return ratio
    return 0.0


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return whether any term appears in text."""

    return any(term in text for term in terms)


def count_terms(text: str, terms: tuple[str, ...]) -> int:
    """Count simple literal term occurrences."""

    return sum(text.count(term) for term in terms)


def count_citations(text: str) -> int:
    """Count Arabic-numbered citation markers in Chinese technical papers."""

    bracket_refs = re.findall(r"\[\d+(?:[-,，]\d+)*\]", text)
    return len(bracket_refs)
