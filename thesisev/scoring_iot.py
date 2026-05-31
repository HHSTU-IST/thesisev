"""Local scoring heuristics for IoT report rubrics."""

from __future__ import annotations

from typing import Any

from thesisev.models import Issue, TechnologyStackItem, ThesisDocument
from thesisev.resources import load_json_resource
from thesisev.scoring import build_criterion
from thesisev.scoring_content import count_terms, ratio_from_thresholds
from thesisev.scoring_format import (
    extract_format_rules,
    normalize_format_spec_payload,
    score_format_rules,
)

DEFAULT_IOT_FORMAT_RUBRIC = "score_report_iot_f.json"


def score_iot_item_locally(
    *,
    document: ThesisDocument,
    format_issues: list[Issue],
    writing_issues: list[Issue],
    technology_details: list[TechnologyStackItem],
    format_requirements: dict[str, Any] | None,
    rubric_item,
):
    """Score a configured IoT report rubric item, if this module owns it."""

    if rubric_item.name == "调研背景与意义":
        return score_iot_background(document, rubric_item)
    if rubric_item.name == "调研方法和思路":
        return score_iot_method(document, rubric_item)
    if rubric_item.name == "软件选型":
        return score_iot_software(document, technology_details, rubric_item)
    if rubric_item.name == "硬件选型":
        return score_iot_hardware(document, technology_details, rubric_item)
    if rubric_item.name == "成本核算":
        return score_iot_cost(document, rubric_item)
    if rubric_item.name == "未来展望":
        return score_iot_outlook(document, rubric_item)
    if rubric_item.name == "篇幅":
        return score_iot_word_count(document, rubric_item)
    if rubric_item.name == "报告撰写":
        return score_iot_writing(document, writing_issues, rubric_item)
    if rubric_item.name == "格式规范":
        return score_iot_format(
            document, format_issues, format_requirements, rubric_item
        )
    return None


def score_iot_background(document: ThesisDocument, rubric_item):
    """Score IoT report background and significance."""

    text = document.cleaned_text
    terms = count_terms(text, ("物联网", "图像处理", "AI", "应用", "场景", "意义"))
    score = rubric_item.max_score * ratio_from_thresholds(
        terms, ((12, 0.6), (6, 0.42), (2, 0.2))
    )
    return build_criterion(
        key="iot_background",
        rubric_item=rubric_item,
        score=score,
        evidence=[f"相关表述 {terms} 处"],
        deductions=[] if terms >= 4 else ["背景与意义展开不足"],
        suggestions=[] if terms >= 4 else ["补充调研背景、行业意义和应用场景"],
    )


def score_iot_method(document: ThesisDocument, rubric_item):
    """Score IoT report method and thinking."""

    text = document.cleaned_text
    terms = count_terms(text, ("关键字", "网站", "资料", "方法", "思路", "调研"))
    score = rubric_item.max_score * ratio_from_thresholds(
        terms, ((10, 0.6), (5, 0.4), (2, 0.18))
    )
    return build_criterion(
        key="iot_method",
        rubric_item=rubric_item,
        score=score,
        evidence=[f"方法相关表述 {terms} 处"],
        deductions=[] if terms >= 3 else ["调研方法或思路描述较少"],
        suggestions=[] if terms >= 3 else ["补充信息来源、检索关键词和调研路径"],
    )


def score_iot_software(
    document: ThesisDocument,
    technology_details: list[TechnologyStackItem],
    rubric_item,
):
    """Score software selection."""

    text = document.cleaned_text
    terms = count_terms(text, ("软件", "架构", "平台", "框架", "对比", "理由"))
    tech_hits = sum(
        1 for item in technology_details if item.category in {"software", "platform"}
    )
    score = rubric_item.max_score * min(
        1.0,
        ratio_from_thresholds(terms, ((10, 0.5), (5, 0.35), (2, 0.18)))
        + ratio_from_thresholds(tech_hits, ((3, 0.4), (1, 0.2))),
    )
    return build_criterion(
        key="iot_software",
        rubric_item=rubric_item,
        score=score,
        evidence=[f"软件选型表述 {terms} 处", f"识别到软件类技术 {tech_hits} 项"],
        deductions=[] if terms >= 3 else ["软件选型理由不足"],
        suggestions=[] if terms >= 3 else ["补充方案对比和选择理由"],
    )


def score_iot_hardware(
    document: ThesisDocument,
    technology_details: list[TechnologyStackItem],
    rubric_item,
):
    """Score hardware selection."""

    text = document.cleaned_text
    terms = count_terms(
        text, ("硬件", "设备", "传感器", "主控", "模块", "对比", "理由")
    )
    tech_hits = sum(
        1 for item in technology_details if item.category in {"hardware", "device"}
    )
    score = rubric_item.max_score * min(
        1.0,
        ratio_from_thresholds(terms, ((10, 0.5), (5, 0.35), (2, 0.18)))
        + ratio_from_thresholds(tech_hits, ((3, 0.4), (1, 0.2))),
    )
    return build_criterion(
        key="iot_hardware",
        rubric_item=rubric_item,
        score=score,
        evidence=[f"硬件选型表述 {terms} 处", f"识别到硬件类技术 {tech_hits} 项"],
        deductions=[] if terms >= 3 else ["硬件选型理由不足"],
        suggestions=[] if terms >= 3 else ["补充硬件对比和选择理由"],
    )


def score_iot_cost(document: ThesisDocument, rubric_item):
    """Score cost accounting."""

    text = document.cleaned_text
    terms = count_terms(text, ("成本", "核算", "预算", "费用", "单价", "总价"))
    score = rubric_item.max_score * ratio_from_thresholds(
        terms, ((8, 0.7), (4, 0.45), (2, 0.2))
    )
    return build_criterion(
        key="iot_cost",
        rubric_item=rubric_item,
        score=score,
        evidence=[f"成本核算表述 {terms} 处"],
        deductions=[] if terms >= 3 else ["成本核算方法不够充分"],
        suggestions=[] if terms >= 3 else ["补充成本估算依据和计算过程"],
    )


def score_iot_outlook(document: ThesisDocument, rubric_item):
    """Score future outlook."""

    text = document.cleaned_text
    terms = count_terms(text, ("展望", "未来", "发展", "趋势", "行业", "应用"))
    score = rubric_item.max_score * ratio_from_thresholds(
        terms, ((8, 0.7), (4, 0.45), (2, 0.2))
    )
    return build_criterion(
        key="iot_outlook",
        rubric_item=rubric_item,
        score=score,
        evidence=[f"展望相关表述 {terms} 处"],
        deductions=[] if terms >= 3 else ["未来展望偏少"],
        suggestions=[] if terms >= 3 else ["补充与物联网和图像处理相关的未来方向"],
    )


def score_iot_word_count(document: ThesisDocument, rubric_item):
    """Score report length."""

    word_count = document.total_word_count
    score = rubric_item.max_score * ratio_from_thresholds(
        word_count, ((3500, 0.8), (2500, 0.5), (1500, 0.2))
    )
    return build_criterion(
        key="iot_word_count",
        rubric_item=rubric_item,
        score=score,
        evidence=[f"篇幅 {word_count}"],
        deductions=[] if word_count >= 2500 else ["篇幅不足"],
        suggestions=[] if word_count >= 2500 else ["补充图表说明和调研内容"],
    )


def score_iot_writing(
    document: ThesisDocument, writing_issues: list[Issue], rubric_item
):
    """Score report writing quality."""

    issue_penalty = sum(
        0.5 if issue.severity == "low" else 1.0 for issue in writing_issues
    )
    score = max(
        0.0, rubric_item.max_score - min(rubric_item.max_score * 0.8, issue_penalty)
    )
    return build_criterion(
        key="iot_writing",
        rubric_item=rubric_item,
        score=score,
        evidence=[f"书面表达问题 {len(writing_issues)} 项"],
        deductions=[] if not writing_issues else ["存在口语化表达问题"],
        suggestions=[] if not writing_issues else ["统一正文书面表达"],
    )


def score_iot_format(
    document: ThesisDocument,
    format_issues: list[Issue],
    format_requirements: dict[str, Any] | None,
    rubric_item,
):
    """Score format compliance."""

    format_spec = load_json_resource(DEFAULT_IOT_FORMAT_RUBRIC)
    format_spec = normalize_format_spec_payload(format_spec)
    rules = extract_format_rules(format_spec)
    if not rules:
        raise ValueError("format rubric rules are empty")
    summary = score_format_rules(
        document=document,
        format_issues=format_issues,
        rules=rules,
        format_requirements=format_requirements,
    )
    format_count = (
        len(format_requirements.get("items", [])) if format_requirements else 0
    )
    score = max(
        0.0,
        rubric_item.max_score - min(rubric_item.max_score * 0.8, summary["deduction"]),
    )
    criterion = build_criterion(
        key="iot_format",
        rubric_item=rubric_item,
        score=score,
        evidence=[
            f"格式规范来源 {DEFAULT_IOT_FORMAT_RUBRIC}",
            f"格式要求条目 {format_count}",
            f"格式问题 {len(format_issues)} 项",
        ]
        + summary["evidence"],
        deductions=summary["deductions"],
        suggestions=summary["suggestions"],
    )
    criterion.evaluation = "local_program"
    return criterion
