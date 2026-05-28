"""Comment generation helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def generate_comment(
    title: str,
    keywords: list[str],
    technology_details,
    topic_keywords: list[str],
    topic_relevance_ratio: float,
    score: int,
    issues,
    root_sections,
) -> tuple[str, dict[str, Any]]:
    """Generate a concise thesis evaluation comment and validation checks."""

    focus_keywords = select_comment_keywords(title=title, fallback_keywords=keywords)
    quality_phrase = score_to_phrase(score)
    structure_summary = summarize_structure(root_sections)
    topic_summary = summarize_topic_relevance(topic_keywords, topic_relevance_ratio)
    technology_summary = summarize_technology(technology_details)
    issue_summary = summarize_issues(issues)
    score_summary = summarize_score(score)

    focus_text = "、".join(focus_keywords) if focus_keywords else "研究主题"
    comment = (
        f"论文围绕{focus_text}等内容展开，{structure_summary}，整体质量{quality_phrase}。"
        f"{topic_summary}{technology_summary}{issue_summary}{score_summary}当前评分约为 {score} 分。"
    )
    comment = comment.replace(title, focus_text)
    checks = assess_comment(
        comment=comment, title=title, keywords=focus_keywords, score=score
    )
    if not checks["passes_keyword_coverage"]:
        comment = reinforce_keyword_coverage(comment, checks["missing_keywords"])
        checks = assess_comment(
            comment=comment, title=title, keywords=focus_keywords, score=score
        )
    return comment, checks


def score_to_phrase(score: int) -> str:
    """Map a score to a short quality phrase."""

    if score >= 90:
        return "较为完整"
    if score >= 80:
        return "基本清晰"
    if score >= 70:
        return "较为基础"
    return "仍需加强"


def select_comment_keywords(title: str, fallback_keywords: list[str]) -> list[str]:
    """Pick a small set of title-related keywords for the comment."""

    title_keywords = extract_title_keywords(title)
    selected = deduplicate_preserving_order(title_keywords)
    if len(selected) < 2:
        for keyword in fallback_keywords:
            if keyword == title or keyword in GENERIC_TITLE_TERMS:
                continue
            selected.append(keyword)
    compact = [keyword for keyword in selected if keyword not in GENERIC_TITLE_TERMS]
    return compact[:3]


def extract_title_keywords(title: str) -> list[str]:
    """Extract compact title keywords while skipping generic thesis terms."""

    parts = re.split(r"[的与及和：:（）()\-\s]+", title)
    keywords = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if stripped in GENERIC_TITLE_TERMS:
            continue
        if stripped.isdigit():
            continue
        if len(stripped) == 1 and not re.search(r"[A-Za-z]", stripped):
            continue
        keywords.append(stripped)
    return keywords


def summarize_structure(root_sections) -> str:
    """Summarize structural completeness from section distribution."""

    if not root_sections:
        return "结构信息暂不充分"
    top_ratio = max(section.ratio for section in root_sections)
    section_count = len(root_sections)
    if section_count >= 3 and top_ratio <= 0.45:
        return "章节安排较为均衡"
    if top_ratio >= 0.6:
        return "章节分配略有失衡"
    if section_count >= 2:
        return "章节结构基本完整"
    return "结构层次仍可进一步完善"


def summarize_technology(technology_details) -> str:
    """Summarize technology extraction results for the comment."""

    if not technology_details:
        return "技术方案描述仍可进一步具体化。"
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in technology_details:
        grouped[item.category].append(item.name)
    category_names = list(grouped)
    first_category = category_names[0]
    tech_names = grouped[first_category][:2]
    tech_text = "、".join(tech_names)
    if len(category_names) >= 2:
        return (
            f"技术方案中体现出对{first_category}与{category_names[1]}的关注，"
            f"如{tech_text}等要素已有一定体现。"
        )
    return f"技术方案中已体现出对{first_category}的关注，如{tech_text}等要素。"


def summarize_topic_relevance(
    topic_keywords: list[str], topic_relevance_ratio: float
) -> str:
    """Summarize topical focus from the relevance analysis."""

    if not topic_keywords:
        return "主题关键词仍可进一步明确。"
    keyword_text = "、".join(topic_keywords[:2])
    if topic_relevance_ratio >= 0.7:
        return f"主题聚焦度较好，与{keyword_text}相关的内容占比较高。"
    if topic_relevance_ratio >= 0.45:
        return f"主题关联内容占比基本合理，核心内容能够围绕{keyword_text}展开。"
    return f"与{keyword_text}相关的内容占比仍然偏低，存在一定偏题风险。"


def summarize_issues(issues) -> str:
    """Summarize issue detection results for the comment."""

    issue_count = len(issues)
    if issue_count == 0:
        return "全文表达较为规范，暂未发现明显格式或口语化问题。"
    medium_count = sum(issue.severity == "medium" for issue in issues)
    categories = {issue.category for issue in issues}
    if medium_count <= 2 and issue_count <= 3:
        return "文中存在少量格式或表达细节需要润色，但整体不影响主要内容呈现。"
    if {"标点误用", "口语化表达"} <= categories:
        return "文中在标点规范性与学术化表达方面仍有较明显的优化空间，建议集中修订。"
    return "文中仍存在若干表达与规范问题，建议在正式提交前统一校正。"


def summarize_score(score: int) -> str:
    """Generate a score-consistent closing sentence fragment."""

    if score >= 90:
        return "综合来看，该文完成度较高，"
    if score >= 80:
        return "综合来看，该文整体表现较为稳健，"
    if score >= 70:
        return "综合来看，该文具有一定完成度，"
    return "综合来看，该文仍需进一步打磨，"


def assess_comment(
    *, comment: str, title: str, keywords: list[str], score: int
) -> dict[str, Any]:
    """Validate keyword coverage, title repetition, and score alignment."""

    covered_keywords = [
        keyword for keyword in keywords if keyword and keyword in comment
    ]
    missing_keywords = [
        keyword for keyword in keywords if keyword and keyword not in comment
    ]
    expected_keyword_count = min(2, len(keywords))
    passes_keyword_coverage = len(covered_keywords) >= expected_keyword_count
    repeats_title = title in comment
    mentions_score = f"{score} 分" in comment
    expected_score_summary = summarize_score(score).strip("，")
    has_score_alignment = expected_score_summary in comment and mentions_score
    return {
        "covered_keywords": covered_keywords,
        "missing_keywords": missing_keywords,
        "passes_keyword_coverage": passes_keyword_coverage,
        "repeats_title": repeats_title,
        "has_score_alignment": has_score_alignment,
    }


def reinforce_keyword_coverage(comment: str, missing_keywords: list[str]) -> str:
    """Append missing keywords to the opening sentence when needed."""

    if not missing_keywords:
        return comment
    addition = "、".join(missing_keywords[:2])
    return comment.replace("等内容展开", f"、{addition}等内容展开", 1)


def deduplicate_preserving_order(values: list[str]) -> list[str]:
    """Deduplicate a list while preserving order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


GENERIC_TITLE_TERMS = {
    "论文",
    "设计",
    "实现",
    "系统",
    "研究",
    "分析",
    "评价",
    "助手",
    "基于",
    "面向",
    "方法",
}
