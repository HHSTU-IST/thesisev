"""Comment generation helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from thesisev.llm import ModelConfig, create_chat_model


def generate_comment(
    title: str,
    keywords: list[str],
    technology_details,
    topic_keywords: list[str],
    topic_relevance_ratio: float,
    root_sections,
    model_config: ModelConfig | None = None,
) -> tuple[str, dict[str, Any], str]:
    """Generate a concise content evaluation and validation checks."""

    focus_keywords = select_comment_keywords(title=title, fallback_keywords=keywords)
    comment, comment_source = generate_comment_with_llm(
        title=title,
        focus_keywords=focus_keywords,
        technology_details=technology_details,
        topic_keywords=topic_keywords,
        topic_relevance_ratio=topic_relevance_ratio,
        root_sections=root_sections,
        model_config=model_config,
    )
    checks = assess_comment(comment=comment, title=title, keywords=focus_keywords)
    if not checks["passes_keyword_coverage"]:
        comment = reinforce_keyword_coverage(comment, checks["missing_keywords"])
        checks = assess_comment(comment=comment, title=title, keywords=focus_keywords)
    return comment, checks, comment_source


def generate_comment_with_llm(
    *,
    title: str,
    focus_keywords: list[str],
    technology_details,
    topic_keywords: list[str],
    topic_relevance_ratio: float,
    root_sections,
    model_config: ModelConfig | None,
) -> tuple[str, str]:
    """Generate content commentary with an LLM and fall back to rule-based text."""

    fallback = build_rule_based_comment(
        title=title,
        focus_keywords=focus_keywords,
        technology_details=technology_details,
        topic_keywords=topic_keywords,
        topic_relevance_ratio=topic_relevance_ratio,
        root_sections=root_sections,
    )
    if model_config is None or not model_config.is_available():
        return fallback, "fallback"

    prompt = build_comment_prompt(
        title=title,
        focus_keywords=focus_keywords,
        technology_details=technology_details,
        topic_keywords=topic_keywords,
        topic_relevance_ratio=topic_relevance_ratio,
        root_sections=root_sections,
    )
    try:
        model = create_chat_model(model_config)
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是一名严谨的中文论文评审助手。"
                        "请根据给定分析结果生成一段 120 到 180 字的内容评价。"
                        "评价要聚焦选题、论证、方案、技术路线和创新价值。"
                        "不要评价格式、排版、标点规范，也不要给出分数。"
                        "必须包含至少两个关键词，不要直接复述完整论文标题。"
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
    except Exception:
        return fallback, "fallback"

    content = extract_response_text(response).strip()
    if not content:
        return fallback, "fallback"
    return content.replace(title, "、".join(focus_keywords) or "研究主题"), "llm"


def build_rule_based_comment(
    *,
    title: str,
    focus_keywords: list[str],
    technology_details,
    topic_keywords: list[str],
    topic_relevance_ratio: float,
    root_sections,
) -> str:
    """Build the original deterministic comment as a safe fallback."""

    structure_summary = summarize_structure(root_sections)
    topic_summary = summarize_topic_relevance(topic_keywords, topic_relevance_ratio)
    technology_summary = summarize_technology(technology_details)
    focus_text = "、".join(focus_keywords) if focus_keywords else "研究主题"
    comment = (
        f"论文围绕{focus_text}等内容展开，{structure_summary}。"
        f"{topic_summary}{technology_summary}"
    )
    return comment.replace(title, focus_text)


def build_comment_prompt(
    *,
    title: str,
    focus_keywords: list[str],
    technology_details,
    topic_keywords: list[str],
    topic_relevance_ratio: float,
    root_sections,
) -> str:
    """Build an LLM prompt from content-related thesis signals."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for item in technology_details:
        grouped[item.category].append(item.name)
    technology_summary = (
        "；".join(
            f"{category}: {', '.join(names[:4])}" for category, names in grouped.items()
        )
        or "未提取到明确技术栈"
    )
    section_summary = (
        "；".join(
            f"{section.title}({section.ratio * 100:.1f}%)"
            for section in root_sections[:6]
        )
        or "未识别到稳定章节结构"
    )
    return (
        f"论文标题：{title}\n"
        f"建议覆盖关键词：{'、'.join(focus_keywords) or '研究主题'}\n"
        f"主题关键词：{'、'.join(topic_keywords) or '无'}\n"
        f"主题相关内容占比：{topic_relevance_ratio * 100:.1f}%\n"
        f"章节分布：{section_summary}\n"
        f"技术栈：{technology_summary}\n"
        "请只评价论文内容质量，不要讨论格式、标点、排版或评分。"
        "请输出一段中文内容评价，不要分点。"
    )


def extract_response_text(response: Any) -> str:
    """Extract text content from a LangChain response object."""

    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


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


def assess_comment(*, comment: str, title: str, keywords: list[str]) -> dict[str, Any]:
    """Validate keyword coverage and title repetition."""

    covered_keywords = [
        keyword for keyword in keywords if keyword and keyword in comment
    ]
    missing_keywords = [
        keyword for keyword in keywords if keyword and keyword not in comment
    ]
    expected_keyword_count = min(2, len(keywords))
    passes_keyword_coverage = len(covered_keywords) >= expected_keyword_count
    repeats_title = title in comment
    return {
        "covered_keywords": covered_keywords,
        "missing_keywords": missing_keywords,
        "passes_keyword_coverage": passes_keyword_coverage,
        "repeats_title": repeats_title,
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
