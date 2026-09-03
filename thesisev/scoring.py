"""Rule-based scoring for thesis evaluation rubrics."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from thesisev.llm import (
    ModelConfig,
    create_chat_model,
    extract_response_text,
    invoke_chat_model_with_retry,
)
from thesisev.models import Issue, TechnologyStackItem, ThesisDocument
from thesisev.resources import load_json_resource
from thesisev.rubric_utils import (
    FORMAT_RUBRIC_KEY,
    RubricItem,
    ScoreCriterion,
    build_criterion,
    merge_rubric_items,
    normalize_rubric_items,
    normalize_rubric_payload,
    parse_score_value,
)
from thesisev.scoring_content import (
    score_experiment_analysis,
    score_innovation,
    score_research_argument,
    score_topic_workload,
    score_translation,
    score_writing_quality,
)
from thesisev.scoring_format import (
    extract_format_rules,
    format_expected_value,
    get_rule_expected,
    normalize_format_spec_payload,
    parse_float_value,
    score_format_compliance,
)
from thesisev.scoring_iot import OWNED_IOT_ITEM_NAMES, score_iot_item_locally

logger = logging.getLogger(__name__)

DEFAULT_THESIS_TECH_RUBRIC = "score_thesis_tech.json"
FORMAT_RUBRIC_BY_SCORE_RUBRIC = {
    "score_report_iot.json": "score_report_iot_f.json",
    "score_thesis_tech.json": "score_thesis_tech_f.json",
}

#: Stable keys implemented by the local content scorers in scoring_content.py.
THESIS_LOCAL_SCORER_KEYS = frozenset(
    {
        "topic_workload",
        "research_argument",
        "translation",
        "experiment_analysis",
        "writing_quality",
        "innovation",
    }
)


class ScoreReport:
    """Normalized score report derived from rubric criteria."""

    def __init__(
        self,
        *,
        score: int,
        raw_score: float,
        raw_total: float,
        criteria: list[ScoreCriterion],
        rubric_source: str,
        score_source: str,
    ) -> None:
        self.score = score
        self.raw_score = raw_score
        self.raw_total = raw_total
        self.criteria = criteria
        self.rubric_source = rubric_source
        self.score_source = score_source

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-friendly dictionary."""

        return {
            "score": self.score,
            "raw_score": self.raw_score,
            "raw_total": self.raw_total,
            "rubric_source": self.rubric_source,
            "score_source": self.score_source,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
        }


def calculate_score_report(
    *,
    document: ThesisDocument,
    topic_analysis: dict[str, Any],
    format_issues: list[Issue],
    writing_issues: list[Issue],
    content_context: dict[str, Any],
    keywords: list[str],
    technology_details: list[TechnologyStackItem],
    format_requirements: dict[str, Any] | None = None,
    rubric: dict[str, Any] | None = None,
    rubric_filename: str = DEFAULT_THESIS_TECH_RUBRIC,
    model_config: ModelConfig | None = None,
) -> ScoreReport:
    """Calculate a rubric-based percentage score from rubric config."""

    rubric_items, rubric_source = resolve_rubric_items(
        rubric=rubric, rubric_filename=rubric_filename
    )
    criteria = score_rubric_items(
        document=document,
        topic_analysis=topic_analysis,
        format_issues=format_issues,
        writing_issues=writing_issues,
        content_context=content_context,
        keywords=keywords,
        technology_details=technology_details,
        format_requirements=format_requirements,
        rubric_items=rubric_items,
        rubric_source=rubric_source,
        rubric_filename=rubric_filename,
        model_config=model_config,
    )
    return build_score_report(
        criteria=criteria,
        rubric_source=rubric_source,
        score_source=determine_score_source(criteria),
    )


def score_rubric_items(
    *,
    document: ThesisDocument,
    topic_analysis: dict[str, Any],
    format_issues: list[Issue],
    writing_issues: list[Issue],
    content_context: dict[str, Any],
    keywords: list[str],
    technology_details: list[TechnologyStackItem],
    format_requirements: dict[str, Any] | None,
    rubric_items: list[RubricItem],
    rubric_source: str,
    rubric_filename: str,
    model_config: ModelConfig | None,
) -> list[ScoreCriterion]:
    """Score each rubric item according to its configured evaluation method."""

    validate_rubric_local_scorability(rubric_items, rubric_filename=rubric_filename)
    local_kwargs: dict[str, Any] = {
        "document": document,
        "topic_analysis": topic_analysis,
        "format_issues": format_issues,
        "writing_issues": writing_issues,
        "keywords": keywords,
        "technology_details": technology_details,
        "format_requirements": format_requirements,
    }
    criteria: list[ScoreCriterion] = []
    for item in rubric_items:
        method = item.evaluation.strip().lower()
        if method == "llm" and model_config is not None and model_config.is_available():
            try:
                criteria.append(
                    score_item_with_llm(
                        content_context=content_context,
                        rubric_item=item,
                        rubric_filename=rubric_filename,
                        model_config=model_config,
                    )
                )
            except Exception as exc:  # degrade one item, never fail the review
                logger.warning(
                    "llm scoring failed for %s (rubric=%s); falling back to local: %s",
                    item.name,
                    rubric_filename,
                    exc,
                )
                criteria.append(
                    mark_llm_fallback_if_needed(
                        score_item_locally(
                            rubric_item=item,
                            rubric_filename=rubric_filename,
                            **local_kwargs,
                        ),
                        requested_method="llm",
                    )
                )
            continue
        criteria.append(
            mark_llm_fallback_if_needed(
                score_item_locally(
                    rubric_item=item, rubric_filename=rubric_filename, **local_kwargs
                ),
                requested_method=method,
            )
        )
    return criteria


def validate_rubric_local_scorability(
    rubric_items: list[RubricItem], *, rubric_filename: str
) -> None:
    """Raise early when a rubric item has no local scorer behind it.

    LLM-configured items may be answered by any model, so they are exempt
    unless the item belongs to a built-in thesis rubric whose local fallback
    must stay complete. Local items (or thesis items, which can silently fall
    back to local when no API key is configured) must resolve to an
    implemented scorer, otherwise an unknown criterion would previously have
    been reported as a silent zero score.
    """

    thesis_tech = rubric_filename.startswith("score_thesis_tech")
    supported = (
        THESIS_LOCAL_SCORER_KEYS | {FORMAT_RUBRIC_KEY} | set(OWNED_IOT_ITEM_NAMES)
    )
    for item in rubric_items:
        if item.evaluation == "llm" and not thesis_tech:
            continue
        if (item.key or item.name) in supported or item.name in supported:
            continue
        msg = (
            "rubric item has no local scorer: "
            f"name={item.name!r} key={item.key!r} "
            f"(rubric={rubric_filename})"
        )
        raise ValueError(msg)


def mark_llm_fallback_if_needed(
    criterion: ScoreCriterion, *, requested_method: str
) -> ScoreCriterion:
    """Preserve method transparency when an LLM-configured item falls back locally."""

    if requested_method == "llm" and criterion.evaluation == "local":
        criterion.evaluation = "llm_fallback_local"
    return criterion


def score_item_with_llm(
    *,
    content_context: dict[str, Any],
    rubric_item: RubricItem,
    rubric_filename: str,
    model_config: ModelConfig,
) -> ScoreCriterion:
    """Score a single rubric item with the LLM."""

    prompt = build_score_prompt(
        content_context=content_context,
        rubric_items=[rubric_item],
        rubric_filename=rubric_filename,
    )
    model = create_chat_model(model_config)
    response = invoke_chat_model_with_retry(
        model,
        [
            SystemMessage(
                content=(
                    "你是一名严谨的中文论文评分老师。"
                    "你只负责单个评分项打分，不负责格式检测。"
                    "请仅根据提供的内容证据和评分标准，给出该项分数。"
                    "必须只输出纯 JSON，不要 Markdown，不要解释。"
                )
            ),
            HumanMessage(content=prompt),
        ],
    )
    payload = parse_llm_json_response(response)
    criteria = normalize_llm_score_criteria(payload, [rubric_item])
    return criteria[0]


def score_item_locally(
    *,
    document: ThesisDocument,
    topic_analysis: dict[str, Any],
    format_issues: list[Issue],
    writing_issues: list[Issue],
    keywords: list[str],
    technology_details: list[TechnologyStackItem],
    format_requirements: dict[str, Any] | None,
    rubric_item: RubricItem,
    rubric_filename: str,
) -> ScoreCriterion:
    """Score a single rubric item using local heuristics.

    Dispatch is driven by the stable rubric key (see ``RubricItem.key``) so it
    never depends on a Chinese label that may drift between rubric versions.
    """

    key = rubric_item.key or rubric_item.name
    if key == "topic_workload":
        return score_topic_workload(
            document, topic_analysis, technology_details, rubric_item
        )
    if key == "research_argument":
        return score_research_argument(document, keywords, rubric_item)
    if key == "translation":
        return score_translation(document, rubric_item)
    if key == "experiment_analysis":
        return score_experiment_analysis(document, technology_details, rubric_item)
    if key == "writing_quality":
        return score_writing_quality(document, writing_issues, rubric_item)
    if key == "innovation":
        return score_innovation(document, technology_details, rubric_item)
    if key == FORMAT_RUBRIC_KEY or rubric_item.name == "格式规范":
        format_filename = FORMAT_RUBRIC_BY_SCORE_RUBRIC.get(rubric_filename)
        if format_filename is None:
            return build_criterion(
                key=FORMAT_RUBRIC_KEY,
                rubric_item=rubric_item,
                score=0,
                evidence=["该评分预设未配置内置格式规范文件"],
                deductions=["缺少格式规范文件映射"],
                suggestions=["为该预设补充格式规范文件"],
            )
        return score_format_compliance(
            document=document,
            format_issues=format_issues,
            format_requirements=format_requirements,
            rubric_item=rubric_item,
            format_filename=format_filename,
        )
    if key in OWNED_IOT_ITEM_NAMES or rubric_item.name in OWNED_IOT_ITEM_NAMES:
        iot_criterion = score_iot_item_locally(
            document=document,
            format_issues=format_issues,
            writing_issues=writing_issues,
            technology_details=technology_details,
            format_requirements=format_requirements,
            rubric_item=rubric_item,
        )
        if iot_criterion is not None:
            return iot_criterion
    return build_criterion(
        key=key,
        rubric_item=rubric_item,
        score=0,
        evidence=["未实现本地评分逻辑"],
        deductions=["缺少本地评分映射"],
        suggestions=["补充该评分项的本地规则"],
    )


def build_score_report(
    *, criteria: list[ScoreCriterion], rubric_source: str, score_source: str
) -> ScoreReport:
    """Build a normalized score report from criterion scores."""

    raw_score = round(sum(item.score for item in criteria), 2)
    raw_total = round(sum(item.max_score for item in criteria), 2)
    score = round(raw_score / max(raw_total, 1) * 100)
    return ScoreReport(
        score=max(0, min(100, score)),
        raw_score=raw_score,
        raw_total=raw_total,
        criteria=criteria,
        rubric_source=rubric_source,
        score_source=score_source,
    )


def determine_score_source(criteria: list[ScoreCriterion]) -> str:
    """Summarize how the final score was produced."""

    sources = {item.evaluation or "local" for item in criteria}
    if sources == {"llm"}:
        return "llm"
    if sources <= {"local"}:
        return "local"
    return "mixed"


def build_content_context(
    *,
    document: ThesisDocument,
    topic_analysis: dict[str, Any],
    keywords: list[str],
    technology_details: list[TechnologyStackItem],
) -> dict[str, Any]:
    """Build bounded content-only evidence for LLM scoring."""

    tech_summary = [
        {
            "name": item.name,
            "category": item.category,
            "matched_terms": item.matched_terms,
        }
        for item in technology_details
    ]
    section_summary = [
        {
            "identifier": section.identifier,
            "level": section.level,
            "title": section.title,
            "word_count": section.word_count,
            "excerpt": truncate_context_text(section.content, limit=600),
        }
        for section in document.sections[:20]
        if not section.is_mermaid_code
    ]
    return {
        "title": document.title,
        "source_type": document.source_type,
        "total_word_count": document.total_word_count,
        "section_count": len(document.sections),
        "abstract": truncate_context_text(document.abstract, limit=1200),
        "topic_analysis": {
            "document_ratio": topic_analysis.get("document_ratio", 0),
            "topic_keywords": topic_analysis.get("topic_keywords", []),
        },
        "keywords": keywords,
        "technology_details": tech_summary,
        "sections": section_summary,
    }


def truncate_context_text(text: str, *, limit: int) -> str:
    """Keep prompt evidence bounded while preserving readable excerpts."""

    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


def build_score_prompt(
    *,
    content_context: dict[str, Any],
    rubric_items: list[RubricItem],
    rubric_filename: str,
) -> str:
    """Build an LLM prompt that exposes content evidence only."""

    rubric_summary = [
        {"name": item.name, "score": item.max_score, "standards": item.standards}
        for item in rubric_items
    ]
    payload = {
        "content_context": content_context,
        "rubric_source": rubric_filename,
        "rubric": rubric_summary,
    }
    return (
        "请根据以下论文信息，为评分标准中列出的每一项打分，并输出严格 JSON。\n"
        "JSON 结构必须为：\n"
        "{"
        '"criteria":[{"key":"选题及工作量","name":"选题及工作量","score":0,"max_score":20,'
        '"evidence":["..."],"deductions":["..."],"suggestions":["..."]}...],'
        '"raw_score":0,"raw_total":0,"score":0'
        "}\n"
        "要求：\n"
        "1. criteria 必须覆盖 rubric 中的全部项目。\n"
        "2. score 为百分制总分，raw_score 为各项原始分总和，raw_total 为各项满分总和。\n"
        "3. 评分标准和评价方法必须来自 rubric_source 对应的配置文件。\n"
        "4. 评分必须参考评分标准，但分数由你综合判断。\n"
        "5. 证据、扣分原因、建议都要简洁具体；"
        "任何低于满分的评分项，deductions 必须给出具体扣分理由，不能留空。\n"
        "6. 只能依据 content_context 中的内容证据评分；"
        "不得推测或评价格式、标点和口语化表达。\n"
        f"内容证据：{json.dumps(payload, ensure_ascii=False)}"
    )


def parse_llm_json_response(response: Any) -> dict[str, Any]:
    """Parse JSON from an LLM response."""

    text = extract_response_text(response).strip()
    if not text:
        raise ValueError("empty llm response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def normalize_llm_score_criteria(
    payload: dict[str, Any], rubric_items: list[RubricItem]
) -> list[ScoreCriterion]:
    """Normalize LLM score payload into score criteria."""

    rubric_by_name = {item.name: item for item in rubric_items}
    item_by_key = {item.name: item.name for item in rubric_items}
    criteria_payload = payload.get("criteria", [])
    if not isinstance(criteria_payload, list):
        raise ValueError("llm score payload criteria must be a list")

    criteria: list[ScoreCriterion] = []
    for entry in criteria_payload:
        if not isinstance(entry, dict):
            raise ValueError("llm score payload criteria entry must be an object")
        name = str(entry.get("name") or entry.get("key") or "").strip()
        if name not in rubric_by_name:
            raise ValueError(f"unknown rubric criterion: {name}")
        rubric_item = rubric_by_name[name]
        raw_score = parse_score_value(entry.get("score", 0))
        clamped_score = round(min(max(raw_score, 0.0), rubric_item.max_score), 2)
        evidence = parse_string_list(entry.get("evidence", []))
        if clamped_score != round(raw_score, 2):
            logger.warning(
                "llm returned out-of-range score for %s: %.2f not in [0, %s]; clamped to %.2f",
                rubric_item.name,
                raw_score,
                rubric_item.max_score,
                clamped_score,
            )
            evidence.append(
                f"LLM 返回分数 {round(raw_score, 2)} 超出 [0, {rubric_item.max_score}]，"
                f"已修正为 {clamped_score}"
            )
        score = clamped_score
        deductions = parse_string_list(entry.get("deductions", []))
        validate_llm_deductions(
            criterion_name=rubric_item.name,
            score=score,
            max_score=rubric_item.max_score,
            deductions=deductions,
        )
        criteria.append(
            ScoreCriterion(
                key=str(entry.get("key") or item_by_key[name]),
                name=rubric_item.name,
                score=score,
                max_score=rubric_item.max_score,
                standards=rubric_item.standards,
                evaluation="llm",
                evidence=evidence,
                deductions=deductions,
                suggestions=parse_string_list(entry.get("suggestions", [])),
            )
        )
    if len(criteria) != len(rubric_items):
        raise ValueError("llm score payload criteria count does not match rubric items")
    return criteria


def parse_string_list(value: Any) -> list[str]:
    """Parse a JSON value into a compact string list."""

    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def validate_llm_deductions(
    *, criterion_name: str, score: float, max_score: float, deductions: list[str]
) -> None:
    """Require LLM scoring to explain every non-full score."""

    if score < max_score and not deductions:
        msg = f"llm score criterion {criterion_name} is below max but deductions are empty"
        raise ValueError(msg)


def resolve_rubric_items(
    *, rubric: dict[str, Any] | None, rubric_filename: str
) -> tuple[list[RubricItem], str]:
    """Resolve rubric items from upload metadata or bundled config."""

    default_items = append_builtin_format_rubric_item(
        normalize_rubric_items(
            normalize_rubric_payload(load_json_resource(rubric_filename))
        ),
        rubric_filename=rubric_filename,
    )
    if rubric and rubric.get("items"):
        return merge_rubric_items(
            default_items, normalize_rubric_items(rubric["items"], require_all=False)
        ), str(rubric.get("source_name") or "uploaded_rubric.json")
    return default_items, rubric_filename


def append_builtin_format_rubric_item(
    items: list[RubricItem], *, rubric_filename: str
) -> list[RubricItem]:
    """Append built-in format scoring derived from the paired format JSON."""

    if any(item.name == "格式规范" for item in items):
        return items
    format_filename = FORMAT_RUBRIC_BY_SCORE_RUBRIC.get(rubric_filename)
    if not format_filename:
        return items
    return [*items, build_format_rubric_item(format_filename)]


def build_format_rubric_item(format_filename: str) -> RubricItem:
    """Build a local format rubric item from a bundled format specification."""

    format_spec = normalize_format_spec_payload(load_json_resource(format_filename))
    rules = extract_format_rules(format_spec)
    if not rules:
        msg = f"format rubric rules are empty: {format_filename}"
        raise ValueError(msg)
    return RubricItem(
        name="格式规范",
        standards=build_format_standards(rules),
        evaluation="local",
        max_score=sum_format_rule_points(rules),
        key=FORMAT_RUBRIC_KEY,
    )


def sum_format_rule_points(rules: list[dict[str, Any]]) -> float:
    """Calculate the format full score from rule points and section weights."""

    total = sum(
        parse_float_value(rule.get("points", 0))
        * parse_float_value(rule.get("section_weight", 1))
        for rule in rules
    )
    return round(total, 4)


def build_format_standards(rules: list[dict[str, Any]]) -> list[str]:
    """Create compact human-readable standards from structured format rules."""

    standards: list[str] = []
    for rule in rules:
        label = str(rule.get("label") or rule.get("id") or "").strip()
        expected = format_expected_value(get_rule_expected(rule))
        if label and expected:
            standards.append(f"{label}: {expected}")
        elif label:
            standards.append(label)
    return standards
