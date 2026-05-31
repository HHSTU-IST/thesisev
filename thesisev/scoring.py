"""Rule-based scoring for thesis evaluation rubrics."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from thesisev.llm import ModelConfig, create_chat_model
from thesisev.models import Issue, TechnologyStackItem, ThesisDocument
from thesisev.resources import load_json_resource

DEFAULT_THESIS_TECH_RUBRIC = "score_thesis_tech.json"
FORMAT_RUBRIC_BY_SCORE_RUBRIC = {
    "score_report_iot.json": "score_report_iot_f.json",
}
REQUIRED_CRITERIA = (
    "选题及工作量",
    "调查论证",
    "译文",
    "实验方案、分析与技能",
    "论文质量",
    "创新",
)


@dataclass(slots=True)
class ScoreCriterion:
    """Per-criterion score with evidence and improvement hints."""

    key: str
    name: str
    score: float
    max_score: float
    standards: list[str] = field(default_factory=list)
    evaluation: str = ""
    evidence: list[str] = field(default_factory=list)
    deductions: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the criterion to a JSON-friendly dictionary."""

        return asdict(self)


@dataclass(slots=True)
class ScoreReport:
    """Normalized score report derived from rubric criteria."""

    score: int
    raw_score: float
    raw_total: float
    criteria: list[ScoreCriterion]
    rubric_source: str
    score_source: str

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


@dataclass(slots=True)
class RubricItem:
    """Normalized rubric item loaded from JSON or upload metadata."""

    name: str
    standards: list[str]
    evaluation: str
    max_score: float


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


def calculate_score_report_with_llm(
    *,
    content_context: dict[str, Any],
    rubric_items: list[RubricItem],
    rubric_source: str,
    rubric_filename: str,
    model_config: ModelConfig,
) -> ScoreReport:
    """Calculate rubric scores by asking an LLM for each criterion."""

    prompt = build_score_prompt(
        content_context=content_context,
        rubric_items=rubric_items,
        rubric_filename=rubric_filename,
    )
    model = create_chat_model(model_config)
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "你是一名严谨的中文论文评分老师。"
                    "你只负责六项评分标准打分，不负责格式检测。"
                    "请仅根据提供的内容证据和评分标准，给出六项标准的分数。"
                    "必须只输出纯 JSON，不要 Markdown，不要解释。"
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    payload = parse_llm_json_response(response)
    criteria = normalize_llm_score_criteria(payload, rubric_items)
    return build_score_report(
        criteria=criteria,
        rubric_source=rubric_source,
        score_source="llm",
    )


def calculate_score_report_local(
    *,
    document: ThesisDocument,
    topic_analysis: dict[str, Any],
    writing_issues: list[Issue],
    keywords: list[str],
    technology_details: list[TechnologyStackItem],
    rubric_items: list[RubricItem],
    rubric_source: str,
) -> ScoreReport:
    """Fallback deterministic score calculation."""

    item_by_name = {item.name: item for item in rubric_items}
    criteria = [
        score_topic_workload(
            document,
            topic_analysis,
            technology_details,
            item_by_name["选题及工作量"],
        ),
        score_research_argument(document, keywords, item_by_name["调查论证"]),
        score_translation(document, item_by_name["译文"]),
        score_experiment_analysis(
            document,
            technology_details,
            item_by_name["实验方案、分析与技能"],
        ),
        score_writing_quality(
            document,
            writing_issues,
            item_by_name["论文质量"],
        ),
        score_innovation(document, technology_details, item_by_name["创新"]),
    ]
    return build_score_report(
        criteria=criteria,
        rubric_source=rubric_source,
        score_source="local",
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

    item_by_name = {item.name: item for item in rubric_items}
    criteria: list[ScoreCriterion] = []
    for item in rubric_items:
        method = item.evaluation.strip().lower()
        if method == "llm" and model_config is not None and model_config.is_available():
            criteria.append(
                score_item_with_llm(
                    content_context=content_context,
                    rubric_item=item,
                    rubric_filename=rubric_filename,
                    model_config=model_config,
                )
            )
            continue
        criteria.append(
            mark_llm_fallback_if_needed(
                score_item_locally(
                    document=document,
                    topic_analysis=topic_analysis,
                    format_issues=format_issues,
                    writing_issues=writing_issues,
                    keywords=keywords,
                    technology_details=technology_details,
                    format_requirements=format_requirements,
                    rubric_item=item,
                    item_by_name=item_by_name,
                ),
                requested_method=method,
            )
        )
    return criteria


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
    response = model.invoke(
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
        ]
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
    item_by_name: dict[str, RubricItem],
) -> ScoreCriterion:
    """Score a single rubric item using local heuristics."""

    if rubric_item.name == "选题及工作量":
        return score_topic_workload(
            document, topic_analysis, technology_details, rubric_item
        )
    if rubric_item.name == "调查论证":
        return score_research_argument(document, keywords, rubric_item)
    if rubric_item.name == "译文":
        return score_translation(document, rubric_item)
    if rubric_item.name == "实验方案、分析与技能":
        return score_experiment_analysis(document, technology_details, rubric_item)
    if rubric_item.name == "论文质量":
        return score_writing_quality(document, writing_issues, rubric_item)
    if rubric_item.name == "创新":
        return score_innovation(document, technology_details, rubric_item)
    from thesisev.scoring_iot import score_iot_item_locally

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
        key=normalize_criterion_name(rubric_item.name),
        rubric_item=rubric_item,
        score=0,
        evidence=["未实现本地评分逻辑"],
        deductions=["缺少本地评分映射"],
        suggestions=["补充该评分项的本地规则"],
    )


def build_score_report(
    *,
    criteria: list[ScoreCriterion],
    rubric_source: str,
    score_source: str,
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
        {
            "name": item.name,
            "score": item.max_score,
            "standards": item.standards,
        }
        for item in rubric_items
    ]
    payload = {
        "content_context": content_context,
        "rubric_source": rubric_filename,
        "rubric": rubric_summary,
    }
    return (
        "请根据以下论文信息，为六项评分标准分别打分，并输出严格 JSON。\n"
        "JSON 结构必须为：\n"
        "{"
        '"criteria":[{"key":"选题及工作量","name":"选题及工作量","score":0,"max_score":20,'
        '"evidence":["..."],"deductions":["..."],"suggestions":["..."]}...],'
        '"raw_score":0,"raw_total":0,"score":0'
        "}\n"
        "要求：\n"
        "1. 六项 criteria 必须全部返回。\n"
        "2. score 为百分制总分，raw_score 为六项原始分总和，raw_total 为六项满分总和。\n"
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
        score = round(parse_score_value(entry.get("score", 0)), 2)
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
                evidence=parse_string_list(entry.get("evidence", [])),
                deductions=deductions,
                suggestions=parse_string_list(entry.get("suggestions", [])),
            )
        )
    if len(criteria) != len(rubric_items):
        raise ValueError("llm score payload must include six criteria")
    return criteria


def parse_string_list(value: Any) -> list[str]:
    """Parse a JSON value into a compact string list."""

    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def validate_llm_deductions(
    *,
    criterion_name: str,
    score: float,
    max_score: float,
    deductions: list[str],
) -> None:
    """Require LLM scoring to explain every non-full score."""

    if score < max_score and not deductions:
        msg = f"llm score criterion {criterion_name} is below max but deductions are empty"
        raise ValueError(msg)


def deduplicate_preserving_order(values: list[str]) -> list[str]:
    """Return a list with duplicates removed while keeping first-seen order."""

    seen: set[str] = set()
    deduplicated: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return deduplicated


def score_topic_workload(
    document: ThesisDocument,
    topic_analysis: dict[str, Any],
    technology_details: list[TechnologyStackItem],
    rubric_item: RubricItem,
) -> ScoreCriterion:
    """Score topic value, workload, and overall completeness."""

    from thesisev.scoring_content import score_topic_workload as impl

    return impl(document, topic_analysis, technology_details, rubric_item)


def score_research_argument(
    document: ThesisDocument,
    keywords: list[str],
    rubric_item: RubricItem,
) -> ScoreCriterion:
    """Score literature use and argumentation signals."""

    from thesisev.scoring_content import score_research_argument as impl

    return impl(document, keywords, rubric_item)


def score_translation(
    document: ThesisDocument,
    rubric_item: RubricItem,
) -> ScoreCriterion:
    """Score Chinese-English abstract translation completeness."""

    from thesisev.scoring_content import score_translation as impl

    return impl(document, rubric_item)


def score_experiment_analysis(
    document: ThesisDocument,
    technology_details: list[TechnologyStackItem],
    rubric_item: RubricItem,
) -> ScoreCriterion:
    """Score design, data, analysis, feasibility, and benefit signals."""

    from thesisev.scoring_content import score_experiment_analysis as impl

    return impl(document, technology_details, rubric_item)


def score_writing_quality(
    document: ThesisDocument,
    writing_issues: list[Issue],
    rubric_item: RubricItem,
) -> ScoreCriterion:
    """Score writing quality from locally detected writing issues."""

    from thesisev.scoring_content import score_writing_quality as impl

    return impl(document, writing_issues, rubric_item)


def score_innovation(
    document: ThesisDocument,
    technology_details: list[TechnologyStackItem],
    rubric_item: RubricItem,
) -> ScoreCriterion:
    """Score innovation and application-value signals."""

    from thesisev.scoring_content import score_innovation as impl

    return impl(document, technology_details, rubric_item)


def extract_format_rules(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import extract_format_rules as impl

    return impl(*args, **kwargs)


def summarize_format_spec(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import summarize_format_spec as impl

    return impl(*args, **kwargs)


def score_format_rules(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import score_format_rules as impl

    return impl(*args, **kwargs)


def build_format_suggestion(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import build_format_suggestion as impl

    return impl(*args, **kwargs)


def format_expected_value(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import format_expected_value as impl

    return impl(*args, **kwargs)


def format_expected_items(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import format_expected_items as impl

    return impl(*args, **kwargs)


def evaluate_docx_expected_rule(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import evaluate_docx_expected_rule as impl

    return impl(*args, **kwargs)


def build_rule_suggestion(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import build_rule_suggestion as impl

    return impl(*args, **kwargs)


def select_docx_target_snapshot(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import select_docx_target_snapshot as impl

    return impl(*args, **kwargs)


def select_matching_paragraph(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import select_matching_paragraph as impl

    return impl(*args, **kwargs)


def lookup_docx_snapshot_value(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import lookup_docx_snapshot_value as impl

    return impl(*args, **kwargs)


def resolve_from_first_paragraph(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import resolve_from_first_paragraph as impl

    return impl(*args, **kwargs)


def resolve_from_first_run(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import resolve_from_first_run as impl

    return impl(*args, **kwargs)


def resolve_from_first_table(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import resolve_from_first_table as impl

    return impl(*args, **kwargs)


def resolve_from_first_section(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import resolve_from_first_section as impl

    return impl(*args, **kwargs)


def compare_docx_expected_value(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import compare_docx_expected_value as impl

    return impl(*args, **kwargs)


def compare_numeric_value(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import compare_numeric_value as impl

    return impl(*args, **kwargs)


def compare_length_value(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import compare_length_value as impl

    return impl(*args, **kwargs)


def parse_flexible_numeric_value(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import parse_flexible_numeric_value as impl

    return impl(*args, **kwargs)


def parse_length_to_points(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import parse_length_to_points as impl

    return impl(*args, **kwargs)


def normalize_docx_expected_token(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import normalize_docx_expected_token as impl

    return impl(*args, **kwargs)


def format_docx_expected_scalar(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import format_docx_expected_scalar as impl

    return impl(*args, **kwargs)


def normalize_string_list(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import normalize_string_list as impl

    return impl(*args, **kwargs)


def parse_optional_float(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import parse_optional_float as impl

    return impl(*args, **kwargs)


def normalize_rule_check(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import normalize_rule_check as impl

    return impl(*args, **kwargs)


def get_rule_expected(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import get_rule_expected as impl

    return impl(*args, **kwargs)


def parse_float_value(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import parse_float_value as impl

    return impl(*args, **kwargs)


def normalize_format_spec_payload(*args, **kwargs):
    """Compatibility wrapper for :mod:`thesisev.scoring_format`."""

    from thesisev.scoring_format import normalize_format_spec_payload as impl

    return impl(*args, **kwargs)


def resolve_rubric_items(
    *,
    rubric: dict[str, Any] | None,
    rubric_filename: str,
) -> tuple[list[RubricItem], str]:
    """Resolve rubric items from upload metadata or bundled config."""

    default_items = append_builtin_format_rubric_item(
        normalize_rubric_payload(load_json_resource(rubric_filename)),
        rubric_filename=rubric_filename,
    )
    if rubric and rubric.get("items"):
        return merge_rubric_items(
            default_items,
            normalize_rubric_items(rubric["items"], require_all=False),
        ), rubric.get("source_name", "uploaded_rubric.json")
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


def normalize_rubric_payload(payload: Any) -> list[RubricItem]:
    """Normalize a bundled rubric JSON payload."""

    if not isinstance(payload, dict):
        msg = "score rubric JSON must be an object"
        raise ValueError(msg)
    return normalize_rubric_items(
        [
            {"criterion": criterion, **parse_rubric_value(value)}
            for criterion, value in payload.items()
        ]
    )


def parse_rubric_item_payload(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single rubric item payload."""

    return {
        "criterion": item.get("criterion", ""),
        "score": item.get("score", 0),
        "standard": item.get("standard", item.get("standards", [])),
    }


def normalize_rubric_items(
    items: list[dict[str, Any]], *, require_all: bool = True
) -> list[RubricItem]:
    """Normalize rubric item dictionaries preserving configured order."""

    normalized = [
        RubricItem(
            name=normalize_criterion_name(item.get("criterion", "")),
            standards=parse_standards(item.get("standard", item.get("standards", []))),
            evaluation=str(item.get("evaluation") or "llm").strip().lower(),
            max_score=parse_score_value(item.get("score", 0)),
        )
        for item in items
    ]
    if require_all and not normalized:
        msg = "score rubric must contain at least one criterion"
        raise ValueError(msg)
    return normalized


def merge_rubric_items(
    default_items: list[RubricItem], uploaded_items: list[RubricItem]
) -> list[RubricItem]:
    """Overlay uploaded rubric scores onto default criteria."""

    item_by_name = {item.name: item for item in default_items}
    for item in uploaded_items:
        if item.name in item_by_name:
            item_by_name[item.name] = item
    return [item_by_name[item.name] for item in default_items]


def parse_rubric_value(value: Any) -> dict[str, Any]:
    """Parse either flat score values or nested standard-score objects."""

    if isinstance(value, int | float) and not isinstance(value, bool):
        return {"score": float(value), "standard": []}
    if isinstance(value, dict):
        score = value.get("score", value.get("分数"))
        standard = value.get("standard", value.get("standards", value.get("标准", [])))
        return {
            "score": parse_score_value(score),
            "standard": parse_standards(standard),
        }
    msg = "score rubric item must be numeric or an object with score"
    raise ValueError(msg)


def parse_score_value(value: Any) -> float:
    """Parse a numeric score value."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = "rubric score must be numeric"
        raise ValueError(msg)
    return float(value)


def parse_standards(value: Any) -> list[str]:
    """Parse rubric standard descriptions."""

    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def normalize_criterion_name(criterion: Any) -> str:
    """Normalize criterion labels to their top-level rubric name."""

    if not isinstance(criterion, str) or not criterion.strip():
        msg = "rubric criterion must be a non-empty string"
        raise ValueError(msg)
    return re.split(r"[：:]", criterion.strip(), maxsplit=1)[0].strip()


def build_criterion(
    *,
    key: str,
    rubric_item: RubricItem,
    score: float,
    evidence: list[str],
    deductions: list[str],
    suggestions: list[str],
) -> ScoreCriterion:
    """Clamp and round a criterion score."""

    clamped_score = round(max(0.0, min(rubric_item.max_score, score)), 2)
    return ScoreCriterion(
        key=key,
        name=rubric_item.name,
        score=clamped_score,
        max_score=rubric_item.max_score,
        standards=rubric_item.standards,
        evaluation="local",
        evidence=evidence,
        deductions=ensure_deduction_visibility(
            score=clamped_score,
            max_score=rubric_item.max_score,
            deductions=deductions,
        ),
        suggestions=suggestions,
    )


def ensure_deduction_visibility(
    *, score: float, max_score: float, deductions: list[str]
) -> list[str]:
    """Ensure non-full scores always expose a visible deduction reason."""

    if deductions or score >= max_score:
        return deductions
    lost_points = round(max_score - score, 2)
    return [f"未达到满分，扣 {lost_points:g} 分"]


def ratio_from_thresholds(
    value: float, thresholds: tuple[tuple[float, float], ...]
) -> float:
    """Return the first ratio whose threshold is met."""

    from thesisev.scoring_content import ratio_from_thresholds as impl

    return impl(value, thresholds)


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return whether any term appears in text."""

    from thesisev.scoring_content import has_any as impl

    return impl(text, terms)


def count_terms(text: str, terms: tuple[str, ...]) -> int:
    """Count simple literal term occurrences."""

    from thesisev.scoring_content import count_terms as impl

    return impl(text, terms)


def count_citations(text: str) -> int:
    """Count Arabic-numbered citation markers in Chinese technical papers."""

    from thesisev.scoring_content import count_citations as impl

    return impl(text)
