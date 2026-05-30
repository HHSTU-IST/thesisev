"""Rule-based scoring for thesis evaluation rubrics."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from thesisev.llm import ModelConfig, create_chat_model
from thesisev.models import Issue, TechnologyStackItem, ThesisDocument
from thesisev.resources import load_json_resource

DEFAULT_THESIS_TECH_RUBRIC = "score_thesis_tech.json"
DEFAULT_IOT_FORMAT_RUBRIC = "score_report_iot_f.json"
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
    issues: list[Issue],
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
        issues=issues,
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
    document: ThesisDocument,
    topic_analysis: dict[str, Any],
    issues: list[Issue],
    keywords: list[str],
    technology_details: list[TechnologyStackItem],
    format_requirements: dict[str, Any] | None,
    rubric_items: list[RubricItem],
    rubric_source: str,
    rubric_filename: str,
    model_config: ModelConfig,
) -> ScoreReport:
    """Calculate rubric scores by asking an LLM for each criterion."""

    prompt = build_score_prompt(
        document=document,
        topic_analysis=topic_analysis,
        issues=issues,
        keywords=keywords,
        technology_details=technology_details,
        format_requirements=format_requirements,
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
                    "请根据提供的论文信息、本地检测结果和评分标准，给出六项标准的分数。"
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
    issues: list[Issue],
    keywords: list[str],
    technology_details: list[TechnologyStackItem],
    format_requirements: dict[str, Any] | None,
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
            issues,
            format_requirements,
            item_by_name["论文质量"],
        ),
        score_innovation(document, technology_details, item_by_name["创新"]),
    ]
    return build_score_report(
        criteria=criteria,
        rubric_source=rubric_source,
        score_source="local_program",
    )


def score_rubric_items(
    *,
    document: ThesisDocument,
    topic_analysis: dict[str, Any],
    issues: list[Issue],
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
                    document=document,
                    topic_analysis=topic_analysis,
                    issues=issues,
                    keywords=keywords,
                    technology_details=technology_details,
                    format_requirements=format_requirements,
                    rubric_item=item,
                    rubric_source=rubric_source,
                    rubric_filename=rubric_filename,
                    model_config=model_config,
                )
            )
            continue
        criteria.append(
            score_item_locally(
                document=document,
                topic_analysis=topic_analysis,
                issues=issues,
                keywords=keywords,
                technology_details=technology_details,
                format_requirements=format_requirements,
                rubric_item=item,
                item_by_name=item_by_name,
            )
        )
    return criteria


def score_item_with_llm(
    *,
    document: ThesisDocument,
    topic_analysis: dict[str, Any],
    issues: list[Issue],
    keywords: list[str],
    technology_details: list[TechnologyStackItem],
    format_requirements: dict[str, Any] | None,
    rubric_item: RubricItem,
    rubric_source: str,
    rubric_filename: str,
    model_config: ModelConfig,
) -> ScoreCriterion:
    """Score a single rubric item with the LLM."""

    prompt = build_score_prompt(
        document=document,
        topic_analysis=topic_analysis,
        issues=issues,
        keywords=keywords,
        technology_details=technology_details,
        format_requirements=format_requirements,
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
                    "请根据提供的论文信息、本地检测结果和评分标准，给出该项分数。"
                    "必须只输出纯 JSON，不要 Markdown，不要解释。"
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    payload = parse_llm_json_response(response)
    criteria = normalize_llm_score_criteria(payload, [rubric_item])
    criterion = criteria[0]
    criterion.source = "llm"
    return criterion


def score_item_locally(
    *,
    document: ThesisDocument,
    topic_analysis: dict[str, Any],
    issues: list[Issue],
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
        return score_writing_quality(document, issues, format_requirements, rubric_item)
    if rubric_item.name == "创新":
        return score_innovation(document, technology_details, rubric_item)
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
        return score_iot_writing(document, issues, rubric_item)
    if rubric_item.name == "格式规范":
        return score_iot_format(document, issues, format_requirements, rubric_item)
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

    sources = {item.source or item.evaluation or "local_program" for item in criteria}
    if sources == {"llm"}:
        return "llm"
    if sources <= {"local", "local_program"}:
        return "local_program"
    return "mixed"


def build_score_prompt(
    *,
    document: ThesisDocument,
    topic_analysis: dict[str, Any],
    issues: list[Issue],
    keywords: list[str],
    technology_details: list[TechnologyStackItem],
    format_requirements: dict[str, Any] | None,
    rubric_items: list[RubricItem],
    rubric_filename: str,
) -> str:
    """Build an LLM prompt for rubric scoring."""

    issue_summary = summarize_issues(issues)
    tech_summary = [
        {
            "name": item.name,
            "category": item.category,
            "matched_terms": item.matched_terms,
        }
        for item in technology_details
    ]
    rubric_summary = [
        {
            "name": item.name,
            "score": item.max_score,
            "standards": item.standards,
        }
        for item in rubric_items
    ]
    payload = {
        "title": document.title,
        "source_type": document.source_type,
        "total_word_count": document.total_word_count,
        "section_count": len(document.sections),
        "topic_analysis": {
            "document_ratio": topic_analysis.get("document_ratio", 0),
            "topic_keywords": topic_analysis.get("topic_keywords", []),
        },
        "keywords": keywords,
        "technology_details": tech_summary,
        "issues": issue_summary,
        "format_requirements": format_requirements,
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
        "5. 证据、扣分原因、建议都要简洁具体。\n"
        f"论文信息：{json.dumps(payload, ensure_ascii=False)}"
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
        criteria.append(
            ScoreCriterion(
                key=str(entry.get("key") or item_by_key[name]),
                name=rubric_item.name,
                score=round(parse_score_value(entry.get("score", 0)), 2),
                max_score=rubric_item.max_score,
                standards=rubric_item.standards,
                evaluation=rubric_item.evaluation,
                evidence=parse_string_list(entry.get("evidence", [])),
                deductions=parse_string_list(entry.get("deductions", [])),
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


def summarize_issues(issues: list[Issue]) -> list[dict[str, Any]]:
    """Summarize local formatting issues for the LLM prompt."""

    counts = Counter(issue.category for issue in issues)
    samples = defaultdict(list)
    for issue in issues[:20]:
        samples[issue.category].append(
            {
                "message": issue.message,
                "suggestion": issue.suggestion,
                "section": issue.section_title,
            }
        )
    return [
        {
            "category": category,
            "count": count,
            "samples": samples.get(category, []),
        }
        for category, count in counts.items()
    ]


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
    document: ThesisDocument,
    keywords: list[str],
    rubric_item: RubricItem,
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
    document: ThesisDocument,
    rubric_item: RubricItem,
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
    document: ThesisDocument,
    issues: list[Issue],
    format_requirements: dict[str, Any] | None,
    rubric_item: RubricItem,
) -> ScoreCriterion:
    """Score writing quality and formatting from detected issues."""

    max_score = rubric_item.max_score
    severity_weights = {"low": 0.35, "medium": 0.65, "high": 1.0}
    deduction = sum(severity_weights.get(issue.severity, 0.8) for issue in issues)
    if len(document.sections) <= 1:
        deduction += 1.2
    if not document.abstract:
        deduction += 0.8
    score = max_score - min(max_score * 0.8, deduction)

    format_count = (
        len(format_requirements.get("items", [])) if format_requirements else 0
    )
    evidence = [
        f"检测问题 {len(issues)} 项",
        f"章节数 {len(document.sections)}",
        f"格式要求条目 {format_count}",
    ]
    deductions: list[str] = []
    suggestions: list[str] = []
    if issues:
        deductions.append("存在标点或口语化表达问题")
        suggestions.append("统一技术论文书面表达和中英文标点规范")
    if len(document.sections) <= 1:
        deductions.append("章节结构不稳定，影响条理性判断")
    if not document.abstract:
        deductions.append("未识别到摘要")
    if format_requirements:
        evidence.append("已读取上传格式要求，可用于人工复核")

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


def score_iot_background(
    document: ThesisDocument, rubric_item: RubricItem
) -> ScoreCriterion:
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


def score_iot_method(
    document: ThesisDocument, rubric_item: RubricItem
) -> ScoreCriterion:
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
    rubric_item: RubricItem,
) -> ScoreCriterion:
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
    rubric_item: RubricItem,
) -> ScoreCriterion:
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


def score_iot_cost(document: ThesisDocument, rubric_item: RubricItem) -> ScoreCriterion:
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


def score_iot_outlook(
    document: ThesisDocument, rubric_item: RubricItem
) -> ScoreCriterion:
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


def score_iot_word_count(
    document: ThesisDocument, rubric_item: RubricItem
) -> ScoreCriterion:
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
    document: ThesisDocument, issues: list[Issue], rubric_item: RubricItem
) -> ScoreCriterion:
    """Score report writing quality."""

    issue_penalty = sum(0.5 if issue.severity == "low" else 1.0 for issue in issues)
    score = max(
        0.0, rubric_item.max_score - min(rubric_item.max_score * 0.8, issue_penalty)
    )
    return build_criterion(
        key="iot_writing",
        rubric_item=rubric_item,
        score=score,
        evidence=[f"检测问题 {len(issues)} 项"],
        deductions=[] if not issues else ["存在格式或表达问题"],
        suggestions=[] if not issues else ["统一正文表达并修正标点/排版问题"],
    )


def score_iot_format(
    document: ThesisDocument,
    issues: list[Issue],
    format_requirements: dict[str, Any] | None,
    rubric_item: RubricItem,
) -> ScoreCriterion:
    """Score format compliance."""

    format_spec = load_json_resource(DEFAULT_IOT_FORMAT_RUBRIC)
    format_spec = normalize_format_spec_payload(format_spec)
    rules = extract_format_rules(format_spec)
    if not rules:
        raise ValueError("format rubric rules are empty")
    summary = score_format_rules(
        document=document,
        issues=issues,
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
    return build_criterion(
        key="iot_format",
        rubric_item=rubric_item,
        score=score,
        evidence=[f"格式要求条目 {format_count}", f"检测问题 {len(issues)} 项"]
        + summary["evidence"],
        deductions=summary["deductions"],
        suggestions=summary["suggestions"],
    )


def extract_format_rules(format_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract normalized format rules from the bundled format spec."""

    sections = format_spec.get("sections", [])
    if not isinstance(sections, list):
        raise ValueError("format rubric sections must be a list")

    rules: list[dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        section_name = str(section.get("name") or section.get("id") or "").strip()
        section_weight = parse_float_value(section.get("weight", 0))
        for rule in section.get("rules", []):
            if not isinstance(rule, dict):
                continue
            check = normalize_rule_check(rule)
            rules.append(
                {
                    "section": section_name,
                    "section_weight": section_weight,
                    "id": str(rule.get("id") or "").strip(),
                    "label": str(rule.get("label") or "").strip(),
                    "points": parse_float_value(rule.get("points", 1)),
                    "check": check,
                }
            )
    return rules


def summarize_format_spec(
    format_spec: dict[str, Any], *, source_name: str
) -> dict[str, Any]:
    """Convert the bundled format spec into UI-friendly summary data."""

    rules = extract_format_rules(format_spec)
    sections = format_spec.get("sections", [])
    section_summary = [
        {
            "label": str(section.get("name") or section.get("id") or "").strip(),
            "weight": parse_float_value(section.get("weight", 0)),
            "rule_count": len(section.get("rules", []))
            if isinstance(section, dict)
            else 0,
        }
        for section in sections
        if isinstance(section, dict)
    ]
    return {
        "source_name": source_name,
        "item_count": len(rules),
        "items": [
            {
                "label": f"{rule['section']} / {rule['label']}",
                "value": format_expected_value(get_rule_expected(rule)) or rule["id"],
            }
            for rule in rules
        ],
        "sections": section_summary,
    }


def score_format_rules(
    *,
    document: ThesisDocument,
    issues: list[Issue],
    rules: list[dict[str, Any]],
    format_requirements: dict[str, Any] | None,
) -> dict[str, Any]:
    """Score bundled format rules against document signals."""

    issue_categories = Counter(issue.category for issue in issues)
    evidence: list[str] = []
    deductions: list[str] = []
    suggestions: list[str] = []
    deduction = 0.0

    for rule in rules:
        signal = normalize_rule_check(rule)
        signal_type = str(signal.get("type", "")).strip()
        matched = False
        rule_evidence = ""
        suggestion = build_rule_suggestion(rule)

        if signal_type == "word_count_range":
            min_words = parse_optional_float(signal.get("min"))
            max_words = parse_optional_float(signal.get("max"))
            word_count = document.total_word_count
            rule_evidence = f"{rule['label']}: {word_count} 字"
            matched = (min_words is not None and word_count < min_words) or (
                max_words is not None and word_count > max_words
            )
        elif signal_type == "section_count_min":
            min_sections = parse_optional_float(signal.get("min")) or 0
            section_count = len(document.sections)
            rule_evidence = f"{rule['label']}: {section_count} 个章节"
            matched = section_count < min_sections
        elif signal_type == "text_contains_any":
            terms = normalize_string_list(signal.get("terms", []))
            matched_terms = [term for term in terms if term in document.cleaned_text]
            rule_evidence = f"{rule['label']}: {'，'.join(matched_terms) if matched_terms else '未命中'}"
            matched = not matched_terms
        elif signal_type == "issue_category":
            categories = normalize_string_list(signal.get("categories", []))
            hit_count = sum(
                issue_categories.get(category, 0) for category in categories
            )
            rule_evidence = f"{rule['label']}: {hit_count} 项问题"
            matched = hit_count > 0
        elif signal_type == "text_term_count_min":
            terms = normalize_string_list(signal.get("terms", []))
            min_hits = parse_optional_float(signal.get("min")) or 0
            hit_count = count_terms(document.cleaned_text, tuple(terms))
            rule_evidence = f"{rule['label']}: {hit_count} 次关键词命中"
            matched = hit_count < min_hits
        elif signal_type == "manual_review":
            rule_evidence = f"{rule['label']}: 需人工核对"
            matched = False
        elif signal_type == "docx_expected":
            passed, rule_evidence = evaluate_docx_expected_rule(document, rule)
            matched = not passed
        else:
            rule_evidence = f"{rule['label']}: 未定义检查类型，需人工核对"
            matched = False

        expected_lines = format_expected_items(get_rule_expected(rule))
        expected_text = "；".join(expected_lines)
        if expected_text:
            rule_evidence = f"{rule_evidence}；期望 {expected_text}"
        evidence.append(rule_evidence)
        if matched:
            deduction += parse_float_value(rule.get("points", 1)) * parse_float_value(
                rule.get("section_weight", 1)
            )
            deductions.append(f"{rule['label']} 不符合要求")
            if suggestion:
                suggestions.append(str(suggestion))

    if format_requirements:
        evidence.append("已读取内置格式要求，可用于人工复核")

    return {
        "deduction": deduction,
        "evidence": evidence,
        "deductions": deduplicate_preserving_order(deductions),
        "suggestions": deduplicate_preserving_order(suggestions),
    }


def build_format_suggestion(rule: dict[str, Any]) -> str:
    """Build a short suggestion for a failed format rule."""

    label = str(rule.get("label", "")).strip() or str(rule.get("id", "")).strip()
    expected = format_expected_value(get_rule_expected(rule))
    if expected:
        return f"对照规范检查{label}：{expected}"
    return f"对照规范检查{label}"


def format_expected_value(value: Any) -> str:
    """Format a rule expectation for display."""

    if value in (None, ""):
        return ""
    if isinstance(value, dict):
        parts = [f"{key}={value[key]}" for key in value]
        return ", ".join(parts)
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            parts = []
            for item in value:
                path = str(item.get("path", "")).strip()
                expected = item.get("value", "")
                if path:
                    parts.append(f"{path}={expected}")
            if parts:
                return "; ".join(parts)
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def format_expected_items(value: Any) -> list[str]:
    """Convert expected entries into display strings."""

    if value in (None, ""):
        return []
    if isinstance(value, list):
        items: list[str] = []
        for entry in value:
            if isinstance(entry, dict):
                path = str(entry.get("path", "")).strip()
                expected = entry.get("value", "")
                if path:
                    items.append(f"{path} = {expected}")
            else:
                text = str(entry).strip()
                if text:
                    items.append(text)
        return items
    if isinstance(value, dict):
        return [f"{key} = {value[key]}" for key in value]
    text = str(value).strip()
    return [text] if text else []


def evaluate_docx_expected_rule(
    document: ThesisDocument, rule: dict[str, Any]
) -> tuple[bool, str]:
    """Evaluate a docx-style expected rule against the stored snapshot."""

    snapshot = document.format_snapshot or {}
    expected_items = get_rule_expected(rule) or []
    if not isinstance(expected_items, list):
        return False, f"{rule['label']}: 期望项格式无效"

    target_snapshot = select_docx_target_snapshot(snapshot, expected_items)
    results: list[str] = []
    all_matched = True
    for item in expected_items:
        if not isinstance(item, dict):
            all_matched = False
            continue
        path = str(item.get("path", "")).strip()
        expected = item.get("value")
        actual = lookup_docx_snapshot_value(target_snapshot, path)
        matched = compare_docx_expected_value(path, actual, expected)
        results.append(
            f"{path}: 期望 {format_docx_expected_scalar(expected)}，实际 {format_docx_expected_scalar(actual)}"
        )
        if not matched:
            all_matched = False
    return all_matched, f"{rule['label']}: {'；'.join(results)}"


def build_rule_suggestion(rule: dict[str, Any]) -> str:
    """Build a short rule-based suggestion without hint text."""

    label = str(rule.get("label", "")).strip() or str(rule.get("id", "")).strip()
    expected = format_expected_value(get_rule_expected(rule))
    if expected:
        return f"请核对{label}：{expected}"
    return f"请核对{label}"


def select_docx_target_snapshot(
    snapshot: dict[str, Any], expected_items: list[dict[str, Any]]
) -> dict[str, Any]:
    """Pick the most relevant snapshot bucket for a rule."""

    paragraphs = snapshot.get("paragraphs", [])
    tables = snapshot.get("tables", [])
    sections = snapshot.get("sections", [])

    paragraph_expectations = {
        str(item.get("path", "")).strip(): item.get("value")
        for item in expected_items
        if isinstance(item, dict)
    }

    if any(path.startswith("table.") for path in paragraph_expectations):
        return {
            "paragraphs": [],
            "tables": [tables[0]] if isinstance(tables, list) and tables else [],
            "sections": sections if isinstance(sections, list) else [],
            "word_count": snapshot.get("word_count"),
            "section_count": snapshot.get("section_count"),
        }

    if any(path.startswith("section.") for path in paragraph_expectations):
        return {
            "paragraphs": [],
            "tables": tables if isinstance(tables, list) else [],
            "sections": [sections[0]]
            if isinstance(sections, list) and sections
            else [],
            "word_count": snapshot.get("word_count"),
            "section_count": snapshot.get("section_count"),
        }

    target_paragraph = select_matching_paragraph(paragraphs, paragraph_expectations)
    return {
        "paragraphs": [target_paragraph] if target_paragraph else [],
        "tables": tables if isinstance(tables, list) else [],
        "sections": sections if isinstance(sections, list) else [],
        "word_count": snapshot.get("word_count"),
        "section_count": snapshot.get("section_count"),
    }


def select_matching_paragraph(
    paragraphs: Any, expectations: dict[str, Any]
) -> dict[str, Any] | None:
    """Select the paragraph most likely to match a rule."""

    if not isinstance(paragraphs, list) or not paragraphs:
        return None

    style_expected = expectations.get("paragraph.style")
    if style_expected is not None:
        normalized_expected = normalize_docx_expected_token(style_expected)
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                continue
            if (
                normalize_docx_expected_token(paragraph.get("style"))
                == normalized_expected
            ):
                return paragraph

    if any(path.startswith("run.") for path in expectations):
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                continue
            if paragraph.get("run") or paragraph.get("runs"):
                return paragraph

    for paragraph in paragraphs:
        if isinstance(paragraph, dict):
            return paragraph
    return None


def lookup_docx_snapshot_value(snapshot: dict[str, Any], path: str) -> Any:
    """Resolve a dotted path against the stored docx snapshot."""

    if not path:
        return None
    root, _, remainder = path.partition(".")
    if root == "document":
        if remainder == "word_count":
            return snapshot.get("word_count")
        if remainder == "section_count":
            return snapshot.get("section_count")
        return snapshot.get(remainder)
    if root == "paragraph":
        return resolve_from_first_paragraph(snapshot, remainder)
    if root == "paragraph_format":
        return resolve_from_first_paragraph(snapshot, remainder)
    if root == "run":
        return resolve_from_first_run(snapshot, remainder)
    if root == "table":
        return resolve_from_first_table(snapshot, remainder)
    if root == "section":
        return resolve_from_first_section(snapshot, remainder)
    return None


def resolve_from_first_paragraph(snapshot: dict[str, Any], remainder: str) -> Any:
    """Resolve a paragraph property from the first stored paragraph."""

    paragraphs = snapshot.get("paragraphs", [])
    if not isinstance(paragraphs, list) or not paragraphs:
        return None
    paragraph = paragraphs[0]
    if not isinstance(paragraph, dict):
        return None
    if not remainder:
        return paragraph
    if remainder == "style":
        return paragraph.get("style")
    if remainder == "alignment":
        return paragraph.get("alignment")
    if remainder == "line_spacing":
        return paragraph.get("line_spacing")
    if remainder == "paragraph_format.line_spacing":
        return paragraph.get("line_spacing")
    if remainder == "space_before":
        return paragraph.get("space_before_pt")
    if remainder == "paragraph_format.space_before":
        return paragraph.get("space_before_pt")
    if remainder == "space_after":
        return paragraph.get("space_after_pt")
    if remainder == "paragraph_format.space_after":
        return paragraph.get("space_after_pt")
    if remainder == "first_line_indent":
        return paragraph.get("first_line_indent_pt")
    if remainder == "paragraph_format.first_line_indent":
        return paragraph.get("first_line_indent_pt")
    return paragraph.get(remainder)


def resolve_from_first_run(snapshot: dict[str, Any], remainder: str) -> Any:
    """Resolve a run property from the first stored run."""

    paragraphs = snapshot.get("paragraphs", [])
    if not isinstance(paragraphs, list) or not paragraphs:
        return None
    first_paragraph = paragraphs[0]
    if not isinstance(first_paragraph, dict):
        return None
    run = first_paragraph.get("run", {})
    if not isinstance(run, dict):
        return None
    if remainder == "font.name":
        return run.get("font_name")
    if remainder == "font.size":
        return run.get("font_size_pt")
    if remainder == "font.bold":
        return run.get("bold")
    return run.get(remainder)


def resolve_from_first_table(snapshot: dict[str, Any], remainder: str) -> Any:
    """Resolve a table property from the first table snapshot."""

    tables = snapshot.get("tables", [])
    if not isinstance(tables, list) or not tables:
        return None
    table = tables[0]
    if not isinstance(table, dict):
        return None
    if remainder == "alignment":
        return table.get("alignment")
    return table.get(remainder)


def resolve_from_first_section(snapshot: dict[str, Any], remainder: str) -> Any:
    """Resolve a section property from the first section snapshot."""

    sections = snapshot.get("sections", [])
    if not isinstance(sections, list) or not sections:
        return None
    section = sections[0]
    if not isinstance(section, dict):
        return None
    mapping = {
        "top_margin": "top_margin_pt",
        "bottom_margin": "bottom_margin_pt",
        "left_margin": "left_margin_pt",
        "right_margin": "right_margin_pt",
    }
    return section.get(mapping.get(remainder, remainder))


def compare_docx_expected_value(path: str, actual: Any, expected: Any) -> bool:
    """Compare actual and expected values for docx-style paths."""

    if actual is None:
        return False
    if path.endswith("font.size"):
        return compare_numeric_value(actual, expected, tolerance=0.25)
    if path.endswith("line_spacing"):
        return compare_numeric_value(actual, expected, tolerance=0.1)
    if path.endswith(("space_before", "space_after")):
        return compare_length_value(actual, expected)
    if path.endswith("first_line_indent"):
        return compare_length_value(actual, expected)
    if path.endswith("alignment"):
        return normalize_docx_expected_token(actual) == normalize_docx_expected_token(
            expected
        )
    if path.endswith("style"):
        return normalize_docx_expected_token(actual) == normalize_docx_expected_token(
            expected
        )
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, int | float) and not isinstance(expected, bool):
        return compare_numeric_value(actual, expected, tolerance=0.25)
    return str(actual).strip() == str(expected).strip()


def compare_numeric_value(actual: Any, expected: Any, *, tolerance: float) -> bool:
    """Compare numeric values with tolerance."""

    actual_value = parse_flexible_numeric_value(actual)
    expected_value = parse_flexible_numeric_value(expected)
    if actual_value is None or expected_value is None:
        return False
    return abs(actual_value - expected_value) <= tolerance


def compare_length_value(actual: Any, expected: Any) -> bool:
    """Compare typographic lengths after normalization."""

    actual_value = parse_length_to_points(actual)
    expected_value = parse_length_to_points(expected)
    if actual_value is None or expected_value is None:
        return False
    return abs(actual_value - expected_value) <= 0.5


def parse_flexible_numeric_value(value: Any) -> float | None:
    """Parse numeric-like values from strings or numbers."""

    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_length_to_points(value: Any) -> float | None:
    """Parse common document length expressions into points."""

    if value in (None, ""):
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    magnitude = float(match.group(0))
    if "cm" in text:
        return round(magnitude * 28.3465, 2)
    if "pt" in text:
        return round(magnitude, 2)
    if "in" in text:
        return round(magnitude * 72, 2)
    return round(magnitude, 2)


def normalize_docx_expected_token(value: Any) -> str:
    """Normalize docx enum-like expected values."""

    text = str(value).strip().lower()
    text = text.replace("wd_align_paragraph.", "")
    text = text.replace("wd_table_alignment.", "")
    return text.replace("_", "")


def format_docx_expected_scalar(value: Any) -> str:
    """Format a scalar value for docx evidence output."""

    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def normalize_string_list(value: Any) -> list[str]:
    """Normalize a value to a compact non-empty string list."""

    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def parse_optional_float(value: Any) -> float | None:
    """Parse an optional numeric value."""

    if value in (None, ""):
        return None
    return parse_float_value(value)


def normalize_rule_check(rule: dict[str, Any]) -> dict[str, Any]:
    """Normalize a rule's check payload into a dict."""

    check = rule.get("check", rule.get("signal", {}))
    if not isinstance(check, dict):
        return {}
    return check


def get_rule_expected(rule: dict[str, Any]) -> Any:
    """Fetch expected values from the rule check payload."""

    check = normalize_rule_check(rule)
    return check.get("expected")


def parse_float_value(value: Any) -> float:
    """Parse a numeric value with validation."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = "numeric format rubric value required"
        raise ValueError(msg)
    return float(value)


def normalize_format_spec_payload(payload: Any) -> dict[str, Any]:
    """Normalize format spec payloads into the internal structured shape."""

    if isinstance(payload, list):
        return {"sections": payload}
    if isinstance(payload, dict) and "sections" in payload:
        return payload
    msg = "format rubric JSON must be a list or an object with sections"
    raise ValueError(msg)


def resolve_rubric_items(
    *,
    rubric: dict[str, Any] | None,
    rubric_filename: str,
) -> tuple[list[RubricItem], str]:
    """Resolve rubric items from upload metadata or bundled config."""

    default_items = normalize_rubric_payload(load_json_resource(rubric_filename))
    if rubric and rubric.get("items"):
        return merge_rubric_items(
            default_items,
            normalize_rubric_items(rubric["items"], require_all=False),
        ), rubric.get("source_name", "uploaded_rubric.json")
    return default_items, rubric_filename


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
    """Normalize rubric item dictionaries and ensure required criteria exist."""

    normalized = [
        RubricItem(
            name=normalize_criterion_name(item.get("criterion", "")),
            standards=parse_standards(item.get("standard", item.get("standards", []))),
            max_score=parse_score_value(item.get("score", 0)),
        )
        for item in items
    ]
    item_by_name = {item.name: item for item in normalized}
    missing = [name for name in REQUIRED_CRITERIA if name not in item_by_name]
    if missing:
        if not require_all:
            return normalized
        msg = f"score rubric missing required criteria: {', '.join(missing)}"
        raise ValueError(msg)
    return [item_by_name[name] for name in REQUIRED_CRITERIA]


def merge_rubric_items(
    default_items: list[RubricItem], uploaded_items: list[RubricItem]
) -> list[RubricItem]:
    """Overlay uploaded rubric scores onto required default criteria."""

    item_by_name = {item.name: item for item in default_items}
    for item in uploaded_items:
        if item.name in item_by_name:
            item_by_name[item.name] = item
    return [item_by_name[name] for name in REQUIRED_CRITERIA]


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

    return ScoreCriterion(
        key=key,
        name=rubric_item.name,
        score=round(max(0.0, min(rubric_item.max_score, score)), 2),
        max_score=rubric_item.max_score,
        standards=rubric_item.standards,
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
    """Count common citation markers in Chinese technical papers."""

    bracket_refs = re.findall(
        r"\[(?:\d+|[一二三四五六七八九十]+)(?:[-,，]\d+)?\]", text
    )
    author_year = re.findall(r"[（(][^）)]{1,12}(?:19|20)\d{2}[^）)]*[）)]", text)
    reference_lines = re.findall(r"^\s*\[\d+\].+$", text, flags=re.MULTILINE)
    return len(bracket_refs) + len(author_year) + len(reference_lines)
