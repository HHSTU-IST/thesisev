"""Rule-based scoring for thesis evaluation rubrics."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from thesisev.models import Issue, TechnologyStackItem, ThesisDocument
from thesisev.resources import load_json_resource

DEFAULT_THESIS_TECH_RUBRIC = "score_thesis_tech.json"
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-friendly dictionary."""

        return {
            "score": self.score,
            "raw_score": self.raw_score,
            "raw_total": self.raw_total,
            "rubric_source": self.rubric_source,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
        }


@dataclass(slots=True)
class RubricItem:
    """Normalized rubric item loaded from JSON or upload metadata."""

    name: str
    standards: list[str]
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
) -> ScoreReport:
    """Calculate a rubric-based percentage score from deterministic signals."""

    rubric_items, rubric_source = resolve_rubric_items(
        rubric=rubric, rubric_filename=rubric_filename
    )
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
    raw_score = round(sum(item.score for item in criteria), 2)
    raw_total = round(sum(item.max_score for item in criteria), 2)
    score = round(raw_score / max(raw_total, 1) * 100)
    return ScoreReport(
        score=max(0, min(100, score)),
        raw_score=raw_score,
        raw_total=raw_total,
        criteria=criteria,
        rubric_source=rubric_source,
    )


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
        f"总字数 {document.total_word_count}",
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
