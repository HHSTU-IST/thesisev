"""Deterministic cross-section logic checks.

The local pipeline previously covered formatting and wording only; argument
chains and data consistency were entirely delegated to the LLM.  These
checks give reviewers deterministic, explainable signals for the most common
structural flaws:

- a conclusion/summary chapter that has no evidence chapter behind it
  (experiment / test / result / analysis);
- numerical claims about the same metric that contradict each other across
  different sections (e.g. "识别准确率为 95%" vs "识别准确率约为 88%").

Everything here is heuristic and severity-tagged so reviewers can triage
quickly; it never auto-scores content.
"""

from __future__ import annotations

import re
from collections import defaultdict

from thesisev.models import Issue, Paragraph, Section, ThesisDocument

CATEGORY = "逻辑问题"

#: Chapter titles that assert a conclusion.
_CONCLUSION_TITLE = re.compile(r"^(?:结论|总结|结束语|结语|结 论)")
#: Chapter titles that could supply evidence for a conclusion.
_EVIDENCE_TITLE = re.compile(
    r"(实验|试验|验证|测试|仿真|结果|分析|性能|对比|实测|评估)"
)
#: Chapters whose figures must never be treated as claims (citations, thanks).
_IGNORED_SECTION = re.compile(r"(参考文献|致谢|附录|声明)")

#: A metric claim such as ``识别准确率为 95.2%`` or ``平均时延约 12ms``.
_METRIC_CLAIM = re.compile(
    r"([\u4e00-\u9fff]{2,8}"
    r"(?:准确率|检出率|覆盖率|有效率|成功率|率|精度|时延|延迟|耗时|误差"
    r"|成本|功耗|内存|温度|带宽))"
    r"\s*(?:约为|约|为|是|达到|达|：|:|=)?\s*"
    r"(\d{1,3}(?:\.\d+)?)\s*(%|％|ms|毫秒|MB|GB|s|秒)?"
)

MIN_SECTIONS_FOR_LOGIC = 3
MIN_CLAIMS_FOR_CONSISTENCY = 4
MAX_CONSISTENCY_ISSUES = 8


def detect_logic_issues(document: ThesisDocument) -> list[Issue]:
    """Detect cross-section argument-chain and data-consistency issues."""

    sections = [section for section in document.sections if not section.skip_format_check]
    if len(sections) < MIN_SECTIONS_FOR_LOGIC:
        return []

    issues: list[Issue] = []
    issues.extend(_detect_conclusion_without_evidence(sections))
    issues.extend(_detect_conflicting_metric_claims(sections))
    return issues


def _detect_conclusion_without_evidence(
    sections: list[Section],
) -> list[Issue]:
    """Flag a conclusion chapter when no evidence chapter exists anywhere."""

    conclusion_sections = [
        section
        for section in sections
        if _CONCLUSION_TITLE.match(section.title.strip())
        and not _IGNORED_SECTION.match(section.title.strip())
    ]
    if not conclusion_sections:
        return []

    evidence_sections = [
        section
        for section in sections
        if _EVIDENCE_TITLE.search(section.title.strip())
        and not _IGNORED_SECTION.match(section.title.strip())
    ]
    if evidence_sections:
        return []

    issues: list[Issue] = []
    for section in conclusion_sections[:1]:
        issues.append(
            _build_logic_issue(
                section=section,
                rule_id="conclusion_without_evidence",
                severity="medium",
                message=(
                    f"章节「{section.title}」给出结论性表述，"
                    "但全文未找到实验、测试、验证或结果分析类章节作为支撑依据。"
                ),
                suggestion=(
                    "建议补充实验/测试/仿真与结果分析章节，"
                    "或在结论中说明依据来源，避免结论缺乏证据链。"
                ),
                matched_text=section.title,
                excerpt=_section_excerpt(section),
            )
        )
    return issues


def _detect_conflicting_metric_claims(
    sections: list[Section],
) -> list[Issue]:
    """Flag the same metric reported with conflicting values across sections."""

    claims = _collect_metric_claims(sections)
    if len(claims) < MIN_CLAIMS_FOR_CONSISTENCY:
        return []

    grouped: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for claim in claims:
        core = _metric_core(claim["metric"])
        grouped[core][claim["value"]].add(claim["section_title"])

    issues: list[Issue] = []
    for core, values_by_value in grouped.items():
        distinct_values = [_claim_number(value) for value in values_by_value]
        distinct_sections = {
            title for titles in values_by_value.values() for title in titles
        }
        if len(distinct_values) < 2 or len(distinct_sections) < 2:
            continue
        issues.extend(
            _build_conflict_issues(core, values_by_value, distinct_sections)
        )
        if len(issues) >= MAX_CONSISTENCY_ISSUES:
            break
    return issues


def _collect_metric_claims(sections: list[Section]) -> list[dict[str, str]]:
    """Collect ``{metric, value, section_title, anchor}`` claims body-wide."""

    claims: list[dict[str, str]] = []
    for section in sections:
        title = section.title.strip()
        if _IGNORED_SECTION.match(title):
            continue
        for paragraph in section.paragraphs:
            if paragraph.skip_format_check or paragraph.is_mermaid_code:
                continue
            for sentence in paragraph.sentences:
                for match in _METRIC_CLAIM.finditer(sentence.text):
                    metric = match.group(1).strip()
                    unit = match.group(3) or ""
                    claims.append(
                        {
                            "metric": metric,
                            "value": f"{match.group(2)}{unit}",
                            "section_title": title,
                            "anchor": _clip(sentence.text),
                        }
                    )
    return claims


def _build_conflict_issues(
    core: str,
    values_by_value: dict[str, set[str]],
    distinct_sections: set[str],
) -> list[Issue]:
    """Build a single issue describing the widest conflicting pair."""

    ordered_values = sorted(
        values_by_value, key=lambda value: _claim_number(value), reverse=True
    )
    lowest = ordered_values[-1]
    highest = ordered_values[0]
    if _claim_number(lowest) == _claim_number(highest):
        return []
    high_sections = "、".join(sorted(values_by_value[highest])) or "前文"
    low_sections = "、".join(sorted(values_by_value[lowest])) or "后文"
    section_names = "、".join(sorted(distinct_sections))[:40]
    return [
        Issue(
            category=CATEGORY,
            rule_id="conflicting_metric_value",
            severity="low",
            message=(
                f"指标「{core}」在不同章节出现不一致数值："
                f"{high_sections} 中为 {highest}，{low_sections} 中为 {lowest}。"
                "若属于不同实验条件或统计口径，请明确交代差异原因。"
            ),
            suggestion=(
                "请核对两处数据来源；如为不同条件，建议在同一指标旁标注"
                "（如“训练集 / 测试集”）以免被误读为矛盾。"
            ),
            section_identifier="",
            section_title=section_names,
            paragraph_index=-1,
            sentence_index=-1,
            matched_text=f"{core} {highest} / {lowest}",
            excerpt=_clip(
                f"相关章节：{section_names}（{highest} vs {lowest}）"
            ),
        )
    ]


def _metric_core(metric: str) -> str:
    """Merge near-synonym metric names onto a short stable core."""

    return metric[-3:] if len(metric) >= 3 else metric


def _claim_number(value: str) -> float:
    """Extract the numeric part of a claim such as ``95.2%`` or ``12ms``."""

    cleaned = re.sub(r"[^\d.]", "", value)
    try:
        return float(cleaned)
    except ValueError:
        return float("inf")


def _build_logic_issue(
    *,
    section: Section,
    rule_id: str,
    severity: str,
    message: str,
    suggestion: str,
    matched_text: str,
    excerpt: str,
) -> Issue:
    return Issue(
        category=CATEGORY,
        rule_id=rule_id,
        severity=severity,
        message=message,
        suggestion=suggestion,
        section_identifier=section.identifier,
        section_title=section.title,
        paragraph_index=-1,
        sentence_index=-1,
        matched_text=matched_text,
        excerpt=excerpt,
    )


def _section_excerpt(section: Section) -> str:
    text = _clip(section.content)
    return text or section.title


def _clip(text: str, *, limit: int = 80) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."
