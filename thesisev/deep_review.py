"""Optional LLM deep review for logic and tone.

Deterministic checks in :mod:`thesisev.logic_review` always run and cover the
most mechanical flaws.  When a model is configured, :func:`run_deep_review`
adds one extra bounded call that judges the *same* two dimensions with
context: cross-chapter evidence chains, contradictions and wording that a
word list cannot decide.  Any failure (no API key, transport error, invalid
JSON, empty findings) degrades to an empty list, so the pipeline behaves
exactly as before when the LLM is unavailable.
"""

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
from thesisev.models import Issue, ThesisDocument

logger = logging.getLogger(__name__)

CATEGORY_LOGIC = "逻辑问题"
CATEGORY_TONE = "口语化表达"
SUPPORTED_TYPES = {"logic", "tone"}
SEVERITIES = {"low", "medium", "high"}
DEFAULT_MAX_FINDINGS = 6
SECTION_EXCERPT_LIMIT = 320
MAX_SECTIONS_IN_PROMPT = 30


def run_deep_review(
    *,
    document: ThesisDocument,
    model_config: ModelConfig,
    existing_issues: list[Issue] | None = None,
    max_findings: int = DEFAULT_MAX_FINDINGS,
) -> list[Issue]:
    """Run the optional LLM logic/tone review, degrading safely to empty."""

    if model_config is None or not model_config.is_available():
        return []
    prompt = build_deep_review_prompt(document)
    try:
        model = create_chat_model(model_config)
        response = invoke_chat_model_with_retry(
            model,
            [
                SystemMessage(
                    content=(
                        "你是一名严谨的中文论文评审老师。"
                        "你只负责逻辑与语气两类问题的深度复核，"
                        "不负责格式、标点、评分。"
                        "请严格按要求的 JSON 结构输出，不要 Markdown，不要解释。"
                    )
                ),
                HumanMessage(content=prompt),
            ],
            attempts=2,
        )
        payload = parse_deep_review_json(response)
        findings = normalize_deep_review_findings(payload)
    except Exception as exc:
        logger.warning("deep review skipped after LLM failure: %s", exc)
        return []
    deep_issues = [
        to_deep_review_issue(document=document, finding=finding) for finding in findings
    ]
    seen = {(issue.category, issue.message) for issue in (existing_issues or [])}
    added: list[Issue] = []
    for issue in deep_issues:
        key = (issue.category, issue.message)
        if key in seen:
            continue
        seen.add(key)
        added.append(issue)
        if len(added) >= max_findings:
            break
    return added


def build_deep_review_prompt(document: ThesisDocument) -> str:
    """Build a bounded prompt exposing section-level evidence only."""

    sections = [
        section
        for section in document.sections
        if not section.is_mermaid_code and not section.skip_format_check
    ]
    section_summary = [
        {
            "identifier": section.identifier,
            "title": section.title,
            "word_count": section.word_count,
            "excerpt": _clip(section.content, limit=SECTION_EXCERPT_LIMIT),
        }
        for section in sections[:MAX_SECTIONS_IN_PROMPT]
    ]
    payload = {
        "title": document.title,
        "total_word_count": document.total_word_count,
        "abstract": _clip(document.abstract, limit=800),
        "sections": section_summary,
    }
    return (
        "请基于以下论文结构信息做一次逻辑与语气深度复核，并输出严格 JSON。\n"
        "JSON 结构必须为：\n"
        '{"findings":[{"type":"logic","severity":"medium",'
        '"message":"一句话指出问题","suggestion":"修改建议",'
        '"section":"对应章节标题或编号"}...]}\n'
        "要求：\n"
        '1. type 只能是 "logic" 或 "tone"：'
        "logic 针对前后数据/结论矛盾、论证链缺失、结论无实验支撑、章节衔接断裂；"
        "tone 针对口语化或主观化表述（如程度词滥用、随意连接词）及其上下文是否确实不妥。\n"
        '2. severity 只能是 "low" / "medium" / "high"。\n'
        "3. message 必须能对应到正文证据，空泛的套话不要列；"
        "每类最多 3 条，没有则返回空数组。\n"
        "4. 不要评价格式、排版、标点；不要给出分数。\n"
        f"论文信息：{json.dumps(payload, ensure_ascii=False)}"
    )


def parse_deep_review_json(response: Any) -> dict[str, Any]:
    """Parse the strict JSON payload from a deep review response."""

    text = extract_response_text(response).strip()
    if not text:
        raise ValueError("deep review returned empty response")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("deep review response contains no JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("deep review payload must be an object")
    return payload


def normalize_deep_review_findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize raw findings from the LLM payload."""

    raw_findings = payload.get("findings")
    if not isinstance(raw_findings, list):
        return []
    findings: list[dict[str, Any]] = []
    for entry in raw_findings:
        if not isinstance(entry, dict):
            continue
        finding_type = str(entry.get("type") or "").strip().lower()
        severity = str(entry.get("severity") or "").strip().lower()
        message = str(entry.get("message") or "").strip()
        suggestion = str(entry.get("suggestion") or "").strip()
        section = str(entry.get("section") or "").strip()
        if finding_type not in SUPPORTED_TYPES:
            continue
        if severity not in SEVERITIES:
            continue
        if not message:
            continue
        findings.append(
            {
                "type": finding_type,
                "severity": severity,
                "message": message,
                "suggestion": suggestion or "请结合上下文复核并修正。",
                "section": section,
            }
        )
    return findings


def to_deep_review_issue(*, document: ThesisDocument, finding: dict[str, Any]) -> Issue:
    """Map one normalized LLM finding onto an Issue for the existing UI."""

    finding_type = finding["type"]
    section = _locate_section(document, finding.get("section", ""))
    category = CATEGORY_LOGIC if finding_type == "logic" else CATEGORY_TONE
    return Issue(
        category=category,
        rule_id=f"llm_{finding_type}",
        severity=finding["severity"],
        message=finding["message"],
        suggestion=finding["suggestion"],
        section_identifier=section.identifier if section else "",
        section_title=section.title if section else "",
        paragraph_index=-1,
        sentence_index=-1,
        matched_text=finding.get("section", ""),
        excerpt=_clip(section.content if section else finding["message"], limit=120),
    )


def merge_deep_review_issues(
    existing: list[Issue], deep_issues: list[Issue]
) -> list[Issue]:
    """Append deep-review issues, dropping exact duplicates by message."""

    seen = {(issue.category, issue.message) for issue in existing}
    merged = list(existing)
    for issue in deep_issues:
        key = (issue.category, issue.message)
        if key in seen:
            continue
        seen.add(key)
        merged.append(issue)
    return merged


def _locate_section(document: ThesisDocument, label: str) -> Any | None:
    """Find the best-matching section for a human-written label."""

    if not label:
        return None
    for section in document.sections:
        if label == section.title or label == section.identifier:
            return section
    for section in document.sections:
        if section.title and section.title in label:
            return section
        if section.identifier and label.startswith(section.identifier):
            return section
    return None


def _clip(text: str, *, limit: int) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."
