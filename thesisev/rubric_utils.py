"""Shared rubric parsing and scoring helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


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
class RubricItem:
    """Normalized rubric item loaded from JSON or upload metadata."""

    name: str
    standards: list[str]
    evaluation: str
    max_score: float


def normalize_criterion_name(criterion: Any) -> str:
    """Normalize criterion labels to their top-level rubric name."""

    if not isinstance(criterion, str) or not criterion.strip():
        msg = "rubric criterion must be a non-empty string"
        raise ValueError(msg)
    return re.split(r"[：:]", criterion.strip(), maxsplit=1)[0].strip()


def parse_standards(value: Any) -> list[str]:
    """Parse rubric standard descriptions."""

    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def parse_score_value(value: Any) -> float:
    """Parse a numeric score value."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = "rubric score must be numeric"
        raise ValueError(msg)
    return float(value)


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
            score=clamped_score, max_score=rubric_item.max_score, deductions=deductions
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


def parse_rubric_criterion(criterion: Any) -> str:
    """Parse a rubric criterion as non-empty text."""

    if not isinstance(criterion, str) or not criterion.strip():
        msg = "rubric criterion must be a non-empty string"
        raise ValueError(msg)
    return criterion.strip()


def parse_rubric_score(score: Any) -> float:
    """Parse a rubric score as a numeric value."""

    if isinstance(score, bool):
        msg = "rubric score must be numeric"
        raise ValueError(msg)
    if isinstance(score, int | float):
        return float(score)
    msg = "rubric score must be numeric"
    raise ValueError(msg)


def parse_rubric_standard(standard: Any) -> list[str]:
    """Parse rubric standard descriptions into a text list."""

    if standard in (None, ""):
        return []
    if isinstance(standard, str):
        stripped = standard.strip()
        return [stripped] if stripped else []
    if isinstance(standard, list):
        return [str(item).strip() for item in standard if str(item).strip()]
    return [str(standard).strip()] if str(standard).strip() else []


def parse_rubric_item(*, criterion: Any, value: Any) -> dict[str, Any]:
    """Parse one rubric item from flat or nested JSON shapes."""

    item = {"criterion": parse_rubric_criterion(criterion)}
    if isinstance(value, dict):
        item["score"] = parse_rubric_score(value.get("score", value.get("分数")))
        item["standard"] = parse_rubric_standard(
            value.get("standard", value.get("standards", value.get("标准", [])))
        )
        if value.get("evaluation"):
            item["evaluation"] = str(value["evaluation"]).strip().lower()
        return item
    item["score"] = parse_rubric_score(value)
    item["standard"] = []
    return item


def normalize_rubric_payload(payload: Any) -> list[dict[str, Any]]:
    """Normalize rubric JSON into criterion-score items."""

    if isinstance(payload, dict):
        return [
            parse_rubric_item(criterion=criterion, value=value)
            for criterion, value in payload.items()
        ]

    if isinstance(payload, list):
        items: list[dict[str, Any]] = []
        for entry in payload:
            if not isinstance(entry, dict) or len(entry) != 1:
                msg = "rubric list entries must be single-key objects"
                raise ValueError(msg)
            criterion, value = next(iter(entry.items()))
            items.append(parse_rubric_item(criterion=criterion, value=value))
        return items

    msg = "rubric JSON must be an object or a list of single-key objects"
    raise ValueError(msg)


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


def parse_format_requirement_label(label: Any) -> str:
    """Parse a format-requirements item label."""

    if not isinstance(label, str) or not label.strip():
        msg = "format requirement label must be a non-empty string"
        raise ValueError(msg)
    return label.strip()


def stringify_format_requirement_value(value: Any) -> str:
    """Convert a format-requirements value into compact display text."""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            msg = "format requirement value must not be empty"
            raise ValueError(msg)
        return text
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_format_requirements_payload(payload: Any) -> list[dict[str, str]]:
    """Normalize format-requirements JSON into display-friendly items."""

    if isinstance(payload, dict):
        return [
            {
                "label": parse_format_requirement_label(label),
                "value": stringify_format_requirement_value(value),
            }
            for label, value in payload.items()
        ]

    if isinstance(payload, list):
        items: list[dict[str, str]] = []
        for index, entry in enumerate(payload, start=1):
            if isinstance(entry, str):
                items.append({"label": f"要求 {index}", "value": entry.strip()})
                continue
            if isinstance(entry, dict) and len(entry) == 1:
                label, value = next(iter(entry.items()))
                items.append(
                    {
                        "label": parse_format_requirement_label(label),
                        "value": stringify_format_requirement_value(value),
                    }
                )
                continue
            items.append(
                {
                    "label": f"要求 {index}",
                    "value": stringify_format_requirement_value(entry),
                }
            )
        if items:
            return items
        msg = "format requirements JSON list must not be empty"
        raise ValueError(msg)

    msg = "format requirements JSON must be an object or a list"
    raise ValueError(msg)


def normalize_structured_format_requirements(
    payload: dict[str, Any], *, source_name: str
) -> dict[str, Any]:
    """Normalize the structured format rubric into a UI-friendly summary."""

    sections = payload.get("sections", [])
    if not isinstance(sections, list):
        raise ValueError("format rubric sections must be a list")

    section_items: list[dict[str, Any]] = []
    display_items: list[dict[str, str]] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        section_label = str(section.get("name") or section.get("id") or "").strip()
        rule_count = 0
        for rule in section.get("rules", []):
            if not isinstance(rule, dict):
                continue
            rule_count += 1
            check = rule.get("check", {})
            if not isinstance(check, dict):
                check = {}
            display_items.append(
                {
                    "label": f"{section_label} / {str(rule.get('label') or rule.get('id') or '').strip()}",
                    "value": stringify_format_requirement_value(
                        check.get("expected", "")
                    ),
                }
            )
        section_items.append(
            {
                "label": section_label,
                "weight": section.get("weight", 0),
                "rule_count": rule_count,
            }
        )

    return {
        "source_name": source_name,
        "item_count": len(display_items),
        "items": display_items,
        "sections": section_items,
    }
