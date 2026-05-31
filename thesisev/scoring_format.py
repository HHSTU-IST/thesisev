"""Local format scoring rules and DOCX expectation checks."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from thesisev.models import Issue, ThesisDocument
from thesisev.scoring_content import count_terms


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
    format_issues: list[Issue],
    rules: list[dict[str, Any]],
    format_requirements: dict[str, Any] | None,
) -> dict[str, Any]:
    """Score bundled format rules against document signals."""

    issue_categories = Counter(issue.category for issue in format_issues)
    evidence: list[str] = []
    deductions: list[str] = []
    suggestions: list[str] = []
    deduction = 0.0
    seen_penalty_keys: set[str] = set()

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
            penalty_key = str(rule.get("id") or rule.get("label") or "").strip()
            if penalty_key in seen_penalty_keys:
                continue
            seen_penalty_keys.add(penalty_key)
            rule_penalty = parse_float_value(rule.get("points", 1)) * parse_float_value(
                rule.get("section_weight", 1)
            )
            deduction += min(rule_penalty, 3.0)
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
