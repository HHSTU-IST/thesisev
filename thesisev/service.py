"""High-level service API for thesis evaluation."""

from __future__ import annotations

from pathlib import Path

from thesisev.analyzers import (
    annotate_section_statistics,
    build_statistics,
    calculate_score,
    detect_issues,
    extract_keywords,
    extract_technology_details,
    extract_technology_stack,
)
from thesisev.commentary import generate_comment
from thesisev.models import EvaluationResult, ThesisDocument
from thesisev.parser import load_document


def evaluate_document(path: str | Path) -> EvaluationResult:
    """Evaluate a thesis document from a local path."""

    document = load_document(path)
    annotate_section_statistics(document)
    statistics = build_statistics(document)
    issues = detect_issues(document)
    keywords = extract_keywords(document)
    technology_details = extract_technology_details(document)
    technology_stack = extract_technology_stack(document)
    score = calculate_score(issues, len(document.sections))
    comment = generate_comment(
        title=document.title,
        keywords=keywords,
        technology_stack=technology_stack,
        score=score,
        issue_count=len(issues),
    )
    return EvaluationResult(
        document=document,
        statistics=statistics,
        issues=issues,
        keywords=keywords,
        technology_stack=technology_stack,
        technology_details=technology_details,
        score=score,
        comment=comment,
        metadata={"version": "0.1.0"},
    )


def structure_document(path: str | Path) -> ThesisDocument:
    """Load a thesis document and return only its structured representation."""

    document = load_document(path)
    annotate_section_statistics(document)
    return document
