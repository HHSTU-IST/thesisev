"""High-level service API for thesis evaluation."""

from __future__ import annotations

from pathlib import Path

from thesisev.analyzers import (
    annotate_section_statistics,
    annotate_topic_relevance,
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
    topic_analysis = annotate_topic_relevance(document)
    statistics = build_statistics(document)
    issues = detect_issues(document)
    keywords = extract_keywords(document)
    technology_details = extract_technology_details(document)
    technology_stack = extract_technology_stack(document)
    score = calculate_score(issues, len(document.sections))
    comment, comment_checks = generate_comment(
        title=document.title,
        keywords=keywords,
        technology_details=technology_details,
        topic_keywords=topic_analysis["topic_keywords"],
        topic_relevance_ratio=topic_analysis["document_ratio"],
        score=score,
        issues=issues,
        root_sections=document.root_sections,
    )
    return EvaluationResult(
        document=document,
        statistics=statistics,
        issues=issues,
        keywords=keywords,
        technology_stack=technology_stack,
        technology_details=technology_details,
        topic_keywords=topic_analysis["topic_keywords"],
        topic_relevance_ratio=topic_analysis["document_ratio"],
        score=score,
        comment=comment,
        comment_checks=comment_checks,
        metadata={"version": "0.1.0", "topic_analysis": topic_analysis},
    )


def structure_document(path: str | Path) -> ThesisDocument:
    """Load a thesis document and return only its structured representation."""

    document = load_document(path)
    annotate_section_statistics(document)
    annotate_topic_relevance(document)
    return document
