"""Framework-free evaluation pipeline.

This module owns the end-to-end ``parse -> analyze -> score -> report`` flow
(`:func:`evaluate_document`, :func:`structure_document`) without importing any
web/ASGI machinery, so the pipeline is reusable from scripts and tests and the
package root no longer pulls FastAPI/uvicorn on ``import thesisev``.
``thesisev.api`` is a thin HTTP adapter on top of these functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thesisev.analyzers import (
    annotate_section_statistics,
    annotate_topic_relevance,
    build_statistics,
    detect_issue_groups,
    extract_keywords,
    extract_technology_details,
    extract_technology_stack,
    split_technology_stack,
)
from thesisev.commentary import generate_comment
from thesisev.deep_review import run_deep_review
from thesisev.llm import ModelConfig, build_model_config
from thesisev.models import EvaluationResult, Issue, ThesisDocument
from thesisev.parser import load_document
from thesisev.rubric_utils import DEFAULT_THESIS_TECH_RUBRIC, PRESET_CONFIGS
from thesisev.scoring import build_content_context, calculate_score_report

__all__ = ["evaluate_document", "resolve_preset_files", "structure_document"]


def resolve_preset_files(preset: str) -> tuple[str, str | None]:
    """Resolve bundled rubric filenames for a preset.

    Raises ``ValueError`` for an unknown or empty preset; the HTTP adapter
    translates that into a ``400`` while the CLI lets it surface as a plain
    usage error.
    """

    config = PRESET_CONFIGS.get(preset)
    if config is None or len(config) == 0:
        msg = f"unknown preset: {preset}"
        raise ValueError(msg)
    return config["rubric"], config["format"] or None


def evaluate_document(
    path: str | Path,
    *,
    engine: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    timeout: int = 60,
    model_config: ModelConfig | None = None,
    rubric_filename: str = DEFAULT_THESIS_TECH_RUBRIC,
    rubric: dict[str, Any] | None = None,
    format_requirements: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Evaluate a thesis document from a local path.

    ``model_config`` short-circuits the individual knobs.  When it is omitted,
    a config is built from ``engine``/``provider``; callers that intend to run
    a local engine should probe the runtime first (see
    :func:`thesisev.llm.resolve_runtime`) so a stopped server degrades at once.
    """

    document = load_document(path)
    topic_analysis = annotate_topic_relevance(document)
    statistics = build_statistics(document, topic_analysis=dict(topic_analysis))
    issue_groups = detect_issue_groups(document)
    format_issues = issue_groups.format_issues
    writing_issues = issue_groups.writing_issues
    issues = issue_groups.all_issues
    keywords = extract_keywords(
        document, topic_keywords=topic_analysis["topic_keywords"]
    )
    technology_details = extract_technology_details(document)
    technology_stack = extract_technology_stack(document)
    technology_groups = split_technology_stack(technology_details)
    content_context = build_content_context(
        document=document,
        topic_analysis=dict(topic_analysis),
        keywords=keywords,
        technology_details=technology_details,
    )
    runtime_model_config = model_config or build_model_config(
        engine=engine,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    score_report = calculate_score_report(
        document=document,
        topic_analysis=dict(topic_analysis),
        format_issues=format_issues,
        writing_issues=writing_issues,
        content_context=content_context,
        keywords=keywords,
        technology_details=technology_details,
        format_requirements=format_requirements,
        rubric=rubric,
        rubric_filename=rubric_filename,
        model_config=runtime_model_config,
    )
    score = score_report.score
    comment, comment_checks, comment_source = generate_comment(
        title=document.title,
        keywords=keywords,
        technology_details=technology_details,
        topic_keywords=topic_analysis["topic_keywords"],
        topic_relevance_ratio=topic_analysis["document_ratio"],
        root_sections=document.root_sections,
        model_config=runtime_model_config,
    )
    available = runtime_model_config.is_available()
    if available:
        deep_issues = run_deep_review(
            document=document, model_config=runtime_model_config, existing_issues=issues
        )
        if deep_issues:
            issues = [*issues, *deep_issues]
    else:
        deep_issues: list[Issue] = []
    return EvaluationResult(
        document=document,
        statistics=statistics,
        issues=issues,
        format_issues=format_issues,
        writing_issues=writing_issues,
        content_context=content_context,
        keywords=keywords,
        technology_stack=technology_stack,
        software_technology_stack=technology_groups["software_technology_stack"],
        hardware_technology_stack=technology_groups["hardware_technology_stack"],
        technology_details=technology_details,
        topic_keywords=topic_analysis["topic_keywords"],
        topic_relevance_ratio=topic_analysis["document_ratio"],
        score=score,
        comment=comment,
        comment_checks=comment_checks,
        metadata={
            "version": "0.1.0",
            "topic_analysis": topic_analysis,
            "score_source": score_report.score_source,
            "score_detail": score_report.to_dict(),
            "comment_source": comment_source,
            "rubric": rubric,
            "format_requirements": format_requirements,
            "evaluation_roles": {
                "format_detection": "local",
                "format_evaluation": "local",
                "writing_detection": "local",
                "writing_evaluation": "local",
                "logic_detection": "local",
                "logic_deep_review": ("llm" if available else "skipped"),
                "content_evaluation": comment_source,
                "llm_input": "content_context_only",
            },
            "model": runtime_model_config.to_metadata(),
            "deep_review": {
                "source": "llm" if available else "skipped",
                "added_count": len(deep_issues),
            },
        },
    )


def structure_document(path: str | Path) -> ThesisDocument:
    """Load a thesis document and return only its structured representation."""

    document = load_document(path)
    annotate_section_statistics(document)
    annotate_topic_relevance(document)
    return document
