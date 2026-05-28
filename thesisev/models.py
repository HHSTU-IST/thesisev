"""Core data models for thesis evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Section:
    """A parsed thesis section."""

    level: int
    title: str
    content: str
    paragraphs: list[str]
    sentences: list[str]
    word_count: int
    ratio: float = 0.0


@dataclass(slots=True)
class ThesisDocument:
    """Structured thesis content used across the pipeline."""

    title: str
    source_path: str
    raw_text: str
    cleaned_text: str
    sections: list[Section]
    paragraphs: list[str]
    sentences: list[str]
    total_word_count: int


@dataclass(slots=True)
class Issue:
    """A detected writing or formatting issue."""

    category: str
    severity: str
    message: str
    suggestion: str
    section_title: str
    excerpt: str


@dataclass(slots=True)
class Statistic:
    """A calculated statistic for the document."""

    label: str
    value: str


@dataclass(slots=True)
class EvaluationResult:
    """Full evaluation output for a thesis document."""

    document: ThesisDocument
    statistics: list[Statistic]
    issues: list[Issue]
    keywords: list[str]
    technology_stack: list[str]
    score: int
    comment: str
    metadata: dict[str, str] = field(default_factory=dict)
