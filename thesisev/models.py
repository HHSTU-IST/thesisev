"""Core data models for thesis evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Sentence:
    """A sentence extracted from the thesis."""

    index: int
    text: str


@dataclass(slots=True)
class Paragraph:
    """A paragraph extracted from the thesis."""

    index: int
    text: str
    sentences: list[Sentence]
    word_count: int


@dataclass(slots=True)
class Section:
    """A parsed thesis section."""

    identifier: str
    level: int
    title: str
    heading: str
    numbering: str
    content: str
    paragraphs: list[Paragraph]
    sentences: list[str]
    word_count: int
    children: list[Section] = field(default_factory=list)
    ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the section to a JSON-friendly dictionary."""

        data = asdict(self)
        data["children"] = [child.to_dict() for child in self.children]
        return data


@dataclass(slots=True)
class ThesisDocument:
    """Structured thesis content used across the pipeline."""

    title: str
    source_path: str
    source_type: str
    raw_text: str
    cleaned_text: str
    front_matter: str
    abstract: str
    sections: list[Section]
    root_sections: list[Section]
    paragraphs: list[Paragraph]
    sentences: list[str]
    total_word_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the document to a JSON-friendly dictionary."""

        return {
            "title": self.title,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "front_matter": self.front_matter,
            "abstract": self.abstract,
            "cleaned_text": self.cleaned_text,
            "total_word_count": self.total_word_count,
            "paragraphs": [asdict(paragraph) for paragraph in self.paragraphs],
            "sentences": self.sentences,
            "sections": [section.to_dict() for section in self.sections],
            "root_sections": [section.to_dict() for section in self.root_sections],
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the evaluation result to a JSON-friendly dictionary."""

        return {
            "document": self.document.to_dict(),
            "statistics": [asdict(statistic) for statistic in self.statistics],
            "issues": [asdict(issue) for issue in self.issues],
            "keywords": self.keywords,
            "technology_stack": self.technology_stack,
            "score": self.score,
            "comment": self.comment,
            "metadata": self.metadata,
        }
