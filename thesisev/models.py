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
    is_mermaid_code: bool = False
    skip_format_check: bool = False
    topic_relevance_score: float = 0.0
    topic_matched_keywords: list[str] = field(default_factory=list)
    topic_is_relevant: bool = False


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
    is_mermaid_code: bool = False
    skip_format_check: bool = False
    children: list[Section] = field(default_factory=list)
    subtree_word_count: int = 0
    ratio: float = 0.0
    parent_ratio: float = 0.0
    topic_relevance_score: float = 0.0
    topic_relevant_word_count: int = 0
    topic_matched_keywords: list[str] = field(default_factory=list)

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
    format_snapshot: dict[str, Any] = field(default_factory=dict)

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
            "format_snapshot": self.format_snapshot,
            "sections": [section.to_dict() for section in self.sections],
            "root_sections": [section.to_dict() for section in self.root_sections],
        }


@dataclass(slots=True)
class Issue:
    """A detected writing or formatting issue."""

    category: str
    rule_id: str
    severity: str
    message: str
    suggestion: str
    section_identifier: str
    section_title: str
    paragraph_index: int
    sentence_index: int
    matched_text: str
    excerpt: str


@dataclass(slots=True)
class Statistic:
    """A calculated statistic for the document."""

    label: str
    value: str


@dataclass(slots=True)
class TechnologyStackItem:
    """A detected technology mention."""

    name: str
    category: str
    matched_terms: list[str]


@dataclass(slots=True)
class EvaluationResult:
    """Full evaluation output for a thesis document."""

    document: ThesisDocument
    statistics: list[Statistic]
    issues: list[Issue]
    keywords: list[str]
    technology_stack: list[str]
    technology_details: list[TechnologyStackItem]
    topic_keywords: list[str]
    topic_relevance_ratio: float
    score: int
    comment: str
    comment_checks: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    format_issues: list[Issue] = field(default_factory=list)
    writing_issues: list[Issue] = field(default_factory=list)
    content_context: dict[str, Any] = field(default_factory=dict)
    software_technology_stack: list[str] = field(default_factory=list)
    hardware_technology_stack: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the evaluation result to a JSON-friendly dictionary."""

        return {
            "document": self.document.to_dict(),
            "statistics": [asdict(statistic) for statistic in self.statistics],
            "issues": [asdict(issue) for issue in self.issues],
            "format_issues": [asdict(issue) for issue in self.format_issues],
            "writing_issues": [asdict(issue) for issue in self.writing_issues],
            "content_context": self.content_context,
            "keywords": self.keywords,
            "technology_stack": self.technology_stack,
            "software_technology_stack": self.software_technology_stack,
            "hardware_technology_stack": self.hardware_technology_stack,
            "technology_details": [
                asdict(technology_item) for technology_item in self.technology_details
            ],
            "topic_keywords": self.topic_keywords,
            "topic_relevance_ratio": self.topic_relevance_ratio,
            "score": self.score,
            "comment": self.comment,
            "comment_checks": self.comment_checks,
            "metadata": self.metadata,
        }
