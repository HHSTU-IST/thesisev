"""Parsing and preprocessing utilities."""

from __future__ import annotations

import re
from pathlib import Path

from thesisev.models import Section, ThesisDocument

SECTION_PATTERN = re.compile(
    r"^(?P<prefix>(?:第[一二三四五六七八九十]+[章节部分]|"
    r"[0-9]+(?:\.[0-9]+){0,2}|"
    r"[IVXLC]+(?:\.[IVXLC]+){0,2}))[\s、.\-:：]+(?P<title>.+)$"
)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?；;])\s*")


def load_document(path: str | Path) -> ThesisDocument:
    """Load and parse a thesis from a txt or markdown file."""

    source_path = Path(path)
    raw_text = source_path.read_text(encoding="utf-8")
    cleaned_text = clean_text(raw_text)
    lines = [line.strip() for line in cleaned_text.splitlines()]
    title = next((line for line in lines if line), source_path.stem)
    sections = parse_sections(cleaned_text)
    paragraphs = split_paragraphs(cleaned_text)
    sentences = split_sentences(cleaned_text)
    total_word_count = count_words(cleaned_text)
    return ThesisDocument(
        title=title,
        source_path=str(source_path),
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        sections=sections,
        paragraphs=paragraphs,
        sentences=sentences,
        total_word_count=total_word_count,
    )


def clean_text(text: str) -> str:
    """Normalize spacing and drop obvious markdown noise."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    markdown_heading = re.compile(r"^\s*#+\s*", flags=re.MULTILINE)
    normalized = markdown_heading.sub("", normalized)
    return normalized.strip()


def parse_sections(text: str) -> list[Section]:
    """Split text into thesis sections using common heading patterns."""

    lines = text.splitlines()
    heading_indexes: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        heading = match_section_heading(stripped)
        if heading is None:
            continue
        heading_indexes.append((index, heading[0], heading[1]))

    if not heading_indexes:
        return [build_section(level=1, title="全文", content=text)]

    sections: list[Section] = []
    for position, (start_index, level, title) in enumerate(heading_indexes):
        end_index = len(lines)
        if position + 1 < len(heading_indexes):
            end_index = heading_indexes[position + 1][0]
        content_lines = lines[start_index + 1 : end_index]
        content = "\n".join(content_lines).strip()
        sections.append(build_section(level=level, title=title, content=content))
    return sections


def match_section_heading(line: str) -> tuple[int, str] | None:
    """Match a line against supported section heading formats."""

    match = SECTION_PATTERN.match(line)
    if match is not None:
        prefix = match.group("prefix")
        title = match.group("title").strip()
        return infer_level(prefix), title

    if len(line) <= 24 and not line.endswith(("。", ".", "!", "！", "?", "？")):
        if not any(char in line for char in ("，", ",", "；", ";")):
            return 1, line
    return None


def infer_level(prefix: str) -> int:
    """Infer a section depth from its numbering prefix."""

    if "第" in prefix and any(token in prefix for token in ("章", "部分")):
        return 1
    if "第" in prefix and "节" in prefix:
        return 2
    if "." in prefix:
        return prefix.count(".") + 1
    if prefix.isdigit():
        return 1
    return 1


def build_section(level: int, title: str, content: str) -> Section:
    """Construct a section model from raw content."""

    paragraphs = split_paragraphs(content)
    sentences = split_sentences(content)
    word_count = count_words(content)
    return Section(
        level=level,
        title=title,
        content=content,
        paragraphs=paragraphs,
        sentences=sentences,
        word_count=word_count,
    )


def split_paragraphs(text: str) -> list[str]:
    """Split text into non-empty paragraphs."""

    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentences."""

    return [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]


def count_words(text: str) -> int:
    """Count Chinese characters and Latin tokens as a rough word count."""

    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    latin_tokens = re.findall(r"[A-Za-z0-9_]+", text)
    return len(chinese_chars) + len(latin_tokens)
