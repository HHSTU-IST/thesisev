"""Parsing and preprocessing utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from thesisev.models import Paragraph, Section, Sentence, ThesisDocument

CHINESE_NUMERALS = "一二三四五六七八九十百千万零"
SECTION_PATTERN = re.compile(
    r"^(?P<prefix>("
    r"第[" + CHINESE_NUMERALS + r"]+[章节部分]"
    r"|第[" + CHINESE_NUMERALS + r"]+节"
    r"|[0-9]+(?:\.[0-9]+){0,3}"
    r"|[IVXLC]+(?:\.[IVXLC]+){0,3}"
    r"|[(（]?[0-9一二三四五六七八九十]+[)）]"
    r"))[\s、.\-:：]+(?P<title>.+)$",
    flags=re.IGNORECASE,
)
MARKDOWN_HEADING_PATTERN = re.compile(r"^\s*#+\s*(.+?)\s*$")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?；;])\s*")
WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


@dataclass(slots=True)
class HeadingMatch:
    """A detected heading line."""

    line_index: int
    level: int
    title: str
    numbering: str
    heading: str


def load_document(path: str | Path) -> ThesisDocument:
    """Load and parse a thesis from txt, markdown, or docx."""

    source_path = Path(path)
    raw_text = read_source_text(source_path)
    cleaned_text = clean_text(raw_text)
    lines = [line.strip() for line in cleaned_text.splitlines()]
    title = next((line for line in lines if line), source_path.stem)
    body_text = remove_title_from_text(cleaned_text, title)
    front_matter, sections = parse_sections(body_text)
    root_sections = build_section_tree(sections)
    paragraphs = collect_document_paragraphs(front_matter, sections)
    sentences = flatten_sentence_text(paragraphs)
    total_word_count = sum(paragraph.word_count for paragraph in paragraphs)
    abstract = find_abstract(sections, front_matter)
    return ThesisDocument(
        title=title,
        source_path=str(source_path),
        source_type=source_path.suffix.lower().lstrip("."),
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        front_matter=front_matter,
        abstract=abstract,
        sections=sections,
        root_sections=root_sections,
        paragraphs=paragraphs,
        sentences=sentences,
        total_word_count=total_word_count,
    )


def read_source_text(path: Path) -> str:
    """Read thesis text from a supported source file."""

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        return read_docx_text(path)
    msg = f"unsupported file type: {path.suffix or 'unknown'}"
    raise ValueError(msg)


def read_docx_text(path: Path) -> str:
    """Extract plain text from a docx file using the OpenXML document body."""

    with ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", WORD_NAMESPACE):
        runs = paragraph.findall(".//w:t", WORD_NAMESPACE)
        text = "".join(run.text or "" for run in runs).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def clean_text(text: str) -> str:
    """Normalize spacing and drop obvious markdown noise."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"^\s*#+\s*", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"^\s*[-*_]{3,}\s*$", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"^\s*>\s?", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"^\s*\|\s*", "", normalized, flags=re.MULTILINE)
    return normalized.strip()


def remove_title_from_text(text: str, title: str) -> str:
    """Remove the title line from the text body if present."""

    lines = text.splitlines()
    skipped = False
    kept_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not skipped and stripped == title:
            skipped = True
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def parse_sections(text: str) -> tuple[str, list[Section]]:
    """Split text into thesis sections using common heading patterns."""

    lines = text.splitlines()
    headings: list[HeadingMatch] = []
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        heading = match_section_heading(stripped)
        if heading is None:
            continue
        headings.append(
            HeadingMatch(
                line_index=index,
                level=heading[0],
                title=heading[1],
                numbering=heading[2],
                heading=heading[3],
            )
        )

    if not headings:
        section = build_section(
            identifier="1",
            level=1,
            title="全文",
            heading="全文",
            numbering="",
            content=text,
        )
        return "", [section]

    first_heading_line = headings[0].line_index
    front_matter = "\n".join(lines[:first_heading_line]).strip()
    sections: list[Section] = []
    for position, heading in enumerate(headings):
        end_index = len(lines)
        if position + 1 < len(headings):
            end_index = headings[position + 1].line_index
        content_lines = lines[heading.line_index + 1 : end_index]
        content = "\n".join(content_lines).strip()
        sections.append(
            build_section(
                identifier=str(position + 1),
                level=heading.level,
                title=heading.title,
                heading=heading.heading,
                numbering=heading.numbering,
                content=content,
            )
        )
    return front_matter, sections


def match_section_heading(line: str) -> tuple[int, str, str, str] | None:
    """Match a line against supported section heading formats."""

    markdown_match = MARKDOWN_HEADING_PATTERN.match(line)
    if markdown_match is not None:
        line = markdown_match.group(1).strip()

    match = SECTION_PATTERN.match(line)
    if match is not None:
        numbering = match.group("prefix").strip()
        title = match.group("title").strip()
        return infer_level(numbering), title, numbering, line

    if len(line) <= 24 and not line.endswith(("。", ".", "!", "！", "?", "？")):
        if not any(char in line for char in ("，", ",", "；", ";")):
            return 1, line, "", line
    return None


def infer_level(prefix: str) -> int:
    """Infer a section depth from its numbering prefix."""

    normalized = prefix.strip()
    if normalized.startswith("第") and any(
        token in normalized for token in ("章", "部分")
    ):
        return 1
    if normalized.startswith("第") and "节" in normalized:
        return 2
    if normalized.startswith(("(", "（")) and normalized.endswith((")", "）")):
        return 3
    if "." in normalized:
        return normalized.count(".") + 1
    if normalized.isdigit():
        return 1
    return 1


def build_section(
    identifier: str, level: int, title: str, heading: str, numbering: str, content: str
) -> Section:
    """Construct a section model from raw content."""

    paragraphs = build_paragraphs(content)
    sentences = flatten_sentence_text(paragraphs)
    word_count = count_words(content)
    return Section(
        identifier=identifier,
        level=level,
        title=title,
        heading=heading,
        numbering=numbering,
        content=content,
        paragraphs=paragraphs,
        sentences=sentences,
        word_count=word_count,
    )


def build_section_tree(sections: list[Section]) -> list[Section]:
    """Build a hierarchy of sections from a flat sequence."""

    root_sections: list[Section] = []
    stack: list[Section] = []
    counters: dict[int, int] = {}
    for section in sections:
        section.children = []
        counters[section.level] = counters.get(section.level, 0) + 1
        for deeper_level in list(counters):
            if deeper_level > section.level:
                counters.pop(deeper_level)
        section.identifier = ".".join(
            str(counters[level]) for level in sorted(counters) if level <= section.level
        )
        while stack and stack[-1].level >= section.level:
            stack.pop()
        if stack:
            stack[-1].children.append(section)
        else:
            root_sections.append(section)
        stack.append(section)
    return root_sections


def collect_document_paragraphs(
    front_matter: str, sections: list[Section]
) -> list[Paragraph]:
    """Collect document paragraphs excluding heading lines."""

    paragraph_texts: list[str] = []
    if front_matter.strip():
        paragraph_texts.extend(split_paragraphs(front_matter))
    for section in sections:
        paragraph_texts.extend(paragraph.text for paragraph in section.paragraphs)
    combined_text = "\n\n".join(paragraph_texts)
    return build_paragraphs(combined_text)


def build_paragraphs(text: str) -> list[Paragraph]:
    """Split text into non-empty paragraphs with sentence objects."""

    paragraphs: list[Paragraph] = []
    for index, part in enumerate(re.split(r"\n\s*\n", text), start=1):
        paragraph_text = part.strip()
        if not paragraph_text:
            continue
        sentence_objects = build_sentence_objects(paragraph_text)
        paragraphs.append(
            Paragraph(
                index=index,
                text=paragraph_text,
                sentences=sentence_objects,
                word_count=count_words(paragraph_text),
            )
        )
    return paragraphs


def build_sentence_objects(text: str) -> list[Sentence]:
    """Split text into sentence objects."""

    sentence_parts = [
        part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()
    ]
    return [
        Sentence(index=index, text=part)
        for index, part in enumerate(sentence_parts, start=1)
    ]


def flatten_sentence_text(paragraphs: list[Paragraph]) -> list[str]:
    """Flatten sentence objects to plain strings for compatibility."""

    return [
        sentence.text for paragraph in paragraphs for sentence in paragraph.sentences
    ]


def split_paragraphs(text: str) -> list[str]:
    """Split text into non-empty paragraph strings."""

    return [paragraph.text for paragraph in build_paragraphs(text)]


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentence strings."""

    return flatten_sentence_text(build_paragraphs(text))


def find_abstract(sections: list[Section], front_matter: str) -> str:
    """Extract abstract-like content from the document."""

    for section in sections:
        normalized = section.title.strip().lower()
        if normalized in {"摘要", "abstract"}:
            return section.content
    return front_matter


def count_words(text: str) -> int:
    """Count Chinese characters and Latin tokens as a rough word count."""

    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    latin_tokens = re.findall(r"[A-Za-z0-9_]+", text)
    return len(chinese_chars) + len(latin_tokens)
