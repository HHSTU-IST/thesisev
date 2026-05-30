"""Parsing and preprocessing utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from defusedxml import ElementTree

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
    """Load and parse a thesis from markdown or docx."""

    source_path = Path(path)
    raw_text = read_source_text(source_path)
    format_snapshot = (
        read_docx_format_snapshot(source_path)
        if source_path.suffix.lower() == ".docx"
        else {}
    )
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
    if isinstance(format_snapshot, dict):
        format_snapshot["word_count"] = total_word_count
        format_snapshot["section_count"] = len(sections)

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
        format_snapshot=format_snapshot,
    )


def read_source_text(path: Path) -> str:
    """Read thesis text from a supported source file."""

    suffix = path.suffix.lower()
    if suffix == ".md":
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


def read_docx_format_snapshot(path: Path) -> dict[str, Any]:
    """Extract a compact formatting snapshot from a docx file."""

    with ZipFile(path) as archive:
        document_xml = ElementTree.fromstring(archive.read("word/document.xml"))

    paragraphs: list[dict[str, Any]] = []
    tables = [
        extract_docx_table_snapshot(table)
        for table in document_xml.findall(".//w:tbl", WORD_NAMESPACE)
    ]
    sections = [
        extract_docx_section_snapshot(sect_pr)
        for sect_pr in document_xml.findall(".//w:sectPr", WORD_NAMESPACE)
    ]

    paragraphs.extend(
        snapshot
        for paragraph in document_xml.findall(".//w:p", WORD_NAMESPACE)
        if (snapshot := extract_docx_paragraph_snapshot(paragraph))["text"]
        or snapshot["style"]
        or snapshot["runs"]
    )

    if not sections:
        sections.append({})

    return {
        "paragraphs": paragraphs,
        "tables": tables,
        "sections": sections,
    }


def extract_docx_paragraph_snapshot(paragraph: ElementTree.Element) -> dict[str, Any]:
    """Extract a paragraph-level formatting snapshot."""

    paragraph_props = paragraph.find("w:pPr", WORD_NAMESPACE)
    runs: list[dict[str, Any]] = []
    for run in paragraph.findall("w:r", WORD_NAMESPACE):
        run_snapshot = extract_docx_run_snapshot(run)
        if run_snapshot["text"] or any(
            run_snapshot[key] is not None
            for key in ("font_name", "font_size_pt", "bold")
        ):
            runs.append(run_snapshot)

    text = "".join(run.get("text", "") for run in runs).strip()
    primary_run = next(
        (run for run in runs if run.get("text")), runs[0] if runs else {}
    )
    return {
        "text": text,
        "style": normalize_docx_style_name(
            get_docx_attr_value(paragraph_props, "pStyle")
        ),
        "alignment": normalize_docx_alignment(
            get_docx_attr_value(paragraph_props, "jc")
        ),
        "line_spacing": parse_docx_line_spacing(paragraph_props),
        "space_before_pt": parse_docx_twips(
            get_docx_child(paragraph_props, "spacing"), "before"
        ),
        "space_after_pt": parse_docx_twips(
            get_docx_child(paragraph_props, "spacing"), "after"
        ),
        "first_line_indent_pt": parse_docx_indentation(
            get_docx_child(paragraph_props, "ind"), "firstLine"
        ),
        "runs": runs,
        "run": primary_run,
    }


def extract_docx_run_snapshot(run: ElementTree.Element) -> dict[str, Any]:
    """Extract a run-level formatting snapshot."""

    run_props = run.find("w:rPr", WORD_NAMESPACE)
    text = "".join(
        node.text or "" for node in run.findall("w:t", WORD_NAMESPACE)
    ).strip()
    return {
        "text": text,
        "font_name": parse_docx_font_name(run_props),
        "font_size_pt": parse_docx_font_size(run_props),
        "bold": parse_docx_bool(get_docx_child(run_props, "b")),
    }


def extract_docx_table_snapshot(table: ElementTree.Element) -> dict[str, Any]:
    """Extract a table-level formatting snapshot."""

    table_props = table.find("w:tblPr", WORD_NAMESPACE)
    return {
        "alignment": normalize_docx_alignment(get_docx_attr_value(table_props, "jc")),
    }


def extract_docx_section_snapshot(section: ElementTree.Element) -> dict[str, Any]:
    """Extract a section-level formatting snapshot."""

    margins = get_docx_child(section, "pgMar")
    return {
        "top_margin_pt": parse_docx_twips(margins, "top"),
        "bottom_margin_pt": parse_docx_twips(margins, "bottom"),
        "left_margin_pt": parse_docx_twips(margins, "left"),
        "right_margin_pt": parse_docx_twips(margins, "right"),
    }


def get_docx_child(
    parent: ElementTree.Element | None, name: str
) -> ElementTree.Element | None:
    """Return a child element by local name."""

    if parent is None:
        return None
    return parent.find(f"w:{name}", WORD_NAMESPACE)


def get_docx_attr_value(parent: ElementTree.Element | None, name: str) -> str | None:
    """Return the w:val attribute from a docx child element."""

    child = get_docx_child(parent, name)
    if child is None:
        return None
    value = child.get(f"{{{WORD_NAMESPACE['w']}}}val")
    if value is None:
        return None
    return value.strip()


def parse_docx_bool(element: ElementTree.Element | None) -> bool | None:
    """Parse a w:b-like boolean element."""

    if element is None:
        return None
    value = element.get(f"{{{WORD_NAMESPACE['w']}}}val")
    if value is None:
        return True
    return value not in {"0", "false", "off"}


def parse_docx_font_name(run_props: ElementTree.Element | None) -> str | None:
    """Parse the first usable font name from run properties."""

    fonts = get_docx_child(run_props, "rFonts")
    if fonts is None:
        return None
    for attr in ("eastAsia", "ascii", "hAnsi"):
        value = fonts.get(f"{{{WORD_NAMESPACE['w']}}}{attr}")
        if value:
            return value.strip()
    return None


def parse_docx_font_size(run_props: ElementTree.Element | None) -> float | None:
    """Parse run font size in points."""

    size_element = get_docx_child(run_props, "sz")
    if size_element is None:
        return None
    raw_value = size_element.get(f"{{{WORD_NAMESPACE['w']}}}val")
    if not raw_value:
        return None
    try:
        return round(float(raw_value) / 2, 2)
    except ValueError:
        return None


def parse_docx_line_spacing(
    paragraph_props: ElementTree.Element | None,
) -> float | None:
    """Parse paragraph line spacing into a float ratio when possible."""

    spacing = get_docx_child(paragraph_props, "spacing")
    if spacing is None:
        return None
    line_rule = spacing.get(f"{{{WORD_NAMESPACE['w']}}}lineRule")
    raw_value = spacing.get(f"{{{WORD_NAMESPACE['w']}}}line")
    if not raw_value:
        return None
    try:
        numeric_value = float(raw_value)
    except ValueError:
        return None
    if line_rule in {"auto", "autoSpacing"}:
        return round(numeric_value / 240, 2)
    return round(numeric_value / 20, 2)


def parse_docx_indentation(
    paragraph_indent: ElementTree.Element | None, attr_name: str
) -> float | None:
    """Parse paragraph indentation from twips to points."""

    if paragraph_indent is None:
        return None
    raw_value = paragraph_indent.get(f"{{{WORD_NAMESPACE['w']}}}{attr_name}")
    if raw_value is None:
        return None
    try:
        return round(float(raw_value) / 20, 2)
    except ValueError:
        return None


def parse_docx_twips(
    element: ElementTree.Element | None, attr_name: str
) -> float | None:
    """Parse a twips-based docx attribute into points."""

    if element is None:
        return None
    raw_value = element.get(f"{{{WORD_NAMESPACE['w']}}}{attr_name}")
    if raw_value is None:
        return None
    try:
        return round(float(raw_value) / 20, 2)
    except ValueError:
        return None


def normalize_docx_style_name(value: str | None) -> str | None:
    """Normalize a docx style identifier into a display name."""

    if not value:
        return None
    normalized = value.replace("_", " ").strip()
    return re.sub(r"(?<=\D)(\d+)$", r" \1", normalized)


def normalize_docx_alignment(value: str | None) -> str | None:
    """Normalize docx alignment values to a canonical token."""

    if not value:
        return None
    token = value.strip().lower()
    aliases = {
        "both": "justify",
        "justified": "justify",
        "center": "center",
        "centre": "center",
        "left": "left",
        "right": "right",
        "distribute": "distribute",
    }
    return aliases.get(token, token)


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
    headings = [
        HeadingMatch(
            line_index=index,
            level=heading[0],
            title=heading[1],
            numbering=heading[2],
            heading=heading[3],
        )
        for index, raw_line in enumerate(lines)
        if (stripped := raw_line.strip())
        and (heading := match_section_heading(stripped)) is not None
    ]

    if not headings:
        msg = "markdown file must contain explicit heading markers"
        raise ValueError(msg)

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
