"""Shared fixtures for thesis evaluation tests."""

from __future__ import annotations

from typing import Any

import pytest

from thesisev.models import Paragraph, Section, Sentence, ThesisDocument


def build_paragraph(
    text: str,
    *,
    index: int = 0,
    skip_format_check: bool = False,
) -> Paragraph:
    """Build a minimal paragraph model with naive sentence splitting."""
    sentences = [
        Sentence(index=offset, text=part)
        for offset, part in enumerate(
            part.strip()
            for part in text.replace("。", "。\n").split("\n")
            if part.strip()
        )
    ]
    return Paragraph(
        index=index,
        text=text,
        sentences=sentences,
        word_count=len(text),
        skip_format_check=skip_format_check,
    )


def build_section(
    title: str,
    content: str,
    *,
    level: int = 1,
    identifier: str | None = None,
    children: list[Section] | None = None,
) -> Section:
    """Build a minimal section model."""
    paragraphs = [build_paragraph(content, index=0)]
    return Section(
        identifier=identifier or title,
        level=level,
        title=title,
        heading=title,
        numbering="",
        content=content,
        paragraphs=paragraphs,
        sentences=[
            sentence.text
            for paragraph in paragraphs
            for sentence in paragraph.sentences
        ],
        word_count=len(content),
        children=children or [],
        subtree_word_count=len(content),
    )


def build_thesis_document(
    *,
    title: str = "基于边缘计算的目标检测系统设计与实现",
    text: str = "",
    source_type: str = "md",
    format_snapshot: dict[str, Any] | None = None,
    total_word_count: int | None = None,
) -> ThesisDocument:
    """Build a ThesisDocument with sane defaults for scoring tests."""
    cleaned = text or (
        "摘要 本文研究基于边缘计算的目标检测系统。"
        "Abstract This paper studies an edge computing based detection system. "
        + "方案 设计 架构 数据 采集 计算 处理 结果 验证 测试 分析 论证 对比 评估 推导 "
        * 2
        + "因此 表明 综上 依据 分析 创新 提出 改进 优化 应用价值 意义 " * 2
    )
    section = build_section("1 系统总体设计", "正文内容 边缘计算 目标检测 系统实现")
    return ThesisDocument(
        title=title,
        source_path="thesis." + source_type,
        source_type=source_type,
        raw_text=cleaned,
        cleaned_text=cleaned,
        front_matter="",
        abstract="摘要 边缘计算 目标检测",
        sections=[section],
        root_sections=[section],
        paragraphs=[build_paragraph(section.content)],
        sentences=[],
        total_word_count=total_word_count
        if total_word_count is not None
        else len(cleaned),
        format_snapshot=format_snapshot or {},
    )


def build_docx_paragraph(
    text: str,
    *,
    style: str = "Normal",
    font: str = "宋体",
    size: float = 12.0,
    spacing: float = 1.5,
    space_before: float = 0.0,
    space_after: float = 0.0,
    indent: float | None = 24.0,
) -> dict[str, Any]:
    """Build one docx snapshot paragraph entry for format-rule tests."""
    return {
        "text": text,
        "style": style,
        "run": {"font_name": font, "font_size_pt": size},
        "line_spacing": spacing,
        "space_before_pt": space_before,
        "space_after_pt": space_after,
        "first_line_indent_pt": indent,
    }


@pytest.fixture
def thesis_document() -> ThesisDocument:
    """A basic markdown thesis document used across scoring tests."""
    return build_thesis_document()
