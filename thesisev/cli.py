"""CLI entrypoint for thesisev."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from thesisev.service import evaluate_document, structure_document


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="thesisev",
        description="Analyze a thesis-like file and produce a structure or evaluation report.",
    )
    parser.add_argument("path", help="Path to a txt, markdown, or docx thesis file.")
    parser.add_argument(
        "--output",
        choices=("report", "structure"),
        default="report",
        help="Choose whether to output the evaluation report or the parsed structure.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the human-readable output.",
    )
    return parser


def main() -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args()
    source = Path(args.path)
    if not source.exists():
        parser.error(f"file not found: {source}")
    if args.output == "structure":
        document = structure_document(source)
        if args.json:
            print(json.dumps(document.to_dict(), ensure_ascii=False, indent=2))
        else:
            print_structure(document)
        return 0

    result = evaluate_document(source)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print_report(result)
    return 0


def print_report(result) -> None:
    """Print a human-readable evaluation report."""

    print(f"Title: {result.document.title}")
    print(f"Source: {result.document.source_path}")
    print(f"Type: {result.document.source_type}")
    print()
    print("Statistics:")
    for item in result.statistics:
        print(f"- {item.label}: {item.value}")
    print()
    print("Keywords:")
    print(f"- {'、'.join(result.keywords) if result.keywords else 'None'}")
    print()
    print("Technology Stack:")
    print(
        f"- {'、'.join(result.technology_stack) if result.technology_stack else 'None'}"
    )
    print()
    print(f"Score: {result.score}")
    print("Comment:")
    print(result.comment)
    print()
    print("Issues:")
    if not result.issues:
        print("- None")
        return
    for issue in result.issues:
        print(
            f"- [{issue.category}] {issue.section_title}: {issue.message}\n"
            f"  Suggestion: {issue.suggestion}\n"
            f"  Excerpt: {issue.excerpt}"
        )


def print_structure(document) -> None:
    """Print a human-readable structure report."""

    print(f"Title: {document.title}")
    print(f"Source: {document.source_path}")
    print(f"Type: {document.source_type}")
    print(f"Total Words: {document.total_word_count}")
    print(f"Paragraphs: {len(document.paragraphs)}")
    print(f"Sentences: {len(document.sentences)}")
    print()
    if document.front_matter:
        print("Front Matter:")
        print(document.front_matter)
        print()
    print("Section Tree:")
    for section in document.root_sections:
        print_section_tree(section)


def print_section_tree(section, indent: int = 0) -> None:
    """Print a section and its children recursively."""

    prefix = "  " * indent
    print(
        f"{prefix}- [{section.identifier}] {section.title} "
        f"(L{section.level}, paragraphs={len(section.paragraphs)}, words={section.word_count})"
    )
    for child in section.children:
        print_section_tree(child, indent + 1)


if __name__ == "__main__":
    sys.exit(main())
