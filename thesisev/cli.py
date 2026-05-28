"""CLI entrypoint for thesisev."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thesisev.service import evaluate_document


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="thesisev",
        description="Evaluate a thesis-like text file and produce a basic report.",
    )
    parser.add_argument("path", help="Path to a txt or markdown thesis file.")
    return parser


def main() -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args()
    source = Path(args.path)
    if not source.exists():
        parser.error(f"file not found: {source}")
    result = evaluate_document(source)
    print_report(result)
    return 0


def print_report(result) -> None:
    """Print a human-readable evaluation report."""

    print(f"Title: {result.document.title}")
    print(f"Source: {result.document.source_path}")
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


if __name__ == "__main__":
    sys.exit(main())
