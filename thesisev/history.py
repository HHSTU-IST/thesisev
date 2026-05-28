"""Lightweight local history storage for thesis evaluations."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from thesisev.models import EvaluationResult

HISTORY_DIR = Path(__file__).resolve().parent / "data"
HISTORY_PATH = HISTORY_DIR / "history.json"
MAX_HISTORY_ITEMS = 20


def append_history(result: EvaluationResult) -> None:
    """Append a compact evaluation summary to local history."""

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    items = read_history()
    items.insert(0, build_history_entry(result))
    HISTORY_PATH.write_text(
        json.dumps(items[:MAX_HISTORY_ITEMS], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_history() -> list[dict[str, Any]]:
    """Read stored history entries from disk."""

    if not HISTORY_PATH.exists():
        return []
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))


def build_history_entry(result: EvaluationResult) -> dict[str, Any]:
    """Build a compact serialized history entry."""

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "title": result.document.title,
        "source_type": result.document.source_type,
        "score": result.score,
        "comment": result.comment,
        "issue_count": len(result.issues),
        "topic_relevance_ratio": result.topic_relevance_ratio,
        "technology_stack": result.technology_stack,
        "model": result.metadata.get("model", {}),
    }
