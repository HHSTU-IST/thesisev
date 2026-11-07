"""Tunable numeric operating knobs loaded from ``config/settings.toml``.

The pipeline carries a long tail of small numeric constants -- upload size
caps, background worker counts, prompt-evidence budgets, DOCX zip-bomb guards
and deduction ceilings -- that operators occasionally need to retune without
editing code.  Those values are gathered here so they can be overridden from a
single TOML table, while the *logic* that reads them stays in each module.

Scope is intentionally narrow: only plain numeric tuning knobs live here.
Regular expressions, identity strings/labels, category names, ``Form``/``File``
default objects and derived paths deliberately remain in the modules where they
are used, because externalising them would not make them any easier to reason
about.

The loader is forgiving by design.  A missing file, an unreadable file or
malformed TOML is never fatal: every knob falls back to the built-in default
mirrored in :data:`DEFAULTS`, with one warning so a typo cannot silently change
behaviour *and* stay invisible.  A key present with the wrong type (a string
where an integer is expected, or a ``bool`` masquerading as a number) falls back
to that key's default with a warning in the same way.

:data:`DEFAULTS` is the single source of truth for the fallback behaviour, and
its numbers are kept identical to the literals they replaced so that extraction
is behaviour-preserving even when ``config/settings.toml`` is absent.
"""

from __future__ import annotations

import logging
import tomllib
from typing import Any

from thesisev.paths import config_dir

logger = logging.getLogger(__name__)

#: Filename (inside the config directory) that may override the defaults below.
SETTINGS_FILENAME = "settings.toml"

#: Built-in defaults mirroring the literals that used to sit in each module.
#: These MUST stay equal to the previous hard-coded numbers: the bundled
#: ``config/settings.toml`` reproduces them, and this table is what the code
#: falls back to when that file is missing, unreadable or malformed.
DEFAULTS: dict[str, dict[str, int | float]] = {
    "api": {
        "max_history_items": 20,
        "max_upload_bytes": 20 * 1024 * 1024,
        "upload_read_chunk_bytes": 1024 * 1024,
        "evaluate_max_workers": 2,
        "max_jobs": 20,
        "job_timeout_seconds": 300,
    },
    "llm": {
        "fallback_max_tokens": 400,
        "min_recommended_local_context": 16384,
    },
    "deep_review": {
        "default_max_findings": 6,
        "section_excerpt_limit": 320,
        "max_sections_in_prompt": 30,
    },
    "scoring": {
        "max_context_sections": 20,
        "section_excerpt_limit": 600,
        "abstract_excerpt_limit": 1200,
    },
    "scoring_format": {
        "max_format_deduction_ratio": 0.8,
        "max_rule_penalty": 3.0,
        "docx_paragraph_compliance_ratio": 0.95,
    },
    "parser": {
        "max_archive_bytes": 25 * 1024 * 1024,
        "max_member_count": 512,
        "max_total_uncompressed_bytes": 80 * 1024 * 1024,
        "max_member_bytes": 30 * 1024 * 1024,
        "max_compression_ratio": 100,
        "max_text_paragraphs": 5000,
        "max_snapshot_paragraphs": 5000,
        "max_snapshot_tables": 500,
        "max_snapshot_sections": 200,
        "max_runs_per_paragraph": 500,
    },
    "logic_review": {
        "min_sections_for_logic": 3,
        "min_claims_for_consistency": 4,
        "max_consistency_issues": 8,
    },
}


class _Missing:
    """Sentinel type marking "the key was not in the file" (vs. a real value)."""


_MISSING = _Missing()

#: Parsed contents of ``settings.toml`` (empty dict when absent/invalid).
_RAW: dict[str, Any] = {}


def _load_raw() -> dict[str, Any]:
    """Read and parse the TOML file, degrading to ``{}`` on any failure."""

    config_path = config_dir() / SETTINGS_FILENAME
    try:
        with config_path.open("rb") as file:
            return tomllib.load(file)
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning(
            "settings file %s unreadable (%s); using built-in defaults",
            config_path,
            exc,
        )
        return {}


def reload() -> None:
    """(Re)read ``settings.toml`` into the module cache.

    Called once at import so the module-level constants in dependent modules
    resolve against the on-disk overrides; tests may point :func:`config_dir`
    at a temporary directory and call this again to exercise overrides.
    """

    global _RAW
    _RAW = _load_raw()


def _default_value(section: str, key: str) -> int | float:
    try:
        return DEFAULTS[section][key]
    except KeyError as exc:
        raise KeyError(f"unknown settings knob: {section}.{key}") from exc


def _lookup(section: str, key: str) -> Any:
    section_table = _RAW.get(section)
    if isinstance(section_table, dict) and key in section_table:
        return section_table[key]
    return _MISSING


def _warn_bad_type(section: str, key: str, value: object, expected: str) -> None:
    logger.warning(
        "settings knob %s.%s expected %s but got %s (%r); using default",
        section,
        key,
        expected,
        type(value).__name__,
        value,
    )


def get_int(section: str, key: str) -> int:
    """Return an integer knob, falling back to the default on any mismatch."""

    default = int(_default_value(section, key))
    value = _lookup(section, key)
    if value is _MISSING:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        _warn_bad_type(section, key, value, "an integer")
        return default
    return value


def get_float(section: str, key: str) -> float:
    """Return a float knob, accepting ints too, falling back on any mismatch."""

    default = float(_default_value(section, key))
    value = _lookup(section, key)
    if value is _MISSING:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _warn_bad_type(section, key, value, "a number")
        return default
    return float(value)


reload()
