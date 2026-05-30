"""Shared path helpers for source and installed package layouts."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the active project root for source or installed usage."""

    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists() and (cwd / "thesisev").is_dir():
        return cwd
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Return the runtime data directory."""

    return project_root() / "data"


def static_dir() -> Path:
    """Return the static assets directory."""

    return project_root() / "static"


def templates_dir() -> Path:
    """Return the templates directory."""

    return project_root() / "templates"
