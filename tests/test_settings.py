"""Tests for the tunable numeric-knob loader in :mod:`thesisev.settings`.

The loader must be forgiving: a missing or malformed ``config/settings.toml``
falls back to the built-in defaults (with a warning), a per-key override is
honoured, a value of the wrong type falls back to that key's default rather
than crashing, and an unknown knob is a programming error (``KeyError``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from thesisev import settings


@pytest.fixture
def isolated_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Point the loader at an empty temp config dir; restore on teardown."""

    monkeypatch.setattr(settings, "config_dir", lambda: tmp_path)
    yield tmp_path
    # Reload the real on-disk table so no later test sees the temp state.
    settings.reload()


def _write(tmp_path: Path, text: str) -> None:
    (tmp_path / settings.SETTINGS_FILENAME).write_text(text, encoding="utf-8")


def test_missing_file_falls_back_to_defaults(
    isolated_settings: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        settings.reload()
    assert settings.get_int("api", "max_history_items") == 20
    assert "built-in defaults" in caplog.text


def test_override_is_read(isolated_settings: Path) -> None:
    _write(
        isolated_settings,
        "[api]\nmax_history_items = 99\n[scoring_format]\nmax_rule_penalty = 1.5\n",
    )
    settings.reload()
    assert settings.get_int("api", "max_history_items") == 99
    assert settings.get_float("scoring_format", "max_rule_penalty") == 1.5


def test_partial_override_leaves_other_knobs_at_default(
    isolated_settings: Path,
) -> None:
    _write(isolated_settings, "[api]\nmax_jobs = 7\n")
    settings.reload()
    assert settings.get_int("api", "max_jobs") == 7
    assert settings.get_int("api", "max_history_items") == 20


def test_wrong_type_falls_back_to_default(
    isolated_settings: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # bool is not accepted as an int; a string is neither.
    _write(
        isolated_settings,
        '[api]\nmax_history_items = true\nmax_jobs = "twenty"\n',
    )
    with caplog.at_level(logging.WARNING):
        settings.reload()
    assert settings.get_int("api", "max_history_items") == 20
    assert settings.get_int("api", "max_jobs") == 20
    assert "expected an integer" in caplog.text


def test_float_accepts_integer_literal(isolated_settings: Path) -> None:
    _write(isolated_settings, "[scoring_format]\nmax_rule_penalty = 2\n")
    settings.reload()
    value = settings.get_float("scoring_format", "max_rule_penalty")
    assert value == 2.0
    assert isinstance(value, float)


def test_malformed_toml_falls_back_to_defaults(
    isolated_settings: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(isolated_settings, "this is = = not valid toml")
    with caplog.at_level(logging.WARNING):
        settings.reload()
    assert settings.get_int("llm", "fallback_max_tokens") == 400
    assert "built-in defaults" in caplog.text


def test_bundled_file_reproduces_every_default() -> None:
    """The shipped config/settings.toml must equal the code defaults exactly."""

    settings.reload()
    for section, knobs in settings.DEFAULTS.items():
        for key, default in knobs.items():
            if isinstance(default, float):
                assert settings.get_float(section, key) == default
            else:
                got = settings.get_int(section, key)
                assert got == default
                assert isinstance(got, int)


def test_unknown_knob_raises() -> None:
    with pytest.raises(KeyError, match="unknown settings knob"):
        settings.get_int("api", "does_not_exist")
    with pytest.raises(KeyError, match="unknown settings knob"):
        settings.get_float("nope", "nope")
