"""Regression tests for the optional LLM deep review module.

The deep review must be strictly additive: no model configured or any LLM
failure degrades to an empty list, findings map onto the existing Issue UI
shape, and duplicates against deterministic issues are dropped.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from tests.conftest import build_section, build_thesis_document
from thesisev.deep_review import (
    CATEGORY_LOGIC,
    CATEGORY_TONE,
    build_deep_review_prompt,
    normalize_deep_review_findings,
    parse_deep_review_json,
    run_deep_review,
    to_deep_review_issue,
)
from thesisev.llm import ModelConfig
from thesisev.models import Issue


def _always_available(_self: Any) -> bool:
    """is_available() replacement that always reports the model as ready."""
    return True


def _fake_model_factory(model_cls: type[Any]) -> Any:
    """Build a create_chat_model stand-in that ignores its config argument."""

    def factory(_config: Any) -> Any:
        return model_cls()

    return factory


def _available_config(monkeypatch: pytest.MonkeyPatch) -> ModelConfig:
    """Return a ModelConfig whose availability check passes via monkeypatch."""
    config = ModelConfig()
    monkeypatch.setattr(ModelConfig, "is_available", _always_available)
    return config


def _existing_issue(message: str) -> Issue:
    return Issue(
        category=CATEGORY_LOGIC,
        rule_id="conclusion_without_evidence",
        severity="medium",
        message=message,
        suggestion="补充支撑证据。",
        section_identifier="3 实验分析",
        section_title="3 实验分析",
        paragraph_index=-1,
        sentence_index=-1,
        matched_text="",
        excerpt="",
    )


def test_prompt_is_bounded_and_parsable() -> None:
    """Prompt payload caps sections, clips excerpts and stays valid JSON."""
    sections = [
        build_section(f"第{i}章 长标题内容", "内容。" * 200, level=1) for i in range(45)
    ]
    document = build_thesis_document()
    document.sections = sections
    prompt = build_deep_review_prompt(document)
    # Embedded JSON payload after the final line must parse.
    payload_text = prompt.split("\n")[-1].removeprefix("论文信息：")
    payload = json.loads(payload_text)
    assert isinstance(payload, dict)
    assert len(payload["sections"]) == 30  # capped at MAX_SECTIONS_IN_PROMPT
    long_excerpt = payload["sections"][0]["excerpt"]
    assert len(long_excerpt) <= 320 + 3  # clip limit plus ellipsis
    assert payload["abstract"]  # bounded excerpt of abstract


def test_parse_deep_review_json_accepts_strict_payload() -> None:
    raw = json.dumps(
        {"findings": [{"type": "logic", "severity": "high", "message": "矛盾"}]},
        ensure_ascii=False,
    )
    response = SimpleNamespace(content=raw)
    assert parse_deep_review_json(response)["findings"]


def test_parse_deep_review_json_rejects_bad_payloads() -> None:
    for content in ("", "no json here", "[]"):
        with pytest.raises(ValueError):
            parse_deep_review_json(SimpleNamespace(content=content))
    # content list shape also handled by extract_response_text
    with pytest.raises(ValueError):
        parse_deep_review_json(SimpleNamespace(content=[]))
    # wrapped in markdown fences must still parse
    wrapped = SimpleNamespace(content='```json\n{"findings": []}\n```')
    assert parse_deep_review_json(wrapped) == {"findings": []}
    # structural parse only guarantees an object; malformed findings are
    # filtered later by normalize_deep_review_findings
    non_list = parse_deep_review_json(SimpleNamespace(content='{"findings": 1}'))
    assert normalize_deep_review_findings(non_list) == []


def test_normalize_findings_filters_invalid_entries() -> None:
    payload = {
        "findings": [
            {"type": "logic", "severity": "high", "message": "ok", "suggestion": "改"},
            {"type": "bogus", "severity": "high", "message": "dropped type"},
            {"type": "logic", "severity": "extreme", "message": "dropped severity"},
            {"type": "tone", "severity": "low", "message": ""},
            {"type": "tone", "severity": "medium", "message": "no suggestion"},
            "not-a-dict",
        ]
    }
    findings = normalize_deep_review_findings(payload)
    assert len(findings) == 2
    assert findings[0]["type"] == "logic"
    # missing suggestion gets a safe default
    assert findings[1]["suggestion"]
    # empty / invalid findings -> []
    assert normalize_deep_review_findings({}) == []
    assert normalize_deep_review_findings({"findings": "nope"}) == []


def test_to_deep_review_issue_maps_category_and_section() -> None:
    document = build_thesis_document()
    section = build_section("3 实验分析", "实验数据支撑结论。", level=1)
    document.sections = [section]

    logic = to_deep_review_issue(
        document=document,
        finding={
            "type": "logic",
            "severity": "high",
            "message": "结论缺数据",
            "suggestion": "补",
            "section": "3 实验分析",
        },
    )
    assert logic.category == CATEGORY_LOGIC
    assert logic.section_title == "3 实验分析"
    assert logic.rule_id == "llm_logic"

    tone = to_deep_review_issue(
        document=document,
        finding={
            "type": "tone",
            "severity": "low",
            "message": "程度词滥用",
            "suggestion": "删",
            "section": "不存在的章节",
        },
    )
    assert tone.category == CATEGORY_TONE
    assert tone.rule_id == "llm_tone"
    assert tone.section_identifier == ""  # unmatched label is safe


def test_run_deep_review_skips_when_model_unavailable() -> None:
    """A local engine demoted by the runtime probe never reaches the model."""
    document = build_thesis_document()
    config = ModelConfig()
    # The default engine is served from this host and needs no credential, so
    # it only counts as unavailable once the runtime probe has reported the
    # server down.  Availability must not be asserted through the environment:
    # without the demotion the call goes out to whatever Ollama happens to be
    # listening on this host.
    assert config.is_available()
    config.runtime_reachable = False
    assert not config.is_available()
    assert run_deep_review(document=document, model_config=config) == []


def test_run_deep_review_calls_a_reachable_local_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probed-reachable local engine is available and does issue the call."""
    document = build_thesis_document()
    config = ModelConfig()
    config.runtime_reachable = True
    assert config.is_available()
    calls: list[Any] = []

    class FakeModel:
        def invoke(self, messages: list[Any]) -> SimpleNamespace:
            calls.append(messages)
            return SimpleNamespace(content='{"findings": []}')

    monkeypatch.setattr(
        "thesisev.deep_review.create_chat_model",
        _fake_model_factory(FakeModel),
    )
    assert run_deep_review(document=document, model_config=config) == []
    assert len(calls) == 1


def test_run_deep_review_degrades_on_llm_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = build_thesis_document()
    config = _available_config(monkeypatch)

    def _boom(
        model: Any,
        messages: list[Any],
        *,
        attempts: int = 3,
        base_delay: float = 1.0,
    ) -> Any:
        raise RuntimeError("provider down")

    monkeypatch.setattr("thesisev.deep_review.invoke_chat_model_with_retry", _boom)
    assert run_deep_review(document=document, model_config=config) == []


def test_run_deep_review_merges_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = build_thesis_document()
    section = build_section("2 系统设计", "设计了缓存层。", level=1)
    document.sections = [section]
    config = _available_config(monkeypatch)

    findings = {
        "findings": [
            {
                "type": "logic",
                "severity": "high",
                "message": "设计结论缺少数据支撑",
                "suggestion": "补实验",
                "section": "2 系统设计",
            },
            {
                "type": "tone",
                "severity": "low",
                "message": "『特别』在正式句中使用",
                "suggestion": "删",
                "section": "",
            },
        ]
    }

    class FakeModel:
        def invoke(self, messages: list[Any]) -> SimpleNamespace:
            return SimpleNamespace(content=json.dumps(findings, ensure_ascii=False))

    monkeypatch.setattr(
        "thesisev.deep_review.create_chat_model",
        _fake_model_factory(FakeModel),
    )
    added = run_deep_review(
        document=document,
        model_config=config,
        existing_issues=[
            _existing_issue("设计结论缺少数据支撑")
        ],  # same message dropped
        max_findings=6,
    )
    assert len(added) == 1  # tone finding added, logic dup dropped
    assert added[0].category == CATEGORY_TONE


def test_run_deep_review_respects_max_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = build_thesis_document()
    config = _available_config(monkeypatch)

    class FakeModel:
        def invoke(self, messages: list[Any]) -> SimpleNamespace:
            raw = {
                "findings": [
                    {
                        "type": "logic",
                        "severity": "low",
                        "message": f"问题 {i}",
                        "suggestion": "改",
                        "section": "",
                    }
                    for i in range(8)
                ]
            }
            return SimpleNamespace(content=json.dumps(raw, ensure_ascii=False))

    monkeypatch.setattr(
        "thesisev.deep_review.create_chat_model",
        _fake_model_factory(FakeModel),
    )
    added = run_deep_review(document=document, model_config=config, max_findings=3)
    assert len(added) == 3
