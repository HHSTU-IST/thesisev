"""B1: LLM 链路韧性 — 重试 / 单项目降级 / 分数钳制 / 截断可见性。

另覆盖两个只在真实远程调用中暴露的缺陷：输出预算过小导致 JSON 被截断而
静默降级；单项目 prompt 的「score 为百分制」使模型把逐项分数也写成百分数，
从而被钳制到满分、总分虚高。
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from thesisev.llm import ModelConfig, invoke_chat_model_with_retry
from thesisev.models import ThesisDocument
from thesisev.rubric_utils import RubricItem, normalize_rubric_items
from thesisev.scoring import (
    build_score_prompt,
    normalize_llm_score_criteria,
    score_rubric_items,
    validate_llm_deductions,
)

WRITING_ITEM = RubricItem(
    name="论文质量",
    key="writing_quality",
    standards=["表达规范"],
    evaluation="llm",
    max_score=20.0,
)


def _always_available(_self: ModelConfig) -> bool:
    """is_available() replacement that always reports the model as ready."""
    return True


def make_document() -> ThesisDocument:
    return ThesisDocument(
        title="示例论文",
        source_path="smoke.md",
        source_type="md",
        raw_text="正文内容",
        cleaned_text="正文内容",
        front_matter="",
        abstract="",
        sections=[],
        root_sections=[],
        paragraphs=[],
        sentences=[],
        total_word_count=1000,
    )


class _FakeModelConfig:
    """Model config stub whose availability is forced by the test."""

    def __init__(self, available: bool = True) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeModel:
    """A chat-model double that raises ``n`` times then returns a value."""

    def __init__(self, failures: int, value: str = "ok") -> None:
        self.failures = failures
        self.value = value
        self.calls = 0

    def invoke(self, messages: list[Any]) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError(f"transient failure {self.calls}")
        return self.value


def test_retry_transient_then_success() -> None:
    model = _FakeModel(failures=2)
    result = invoke_chat_model_with_retry(model, [], attempts=3, base_delay=0.0)
    assert result == "ok"
    assert model.calls == 3


def test_retry_exhausted_raises_last_error() -> None:
    model = _FakeModel(failures=5)
    with pytest.raises(ConnectionError, match="transient failure"):
        invoke_chat_model_with_retry(model, [], attempts=3, base_delay=0.0)
    assert model.calls == 3


def test_retry_single_attempt_no_extra_calls() -> None:
    model = _FakeModel(failures=0)
    assert invoke_chat_model_with_retry(model, [], attempts=1) == "ok"
    assert model.calls == 1


def test_llm_item_degrades_to_local_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("thesisev.scoring.score_item_with_llm", boom)
    monkeypatch.setattr(ModelConfig, "is_available", _always_available)
    criteria = score_rubric_items(
        document=make_document(),
        topic_analysis={"document_ratio": 0.5, "topic_keywords": []},
        format_issues=[],
        writing_issues=[],
        content_context={},
        keywords=[],
        technology_details=[],
        format_requirements=None,
        rubric_items=[WRITING_ITEM],
        rubric_source="score_thesis_tech.json",
        rubric_filename="score_thesis_tech.json",
        model_config=ModelConfig(),
    )
    assert len(criteria) == 1
    criterion = criteria[0]
    # Transparent degradation marker, never a hard 500.
    assert criterion.evaluation == "llm_fallback_local"
    assert criterion.key == "writing_quality"
    assert 0 <= criterion.score <= criterion.max_score


def test_clamp_llm_score_to_max_score() -> None:
    payload: dict[str, Any] = {
        "criteria": [
            {
                "name": "论文质量",
                "score": 25.0,
                "evidence": [],
                "deductions": ["示例扣分"],
                "suggestions": [],
            }
        ]
    }
    criteria = normalize_llm_score_criteria(payload, [WRITING_ITEM])
    assert criteria[0].score == WRITING_ITEM.max_score
    assert any("超出" in note for note in criteria[0].evidence)


def test_clamp_llm_negative_score_to_zero() -> None:
    payload = {"criteria": [{"name": "论文质量", "score": -3, "deductions": ["示例"]}]}
    criteria = normalize_llm_score_criteria(payload, [WRITING_ITEM])
    assert criteria[0].score == 0.0


def test_score_prompt_pins_each_item_to_its_own_scale() -> None:
    """A one-item prompt must not leave the item scale open to percentage.

    The top-level field and the per-item field are both named ``score``.  A
    bare "``score`` 为百分制总分" was read as the scale of the requested item,
    and models answered 60 / 78 / 80 against a ``max_score`` of 10 or 5 -- every
    score clamped to full marks and the report inflated to a near-perfect total.
    """

    prompt = build_score_prompt(
        content_context={"title": "示例论文"},
        rubric_items=[WRITING_ITEM],
        rubric_filename="score_thesis_tech.json",
    )
    assert "[0, 该项 max_score]" in prompt
    assert "不得写成百分制" in prompt
    assert "顶层的 score 才是百分制总分" in prompt
    # The item's real ceiling must still reach the model as a concrete number.
    assert '"max_score": 20.0' in prompt


class _FakeStubResponse:
    """Response double carrying only the provider's finish reason."""

    def __init__(self, finish_reason: str) -> None:
        self.response_metadata = {"finish_reason": finish_reason}


class _FakeFinishingModel:
    """Chat-model double that reports how the provider ended the reply."""

    def __init__(self, finish_reason: str) -> None:
        self.finish_reason = finish_reason
        self.calls = 0

    def invoke(self, messages: list[Any]) -> _FakeStubResponse:
        self.calls += 1
        return _FakeStubResponse(self.finish_reason)


def test_truncated_output_is_logged_instead_of_staying_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``finish_reason=length`` must be visible: it only shows up as a parse error.

    The reply simply ends mid-string, so the caller degrades to the local rules
    while the report still looks complete -- the cause has to be logged, and the
    call is not retried because the budget, not the transport, was the problem.
    """

    model = _FakeFinishingModel("length")
    with caplog.at_level(logging.WARNING, logger="thesisev.llm"):
        invoke_chat_model_with_retry(model, [], attempts=3, base_delay=0.0)
    assert model.calls == 1
    assert "finish_reason=length" in caplog.text


def test_complete_output_logs_no_truncation_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _FakeFinishingModel("stop")
    with caplog.at_level(logging.WARNING, logger="thesisev.llm"):
        invoke_chat_model_with_retry(model, [], attempts=1)
    assert "finish_reason" not in caplog.text


def test_below_max_requires_deductions() -> None:
    with pytest.raises(ValueError, match="deductions are empty"):
        validate_llm_deductions(
            criterion_name="论文质量", score=12.0, max_score=20.0, deductions=[]
        )


def test_rubric_items_parse_unknown_evaluation_ignored() -> None:
    """Uploaded rubric items must not break validation when evaluation is llm."""
    items = normalize_rubric_items(
        [
            {
                "criterion": "论文质量",
                "key": "writing_quality",
                "standard": [],
                "score": 20,
                "evaluation": "llm",
            }
        ]
    )
    assert items[0].key == "writing_quality"
