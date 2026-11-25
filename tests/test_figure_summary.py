"""Tests for extractive and optional OpenRouter figure LLM summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from rag.config import (
    resolve_figure_summary_llm,
    resolve_figure_summary_llm_provider,
)
from rag.figure_summary import FigureLlmBudget, build_figure_summary
from rag.openrouter import OPENROUTER_BASE_URL, OPENROUTER_FIGURE_MODEL

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture(autouse=True)
def _clear_figure_llm_env(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Isolate figure LLM env from the host and use a temp RAG root."""
    root = tmp_path / "rag"
    root.mkdir(exist_ok=True)
    monkeypatch.setenv("RAG_DATA_ROOT", str(root))
    monkeypatch.delenv("FIGURE_SUMMARY_LLM", raising=False)
    monkeypatch.delenv("FIGURE_SUMMARY_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("FIGURE_SUMMARY_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)


def _castle_spans() -> tuple[list[dict[str, object]], list[str]]:
    """Return grounded figure refs and matching span text."""
    text = "The sandstone castle stands on the hill above the river."
    return ([{"kind": "same_page", "score": 1.0}], [text])


def test_resolve_figure_summary_llm_accepts_openrouter(
    monkeypatch: MonkeyPatch,
) -> None:
    """``openrouter`` is a valid backend; unknown values fail closed to ``off``."""
    monkeypatch.setenv("FIGURE_SUMMARY_LLM", "openrouter")
    assert resolve_figure_summary_llm() == "openrouter"
    monkeypatch.setenv("FIGURE_SUMMARY_LLM", "gateway")
    assert resolve_figure_summary_llm() == "off"


def test_resolve_figure_summary_llm_provider_fails_closed(
    monkeypatch: MonkeyPatch,
) -> None:
    """Unknown provider slugs fall back to Novita BYOK."""
    assert resolve_figure_summary_llm_provider() == "novita"
    monkeypatch.setenv("FIGURE_SUMMARY_LLM_PROVIDER", "nvidia")
    assert resolve_figure_summary_llm_provider() == "nvidia"
    monkeypatch.setenv("FIGURE_SUMMARY_LLM_PROVIDER", "openai")
    assert resolve_figure_summary_llm_provider() == "novita"


def test_build_figure_summary_default_is_extractive() -> None:
    """Default ``FIGURE_SUMMARY_LLM=off`` never calls a chat API."""
    refs, spans = _castle_spans()
    summary, sentences, method, model = build_figure_summary(
        caption=None,
        related_text_refs=refs,
        span_texts=spans,
        ocr_text=None,
        image_sha256="abc",
        llm_budget=FigureLlmBudget(),
    )
    assert method == "extractive"
    assert model is None
    assert summary
    assert sentences


def test_build_figure_summary_openrouter_posts_provider_only(
    monkeypatch: MonkeyPatch, mocker: MockerFixture
) -> None:
    """OpenRouter mode posts chat completions with ``provider.only``."""
    monkeypatch.setenv("FIGURE_SUMMARY_LLM", "openrouter")
    monkeypatch.setenv("FIGURE_SUMMARY_LLM_PROVIDER", "novita")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-project-key")
    refs, spans = _castle_spans()
    payload = [{"text": spans[0], "ref": 0}]
    response = mocker.Mock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(payload)}}]
    }
    mocked_post = mocker.patch("rag.openrouter.requests.post", return_value=response)

    summary, sentences, method, model = build_figure_summary(
        caption=None,
        related_text_refs=refs,
        span_texts=spans,
        ocr_text=None,
        image_sha256="abc",
        llm_budget=FigureLlmBudget(),
    )

    assert method == "llm"
    assert model == OPENROUTER_FIGURE_MODEL
    assert sentences
    mocked_post.assert_called_once()
    args, kwargs = mocked_post.call_args
    assert args[0] == f"{OPENROUTER_BASE_URL}/chat/completions"
    assert kwargs["json"]["provider"] == {"only": ["novita"]}
    assert kwargs["json"]["model"] == OPENROUTER_FIGURE_MODEL
    assert kwargs["headers"]["Authorization"] == "Bearer or-project-key"
    assert "or-project-key" not in summary


def test_build_figure_summary_openrouter_missing_key_is_extractive(
    monkeypatch: MonkeyPatch, mocker: MockerFixture
) -> None:
    """Missing ``OPENROUTER_API_KEY`` falls back to extractive summaries."""
    monkeypatch.setenv("FIGURE_SUMMARY_LLM", "openrouter")
    mocked_post = mocker.patch("rag.openrouter.requests.post")
    refs, spans = _castle_spans()

    _summary, _sentences, method, model = build_figure_summary(
        caption=None,
        related_text_refs=refs,
        span_texts=spans,
        ocr_text=None,
        image_sha256="abc",
        llm_budget=FigureLlmBudget(),
    )

    assert method == "extractive"
    assert model is None
    mocked_post.assert_not_called()
