"""Tests for OpenRouter credential resolution and public status."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from rag.openrouter import (
    OPENROUTER_BASE_URL,
    list_openrouter_models,
    openrouter_configured,
    openrouter_status,
    resolve_openrouter_api_key,
    resolve_openrouter_model,
)
from tests.conftest import AUTH_HEADERS, patched_embeddings

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture(autouse=True)
def _clear_openrouter_env(monkeypatch: MonkeyPatch) -> None:
    """Keep OpenRouter tests isolated from the agent or host environment."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)


def test_resolve_openrouter_api_key(monkeypatch: MonkeyPatch) -> None:
    """``OPENROUTER_API_KEY`` is the only OpenRouter credential name."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-project-key")

    assert resolve_openrouter_api_key() == "or-project-key"
    assert openrouter_configured() is True


def test_resolve_openrouter_api_key_unset(monkeypatch: MonkeyPatch) -> None:
    """Empty or missing OpenRouter vars mean the client is not configured."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "  ")

    assert resolve_openrouter_api_key() is None
    assert openrouter_configured() is False
    status = openrouter_status()
    assert status["configured"] is False
    assert status["base_url"] == OPENROUTER_BASE_URL
    assert status["model"] is None


def test_resolve_openrouter_model(monkeypatch: MonkeyPatch) -> None:
    """Optional model pin is a catalog id, not a secret."""
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    assert resolve_openrouter_model() == "openai/gpt-4o-mini"
    assert openrouter_status()["model"] == "openai/gpt-4o-mini"


def test_list_openrouter_models_requires_key(monkeypatch: MonkeyPatch) -> None:
    """Listing models fails closed when no OpenRouter key is set."""
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        list_openrouter_models()


def test_list_openrouter_models_returns_ids(
    monkeypatch: MonkeyPatch, mocker: MockerFixture
) -> None:
    """``GET /api/v1/models`` returns catalog ids and never needs the raw key in asserts."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-project-key")
    response = Mock()
    response.json.return_value = {
        "data": [{"id": "openai/gpt-4o-mini"}, {"id": "  "}, "skip"]
    }
    mocked_get = mocker.patch("rag.openrouter.requests.get", return_value=response)

    assert list_openrouter_models() == ["openai/gpt-4o-mini"]
    mocked_get.assert_called_once()
    args, kwargs = mocked_get.call_args
    assert args[0] == f"{OPENROUTER_BASE_URL}/models"
    assert kwargs["headers"]["Authorization"] == "Bearer or-project-key"


def test_health_reports_openrouter_configured(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> None:
    """``GET /health`` exposes a boolean OpenRouter flag and never the key."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-project-key")
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app, headers=AUTH_HEADERS) as client:
            body = client.get("/health").json()
            assert body["openrouter"]["configured"] is True
            assert body["openrouter"]["base_url"] == OPENROUTER_BASE_URL
            assert "or-project-key" not in str(body)
