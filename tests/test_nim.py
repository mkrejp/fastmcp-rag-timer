"""Tests for hosted NVIDIA NIM credential resolution and public status."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from rag.nim import (
    NVIDIA_NIM_BASE_URL,
    list_nim_models,
    nvidia_nim_configured,
    nvidia_nim_status,
    resolve_nvidia_nim_api_key,
    resolve_nvidia_nim_model,
)
from tests.conftest import AUTH_HEADERS, patched_embeddings

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture(autouse=True)
def _clear_nim_env(monkeypatch: MonkeyPatch) -> None:
    """Keep NIM tests isolated from the agent or host environment."""
    monkeypatch.delenv("NVIDIA_NIM_API", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_NIM_MODEL", raising=False)


def test_resolve_nvidia_nim_api_key_prefers_nvidia_nim_api(
    monkeypatch: MonkeyPatch,
) -> None:
    """``NVIDIA_NIM_API`` wins over the catalog ``NVIDIA_API_KEY`` name."""
    monkeypatch.setenv("NVIDIA_NIM_API", "nim-project-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "catalog-key")

    assert resolve_nvidia_nim_api_key() == "nim-project-key"
    assert nvidia_nim_configured() is True


def test_resolve_nvidia_nim_api_key_falls_back_to_nvidia_api_key(
    monkeypatch: MonkeyPatch,
) -> None:
    """Catalog ``NVIDIA_API_KEY`` is accepted when ``NVIDIA_NIM_API`` is unset."""
    monkeypatch.setenv("NVIDIA_API_KEY", "catalog-key")

    assert resolve_nvidia_nim_api_key() == "catalog-key"


def test_resolve_nvidia_nim_api_key_unset(monkeypatch: MonkeyPatch) -> None:
    """Empty or missing NIM vars mean the client is not configured."""
    monkeypatch.setenv("NVIDIA_NIM_API", "  ")

    assert resolve_nvidia_nim_api_key() is None
    assert nvidia_nim_configured() is False
    status = nvidia_nim_status()
    assert status["configured"] is False
    assert status["base_url"] == NVIDIA_NIM_BASE_URL
    assert status["model"] is None


def test_resolve_nvidia_nim_model(monkeypatch: MonkeyPatch) -> None:
    """Optional model pin is a catalog id, not a secret."""
    monkeypatch.setenv("NVIDIA_NIM_MODEL", "nvidia/llama-3.1-70b-instruct")

    assert resolve_nvidia_nim_model() == "nvidia/llama-3.1-70b-instruct"
    assert nvidia_nim_status()["model"] == "nvidia/llama-3.1-70b-instruct"


def test_list_nim_models_requires_key(monkeypatch: MonkeyPatch) -> None:
    """Listing models fails closed when no NIM key is set."""
    with pytest.raises(RuntimeError, match="NVIDIA_NIM_API"):
        list_nim_models()


def test_list_nim_models_returns_ids(
    monkeypatch: MonkeyPatch, mocker: MockerFixture
) -> None:
    """``GET /v1/models`` returns catalog ids and never needs the raw key in asserts."""
    monkeypatch.setenv("NVIDIA_NIM_API", "nim-project-key")
    response = Mock()
    response.json.return_value = {
        "data": [{"id": "nvidia/llama-3.1-70b-instruct"}, {"id": "  "}, "skip"]
    }
    mocked_get = mocker.patch("rag.nim.requests.get", return_value=response)

    assert list_nim_models() == ["nvidia/llama-3.1-70b-instruct"]
    mocked_get.assert_called_once()
    args, kwargs = mocked_get.call_args
    assert args[0] == f"{NVIDIA_NIM_BASE_URL}/models"
    assert kwargs["headers"]["Authorization"] == "Bearer nim-project-key"


def test_health_reports_nvidia_nim_configured(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> None:
    """``GET /health`` exposes a boolean NIM flag and never the key."""
    monkeypatch.setenv("NVIDIA_NIM_API", "nim-project-key")
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app, headers=AUTH_HEADERS) as client:
            body = client.get("/health").json()
            assert body["nvidia_nim"]["configured"] is True
            assert body["nvidia_nim"]["base_url"] == NVIDIA_NIM_BASE_URL
            assert "nim-project-key" not in str(body)
