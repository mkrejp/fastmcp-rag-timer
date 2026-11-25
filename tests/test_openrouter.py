"""Tests for OpenRouter credential resolution and public status."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from rag.openrouter import (
    OPENROUTER_BASE_URL,
    ensure_openrouter_byok_credential,
    list_openrouter_byok_providers,
    list_openrouter_models,
    openrouter_configured,
    openrouter_management_configured,
    openrouter_status,
    resolve_openrouter_api_key,
    resolve_openrouter_management_key,
    resolve_openrouter_model,
    resolve_openrouter_novita_byok_key,
    resolve_openrouter_nvidia_byok_key,
    sync_openrouter_nvidia_novita_byok,
)
from tests.conftest import AUTH_HEADERS, patched_embeddings

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture(autouse=True)
def _clear_openrouter_env(monkeypatch: MonkeyPatch) -> None:
    """Keep OpenRouter tests isolated from the agent or host environment."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MANAGEMENT_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("NVIDIA_NIM_API", raising=False)
    monkeypatch.delenv("NVIDIA_NIM_APU", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("BGE_M3_NOVITA_API_KEY", raising=False)
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)


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
    assert status["management_configured"] is False
    assert status["base_url"] == OPENROUTER_BASE_URL
    assert status["model"] is None
    assert status["nvidia_provider"] == "nvidia"
    assert status["novita_provider"] == "novita"


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
            assert body["openrouter"]["management_configured"] is False
            assert body["openrouter"]["base_url"] == OPENROUTER_BASE_URL
            assert body["openrouter"]["nvidia_provider"] == "nvidia"
            assert body["openrouter"]["novita_provider"] == "novita"
            assert "or-project-key" not in str(body)


def test_resolve_openrouter_management_key(monkeypatch: MonkeyPatch) -> None:
    """Management key is separate from the inference key."""
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", "or-mgmt-key")

    assert resolve_openrouter_management_key() == "or-mgmt-key"
    assert openrouter_management_configured() is True
    assert openrouter_status()["management_configured"] is True


def test_resolve_openrouter_byok_keys_prefer_project_names(
    monkeypatch: MonkeyPatch,
) -> None:
    """NVIDIA and Novita BYOK keys use the project secret names."""
    monkeypatch.setenv("NVIDIA_NIM_APU", "apu-key")
    monkeypatch.setenv("BGE_M3_NOVITA_API_KEY", "novita-key")

    assert resolve_openrouter_nvidia_byok_key() == "apu-key"
    assert resolve_openrouter_novita_byok_key() == "novita-key"


def test_list_openrouter_byok_providers_requires_management_key() -> None:
    """BYOK listing fails closed without a management key."""
    with pytest.raises(RuntimeError, match="OPENROUTER_MANAGEMENT_KEY"):
        list_openrouter_byok_providers()


def test_list_openrouter_byok_providers_skips_disabled(
    monkeypatch: MonkeyPatch, mocker: MockerFixture
) -> None:
    """Enabled provider slugs are returned; disabled rows and labels are dropped."""
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", "or-mgmt-key")
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "data": [
            {"provider": "nvidia", "disabled": False, "label": "nvapi-secret"},
            {"provider": "novita", "disabled": True, "label": "sk-secret"},
            {"provider": "openai", "disabled": False},
        ]
    }
    mocked_get = mocker.patch("rag.openrouter.requests.get", return_value=response)

    assert list_openrouter_byok_providers() == ["nvidia", "openai"]
    mocked_get.assert_called_once()
    args, kwargs = mocked_get.call_args
    assert args[0] == f"{OPENROUTER_BASE_URL}/byok"
    assert kwargs["headers"]["Authorization"] == "Bearer or-mgmt-key"


def test_ensure_openrouter_byok_credential_creates(
    monkeypatch: MonkeyPatch, mocker: MockerFixture
) -> None:
    """A missing nvidia credential is created and the raw key is not returned."""
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", "or-mgmt-key")
    get_response = Mock()
    get_response.status_code = 200
    get_response.json.return_value = {"data": []}
    post_response = Mock()
    post_response.status_code = 201
    post_response.json.return_value = {"data": {"id": "cred-1", "provider": "nvidia"}}
    mocker.patch("rag.openrouter.requests.get", return_value=get_response)
    mocked_post = mocker.patch(
        "rag.openrouter.requests.post", return_value=post_response
    )

    result = ensure_openrouter_byok_credential(
        "nvidia", key="nvapi-secret", name="NVIDIA NIM (Cursor dev)"
    )

    assert result == "created"
    _args, kwargs = mocked_post.call_args
    assert kwargs["json"]["provider"] == "nvidia"
    assert kwargs["json"]["key"] == "nvapi-secret"
    assert "nvapi-secret" not in result


def test_ensure_openrouter_byok_credential_exists(
    monkeypatch: MonkeyPatch, mocker: MockerFixture
) -> None:
    """An already-enabled provider is not posted again."""
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", "or-mgmt-key")
    get_response = Mock()
    get_response.status_code = 200
    get_response.json.return_value = {
        "data": [{"provider": "novita", "disabled": False}]
    }
    mocked_get = mocker.patch("rag.openrouter.requests.get", return_value=get_response)
    mocked_post = mocker.patch("rag.openrouter.requests.post")

    assert (
        ensure_openrouter_byok_credential(
            "novita", key="novita-secret", name="Novita prepaid (Cursor dev)"
        )
        == "exists"
    )
    mocked_get.assert_called_once()
    mocked_post.assert_not_called()


def test_sync_openrouter_nvidia_novita_byok_without_management(
    monkeypatch: MonkeyPatch,
) -> None:
    """Sync no-ops when the management key is missing, even if provider keys exist."""
    monkeypatch.setenv("NVIDIA_NIM_API", "nim-key")
    monkeypatch.setenv("BGE_M3_NOVITA_API_KEY", "novita-key")

    assert sync_openrouter_nvidia_novita_byok() == {
        "nvidia": "skipped_no_management",
        "novita": "skipped_no_management",
    }


def test_sync_openrouter_nvidia_novita_byok_creates_both(
    monkeypatch: MonkeyPatch, mocker: MockerFixture
) -> None:
    """Sync registers both providers when management and provider keys are set."""
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", "or-mgmt-key")
    monkeypatch.setenv("NVIDIA_NIM_API", "nim-key")
    monkeypatch.setenv("BGE_M3_NOVITA_API_KEY", "novita-key")
    mocked = mocker.patch(
        "rag.openrouter.ensure_openrouter_byok_credential",
        side_effect=["created", "created"],
    )

    assert sync_openrouter_nvidia_novita_byok() == {
        "nvidia": "created",
        "novita": "created",
    }
    assert mocked.call_count == 2
    first_provider = mocked.call_args_list[0].args[0]
    second_provider = mocked.call_args_list[1].args[0]
    assert first_provider == "nvidia"
    assert second_provider == "novita"
