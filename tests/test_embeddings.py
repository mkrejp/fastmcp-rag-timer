"""Tests for the Novita BGE-M3 embedding client."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from rag.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_URL,
    get_embedding,
    resolve_api_key,
)


def test_resolve_api_key_prefers_bge_m3_runtime_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dedicated Railway variable takes precedence over legacy names."""
    monkeypatch.setenv("BGE_M3_NOVITA_API_KEY", "novita-key")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")

    assert resolve_api_key() == "novita-key"


def test_get_embedding_uses_novita_openai_compatible_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding requests use the configured key, model, and endpoint."""
    vector = [0.25] * EMBEDDING_DIM
    response = Mock()
    response.json.return_value = {"data": [{"embedding": vector}]}
    response.raise_for_status.return_value = None
    post = Mock(return_value=response)
    monkeypatch.setattr("rag.embeddings.requests.post", post)

    result = get_embedding("sample text", api_key="test-key")

    assert result == vector
    post.assert_called_once_with(
        EMBEDDING_URL,
        json={"input": "sample text", "model": EMBEDDING_MODEL},
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        timeout=60,
    )


def test_get_embedding_rejects_unexpected_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected model dimensions fail before vectors are stored."""
    response = Mock()
    response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
    response.raise_for_status.return_value = None
    monkeypatch.setattr("rag.embeddings.requests.post", Mock(return_value=response))

    with pytest.raises(ValueError, match="expected 1024"):
        get_embedding("sample text", api_key="test-key")


def test_get_embedding_requires_runtime_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing credentials fail with guidance naming the dedicated variable."""
    for name in (
        "BGE_M3_NOVITA_API_KEY",
        "NOVITA_API_KEY",
        "OPENAI_TOKEN",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="BGE_M3_NOVITA_API_KEY"):
        get_embedding("sample text")
