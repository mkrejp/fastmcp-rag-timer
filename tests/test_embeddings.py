"""Tests for Novita and FastEmbed embedding providers."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rag.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_URL,
    FASTEMBED_DIM,
    FASTEMBED_MODEL,
    embedding_dim,
    embedding_model,
    embedding_url,
    get_embedding,
    get_embeddings,
    resolve_api_key,
    resolve_embedding_provider,
)


def test_resolve_api_key_prefers_bge_m3_runtime_variable(
    monkeypatch: MonkeyPatch,
) -> None:
    """BGE_M3_NOVITA_API_KEY takes precedence over other credential names."""
    monkeypatch.setenv("BGE_M3_NOVITA_API_KEY", "novita-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-key")

    assert resolve_api_key() == "novita-key"


def test_resolve_embedding_provider_defaults_to_novita(
    monkeypatch: MonkeyPatch,
) -> None:
    """Unset EMBEDDING_PROVIDER keeps the production Novita backend."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)

    assert resolve_embedding_provider() == "novita"
    assert embedding_model() == EMBEDDING_MODEL
    assert embedding_url() == EMBEDDING_URL
    assert embedding_dim() == EMBEDDING_DIM


@pytest.mark.parametrize("value", ["novita", "NOVITA", " Novita "])
def test_resolve_embedding_provider_accepts_novita(
    monkeypatch: MonkeyPatch, value: str
) -> None:
    """Explicit novita values, including mixed case, select the HTTP backend."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", value)

    assert resolve_embedding_provider() == "novita"


@pytest.mark.parametrize("value", ["fastembed", "FASTEMBED", " FastEmbed "])
def test_resolve_embedding_provider_accepts_fastembed(
    monkeypatch: MonkeyPatch, value: str
) -> None:
    """fastembed values select the local backend and 1024-dim BGE-M3 model."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", value)

    assert resolve_embedding_provider() == "fastembed"
    assert embedding_model() == FASTEMBED_MODEL
    assert embedding_url() is None
    assert embedding_dim() == FASTEMBED_DIM


def test_resolve_embedding_provider_rejects_unknown(
    monkeypatch: MonkeyPatch,
) -> None:
    """Unsupported provider names fail before any embed call."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")

    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        resolve_embedding_provider()


def test_get_embedding_uses_novita_openai_compatible_api(
    monkeypatch: MonkeyPatch,
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
    monkeypatch: MonkeyPatch,
) -> None:
    """Unexpected model dimensions fail before vectors are stored."""
    response = Mock()
    response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
    response.raise_for_status.return_value = None
    monkeypatch.setattr("rag.embeddings.requests.post", Mock(return_value=response))

    with pytest.raises(ValueError, match="expected 1024"):
        get_embedding("sample text", api_key="test-key")


def test_get_embedding_requires_runtime_key(monkeypatch: MonkeyPatch) -> None:
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


class _FakeFastEmbed:
    """Minimal FastEmbed stand-in that yields fixed-length dense vectors."""

    def __init__(self, dim: int = FASTEMBED_DIM) -> None:
        """Store the vector length returned by ``embed``."""
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, documents: list[str], **_kwargs: object) -> list[list[float]]:
        """Return one dense vector per document."""
        texts = list(documents)
        self.calls.append(texts)
        return [[0.25] * self.dim for _ in texts]


def test_fastembed_get_embedding_returns_expected_length(
    monkeypatch: MonkeyPatch,
) -> None:
    """Local FastEmbed path returns 1024-dim vectors and skips the HTTP API."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    fake = _FakeFastEmbed()
    post = Mock(side_effect=AssertionError("Novita must not be called"))
    monkeypatch.setattr("rag.embeddings.requests.post", post)
    monkeypatch.setattr("rag.embeddings._get_fastembed_model", lambda: fake)

    result = get_embedding("sample text")

    assert len(result) == FASTEMBED_DIM
    assert result == [0.25] * FASTEMBED_DIM
    assert fake.calls == [["sample text"]]
    post.assert_not_called()


def test_fastembed_get_embeddings_batches_texts(
    monkeypatch: MonkeyPatch,
) -> None:
    """FastEmbed encodes a batch in one local model call."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    fake = _FakeFastEmbed()
    monkeypatch.setattr("rag.embeddings._get_fastembed_model", lambda: fake)

    vectors = get_embeddings(["one", "two"])

    assert len(vectors) == 2
    assert all(len(vector) == FASTEMBED_DIM for vector in vectors)
    assert fake.calls == [["one", "two"]]


def test_fastembed_does_not_require_api_key(
    monkeypatch: MonkeyPatch,
) -> None:
    """Local embeddings work without Novita or OpenAI credentials."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    for name in (
        "BGE_M3_NOVITA_API_KEY",
        "NOVITA_API_KEY",
        "OPENAI_TOKEN",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("rag.embeddings._get_fastembed_model", lambda: _FakeFastEmbed())

    vector = get_embedding("no key needed")

    assert len(vector) == FASTEMBED_DIM


def test_fastembed_rejects_unexpected_dimension(
    monkeypatch: MonkeyPatch,
) -> None:
    """Local vectors with the wrong length fail before they are stored."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setattr(
        "rag.embeddings._get_fastembed_model", lambda: _FakeFastEmbed(dim=8)
    )

    with pytest.raises(ValueError, match="expected 1024"):
        get_embedding("sample text")


def test_fastembed_missing_package_has_clear_error(
    monkeypatch: MonkeyPatch,
) -> None:
    """A missing fastembed install explains how to keep using Novita."""
    import builtins

    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        """Fail only the FastEmbed import used by the local backend."""
        if name == "fastembed" or name.startswith("fastembed."):
            raise ImportError("No module named 'fastembed'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="requires the fastembed package"):
        get_embedding("sample text")
