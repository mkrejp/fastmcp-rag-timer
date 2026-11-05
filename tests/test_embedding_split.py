"""Tests for ingest vs query embedding provider split (phase D)."""

from __future__ import annotations

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rag.embeddings import (
    resolve_embedding_provider_ingest,
    resolve_embedding_provider_query_unvalidated,
    validate_embedding_configuration,
)


def test_provider_fallback_to_embedding_provider(
    monkeypatch: MonkeyPatch,
) -> None:
    """Unset INGEST/QUERY fall back to EMBEDDING_PROVIDER."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.delenv("EMBEDDING_PROVIDER_INGEST", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER_QUERY", raising=False)
    assert resolve_embedding_provider_ingest() == "fastembed"
    assert resolve_embedding_provider_query_unvalidated() == "fastembed"


def test_ingest_override(monkeypatch: MonkeyPatch) -> None:
    """EMBEDDING_PROVIDER_INGEST overrides the default."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("EMBEDDING_PROVIDER_INGEST", "novita")
    monkeypatch.setenv("EMBEDDING_PROVIDER_QUERY", "fastembed")
    assert resolve_embedding_provider_ingest() == "novita"


def test_query_search_alias(monkeypatch: MonkeyPatch) -> None:
    """EMBEDDING_PROVIDER_SEARCH is an alias for QUERY."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "novita")
    monkeypatch.setenv("EMBEDDING_PROVIDER_SEARCH", "fastembed")
    assert resolve_embedding_provider_query_unvalidated() == "fastembed"


def test_validate_refuses_remote_query(monkeypatch: MonkeyPatch) -> None:
    """QUERY=novita must fail validation (zero tokens on search)."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "novita")
    monkeypatch.setenv("EMBEDDING_PROVIDER_QUERY", "novita")
    with pytest.raises(ValueError, match="fastembed"):
        validate_embedding_configuration()


def test_validate_requires_explicit_query_when_default_is_novita(
    monkeypatch: MonkeyPatch,
) -> None:
    """Novita default without QUERY override is rejected."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "novita")
    monkeypatch.delenv("EMBEDDING_PROVIDER_QUERY", raising=False)
    with pytest.raises(ValueError, match="fastembed"):
        validate_embedding_configuration()


def test_validate_accepts_novita_ingest_fastembed_query(
    monkeypatch: MonkeyPatch,
) -> None:
    """Split remote ingest + local query passes validation."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "novita")
    monkeypatch.setenv("EMBEDDING_PROVIDER_INGEST", "novita")
    monkeypatch.setenv("EMBEDDING_PROVIDER_QUERY", "fastembed")
    validate_embedding_configuration()
