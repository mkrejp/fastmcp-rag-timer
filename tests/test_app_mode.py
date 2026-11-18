"""Tests for APP_MODE reader/ingester split."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import pytest
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

from rag.app_mode import resolve_app_mode
from rag.embeddings import (
    resolve_embedding_provider_query_unvalidated,
    validate_embedding_configuration,
)
from rag.node_info import (
    ROLE_READ,
    ROLE_WRITE,
    enforce_node_role,
    resolve_node_role_override,
)
from tests.conftest import AUTH_HEADERS, patched_embeddings


def test_resolve_app_mode_aliases(monkeypatch: MonkeyPatch) -> None:
    """Shortcuts map onto canonical modes."""
    monkeypatch.setenv("APP_MODE", "readonly")
    assert resolve_app_mode() == "reader"
    monkeypatch.setenv("APP_MODE", "ingest")
    assert resolve_app_mode() == "ingester"
    monkeypatch.delenv("APP_MODE", raising=False)
    monkeypatch.delenv("RAG_APP_MODE", raising=False)
    assert resolve_app_mode() == "all"


def test_reader_defaults_query_to_novita(monkeypatch: MonkeyPatch) -> None:
    """Unset query provider in reader mode selects Novita (low RAM)."""
    monkeypatch.setenv("APP_MODE", "reader")
    monkeypatch.delenv("EMBEDDING_PROVIDER_QUERY", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER_SEARCH", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "novita")
    assert resolve_embedding_provider_query_unvalidated() == "novita"
    validate_embedding_configuration()


def test_reader_forces_enforce_and_read_role(monkeypatch: MonkeyPatch) -> None:
    """Reader mode blocks writes even when RAG_ENFORCE_NODE_ROLE is unset."""
    monkeypatch.setenv("APP_MODE", "reader")
    monkeypatch.delenv("RAG_ENFORCE_NODE_ROLE", raising=False)
    monkeypatch.delenv("RAG_NODE_ROLE", raising=False)
    assert enforce_node_role() is True
    assert resolve_node_role_override() == frozenset({ROLE_READ})


def test_ingester_default_roles(monkeypatch: MonkeyPatch) -> None:
    """Ingester mode defaults to write+read when RAG_NODE_ROLE unset."""
    monkeypatch.setenv("APP_MODE", "ingester")
    monkeypatch.delenv("RAG_NODE_ROLE", raising=False)
    assert resolve_node_role_override() == frozenset({ROLE_WRITE, ROLE_READ})


@pytest.fixture()
def reader_client(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> Iterator[TestClient]:
    """Build a TestClient in APP_MODE=reader with mocked embeddings."""
    monkeypatch.setenv("APP_MODE", "reader")
    monkeypatch.setenv("EMBEDDING_PROVIDER_QUERY", "novita")
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app, headers=AUTH_HEADERS) as test_client:
            yield test_client


def test_reader_health_reports_mode(reader_client: TestClient) -> None:
    """Health includes app_mode and write flags for readers."""
    response = reader_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["app_mode"] == "reader"
    assert body["writes_enabled"] is False
    assert body["status"] == "ok"


def test_reader_rejects_upload(reader_client: TestClient) -> None:
    """Mutating HTTP routes return 403 on a reader process."""
    response = reader_client.post(
        "/upload",
        files={"file": ("note.txt", b"hello reader", "text/plain")},
    )
    assert response.status_code == 403
    assert "Writes are disabled" in response.json()["detail"]
