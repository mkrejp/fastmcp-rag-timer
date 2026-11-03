"""Tests for shared-token write authentication."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def test_upload_rejected_without_token(
    rag_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When RAG_API_TOKEN is set, uploads without credentials must fail."""
    monkeypatch.setenv("RAG_API_TOKEN", "secret-write-token")
    with patch("rag.ingest.get_embedding", side_effect=fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root))
        with TestClient(app) as client:
            response = client.post(
                "/upload",
                files={"file": ("a.txt", b"hello", "text/plain")},
            )
            assert response.status_code == 401


def test_upload_accepted_with_bearer_token(
    rag_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bearer token matching RAG_API_TOKEN should authorize uploads."""
    monkeypatch.setenv("RAG_API_TOKEN", "secret-write-token")
    with patch("rag.ingest.get_embedding", side_effect=fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root))
        with TestClient(app) as client:
            response = client.post(
                "/upload",
                headers={"Authorization": "Bearer secret-write-token"},
                files={"file": ("a.txt", b"hello auth", "text/plain")},
            )
            assert response.status_code == 200
            assert response.json()["status"] == "Success"


def test_upload_accepted_with_x_api_key(
    rag_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """X-API-Key header should authorize uploads."""
    monkeypatch.setenv("RAG_API_TOKEN", "secret-write-token")
    with patch("rag.ingest.get_embedding", side_effect=fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root))
        with TestClient(app) as client:
            response = client.post(
                "/upload",
                headers={"X-API-Key": "secret-write-token"},
                files={"file": ("b.txt", b"hello key", "text/plain")},
            )
            assert response.status_code == 200


def test_reads_remain_open_with_token_configured(
    rag_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document reads should not require the write token."""
    monkeypatch.setenv("RAG_API_TOKEN", "secret-write-token")
    with patch("rag.ingest.get_embedding", side_effect=fake_embed):
        from rag.app import create_app
        from rag.storage import RagStorage

        storage = RagStorage(rag_root)
        storage.ensure_layout()
        storage.save_original("doc-1", "note.txt", b"visible")
        storage.write_processed(
            doc_id="doc-1",
            filename="note.txt",
            mime="text/plain",
            sha256="aa",
            fulltext="visible",
            chunks=[{"index": 0, "text": "visible", "page": 0}],
        )
        app = create_app(str(rag_root))
        with TestClient(app) as client:
            listed = client.get("/documents")
            assert listed.status_code == 200
            text = client.get("/documents/doc-1/text")
            assert text.status_code == 200
            assert text.json()["text"] == "visible"


def test_verify_write_token_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """verify_write_token should accept matches and reject mismatches."""
    from rag.auth import verify_write_token

    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    verify_write_token(None)

    monkeypatch.setenv("RAG_API_TOKEN", "abc")
    verify_write_token("abc")
    with pytest.raises(PermissionError):
        verify_write_token("wrong")
    with pytest.raises(PermissionError):
        verify_write_token(None)
