"""HTTP and MCP tool tests for the FastAPI application factory."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from tests.conftest import AUTH_HEADERS, patched_embeddings


@pytest.fixture()
def client(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> Iterator[TestClient]:
    """Build a TestClient with temp storage and mocked embeddings."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app, headers=AUTH_HEADERS) as test_client:
            yield test_client


def test_health(client: TestClient, rag_root: Path) -> None:
    """Health endpoint should report the configured RAG root."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["rag_data_root"] == str(rag_root)


def test_upload_text_and_fetch_processed(client: TestClient) -> None:
    """Uploading text should expose catalog and processed fulltext endpoints."""
    response = client.post(
        "/upload",
        files={
            "file": (
                "domain.txt",
                b"Railway volume mounts at /data.",
                "text/plain",
            )
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "Success"
    doc_id = payload["doc_id"]

    listed = client.get("/documents")
    assert listed.status_code == 200
    assert any(d["doc_id"] == doc_id for d in listed.json()["documents"])

    text_resp = client.get(f"/documents/{doc_id}/text")
    assert text_resp.status_code == 200
    assert "Railway volume" in text_resp.json()["text"]


def test_processed_text_available_after_upload(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """After upload, processed fulltext should be readable without embeddings."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app
        from rag.storage import RagStorage

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app, headers=AUTH_HEADERS) as client:
            upload = client.post(
                "/upload",
                files={
                    "file": (
                        "tooling.txt",
                        b"processed text for agents",
                        "text/plain",
                    )
                },
            )
            doc_id = upload.json()["doc_id"]

        storage = RagStorage(rag_root)
        assert storage.list_documents()
        assert storage.read_fulltext(doc_id) == "processed text for agents"


def test_mcp_mount_registered(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """create_app should mount the MCP ASGI app at ``/mcp``."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        mount_paths = [
            getattr(route, "path", None)
            for route in app.routes
            if route.__class__.__name__ == "Mount"
        ]
        assert "/mcp" in mount_paths
