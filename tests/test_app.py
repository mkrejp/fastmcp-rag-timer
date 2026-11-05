"""HTTP and MCP tool tests for the FastAPI application factory."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.config import COLLECTION_NAME
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


def test_delete_document_removes_catalog_and_text(client: TestClient) -> None:
    """Authenticated DELETE should remove a document from catalog and disk."""
    upload = client.post(
        "/upload",
        files={
            "file": (
                "obsolete.txt",
                b"This uploaded file should be removable.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]

    deleted = client.delete(f"/documents/{doc_id}")
    assert deleted.status_code == 200
    body = deleted.json()
    assert body["status"] == "deleted"
    assert body["doc_id"] == doc_id
    assert body["filename"] == "obsolete.txt"
    assert body["chunks_removed"] >= 1

    listed = client.get("/documents")
    assert listed.status_code == 200
    assert all(d["doc_id"] != doc_id for d in listed.json()["documents"])
    assert client.get(f"/documents/{doc_id}").status_code == 404
    assert client.get(f"/documents/{doc_id}/text").status_code == 404

    storage = client.app.state.storage
    assert not storage.original_dir(doc_id).exists()
    assert not storage.processed_doc_dir(doc_id).exists()
    remaining = client.app.state.qdrant.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        ),
        exact=True,
    )
    assert remaining.count == 0


def test_delete_missing_document_returns_404(client: TestClient) -> None:
    """DELETE of an unknown or invalid doc_id should return 404."""
    missing = client.delete("/documents/00000000-0000-0000-0000-000000000000")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Document not found"
    invalid = client.delete("/documents/missing-doc-id")
    assert invalid.status_code == 404
    assert invalid.json()["detail"] == "Document not found"


def test_delete_document_requires_bearer(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """DELETE /documents/{doc_id} should reject callers without a bearer token."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as unauth:
            response = unauth.delete("/documents/any-doc-id")
            assert response.status_code == 401
            assert response.json()["detail"] == "Missing or invalid bearer token"


def test_openapi_documents_delete_route(client: TestClient) -> None:
    """OpenAPI should advertise authenticated DELETE /documents/{doc_id}."""
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    payload = spec.json()
    delete_op = payload["paths"]["/documents/{doc_id}"]["delete"]
    schema = delete_op["responses"]["200"]["content"]["application/json"]["schema"]
    ref = schema.get("$ref", "")
    if not ref:
        ref = schema.get("anyOf", [{}])[0].get("$ref", "")
    assert ref.endswith("DeleteDocumentResponse")
    assert "DeleteDocumentResponse" in payload["components"]["schemas"]


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
