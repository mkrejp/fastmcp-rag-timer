"""Tests for PATCH /documents/{doc_id}/metadata and ingest payload fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.document_patch import patch_document_metadata
from rag.ingest import ingest_bytes, stable_point_id
from rag.storage import RagStorage
from rag.verification_meta import (
    MAX_OVERENI_SOUHRN_LEN,
    MAX_QUOTE_LEN,
    DocumentMetadataError,
)
from tests.conftest import AUTH_HEADERS, patched_embeddings

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

COLLECTION = "client_domain_knowledge"
MCP_HEADERS = {
    **AUTH_HEADERS,
    "Accept": "application/json, text/event-stream",
    "Host": "localhost",
}


def _scroll_doc(qdrant_client: QdrantClient, doc_id: str) -> list[Any]:
    """Return knowledge points for ``doc_id``."""
    points, _offset = qdrant_client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=64,
        with_payload=True,
        with_vectors=True,
    )
    return list(points)


def _item() -> dict[str, Any]:
    """Return a valid verification[] row including the new optional fields."""
    return {
        "agent": "TesterVerifier",
        "check": "gps",
        "tool": "qgate.py",
        "date": "2026-11-12T12:00:00+01:00",
        "result": "potvrzeno-wiki",
        "source_url": "https://example.test/wiki",
        "quote": "Souřadnice u náměstí.",
        "independent": False,
        "source_key": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "method": "haversine",
        "compared_value": 80,
        "found_value": 76.4,
        "threshold": 100,
        "evidence_type": "gps",
    }


def test_ingest_writes_provided_verification_fields(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """New uploads should persist provided flat fields and default reliability."""
    long_souhrn = "x" * (MAX_OVERENI_SOUHRN_LEN + 25)
    result = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="fact.txt",
        data=b"A stored fact about a facade.",
        embed_fn=fake_embed,
        metadata={
            "qg_verze": "2026-11-12:abcd",
            "overeni_souhrn": long_souhrn,
            "verification": [_item()],
            "source_key": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        },
    )
    doc_id = str(result["doc_id"])
    payload = (
        qdrant_client.retrieve(
            collection_name=COLLECTION,
            ids=[stable_point_id(doc_id, 0)],
            with_payload=True,
            with_vectors=True,
        )[0].payload
        or {}
    )
    assert payload["reliability"] == 1
    assert payload["qg_verze"] == "2026-11-12:abcd"
    assert payload["overeni_souhrn"] == "x" * MAX_OVERENI_SOUHRN_LEN
    assert payload["verification"][0]["method"] == "haversine"
    assert payload["verification"][0]["compared_value"] == 80
    assert payload["verification"][0]["found_value"] == 76.4
    assert payload["verification"][0]["threshold"] == 100
    assert payload["verification"][0]["evidence_type"] == "gps"
    assert payload["verification"][0]["result"] == "potvrzeno-wiki"
    assert "truncated" not in payload
    assert result["truncated"] == ["overeni_souhrn"]
    meta = rag_storage.read_meta(doc_id)
    assert meta is not None
    assert meta.extra["qg_verze"] == "2026-11-12:abcd"
    assert meta.doc_id == doc_id


def test_patch_document_metadata_set_payload_no_reembed(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """PATCH metadata should update payloads and leave vectors and doc_id."""
    ingested = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="old.txt",
        data=b"Patch verification fields in place.",
        embed_fn=fake_embed,
    )
    doc_id = str(ingested["doc_id"])
    before = _scroll_doc(qdrant_client, doc_id)
    assert before
    before_vectors = {point.id: point.vector for point in before}
    before_text = (before[0].payload or {}).get("text")

    result = patch_document_metadata(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        metadata={
            "reliability": 2,
            "qg_verze": "2026-11-12:ef01",
            "overeni_souhrn": "GPS lt100, wiki strop 2",
            "verification_status": "potvrzeno-wiki",
            "verification": [_item()],
        },
    )
    assert result["doc_id"] == doc_id
    assert "reliability" in result["updated_fields"]
    assert result["points_updated"] >= 1
    assert result["truncated"] == []

    after = _scroll_doc(qdrant_client, doc_id)
    assert {point.id: point.vector for point in after} == before_vectors
    payload = after[0].payload or {}
    assert payload["reliability"] == 2
    assert payload["qg_verze"] == "2026-11-12:ef01"
    assert payload["overeni_souhrn"] == "GPS lt100, wiki strop 2"
    assert payload["verification"][0]["method"] == "haversine"
    assert payload["text"] == before_text
    assert payload["doc_id"] == doc_id


def test_patch_document_metadata_http(
    client: TestClient,
) -> None:
    """Authenticated PATCH /documents/{id}/metadata should write payload fields."""
    upload = client.post(
        "/upload",
        files={"file": ("meta.txt", b"HTTP metadata patch target.", "text/plain")},
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]

    patched = client.patch(
        f"/documents/{doc_id}/metadata",
        json={
            "reliability": 3,
            "qg_verze": "2026-11-12:http",
            "overeni_souhrn": "primární zdroj, shoda 2",
            "verification": [_item()],
        },
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["status"] == "updated"
    assert body["doc_id"] == doc_id
    assert body["meta"]["reliability"] == 3
    assert body["meta"]["overeni_souhrn"] == "primární zdroj, shoda 2"
    assert body["truncated"] == []

    fetched = client.get(f"/documents/{doc_id}")
    assert fetched.status_code == 200
    assert fetched.json()["reliability"] == 3

    missing = client.patch(
        "/documents/00000000-0000-0000-0000-000000000000/metadata",
        json={"reliability": 1},
    )
    assert missing.status_code == 404
    bad = client.patch(f"/documents/{doc_id}/metadata", json={"doc_id": "other"})
    assert bad.status_code == 400


def test_patch_document_metadata_requires_bearer(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """PATCH /documents/{doc_id}/metadata should require the same auth as DELETE."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as unauth:
            response = unauth.patch(
                "/documents/any-doc-id/metadata", json={"reliability": 1}
            )
            assert response.status_code == 401


def test_upload_metadata_form_field(client: TestClient) -> None:
    """POST /upload may send verification JSON in the metadata form field."""
    response = client.post(
        "/upload",
        files={"file": ("form.txt", b"Upload with metadata form.", "text/plain")},
        data={
            "metadata": json.dumps(
                {
                    "qg_verze": "form-1",
                    "overeni_souhrn": "form upload",
                    "verification": [_item()],
                }
            )
        },
    )
    assert response.status_code == 200
    doc_id = response.json()["doc_id"]
    fetched = client.get(f"/documents/{doc_id}").json()
    assert fetched["qg_verze"] == "form-1"
    assert fetched["overeni_souhrn"] == "form upload"
    assert fetched["verification"][0]["evidence_type"] == "gps"
    assert response.json()["truncated"] == []


def test_update_document_metadata_mcp_tool(client: TestClient) -> None:
    """MCP update_document_metadata should patch fields and reject unknown ids."""
    upload = client.post(
        "/upload",
        files={"file": ("mcp-meta.txt", b"MCP verification patch.", "text/plain")},
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]

    init = client.post(
        "/mcp/",
        headers=MCP_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"},
            },
        },
    )
    assert init.status_code == 200
    call = client.post(
        "/mcp/",
        headers=MCP_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "update_document_metadata",
                "arguments": {
                    "doc_id": doc_id,
                    "metadata": {
                        "reliability": 2,
                        "overeni_souhrn": "mcp",
                        "verification": [_item()],
                    },
                },
            },
        },
    )
    assert call.status_code == 200
    text = call.text
    assert "updated" in text
    fetched = client.get(f"/documents/{doc_id}").json()
    assert fetched["reliability"] == 2
    assert fetched["verification"][0]["method"] == "haversine"


def test_ingest_reports_truncated_field_paths(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Ingest metadata responses list every field that was clipped."""
    long_souhrn = "s" * (MAX_OVERENI_SOUHRN_LEN + 12)
    long_quote = "q" * (MAX_QUOTE_LEN + 8)
    long_verified = "v" * (MAX_QUOTE_LEN + 4)
    item = _item()
    item["quote"] = long_quote
    ingested = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="trunc.txt",
        data=b"Truncation must be reported.",
        embed_fn=fake_embed,
        metadata={
            "overeni_souhrn": long_souhrn,
            "verified_quote": long_verified,
            "verification": [item],
        },
    )
    assert ingested["truncated"] == [
        "overeni_souhrn",
        "verified_quote",
        "verification[0].quote",
    ]
    payload = (
        qdrant_client.retrieve(
            collection_name=COLLECTION,
            ids=[stable_point_id(str(ingested["doc_id"]), 0)],
            with_payload=True,
        )[0].payload
        or {}
    )
    assert payload["overeni_souhrn"] == "s" * MAX_OVERENI_SOUHRN_LEN
    assert "truncated" not in payload


def test_patch_http_reports_truncated_field_paths(client: TestClient) -> None:
    """PATCH /documents/{id}/metadata lists every field that was clipped."""
    long_souhrn = "s" * (MAX_OVERENI_SOUHRN_LEN + 12)
    long_quote = "q" * (MAX_QUOTE_LEN + 8)
    long_verified = "v" * (MAX_QUOTE_LEN + 4)
    item = _item()
    item["quote"] = long_quote
    upload = client.post(
        "/upload",
        files={"file": ("http-trunc.txt", b"HTTP truncation report.", "text/plain")},
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]
    patched = client.patch(
        f"/documents/{doc_id}/metadata",
        json={
            "overeni_souhrn": long_souhrn,
            "verified_quote": long_verified,
            "verification": [item],
        },
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["truncated"] == [
        "overeni_souhrn",
        "verified_quote",
        "verification[0].quote",
    ]
    assert body["meta"]["overeni_souhrn"] == "s" * MAX_OVERENI_SOUHRN_LEN
    assert body["meta"]["verified_quote"] == "v" * MAX_QUOTE_LEN


def test_patch_rejects_doc_id_change(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """doc_id must not be accepted as a metadata field."""
    ingested = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="id.txt",
        data=b"Do not change doc_id.",
        embed_fn=fake_embed,
    )
    with pytest.raises(DocumentMetadataError, match="doc_id"):
        patch_document_metadata(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            doc_id=str(ingested["doc_id"]),
            metadata={"doc_id": "new-id"},
        )


@pytest.fixture()
def client(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> Iterator[TestClient]:
    """Authenticated TestClient with temp storage."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app, headers=AUTH_HEADERS) as test_client:
            yield test_client
