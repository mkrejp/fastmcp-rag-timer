"""Tests for PATCH /documents/{doc_id}/metadata and ingest payload fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator
from unittest.mock import patch

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
from tests.fixtures.pdf_factory import build_image_rich_pdf

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


def _mcp_tool_call(client: TestClient, name: str, arguments: dict[str, Any]) -> Any:
    """Initialize MCP and call one tool; return the tools/call response."""
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
            "params": {"name": name, "arguments": arguments},
        },
    )
    assert call.status_code == 200
    return call


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
    assert result["deleted_fields"] == []

    after = _scroll_doc(qdrant_client, doc_id)
    assert {point.id: point.vector for point in after} == before_vectors
    payload = after[0].payload or {}
    assert payload["reliability"] == 2
    assert payload["qg_verze"] == "2026-11-12:ef01"
    assert payload["overeni_souhrn"] == "GPS lt100, wiki strop 2"
    assert payload["verification"][0]["method"] == "haversine"
    assert payload["text"] == before_text
    assert payload["doc_id"] == doc_id

    cleared = patch_document_metadata(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        metadata={"verification": None, "overeni_souhrn": None},
    )
    assert cleared["deleted_fields"] == ["overeni_souhrn", "verification"]
    after_delete = _scroll_doc(qdrant_client, doc_id)
    assert {point.id: point.vector for point in after_delete} == before_vectors
    deleted_payload = after_delete[0].payload or {}
    assert "verification" not in deleted_payload
    assert "overeni_souhrn" not in deleted_payload
    assert deleted_payload["text"] == before_text
    assert deleted_payload["doc_id"] == doc_id
    assert deleted_payload["reliability"] == 2


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
    assert body["deleted_fields"] == []

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

    call = _mcp_tool_call(
        client,
        "update_document_metadata",
        {
            "doc_id": doc_id,
            "metadata": {
                "reliability": 2,
                "overeni_souhrn": "mcp",
                "verification": [_item()],
            },
        },
    )
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


def test_patch_null_deletes_whitelist_field_from_get(client: TestClient) -> None:
    """JSON null on a whitelist key should drop it from GET and Qdrant."""
    upload = client.post(
        "/upload",
        files={
            "file": ("del.txt", b"Null deletes a verification field.", "text/plain")
        },
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]
    seeded = client.patch(
        f"/documents/{doc_id}/metadata",
        json={
            "verified_by": "auditor",
            "overeni_souhrn": "keep then drop",
            "reliability": 2,
        },
    )
    assert seeded.status_code == 200
    before = client.get(f"/documents/{doc_id}")
    assert before.status_code == 200
    assert before.json()["verified_by"] == "auditor"

    deleted = client.patch(
        f"/documents/{doc_id}/metadata",
        json={"verified_by": None},
    )
    assert deleted.status_code == 200
    body = deleted.json()
    assert body["deleted_fields"] == ["verified_by"]
    assert "verified_by" not in body["updated_fields"]
    fetched = client.get(f"/documents/{doc_id}").json()
    assert "verified_by" not in fetched
    assert fetched["overeni_souhrn"] == "keep then drop"
    assert fetched["reliability"] == 2
    assert fetched["doc_id"] == doc_id


def test_patch_null_non_whitelist_is_400(client: TestClient) -> None:
    """Null on tags / obec_kod / gps_presnost / extrakce_model / fakt_ref is 400."""
    upload = client.post(
        "/upload",
        files={"file": ("bad-null.txt", b"Reject non-whitelist nulls.", "text/plain")},
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]
    for field in (
        "tags",
        "obec_kod",
        "gps_presnost",
        "extrakce_model",
        "fakt_ref",
    ):
        response = client.patch(
            f"/documents/{doc_id}/metadata",
            json={field: None},
        )
        assert response.status_code == 400, field


def test_patch_source_key_content_hash_null_requires_verification_null(
    client: TestClient,
) -> None:
    """source_key/content_hash null is 400 unless verification is also null."""
    upload = client.post(
        "/upload",
        files={
            "file": (
                "rollback.txt",
                b"Rollback source_key with verification.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]
    seeded = client.patch(
        f"/documents/{doc_id}/metadata",
        json={
            "source_key": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "content_hash": "abc123",
            "verification": [_item()],
            "verified_by": "keeper",
        },
    )
    assert seeded.status_code == 200
    assert (
        client.patch(
            f"/documents/{doc_id}/metadata", json={"source_key": None}
        ).status_code
        == 400
    )
    assert (
        client.patch(
            f"/documents/{doc_id}/metadata", json={"content_hash": None}
        ).status_code
        == 400
    )

    rolled = client.patch(
        f"/documents/{doc_id}/metadata",
        json={
            "verification": None,
            "source_key": None,
            "content_hash": None,
        },
    )
    assert rolled.status_code == 200
    assert rolled.json()["deleted_fields"] == [
        "content_hash",
        "source_key",
        "verification",
    ]
    fetched = client.get(f"/documents/{doc_id}").json()
    assert "verification" not in fetched
    assert "source_key" not in fetched
    assert "content_hash" not in fetched
    assert fetched["verified_by"] == "keeper"


def test_update_document_metadata_mcp_null_delete(client: TestClient) -> None:
    """MCP update_document_metadata should delete whitelist keys on null."""
    upload = client.post(
        "/upload",
        files={"file": ("mcp-del.txt", b"MCP null delete.", "text/plain")},
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]
    client.patch(
        f"/documents/{doc_id}/metadata",
        json={"verified_quote": "drop me", "reliability": 2},
    )
    call = _mcp_tool_call(
        client,
        "update_document_metadata",
        {"doc_id": doc_id, "metadata": {"verified_quote": None}},
    )
    assert "deleted_fields" in call.text
    fetched = client.get(f"/documents/{doc_id}").json()
    assert "verified_quote" not in fetched
    assert fetched["reliability"] == 2


def test_ingest_pdf_images_omit_nested_verification(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """PDF image/figure points should get flat figure fields, not verification[]."""
    pdf_bytes = build_image_rich_pdf()
    with patch(
        "rag.ingest.ocr_image_file",
        return_value=("indexed ocr phrase long enough", 90.0),
    ):
        result = ingest_bytes(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            filename="fixture.pdf",
            data=pdf_bytes,
            embed_fn=fake_embed,
            metadata={
                "reliability": 2,
                "overeni_souhrn": "flat only on figures",
                "verification": [_item()],
            },
        )
    assert result["status"] == "Success"
    doc_id = str(result["doc_id"])
    by_type: dict[str, list[Any]] = {}
    for point in _scroll_doc(qdrant_client, doc_id):
        payload = point.payload or {}
        source_type = str(payload.get("source_type") or "")
        by_type.setdefault(source_type, []).append(payload)
        assert payload.get("text")
        assert payload["doc_id"] == doc_id
    chunks = by_type.get("pdf_chunk") or []
    assert chunks
    assert chunks[0]["verification"][0]["method"] == "haversine"
    figure_like = (by_type.get("pdf_image") or []) + (by_type.get("pdf_figure") or [])
    assert figure_like
    for payload in figure_like:
        assert "verification" not in payload
        assert payload.get("reliability") == 2
        assert payload.get("overeni_souhrn") == "flat only on figures"


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
