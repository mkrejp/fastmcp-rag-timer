"""Tests for in-place document metadata patches (no re-embed, no deletes)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.document_delete import DocumentNotFoundError
from rag.document_patch import DocumentPatchError, patch_document
from rag.ingest import ingest_bytes, stable_point_id
from rag.storage import RagStorage
from tests.conftest import AUTH_HEADERS, patched_embeddings

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

COLLECTION = "client_domain_knowledge"
MCP_HEADERS = {
    **AUTH_HEADERS,
    "Accept": "application/json, text/event-stream",
    "Host": "localhost",
}


def _scroll_doc(
    qdrant_client: QdrantClient, doc_id: str, *, with_vectors: bool = False
) -> list[Any]:
    """Return all knowledge points for ``doc_id``."""
    points, _offset = qdrant_client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=64,
        with_payload=True,
        with_vectors=with_vectors,
    )
    return list(points)


def test_patch_document_renames_catalog_and_payloads(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Renaming should update catalog + payloads and leave vectors unchanged."""
    neighbor = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="keep.txt",
        data=b"Neighbor document must keep its filename.",
        embed_fn=fake_embed,
    )
    target = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="old-name.txt",
        data=b"Rename this stored document in place.",
        embed_fn=fake_embed,
    )
    doc_id = str(target["doc_id"])
    keep_id = str(neighbor["doc_id"])
    before = _scroll_doc(qdrant_client, doc_id, with_vectors=True)
    assert before
    before_vectors = {point.id: point.vector for point in before}
    before_text = (before[0].payload or {}).get("text")

    result = patch_document(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        filename="renamed-note.txt",
        extra={"akon_id": "AK110_057"},
    )
    assert result["status"] == "updated"
    assert result["filename"] == "renamed-note.txt"
    assert result["updated_fields"] == ["filename", "extra"]
    assert result["points_updated"] >= 1
    assert result["meta"]["akon_id"] == "AK110_057"

    meta = rag_storage.read_meta(doc_id)
    assert meta is not None
    assert meta.filename == "renamed-note.txt"
    assert meta.extra.get("akon_id") == "AK110_057"
    catalog = rag_storage.list_documents()
    assert any(
        entry["doc_id"] == doc_id and entry["filename"] == "renamed-note.txt"
        for entry in catalog
    )
    assert any(
        entry["doc_id"] == keep_id and entry["filename"] == "keep.txt"
        for entry in catalog
    )

    after = _scroll_doc(qdrant_client, doc_id, with_vectors=True)
    assert len(after) == len(before)
    for point in after:
        payload = point.payload or {}
        assert payload["parent_document"] == "renamed-note.txt"
        assert payload["source"] == "renamed-note.txt"
        assert payload["extra"]["akon_id"] == "AK110_057"
        assert payload["text"] == before_text
        assert point.vector == before_vectors[point.id]

    keep_points = _scroll_doc(qdrant_client, keep_id)
    assert keep_points
    keep_payload = keep_points[0].payload or {}
    assert keep_payload["parent_document"] == "keep.txt"
    assert "extra" not in keep_payload


def test_patch_document_removes_extra_key(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """A null extra value should drop that custom key from catalog and payload."""
    ingested = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="tagged.txt",
        data=b"Document with custom extra metadata.",
        embed_fn=fake_embed,
    )
    doc_id = str(ingested["doc_id"])
    patch_document(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        extra={"akon_id": "AK1", "title": "Keep me"},
    )
    removed = patch_document(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        extra={"akon_id": None},
    )
    assert removed["updated_fields"] == ["extra"]
    meta = rag_storage.read_meta(doc_id)
    assert meta is not None
    assert "akon_id" not in meta.extra
    assert meta.extra.get("title") == "Keep me"
    payload = _scroll_doc(qdrant_client, doc_id)[0].payload or {}
    assert "akon_id" not in (payload.get("extra") or {})
    assert payload["extra"]["title"] == "Keep me"


def test_patch_document_rejects_invalid_fields(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Suffix changes, paths, reserved extra keys, and empty patches fail."""
    ingested = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="note.txt",
        data=b"Validation cases for metadata patch.",
        embed_fn=fake_embed,
    )
    doc_id = str(ingested["doc_id"])
    with pytest.raises(DocumentPatchError, match="filename or extra"):
        patch_document(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            doc_id=doc_id,
        )
    with pytest.raises(DocumentPatchError, match="suffix"):
        patch_document(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            doc_id=doc_id,
            filename="note.pdf",
        )
    with pytest.raises(DocumentPatchError, match="path"):
        patch_document(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            doc_id=doc_id,
            filename="../note.txt",
        )
    with pytest.raises(DocumentPatchError, match="reserved"):
        patch_document(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            doc_id=doc_id,
            extra={"sha256": "nope"},
        )
    point_id = stable_point_id(doc_id, 0)
    before = qdrant_client.retrieve(
        collection_name=COLLECTION,
        ids=[point_id],
        with_payload=True,
        with_vectors=True,
    )[0]
    after_meta = rag_storage.read_meta(doc_id)
    assert after_meta is not None
    assert after_meta.filename == "note.txt"
    after = qdrant_client.retrieve(
        collection_name=COLLECTION,
        ids=[point_id],
        with_payload=True,
        with_vectors=True,
    )[0]
    assert after.vector == before.vector
    assert (after.payload or {})["parent_document"] == "note.txt"


def test_patch_document_missing_id_raises(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
) -> None:
    """Unknown or invalid ids should raise ``DocumentNotFoundError``."""
    missing = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(DocumentNotFoundError, match=missing):
        patch_document(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            doc_id=missing,
            filename="x.txt",
        )
    with pytest.raises(DocumentNotFoundError, match="missing-doc"):
        patch_document(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            doc_id="missing-doc",
            filename="x.txt",
        )


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


def _sse_or_json_payload(response: Any) -> dict[str, Any]:
    """Parse a streamable-HTTP MCP response body into a JSON-RPC object."""
    content_type = response.headers.get("content-type", "")
    text = response.text
    if "text/event-stream" in content_type:
        for line in text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        raise AssertionError(f"No SSE data line in response: {text!r}")
    return response.json()


def _call_tool(
    client: TestClient,
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: int = 2,
) -> str:
    """Invoke an MCP tool and return joined text content."""
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
                "clientInfo": {"name": "pytest", "version": "1"},
            },
        },
    )
    assert init.status_code == 200
    response = client.post(
        "/mcp/",
        headers=MCP_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": call_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    assert response.status_code == 200
    payload = _sse_or_json_payload(response)
    assert "error" not in payload
    result = payload["result"]
    parts = [
        block.get("text", "")
        for block in result.get("content", [])
        if isinstance(block, dict)
    ]
    return "\n".join(parts)


def test_update_document_mcp_tool(client: TestClient) -> None:
    """MCP ``update_document`` should patch metadata and reject unknown ids."""
    upload = client.post(
        "/upload",
        files={"file": ("mcp-old.txt", b"MCP metadata patch target.", "text/plain")},
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]

    raw = _call_tool(
        client,
        "update_document",
        {"doc_id": doc_id, "filename": "mcp-new.txt", "extra": {"series": "A"}},
        call_id=2,
    )
    body = json.loads(raw)
    assert body["status"] == "updated"
    assert body["filename"] == "mcp-new.txt"
    assert body["meta"]["series"] == "A"

    listed = client.get("/documents").json()["documents"]
    assert any(d["doc_id"] == doc_id and d["filename"] == "mcp-new.txt" for d in listed)

    missing = _call_tool(
        client,
        "update_document",
        {"doc_id": "00000000-0000-0000-0000-000000000000", "filename": "nope.txt"},
        call_id=3,
    )
    assert missing.startswith("Error update_document:")
