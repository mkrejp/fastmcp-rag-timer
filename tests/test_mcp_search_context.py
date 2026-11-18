"""MCP tools that attach reader-local context around search hits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from tests.conftest import AUTH_HEADERS, patched_embeddings

MCP_HEADERS = {
    **AUTH_HEADERS,
    "Accept": "application/json, text/event-stream",
    "Host": "localhost",
}


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


def _tool_text(payload: dict[str, Any]) -> str:
    """Join text content blocks from an MCP tools/call result."""
    result = payload["result"]
    parts = [
        block.get("text", "")
        for block in result.get("content", [])
        if isinstance(block, dict)
    ]
    return "\n".join(parts)


def _initialize(client: TestClient) -> None:
    """Complete the MCP initialize handshake."""
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


def _call_tool(
    client: TestClient,
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: int = 2,
) -> str:
    """Invoke an MCP tool and return joined text content."""
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
    return _tool_text(payload)


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


def test_search_with_context_attaches_catalog_and_neighbors(
    client: TestClient,
) -> None:
    """search_with_context returns JSON hits with catalog and neighbor chunks."""
    upload = client.post(
        "/upload",
        files={
            "file": (
                "domain.txt",
                (
                    b"Prague timer RAG domain knowledge for agents. "
                    b"Second chunk expands on municipal locality notes and "
                    b"keeps enough text to form a second processed segment."
                ),
                "text/plain",
            )
        },
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]

    _initialize(client)
    link = _call_tool(
        client,
        "link_locality",
        {
            "links": [
                {
                    "obec_id": "554782",
                    "name": "Praha",
                    "doc_id": doc_id,
                    "chunk_index": 0,
                    "matched_text": "Prague",
                    "char_start": 0,
                    "char_end": 6,
                    "source_type": "text_chunk",
                    "method": "exact",
                }
            ]
        },
        call_id=2,
    )
    assert json.loads(link)["upserted"] >= 1

    raw = _call_tool(
        client,
        "search_with_context",
        {"query": "timer RAG", "limit": 2},
        call_id=3,
    )
    assert not raw.startswith("Error")
    hits = json.loads(raw)
    assert isinstance(hits, list)
    assert hits
    first = hits[0]
    assert first.get("doc_id") == doc_id
    assert first.get("catalog", {}).get("filename") == "domain.txt"
    assert "versions" in first
    assert first.get("locality_links", {}).get("returned", 0) >= 1
    assert "related_documents" in first


def test_get_document_context_and_catalog_search(client: TestClient) -> None:
    """Non-embedding context tools work from catalog/locality alone."""
    upload = client.post(
        "/upload",
        files={
            "file": (
                "catalog-search.txt",
                b"Catalog search fixture text for reader context tools.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]

    _initialize(client)
    context_raw = _call_tool(
        client,
        "get_document_context",
        {"doc_id": doc_id, "chunk_index": 0},
        call_id=2,
    )
    context = json.loads(context_raw)
    assert context["doc_id"] == doc_id
    assert context["catalog"]["filename"] == "catalog-search.txt"
    assert "versions" in context
    assert "locality_links" in context
    assert "related_documents" in context

    missing = _call_tool(
        client,
        "get_document_context",
        {"doc_id": "99999999-9999-9999-9999-999999999999"},
        call_id=3,
    )
    assert missing.startswith("Document not found:")

    catalog_raw = _call_tool(
        client,
        "search_catalog_documents",
        {"query": "catalog-search", "limit": 5, "include_context": True},
        call_id=4,
    )
    catalog_hits = json.loads(catalog_raw)
    assert len(catalog_hits) == 1
    assert catalog_hits[0]["catalog"]["doc_id"] == doc_id


def test_search_with_context_available_in_reader_mode(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reader mode still registers the new read-only context search tools."""
    monkeypatch.setenv("APP_MODE", "reader")
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app, headers=AUTH_HEADERS) as client:
            upload = client.post(
                "/upload",
                files={"file": ("blocked.txt", b"should fail", "text/plain")},
            )
            assert upload.status_code == 403

            # Seed via storage bypass so search tools have something to read.
            from rag.storage import RagStorage

            storage = RagStorage(str(rag_root))
            storage.ensure_layout()
            meta = storage.write_processed(
                doc_id="55555555-5555-5555-5555-555555555555",
                filename="reader.txt",
                mime="text/plain",
                sha256="deadbeef",
                fulltext="reader mode catalog text",
                chunks=[{"index": 0, "text": "reader mode catalog text", "page": 0}],
            )

            _initialize(client)
            catalog_raw = _call_tool(
                client,
                "search_catalog_documents",
                {"query": "reader", "limit": 5},
                call_id=2,
            )
            hits = json.loads(catalog_raw)
            assert hits[0]["doc_id"] == meta.doc_id

            context_raw = _call_tool(
                client,
                "get_document_context",
                {"doc_id": meta.doc_id},
                call_id=3,
            )
            assert json.loads(context_raw)["catalog"]["filename"] == "reader.txt"
