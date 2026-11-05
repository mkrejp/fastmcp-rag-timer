"""MCP search_domain_knowledge uses the current Qdrant query API."""

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


def test_search_domain_knowledge_returns_hits(client: TestClient) -> None:
    """Ingest then search should return content, not a Qdrant AttributeError."""
    upload = client.post(
        "/upload",
        files={
            "file": (
                "domain.txt",
                b"Prague timer RAG domain knowledge for agents.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 200

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

    call = client.post(
        "/mcp/",
        headers=MCP_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "search_domain_knowledge",
                "arguments": {"query": "timer RAG", "limit": 2},
            },
        },
    )
    assert call.status_code == 200
    payload = _sse_or_json_payload(call)
    assert "error" not in payload
    result = payload["result"]
    text_parts = [
        block.get("text", "")
        for block in result.get("content", [])
        if isinstance(block, dict)
    ]
    joined = "\n".join(text_parts)
    assert "Error executing RAG search" not in joined
    assert "AttributeError" not in joined
    assert "has no attribute 'search'" not in joined
    assert "Prague timer" in joined or "No relevant facts found." != joined


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


def _tool_text(payload: dict[str, Any]) -> str:
    """Join text content blocks from an MCP tools/call result."""
    result = payload["result"]
    parts = [
        block.get("text", "")
        for block in result.get("content", [])
        if isinstance(block, dict)
    ]
    return "\n".join(parts)


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


def test_search_domain_knowledge_optional_locality_filter(
    client: TestClient,
) -> None:
    """obec_id / cast_obce should restrict hits without changing unfiltered search."""
    praha = client.post(
        "/upload",
        files={
            "file": (
                "praha.txt",
                b"Prague timer RAG domain knowledge for agents.",
                "text/plain",
            )
        },
    )
    brno = client.post(
        "/upload",
        files={
            "file": (
                "brno.txt",
                b"Brno factory notes that are a much longer string so vectors differ.",
                "text/plain",
            )
        },
    )
    assert praha.status_code == 200
    assert brno.status_code == 200
    praha_id = praha.json()["doc_id"]

    _initialize(client)
    unfiltered = _call_tool(
        client,
        "search_domain_knowledge",
        {"query": "timer RAG", "limit": 5},
        call_id=2,
    )
    assert "Error executing RAG search" not in unfiltered
    assert "Prague timer" in unfiltered

    link = _call_tool(
        client,
        "link_locality",
        {
            "links": [
                {
                    "obec_id": "554782",
                    "name": "Praha",
                    "doc_id": praha_id,
                    "chunk_index": 0,
                    "matched_text": "Prague",
                    "char_start": 0,
                    "char_end": 6,
                    "source_type": "pdf_chunk",
                    "method": "exact",
                    "locality_level": "cast_obce",
                    "cast_obce": "400637",
                    "cast_obce_name": "Karlín",
                }
            ]
        },
        call_id=3,
    )
    link_payload = json.loads(link)
    assert link_payload["chunks_updated"] == 1

    filtered = _call_tool(
        client,
        "search_domain_knowledge",
        {"query": "timer RAG", "limit": 5, "obec_id": "554782"},
        call_id=4,
    )
    assert "Prague timer" in filtered
    assert "Brno factory" not in filtered

    missing = _call_tool(
        client,
        "search_domain_knowledge",
        {"query": "timer RAG", "limit": 5, "obec_id": "582786"},
        call_id=5,
    )
    assert missing == "No relevant facts found."

    by_part = _call_tool(
        client,
        "search_domain_knowledge",
        {"query": "timer RAG", "limit": 5, "cast_obce": "400637"},
        call_id=6,
    )
    assert "Prague timer" in by_part


def test_list_documents_since_via_mcp(client: TestClient) -> None:
    """list_documents(since=) filters uploaded_at and keeps the empty-catalog string."""
    _initialize(client)
    empty = _call_tool(client, "list_documents", {}, call_id=2)
    assert empty == "No documents stored."

    upload = client.post(
        "/upload",
        files={"file": ("domain.txt", b"catalog since filter", "text/plain")},
    )
    assert upload.status_code == 200

    all_docs = _call_tool(client, "list_documents", {}, call_id=3)
    catalog = json.loads(all_docs)
    assert len(catalog) == 1
    uploaded_at = str(catalog[0]["uploaded_at"])

    future = _call_tool(
        client,
        "list_documents",
        {"since": "2099-01-01T00:00:00Z"},
        call_id=4,
    )
    assert json.loads(future) == []

    past = _call_tool(
        client,
        "list_documents",
        {"since": uploaded_at},
        call_id=5,
    )
    assert len(json.loads(past)) == 1

    bad = _call_tool(
        client,
        "list_documents",
        {"since": "not-a-date"},
        call_id=6,
    )
    assert bad.startswith("Error list_documents:")
