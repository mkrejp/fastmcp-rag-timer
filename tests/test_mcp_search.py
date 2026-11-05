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
