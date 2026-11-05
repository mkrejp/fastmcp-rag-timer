"""MCP host allow-list (no 421 on Railway) and bearer auth on /mcp and embedding routes."""

import pytest
from fastapi.testclient import TestClient

from rag.app import create_app

INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "t", "version": "1"},
    },
}
LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
HDRS = {
    "Accept": "application/json, text/event-stream",
    "Host": "rag-timer-00001-production.up.railway.app",
}
AUTH = {"Authorization": "Bearer test-mcp-token"}


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "rag"), accnt_data_root=str(tmp_path / "accnt"))
    with TestClient(app) as c:
        yield c


def test_mcp_accepts_railway_host_with_token(client):
    assert client.post("/mcp/", json=INIT, headers={**HDRS, **AUTH}).status_code == 200
    r = client.post("/mcp/", json=LIST, headers={**HDRS, **AUTH})
    assert r.status_code == 200
    for name in (
        "list_documents",
        "get_document_text",
        "get_document_chunks",
        "ingest_text",
        "search_domain_knowledge",
    ):
        assert name in r.text


def test_mcp_rejects_unknown_host(client):
    assert (
        client.post(
            "/mcp/", json=INIT, headers={**HDRS, **AUTH, "Host": "evil.example"}
        ).status_code
        == 421
    )


def test_mcp_requires_bearer(client):
    assert client.post("/mcp/", json=INIT, headers=HDRS).status_code == 401
    assert (
        client.post(
            "/mcp/", json=INIT, headers={**HDRS, "Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )


def test_upload_requires_bearer(client):
    files = {"file": ("a.txt", b"hello", "text/plain")}
    assert client.post("/upload", files=files).status_code == 401


def test_delete_document_requires_bearer(client):
    """DELETE /documents/{doc_id} should reject missing bearer tokens."""
    assert client.delete("/documents/not-a-real-doc").status_code == 401


def test_fail_closed_without_token(client, monkeypatch):
    monkeypatch.delenv("MCP_API_TOKEN", raising=False)
    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    assert client.post("/mcp/", json=INIT, headers={**HDRS, **AUTH}).status_code == 503
    assert (
        client.post(
            "/upload", files={"file": ("a.txt", b"x", "text/plain")}
        ).status_code
        == 503
    )
    assert client.delete("/documents/not-a-real-doc").status_code == 503
