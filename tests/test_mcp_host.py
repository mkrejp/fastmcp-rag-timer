"""MCP must accept the public Railway Host header (no 421)."""

from fastapi.testclient import TestClient

from rag.app import create_app

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
HDRS = {"Accept": "application/json, text/event-stream", "Host": "rag-timer-00001-production.up.railway.app"}


def test_mcp_accepts_railway_host(tmp_path):
    app = create_app(str(tmp_path / "rag"), accnt_data_root=str(tmp_path / "accnt"))
    with TestClient(app) as client:
        r = client.post("/mcp/", json=INIT, headers=HDRS)
        assert r.status_code == 200, r.text
        r = client.post("/mcp/", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, headers=HDRS)
        assert r.status_code == 200
        for name in ("list_documents", "search_domain_knowledge"):
            assert name in r.text


def test_mcp_rejects_unknown_host(tmp_path):
    app = create_app(str(tmp_path / "rag"), accnt_data_root=str(tmp_path / "accnt"))
    with TestClient(app) as client:
        r = client.post("/mcp/", json=INIT, headers={**HDRS, "Host": "evil.example"})
        assert r.status_code == 421
