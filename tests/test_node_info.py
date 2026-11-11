"""Tests for public node identity, login HTML, and optional write blocking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from rag.node_info import (
    ROLE_READ,
    ROLE_STOP,
    ROLE_WRITE,
    current_roles,
    format_role_text,
    is_http_write_request,
    node_info_payload,
    parse_rag_nodes,
    parse_roles,
    writes_allowed_for,
)
from tests.conftest import AUTH_HEADERS, patched_embeddings

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

SECRET_ENV = {
    "RAG_API_TOKEN": "super-secret-token",
    "GOOGLE_CLIENT_SECRET": "google-secret",
    "SESSION_SECRET": "session-secret",
    "NOVITA_API_KEY": "novita-secret",
    "MCP_API_TOKEN": "mcp-secret-should-not-leak",
}


def test_parse_roles_czech_and_aliases() -> None:
    """Czech labels, English aliases, and combinations should map to role codes."""
    assert parse_roles("ZÁPIS+ČTENÍ") == frozenset({ROLE_WRITE, ROLE_READ})
    assert parse_roles("ČTENÍ") == frozenset({ROLE_READ})
    assert parse_roles("STOP") == frozenset({ROLE_STOP})
    assert parse_roles("primary") == frozenset({ROLE_WRITE, ROLE_READ})
    assert parse_roles("replica") == frozenset({ROLE_READ})
    assert parse_roles(["write", "read"]) == frozenset({ROLE_WRITE, ROLE_READ})
    assert parse_roles("ZÁPIS, STOP") == frozenset({ROLE_STOP})
    assert format_role_text({ROLE_WRITE, ROLE_READ}) == "ZÁPIS + ČTENÍ"
    assert writes_allowed_for({ROLE_WRITE, ROLE_READ}) is True
    assert writes_allowed_for({ROLE_READ}) is False
    assert writes_allowed_for({ROLE_STOP}) is False


def test_parse_rag_nodes_defaults() -> None:
    """Unset RAG_NODES should yield the two production nodes and their roles."""
    nodes = parse_rag_nodes("")
    assert len(nodes) == 2
    assert nodes[0].name == "rag-timer-00001"
    assert nodes[0].url == "https://rag-timer-00001-production.up.railway.app"
    assert nodes[0].roles == frozenset({ROLE_WRITE, ROLE_READ})
    assert nodes[1].name == "rag-timer-00002"
    assert nodes[1].url == "https://rag-artnouveau.up.railway.app"
    assert nodes[1].roles == frozenset({ROLE_READ})


def test_parse_rag_nodes_json(monkeypatch: MonkeyPatch) -> None:
    """JSON RAG_NODES should ignore unknown keys and keep only public fields."""
    monkeypatch.setenv(
        "RAG_NODES",
        json.dumps(
            [
                {
                    "name": "alpha",
                    "url": "https://alpha.example",
                    "role": "ZÁPIS",
                    "api_key": "should-not-be-copied",
                },
                {
                    "name": "beta",
                    "url": "https://beta.example",
                    "roles": ["ČTENÍ"],
                },
            ]
        ),
    )
    nodes = parse_rag_nodes()
    assert [(node.name, node.url, node.roles) for node in nodes] == [
        ("alpha", "https://alpha.example", frozenset({ROLE_WRITE})),
        ("beta", "https://beta.example", frozenset({ROLE_READ})),
    ]


def test_parse_rag_nodes_pipe_list() -> None:
    """Simple name|url|role lists should parse without JSON."""
    raw = (
        "rag-timer-00001|https://rag-timer-00001-production.up.railway.app|ZÁPIS+ČTENÍ,"
        "rag-timer-00002|https://rag-artnouveau.up.railway.app|ČTENÍ"
    )
    nodes = parse_rag_nodes(raw)
    assert nodes[0].roles == frozenset({ROLE_WRITE, ROLE_READ})
    assert nodes[1].name == "rag-timer-00002"


def test_parse_rag_nodes_invalid_falls_back() -> None:
    """Garbage RAG_NODES should not crash; defaults remain available."""
    nodes = parse_rag_nodes("::::not-nodes::::")
    assert nodes[0].name == "rag-timer-00001"


def test_current_roles_match_railway_service(monkeypatch: MonkeyPatch) -> None:
    """RAILWAY_SERVICE_NAME fastmcp-rag-timer-00002 should map to the replica."""
    monkeypatch.delenv("RAG_NODE_ROLE", raising=False)
    monkeypatch.delenv("RAG_NODE_NAME", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "fastmcp-rag-timer-00002")
    assert current_roles() == frozenset({ROLE_READ})


def test_current_roles_override(monkeypatch: MonkeyPatch) -> None:
    """RAG_NODE_ROLE should win over catalog matching."""
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "rag-timer-00001")
    monkeypatch.setenv("RAG_NODE_ROLE", "STOP")
    assert current_roles() == frozenset({ROLE_STOP})


def test_node_info_payload_exposes_railway_public_fields(
    monkeypatch: MonkeyPatch,
) -> None:
    """Railway identity fields should appear; other env values must not."""
    monkeypatch.setenv(
        "PUBLIC_BASE_URL", "https://rag-timer-00001-production.up.railway.app"
    )
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "rag-timer-00001")
    monkeypatch.setenv("RAILWAY_REPLICA_REGION", "europe-west4-drams3a")
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_NAME", "production")
    monkeypatch.setenv("RAILWAY_PROJECT_ID", "should-not-appear")
    for key, value in SECRET_ENV.items():
        monkeypatch.setenv(key, value)
    payload = node_info_payload()
    dumped = json.dumps(payload)
    this = payload["this"]
    assert this["name"] == "rag-timer-00001"
    assert this["url"] == "https://rag-timer-00001-production.up.railway.app"
    assert this["railway_service_name"] == "rag-timer-00001"
    assert this["railway_replica_region"] == "europe-west4-drams3a"
    assert this["railway_environment_name"] == "production"
    assert this["role"] == "ZÁPIS + ČTENÍ"
    assert payload["nodes"][0]["is_this"] is True
    assert payload["nodes"][1]["role"] == "ČTENÍ"
    assert payload["enforce_node_role"] is False
    assert payload["writes_allowed"] is True
    for secret in SECRET_ENV.values():
        assert secret not in dumped
    assert "should-not-appear" not in dumped
    assert "RAILWAY_PROJECT_ID" not in dumped


def test_is_http_write_request_covers_mutating_routes() -> None:
    """Upload, ingest, and document delete are writes; health and catalog are not."""
    assert is_http_write_request("POST", "/upload") is True
    assert is_http_write_request("POST", "/upload/init") is True
    assert is_http_write_request("PUT", "/upload/abc/chunks/0") is True
    assert is_http_write_request("DELETE", "/upload/abc") is True
    assert is_http_write_request("POST", "/ingest-json") is True
    assert is_http_write_request("DELETE", "/documents/abc") is True
    assert is_http_write_request("GET", "/upload/abc") is False
    assert is_http_write_request("GET", "/health") is False
    assert is_http_write_request("GET", "/node-info") is False
    assert is_http_write_request("GET", "/documents") is False
    assert is_http_write_request("POST", "/mcp/") is False


@pytest.fixture()
def node_client(
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


def test_node_info_and_health_endpoints(node_client: TestClient) -> None:
    """``/node-info`` is public JSON; ``/health`` stays a liveness probe."""
    info = node_client.get("/node-info")
    assert info.status_code == 200
    body = info.json()
    assert "this" in body
    assert len(body["nodes"]) == 2
    assert body["nodes"][0]["url"].startswith("https://")
    health = node_client.get("/health")
    assert health.status_code == 200
    health_body = health.json()
    assert health_body["status"] == "ok"
    assert "rag_data_root" in health_body
    assert "nodes" not in health_body
    assert "this" not in health_body


def test_login_page_shows_czech_node_block(
    node_client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    """The Google sign-in page should list both nodes in Czech with links."""
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "rag-timer-00001")
    monkeypatch.setenv("RAILWAY_REPLICA_REGION", "europe-west4-drams3a")
    monkeypatch.setenv(
        "PUBLIC_BASE_URL", "https://rag-timer-00001-production.up.railway.app"
    )
    response = node_client.get("/auth/login", follow_redirects=False)
    assert response.status_code == 200
    html = response.text
    assert "Informace o této instanci" in html
    assert "Všechny uzly" in html
    assert "ZÁPIS" in html
    assert "ČTENÍ" in html
    assert 'href="https://rag-timer-00001-production.up.railway.app"' in html or (
        "href='https://rag-timer-00001-production.up.railway.app'" in html
    )
    assert "https://rag-artnouveau.up.railway.app" in html
    assert "europe-west4-drams3a" in html
    assert "Služba Railway" in html
    assert "rag-timer-00001" in html
    assert "rag-timer-00002" in html


def test_login_page_escapes_html(
    monkeypatch: MonkeyPatch, node_client: TestClient
) -> None:
    """Node names from env must be escaped on the login page."""
    monkeypatch.setenv(
        "RAG_NODES",
        json.dumps(
            [
                {
                    "name": "<script>alert(1)</script>",
                    "url": "https://safe.example",
                    "role": "ČTENÍ",
                }
            ]
        ),
    )
    response = node_client.get("/auth/login")
    assert response.status_code == 200
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text


def test_write_blocking_off_by_default_on_replica(
    monkeypatch: MonkeyPatch,
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """A ČTENÍ node should still accept writes until the feature flag is on."""
    monkeypatch.setenv("RAG_NODE_ROLE", "ČTENÍ")
    monkeypatch.delenv("RAG_ENFORCE_NODE_ROLE", raising=False)
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app, headers=AUTH_HEADERS) as client:
            upload = client.post(
                "/upload",
                files={"file": ("note.txt", b"still writable", "text/plain")},
            )
            assert upload.status_code == 200


def _app_client(rag_root: Path, accnt_root: Path) -> TestClient:
    """Create a TestClient for the current env (caller holds embeddings patch)."""
    from rag.app import create_app

    app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
    return TestClient(app, headers=AUTH_HEADERS)


def test_write_blocking_on_read_node(
    monkeypatch: MonkeyPatch,
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """When enforcement is on, ČTENÍ nodes reject HTTP writes but keep reads."""
    monkeypatch.setenv("RAG_NODE_ROLE", "ČTENÍ")
    monkeypatch.setenv("RAG_ENFORCE_NODE_ROLE", "1")
    monkeypatch.setenv("RAG_API_TOKEN", "test-rag-token")
    with patched_embeddings(fake_embed):
        with _app_client(rag_root, accnt_root) as client:
            upload = client.post(
                "/upload",
                files={"file": ("note.txt", b"blocked", "text/plain")},
            )
            assert upload.status_code == 403
            body = upload.json()
            assert body["role"] == "ČTENÍ"
            assert "write_nodes" in body
            ingest = client.post(
                "/ingest-json",
                json={"url": "https://example.com/x.json"},
                headers={**AUTH_HEADERS, "Authorization": "Bearer test-rag-token"},
            )
            assert ingest.status_code == 403
            deleted = client.delete("/documents/00000000-0000-0000-0000-000000000000")
            assert deleted.status_code == 403
            listed = client.get("/documents")
            assert listed.status_code == 200
            health = client.get("/health")
            assert health.status_code == 200
            info = client.get("/node-info")
            assert info.status_code == 200
            assert info.json()["enforce_node_role"] is True
            assert info.json()["writes_allowed"] is False


def test_write_blocking_on_stop_node(
    monkeypatch: MonkeyPatch,
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """STOP nodes reject writes when enforcement is enabled."""
    monkeypatch.setenv("RAG_NODE_ROLE", "STOP")
    monkeypatch.setenv("RAG_ENFORCE_NODE_ROLE", "true")
    with patched_embeddings(fake_embed):
        with _app_client(rag_root, accnt_root) as client:
            upload = client.post(
                "/upload",
                files={"file": ("note.txt", b"blocked", "text/plain")},
            )
            assert upload.status_code == 403
            assert upload.json()["role"] == "STOP"


def test_write_blocking_allows_primary(
    monkeypatch: MonkeyPatch,
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Enforcement on a ZÁPIS+ČTENÍ node should still allow uploads."""
    monkeypatch.setenv("RAG_NODE_ROLE", "ZÁPIS+ČTENÍ")
    monkeypatch.setenv("RAG_ENFORCE_NODE_ROLE", "1")
    with patched_embeddings(fake_embed):
        with _app_client(rag_root, accnt_root) as client:
            upload = client.post(
                "/upload",
                files={"file": ("ok.txt", b"primary still writes", "text/plain")},
            )
            assert upload.status_code == 200


def _initialize_mcp(client: TestClient) -> None:
    """Complete the MCP initialize handshake."""
    init = client.post(
        "/mcp/",
        headers={
            **AUTH_HEADERS,
            "Accept": "application/json, text/event-stream",
            "Host": "localhost",
        },
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


def test_mcp_write_blocked_read_allowed(
    monkeypatch: MonkeyPatch,
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Enforced ČTENÍ nodes reject MCP ingest_text but still list documents."""
    monkeypatch.setenv("RAG_NODE_ROLE", "ČTENÍ")
    monkeypatch.setenv("RAG_ENFORCE_NODE_ROLE", "1")
    with patched_embeddings(fake_embed):
        with _app_client(rag_root, accnt_root) as client:
            _initialize_mcp(client)
            headers = {
                **AUTH_HEADERS,
                "Accept": "application/json, text/event-stream",
                "Host": "localhost",
            }
            ingest = client.post(
                "/mcp/",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "ingest_text",
                        "arguments": {"filename": "x.txt", "content": "nope"},
                    },
                },
            )
            assert ingest.status_code == 200
            ingest_payload = _sse_or_json_payload(ingest)
            joined = json.dumps(ingest_payload)
            assert "Writes are disabled" in joined
            listed = client.post(
                "/mcp/",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "list_documents", "arguments": {}},
                },
            )
            assert listed.status_code == 200
            listed_payload = _sse_or_json_payload(listed)
            listed_text = json.dumps(listed_payload)
            assert "Writes are disabled" not in listed_text
            assert "Error list_documents" not in listed_text
