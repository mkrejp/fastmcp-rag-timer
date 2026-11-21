"""Optional MCP_READ_TOKEN: read-only /mcp access without HTTP write rights."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

from rag.auth import JSONRPC_FORBIDDEN, classify_mcp_bearer
from rag.config import resolve_mcp_read_token, resolve_mcp_token
from rag.node_info import MCP_WRITE_TOOLS
from tests.conftest import AUTH_HEADERS, TEST_MCP_TOKEN, patched_embeddings

TEST_READ_TOKEN = "test-mcp-read-token"
READ_AUTH = {"Authorization": f"Bearer {TEST_READ_TOKEN}"}
MCP_HOST_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Host": "localhost",
}
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
READ_TOOL_NAMES = (
    "list_documents",
    "get_document_text",
    "search_domain_knowledge",
    "list_links",
    "link_stats",
    "list_geo_refinements",
)


def _sse_or_json_payload(response: Any) -> dict[str, Any]:
    """Parse a streamable-HTTP MCP response body into a JSON-RPC object."""
    content_type = response.headers.get("content-type", "")
    text = response.text
    if "text/event-stream" in content_type:
        for line in text.splitlines():
            if line.startswith("data:"):
                parsed: object = json.loads(line[5:].strip())
                assert isinstance(parsed, dict)
                return parsed
        raise AssertionError(f"No SSE data line in response: {text!r}")
    body = response.json()
    assert isinstance(body, dict)
    return body


def _tool_names(payload: dict[str, Any]) -> set[str]:
    """Return advertised MCP tool names from a tools/list JSON-RPC result."""
    result = payload.get("result")
    assert isinstance(result, dict)
    tools = result.get("tools")
    assert isinstance(tools, list)
    names: set[str] = set()
    for tool in tools:
        assert isinstance(tool, dict)
        name = tool.get("name")
        assert isinstance(name, str)
        names.add(name)
    return names


def _initialize(client: TestClient, headers: dict[str, str]) -> None:
    """Complete the MCP initialize handshake."""
    init = client.post("/mcp/", json=INIT, headers={**MCP_HOST_HEADERS, **headers})
    assert init.status_code == 200


@pytest.fixture()
def dual_token_client(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> Iterator[TestClient]:
    """App with write MCP token (autouse) and MCP_READ_TOKEN both set."""
    monkeypatch.setenv("MCP_READ_TOKEN", TEST_READ_TOKEN)
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture()
def ingester_dual_token_client(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> Iterator[TestClient]:
    """Ingester app with write tools registered and MCP_READ_TOKEN set."""
    monkeypatch.setenv("APP_MODE", "ingester")
    monkeypatch.setenv("MCP_READ_TOKEN", TEST_READ_TOKEN)
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as test_client:
            yield test_client


def test_resolve_mcp_read_token_unset(monkeypatch: MonkeyPatch) -> None:
    """Empty or missing MCP_READ_TOKEN is treated as unset."""
    monkeypatch.delenv("MCP_READ_TOKEN", raising=False)
    assert resolve_mcp_read_token() is None
    monkeypatch.setenv("MCP_READ_TOKEN", "  ")
    assert resolve_mcp_read_token() is None


def test_resolve_mcp_read_token_set(monkeypatch: MonkeyPatch) -> None:
    """Configured MCP_READ_TOKEN is returned stripped."""
    monkeypatch.setenv("MCP_READ_TOKEN", f"  {TEST_READ_TOKEN}  ")
    assert resolve_mcp_read_token() == TEST_READ_TOKEN


def test_resolve_mcp_token_unchanged_with_read_token(
    monkeypatch: MonkeyPatch,
) -> None:
    """MCP_READ_TOKEN must not replace MCP_API_TOKEN / RAG_API_TOKEN."""
    monkeypatch.setenv("MCP_API_TOKEN", TEST_MCP_TOKEN)
    monkeypatch.setenv("MCP_READ_TOKEN", TEST_READ_TOKEN)
    assert resolve_mcp_token() == TEST_MCP_TOKEN
    monkeypatch.delenv("MCP_API_TOKEN", raising=False)
    monkeypatch.setenv("RAG_API_TOKEN", "fallback-write-token")
    assert resolve_mcp_token() == "fallback-write-token"


def test_classify_mcp_bearer_write_wins(monkeypatch: MonkeyPatch) -> None:
    """Matching write token is write-capable even when a read token is set."""
    monkeypatch.setenv("MCP_API_TOKEN", TEST_MCP_TOKEN)
    monkeypatch.setenv("MCP_READ_TOKEN", TEST_READ_TOKEN)
    assert classify_mcp_bearer(f"Bearer {TEST_MCP_TOKEN}") == "write"
    assert classify_mcp_bearer(f"Bearer {TEST_READ_TOKEN}") == "read"
    assert classify_mcp_bearer("Bearer wrong-token") is None
    assert classify_mcp_bearer(None) is None


def test_write_token_still_lists_and_calls_write_tools(
    dual_token_client: TestClient,
) -> None:
    """Existing MCP_API_TOKEN clients keep full access when MCP_READ_TOKEN is set."""
    _initialize(dual_token_client, AUTH_HEADERS)
    listed = dual_token_client.post(
        "/mcp/",
        json=LIST,
        headers={**MCP_HOST_HEADERS, **AUTH_HEADERS},
    )
    assert listed.status_code == 200
    names = _tool_names(_sse_or_json_payload(listed))
    assert "list_documents" in names
    assert "ingest_text" in names
    for write_name in MCP_WRITE_TOOLS:
        assert write_name in names

    called = dual_token_client.post(
        "/mcp/",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "ingest_text",
                "arguments": {"filename": "note.txt", "content": "hello write token"},
            },
        },
        headers={**MCP_HOST_HEADERS, **AUTH_HEADERS},
    )
    assert called.status_code == 200
    payload = _sse_or_json_payload(called)
    assert "error" not in payload
    dumped = json.dumps(payload)
    assert "hello write token" in dumped or "Success" in dumped or "doc_id" in dumped


def test_rag_api_token_fallback_still_works(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> None:
    """RAG_API_TOKEN remains the write MCP token when MCP_API_TOKEN is unset."""
    monkeypatch.delenv("MCP_API_TOKEN", raising=False)
    monkeypatch.setenv("RAG_API_TOKEN", "fallback-write-token")
    monkeypatch.setenv("MCP_READ_TOKEN", TEST_READ_TOKEN)
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        headers = {"Authorization": "Bearer fallback-write-token"}
        with TestClient(app) as client:
            _initialize(client, headers)
            listed = client.post(
                "/mcp/",
                json=LIST,
                headers={**MCP_HOST_HEADERS, **headers},
            )
            assert listed.status_code == 200
            names = _tool_names(_sse_or_json_payload(listed))
            assert "ingest_text" in names


def test_read_token_lists_only_read_tools(dual_token_client: TestClient) -> None:
    """MCP_READ_TOKEN tools/list omits mutating tools."""
    _initialize(dual_token_client, READ_AUTH)
    listed = dual_token_client.post(
        "/mcp/",
        json=LIST,
        headers={**MCP_HOST_HEADERS, **READ_AUTH},
    )
    assert listed.status_code == 200
    names = _tool_names(_sse_or_json_payload(listed))
    for read_name in READ_TOOL_NAMES:
        assert read_name in names
    for write_name in MCP_WRITE_TOOLS:
        assert write_name not in names


def test_read_token_can_call_read_tools(dual_token_client: TestClient) -> None:
    """MCP_READ_TOKEN may call read tools such as list_documents."""
    _initialize(dual_token_client, READ_AUTH)
    called = dual_token_client.post(
        "/mcp/",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "list_documents", "arguments": {}},
        },
        headers={**MCP_HOST_HEADERS, **READ_AUTH},
    )
    assert called.status_code == 200
    payload = _sse_or_json_payload(called)
    assert "error" not in payload
    dumped = json.dumps(payload)
    assert "Writes are disabled" not in dumped
    assert "Read-only token" not in dumped


@pytest.mark.parametrize("tool_name", sorted(MCP_WRITE_TOOLS))
def test_read_token_rejected_for_write_tools(
    dual_token_client: TestClient, tool_name: str
) -> None:
    """MCP_READ_TOKEN tools/call of every mutating tool is JSON-RPC 403."""
    _initialize(dual_token_client, READ_AUTH)
    called = dual_token_client.post(
        "/mcp/",
        json={
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": {}},
        },
        headers={**MCP_HOST_HEADERS, **READ_AUTH},
    )
    assert called.status_code == 403
    payload = called.json()
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == 9
    assert payload["error"]["code"] == JSONRPC_FORBIDDEN
    assert "write" in payload["error"]["message"].lower()
    assert TEST_READ_TOKEN not in called.text
    assert TEST_MCP_TOKEN not in called.text


def test_read_token_rejected_for_write_tools_on_ingester(
    ingester_dual_token_client: TestClient,
) -> None:
    """Read-only MCP still applies when APP_MODE is ingester (write tools exist)."""
    _initialize(ingester_dual_token_client, READ_AUTH)
    listed = ingester_dual_token_client.post(
        "/mcp/",
        json=LIST,
        headers={**MCP_HOST_HEADERS, **READ_AUTH},
    )
    assert listed.status_code == 200
    names = _tool_names(_sse_or_json_payload(listed))
    assert "list_documents" in names
    assert "ingest_text" not in names
    called = ingester_dual_token_client.post(
        "/mcp/",
        json={
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "ingest_text",
                "arguments": {"filename": "x.txt", "content": "should not ingest"},
            },
        },
        headers={**MCP_HOST_HEADERS, **READ_AUTH},
    )
    assert called.status_code == 403
    assert called.json()["error"]["code"] == JSONRPC_FORBIDDEN


def test_read_token_rejected_for_http_write_routes(
    dual_token_client: TestClient,
) -> None:
    """MCP_READ_TOKEN must not authorize upload, PDF upload, or document mutations."""
    headers = READ_AUTH
    upload = dual_token_client.post(
        "/upload",
        files={"file": ("a.txt", b"nope", "text/plain")},
        headers=headers,
    )
    assert upload.status_code == 401
    pdf = dual_token_client.post(
        "/upload-pdf",
        files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        headers=headers,
    )
    assert pdf.status_code == 401
    deleted = dual_token_client.delete("/documents/not-a-real-doc", headers=headers)
    assert deleted.status_code == 401
    patched = dual_token_client.patch(
        "/documents/not-a-real-doc",
        json={"filename": "x.txt"},
        headers=headers,
    )
    assert patched.status_code == 401
    metadata = dual_token_client.patch(
        "/documents/not-a-real-doc/metadata",
        json={"reliability": 1},
        headers=headers,
    )
    assert metadata.status_code == 401
    ingest_json = dual_token_client.post(
        "/ingest-json",
        json={"url": "https://example.com/data.json"},
        headers=headers,
    )
    assert ingest_json.status_code in {401, 503}


def test_read_token_rejected_for_sync_and_admin(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> None:
    """MCP_READ_TOKEN must not grant reader sync or Google admin routes."""
    monkeypatch.setenv("APP_MODE", "reader")
    monkeypatch.setenv("EMBEDDING_PROVIDER_QUERY", "novita")
    monkeypatch.setenv("MCP_READ_TOKEN", TEST_READ_TOKEN)
    monkeypatch.setenv("RAG_SYNC_TOKEN", "real-sync-token")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "dummy-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "dummy-client-secret")
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as client:
            sync = client.get("/sync/v1/watermark", headers=READ_AUTH)
            assert sync.status_code == 401
            admin = client.get("/api/admin/mcp", headers=READ_AUTH)
            assert admin.status_code == 401


def test_wrong_token_is_401(dual_token_client: TestClient) -> None:
    """A bearer that matches neither write nor read token is 401."""
    headers = {**MCP_HOST_HEADERS, "Authorization": "Bearer wrong-token"}
    response = dual_token_client.post("/mcp/", json=INIT, headers=headers)
    assert response.status_code == 401
    assert "invalid" in response.json()["detail"].lower()
    assert TEST_READ_TOKEN not in response.text
    assert TEST_MCP_TOKEN not in response.text


def test_read_token_only_allows_mcp_not_http_writes(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> None:
    """MCP_READ_TOKEN alone is enough for read /mcp; HTTP writes stay fail-closed."""
    monkeypatch.delenv("MCP_API_TOKEN", raising=False)
    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    monkeypatch.setenv("MCP_READ_TOKEN", TEST_READ_TOKEN)
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as client:
            _initialize(client, READ_AUTH)
            listed = client.post(
                "/mcp/",
                json=LIST,
                headers={**MCP_HOST_HEADERS, **READ_AUTH},
            )
            assert listed.status_code == 200
            names = _tool_names(_sse_or_json_payload(listed))
            assert "list_documents" in names
            assert "ingest_text" not in names
            upload = client.post(
                "/upload",
                files={"file": ("a.txt", b"x", "text/plain")},
                headers=READ_AUTH,
            )
            assert upload.status_code == 503


def test_read_token_unset_behaviour_unchanged(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> None:
    """When MCP_READ_TOKEN is unset, only the write token works and lists write tools."""
    monkeypatch.delenv("MCP_READ_TOKEN", raising=False)
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as client:
            rejected = client.post(
                "/mcp/",
                json=INIT,
                headers={**MCP_HOST_HEADERS, **READ_AUTH},
            )
            assert rejected.status_code == 401
            _initialize(client, AUTH_HEADERS)
            listed = client.post(
                "/mcp/",
                json=LIST,
                headers={**MCP_HOST_HEADERS, **AUTH_HEADERS},
            )
            assert listed.status_code == 200
            names = _tool_names(_sse_or_json_payload(listed))
            assert "ingest_text" in names
            assert "list_documents" in names


def test_tokens_are_never_logged(
    dual_token_client: TestClient,
    caplog: LogCaptureFixture,
) -> None:
    """Write and read bearer values must not appear in log output."""
    with caplog.at_level(logging.DEBUG):
        _initialize(dual_token_client, READ_AUTH)
        dual_token_client.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "ingest_text", "arguments": {}},
            },
            headers={**MCP_HOST_HEADERS, **READ_AUTH},
        )
        dual_token_client.post(
            "/mcp/",
            json=INIT,
            headers={**MCP_HOST_HEADERS, "Authorization": "Bearer wrong-token"},
        )
        dual_token_client.post(
            "/upload",
            files={"file": ("a.txt", b"x", "text/plain")},
            headers=READ_AUTH,
        )
    joined = caplog.text
    assert TEST_READ_TOKEN not in joined
    assert TEST_MCP_TOKEN not in joined
