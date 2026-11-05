"""MCP Trails forwarding: mocked HTTP, outbox, dry_run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import AUTH_HEADERS

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


def _sample_link(**overrides: object) -> dict[str, object]:
    """Return a valid locality-link object for MCP tool arguments."""
    payload: dict[str, object] = {
        "obec_id": "554782",
        "name": "Praha",
        "doc_id": "doc-trails-1",
        "chunk_index": 0,
        "matched_text": "Praha",
        "char_start": 0,
        "char_end": 5,
        "source_type": "pdf_chunk",
        "method": "exact",
        "chunk_sha256": "deadbeef",
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def client(
    rag_root: Path, accnt_root: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Build a TestClient with Trails forwarding enabled."""
    monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
    monkeypatch.setenv("TRAILS_API_TOKEN", "super-secret-token")
    from rag.app import create_app

    app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
    with TestClient(app, headers=AUTH_HEADERS) as test_client:
        yield test_client


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
) -> dict[str, Any]:
    """Invoke an MCP tool and return the JSON-RPC payload."""
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
    return payload


def test_link_locality_forwards_to_trails(client: TestClient) -> None:
    """link_locality should POST Trails and still return local upsert ok."""
    _initialize(client)
    trails_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    def fake_request(method: str, url: str, **kwargs: Any) -> Any:
        assert "super-secret-token" in kwargs["headers"]["Authorization"]
        assert method == "POST"
        response = type("Resp", (), {})()
        response.status_code = 200
        response.content = b"{}"
        response.text = "{}"
        response.json = lambda: {
            "inserted": 1,
            "updated": 0,
            "ids": [trails_id],
        }
        return response

    with (
        patch("rag.trails_sync.requests.request", side_effect=fake_request),
        patch("rag.trails_sync.time.sleep"),
    ):
        payload = _call_tool(client, "link_locality", {"links": [_sample_link()]})
    body = json.loads(_tool_text(payload))
    assert body["status"] == "ok"
    assert body["inserted"] == 1
    assert body["trails"]["enabled"] is True
    assert body["trails"]["inserted"] == 1
    assert "super-secret-token" not in _tool_text(payload)


def test_link_locality_survives_trails_500(client: TestClient) -> None:
    """Trails 500 must not fail link_locality; the outbox should hold the id."""
    _initialize(client)

    def fake_request(*_args: object, **_kwargs: object) -> Any:
        response = type("Resp", (), {})()
        response.status_code = 500
        response.content = b"down"
        response.text = "down"
        response.json = lambda: {}
        return response

    with (
        patch("rag.trails_sync.requests.request", side_effect=fake_request),
        patch("rag.trails_sync.time.sleep"),
    ):
        payload = _call_tool(client, "link_locality", {"links": [_sample_link()]})
    body = json.loads(_tool_text(payload))
    assert body["status"] == "ok"
    assert body["trails"]["queued"] == 1
    outbox = Path(str(client.app.state.storage.trails_outbox_path))
    saved = json.loads(outbox.read_text(encoding="utf-8"))
    assert saved["items"][0]["link_ids"] == body["ids"]
    assert "super-secret-token" not in outbox.read_text(encoding="utf-8")


def test_verify_link_and_sync_dry_run(client: TestClient) -> None:
    """verify_link PATCHes Trails; sync_links_to_trails dry_run counts locally."""
    _initialize(client)
    methods: list[str] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> Any:
        methods.append(method)
        response = type("Resp", (), {})()
        response.status_code = 200
        response.content = b"{}"
        response.text = "{}"
        if method == "POST":
            response.json = lambda: {
                "inserted": 1,
                "updated": 0,
                "ids": ["bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"],
            }
        else:
            assert kwargs["json"]["status"] == "verified"
            response.json = lambda: {}
        return response

    with (
        patch("rag.trails_sync.requests.request", side_effect=fake_request),
        patch("rag.trails_sync.time.sleep"),
    ):
        created = _call_tool(
            client, "link_locality", {"links": [_sample_link()]}, call_id=2
        )
        link_id = json.loads(_tool_text(created))["ids"][0]
        verified = _call_tool(
            client,
            "verify_link",
            {"link_id": link_id, "status": "verified", "verified_by": "agent"},
            call_id=3,
        )
        dry = _call_tool(
            client, "sync_links_to_trails", {"dry_run": True}, call_id=4
        )
    verified_body = json.loads(_tool_text(verified))
    assert verified_body["link"]["verification_status"] == "verified"
    assert verified_body["trails"]["verified"] == 1
    assert "PATCH" in methods
    dry_body = json.loads(_tool_text(dry))
    assert dry_body["dry_run"] is True
    assert dry_body["scanned"] == 1
    assert dry_body["would_verify"] == 1
