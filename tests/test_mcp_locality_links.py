"""MCP link_locality and list_links against the payload-only collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from rag.config import LOCALITY_LINKS_COLLECTION
from rag.mcp_tools import _coerce_locality_links
from rag.payloads import LocalityLinkPayloadError
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
        "okres": "Praha",
        "kraj": "Hlavní město Praha",
        "lau_id": "CZ0100554782",
        "doc_id": "doc-mcp-1",
        "parent_document": "obce.pdf",
        "chunk_index": 2,
        "page_number": 3,
        "matched_text": "Praha",
        "char_start": 40,
        "char_end": 45,
        "source_type": "pdf_chunk",
        "method": "exact",
        "chunk_sha256": "deadbeef",
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def client(rag_root: Path, accnt_root: Path) -> Iterator[TestClient]:
    """Build a TestClient without embedding mocks (these tools must not embed)."""
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


def test_mcp_lists_locality_link_tools(client: TestClient) -> None:
    """tools/list should advertise link_locality and list_links."""
    _initialize(client)
    listed = client.post(
        "/mcp/",
        headers=MCP_HEADERS,
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )
    assert listed.status_code == 200
    body = listed.text
    assert "link_locality" in body
    assert "list_links" in body
    assert "verify_link" in body
    assert "link_stats" in body


def test_link_locality_and_list_links_round_trip(client: TestClient) -> None:
    """link_locality should upsert payload-only points that list_links can filter."""
    _initialize(client)
    upsert = _call_tool(
        client,
        "link_locality",
        {
            "links": [
                _sample_link(),
                _sample_link(
                    obec_id="582786",
                    name="Brno",
                    matched_text="Brno",
                    char_start=80,
                    char_end=84,
                    method="lemma",
                    ambiguous=True,
                ),
            ]
        },
    )
    upsert_body = json.loads(_tool_text(upsert))
    assert upsert_body["status"] == "ok"
    assert upsert_body["inserted"] == 2
    assert upsert_body["updated"] == 0
    assert len(upsert_body["ids"]) == 2

    qdrant = client.app.state.qdrant
    info = qdrant.get_collection(LOCALITY_LINKS_COLLECTION)
    assert info.config.params.vectors == {}
    assert int(info.points_count or 0) == 2

    listed = _call_tool(
        client,
        "list_links",
        {"doc_id": "doc-mcp-1", "limit": 10, "offset": 0},
        call_id=3,
    )
    listed_body = json.loads(_tool_text(listed))
    assert listed_body["returned"] == 2

    brno = _call_tool(
        client,
        "list_links",
        {"obec_id": "582786", "ambiguous": True},
        call_id=4,
    )
    brno_body = json.loads(_tool_text(brno))
    assert brno_body["returned"] == 1
    assert brno_body["links"][0]["name"] == "Brno"
    assert brno_body["links"][0]["ambiguous"] is True

    page = _call_tool(
        client,
        "list_links",
        {"doc_id": "doc-mcp-1", "limit": 1, "offset": 0},
        call_id=5,
    )
    page_body = json.loads(_tool_text(page))
    assert page_body["returned"] == 1
    assert page_body["has_more"] is True

    again = _call_tool(
        client,
        "link_locality",
        {"links": [_sample_link(notes="updated")]},
        call_id=6,
    )
    again_body = json.loads(_tool_text(again))
    assert again_body["inserted"] == 0
    assert again_body["updated"] == 1
    assert again_body["ids"] == [upsert_body["ids"][0]]


def test_coerce_locality_links_accepts_object_array_and_json_string() -> None:
    """The tool argument helper should accept a dict, list, or JSON string."""
    single = _coerce_locality_links(_sample_link())
    assert len(single) == 1
    assert single[0]["obec_id"] == "554782"
    encoded = json.dumps([_sample_link(), _sample_link(obec_id="582786")])
    parsed = _coerce_locality_links(encoded)
    assert len(parsed) == 2
    with pytest.raises(LocalityLinkPayloadError, match="at least one"):
        _coerce_locality_links([])


def test_link_locality_rejects_invalid_payload(client: TestClient) -> None:
    """Missing required fields should return a tool-visible error string."""
    _initialize(client)
    payload = _call_tool(
        client,
        "link_locality",
        {"links": [{"doc_id": "doc-mcp-1", "name": "Praha"}]},
    )
    text = _tool_text(payload)
    assert text.startswith("Error link_locality:")
    assert "missing required field" in text


def test_verify_link_and_link_stats_round_trip(client: TestClient) -> None:
    """verify_link should patch an existing point; link_stats should aggregate it."""
    _initialize(client)
    upsert = _call_tool(
        client,
        "link_locality",
        {
            "links": [
                _sample_link(),
                _sample_link(
                    locality_level="cast_obce",
                    cast_obce={"kod": "400637", "name": "Karlín"},
                    matched_text="Karlín",
                    ambiguous=True,
                ),
            ]
        },
    )
    upsert_body = json.loads(_tool_text(upsert))
    praha_id = upsert_body["ids"][0]
    verified = _call_tool(
        client,
        "verify_link",
        {
            "link_id": praha_id,
            "status": "verified",
            "verified_by": "pytest",
            "notes": "ok",
        },
        call_id=3,
    )
    verified_body = json.loads(_tool_text(verified))
    assert verified_body["link"]["verification_status"] == "verified"
    assert verified_body["link"]["verified_by"] == "pytest"
    assert verified_body["link"]["notes"] == "ok"

    listed = _call_tool(
        client,
        "list_links",
        {"locality_level": "cast_obce", "cast_obce": "400637"},
        call_id=4,
    )
    listed_body = json.loads(_tool_text(listed))
    assert listed_body["returned"] == 1
    assert listed_body["links"][0]["cast_obce_name"] == "Karlín"

    stats = _call_tool(client, "link_stats", {}, call_id=5)
    stats_body = json.loads(_tool_text(stats))
    assert stats_body["total"] == 2
    assert stats_body["by_status"]["verified"] == 1
    assert stats_body["by_locality_level"]["cast_obce"] == 1
    assert stats_body["ambiguous"]["true"] == 1

    missing = _call_tool(
        client,
        "verify_link",
        {
            "link_id": "00000000-0000-0000-0000-000000000000",
            "status": "rejected",
        },
        call_id=6,
    )
    assert "Error verify_link:" in _tool_text(missing)
    assert "not found" in _tool_text(missing)
