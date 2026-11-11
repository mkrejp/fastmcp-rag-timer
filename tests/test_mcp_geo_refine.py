"""MCP refine_geo / refine_geo_batch / list_geo_refinements and prefer-refined hits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import PointStruct
from rag.config import COLLECTION_NAME
from rag.geo_index import MAX_GEO_BATCH_SIZE
from rag.ingest import stable_point_id
from rag.payloads import stable_figure_point_id
from rag.storage import RagStorage
from rag.vectorstore import build_named_vector
from tests.conftest import AUTH_HEADERS, patched_embeddings
from tests.test_geo_refined import _sample_geo

MCP_HEADERS = {
    **AUTH_HEADERS,
    "Accept": "application/json, text/event-stream",
    "Host": "localhost",
}

FIGURE_DOC_ID = "0b5019ec-dab5-4307-8161-37a6ef474d04"
FIGURE_IMAGE_ID = "9981e4e524e1b54c"


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


def _upload_text(client: TestClient, name: str, body: bytes) -> str:
    """Upload a text document and return its ``doc_id``."""
    response = client.post(
        "/upload",
        files={"file": (name, body, "text/plain")},
    )
    assert response.status_code == 200
    return str(response.json()["doc_id"])


def _set_original_coords(
    client: TestClient,
    doc_id: str,
    *,
    lat: float,
    lon: float,
    chunk_index: int = 0,
) -> None:
    """Write original lat/lon onto a knowledge chunk without touching geo_refined."""
    qdrant = client.app.state.qdrant
    point_id = stable_point_id(doc_id, chunk_index)
    qdrant.set_payload(
        collection_name=COLLECTION_NAME,
        payload={"lat": lat, "lon": lon},
        points=[point_id],
        wait=True,
    )
    storage: RagStorage = client.app.state.storage
    meta = storage.read_meta(doc_id)
    assert meta is not None
    meta.lat = lat
    meta.lon = lon
    storage.write_meta(meta)
    chunks = storage.read_chunks(doc_id) or []
    if chunks and isinstance(chunks[0], dict):
        chunks[0]["lat"] = lat
        chunks[0]["lon"] = lon
        storage.write_chunks(doc_id, chunks)


def _seed_figure(
    client: TestClient,
    rag_root: Path,
    fake_embed: Callable[[str], list[float]],
    *,
    lat: float | None = None,
    lon: float | None = None,
    fulltext: str = "postcard",
    extra: dict[str, Any] | None = None,
) -> None:
    """Insert one pdf_figure point, optionally with original coordinates."""
    storage = RagStorage(rag_root)
    storage.save_original(FIGURE_DOC_ID, "postcard.pdf", b"%PDF-1.4")
    storage.write_processed(
        doc_id=FIGURE_DOC_ID,
        filename="postcard.pdf",
        mime="application/pdf",
        sha256="abc",
        fulltext=fulltext,
        chunks=[{"index": 0, "text": fulltext, "page": 0}],
        images=[],
    )
    meta = storage.read_meta(FIGURE_DOC_ID)
    assert meta is not None
    if lat is not None:
        meta.lat = lat
    if lon is not None:
        meta.lon = lon
    if extra:
        meta.extra.update(extra)
    storage.write_meta(meta)
    qdrant = client.app.state.qdrant
    figure_id = "f_geo_p1_0"
    payload: dict[str, Any] = {
        "source_type": "pdf_figure",
        "figure_id": figure_id,
        "doc_id": FIGURE_DOC_ID,
        "parent_document": "postcard.pdf",
        "image_id": FIGURE_IMAGE_ID,
        "page_number": 0,
        "caption": "Brno Deutsches Haus",
        "text": "Brno Deutsches Haus",
        "verification_status": "unverified",
    }
    if lat is not None:
        payload["lat"] = lat
    if lon is not None:
        payload["lon"] = lon
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=stable_figure_point_id(FIGURE_DOC_ID, figure_id),
                vector=build_named_vector(text_vector=fake_embed(str(payload["text"]))),
                payload=payload,
            )
        ],
        wait=True,
    )


def test_mcp_lists_geo_tools(client: TestClient) -> None:
    """tools/list should advertise refine_geo, refine_geo_batch, and list_geo_refinements."""
    _initialize(client)
    listed = client.post(
        "/mcp/",
        headers=MCP_HEADERS,
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )
    assert listed.status_code == 200
    body = listed.text
    assert "refine_geo" in body
    assert "refine_geo_batch" in body
    assert "list_geo_refinements" in body


def test_refine_geo_dry_run_does_not_write(client: TestClient) -> None:
    """dry_run should report would_change without persisting geo_refined."""
    doc_id = _upload_text(client, "brno.txt", b"Brno Deutsches Haus postcard notes.")
    _set_original_coords(client, doc_id, lat=49.1952, lon=16.6087)
    _initialize(client)
    payload = _call_tool(
        client,
        "refine_geo",
        {
            "doc_id": doc_id,
            "geo_refined": _sample_geo(),
            "dry_run": True,
        },
    )
    body = json.loads(_tool_text(payload))
    assert body["dry_run"] is True
    assert body["written"] is False
    assert body["would_change"] is True
    assert body["changed"] is False
    storage: RagStorage = client.app.state.storage
    meta = storage.read_meta(doc_id)
    assert meta is not None
    assert meta.geo_refined is None
    assert meta.lat == pytest.approx(49.1952)
    assert meta.lon == pytest.approx(16.6087)


def test_refine_geo_write_history_and_idempotent(client: TestClient) -> None:
    """Writes append history on change and stay idempotent for the same object."""
    doc_id = _upload_text(client, "brno.txt", b"Brno Deutsches Haus postcard notes.")
    _set_original_coords(client, doc_id, lat=49.1952, lon=16.6087)
    _initialize(client)
    first = json.loads(
        _tool_text(
            _call_tool(
                client,
                "refine_geo",
                {
                    "doc_id": doc_id,
                    "chunk_index": 0,
                    "geo_refined": _sample_geo(),
                    "dry_run": False,
                },
            )
        )
    )
    assert first["written"] is True
    assert first["changed"] is True
    assert first["idempotent"] is False

    qdrant = client.app.state.qdrant
    point_id = stable_point_id(doc_id, 0)
    stored = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[point_id])[0]
    payload = stored.payload or {}
    assert payload["lat"] == pytest.approx(49.1952)
    assert payload["lon"] == pytest.approx(16.6087)
    assert payload["geo_refined"]["lat"] == pytest.approx(49.1951)
    assert payload["geo_refined_history"] == []

    again = json.loads(
        _tool_text(
            _call_tool(
                client,
                "refine_geo",
                {
                    "doc_id": doc_id,
                    "chunk_index": 0,
                    "geo_refined": _sample_geo(),
                    "dry_run": False,
                },
                call_id=3,
            )
        )
    )
    assert again["written"] is False
    assert again["changed"] is False
    assert again["idempotent"] is True

    moved = json.loads(
        _tool_text(
            _call_tool(
                client,
                "refine_geo",
                {
                    "doc_id": doc_id,
                    "chunk_index": 0,
                    "geo_refined": _sample_geo(lat=49.1953, note="second pass"),
                    "dry_run": False,
                },
                call_id=4,
            )
        )
    )
    assert moved["changed"] is True
    stored = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[point_id])[0]
    payload = stored.payload or {}
    assert payload["lat"] == pytest.approx(49.1952)
    assert len(payload["geo_refined_history"]) == 1
    assert payload["geo_refined"]["lat"] == pytest.approx(49.1953)


def test_refine_geo_rejects_invalid_schema(client: TestClient) -> None:
    """Invalid lat should return a tool-visible error and not write."""
    doc_id = _upload_text(client, "bad.txt", b"Invalid geo refinement target.")
    _initialize(client)
    text = _tool_text(
        _call_tool(
            client,
            "refine_geo",
            {
                "doc_id": doc_id,
                "geo_refined": _sample_geo(lat=99),
                "dry_run": False,
            },
        )
    )
    assert text.startswith("Error refine_geo:")
    assert "lat" in text
    storage: RagStorage = client.app.state.storage
    meta = storage.read_meta(doc_id)
    assert meta is not None
    assert meta.geo_refined is None


def test_list_geo_refinements_filters_and_pagination(client: TestClient) -> None:
    """list_geo_refinements should filter by precision/status and paginate."""
    building = _upload_text(client, "building.txt", b"Building level GPS note.")
    street = _upload_text(client, "street.txt", b"Street level GPS note here.")
    _initialize(client)
    _call_tool(
        client,
        "refine_geo",
        {
            "doc_id": building,
            "geo_refined": _sample_geo(precision="building", precision_m=10),
            "dry_run": False,
        },
        call_id=2,
    )
    _call_tool(
        client,
        "refine_geo",
        {
            "doc_id": street,
            "geo_refined": _sample_geo(
                precision="street",
                precision_m=40,
                confidence_level="medium",
                confidence=0.5,
                status="priblizne",
            ),
            "dry_run": False,
        },
        call_id=3,
    )
    listed = json.loads(
        _tool_text(
            _call_tool(
                client,
                "list_geo_refinements",
                {"filter": {"precision": "building"}},
                call_id=4,
            )
        )
    )
    assert listed["returned"] >= 1
    assert all(
        (row.get("geo_refined") or {}).get("precision") == "building"
        for row in listed["records"]
    )
    needs = json.loads(
        _tool_text(
            _call_tool(
                client,
                "list_geo_refinements",
                {
                    "needs_refinement": True,
                    "doc_id": street,
                },
                call_id=5,
            )
        )
    )
    assert needs["returned"] >= 1
    assert all(row.get("doc_id") == street for row in needs["records"])
    page = json.loads(
        _tool_text(
            _call_tool(
                client,
                "list_geo_refinements",
                {"limit": 1, "offset": 0},
                call_id=6,
            )
        )
    )
    assert page["returned"] == 1
    assert page["has_more"] is True
    assert page["limit"] == 1


def test_search_prefers_refined_coordinates(client: TestClient) -> None:
    """search_domain_knowledge JSON should prefer geo_refined and expose both."""
    doc_id = _upload_text(
        client,
        "prefer.txt",
        b"UniquePreferRefinedPhrase for GPS search hits.",
    )
    _set_original_coords(client, doc_id, lat=49.2, lon=16.6)
    _initialize(client)
    before = json.loads(
        _tool_text(
            _call_tool(
                client,
                "search_domain_knowledge",
                {
                    "query": "UniquePreferRefinedPhrase",
                    "limit": 3,
                    "format": "json",
                },
                call_id=2,
            )
        )
    )
    hit = next(
        item
        for item in before
        if "UniquePreferRefinedPhrase" in str(item.get("content") or "")
    )
    assert hit["geo_source"] == "original"
    assert hit["lat"] == pytest.approx(49.2)
    assert hit["geo_original"]["lat"] == pytest.approx(49.2)
    assert hit["geo_original"]["lon"] == pytest.approx(16.6)
    assert hit["geo_refined"] is None

    _call_tool(
        client,
        "refine_geo",
        {
            "doc_id": doc_id,
            "chunk_index": 0,
            "geo_refined": _sample_geo(),
            "dry_run": False,
        },
        call_id=3,
    )
    after = json.loads(
        _tool_text(
            _call_tool(
                client,
                "search_domain_knowledge",
                {
                    "query": "UniquePreferRefinedPhrase",
                    "limit": 3,
                    "format": "json",
                },
                call_id=4,
            )
        )
    )
    hit = next(
        item
        for item in after
        if "UniquePreferRefinedPhrase" in str(item.get("content") or "")
    )
    assert hit["geo_source"] == "refined"
    assert hit["lat"] == pytest.approx(49.1951)
    assert hit["lon"] == pytest.approx(16.6080)
    assert hit["geo_original"]["lat"] == pytest.approx(49.2)
    assert hit["geo_original"]["lon"] == pytest.approx(16.6)
    assert hit["geo_refined"]["precision"] == "building"


def test_list_image_records_prefers_refined_and_filters(
    client: TestClient,
    rag_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Image records should expose preferred coords and honor precision filters."""
    _seed_figure(client, rag_root, fake_embed, lat=49.2, lon=16.6)
    _initialize(client)
    _call_tool(
        client,
        "refine_geo",
        {
            "doc_id": FIGURE_DOC_ID,
            "image_id": FIGURE_IMAGE_ID,
            "geo_refined": _sample_geo(),
            "dry_run": False,
        },
    )
    listed = json.loads(
        _tool_text(
            _call_tool(
                client,
                "list_image_records",
                {"doc_id": FIGURE_DOC_ID, "precision": "building"},
                call_id=3,
            )
        )
    )
    records = listed["records"]
    assert records
    record = records[0]
    assert record["geo_source"] == "refined"
    assert record["lat"] == pytest.approx(49.1951)
    assert record["geo_original"]["lat"] == pytest.approx(49.2)
    qdrant = client.app.state.qdrant
    stored = qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[stable_figure_point_id(FIGURE_DOC_ID, "f_geo_p1_0")],
    )[0]
    assert (stored.payload or {})["lat"] == pytest.approx(49.2)
    street = json.loads(
        _tool_text(
            _call_tool(
                client,
                "list_image_records",
                {"doc_id": FIGURE_DOC_ID, "precision": "street"},
                call_id=4,
            )
        )
    )
    assert street["returned"] == 0


def test_list_geo_refinements_reads_souradnice_when_payload_has_no_lat_lon(
    client: TestClient,
    rag_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """AKON-style figures store originals in Souradnice text, not payload lat/lon."""
    _seed_figure(
        client,
        rag_root,
        fake_embed,
        fulltext="Brünn Deutsches Haus\nSouradnice: 49.195220, 16.607960",
    )
    _initialize(client)
    _call_tool(
        client,
        "refine_geo",
        {
            "doc_id": FIGURE_DOC_ID,
            "image_id": FIGURE_IMAGE_ID,
            "geo_refined": _sample_geo(),
            "dry_run": False,
        },
    )
    listed = json.loads(
        _tool_text(
            _call_tool(
                client,
                "list_geo_refinements",
                {"doc_id": FIGURE_DOC_ID},
                call_id=3,
            )
        )
    )
    image_rows = [
        row for row in listed["records"] if row.get("image_id") == FIGURE_IMAGE_ID
    ]
    assert image_rows
    original = image_rows[0].get("geo_original")
    assert original is not None
    assert original["lat"] == pytest.approx(49.195220)
    assert original["lon"] == pytest.approx(16.607960)
    assert image_rows[0]["geo_source"] == "refined"
    qdrant = client.app.state.qdrant
    stored = qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[stable_figure_point_id(FIGURE_DOC_ID, "f_geo_p1_0")],
    )[0]
    payload = stored.payload or {}
    assert "lat" not in payload
    assert "lon" not in payload
    assert payload["geo_refined"]["lat"] == pytest.approx(49.1951)


def test_refine_geo_batch_dry_run_write_and_per_item_errors(
    client: TestClient,
) -> None:
    """Batch should isolate item errors and honor dry_run then write."""
    doc_a = _upload_text(client, "a.txt", b"Batch document alpha GPS target.")
    doc_b = _upload_text(client, "b.txt", b"Batch document beta GPS target.")
    _set_original_coords(client, doc_a, lat=50.0, lon=14.4)
    _initialize(client)
    dry = json.loads(
        _tool_text(
            _call_tool(
                client,
                "refine_geo_batch",
                {
                    "dry_run": True,
                    "items": [
                        {"doc_id": doc_a, "geo_refined": _sample_geo()},
                        {"doc_id": doc_b, "geo_refined": _sample_geo(lat=91)},
                        {
                            "doc_id": "00000000-0000-0000-0000-000000000000",
                            "geo_refined": _sample_geo(),
                        },
                    ],
                },
            )
        )
    )
    assert dry["dry_run"] is True
    assert dry["received"] == 3
    assert dry["ok"] == 1
    assert dry["errors"] == 2
    assert dry["would_change"] == 1
    assert dry["changed"] == 0
    assert dry["items"][0]["ok"] is True
    assert dry["items"][0]["would_change"] is True
    assert dry["items"][0]["changed"] is False
    assert dry["items"][1]["ok"] is False
    assert "lat" in str(dry["items"][1]["error"])
    assert dry["items"][2]["ok"] is False
    assert "not found" in str(dry["items"][2]["error"])
    storage: RagStorage = client.app.state.storage
    meta_before = storage.read_meta(doc_a)
    assert meta_before is not None
    assert meta_before.geo_refined is None

    written = json.loads(
        _tool_text(
            _call_tool(
                client,
                "refine_geo_batch",
                {
                    "dry_run": False,
                    "items": [
                        {"doc_id": doc_a, "geo_refined": _sample_geo()},
                        {
                            "doc_id": doc_b,
                            "geo_refined": _sample_geo(precision="street"),
                        },
                    ],
                },
                call_id=3,
            )
        )
    )
    assert written["dry_run"] is False
    assert written["ok"] == 2
    assert written["changed"] == 2
    assert written["would_change"] == 0
    meta_a = storage.read_meta(doc_a)
    assert meta_a is not None
    assert meta_a.geo_refined is not None
    assert meta_a.lat == pytest.approx(50.0)
    meta_b = storage.read_meta(doc_b)
    assert meta_b is not None
    assert meta_b.geo_refined is not None

    again = json.loads(
        _tool_text(
            _call_tool(
                client,
                "refine_geo_batch",
                {
                    "dry_run": False,
                    "items": [{"doc_id": doc_a, "geo_refined": _sample_geo()}],
                },
                call_id=4,
            )
        )
    )
    assert again["changed"] == 0
    assert again["items"][0]["idempotent"] is True
    assert again["items"][0]["changed"] is False


def test_refine_geo_batch_rejects_oversize(client: TestClient) -> None:
    """A batch larger than the max size should fail the whole call."""
    _initialize(client)
    items = [
        {"doc_id": f"{i:08d}-0000-0000-0000-000000000000", "geo_refined": _sample_geo()}
        for i in range(MAX_GEO_BATCH_SIZE + 1)
    ]
    text = _tool_text(
        _call_tool(client, "refine_geo_batch", {"items": items, "dry_run": True})
    )
    assert text.startswith("Error refine_geo_batch:")
    assert "at most" in text
