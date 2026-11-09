"""MCP tests for figure-clip ``get_image`` and ``get_document_images``."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from qdrant_client.models import PointStruct

from rag.config import COLLECTION_NAME
from rag.payloads import stable_figure_point_id
from rag.storage import RagStorage
from rag.vectorstore import build_named_vector
from tests.conftest import AUTH_HEADERS, patched_embeddings
from tests.fixtures.pdf_figure_factory import build_figure_caption_pdf

MCP_HEADERS = {
    **AUTH_HEADERS,
    "Accept": "application/json, text/event-stream",
    "Host": "localhost",
}

DOC_ID = "0b5019ec-dab5-4307-8161-37a6ef474d04"
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


def _call_tool_payload(
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


def _tool_text(payload: dict[str, Any]) -> str:
    """Join text content blocks from an MCP tools/call result."""
    result = payload["result"]
    parts = [
        block.get("text", "")
        for block in result.get("content", [])
        if isinstance(block, dict)
    ]
    return "\n".join(parts)


def _png_bytes() -> bytes:
    """Return a tiny PNG used as a figure clip stand-in."""
    image = Image.new("RGB", (16, 16), "maroon")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _seed_orphan_figure(storage: RagStorage, png: bytes) -> None:
    """Write a document whose figure PNG is on disk but not in meta.images."""
    storage.save_original(DOC_ID, "wiener-secession-1898.pdf", b"%PDF-1.4")
    storage.write_processed(
        doc_id=DOC_ID,
        filename="wiener-secession-1898.pdf",
        mime="application/pdf",
        sha256="abc",
        fulltext="secession",
        chunks=[{"index": 0, "text": "secession", "page": 0}],
        images=[],
    )
    storage.save_image_bytes(
        DOC_ID, image_id=FIGURE_IMAGE_ID, extension=".png", data=png
    )


def _upsert_figure_point(client: TestClient, fake_embed: Callable[[str], list[float]]) -> None:
    """Insert one ``pdf_figure`` Qdrant point for ``DOC_ID``."""
    qdrant = client.app.state.qdrant
    figure_id = "f_test_p1_0"
    payload = {
        "source_type": "pdf_figure",
        "figure_id": figure_id,
        "doc_id": DOC_ID,
        "parent_document": "wiener-secession-1898.pdf",
        "image_id": FIGURE_IMAGE_ID,
        "image_sha256": "abc",
        "file_path": f"processed/{DOC_ID}/images/{FIGURE_IMAGE_ID}.png",
        "page_number": 0,
        "caption": "Obr. 1 Wiener Secession",
        "summary": "Exhibition catalog figure.",
        "ocr_text": "",
        "ocr_confidence": None,
        "text": "Obr. 1 Wiener Secession",
    }
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=stable_figure_point_id(DOC_ID, figure_id),
                vector=build_named_vector(text_vector=fake_embed(str(payload["text"]))),
                payload=payload,
            )
        ],
        wait=True,
    )


def test_get_image_serves_figure_png(
    client: TestClient,
    rag_root: Path,
) -> None:
    """MCP ``get_image`` should return figure clip bytes when id is not in meta.images."""
    png = _png_bytes()
    _seed_orphan_figure(RagStorage(rag_root), png)
    _initialize(client)
    payload = _call_tool_payload(client, "get_image", {"image_id": FIGURE_IMAGE_ID})
    blocks = payload["result"]["content"]
    image_blocks = [
        block
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "image"
    ]
    assert image_blocks, f"expected image content, got {blocks!r}"
    assert str(image_blocks[0].get("mimeType") or "").startswith("image/")
    assert image_blocks[0].get("data")


def test_get_image_unknown_id_message(client: TestClient) -> None:
    """MCP ``get_image`` should distinguish unknown ids from missing files."""
    _initialize(client)
    text = _tool_text(
        _call_tool_payload(client, "get_image", {"image_id": "abcd1234abcd1234"})
    )
    assert "Unknown image id" in text
    assert "abcd1234abcd1234" in text


def test_get_document_images_include_figures(
    client: TestClient,
    rag_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """``include_figures`` should append pdf_figure items; default stays extracted-only."""
    png = _png_bytes()
    _seed_orphan_figure(RagStorage(rag_root), png)
    _upsert_figure_point(client, fake_embed)
    _initialize(client)

    default_text = _tool_text(
        _call_tool_payload(
            client,
            "get_document_images",
            {"doc_id": DOC_ID},
            call_id=2,
        )
    )
    default_items = json.loads(default_text)
    assert default_items == []

    with_figures = json.loads(
        _tool_text(
            _call_tool_payload(
                client,
                "get_document_images",
                {"doc_id": DOC_ID, "include_figures": True},
                call_id=3,
            )
        )
    )
    assert len(with_figures) == 1
    item = with_figures[0]
    assert item["image_id"] == FIGURE_IMAGE_ID
    assert item["page_number"] == 0
    assert item["caption"] == "Obr. 1 Wiener Secession"
    assert item["summary"] == "Exhibition catalog figure."
    assert item["source_type"] == "pdf_figure"
    assert "url" in item


def test_mcp_ingested_figure_round_trip(
    client: TestClient,
    rag_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Ingest a figure PDF, then fetch and list the clip through MCP tools."""

    def fake_many(texts: list[str], **_kwargs: object) -> list[list[float]]:
        """Map a batch of texts through ``fake_embed``."""
        return [fake_embed(text) for text in texts]

    with (
        patch("rag.figure_index.get_ingest_embeddings", side_effect=fake_many),
        patch(
            "rag.pdf_figures.ocr_pil_image",
            return_value=("", "", None, 0, "none"),
        ),
    ):
        upload = client.post(
            "/upload",
            files={
                "file": (
                    "figure.pdf",
                    build_figure_caption_pdf(),
                    "application/pdf",
                )
            },
        )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]
    _initialize(client)

    records_text = _tool_text(
        _call_tool_payload(
            client,
            "list_image_records",
            {"doc_id": doc_id, "limit": 20},
            call_id=2,
        )
    )
    records = json.loads(records_text)
    figure_records = records.get("records") or []
    assert figure_records, records_text
    image_id = str(figure_records[0]["image_id"])

    listed_default = json.loads(
        _tool_text(
            _call_tool_payload(
                client, "get_document_images", {"doc_id": doc_id}, call_id=3
            )
        )
    )
    listed_ids = {item.get("image_id") for item in listed_default}
    assert image_id not in listed_ids

    listed_figures = json.loads(
        _tool_text(
            _call_tool_payload(
                client,
                "get_document_images",
                {"doc_id": doc_id, "include_figures": True},
                call_id=4,
            )
        )
    )
    figure_items = [
        item for item in listed_figures if item.get("source_type") == "pdf_figure"
    ]
    assert any(item.get("image_id") == image_id for item in figure_items)
    assert figure_items[0].get("page_number") is not None
    assert "url" in figure_items[0]

    payload = _call_tool_payload(
        client, "get_image", {"image_id": image_id}, call_id=5
    )
    blocks = payload["result"]["content"]
    image_blocks = [
        block
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "image"
    ]
    assert image_blocks, f"expected image content, got {blocks!r}"
