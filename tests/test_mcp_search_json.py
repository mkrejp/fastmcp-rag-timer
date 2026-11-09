"""MCP search JSON format and page_image_id headers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from tests.conftest import AUTH_HEADERS, patched_embeddings
from tests.fixtures.pdf_page_factory import build_text_layer_pdf
from tests.test_mcp_search import MCP_HEADERS, _call_tool, _initialize


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


def test_search_text_output_includes_page_image_id(client: TestClient) -> None:
    """Default text format adds page_image_id without removing legacy fields."""
    upload = client.post(
        "/upload",
        files={
            "file": (
                "domain.pdf",
                build_text_layer_pdf(phrase="Timer domain knowledge chunk."),
                "application/pdf",
            )
        },
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]

    _initialize(client)
    text = _call_tool(
        client,
        "search_domain_knowledge",
        {"query": "Timer domain", "limit": 2},
        call_id=2,
    )
    assert "page_image_id=page:" in text
    assert f"doc_id={doc_id}" in text
    assert "Content:" in text

    plain = client.post(
        "/upload",
        files={
            "file": (
                "plain.txt",
                b"UniquePlainTextOnlyPhrase for non-PDF hits.",
                "text/plain",
            )
        },
    )
    assert plain.status_code == 200
    json_plain = _call_tool(
        client,
        "search_domain_knowledge",
        {
            "query": "UniquePlainTextOnlyPhrase",
            "limit": 3,
            "format": "json",
        },
        call_id=3,
    )
    for hit in json.loads(json_plain):
        if "UniquePlainTextOnlyPhrase" in str(hit.get("content", "")):
            assert hit.get("page_image_id") is None


def test_search_domain_knowledge_json_format(client: TestClient) -> None:
    """format=json returns structured hits with page_image_id and score."""
    upload = client.post(
        "/upload",
        files={
            "file": (
                "json.pdf",
                build_text_layer_pdf(phrase="Structured JSON search phrase."),
                "application/pdf",
            )
        },
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]

    _initialize(client)
    raw = _call_tool(
        client,
        "search_domain_knowledge",
        {"query": "Structured JSON", "limit": 1, "format": "json"},
        call_id=2,
    )
    hits: list[dict[str, Any]] = json.loads(raw)
    assert len(hits) >= 1
    hit = hits[0]
    assert hit["doc_id"] == doc_id
    assert hit["page_image_id"] == f"page:{doc_id}:1"
    assert hit["content"]
    assert hit["score"] is not None
