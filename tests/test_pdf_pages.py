"""Tests for lazy page renders and highlight normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from rag.page_ids import format_page_image_id
from rag.pdf_pages import ensure_page_render, get_page_highlights, strip_diacritics
from rag.storage import RagStorage
from tests.conftest import AUTH_HEADERS, patched_embeddings
from tests.fixtures.pdf_page_factory import build_text_layer_pdf


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


def _ingest_pdf(client: TestClient) -> str:
    response = client.post(
        "/upload",
        files={
            "file": (
                "pages.pdf",
                build_text_layer_pdf(),
                "application/pdf",
            )
        },
    )
    assert response.status_code == 200
    return str(response.json()["doc_id"])


def test_page_render_caches_jpeg(client: TestClient, rag_root: Path) -> None:
    """Rendering writes ``processed/{doc}/pages/{n}.jpg`` once."""
    doc_id = _ingest_pdf(client)
    storage = RagStorage(rag_root)
    path1 = ensure_page_render(storage, doc_id, 1)
    assert path1.is_file()
    mtime = path1.stat().st_mtime
    path2 = ensure_page_render(storage, doc_id, 1)
    assert path2 == path1
    assert path2.stat().st_mtime == mtime


def test_highlights_normalized_and_diacritics(client: TestClient, rag_root: Path) -> None:
    """Text-layer search returns normalized boxes; diacritic-insensitive fallback works."""
    doc_id = _ingest_pdf(client)
    storage = RagStorage(rag_root)
    folded = strip_diacritics("Pražský")
    assert "Prazsky" in folded or folded.startswith("Prazsk")
    result = get_page_highlights(storage, doc_id, 1, "Prazsky", max_hits=5)
    assert result["page_width"] > 0
    assert result["page_height"] > 0
    assert result["highlights"]
    box = result["highlights"][0]
    for key in ("x0", "y0", "x1", "y1"):
        assert 0.0 <= float(box[key]) <= 1.0
        assert box["x1"] >= box["x0"]
        assert box["y1"] >= box["y0"]


def test_page_image_http_route(client: TestClient) -> None:
    """HTTP image route accepts page composite ids."""
    doc_id = _ingest_pdf(client)
    image_id = format_page_image_id(doc_id, 1)
    response = client.get(f"/images/{doc_id}/{image_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 500
