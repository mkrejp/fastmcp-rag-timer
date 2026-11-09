"""HTTP auth and delivery tests for extracted images and figure clips."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rag.image_delivery import (
    MISSING_IMAGE_FILE_MESSAGE,
    UNKNOWN_IMAGE_ID_MESSAGE,
)
from rag.storage import ImageMeta, RagStorage
from tests.conftest import AUTH_HEADERS, patched_embeddings
from tests.fixtures.pdf_factory import build_image_rich_pdf
from tests.fixtures.pdf_figure_factory import build_figure_caption_pdf

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


def _ingest_fixture_pdf(client: TestClient) -> tuple[str, str]:
    """Upload the image-rich PDF and return ``doc_id`` and one ``image_id``."""
    return _ingest_fixture_pdf_with_headers(client, AUTH_HEADERS)


def _ingest_fixture_pdf_with_headers(
    client: TestClient,
    headers: dict[str, str],
) -> tuple[str, str]:
    """Upload the fixture PDF using explicit request headers."""
    with patch(
        "rag.ingest.ocr_image_file",
        return_value=("placeholder ocr text long enough", 90.0),
    ):
        upload = client.post(
            "/upload",
            files={
                "file": (
                    "fixture.pdf",
                    build_image_rich_pdf(),
                    "application/pdf",
                )
            },
            headers=headers,
        )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]
    meta = client.get(f"/documents/{doc_id}").json()
    image_id = meta["images"][0]["image_id"]
    return doc_id, image_id


def test_image_http_requires_auth(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Image downloads should reject missing or invalid bearer tokens."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as http:
            doc_id, image_id = _ingest_fixture_pdf_with_headers(http, AUTH_HEADERS)
            missing = http.get(f"/images/{doc_id}/{image_id}")
            assert missing.status_code == 401
            bad = http.get(
                f"/images/{doc_id}/{image_id}",
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert bad.status_code == 401


def test_image_http_success_with_auth(client: TestClient) -> None:
    """Authenticated clients should receive the image with a matching Content-Type."""
    doc_id, image_id = _ingest_fixture_pdf(client)
    response = client.get(f"/images/{doc_id}/{image_id}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    assert len(response.content) > 100


def test_image_http_404_unknown(client: TestClient) -> None:
    """Unknown image ids should return 404 with an unknown-id message."""
    doc_id = "11111111-1111-1111-1111-111111111111"
    response = client.get(f"/images/{doc_id}/abcd1234abcd1234")
    assert response.status_code == 404
    assert response.json()["detail"] == UNKNOWN_IMAGE_ID_MESSAGE


def _png_bytes() -> bytes:
    """Return a tiny PNG used as a figure clip stand-in."""
    image = Image.new("RGB", (16, 16), "purple")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_image_http_serves_figure_png_absent_from_meta(
    client: TestClient,
    rag_root: Path,
) -> None:
    """HTTP should serve figure PNGs that exist on disk but not in meta.images."""
    storage = RagStorage(rag_root)
    png = _png_bytes()
    storage.save_original(FIGURE_DOC_ID, "wiener-secession-1898.pdf", b"%PDF-1.4")
    storage.write_processed(
        doc_id=FIGURE_DOC_ID,
        filename="wiener-secession-1898.pdf",
        mime="application/pdf",
        sha256="abc",
        fulltext="secession",
        chunks=[{"index": 0, "text": "secession", "page": 0}],
        images=[],
    )
    storage.save_image_bytes(
        FIGURE_DOC_ID, image_id=FIGURE_IMAGE_ID, extension=".png", data=png
    )
    assert storage.find_image(FIGURE_DOC_ID, FIGURE_IMAGE_ID) is None

    response = client.get(f"/images/{FIGURE_DOC_ID}/{FIGURE_IMAGE_ID}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content == png


def test_image_http_404_missing_on_disk(client: TestClient, rag_root: Path) -> None:
    """Catalogued images whose files are gone should report a missing file."""
    storage = RagStorage(rag_root)
    storage.save_original(FIGURE_DOC_ID, "book.pdf", b"%PDF-1.4")
    storage.write_processed(
        doc_id=FIGURE_DOC_ID,
        filename="book.pdf",
        mime="application/pdf",
        sha256="abc",
        fulltext="book",
        chunks=[{"index": 0, "text": "book", "page": 0}],
        images=[
            ImageMeta(
                image_id=FIGURE_IMAGE_ID,
                sha256="deadbeef",
                file_path=f"processed/{FIGURE_DOC_ID}/images/{FIGURE_IMAGE_ID}.png",
                page_number=0,
                width=1,
                height=1,
                mime="image/png",
            )
        ],
    )
    response = client.get(f"/images/{FIGURE_DOC_ID}/{FIGURE_IMAGE_ID}")
    assert response.status_code == 404
    assert response.json()["detail"] == MISSING_IMAGE_FILE_MESSAGE


def test_image_http_invalid_id_stays_not_found(client: TestClient) -> None:
    """Invalid ids must still 404 without treating them as a disk lookup."""
    response = client.get(f"/images/{FIGURE_DOC_ID}/not-a-valid-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"


def test_image_http_serves_ingested_pdf_figure(
    client: TestClient,
    rag_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Ingested pdf_figure clips should be fetchable even when skipped in meta.images."""

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
    meta = client.get(f"/documents/{doc_id}").json()
    meta_ids = {item["image_id"] for item in meta.get("images") or []}
    images_dir = rag_root / "processed" / doc_id / "images"
    on_disk = [path.stem for path in images_dir.glob("*") if path.is_file()]
    figure_ids = [image_id for image_id in on_disk if image_id not in meta_ids]
    assert figure_ids, f"expected figure PNG on disk; meta={meta_ids} disk={on_disk}"

    figure_id = figure_ids[0]
    response = client.get(f"/images/{doc_id}/{figure_id}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    assert len(response.content) > 100
