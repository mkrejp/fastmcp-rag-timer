"""HTTP auth and delivery tests for extracted images."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import AUTH_HEADERS, patched_embeddings
from tests.fixtures.pdf_factory import build_image_rich_pdf


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
    """Unknown image ids should return 404."""
    doc_id = "11111111-1111-1111-1111-111111111111"
    response = client.get(f"/images/{doc_id}/abcd1234abcd1234")
    assert response.status_code == 404
