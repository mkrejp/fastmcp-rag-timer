"""Tests for per-image binary restriction without deletes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.image_visibility import (
    IMAGE_BINARY_DENIED_DETAIL,
    audit_log_path,
    image_binary_serving_allowed,
)
from rag.storage import RagStorage
from tests.conftest import AUTH_HEADERS, patched_embeddings
from tests.fixtures.pdf_factory import build_image_rich_pdf
from tests.fixtures.pdf_figure_factory import build_figure_caption_pdf
from tests.test_image_http import _ingest_fixture_pdf


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


def _patch_restrict(
    client: TestClient,
    doc_id: str,
    image_id: str,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    """Restrict one image with provenance metadata on the document."""
    body = {
        "restricted": True,
        "license_status": "restricted",
        "reason": "© unclear",
        "decided_by": "test-operator",
    }
    url = f"/images/{doc_id}/{image_id}/visibility"
    if dry_run:
        url = f"{url}?dry_run=true"
    response = client.patch(url, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _publish_document_for_hosting(client: TestClient, doc_id: str) -> None:
    """Mark document/images hostable (matches PR #59 fail-closed defaults)."""
    response = client.patch(
        f"/documents/{doc_id}/metadata",
        json={
            "source_url": "https://kramerius.example/uuid",
            "source_name": "Kramerius NKP",
            "attribution_text": "Test attribution",
            "license_id": "CC0-1.0",
            "license_status": "published",
        },
    )
    assert response.status_code == 200


def test_restricted_image_returns_403_and_restores_200(client: TestClient) -> None:
    """Restricted binaries return 403; un-restrict restores 200."""
    doc_id, image_id = _ingest_fixture_pdf(client)
    _publish_document_for_hosting(client, doc_id)
    assert client.get(f"/images/{doc_id}/{image_id}").status_code == 200

    _patch_restrict(client, doc_id, image_id)
    denied = client.get(f"/images/{doc_id}/{image_id}")
    assert denied.status_code == 403

    restore = client.patch(
        f"/images/{doc_id}/{image_id}/visibility",
        json={
            "restricted": False,
            "license_status": "published",
            "reason": "cleared",
            "decided_by": "test-operator",
        },
    )
    assert restore.status_code == 200
    assert client.get(f"/images/{doc_id}/{image_id}").status_code == 200


def test_metadata_keeps_source_and_null_urls(
    client: TestClient,
    rag_root: Path,
) -> None:
    """MCP/list JSON keeps provenance but clears delivery URLs when restricted."""
    doc_id, image_id = _ingest_fixture_pdf(client)
    _publish_document_for_hosting(client, doc_id)
    _patch_restrict(client, doc_id, image_id)

    storage = RagStorage(str(rag_root))
    meta = storage.read_meta(doc_id)
    assert meta is not None
    row = next(img for img in meta.images if img.image_id == image_id)
    image_dict = row.to_dict()
    assert image_dict.get("license_status") == "restricted"

    from rag.mcp_tools import _image_meta_with_url

    item = _image_meta_with_url(doc_id, row, fallback=meta.to_dict())
    assert item.get("source_url") == "https://kramerius.example/uuid"
    assert item.get("source_name") == "Kramerius NKP"
    assert item.get("url") is None
    assert item.get("thumb_url") is None
    assert item.get("large_url") is None


def test_visibility_patch_requires_auth(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Visibility writes reject missing bearer tokens."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as http:
            doc_id, image_id = _ingest_fixture_pdf(http)
            response = http.patch(
                f"/images/{doc_id}/{image_id}/visibility",
                json={"restricted": True, "reason": "x", "decided_by": "y"},
            )
            assert response.status_code == 401


def test_restrict_does_not_delete_files(
    client: TestClient,
    rag_root: Path,
) -> None:
    """On-disk image bytes remain after restriction."""
    doc_id, image_id = _ingest_fixture_pdf(client)
    storage = RagStorage(str(rag_root))
    path_before = storage.locate_image_file(doc_id, image_id)
    assert path_before is not None and path_before.is_file()
    _patch_restrict(client, doc_id, image_id)
    path_after = storage.locate_image_file(doc_id, image_id)
    assert path_after is not None and path_after.is_file()
    assert path_after.read_bytes() == path_before.read_bytes()


def test_dry_run_writes_no_audit(
    client: TestClient,
    rag_root: Path,
) -> None:
    """``dry_run=true`` skips catalog writes and audit append."""
    doc_id, image_id = _ingest_fixture_pdf(client)
    storage = RagStorage(str(rag_root))
    audit = audit_log_path(storage)
    _patch_restrict(client, doc_id, image_id, dry_run=True)
    meta = storage.read_meta(doc_id)
    assert meta is not None
    row = next(img for img in meta.images if img.image_id == image_id)
    assert row.to_dict().get("license_status") is None
    assert not audit.exists()


def test_batch_visibility_patch(client: TestClient, rag_root: Path) -> None:
    """Batch endpoint updates multiple images and Qdrant payloads."""
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
        )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]
    meta = client.get(f"/documents/{doc_id}").json()
    image_ids = [img["image_id"] for img in meta["images"][:2]]
    assert len(image_ids) >= 1

    response = client.patch(
        f"/images/{doc_id}/visibility",
        json={
            "image_ids": image_ids,
            "restricted": True,
            "license_status": "restricted",
            "reason": "batch",
            "decided_by": "test-operator",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == len(image_ids)

    filt = Filter(
        must=[
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            FieldCondition(key="image_id", match=MatchValue(value=image_ids[0])),
        ]
    )
    qdrant = client.app.state.qdrant
    points, _ = qdrant.scroll(
        collection_name="client_domain_knowledge",
        scroll_filter=filt,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )
    assert points
    payload = points[0].payload or {}
    assert payload.get("license_status") == "restricted"

    audit = audit_log_path(RagStorage(str(rag_root)))
    assert audit.is_file()
    lines = audit.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    last = json.loads(lines[-1])
    assert last.get("action") == "restrict"
    assert "previous_license_status" in last


def test_http_get_denies_copyright_even_when_published(client: TestClient) -> None:
    """REST binary gate rejects ``Copyright`` per media-licence §6 (no published bypass)."""
    doc_id, image_id = _ingest_fixture_pdf(client)
    response = client.patch(
        f"/documents/{doc_id}/metadata",
        json={
            "attribution_text": "fixture test",
            "license_id": "Copyright",
            "license_status": "published",
        },
    )
    assert response.status_code == 200
    denied = client.get(f"/images/{doc_id}/{image_id}")
    assert denied.status_code == 403
    assert IMAGE_BINARY_DENIED_DETAIL in denied.json().get("detail", "")


def test_image_binary_serving_fail_closed_on_empty_status() -> None:
    """Missing ``license_status`` must not allow binary delivery."""
    assert image_binary_serving_allowed({}) is False
    assert image_binary_serving_allowed({"license_id": "CC0-1.0"}) is False


def test_image_binary_serving_denies_copyright_and_unknown_id() -> None:
    """§6: ``Copyright`` / ``Unknown`` ``license_id`` blocks bytes even when published."""
    assert (
        image_binary_serving_allowed(
            {"license_id": "Copyright", "license_status": "published"}
        )
        is False
    )
    assert (
        image_binary_serving_allowed(
            {"license_id": "Unknown", "license_status": "verified"}
        )
        is False
    )


def test_image_binary_serving_allows_intake_after_ingest_default() -> None:
    """Figures from the #59 inherit path (``intake`` only) may serve authenticated bytes."""
    assert image_binary_serving_allowed({"license_status": "intake"}) is True


def test_visibility_audit_records_previous_on_unrestrict(client: TestClient) -> None:
    """Audit lines carry prior status and operator fields for unrestrict."""
    doc_id, image_id = _ingest_fixture_pdf(client)
    _publish_document_for_hosting(client, doc_id)
    _patch_restrict(client, doc_id, image_id)
    restore = client.patch(
        f"/images/{doc_id}/{image_id}/visibility",
        json={
            "restricted": False,
            "license_status": "published",
            "reason": "cleared",
            "decided_by": "test-operator",
        },
    )
    assert restore.status_code == 200
    audit = audit_log_path(client.app.state.storage)
    lines = audit.read_text(encoding="utf-8").strip().splitlines()
    last = json.loads(lines[-1])
    assert last.get("action") == "unrestrict"
    assert last.get("previous_license_status") == "restricted"
    assert last.get("decided_by") == "test-operator"
    assert last.get("reason") == "cleared"


def test_ingested_figure_serves_bytes_before_visibility_patch(
    client: TestClient,
    rag_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Fresh pdf_figure clips with ``intake`` provenance return bytes (no PATCH required)."""

    def fake_many(texts: list[str], **_kwargs: object) -> list[list[float]]:
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
    response = client.get(f"/images/{doc_id}/{figure_ids[0]}")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("image/")
