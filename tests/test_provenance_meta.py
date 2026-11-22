"""Tests for provenance metadata, URL building, and /upload-media deprecation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.config import upload_media_deprecated
from rag.ingest import ingest_bytes
from rag.provenance_meta import (
    ProvenanceMetadataError,
    attach_image_api_fields,
    attribution_allows_public_urls,
    build_image_delivery_urls,
    inherit_provenance_for_extracted,
    merge_ingest_metadata,
    normalize_provenance_metadata,
)
from rag.storage import RagStorage

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


def test_provenance_rejects_legacy_licence() -> None:
    """Legacy ``licence`` must not be accepted on ingest metadata."""
    with pytest.raises(ProvenanceMetadataError, match="legacy provenance"):
        normalize_provenance_metadata({"licence": "public"})


def test_license_status_separate_from_doc_status() -> None:
    """``license_status`` and ``doc_status`` are independent fields."""
    meta = {
        "doc_status": "active",
        "license_status": "restricted",
        "license_id": "Unknown",
        "attribution_text": "neuvedeno",
    }
    combined, _ = merge_ingest_metadata(meta, apply_verification_defaults=False)
    assert combined["doc_status"] == "active"
    assert combined["license_status"] == "restricted"
    assert combined["license_id"] == "Unknown"


def test_inherit_provenance_sets_image_sha256() -> None:
    """Extracted images inherit parent fields but use image digest for ``sha256``."""
    parent = {
        "license_id": "CC0-1.0",
        "license_status": "published",
        "attribution_text": "Public domain",
        "sha256": "a" * 64,
        "source_url": "https://example.test/doc",
    }
    child = inherit_provenance_for_extracted(
        parent,
        image_sha256="b" * 64,
        content_type="image",
    )
    assert child["license_id"] == "CC0-1.0"
    assert child["attribution_text"] == "Public domain"
    assert child["sha256"] == "b" * 64
    assert child["sha256"] != parent["sha256"]


def test_fail_closed_without_attribution() -> None:
    """Missing ``attribution_text`` must not expose RAG or media URLs."""
    record = {"license_id": "CC0-1.0", "license_status": "published"}
    assert attribution_allows_public_urls(record) is False
    urls = build_image_delivery_urls(
        record,
        doc_id="00000000-0000-0000-0000-000000000001",
        image_id="abcd1234abcd1234",
        public_base_url="https://rag.example",
        media_public_base_url=None,
    )
    assert urls["url"] is None
    assert urls["thumb_url"] is None


def test_media_asset_id_builds_thumb_large_urls() -> None:
    """When ``media_asset_id`` is set, thumb/large point at the media service."""
    asset_id = "3d1c0e2a-7c4b-4f11-9c0e-0b1a2c3d4e5f"
    record = {
        "media_asset_id": asset_id,
        "license_id": "CC-BY-4.0",
        "license_status": "published",
        "attribution_text": "Author, CC BY 4.0",
    }
    urls = build_image_delivery_urls(
        record,
        doc_id="doc",
        image_id="img",
        public_base_url="https://rag.example",
        media_public_base_url="https://media.example",
    )
    assert urls["thumb_url"] == f"https://media.example/v1/assets/{asset_id}/thumb"
    assert urls["large_url"] == f"https://media.example/v1/assets/{asset_id}/large"
    assert urls["url"] == urls["thumb_url"]


def test_rag_fallback_url_only_with_attribution() -> None:
    """Without media id, ``PUBLIC_BASE_URL`` RAG path is used when attributed."""
    record = {
        "attribution_text": "TASL line",
        "license_status": "published",
        "license_id": "CC-BY-4.0",
    }
    urls = build_image_delivery_urls(
        record,
        doc_id="00000000-0000-0000-0000-000000000002",
        image_id="abcd1234abcd1234",
        public_base_url="https://rag.example",
        media_public_base_url=None,
    )
    assert urls["url"] == (
        "https://rag.example/images/"
        "00000000-0000-0000-0000-000000000002/abcd1234abcd1234"
    )


@pytest.fixture()
def client(rag_root, accnt_root, fake_embed) -> TestClient:  # type: ignore[no-untyped-def]
    """TestClient with temp storage."""
    from tests.conftest import patched_embeddings

    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as test_client:
            yield test_client


def test_upload_media_returns_410_by_default(client: TestClient) -> None:
    """``POST /upload-media`` is closed unless ``RAG_UPLOAD_MEDIA_DEPRECATED=0``."""
    assert upload_media_deprecated() is True
    response = client.post(
        "/upload-media",
        files={"file": ("x.bin", b"data", "application/octet-stream")},
    )
    assert response.status_code == 410
    body = response.json()
    assert body["detail"]["error"] == "gone"
    assert "geo-media" in body["detail"]["message"]


def test_ingest_accepts_provenance_metadata(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Provenance keys on ingest land in catalog extra without breaking verification."""
    metadata = {
        "license_id": "CC0-1.0",
        "license_status": "published",
        "attribution_text": "Public domain scan",
        "source_url": "https://commons.wikimedia.org/wiki/File:Test.jpg",
        "content_type": "document",
    }
    result = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name="client_domain_knowledge",
        filename="note.txt",
        data=b"Provenance ingest test body.",
        embed_fn=fake_embed,
        metadata=metadata,
    )
    doc_id = str(result["doc_id"])
    meta = rag_storage.read_meta(doc_id)
    assert meta is not None
    assert meta.extra.get("license_id") == "CC0-1.0"
    assert meta.extra.get("attribution_text") == "Public domain scan"
    assert meta.extra.get("license_status") == "published"
    assert meta.extra.get("verification_status") == "nejiste"


def test_pdf_image_inherits_parent_provenance(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Extracted ``pdf_image`` points inherit provenance and image ``sha256``."""
    from tests.fixtures.pdf_factory import build_image_rich_pdf

    metadata = {
        "license_id": "CC0-1.0",
        "license_status": "published",
        "attribution_text": "Parent document attribution",
        "content_type": "document",
    }
    with patch(
        "rag.ingest.ocr_image_file",
        return_value=("visible text", 0.95),
    ):
        result = ingest_bytes(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name="client_domain_knowledge",
            filename="figures.pdf",
            data=build_image_rich_pdf(),
            embed_fn=fake_embed,
            metadata=metadata,
        )
    doc_id = str(result["doc_id"])
    scroll_filter = Filter(
        must=[
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            FieldCondition(key="source_type", match=MatchValue(value="pdf_image")),
        ]
    )
    points, _next = qdrant_client.scroll(
        collection_name="client_domain_knowledge",
        scroll_filter=scroll_filter,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    assert points
    payload = dict(points[0].payload or {})
    doc_meta = rag_storage.read_meta(doc_id)
    assert doc_meta is not None
    assert payload.get("attribution_text") == "Parent document attribution"
    assert payload.get("license_status") == "published"
    image_sha = doc_meta.images[0].sha256
    assert payload.get("sha256") == image_sha
    assert payload.get("sha256") != doc_meta.sha256


def test_pdf_image_without_parent_attribution_is_restricted(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Images extracted without parent attribution default to restricted (fail-closed)."""
    from tests.fixtures.pdf_factory import build_image_rich_pdf

    with patch(
        "rag.ingest.ocr_image_file",
        return_value=("visible text", 0.95),
    ):
        result = ingest_bytes(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name="client_domain_knowledge",
            filename="raw.pdf",
            data=build_image_rich_pdf(),
            embed_fn=fake_embed,
        )
    doc_id = str(result["doc_id"])
    scroll_filter = Filter(
        must=[
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            FieldCondition(key="source_type", match=MatchValue(value="pdf_image")),
        ]
    )
    points, _next = qdrant_client.scroll(
        collection_name="client_domain_knowledge",
        scroll_filter=scroll_filter,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )
    assert points
    payload = dict(points[0].payload or {})
    assert payload.get("license_id") == "Unknown"
    assert payload.get("license_status") == "restricted"
    urls = build_image_delivery_urls(
        payload,
        doc_id=doc_id,
        image_id=str(payload.get("image_id")),
        public_base_url="https://rag.example",
        media_public_base_url=None,
    )
    assert urls["url"] is None