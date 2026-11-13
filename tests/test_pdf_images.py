"""Tests for PDF image extraction, storage, payloads, and delete cleanup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
from unittest.mock import patch

import pytest

from rag.document_delete import delete_document
from rag.ingest import ingest_bytes, stable_image_point_id
from rag.ocr import tesseract_is_available
from rag.payloads import build_chunk_payload, build_image_payload
from rag.pdf_images import extract_pdf_images
from rag.storage import DocumentMeta, RagStorage
from tests.fixtures.pdf_factory import build_image_rich_pdf

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


def test_extract_pdf_images_filters_tiny_and_dedupes() -> None:
    """Fixture PDF should yield two large unique images (phrase + photo), not the icon."""
    pdf_bytes = build_image_rich_pdf()
    extracted = extract_pdf_images(pdf_bytes)
    assert len(extracted) == 2
    sha_set = {item.sha256 for item in extracted}
    assert len(sha_set) == 2
    for item in extracted:
        assert item.width >= 64 and item.height >= 64
        assert len(item.data) >= 4096
        assert item.image_id == item.sha256[:16]


def test_image_meta_backward_compatible_deserialize() -> None:
    """DocumentMeta without ``images`` should deserialize with an empty list."""
    legacy = {
        "doc_id": "11111111-1111-1111-1111-111111111111",
        "filename": "old.pdf",
        "mime": "application/pdf",
        "sha256": "abc",
        "uploaded_at": "2020-01-01T00:00:00+00:00",
        "chunk_count": 1,
        "status": "ready",
        "original_path": "/tmp/x",
        "fulltext_path": "/tmp/y",
        "chunks_path": "/tmp/z",
    }
    meta = DocumentMeta.from_dict(legacy)
    assert meta.images == []


def test_payload_builders_include_source_type() -> None:
    """Chunk and image payloads should expose the new RAG fields."""
    chunk = build_chunk_payload(
        doc_id="d",
        parent_document="f.pdf",
        page_number=2,
        doc_sha256="dead",
        text="hello",
        chunk_index=0,
    )
    assert chunk["source_type"] == "pdf_chunk"
    assert chunk["content_summary"] == "hello"
    assert chunk["chunk_sha256"]
    assert chunk["localities"] == []
    assert chunk["cast_obce"] == []
    image = build_image_payload(
        doc_id="d",
        parent_document="f.pdf",
        page_number=1,
        file_path="processed/d/images/ab.png",
        image_id="abcd1234abcd1234",
        image_sha256="abcd",
        content_summary="OCR text",
    )
    assert image["source_type"] == "pdf_image"
    assert "text" not in image or image["text"] == "OCR text"
    assert image["chunk_sha256"]
    assert image["localities"] == []
    assert image["cast_obce"] == []


def test_path_traversal_rejected(rag_storage: RagStorage) -> None:
    """Unsafe doc/image ids must not resolve to paths outside the RAG root."""
    with pytest.raises(ValueError):
        rag_storage.validate_doc_id("../etc/passwd")
    with pytest.raises(ValueError):
        rag_storage.validate_image_id("../../bad")
    with pytest.raises(ValueError):
        rag_storage.resolve_under_root("../../etc/passwd")


def test_ingest_pdf_stores_images_with_mock_ocr(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PDF ingest should persist images and index OCR summaries when OCR returns text.

    Figure indexing is disabled here: the photo placement is above the
    ``pdf_figure`` area threshold and would otherwise be skipped from
    ``meta.images`` (one image instead of two).
    """
    monkeypatch.setenv("FIGURE_INDEX_ENABLED", "false")
    pdf_bytes = build_image_rich_pdf()
    with patch(
        "rag.ingest.ocr_image_file",
        side_effect=lambda path: (
            ("UNIQUEOCRPHRASE12345 long enough for indexing", 88.0)
            if "images" in str(path)
            else ("", None)
        ),
    ):
        result = ingest_bytes(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name="client_domain_knowledge",
            filename="fixture.pdf",
            data=pdf_bytes,
            embed_fn=fake_embed,
        )
    assert result["status"] == "Success"
    doc_id = str(result["doc_id"])
    meta = rag_storage.read_meta(doc_id)
    assert meta is not None
    assert len(meta.images) == 2
    images_dir = rag_storage.images_dir(doc_id)
    assert images_dir.is_dir()
    indexed = [img for img in meta.images if img.ocr_text and len(img.ocr_text) >= 20]
    assert result["images_indexed"] == len(indexed)


def test_delete_document_removes_images_and_points(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Deleting a document should remove files and all Qdrant points for that doc."""
    pdf_bytes = build_image_rich_pdf()
    with patch(
        "rag.ingest.ocr_image_file",
        return_value=("indexed ocr phrase long enough", 90.0),
    ):
        result = ingest_bytes(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name="client_domain_knowledge",
            filename="fixture.pdf",
            data=pdf_bytes,
            embed_fn=fake_embed,
        )
    doc_id = str(result["doc_id"])
    info_before = qdrant_client.get_collection("client_domain_knowledge")
    assert int(info_before.points_count or 0) > 0

    delete_document(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name="client_domain_knowledge",
        doc_id=doc_id,
    )
    assert rag_storage.read_meta(doc_id) is None
    assert not rag_storage.processed_doc_dir(doc_id).exists()
    info_after = qdrant_client.get_collection("client_domain_knowledge")
    assert int(info_after.points_count or 0) == 0


@pytest.mark.integration
@pytest.mark.skipif(
    not tesseract_is_available(),
    reason="Tesseract binary not installed",
)
def test_integration_image_only_phrase_is_searchable(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Phrase present only in an embedded image should appear as a pdf_image hit."""
    phrase = "ZEBRAINTEGRATIONONLY98765"
    pdf_bytes = build_image_rich_pdf(ocr_phrase=phrase)
    ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name="client_domain_knowledge",
        filename="integration.pdf",
        data=pdf_bytes,
        embed_fn=fake_embed,
    )
    from rag.vectorstore import search_similar

    hits = search_similar(
        qdrant_client,
        query_vector=fake_embed(phrase),
        limit=5,
    )
    image_hits = [
        h for h in hits if (h.payload or {}).get("source_type") == "pdf_image"
    ]
    assert image_hits
    assert phrase in str((image_hits[0].payload or {}).get("content_summary", ""))


def test_stable_image_point_id_is_deterministic() -> None:
    """Image point ids should be stable for re-ingest idempotency expectations."""
    first = stable_image_point_id(
        "11111111-1111-1111-1111-111111111111", "abcd1234abcd1234"
    )
    second = stable_image_point_id(
        "11111111-1111-1111-1111-111111111111", "abcd1234abcd1234"
    )
    assert first == second
