"""Tests for document ingest into storage and Qdrant."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from rag.ingest import ingest_bytes, stable_point_id
from rag.payloads import chunk_text_sha256
from rag.storage import RagStorage

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


def test_ingest_text_writes_originals_and_processed(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Text ingest should persist originals, processed artifacts, and vectors."""
    result = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name="client_domain_knowledge",
        filename="facts.txt",
        data=b"The timer service stores documents under /data/rag.",
        embed_fn=fake_embed,
    )
    assert result["status"] == "Success"
    assert result["truncated"] == []
    doc_id = str(result["doc_id"])
    assert (rag_storage.original_dir(doc_id) / "facts.txt").is_file()
    assert rag_storage.read_fulltext(doc_id) is not None
    assert rag_storage.read_chunks(doc_id) is not None
    assert rag_storage.list_documents()
    points = qdrant_client.retrieve(
        collection_name="client_domain_knowledge",
        ids=[stable_point_id(doc_id, 0)],
        with_payload=True,
        with_vectors=True,
    )
    assert len(points) == 1
    payload = points[0].payload or {}
    text = "The timer service stores documents under /data/rag."
    assert payload["chunk_sha256"] == chunk_text_sha256(text)
    assert payload["sha256"]
    assert payload["localities"] == []
    assert payload["cast_obce"] == []
    assert payload["reliability"] == 1
    assert payload["verification_status"] == "nejiste"
    assert points[0].vector is not None
    stored = rag_storage.read_meta(doc_id)
    assert stored is not None
    assert stored.extra.get("reliability") == 1


def test_ingest_skips_duplicate_sha256(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Re-uploading identical bytes should reuse the existing document."""
    payload = b"duplicate payload for hashing"
    first = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name="client_domain_knowledge",
        filename="one.txt",
        data=payload,
        embed_fn=fake_embed,
    )
    second = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name="client_domain_knowledge",
        filename="two.txt",
        data=payload,
        embed_fn=fake_embed,
    )
    assert first["reused"] is False
    assert second["reused"] is True
    assert second["doc_id"] == first["doc_id"]
    assert len(rag_storage.list_documents()) == 1


def test_get_document_text_does_not_need_embeddings(
    rag_storage: RagStorage,
) -> None:
    """Reading processed text is a pure filesystem operation."""
    doc_id = "33333333-3333-3333-3333-333333333333"
    rag_storage.save_original(doc_id, "cached.txt", b"fast access")
    rag_storage.write_processed(
        doc_id=doc_id,
        filename="cached.txt",
        mime="text/plain",
        sha256="ff",
        fulltext="fast access",
        chunks=[{"index": 0, "text": "fast access", "page": 0}],
    )
    assert rag_storage.read_fulltext(doc_id) == "fast access"


def test_ingest_pdf_bytes(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """PDF ingest should store the original and create processed metadata."""
    # Minimal valid PDF (may yield empty extractable text).
    pdf_data = (
        b"%PDF-1.0\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000052 00000 n \n0000000101 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n149\n%%EOF\n"
    )
    result = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name="client_domain_knowledge",
        filename="blank.pdf",
        data=pdf_data,
        embed_fn=fake_embed,
    )
    assert result["status"] == "Success"
    doc_id = str(result["doc_id"])
    assert (rag_storage.original_dir(doc_id) / "blank.pdf").is_file()
    meta = rag_storage.read_meta(doc_id)
    assert meta is not None
    assert meta.mime == "application/pdf"
