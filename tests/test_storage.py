"""Tests for persistent RAG filesystem storage."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from storage import RagStorage

if TYPE_CHECKING:
    pass


def test_ensure_layout_creates_tree(rag_root: Path) -> None:
    """Creating storage should materialize originals, processed, qdrant, index."""
    storage = RagStorage(rag_root)
    storage.ensure_layout()

    assert (rag_root / "originals").is_dir()
    assert (rag_root / "processed").is_dir()
    assert (rag_root / "qdrant").is_dir()
    assert (rag_root / "index.json").is_file()
    assert storage.list_documents() == []


def test_write_and_read_processed_artifacts(rag_storage: RagStorage) -> None:
    """Processed fulltext, chunks, and meta should round-trip on disk."""
    doc_id = "11111111-1111-1111-1111-111111111111"
    rag_storage.save_original(doc_id, "note.txt", b"hello world")
    meta = rag_storage.write_processed(
        doc_id=doc_id,
        filename="note.txt",
        mime="text/plain",
        sha256="abc",
        fulltext="hello world",
        chunks=[{"index": 0, "text": "hello world", "page": 0}],
    )

    assert meta.chunk_count == 1
    assert rag_storage.read_fulltext(doc_id) == "hello world"
    chunks = rag_storage.read_chunks(doc_id)
    assert chunks is not None
    assert chunks[0]["text"] == "hello world"
    loaded = rag_storage.read_meta(doc_id)
    assert loaded is not None
    assert loaded.filename == "note.txt"
    catalog = rag_storage.list_documents()
    assert len(catalog) == 1
    assert catalog[0]["doc_id"] == doc_id


def test_find_by_sha256(rag_storage: RagStorage) -> None:
    """Catalog lookup by content hash should return the matching document."""
    doc_id = "22222222-2222-2222-2222-222222222222"
    rag_storage.save_original(doc_id, "a.txt", b"same")
    rag_storage.write_processed(
        doc_id=doc_id,
        filename="a.txt",
        mime="text/plain",
        sha256="deadbeef",
        fulltext="same",
        chunks=[{"index": 0, "text": "same", "page": 0}],
    )
    found = rag_storage.find_by_sha256("deadbeef")
    assert found is not None
    assert found.doc_id == doc_id
    assert rag_storage.find_by_sha256("missing") is None
