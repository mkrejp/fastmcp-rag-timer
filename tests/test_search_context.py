"""Unit tests for reader-local search context helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from qdrant_client import QdrantClient

from rag.document_replace import save_version_snapshot
from rag.search_context import (
    build_document_context,
    catalog_entry_for_doc,
    neighbor_chunks_for_hit,
    search_catalog,
)
from rag.storage import DocumentMeta, RagStorage
from rag.vectorstore import ensure_locality_links_collection, upsert_locality_links


@pytest.fixture()
def storage(rag_root: Path) -> RagStorage:
    """Initialized storage under the temp RAG root."""
    store = RagStorage(rag_root)
    store.ensure_layout()
    return store


@pytest.fixture()
def qdrant(rag_storage: RagStorage) -> Iterator[QdrantClient]:
    """Local Qdrant client with locality_links ready."""
    client = QdrantClient(path=str(rag_storage.qdrant_dir))
    ensure_locality_links_collection(client)
    try:
        yield client
    finally:
        client.close()


def _write_doc(
    storage: RagStorage, doc_id: str, *, filename: str = "note.txt"
) -> DocumentMeta:
    """Persist a small multi-chunk document for context tests."""
    chunks = [
        {"index": 0, "text": "alpha chunk about Prague", "page": 0},
        {"index": 1, "text": "beta chunk about timers", "page": 0},
        {"index": 2, "text": "gamma chunk about RAG", "page": 1},
    ]
    return storage.write_processed(
        doc_id=doc_id,
        filename=filename,
        mime="text/plain",
        sha256="abc123",
        fulltext="alpha chunk about Prague\nbeta chunk about timers\ngamma chunk about RAG",
        chunks=chunks,
    )


def test_catalog_and_neighbor_helpers(storage: RagStorage) -> None:
    """Catalog summary and neighbor chunks come from disk without embeddings."""
    doc_id = "11111111-1111-1111-1111-111111111111"
    _write_doc(storage, doc_id, filename="prague-timer.txt")

    catalog = catalog_entry_for_doc(storage, doc_id)
    assert catalog is not None
    assert catalog["filename"] == "prague-timer.txt"
    assert catalog["chunk_count"] == 3

    neighbors = neighbor_chunks_for_hit(storage, doc_id, 1, radius=1)
    assert [item["chunk_index"] for item in neighbors] == [0, 2]
    assert "Prague" in neighbors[0]["text"]
    assert "RAG" in neighbors[1]["text"]


def test_search_catalog_substring(storage: RagStorage) -> None:
    """Catalog search matches filename substrings and respects limit."""
    _write_doc(
        storage,
        "22222222-2222-2222-2222-222222222222",
        filename="prague-guide.txt",
    )
    _write_doc(
        storage,
        "33333333-3333-3333-3333-333333333333",
        filename="brno-notes.txt",
    )
    hits = search_catalog(storage, query="prague", limit=10)
    assert len(hits) == 1
    assert hits[0]["filename"] == "prague-guide.txt"


def test_build_document_context_includes_versions_and_locality(
    storage: RagStorage,
    qdrant: QdrantClient,
) -> None:
    """Document context bundles versions and locality links from reader data."""
    doc_id = "44444444-4444-4444-4444-444444444444"
    meta = _write_doc(storage, doc_id, filename="linked.txt")
    save_version_snapshot(
        storage,
        meta=meta,
        fulltext="archived body",
        author="tester",
        reason="unit-test",
        created_at="2024-01-01T00:00:00+00:00",
        generation=1,
    )
    upsert_locality_links(
        qdrant,
        [
            {
                "obec_id": "554782",
                "name": "Praha",
                "doc_id": doc_id,
                "chunk_index": 0,
                "matched_text": "Prague",
                "char_start": 0,
                "char_end": 6,
                "source_type": "text_chunk",
                "method": "exact",
            }
        ],
    )

    context = build_document_context(
        storage,
        qdrant,
        doc_id,
        chunk_index=1,
        neighbor_radius=1,
    )
    assert context is not None
    assert context["catalog"]["filename"] == "linked.txt"
    assert context["versions"]["count"] == 1
    assert context["locality_links"]["returned"] == 1
    assert context["locality_links"]["links"][0]["obec_id"] == "554782"
    assert [item["chunk_index"] for item in context["neighbor_chunks"]] == [0, 2]
