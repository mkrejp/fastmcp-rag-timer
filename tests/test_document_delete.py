"""Tests for removing a document from disk and Qdrant."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.document_delete import DocumentNotFoundError, delete_document
from rag.ingest import ingest_bytes
from rag.storage import RagStorage

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

COLLECTION = "client_domain_knowledge"


def _count_points(qdrant_client: QdrantClient, doc_id: str) -> int:
    """Return the number of Qdrant points stored for ``doc_id``."""
    result = qdrant_client.count(
        collection_name=COLLECTION,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        ),
        exact=True,
    )
    return int(result.count)


def test_delete_document_removes_files_and_vectors(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Deleting a document should drop files, catalog row, and Qdrant points."""
    first = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="keep.txt",
        data=b"Keep this neighboring document.",
        embed_fn=fake_embed,
    )
    second = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="drop.txt",
        data=b"Remove this obsolete uploaded file.",
        embed_fn=fake_embed,
    )
    drop_id = str(second["doc_id"])
    keep_id = str(first["doc_id"])
    assert _count_points(qdrant_client, drop_id) >= 1

    result = delete_document(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=drop_id,
    )
    assert result["status"] == "deleted"
    assert result["doc_id"] == drop_id
    assert result["filename"] == "drop.txt"
    assert result["chunks_removed"] >= 1

    assert rag_storage.read_meta(drop_id) is None
    assert not rag_storage.original_dir(drop_id).exists()
    assert not rag_storage.processed_doc_dir(drop_id).exists()
    assert all(entry["doc_id"] != drop_id for entry in rag_storage.list_documents())
    assert _count_points(qdrant_client, drop_id) == 0

    assert rag_storage.read_meta(keep_id) is not None
    assert _count_points(qdrant_client, keep_id) >= 1


def test_delete_document_missing_id_raises(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
) -> None:
    """A missing catalog id should raise ``DocumentNotFoundError``."""
    with pytest.raises(DocumentNotFoundError, match="missing-doc"):
        delete_document(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            doc_id="missing-doc",
        )
