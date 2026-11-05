"""Remove a document from disk and from the Qdrant index."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from rag.logging_setup import get_logger
from rag.storage import RagStorage

logger = get_logger("rag.document_delete")


class DocumentNotFoundError(LookupError):
    """Raised when ``doc_id`` is not present in persistent storage."""

    def __init__(self, doc_id: str) -> None:
        """Record the missing document id on the exception."""
        self.doc_id = doc_id
        super().__init__(f"Document not found: {doc_id}")


def _doc_id_filter(doc_id: str) -> Filter:
    """Return a Qdrant payload filter matching ``doc_id``."""
    return Filter(
        must=[
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=doc_id),
            )
        ]
    )


def delete_qdrant_points_for_doc(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
) -> int:
    """Delete all vector points whose payload ``doc_id`` matches.

    Returns the number of points removed (text chunks and image points).
    """
    if not qdrant.collection_exists(collection_name):
        return 0
    selector = _doc_id_filter(doc_id)
    removed = int(
        qdrant.count(
            collection_name=collection_name,
            count_filter=selector,
            exact=True,
        ).count
    )
    qdrant.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(filter=selector),
        wait=True,
    )
    return removed


def delete_document(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    doc_id: str,
) -> dict[str, Any]:
    """Delete one document's Qdrant points and on-disk artifacts.

    Vectors are removed first so a retry can still find catalog metadata if
    filesystem cleanup fails. Missing or invalid ids raise
    ``DocumentNotFoundError``.
    """
    try:
        storage.validate_doc_id(doc_id)
    except ValueError as exc:
        raise DocumentNotFoundError(doc_id) from exc

    meta = storage.read_meta(doc_id)
    if meta is None:
        raise DocumentNotFoundError(doc_id)

    chunks_removed = delete_qdrant_points_for_doc(
        qdrant, collection_name=collection_name, doc_id=doc_id
    )
    storage.remove_document_files(doc_id)
    logger.info(
        "Deleted document doc_id=%s filename=%s chunks_removed=%s",
        doc_id,
        meta.filename,
        chunks_removed,
    )
    return {
        "status": "deleted",
        "doc_id": doc_id,
        "filename": meta.filename,
        "chunks_removed": chunks_removed,
    }
