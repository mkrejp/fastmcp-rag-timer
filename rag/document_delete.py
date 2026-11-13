"""Remove a document from disk and from the Qdrant index."""

from __future__ import annotations

from typing import Any, Mapping

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from rag.doc_locks import doc_lock
from rag.logging_setup import get_logger
from rag.storage import RagStorage
from rag.verification_meta import DocumentMetadataError, supersedes_ids

logger = get_logger("rag.document_delete")


class DocumentNotFoundError(LookupError):
    """Raised when ``doc_id`` is not present in persistent storage."""

    def __init__(self, doc_id: str) -> None:
        """Record the missing document id on the exception."""
        self.doc_id = doc_id
        super().__init__(f"Document not found: {doc_id}")


class DocumentLinkedError(ValueError):
    """Raised when DELETE would leave dangling supersession links."""

    def __init__(self, doc_id: str, linked_ids: list[str]) -> None:
        """Record the document and the peers that still reference it."""
        self.doc_id = doc_id
        self.linked_ids = list(linked_ids)
        listed = ", ".join(self.linked_ids)
        super().__init__(
            f"document {doc_id} is linked to {listed}; "
            "unlink first via POST /documents/{doc_id}/unsupersede"
        )


def _peer_ids_from_extra(extra: Mapping[str, Any]) -> list[str]:
    """Return ``superseded_by`` and ``supersedes`` IDs from catalog extra."""
    peers: list[str] = []
    superseded_by = str(extra.get("superseded_by") or "").strip()
    if superseded_by:
        peers.append(superseded_by)
    try:
        listed = supersedes_ids(extra.get("supersedes"))
    except DocumentMetadataError:
        listed = []
    for item in listed:
        if item not in peers:
            peers.append(item)
    return peers


def collect_supersession_peers(storage: RagStorage, doc_id: str) -> list[str]:
    """Return other catalog ids linked to ``doc_id`` via supersession fields.

    Includes this document's own ``superseded_by`` / ``supersedes`` peers and
    any other document that points at ``doc_id``.
    """
    linked: list[str] = []
    seen: set[str] = set()

    def _add(other: str) -> None:
        text = other.strip()
        if text and text != doc_id and text not in seen:
            seen.add(text)
            linked.append(text)

    target = storage.read_meta(doc_id)
    if target is not None:
        for peer in _peer_ids_from_extra(target.extra):
            _add(peer)
    for entry in storage.list_documents():
        other_id = str(entry.get("doc_id") or "").strip()
        if not other_id or other_id == doc_id:
            continue
        meta = storage.read_meta(other_id)
        if meta is None:
            continue
        if doc_id in _peer_ids_from_extra(meta.extra):
            _add(other_id)
    return linked


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
    ``DocumentNotFoundError``. Documents still referenced by
    ``superseded_by`` / ``supersedes`` raise ``DocumentLinkedError``.
    """
    try:
        storage.validate_doc_id(doc_id)
    except ValueError as exc:
        raise DocumentNotFoundError(doc_id) from exc

    meta = storage.read_meta(doc_id)
    if meta is None:
        raise DocumentNotFoundError(doc_id)

    peers = collect_supersession_peers(storage, doc_id)
    with doc_lock(doc_id, *peers):
        meta = storage.read_meta(doc_id)
        if meta is None:
            raise DocumentNotFoundError(doc_id)
        linked = collect_supersession_peers(storage, doc_id)
        if linked:
            raise DocumentLinkedError(doc_id, linked)

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
