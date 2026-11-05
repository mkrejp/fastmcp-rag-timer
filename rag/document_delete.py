"""Remove a document from disk and from the Qdrant index."""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from rag.storage import RagStorage


def delete_qdrant_points_for_doc(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
) -> None:
    """Delete all vector points whose payload ``doc_id`` matches."""
    if not qdrant.collection_exists(collection_name):
        return
    qdrant.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                ]
            )
        ),
    )


def delete_document(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    doc_id: str,
) -> None:
    """Remove filesystem artifacts and all Qdrant points for ``doc_id``."""
    storage.validate_doc_id(doc_id)
    delete_qdrant_points_for_doc(
        qdrant, collection_name=collection_name, doc_id=doc_id
    )
    storage.remove_document_files(doc_id)
