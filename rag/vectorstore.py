"""Local Qdrant vector store helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from qdrant_client.models import Distance, VectorParams
from rag.config import COLLECTION_NAME
from rag.embeddings import EMBEDDING_DIM


def open_qdrant(
    qdrant_dir: Path | str,
    *,
    collection_name: str = COLLECTION_NAME,
) -> QdrantClient:
    """Open or create the local collection with the active embedding dimension.

    If the stored collection's vector size is not ``EMBEDDING_DIM``, delete and
    recreate it. Existing vectors in that collection are discarded; they cannot
    be used with the active model. Re-ingest documents to rebuild the index.
    """
    client = QdrantClient(path=str(qdrant_dir))
    if client.collection_exists(collection_name):
        collection = client.get_collection(collection_name)
        vectors = collection.config.params.vectors
        current_size = getattr(vectors, "size", None)
        if current_size is not None and current_size != EMBEDDING_DIM:
            client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return client


def search_similar(
    client: QdrantClient,
    *,
    query_vector: Sequence[float],
    limit: int = 3,
    collection_name: str = COLLECTION_NAME,
) -> list[ScoredPoint]:
    """Return nearest points for ``query_vector`` via the current Qdrant query API.

    ``QdrantClient.search`` was removed in qdrant-client 1.16+; callers must use
    ``query_points`` and read ``.points``.
    """
    response = client.query_points(
        collection_name=collection_name,
        query=list(query_vector),
        limit=limit,
        with_payload=True,
    )
    return list(response.points)
