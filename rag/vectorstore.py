"""Local Qdrant vector store helpers."""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
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
