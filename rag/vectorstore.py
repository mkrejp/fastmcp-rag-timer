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

    A dimension change requires re-embedding all documents. If the stored
    collection uses an older dimension, replace it; its vectors cannot be used
    with the active model.
    """
    client = QdrantClient(path=str(qdrant_dir))
    if client.collection_exists(collection_name):
        collection = client.get_collection(collection_name)
        current_size = collection.config.params.vectors.size
        if current_size != EMBEDDING_DIM:
            client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return client
