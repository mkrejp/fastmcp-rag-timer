"""Local Qdrant vector store helpers."""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rag.config import COLLECTION_NAME
from rag.embeddings import EMBEDDING_DIM


def open_qdrant(qdrant_dir: Path | str, *, collection_name: str = COLLECTION_NAME) -> QdrantClient:
    """Open (or create) the on-disk Qdrant collection used by the RAG engine."""
    client = QdrantClient(path=str(qdrant_dir))
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return client
