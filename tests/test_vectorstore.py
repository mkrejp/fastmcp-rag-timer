"""Tests for Qdrant embedding-dimension initialization."""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rag.embeddings import EMBEDDING_DIM
from rag.vectorstore import open_qdrant


def test_open_qdrant_creates_active_embedding_dimension(tmp_path: Path) -> None:
    """A new collection matches the active BGE-M3 vector dimension."""
    client = open_qdrant(tmp_path / "qdrant")
    try:
        info = client.get_collection("client_domain_knowledge")
        assert info.config.params.vectors.size == EMBEDDING_DIM
    finally:
        client.close()


def test_open_qdrant_recreates_collection_with_old_dimension(tmp_path: Path) -> None:
    """A persisted collection with an old dimension is replaced on startup."""
    qdrant_dir = tmp_path / "qdrant"
    old_client = QdrantClient(path=str(qdrant_dir))
    old_client.create_collection(
        collection_name="client_domain_knowledge",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    old_client.close()

    client = open_qdrant(qdrant_dir)
    try:
        info = client.get_collection("client_domain_knowledge")
        assert info.config.params.vectors.size == EMBEDDING_DIM
    finally:
        client.close()
