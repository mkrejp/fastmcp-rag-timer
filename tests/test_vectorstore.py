"""Tests for Qdrant embedding-dimension initialization."""

from __future__ import annotations

import json
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rag.embeddings import EMBEDDING_DIM, FASTEMBED_DIM, FASTEMBED_MODEL
from rag.vectorstore import BACKEND_MARKER_NAME, open_qdrant


def _marker_payload(qdrant_dir: Path) -> dict[str, object]:
    """Read the embedding backend marker written next to ``qdrant_dir``."""
    path = qdrant_dir.parent / BACKEND_MARKER_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def test_open_qdrant_creates_active_embedding_dimension(tmp_path: Path) -> None:
    """A new collection matches the active BGE-M3 vector dimension."""
    qdrant_dir = tmp_path / "qdrant"
    client = open_qdrant(qdrant_dir)
    try:
        info = client.get_collection("client_domain_knowledge")
        assert info.config.params.vectors.size == EMBEDDING_DIM
        assert _marker_payload(qdrant_dir) == {
            "provider": "novita",
            "model": "baai/bge-m3",
            "dim": EMBEDDING_DIM,
        }
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


def test_open_qdrant_keeps_unmarked_novita_collection(tmp_path: Path) -> None:
    """Pre-switch Novita indexes are kept when the provider stays Novita."""
    qdrant_dir = tmp_path / "qdrant"
    old_client = QdrantClient(path=str(qdrant_dir))
    old_client.create_collection(
        collection_name="client_domain_knowledge",
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    old_client.close()
    assert not (tmp_path / BACKEND_MARKER_NAME).exists()

    client = open_qdrant(qdrant_dir)
    try:
        info = client.get_collection("client_domain_knowledge")
        assert info.config.params.vectors.size == EMBEDDING_DIM
        assert _marker_payload(qdrant_dir)["provider"] == "novita"
    finally:
        client.close()


def test_open_qdrant_recreates_collection_when_provider_changes(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Switching to FastEmbed rebuilds the collection even at the same dimension."""
    qdrant_dir = tmp_path / "qdrant"
    first = open_qdrant(qdrant_dir)
    first.close()
    assert _marker_payload(qdrant_dir)["provider"] == "novita"

    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    client = open_qdrant(qdrant_dir)
    try:
        info = client.get_collection("client_domain_knowledge")
        assert info.config.params.vectors.size == FASTEMBED_DIM
        assert _marker_payload(qdrant_dir) == {
            "provider": "fastembed",
            "model": FASTEMBED_MODEL,
            "dim": FASTEMBED_DIM,
        }
    finally:
        client.close()


def test_open_qdrant_uses_embedding_dim_function(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Collection size follows ``embedding_dim()``, not a hardcoded constant."""
    monkeypatch.setattr("rag.vectorstore.embedding_dim", lambda: 384)
    monkeypatch.setattr(
        "rag.vectorstore.resolve_embedding_provider", lambda: "fastembed"
    )
    monkeypatch.setattr("rag.vectorstore.embedding_model", lambda: "tiny-test")

    client = open_qdrant(tmp_path / "qdrant")
    try:
        info = client.get_collection("client_domain_knowledge")
        assert info.config.params.vectors.size == 384
    finally:
        client.close()
