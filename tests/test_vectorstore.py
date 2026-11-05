"""Tests for Qdrant named-vector initialization."""

from __future__ import annotations

import json
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rag.embeddings import EMBEDDING_DIM, FASTEMBED_MODEL
from rag.image_embed import IMAGE_EMBEDDING_DIM
from rag.vectorstore import (
    BACKEND_MARKER_NAME,
    IMAGE_VECTOR_NAME,
    MARKER_SCHEMA_VERSION,
    TEXT_VECTOR_NAME,
    open_qdrant,
)


def _marker_payload(qdrant_dir: Path) -> dict[str, object]:
    """Read the embedding backend marker written next to ``qdrant_dir``."""
    path = qdrant_dir.parent / BACKEND_MARKER_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def test_open_qdrant_creates_named_vectors(tmp_path: Path) -> None:
    """A new collection exposes text_vector and image_vector with expected sizes."""
    qdrant_dir = tmp_path / "qdrant"
    client = open_qdrant(qdrant_dir)
    try:
        info = client.get_collection("client_domain_knowledge")
        vectors = info.config.params.vectors
        assert isinstance(vectors, dict)
        assert vectors[TEXT_VECTOR_NAME].size == EMBEDDING_DIM
        assert vectors[IMAGE_VECTOR_NAME].size == IMAGE_EMBEDDING_DIM
        marker = _marker_payload(qdrant_dir)
        assert marker["schema_version"] == MARKER_SCHEMA_VERSION
        assert marker["text_dim"] == EMBEDDING_DIM
        assert marker["image_dim"] == IMAGE_EMBEDDING_DIM
    finally:
        client.close()


def test_open_qdrant_recreates_legacy_single_vector_collection(tmp_path: Path) -> None:
    """A persisted single-vector collection is replaced with named vectors."""
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
        vectors = info.config.params.vectors
        assert isinstance(vectors, dict)
        assert vectors[TEXT_VECTOR_NAME].size == EMBEDDING_DIM
    finally:
        client.close()


def test_open_qdrant_keeps_matching_named_collection(tmp_path: Path) -> None:
    """An existing named-vector collection with the expected schema is kept."""
    qdrant_dir = tmp_path / "qdrant"
    old_client = QdrantClient(path=str(qdrant_dir))
    old_client.create_collection(
        collection_name="client_domain_knowledge",
        vectors_config={
            TEXT_VECTOR_NAME: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            IMAGE_VECTOR_NAME: VectorParams(
                size=IMAGE_EMBEDDING_DIM, distance=Distance.COSINE
            ),
        },
    )
    old_client.close()

    client = open_qdrant(qdrant_dir)
    try:
        info = client.get_collection("client_domain_knowledge")
        vectors = info.config.params.vectors
        assert isinstance(vectors, dict)
        assert vectors[TEXT_VECTOR_NAME].size == EMBEDDING_DIM
    finally:
        client.close()


def test_open_qdrant_allows_ingest_provider_switch_without_recreate(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Switching ingest novita→fastembed keeps the named collection (same BGE-M3 dim)."""
    monkeypatch.setenv("EMBEDDING_PROVIDER_QUERY", "fastembed")
    qdrant_dir = tmp_path / "qdrant"
    first = open_qdrant(qdrant_dir)
    first.close()
    assert _marker_payload(qdrant_dir)["ingest_provider"] == "novita"

    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    client = open_qdrant(qdrant_dir)
    try:
        info = client.get_collection("client_domain_knowledge")
        vectors = info.config.params.vectors
        assert isinstance(vectors, dict)
        assert _marker_payload(qdrant_dir)["ingest_provider"] == "fastembed"
        assert _marker_payload(qdrant_dir)["ingest_model"] == FASTEMBED_MODEL
        assert info.points_count == 0
    finally:
        client.close()
