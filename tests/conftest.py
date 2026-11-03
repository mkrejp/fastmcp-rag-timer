"""Shared pytest fixtures for RAG storage tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

# Prevent main.py from eagerly locking the production Qdrant path during imports.
os.environ["RAG_SKIP_DEFAULT_APP"] = "1"

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from embeddings import EMBEDDING_DIM
from storage import RagStorage

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture()
def rag_root(tmp_path: Path) -> Path:
    """Provide an isolated RAG data root directory."""
    root = tmp_path / "rag"
    root.mkdir()
    return root


@pytest.fixture()
def rag_storage(rag_root: Path) -> RagStorage:
    """Provide initialized ``RagStorage`` under a temp root."""
    storage = RagStorage(rag_root)
    storage.ensure_layout()
    return storage


@pytest.fixture()
def qdrant_client(rag_storage: RagStorage) -> Iterator[QdrantClient]:
    """Provide a local Qdrant client bound to the temp RAG root."""
    client = QdrantClient(path=str(rag_storage.qdrant_dir))
    client.create_collection(
        collection_name="client_domain_knowledge",
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    try:
        yield client
    finally:
        client.close()


@pytest.fixture()
def fake_embed() -> object:
    """Return a deterministic embedding function for tests."""

    def _embed(text: str) -> list[float]:
        """Map text length into a fixed-size pseudo-embedding."""
        base = float(len(text) % 97) / 97.0
        return [base] * EMBEDDING_DIM

    return _embed
