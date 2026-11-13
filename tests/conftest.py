"""Shared pytest fixtures for RAG storage tests."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator
from unittest.mock import patch

# Prevent main.py from eagerly locking the production Qdrant path during imports.
os.environ["RAG_SKIP_DEFAULT_APP"] = "1"

import pytest  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, VectorParams  # noqa: E402

from rag.embeddings import EMBEDDING_DIM, reset_fastembed_runtime  # noqa: E402
from rag.image_embed import IMAGE_EMBEDDING_DIM, reset_image_embed_runtime  # noqa: E402
from rag.storage import RagStorage  # noqa: E402
from rag.vectorstore import IMAGE_VECTOR_NAME, TEXT_VECTOR_NAME  # noqa: E402

TEST_MCP_TOKEN = "test-mcp-token"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_MCP_TOKEN}"}


@contextmanager
def patched_embeddings(
    fake_embed: Callable[[str], list[float]],
) -> Iterator[None]:
    """Patch embedding call sites used by HTTP, ingest, and MCP paths."""

    def fake_many(texts: list[str], **_kwargs: object) -> list[list[float]]:
        """Map a batch of texts through ``fake_embed``."""
        return [fake_embed(text) for text in texts]

    with (
        patch("rag.embeddings.get_embedding", side_effect=fake_embed),
        patch("rag.embeddings.get_embeddings", side_effect=fake_many),
        patch("rag.ingest.get_ingest_embeddings", side_effect=fake_many),
        patch("rag.figure_index.get_ingest_embeddings", side_effect=fake_many),
        patch("rag.mcp_tools.get_query_embedding", side_effect=fake_embed),
    ):
        yield


@pytest.fixture(autouse=True)
def _mcp_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the bearer token for /mcp and embedding routes."""
    monkeypatch.setenv("MCP_API_TOKEN", TEST_MCP_TOKEN)
    monkeypatch.delenv("TRAILS_API_URL", raising=False)
    monkeypatch.delenv("TRAILS_API_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def _default_embedding_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests on Novita ingest + local fastembed query unless overridden."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER_INGEST", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER_QUERY", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER_SEARCH", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER_QUERY", "fastembed")
    monkeypatch.setenv("IMAGE_EMBEDDING_ENABLED", "false")
    reset_fastembed_runtime()
    reset_image_embed_runtime()


@pytest.fixture(autouse=True)
def _stub_figure_index_embeddings() -> Iterator[None]:
    """Keep ``pdf_figure`` ingest off the live Novita/FastEmbed backends.

    ``ingest_bytes(embed_fn=...)`` stubs chunk/image vectors, but figure
    indexing calls ``rag.figure_index.get_ingest_embeddings`` directly.
    Without this stub the figure pipeline raises when no API key/model is
    present and is swallowed; with a working backend it indexes large
    placements and skips them from ``meta.images``.
    """

    def fake_many(texts: list[str], **_kwargs: object) -> list[list[float]]:
        """Map texts to deterministic vectors without a live backend."""
        return [[float(len(text) % 97) / 97.0] * EMBEDDING_DIM for text in texts]

    with patch("rag.figure_index.get_ingest_embeddings", side_effect=fake_many):
        yield


@pytest.fixture()
def rag_root(tmp_path: Path) -> Path:
    """Provide an isolated RAG data root directory."""
    root = tmp_path / "rag"
    root.mkdir()
    return root


@pytest.fixture()
def accnt_root(tmp_path: Path) -> Path:
    """Provide an isolated accounting data root directory."""
    root = tmp_path / "accnt"
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
        vectors_config={
            TEXT_VECTOR_NAME: VectorParams(
                size=EMBEDDING_DIM, distance=Distance.COSINE
            ),
            IMAGE_VECTOR_NAME: VectorParams(
                size=IMAGE_EMBEDDING_DIM, distance=Distance.COSINE
            ),
        },
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
