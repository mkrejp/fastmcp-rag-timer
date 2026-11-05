"""Local Qdrant vector store helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from rag.config import COLLECTION_NAME
from rag.embeddings import (
    embedding_dim,
    embedding_model,
    resolve_embedding_provider,
)
from rag.logging_setup import get_logger

BACKEND_MARKER_NAME = "embedding_backend.json"
logger = get_logger("rag.vectorstore")


def backend_marker_path(qdrant_dir: Path | str) -> Path:
    """Return the sidecar file that records the provider used for this index."""
    return Path(qdrant_dir).resolve().parent / BACKEND_MARKER_NAME


def _read_backend_marker(path: Path) -> dict[str, Any] | None:
    """Load the embedding backend marker, or ``None`` if missing/invalid."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_backend_marker(path: Path, *, provider: str, model: str, dim: int) -> None:
    """Persist the active embedding backend next to the Qdrant directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provider": provider, "model": model, "dim": dim}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _backend_mismatch(
    marker: dict[str, Any] | None,
    *,
    provider: str,
    model: str,
    dim: int,
) -> bool:
    """Return whether the on-disk index was built with a different backend.

    A missing marker is treated as the historical Novita BGE-M3 index so a
    default deploy does not wipe production vectors. Switching to FastEmbed
    (or any other provider/model/dim) recreates the collection.
    """
    if marker is None:
        return provider != "novita"
    marker_dim = marker.get("dim")
    try:
        stored_dim = int(marker_dim) if marker_dim is not None else None
    except (TypeError, ValueError):
        stored_dim = None
    return (
        str(marker.get("provider") or "") != provider
        or str(marker.get("model") or "") != model
        or stored_dim != dim
    )


def ensure_payload_indexes(
    client: QdrantClient,
    *,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Create keyword indexes used by filters and admin tooling."""
    if not client.collection_exists(collection_name):
        return
    for field_name in ("source_type", "doc_id"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:  # noqa: BLE001 - index may already exist
            logger.debug(
                "Payload index %s on %s skipped: %s",
                field_name,
                collection_name,
                exc,
            )


def open_qdrant(
    qdrant_dir: Path | str,
    *,
    collection_name: str = COLLECTION_NAME,
) -> QdrantClient:
    """Open or create the local collection with the active embedding dimension.

    If the stored collection's vector size does not match ``embedding_dim()``,
    or the recorded provider/model differs from the active backend, delete and
    recreate the collection. Existing vectors in that collection are discarded;
    they cannot be used with the active model. Re-ingest documents to rebuild
    the index.
    """
    dim = embedding_dim()
    provider = resolve_embedding_provider()
    model = embedding_model()
    marker_path = backend_marker_path(qdrant_dir)
    marker = _read_backend_marker(marker_path)

    client = QdrantClient(path=str(qdrant_dir))
    if client.collection_exists(collection_name):
        collection = client.get_collection(collection_name)
        vectors = collection.config.params.vectors
        current_size = getattr(vectors, "size", None)
        size_mismatch = current_size is not None and current_size != dim
        provider_mismatch = _backend_mismatch(
            marker, provider=provider, model=model, dim=dim
        )
        if size_mismatch or provider_mismatch:
            logger.warning(
                "Recreating Qdrant collection %s (size %s -> %s, provider %s, model %s)",
                collection_name,
                current_size,
                dim,
                provider,
                model,
            )
            client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    _write_backend_marker(marker_path, provider=provider, model=model, dim=dim)
    ensure_payload_indexes(client, collection_name=collection_name)
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
