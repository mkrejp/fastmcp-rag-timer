"""Local Qdrant vector store helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from rag.config import COLLECTION_NAME
from rag.embeddings import (
    BGE_M3_MODEL_FAMILY,
    TEXT_EMBEDDING_DIM,
    provider_text_dim,
    provider_text_model,
    resolve_embedding_provider_ingest,
)
from rag.image_embed import (
    CLIP_TEXT_MODEL,
    CLIP_VISION_MODEL,
    IMAGE_EMBEDDING_DIM,
    image_embedding_enabled,
)
from rag.logging_setup import get_logger

BACKEND_MARKER_NAME = "embedding_backend.json"
MARKER_SCHEMA_VERSION = 2
TEXT_VECTOR_NAME = "text_vector"
IMAGE_VECTOR_NAME = "image_vector"

logger = get_logger("rag.vectorstore")


def backend_marker_path(qdrant_dir: Path | str) -> Path:
    """Return the sidecar file that records the provider used for this index."""
    return Path(qdrant_dir).resolve().parent / BACKEND_MARKER_NAME


def expected_marker_payload() -> dict[str, Any]:
    """Build the marker body that matches the active configuration."""
    ingest = resolve_embedding_provider_ingest()
    return {
        "schema_version": MARKER_SCHEMA_VERSION,
        "text_model_family": BGE_M3_MODEL_FAMILY,
        "ingest_provider": ingest,
        "ingest_model": provider_text_model(ingest),
        "text_dim": TEXT_EMBEDDING_DIM,
        "text_vector": TEXT_VECTOR_NAME,
        "image_vector": IMAGE_VECTOR_NAME,
        "image_dim": IMAGE_EMBEDDING_DIM,
        "image_vision_model": CLIP_VISION_MODEL,
        "image_text_model": CLIP_TEXT_MODEL,
        "image_embedding_enabled": image_embedding_enabled(),
        # Duplicate provider/model/dim keys for marker readers that use those names.
        "provider": ingest,
        "model": provider_text_model(ingest),
        "dim": TEXT_EMBEDDING_DIM,
    }


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


def _write_backend_marker(path: Path, payload: dict[str, Any]) -> None:
    """Persist the active embedding backend next to the Qdrant directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _collection_vector_layout(
    client: QdrantClient, collection_name: str
) -> tuple[bool, dict[str, int] | None]:
    """Return whether the collection uses a single unnamed vector, plus named-vector sizes."""
    if not client.collection_exists(collection_name):
        return False, None
    collection = client.get_collection(collection_name)
    vectors = collection.config.params.vectors
    if isinstance(vectors, dict):
        sizes = {
            str(name): int(params.size)
            for name, params in vectors.items()
            if getattr(params, "size", None) is not None
        }
        return False, sizes
    size = getattr(vectors, "size", None)
    if size is not None:
        return True, {"default": int(size)}
    return True, None


def _expected_named_sizes() -> dict[str, int]:
    """Return required named vector dimensions."""
    return {
        TEXT_VECTOR_NAME: TEXT_EMBEDDING_DIM,
        IMAGE_VECTOR_NAME: IMAGE_EMBEDDING_DIM,
    }


def _backend_mismatch(marker: dict[str, Any] | None) -> bool:
    """Return whether the on-disk marker differs from the active schema."""
    expected = expected_marker_payload()
    if marker is None:
        # Layout checks handle unnamed single-vector stores; a missing marker
        # does not force a recreate when named vectors already match.
        return False
    try:
        schema_version = int(marker.get("schema_version", 0))
    except (TypeError, ValueError):
        schema_version = 0
    if schema_version < MARKER_SCHEMA_VERSION:
        return True
    ingest = str(marker.get("ingest_provider") or marker.get("provider") or "")
    if ingest and ingest != expected["ingest_provider"]:
        # Ingest provider switches (novita↔fastembed) do not require recreate.
        pass
    text_dim = marker.get("text_dim", marker.get("dim"))
    try:
        stored_text_dim = int(text_dim) if text_dim is not None else None
    except (TypeError, ValueError):
        stored_text_dim = None
    if stored_text_dim != expected["text_dim"]:
        return True
    if str(marker.get("text_model_family") or "") != BGE_M3_MODEL_FAMILY:
        return True
    if int(marker.get("image_dim") or 0) != IMAGE_EMBEDDING_DIM:
        return True
    if str(marker.get("image_vision_model") or "") != CLIP_VISION_MODEL:
        return True
    if bool(marker.get("image_embedding_enabled")) != image_embedding_enabled():
        # Toggling IMAGE_EMBEDDING_ENABLED does not require recreate; only schema.
        pass
    return False


def _layout_mismatch(named_sizes: dict[str, int] | None) -> bool:
    """Return whether named vector sizes differ from the expected schema."""
    if named_sizes is None:
        return True
    expected = _expected_named_sizes()
    for name, size in expected.items():
        if named_sizes.get(name) != size:
            return True
    return set(named_sizes.keys()) != set(expected.keys())


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
    """Open or create the local collection with named ``text_vector`` + ``image_vector``.

    Collections with a single unnamed vector are recreated. Re-ingest after recreate.
    """
    marker_path = backend_marker_path(qdrant_dir)
    marker = _read_backend_marker(marker_path)
    expected_marker = expected_marker_payload()

    client = QdrantClient(path=str(qdrant_dir))
    legacy_layout = False
    named_sizes: dict[str, int] | None = None
    if client.collection_exists(collection_name):
        legacy_layout, named_sizes = _collection_vector_layout(client, collection_name)
        provider_mismatch = _backend_mismatch(marker)
        layout_bad = legacy_layout or _layout_mismatch(named_sizes)
        if layout_bad or provider_mismatch:
            logger.warning(
                "Recreating Qdrant collection %s (legacy=%s layout=%s marker_mismatch=%s)",
                collection_name,
                legacy_layout,
                named_sizes,
                provider_mismatch,
            )
            client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                TEXT_VECTOR_NAME: VectorParams(
                    size=TEXT_EMBEDDING_DIM, distance=Distance.COSINE
                ),
                IMAGE_VECTOR_NAME: VectorParams(
                    size=IMAGE_EMBEDDING_DIM, distance=Distance.COSINE
                ),
            },
        )
        logger.warning(
            "Created named-vector collection %s (%s=%s, %s=%s). Re-ingest required.",
            collection_name,
            TEXT_VECTOR_NAME,
            TEXT_EMBEDDING_DIM,
            IMAGE_VECTOR_NAME,
            IMAGE_EMBEDDING_DIM,
        )
    _write_backend_marker(marker_path, expected_marker)
    ensure_payload_indexes(client, collection_name=collection_name)
    return client


def search_similar(
    client: QdrantClient,
    *,
    query_vector: Sequence[float],
    limit: int = 3,
    collection_name: str = COLLECTION_NAME,
    vector_name: str = TEXT_VECTOR_NAME,
) -> list[ScoredPoint]:
    """Return nearest points for ``query_vector`` on ``vector_name``."""
    response = client.query_points(
        collection_name=collection_name,
        query=list(query_vector),
        using=vector_name,
        limit=limit,
        with_payload=True,
    )
    return list(response.points)


def build_named_vector(
    *,
    text_vector: Sequence[float] | None = None,
    image_vector: Sequence[float] | None = None,
) -> Mapping[str, Sequence[float]]:
    """Build a Qdrant named-vector payload for upsert."""
    payload: dict[str, Sequence[float]] = {}
    if text_vector is not None:
        if len(text_vector) != TEXT_EMBEDDING_DIM:
            raise ValueError(
                f"text_vector length {len(text_vector)} != {TEXT_EMBEDDING_DIM}"
            )
        payload[TEXT_VECTOR_NAME] = list(text_vector)
    if image_vector is not None:
        if len(image_vector) != IMAGE_EMBEDDING_DIM:
            raise ValueError(
                f"image_vector length {len(image_vector)} != {IMAGE_EMBEDDING_DIM}"
            )
        payload[IMAGE_VECTOR_NAME] = list(image_vector)
    if not payload:
        raise ValueError("At least one named vector is required")
    return payload


def ingest_text_dim() -> int:
    """Return ingest-side text dimension (BGE-M3 family)."""
    return provider_text_dim(resolve_embedding_provider_ingest())
