"""Local Qdrant vector store helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag.config import COLLECTION_NAME, LOCALITY_LINKS_COLLECTION
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
from rag.payloads import (
    LOCALITY_VERIFICATION_STATUSES,
    LocalityLinkPayloadError,
    build_locality_link_payload,
)

BACKEND_MARKER_NAME = "embedding_backend.json"
MARKER_SCHEMA_VERSION = 2
TEXT_VECTOR_NAME = "text_vector"
IMAGE_VECTOR_NAME = "image_vector"
MAX_LOCALITY_LIST_LIMIT = 200
DEFAULT_LOCALITY_LIST_LIMIT = 50
_LOCALITY_LINK_INDEXES: tuple[tuple[str, PayloadSchemaType], ...] = (
    ("doc_id", PayloadSchemaType.KEYWORD),
    ("obec_id", PayloadSchemaType.KEYWORD),
    ("verification_status", PayloadSchemaType.KEYWORD),
    ("ambiguous", PayloadSchemaType.BOOL),
)

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


def ensure_locality_links_collection(client: QdrantClient) -> None:
    """Create the payload-only ``locality_links`` collection if it is missing.

    Never deletes or recreates an existing collection. Indexes are best-effort
    (no-ops on embedded Qdrant). No embedding vectors are stored or queried.
    """
    if not client.collection_exists(LOCALITY_LINKS_COLLECTION):
        client.create_collection(
            collection_name=LOCALITY_LINKS_COLLECTION,
            vectors_config={},
        )
        logger.info(
            "Created payload-only collection %s (no vectors)",
            LOCALITY_LINKS_COLLECTION,
        )
    _ensure_locality_link_indexes(client)


def _ensure_locality_link_indexes(client: QdrantClient) -> None:
    """Create filter indexes on ``locality_links`` when the backend supports them."""
    if not client.collection_exists(LOCALITY_LINKS_COLLECTION):
        return
    for field_name, field_schema in _LOCALITY_LINK_INDEXES:
        try:
            client.create_payload_index(
                collection_name=LOCALITY_LINKS_COLLECTION,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as exc:  # noqa: BLE001 - index may already exist
            logger.debug(
                "Payload index %s on %s skipped: %s",
                field_name,
                LOCALITY_LINKS_COLLECTION,
                exc,
            )


def _locality_point_vector(
    client: QdrantClient,
) -> Mapping[str, Sequence[float]] | list[float] | dict[str, Any]:
    """Return the vector payload required by the existing ``locality_links`` layout.

    Payload-only collections (the default) use ``{}``. A dummy 1-dim collection
    that already exists is written with a zero vector and is never queried by
    similarity. Existing collections are never deleted to change layout.
    """
    _legacy, named_sizes = _collection_vector_layout(
        client, LOCALITY_LINKS_COLLECTION
    )
    if not named_sizes:
        return {}
    if set(named_sizes.keys()) == {"default"} and named_sizes["default"] == 1:
        return [0.0]
    if len(named_sizes) == 1:
        name, size = next(iter(named_sizes.items()))
        if size == 1:
            return {name: [0.0]}
    return {}


def _locality_links_filter(
    *,
    doc_id: str | None = None,
    obec_id: str | None = None,
    status: str | None = None,
    ambiguous: bool | None = None,
) -> Filter | None:
    """Build a Qdrant payload filter for ``list_locality_links``."""
    must: list[FieldCondition] = []
    if doc_id:
        must.append(
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
        )
    if obec_id:
        must.append(
            FieldCondition(key="obec_id", match=MatchValue(value=obec_id))
        )
    if status:
        if status not in LOCALITY_VERIFICATION_STATUSES:
            allowed = ", ".join(sorted(LOCALITY_VERIFICATION_STATUSES))
            raise LocalityLinkPayloadError(
                f"status must be one of: {allowed}"
            )
        must.append(
            FieldCondition(
                key="verification_status",
                match=MatchValue(value=status),
            )
        )
    if ambiguous is not None:
        must.append(
            FieldCondition(
                key="ambiguous",
                match=MatchValue(value=bool(ambiguous)),
            )
        )
    if not must:
        return None
    return Filter(must=must)


def upsert_locality_links(
    client: QdrantClient,
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Upsert locality-link payloads into ``locality_links`` (no embeddings)."""
    if not items:
        raise LocalityLinkPayloadError("at least one link is required")
    ensure_locality_links_collection(client)
    raw_items: list[Mapping[str, Any]] = []
    tentative_ids: list[str] = []
    for raw in items:
        if not isinstance(raw, Mapping):
            raise LocalityLinkPayloadError("each link must be a JSON object")
        link_id, _payload = build_locality_link_payload(raw)
        raw_items.append(raw)
        tentative_ids.append(link_id)
    existing_points = client.retrieve(
        collection_name=LOCALITY_LINKS_COLLECTION,
        ids=tentative_ids,
        with_payload=True,
        with_vectors=False,
    )
    existing_by_id = {
        str(point.id): (point.payload or {}) for point in existing_points
    }
    points: list[PointStruct] = []
    ids: list[str] = []
    inserted = 0
    updated = 0
    vector = _locality_point_vector(client)
    for raw, tentative_id in zip(raw_items, tentative_ids, strict=True):
        existing = existing_by_id.get(tentative_id)
        link_id, payload = build_locality_link_payload(raw, existing=existing)
        if existing:
            updated += 1
        else:
            inserted += 1
        points.append(PointStruct(id=link_id, vector=vector, payload=payload))
        ids.append(link_id)
    client.upsert(
        collection_name=LOCALITY_LINKS_COLLECTION,
        points=points,
        wait=True,
    )
    return {
        "status": "ok",
        "inserted": inserted,
        "updated": updated,
        "upserted": len(ids),
        "ids": ids,
    }


def list_locality_links(
    client: QdrantClient,
    *,
    doc_id: str | None = None,
    obec_id: str | None = None,
    status: str | None = None,
    ambiguous: bool | None = None,
    limit: int = DEFAULT_LOCALITY_LIST_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """Return a page of locality links matching the given payload filters."""
    ensure_locality_links_collection(client)
    try:
        limit_n = int(limit)
    except (TypeError, ValueError) as exc:
        raise LocalityLinkPayloadError("limit must be an integer") from exc
    try:
        offset_n = int(offset)
    except (TypeError, ValueError) as exc:
        raise LocalityLinkPayloadError("offset must be an integer") from exc
    limit_n = min(max(limit_n, 1), MAX_LOCALITY_LIST_LIMIT)
    offset_n = max(offset_n, 0)
    scroll_filter = _locality_links_filter(
        doc_id=doc_id or None,
        obec_id=obec_id or None,
        status=status or None,
        ambiguous=ambiguous,
    )
    gathered: list[Any] = []
    next_offset: Any = None
    target = offset_n + limit_n + 1
    while len(gathered) < target:
        batch, next_offset = client.scroll(
            collection_name=LOCALITY_LINKS_COLLECTION,
            scroll_filter=scroll_filter,
            limit=min(100, target - len(gathered)),
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        gathered.extend(batch)
        if next_offset is None:
            break
    page = gathered[offset_n:offset_n + limit_n]
    links: list[dict[str, Any]] = []
    for point in page:
        payload = dict(point.payload or {})
        payload.setdefault("link_id", str(point.id))
        links.append(payload)
    return {
        "links": links,
        "limit": limit_n,
        "offset": offset_n,
        "returned": len(links),
        "has_more": len(gathered) > offset_n + limit_n,
    }


def open_qdrant(
    qdrant_dir: Path | str,
    *,
    collection_name: str = COLLECTION_NAME,
) -> QdrantClient:
    """Open or create the local collection with named ``text_vector`` + ``image_vector``.

    Collections with a single unnamed vector are recreated. Re-ingest after recreate.
    Also ensures the payload-only ``locality_links`` collection exists (never
    deletes it).
    """
    marker_path = backend_marker_path(qdrant_dir)
    marker = _read_backend_marker(marker_path)
    expected_marker = expected_marker_payload()

    client = QdrantClient(path=str(qdrant_dir))
    if collection_name == LOCALITY_LINKS_COLLECTION:
        ensure_locality_links_collection(client)
        return client
    legacy_layout = False
    named_sizes: dict[str, int] | None = None
    if (
        client.collection_exists(collection_name)
        and collection_name != LOCALITY_LINKS_COLLECTION
    ):
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
    ensure_locality_links_collection(client)
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
