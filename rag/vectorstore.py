"""Local Qdrant vector store helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

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
    LOCALITY_LEVELS,
    LOCALITY_VERIFICATION_STATUSES,
    LocalityLinkNotFoundError,
    LocalityLinkPayloadError,
    apply_locality_link_verification,
    build_locality_link_payload,
    chunk_text_sha256,
    knowledge_point_id_for_link,
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
    ("locality_level", PayloadSchemaType.KEYWORD),
    ("cast_obce_kod", PayloadSchemaType.KEYWORD),
    ("cast_obce", PayloadSchemaType.KEYWORD),
)
DEFAULT_TOP_OBCE = 20

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
    for field_name, schema in (
        ("source_type", PayloadSchemaType.KEYWORD),
        ("doc_id", PayloadSchemaType.KEYWORD),
        ("chunk_sha256", PayloadSchemaType.KEYWORD),
        ("localities", PayloadSchemaType.KEYWORD),
        ("cast_obce", PayloadSchemaType.KEYWORD),
        ("verification_status", PayloadSchemaType.KEYWORD),
        ("figure_number", PayloadSchemaType.KEYWORD),
        ("page_number", PayloadSchemaType.INTEGER),
        ("verified", PayloadSchemaType.BOOL),
        ("image_id", PayloadSchemaType.KEYWORD),
        ("geo_refined.precision", PayloadSchemaType.KEYWORD),
        ("geo_refined.status", PayloadSchemaType.KEYWORD),
        ("geo_refined.confidence_level", PayloadSchemaType.KEYWORD),
        ("geo_refined.needs_refinement", PayloadSchemaType.BOOL),
        ("geo_refined.verified", PayloadSchemaType.BOOL),
    ):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
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


def _as_keyword_list(value: object) -> list[str]:
    """Normalize a payload value into a de-duplicated list of strings."""
    items: list[str] = []
    if isinstance(value, list):
        raw_items: Sequence[object] = value
    elif value in (None, ""):
        raw_items = []
    else:
        raw_items = [value]
    for item in raw_items:
        text = str(item).strip()
        if text and text not in items:
            items.append(text)
    return items


def _merge_keyword_list(existing: object, value: str) -> list[str]:
    """Append ``value`` to ``existing`` without duplicates."""
    merged = _as_keyword_list(existing)
    text = value.strip()
    if text and text not in merged:
        merged.append(text)
    return merged


def backfill_chunk_payload_fields(
    client: QdrantClient,
    *,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, int]:
    """Fill missing ``chunk_sha256`` / ``localities`` / ``cast_obce`` via set_payload.

    Does not upsert vectors, re-embed, or delete points. Existing non-empty
    locality lists and hashes are left unchanged.
    """
    if not client.collection_exists(collection_name):
        return {"scanned": 0, "updated": 0}
    scanned = 0
    updated = 0
    next_offset: Any = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection_name,
            limit=64,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        for point in batch:
            scanned += 1
            payload = point.payload or {}
            patch: dict[str, Any] = {}
            if "localities" not in payload or payload.get("localities") is None:
                patch["localities"] = []
            if "cast_obce" not in payload or payload.get("cast_obce") is None:
                patch["cast_obce"] = []
            if not payload.get("chunk_sha256"):
                text = payload.get("text") or payload.get("content_summary") or ""
                if text:
                    patch["chunk_sha256"] = chunk_text_sha256(str(text))
            if not patch:
                continue
            client.set_payload(
                collection_name=collection_name,
                payload=patch,
                points=[point.id],
                wait=True,
            )
            updated += 1
        if next_offset is None:
            break
    if updated:
        logger.info(
            "Backfilled chunk payload fields on %s: scanned=%s updated=%s",
            collection_name,
            scanned,
            updated,
        )
    return {"scanned": scanned, "updated": updated}


def denormalize_knowledge_localities(
    client: QdrantClient,
    payloads: Sequence[Mapping[str, Any]],
    *,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Merge obec_id / cast_obce onto matching knowledge points via set_payload.

    Missing knowledge points are skipped. Vectors are never rewritten.
    """
    if not payloads or not client.collection_exists(collection_name):
        return 0
    updated = 0
    for payload in payloads:
        point_id = knowledge_point_id_for_link(payload)
        if not point_id:
            continue
        found = client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not found:
            continue
        existing = dict(found[0].payload or {})
        obec_id = str(payload.get("obec_id") or "").strip()
        if not obec_id:
            continue
        patch: dict[str, Any] = {
            "localities": _merge_keyword_list(existing.get("localities"), obec_id),
        }
        kod = str(
            payload.get("cast_obce_kod") or payload.get("cast_obce") or ""
        ).strip()
        if kod:
            patch["cast_obce"] = _merge_keyword_list(existing.get("cast_obce"), kod)
        elif "cast_obce" not in existing or existing.get("cast_obce") is None:
            patch["cast_obce"] = _as_keyword_list(existing.get("cast_obce"))
        if not existing.get("chunk_sha256"):
            text = existing.get("text") or existing.get("content_summary") or ""
            if text:
                patch["chunk_sha256"] = chunk_text_sha256(str(text))
        client.set_payload(
            collection_name=collection_name,
            payload=patch,
            points=[found[0].id],
            wait=True,
        )
        updated += 1
    return updated


def knowledge_locality_filter(
    *,
    obec_id: str | None = None,
    cast_obce: str | None = None,
) -> Filter | None:
    """Build a payload filter for ``search_domain_knowledge`` locality args."""
    must: list[FieldCondition] = []
    if obec_id:
        must.append(FieldCondition(key="localities", match=MatchValue(value=obec_id)))
    if cast_obce:
        must.append(FieldCondition(key="cast_obce", match=MatchValue(value=cast_obce)))
    if not must:
        return None
    return Filter(must=must)


def _locality_point_vector(
    client: QdrantClient,
) -> Mapping[str, Sequence[float]] | list[float] | dict[str, Any]:
    """Return the vector payload required by the existing ``locality_links`` layout.

    Payload-only collections (the default) use ``{}``. A dummy 1-dim collection
    that already exists is written with a zero vector and is never queried by
    similarity. Existing collections are never deleted to change layout.
    """
    _legacy, named_sizes = _collection_vector_layout(client, LOCALITY_LINKS_COLLECTION)
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
    locality_level: str | None = None,
    cast_obce: str | None = None,
) -> Filter | None:
    """Build a Qdrant payload filter for ``list_locality_links``."""
    must: list[FieldCondition] = []
    if doc_id:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
    if obec_id:
        must.append(FieldCondition(key="obec_id", match=MatchValue(value=obec_id)))
    if status:
        if status not in LOCALITY_VERIFICATION_STATUSES:
            allowed = ", ".join(sorted(LOCALITY_VERIFICATION_STATUSES))
            raise LocalityLinkPayloadError(f"status must be one of: {allowed}")
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
    if locality_level:
        if locality_level not in LOCALITY_LEVELS:
            allowed = ", ".join(sorted(LOCALITY_LEVELS))
            raise LocalityLinkPayloadError(f"locality_level must be one of: {allowed}")
        must.append(
            FieldCondition(
                key="locality_level",
                match=MatchValue(value=locality_level),
            )
        )
    if cast_obce:
        must.append(
            FieldCondition(
                key="cast_obce_kod",
                match=MatchValue(value=cast_obce),
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
    existing_by_id = {str(point.id): (point.payload or {}) for point in existing_points}
    points: list[PointStruct] = []
    ids: list[str] = []
    built_payloads: list[dict[str, Any]] = []
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
        built_payloads.append(payload)
    client.upsert(
        collection_name=LOCALITY_LINKS_COLLECTION,
        points=points,
        wait=True,
    )
    chunks_updated = denormalize_knowledge_localities(client, built_payloads)
    return {
        "status": "ok",
        "inserted": inserted,
        "updated": updated,
        "upserted": len(ids),
        "ids": ids,
        "chunks_updated": chunks_updated,
    }


def list_locality_links(
    client: QdrantClient,
    *,
    doc_id: str | None = None,
    obec_id: str | None = None,
    status: str | None = None,
    ambiguous: bool | None = None,
    locality_level: str | None = None,
    cast_obce: str | None = None,
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
        locality_level=locality_level or None,
        cast_obce=cast_obce or None,
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
    page = gathered[offset_n : offset_n + limit_n]
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


def _iter_locality_link_points(
    client: QdrantClient,
) -> Iterator[Any]:
    """Yield every locality_links point (payload only, no vectors)."""
    ensure_locality_links_collection(client)
    next_offset: Any = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=LOCALITY_LINKS_COLLECTION,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        yield from batch
        if next_offset is None:
            break


def verify_locality_link(
    client: QdrantClient,
    *,
    link_id: str,
    status: str,
    verified_by: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Set verification fields on an existing locality_links point."""
    trimmed = str(link_id).strip()
    if not trimmed:
        raise LocalityLinkPayloadError("link_id is required")
    ensure_locality_links_collection(client)
    found = client.retrieve(
        collection_name=LOCALITY_LINKS_COLLECTION,
        ids=[trimmed],
        with_payload=True,
        with_vectors=False,
    )
    if not found:
        raise LocalityLinkNotFoundError(f"link not found: {trimmed}")
    point = found[0]
    existing = dict(point.payload or {})
    existing.setdefault("link_id", str(point.id))
    payload = apply_locality_link_verification(
        existing,
        status=status,
        verified_by=verified_by,
        notes=notes,
    )
    client.upsert(
        collection_name=LOCALITY_LINKS_COLLECTION,
        points=[
            PointStruct(
                id=str(point.id),
                vector=_locality_point_vector(client),
                payload=payload,
            )
        ],
        wait=True,
    )
    return {"status": "ok", "link": payload}


def locality_link_stats(
    client: QdrantClient,
    *,
    top_obce: int = DEFAULT_TOP_OBCE,
) -> dict[str, Any]:
    """Aggregate locality_links counts by status, method, obec, and ambiguous."""
    ensure_locality_links_collection(client)
    try:
        top_n = int(top_obce)
    except (TypeError, ValueError) as exc:
        raise LocalityLinkPayloadError("top_obce must be an integer") from exc
    top_n = min(max(top_n, 1), 100)
    by_status: Counter[str] = Counter()
    by_method: Counter[str] = Counter()
    by_level: Counter[str] = Counter()
    ambiguous_true = 0
    obec_counts: Counter[str] = Counter()
    obec_names: dict[str, str] = {}
    docs: set[str] = set()
    total = 0
    for point in _iter_locality_link_points(client):
        payload = point.payload or {}
        total += 1
        status = str(payload.get("verification_status") or "unverified")
        by_status[status] += 1
        method = str(payload.get("method") or "")
        if method:
            by_method[method] += 1
        level = str(payload.get("locality_level") or "obec")
        by_level[level] += 1
        if payload.get("ambiguous"):
            ambiguous_true += 1
        obec_id = str(payload.get("obec_id") or "")
        if obec_id and status != "rejected":
            obec_counts[obec_id] += 1
            name = payload.get("name")
            if isinstance(name, str) and name and obec_id not in obec_names:
                obec_names[obec_id] = name
        doc_id = payload.get("doc_id")
        if doc_id and status != "rejected":
            docs.add(str(doc_id))
    ranked = obec_counts.most_common(top_n)
    return {
        "total": total,
        "by_status": {
            "unverified": by_status.get("unverified", 0),
            "verified": by_status.get("verified", 0),
            "rejected": by_status.get("rejected", 0),
        },
        "by_method": dict(by_method),
        "by_locality_level": {
            "obec": by_level.get("obec", 0),
            "cast_obce": by_level.get("cast_obce", 0),
        },
        "ambiguous": {
            "true": ambiguous_true,
            "false": total - ambiguous_true,
        },
        "docs_covered": len(docs),
        "top_obce": [
            {
                "obec_id": obec_id,
                "name": obec_names.get(obec_id, ""),
                "count": count,
            }
            for obec_id, count in ranked
        ],
    }


def open_qdrant(
    qdrant_dir: Path | str,
    *,
    collection_name: str = COLLECTION_NAME,
) -> QdrantClient:
    """Open or create the local collection with named ``text_vector`` + ``image_vector``.

    Collections with a single unnamed vector are recreated. Re-ingest after recreate.
    Also ensures the payload-only ``locality_links`` collection exists (never
    deletes it). Existing knowledge points missing ``chunk_sha256`` /
    ``localities`` / ``cast_obce`` are updated with ``set_payload`` only.
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
    backfill_chunk_payload_fields(client, collection_name=collection_name)
    ensure_locality_links_collection(client)
    return client


def search_similar(
    client: QdrantClient,
    *,
    query_vector: Sequence[float],
    limit: int = 3,
    collection_name: str = COLLECTION_NAME,
    vector_name: str = TEXT_VECTOR_NAME,
    query_filter: Filter | None = None,
) -> list[ScoredPoint]:
    """Return nearest points for ``query_vector`` on ``vector_name``."""
    response = client.query_points(
        collection_name=collection_name,
        query=list(query_vector),
        using=vector_name,
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
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
