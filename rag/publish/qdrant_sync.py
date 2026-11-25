"""Export/import Qdrant points by ``doc_id`` for incremental publishes.

Embedded ``QdrantClient(path=...)`` has no snapshot API; official guidance is
scroll (+filter) + upsert, or rsync the data dir after ``client.close()``.
Named-vector upserts must resend every vector present on the point — an omitted
named vector is nulled. Point ids in this project are UUID strings (uuid5).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from rag.config import COLLECTION_NAME, LOCALITY_LINKS_COLLECTION
from rag.embeddings import TEXT_EMBEDDING_DIM
from rag.image_embed import IMAGE_EMBEDDING_DIM
from rag.logging_setup import get_logger
from rag.vectorstore import (
    IMAGE_VECTOR_NAME,
    TEXT_VECTOR_NAME,
    _iter_locality_link_points,
    _locality_point_vector,
    ensure_locality_links_collection,
    ensure_payload_indexes,
)

logger = get_logger("rag.publish.qdrant_sync")

POINTS_REL_PATH = "qdrant/points.jsonl"
_SCROLL_LIMIT = 256
DEFAULT_SYNC_COLLECTIONS: tuple[str, ...] = (
    COLLECTION_NAME,
    LOCALITY_LINKS_COLLECTION,
)


def doc_id_filter(doc_id: str) -> Filter:
    """Return a payload filter matching ``doc_id``."""
    return Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])


def open_qdrant_for_sync(qdrant_dir: Path | str) -> QdrantClient:
    """Open embedded Qdrant for publish/apply without deleting existing data.

    Creates missing collections with the expected named-vector schema. Does
    **not** recreate collections on marker mismatch (unlike ``open_qdrant``).
    Caller must ensure the writer/reader process is stopped (single-writer lock).
    """
    path = Path(qdrant_dir)
    path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(path))
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                TEXT_VECTOR_NAME: VectorParams(
                    size=TEXT_EMBEDDING_DIM, distance=Distance.COSINE
                ),
                IMAGE_VECTOR_NAME: VectorParams(
                    size=IMAGE_EMBEDDING_DIM, distance=Distance.COSINE
                ),
            },
        )
    ensure_payload_indexes(client, collection_name=COLLECTION_NAME)
    ensure_locality_links_collection(client)
    return client


def scroll_points_for_doc(
    client: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    with_vectors: bool = True,
) -> list[Any]:
    """Scroll every point in ``collection_name`` with payload ``doc_id``."""
    if not client.collection_exists(collection_name):
        return []
    points: list[Any] = []
    offset: Any = None
    selector = doc_id_filter(doc_id)
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=selector,
            limit=_SCROLL_LIMIT,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def _coerce_vector(raw: Any) -> Any:
    """Normalize a Qdrant vector value for JSON + PointStruct round-trip."""
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        out: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None:
                out[str(key)] = None
            elif hasattr(value, "tolist"):
                out[str(key)] = [float(x) for x in value.tolist()]
            else:
                out[str(key)] = [float(x) for x in value]
        return out
    if hasattr(raw, "tolist"):
        return [float(x) for x in raw.tolist()]
    return [float(x) for x in raw]


def serialize_point_record(collection_name: str, point: Any) -> dict[str, Any]:
    """Serialize one scrolled point for JSONL transport."""
    payload = point.payload if isinstance(point.payload, Mapping) else {}
    return {
        "collection": collection_name,
        "id": str(point.id),
        "vector": _coerce_vector(point.vector),
        "payload": dict(payload),
    }


def export_points_for_docs(
    client: QdrantClient,
    doc_ids: Sequence[str],
    *,
    collections: Sequence[str] = DEFAULT_SYNC_COLLECTIONS,
    payload_only: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Export all points for ``doc_ids`` across ``collections``.

    Returns ``(records, point_ids)`` suitable for JSONL staging and the
    release manifest ``qdrant_point_ids_upserted`` field.
    """
    records: list[dict[str, Any]] = []
    point_ids: list[str] = []
    seen: set[str] = set()
    for doc_id in doc_ids:
        if not doc_id:
            continue
        for collection_name in collections:
            with_vectors = (
                not payload_only and collection_name != LOCALITY_LINKS_COLLECTION
            )
            for point in scroll_points_for_doc(
                client,
                collection_name=collection_name,
                doc_id=doc_id,
                with_vectors=with_vectors,
            ):
                record = serialize_point_record(collection_name, point)
                records.append(record)
                pid = str(record["id"])
                if pid not in seen:
                    seen.add(pid)
                    point_ids.append(pid)
    return records, point_ids


def locality_links_catalog_fingerprint(client: QdrantClient) -> dict[str, str]:
    """Return ``doc_id →`` fingerprint of that document's ``locality_links``.

    Scrolls the payload-only collection once (no embedding vectors) and hashes
    a canonical ``[{id, payload}, ...]`` per ``doc_id``. Safe for large
    knowledge collections: only link payloads are loaded.
    """
    ensure_locality_links_collection(client)
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for point in _iter_locality_link_points(client):
        payload = point.payload if isinstance(point.payload, Mapping) else {}
        doc_id = str(payload.get("doc_id") or "").strip()
        if not doc_id:
            continue
        by_doc.setdefault(doc_id, []).append(
            {
                "id": str(point.id),
                "payload": dict(payload),
            }
        )
    out: dict[str, str] = {}
    for doc_id, items in by_doc.items():
        items.sort(key=lambda row: str(row["id"]))
        canonical = json.dumps(
            items,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        out[doc_id] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return out


def write_points_jsonl(path: Path | str, records: Iterable[Mapping[str, Any]]) -> int:
    """Write point records as JSONL. Returns the number of lines written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_points_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Load point records from a JSONL file."""
    target = Path(path)
    if not target.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def delete_points_for_docs(
    client: QdrantClient,
    doc_ids: Sequence[str],
    *,
    collections: Sequence[str] = DEFAULT_SYNC_COLLECTIONS,
) -> int:
    """Delete every point matching ``doc_ids`` in ``collections``. Returns count."""
    removed = 0
    for doc_id in doc_ids:
        if not doc_id:
            continue
        selector = doc_id_filter(doc_id)
        for collection_name in collections:
            if not client.collection_exists(collection_name):
                continue
            count = int(
                client.count(
                    collection_name=collection_name,
                    count_filter=selector,
                    exact=True,
                ).count
            )
            if count:
                client.delete(
                    collection_name=collection_name,
                    points_selector=FilterSelector(filter=selector),
                    wait=True,
                )
                removed += count
    return removed


def _coerce_point_id(raw: Any) -> int | str | uuid.UUID:
    """Accept uuid5 strings, UUID objects, or uint64 point ids."""
    if isinstance(raw, uuid.UUID):
        return raw
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    try:
        return uuid.UUID(text)
    except ValueError:
        pass
    if text.isdigit():
        return int(text)
    # Last resort: pass through (Qdrant will reject illegal ids).
    return text


def _knowledge_vector_from_record(raw: Any) -> Mapping[str, Any]:
    """Preserve the full named-vector map from export (omit ⇒ null on upsert)."""
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(
            "knowledge point export missing named vectors; "
            "re-export with scroll with_vectors=True"
        )
    out: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key)
        if value is None:
            # Explicit null keeps the "present but empty" case distinct from omit.
            out[name] = None
            continue
        if name not in {TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME}:
            out[name] = value
            continue
        if hasattr(value, "tolist"):
            out[name] = [float(x) for x in value.tolist()]
        else:
            out[name] = [float(x) for x in value]
    if TEXT_VECTOR_NAME not in out and IMAGE_VECTOR_NAME not in out:
        raise ValueError("knowledge point has neither text_vector nor image_vector")
    return out


def _point_struct_from_record(
    client: QdrantClient,
    record: Mapping[str, Any],
    *,
    collection_name: str,
) -> PointStruct:
    """Build a ``PointStruct`` from a serialized record."""
    vector = record.get("vector")
    if collection_name == LOCALITY_LINKS_COLLECTION:
        # Match whatever layout already exists on the reader ({} or dummy 1-d).
        vector = _locality_point_vector(client)
    elif collection_name == COLLECTION_NAME:
        vector = _knowledge_vector_from_record(vector)
    elif vector is None:
        vector = {}
    return PointStruct(
        id=_coerce_point_id(record["id"]),
        vector=vector,
        payload=dict(record.get("payload") or {}),
    )


def apply_payload_only_records(
    client: QdrantClient,
    records: Sequence[Mapping[str, Any]],
) -> int:
    """Merge payloads onto existing points via ``set_payload`` (no vector I/O).

    Records must include ``collection``, ``id``, and ``payload``. Missing points
    are skipped (logged at debug). Returns the number of points touched.
    """
    written = 0
    for record in records:
        collection = str(record.get("collection") or COLLECTION_NAME)
        if not client.collection_exists(collection):
            continue
        payload = record.get("payload")
        if not isinstance(payload, Mapping) or not payload:
            continue
        point_id = _coerce_point_id(record["id"])
        try:
            client.set_payload(
                collection_name=collection,
                payload=dict(payload),
                points=[point_id],
                wait=True,
            )
            written += 1
        except Exception:
            logger.debug(
                "set_payload skipped collection=%s id=%s",
                collection,
                record.get("id"),
                exc_info=True,
            )
    return written


def upsert_point_records(
    client: QdrantClient,
    records: Sequence[Mapping[str, Any]],
) -> int:
    """Upsert serialized point records. Returns the number of points written."""
    by_collection: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        collection = str(record.get("collection") or COLLECTION_NAME)
        by_collection.setdefault(collection, []).append(record)
    written = 0
    for collection_name, collection_records in by_collection.items():
        if collection_name == LOCALITY_LINKS_COLLECTION:
            ensure_locality_links_collection(client)
        elif not client.collection_exists(collection_name):
            # Knowledge collection should already exist via open_qdrant_for_sync.
            raise RuntimeError(f"missing Qdrant collection: {collection_name}")
        points = [
            _point_struct_from_record(client, record, collection_name=collection_name)
            for record in collection_records
        ]
        # Batch upserts to keep memory bounded for large docs.
        batch_size = 64
        for start in range(0, len(points), batch_size):
            end = start + batch_size
            chunk = points[start:end]
            client.upsert(collection_name=collection_name, points=chunk, wait=True)
            written += len(chunk)
    return written


def apply_qdrant_release_with_client(
    client: QdrantClient,
    *,
    upsert_doc_ids: Sequence[str],
    delete_doc_ids: Sequence[str],
    metadata_only_doc_ids: Sequence[str] = (),
    locality_link_doc_ids: Sequence[str] = (),
    points_jsonl: Path | str | None = None,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Delete then upsert points using an already-open ``client``.

    Does **not** close ``client`` (safe for the live reader process). Prefer this
    over :func:`apply_qdrant_release` whenever a Qdrant handle is already held.

    Catalog upserts/deletes purge **both** knowledge and locality collections.
    ``locality_link_doc_ids`` (link-only sync) purge **only** ``locality_links``
    so knowledge vectors are not reloaded for a link change.
    """
    loaded: Sequence[Mapping[str, Any]]
    if records is not None:
        loaded = records
    elif points_jsonl:
        loaded = read_points_jsonl(points_jsonl)
    else:
        loaded = ()
    metadata_only_set = {str(d) for d in metadata_only_doc_ids if d}
    vector_upsert_ids = sorted(
        {
            str(d)
            for d in upsert_doc_ids
            if d and str(d) not in metadata_only_set
        }
    )
    full_ids = sorted(
        {str(d) for d in vector_upsert_ids + list(delete_doc_ids) if d}
    )
    full_set = set(full_ids) | metadata_only_set
    link_only = sorted(
        {str(d) for d in locality_link_doc_ids if d and str(d) not in full_set}
    )
    deleted = delete_points_for_docs(client, full_ids)
    if link_only:
        deleted += delete_points_for_docs(
            client,
            link_only,
            collections=(LOCALITY_LINKS_COLLECTION,),
        )
    if loaded and metadata_only_set:
        meta_ids = metadata_only_set
        vector_records = [
            rec
            for rec in loaded
            if str((rec.get("payload") or {}).get("doc_id") or "") not in meta_ids
        ]
        payload_records = [
            rec
            for rec in loaded
            if str((rec.get("payload") or {}).get("doc_id") or "") in meta_ids
        ]
    else:
        vector_records = list(loaded)
        payload_records = []
    written = upsert_point_records(client, vector_records) if vector_records else 0
    payload_merged = (
        apply_payload_only_records(client, payload_records) if payload_records else 0
    )
    # Re-denormalize obec lists onto knowledge payloads from applied links
    # without re-shipping embedding vectors.
    link_payloads = [
        dict(rec.get("payload") or {})
        for rec in loaded
        if str(rec.get("collection") or "") == LOCALITY_LINKS_COLLECTION
        and isinstance(rec.get("payload"), Mapping)
    ]
    chunks_updated = 0
    if link_payloads:
        from rag.vectorstore import denormalize_knowledge_localities

        chunks_updated = denormalize_knowledge_localities(client, link_payloads)
    touch_count = len(full_set | set(link_only))
    logger.info(
        "Qdrant sync (in-process) docs=%s link_only=%s points_deleted=%s "
        "points_upserted=%s chunks_denorm=%s",
        len(full_ids),
        len(link_only),
        deleted,
        written,
        chunks_updated,
    )
    return {
        "docs_touched": touch_count,
        "points_deleted": deleted,
        "points_upserted": written,
        "points_payload_merged": payload_merged,
        "records_in_bundle": len(loaded),
        "metadata_only_docs": len(metadata_only_set),
        "locality_link_docs": len(link_only),
        "chunks_denormalized": chunks_updated,
    }


def apply_qdrant_release(
    qdrant_dir: Path | str,
    *,
    upsert_doc_ids: Sequence[str],
    delete_doc_ids: Sequence[str],
    metadata_only_doc_ids: Sequence[str] = (),
    locality_link_doc_ids: Sequence[str] = (),
    points_jsonl: Path | str | None,
    client: QdrantClient | None = None,
) -> dict[str, Any]:
    """Delete then upsert points for a release onto ``qdrant_dir``.

    For upserted docs, existing points are removed first so replacements do not
    leave stale chunk/image ids. Deleted docs only remove points.

    When ``client`` is provided it is reused and **not** closed (live reader).
    Otherwise opens a temporary client on ``qdrant_dir`` and closes it (CLI
    stop-apply path).
    """
    if client is not None:
        return apply_qdrant_release_with_client(
            client,
            upsert_doc_ids=upsert_doc_ids,
            delete_doc_ids=delete_doc_ids,
            metadata_only_doc_ids=metadata_only_doc_ids,
            locality_link_doc_ids=locality_link_doc_ids,
            points_jsonl=points_jsonl,
        )
    owned = open_qdrant_for_sync(qdrant_dir)
    try:
        stats = apply_qdrant_release_with_client(
            owned,
            upsert_doc_ids=upsert_doc_ids,
            delete_doc_ids=delete_doc_ids,
            metadata_only_doc_ids=metadata_only_doc_ids,
            locality_link_doc_ids=locality_link_doc_ids,
            points_jsonl=points_jsonl,
        )
        logger.info(
            "Qdrant sync qdrant_dir=%s docs=%s points_deleted=%s points_upserted=%s",
            qdrant_dir,
            stats["docs_touched"],
            stats["points_deleted"],
            stats["points_upserted"],
        )
        return stats
    finally:
        owned.close()
