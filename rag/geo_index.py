"""Read/write ``geo_refined`` on documents, chunks, and image records.

Uses Qdrant ``set_payload`` (no re-embed, no deletes) and optional disk
metadata. Original ``lat`` / ``lon`` fields are never included in writes.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.config import COLLECTION_NAME
from rag.geo_refined import (
    GeoRefinedNotFoundError,
    GeoRefinedPayloadError,
    ORIGINAL_COORD_KEYS,
    apply_geo_refined,
    coerce_filter_argument,
    coerce_geo_refined_argument,
    geo_patch,
    matches_geo_filter,
    original_lat_lon,
    parse_geo_refined,
    preferred_geo,
    utc_now_iso,
    validate_confidence_level,
    validate_precision,
    validate_status,
)
from rag.ingest import stable_image_point_id, stable_point_id
from rag.logging_setup import get_logger
from rag.storage import ImageMeta, RagStorage

logger = get_logger("rag.geo_index")

DEFAULT_GEO_LIST_LIMIT = 50
MAX_GEO_LIST_LIMIT = 200
MAX_GEO_BATCH_SIZE = 200
_IMAGE_SOURCE_TYPES = frozenset({"pdf_figure", "pdf_image"})


def _as_optional_bool(value: object, *, field: str) -> bool | None:
    """Coerce an optional boolean filter value."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise GeoRefinedPayloadError(f"{field} must be a boolean")


def geo_refined_filter(
    *,
    doc_id: str | None = None,
    precision: str | None = None,
    status: str | None = None,
    confidence_level: str | None = None,
    needs_refinement: bool | None = None,
    verified: bool | None = None,
) -> Filter | None:
    """Build a Qdrant payload filter for ``geo_refined`` nested fields."""
    must: list[FieldCondition] = []
    if doc_id:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
    if precision:
        must.append(
            FieldCondition(
                key="geo_refined.precision",
                match=MatchValue(value=precision),
            )
        )
    if status:
        must.append(
            FieldCondition(
                key="geo_refined.status",
                match=MatchValue(value=status),
            )
        )
    if confidence_level:
        must.append(
            FieldCondition(
                key="geo_refined.confidence_level",
                match=MatchValue(value=confidence_level),
            )
        )
    if needs_refinement is not None:
        must.append(
            FieldCondition(
                key="geo_refined.needs_refinement",
                match=MatchValue(value=bool(needs_refinement)),
            )
        )
    if verified is not None:
        must.append(
            FieldCondition(
                key="geo_refined.verified",
                match=MatchValue(value=bool(verified)),
            )
        )
    if not must:
        return None
    return Filter(must=must)


def _assert_no_original_overwrite(patch: Mapping[str, Any]) -> None:
    """Refuse patches that would write original coordinate keys."""
    for key in ORIGINAL_COORD_KEYS:
        if key in patch:
            raise GeoRefinedPayloadError(f"refusing to overwrite original {key}")


def _set_geo_payload(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    point_id: str | int,
    patch: Mapping[str, Any],
) -> None:
    """``set_payload`` only ``geo_refined`` fields (no vector rewrite)."""
    _assert_no_original_overwrite(patch)
    qdrant.set_payload(
        collection_name=collection_name,
        payload=dict(patch),
        points=[point_id],
        wait=True,
    )


def _target_view(
    *,
    doc_id: str,
    source_type: str,
    existing: Mapping[str, Any],
    updated: Mapping[str, Any],
    changed: bool,
    point_id: str | None = None,
    chunk_index: int | None = None,
    image_id: str | None = None,
) -> dict[str, Any]:
    """Shape one dry-run / write target for MCP/REST responses."""
    old = existing.get("geo_refined") if isinstance(existing.get("geo_refined"), Mapping) else None
    new = updated.get("geo_refined") if isinstance(updated.get("geo_refined"), Mapping) else None
    geo = preferred_geo(updated)
    return {
        "doc_id": doc_id,
        "source_type": source_type,
        "point_id": point_id,
        "chunk_index": chunk_index,
        "image_id": image_id,
        "old": dict(old) if old else None,
        "new": dict(new) if new else None,
        "changed": changed,
        "history_len": len(list(updated.get("geo_refined_history") or [])),
        "lat": geo["lat"],
        "lon": geo["lon"],
        "geo_source": geo["geo_source"],
        "geo_original": geo["geo_original"],
    }


def _retrieve_point(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    point_id: str,
) -> Any | None:
    """Return one Qdrant point or ``None``."""
    found = qdrant.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )
    return found[0] if found else None


def _scroll_doc_points(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    source_types: frozenset[str] | None = None,
    image_id: str | None = None,
) -> list[Any]:
    """Scroll knowledge points for ``doc_id`` with optional type/image filters."""
    must = [FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    if image_id:
        must.append(FieldCondition(key="image_id", match=MatchValue(value=image_id)))
    scroll_filter = Filter(must=must)
    points: list[Any] = []
    next_offset: Any = None
    while True:
        batch, next_offset = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=64,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        for point in batch:
            payload = point.payload or {}
            source_type = str(payload.get("source_type") or "")
            if source_types and source_type not in source_types:
                continue
            points.append(point)
        if next_offset is None:
            break
    return points


def _apply_image_meta(
    image: ImageMeta,
    geo_refined: Mapping[str, Any],
    *,
    now: str,
) -> tuple[ImageMeta, bool]:
    """Return updated ``ImageMeta`` without touching original lat/lon."""
    payload = image.to_dict()
    updated, changed = apply_geo_refined(payload, geo_refined, now=now)
    if not changed:
        return image, False
    return ImageMeta.from_dict(updated), True


def _write_document_meta(
    storage: RagStorage,
    meta: Any,
    geo_refined: Mapping[str, Any],
    *,
    now: str,
    dry_run: bool,
) -> tuple[dict[str, Any], bool]:
    """Apply geo_refined on document metadata (original lat/lon preserved)."""
    payload = meta.to_dict()
    original_lat = payload.get("lat")
    original_lon = payload.get("lon")
    updated, changed = apply_geo_refined(payload, geo_refined, now=now)
    view = _target_view(
        doc_id=meta.doc_id,
        source_type="document",
        existing=payload,
        updated=updated,
        changed=changed,
    )
    if dry_run or not changed:
        return view, changed
    meta.geo_refined = updated.get("geo_refined")
    meta.geo_refined_history = list(updated.get("geo_refined_history") or [])
    storage.write_meta(meta)
    written = storage.read_meta(meta.doc_id)
    if written is not None:
        after = written.to_dict()
        if original_lat is not None and after.get("lat") != original_lat:
            raise GeoRefinedPayloadError("original lat was overwritten")
        if original_lon is not None and after.get("lon") != original_lon:
            raise GeoRefinedPayloadError("original lon was overwritten")
    return view, changed


def _write_chunks_file(
    storage: RagStorage,
    doc_id: str,
    chunk_index: int,
    geo_refined: Mapping[str, Any],
    *,
    now: str,
    dry_run: bool,
) -> tuple[dict[str, Any], bool] | None:
    """Update one chunk in ``chunks.json`` when the file exists."""
    chunks = storage.read_chunks(doc_id)
    if chunks is None:
        return None
    match: dict[str, Any] | None = None
    match_i: int | None = None
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        try:
            current_index = int(chunk.get("index", index))
        except (TypeError, ValueError):
            current_index = index
        if current_index == chunk_index:
            match = chunk
            match_i = index
            break
    if match is None or match_i is None:
        return None
    original_lat = match.get("lat")
    original_lon = match.get("lon")
    updated, changed = apply_geo_refined(match, geo_refined, now=now)
    view = _target_view(
        doc_id=doc_id,
        source_type="pdf_chunk",
        existing=match,
        updated=updated,
        changed=changed,
        chunk_index=chunk_index,
    )
    if dry_run or not changed:
        return view, changed
    for key in ORIGINAL_COORD_KEYS:
        if key in match:
            updated[key] = match[key]
    chunks[match_i] = updated
    storage.write_chunks(doc_id, chunks)
    if original_lat is not None and updated.get("lat") != original_lat:
        raise GeoRefinedPayloadError("original lat was overwritten")
    if original_lon is not None and updated.get("lon") != original_lon:
        raise GeoRefinedPayloadError("original lon was overwritten")
    return view, changed


def _write_point(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    point: Any,
    geo_refined: Mapping[str, Any],
    now: str,
    dry_run: bool,
) -> tuple[dict[str, Any], bool]:
    """Apply geo_refined on one Qdrant knowledge point via ``set_payload``."""
    existing = dict(point.payload or {})
    original = {key: existing[key] for key in ORIGINAL_COORD_KEYS if key in existing}
    updated, changed = apply_geo_refined(existing, geo_refined, now=now)
    payload = dict(point.payload or {})
    chunk_index = payload.get("chunk_index")
    try:
        chunk_index_n = int(chunk_index) if chunk_index is not None else None
    except (TypeError, ValueError):
        chunk_index_n = None
    image_id = str(payload.get("image_id") or "") or None
    view = _target_view(
        doc_id=str(payload.get("doc_id") or ""),
        source_type=str(payload.get("source_type") or "pdf_chunk"),
        existing=existing,
        updated=updated,
        changed=changed,
        point_id=str(point.id),
        chunk_index=chunk_index_n,
        image_id=image_id,
    )
    if dry_run or not changed:
        return view, changed
    patch = geo_patch(updated)
    _set_geo_payload(
        qdrant,
        collection_name=collection_name,
        point_id=point.id,
        patch=patch,
    )
    stored = _retrieve_point(
        qdrant, collection_name=collection_name, point_id=str(point.id)
    )
    if stored is not None:
        stored_payload = stored.payload or {}
        for key, value in original.items():
            if stored_payload.get(key) != value:
                raise GeoRefinedPayloadError(f"original {key} was overwritten")
    return view, changed


def refine_geo(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    doc_id: str,
    geo_refined: object,
    chunk_index: int | None = None,
    image_id: str | None = None,
    dry_run: bool = True,
    now: str | None = None,
) -> dict[str, Any]:
    """Validate and optionally persist a ``geo_refined`` object.

    ``dry_run=true`` (the default) returns old vs new without writing.
    Writes are idempotent: repeating the same refinement does not append
    history. Original lat/lon on the target are never modified.
    """
    trimmed_doc = str(doc_id or "").strip()
    if not trimmed_doc:
        raise GeoRefinedPayloadError("doc_id is required")
    meta = storage.read_meta(trimmed_doc)
    if meta is None:
        raise GeoRefinedNotFoundError(f"document not found: {trimmed_doc}")
    raw_geo = coerce_geo_refined_argument(geo_refined)
    timestamp = now or utc_now_iso()
    image_id_n = str(image_id or "").strip() or None
    chunk_index_n = chunk_index
    # Validate up-front so dry_run and write share the same error path.
    parse_geo_refined(
        raw_geo,
        now=timestamp,
        original=original_lat_lon(meta.to_dict()),
    )

    targets: list[dict[str, Any]] = []
    any_changed = False

    if image_id_n:
        points = _scroll_doc_points(
            qdrant,
            collection_name=collection_name,
            doc_id=trimmed_doc,
            source_types=_IMAGE_SOURCE_TYPES,
            image_id=image_id_n,
        )
        if not points:
            points = [
                point
                for point in _scroll_doc_points(
                    qdrant,
                    collection_name=collection_name,
                    doc_id=trimmed_doc,
                    source_types=_IMAGE_SOURCE_TYPES,
                )
                if str((point.payload or {}).get("image_id") or "") == image_id_n
            ]
        if not points:
            point_id = stable_image_point_id(trimmed_doc, image_id_n)
            found = _retrieve_point(
                qdrant, collection_name=collection_name, point_id=point_id
            )
            if found is not None:
                points = [found]
        if not points:
            image_meta = storage.find_image(trimmed_doc, image_id_n)
            if image_meta is None:
                raise GeoRefinedNotFoundError(
                    f"image record not found: doc_id={trimmed_doc} image_id={image_id_n}"
                )
        for point in points:
            view, changed = _write_point(
                qdrant,
                collection_name=collection_name,
                point=point,
                geo_refined=raw_geo,
                now=timestamp,
                dry_run=dry_run,
            )
            targets.append(view)
            any_changed = any_changed or changed
        image_meta = storage.find_image(trimmed_doc, image_id_n)
        if image_meta is not None:
            updated_image, changed = _apply_image_meta(
                image_meta, raw_geo, now=timestamp
            )
            view = _target_view(
                doc_id=trimmed_doc,
                source_type="image_meta",
                existing=image_meta.to_dict(),
                updated=updated_image.to_dict(),
                changed=changed,
                image_id=image_id_n,
            )
            targets.append(view)
            any_changed = any_changed or changed
            if not dry_run and changed:
                images = []
                for img in meta.images:
                    images.append(
                        updated_image if img.image_id == image_id_n else img
                    )
                meta.images = images
                storage.write_meta(meta)

    if chunk_index_n is not None:
        point_id = stable_point_id(trimmed_doc, int(chunk_index_n))
        point = _retrieve_point(
            qdrant, collection_name=collection_name, point_id=point_id
        )
        if point is None:
            raise GeoRefinedNotFoundError(
                f"chunk not found: doc_id={trimmed_doc} chunk_index={chunk_index_n}"
            )
        view, changed = _write_point(
            qdrant,
            collection_name=collection_name,
            point=point,
            geo_refined=raw_geo,
            now=timestamp,
            dry_run=dry_run,
        )
        targets.append(view)
        any_changed = any_changed or changed
        disk = _write_chunks_file(
            storage,
            trimmed_doc,
            int(chunk_index_n),
            raw_geo,
            now=timestamp,
            dry_run=dry_run,
        )
        if disk is not None:
            disk_view, disk_changed = disk
            targets.append(disk_view)
            any_changed = any_changed or disk_changed

    if image_id_n is None and chunk_index_n is None:
        view, changed = _write_document_meta(
            storage, meta, raw_geo, now=timestamp, dry_run=dry_run
        )
        targets.append(view)
        any_changed = any_changed or changed
        points = _scroll_doc_points(
            qdrant,
            collection_name=collection_name,
            doc_id=trimmed_doc,
            source_types=_IMAGE_SOURCE_TYPES,
        )
        for point in points:
            view, changed = _write_point(
                qdrant,
                collection_name=collection_name,
                point=point,
                geo_refined=raw_geo,
                now=timestamp,
                dry_run=dry_run,
            )
            targets.append(view)
            any_changed = any_changed or changed
        if meta.images:
            new_images: list[ImageMeta] = []
            images_changed = False
            for img in meta.images:
                updated_image, changed = _apply_image_meta(img, raw_geo, now=timestamp)
                new_images.append(updated_image)
                view = _target_view(
                    doc_id=trimmed_doc,
                    source_type="image_meta",
                    existing=img.to_dict(),
                    updated=updated_image.to_dict(),
                    changed=changed,
                    image_id=img.image_id,
                )
                targets.append(view)
                any_changed = any_changed or changed
                images_changed = images_changed or changed
            if not dry_run and images_changed:
                # Re-read in case document-level write already persisted.
                latest = storage.read_meta(trimmed_doc) or meta
                latest.images = new_images
                storage.write_meta(latest)

    written = bool(any_changed) and not dry_run
    return {
        "status": "dry_run" if dry_run else "ok",
        "dry_run": bool(dry_run),
        "written": written,
        "would_change": bool(any_changed) if dry_run else False,
        "changed": written,
        "idempotent": not any_changed,
        "doc_id": trimmed_doc,
        "chunk_index": chunk_index_n,
        "image_id": image_id_n,
        "targets": targets,
        "changed_targets": sum(1 for item in targets if item.get("changed")),
    }


def coerce_geo_batch_items(items: object) -> list[dict[str, Any]]:
    """Accept a list of objects, a single object, or a JSON string of either."""
    parsed: object = items
    if isinstance(items, str):
        try:
            parsed = json.loads(items)
        except json.JSONDecodeError as exc:
            raise GeoRefinedPayloadError(
                "items must be a JSON object or array of objects"
            ) from exc
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise GeoRefinedPayloadError("items must be a JSON object or array of objects")
    if not parsed:
        raise GeoRefinedPayloadError("at least one item is required")
    if len(parsed) > MAX_GEO_BATCH_SIZE:
        raise GeoRefinedPayloadError(
            f"at most {MAX_GEO_BATCH_SIZE} items per batch; got {len(parsed)}"
        )
    coerced: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise GeoRefinedPayloadError("each item must be a JSON object")
        coerced.append(item)
    return coerced


def _batch_chunk_index(raw: Mapping[str, Any]) -> int | None:
    """Return optional ``chunk_index`` from a batch item."""
    if "chunk_index" not in raw or raw["chunk_index"] in (None, ""):
        return None
    try:
        return int(raw["chunk_index"])
    except (TypeError, ValueError) as exc:
        raise GeoRefinedPayloadError("chunk_index must be an integer") from exc


def refine_geo_batch(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    items: object,
    dry_run: bool = True,
    now: str | None = None,
) -> dict[str, Any]:
    """Apply ``refine_geo`` to many items with per-item success/error results.

    Same idempotency, history, and original-coordinate guarantees as
    ``refine_geo``. A single invalid item does not abort the rest of the
    batch. Rejects the whole call when ``items`` is empty or larger than
    ``MAX_GEO_BATCH_SIZE``.
    """
    coerced = coerce_geo_batch_items(items)
    timestamp = now or utc_now_iso()
    rows: list[dict[str, Any]] = []
    ok_count = 0
    error_count = 0
    would_change_count = 0
    changed_count = 0
    for index, raw in enumerate(coerced):
        doc_id = str(raw.get("doc_id") or "").strip()
        image_id = str(raw.get("image_id") or "").strip() or None
        row: dict[str, Any] = {
            "index": index,
            "doc_id": doc_id or None,
            "chunk_index": None,
            "image_id": image_id,
            "ok": False,
            "would_change": False,
            "changed": False,
            "idempotent": False,
            "error": None,
            "result": None,
        }
        try:
            chunk_index = _batch_chunk_index(raw)
            row["chunk_index"] = chunk_index
            if "geo_refined" not in raw:
                raise GeoRefinedPayloadError("missing required field: geo_refined")
            result = refine_geo(
                storage=storage,
                qdrant=qdrant,
                collection_name=collection_name,
                doc_id=doc_id,
                geo_refined=raw.get("geo_refined"),
                chunk_index=chunk_index,
                image_id=image_id,
                dry_run=dry_run,
                now=timestamp,
            )
            row["ok"] = True
            row["would_change"] = bool(result.get("would_change"))
            row["changed"] = bool(result.get("changed"))
            row["idempotent"] = bool(result.get("idempotent"))
            row["result"] = result
            ok_count += 1
            if row["would_change"]:
                would_change_count += 1
            if row["changed"]:
                changed_count += 1
        except (GeoRefinedPayloadError, GeoRefinedNotFoundError) as exc:
            row["error"] = str(exc)
            error_count += 1
        rows.append(row)
    return {
        "status": "dry_run" if dry_run else "ok",
        "dry_run": bool(dry_run),
        "max_batch_size": MAX_GEO_BATCH_SIZE,
        "received": len(coerced),
        "ok": ok_count,
        "errors": error_count,
        "would_change": would_change_count,
        "changed": changed_count,
        "items": rows,
    }


def _document_fallback_for_payload(
    payload: Mapping[str, Any],
    fallback: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Merge matching image-meta originals onto a document geo fallback."""
    if not fallback:
        return fallback
    image_id = str(payload.get("image_id") or "")
    if not image_id:
        return fallback
    images = fallback.get("images")
    if not isinstance(images, list):
        return fallback
    for image in images:
        if not isinstance(image, Mapping):
            continue
        if str(image.get("image_id") or "") != image_id:
            continue
        merged = dict(fallback)
        for key, value in image.items():
            if key in {"geo_refined", "geo_refined_history", "images"}:
                continue
            merged[key] = value
        return merged
    return fallback


def _list_record_from_payload(
    payload: Mapping[str, Any],
    *,
    point_id: str | None = None,
    fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one list_geo_refinements row from a payload."""
    geo = preferred_geo(payload, fallback=fallback)
    chunk_index = payload.get("chunk_index")
    try:
        chunk_index_n = int(chunk_index) if chunk_index is not None else None
    except (TypeError, ValueError):
        chunk_index_n = None
    history = payload.get("geo_refined_history") or (
        (fallback or {}).get("geo_refined_history") if fallback else None
    )
    return {
        "doc_id": str(payload.get("doc_id") or (fallback or {}).get("doc_id") or ""),
        "source_type": str(payload.get("source_type") or "document"),
        "point_id": point_id,
        "chunk_index": chunk_index_n,
        "image_id": str(payload.get("image_id") or "") or None,
        "lat": geo["lat"],
        "lon": geo["lon"],
        "geo_source": geo["geo_source"],
        "geo_refined": geo["geo_refined"],
        "geo_original": geo["geo_original"],
        "history_len": len(list(history or [])),
    }


def list_geo_refinements(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    filters: object = None,
    precision: str | None = None,
    status: str | None = None,
    confidence_level: str | None = None,
    needs_refinement: bool | None = None,
    verified: bool | None = None,
    doc_id: str | None = None,
    limit: int = DEFAULT_GEO_LIST_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """List stored ``geo_refined`` records with optional filters and pagination."""
    merged = coerce_filter_argument(filters)
    precision_n = validate_precision(
        str(merged.get("precision") or precision or "") or None
    )
    status_n = validate_status(str(merged.get("status") or status or "") or None)
    level_n = validate_confidence_level(
        str(merged.get("confidence_level") or confidence_level or "") or None
    )
    needs_n = _as_optional_bool(
        merged["needs_refinement"] if "needs_refinement" in merged else needs_refinement,
        field="needs_refinement",
    )
    verified_n = _as_optional_bool(
        merged["verified"] if "verified" in merged else verified,
        field="verified",
    )
    doc_n = str(merged.get("doc_id") or doc_id or "").strip() or None
    try:
        limit_n = int(merged.get("limit", limit))
    except (TypeError, ValueError) as exc:
        raise GeoRefinedPayloadError("limit must be an integer") from exc
    try:
        offset_n = int(merged.get("offset", offset))
    except (TypeError, ValueError) as exc:
        raise GeoRefinedPayloadError("offset must be an integer") from exc
    limit_n = min(max(limit_n, 1), MAX_GEO_LIST_LIMIT)
    offset_n = max(offset_n, 0)

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    doc_fallback_cache: dict[str, Mapping[str, Any] | None] = {}
    scroll_filter = geo_refined_filter(doc_id=doc_n)
    if qdrant.collection_exists(collection_name):
        next_offset: Any = None
        while True:
            batch, next_offset = qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=100,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            if not batch:
                break
            for point in batch:
                payload = dict(point.payload or {})
                if not matches_geo_filter(
                    payload,
                    precision=precision_n,
                    status=status_n,
                    confidence_level=level_n,
                    needs_refinement=needs_n,
                    verified=verified_n,
                ):
                    continue
                key = (
                    str(payload.get("doc_id") or ""),
                    str(payload.get("source_type") or ""),
                    str(point.id),
                )
                if key in seen:
                    continue
                seen.add(key)
                point_doc = str(payload.get("doc_id") or "")
                if point_doc and point_doc not in doc_fallback_cache:
                    doc_fallback_cache[point_doc] = document_geo_fallback(
                        storage, point_doc
                    )
                records.append(
                    _list_record_from_payload(
                        payload,
                        point_id=str(point.id),
                        fallback=_document_fallback_for_payload(
                            payload, doc_fallback_cache.get(point_doc)
                        ),
                    )
                )
            if next_offset is None:
                break

    catalog = storage.list_documents()
    for entry in catalog:
        catalog_doc = str(entry.get("doc_id") or "")
        if not catalog_doc:
            continue
        if doc_n and catalog_doc != doc_n:
            continue
        if catalog_doc not in doc_fallback_cache:
            doc_fallback_cache[catalog_doc] = document_geo_fallback(
                storage, catalog_doc
            )
        meta_payload = dict(doc_fallback_cache[catalog_doc] or {})
        if not meta_payload:
            meta = storage.read_meta(catalog_doc)
            if meta is None:
                continue
            meta_payload = meta.to_dict()
        meta_payload["doc_id"] = catalog_doc
        meta_payload["source_type"] = "document"
        if not matches_geo_filter(
            meta_payload,
            precision=precision_n,
            status=status_n,
            confidence_level=level_n,
            needs_refinement=needs_n,
            verified=verified_n,
        ):
            continue
        key = (catalog_doc, "document", "")
        if key in seen:
            continue
        seen.add(key)
        records.append(_list_record_from_payload(meta_payload, point_id=None))

    records.sort(
        key=lambda item: (
            str(item.get("doc_id") or ""),
            str(item.get("source_type") or ""),
            str(item.get("image_id") or ""),
            item.get("chunk_index") if item.get("chunk_index") is not None else -1,
            str(item.get("point_id") or ""),
        )
    )
    page = records[offset_n : offset_n + limit_n]
    return {
        "records": page,
        "limit": limit_n,
        "offset": offset_n,
        "returned": len(page),
        "has_more": len(records) > offset_n + limit_n,
        "total": len(records),
    }


def document_geo_fallback(storage: RagStorage, doc_id: str) -> dict[str, Any] | None:
    """Return document-level geo metadata for search-hit and list fallback.

    Includes disk meta (``lat``/``lon`` and extra keys such as ``Souradnice``)
    plus fulltext so labeled original coordinates can be recovered when image
    payloads omit structured lat/lon.
    """
    if not doc_id:
        return None
    payload: dict[str, Any] = {}
    meta = storage.read_meta(doc_id)
    if meta is not None:
        payload.update(meta.to_dict())
    fulltext = storage.read_fulltext(doc_id)
    if fulltext and "fulltext" not in payload:
        payload["fulltext"] = fulltext
    return payload or None
