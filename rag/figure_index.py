"""Upsert, reindex, and verify ``pdf_figure`` points in Qdrant."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from rag.config import FIGURE_INDEX_VERSION, resolve_figure_batch_size
from rag.embeddings import get_ingest_embeddings
from rag.figure_summary import FigureLlmBudget
from rag.geo_refined import attach_preferred_geo, matches_geo_filter, validate_precision
from rag.image_embed import embed_image_paths, image_embedding_enabled
from rag.logging_setup import get_logger
from rag.payloads import (
    ImageRecordNotFoundError,
    ImageRecordPayloadError,
    apply_image_record_verification,
    stable_figure_point_id,
)
from rag.pdf_figures import BuiltFigure, iterate_figure_records
from rag.storage import RagStorage
from rag.vectorstore import (
    build_named_vector,
    document_lifecycle_filter,
    merge_query_filters,
)

logger = get_logger("rag.figure_index")


def _figure_filter(
    *,
    doc_id: str | None = None,
    verification_status: str | None = None,
    precision: str | None = None,
    needs_refinement: bool | None = None,
) -> Filter | None:
    """Build a Qdrant filter for pdf_figure records."""
    must: list[FieldCondition] = [
        FieldCondition(key="source_type", match=MatchValue(value="pdf_figure"))
    ]
    if doc_id:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
    if verification_status:
        must.append(
            FieldCondition(
                key="verification_status",
                match=MatchValue(value=verification_status),
            )
        )
    if precision:
        must.append(
            FieldCondition(
                key="geo_refined.precision",
                match=MatchValue(value=precision),
            )
        )
    if needs_refinement is not None:
        must.append(
            FieldCondition(
                key="geo_refined.needs_refinement",
                match=MatchValue(value=bool(needs_refinement)),
            )
        )
    return Filter(must=must)


def _load_existing_figure_payloads(
    client: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
) -> dict[str, dict[str, Any]]:
    """Scroll existing pdf_figure payloads for ``doc_id`` keyed by figure_id."""
    result: dict[str, dict[str, Any]] = {}
    scroll_filter = _figure_filter(doc_id=doc_id)
    next_offset: Any = None
    while True:
        batch, next_offset = client.scroll(
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
            payload = dict(point.payload or {})
            figure_id = str(payload.get("figure_id") or "")
            if figure_id:
                result[figure_id] = payload
        if next_offset is None:
            break
    return result


def _merge_preserved_geo(
    new_payload: dict[str, Any],
    existing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep original coordinates and ``geo_refined`` across figure reindex."""
    if not existing:
        return new_payload
    merged = dict(new_payload)
    for key in ("lat", "lon", "latitude", "longitude", "geom"):
        if key in existing and key not in merged:
            merged[key] = existing[key]
    if existing.get("geo_refined") is not None:
        merged["geo_refined"] = existing.get("geo_refined")
    if existing.get("geo_refined_history"):
        merged["geo_refined_history"] = existing.get("geo_refined_history")
    return merged


def _merge_preserved_verification(
    new_payload: dict[str, Any],
    existing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep human verified/rejected state and overrides from ``existing``."""
    if not existing:
        return new_payload
    merged = _merge_preserved_geo(new_payload, existing)
    status = str(existing.get("verification_status") or "")
    if status not in ("verified", "rejected"):
        return merged
    merged["verification_status"] = status
    merged["verified"] = status == "verified"
    verification = dict(existing.get("verification") or {})
    notes = verification.get("notes")
    if notes:
        verification = dict(verification)
        verification["notes"] = f"{notes}; stale"
    merged["verification"] = verification
    overrides = (verification.get("human_overrides") or {}) if verification else {}
    if overrides.get("caption"):
        merged["caption"] = overrides["caption"]
    if overrides.get("summary"):
        merged["summary"] = overrides["summary"]
    return merged


def upsert_figure_batch(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    built: Sequence[BuiltFigure],
    existing_by_figure: Mapping[str, Mapping[str, Any]] | None = None,
) -> int:
    """Embed and upsert a batch of built figures; return count upserted."""
    if not built:
        return 0
    existing_by_figure = existing_by_figure or {}
    batch_size = resolve_figure_batch_size()
    upserted = 0
    for start in range(0, len(built), batch_size):
        chunk = built[start : start + batch_size]
        texts = [str(item.payload.get("text") or "") for item in chunk]
        text_vectors = get_ingest_embeddings(texts)

        image_paths = []
        image_items: list[BuiltFigure] = []
        if image_embedding_enabled():
            import tempfile
            from pathlib import Path

            tmp_dir = Path(tempfile.mkdtemp(prefix="fig_clip_"))
            for item in chunk:
                path = tmp_dir / f"{item.payload['image_id']}.png"
                path.write_bytes(item.png_bytes)
                image_paths.append(path)
                image_items.append(item)
            try:
                clip_vectors = embed_image_paths(image_paths) if image_paths else []
            except Exception as exc:  # noqa: BLE001
                logger.warning("CLIP batch failed: %s", exc)
                clip_vectors = [[] for _ in image_paths]
        else:
            clip_vectors = [[] for _ in chunk]

        points: list[PointStruct] = []
        for item, text_vector, clip_vector in zip(
            chunk, text_vectors, clip_vectors, strict=True
        ):
            payload = _merge_preserved_verification(
                item.payload,
                existing_by_figure.get(item.figure_id),
            )
            point_id = stable_figure_point_id(doc_id, item.figure_id)
            image_vector = clip_vector if clip_vector else None
            points.append(
                PointStruct(
                    id=point_id,
                    vector=build_named_vector(
                        text_vector=text_vector,
                        image_vector=image_vector,
                    ),
                    payload=payload,
                )
            )
        qdrant.upsert(collection_name=collection_name, points=points)
        upserted += len(points)
    return upserted


def index_pdf_figures(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    doc_id: str,
    parent_document: str,
    doc_sha256: str,
    pdf_bytes: bytes,
    chunks: Sequence[Mapping[str, Any]],
    preserve_existing: bool = True,
) -> tuple[int, set[tuple[int, int]]]:
    """Run figure pipeline, store PNGs, upsert Qdrant; return count and skip keys."""
    existing = (
        _load_existing_figure_payloads(
            qdrant, collection_name=collection_name, doc_id=doc_id
        )
        if preserve_existing
        else {}
    )
    llm_budget = FigureLlmBudget()
    built: list[BuiltFigure] = []
    skip_keys: set[tuple[int, int]] = set()
    for figure in iterate_figure_records(
        pdf_bytes,
        doc_id=doc_id,
        parent_document=parent_document,
        doc_sha256=doc_sha256,
        chunks=chunks,
        llm_budget=llm_budget,
    ):
        image_id = str(figure.payload["image_id"])
        dest = storage.save_image_bytes(
            doc_id,
            image_id=image_id,
            extension=".png",
            data=figure.png_bytes,
        )
        figure.payload["file_path"] = storage.relative_to_root(dest)
        built.append(figure)
        skip_keys.update(figure.skip_pdf_image_keys)

    count = upsert_figure_batch(
        qdrant,
        collection_name=collection_name,
        doc_id=doc_id,
        built=built,
        existing_by_figure=existing,
    )
    if built:
        seen_ids = {item.figure_id for item in built}
        for figure_id in existing:
            if figure_id not in seen_ids:
                qdrant.delete(
                    collection_name=collection_name,
                    points_selector=[stable_figure_point_id(doc_id, figure_id)],
                    wait=True,
                )
    return count, skip_keys


def reindex_figures_for_doc(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    doc_id: str,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Re-run figure pipeline from stored original + chunks (no chunk re-embed)."""
    meta = storage.read_meta(doc_id)
    if meta is None:
        raise ImageRecordPayloadError(f"document not found: {doc_id}")
    if not force and getattr(meta, "figure_index_version", 0) >= FIGURE_INDEX_VERSION:
        return {
            "status": "skipped",
            "doc_id": doc_id,
            "reason": "figure_index_version current",
            "figures_reindexed": 0,
        }
    from pathlib import Path

    original = Path(meta.original_path)
    if not original.is_file():
        original = storage.resolve_under_root(meta.original_path)
    if not original.is_file():
        raise ImageRecordPayloadError(f"missing original for doc_id={doc_id}")
    chunks = storage.read_chunks(doc_id) or []
    pdf_bytes = original.read_bytes()
    if dry_run:
        total = sum(
            1
            for _ in iterate_figure_records(
                pdf_bytes,
                doc_id=doc_id,
                parent_document=meta.filename,
                doc_sha256=meta.sha256,
                chunks=chunks,
            )
        )
        return {"status": "dry_run", "doc_id": doc_id, "figures_reindexed": total}
    count, _skip = index_pdf_figures(
        storage=storage,
        qdrant=qdrant,
        collection_name=collection_name,
        doc_id=doc_id,
        parent_document=meta.filename,
        doc_sha256=meta.sha256,
        pdf_bytes=pdf_bytes,
        chunks=chunks,
        preserve_existing=True,
    )
    meta.figure_index_version = FIGURE_INDEX_VERSION
    meta.figures_count = count
    meta.figures_status = "done"
    storage.write_meta(meta)
    return {"status": "ok", "doc_id": doc_id, "figures_reindexed": count}


def verify_image_record(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    point_id: str | None = None,
    doc_id: str | None = None,
    figure_id: str | None = None,
    status: str,
    verified_by: str | None = None,
    notes: str | None = None,
    caption: str | None = None,
    summary: str | None = None,
    embed_fn: Any | None = None,
) -> dict[str, Any]:
    """Verify or reject a pdf_figure point and re-embed text when summary changes."""
    pid = (point_id or "").strip()
    if not pid and doc_id and figure_id:
        pid = stable_figure_point_id(doc_id, figure_id)
    if not pid:
        raise ImageRecordPayloadError("point_id or doc_id+figure_id required")
    found = qdrant.retrieve(
        collection_name=collection_name,
        ids=[pid],
        with_payload=True,
        with_vectors=True,
    )
    if not found:
        raise ImageRecordNotFoundError(f"figure point not found: {pid}")
    point = found[0]
    existing = dict(point.payload or {})
    if existing.get("source_type") != "pdf_figure":
        raise ImageRecordPayloadError("point is not pdf_figure")
    payload = apply_image_record_verification(
        existing,
        status=status,
        verified_by=verified_by,
        notes=notes,
        caption=caption,
        summary=summary,
    )
    text = str(payload.get("text") or "")
    if embed_fn is not None:
        text_vector = embed_fn(text)
    else:
        text_vector = get_ingest_embeddings([text])[0]
    vectors = dict(point.vector or {}) if isinstance(point.vector, dict) else {}
    image_vector = vectors.get("image_vector")
    qdrant.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=pid,
                vector=build_named_vector(
                    text_vector=text_vector,
                    image_vector=image_vector,
                ),
                payload=payload,
            )
        ],
        wait=True,
    )
    return {"status": "ok", "figure": payload}


def list_figure_payloads_for_doc(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
) -> list[dict[str, Any]]:
    """Return all ``pdf_figure`` payloads for ``doc_id`` (unbounded scroll)."""
    existing = _load_existing_figure_payloads(
        qdrant, collection_name=collection_name, doc_id=doc_id
    )
    records = list(existing.values())
    records.sort(
        key=lambda payload: (
            int(payload.get("page_number") or 0),
            str(payload.get("figure_id") or ""),
        )
    )
    return records


def list_image_records(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str | None = None,
    status: str | None = None,
    precision: str | None = None,
    needs_refinement: bool | None = None,
    limit: int = 50,
    include_superseded: bool = False,
) -> dict[str, Any]:
    """List pdf_figure payloads with optional filters.

    Superseded and retracted documents are omitted unless
    ``include_superseded`` is true. Missing ``doc_status`` counts as active.
    """
    precision_n = None
    if precision:
        try:
            precision_n = validate_precision(precision)
        except Exception as exc:  # noqa: BLE001
            raise ImageRecordPayloadError(str(exc)) from exc
    scroll_filter = merge_query_filters(
        _figure_filter(
            doc_id=doc_id,
            verification_status=status or None,
        ),
        document_lifecycle_filter(include_superseded=include_superseded),
    )
    fetch_limit = min(max(limit, 1), 200)
    if precision_n is not None or needs_refinement is not None:
        fetch_limit = 200
    batch, _ = qdrant.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=fetch_limit,
        with_payload=True,
        with_vectors=False,
    )
    records = []
    geo_filter = precision_n is not None or needs_refinement is not None
    for point in batch:
        payload = dict(point.payload or {})
        if geo_filter and not matches_geo_filter(
            payload,
            precision=precision_n,
            needs_refinement=needs_refinement,
        ):
            continue
        records.append(attach_preferred_geo(payload))
        if len(records) >= min(max(limit, 1), 200):
            break
    return {"records": records, "returned": len(records), "limit": limit}


def image_record_stats(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
) -> dict[str, Any]:
    """Aggregate pdf_figure counts by verification_status for one document."""
    scroll_filter = _figure_filter(doc_id=doc_id)
    by_status: dict[str, int] = {}
    total = 0
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
            payload = point.payload or {}
            status = str(payload.get("verification_status") or "unverified")
            by_status[status] = by_status.get(status, 0) + 1
            total += 1
        if next_offset is None:
            break
    return {"doc_id": doc_id, "total": total, "by_status": by_status}


def knowledge_figure_verification_filter(
    *,
    only_verified: bool = False,
    min_status: str = "",
) -> Filter | None:
    """Build optional verification filter for knowledge / image search."""
    if only_verified:
        return Filter(
            must=[
                FieldCondition(
                    key="verification_status",
                    match=MatchValue(value="verified"),
                )
            ]
        )
    min_status = (min_status or "").strip()
    if min_status == "auto_checked":
        return Filter(
            should=[
                FieldCondition(
                    key="verification_status",
                    match=MatchValue(value="auto_checked"),
                ),
                FieldCondition(
                    key="verification_status",
                    match=MatchValue(value="verified"),
                ),
            ]
        )
    return None


def merge_knowledge_filters(*filters: Filter | None) -> Filter | None:
    """Combine filters by AND-ing their ``must`` and ``must_not`` clauses."""
    return merge_query_filters(*filters)
