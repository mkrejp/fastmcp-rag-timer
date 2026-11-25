"""Restrict image binary serving without deleting files or Qdrant points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from rag.accounting import utc_now_iso
from rag.doc_locks import doc_lock
from rag.document_delete import DocumentNotFoundError
from rag.logging_setup import get_logger
from rag.provenance_meta import (
    HIDDEN_LICENSE_STATUSES,
    LICENSE_STATUS_VALUES,
    PROVENANCE_KEYS,
    document_provenance_from_extra,
    normalize_provenance_metadata,
)
from rag.page_ids import parse_page_image_id
from rag.storage import DocumentMeta, ImageMeta, RagStorage

logger = get_logger("rag.image_visibility")

IMAGE_BINARY_DENIED_DETAIL = "Image binary access denied (license_status)"
VISIBILITY_EXTRA_KEYS = frozenset(
    {"visibility_reason", "decided_by", "visibility_updated_at"}
)
RESTRICT_DEFAULT_STATUS = "restricted"
UNRESTRICT_DEFAULT_STATUS = "published"


class ImageVisibilityError(ValueError):
    """Raised when a visibility patch is invalid or the image is unknown."""


class ImageNotFoundError(ImageVisibilityError):
    """Raised when ``image_id`` cannot be resolved for ``doc_id``."""


def image_provenance_record(
    *,
    image_row: Mapping[str, Any] | None,
    document_extra: Mapping[str, Any] | None,
    qdrant_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge document-, catalog-image-, and Qdrant-level provenance (image wins)."""
    doc = document_provenance_from_extra(document_extra)
    img = document_provenance_from_extra(image_row)
    qdrant = document_provenance_from_extra(qdrant_payload)
    merged: dict[str, Any] = {**doc, **img, **qdrant}
    for key in ("visibility_reason", "decided_by", "visibility_updated_at"):
        for source in (image_row, qdrant_payload):
            if source and key in source:
                merged[key] = source[key]
    return merged


def image_binary_serving_allowed(record: Mapping[str, Any]) -> bool:
    """Return whether authenticated HTTP/MCP may return image bytes."""
    status = str(record.get("license_status") or "").strip()
    return status not in HIDDEN_LICENSE_STATUSES


def audit_log_path(storage: RagStorage) -> Path:
    """Return the append-only audit path under the RAG data root."""
    dest = storage.root / "audit" / "image_visibility.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def append_visibility_audit(
    storage: RagStorage,
    entry: Mapping[str, Any],
    *,
    dry_run: bool,
) -> None:
    """Append one audit line (no-op when ``dry_run``)."""
    if dry_run:
        return
    path = audit_log_path(storage)
    line = json.dumps(dict(entry), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _image_id_filter(doc_id: str, image_id: str) -> Filter:
    """Qdrant filter for one document image or figure."""
    return Filter(
        must=[
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            FieldCondition(key="image_id", match=MatchValue(value=image_id)),
        ]
    )


def _scroll_image_payloads(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    image_id: str,
) -> list[dict[str, Any]]:
    """Return pdf_image / pdf_figure payloads matching ``image_id``."""
    if not qdrant.collection_exists(collection_name):
        return []
    filt = _image_id_filter(doc_id, image_id)
    payloads: list[dict[str, Any]] = []
    next_offset: Any = None
    while True:
        batch, next_offset = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=filt,
            limit=32,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        for point in batch:
            payload = dict(point.payload or {})
            source_type = str(payload.get("source_type") or "")
            if source_type in ("pdf_image", "pdf_figure"):
                payloads.append(payload)
        if next_offset is None:
            break
    return payloads


def resolve_image_provenance(
    storage: RagStorage,
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    image_id: str,
) -> tuple[dict[str, Any], ImageMeta | None, list[dict[str, Any]]]:
    """Load merged provenance plus catalog row and Qdrant payloads."""
    meta = storage.read_meta(doc_id)
    if meta is None:
        raise DocumentNotFoundError(doc_id)
    parsed_page = parse_page_image_id(image_id)
    if parsed_page is not None:
        pid, _page = parsed_page
        if pid != doc_id:
            raise ImageNotFoundError("doc_id does not match page image id")
        record = image_provenance_record(
            image_row=None,
            document_extra=meta.extra,
        )
        return record, None, []

    storage.validate_image_id(image_id)
    image_meta = storage.find_image(doc_id, image_id)
    image_row = image_meta.to_dict() if image_meta is not None else None
    qdrant_payloads = _scroll_image_payloads(
        qdrant,
        collection_name=collection_name,
        doc_id=doc_id,
        image_id=image_id,
    )
    on_disk = storage.locate_image_file(doc_id, image_id) is not None
    if image_meta is None and not qdrant_payloads and not on_disk:
        raise ImageNotFoundError(f"Unknown image id: {image_id}")
    qdrant_primary = qdrant_payloads[0] if qdrant_payloads else None
    record = image_provenance_record(
        image_row=image_row,
        document_extra=meta.extra,
        qdrant_payload=qdrant_primary,
    )
    return record, image_meta, qdrant_payloads


def _validate_visibility_body(body: Mapping[str, Any], *, restrict: bool) -> dict[str, Any]:
    """Normalize and validate a single-image visibility patch body."""
    reason = body.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ImageVisibilityError("reason must be a string")
    decided_by = body.get("decided_by")
    if decided_by is not None and not isinstance(decided_by, str):
        raise ImageVisibilityError("decided_by must be a string")
    if restrict and not (str(decided_by or "").strip()):
        raise ImageVisibilityError("decided_by is required when restricting")
    if restrict and not (str(reason or "").strip()):
        raise ImageVisibilityError("reason is required when restricting")

    status_raw = body.get("license_status")
    if status_raw is not None:
        status = str(status_raw).strip()
        if status not in LICENSE_STATUS_VALUES:
            choices = ", ".join(sorted(LICENSE_STATUS_VALUES))
            raise ImageVisibilityError(f"license_status must be one of: {choices}")
    elif restrict:
        status = RESTRICT_DEFAULT_STATUS
    else:
        status = UNRESTRICT_DEFAULT_STATUS

    if restrict and status not in HIDDEN_LICENSE_STATUSES:
        hidden = ", ".join(sorted(HIDDEN_LICENSE_STATUSES))
        raise ImageVisibilityError(
            f"restrict requires license_status in {{{hidden}}}"
        )
    if not restrict and status in HIDDEN_LICENSE_STATUSES:
        raise ImageVisibilityError(
            "unrestrict must set a non-hidden license_status (e.g. published)"
        )

    now = utc_now_iso()
    patch: dict[str, Any] = {
        "license_status": status,
        "visibility_updated_at": now,
    }
    if reason is not None:
        patch["visibility_reason"] = str(reason).strip()
    if decided_by is not None:
        patch["decided_by"] = str(decided_by).strip()
    return patch


def _apply_catalog_image_patch(
    meta: DocumentMeta,
    image_id: str,
    patch: Mapping[str, Any],
) -> bool:
    """Merge ``patch`` onto the matching ``ImageMeta`` row."""
    for index, image in enumerate(meta.images):
        if image.image_id != image_id:
            continue
        row = image.to_dict()
        for key, value in patch.items():
            if key in PROVENANCE_KEYS or key in VISIBILITY_EXTRA_KEYS:
                row[key] = value
        meta.images[index] = ImageMeta.from_dict(row)
        return True
    return False


def _set_payload_image_points(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    image_id: str,
    payload: Mapping[str, Any],
) -> int:
    """``set_payload`` on pdf_image / pdf_figure points for one ``image_id``."""
    if not payload:
        return 0
    if not qdrant.collection_exists(collection_name):
        return 0
    filt = _image_id_filter(doc_id, image_id)
    count = int(
        qdrant.count(
            collection_name=collection_name,
            count_filter=filt,
            exact=True,
        ).count
    )
    if count == 0:
        return 0
    qdrant.set_payload(
        collection_name=collection_name,
        payload=dict(payload),
        points=FilterSelector(filter=filt),
    )
    return count


def _apply_one_image_visibility(
    storage: RagStorage,
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    image_id: str,
    patch: Mapping[str, Any],
    restrict: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Restrict or un-restrict one image (catalog + Qdrant, no deletes)."""
    record, image_meta, qdrant_payloads = resolve_image_provenance(
        storage,
        qdrant,
        collection_name=collection_name,
        doc_id=doc_id,
        image_id=image_id,
    )
    _ = record  # used for pre-check logging only
    meta = storage.read_meta(doc_id)
    assert meta is not None

    catalog_updated = False
    if image_meta is not None:
        catalog_updated = _apply_catalog_image_patch(meta, image_id, patch)
        if catalog_updated and not dry_run:
            storage.write_meta(meta)

    prov_patch, _trunc = normalize_provenance_metadata(
        {k: patch[k] for k in patch if k in PROVENANCE_KEYS}
    )
    qdrant_patch = {**prov_patch}
    for key in VISIBILITY_EXTRA_KEYS:
        if key in patch:
            qdrant_patch[key] = patch[key]

    points_updated = 0
    if qdrant_patch and not dry_run:
        points_updated = _set_payload_image_points(
            qdrant,
            collection_name=collection_name,
            doc_id=doc_id,
            image_id=image_id,
            payload=qdrant_patch,
        )

    action = "restrict" if restrict else "unrestrict"
    audit = {
        "ts": patch.get("visibility_updated_at") or utc_now_iso(),
        "doc_id": doc_id,
        "image_id": image_id,
        "action": action,
        "license_status": patch.get("license_status"),
        "reason": patch.get("visibility_reason"),
        "decided_by": patch.get("decided_by"),
        "dry_run": dry_run,
        "catalog_updated": catalog_updated,
        "qdrant_points_updated": points_updated,
        "had_qdrant_payload": bool(qdrant_payloads),
    }
    append_visibility_audit(storage, audit, dry_run=dry_run)
    logger.info(
        "image_visibility %s doc_id=%s image_id=%s status=%s dry_run=%s points=%s",
        action,
        doc_id,
        image_id,
        patch.get("license_status"),
        dry_run,
        points_updated,
    )
    merged = image_provenance_record(
        image_row=image_meta.to_dict() if image_meta else None,
        document_extra=meta.extra,
        qdrant_payload={**qdrant_payloads[0], **qdrant_patch} if qdrant_payloads else qdrant_patch,
    )
    return {
        "doc_id": doc_id,
        "image_id": image_id,
        "restricted": restrict,
        "license_status": merged.get("license_status"),
        "catalog_updated": catalog_updated,
        "qdrant_points_updated": points_updated,
        "audit": audit,
        "provenance": document_provenance_from_extra(merged),
    }


def patch_image_visibility(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    doc_id: str,
    image_id: str,
    body: Mapping[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply visibility change to one image under ``doc_lock``."""
    if "restricted" not in body:
        raise ImageVisibilityError("restricted (boolean) is required")
    restrict = body["restricted"]
    if not isinstance(restrict, bool):
        raise ImageVisibilityError("restricted must be a boolean")
    patch = _validate_visibility_body(body, restrict=restrict)
    with doc_lock(doc_id):
        try:
            return _apply_one_image_visibility(
                storage,
                qdrant,
                collection_name=collection_name,
                doc_id=doc_id,
                image_id=image_id,
                patch=patch,
                restrict=restrict,
                dry_run=dry_run,
            )
        except DocumentNotFoundError as exc:
            raise DocumentNotFoundError(exc.doc_id) from exc


def patch_images_visibility_batch(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    doc_id: str,
    image_ids: Sequence[str],
    body: Mapping[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply the same visibility patch to many ``image_ids`` of one document."""
    if not image_ids:
        raise ImageVisibilityError("image_ids must be a non-empty list")
    if "restricted" not in body:
        raise ImageVisibilityError("restricted (boolean) is required")
    restrict = body["restricted"]
    if not isinstance(restrict, bool):
        raise ImageVisibilityError("restricted must be a boolean")
    patch = _validate_visibility_body(body, restrict=restrict)
    results: list[dict[str, Any]] = []
    with doc_lock(doc_id):
        meta = storage.read_meta(doc_id)
        if meta is None:
            raise DocumentNotFoundError(doc_id)
        for image_id in image_ids:
            image_id_text = str(image_id).strip()
            storage.validate_image_id(image_id_text)
            results.append(
                _apply_one_image_visibility(
                    storage,
                    qdrant,
                    collection_name=collection_name,
                    doc_id=doc_id,
                    image_id=image_id_text,
                    patch=patch,
                    restrict=restrict,
                    dry_run=dry_run,
                )
            )
    return {
        "status": "updated",
        "doc_id": doc_id,
        "dry_run": dry_run,
        "count": len(results),
        "images": results,
    }