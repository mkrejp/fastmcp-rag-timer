"""Update stored document metadata without re-embedding or deleting points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from rag.config import LOCALITY_LINKS_COLLECTION
from rag.document_delete import DocumentNotFoundError
from rag.logging_setup import get_logger
from rag.processing import guess_mime, is_supported_filename
from rag.storage import DocumentMeta, RagStorage
from rag.verification_meta import (
    FLAT_PAYLOAD_KEYS,
    DocumentMetadataError,
    chunk_payload_fields,
    figure_payload_fields,
    normalize_verification_metadata,
    split_verification_deletes,
)

logger = get_logger("rag.document_patch")

ALLOWED_PATCH_FIELDS = frozenset({"filename", "extra"})
MAX_EXTRA_KEYS = 32
MAX_EXTRA_JSON_BYTES = 8192
_EXTRA_KEY_MAX_LEN = 64

# Payload keys that must not be overwritten via ``extra`` (identity, vectors,
# derived ingest fields, and geo_refined which has its own write path).
_RESERVED_EXTRA_KEYS = (
    frozenset(ALLOWED_PATCH_FIELDS)
    | {
        "bbox",
        "caption",
        "cast_obce",
        "chunk_count",
        "chunk_index",
        "chunk_sha256",
        "chunks_path",
        "content_summary",
        "doc_id",
        "file_path",
        "figure_id",
        "figure_index_version",
        "figures_count",
        "figures_status",
        "fulltext_path",
        "geo_refined",
        "geo_refined_history",
        "id",
        "image_id",
        "image_sha256",
        "images",
        "lat",
        "localities",
        "lon",
        "mime",
        "original_path",
        "page_number",
        "parent_document",
        "sha256",
        "source",
        "source_type",
        "source_url",
        "status",
        "text",
        "uploaded_at",
        "vectors",
    }
    | FLAT_PAYLOAD_KEYS
    | frozenset({"source_axis", "overeni_souhrn"})
)


class DocumentPatchError(ValueError):
    """Raised when a metadata patch is empty, unknown, or otherwise invalid."""


def _doc_id_filter(doc_id: str) -> Filter:
    """Return a Qdrant payload filter matching ``doc_id``."""
    return Filter(
        must=[
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=doc_id),
            )
        ]
    )


def _count_points(
    qdrant: QdrantClient,
    collection_name: str,
    doc_id: str,
    payload_filter: Filter | None = None,
) -> int:
    """Return the number of points in ``collection_name`` for ``doc_id``."""
    if not qdrant.collection_exists(collection_name):
        return 0
    return int(
        qdrant.count(
            collection_name=collection_name,
            count_filter=payload_filter or _doc_id_filter(doc_id),
            exact=True,
        ).count
    )


def _not_figure_filter(doc_id: str) -> Filter:
    """Match knowledge points for ``doc_id`` that are text chunks."""
    return Filter(
        must=[
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=doc_id),
            )
        ],
        must_not=[
            FieldCondition(
                key="source_type",
                match=MatchValue(value="pdf_figure"),
            ),
            FieldCondition(
                key="source_type",
                match=MatchValue(value="pdf_image"),
            ),
        ],
    )


def _source_type_filter(doc_id: str, source_type: str) -> Filter:
    """Match knowledge points for ``doc_id`` with ``source_type``."""
    return Filter(
        must=[
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=doc_id),
            ),
            FieldCondition(
                key="source_type",
                match=MatchValue(value=source_type),
            ),
        ]
    )


def _prepare_payload_selector(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    payload_filter: Filter | None,
) -> tuple[int, Filter] | None:
    """Return ``(matched_count, filter)`` or ``None`` when there is nothing to write."""
    if not qdrant.collection_exists(collection_name):
        return None
    filt = payload_filter or _doc_id_filter(doc_id)
    matched = _count_points(qdrant, collection_name, doc_id, payload_filter=filt)
    if matched == 0:
        return None
    return matched, filt


def _set_payload_for_doc(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    payload: Mapping[str, Any],
    payload_filter: Filter | None = None,
) -> int:
    """``set_payload`` on every point matching ``doc_id`` (no vector rewrite).

    Returns the number of matching points. Missing collections are a no-op.
    """
    if not payload:
        return 0
    prepared = _prepare_payload_selector(
        qdrant,
        collection_name=collection_name,
        doc_id=doc_id,
        payload_filter=payload_filter,
    )
    if prepared is None:
        return 0
    matched, filt = prepared
    qdrant.set_payload(
        collection_name=collection_name,
        payload=dict(payload),
        points=FilterSelector(filter=filt),
        wait=True,
    )
    return matched


def _delete_payload_for_doc(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    keys: list[str],
    payload_filter: Filter | None = None,
) -> int:
    """``delete_payload`` on matching points (no vector rewrite, no point deletes)."""
    if not keys:
        return 0
    prepared = _prepare_payload_selector(
        qdrant,
        collection_name=collection_name,
        doc_id=doc_id,
        payload_filter=payload_filter,
    )
    if prepared is None:
        return 0
    matched, filt = prepared
    qdrant.delete_payload(
        collection_name=collection_name,
        keys=list(keys),
        points=FilterSelector(filter=filt),
        wait=True,
    )
    return matched


def write_verification_payload(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    fields: Mapping[str, Any],
) -> tuple[int, int]:
    """Write verification fields onto chunks/images and figure-safe keys.

    Does not change ``doc_id``, rewrite vectors, or delete points.
    Returns ``(chunk_points_updated, figure_points_updated)``.
    """
    chunk_fields = chunk_payload_fields(fields)
    figure_fields = figure_payload_fields(fields)
    points_updated = _set_payload_for_doc(
        qdrant,
        collection_name=collection_name,
        doc_id=doc_id,
        payload=chunk_fields,
        payload_filter=_not_figure_filter(doc_id),
    )
    figures_updated = 0
    for source_type in ("pdf_figure", "pdf_image"):
        updated = _set_payload_for_doc(
            qdrant,
            collection_name=collection_name,
            doc_id=doc_id,
            payload=figure_fields,
            payload_filter=_source_type_filter(doc_id, source_type),
        )
        if source_type == "pdf_figure":
            figures_updated = updated
    return points_updated, figures_updated


def _jsonable(value: object, *, path: str) -> None:
    """Reject values that cannot round-trip through JSON."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _jsonable(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DocumentPatchError(f"{path} keys must be strings")
            _jsonable(item, path=f"{path}.{key}")
        return
    raise DocumentPatchError(f"{path} must be JSON-serializable")


def validate_extra_patch(extra: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``extra`` after key/value validation.

    ``None`` values mean "remove this extra key". Other values must be
    JSON-serializable. Reserved catalog and Qdrant payload names are rejected.
    """
    if not isinstance(extra, Mapping):
        raise DocumentPatchError("extra must be an object")
    if len(extra) > MAX_EXTRA_KEYS:
        raise DocumentPatchError(f"extra may contain at most {MAX_EXTRA_KEYS} keys")
    cleaned: dict[str, Any] = {}
    for raw_key, value in extra.items():
        key = str(raw_key).strip()
        if not key or len(key) > _EXTRA_KEY_MAX_LEN:
            raise DocumentPatchError(
                f"extra key must be 1–{_EXTRA_KEY_MAX_LEN} characters"
            )
        if not key.isidentifier() or key.startswith("__"):
            raise DocumentPatchError(
                f"extra key {key!r} must be a Python identifier without dunder prefix"
            )
        if key in _RESERVED_EXTRA_KEYS:
            raise DocumentPatchError(f"extra key {key!r} is reserved")
        if value is not None:
            _jsonable(value, path=f"extra.{key}")
        cleaned[key] = value
    encoded = json.dumps(cleaned, ensure_ascii=False, default=str)
    if len(encoded.encode("utf-8")) > MAX_EXTRA_JSON_BYTES:
        raise DocumentPatchError(
            f"extra JSON must be at most {MAX_EXTRA_JSON_BYTES} bytes"
        )
    return cleaned


def validate_filename_patch(filename: str, *, current: str) -> str:
    """Return a sanitized filename that keeps the existing document type."""
    text = filename.strip()
    if not text:
        raise DocumentPatchError("filename must be a non-empty string")
    if Path(text).name != text:
        raise DocumentPatchError("filename must not contain a path")
    if not is_supported_filename(text):
        raise DocumentPatchError(
            "Only PDF, TXT, Markdown, JSON, and GeoJSON files are supported."
        )
    current_suffix = Path(current).suffix.lower()
    new_suffix = Path(text).suffix.lower()
    if current_suffix and new_suffix != current_suffix:
        raise DocumentPatchError(
            f"filename suffix must remain {current_suffix} (no re-parse)"
        )
    return text


def _merge_extra(
    existing: Mapping[str, Any], patch: Mapping[str, Any]
) -> dict[str, Any]:
    """Return ``existing`` with ``patch`` applied; ``None`` values delete keys."""
    merged = dict(existing)
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def _load_document(storage: RagStorage, doc_id: str) -> DocumentMeta:
    """Return stored metadata or raise ``DocumentNotFoundError``."""
    try:
        storage.validate_doc_id(doc_id)
    except ValueError as exc:
        raise DocumentNotFoundError(doc_id) from exc
    meta = storage.read_meta(doc_id)
    if meta is None:
        raise DocumentNotFoundError(doc_id)
    return meta


def patch_document(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    doc_id: str,
    filename: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Patch catalog metadata and matching Qdrant payloads in place.

    Does not re-embed, upsert vectors, or delete points. Unknown or reserved
    fields raise ``DocumentPatchError``. Missing ids raise
    ``DocumentNotFoundError``.
    """
    if filename is None and extra is None:
        raise DocumentPatchError(
            "at least one of filename or extra is required; "
            f"allowed fields: {', '.join(sorted(ALLOWED_PATCH_FIELDS))}"
        )

    meta = _load_document(storage, doc_id)
    updated_fields: list[str] = []
    payload_patch: dict[str, Any] = {}

    if filename is not None:
        new_name = validate_filename_patch(filename, current=meta.filename)
        if new_name != meta.filename:
            meta.filename = new_name
            meta.mime = guess_mime(new_name)
            payload_patch["parent_document"] = new_name
            payload_patch["source"] = new_name
            updated_fields.append("filename")

    if extra is not None:
        extra_patch = validate_extra_patch(extra)
        merged = _merge_extra(meta.extra, extra_patch)
        if merged != meta.extra:
            meta.extra = merged
            payload_patch["extra"] = dict(merged)
            updated_fields.append("extra")

    points_updated = 0
    locality_links_updated = 0
    if payload_patch:
        storage.write_meta(meta)
        knowledge_payload = dict(payload_patch)
        points_updated = _set_payload_for_doc(
            qdrant,
            collection_name=collection_name,
            doc_id=doc_id,
            payload=knowledge_payload,
        )
        link_payload = {
            key: payload_patch[key]
            for key in ("parent_document",)
            if key in payload_patch
        }
        if link_payload:
            locality_links_updated = _set_payload_for_doc(
                qdrant,
                collection_name=LOCALITY_LINKS_COLLECTION,
                doc_id=doc_id,
                payload=link_payload,
            )
        logger.info(
            "Patched document doc_id=%s filename=%s fields=%s points_updated=%s",
            doc_id,
            meta.filename,
            ",".join(updated_fields) or "-",
            points_updated,
        )
    else:
        logger.info("Document patch was a no-op doc_id=%s", doc_id)

    return {
        "status": "updated",
        "doc_id": doc_id,
        "filename": meta.filename,
        "updated_fields": updated_fields,
        "points_updated": points_updated,
        "locality_links_updated": locality_links_updated,
        "meta": meta.to_dict(),
    }


def patch_document_metadata(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    doc_id: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Patch verification fields onto catalog extra and Qdrant payloads.

    Non-null fields use ``set_payload``. JSON ``null`` on a whitelist key
    uses ``delete_payload`` (chunks and figures) and pops catalog extra.
    Does not change ``doc_id``, re-embed, or delete points. Empty bodies
    apply ``reliability=1``.
    """
    meta = _load_document(storage, doc_id)
    remaining, deleted = split_verification_deletes(metadata)
    apply_defaults = not metadata
    if remaining:
        fields, truncated = normalize_verification_metadata(
            remaining, apply_defaults=False
        )
    elif apply_defaults:
        fields, truncated = normalize_verification_metadata(None, apply_defaults=True)
    elif deleted:
        fields, truncated = {}, []
    else:
        raise DocumentMetadataError("at least one metadata field is required")
    if "doc_id" in fields:
        raise DocumentMetadataError("doc_id cannot be changed")

    merged_extra = dict(meta.extra)
    merged_extra.update(fields)
    for key in deleted:
        merged_extra.pop(key, None)
    meta.extra = merged_extra
    storage.write_meta(meta)

    points_updated = 0
    figures_updated = 0
    if fields:
        points_updated, figures_updated = write_verification_payload(
            qdrant,
            collection_name=collection_name,
            doc_id=doc_id,
            fields=fields,
        )
    if deleted:
        _delete_payload_for_doc(
            qdrant,
            collection_name=collection_name,
            doc_id=doc_id,
            keys=deleted,
        )
        if not fields:
            points_updated = _count_points(
                qdrant,
                collection_name,
                doc_id,
                payload_filter=_not_figure_filter(doc_id),
            )
            figures_updated = _count_points(
                qdrant,
                collection_name,
                doc_id,
                payload_filter=_source_type_filter(doc_id, "pdf_figure"),
            )

    logger.info(
        "Patched document metadata doc_id=%s fields=%s deleted=%s "
        "points_updated=%s figures_updated=%s",
        doc_id,
        ",".join(sorted(fields)) or "-",
        ",".join(deleted) or "-",
        points_updated,
        figures_updated,
    )
    return {
        "status": "updated",
        "doc_id": doc_id,
        "filename": meta.filename,
        "updated_fields": sorted(fields),
        "deleted_fields": sorted(deleted),
        "points_updated": points_updated,
        "figures_updated": figures_updated,
        "truncated": truncated,
        "meta": meta.to_dict(),
    }
