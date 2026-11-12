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

logger = get_logger("rag.document_patch")

ALLOWED_PATCH_FIELDS = frozenset({"filename", "extra"})
MAX_EXTRA_KEYS = 32
MAX_EXTRA_JSON_BYTES = 8192
_EXTRA_KEY_MAX_LEN = 64

# Payload keys that must not be overwritten via ``extra`` (identity, vectors,
# derived ingest fields, and geo_refined which has its own write path).
_RESERVED_EXTRA_KEYS = frozenset(ALLOWED_PATCH_FIELDS) | {
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


def _count_points(qdrant: QdrantClient, collection_name: str, doc_id: str) -> int:
    """Return the number of points in ``collection_name`` for ``doc_id``."""
    if not qdrant.collection_exists(collection_name):
        return 0
    return int(
        qdrant.count(
            collection_name=collection_name,
            count_filter=_doc_id_filter(doc_id),
            exact=True,
        ).count
    )


def _set_payload_for_doc(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    payload: Mapping[str, Any],
) -> int:
    """``set_payload`` on every point matching ``doc_id`` (no vector rewrite).

    Returns the number of matching points. Missing collections are a no-op.
    """
    if not payload:
        return 0
    if not qdrant.collection_exists(collection_name):
        return 0
    matched = _count_points(qdrant, collection_name, doc_id)
    if matched == 0:
        return 0
    qdrant.set_payload(
        collection_name=collection_name,
        payload=dict(payload),
        points=FilterSelector(filter=_doc_id_filter(doc_id)),
        wait=True,
    )
    return matched


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
