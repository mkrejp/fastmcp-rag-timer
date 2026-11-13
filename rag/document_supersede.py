"""Atomic document supersession links (both sides, no silent peer overwrites)."""

from __future__ import annotations

from typing import Any, Mapping

from qdrant_client import QdrantClient

from rag.document_delete import DocumentNotFoundError
from rag.document_patch import (
    delete_payload_keys_for_doc,
    write_verification_payload,
)
from rag.logging_setup import get_logger
from rag.storage import DocumentMeta, RagStorage
from rag.verification_meta import (
    UNSUPERSEDE_ENDPOINT,
    DocumentMetadataError,
    normalize_verification_metadata,
)

logger = get_logger("rag.document_supersede")

_MAX_CHAIN_LEN = 64


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


def _as_supersedes_list(value: object) -> list[str]:
    """Return a de-duplicated list of doc ids from ``supersedes`` storage."""
    if value is None or value == "":
        raw: list[object] = []
    elif isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = list(value)
    else:
        raise DocumentMetadataError("supersedes must be a list of doc_ids")
    out: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _superseded_by_chain(storage: RagStorage, start_id: str) -> list[str]:
    """Follow ``superseded_by`` from ``start_id`` until the chain ends."""
    chain: list[str] = []
    current = start_id.strip()
    seen: set[str] = set()
    for _ in range(_MAX_CHAIN_LEN):
        if not current or current in seen:
            break
        seen.add(current)
        chain.append(current)
        meta = storage.read_meta(current)
        if meta is None:
            break
        current = str(meta.extra.get("superseded_by") or "").strip()
    return chain


def _apply_doc_fields(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    meta: DocumentMeta,
    set_fields: Mapping[str, Any],
    delete_keys: list[str],
) -> tuple[int, int]:
    """Write catalog extra and matching Qdrant payloads for one document."""
    extra = dict(meta.extra)
    extra.update(dict(set_fields))
    for key in delete_keys:
        extra.pop(key, None)
    meta.extra = extra
    storage.write_meta(meta)
    points_updated = 0
    figures_updated = 0
    if set_fields:
        points_updated, figures_updated = write_verification_payload(
            qdrant,
            collection_name=collection_name,
            doc_id=meta.doc_id,
            fields=set_fields,
        )
    if delete_keys:
        delete_payload_keys_for_doc(
            qdrant,
            collection_name=collection_name,
            doc_id=meta.doc_id,
            keys=delete_keys,
        )
    return points_updated, figures_updated


def _normalize_new_doc_id(storage: RagStorage, new_doc_id: object) -> str:
    """Return a validated replacement ``doc_id`` or raise ``DocumentMetadataError``."""
    if new_doc_id is None:
        raise DocumentMetadataError("new_doc_id is required")
    text = str(new_doc_id).strip()
    if not text:
        raise DocumentMetadataError("new_doc_id must be a non-empty document id")
    try:
        storage.validate_doc_id(text)
    except ValueError as exc:
        raise DocumentMetadataError(
            f"new_doc_id is not a valid document id: {text!r}"
        ) from exc
    return text


def supersede_document(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    old_doc_id: str,
    new_doc_id: object,
    correction_note: object | None = None,
) -> dict[str, Any]:
    """Link ``old_doc_id`` as superseded by ``new_doc_id`` on both documents.

    Sets ``old.superseded_by``, ``old.doc_status=superseded``, appends old to
    ``new.supersedes`` (list, no duplicates), and optionally stores
    ``correction_note`` on the old document. Does not change ``new.doc_status``,
    re-embed, or delete points. Idempotent for the same pair.
    """
    old_meta = _load_document(storage, old_doc_id)
    replacement_id = _normalize_new_doc_id(storage, new_doc_id)
    if replacement_id == old_doc_id:
        raise DocumentMetadataError("a document may not supersede itself")
    try:
        new_meta = _load_document(storage, replacement_id)
    except DocumentNotFoundError as exc:
        raise DocumentNotFoundError(replacement_id) from exc

    existing = str(old_meta.extra.get("superseded_by") or "").strip()
    if existing and existing != replacement_id:
        raise DocumentMetadataError(
            f"document {old_doc_id} is already superseded by {existing}; "
            f"use {UNSUPERSEDE_ENDPOINT} first"
        )
    chain = _superseded_by_chain(storage, replacement_id)
    if old_doc_id in chain:
        raise DocumentMetadataError(
            f"supersession cycle: {old_doc_id} already appears in the "
            f"superseded_by chain of {replacement_id}"
        )

    truncated: list[str] = []
    old_fields: dict[str, Any] = {
        "superseded_by": replacement_id,
        "doc_status": "superseded",
    }
    note_text = None if correction_note is None else str(correction_note)
    if note_text is not None and note_text.strip():
        note_fields, truncated = normalize_verification_metadata(
            {"correction_note": note_text},
            apply_defaults=False,
        )
        old_fields.update(note_fields)

    new_supersedes = _as_supersedes_list(new_meta.extra.get("supersedes"))
    if old_doc_id not in new_supersedes:
        new_supersedes.append(old_doc_id)

    old_points, old_figures = _apply_doc_fields(
        storage=storage,
        qdrant=qdrant,
        collection_name=collection_name,
        meta=old_meta,
        set_fields=old_fields,
        delete_keys=[],
    )
    new_points, new_figures = _apply_doc_fields(
        storage=storage,
        qdrant=qdrant,
        collection_name=collection_name,
        meta=new_meta,
        set_fields={"supersedes": new_supersedes},
        delete_keys=[],
    )
    logger.info(
        "Superseded doc_id=%s by new_doc_id=%s old_points=%s new_points=%s",
        old_doc_id,
        replacement_id,
        old_points,
        new_points,
    )
    return {
        "status": "updated",
        "old_doc_id": old_doc_id,
        "new_doc_id": replacement_id,
        "doc_status": "superseded",
        "superseded_by": replacement_id,
        "supersedes": list(new_supersedes),
        "correction_note": old_meta.extra.get("correction_note"),
        "truncated": truncated,
        "old_points_updated": old_points,
        "old_figures_updated": old_figures,
        "new_points_updated": new_points,
        "new_figures_updated": new_figures,
        "old": old_meta.to_dict(),
        "new": new_meta.to_dict(),
    }


def unsupersede_document(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    old_doc_id: str,
) -> dict[str, Any]:
    """Remove the supersession link from ``old_doc_id`` and its replacement.

    Clears ``superseded_by`` and ``doc_status`` on the old document and removes
    it from the replacement's ``supersedes`` list. Leaves ``correction_note``,
    verification fields, and vectors untouched. Idempotent when no link exists.
    """
    old_meta = _load_document(storage, old_doc_id)
    replacement_id = str(old_meta.extra.get("superseded_by") or "").strip()
    new_meta: DocumentMeta | None = None
    new_points = 0
    new_figures = 0
    remaining: list[str] = []
    if replacement_id:
        try:
            new_meta = _load_document(storage, replacement_id)
        except DocumentNotFoundError:
            new_meta = None
        if new_meta is not None:
            remaining = [
                item
                for item in _as_supersedes_list(new_meta.extra.get("supersedes"))
                if item != old_doc_id
            ]
            if remaining:
                new_points, new_figures = _apply_doc_fields(
                    storage=storage,
                    qdrant=qdrant,
                    collection_name=collection_name,
                    meta=new_meta,
                    set_fields={"supersedes": remaining},
                    delete_keys=[],
                )
            else:
                new_points, new_figures = _apply_doc_fields(
                    storage=storage,
                    qdrant=qdrant,
                    collection_name=collection_name,
                    meta=new_meta,
                    set_fields={},
                    delete_keys=["supersedes"],
                )

    # Always delete payload keys so Qdrant matches catalog even if extra
    # already lacked one of them.
    old_delete_keys = ["superseded_by", "doc_status"]
    old_points, old_figures = _apply_doc_fields(
        storage=storage,
        qdrant=qdrant,
        collection_name=collection_name,
        meta=old_meta,
        set_fields={},
        delete_keys=old_delete_keys,
    )
    logger.info(
        "Unsuperseded doc_id=%s former_new_doc_id=%s",
        old_doc_id,
        replacement_id or "-",
    )
    return {
        "status": "updated",
        "old_doc_id": old_doc_id,
        "new_doc_id": replacement_id or None,
        "unlinked": True,
        "truncated": [],
        "old_points_updated": old_points,
        "old_figures_updated": old_figures,
        "new_points_updated": new_points,
        "new_figures_updated": new_figures,
        "old": old_meta.to_dict(),
        "new": new_meta.to_dict() if new_meta is not None else None,
    }
