"""Atomic document supersession links (both sides, no silent peer overwrites)."""

from __future__ import annotations

from typing import Any, Mapping

from qdrant_client import QdrantClient

from rag.doc_locks import doc_lock
from rag.document_delete import DocumentNotFoundError
from rag.document_patch import delete_payload_keys_for_doc, write_verification_payload
from rag.logging_setup import get_logger
from rag.storage import DocumentMeta, RagStorage
from rag.verification_meta import (
    UNSUPERSEDE_ENDPOINT,
    DocumentMetadataError,
    is_hidden_doc_status,
    normalize_verification_metadata,
    supersedes_ids,
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


def _restore_doc_fields(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    meta: DocumentMeta,
    extra_before: Mapping[str, Any],
    touched: list[str],
) -> None:
    """Put ``touched`` catalog/payload keys back to ``extra_before``."""
    if not touched:
        return
    set_fields = {key: extra_before[key] for key in touched if key in extra_before}
    delete_keys = [key for key in touched if key not in extra_before]
    _apply_doc_fields(
        storage=storage,
        qdrant=qdrant,
        collection_name=collection_name,
        meta=meta,
        set_fields=set_fields,
        delete_keys=delete_keys,
    )


def _apply_pair_with_rollback(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    first: tuple[DocumentMeta, Mapping[str, Any], list[str]],
    second: tuple[DocumentMeta, Mapping[str, Any], list[str]] | None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Write ``first`` then ``second``; restore both snapshots if either fails."""
    first_meta, first_set, first_delete = first
    snapshots: list[tuple[DocumentMeta, dict[str, Any], list[str]]] = [
        (first_meta, dict(first_meta.extra), list(first_set) + list(first_delete)),
    ]
    if second is not None:
        second_meta, second_set, second_delete = second
        snapshots.append(
            (
                second_meta,
                dict(second_meta.extra),
                list(second_set) + list(second_delete),
            )
        )
    try:
        first_counts = _apply_doc_fields(
            storage=storage,
            qdrant=qdrant,
            collection_name=collection_name,
            meta=first_meta,
            set_fields=first_set,
            delete_keys=first_delete,
        )
        second_counts = (0, 0)
        if second is not None:
            second_meta, second_set, second_delete = second
            second_counts = _apply_doc_fields(
                storage=storage,
                qdrant=qdrant,
                collection_name=collection_name,
                meta=second_meta,
                set_fields=second_set,
                delete_keys=second_delete,
            )
        return first_counts, second_counts
    except Exception:
        for meta, extra_before, touched in snapshots:
            try:
                _restore_doc_fields(
                    storage=storage,
                    qdrant=qdrant,
                    collection_name=collection_name,
                    meta=meta,
                    extra_before=extra_before,
                    touched=touched,
                )
            except Exception:
                logger.exception(
                    "Failed to roll back supersession fields doc_id=%s",
                    meta.doc_id,
                )
        raise


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
    replacement_id = _normalize_new_doc_id(storage, new_doc_id)
    if replacement_id == old_doc_id:
        raise DocumentMetadataError("a document may not supersede itself")

    with doc_lock(old_doc_id, replacement_id):
        return _supersede_document_locked(
            storage=storage,
            qdrant=qdrant,
            collection_name=collection_name,
            old_doc_id=old_doc_id,
            replacement_id=replacement_id,
            correction_note=correction_note,
        )


def _supersede_document_locked(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    old_doc_id: str,
    replacement_id: str,
    correction_note: object | None,
) -> dict[str, Any]:
    """Apply supersession while ``doc_lock(old, new)`` is held."""
    old_meta = _load_document(storage, old_doc_id)
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
    new_status = str(new_meta.extra.get("doc_status") or "").strip()
    if (
        is_hidden_doc_status(new_status)
        or str(new_meta.extra.get("superseded_by") or "").strip()
    ):
        status_label = new_status or "superseded"
        raise DocumentMetadataError(
            f"new_doc_id {replacement_id} is {status_label}; "
            "choose an active replacement"
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

    new_supersedes = supersedes_ids(new_meta.extra.get("supersedes"))
    if old_doc_id not in new_supersedes:
        new_supersedes.append(old_doc_id)

    (old_points, old_figures), (new_points, new_figures) = _apply_pair_with_rollback(
        storage=storage,
        qdrant=qdrant,
        collection_name=collection_name,
        first=(old_meta, old_fields, []),
        second=(new_meta, {"supersedes": new_supersedes}, []),
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


class _RetryDocLock(Exception):
    """Internal: unsupersede should drop locks and retry with ``hint``."""

    def __init__(self, hint: str) -> None:
        self.hint = hint
        super().__init__(hint)


def unsupersede_document(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    old_doc_id: str,
) -> dict[str, Any]:
    """Remove the supersession link from ``old_doc_id`` and its replacement.

    Clears ``superseded_by`` on the old document and removes it from the
    replacement's ``supersedes`` list. Deletes ``doc_status`` only when it is
    ``superseded`` (keeps ``retracted``). Leaves ``correction_note``,
    verification fields, and vectors untouched. Idempotent when no link exists.
    """
    old_meta = _load_document(storage, old_doc_id)
    hint = str(old_meta.extra.get("superseded_by") or "").strip()
    while True:
        try:
            with doc_lock(old_doc_id, hint):
                return _unsupersede_document_locked(
                    storage=storage,
                    qdrant=qdrant,
                    collection_name=collection_name,
                    old_doc_id=old_doc_id,
                    expected_replacement=hint,
                )
        except _RetryDocLock as retry:
            hint = retry.hint


def _unsupersede_document_locked(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    old_doc_id: str,
    expected_replacement: str,
) -> dict[str, Any]:
    """Unlink while locks for ``old_doc_id`` and ``expected_replacement`` are held."""
    old_meta = _load_document(storage, old_doc_id)
    replacement_id = str(old_meta.extra.get("superseded_by") or "").strip()
    if replacement_id != expected_replacement:
        raise _RetryDocLock(replacement_id)

    new_meta: DocumentMeta | None = None
    new_set: dict[str, Any] = {}
    new_delete: list[str] = []
    if replacement_id:
        try:
            new_meta = _load_document(storage, replacement_id)
        except DocumentNotFoundError:
            new_meta = None
        if new_meta is not None:
            remaining = [
                item
                for item in supersedes_ids(new_meta.extra.get("supersedes"))
                if item != old_doc_id
            ]
            if remaining:
                new_set = {"supersedes": remaining}
            else:
                new_delete = ["supersedes"]

    old_delete_keys = ["superseded_by"]
    if str(old_meta.extra.get("doc_status") or "").strip() == "superseded":
        old_delete_keys.append("doc_status")

    new_step: tuple[DocumentMeta, Mapping[str, Any], list[str]] | None = None
    if new_meta is not None:
        new_step = (new_meta, new_set, new_delete)
    first_counts, second_counts = _apply_pair_with_rollback(
        storage=storage,
        qdrant=qdrant,
        collection_name=collection_name,
        first=new_step if new_step is not None else (old_meta, {}, old_delete_keys),
        second=(old_meta, {}, old_delete_keys) if new_step is not None else None,
    )
    if new_step is not None:
        new_points, new_figures = first_counts
        old_points, old_figures = second_counts
    else:
        old_points, old_figures = first_counts
        new_points, new_figures = 0, 0
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
