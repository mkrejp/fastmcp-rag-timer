"""In-place document text replacement with durable versions and Qdrant rollback.

Qdrant has no multi-point transactions. Replacement therefore:

1. Archives the current processed text under ``{RAG_DATA_ROOT}/versions/{doc_id}/``.
2. Embeds and **inserts** new ``pdf_chunk`` points with generation-scoped ids
   (so they never overwrite the live points).
3. Writes the new processed files (and text originals).
4. **Deletes** the previous text-chunk points.
5. Updates ``content_hash`` / ``updated_at`` on surviving ``pdf_image`` /
   ``pdf_figure`` points (those points are not rebuilt).

On any failure before the old points are dropped, newly inserted points are
deleted and the archived version snapshot is removed so the live document is
unchanged. Image and figure points are never deleted by this path.

``locality_links`` rows are counted (``affected_locality_links``) but never
deleted. After replace, generation-scoped chunk ids no longer equal
``stable_point_id(doc_id, chunk_index)``; callers must resolve by
``doc_id`` + ``chunk_index`` payload. Per-chunk payload keys that differ
across old points are not copied onto the new generation
(``dropped_chunk_fields``). Concurrent replace / metadata PATCH / DELETE
share process-local ``doc_lock`` (not cross-process or cross-node).
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, cast

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    PointStruct,
)

from rag.config import LOCALITY_LINKS_COLLECTION
from rag.doc_locks import doc_lock
from rag.document_delete import DocumentNotFoundError
from rag.embeddings import get_ingest_embeddings
from rag.ingest import content_sha256
from rag.logging_setup import get_logger
from rag.payloads import build_chunk_payload
from rag.processing import (
    JSON_EXTENSIONS,
    PDF_EXTENSIONS,
    TEXT_EXTENSIONS,
    process_document,
)
from rag.storage import DocumentMeta, RagStorage, utc_now_iso
from rag.vectorstore import build_named_vector
from rag.verification_meta import chunk_payload_fields

EmbedFn = Callable[[str], list[float]]
logger = get_logger("rag.document_replace")

TEXT_CHUNK_SOURCE = "pdf_chunk"
IMAGE_SOURCE_TYPES = ("pdf_image", "pdf_figure")
MAX_REASON_LEN = 2000
_SCROLL_LIMIT = 64
_OLD_DELETE_ATTEMPTS = 3
_ALWAYS_OVERWRITE_KEYS = frozenset({"content_hash", "updated_at"})

# Payload keys that belong to chunk identity / body, not document metadata.
_CHUNK_BODY_KEYS = frozenset(
    {
        "text",
        "content_summary",
        "chunk_index",
        "chunk_sha256",
        "sha256",
        "page_number",
        "source_type",
        "doc_id",
        "parent_document",
        "source",
        "localities",
        "cast_obce",
    }
)


class DocumentReplaceError(ValueError):
    """Raised when a content replace or restore request is invalid."""


class DocumentVersionNotFoundError(LookupError):
    """Raised when a requested version number is missing on disk."""

    def __init__(self, doc_id: str, version: int) -> None:
        """Record the missing version identity."""
        self.doc_id = doc_id
        self.version = version
        super().__init__(f"Version {version} not found for document {doc_id}")


@dataclass(frozen=True)
class VersionRecord:
    """One archived snapshot of a document's processed text."""

    version: int
    created_at: str
    author: str
    reason: str
    content_hash: str
    chunk_count: int
    filename: str
    generation: int

    def to_dict(self, *, text: str | None = None) -> dict[str, Any]:
        """Serialize this record; include ``text`` when provided."""
        payload: dict[str, Any] = {
            "version": self.version,
            "created_at": self.created_at,
            "author": self.author,
            "reason": self.reason,
            "content_hash": self.content_hash,
            "chunk_count": self.chunk_count,
            "filename": self.filename,
            "generation": self.generation,
        }
        if text is not None:
            payload["text"] = text
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VersionRecord:
        """Deserialize a version manifest dictionary."""
        return cls(
            version=int(data["version"]),
            created_at=str(data["created_at"]),
            author=str(data.get("author") or ""),
            reason=str(data.get("reason") or ""),
            content_hash=str(data["content_hash"]),
            chunk_count=int(data.get("chunk_count") or 0),
            filename=str(data.get("filename") or ""),
            generation=int(data.get("generation") or data["version"]),
        )


def replace_point_id(doc_id: str, generation: int, chunk_index: int) -> str:
    """Build a generation-scoped UUID for a replacement text chunk."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:g{generation}:{chunk_index}"))


def normalize_reason(reason: str | None) -> str:
    """Return a stripped non-empty reason or raise ``DocumentReplaceError``."""
    text = (reason or "").strip()
    if not text:
        raise DocumentReplaceError("reason is required and must be non-empty")
    if len(text) > MAX_REASON_LEN:
        raise DocumentReplaceError(
            f"reason must be at most {MAX_REASON_LEN} characters"
        )
    return text


def normalize_content(content: str | None) -> str:
    """Return replacement text or raise ``DocumentReplaceError``."""
    if content is None:
        raise DocumentReplaceError("content is required")
    if not isinstance(content, str):
        raise DocumentReplaceError("content must be a string")
    if not content.strip():
        raise DocumentReplaceError("content must be a non-empty string")
    return content


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


def _as_txt_name(filename: str) -> str:
    """Return ``filename`` with a ``.txt`` suffix for text-only processing."""
    stem = Path(filename).stem or "document"
    return f"{stem}.txt"


def prepare_replacement(
    filename: str, content: str
) -> tuple[str, list[dict[str, Any]], bool]:
    """Chunk ``content`` with the ingest pipeline used at upload.

    PDFs are always processed as plain text so a restore of extracted
    fulltext does not have to be valid PDF bytes. JSON/GeoJSON is parsed
    when the payload is valid; otherwise it falls back to plain text.
    ``overwrite_original`` is true when the original file should be
    rewritten to match the new content (text and successfully parsed JSON).
    """
    data = content.encode("utf-8")
    suffix = Path(filename).suffix.lower()
    if suffix in PDF_EXTENSIONS:
        fulltext, chunks = process_document(_as_txt_name(filename), data)
        return fulltext, chunks, False
    try:
        fulltext, chunks = process_document(filename, data)
    except ValueError:
        fulltext, chunks = process_document(_as_txt_name(filename), data)
        return fulltext, chunks, suffix in TEXT_EXTENSIONS
    overwrite = suffix in TEXT_EXTENSIONS or suffix in JSON_EXTENSIONS
    return fulltext, chunks, overwrite


def _source_type_filter(doc_id: str, source_type: str) -> Filter:
    """Return a filter for ``doc_id`` + ``source_type``."""
    return Filter(
        must=[
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            FieldCondition(key="source_type", match=MatchValue(value=source_type)),
        ]
    )


def _scroll_points(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    payload_filter: Filter,
    with_vectors: bool = False,
) -> list[Any]:
    """Scroll every point matching ``payload_filter``."""
    if not qdrant.collection_exists(collection_name):
        return []
    points: list[Any] = []
    offset: Any = None
    while True:
        batch, offset = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=payload_filter,
            limit=_SCROLL_LIMIT,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def _point_ids(points: list[Any]) -> list[int | str | uuid.UUID]:
    """Return Qdrant point ids from scrolled records."""
    ids: list[int | str | uuid.UUID] = []
    for point in points:
        raw = point.id
        if isinstance(raw, int):
            ids.append(raw)
        else:
            ids.append(str(raw))
    return ids


def _delete_point_ids(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    point_ids: Sequence[int | str | uuid.UUID],
) -> None:
    """Delete the given point ids. No-op when the list is empty."""
    if not point_ids or not qdrant.collection_exists(collection_name):
        return
    qdrant.delete(
        collection_name=collection_name,
        points_selector=PointIdsList(points=list(point_ids)),
        wait=True,
    )


def _remaining_point_ids(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    point_ids: Sequence[int | str | uuid.UUID],
) -> list[int | str | uuid.UUID]:
    """Return the subset of ``point_ids`` that still exist in Qdrant."""
    if not point_ids or not qdrant.collection_exists(collection_name):
        return []
    found = qdrant.retrieve(
        collection_name=collection_name,
        ids=list(point_ids),
        with_payload=False,
        with_vectors=False,
    )
    return _point_ids(list(found))


def _delete_old_chunk_points(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    point_ids: Sequence[int | str | uuid.UUID],
) -> list[int | str | uuid.UUID]:
    """Delete previous chunk ids, retrying, and return leftovers if any remain.

    After new points are committed, leftover old ids must not fail the
    replace. Callers report ``cleanup_pending`` instead of raising.
    """
    remaining = list(point_ids)
    for attempt in range(1, _OLD_DELETE_ATTEMPTS + 1):
        if not remaining:
            return []
        try:
            _delete_point_ids(
                qdrant, collection_name=collection_name, point_ids=remaining
            )
        except Exception:  # noqa: BLE001 - leftover cleanup is best-effort
            logger.exception(
                "Deleting replaced chunk points failed attempt=%s/%s "
                "doc_id=%s leftover_count=%s leftover_ids=%s",
                attempt,
                _OLD_DELETE_ATTEMPTS,
                doc_id,
                len(remaining),
                remaining,
            )
        remaining = _remaining_point_ids(
            qdrant, collection_name=collection_name, point_ids=remaining
        )
        if not remaining:
            return []
    logger.error(
        "Leftover old chunk points after replace doc_id=%s leftover_count=%s "
        "leftover_ids=%s; new generation is live — retry DELETE of these "
        "point ids or re-run replace",
        doc_id,
        len(remaining),
        remaining,
    )
    return remaining


def _count_locality_links(qdrant: QdrantClient, doc_id: str) -> int:
    """Return how many ``locality_links`` rows reference ``doc_id``.

    Does not create or delete the collection. Links are not rewritten on
    replace; they need a later re-link against the new chunk generation.
    """
    if not qdrant.collection_exists(LOCALITY_LINKS_COLLECTION):
        return 0
    return int(
        qdrant.count(
            collection_name=LOCALITY_LINKS_COLLECTION,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
            exact=True,
        ).count
    )


def _payload_values_equal(left: object, right: object) -> bool:
    """Return whether two payload values compare equal after JSON round-trip."""
    if left is right:
        return True
    try:
        return json.dumps(left, sort_keys=True, default=str) == json.dumps(
            right, sort_keys=True, default=str
        )
    except (TypeError, ValueError):
        return left == right


def _shared_chunk_fields(
    existing_chunks: list[Any],
) -> tuple[dict[str, Any], list[str]]:
    """Split old chunk metadata into identical-across-chunks vs dropped keys.

    Body identity keys and ``content_hash`` / ``updated_at`` are ignored.
    Keys missing from any chunk, or whose values differ, are dropped.
    """
    skip = _CHUNK_BODY_KEYS | _ALWAYS_OVERWRITE_KEYS
    maps: list[dict[str, Any]] = []
    for point in existing_chunks:
        payload = dict(point.payload or {})
        maps.append({key: value for key, value in payload.items() if key not in skip})
    if not maps:
        return {}, []
    keys: set[str] = set()
    for item in maps:
        keys.update(item)
    shared: dict[str, Any] = {}
    dropped: list[str] = []
    for key in sorted(keys):
        if any(key not in item for item in maps):
            dropped.append(key)
            continue
        first = maps[0][key]
        if all(_payload_values_equal(item[key], first) for item in maps[1:]):
            if first is not None:
                shared[key] = first
        else:
            dropped.append(key)
    return shared, dropped


def _side_effect_fields(
    *,
    dropped_chunk_fields: Sequence[str],
    affected_locality_links: int,
    leftover_old_ids: Sequence[int | str | uuid.UUID] = (),
) -> dict[str, Any]:
    """Return replace-response keys for dropped fields, links, and leftovers."""
    leftover = [str(point_id) for point_id in leftover_old_ids]
    return {
        "dropped_chunk_fields": list(dropped_chunk_fields),
        "affected_locality_links": int(affected_locality_links),
        "cleanup_pending": bool(leftover),
        "leftover_old_point_ids": leftover,
        "leftover_old_point_count": len(leftover),
    }


def _versions_index_path(storage: RagStorage, doc_id: str) -> Path:
    """Return the per-document versions catalog path."""
    return storage.versions_doc_dir(doc_id) / "index.json"


def _version_dir(storage: RagStorage, doc_id: str, version: int) -> Path:
    """Return the directory for one archived version."""
    return storage.versions_doc_dir(doc_id) / f"v{version:04d}"


def _read_versions_index(storage: RagStorage, doc_id: str) -> list[dict[str, Any]]:
    """Load the versions catalog, or an empty list."""
    path = _versions_index_path(storage, doc_id)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _write_versions_index(
    storage: RagStorage, doc_id: str, entries: list[dict[str, Any]]
) -> None:
    """Atomically write the per-document versions catalog."""
    dest_dir = storage.versions_doc_dir(doc_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = _versions_index_path(storage, doc_id)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def next_version_number(storage: RagStorage, doc_id: str) -> int:
    """Return the next 1-based version number for ``doc_id``."""
    entries = _read_versions_index(storage, doc_id)
    if not entries:
        return 1
    return max(int(item.get("version") or 0) for item in entries) + 1


def save_version_snapshot(
    storage: RagStorage,
    *,
    meta: DocumentMeta,
    fulltext: str,
    author: str,
    reason: str,
    created_at: str,
    generation: int,
) -> VersionRecord:
    """Persist one archived version (manifest + fulltext) on the data volume."""
    version = next_version_number(storage, meta.doc_id)
    dest = _version_dir(storage, meta.doc_id, version)
    dest.mkdir(parents=True, exist_ok=True)
    record = VersionRecord(
        version=version,
        created_at=created_at,
        author=author,
        reason=reason,
        content_hash=content_sha256(fulltext.encode("utf-8")),
        chunk_count=meta.chunk_count,
        filename=meta.filename,
        generation=generation,
    )
    (dest / "fulltext.txt").write_text(fulltext, encoding="utf-8")
    (dest / "manifest.json").write_text(
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    entries = _read_versions_index(storage, meta.doc_id)
    entries.append(record.to_dict())
    entries.sort(key=lambda item: int(item.get("version") or 0))
    _write_versions_index(storage, meta.doc_id, entries)
    return record


def discard_version_snapshot(storage: RagStorage, doc_id: str, version: int) -> None:
    """Remove a version directory and drop it from the catalog (rollback)."""
    dest = _version_dir(storage, doc_id, version)
    if dest.exists():
        shutil.rmtree(dest)
    entries = [
        item
        for item in _read_versions_index(storage, doc_id)
        if int(item.get("version") or 0) != version
    ]
    if entries:
        _write_versions_index(storage, doc_id, entries)
    else:
        index_path = _versions_index_path(storage, doc_id)
        if index_path.exists():
            index_path.unlink()
        doc_dir = storage.versions_doc_dir(doc_id)
        if doc_dir.is_dir() and not any(doc_dir.iterdir()):
            doc_dir.rmdir()


def read_version_text(storage: RagStorage, doc_id: str, version: int) -> str:
    """Return archived fulltext for ``version``."""
    path = _version_dir(storage, doc_id, version) / "fulltext.txt"
    if not path.is_file():
        raise DocumentVersionNotFoundError(doc_id, version)
    return path.read_text(encoding="utf-8")


def read_version_record(
    storage: RagStorage, doc_id: str, version: int
) -> VersionRecord:
    """Return the archived manifest for ``version``."""
    path = _version_dir(storage, doc_id, version) / "manifest.json"
    if not path.is_file():
        raise DocumentVersionNotFoundError(doc_id, version)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DocumentVersionNotFoundError(doc_id, version)
    return VersionRecord.from_dict(data)


def list_document_versions(
    *,
    storage: RagStorage,
    doc_id: str,
    include_text: bool = False,
) -> dict[str, Any]:
    """List archived versions for ``doc_id`` (404 if the document is missing)."""
    _load_document(storage, doc_id)
    versions: list[dict[str, Any]] = []
    for item in _read_versions_index(storage, doc_id):
        record = VersionRecord.from_dict(item)
        text: str | None = None
        if include_text:
            try:
                text = read_version_text(storage, doc_id, record.version)
            except DocumentVersionNotFoundError:
                text = ""
        versions.append(record.to_dict(text=text))
    return {
        "doc_id": doc_id,
        "versions": versions,
        "next_version": next_version_number(storage, doc_id),
    }


def current_fulltext(storage: RagStorage, meta: DocumentMeta) -> str:
    """Return the live processed text, or empty string if missing."""
    text = storage.read_fulltext(meta.doc_id)
    return text if text is not None else ""


def _preserved_payload(
    meta: DocumentMeta,
    *,
    existing_chunks: list[Any],
    new_hash: str,
    updated_at: str,
) -> tuple[dict[str, Any], list[str]]:
    """Return document-level payload fields plus names of dropped per-chunk keys.

    Catalog ``meta.extra`` is always applied. Additional keys are copied from
    old chunks only when every old chunk has the same value. Differing
    per-chunk fields are listed in ``dropped_chunk_fields`` and are not
    smeared onto the new generation.
    """
    fields = chunk_payload_fields(meta.extra)
    shared, dropped = _shared_chunk_fields(existing_chunks)
    for key, value in shared.items():
        fields.setdefault(key, value)
    if meta.lat is not None:
        fields.setdefault("lat", meta.lat)
    if meta.lon is not None:
        fields.setdefault("lon", meta.lon)
    if meta.geo_refined:
        fields.setdefault("geo_refined", dict(meta.geo_refined))
    fields["content_hash"] = new_hash
    fields["updated_at"] = updated_at
    return fields, dropped


def _image_payload_update(*, new_hash: str, updated_at: str) -> dict[str, Any]:
    """Fields written onto surviving image/figure points."""
    return {"content_hash": new_hash, "updated_at": updated_at}


def _figure_payload_update(*, new_hash: str, updated_at: str) -> dict[str, Any]:
    """Figure points also store the document digest as ``sha256``."""
    return {
        "content_hash": new_hash,
        "updated_at": updated_at,
        "sha256": new_hash,
    }


def _set_payload_for_source(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    doc_id: str,
    source_type: str,
    payload: Mapping[str, Any],
) -> int:
    """``set_payload`` on every point of ``source_type`` for ``doc_id``."""
    if not payload or not qdrant.collection_exists(collection_name):
        return 0
    filt = _source_type_filter(doc_id, source_type)
    matched = int(
        qdrant.count(
            collection_name=collection_name,
            count_filter=filt,
            exact=True,
        ).count
    )
    if matched == 0:
        return 0
    qdrant.set_payload(
        collection_name=collection_name,
        payload=dict(payload),
        points=FilterSelector(filter=filt),
        wait=True,
    )
    return matched


def _commit_processed(
    storage: RagStorage,
    *,
    meta: DocumentMeta,
    fulltext: str,
    chunks: list[dict[str, Any]],
    new_hash: str,
    updated_at: str,
    overwrite_original: bool,
    content: str,
) -> None:
    """Write processed fulltext/chunks/meta (and text originals) for the doc."""
    dest = storage.processed_doc_dir(meta.doc_id)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "fulltext.txt").write_text(fulltext, encoding="utf-8")
    storage.write_chunks(meta.doc_id, chunks)
    meta.sha256 = new_hash
    meta.chunk_count = len(chunks)
    extra = dict(meta.extra)
    extra["content_hash"] = new_hash
    extra["updated_at"] = updated_at
    meta.extra = extra
    storage.write_meta(meta)
    if overwrite_original:
        storage.save_original(meta.doc_id, meta.filename, content.encode("utf-8"))


def replace_document_content(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    doc_id: str,
    content: str,
    reason: str,
    author: str = "bearer",
    dry_run: bool = False,
    embed_fn: EmbedFn | None = None,
) -> dict[str, Any]:
    """Replace indexed text for ``doc_id`` while preserving metadata.

    Same ingest embedding backend as upload (``embed_fn`` or
    ``get_ingest_embeddings``). Identical content is a no-op. ``dry_run``
    reports hashes and counts without writing. Holds the process-local
    ``doc_lock`` for ``doc_id``.
    """
    with doc_lock(doc_id):
        return _replace_document_content(
            storage=storage,
            qdrant=qdrant,
            collection_name=collection_name,
            doc_id=doc_id,
            content=content,
            reason=reason,
            author=author,
            dry_run=dry_run,
            embed_fn=embed_fn,
        )


def _replace_document_content(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    doc_id: str,
    content: str,
    reason: str,
    author: str = "bearer",
    dry_run: bool = False,
    embed_fn: EmbedFn | None = None,
) -> dict[str, Any]:
    """Replace indexed text while the caller holds ``doc_lock(doc_id)``."""
    meta = _load_document(storage, doc_id)
    reason_text = normalize_reason(reason)
    new_content = normalize_content(content)
    current_text = current_fulltext(storage, meta)
    fulltext, chunks, overwrite_original = prepare_replacement(
        meta.filename, new_content
    )
    old_hash = content_sha256(current_text.encode("utf-8"))
    new_hash = content_sha256(fulltext.encode("utf-8"))
    next_version = next_version_number(storage, doc_id)
    old_chunks = _scroll_points(
        qdrant,
        collection_name=collection_name,
        payload_filter=_source_type_filter(doc_id, TEXT_CHUNK_SOURCE),
    )
    image_points = 0
    for source_type in IMAGE_SOURCE_TYPES:
        image_points += len(
            _scroll_points(
                qdrant,
                collection_name=collection_name,
                payload_filter=_source_type_filter(doc_id, source_type),
            )
        )
    affected_links = _count_locality_links(qdrant, doc_id)
    extra_payload, dropped_chunk_fields = _preserved_payload(
        meta,
        existing_chunks=old_chunks,
        new_hash=new_hash,
        updated_at="",
    )
    side = _side_effect_fields(
        dropped_chunk_fields=dropped_chunk_fields,
        affected_locality_links=affected_links,
    )

    if old_hash == new_hash:
        logger.info("Content replace no-op doc_id=%s (identical hash)", doc_id)
        return {
            "status": "unchanged",
            "doc_id": doc_id,
            "filename": meta.filename,
            "content_hash": old_hash,
            "chunk_count": len(old_chunks),
            "image_points_preserved": image_points,
            **side,
        }

    if dry_run:
        return {
            "status": "dry_run",
            "doc_id": doc_id,
            "filename": meta.filename,
            "next_version": next_version,
            "old_content_hash": old_hash,
            "new_content_hash": new_hash,
            "old_chunk_count": len(old_chunks),
            "new_chunk_count": len(chunks),
            "image_points_preserved": image_points,
            **side,
        }

    updated_at = utc_now_iso()
    extra_payload["content_hash"] = new_hash
    extra_payload["updated_at"] = updated_at
    source_url = extra_payload.pop("source_url", None)
    previous_sha = meta.sha256
    previous_count = meta.chunk_count
    previous_extra = dict(meta.extra)
    previous_chunks = storage.read_chunks(doc_id)
    snapshot = save_version_snapshot(
        storage,
        meta=meta,
        fulltext=current_text,
        author=author,
        reason=reason_text,
        created_at=updated_at,
        generation=next_version,
    )

    texts = [str(chunk["text"]) for chunk in chunks]
    new_ids: list[int | str | uuid.UUID] = []
    try:
        if embed_fn is not None:
            vectors = [embed_fn(text) for text in texts]
        else:
            vectors = get_ingest_embeddings(texts) if texts else []
        points: list[PointStruct] = []
        for chunk, text, vector in zip(chunks, texts, vectors, strict=True):
            point_id = replace_point_id(
                doc_id, snapshot.generation, int(chunk["index"])
            )
            new_ids.append(point_id)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=cast(Any, build_named_vector(text_vector=vector)),
                    payload=build_chunk_payload(
                        doc_id=doc_id,
                        parent_document=meta.filename,
                        page_number=int(chunk.get("page", 0)),
                        doc_sha256=new_hash,
                        text=text,
                        chunk_index=int(chunk["index"]),
                        source_url=str(source_url) if source_url else None,
                        extra_payload=extra_payload,
                    ),
                )
            )
        if points:
            qdrant.upsert(collection_name=collection_name, points=points, wait=True)
        _commit_processed(
            storage,
            meta=meta,
            fulltext=fulltext,
            chunks=chunks,
            new_hash=new_hash,
            updated_at=updated_at,
            overwrite_original=overwrite_original,
            content=new_content,
        )
    except Exception:
        _delete_point_ids(qdrant, collection_name=collection_name, point_ids=new_ids)
        try:
            dest = storage.processed_doc_dir(doc_id)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "fulltext.txt").write_text(current_text, encoding="utf-8")
            if previous_chunks is not None:
                storage.write_chunks(doc_id, previous_chunks)
            meta.sha256 = previous_sha
            meta.chunk_count = previous_count
            meta.extra = previous_extra
            storage.write_meta(meta)
        except Exception:  # noqa: BLE001 - best-effort disk rollback
            logger.exception("Processed-file rollback failed doc_id=%s", doc_id)
        discard_version_snapshot(storage, doc_id, snapshot.version)
        logger.exception(
            "Content replace rolled back doc_id=%s version=%s",
            doc_id,
            snapshot.version,
        )
        raise

    old_ids = _point_ids(old_chunks)
    leftover_old_ids = _delete_old_chunk_points(
        qdrant,
        collection_name=collection_name,
        doc_id=doc_id,
        point_ids=old_ids,
    )
    try:
        figures_updated = _set_payload_for_source(
            qdrant,
            collection_name=collection_name,
            doc_id=doc_id,
            source_type="pdf_figure",
            payload=_figure_payload_update(new_hash=new_hash, updated_at=updated_at),
        )
        images_updated = _set_payload_for_source(
            qdrant,
            collection_name=collection_name,
            doc_id=doc_id,
            source_type="pdf_image",
            payload=_image_payload_update(new_hash=new_hash, updated_at=updated_at),
        )
    except Exception:  # noqa: BLE001 - chunk swap already committed
        logger.exception(
            "Image/figure payload update failed after replace doc_id=%s",
            doc_id,
        )
        figures_updated = 0
        images_updated = 0
    logger.info(
        "Replaced document content doc_id=%s version=%s new_chunks=%s "
        "old_chunks=%s leftover_old=%s images_updated=%s figures_updated=%s "
        "affected_locality_links=%s dropped_chunk_fields=%s",
        doc_id,
        snapshot.version,
        len(chunks),
        len(old_ids),
        len(leftover_old_ids),
        images_updated,
        figures_updated,
        affected_links,
        ",".join(dropped_chunk_fields) or "-",
    )
    return {
        "status": "replaced",
        "doc_id": doc_id,
        "filename": meta.filename,
        "version": snapshot.version,
        "content_hash": new_hash,
        "previous_content_hash": old_hash,
        "chunks_indexed": len(chunks),
        "chunks_removed": len(old_ids) - len(leftover_old_ids),
        "image_points_preserved": image_points,
        "author": author,
        "reason": reason_text,
        "updated_at": updated_at,
        **_side_effect_fields(
            dropped_chunk_fields=dropped_chunk_fields,
            affected_locality_links=affected_links,
            leftover_old_ids=leftover_old_ids,
        ),
    }


def restore_document_version(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    doc_id: str,
    version: int,
    reason: str,
    author: str = "bearer",
    dry_run: bool = False,
    embed_fn: EmbedFn | None = None,
) -> dict[str, Any]:
    """Restore archived version ``n`` through the same safe replace path.

    Archives the live text as a new version before applying the restore.
    Holds the process-local ``doc_lock`` for ``doc_id`` (re-entrant with
    the inner replace).
    """
    with doc_lock(doc_id):
        _load_document(storage, doc_id)
        if version < 1:
            raise DocumentReplaceError("version must be a positive integer")
        record = read_version_record(storage, doc_id, version)
        text = read_version_text(storage, doc_id, version)
        restore_reason = normalize_reason(reason)
        result = _replace_document_content(
            storage=storage,
            qdrant=qdrant,
            collection_name=collection_name,
            doc_id=doc_id,
            content=text,
            reason=restore_reason,
            author=author,
            dry_run=dry_run,
            embed_fn=embed_fn,
        )
        if result.get("status") == "unchanged":
            return result
        if dry_run:
            result["restore_version"] = version
            result["restore_content_hash"] = record.content_hash
            return result
        result["status"] = "restored"
        result["restored_version"] = version
        return result
