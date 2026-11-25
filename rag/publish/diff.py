"""Compute catalog/file diffs for incremental publish."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rag.publish.manifest import FileChange, ReleaseManifest, new_release_id
from rag.storage import DocumentMeta, RagStorage, utc_now_iso

# Paths under the RAG root that are never published to readers.
SKIP_PREFIXES = (
    "uploads/",
    "fastembed/",
    "clip/",
    "qdrant/",  # points synced separately; directory replace is baseline-only
)

# Fields that affect reader visibility / catalog parity beyond content bytes.
# Content ``sha256`` is always included separately.
_FINGERPRINT_META_KEYS = (
    "filename",
    "doc_status",
    "superseded_by",
    "supersedes",
    "correction_note",
    "verification_status",
    "reliability",
    "source_key",
    "source_axis",
    "verification_checks",
    "evidence_source_type",
    "tags",
)

# Document-level fields stored on ``meta.json`` (not only ``extra``) that affect
# reader catalog parity and search filters.
_FINGERPRINT_TOP_LEVEL_KEYS = (
    "lat",
    "lon",
    "geo_refined",
    "geo_refined_history",
)


def _sha256_file(path: Path) -> tuple[str, int]:
    """Return ``(hex digest, size)`` for ``path``."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def _rel_posix(root: Path, path: Path) -> str:
    """Return a POSIX relative path from ``root``."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def _should_skip(rel: str) -> bool:
    """Return whether ``rel`` is excluded from publish."""
    if rel in {"release.json", "index.json"}:
        return False
    return any(
        rel == prefix.rstrip("/") or rel.startswith(prefix) for prefix in SKIP_PREFIXES
    )


def iter_publishable_files(rag_root: Path) -> Iterable[Path]:
    """Yield on-disk files under ``rag_root`` that belong in a release."""
    root = rag_root.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = _rel_posix(root, path)
        if _should_skip(rel):
            continue
        if rel.startswith("."):
            continue
        yield path


def _meta_fingerprint_payload(meta: DocumentMeta) -> dict[str, Any]:
    """Return the stable subset of metadata hashed into a catalog fingerprint."""
    extra = meta.extra if isinstance(meta.extra, Mapping) else {}
    payload: dict[str, Any] = {
        "sha256": str(meta.sha256 or ""),
        "filename": str(meta.filename or ""),
    }
    for key in _FINGERPRINT_TOP_LEVEL_KEYS:
        value = getattr(meta, key, None)
        if key == "geo_refined_history":
            if value:
                payload[key] = value
        elif value is not None:
            payload[key] = value
    for key in _FINGERPRINT_META_KEYS:
        if key in {"sha256", "filename"}:
            continue
        if key in extra:
            payload[key] = extra[key]
    return payload


def document_catalog_fingerprint(meta: DocumentMeta) -> str:
    """Return a composite fingerprint for one document.

    Hashes content ``sha256`` together with lifecycle / verification fields so
    metadata-only patches (e.g. ``reliability``) still produce a catalog diff
    when the file content digest is unchanged.
    """
    canonical = json.dumps(
        _meta_fingerprint_payload(meta),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _legacy_content_sha_fingerprint(sha: str) -> str:
    """Hash index ``sha256`` when ``meta.json`` is missing."""
    return hashlib.sha256(
        json.dumps(
            {"sha256": sha},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _fingerprint_map_for_storage(
    storage: RagStorage,
    *,
    from_meta: Callable[[DocumentMeta], str],
) -> dict[str, str]:
    """Build doc_id→fingerprint using ``from_meta`` on each ``meta.json``."""
    out: dict[str, str] = {}
    for doc in storage.list_documents():
        doc_id = str(doc.get("doc_id") or "")
        if not doc_id:
            continue
        meta = storage.read_meta(doc_id)
        if meta is not None:
            out[doc_id] = str(from_meta(meta))
            continue
        sha = str(doc.get("sha256") or "")
        if sha:
            out[doc_id] = _legacy_content_sha_fingerprint(sha)
    return out


def document_content_fingerprint(meta: DocumentMeta) -> str:
    """Return a content-only digest (processed bytes ``sha256``).

    Used with the reader ``documents_content`` catalog map to detect
    metadata-only drift without re-shipping embedding vectors.
    """
    sha = str(meta.sha256 or "")
    canonical = json.dumps(
        {"sha256": sha},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def content_catalog_fingerprint(storage: RagStorage) -> dict[str, str]:
    """Map ``doc_id`` → content-only fingerprint."""
    return _fingerprint_map_for_storage(
        storage,
        from_meta=document_content_fingerprint,
    )


def catalog_fingerprint(storage: RagStorage) -> dict[str, str]:
    """Map ``doc_id`` → composite catalog fingerprint.

    Prefer ``meta.json`` (includes verification / lifecycle extras). Fall back
    to the index ``sha256`` only when meta is missing.
    """
    return _fingerprint_map_for_storage(
        storage,
        from_meta=document_catalog_fingerprint,
    )


def _metadata_only_doc_ids(
    storage: RagStorage,
    upserted: Sequence[str],
    *,
    reader_catalog: Mapping[str, str],
    reader_content_catalog: Mapping[str, str] | None,
) -> list[str]:
    """Return upserted docs whose processed content digest already matches the reader."""
    if not upserted or not reader_content_catalog:
        return []
    writer_content = content_catalog_fingerprint(storage)
    out: list[str] = []
    for doc_id in upserted:
        if doc_id not in reader_catalog:
            continue
        reader_digest = reader_content_catalog.get(doc_id)
        writer_digest = writer_content.get(doc_id)
        if reader_digest and writer_digest and reader_digest == writer_digest:
            out.append(doc_id)
    return sorted(out)


def _paths_for_metadata_only_doc(root: Path, doc_id: str) -> list[Path]:
    """Return minimal on-disk artifacts needed for a metadata-only publish."""
    paths: list[Path] = []
    proc = root / "processed" / doc_id
    for name in ("meta.json", "chunks.json"):
        candidate = proc / name
        if candidate.is_file():
            paths.append(candidate)
    return paths


def build_file_diff(
    storage: RagStorage,
    *,
    parent_release_id: str | None,
    reader_catalog: Mapping[str, str] | None = None,
    reader_content_catalog: Mapping[str, str] | None = None,
    reader_locality_catalog: Mapping[str, str] | None = None,
    writer_locality_catalog: Mapping[str, str] | None = None,
    writer_host: str | None = None,
    force_baseline: bool = False,
) -> ReleaseManifest:
    """Build a file+catalog (+ optional locality_links) diff.

    When ``reader_catalog`` is empty/None or ``force_baseline`` is true, every
    publishable file is listed (baseline). Otherwise only docs whose catalog
    fingerprint changed (or are new/deleted) contribute file paths under that
    doc — including metadata-only changes.

    ``locality_link_doc_ids_upserted`` lists docs whose ``locality_links``
    fingerprint differs. Link-only docs do **not** re-list file blobs (memory
    safe): the push path exports payload-only points for those docs.
    """
    root = Path(storage.root)
    writer_fp = catalog_fingerprint(storage)
    reader_fp = dict(reader_catalog or {})
    baseline = force_baseline or not reader_fp

    upserted = sorted(
        doc_id
        for doc_id, digest in writer_fp.items()
        if baseline or reader_fp.get(doc_id) != digest
    )
    deleted = sorted(doc_id for doc_id in reader_fp if doc_id not in writer_fp)
    metadata_only = (
        []
        if baseline
        else _metadata_only_doc_ids(
            storage,
            upserted,
            reader_catalog=reader_fp,
            reader_content_catalog=reader_content_catalog,
        )
    )
    metadata_only_set = set(metadata_only)

    writer_links = dict(writer_locality_catalog or {})
    reader_links = dict(reader_locality_catalog or {})
    link_docs = sorted(
        {
            *(
                doc_id
                for doc_id, digest in writer_links.items()
                if baseline or reader_links.get(doc_id) != digest
            ),
            *(doc_id for doc_id in reader_links if doc_id not in writer_links),
        }
    )

    files_added: list[FileChange] = []
    files_deleted: list[str] = []

    if baseline:
        for path in iter_publishable_files(root):
            rel = _rel_posix(root, path)
            digest, size = _sha256_file(path)
            doc_id = _doc_id_from_rel(rel)
            files_added.append(
                FileChange(path=rel, sha256=digest, size=size, doc_id=doc_id)
            )
    else:
        # Delete processed/originals trees for removed docs.
        for doc_id in deleted:
            for prefix in ("originals", "processed", "versions"):
                files_deleted.append(f"{prefix}/{doc_id}")
        # Include top-level index.json only when the catalog actually changed.
        # Always appending it made no-op cron/parity runs look non-empty
        # (files_added=1) and open a spurious release (V2).
        if upserted or deleted:
            index_path = root / "index.json"
            if index_path.is_file():
                digest, size = _sha256_file(index_path)
                files_added.append(
                    FileChange(path="index.json", sha256=digest, size=size, doc_id=None)
                )
        for doc_id in upserted:
            if doc_id in metadata_only_set:
                for path in _paths_for_metadata_only_doc(root, doc_id):
                    rel = _rel_posix(root, path)
                    digest, size = _sha256_file(path)
                    files_added.append(
                        FileChange(path=rel, sha256=digest, size=size, doc_id=doc_id)
                    )
                continue
            for prefix in ("originals", "processed", "versions"):
                base = root / prefix / doc_id
                if not base.exists():
                    continue
                if base.is_file():
                    paths = [base]
                else:
                    paths = [p for p in base.rglob("*") if p.is_file()]
                for path in paths:
                    rel = _rel_posix(root, path)
                    digest, size = _sha256_file(path)
                    files_added.append(
                        FileChange(path=rel, sha256=digest, size=size, doc_id=doc_id)
                    )

    return ReleaseManifest(
        release_id=new_release_id(),
        parent_release_id=parent_release_id,
        created_at=utc_now_iso(),
        writer_host=writer_host,
        files_added=files_added,
        files_deleted=files_deleted,
        catalog_doc_ids_upserted=upserted,
        catalog_doc_ids_deleted=deleted,
        catalog_doc_ids_metadata_only=sorted(metadata_only),
        locality_link_doc_ids_upserted=link_docs,
        full_baseline=baseline,
    )


def _doc_id_from_rel(rel: str) -> str | None:
    """Extract doc_id from ``originals|processed|versions/<doc_id>/...``."""
    parts = rel.split("/")
    if len(parts) >= 2 and parts[0] in {"originals", "processed", "versions"}:
        return parts[1]
    return None


def summarize_diff(manifest: ReleaseManifest) -> dict[str, Any]:
    """Return compact counts for logs/health."""
    bytes_out = sum(item.size for item in manifest.files_added)
    return {
        "release_id": manifest.release_id,
        "parent_release_id": manifest.parent_release_id,
        "full_baseline": manifest.full_baseline,
        "files_added": len(manifest.files_added),
        "files_deleted": len(manifest.files_deleted),
        "docs_upserted": len(manifest.catalog_doc_ids_upserted),
        "docs_metadata_only": len(manifest.catalog_doc_ids_metadata_only),
        "docs_deleted": len(manifest.catalog_doc_ids_deleted),
        "locality_link_docs": len(manifest.locality_link_doc_ids_upserted),
        "qdrant_points_upserted": len(manifest.qdrant_point_ids_upserted),
        "qdrant_points_deleted": len(manifest.qdrant_point_ids_deleted),
        "bytes_added": bytes_out,
    }
