"""Compute catalog/file diffs for incremental publish."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from rag.publish.manifest import FileChange, ReleaseManifest, new_release_id
from rag.storage import RagStorage, utc_now_iso

# Paths under the RAG root that are never published to readers.
SKIP_PREFIXES = (
    "uploads/",
    "fastembed/",
    "clip/",
    "qdrant/",  # points synced separately; directory replace is baseline-only
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
    return any(rel == prefix.rstrip("/") or rel.startswith(prefix) for prefix in SKIP_PREFIXES)


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


def catalog_fingerprint(storage: RagStorage) -> dict[str, str]:
    """Map ``doc_id`` → content ``sha256`` from the catalog."""
    out: dict[str, str] = {}
    for doc in storage.list_documents():
        doc_id = str(doc.get("doc_id") or "")
        sha = str(doc.get("sha256") or "")
        if doc_id and sha:
            out[doc_id] = sha
    return out


def build_file_diff(
    storage: RagStorage,
    *,
    parent_release_id: str | None,
    reader_catalog: Mapping[str, str] | None = None,
    writer_host: str | None = None,
    force_baseline: bool = False,
) -> ReleaseManifest:
    """Build a file+catalog diff against ``reader_catalog`` (doc_id→sha256).

    When ``reader_catalog`` is empty/None or ``force_baseline`` is true, every
    publishable file is listed (baseline). Otherwise only docs whose sha256
    changed (or are new/deleted) contribute file paths under that doc.
    """
    root = Path(storage.root)
    writer_fp = catalog_fingerprint(storage)
    reader_fp = dict(reader_catalog or {})
    baseline = force_baseline or not reader_fp

    upserted = sorted(
        doc_id
        for doc_id, sha in writer_fp.items()
        if baseline or reader_fp.get(doc_id) != sha
    )
    deleted = sorted(doc_id for doc_id in reader_fp if doc_id not in writer_fp)

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
        touched = set(upserted) | set(deleted)
        # Delete processed/originals trees for removed docs.
        for doc_id in deleted:
            for prefix in ("originals", "processed", "versions"):
                files_deleted.append(f"{prefix}/{doc_id}")
        # Re-hash files under touched upserted docs + top-level index.json.
        index_path = root / "index.json"
        if index_path.is_file():
            digest, size = _sha256_file(index_path)
            files_added.append(
                FileChange(path="index.json", sha256=digest, size=size, doc_id=None)
            )
        for doc_id in upserted:
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
                        FileChange(
                            path=rel, sha256=digest, size=size, doc_id=doc_id
                        )
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
        "docs_deleted": len(manifest.catalog_doc_ids_deleted),
        "bytes_added": bytes_out,
    }
