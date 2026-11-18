"""Apply a release manifest onto a reader data root (files + Qdrant points)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient

from rag.logging_setup import get_logger
from rag.publish.manifest import ReleaseManifest, load_manifest
from rag.publish.qdrant_sync import (
    POINTS_REL_PATH,
    apply_qdrant_release,
    apply_qdrant_release_with_client,
    export_points_for_docs,
    write_points_jsonl,
)
from rag.publish.release_state import write_release_state
from rag.storage import RagStorage, utc_now_iso

logger = get_logger("rag.publish.apply")

_DOC_TREE_NAMES: tuple[str, ...] = ("originals", "processed", "versions")
_SNAPSHOT_META_NAME = "snapshot.json"
_SNAPSHOT_POINTS_REL = "qdrant/points.jsonl"
_SNAPSHOT_CATALOG_REL = "catalog_entries.json"


def _sync_qdrant_for_release(
    storage: RagStorage,
    *,
    upsert_ids: list[str],
    delete_ids: list[str],
    points_path: Path | None,
    qdrant_client: QdrantClient | None,
) -> dict[str, Any]:
    """Run point sync either in-process or via the open+close CLI wrapper."""
    if qdrant_client is not None:
        return apply_qdrant_release_with_client(
            qdrant_client,
            upsert_doc_ids=upsert_ids,
            delete_doc_ids=delete_ids,
            points_jsonl=points_path,
        )
    return apply_qdrant_release(
        storage.qdrant_dir,
        upsert_doc_ids=upsert_ids,
        delete_doc_ids=delete_ids,
        points_jsonl=points_path,
    )


def apply_file_release(
    rag_root: Path | str,
    *,
    manifest: ReleaseManifest,
    blob_root: Path | str,
    source: str | None = None,
    points_jsonl: Path | str | None = None,
    sync_qdrant: bool = True,
    qdrant_client: QdrantClient | None = None,
) -> dict[str, Any]:
    """Apply file add/delete ops from ``manifest`` using blobs under ``blob_root``.

    Expected layout: ``blob_root/<relative path>`` matches ``FileChange.path``.
    When ``qdrant_client`` is ``None`` (CLI / ``rag-apply-release.sh``), Qdrant is
    opened and closed for the sync — stop the reader first (single-writer lock).
    When ``qdrant_client`` is provided, points are applied in-process on that
    client (no second ``path=`` open).

    When ``sync_qdrant`` is true, points for deleted/upserted docs are synced
    from ``points_jsonl`` (default: sibling ``qdrant/points.jsonl``).
    """
    root = Path(rag_root)
    blobs = Path(blob_root)
    root.mkdir(parents=True, exist_ok=True)

    for rel in manifest.files_deleted:
        target = root / rel
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()

    copied = 0
    for change in manifest.files_added:
        src = blobs / change.path
        if not src.is_file():
            raise FileNotFoundError(f"missing blob for {change.path}: {src}")
        dest = root / change.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1

    qdrant_stats: dict[str, Any] = {}
    if sync_qdrant:
        candidate = (
            Path(points_jsonl)
            if points_jsonl is not None
            else blobs.parent / POINTS_REL_PATH
        )
        has_points = candidate.is_file()
        points_path: Path | None = candidate if has_points else None
        if not has_points and manifest.catalog_doc_ids_upserted:
            logger.warning(
                "Release %s has upserted docs but no %s; "
                "file/catalog applied without vector upsert",
                manifest.release_id,
                POINTS_REL_PATH,
            )
        # Only purge upserted docs when we also have replacement points.
        upsert_ids = list(manifest.catalog_doc_ids_upserted) if has_points else []
        delete_ids = list(manifest.catalog_doc_ids_deleted)
        if upsert_ids or delete_ids or has_points:
            storage = RagStorage(str(root))
            storage.ensure_layout()
            qdrant_stats = _sync_qdrant_for_release(
                storage,
                upsert_ids=upsert_ids,
                delete_ids=delete_ids,
                points_path=points_path,
                qdrant_client=qdrant_client,
            )

    state = write_release_state(
        root,
        release_id=manifest.release_id,
        parent_release_id=manifest.parent_release_id,
        source=source or manifest.writer_host,
        extra={
            "full_baseline": manifest.full_baseline,
            "files_copied": copied,
            "files_deleted": len(manifest.files_deleted),
            **qdrant_stats,
        },
    )
    logger.info(
        "Applied release %s files_copied=%s files_deleted=%s qdrant=%s",
        manifest.release_id,
        copied,
        len(manifest.files_deleted),
        qdrant_stats,
    )
    return state


def apply_release_dir(
    rag_root: Path | str,
    release_dir: Path | str,
    *,
    qdrant_client: QdrantClient | None = None,
) -> dict[str, Any]:
    """Apply ``release_dir/manifest.json`` with blobs rooted at ``release_dir``."""
    release_path = Path(release_dir)
    manifest = load_manifest(release_path / "manifest.json")
    points = release_path / POINTS_REL_PATH
    return apply_file_release(
        rag_root,
        manifest=manifest,
        blob_root=release_path / "blobs",
        points_jsonl=points if points.is_file() else None,
        qdrant_client=qdrant_client,
    )


def snapshot_touched_docs(
    rag_root: Path | str,
    *,
    client: QdrantClient,
    doc_ids: Sequence[str],
    snapshot_dir: Path | str,
) -> dict[str, Any]:
    """Snapshot files + Qdrant points for ``doc_ids`` under ``snapshot_dir``.

    Library-only helper for optional pre-apply rollback. Layout::

        snapshot_dir/
          snapshot.json
          catalog_entries.json
          files/{originals,processed,versions}/<doc_id>/...
          qdrant/points.jsonl

    Does not prune older snapshots (callers enforce ``RAG_SYNC_ROLLBACK_KEEP``).
    """
    storage = RagStorage(str(rag_root))
    dest = Path(snapshot_dir)
    if dest.exists():
        shutil.rmtree(dest)
    files_root = dest / "files"
    files_root.mkdir(parents=True, exist_ok=True)

    unique_ids = sorted({str(d) for d in doc_ids if d})
    catalog_by_id = {
        str(entry.get("doc_id")): entry
        for entry in storage.list_documents()
        if entry.get("doc_id")
    }
    catalog_entries: list[dict[str, Any]] = []
    trees_copied = 0
    for doc_id in unique_ids:
        entry = catalog_by_id.get(doc_id)
        if entry is not None:
            catalog_entries.append(dict(entry))
        for tree_name in _DOC_TREE_NAMES:
            src = storage.root / tree_name / doc_id
            if not src.exists():
                continue
            target = files_root / tree_name / doc_id
            target.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, target)
            else:
                shutil.copy2(src, target)
            trees_copied += 1

    records, point_ids = export_points_for_docs(client, unique_ids)
    points_path = dest / _SNAPSHOT_POINTS_REL
    write_points_jsonl(points_path, records)

    catalog_path = dest / _SNAPSHOT_CATALOG_REL
    catalog_path.write_text(
        json.dumps(catalog_entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    meta: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "doc_ids": unique_ids,
        "trees_copied": trees_copied,
        "points_exported": len(records),
        "point_ids": point_ids,
        "catalog_entries": len(catalog_entries),
    }
    (dest / _SNAPSHOT_META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Rollback snapshot dir=%s docs=%s trees=%s points=%s",
        dest,
        len(unique_ids),
        trees_copied,
        len(records),
    )
    return meta


def restore_touched_docs_snapshot(
    rag_root: Path | str,
    *,
    client: QdrantClient,
    snapshot_dir: Path | str,
) -> dict[str, Any]:
    """Restore a :func:`snapshot_touched_docs` directory onto ``rag_root``.

    Replaces per-doc file trees from the snapshot, merges catalog rows for those
    docs, then deletes + re-upserts Qdrant points for the snapshotted ``doc_ids``.
    Library only — not wired to HTTP.
    """
    storage = RagStorage(str(rag_root))
    storage.ensure_layout()
    source = Path(snapshot_dir)
    meta_path = source / _SNAPSHOT_META_NAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing snapshot meta: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("snapshot meta must be an object")
    doc_ids = [str(d) for d in meta.get("doc_ids") or [] if d]
    files_root = source / "files"

    trees_restored = 0
    for doc_id in doc_ids:
        for tree_name in _DOC_TREE_NAMES:
            live = storage.root / tree_name / doc_id
            snap = files_root / tree_name / doc_id
            if live.exists():
                shutil.rmtree(live) if live.is_dir() else live.unlink()
            if not snap.exists():
                continue
            live.parent.mkdir(parents=True, exist_ok=True)
            if snap.is_dir():
                shutil.copytree(snap, live)
            else:
                shutil.copy2(snap, live)
            trees_restored += 1

    catalog_path = source / _SNAPSHOT_CATALOG_REL
    if catalog_path.is_file():
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        snap_entries = (
            [e for e in raw if isinstance(e, dict)] if isinstance(raw, list) else []
        )
        snap_by_id = {str(e["doc_id"]): e for e in snap_entries if e.get("doc_id")}
        kept = [
            e
            for e in storage.list_documents()
            if str(e.get("doc_id")) not in set(doc_ids)
        ]
        for doc_id in doc_ids:
            if doc_id in snap_by_id:
                kept.append(snap_by_id[doc_id])
        kept.sort(key=lambda e: str(e.get("uploaded_at", "")), reverse=True)
        tmp_path = storage.index_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(kept, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, storage.index_path)

    qdrant_stats = apply_qdrant_release_with_client(
        client,
        upsert_doc_ids=doc_ids,
        delete_doc_ids=[],
        points_jsonl=source / _SNAPSHOT_POINTS_REL,
    )
    result = {
        "doc_ids": doc_ids,
        "trees_restored": trees_restored,
        **qdrant_stats,
    }
    logger.info("Restored rollback snapshot dir=%s result=%s", source, result)
    return result
