"""Soft rollback snapshots for live sync applies (v1).

Before applying a push, copy touched document trees, export their Qdrant
points, and snapshot ``release.json`` under ``.sync/rollback/<release_id>/``.
Restore puts docs, vectors, **and the previous watermark** back so the next
push can continue from the pre-commit parent.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient

from rag.logging_setup import get_logger
from rag.publish.qdrant_sync import (
    POINTS_REL_PATH,
    delete_points_for_docs,
    export_points_for_docs,
    read_points_jsonl,
    upsert_point_records,
    write_points_jsonl,
)
from rag.publish.release_state import RELEASE_STATE_NAME, read_release_state
from rag.publish.sync_constants import ROLLBACK_DIR_NAME, SYNC_DIR_NAME
from rag.storage import RagStorage, utc_now_iso

logger = get_logger("rag.publish.sync_rollback")

META_NAME = "rollback.json"


def rollback_root(rag_root: Path | str, release_id: str) -> Path:
    """Return ``.sync/rollback/<release_id>`` under ``rag_root``."""
    return Path(rag_root) / SYNC_DIR_NAME / ROLLBACK_DIR_NAME / release_id


def snapshot_touched_docs(
    rag_root: Path | str,
    *,
    client: QdrantClient,
    release_id: str,
    doc_ids: Sequence[str],
) -> Path:
    """Snapshot files, points, and ``release.json`` before a sync apply.

    Empty ``doc_ids`` still writes a meta marker (and watermark snapshot) so
    commit can record that no document trees needed copying.
    """
    root = Path(rag_root)
    dest = rollback_root(root, release_id)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    files_root = dest / "files"
    files_root.mkdir(parents=True, exist_ok=True)

    unique = sorted({str(d) for d in doc_ids if d})
    copied_paths: list[str] = []
    for doc_id in unique:
        for prefix in ("originals", "processed", "versions"):
            src = root / prefix / doc_id
            if not src.exists():
                continue
            target = files_root / prefix / doc_id
            if src.is_dir():
                shutil.copytree(src, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
            copied_paths.append(f"{prefix}/{doc_id}")

    # Also snapshot index.json when any docs change (catalog mutation).
    index_src = root / "index.json"
    if unique and index_src.is_file():
        shutil.copy2(index_src, files_root / "index.json")
        copied_paths.append("index.json")

    # Capture pre-commit watermark so rollback can restore parent_release_id.
    prior_release = read_release_state(root)
    release_src = root / RELEASE_STATE_NAME
    had_release_json = release_src.is_file()
    if had_release_json:
        shutil.copy2(release_src, dest / RELEASE_STATE_NAME)

    records, point_ids = export_points_for_docs(client, unique)
    write_points_jsonl(dest / POINTS_REL_PATH, records)

    meta = {
        "release_id": release_id,
        "created_at": utc_now_iso(),
        "doc_ids": unique,
        "files_copied": copied_paths,
        "point_ids": point_ids,
        "points_count": len(records),
        "had_release_json": had_release_json,
        "prior_release_id": (
            prior_release.get("release_id") if prior_release else None
        ),
        "prior_parent_release_id": (
            prior_release.get("parent_release_id") if prior_release else None
        ),
    }
    (dest / META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Rollback snapshot release=%s docs=%s files=%s points=%s prior_release=%s",
        release_id,
        len(unique),
        len(copied_paths),
        len(records),
        meta["prior_release_id"],
    )
    return dest


def restore_rollback(
    rag_root: Path | str,
    *,
    client: QdrantClient,
    release_id: str,
) -> dict[str, Any]:
    """Restore files, points, and ``release.json`` from a soft rollback snapshot.

    Replaces current trees for snapshotted doc ids, re-upserts exported
    points (after deleting current points for those docs), and restores the
    pre-commit watermark atomically with the filesystem copy.
    """
    root = Path(rag_root)
    snap = rollback_root(root, release_id)
    meta_path = snap / META_NAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"no rollback snapshot for release {release_id}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    doc_ids = [str(d) for d in (meta.get("doc_ids") or [])]
    files_root = snap / "files"

    # Remove current trees then restore from snapshot.
    for doc_id in doc_ids:
        for prefix in ("originals", "processed", "versions"):
            current = root / prefix / doc_id
            if current.is_dir():
                shutil.rmtree(current, ignore_errors=True)
            elif current.exists():
                current.unlink()
            snap_path = files_root / prefix / doc_id
            if snap_path.exists():
                if snap_path.is_dir():
                    shutil.copytree(snap_path, current, dirs_exist_ok=True)
                else:
                    current.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(snap_path, current)

    index_snap = files_root / "index.json"
    if index_snap.is_file():
        shutil.copy2(index_snap, root / "index.json")

    points_path = snap / POINTS_REL_PATH
    records: list[dict[str, Any]] = []
    if points_path.is_file():
        records = read_points_jsonl(points_path)
    deleted = delete_points_for_docs(client, doc_ids) if doc_ids else 0
    written = upsert_point_records(client, records) if records else 0

    # Restore pre-commit watermark (or clear it if the snapshot had none).
    release_dest = root / RELEASE_STATE_NAME
    release_snap = snap / RELEASE_STATE_NAME
    had_release = bool(meta.get("had_release_json"))
    if had_release and release_snap.is_file():
        tmp = release_dest.with_suffix(".json.tmp")
        tmp.write_bytes(release_snap.read_bytes())
        tmp.replace(release_dest)
        restored_wm = read_release_state(root)
    else:
        if release_dest.exists():
            release_dest.unlink()
        restored_wm = None

    # Refresh catalog from restored processed metas when possible.
    storage = RagStorage(str(root))
    storage.ensure_layout()

    result = {
        "release_id": release_id,
        "docs_restored": len(doc_ids),
        "points_deleted": deleted,
        "points_upserted": written,
        "restored_at": utc_now_iso(),
        "watermark_restored": restored_wm is not None or not had_release,
        "restored_release_id": (restored_wm.get("release_id") if restored_wm else None),
        "restored_parent_release_id": (
            restored_wm.get("parent_release_id") if restored_wm else None
        ),
    }
    logger.info("Restored rollback %s: %s", release_id, result)
    return result
