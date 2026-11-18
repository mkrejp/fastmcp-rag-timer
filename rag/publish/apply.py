"""Apply a release manifest onto a reader data root (files + Qdrant points)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from rag.logging_setup import get_logger
from rag.publish.manifest import ReleaseManifest, load_manifest
from rag.publish.qdrant_sync import POINTS_REL_PATH, apply_qdrant_release
from rag.publish.release_state import write_release_state
from rag.storage import RagStorage

logger = get_logger("rag.publish.apply")


def apply_file_release(
    rag_root: Path | str,
    *,
    manifest: ReleaseManifest,
    blob_root: Path | str,
    source: str | None = None,
    points_jsonl: Path | str | None = None,
    sync_qdrant: bool = True,
) -> dict[str, Any]:
    """Apply file add/delete ops from ``manifest`` using blobs under ``blob_root``.

    Expected layout: ``blob_root/<relative path>`` matches ``FileChange.path``.
    Caller must stop the reader app first (embedded Qdrant single-writer lock).
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
        points_path = (
            Path(points_jsonl)
            if points_jsonl is not None
            else blobs.parent / POINTS_REL_PATH
        )
        has_points = points_path.is_file()
        if not has_points:
            points_path = None
            if manifest.catalog_doc_ids_upserted:
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
            qdrant_stats = apply_qdrant_release(
                storage.qdrant_dir,
                upsert_doc_ids=upsert_ids,
                delete_doc_ids=delete_ids,
                points_jsonl=points_path,
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


def apply_release_dir(rag_root: Path | str, release_dir: Path | str) -> dict[str, Any]:
    """Apply ``release_dir/manifest.json`` with blobs rooted at ``release_dir``."""
    release_path = Path(release_dir)
    manifest = load_manifest(release_path / "manifest.json")
    points = release_path / POINTS_REL_PATH
    return apply_file_release(
        rag_root,
        manifest=manifest,
        blob_root=release_path / "blobs",
        points_jsonl=points if points.is_file() else None,
    )
