"""Apply a release manifest onto a reader data root (files only)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from rag.logging_setup import get_logger
from rag.publish.manifest import ReleaseManifest, load_manifest
from rag.publish.release_state import write_release_state

logger = get_logger("rag.publish.apply")


def apply_file_release(
    rag_root: Path | str,
    *,
    manifest: ReleaseManifest,
    blob_root: Path | str,
    source: str | None = None,
) -> dict[str, Any]:
    """Apply file add/delete ops from ``manifest`` using blobs under ``blob_root``.

    Expected layout: ``blob_root/<relative path>`` matches ``FileChange.path``.
    Does not touch the live Qdrant directory (caller stops the app first).
    Qdrant point upsert/delete is left to a later phase / operator tooling.
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

    state = write_release_state(
        root,
        release_id=manifest.release_id,
        parent_release_id=manifest.parent_release_id,
        source=source or manifest.writer_host,
        extra={
            "full_baseline": manifest.full_baseline,
            "files_copied": copied,
            "files_deleted": len(manifest.files_deleted),
        },
    )
    logger.info(
        "Applied release %s files_copied=%s files_deleted=%s",
        manifest.release_id,
        copied,
        len(manifest.files_deleted),
    )
    return state


def apply_release_dir(rag_root: Path | str, release_dir: Path | str) -> dict[str, Any]:
    """Apply ``release_dir/manifest.json`` with blobs rooted at ``release_dir``."""
    release_path = Path(release_dir)
    manifest = load_manifest(release_path / "manifest.json")
    return apply_file_release(
        rag_root, manifest=manifest, blob_root=release_path / "blobs"
    )
