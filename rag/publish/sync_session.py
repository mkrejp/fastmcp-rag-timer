"""On-disk session state for an in-flight live sync push."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any

from rag.publish.manifest import ReleaseManifest
from rag.publish.sync_chunk import PackedChunk, decode_chunk, read_chunk_file
from rag.publish.sync_constants import (
    ASSEMBLED_DIR_NAME,
    CHUNKS_DIR_NAME,
    INCOMING_DIR_NAME,
    SESSION_FILENAME,
    STAGE_NAMES,
    SYNC_DIR_NAME,
)
from rag.storage import utc_now_iso

_LOCK = threading.RLock()

STATUS_RECEIVING = "receiving"
STATUS_COMMITTING = "committing"
STATUS_COMMITTED = "committed"
STATUS_ABORTED = "aborted"
STATUS_DRY_RUN_OK = "dry_run_ok"
STATUS_DRY_RUN_FAILED = "dry_run_failed"

# Only one live push may occupy the reader at a time.
ACTIVE_PUSH_STATUSES = frozenset({STATUS_RECEIVING, STATUS_COMMITTING})


class PushConflictError(RuntimeError):
    """Raised when another push session already holds the global reader lock."""

    def __init__(self, active_release_id: str) -> None:
        """Record the blocking ``active_release_id``."""
        self.active_release_id = active_release_id
        super().__init__(
            f"another push session is active: release_id={active_release_id!r} "
            f"(status receiving/committing); abort or commit it first"
        )


def sync_incoming_root(rag_root: Path | str) -> Path:
    """Return ``.sync/incoming`` under ``rag_root``."""
    return Path(rag_root) / SYNC_DIR_NAME / INCOMING_DIR_NAME


def session_dir(rag_root: Path | str, release_id: str) -> Path:
    """Return the incoming directory for one release."""
    return sync_incoming_root(rag_root) / release_id


def session_path(rag_root: Path | str, release_id: str) -> Path:
    """Return ``session.json`` for a release."""
    return session_dir(rag_root, release_id) / SESSION_FILENAME


def chunk_path(rag_root: Path | str, release_id: str, stage: str, seq: int) -> Path:
    """Return the on-disk path for ``(release_id, stage, seq)``."""
    return (
        session_dir(rag_root, release_id) / CHUNKS_DIR_NAME / f"{stage}.{seq:05d}.chunk"
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def load_session(rag_root: Path | str, release_id: str) -> dict[str, Any] | None:
    """Load session JSON or ``None`` when missing."""
    path = session_path(rag_root, release_id)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def save_session(rag_root: Path | str, session: dict[str, Any]) -> dict[str, Any]:
    """Persist session and return it."""
    release_id = str(session["release_id"])
    session["updated_at"] = utc_now_iso()
    _atomic_write_json(session_path(rag_root, release_id), session)
    return session


def find_active_push_session(
    rag_root: Path | str,
    *,
    exclude_release_id: str | None = None,
) -> dict[str, Any] | None:
    """Return the first session in ``receiving``/``committing``, if any."""
    base = sync_incoming_root(rag_root)
    if not base.is_dir():
        return None
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if exclude_release_id and child.name == exclude_release_id:
            continue
        session = load_session(rag_root, child.name)
        if session and session.get("status") in ACTIVE_PUSH_STATUSES:
            return session
    return None


def begin_session(
    rag_root: Path | str,
    *,
    manifest: ReleaseManifest,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create or resume a receiving session for ``manifest.release_id``.

    Enforces a **single global** active push on the reader: a second
    ``begin`` for a different ``release_id`` while another session is
    ``receiving`` or ``committing`` raises :class:`PushConflictError`.
    Terminal statuses (``committed``, ``aborted``, ``dry_run_*``) do not
    block. Same-``release_id`` resume remains idempotent.
    """
    with _LOCK:
        existing = load_session(rag_root, manifest.release_id)
        if existing is not None:
            if existing.get("status") == STATUS_COMMITTED:
                raise ValueError(f"release {manifest.release_id} already committed")
            if existing.get("status") in {
                STATUS_ABORTED,
                STATUS_DRY_RUN_OK,
                STATUS_DRY_RUN_FAILED,
            }:
                # Allow re-begin after abort / dry-run by clearing the tree.
                shutil.rmtree(
                    session_dir(rag_root, manifest.release_id), ignore_errors=True
                )
            elif existing.get("status") in ACTIVE_PUSH_STATUSES:
                # Idempotent begin: same release_id resumes (keep dry_run flag).
                return existing

        blocker = find_active_push_session(
            rag_root, exclude_release_id=manifest.release_id
        )
        if blocker is not None:
            raise PushConflictError(str(blocker.get("release_id") or "unknown"))

        root = session_dir(rag_root, manifest.release_id)
        root.mkdir(parents=True, exist_ok=True)
        (root / CHUNKS_DIR_NAME).mkdir(parents=True, exist_ok=True)
        session: dict[str, Any] = {
            "release_id": manifest.release_id,
            "parent_release_id": manifest.parent_release_id,
            "status": STATUS_RECEIVING,
            "dry_run": bool(dry_run),
            "begun_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "manifest": manifest.to_dict(),
            "chunks": {},
        }
        return save_session(rag_root, session)


def store_chunk(
    rag_root: Path | str,
    *,
    release_id: str,
    raw: bytes,
) -> dict[str, Any]:
    """Validate and store a chunk; idempotent on ``(release_id, stage, seq)``.

    Returns a receipt dict. Rejects conflicting sha256 for the same key.
    """
    chunk = decode_chunk(raw)
    if chunk.release_id and chunk.release_id != release_id:
        raise ValueError(
            f"chunk release_id {chunk.release_id!r} != session {release_id!r}"
        )
    if chunk.stage not in STAGE_NAMES:
        raise ValueError(f"unknown stage {chunk.stage!r}")

    with _LOCK:
        session = load_session(rag_root, release_id)
        if session is None:
            raise FileNotFoundError(f"no push session for {release_id}")
        if session.get("status") not in {STATUS_RECEIVING}:
            raise ValueError(
                f"session {release_id} is {session.get('status')}, not receiving"
            )

        key = f"{chunk.stage}:{chunk.seq}"
        prior = (session.get("chunks") or {}).get(key)
        dest = chunk_path(rag_root, release_id, chunk.stage, chunk.seq)
        if prior is not None:
            if str(prior.get("sha256")) != chunk.sha256:
                raise ValueError(
                    f"idempotent conflict for {key}: "
                    f"have {prior.get('sha256')} got {chunk.sha256}"
                )
            return {
                "release_id": release_id,
                "stage": chunk.stage,
                "seq": chunk.seq,
                "sha256": chunk.sha256,
                "bytes": len(chunk.payload),
                "duplicate": True,
            }

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        session.setdefault("chunks", {})[key] = {
            "stage": chunk.stage,
            "seq": chunk.seq,
            "sha256": chunk.sha256,
            "bytes": len(chunk.payload),
            "entries": len(chunk.entries),
            "path": dest.name,
        }
        save_session(rag_root, session)
        return {
            "release_id": release_id,
            "stage": chunk.stage,
            "seq": chunk.seq,
            "sha256": chunk.sha256,
            "bytes": len(chunk.payload),
            "duplicate": False,
        }


def list_received_chunks(rag_root: Path | str, release_id: str) -> list[PackedChunk]:
    """Load every stored chunk for a session (sorted by stage then seq)."""
    session = load_session(rag_root, release_id)
    if session is None:
        return []
    chunks: list[PackedChunk] = []
    for key, meta in sorted((session.get("chunks") or {}).items()):
        stage = str(meta.get("stage") or key.split(":", 1)[0])
        seq = int(
            meta.get("seq") if meta.get("seq") is not None else key.split(":")[-1]
        )
        path = chunk_path(rag_root, release_id, stage, seq)
        if path.is_file():
            chunks.append(read_chunk_file(path))
    return chunks


def mark_status(
    rag_root: Path | str, release_id: str, status: str, **extra: Any
) -> dict[str, Any]:
    """Update session status and optional extra fields."""
    with _LOCK:
        session = load_session(rag_root, release_id)
        if session is None:
            raise FileNotFoundError(f"no push session for {release_id}")
        session["status"] = status
        session.update(extra)
        return save_session(rag_root, session)


def abort_session(rag_root: Path | str, release_id: str) -> dict[str, Any]:
    """Mark aborted and remove staging files (keep a tiny marker)."""
    with _LOCK:
        session = load_session(rag_root, release_id)
        if session is None:
            return {
                "release_id": release_id,
                "status": STATUS_ABORTED,
                "missing": True,
            }
        root = session_dir(rag_root, release_id)
        chunks_dir = root / CHUNKS_DIR_NAME
        assembled = root / ASSEMBLED_DIR_NAME
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir, ignore_errors=True)
        if assembled.exists():
            shutil.rmtree(assembled, ignore_errors=True)
        session["status"] = STATUS_ABORTED
        session["chunks"] = {}
        session["aborted_at"] = utc_now_iso()
        return save_session(rag_root, session)


def active_sessions(rag_root: Path | str) -> list[dict[str, Any]]:
    """Return summaries of all incoming sessions."""
    base = sync_incoming_root(rag_root)
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        session = load_session(rag_root, child.name)
        if session:
            out.append(
                {
                    "release_id": session.get("release_id"),
                    "status": session.get("status"),
                    "parent_release_id": session.get("parent_release_id"),
                    "chunks_received": len(session.get("chunks") or {}),
                    "begun_at": session.get("begun_at"),
                    "updated_at": session.get("updated_at"),
                }
            )
    return out
