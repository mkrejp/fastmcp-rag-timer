"""Authenticated live diff-push API on the reader (in-process apply).

Routes (Bearer ``RAG_SYNC_TOKEN``)::

    GET  /sync/v1/watermark
    GET  /sync/v1/catalog
    POST /sync/v1/push/begin
    PUT  /sync/v1/push/chunk
    POST /sync/v1/push/commit
    POST /sync/v1/push/abort
    GET  /sync/v1/push/status
    POST /sync/v1/rollback/{release_id}   # soft rollback restore

Vectors travel with the push; the reader never re-embeds.
"""

from __future__ import annotations

import hmac
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient

from rag.app_mode import is_reader_mode, resolve_app_mode
from rag.config import resolve_sync_token
from rag.logging_setup import get_logger
from rag.publish.apply import apply_file_release
from rag.publish.diff import catalog_fingerprint
from rag.publish.manifest import ReleaseManifest
from rag.publish.qdrant_sync import POINTS_REL_PATH
from rag.publish.release_state import read_release_state
from rag.publish.sync_chunk import assemble_blob_files, assemble_points_jsonl
from rag.publish.sync_constants import ASSEMBLED_DIR_NAME, BLOBS_STAGE, POINTS_STAGE
from rag.publish.sync_rollback import restore_rollback, snapshot_touched_docs
from rag.publish.sync_session import (
    STATUS_ABORTED,
    STATUS_COMMITTED,
    STATUS_COMMITTING,
    STATUS_RECEIVING,
    abort_session,
    active_sessions,
    begin_session,
    list_received_chunks,
    load_session,
    mark_status,
    session_dir,
    store_chunk,
)
from rag.storage import RagStorage

logger = get_logger("rag.publish.sync_api")

router = APIRouter(prefix="/sync/v1", tags=["sync"])
_bearer = HTTPBearer(auto_error=False)


class BeginPushBody(BaseModel):
    """POST /push/begin payload wrapping a release manifest."""

    model_config = ConfigDict(extra="forbid")

    manifest: dict[str, Any]


class CommitPushBody(BaseModel):
    """Optional commit hints."""

    model_config = ConfigDict(extra="forbid")

    release_id: str
    source: str | None = None


class AbortPushBody(BaseModel):
    """Abort a receiving session."""

    model_config = ConfigDict(extra="forbid")

    release_id: str


class ChunkQuery(BaseModel):
    """Query params for PUT /push/chunk when not embedded in the body header."""

    model_config = ConfigDict(extra="forbid")

    release_id: str = Field(..., min_length=1)


def require_sync_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """Require ``Authorization: Bearer`` matching ``RAG_SYNC_TOKEN``."""
    expected = resolve_sync_token()
    if expected is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG_SYNC_TOKEN is not configured",
        )
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not credentials.credentials
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    provided = credentials.credentials.encode("utf-8")
    wanted = expected.encode("utf-8")
    if not hmac.compare_digest(provided, wanted):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return expected


def _require_reader() -> None:
    """Reject sync apply routes when not running as reader."""
    if not is_reader_mode():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Live sync API is only enabled for APP_MODE=reader "
                f"(current={resolve_app_mode()})"
            ),
        )


def _storage(request: Request) -> RagStorage:
    """Return the app RagStorage."""
    return request.app.state.storage


def _qdrant(request: Request) -> QdrantClient:
    """Return the already-open app Qdrant client (never open a second path=)."""
    return request.app.state.qdrant


def _rag_root(request: Request) -> Path:
    """Return the RAG data root path."""
    return Path(request.app.state.rag_data_root)


@router.get("/watermark")
def get_watermark(
    request: Request,
    _token: str = Depends(require_sync_token),
) -> dict[str, Any]:
    """Return the reader's current ``release.json`` watermark (or empty)."""
    _require_reader()
    state = read_release_state(_storage(request).root)
    if not state:
        return {
            "release_id": None,
            "parent_release_id": None,
            "applied_at": None,
            "present": False,
        }
    return {
        "release_id": state.get("release_id"),
        "parent_release_id": state.get("parent_release_id"),
        "applied_at": state.get("applied_at"),
        "present": True,
        "extra": {
            k: v
            for k, v in state.items()
            if k
            not in {
                "release_id",
                "parent_release_id",
                "applied_at",
                "source",
            }
        },
        "source": state.get("source"),
    }


@router.get("/catalog")
def get_catalog(
    request: Request,
    _token: str = Depends(require_sync_token),
) -> dict[str, Any]:
    """Return ``doc_id → sha256`` fingerprint of the reader catalog."""
    _require_reader()
    storage = _storage(request)
    fp = catalog_fingerprint(storage)
    return {"documents": fp, "count": len(fp)}


@router.post("/push/begin")
def push_begin(
    body: BeginPushBody,
    request: Request,
    _token: str = Depends(require_sync_token),
) -> dict[str, Any]:
    """Start (or resume) a push; ``parent_release_id`` must match watermark."""
    _require_reader()
    try:
        manifest = ReleaseManifest.from_dict(body.manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid manifest: {exc}") from exc

    watermark = read_release_state(_storage(request).root)
    current_id = (
        str(watermark["release_id"])
        if watermark and watermark.get("release_id")
        else None
    )
    parent = manifest.parent_release_id
    if current_id != parent:
        raise HTTPException(
            status_code=409,
            detail=(
                f"parent_release_id mismatch: manifest parent={parent!r} "
                f"reader watermark={current_id!r}"
            ),
        )

    try:
        session = begin_session(_rag_root(request), manifest=manifest)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "release_id": session["release_id"],
        "status": session["status"],
        "parent_release_id": session.get("parent_release_id"),
        "chunks_received": len(session.get("chunks") or {}),
    }


@router.put("/push/chunk")
async def push_chunk(
    request: Request,
    release_id: str,
    _token: str = Depends(require_sync_token),
) -> dict[str, Any]:
    """Store one chunk under ``.sync/incoming/<release_id>/`` (idempotent)."""
    _require_reader()
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="empty chunk body")
    try:
        receipt = store_chunk(_rag_root(request), release_id=release_id, raw=raw)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return receipt


@router.get("/push/status")
def push_status(
    request: Request,
    release_id: str | None = None,
    _token: str = Depends(require_sync_token),
) -> dict[str, Any]:
    """Return one session or a list of active incoming sessions."""
    _require_reader()
    root = _rag_root(request)
    if release_id:
        session = load_session(root, release_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return {
            "release_id": session.get("release_id"),
            "status": session.get("status"),
            "parent_release_id": session.get("parent_release_id"),
            "chunks_received": len(session.get("chunks") or {}),
            "begun_at": session.get("begun_at"),
            "updated_at": session.get("updated_at"),
            "chunks": session.get("chunks") or {},
            "commit": session.get("commit"),
        }
    return {"sessions": active_sessions(root)}


@router.post("/push/abort")
def push_abort(
    body: AbortPushBody,
    request: Request,
    _token: str = Depends(require_sync_token),
) -> dict[str, Any]:
    """Abort a receiving session and drop staged chunks."""
    _require_reader()
    return abort_session(_rag_root(request), body.release_id)


@router.post("/push/commit")
def push_commit(
    body: CommitPushBody,
    request: Request,
    _token: str = Depends(require_sync_token),
) -> dict[str, Any]:
    """Assemble chunks, soft-rollback snapshot, apply in-process, ACK watermark."""
    _require_reader()
    root = _rag_root(request)
    release_id = body.release_id
    session = load_session(root, release_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    if session.get("status") == STATUS_COMMITTED:
        return {
            "release_id": release_id,
            "status": STATUS_COMMITTED,
            "duplicate": True,
            "commit": session.get("commit"),
        }
    if session.get("status") == STATUS_ABORTED:
        raise HTTPException(status_code=409, detail="session was aborted")
    if session.get("status") not in {STATUS_RECEIVING, STATUS_COMMITTING}:
        raise HTTPException(
            status_code=409, detail=f"cannot commit status={session.get('status')}"
        )

    # Re-check watermark against parent before applying.
    watermark = read_release_state(_storage(request).root)
    current_id = (
        str(watermark["release_id"])
        if watermark and watermark.get("release_id")
        else None
    )
    parent = session.get("parent_release_id")
    if current_id != parent:
        raise HTTPException(
            status_code=409,
            detail=(
                f"watermark changed during push: expected parent={parent!r} "
                f"have={current_id!r}"
            ),
        )

    try:
        mark_status(root, release_id, STATUS_COMMITTING)
        manifest = ReleaseManifest.from_dict(session["manifest"])
        chunks = list_received_chunks(root, release_id)
        assembled = session_dir(root, release_id) / ASSEMBLED_DIR_NAME
        if assembled.exists():
            shutil.rmtree(assembled, ignore_errors=True)
        assembled.mkdir(parents=True, exist_ok=True)
        blob_root = assembled / "blobs"
        blob_root.mkdir(parents=True, exist_ok=True)

        blob_chunks = [c for c in chunks if c.stage == BLOBS_STAGE]
        point_chunks = [c for c in chunks if c.stage == POINTS_STAGE]
        written_paths = assemble_blob_files(blob_chunks, blob_root)

        # Every files_added path must be present after assemble.
        missing = [
            change.path
            for change in manifest.files_added
            if change.path not in set(written_paths)
        ]
        if missing:
            raise ValueError(f"missing assembled blobs: {missing[:10]}")

        points_path = assemble_points_jsonl(
            point_chunks, assembled / POINTS_REL_PATH
        )
        if manifest.catalog_doc_ids_upserted and points_path is None:
            raise ValueError(
                "manifest upserts docs but no points chunks were received"
            )

        touch_ids = sorted(
            set(manifest.catalog_doc_ids_upserted)
            | set(manifest.catalog_doc_ids_deleted)
        )
        snapshot_touched_docs(
            root,
            client=_qdrant(request),
            release_id=release_id,
            doc_ids=touch_ids,
        )

        state = apply_file_release(
            root,
            manifest=manifest,
            blob_root=blob_root,
            points_jsonl=points_path,
            source=body.source or manifest.writer_host or "live-diff-push",
            client=_qdrant(request),
        )

        # Drop assembled staging; keep session marker as committed.
        shutil.rmtree(assembled, ignore_errors=True)
        chunks_dir = session_dir(root, release_id) / "chunks"
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir, ignore_errors=True)

        commit_info = {
            "applied_release_id": state.get("release_id"),
            "files_copied": state.get("files_copied"),
            "files_deleted": state.get("files_deleted"),
            "points_upserted": state.get("points_upserted"),
            "points_deleted": state.get("points_deleted"),
            "blobs_assembled": len(written_paths),
            "points_chunks": len(point_chunks),
        }
        mark_status(
            root,
            release_id,
            STATUS_COMMITTED,
            commit=commit_info,
            chunks={},
        )
        logger.info("Live sync committed release=%s %s", release_id, commit_info)
        return {
            "release_id": release_id,
            "status": STATUS_COMMITTED,
            "duplicate": False,
            "commit": commit_info,
            "release": state,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface apply failures
        logger.exception("Live sync commit failed release=%s", release_id)
        try:
            mark_status(root, release_id, STATUS_RECEIVING, last_error=str(exc))
        except Exception:  # noqa: BLE001
            pass
        raise HTTPException(status_code=500, detail=f"commit failed: {exc}") from exc


@router.post("/rollback/{release_id}")
def rollback_release(
    release_id: str,
    request: Request,
    _token: str = Depends(require_sync_token),
) -> dict[str, Any]:
    """Restore the soft rollback snapshot taken before applying ``release_id``."""
    _require_reader()
    try:
        result = restore_rollback(
            _rag_root(request),
            client=_qdrant(request),
            release_id=release_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"rollback failed: {exc}") from exc
    return result
