"""Admin/browse APIs for volume contents, MCP status, and recent logs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from qdrant_client import QdrantClient

from rag.accounting import AccountingStore, utc_today
from rag.config import COLLECTION_NAME
from rag.embeddings import EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDING_URL
from rag.google_oauth import require_google_user
from rag.logging_setup import get_log_handler, get_logger
from rag.storage import RagStorage

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

router = APIRouter(
    prefix="/api/admin",
    tags=["admin"],
    dependencies=[Depends(require_google_user)],
)
logger = get_logger("rag.admin")


def _storage(request: Request) -> RagStorage:
    """Return the request-scoped storage instance."""
    return request.app.state.storage


def _qdrant(request: Request) -> QdrantClient:
    """Return the request-scoped Qdrant client."""
    return request.app.state.qdrant


def _accounting(request: Request) -> AccountingStore:
    """Return the request-scoped accounting store."""
    return request.app.state.accounting


def _dir_size(path: Path) -> int:
    """Return recursive byte size of ``path``, or 0 if missing."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _list_tree(path: Path, *, max_entries: int = 200) -> list[dict[str, Any]]:
    """List files/directories under ``path`` (non-recursive children first)."""
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        children = sorted(
            path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except OSError as exc:
        logger.warning("Failed to list %s: %s", path, exc)
        return []
    for child in children[:max_entries]:
        item: dict[str, Any] = {
            "name": child.name,
            "path": str(child),
            "is_dir": child.is_dir(),
        }
        try:
            if child.is_file():
                item["size_bytes"] = child.stat().st_size
            else:
                item["size_bytes"] = _dir_size(child)
                item["children"] = len(list(child.iterdir())) if child.is_dir() else 0
        except OSError:
            item["size_bytes"] = None
        entries.append(item)
    return entries


def _collection_stats(qdrant: QdrantClient, collection_name: str) -> dict[str, Any]:
    """Return Qdrant collection stats used by the RAG search tools."""
    try:
        if not qdrant.collection_exists(collection_name):
            return {
                "name": collection_name,
                "exists": False,
                "points_count": 0,
                "vector_size": None,
            }
        info = qdrant.get_collection(collection_name)
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", None)
        return {
            "name": collection_name,
            "exists": True,
            "points_count": int(info.points_count or 0),
            "vector_size": int(size) if size is not None else None,
            "status": str(info.status),
        }
    except Exception as exc:  # noqa: BLE001 - surface as structured status
        logger.exception("Failed to read Qdrant collection stats")
        return {
            "name": collection_name,
            "exists": False,
            "error": str(exc),
        }


@router.get("/status")
def admin_status(request: Request) -> dict[str, Any]:
    """Return overall service, storage, MCP, and vector-index status."""
    storage = _storage(request)
    docs = storage.list_documents()
    mcp_tools = list(getattr(request.app.state, "mcp_tools", []))
    mcp_mounted = bool(getattr(request.app.state, "mcp_mounted", False))
    payload: dict[str, Any] = {
        "service": "Lightweight RAG Engine",
        "rag_data_root": str(request.app.state.rag_data_root),
        "documents": {
            "count": len(docs),
            "chunk_count_total": sum(int(d.get("chunk_count") or 0) for d in docs),
        },
        "volume": {
            "root": str(storage.root),
            "root_exists": storage.root.exists(),
            "size_bytes": _dir_size(storage.root),
            "originals_bytes": _dir_size(storage.originals_dir),
            "processed_bytes": _dir_size(storage.processed_dir),
            "qdrant_bytes": _dir_size(storage.qdrant_dir),
            "index_exists": storage.index_path.exists(),
            "accnt_root": str(getattr(request.app.state, "accnt_data_root", "")),
        },
        "mcp": {
            "mounted": mcp_mounted,
            "mount_path": "/mcp",
            "server_name": "Client RAG Node",
            "transport": "streamable-http",
            "tools": mcp_tools,
            "tool_count": len(mcp_tools),
        },
        "embeddings": {
            "model": EMBEDDING_MODEL,
            "url": EMBEDDING_URL,
            "dimensions": EMBEDDING_DIM,
        },
        "vector_store": _collection_stats(_qdrant(request), COLLECTION_NAME),
    }
    logger.info(
        "Admin status requested: docs=%s vectors=%s mcp_tools=%s",
        payload["documents"]["count"],
        payload["vector_store"].get("points_count"),
        payload["mcp"]["tool_count"],
    )
    return payload


@router.get("/documents")
def admin_documents(request: Request) -> dict[str, Any]:
    """Return the RAG catalog with disk sizes for browse UI."""
    storage = _storage(request)
    enriched: list[dict[str, Any]] = []
    for entry in storage.list_documents():
        doc_id = str(entry.get("doc_id", ""))
        item = dict(entry)
        item["originals_bytes"] = _dir_size(storage.original_dir(doc_id))
        item["processed_bytes"] = _dir_size(storage.processed_doc_dir(doc_id))
        item["has_fulltext"] = (
            storage.processed_doc_dir(doc_id) / "fulltext.txt"
        ).exists()
        item["has_chunks"] = (
            storage.processed_doc_dir(doc_id) / "chunks.json"
        ).exists()
        enriched.append(item)
    return {"documents": enriched, "count": len(enriched)}


@router.get("/documents/{doc_id}")
def admin_document_detail(
    doc_id: str,
    request: Request,
    text_limit: int = Query(4000, ge=200, le=50000),
    chunk_limit: int = Query(20, ge=1, le=200),
) -> dict[str, Any]:
    """Return one document's metadata, text preview, and chunk previews."""
    storage = _storage(request)
    meta = storage.read_meta(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")
    fulltext = storage.read_fulltext(doc_id) or ""
    chunks = storage.read_chunks(doc_id) or []
    originals = _list_tree(storage.original_dir(doc_id))
    truncated = len(fulltext) > text_limit
    preview_chunks = []
    for chunk in chunks[:chunk_limit]:
        text = str(chunk.get("text", ""))
        preview_chunks.append(
            {
                "index": chunk.get("index"),
                "page": chunk.get("page"),
                "chars": len(text),
                "text_preview": text if len(text) <= 500 else text[:500] + "…",
            }
        )
    return {
        "meta": meta.to_dict(),
        "originals": originals,
        "fulltext": {
            "chars": len(fulltext),
            "truncated": truncated,
            "preview": fulltext[:text_limit] + ("…" if truncated else ""),
        },
        "chunks": {
            "total": len(chunks),
            "returned": len(preview_chunks),
            "items": preview_chunks,
        },
    }


@router.get("/volume")
def admin_volume(request: Request) -> dict[str, Any]:
    """Browse top-level directories on the RAG data volume."""
    storage = _storage(request)
    root = storage.root
    return {
        "root": str(root),
        "exists": root.exists(),
        "size_bytes": _dir_size(root),
        "entries": {
            "originals": _list_tree(storage.originals_dir),
            "processed": _list_tree(storage.processed_dir),
            "qdrant": _list_tree(storage.qdrant_dir),
            "index": {
                "path": str(storage.index_path),
                "exists": storage.index_path.exists(),
                "size_bytes": (
                    storage.index_path.stat().st_size
                    if storage.index_path.exists()
                    else 0
                ),
            },
        },
    }


@router.get("/mcp")
def admin_mcp(request: Request) -> dict[str, Any]:
    """Return MCP mount and tool registration status."""
    tools = list(getattr(request.app.state, "mcp_tools", []))
    return {
        "mounted": bool(getattr(request.app.state, "mcp_mounted", False)),
        "mount_path": "/mcp",
        "server_name": "Client RAG Node",
        "transport": "streamable-http",
        "stateless_http": True,
        "tools": [
            {
                "name": name,
                "description": getattr(request.app.state, "mcp_tool_docs", {}).get(
                    name, ""
                ),
            }
            for name in tools
        ],
    }


@router.get("/logs")
def admin_logs(
    limit: int = Query(100, ge=1, le=500),
    level: str | None = Query(None),
) -> dict[str, Any]:
    """Return recent application log lines from the in-memory ring buffer."""
    handler = get_log_handler()
    entries = handler.snapshot(limit=limit, level=level)
    return {"count": len(entries), "limit": limit, "level": level, "entries": entries}


@router.get("/accounts")
def admin_accounts(
    request: Request,
    days: int = Query(30, ge=1, le=366),
) -> dict[str, Any]:
    """Return recent daily accounting rollups plus aggregate totals."""
    store = _accounting(request)
    summary = store.summarize(days=days)
    summary["today"] = utc_today()
    return summary


@router.get("/accounts/{date}")
def admin_accounts_day(
    date: str,
    request: Request,
    include_events: bool = Query(False),
    event_limit: int = Query(200, ge=1, le=2000),
    event_kind: str | None = Query(
        None,
        description="Optional event kind filter (embedding, mcp, http, outbound).",
    ),
) -> dict[str, Any]:
    """Return one UTC day's accounting rollup and optional event tail."""
    if not _DATE_RE.match(date):
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
    store = _accounting(request)
    day = store.read_day(date)
    payload: dict[str, Any] = {
        "date": date,
        "accnt_root": str(store.root),
        "day": day,
        "exists": store.day_path(date).exists(),
    }
    if include_events:
        kind = (event_kind or "").strip() or None
        events = store.read_events(date, limit=event_limit, kind=kind)
        payload["events"] = events
        payload["event_limit"] = event_limit
        payload["event_kind"] = kind
        payload["event_count"] = len(events)
    return payload
