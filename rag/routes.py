"""HTTP routes for document ingest and retrieval."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from rag.accounting import accounting_source, get_global_store
from rag.auth import require_api_token, require_embed_access
from rag.config import COLLECTION_NAME
from rag.ingest import ingest_bytes, meta_to_public
from rag.logging_setup import get_logger
from rag.processing import is_supported_filename
from rag.remote import (
    RemoteFetchError,
    RemoteTooLargeError,
    UnsafeRemoteUrlError,
    fetch_remote_bytes,
    filename_from_url,
)
from rag.storage import RagStorage

router = APIRouter()
logger = get_logger("rag.routes")


class IngestJsonRequest(BaseModel):
    """Body for remote JSON/GeoJSON ingest."""

    url: str = Field(..., min_length=8, max_length=2048)


def _storage(request: Request) -> RagStorage:
    """Return the request-scoped storage instance."""
    return request.app.state.storage


def _qdrant(request: Request) -> QdrantClient:
    """Return the request-scoped Qdrant client."""
    return request.app.state.qdrant


def _rag_data_root(request: Request) -> str:
    """Return the configured RAG data root."""
    return str(request.app.state.rag_data_root)


def _record_http(*, route: str, ok: bool, bytes_in: int = 0, error: str | None = None) -> None:
    """Record an HTTP route invocation when accounting is configured."""
    store = get_global_store()
    if store is None:
        return
    store.record_http(route=route, ok=ok, bytes_in=bytes_in, error=error)


def _ingest_upload(
    request: Request,
    filename: str,
    data: bytes,
    *,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Shared ingest path for HTTP uploads."""
    try:
        with accounting_source("ingest_http"):
            return ingest_bytes(
                storage=_storage(request),
                qdrant=_qdrant(request),
                collection_name=COLLECTION_NAME,
                filename=filename,
                data=data,
                source_url=source_url,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface ingest failures to clients
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    """Liveness probe for container orchestration."""
    try:
        body = {"status": "ok", "rag_data_root": _rag_data_root(request)}
        _record_http(route="health", ok=True)
        return body
    except Exception as exc:  # noqa: BLE001 - still report health accounting failures
        _record_http(route="health", ok=False, error=str(exc))
        raise


@router.post("/upload", dependencies=[Depends(require_embed_access)])
async def upload_document(
    request: Request, file: UploadFile = File(...)
) -> dict[str, Any]:
    """Upload a PDF or text document into persistent RAG storage."""
    filename = file.filename or ""
    if not is_supported_filename(filename):
        _record_http(route="upload", ok=False, error="unsupported filename")
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, Markdown, JSON, and GeoJSON files are supported.",
        )
    data = await file.read()
    try:
        result = _ingest_upload(request, filename, data)
        _record_http(route="upload", ok=True, bytes_in=len(data))
        return result
    except HTTPException as exc:
        _record_http(route="upload", ok=False, bytes_in=len(data), error=str(exc.detail))
        raise


@router.post("/upload-pdf", dependencies=[Depends(require_embed_access)])
async def upload_pdf(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """Compatibility alias for PDF uploads."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        _record_http(route="upload_pdf", ok=False, error="unsupported filename")
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported via this endpoint.",
        )
    data = await file.read()
    try:
        result = _ingest_upload(request, filename, data)
        _record_http(route="upload_pdf", ok=True, bytes_in=len(data))
        return result
    except HTTPException as exc:
        _record_http(route="upload_pdf", ok=False, bytes_in=len(data), error=str(exc.detail))
        raise


@router.post("/ingest-json")
def ingest_json_url(
    body: IngestJsonRequest,
    request: Request,
    _token: str = Depends(require_api_token),
) -> dict[str, Any]:
    """Fetch remote GeoJSON/JSON and index feature attributes into local RAG storage.

    Accepts any similar GeoJSON (FeatureCollection, Feature, geometry, or a list of
    those objects), including GitHub blob URLs such as ``kraje.json`` / ``obce.json``.
    Coordinate arrays are not embedded.
    """
    logger.info("Remote JSON ingest requested url=%s", body.url)
    try:
        data, final_url = fetch_remote_bytes(body.url)
    except UnsafeRemoteUrlError as exc:
        logger.warning("Rejected remote URL: %s", exc)
        _record_http(route="ingest_json", ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RemoteTooLargeError as exc:
        logger.warning("Remote JSON too large: %s", exc)
        _record_http(route="ingest_json", ok=False, error=str(exc))
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except RemoteFetchError as exc:
        logger.error("Remote fetch failed: %s", exc)
        _record_http(route="ingest_json", ok=False, error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    filename = filename_from_url(final_url)
    lowered = filename.lower()
    if not (lowered.endswith(".json") or lowered.endswith(".geojson")):
        filename = f"{filename}.json"
    logger.info(
        "Fetched remote JSON bytes=%s filename=%s final_url=%s",
        len(data),
        filename,
        final_url,
    )
    try:
        result = _ingest_upload(request, filename, data, source_url=final_url)
        _record_http(route="ingest_json", ok=True, bytes_in=len(data))
        return result
    except HTTPException as exc:
        _record_http(route="ingest_json", ok=False, bytes_in=len(data), error=str(exc.detail))
        raise


@router.post("/upload-media")
async def upload_media(
    request: Request, file: UploadFile = File(...)
) -> dict[str, Any]:
    """Store a media asset under originals without extraction or indexing."""
    filename = file.filename or "media.bin"
    data = await file.read()
    try:
        doc_id = str(uuid.uuid4())
        path = _storage(request).save_original(doc_id, filename, data)
        _record_http(route="upload_media", ok=True, bytes_in=len(data))
        return {"status": "Saved", "path": str(path), "doc_id": doc_id}
    except Exception as exc:  # noqa: BLE001 - surface storage failures
        _record_http(route="upload_media", ok=False, bytes_in=len(data), error=str(exc))
        raise


@router.get("/documents")
def list_documents_http(request: Request) -> dict[str, Any]:
    """Return the persistent document catalog."""
    try:
        body = {"documents": _storage(request).list_documents()}
        _record_http(route="documents_list", ok=True)
        return body
    except Exception as exc:  # noqa: BLE001 - meter failures
        _record_http(route="documents_list", ok=False, error=str(exc))
        raise


@router.get("/documents/{doc_id}")
def get_document_http(doc_id: str, request: Request) -> dict[str, Any]:
    """Return metadata for a stored document."""
    meta = _storage(request).read_meta(doc_id)
    if meta is None:
        _record_http(route="documents_get", ok=False, error="not found")
        raise HTTPException(status_code=404, detail="Document not found")
    _record_http(route="documents_get", ok=True)
    return meta_to_public(meta)


@router.get("/documents/{doc_id}/text")
def get_document_text_http(doc_id: str, request: Request) -> dict[str, Any]:
    """Return processed full text without re-parsing the original file."""
    storage = _storage(request)
    text = storage.read_fulltext(doc_id)
    if text is None:
        _record_http(route="documents_text", ok=False, error="not found")
        raise HTTPException(status_code=404, detail="Document text not found")
    meta = storage.read_meta(doc_id)
    _record_http(route="documents_text", ok=True, bytes_in=len(text.encode("utf-8")))
    return {
        "doc_id": doc_id,
        "filename": meta.filename if meta else None,
        "text": text,
    }
