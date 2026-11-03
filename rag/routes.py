"""HTTP routes for document ingest and retrieval."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from qdrant_client import QdrantClient

from rag.config import COLLECTION_NAME
from rag.ingest import ingest_bytes, meta_to_public
from rag.processing import is_supported_filename
from rag.storage import RagStorage

router = APIRouter()


def _storage(request: Request) -> RagStorage:
    """Return the request-scoped storage instance."""
    return request.app.state.storage


def _qdrant(request: Request) -> QdrantClient:
    """Return the request-scoped Qdrant client."""
    return request.app.state.qdrant


def _rag_data_root(request: Request) -> str:
    """Return the configured RAG data root."""
    return str(request.app.state.rag_data_root)


def _ingest_upload(request: Request, filename: str, data: bytes) -> dict[str, Any]:
    """Shared ingest path for HTTP uploads."""
    try:
        return ingest_bytes(
            storage=_storage(request),
            qdrant=_qdrant(request),
            collection_name=COLLECTION_NAME,
            filename=filename,
            data=data,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface ingest failures to clients
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    """Liveness probe for container orchestration."""
    return {"status": "ok", "rag_data_root": _rag_data_root(request)}


@router.post("/upload")
async def upload_document(
    request: Request, file: UploadFile = File(...)
) -> dict[str, Any]:
    """Upload a PDF or text document into persistent RAG storage."""
    filename = file.filename or ""
    if not is_supported_filename(filename):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, and Markdown files are supported.",
        )
    data = await file.read()
    return _ingest_upload(request, filename, data)


@router.post("/upload-pdf")
async def upload_pdf(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """Compatibility alias for PDF uploads."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported via this endpoint.",
        )
    data = await file.read()
    return _ingest_upload(request, filename, data)


@router.post("/upload-media")
async def upload_media(
    request: Request, file: UploadFile = File(...)
) -> dict[str, Any]:
    """Store a non-indexed media asset under originals (legacy endpoint)."""
    filename = file.filename or "media.bin"
    data = await file.read()
    doc_id = str(uuid.uuid4())
    path = _storage(request).save_original(doc_id, filename, data)
    return {"status": "Saved", "path": str(path), "doc_id": doc_id}


@router.get("/documents")
def list_documents_http(request: Request) -> dict[str, Any]:
    """Return the persistent document catalog."""
    return {"documents": _storage(request).list_documents()}


@router.get("/documents/{doc_id}")
def get_document_http(doc_id: str, request: Request) -> dict[str, Any]:
    """Return metadata for a stored document."""
    meta = _storage(request).read_meta(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return meta_to_public(meta)


@router.get("/documents/{doc_id}/text")
def get_document_text_http(doc_id: str, request: Request) -> dict[str, Any]:
    """Return processed full text without re-parsing the original file."""
    storage = _storage(request)
    text = storage.read_fulltext(doc_id)
    if text is None:
        raise HTTPException(status_code=404, detail="Document text not found")
    meta = storage.read_meta(doc_id)
    return {
        "doc_id": doc_id,
        "filename": meta.filename if meta else None,
        "text": text,
    }
