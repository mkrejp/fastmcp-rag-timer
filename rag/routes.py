"""HTTP routes for document ingest, retrieval, replacement, and deletion."""

from __future__ import annotations

import json
import uuid
from typing import Any, NoReturn

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient

from rag.accounting import accounting_source, get_global_store
from rag.auth import require_api_token, require_embed_access
from rag.chunked_upload import (
    ChunkConflictError,
    ChunkedUploadError,
    ChunkedUploadStore,
    UploadExpiredError,
    UploadNotFoundError,
)
from rag.config import COLLECTION_NAME
from rag.document_delete import (
    DocumentLinkedError,
    DocumentNotFoundError,
    delete_document,
)
from rag.document_patch import (
    DocumentPatchError,
    patch_document,
    patch_document_metadata,
)
from rag.document_replace import (
    DocumentReplaceError,
    DocumentVersionNotFoundError,
    list_document_versions,
    replace_document_content,
    restore_document_version,
)
from rag.document_supersede import supersede_document, unsupersede_document
from rag.embeddings import embedding_provider_status, validate_embedding_configuration
from rag.google_oauth import current_user
from rag.image_delivery import resolve_image_delivery
from rag.image_embed import image_embedding_enabled
from rag.ingest import ingest_bytes, meta_to_public
from rag.logging_setup import get_logger
from rag.node_info import node_info_payload
from rag.page_ids import parse_page_image_id
from rag.processing import is_supported_filename
from rag.remote import (
    RemoteFetchError,
    RemoteTooLargeError,
    UnsafeRemoteUrlError,
    fetch_remote_bytes,
    filename_from_url,
)
from rag.storage import RagStorage
from rag.verification_meta import DocumentMetadataError

MAX_IMAGE_HTTP_BYTES = 50 * 1024 * 1024

router = APIRouter()
logger = get_logger("rag.routes")


class UploadInitRequest(BaseModel):
    """Body for ``POST /upload/init``."""

    filename: str = Field(..., min_length=1, max_length=512)
    size: int = Field(..., gt=0)
    sha256: str = Field(..., min_length=64, max_length=64)
    content_type: str | None = Field(default=None, max_length=256)
    chunk_size: int | None = Field(
        default=None,
        gt=0,
        description="Bytes per chunk; server default is ~50 MiB.",
    )


class UploadInitResponse(BaseModel):
    """Response for ``POST /upload/init``."""

    upload_id: str
    chunk_size: int
    chunk_count: int
    expires_at: str


class UploadStatusResponse(BaseModel):
    """Progress for an in-progress chunked upload."""

    upload_id: str
    filename: str
    size: int
    chunk_size: int
    chunk_count: int
    received_indexes: list[int]
    bytes_received: int
    expires_at: str
    complete: bool


class IngestJsonRequest(BaseModel):
    """Body for remote JSON/GeoJSON ingest."""

    url: str = Field(..., min_length=8, max_length=2048)
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional verification / reliability payload fields.",
    )


class DeleteDocumentResponse(BaseModel):
    """Success body for ``DELETE /documents/{doc_id}``."""

    status: str = Field(..., description="Always ``deleted`` on success.")
    doc_id: str
    filename: str
    chunks_removed: int = Field(
        ...,
        description="Number of Qdrant points removed for this document.",
    )


class PatchDocumentRequest(BaseModel):
    """Body for ``PATCH /documents/{doc_id}``. Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    filename: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="Display name; suffix must match the stored document type.",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom metadata to merge. Null values remove a key.",
    )


class PatchDocumentMetadataResponse(BaseModel):
    """Success body for ``PATCH /documents/{doc_id}/metadata``."""

    status: str = Field(..., description="Always ``updated`` on success.")
    doc_id: str
    filename: str
    updated_fields: list[str]
    deleted_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Whitelist keys removed via JSON null (Qdrant delete_payload + "
            "catalog extra pop)."
        ),
    )
    points_updated: int
    figures_updated: int
    truncated: list[str] = Field(
        default_factory=list,
        description=(
            "Field paths clipped to spec limits (e.g. overeni_souhrn, "
            "verification[0].quote, verified_quote)."
        ),
    )
    meta: dict[str, Any]


class SupersedeDocumentRequest(BaseModel):
    """Body for ``POST /documents/{doc_id}/supersede``."""

    model_config = ConfigDict(extra="forbid")

    new_doc_id: str = Field(..., min_length=1, max_length=64)
    correction_note: str | None = Field(
        default=None,
        description=(
            "Optional reason for the replacement. Longer than 500 characters "
            "is truncated and listed in ``truncated``."
        ),
    )


class SupersedeDocumentResponse(BaseModel):
    """Success body for ``POST /documents/{doc_id}/supersede``."""

    status: str = Field(..., description="Always ``updated`` on success.")
    old_doc_id: str
    new_doc_id: str
    doc_status: str
    superseded_by: str
    supersedes: list[str]
    correction_note: str | None = None
    truncated: list[str] = Field(default_factory=list)
    old_points_updated: int
    old_figures_updated: int
    new_points_updated: int
    new_figures_updated: int
    old: dict[str, Any]
    new: dict[str, Any]


class UnsupersedeDocumentResponse(BaseModel):
    """Success body for ``POST /documents/{doc_id}/unsupersede``."""

    status: str = Field(..., description="Always ``updated`` on success.")
    old_doc_id: str
    new_doc_id: str | None = None
    unlinked: bool
    truncated: list[str] = Field(default_factory=list)
    old_points_updated: int
    old_figures_updated: int
    new_points_updated: int
    new_figures_updated: int
    old: dict[str, Any]
    new: dict[str, Any] | None = None


class ReplaceDocumentContentRequest(BaseModel):
    """Body for ``PUT /documents/{doc_id}/content``."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(
        ...,
        min_length=1,
        description="Replacement document text (UTF-8). Re-chunked and re-embedded.",
    )
    reason: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description="Required non-empty reason (also accepted as a query param).",
    )
    dry_run: bool = Field(
        default=False,
        description="When true, report hashes and counts without writing.",
    )


class RestoreDocumentVersionRequest(BaseModel):
    """Optional body for ``POST /documents/{doc_id}/versions/{n}/restore``."""

    model_config = ConfigDict(extra="forbid")

    reason: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description="Required non-empty reason (also accepted as a query param).",
    )
    dry_run: bool = Field(
        default=False,
        description="When true, report the restore plan without writing.",
    )


class PatchDocumentResponse(BaseModel):
    """Success body for ``PATCH /documents/{doc_id}``."""

    status: str = Field(..., description="Always ``updated`` on success.")
    doc_id: str
    filename: str
    updated_fields: list[str]
    points_updated: int = Field(
        ...,
        description="Knowledge-collection points whose payload was patched.",
    )
    locality_links_updated: int = Field(
        ...,
        description="locality_links points whose parent_document was patched.",
    )
    meta: dict[str, Any]


def _storage(request: Request) -> RagStorage:
    """Return the request-scoped storage instance."""
    return request.app.state.storage


def _qdrant(request: Request) -> QdrantClient:
    """Return the request-scoped Qdrant client."""
    return request.app.state.qdrant


def _rag_data_root(request: Request) -> str:
    """Return the configured RAG data root."""
    return str(request.app.state.rag_data_root)


def _chunked_uploads(request: Request) -> ChunkedUploadStore:
    """Return the request-scoped chunked upload store."""
    return request.app.state.chunked_uploads


def _chunked_upload_http_error(exc: Exception) -> HTTPException:
    """Map chunked-upload domain errors to HTTP responses."""
    if isinstance(exc, UploadNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, UploadExpiredError):
        return HTTPException(status_code=410, detail=str(exc))
    if isinstance(exc, ChunkConflictError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def _record_http(
    *, route: str, ok: bool, bytes_in: int = 0, error: str | None = None
) -> None:
    """Record an HTTP route invocation when accounting is configured."""
    store = get_global_store()
    if store is None:
        return
    store.record_http(route=route, ok=ok, bytes_in=bytes_in, error=error)


def _caller_identity(request: Request) -> str:
    """Return Google email or ``bearer`` for version authorship."""
    if "session" not in request.scope:
        return "bearer"
    user = current_user(request)
    if not user:
        return "bearer"
    email = str(user.get("email") or "").strip()
    if email:
        return email
    sub = str(user.get("sub") or "").strip()
    if sub:
        return f"google:{sub}"
    return "bearer"


def _resolve_reason(*candidates: str | None) -> str:
    """Return the first non-empty reason among ``candidates``."""
    for raw in candidates:
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text
    return ""


def _parse_metadata_form(raw: str | None) -> dict[str, Any] | None:
    """Parse optional multipart ``metadata`` JSON object."""
    if raw is None or not str(raw).strip():
        return None
    try:
        loaded: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="metadata must be valid JSON"
        ) from exc
    if not isinstance(loaded, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    return loaded


def _ingest_upload(
    request: Request,
    filename: str,
    data: bytes,
    *,
    source_url: str | None = None,
    metadata: dict[str, Any] | None = None,
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
                metadata=metadata,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface ingest failures to clients
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc


@router.get("/node-info")
def node_info(request: Request) -> dict[str, Any]:
    """Public catalog of RAG nodes, this instance, and Czech role labels.

    Unauthenticated. Never includes secrets or non-allow-listed env values.
    """
    try:
        body = node_info_payload(request)
        _record_http(route="node_info", ok=True)
        return body
    except Exception as exc:  # noqa: BLE001 - still report accounting failures
        _record_http(route="node_info", ok=False, error=str(exc))
        raise


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    """Liveness probe for container orchestration."""
    try:
        validate_embedding_configuration()
        body: dict[str, object] = {
            "status": "ok",
            "rag_data_root": _rag_data_root(request),
            "embeddings": embedding_provider_status(),
            "image_embedding_enabled": image_embedding_enabled(),
        }
        _record_http(route="health", ok=True)
        return body
    except Exception as exc:  # noqa: BLE001 - still report health accounting failures
        _record_http(route="health", ok=False, error=str(exc))
        raise


@router.post("/upload", dependencies=[Depends(require_embed_access)])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    metadata: str | None = Form(default=None),
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
        result = _ingest_upload(
            request, filename, data, metadata=_parse_metadata_form(metadata)
        )
        _record_http(route="upload", ok=True, bytes_in=len(data))
        return result
    except HTTPException as exc:
        _record_http(
            route="upload", ok=False, bytes_in=len(data), error=str(exc.detail)
        )
        raise


@router.post(
    "/upload/init",
    dependencies=[Depends(require_embed_access)],
    response_model=UploadInitResponse,
)
def upload_init(
    body: UploadInitRequest,
    request: Request,
) -> UploadInitResponse:
    """Start a chunked upload session (see ``PUT /upload/{upload_id}/chunks/{index}``)."""
    if not is_supported_filename(body.filename):
        _record_http(route="upload_init", ok=False, error="unsupported filename")
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, Markdown, JSON, and GeoJSON files are supported.",
        )
    store = _chunked_uploads(request)
    try:
        meta = store.create_session(
            filename=body.filename,
            size=body.size,
            sha256=body.sha256,
            content_type=body.content_type,
            chunk_size=body.chunk_size,
        )
    except ValueError as exc:
        _record_http(route="upload_init", ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _record_http(route="upload_init", ok=True)
    return UploadInitResponse(
        upload_id=meta.upload_id,
        chunk_size=meta.chunk_size,
        chunk_count=meta.chunk_count,
        expires_at=meta.expires_at,
    )


@router.put(
    "/upload/{upload_id}/chunks/{index}",
    dependencies=[Depends(require_embed_access)],
)
async def upload_chunk(
    upload_id: str,
    index: int,
    request: Request,
    x_chunk_sha256: str | None = Header(default=None, alias="X-Chunk-SHA256"),
) -> dict[str, Any]:
    """Upload one raw chunk body; idempotent when the same bytes are resent."""
    data = await request.body()
    store = _chunked_uploads(request)
    try:
        meta = store.store_chunk(
            upload_id,
            index,
            data,
            chunk_sha256=x_chunk_sha256,
        )
    except ChunkedUploadError as exc:
        _record_http(route="upload_chunk", ok=False, error=str(exc))
        raise _chunked_upload_http_error(exc) from exc
    except ValueError as exc:
        _record_http(route="upload_chunk", ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _record_http(route="upload_chunk", ok=True, bytes_in=len(data))
    summary = store.received_summary(meta)
    return {"status": "received", **summary}


@router.get(
    "/upload/{upload_id}",
    dependencies=[Depends(require_embed_access)],
    response_model=UploadStatusResponse,
)
def upload_status(upload_id: str, request: Request) -> UploadStatusResponse:
    """Return received chunk indexes and byte progress for an upload session."""
    store = _chunked_uploads(request)
    try:
        meta = store.get_session(upload_id)
    except ChunkedUploadError as exc:
        _record_http(route="upload_status", ok=False, error=str(exc))
        raise _chunked_upload_http_error(exc) from exc
    _record_http(route="upload_status", ok=True)
    summary = store.received_summary(meta)
    return UploadStatusResponse(**summary)


@router.post(
    "/upload/{upload_id}/complete",
    dependencies=[Depends(require_embed_access)],
)
def upload_complete(upload_id: str, request: Request) -> dict[str, Any]:
    """Verify chunks, assemble on disk, ingest, and remove staging."""
    store = _chunked_uploads(request)
    try:
        assembled_path, meta = store.assemble_to_file(upload_id)
    except ChunkedUploadError as exc:
        _record_http(route="upload_complete", ok=False, error=str(exc))
        raise _chunked_upload_http_error(exc) from exc
    except ValueError as exc:
        _record_http(route="upload_complete", ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data = assembled_path.read_bytes()
    try:
        result = _ingest_upload(request, meta.filename, data)
        store.delete_session(upload_id)
        _record_http(route="upload_complete", ok=True, bytes_in=len(data))
        return result
    except HTTPException as exc:
        _record_http(
            route="upload_complete",
            ok=False,
            bytes_in=len(data),
            error=str(exc.detail),
        )
        raise


@router.delete(
    "/upload/{upload_id}",
    dependencies=[Depends(require_embed_access)],
    status_code=204,
)
def upload_abort(upload_id: str, request: Request) -> Response:
    """Delete staged chunks and metadata for an upload session."""
    store = _chunked_uploads(request)
    removed = store.delete_session(upload_id)
    if not removed:
        _record_http(route="upload_delete", ok=False, error="not found")
        raise HTTPException(status_code=404, detail="Upload session not found")
    _record_http(route="upload_delete", ok=True)
    return Response(status_code=204)


@router.post("/upload-pdf", dependencies=[Depends(require_embed_access)])
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    metadata: str | None = Form(default=None),
) -> dict[str, Any]:
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
        result = _ingest_upload(
            request, filename, data, metadata=_parse_metadata_form(metadata)
        )
        _record_http(route="upload_pdf", ok=True, bytes_in=len(data))
        return result
    except HTTPException as exc:
        _record_http(
            route="upload_pdf", ok=False, bytes_in=len(data), error=str(exc.detail)
        )
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
        result = _ingest_upload(
            request,
            filename,
            data,
            source_url=final_url,
            metadata=body.metadata,
        )
        _record_http(route="ingest_json", ok=True, bytes_in=len(data))
        return result
    except HTTPException as exc:
        _record_http(
            route="ingest_json", ok=False, bytes_in=len(data), error=str(exc.detail)
        )
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


@router.get(
    "/images/{doc_id}/{image_id}",
    dependencies=[Depends(require_embed_access)],
)
def get_image_http(doc_id: str, image_id: str, request: Request) -> FileResponse:
    """Return one extracted PDF image (same auth as ``POST /upload``)."""
    storage = _storage(request)
    try:
        storage.validate_doc_id(doc_id)
        storage.validate_image_route_id(image_id)
        parsed = parse_page_image_id(image_id)
        if parsed is not None:
            pid, _page = parsed
            if pid != doc_id:
                raise ValueError("doc_id mismatch")
        path, mime = resolve_image_delivery(
            storage,
            doc_id=doc_id,
            image_id=image_id,
        )
    except FileNotFoundError as exc:
        _record_http(route="images_get", ok=False, error=str(exc))
        detail = str(exc) or "Image not found"
        raise HTTPException(status_code=404, detail=detail) from exc
    except ValueError as exc:
        _record_http(route="images_get", ok=False, error=str(exc))
        raise HTTPException(status_code=404, detail="Image not found") from exc
    if path.stat().st_size > MAX_IMAGE_HTTP_BYTES:
        _record_http(route="images_get", ok=False, error="too large")
        raise HTTPException(status_code=413, detail="Image file too large")

    _record_http(route="images_get", ok=True)
    return FileResponse(path, media_type=mime)


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


@router.put(
    "/documents/{doc_id}/content",
    dependencies=[Depends(require_embed_access)],
)
def replace_document_content_http(
    doc_id: str,
    body: ReplaceDocumentContentRequest,
    request: Request,
    reason: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Replace indexed text in place, archive the previous version, and re-embed.

    Keeps ``doc_id``. New chunk points are inserted first; old text-chunk
    points are deleted afterwards. On embed/upsert failure the new points
    are removed and the live document is left unchanged. ``pdf_image`` /
    ``pdf_figure`` points are kept. Same auth as ``POST /upload``.
    """
    resolved_reason = _resolve_reason(body.reason, reason)
    do_dry_run = bool(body.dry_run or dry_run)
    try:
        result = replace_document_content(
            storage=_storage(request),
            qdrant=_qdrant(request),
            collection_name=COLLECTION_NAME,
            doc_id=doc_id,
            content=body.content,
            reason=resolved_reason,
            author=_caller_identity(request),
            dry_run=do_dry_run,
        )
    except DocumentNotFoundError:
        _record_http(route="documents_content_put", ok=False, error="not found")
        raise HTTPException(status_code=404, detail="Document not found") from None
    except DocumentReplaceError as exc:
        _record_http(route="documents_content_put", ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface replace failures
        _record_http(route="documents_content_put", ok=False, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Replace failed: {exc}") from exc
    _record_http(route="documents_content_put", ok=True)
    return result


@router.get(
    "/documents/{doc_id}/versions",
    dependencies=[Depends(require_embed_access)],
)
def list_document_versions_http(
    doc_id: str,
    request: Request,
    include_text: bool = False,
) -> dict[str, Any]:
    """List archived content versions. Full text is omitted unless requested."""
    try:
        result = list_document_versions(
            storage=_storage(request),
            doc_id=doc_id,
            include_text=include_text,
        )
    except DocumentNotFoundError:
        _record_http(route="documents_versions_list", ok=False, error="not found")
        raise HTTPException(status_code=404, detail="Document not found") from None
    except Exception as exc:  # noqa: BLE001 - surface version-list failures
        _record_http(route="documents_versions_list", ok=False, error=str(exc))
        raise HTTPException(
            status_code=500, detail=f"List versions failed: {exc}"
        ) from exc
    _record_http(route="documents_versions_list", ok=True)
    return result


@router.post(
    "/documents/{doc_id}/versions/{version}/restore",
    dependencies=[Depends(require_embed_access)],
)
def restore_document_version_http(
    doc_id: str,
    version: int,
    request: Request,
    body: RestoreDocumentVersionRequest | None = None,
    reason: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Restore archived version ``n`` through the same safe replace path.

    Archives the live text as a new version before restoring. Same auth as
    ``PUT /documents/{doc_id}/content``.
    """
    payload = body or RestoreDocumentVersionRequest()
    resolved_reason = _resolve_reason(payload.reason, reason)
    do_dry_run = bool(payload.dry_run or dry_run)
    try:
        result = restore_document_version(
            storage=_storage(request),
            qdrant=_qdrant(request),
            collection_name=COLLECTION_NAME,
            doc_id=doc_id,
            version=version,
            reason=resolved_reason,
            author=_caller_identity(request),
            dry_run=do_dry_run,
        )
    except DocumentNotFoundError:
        _record_http(route="documents_versions_restore", ok=False, error="not found")
        raise HTTPException(status_code=404, detail="Document not found") from None
    except DocumentVersionNotFoundError as exc:
        _record_http(route="documents_versions_restore", ok=False, error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentReplaceError as exc:
        _record_http(route="documents_versions_restore", ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface restore failures
        _record_http(route="documents_versions_restore", ok=False, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Restore failed: {exc}") from exc
    _record_http(route="documents_versions_restore", ok=True)
    return result


@router.patch(
    "/documents/{doc_id}/metadata",
    response_model=PatchDocumentMetadataResponse,
    dependencies=[Depends(require_embed_access)],
)
def patch_document_metadata_http(
    doc_id: str,
    request: Request,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write verification fields onto catalog extra and Qdrant chunk payloads.

    Requires the same auth as ``DELETE /documents/{doc_id}``. Non-null
    fields use ``set_payload``. JSON ``null`` on a whitelist key uses
    ``delete_payload`` (no re-embed, no point deletes, ``doc_id`` unchanged).
    ``superseded_by`` / ``supersedes`` are rejected; use
    ``POST /documents/{doc_id}/supersede`` or
    ``POST /documents/{doc_id}/unsupersede``.
    """
    metadata = body or {}
    try:
        result = patch_document_metadata(
            storage=_storage(request),
            qdrant=_qdrant(request),
            collection_name=COLLECTION_NAME,
            doc_id=doc_id,
            metadata=metadata,
        )
    except DocumentNotFoundError:
        _record_http(route="documents_metadata_patch", ok=False, error="not found")
        raise HTTPException(status_code=404, detail="Document not found") from None
    except (DocumentPatchError, DocumentMetadataError) as exc:
        _record_http(route="documents_metadata_patch", ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface patch failures to clients
        _record_http(route="documents_metadata_patch", ok=False, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Patch failed: {exc}") from exc
    _record_http(route="documents_metadata_patch", ok=True)
    return result


def _supersede_http_error(exc: DocumentNotFoundError, *, path_doc_id: str) -> NoReturn:
    """Raise 404 for a missing path id or referenced replacement document."""
    if exc.doc_id == path_doc_id:
        raise HTTPException(status_code=404, detail="Document not found") from None
    raise HTTPException(
        status_code=404,
        detail=f"new_doc_id not found: {exc.doc_id}",
    ) from None


@router.post(
    "/documents/{doc_id}/supersede",
    response_model=SupersedeDocumentResponse,
    dependencies=[Depends(require_embed_access)],
)
def supersede_document_http(
    doc_id: str,
    request: Request,
    body: SupersedeDocumentRequest,
) -> dict[str, Any]:
    """Atomically link ``doc_id`` as superseded by ``body.new_doc_id``.

    Writes both sides of the pair and sets ``doc_status=superseded`` on the
    old document. Does not re-embed or delete points. Same auth as PATCH.
    """
    try:
        result = supersede_document(
            storage=_storage(request),
            qdrant=_qdrant(request),
            collection_name=COLLECTION_NAME,
            old_doc_id=doc_id,
            new_doc_id=body.new_doc_id,
            correction_note=body.correction_note,
        )
    except DocumentNotFoundError as exc:
        _record_http(route="documents_supersede", ok=False, error="not found")
        _supersede_http_error(exc, path_doc_id=doc_id)
    except (DocumentPatchError, DocumentMetadataError) as exc:
        _record_http(route="documents_supersede", ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface patch failures to clients
        _record_http(route="documents_supersede", ok=False, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Supersede failed: {exc}") from exc
    _record_http(route="documents_supersede", ok=True)
    return result


@router.post(
    "/documents/{doc_id}/unsupersede",
    response_model=UnsupersedeDocumentResponse,
    dependencies=[Depends(require_embed_access)],
)
def unsupersede_document_http(doc_id: str, request: Request) -> dict[str, Any]:
    """Atomically remove the supersession link from ``doc_id``.

    Clears ``superseded_by`` on the old document and removes it from the
    replacement's ``supersedes`` list. Deletes ``doc_status`` only when it
    is ``superseded``. Same auth as PATCH.
    """
    try:
        result = unsupersede_document(
            storage=_storage(request),
            qdrant=_qdrant(request),
            collection_name=COLLECTION_NAME,
            old_doc_id=doc_id,
        )
    except DocumentNotFoundError:
        _record_http(route="documents_unsupersede", ok=False, error="not found")
        raise HTTPException(status_code=404, detail="Document not found") from None
    except (DocumentPatchError, DocumentMetadataError) as exc:
        _record_http(route="documents_unsupersede", ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface patch failures to clients
        _record_http(route="documents_unsupersede", ok=False, error=str(exc))
        raise HTTPException(
            status_code=500, detail=f"Unsupersede failed: {exc}"
        ) from exc
    _record_http(route="documents_unsupersede", ok=True)
    return result


@router.patch(
    "/documents/{doc_id}",
    response_model=PatchDocumentResponse,
    dependencies=[Depends(require_embed_access)],
)
def patch_document_http(
    doc_id: str,
    body: PatchDocumentRequest,
    request: Request,
) -> dict[str, Any]:
    """Update catalog metadata and matching Qdrant payloads without re-embedding.

    Requires ``Authorization: Bearer`` matching ``MCP_API_TOKEN`` or
    ``RAG_API_TOKEN``, or a signed-in Google session (same as ``DELETE
    /documents/{doc_id}``). Allowed body fields: ``filename``, ``extra``.
    """
    try:
        result = patch_document(
            storage=_storage(request),
            qdrant=_qdrant(request),
            collection_name=COLLECTION_NAME,
            doc_id=doc_id,
            filename=body.filename,
            extra=body.extra,
        )
    except DocumentNotFoundError:
        _record_http(route="documents_patch", ok=False, error="not found")
        raise HTTPException(status_code=404, detail="Document not found") from None
    except DocumentPatchError as exc:
        _record_http(route="documents_patch", ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface patch failures to clients
        _record_http(route="documents_patch", ok=False, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Patch failed: {exc}") from exc
    _record_http(route="documents_patch", ok=True)
    return result


@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteDocumentResponse,
    dependencies=[Depends(require_embed_access)],
)
def delete_document_http(doc_id: str, request: Request) -> dict[str, Any]:
    """Remove a document from the catalog, disk, and Qdrant index.

    Requires ``Authorization: Bearer`` matching ``MCP_API_TOKEN`` or
    ``RAG_API_TOKEN``, or a signed-in Google session (same as ``POST /upload``).
    Returns **409** when another document still references this id via
    ``superseded_by`` / ``supersedes`` (unlink first).
    """
    try:
        result = delete_document(
            storage=_storage(request),
            qdrant=_qdrant(request),
            collection_name=COLLECTION_NAME,
            doc_id=doc_id,
        )
    except DocumentNotFoundError:
        _record_http(route="documents_delete", ok=False, error="not found")
        raise HTTPException(status_code=404, detail="Document not found") from None
    except DocumentLinkedError as exc:
        _record_http(route="documents_delete", ok=False, error=str(exc))
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "linked_ids": exc.linked_ids},
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface delete failures to clients
        _record_http(route="documents_delete", ok=False, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc
    _record_http(route="documents_delete", ok=True)
    return result
