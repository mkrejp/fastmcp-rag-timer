"""FastMCP tools for AI clients."""

from __future__ import annotations

import json
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import ImageContent, TextContent
from qdrant_client import QdrantClient

from rag.accounting import accounting_source, get_global_store
from rag.config import COLLECTION_NAME, resolve_public_base_url
from rag.embeddings import get_embedding
from rag.ingest import ingest_bytes
from rag.logging_setup import get_logger
from rag.processing import is_supported_filename
from rag.storage import ImageMeta, RagStorage
from rag.vectorstore import search_similar

logger = get_logger("rag.mcp")

MAX_MCP_IMAGE_BYTES = 5 * 1024 * 1024


def _record_mcp(
    *,
    tool: str,
    ok: bool,
    request_chars: int = 0,
    response_chars: int = 0,
    error: str | None = None,
) -> None:
    """Record an MCP tool invocation when accounting is configured."""
    store = get_global_store()
    if store is None:
        return
    store.record_mcp(
        tool=tool,
        ok=ok,
        request_chars=request_chars,
        response_chars=response_chars,
        error=error,
    )


def _run_mcp_tool(
    tool: str,
    *,
    request_chars: int,
    source: str | None,
    fn: Callable[[], str],
) -> str:
    """Execute an MCP tool body with accounting and optional embedding source."""
    try:
        if source is None:
            result = fn()
        else:
            with accounting_source(source):
                result = fn()
        ok = not result.startswith("Error ")
        _record_mcp(
            tool=tool,
            ok=ok,
            request_chars=request_chars,
            response_chars=len(result),
            error=None if ok else result,
        )
        return result
    except Exception as exc:  # noqa: BLE001 - return tool-visible error text
        message = f"Error executing {tool}: {exc}"
        _record_mcp(
            tool=tool,
            ok=False,
            request_chars=request_chars,
            response_chars=len(message),
            error=str(exc),
        )
        return message


def _image_public_url(doc_id: str, image_id: str) -> str | None:
    """Build an absolute image URL when ``PUBLIC_BASE_URL`` is configured."""
    base = resolve_public_base_url()
    if not base:
        return None
    return f"{base}/images/{doc_id}/{image_id}"


def _image_meta_with_url(
    doc_id: str,
    image: ImageMeta,
) -> dict[str, Any]:
    """Serialize ``image`` and attach a public HTTP URL when available."""
    payload = image.to_dict()
    payload["url"] = _image_public_url(doc_id, image.image_id)
    return payload


def register_mcp_tools(
    mcp: FastMCP,
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Register curated MCP tools against the given storage and vector index."""

    @mcp.tool()
    def list_documents() -> str:
        """List documents stored in the persistent RAG catalog."""

        def _body() -> str:
            docs = storage.list_documents()
            if not docs:
                return "No documents stored."
            return json.dumps(docs, ensure_ascii=False, indent=2)

        return _run_mcp_tool("list_documents", request_chars=0, source=None, fn=_body)

    @mcp.tool()
    def get_document_text(doc_id: str) -> str:
        """Return processed full text for a document id (fast disk read)."""

        def _body() -> str:
            text = storage.read_fulltext(doc_id)
            if text is None:
                return f"Document not found: {doc_id}"
            return text

        return _run_mcp_tool(
            "get_document_text",
            request_chars=len(doc_id),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def get_document_chunks(doc_id: str) -> str:
        """Return processed chunks JSON for a document id."""

        def _body() -> str:
            chunks = storage.read_chunks(doc_id)
            if chunks is None:
                return f"Document chunks not found: {doc_id}"
            return json.dumps(chunks, ensure_ascii=False, indent=2)

        return _run_mcp_tool(
            "get_document_chunks",
            request_chars=len(doc_id),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def get_document_images(doc_id: str) -> str:
        """Return extracted PDF image metadata (including optional public URLs)."""

        def _body() -> str:
            meta = storage.read_meta(doc_id)
            if meta is None:
                return f"Document not found: {doc_id}"
            items = [_image_meta_with_url(doc_id, img) for img in meta.images]
            return json.dumps(items, ensure_ascii=False, indent=2)

        return _run_mcp_tool(
            "get_document_images",
            request_chars=len(doc_id),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def get_image(image_id: str) -> list[TextContent | ImageContent]:
        """Return image bytes (≤5MB) or a URL when the file is larger."""
        request_chars = len(image_id)
        try:
            located = storage.find_image_global(image_id)
            if located is None:
                message = f"Image not found: {image_id}"
                _record_mcp(
                    tool="get_image",
                    ok=False,
                    request_chars=request_chars,
                    response_chars=len(message),
                    error=message,
                )
                return [TextContent(type="text", text=message)]

            doc_id, image_meta = located
            path = storage.image_file_path(doc_id, image_meta)
            size = path.stat().st_size if path.is_file() else 0
            url = _image_public_url(doc_id, image_id)
            if not path.is_file():
                message = f"Image file missing on disk: {image_id}"
                _record_mcp(
                    tool="get_image",
                    ok=False,
                    request_chars=request_chars,
                    response_chars=len(message),
                    error=message,
                )
                return [TextContent(type="text", text=message)]

            if size > MAX_MCP_IMAGE_BYTES:
                hint = url or f"/images/{doc_id}/{image_id}"
                message = (
                    f"Image exceeds MCP inline limit ({size} bytes). "
                    f"Fetch via authenticated HTTP: {hint}"
                )
                _record_mcp(
                    tool="get_image",
                    ok=True,
                    request_chars=request_chars,
                    response_chars=len(message),
                )
                return [TextContent(type="text", text=message)]

            image = Image(path=path)
            content = image.to_image_content()
            _record_mcp(
                tool="get_image",
                ok=True,
                request_chars=request_chars,
                response_chars=size,
            )
            return [content]
        except Exception as exc:  # noqa: BLE001
            message = f"Error executing get_image: {exc}"
            _record_mcp(
                tool="get_image",
                ok=False,
                request_chars=request_chars,
                response_chars=len(message),
                error=str(exc),
            )
            return [TextContent(type="text", text=message)]

    @mcp.tool()
    def ingest_text(filename: str, content: str) -> str:
        """Ingest a plain-text or markdown document via MCP without multipart upload."""

        def _body() -> str:
            safe_name = (
                filename if is_supported_filename(filename) else f"{filename}.txt"
            )
            try:
                result = ingest_bytes(
                    storage=storage,
                    qdrant=qdrant,
                    collection_name=collection_name,
                    filename=safe_name,
                    data=content.encode("utf-8"),
                )
                return json.dumps(result, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001 - return tool-visible error text
                return f"Error ingesting text: {exc}"

        return _run_mcp_tool(
            "ingest_text",
            request_chars=len(filename) + len(content),
            source="ingest_mcp",
            fn=_body,
        )

    @mcp.tool()
    def search_domain_knowledge(query: str, limit: int = 3) -> str:
        """Query local RAG vector storage for contextual facts."""

        def _body() -> str:
            try:
                logger.info(
                    "MCP search_domain_knowledge query_chars=%s limit=%s",
                    len(query),
                    limit,
                )
                query_vector = get_embedding(query)
                results = search_similar(
                    qdrant,
                    query_vector=query_vector,
                    limit=limit,
                    collection_name=collection_name,
                )
                logger.info("MCP search returned %s hits", len(results))
                context_blocks: list[str] = []
                for res in results:
                    payload: dict[str, Any] = res.payload or {}
                    source_name = payload.get("parent_document") or payload.get(
                        "source", "unknown"
                    )
                    doc_id = payload.get("doc_id", "")
                    source_type = payload.get("source_type", "pdf_chunk")
                    page_number = payload.get("page_number")
                    image_id = payload.get("image_id")
                    text = payload.get("content_summary") or payload.get("text", "")
                    header = f"Source: {source_name} (source_type={source_type})"
                    if doc_id:
                        header += f" doc_id={doc_id}"
                    if page_number is not None:
                        header += f" page_number={page_number}"
                    if image_id:
                        url = _image_public_url(str(doc_id), str(image_id))
                        header += f" image_id={image_id}"
                        if url:
                            header += f" url={url}"
                    context_blocks.append(f"{header}\nContent: {text}\n")
                return (
                    "\n---\n".join(context_blocks)
                    if context_blocks
                    else "No relevant facts found."
                )
            except Exception as exc:  # noqa: BLE001 - return tool-visible error text
                return f"Error executing RAG search: {exc}"

        return _run_mcp_tool(
            "search_domain_knowledge",
            request_chars=len(query),
            source="search_mcp",
            fn=_body,
        )
