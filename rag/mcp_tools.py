"""FastMCP tools for AI clients."""

from __future__ import annotations

import json
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

from rag.accounting import accounting_source, get_global_store
from rag.config import COLLECTION_NAME
from rag.embeddings import get_embedding
from rag.ingest import ingest_bytes
from rag.logging_setup import get_logger
from rag.processing import is_supported_filename
from rag.storage import RagStorage
from rag.vectorstore import search_similar

logger = get_logger("rag.mcp")


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
                    source_name = payload.get("source", "unknown")
                    doc_id = payload.get("doc_id", "")
                    text = payload.get("text", "")
                    header = f"Source: {source_name}"
                    if doc_id:
                        header += f" (doc_id={doc_id})"
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
