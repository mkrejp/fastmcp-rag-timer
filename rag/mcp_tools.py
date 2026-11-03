"""FastMCP tools for AI clients."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

from rag.auth import verify_write_token
from rag.config import COLLECTION_NAME
from rag.embeddings import get_embedding
from rag.ingest import ingest_bytes
from rag.processing import is_supported_filename
from rag.storage import RagStorage


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
        docs = storage.list_documents()
        if not docs:
            return "No documents stored."
        return json.dumps(docs, ensure_ascii=False, indent=2)

    @mcp.tool()
    def get_document_text(doc_id: str) -> str:
        """Return processed full text for a document id (fast disk read)."""
        text = storage.read_fulltext(doc_id)
        if text is None:
            return f"Document not found: {doc_id}"
        return text

    @mcp.tool()
    def get_document_chunks(doc_id: str) -> str:
        """Return processed chunks JSON for a document id."""
        chunks = storage.read_chunks(doc_id)
        if chunks is None:
            return f"Document chunks not found: {doc_id}"
        return json.dumps(chunks, ensure_ascii=False, indent=2)

    @mcp.tool()
    def ingest_text(filename: str, content: str, auth_token: str = "") -> str:
        """Ingest text/markdown into RAG storage (requires auth_token when RAG_API_TOKEN is set)."""
        try:
            verify_write_token(auth_token or None)
        except PermissionError as exc:
            return f"Unauthorized: {exc}"
        safe_name = filename if is_supported_filename(filename) else f"{filename}.txt"
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

    @mcp.tool()
    def search_domain_knowledge(query: str, limit: int = 3) -> str:
        """Query local RAG vector storage for contextual facts."""
        try:
            query_vector = get_embedding(query)
            results = qdrant.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
            )
            context_blocks: list[str] = []
            for res in results:
                payload = res.payload or {}
                source = payload.get("source", "unknown")
                doc_id = payload.get("doc_id", "")
                text = payload.get("text", "")
                header = f"Source: {source}"
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
