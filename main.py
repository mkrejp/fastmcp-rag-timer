"""Lightweight RAG engine with persistent storage and FastMCP tools."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from embeddings import EMBEDDING_DIM, get_embedding
from ingest import ingest_bytes, meta_to_public
from processing import is_supported_filename
from storage import RagStorage

DEFAULT_RAG_DATA_ROOT = "/data/rag"
COLLECTION_NAME = "client_domain_knowledge"


def create_app(rag_data_root: str | None = None) -> FastAPI:
    """Build the FastAPI application with MCP mounted at ``/mcp``."""
    root = rag_data_root or os.getenv("RAG_DATA_ROOT", DEFAULT_RAG_DATA_ROOT)
    storage = RagStorage(root)
    storage.ensure_layout()

    qdrant = QdrantClient(path=str(storage.qdrant_dir))
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    mcp = FastMCP(
        "Client RAG Node",
        stateless_http=True,
        streamable_http_path="/",
    )
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Run the MCP session manager lifespan alongside FastAPI."""
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    app = FastAPI(title="Lightweight RAG Engine", lifespan=lifespan)
    app.state.storage = storage
    app.state.qdrant = qdrant
    app.state.rag_data_root = root
    app.mount("/mcp", mcp_app)

    def _ingest_upload(filename: str, data: bytes) -> dict[str, Any]:
        """Shared ingest path for HTTP uploads."""
        try:
            return ingest_bytes(
                storage=storage,
                qdrant=qdrant,
                collection_name=COLLECTION_NAME,
                filename=filename,
                data=data,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface ingest failures to clients
            raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe for container orchestration."""
        return {"status": "ok", "rag_data_root": root}

    @app.post("/upload")
    async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
        """Upload a PDF or text document into persistent RAG storage."""
        filename = file.filename or ""
        if not is_supported_filename(filename):
            raise HTTPException(
                status_code=400,
                detail="Only PDF, TXT, and Markdown files are supported.",
            )
        data = await file.read()
        return _ingest_upload(filename, data)

    @app.post("/upload-pdf")
    async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
        """Compatibility alias for PDF uploads."""
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported via this endpoint.",
            )
        data = await file.read()
        return _ingest_upload(filename, data)

    @app.post("/upload-media")
    async def upload_media(file: UploadFile = File(...)) -> dict[str, Any]:
        """Store a non-indexed media asset under originals (legacy endpoint)."""
        filename = file.filename or "media.bin"
        data = await file.read()
        doc_id = str(uuid.uuid4())
        path = storage.save_original(doc_id, filename, data)
        return {"status": "Saved", "path": str(path), "doc_id": doc_id}

    @app.get("/documents")
    def list_documents_http() -> dict[str, Any]:
        """Return the persistent document catalog."""
        return {"documents": storage.list_documents()}

    @app.get("/documents/{doc_id}")
    def get_document_http(doc_id: str) -> dict[str, Any]:
        """Return metadata for a stored document."""
        meta = storage.read_meta(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return meta_to_public(meta)

    @app.get("/documents/{doc_id}/text")
    def get_document_text_http(doc_id: str) -> dict[str, Any]:
        """Return processed full text without re-parsing the original file."""
        text = storage.read_fulltext(doc_id)
        if text is None:
            raise HTTPException(status_code=404, detail="Document text not found")
        meta = storage.read_meta(doc_id)
        return {
            "doc_id": doc_id,
            "filename": meta.filename if meta else None,
            "text": text,
        }

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
    def ingest_text(filename: str, content: str) -> str:
        """Ingest a plain-text or markdown document via MCP without multipart upload."""
        safe_name = filename if is_supported_filename(filename) else f"{filename}.txt"
        try:
            result = ingest_bytes(
                storage=storage,
                qdrant=qdrant,
                collection_name=COLLECTION_NAME,
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
                collection_name=COLLECTION_NAME,
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

    return app


# Skip eager app construction under pytest so tests can open isolated Qdrant paths.
if os.getenv("RAG_SKIP_DEFAULT_APP") == "1":
    app = None  # type: ignore[assignment]
else:
    app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8080)
