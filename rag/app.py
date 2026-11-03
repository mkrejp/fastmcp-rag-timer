"""FastAPI application factory with mounted FastMCP."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

from rag.config import COLLECTION_NAME, resolve_rag_data_root
from rag.mcp_tools import register_mcp_tools
from rag.routes import router
from rag.storage import RagStorage
from rag.vectorstore import open_qdrant


def create_app(rag_data_root: str | None = None) -> FastAPI:
    """Build the FastAPI application with MCP mounted at ``/mcp``."""
    root = resolve_rag_data_root(rag_data_root)
    storage = RagStorage(root)
    storage.ensure_layout()
    qdrant = open_qdrant(storage.qdrant_dir, collection_name=COLLECTION_NAME)

    mcp = FastMCP(
        "Client RAG Node",
        stateless_http=True,
        streamable_http_path="/",
    )
    register_mcp_tools(
        mcp,
        storage=storage,
        qdrant=qdrant,
        collection_name=COLLECTION_NAME,
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
    app.include_router(router)
    app.mount("/mcp", mcp_app)
    return app
