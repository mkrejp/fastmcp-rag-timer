"""FastAPI application factory with mounted FastMCP and Google-authenticated UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.sessions import SessionMiddleware

from rag.accounting import AccountingStore, set_global_store
from rag.admin_routes import router as admin_router
from rag.auth import McpBearerAuth
from rag.config import (
    COLLECTION_NAME,
    google_oauth_configured,
    mcp_allowed_hosts,
    mcp_dns_rebinding_protection,
    resolve_accnt_data_root,
    resolve_public_base_url,
    resolve_rag_data_root,
    resolve_session_secret,
)
from rag.google_oauth import current_user
from rag.google_oauth import router as auth_router
from rag.logging_setup import configure_logging, get_logger
from rag.mcp_tools import register_mcp_tools
from rag.routes import router
from rag.storage import RagStorage
from rag.vectorstore import open_qdrant

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MCP_TOOL_DOCS = {
    "list_documents": "List documents stored in the persistent RAG catalog.",
    "get_document_text": "Return processed full text for a document id.",
    "get_document_chunks": "Return processed chunks JSON for a document id.",
    "ingest_text": "Ingest plain-text or markdown via MCP.",
    "search_domain_knowledge": "Query local RAG vector storage for contextual facts.",
}


def create_app(
    rag_data_root: str | None = None,
    *,
    accnt_data_root: str | None = None,
) -> FastAPI:
    """Build the FastAPI application with MCP mounted at ``/mcp``."""
    configure_logging()
    logger = get_logger("rag.app")
    root = resolve_rag_data_root(rag_data_root)
    storage = RagStorage(root)
    storage.ensure_layout()
    accnt_root = resolve_accnt_data_root(accnt_data_root)
    accounting = AccountingStore(accnt_root)
    accounting.ensure_layout()
    set_global_store(accounting)
    qdrant = open_qdrant(storage.qdrant_dir, collection_name=COLLECTION_NAME)
    logger.info(
        "Opened RAG data root=%s accnt_root=%s collection=%s",
        root,
        accnt_root,
        COLLECTION_NAME,
    )

    mcp = FastMCP(
        "Client RAG Node",
        stateless_http=True,
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=mcp_dns_rebinding_protection(),
            allowed_hosts=mcp_allowed_hosts(),
            allowed_origins=["https://" + h for h in mcp_allowed_hosts()]
            + ["http://localhost:*", "http://127.0.0.1:*"],
        ),
    )
    register_mcp_tools(
        mcp,
        storage=storage,
        qdrant=qdrant,
        collection_name=COLLECTION_NAME,
    )
    mcp_app = mcp.streamable_http_app()
    mcp_tool_names = list(MCP_TOOL_DOCS.keys())

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Run the MCP session manager lifespan alongside FastAPI."""
        logger.info("Starting MCP session manager at /mcp")
        async with mcp_app.router.lifespan_context(mcp_app):
            yield
        logger.info("MCP session manager stopped")

    public_base = resolve_public_base_url() or ""
    app = FastAPI(title="Lightweight RAG Engine", lifespan=lifespan)
    app.add_middleware(
        SessionMiddleware,
        secret_key=resolve_session_secret(),
        same_site="lax",
        https_only=public_base.startswith("https://"),
        max_age=60 * 60 * 12,
    )
    app.state.storage = storage
    app.state.qdrant = qdrant
    app.state.rag_data_root = root
    app.state.accounting = accounting
    app.state.accnt_data_root = accnt_root
    app.state.mcp_mounted = True
    app.state.mcp_tools = mcp_tool_names
    app.state.mcp_tool_docs = MCP_TOOL_DOCS

    app.include_router(router)
    app.include_router(auth_router)
    app.include_router(admin_router)

    @app.get("/")
    def root_redirect() -> RedirectResponse:
        """Send browsers to the authenticated console."""
        return RedirectResponse(url="/ui/", status_code=302)

    @app.get("/ui", response_model=None)
    @app.get("/ui/", response_model=None)
    def ui_index(request: Request) -> FileResponse | RedirectResponse:
        """Serve the RAG console after Google sign-in; otherwise redirect to login."""
        if not google_oauth_configured():
            return RedirectResponse(
                url="/auth/login?error=not_configured", status_code=302
            )
        if current_user(request) is None:
            return RedirectResponse(url="/auth/login", status_code=302)
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.is_dir():
        # CSS/JS are public; HTML shell and /api/admin require Google session.
        app.mount("/ui", StaticFiles(directory=str(STATIC_DIR)), name="ui-static")

    app.mount("/mcp", McpBearerAuth(mcp_app))
    logger.info("Application routes ready; UI at /ui (Google OAuth)")
    return app
