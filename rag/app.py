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
from rag.asgi_shutdown import CompleteResponseOnExit
from rag.auth import McpBearerAuth
from rag.chunked_upload import ChunkedUploadStore
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
from rag.embeddings import (
    embedding_dim,
    embedding_provider_status,
    validate_embedding_configuration,
)
from rag.google_oauth import current_user
from rag.google_oauth import router as auth_router
from rag.image_embed import image_embedding_enabled
from rag.logging_setup import configure_logging, get_logger
from rag.mcp_tools import register_mcp_tools
from rag.node_info import NodeWriteGuard
from rag.routes import router
from rag.storage import RagStorage
from rag.trails_sync import maybe_drain_on_startup
from rag.vectorstore import open_qdrant

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MCP_TOOL_DOCS = {
    "list_documents": "List RAG catalog documents (optional since= timestamp).",
    "get_document_text": "Return processed full text for a document id.",
    "get_document_chunks": "Return processed chunks JSON for a document id.",
    "get_document_images": "List extracted PDF images; optional include_figures adds pdf_figure clips.",
    "get_image": "Fetch one extracted image, pdf_figure PNG, or page render (inline or URL when large).",
    "get_page_image": "Render a full PDF page to JPEG (cached on disk).",
    "get_page_highlights": "Highlight boxes for a search query on one PDF page.",
    "ingest_text": "Ingest plain-text or markdown via MCP.",
    "update_document": "Patch document filename/extra metadata without re-embedding.",
    "update_document_metadata": "Patch verification payload fields without re-embedding.",
    "search_domain_knowledge": "Query RAG facts; optional obec_id/cast_obce filter.",
    "search_images": "Search PDF images by CLIP visual similarity (optional hybrid).",
    "link_locality": "Upsert document-municipality links into locality_links (no embeddings).",
    "list_links": "List locality_links with optional filters and pagination.",
    "verify_link": "Set verification_status on an existing locality_links point.",
    "link_stats": "Aggregate locality_links counts by status, obec, method, ambiguous.",
    "sync_links_to_trails": "Backfill locality_links to TrailsAssemblage (optional dry_run).",
    "verify_image_record": "Set verification on a pdf_figure point.",
    "list_image_records": "List pdf_figure records; optional precision/needs_refinement.",
    "image_record_stats": "Aggregate pdf_figure verification counts for one document.",
    "reindex_figures": "Re-run pdf_figure indexing from stored PDFs.",
    "refine_geo": "Store refined GPS beside original lat/lon (dry_run default).",
    "refine_geo_batch": "Batch refine_geo for backfill (max 200; dry_run default).",
    "list_geo_refinements": "List geo_refined records with filters and pagination.",
}


def create_app(
    rag_data_root: str | None = None,
    *,
    accnt_data_root: str | None = None,
) -> FastAPI:
    """Build the FastAPI application with MCP mounted at ``/mcp``."""
    configure_logging()
    logger = get_logger("rag.app")
    validate_embedding_configuration()
    root = resolve_rag_data_root(rag_data_root)
    storage = RagStorage(root)
    storage.ensure_layout()
    chunked_uploads = ChunkedUploadStore(root)
    chunked_uploads.ensure_layout()
    accnt_root = resolve_accnt_data_root(accnt_data_root)
    accounting = AccountingStore(accnt_root)
    accounting.ensure_layout()
    set_global_store(accounting)
    qdrant = open_qdrant(storage.qdrant_dir, collection_name=COLLECTION_NAME)
    maybe_drain_on_startup(qdrant, storage.trails_outbox_path)
    providers = embedding_provider_status()
    logger.info(
        "Opened RAG data root=%s accnt_root=%s collection=%s ingest=%s query=%s dim=%s "
        "image_embedding=%s",
        root,
        accnt_root,
        COLLECTION_NAME,
        providers.get("ingest_provider"),
        providers.get("query_provider"),
        embedding_dim(),
        image_embedding_enabled(),
    )
    if image_embedding_enabled():
        logger.warning(
            "IMAGE_EMBEDDING_ENABLED loads CLIP beside BGE-M3 fastembed; on 8GB hosts "
            "consider extra RAM or disable visual search."
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
    mcp_tool_names = sorted(MCP_TOOL_DOCS.keys())

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Run the MCP session manager lifespan alongside FastAPI."""
        logger.info("Starting MCP session manager at /mcp")
        try:
            async with mcp_app.router.lifespan_context(mcp_app):
                yield
        finally:
            logger.info("MCP session manager stopped")
            try:
                qdrant.close()
            except Exception:  # noqa: BLE001 - best-effort on shutdown
                logger.debug("Qdrant client close failed", exc_info=True)

    public_base = resolve_public_base_url() or ""
    app = FastAPI(title="Lightweight RAG Engine", lifespan=lifespan)
    app.add_middleware(
        SessionMiddleware,
        secret_key=resolve_session_secret(),
        same_site="lax",
        https_only=public_base.startswith("https://"),
        max_age=60 * 60 * 12,
    )
    # Outer guard; no-ops unless RAG_ENFORCE_NODE_ROLE is enabled.
    app.add_middleware(NodeWriteGuard)
    app.state.storage = storage
    app.state.chunked_uploads = chunked_uploads
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

    app.mount("/mcp", McpBearerAuth(CompleteResponseOnExit(mcp_app)))
    logger.info("Application routes ready; UI at /ui (Google OAuth)")
    return app
