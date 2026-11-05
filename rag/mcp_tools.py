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
from rag.embeddings import get_query_embedding
from rag.image_embed import embed_image_query, image_embedding_enabled
from rag.ingest import ingest_bytes
from rag.logging_setup import get_logger
from rag.payloads import LocalityLinkPayloadError
from rag.processing import is_supported_filename
from rag.storage import ImageMeta, RagStorage
from rag.vectorstore import (
    IMAGE_VECTOR_NAME,
    TEXT_VECTOR_NAME,
    knowledge_locality_filter,
    list_locality_links,
    locality_link_stats,
    search_similar,
    upsert_locality_links,
    verify_locality_link,
)

logger = get_logger("rag.mcp")

MAX_MCP_IMAGE_BYTES = 5 * 1024 * 1024
RRF_K = 60


def _reciprocal_rank_fusion(
    ranked_lists: list[list[Any]],
    *,
    limit: int,
) -> list[Any]:
    """Merge ranked hit lists with reciprocal rank fusion (simple hybrid)."""
    scores: dict[str | int, float] = {}
    by_id: dict[str | int, Any] = {}
    for ranked in ranked_lists:
        for rank, point in enumerate(ranked):
            point_id = point.id
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            by_id[point_id] = point
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [by_id[point_id] for point_id, _ in ordered[:limit]]


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
    def list_documents(since: str = "") -> str:
        """List documents stored in the persistent RAG catalog.

        Optional ``since`` is an ISO-8601 timestamp compared to
        ``uploaded_at`` (inclusive). Empty ``since`` returns the full catalog.
        """

        def _body() -> str:
            cutoff = since.strip() or None
            try:
                docs = storage.list_documents(since=cutoff)
            except ValueError as exc:
                return f"Error list_documents: {exc}"
            if not cutoff and not docs:
                return "No documents stored."
            return json.dumps(docs, ensure_ascii=False, indent=2)

        return _run_mcp_tool(
            "list_documents",
            request_chars=len(since),
            source=None,
            fn=_body,
        )

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
    def search_domain_knowledge(
        query: str,
        limit: int = 3,
        obec_id: str = "",
        cast_obce: str = "",
    ) -> str:
        """Query local RAG vector storage for contextual facts.

        Optional ``obec_id`` / ``cast_obce`` filter hits whose denormalized
        ``localities`` / ``cast_obce`` payload arrays contain those values.
        Empty strings omit the filter (backward compatible).
        """

        def _body() -> str:
            try:
                logger.info(
                    "MCP search_domain_knowledge query_chars=%s limit=%s",
                    len(query),
                    limit,
                )
                query_vector = get_query_embedding(query)
                results = search_similar(
                    qdrant,
                    query_vector=query_vector,
                    limit=limit,
                    collection_name=collection_name,
                    vector_name=TEXT_VECTOR_NAME,
                    query_filter=knowledge_locality_filter(
                        obec_id=obec_id.strip() or None,
                        cast_obce=cast_obce.strip() or None,
                    ),
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
            request_chars=len(query) + len(obec_id) + len(cast_obce),
            source="search_mcp",
            fn=_body,
        )

    @mcp.tool()
    def search_images(query: str, limit: int = 5, hybrid: bool = False) -> str:
        """Search PDF images by visual similarity (local CLIP) with optional text hybrid."""

        def _body() -> str:
            if not image_embedding_enabled():
                return (
                    "IMAGE_EMBEDDING_ENABLED is false. Set it to true and re-ingest "
                    "PDFs to index image_vector fields before using search_images."
                )
            try:
                query_vector = embed_image_query(query)
                visual_hits = search_similar(
                    qdrant,
                    query_vector=query_vector,
                    limit=limit,
                    collection_name=collection_name,
                    vector_name=IMAGE_VECTOR_NAME,
                )
                if hybrid:
                    text_vector = get_query_embedding(query)
                    text_hits = search_similar(
                        qdrant,
                        query_vector=text_vector,
                        limit=limit,
                        collection_name=collection_name,
                        vector_name=TEXT_VECTOR_NAME,
                    )
                    filter_image = [
                        hit
                        for hit in text_hits
                        if (hit.payload or {}).get("source_type") == "pdf_image"
                    ]
                    results = _reciprocal_rank_fusion(
                        [visual_hits, filter_image], limit=limit
                    )
                else:
                    results = visual_hits
                blocks: list[str] = []
                for res in results:
                    payload: dict[str, Any] = res.payload or {}
                    if payload.get("source_type") != "pdf_image":
                        continue
                    doc_id = str(payload.get("doc_id", ""))
                    image_id = str(payload.get("image_id", ""))
                    page_number = payload.get("page_number")
                    parent = payload.get("parent_document") or payload.get("source")
                    url = _image_public_url(doc_id, image_id) if doc_id and image_id else None
                    summary = payload.get("content_summary") or payload.get("text", "")
                    header = f"Source: {parent} (pdf_image)"
                    if doc_id:
                        header += f" doc_id={doc_id}"
                    if page_number is not None:
                        header += f" page_number={page_number}"
                    if image_id:
                        header += f" image_id={image_id}"
                    if url:
                        header += f" url={url}"
                    blocks.append(f"{header}\nContent: {summary}\n")
                return "\n---\n".join(blocks) if blocks else "No relevant images found."
            except Exception as exc:  # noqa: BLE001
                return f"Error executing search_images: {exc}"

        return _run_mcp_tool(
            "search_images",
            request_chars=len(query),
            source="search_mcp",
            fn=_body,
        )

    @mcp.tool()
    def link_locality(links: list[dict[str, Any]]) -> str:
        """Upsert document-municipality links into the payload-only locality_links collection.

        Each object requires obec_id, name, doc_id, chunk_index, matched_text,
        char_start, char_end, source_type (pdf_chunk|pdf_image|text_chunk), and
        method (exact|lemma|fuzzy|llm|manual). Optional fields include lau_id,
        okres, kraj, parent_document, page_number, point_id, chunk_sha256,
        ambiguous, candidates, confidence, confidence_level, notes, geom,
        locality_level (obec|cast_obce), cast_obce / cast_obce_kod / cast_obce_name.
        Point ids are stable UUIDs from (doc_id, chunk_index, obec_id, char_start)
        plus cast_obce_kod when the mention is a municipal part.
        Also denormalizes obec_id / cast_obce onto matching knowledge points
        via set_payload (no re-embedding, no deletes).
        Does not call embedding APIs.
        """

        def _body() -> str:
            try:
                items = _coerce_locality_links(links)
                result = upsert_locality_links(qdrant, items)
                return json.dumps(result, ensure_ascii=False)
            except LocalityLinkPayloadError as exc:
                return f"Error link_locality: {exc}"

        return _run_mcp_tool(
            "link_locality",
            request_chars=len(json.dumps(links, ensure_ascii=False, default=str)),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def list_links(
        doc_id: str = "",
        obec_id: str = "",
        status: str = "",
        ambiguous: bool | None = None,
        locality_level: str = "",
        cast_obce: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        """List locality_links with optional filters and integer pagination.

        Filter by doc_id, obec_id, status (unverified|verified|rejected),
        ambiguous, locality_level (obec|cast_obce), and cast_obce (kod).
        ``limit`` defaults to 50 (max 200); ``offset`` skips that many
        matching points. Does not call embedding APIs.
        """

        def _body() -> str:
            try:
                result = list_locality_links(
                    qdrant,
                    doc_id=doc_id.strip() or None,
                    obec_id=obec_id.strip() or None,
                    status=status.strip() or None,
                    ambiguous=ambiguous,
                    locality_level=locality_level.strip() or None,
                    cast_obce=cast_obce.strip() or None,
                    limit=limit,
                    offset=offset,
                )
                return json.dumps(result, ensure_ascii=False)
            except LocalityLinkPayloadError as exc:
                return f"Error list_links: {exc}"

        return _run_mcp_tool(
            "list_links",
            request_chars=(
                len(doc_id)
                + len(obec_id)
                + len(status)
                + len(locality_level)
                + len(cast_obce)
                + len(str(limit))
                + len(str(offset))
            ),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def verify_link(
        link_id: str,
        status: str,
        verified_by: str = "",
        notes: str = "",
    ) -> str:
        """Set verification_status on an existing locality_links point.

        ``status`` is unverified, verified, or rejected. Optional verified_by
        and notes are stored on the payload. verified is true only for verified;
        verified_at is set on verified/rejected and cleared on unverified.
        Does not call embedding APIs. Does not sync Trails.
        """

        def _body() -> str:
            try:
                result = verify_locality_link(
                    qdrant,
                    link_id=link_id,
                    status=status,
                    verified_by=verified_by if verified_by.strip() else None,
                    notes=notes if notes.strip() else None,
                )
                return json.dumps(result, ensure_ascii=False)
            except LocalityLinkPayloadError as exc:
                return f"Error verify_link: {exc}"

        return _run_mcp_tool(
            "verify_link",
            request_chars=len(link_id) + len(status) + len(verified_by) + len(notes),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def link_stats() -> str:
        """Aggregate locality_links counts by status, method, obec, and ambiguous.

        Also reports locality_level totals, distinct docs covered (excluding
        rejected), and top obce. Does not call embedding APIs.
        """

        def _body() -> str:
            try:
                result = locality_link_stats(qdrant)
                return json.dumps(result, ensure_ascii=False)
            except LocalityLinkPayloadError as exc:
                return f"Error link_stats: {exc}"

        return _run_mcp_tool(
            "link_stats",
            request_chars=0,
            source=None,
            fn=_body,
        )


def _coerce_locality_links(links: object) -> list[dict[str, Any]]:
    """Accept a list of objects, a single object, or a JSON string of either."""
    parsed: object = links
    if isinstance(links, str):
        try:
            parsed = json.loads(links)
        except json.JSONDecodeError as exc:
            raise LocalityLinkPayloadError(
                "links must be a JSON object or array of objects"
            ) from exc
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        if not parsed:
            raise LocalityLinkPayloadError("at least one link is required")
        coerced: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                raise LocalityLinkPayloadError("each link must be a JSON object")
            coerced.append(item)
        return coerced
    raise LocalityLinkPayloadError("links must be a JSON object or array of objects")
