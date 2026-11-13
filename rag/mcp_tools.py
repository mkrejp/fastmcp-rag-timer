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
from rag.document_delete import DocumentNotFoundError
from rag.document_patch import (
    DocumentPatchError,
    patch_document,
    patch_document_metadata,
)
from rag.document_supersede import supersede_document as apply_supersede_document
from rag.document_supersede import unsupersede_document as apply_unsupersede_document
from rag.embeddings import get_query_embedding
from rag.figure_index import image_record_stats as aggregate_image_record_stats
from rag.figure_index import list_figure_payloads_for_doc
from rag.figure_index import list_image_records as scroll_image_records
from rag.figure_index import reindex_figures_for_doc
from rag.figure_index import verify_image_record as apply_verify_image_record
from rag.geo_index import document_geo_fallback
from rag.geo_index import list_geo_refinements as list_geo_refinement_records
from rag.geo_index import refine_geo as apply_refine_geo
from rag.geo_index import refine_geo_batch as apply_refine_geo_batch
from rag.geo_refined import (
    GeoRefinedNotFoundError,
    GeoRefinedPayloadError,
    attach_preferred_geo,
    matches_geo_filter,
    validate_precision,
)
from rag.image_delivery import (
    MISSING_IMAGE_FILE_MESSAGE,
    UNKNOWN_IMAGE_ID_MESSAGE,
    locate_global_image,
    resolve_image_delivery,
)
from rag.image_embed import embed_image_query, image_embedding_enabled
from rag.ingest import ingest_bytes
from rag.logging_setup import get_logger
from rag.node_info import mcp_write_blocked_message
from rag.page_ids import format_page_image_id
from rag.payloads import ImageRecordPayloadError, LocalityLinkPayloadError
from rag.pdf_pages import get_page_highlights as compute_page_highlights
from rag.processing import is_supported_filename
from rag.search_hits import (
    build_image_search_hit,
    build_knowledge_hit,
    geo_header_suffix,
    page_image_id_for_hit,
)
from rag.storage import ImageMeta, RagStorage
from rag.trails_sync import forward_locality_links
from rag.trails_sync import sync_links_to_trails as run_sync_links_to_trails
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
from rag.verification_meta import DocumentMetadataError

logger = get_logger("rag.mcp")


def _safe_trails_forward(
    qdrant: QdrantClient,
    link_ids: list[str],
    *,
    outbox_path: Any,
) -> dict[str, Any]:
    """Forward to Trails; never raise into an MCP tool body."""
    try:
        return forward_locality_links(qdrant, link_ids, outbox_path=outbox_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Trails forward wrapper failed: %s", exc)
        return {"enabled": True, "ok": False, "error": str(exc)[:300]}


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
    blocked = mcp_write_blocked_message(tool)
    if blocked is not None:
        _record_mcp(
            tool=tool,
            ok=False,
            request_chars=request_chars,
            response_chars=len(blocked),
            error=blocked,
        )
        return blocked
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


def _page_image_public_url(doc_id: str, page_one_based: int) -> str | None:
    """Build a public URL for a full-page render id."""
    return _image_public_url(doc_id, format_page_image_id(doc_id, page_one_based))


def _image_meta_with_url(
    doc_id: str,
    image: ImageMeta,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize ``image`` and attach a public HTTP URL when available."""
    payload = image.to_dict()
    payload["url"] = _image_public_url(doc_id, image.image_id)
    return attach_preferred_geo(payload, fallback=fallback)


def _figure_item_with_url(
    doc_id: str,
    payload: dict[str, Any],
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a ``pdf_figure`` payload in the ``get_document_images`` shape."""
    image_id = str(payload.get("image_id") or "")
    item = {
        "image_id": image_id,
        "sha256": str(payload.get("image_sha256") or ""),
        "file_path": str(payload.get("file_path") or ""),
        "page_number": payload.get("page_number"),
        "mime": "image/png",
        "ocr_text": payload.get("ocr_text") or None,
        "ocr_confidence": payload.get("ocr_confidence"),
        "url": _image_public_url(doc_id, image_id) if image_id else None,
        "source_type": "pdf_figure",
        "caption": payload.get("caption"),
        "summary": payload.get("summary"),
        "figure_id": payload.get("figure_id"),
        "lat": payload.get("lat"),
        "lon": payload.get("lon"),
        "geo_refined": payload.get("geo_refined"),
    }
    return attach_preferred_geo(item, fallback=fallback or payload)


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
            fallback = document_geo_fallback(storage, doc_id)
            items = [
                (
                    attach_preferred_geo(chunk, fallback=fallback)
                    if isinstance(chunk, dict)
                    else chunk
                )
                for chunk in chunks
            ]
            return json.dumps(items, ensure_ascii=False, indent=2)

        return _run_mcp_tool(
            "get_document_chunks",
            request_chars=len(doc_id),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def get_document_images(doc_id: str, include_figures: bool = False) -> str:
        """Return extracted PDF image metadata (including optional public URLs).

        When ``include_figures`` is true, also include this document's
        ``pdf_figure`` images (id, page, caption/summary, URL). Default false
        keeps the previous extracted-image-only list.
        """

        def _body() -> str:
            meta = storage.read_meta(doc_id)
            if meta is None:
                return f"Document not found: {doc_id}"
            fallback = meta.to_dict()
            items = [
                _image_meta_with_url(doc_id, img, fallback=fallback)
                for img in meta.images
            ]
            if include_figures:
                try:
                    figures = list_figure_payloads_for_doc(
                        qdrant,
                        collection_name=collection_name,
                        doc_id=doc_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    return f"Error listing figures: {exc}"
                for payload in figures:
                    image_id = str(payload.get("image_id") or "")
                    if not image_id:
                        continue
                    items.append(
                        _figure_item_with_url(doc_id, payload, fallback=fallback)
                    )
            return json.dumps(items, ensure_ascii=False, indent=2)

        return _run_mcp_tool(
            "get_document_images",
            request_chars=len(doc_id) + int(include_figures),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def get_image(image_id: str) -> list[TextContent | ImageContent]:
        """Return image bytes (≤5MB) or a URL when the file is larger.

        Accepts extracted ``image_id``, ``pdf_figure`` clip ids stored on disk,
        or full-page ids ``page:{doc_id}:{page}``.
        """
        request_chars = len(image_id)
        try:
            located = locate_global_image(storage, image_id)
            if located is None:
                message = f"{UNKNOWN_IMAGE_ID_MESSAGE}: {image_id}"
                _record_mcp(
                    tool="get_image",
                    ok=False,
                    request_chars=request_chars,
                    response_chars=len(message),
                    error=message,
                )
                return [TextContent(type="text", text=message)]

            doc_id, path, _mime = located
            size = path.stat().st_size if path.is_file() else 0
            url = _image_public_url(doc_id, image_id)
            if not path.is_file():
                message = f"{MISSING_IMAGE_FILE_MESSAGE}: {image_id}"
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
        except FileNotFoundError as exc:
            detail = str(exc) or UNKNOWN_IMAGE_ID_MESSAGE
            message = f"{detail}: {image_id}" if image_id not in detail else detail
            _record_mcp(
                tool="get_image",
                ok=False,
                request_chars=request_chars,
                response_chars=len(message),
                error=message,
            )
            return [TextContent(type="text", text=message)]
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
    def get_page_image(doc_id: str, page: int) -> list[TextContent | ImageContent]:
        """Return a full-page JPEG render (≤5MB inline, else authenticated URL)."""
        request_chars = len(doc_id) + len(str(page))
        image_id = format_page_image_id(doc_id, page)
        try:
            path, _mime = resolve_image_delivery(
                storage,
                doc_id=doc_id,
                image_id=image_id,
            )
            size = path.stat().st_size
            url = _page_image_public_url(doc_id, page)
            if size > MAX_MCP_IMAGE_BYTES:
                hint = url or f"/images/{doc_id}/{image_id}"
                message = (
                    f"Image exceeds MCP inline limit ({size} bytes). "
                    f"Fetch via authenticated HTTP: {hint}"
                )
                _record_mcp(
                    tool="get_page_image",
                    ok=True,
                    request_chars=request_chars,
                    response_chars=len(message),
                )
                return [TextContent(type="text", text=message)]
            image = Image(path=path)
            content = image.to_image_content()
            _record_mcp(
                tool="get_page_image",
                ok=True,
                request_chars=request_chars,
                response_chars=size,
            )
            return [content]
        except Exception as exc:  # noqa: BLE001
            message = f"Error executing get_page_image: {exc}"
            _record_mcp(
                tool="get_page_image",
                ok=False,
                request_chars=request_chars,
                response_chars=len(message),
                error=str(exc),
            )
            return [TextContent(type="text", text=message)]

    @mcp.tool()
    def get_page_highlights(
        doc_id: str,
        page: int,
        query: str,
        max_hits: int = 20,
    ) -> str:
        """Return JSON highlight boxes for ``query`` on one PDF page."""

        def _body() -> str:
            result = compute_page_highlights(
                storage,
                doc_id,
                page,
                query,
                max_hits=max_hits,
            )
            return json.dumps(result, ensure_ascii=False)

        return _run_mcp_tool(
            "get_page_highlights",
            request_chars=len(doc_id) + len(query) + len(str(page)),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def ingest_text(
        filename: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
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
                    metadata=metadata,
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
    def update_document(
        doc_id: str,
        filename: str = "",
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Patch stored document metadata without re-upload or re-embedding.

        Updates ``index.json`` / ``meta.json`` and matching Qdrant payloads
        (``parent_document`` / ``source`` / nested ``extra``) via set_payload.
        Allowed fields: ``filename`` (same suffix) and ``extra`` (merge;
        JSON ``null`` removes a custom key). Same bearer auth as other MCP
        write tools.
        """

        def _body() -> str:
            try:
                result = patch_document(
                    storage=storage,
                    qdrant=qdrant,
                    collection_name=collection_name,
                    doc_id=doc_id,
                    filename=filename.strip() or None,
                    extra=extra,
                )
                return json.dumps(result, ensure_ascii=False)
            except DocumentNotFoundError as exc:
                return f"Error update_document: {exc}"
            except DocumentPatchError as exc:
                return f"Error update_document: {exc}"
            except Exception as exc:  # noqa: BLE001 - return tool-visible error text
                return f"Error update_document: {exc}"

        extra_chars = 0
        if extra is not None:
            extra_chars = len(json.dumps(extra, ensure_ascii=False, default=str))
        return _run_mcp_tool(
            "update_document",
            request_chars=len(doc_id) + len(filename) + extra_chars,
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def update_document_metadata(
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Patch verification payload fields without re-upload or re-embedding.

        Writes flat RAGloader fields (``reliability``, ``verification[]``,
        ``qg_verze``, ``overeni_souhrn``, ``doc_status``, ``correction_note``,
        …) via ``set_payload`` onto matching chunks. JSON ``null`` on a
        whitelist key deletes that payload key (``deleted_fields``).
        ``superseded_by`` / ``supersedes`` are rejected; use
        ``supersede_document`` / ``unsupersede_document``. Clipped fields
        are listed in ``truncated``. Does not change ``doc_id``.
        """

        def _body() -> str:
            try:
                result = patch_document_metadata(
                    storage=storage,
                    qdrant=qdrant,
                    collection_name=collection_name,
                    doc_id=doc_id,
                    metadata=metadata,
                )
                return json.dumps(result, ensure_ascii=False)
            except DocumentNotFoundError as exc:
                return f"Error update_document_metadata: {exc}"
            except (DocumentPatchError, DocumentMetadataError) as exc:
                return f"Error update_document_metadata: {exc}"
            except Exception as exc:  # noqa: BLE001 - return tool-visible error text
                return f"Error update_document_metadata: {exc}"

        meta_chars = 0
        if metadata is not None:
            meta_chars = len(json.dumps(metadata, ensure_ascii=False, default=str))
        return _run_mcp_tool(
            "update_document_metadata",
            request_chars=len(doc_id) + meta_chars,
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def supersede_document(
        old_doc_id: str,
        new_doc_id: str,
        correction_note: str = "",
    ) -> str:
        """Atomically mark ``old_doc_id`` as superseded by ``new_doc_id``.

        Sets both sides of the link (``old.superseded_by`` and
        ``new.supersedes``) and ``old.doc_status=superseded``. Does not
        re-embed or auto-write other documents. Same bearer as other writes.
        """

        def _body() -> str:
            try:
                note: str | None = correction_note if correction_note.strip() else None
                result = apply_supersede_document(
                    storage=storage,
                    qdrant=qdrant,
                    collection_name=collection_name,
                    old_doc_id=old_doc_id,
                    new_doc_id=new_doc_id,
                    correction_note=note,
                )
                return json.dumps(result, ensure_ascii=False)
            except DocumentNotFoundError as exc:
                return f"Error supersede_document: {exc}"
            except (DocumentPatchError, DocumentMetadataError) as exc:
                return f"Error supersede_document: {exc}"
            except Exception as exc:  # noqa: BLE001 - return tool-visible error text
                return f"Error supersede_document: {exc}"

        return _run_mcp_tool(
            "supersede_document",
            request_chars=len(old_doc_id) + len(new_doc_id) + len(correction_note),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def unsupersede_document(old_doc_id: str) -> str:
        """Atomically remove a supersession link from ``old_doc_id``.

        Clears ``superseded_by`` / ``doc_status`` on the old document and
        removes it from the replacement's ``supersedes`` list. Verification
        fields are left intact.
        """

        def _body() -> str:
            try:
                result = apply_unsupersede_document(
                    storage=storage,
                    qdrant=qdrant,
                    collection_name=collection_name,
                    old_doc_id=old_doc_id,
                )
                return json.dumps(result, ensure_ascii=False)
            except DocumentNotFoundError as exc:
                return f"Error unsupersede_document: {exc}"
            except (DocumentPatchError, DocumentMetadataError) as exc:
                return f"Error unsupersede_document: {exc}"
            except Exception as exc:  # noqa: BLE001 - return tool-visible error text
                return f"Error unsupersede_document: {exc}"

        return _run_mcp_tool(
            "unsupersede_document",
            request_chars=len(old_doc_id),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def search_domain_knowledge(
        query: str,
        limit: int = 3,
        obec_id: str = "",
        cast_obce: str = "",
        include_figures: bool = True,
        only_verified: bool = False,
        min_status: str = "",
        format: str = "text",
        include_superseded: bool = False,
    ) -> str:
        """Query local RAG vector storage for contextual facts.

        Optional ``obec_id`` / ``cast_obce`` filter hits whose denormalized
        ``localities`` / ``cast_obce`` payload arrays contain those values.
        ``only_verified`` / ``min_status`` apply to ``pdf_figure`` hits;
        ``rejected`` figures are always omitted. Empty strings omit filters.
        Superseded and retracted documents are omitted unless
        ``include_superseded`` is true. ``format=text`` (default) returns
        legacy Source/Content blocks; ``format=json`` returns structured
        hits including ``page_image_id``.
        """

        def _body() -> str:
            try:
                logger.info(
                    "MCP search_domain_knowledge query_chars=%s limit=%s",
                    len(query),
                    limit,
                )
                query_vector = get_query_embedding(query)
                locality_filter = knowledge_locality_filter(
                    obec_id=obec_id.strip() or None,
                    cast_obce=cast_obce.strip() or None,
                )
                fetch_limit = (
                    limit * 4 if (only_verified or min_status.strip()) else limit
                )
                results = search_similar(
                    qdrant,
                    query_vector=query_vector,
                    limit=fetch_limit,
                    collection_name=collection_name,
                    vector_name=TEXT_VECTOR_NAME,
                    query_filter=locality_filter,
                    include_superseded=include_superseded,
                )
                logger.info("MCP search returned %s hits", len(results))
                want_json = format.strip().lower() == "json"
                context_blocks: list[str] = []
                json_hits: list[dict[str, Any]] = []
                doc_geo_cache: dict[str, dict[str, Any] | None] = {}
                for res in results:
                    payload: dict[str, Any] = res.payload or {}
                    st = payload.get("source_type", "pdf_chunk")
                    if st == "pdf_figure":
                        if not include_figures:
                            continue
                        vstat = str(payload.get("verification_status") or "unverified")
                        if vstat == "rejected":
                            continue
                        if only_verified and vstat != "verified":
                            continue
                        if min_status.strip() == "auto_checked" and vstat not in (
                            "auto_checked",
                            "verified",
                        ):
                            continue
                    source_name = payload.get("parent_document") or payload.get(
                        "source", "unknown"
                    )
                    if len(context_blocks) >= limit and not want_json:
                        break
                    if len(json_hits) >= limit and want_json:
                        break
                    doc_id = payload.get("doc_id", "")
                    source_type = payload.get("source_type", "pdf_chunk")
                    page_number = payload.get("page_number")
                    image_id = payload.get("image_id")
                    text = payload.get("content_summary") or payload.get("text", "")
                    page_image_id = page_image_id_for_hit(
                        storage,
                        payload,
                        allowed_source_types=frozenset({"pdf_chunk", "pdf_figure"}),
                    )
                    hit_doc_id = str(doc_id or "")
                    if hit_doc_id not in doc_geo_cache:
                        doc_geo_cache[hit_doc_id] = document_geo_fallback(
                            storage, hit_doc_id
                        )
                    document_fallback = doc_geo_cache.get(hit_doc_id)
                    if want_json:
                        json_hits.append(
                            build_knowledge_hit(
                                storage,
                                payload,
                                score=getattr(res, "score", None),
                                text=str(text),
                                image_url_fn=_image_public_url,
                                document_fallback=document_fallback,
                            )
                        )
                        continue
                    header = f"Source: {source_name} (source_type={source_type})"
                    if doc_id:
                        header += f" doc_id={doc_id}"
                    if page_number is not None:
                        header += f" page_number={page_number}"
                    if page_image_id:
                        header += f" page_image_id={page_image_id}"
                        page_url = _image_public_url(str(doc_id), page_image_id)
                        if page_url:
                            header += f" page_url={page_url}"
                    if image_id:
                        url = _image_public_url(str(doc_id), str(image_id))
                        header += f" image_id={image_id}"
                        if url:
                            header += f" url={url}"
                    if st == "pdf_figure":
                        header += (
                            f" verification_status={payload.get('verification_status')}"
                            f" confidence={payload.get('confidence')}"
                        )
                        if payload.get("caption"):
                            header += f" caption={payload.get('caption')}"
                    header += geo_header_suffix(
                        payload, document_fallback=document_fallback
                    )
                    context_blocks.append(f"{header}\nContent: {text}\n")
                if want_json:
                    return json.dumps(json_hits, ensure_ascii=False, indent=2)
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
    def search_images(
        query: str,
        limit: int = 5,
        hybrid: bool = False,
        only_verified: bool = False,
        min_status: str = "",
        precision: str = "",
        needs_refinement: bool | None = None,
        format: str = "text",
        include_superseded: bool = False,
    ) -> str:
        """Search PDF images and figures by CLIP with optional text hybrid.

        Optional ``precision`` (municipality|cast_obce|street|building) and
        ``needs_refinement`` filter hits that have ``geo_refined`` metadata.
        Superseded and retracted documents are omitted unless
        ``include_superseded`` is true.
        """

        def _body() -> str:
            if not image_embedding_enabled():
                return (
                    "IMAGE_EMBEDDING_ENABLED is false. Set it to true and re-ingest "
                    "PDFs to index image_vector fields before using search_images."
                )
            try:
                validate_precision(precision.strip() or None)
                query_vector = embed_image_query(query)
                visual_hits = search_similar(
                    qdrant,
                    query_vector=query_vector,
                    limit=limit * 4,
                    collection_name=collection_name,
                    vector_name=IMAGE_VECTOR_NAME,
                    include_superseded=include_superseded,
                )
                if hybrid:
                    text_vector = get_query_embedding(query)
                    text_hits = search_similar(
                        qdrant,
                        query_vector=text_vector,
                        limit=limit * 4,
                        collection_name=collection_name,
                        vector_name=TEXT_VECTOR_NAME,
                        include_superseded=include_superseded,
                    )
                    filter_image = [
                        hit
                        for hit in text_hits
                        if (hit.payload or {}).get("source_type")
                        in ("pdf_image", "pdf_figure")
                    ]
                    results = _reciprocal_rank_fusion(
                        [visual_hits, filter_image], limit=limit * 4
                    )
                else:
                    results = visual_hits
                precision_n = validate_precision(precision.strip() or None)
                want_json = format.strip().lower() == "json"
                blocks: list[str] = []
                json_hits: list[dict[str, Any]] = []
                doc_geo_cache: dict[str, dict[str, Any] | None] = {}
                for res in results:
                    payload: dict[str, Any] = res.payload or {}
                    st = payload.get("source_type")
                    if st not in ("pdf_image", "pdf_figure"):
                        continue
                    if precision_n or needs_refinement is not None:
                        if not matches_geo_filter(
                            payload,
                            precision=precision_n,
                            needs_refinement=needs_refinement,
                        ):
                            continue
                    if st == "pdf_figure":
                        vstat = str(payload.get("verification_status") or "unverified")
                        if vstat == "rejected":
                            continue
                        if only_verified and vstat != "verified":
                            continue
                        if min_status.strip() == "auto_checked" and vstat not in (
                            "auto_checked",
                            "verified",
                        ):
                            continue
                    if len(blocks) >= limit and not want_json:
                        break
                    if len(json_hits) >= limit and want_json:
                        break
                    doc_id = str(payload.get("doc_id", ""))
                    image_id = str(payload.get("image_id", ""))
                    page_number = payload.get("page_number")
                    parent = payload.get("parent_document") or payload.get("source")
                    url = (
                        _image_public_url(doc_id, image_id)
                        if doc_id and image_id
                        else None
                    )
                    summary = payload.get("content_summary") or payload.get("text", "")
                    page_image_id = page_image_id_for_hit(
                        storage,
                        payload,
                        allowed_source_types=frozenset({"pdf_image", "pdf_figure"}),
                    )
                    hit_doc_id = str(doc_id or "")
                    if hit_doc_id not in doc_geo_cache:
                        doc_geo_cache[hit_doc_id] = document_geo_fallback(
                            storage, hit_doc_id
                        )
                    document_fallback = doc_geo_cache.get(hit_doc_id)
                    if want_json:
                        json_hits.append(
                            build_image_search_hit(
                                storage,
                                payload,
                                score=getattr(res, "score", None),
                                text=str(summary),
                                image_url_fn=_image_public_url,
                                document_fallback=document_fallback,
                            )
                        )
                        continue
                    header = f"Source: {parent} ({st})"
                    if doc_id:
                        header += f" doc_id={doc_id}"
                    if page_number is not None:
                        header += f" page_number={page_number}"
                    if page_image_id:
                        header += f" page_image_id={page_image_id}"
                        if doc_id:
                            page_url = _image_public_url(doc_id, page_image_id)
                            if page_url:
                                header += f" page_url={page_url}"
                    if image_id:
                        header += f" image_id={image_id}"
                    if url:
                        header += f" url={url}"
                    if st == "pdf_figure":
                        header += (
                            f" verification_status={payload.get('verification_status')}"
                            f" confidence={payload.get('confidence')}"
                        )
                        if payload.get("caption"):
                            header += f" caption={payload.get('caption')}"
                    header += geo_header_suffix(
                        payload, document_fallback=document_fallback
                    )
                    blocks.append(f"{header}\nContent: {summary}\n")
                if want_json:
                    return json.dumps(json_hits, ensure_ascii=False, indent=2)
                return "\n---\n".join(blocks) if blocks else "No relevant images found."
            except Exception as exc:  # noqa: BLE001
                return f"Error executing search_images: {exc}"

        return _run_mcp_tool(
            "search_images",
            request_chars=len(query) + len(precision),
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
        When TRAILS_API_URL and TRAILS_API_TOKEN are set, forwards the batch
        to TrailsAssemblage POST /locality-links. Failures are queued on the
        data volume and do not fail this tool. Does not call embedding APIs.
        """

        def _body() -> str:
            try:
                items = _coerce_locality_links(links)
                result = upsert_locality_links(qdrant, items)
                result["trails"] = _safe_trails_forward(
                    qdrant,
                    list(result.get("ids") or []),
                    outbox_path=storage.trails_outbox_path,
                )
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
        When Trails forwarding is enabled, PATCHes Trails after a local upsert
        (failures queue to the outbox and do not fail this tool).
        Does not call embedding APIs.
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
                trails_id = str((result.get("link") or {}).get("link_id") or link_id)
                result["trails"] = _safe_trails_forward(
                    qdrant,
                    [trails_id],
                    outbox_path=storage.trails_outbox_path,
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

    @mcp.tool()
    def verify_image_record(
        point_id: str = "",
        doc_id: str = "",
        figure_id: str = "",
        status: str = "verified",
        verified_by: str = "",
        notes: str = "",
        caption: str = "",
        summary: str = "",
    ) -> str:
        """Set verification on a ``pdf_figure`` point and re-embed text if needed."""

        def _body() -> str:
            try:
                result = apply_verify_image_record(
                    qdrant,
                    collection_name=collection_name,
                    point_id=point_id.strip() or None,
                    doc_id=doc_id.strip() or None,
                    figure_id=figure_id.strip() or None,
                    status=status,
                    verified_by=verified_by.strip() or None,
                    notes=notes.strip() or None,
                    caption=caption.strip() or None,
                    summary=summary.strip() or None,
                )
                return json.dumps(result, ensure_ascii=False)
            except ImageRecordPayloadError as exc:
                return f"Error verify_image_record: {exc}"

        return _run_mcp_tool(
            "verify_image_record",
            request_chars=len(point_id) + len(doc_id) + len(figure_id),
            source="ingest_mcp",
            fn=_body,
        )

    @mcp.tool()
    def list_image_records(
        doc_id: str = "",
        status: str = "",
        precision: str = "",
        needs_refinement: bool | None = None,
        limit: int = 50,
        include_superseded: bool = False,
    ) -> str:
        """List ``pdf_figure`` records with optional doc_id and status filters.

        Optional ``precision`` and ``needs_refinement`` filter on ``geo_refined``.
        Coordinates prefer ``geo_refined`` when present (``geo_source``).
        Superseded and retracted documents are omitted unless
        ``include_superseded`` is true.
        """

        def _body() -> str:
            try:
                result = scroll_image_records(
                    qdrant,
                    collection_name=collection_name,
                    doc_id=doc_id.strip() or None,
                    status=status.strip() or None,
                    precision=precision.strip() or None,
                    needs_refinement=needs_refinement,
                    limit=limit,
                    include_superseded=include_superseded,
                )
                return json.dumps(result, ensure_ascii=False)
            except ImageRecordPayloadError as exc:
                return f"Error list_image_records: {exc}"

        return _run_mcp_tool(
            "list_image_records",
            request_chars=len(doc_id) + len(status) + len(precision),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def refine_geo(
        doc_id: str,
        geo_refined: dict[str, Any],
        chunk_index: int | None = None,
        image_id: str = "",
        dry_run: bool = True,
    ) -> str:
        """Store a refined GPS location beside original coordinates.

        Never overwrites original ``lat`` / ``lon``. ``dry_run=true`` (default)
        returns old vs new without writing. Writes are idempotent and append
        previous ``geo_refined`` values to ``geo_refined_history``.
        Target an image record with ``image_id``, a chunk with ``chunk_index``,
        or omit both to update document metadata and image records for ``doc_id``.
        Same bearer auth as other MCP write tools (e.g. ``link_locality``).
        """

        def _body() -> str:
            try:
                result = apply_refine_geo(
                    storage=storage,
                    qdrant=qdrant,
                    collection_name=collection_name,
                    doc_id=doc_id,
                    geo_refined=geo_refined,
                    chunk_index=chunk_index,
                    image_id=image_id.strip() or None,
                    dry_run=bool(dry_run),
                )
                return json.dumps(result, ensure_ascii=False)
            except GeoRefinedNotFoundError as exc:
                return f"Error refine_geo: {exc}"
            except GeoRefinedPayloadError as exc:
                return f"Error refine_geo: {exc}"

        return _run_mcp_tool(
            "refine_geo",
            request_chars=len(doc_id)
            + len(image_id)
            + len(json.dumps(geo_refined, ensure_ascii=False, default=str)),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def refine_geo_batch(
        items: list[dict[str, Any]],
        dry_run: bool = True,
    ) -> str:
        """Batch ``refine_geo`` for post-deploy backfill of existing records.

        Each item is ``{doc_id, chunk_index?, image_id?, geo_refined}``.
        ``dry_run=true`` (default) reports ``would_change`` without writing.
        Per-item ``ok`` / ``would_change`` / ``changed`` / ``error``; one bad
        item does not abort the rest. Max ``200`` items per call (page larger
        backfills). Same history, idempotency, and auth as ``refine_geo``.
        """

        def _body() -> str:
            try:
                result = apply_refine_geo_batch(
                    storage=storage,
                    qdrant=qdrant,
                    collection_name=collection_name,
                    items=items,
                    dry_run=bool(dry_run),
                )
                return json.dumps(result, ensure_ascii=False)
            except GeoRefinedPayloadError as exc:
                return f"Error refine_geo_batch: {exc}"

        return _run_mcp_tool(
            "refine_geo_batch",
            request_chars=len(json.dumps(items, ensure_ascii=False, default=str)),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def list_geo_refinements(
        filter: dict[str, Any] | None = None,
        precision: str = "",
        status: str = "",
        confidence_level: str = "",
        needs_refinement: bool | None = None,
        verified: bool | None = None,
        doc_id: str = "",
        limit: int = 50,
        offset: int = 0,
        include_superseded: bool = False,
    ) -> str:
        """List stored ``geo_refined`` records with filters and pagination.

        ``filter`` may include precision, status, confidence_level,
        needs_refinement, verified, doc_id, limit, offset, and
        include_superseded. Individual arguments are used when the filter
        omits them. Does not call embeddings. Superseded and retracted
        documents are omitted unless ``include_superseded`` is true.
        """

        def _body() -> str:
            try:
                result = list_geo_refinement_records(
                    storage=storage,
                    qdrant=qdrant,
                    collection_name=collection_name,
                    filters=filter,
                    precision=precision.strip() or None,
                    status=status.strip() or None,
                    confidence_level=confidence_level.strip() or None,
                    needs_refinement=needs_refinement,
                    verified=verified,
                    doc_id=doc_id.strip() or None,
                    limit=limit,
                    offset=offset,
                    include_superseded=include_superseded,
                )
                return json.dumps(result, ensure_ascii=False)
            except GeoRefinedPayloadError as exc:
                return f"Error list_geo_refinements: {exc}"

        return _run_mcp_tool(
            "list_geo_refinements",
            request_chars=len(doc_id) + len(precision) + len(status),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def image_record_stats(doc_id: str) -> str:
        """Aggregate ``pdf_figure`` verification counts for one document."""

        def _body() -> str:
            try:
                result = aggregate_image_record_stats(
                    qdrant,
                    collection_name=collection_name,
                    doc_id=doc_id.strip(),
                )
                return json.dumps(result, ensure_ascii=False)
            except ImageRecordPayloadError as exc:
                return f"Error image_record_stats: {exc}"

        return _run_mcp_tool(
            "image_record_stats",
            request_chars=len(doc_id),
            source=None,
            fn=_body,
        )

    @mcp.tool()
    def reindex_figures(
        doc_id: str = "",
        all: bool = False,
        dry_run: bool = True,
        force: bool = False,
    ) -> str:
        """Re-run ``pdf_figure`` indexing from stored PDFs (no chunk re-embed)."""

        def _body() -> str:
            try:
                if all:
                    docs = storage.list_documents()
                    totals = 0
                    for entry in docs:
                        did = str(entry.get("doc_id") or "")
                        if not did:
                            continue
                        meta = storage.read_meta(did)
                        if meta is None or not meta.filename.lower().endswith(".pdf"):
                            continue
                        result = reindex_figures_for_doc(
                            storage=storage,
                            qdrant=qdrant,
                            collection_name=collection_name,
                            doc_id=did,
                            dry_run=bool(dry_run),
                            force=force,
                        )
                        totals += int(result.get("figures_reindexed") or 0)
                    return json.dumps(
                        {
                            "status": "ok",
                            "figures_reindexed": totals,
                            "dry_run": dry_run,
                        },
                        ensure_ascii=False,
                    )
                if not doc_id.strip():
                    return "Error reindex_figures: doc_id required unless all=true"
                result = reindex_figures_for_doc(
                    storage=storage,
                    qdrant=qdrant,
                    collection_name=collection_name,
                    doc_id=doc_id.strip(),
                    dry_run=bool(dry_run),
                    force=force,
                )
                return json.dumps(result, ensure_ascii=False)
            except ImageRecordPayloadError as exc:
                return f"Error reindex_figures: {exc}"

        return _run_mcp_tool(
            "reindex_figures",
            request_chars=len(doc_id),
            source="ingest_mcp",
            fn=_body,
        )

    @mcp.tool()
    def sync_links_to_trails(dry_run: bool = False) -> str:
        """Backfill all locality_links points to TrailsAssemblage (idempotent).

        POSTs existing points to /locality-links in batches, then PATCHes
        verify for verified/rejected rows. ``dry_run=true`` counts without
        HTTP. Disabled when TRAILS_API_URL or TRAILS_API_TOKEN is unset
        (dry_run still reports local counts). Failures queue to the durable
        outbox. Does not call embedding APIs. Does not delete anything.
        """

        def _body() -> str:
            try:
                result = run_sync_links_to_trails(
                    qdrant,
                    outbox_path=storage.trails_outbox_path,
                    dry_run=bool(dry_run),
                )
                return json.dumps(result, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001
                return f"Error sync_links_to_trails: {exc}"

        return _run_mcp_tool(
            "sync_links_to_trails",
            request_chars=len(str(dry_run)),
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
