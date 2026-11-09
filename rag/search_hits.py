"""Shared helpers for MCP search hit formatting."""

from __future__ import annotations

from typing import Any, Callable

from rag.geo_refined import attach_preferred_geo, preferred_geo
from rag.page_ids import page_image_id_from_payload_page
from rag.storage import RagStorage


def page_image_id_for_hit(
    storage: RagStorage,
    payload: dict[str, Any],
    *,
    allowed_source_types: frozenset[str],
) -> str | None:
    """Return ``page:{doc_id}:{page}`` for PDF hits when a page is known."""
    source_type = str(payload.get("source_type") or "")
    if source_type not in allowed_source_types:
        return None
    doc_id = str(payload.get("doc_id") or "")
    if not doc_id or not storage.document_is_pdf(doc_id):
        return None
    page_number = payload.get("page_number")
    if page_number is None:
        return None
    try:
        page_zb = int(page_number)
    except (TypeError, ValueError):
        return None
    return page_image_id_from_payload_page(doc_id, page_zb)


def geo_fallback_payload(
    payload: dict[str, Any],
    document_fallback: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge document-level geo under point payload (payload wins)."""
    merged = dict(document_fallback or {})
    merged.update(payload)
    return merged


def geo_header_suffix(
    payload: dict[str, Any],
    *,
    document_fallback: dict[str, Any] | None = None,
) -> str:
    """Return ``lat=... lon=... geo_source=...`` when coordinates exist."""
    geo = preferred_geo(payload, fallback=document_fallback)
    if geo["lat"] is None or geo["lon"] is None:
        return ""
    return f" lat={geo['lat']} lon={geo['lon']} geo_source={geo['geo_source']}"


def build_knowledge_hit(
    storage: RagStorage,
    payload: dict[str, Any],
    *,
    score: float | None,
    text: str,
    image_url_fn: Callable[[str, str], str | None],
    document_fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one structured hit for ``search_domain_knowledge`` JSON output."""
    doc_id = str(payload.get("doc_id") or "")
    image_id = payload.get("image_id")
    image_id_str = str(image_id) if image_id else None
    parent = payload.get("parent_document") or payload.get("source")
    source_type = str(payload.get("source_type") or "pdf_chunk")
    page_number = payload.get("page_number")
    page_image_id = page_image_id_for_hit(
        storage,
        payload,
        allowed_source_types=frozenset({"pdf_chunk", "pdf_figure"}),
    )
    url = (
        image_url_fn(doc_id, image_id_str)
        if doc_id and image_id_str
        else None
    )
    hit: dict[str, Any] = {
        "doc_id": doc_id or None,
        "parent_document": parent,
        "source_type": source_type,
        "page_number": page_number,
        "page_image_id": page_image_id,
        "image_id": image_id_str,
        "url": url,
        "score": score,
        "content": text,
    }
    if source_type == "pdf_figure":
        hit["verification_status"] = payload.get("verification_status")
        hit["confidence"] = payload.get("confidence")
        if payload.get("caption"):
            hit["caption"] = payload.get("caption")
    return attach_preferred_geo(
        hit, fallback=geo_fallback_payload(payload, document_fallback)
    )


def build_image_search_hit(
    storage: RagStorage,
    payload: dict[str, Any],
    *,
    score: float | None,
    text: str,
    image_url_fn: Callable[[str, str], str | None],
    document_fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one structured hit for ``search_images`` JSON output."""
    doc_id = str(payload.get("doc_id") or "")
    image_id_str = str(payload.get("image_id") or "")
    parent = payload.get("parent_document") or payload.get("source")
    source_type = str(payload.get("source_type") or "")
    page_number = payload.get("page_number")
    page_image_id = page_image_id_for_hit(
        storage,
        payload,
        allowed_source_types=frozenset({"pdf_image", "pdf_figure"}),
    )
    url = image_url_fn(doc_id, image_id_str) if doc_id and image_id_str else None
    hit: dict[str, Any] = {
        "doc_id": doc_id or None,
        "parent_document": parent,
        "source_type": source_type,
        "page_number": page_number,
        "page_image_id": page_image_id,
        "image_id": image_id_str or None,
        "url": url,
        "score": score,
        "content": text,
    }
    if source_type == "pdf_figure":
        hit["verification_status"] = payload.get("verification_status")
        hit["confidence"] = payload.get("confidence")
        if payload.get("caption"):
            hit["caption"] = payload.get("caption")
    return attach_preferred_geo(
        hit, fallback=geo_fallback_payload(payload, document_fallback)
    )
