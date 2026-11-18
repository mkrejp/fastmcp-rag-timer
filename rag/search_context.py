"""Attach reader-local context around RAG search hits.

Context is loaded from disk catalog metadata, version indexes, locality_links,
supersession peers, and neighboring processed chunks. No extra embedding
models are loaded — query vectors still use the configured query provider
(Novita by default on ``APP_MODE=reader``).
"""

from __future__ import annotations

from typing import Any, Mapping

from qdrant_client import QdrantClient

from rag.document_delete import collect_supersession_peers
from rag.document_replace import DocumentNotFoundError, list_document_versions
from rag.search_hits import build_knowledge_hit
from rag.storage import DocumentMeta, RagStorage
from rag.vectorstore import list_locality_links

DEFAULT_NEIGHBOR_RADIUS = 1
DEFAULT_LOCALITY_LIMIT = 20
DEFAULT_RELATED_LIMIT = 10
MAX_NEIGHBOR_RADIUS = 3
MAX_LOCALITY_LIMIT = 50
MAX_RELATED_LIMIT = 20
_CATALOG_KEYS = (
    "doc_id",
    "filename",
    "mime",
    "sha256",
    "uploaded_at",
    "chunk_count",
    "status",
    "figure_index_version",
    "figures_count",
    "figures_status",
)
_LIFECYCLE_EXTRA_KEYS = (
    "doc_status",
    "superseded_by",
    "supersedes",
    "correction_note",
    "verification_status",
    "reliability",
    "source_key",
    "source_axis",
)


def _clamp_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    """Coerce ``value`` to an int within ``[minimum, maximum]``."""
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)


def _chunk_index_of(chunk: Mapping[str, Any]) -> int | None:
    """Return the chunk index from a processed chunk or Qdrant payload."""
    raw = chunk.get("chunk_index", chunk.get("index"))
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def catalog_entry_for_doc(
    storage: RagStorage, doc_id: str
) -> dict[str, Any] | None:
    """Return a compact catalog entry for ``doc_id``, or ``None`` if missing."""
    trimmed = doc_id.strip()
    if not trimmed:
        return None
    for entry in storage.list_documents():
        if str(entry.get("doc_id") or "") == trimmed:
            return {
                key: entry.get(key)
                for key in _CATALOG_KEYS
                if key in entry and entry.get(key) not in (None, "")
            }
    meta = storage.read_meta(trimmed)
    if meta is None:
        return None
    return _catalog_from_meta(meta)


def _catalog_from_meta(meta: DocumentMeta) -> dict[str, Any]:
    """Build a catalog-shaped summary from ``DocumentMeta``."""
    return {
        "doc_id": meta.doc_id,
        "filename": meta.filename,
        "mime": meta.mime,
        "sha256": meta.sha256,
        "uploaded_at": meta.uploaded_at,
        "chunk_count": meta.chunk_count,
        "status": meta.status,
        "figure_index_version": meta.figure_index_version,
        "figures_count": meta.figures_count,
        "figures_status": meta.figures_status,
    }


def lifecycle_from_meta(meta: DocumentMeta | None) -> dict[str, Any]:
    """Extract lifecycle / verification fields from document ``extra``."""
    if meta is None:
        return {}
    out: dict[str, Any] = {}
    for key in _LIFECYCLE_EXTRA_KEYS:
        value = meta.extra.get(key)
        if value not in (None, "", []):
            out[key] = value
    return out


def version_summary_for_doc(
    storage: RagStorage, doc_id: str
) -> dict[str, Any] | None:
    """Return version index metadata without archived fulltext bodies."""
    trimmed = doc_id.strip()
    if not trimmed:
        return None
    try:
        listed = list_document_versions(
            storage=storage, doc_id=trimmed, include_text=False
        )
    except DocumentNotFoundError:
        return None
    versions = listed.get("versions") or []
    return {
        "count": len(versions),
        "next_version": listed.get("next_version"),
        "versions": versions,
    }


def locality_links_for_doc(
    qdrant: QdrantClient,
    doc_id: str,
    *,
    limit: int = DEFAULT_LOCALITY_LIMIT,
) -> dict[str, Any]:
    """Return a page of locality links for ``doc_id`` (no embeddings)."""
    trimmed = doc_id.strip()
    if not trimmed:
        return {"links": [], "returned": 0, "has_more": False}
    limit_n = _clamp_int(
        limit,
        default=DEFAULT_LOCALITY_LIMIT,
        minimum=1,
        maximum=MAX_LOCALITY_LIMIT,
    )
    result = list_locality_links(qdrant, doc_id=trimmed, limit=limit_n, offset=0)
    compact: list[dict[str, Any]] = []
    for link in result.get("links") or []:
        if not isinstance(link, dict):
            continue
        compact.append(
            {
                key: link.get(key)
                for key in (
                    "link_id",
                    "obec_id",
                    "name",
                    "cast_obce",
                    "cast_obce_kod",
                    "cast_obce_name",
                    "locality_level",
                    "chunk_index",
                    "matched_text",
                    "verification_status",
                    "method",
                    "ambiguous",
                    "confidence",
                    "confidence_level",
                    "page_number",
                )
                if link.get(key) not in (None, "")
            }
        )
    return {
        "links": compact,
        "returned": len(compact),
        "has_more": bool(result.get("has_more")),
        "limit": limit_n,
    }


def related_documents_for_doc(
    storage: RagStorage,
    doc_id: str,
    *,
    limit: int = DEFAULT_RELATED_LIMIT,
) -> list[dict[str, Any]]:
    """Return supersession-related catalog summaries for ``doc_id``."""
    trimmed = doc_id.strip()
    if not trimmed:
        return []
    limit_n = _clamp_int(
        limit,
        default=DEFAULT_RELATED_LIMIT,
        minimum=1,
        maximum=MAX_RELATED_LIMIT,
    )
    peers = collect_supersession_peers(storage, trimmed)[:limit_n]
    related: list[dict[str, Any]] = []
    for peer_id in peers:
        entry = catalog_entry_for_doc(storage, peer_id)
        meta = storage.read_meta(peer_id)
        item: dict[str, Any] = entry or {"doc_id": peer_id}
        lifecycle = lifecycle_from_meta(meta)
        if lifecycle:
            item["lifecycle"] = lifecycle
        related.append(item)
    return related


def neighbor_chunks_for_hit(
    storage: RagStorage,
    doc_id: str,
    chunk_index: int | None,
    *,
    radius: int = DEFAULT_NEIGHBOR_RADIUS,
) -> list[dict[str, Any]]:
    """Return neighboring processed chunks around ``chunk_index`` (disk only)."""
    trimmed = doc_id.strip()
    if not trimmed or chunk_index is None:
        return []
    radius_n = _clamp_int(
        radius,
        default=DEFAULT_NEIGHBOR_RADIUS,
        minimum=0,
        maximum=MAX_NEIGHBOR_RADIUS,
    )
    chunks = storage.read_chunks(trimmed)
    if not chunks:
        return []
    by_index: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        index = _chunk_index_of(chunk)
        if index is None:
            continue
        by_index[index] = chunk
    neighbors: list[dict[str, Any]] = []
    for index in range(chunk_index - radius_n, chunk_index + radius_n + 1):
        if index == chunk_index or index not in by_index:
            continue
        chunk = by_index[index]
        text = str(chunk.get("text") or "")
        neighbors.append(
            {
                "chunk_index": index,
                "page": chunk.get("page"),
                "text": text,
            }
        )
    return neighbors


def build_document_context(
    storage: RagStorage,
    qdrant: QdrantClient,
    doc_id: str,
    *,
    chunk_index: int | None = None,
    include_catalog: bool = True,
    include_versions: bool = True,
    include_locality: bool = True,
    include_related: bool = True,
    include_neighbors: bool = True,
    neighbor_radius: int = DEFAULT_NEIGHBOR_RADIUS,
    locality_limit: int = DEFAULT_LOCALITY_LIMIT,
    related_limit: int = DEFAULT_RELATED_LIMIT,
) -> dict[str, Any] | None:
    """Assemble reader-local context for one document (no embeddings)."""
    trimmed = doc_id.strip()
    if not trimmed:
        return None
    meta = storage.read_meta(trimmed)
    catalog = catalog_entry_for_doc(storage, trimmed)
    if meta is None and catalog is None:
        return None
    context: dict[str, Any] = {"doc_id": trimmed}
    if include_catalog:
        context["catalog"] = catalog or (
            _catalog_from_meta(meta) if meta is not None else None
        )
        lifecycle = lifecycle_from_meta(meta)
        if lifecycle:
            context["lifecycle"] = lifecycle
        if meta is not None:
            geo: dict[str, Any] = {}
            if meta.lat is not None:
                geo["lat"] = meta.lat
            if meta.lon is not None:
                geo["lon"] = meta.lon
            if meta.geo_refined:
                geo["geo_refined"] = meta.geo_refined
            if geo:
                context["geo"] = geo
    if include_versions:
        context["versions"] = version_summary_for_doc(storage, trimmed)
    if include_locality:
        context["locality_links"] = locality_links_for_doc(
            qdrant, trimmed, limit=locality_limit
        )
    if include_related:
        context["related_documents"] = related_documents_for_doc(
            storage, trimmed, limit=related_limit
        )
    if include_neighbors and chunk_index is not None:
        context["neighbor_chunks"] = neighbor_chunks_for_hit(
            storage,
            trimmed,
            chunk_index,
            radius=neighbor_radius,
        )
    return context


def enrich_knowledge_hit(
    storage: RagStorage,
    qdrant: QdrantClient,
    *,
    payload: Mapping[str, Any],
    score: float | None,
    text: str,
    image_url_fn: Any,
    document_fallback: dict[str, Any] | None,
    include_catalog: bool = True,
    include_versions: bool = True,
    include_locality: bool = True,
    include_related: bool = True,
    include_neighbors: bool = True,
    neighbor_radius: int = DEFAULT_NEIGHBOR_RADIUS,
    locality_limit: int = DEFAULT_LOCALITY_LIMIT,
    related_limit: int = DEFAULT_RELATED_LIMIT,
) -> dict[str, Any]:
    """Build a knowledge hit and attach reader-local context sections."""
    hit = build_knowledge_hit(
        storage,
        dict(payload),
        score=score,
        text=text,
        image_url_fn=image_url_fn,
        document_fallback=document_fallback,
    )
    doc_id = str(payload.get("doc_id") or hit.get("doc_id") or "")
    chunk_index = _chunk_index_of(payload)
    if chunk_index is not None:
        hit["chunk_index"] = chunk_index
    context = build_document_context(
        storage,
        qdrant,
        doc_id,
        chunk_index=chunk_index,
        include_catalog=include_catalog,
        include_versions=include_versions,
        include_locality=include_locality,
        include_related=include_related,
        include_neighbors=include_neighbors,
        neighbor_radius=neighbor_radius,
        locality_limit=locality_limit,
        related_limit=related_limit,
    )
    if context is not None:
        # Flatten context sections onto the hit (keep doc_id once).
        for key, value in context.items():
            if key == "doc_id":
                continue
            hit[key] = value
    return hit


def search_catalog(
    storage: RagStorage,
    *,
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Filter the on-disk catalog by substring match (no embeddings).

    Matches ``filename``, ``doc_id``, ``sha256``, ``mime``, and ``status``.
    """
    needle = query.strip().lower()
    limit_n = _clamp_int(limit, default=20, minimum=1, maximum=100)
    if not needle:
        return storage.list_documents()[:limit_n]
    hits: list[dict[str, Any]] = []
    for entry in storage.list_documents():
        haystacks = (
            str(entry.get("filename") or ""),
            str(entry.get("doc_id") or ""),
            str(entry.get("sha256") or ""),
            str(entry.get("mime") or ""),
            str(entry.get("status") or ""),
        )
        if any(needle in part.lower() for part in haystacks):
            hits.append(
                {
                    key: entry.get(key)
                    for key in _CATALOG_KEYS
                    if key in entry and entry.get(key) not in (None, "")
                }
            )
        if len(hits) >= limit_n:
            break
    return hits
