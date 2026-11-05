"""Ingest documents into persistent storage and the vector index."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Callable

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from rag.embeddings import get_embeddings
from rag.logging_setup import get_logger
from rag.processing import guess_mime, is_supported_filename, process_document
from rag.storage import DocumentMeta, RagStorage

EmbedFn = Callable[[str], list[float]]
logger = get_logger("rag.ingest")


def content_sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def stable_point_id(doc_id: str, chunk_index: int) -> str:
    """Build a stable UUID point id for a document chunk."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk_index}"))


def ingest_bytes(
    *,
    storage: RagStorage,
    qdrant: QdrantClient,
    collection_name: str,
    filename: str,
    data: bytes,
    embed_fn: EmbedFn | None = None,
    skip_duplicate: bool = True,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Persist, process, and index a document from raw bytes."""
    if not filename:
        raise ValueError("filename is required")
    if not is_supported_filename(filename):
        raise ValueError(
            "Only PDF, TXT, Markdown, JSON, and GeoJSON files are supported."
        )

    digest = content_sha256(data)
    if skip_duplicate:
        existing = storage.find_by_sha256(digest)
        if existing is not None:
            reused = {
                "status": "exists",
                "doc_id": existing.doc_id,
                "filename": existing.filename,
                "chunks_indexed": existing.chunk_count,
                "sha256": existing.sha256,
                "reused": True,
            }
            if source_url:
                reused["source_url"] = source_url
            logger.info(
                "Ingest reused existing doc_id=%s filename=%s chunks=%s",
                existing.doc_id,
                existing.filename,
                existing.chunk_count,
            )
            return reused

    doc_id = str(uuid.uuid4())
    logger.info(
        "Ingesting filename=%s bytes=%s source_url=%s doc_id=%s",
        filename,
        len(data),
        source_url,
        doc_id,
    )
    storage.save_original(doc_id, filename, data)
    fulltext, chunks = process_document(filename, data)
    mime = guess_mime(filename)
    meta = storage.write_processed(
        doc_id=doc_id,
        filename=filename,
        mime=mime,
        sha256=digest,
        fulltext=fulltext,
        chunks=chunks,
        status="ready",
    )

    texts = [str(chunk["text"]) for chunk in chunks]
    if embed_fn is not None:
        vectors = [embed_fn(text) for text in texts]
    else:
        vectors = get_embeddings(texts)
    points: list[PointStruct] = []
    for chunk, text, vector in zip(chunks, texts, vectors, strict=True):
        points.append(
            PointStruct(
                id=stable_point_id(doc_id, int(chunk["index"])),
                vector=vector,
                payload={
                    "text": text,
                    "source": filename,
                    "doc_id": doc_id,
                    "page": int(chunk.get("page", 0)),
                    "chunk_index": int(chunk["index"]),
                    **({"source_url": source_url} if source_url else {}),
                },
            )
        )
    if points:
        qdrant.upsert(collection_name=collection_name, points=points)

    result = {
        "status": "Success",
        "doc_id": meta.doc_id,
        "filename": meta.filename,
        "chunks_indexed": meta.chunk_count,
        "sha256": meta.sha256,
        "reused": False,
    }
    if source_url:
        result["source_url"] = source_url
    logger.info(
        "Ingest complete doc_id=%s filename=%s chunks_indexed=%s",
        meta.doc_id,
        meta.filename,
        meta.chunk_count,
    )
    return result


def meta_to_public(meta: DocumentMeta) -> dict[str, Any]:
    """Shape document metadata for HTTP/MCP responses."""
    return meta.to_dict()
