"""Ingest documents into persistent storage and the vector index."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, Callable

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from rag.embeddings import get_embeddings
from rag.logging_setup import get_logger
from rag.ocr import build_content_summary, ocr_image_file
from rag.payloads import build_chunk_payload, build_image_payload
from rag.pdf_images import extract_pdf_images
from rag.processing import guess_mime, is_supported_filename, process_document
from rag.storage import DocumentMeta, ImageMeta, RagStorage

EmbedFn = Callable[[str], list[float]]
logger = get_logger("rag.ingest")


def content_sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def stable_point_id(doc_id: str, chunk_index: int) -> str:
    """Build a stable UUID point id for a document chunk."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk_index}"))


def stable_image_point_id(doc_id: str, image_id: str) -> str:
    """Build a stable UUID point id for a PDF image OCR hit."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:img:{image_id}"))


def _is_pdf_filename(filename: str) -> bool:
    """Return whether ``filename`` refers to a PDF upload."""
    return Path(filename).suffix.lower() == ".pdf"


def _process_pdf_images(
    *,
    storage: RagStorage,
    doc_id: str,
    data: bytes,
) -> list[ImageMeta]:
    """Extract, store, and OCR embedded PDF images."""
    images_meta: list[ImageMeta] = []
    try:
        extracted = extract_pdf_images(data)
    except Exception as exc:  # noqa: BLE001 - PDF images are optional enrichment
        logger.warning("PDF image extraction failed doc_id=%s: %s", doc_id, exc)
        return images_meta

    for item in extracted:
        dest = storage.save_image_bytes(
            doc_id,
            image_id=item.image_id,
            extension=item.extension,
            data=item.data,
        )
        rel_path = storage.relative_to_root(dest)
        ocr_text, ocr_confidence = ocr_image_file(dest)
        images_meta.append(
            ImageMeta(
                image_id=item.image_id,
                sha256=item.sha256,
                file_path=rel_path,
                page_number=item.page_number,
                width=item.width,
                height=item.height,
                mime=item.mime,
                ocr_text=ocr_text or None,
                ocr_confidence=ocr_confidence,
            )
        )
    return images_meta


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
    images: list[ImageMeta] = []
    if _is_pdf_filename(filename):
        images = _process_pdf_images(storage=storage, doc_id=doc_id, data=data)

    meta = storage.write_processed(
        doc_id=doc_id,
        filename=filename,
        mime=mime,
        sha256=digest,
        fulltext=fulltext,
        chunks=chunks,
        images=images,
        status="ready",
    )

    texts = [str(chunk["text"]) for chunk in chunks]
    image_summaries: list[tuple[ImageMeta, str]] = []
    for img in images:
        if img.ocr_text:
            image_summaries.append(
                (img, build_content_summary(img.page_number, img.ocr_text))
            )

    embed_inputs = texts + [summary for _, summary in image_summaries]
    if embed_fn is not None:
        vectors = [embed_fn(text) for text in embed_inputs]
    else:
        vectors = get_embeddings(embed_inputs) if embed_inputs else []

    chunk_vectors = vectors[: len(texts)]
    image_vectors = vectors[len(texts) :]

    points: list[PointStruct] = []
    for chunk, text, vector in zip(chunks, texts, chunk_vectors, strict=True):
        page_number = int(chunk.get("page", 0))
        points.append(
            PointStruct(
                id=stable_point_id(doc_id, int(chunk["index"])),
                vector=vector,
                payload=build_chunk_payload(
                    doc_id=doc_id,
                    parent_document=filename,
                    page_number=page_number,
                    doc_sha256=digest,
                    text=text,
                    chunk_index=int(chunk["index"]),
                    source_url=source_url,
                ),
            )
        )

    for (img, summary), vector in zip(image_summaries, image_vectors, strict=True):
        points.append(
            PointStruct(
                id=stable_image_point_id(doc_id, img.image_id),
                vector=vector,
                payload=build_image_payload(
                    doc_id=doc_id,
                    parent_document=filename,
                    page_number=img.page_number,
                    file_path=img.file_path,
                    image_id=img.image_id,
                    image_sha256=img.sha256,
                    content_summary=summary,
                ),
            )
        )

    if points:
        qdrant.upsert(collection_name=collection_name, points=points)

    result = {
        "status": "Success",
        "doc_id": meta.doc_id,
        "filename": meta.filename,
        "chunks_indexed": meta.chunk_count,
        "images_indexed": len(image_summaries),
        "images_extracted": len(images),
        "sha256": meta.sha256,
        "reused": False,
    }
    if source_url:
        result["source_url"] = source_url
    logger.info(
        "Ingest complete doc_id=%s filename=%s chunks_indexed=%s images_indexed=%s",
        meta.doc_id,
        meta.filename,
        meta.chunk_count,
        len(image_summaries),
    )
    return result


def meta_to_public(meta: DocumentMeta) -> dict[str, Any]:
    """Shape document metadata for HTTP/MCP responses."""
    return meta.to_dict()
