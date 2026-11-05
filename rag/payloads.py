"""Qdrant payload shapes for text chunks and PDF image OCR hits."""

from __future__ import annotations

from typing import Any


def build_chunk_payload(
    *,
    doc_id: str,
    parent_document: str,
    page_number: int,
    doc_sha256: str,
    text: str,
    chunk_index: int,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Build the payload for a ``pdf_chunk`` (or non-PDF text chunk) point."""
    payload: dict[str, Any] = {
        "source_type": "pdf_chunk",
        "doc_id": doc_id,
        "parent_document": parent_document,
        "page_number": page_number,
        "sha256": doc_sha256,
        "content_summary": text,
        "text": text,
        "chunk_index": chunk_index,
        "source": parent_document,
    }
    if source_url:
        payload["source_url"] = source_url
    return payload


def build_image_payload(
    *,
    doc_id: str,
    parent_document: str,
    page_number: int,
    file_path: str,
    image_id: str,
    image_sha256: str,
    content_summary: str,
) -> dict[str, Any]:
    """Build the payload for a ``pdf_image`` point (OCR text embedded as vector)."""
    return {
        "source_type": "pdf_image",
        "doc_id": doc_id,
        "parent_document": parent_document,
        "page_number": page_number,
        "file_path": file_path,
        "image_id": image_id,
        "sha256": image_sha256,
        "content_summary": content_summary,
        "text": content_summary,
        "source": parent_document,
    }
