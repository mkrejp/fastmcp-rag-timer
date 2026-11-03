"""Document text extraction and chunking helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from rag.geojson import geojson_to_records

DEFAULT_CHUNK_SIZE = 1000

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".text"}
PDF_EXTENSIONS = {".pdf"}
JSON_EXTENSIONS = {".json", ".geojson"}


def guess_mime(filename: str) -> str:
    """Return a simple MIME type based on filename extension."""
    suffix = Path(filename).suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return "application/pdf"
    if suffix == ".md" or suffix == ".markdown":
        return "text/markdown"
    if suffix in TEXT_EXTENSIONS:
        return "text/plain"
    if suffix == ".geojson":
        return "application/geo+json"
    if suffix in JSON_EXTENSIONS:
        return "application/json"
    return "application/octet-stream"


def is_supported_filename(filename: str) -> bool:
    """Return whether ``filename`` is an allowed upload type."""
    suffix = Path(filename).suffix.lower()
    return suffix in PDF_EXTENSIONS or suffix in TEXT_EXTENSIONS or suffix in JSON_EXTENSIONS


def extract_pdf_text(data: bytes) -> tuple[str, list[dict[str, Any]]]:
    """Extract full text and page-aware segments from PDF bytes."""
    reader = PdfReader(io.BytesIO(data))
    pages: list[dict[str, Any]] = []
    parts: list[str] = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": page_num, "text": text})
        if text:
            parts.append(text)
    fulltext = "\n".join(parts).strip()
    return fulltext, pages


def extract_text_bytes(data: bytes) -> str:
    """Decode plain-text or markdown bytes to a normalized string."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace").strip()


def extract_document(filename: str, data: bytes) -> tuple[str, list[dict[str, Any]]]:
    """Extract full text and optional page metadata for a document."""
    suffix = Path(filename).suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return extract_pdf_text(data)
    if suffix in TEXT_EXTENSIONS:
        text = extract_text_bytes(data)
        return text, [{"page": 0, "text": text}]
    if suffix in JSON_EXTENSIONS:
        return extract_json_document(data)
    raise ValueError(f"Unsupported file type: {filename}")


def extract_json_document(data: bytes) -> tuple[str, list[dict[str, Any]]]:
    """Parse JSON/GeoJSON bytes into fulltext and per-feature segments."""
    raw = extract_text_bytes(data)
    try:
        payload = json.loads(raw) if raw else None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    records = geojson_to_records(payload)
    if not records:
        raise ValueError("JSON did not contain any indexable GeoJSON features or records")
    segments = [{"page": index, "text": record} for index, record in enumerate(records)]
    return "\n".join(records), segments


def chunk_text(
    fulltext: str,
    *,
    page_segments: list[dict[str, Any]] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """Split text into ~``chunk_size`` character chunks with page hints."""
    if not fulltext.strip():
        return []

    if page_segments:
        chunks: list[dict[str, Any]] = []
        current = ""
        current_page = int(page_segments[0].get("page", 0))
        for segment in page_segments:
            page = int(segment.get("page", 0))
            text = str(segment.get("text", ""))
            if not text:
                continue
            piece = text if not current else f"{current}\n{text}"
            if len(current) > 0 and len(piece) > chunk_size:
                chunks.append(
                    {
                        "index": len(chunks),
                        "text": current.strip(),
                        "page": current_page,
                    }
                )
                current = text
                current_page = page
            else:
                if not current:
                    current_page = page
                current = piece if current else text
                while len(current) > chunk_size:
                    chunks.append(
                        {
                            "index": len(chunks),
                            "text": current[:chunk_size].strip(),
                            "page": current_page,
                        }
                    )
                    current = current[chunk_size:]
        if current.strip():
            chunks.append(
                {
                    "index": len(chunks),
                    "text": current.strip(),
                    "page": current_page,
                }
            )
        return chunks

    chunks = []
    remaining = fulltext
    while remaining:
        piece = remaining[:chunk_size]
        remaining = remaining[chunk_size:]
        stripped = piece.strip()
        if stripped:
            chunks.append({"index": len(chunks), "text": stripped, "page": 0})
    return chunks


def process_document(
    filename: str,
    data: bytes,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[str, list[dict[str, Any]]]:
    """Extract and chunk a document, returning fulltext and chunk records."""
    fulltext, page_segments = extract_document(filename, data)
    chunks = chunk_text(fulltext, page_segments=page_segments, chunk_size=chunk_size)
    return fulltext, chunks
