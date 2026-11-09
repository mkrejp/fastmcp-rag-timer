"""Extract embedded images from PDF bytes via PyMuPDF."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def _sha256(data: bytes) -> str:
    """Return hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


MIN_IMAGE_WIDTH = 64
MIN_IMAGE_HEIGHT = 64
MIN_IMAGE_BYTES = 4 * 1024


@dataclass(frozen=True)
class ExtractedPdfImage:
    """One deduplicated embedded image extracted from a PDF page."""

    sha256: str
    image_id: str
    page_number: int
    width: int
    height: int
    mime: str
    extension: str
    data: bytes


def _mime_for_extension(ext: str) -> str:
    """Map PyMuPDF image extension to a MIME type."""
    lowered = ext.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tif": "image/tiff",
        "tiff": "image/tiff",
        "jpx": "image/jp2",
        "jb2": "image/jb2",
    }.get(lowered, "application/octet-stream")


def extract_pdf_images(
    pdf_bytes: bytes,
    *,
    skip_page_xrefs: set[tuple[int, int]] | None = None,
) -> list[ExtractedPdfImage]:
    """Return embedded images from ``pdf_bytes``, skipping tiny duplicates.

    When ``skip_page_xrefs`` is set, placements ``(page_index, xref)`` indexed
    as ``pdf_figure`` are omitted from ``pdf_image`` extraction.
    """
    try:
        import pymupdf as fitz
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "PyMuPDF (pymupdf) is required for PDF image extraction"
        ) from exc

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    seen_sha: set[str] = set()
    extracted: list[ExtractedPdfImage] = []
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            for img in page.get_images(full=True):
                xref = int(img[0])
                if skip_page_xrefs and (page_index, xref) in skip_page_xrefs:
                    continue
                try:
                    base = doc.extract_image(xref)
                except Exception:
                    continue
                data = base.get("image")
                if not data:
                    continue
                width = int(base.get("width") or 0)
                height = int(base.get("height") or 0)
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    continue
                if len(data) < MIN_IMAGE_BYTES:
                    continue
                digest = _sha256(data)
                if digest in seen_sha:
                    continue
                seen_sha.add(digest)
                ext = str(base.get("ext") or "png").lower()
                image_id = digest[:16]
                extracted.append(
                    ExtractedPdfImage(
                        sha256=digest,
                        image_id=image_id,
                        page_number=page_index,
                        width=width,
                        height=height,
                        mime=_mime_for_extension(ext),
                        extension=ext if ext.startswith(".") else f".{ext}",
                        data=data,
                    )
                )
    finally:
        doc.close()
    return extracted
