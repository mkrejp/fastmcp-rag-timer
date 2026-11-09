"""OCR fallback for page highlights (optional Tesseract)."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pytest

from rag.ocr import tesseract_is_available
from rag.pdf_pages import get_page_highlights
from rag.storage import RagStorage
from tests.fixtures.pdf_factory import build_image_rich_pdf

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not tesseract_is_available(), reason="tesseract not installed")
def test_highlights_ocr_fallback_on_image_only_page(rag_root: Path) -> None:
    """When the text layer is empty, highlights can fall back to OCR word boxes."""
    storage = RagStorage(rag_root)
    storage.ensure_layout()
    doc_id = str(uuid.uuid4())
    data = build_image_rich_pdf(ocr_phrase="UNIQUEOCRPHRASE12345")
    digest = hashlib.sha256(data).hexdigest()
    storage.save_original(doc_id, "ocr-page.pdf", data)
    storage.write_processed(
        doc_id=doc_id,
        filename="ocr-page.pdf",
        mime="application/pdf",
        sha256=digest,
        fulltext="",
        chunks=[],
        status="ready",
    )
    highlights = get_page_highlights(
        storage,
        doc_id,
        2,
        "UNIQUEOCRPHRASE",
        max_hits=5,
    )
    assert highlights["highlights"]
    assert highlights["highlights"][0]["method"] == "ocr"
