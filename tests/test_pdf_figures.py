"""Tests for pdf_figure detection, summary, and ingest integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rag.config import resolve_figure_min_area_ratio
from rag.payloads import build_figure_content_summary
from rag.pdf_figures import (
    collect_raw_placements,
    compute_verification,
    iterate_figure_records,
    merge_placements,
)
from rag.processing import process_document
from tests.fixtures.pdf_figure_factory import (
    build_duplicate_placement_pdf,
    build_figure_caption_pdf,
    build_ocr_label_pdf,
    build_small_icon_pdf,
    build_tiled_figure_pdf,
)

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def test_figure_content_summary_fallback_when_empty() -> None:
    """Caption-less figures still get a non-empty page-label text body."""
    text = build_figure_content_summary(
        page_number=2,
        caption=None,
        summary="",
        ocr_text=None,
    )
    assert text
    assert "3" in text


def test_small_icon_yields_no_raw_placements() -> None:
    """Icons below the area threshold must not become figure candidates."""
    pdf = build_small_icon_pdf()
    raw = collect_raw_placements(pdf, min_area_ratio=resolve_figure_min_area_ratio())
    assert raw == []


def test_large_figure_with_caption_and_refs() -> None:
    """50% figure gets caption-related refs and can reach auto_checked."""
    pdf = build_figure_caption_pdf()
    _full, chunks = process_document("book.pdf", pdf)
    figures = list(
        iterate_figure_records(
            pdf,
            doc_id="doc-1",
            parent_document="book.pdf",
            doc_sha256="abc" * 21,
            chunks=chunks,
        )
    )
    assert len(figures) == 1
    payload = figures[0].payload
    assert payload["source_type"] == "pdf_figure"
    assert payload.get("caption")
    assert "Kostel" in (payload.get("caption") or "")
    refs = payload.get("related_text_refs") or []
    kinds = {r.get("kind") for r in refs}
    assert "caption" in kinds
    assert payload["verification_status"] in ("auto_checked", "unverified")
    assert payload["area_ratio"] >= resolve_figure_min_area_ratio()


def test_duplicate_placements_two_records() -> None:
    """Same image on two pages produces two figure records."""
    pdf = build_duplicate_placement_pdf()
    _full, chunks = process_document("dup.pdf", pdf)
    figures = list(
        iterate_figure_records(
            pdf,
            doc_id="doc-dup",
            parent_document="dup.pdf",
            doc_sha256="def" * 21,
            chunks=chunks,
        )
    )
    assert len(figures) == 2
    assert figures[0].payload["image_id"] == figures[1].payload["image_id"]


def test_tiled_merge_single_figure() -> None:
    """Four tiles merge into one figure with tiled=true."""
    pdf = build_tiled_figure_pdf()
    raw = collect_raw_placements(pdf, min_area_ratio=None)
    merged = merge_placements(raw, page_count=1)
    assert len(merged) == 1
    assert merged[0].tiled is True
    _full, chunks = process_document("tile.pdf", pdf)
    figures = list(
        iterate_figure_records(
            pdf,
            doc_id="doc-tile",
            parent_document="tile.pdf",
            doc_sha256="fed" * 21,
            chunks=chunks,
        )
    )
    assert len(figures) == 1
    assert figures[0].payload.get("tiled") is True
    assert figures[0].payload["area_ratio"] >= resolve_figure_min_area_ratio()


def test_summary_sentences_grounded() -> None:
    """Extractive summary sentences must reference valid related_text_refs."""
    refs = [
        {"page": 0, "chunk_index": 0, "kind": "caption", "score": 0.9},
        {"page": 1, "chunk_index": 1, "kind": "explicit_ref", "score": 0.85},
    ]
    sentences = [
        {"text": "Obr. 3 Kostel", "ref": 0},
        {"text": "jak ukazuje obr. 3", "ref": 1},
    ]
    spans = ["Obr. 3 Kostel sv. Jakuba", "jak ukazuje obr. 3, byl kostel"]
    checks, conf, status, _verified = compute_verification(
        area_ratio=0.5,
        caption="Obr. 3 Kostel sv. Jakuba",
        explicit_ref_count=1,
        ocr_text="",
        ocr_confidence=None,
        ocr_source="none",
        summary_sentences=sentences,
        related_text_refs=refs,
        span_texts=spans,
        decorative=False,
        min_ratio=resolve_figure_min_area_ratio(),
    )
    assert checks["summary_grounded"] is True
    assert status in ("auto_checked", "unverified")
    assert conf > 0.5


@pytest.mark.integration
def test_ocr_on_rendered_figure(monkeypatch: MonkeyPatch) -> None:
    """Tesseract OCR on a rendered figure clip (skipped when unavailable)."""
    from rag.ocr import tesseract_is_available

    if not tesseract_is_available():
        pytest.skip("Tesseract not installed")
    pdf = build_ocr_label_pdf()
    _full, chunks = process_document("ocr.pdf", pdf)
    figures = list(
        iterate_figure_records(
            pdf,
            doc_id="doc-ocr",
            parent_document="ocr.pdf",
            doc_sha256="aaa" * 21,
            chunks=chunks,
        )
    )
    assert len(figures) == 1
    payload = figures[0].payload
    raw = (payload.get("ocr_text_raw") or "").upper()
    assert "HOSTINEC" in raw or "LIPY" in raw
