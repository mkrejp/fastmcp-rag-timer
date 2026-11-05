"""Unit tests for OCR filtering logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image

from rag.ocr import build_content_summary, ocr_image_file


def test_ocr_filters_low_confidence_words(tmp_path: Path) -> None:
    """Words below ``OCR_MIN_CONFIDENCE`` should be dropped from the cleaned text."""
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (120, 40), "white").save(img_path)
    fake_data: dict[str, list[Any]] = {
        "text": ["acceptable", "bad", "confidence"],
        "conf": ["90", "10", "85"],
    }
    with patch("pytesseract.image_to_data", return_value=fake_data):
        with patch("rag.ocr.tesseract_is_available", return_value=True):
            with patch("rag.ocr.resolve_ocr_enabled", return_value=True):
                text, conf = ocr_image_file(img_path)
    assert "acceptable" in text and "confidence" in text
    assert "bad" not in text.split()
    assert conf is not None


def test_ocr_short_text_not_indexed(tmp_path: Path) -> None:
    """Text shorter than ``OCR_MIN_CHARS`` should return empty for indexing."""
    img_path = tmp_path / "short.png"
    Image.new("RGB", (120, 40), "white").save(img_path)
    fake_data: dict[str, list[Any]] = {
        "text": ["tiny"],
        "conf": ["99"],
    }
    with patch("pytesseract.image_to_data", return_value=fake_data):
        with patch("rag.ocr.tesseract_is_available", return_value=True):
            with patch("rag.ocr.resolve_ocr_enabled", return_value=True):
                text, _ = ocr_image_file(img_path)
    assert text == ""


def test_build_content_summary_prefix() -> None:
    """Content summaries should include a human-readable page prefix."""
    summary = build_content_summary(0, "sample text")
    assert summary.startswith("Obrázek, strana 1:")
