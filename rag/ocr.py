"""Local OCR for extracted PDF images (Tesseract via pytesseract)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rag.config import (
    resolve_ocr_enabled,
    resolve_ocr_langs,
    resolve_ocr_min_chars,
    resolve_ocr_min_confidence,
)
from rag.logging_setup import get_logger

if TYPE_CHECKING:
    from PIL import Image as PilImage

logger = get_logger("rag.ocr")

_tesseract_checked = False
_tesseract_available = False


def tesseract_is_available() -> bool:
    """Return whether the Tesseract binary is reachable (cached after first check)."""
    global _tesseract_checked, _tesseract_available  # noqa: PLW0603
    if _tesseract_checked:
        return _tesseract_available
    _tesseract_checked = True
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        _tesseract_available = True
    except Exception:
        logger.warning(
            "Tesseract OCR is not available; image files will be stored without OCR points"
        )
        _tesseract_available = False
    return _tesseract_available


def _preprocess_image(image: PilImage.Image) -> PilImage.Image:
    """Apply light preprocessing before OCR (grayscale, upscale small images)."""
    from PIL import Image

    gray = image.convert("L")
    max_dim = max(gray.size)
    if max_dim < 300:
        scale = max(2, 300 // max(1, max_dim))
        new_size = (gray.size[0] * scale, gray.size[1] * scale)
        gray = gray.resize(new_size, Image.Resampling.LANCZOS)
    return gray


def ocr_image_file(path: Path) -> tuple[str, float | None]:
    """Run OCR on ``path`` and return cleaned text plus mean word confidence."""
    if not resolve_ocr_enabled():
        return "", None
    if not tesseract_is_available():
        return "", None

    import pytesseract
    from PIL import Image

    langs = resolve_ocr_langs()
    min_conf = resolve_ocr_min_confidence()
    min_chars = resolve_ocr_min_chars()

    try:
        with Image.open(path) as img:
            prepared = _preprocess_image(img)
            data = pytesseract.image_to_data(
                prepared,
                lang=langs,
                config="--psm 6",
                output_type=pytesseract.Output.DICT,
            )
    except Exception as exc:  # noqa: BLE001 - OCR failures should not abort ingest
        logger.warning("OCR failed for %s: %s", path, exc)
        return "", None

    words: list[str] = []
    confidences: list[int] = []
    for word, conf_raw in zip(data.get("text", []), data.get("conf", []), strict=False):
        stripped = str(word or "").strip()
        if not stripped:
            continue
        try:
            conf = int(float(conf_raw))
        except (TypeError, ValueError):
            continue
        if conf < min_conf:
            continue
        words.append(stripped)
        confidences.append(conf)

    text = " ".join(words).strip()
    if len(text) < min_chars:
        return "", float(sum(confidences) / len(confidences)) if confidences else None

    avg_conf = float(sum(confidences) / len(confidences)) if confidences else None
    return text, avg_conf


def build_content_summary(page_number: int, ocr_text: str) -> str:
    """Prefix OCR text for embedding and search snippets."""
    page_label = page_number + 1
    return f"Obrázek, strana {page_label}: {ocr_text}"
