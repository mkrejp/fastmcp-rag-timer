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

try:
    import pytesseract as pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]

_tesseract_checked = False
_tesseract_available = False


def tesseract_is_available() -> bool:
    """Return whether the Tesseract binary is reachable (cached after first check)."""
    global _tesseract_checked, _tesseract_available  # noqa: PLW0603
    if _tesseract_checked:
        return _tesseract_available
    _tesseract_checked = True
    try:
        if pytesseract is None:
            raise RuntimeError("pytesseract not installed")
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

    if pytesseract is None:
        return "", None
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


def _ocr_data_from_prepared(
    prepared: PilImage.Image,
    *,
    psm: int,
) -> tuple[list[str], list[int], list[str]]:
    """Return word list, confidences, and raw words from Tesseract data output."""
    if pytesseract is None:
        return [], [], []
    langs = resolve_ocr_langs()
    data = pytesseract.image_to_data(
        prepared,
        lang=langs,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
    )
    words: list[str] = []
    confidences: list[int] = []
    raw_words: list[str] = []
    for word, conf_raw in zip(data.get("text", []), data.get("conf", []), strict=False):
        stripped = str(word or "").strip()
        if not stripped:
            continue
        raw_words.append(stripped)
        try:
            conf = int(float(conf_raw))
        except (TypeError, ValueError):
            conf = -1
        words.append(stripped)
        if conf >= 0:
            confidences.append(conf)
    return words, confidences, raw_words


def ocr_pil_image(
    image: PilImage.Image,
    *,
    accept_min_confidence: float | None = None,
) -> tuple[str, str, float | None, int, str]:
    """OCR a PIL image; return filtered text, raw text, confidence, count, source.

    Uses ``--psm 11`` first, then ``--psm 6``. ``accept_min_confidence`` overrides
    the word filter threshold for the filtered ``ocr_text`` only.
    """
    if not resolve_ocr_enabled() or not tesseract_is_available():
        return "", "", None, 0, "none"

    from PIL import Image

    work = image
    max_dim = max(work.size)
    if max_dim > 4000:
        scale = 4000 / max_dim
        new_size = (int(work.size[0] * scale), int(work.size[1] * scale))
        work = work.resize(new_size, Image.Resampling.LANCZOS)

    prepared = _preprocess_image(work)
    min_conf = resolve_ocr_min_confidence()
    accept_conf = (
        float(accept_min_confidence)
        if accept_min_confidence is not None
        else float(min_conf)
    )

    best: tuple[str, str, float | None, int] = ("", "", None, 0)
    for psm in (11, 6):
        _words, confidences, raw_words = _ocr_data_from_prepared(prepared, psm=psm)
        if not raw_words:
            continue
        filtered = [
            word
            for word, conf in zip(_words, confidences, strict=False)
            if conf >= min_conf
        ]
        raw_text = " ".join(raw_words).strip()
        text = " ".join(filtered).strip()
        avg_conf = float(sum(confidences) / len(confidences)) if confidences else None
        if len(text) >= len(best[0]):
            best = (text, raw_text, avg_conf, len(raw_words))
        if text:
            break

    text, raw_text, avg_conf, word_count = best

    if text and avg_conf is not None and avg_conf < accept_conf:
        text = ""
    if text:
        alpha_words = [w for w in text.split() if sum(c.isalpha() for c in w) >= 3]
        if len(alpha_words) < max(1, len(text.split()) * 0.5):
            text = ""

    return text, raw_text, avg_conf, word_count, "tesseract"
