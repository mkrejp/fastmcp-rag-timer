"""Lazy full-page PDF renders and query highlight boxes."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import pymupdf as fitz

from rag.config import (
    resolve_page_render_dpi,
    resolve_page_render_max_pixels,
    resolve_page_render_quality,
    resolve_ocr_langs,
    resolve_ocr_min_confidence,
)
from rag.logging_setup import get_logger
from rag.ocr import tesseract_is_available
from rag.storage import RagStorage

logger = get_logger("rag.pdf_pages")

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]


def strip_diacritics(text: str) -> str:
    """Fold Czech diacritics for case-insensitive text-layer matching."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _find_original_pdf(storage: RagStorage, doc_id: str) -> Path:
    """Locate the stored original PDF for ``doc_id``."""
    meta = storage.read_meta(doc_id)
    if meta is None:
        raise FileNotFoundError(f"Document not found: {doc_id}")
    if not meta.filename.lower().endswith(".pdf"):
        raise ValueError("Document is not a PDF")
    original = storage.resolve_under_root(meta.original_path)
    if not original.is_file():
        raise FileNotFoundError("Original PDF missing on disk")
    return original


def _page_count(pdf_path: Path) -> int:
    """Return page count without keeping the document open."""
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def ensure_page_render(
    storage: RagStorage,
    doc_id: str,
    page_one_based: int,
) -> Path:
    """Render a PDF page to JPEG on disk (cached) and return its path."""
    storage.validate_doc_id(doc_id)
    storage.validate_page_number(page_one_based)
    dest = storage.page_jpeg_path(doc_id, page_one_based)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    pdf_path = _find_original_pdf(storage, doc_id)
    page_index = page_one_based - 1
    count = _page_count(pdf_path)
    if page_index < 0 or page_index >= count:
        raise FileNotFoundError(f"Page {page_one_based} out of range (1..{count})")

    dpi = resolve_page_render_dpi()
    max_pixels = resolve_page_render_max_pixels()
    quality = resolve_page_render_quality()

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".jpg.tmp")

    with fitz.open(pdf_path) as doc:
        page = doc[page_index]
        rect = page.rect
        long_edge = max(rect.width, rect.height)
        scale = min(1.0, (max_pixels * 72.0) / (long_edge * dpi)) if long_edge > 0 else 1.0
        effective_dpi = max(36, int(dpi * scale))
        pix = page.get_pixmap(dpi=effective_dpi, alpha=False)
        pix.save(str(tmp), output="jpeg", jpg_quality=quality)

    tmp.replace(dest)
    return dest


def _normalize_rect(
    rect: fitz.Rect,
    page_width: float,
    page_height: float,
) -> dict[str, float]:
    """Convert a PDF rect to normalized 0–1 coordinates."""
    if page_width <= 0 or page_height <= 0:
        return {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}
    return {
        "x0": max(0.0, min(1.0, rect.x0 / page_width)),
        "y0": max(0.0, min(1.0, rect.y0 / page_height)),
        "x1": max(0.0, min(1.0, rect.x1 / page_width)),
        "y1": max(0.0, min(1.0, rect.y1 / page_height)),
    }


def _dedupe_highlights(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop overlapping duplicate boxes (same text and nearly same coords)."""
    seen: set[tuple[str, int, int, int, int]] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = (
            str(item.get("text", "")),
            int(round(float(item.get("x0", 0)) * 1000)),
            int(round(float(item.get("y0", 0)) * 1000)),
            int(round(float(item.get("x1", 0)) * 1000)),
            int(round(float(item.get("y1", 0)) * 1000)),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _search_text_layer(
    page: fitz.Page,
    query: str,
    *,
    page_width: float,
    page_height: float,
    max_hits: int,
    method: str,
) -> list[dict[str, Any]]:
    """Run ``page.search_for`` and return normalized highlight dicts."""
    highlights: list[dict[str, Any]] = []
    for rect in page.search_for(query):
        if len(highlights) >= max_hits:
            break
        box = _normalize_rect(rect, page_width, page_height)
        highlights.append({**box, "text": query, "method": method})
    return highlights


def _search_text_layer_words(
    page: fitz.Page,
    query: str,
    *,
    page_width: float,
    page_height: float,
    max_hits: int,
) -> list[dict[str, Any]]:
    """Match query tokens against ``get_text('words')`` with diacritic folding."""
    q_norm = strip_diacritics(query).casefold().strip()
    if not q_norm:
        return []
    highlights: list[dict[str, Any]] = []
    for word in page.get_text("words"):
        if len(highlights) >= max_hits:
            break
        if len(word) < 5:
            continue
        x0, y0, x1, y1, token, *_rest = word
        token_str = str(token or "")
        t_norm = strip_diacritics(token_str).casefold()
        if not t_norm:
            continue
        if q_norm not in t_norm and t_norm not in q_norm:
            continue
        rect = fitz.Rect(x0, y0, x1, y1)
        box = _normalize_rect(rect, page_width, page_height)
        highlights.append(
            {**box, "text": token_str, "method": "text_layer_word_match"}
        )
    return highlights


def _ocr_words_cache_path(storage: RagStorage, doc_id: str, page_one_based: int) -> Path:
    """Path for cached OCR word boxes on a rendered page."""
    return storage.pages_dir(doc_id) / f"{page_one_based}_ocr_words.json"


def _load_or_run_ocr_words(
    storage: RagStorage,
    doc_id: str,
    page_one_based: int,
) -> list[dict[str, Any]]:
    """Load cached OCR words or run Tesseract on the page JPEG."""
    cache_path = _ocr_words_cache_path(storage, doc_id, page_one_based)
    if cache_path.is_file():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data

    if not tesseract_is_available() or pytesseract is None:
        return []

    jpeg_path = ensure_page_render(storage, doc_id, page_one_based)
    from PIL import Image

    langs = resolve_ocr_langs()
    min_conf = resolve_ocr_min_confidence()
    words: list[dict[str, Any]] = []
    with Image.open(jpeg_path) as img:
        width, height = img.size
        data = pytesseract.image_to_data(
            img,
            lang=langs,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
        for text, conf_raw, left, top, w, h in zip(
            data.get("text", []),
            data.get("conf", []),
            data.get("left", []),
            data.get("top", []),
            data.get("width", []),
            data.get("height", []),
            strict=False,
        ):
            token = str(text or "").strip()
            if not token:
                continue
            try:
                conf = int(float(conf_raw))
            except (TypeError, ValueError):
                continue
            if conf < min_conf:
                continue
            words.append(
                {
                    "text": token,
                    "x0": float(left) / width if width else 0.0,
                    "y0": float(top) / height if height else 0.0,
                    "x1": float(left + w) / width if width else 0.0,
                    "y1": float(top + h) / height if height else 0.0,
                }
            )

    cache_path.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    return words


def _match_ocr_words(
    words: list[dict[str, Any]],
    query: str,
    *,
    max_hits: int,
) -> list[dict[str, Any]]:
    """Find query tokens in cached OCR word boxes."""
    q_norm = strip_diacritics(query).casefold()
    if not q_norm.strip():
        return []
    hits: list[dict[str, Any]] = []
    for word in words:
        token = str(word.get("text", ""))
        t_norm = strip_diacritics(token).casefold()
        if not t_norm:
            continue
        if q_norm in t_norm or t_norm in q_norm:
            hits.append(
                {
                    "x0": word["x0"],
                    "y0": word["y0"],
                    "x1": word["x1"],
                    "y1": word["y1"],
                    "text": token,
                    "method": "ocr",
                }
            )
        if len(hits) >= max_hits:
            break
    if hits:
        return hits

    for part in query.split():
        if len(part) < 3:
            continue
        p_norm = strip_diacritics(part).casefold()
        for word in words:
            token = str(word.get("text", ""))
            t_norm = strip_diacritics(token).casefold()
            if p_norm and p_norm in t_norm:
                hits.append(
                    {
                        "x0": word["x0"],
                        "y0": word["y0"],
                        "x1": word["x1"],
                        "y1": word["y1"],
                        "text": token,
                        "method": "ocr",
                    }
                )
            if len(hits) >= max_hits:
                return hits
    return hits


def get_page_highlights(
    storage: RagStorage,
    doc_id: str,
    page_one_based: int,
    query: str,
    *,
    max_hits: int = 20,
) -> dict[str, Any]:
    """Return normalized highlight boxes for ``query`` on one PDF page."""
    storage.validate_doc_id(doc_id)
    storage.validate_page_number(page_one_based)
    if max_hits < 1:
        max_hits = 1
    if max_hits > 50:
        max_hits = 50

    pdf_path = _find_original_pdf(storage, doc_id)
    page_index = page_one_based - 1

    highlights: list[dict[str, Any]] = []
    page_width = 1.0
    page_height = 1.0

    with fitz.open(pdf_path) as doc:
        if page_index < 0 or page_index >= doc.page_count:
            raise FileNotFoundError(f"Page {page_one_based} out of range")
        page = doc[page_index]
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        trimmed = query.strip()
        if trimmed:
            highlights.extend(
                _search_text_layer(
                    page,
                    trimmed,
                    page_width=page_width,
                    page_height=page_height,
                    max_hits=max_hits,
                    method="text_layer",
                )
            )
            if len(highlights) < max_hits:
                folded = strip_diacritics(trimmed)
                if folded != trimmed:
                    highlights.extend(
                        _search_text_layer(
                            page,
                            folded,
                            page_width=page_width,
                            page_height=page_height,
                            max_hits=max_hits - len(highlights),
                            method="text_layer_normalized",
                        )
                    )
            if len(highlights) < max_hits:
                for part in trimmed.split():
                    if len(part) < 3:
                        continue
                    highlights.extend(
                        _search_text_layer(
                            page,
                            part,
                            page_width=page_width,
                            page_height=page_height,
                            max_hits=max_hits - len(highlights),
                            method="text_layer_word",
                        )
                    )
                    if len(highlights) >= max_hits:
                        break
                for part in trimmed.split():
                    if len(part) < 3:
                        continue
                    folded_part = strip_diacritics(part)
                    if folded_part == part:
                        continue
                    highlights.extend(
                        _search_text_layer(
                            page,
                            folded_part,
                            page_width=page_width,
                            page_height=page_height,
                            max_hits=max_hits - len(highlights),
                            method="text_layer_word_normalized",
                        )
                    )
                    if len(highlights) >= max_hits:
                        break
            if len(highlights) < max_hits and trimmed:
                highlights.extend(
                    _search_text_layer_words(
                        page,
                        trimmed,
                        page_width=page_width,
                        page_height=page_height,
                        max_hits=max_hits - len(highlights),
                    )
                )

    highlights = _dedupe_highlights(highlights)[:max_hits]

    if not highlights and query.strip():
        ocr_words = _load_or_run_ocr_words(storage, doc_id, page_one_based)
        highlights = _match_ocr_words(ocr_words, query, max_hits=max_hits)[:max_hits]

    return {
        "page_width": page_width,
        "page_height": page_height,
        "highlights": highlights,
    }
