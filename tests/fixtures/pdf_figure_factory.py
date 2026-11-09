"""PyMuPDF-generated PDF fixtures for pdf_figure indexing tests."""

from __future__ import annotations

import io

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            36,
        )
    except OSError:
        return ImageFont.load_default()


def build_figure_caption_pdf() -> bytes:
    """One large figure (≈50% page) with caption and a cross-page explicit ref."""
    doc = fitz.open()
    page_fig = doc.new_page()
    rect = page_fig.rect
    w, h = rect.width, rect.height
    img = Image.new("RGB", (int(w), int(h * 0.52)), (200, 210, 220))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    y1 = 72 + h * 0.52
    page_fig.insert_image(fitz.Rect(72, 72, w - 72, y1), stream=buf.getvalue())
    page_fig.insert_text(
        (72, y1 + 28),
        "Obr. 3 Kostel sv. Jakuba před přestavbou",
        fontsize=11,
    )

    page_ref = doc.new_page()
    page_ref.insert_text(
        (72, 72),
        "Na jiné stránce text jak ukazuje obr. 3, byl kostel zachován.",
        fontsize=11,
    )
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()


def build_small_icon_pdf() -> bytes:
    """Tiny embedded icon only (no pdf_figure)."""
    doc = fitz.open()
    page = doc.new_page()
    tiny = Image.new("RGB", (24, 24), "red")
    buf = io.BytesIO()
    tiny.save(buf, format="PNG")
    page.insert_image(fitz.Rect(72, 72, 96, 96), stream=buf.getvalue())
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()


def build_duplicate_placement_pdf() -> bytes:
    """Same large image on two pages (both ≥25% area)."""
    doc = fitz.open()
    w, h = 595.0, 842.0
    img = Image.new("RGB", (int(w - 144), int(h * 0.5)), (120, 140, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    stream = buf.getvalue()
    for _ in range(2):
        page = doc.new_page()
        page.insert_image(
            fitz.Rect(72, 72, w - 72, 72 + h * 0.5),
            stream=stream,
        )
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()


def build_tiled_figure_pdf() -> bytes:
    """Four tiles that merge into one figure (~40% page)."""
    doc = fitz.open()
    page = doc.new_page()
    w, h = 595.0, 842.0
    tile_w = (w - 144) / 2
    tile_h = h * 0.2
    tile = Image.new("RGB", (int(tile_w), int(tile_h)), (90, 120, 90))
    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    stream = buf.getvalue()
    positions = [
        (72, 72),
        (72 + tile_w, 72),
        (72, 72 + tile_h),
        (72 + tile_w, 72 + tile_h),
    ]
    for x0, y0 in positions:
        page.insert_image(
            fitz.Rect(x0, y0, x0 + tile_w, y0 + tile_h),
            stream=stream,
        )
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()


def build_ocr_label_pdf(*, label: str = "HOSTINEC U LIPY") -> bytes:
    """Large figure with painted label text for OCR tests."""
    doc = fitz.open()
    page = doc.new_page()
    w, h = 595.0, 842.0
    img = Image.new("RGB", (int(w - 144), int(h * 0.45)), (240, 235, 220))
    draw = ImageDraw.Draw(img)
    draw.text((40, 80), label, fill="black", font=_font())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(
        fitz.Rect(72, 72, w - 72, 72 + h * 0.45),
        stream=buf.getvalue(),
    )
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()
