"""Build PDF fixtures with embedded images for ingest tests."""

from __future__ import annotations

import io

import pymupdf as fitz
from PIL import Image, ImageDraw


def build_image_rich_pdf(*, ocr_phrase: str = "UNIQUEOCRPHRASE12345") -> bytes:
    """Return PDF bytes: body text, OCR phrase in an image, photo-like block, tiny icon."""
    doc = fitz.open()
    page_text = doc.new_page()
    page_text.insert_text((72, 72), "Body text without the hidden phrase.")

    phrase_img = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(phrase_img)
    draw.text((24, 110), ocr_phrase, fill="black")
    for x in range(0, 900, 40):
        draw.line([(x, 0), (x, 260)], fill=(230, 230, 230), width=1)
    phrase_buf = io.BytesIO()
    phrase_img.save(phrase_buf, format="PNG")
    page_image = doc.new_page()
    page_image.insert_image(
        fitz.Rect(72, 72, 972, 332),
        stream=phrase_buf.getvalue(),
    )

    photo = Image.new("RGB", (520, 420), color=(40, 90, 160))
    photo_draw = ImageDraw.Draw(photo)
    for y in range(0, 420, 8):
        photo_draw.line([(0, y), (520, y)], fill=(30 + y % 40, 70, 140))
    photo_buf = io.BytesIO()
    photo.save(photo_buf, format="JPEG")
    page_photo = doc.new_page()
    page_photo.insert_image(
        fitz.Rect(72, 72, 592, 492),
        stream=photo_buf.getvalue(),
    )

    tiny = Image.new("RGB", (24, 24), "red")
    tiny_buf = io.BytesIO()
    tiny.save(tiny_buf, format="PNG")
    page_icon = doc.new_page()
    page_icon.insert_image(
        fitz.Rect(72, 72, 96, 96),
        stream=tiny_buf.getvalue(),
    )

    duplicate_page = doc.new_page()
    duplicate_page.insert_image(
        fitz.Rect(72, 72, 972, 332),
        stream=phrase_buf.getvalue(),
    )

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()
