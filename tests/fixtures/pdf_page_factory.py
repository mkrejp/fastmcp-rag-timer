"""PDF fixtures with a searchable text layer for page render tests."""

from __future__ import annotations

import io

import pymupdf as fitz


def build_text_layer_pdf(
    *,
    phrase: str = "Pražský městský timer",
    page_count: int = 2,
) -> bytes:
    """Return PDF bytes with ``phrase`` on page 1 and unrelated text on page 2."""
    doc = fitz.open()
    for index in range(page_count):
        page = doc.new_page()
        body = phrase if index == 0 else "Unrelated content on page two."
        writer = fitz.TextWriter(page.rect)
        writer.append((72, 80), body, fontsize=14)
        writer.write_text(page)
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()
