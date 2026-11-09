"""Identifiers and validation for lazily rendered PDF page images."""

from __future__ import annotations

import re
from typing import Final

from rag.storage import RagStorage

_PAGE_IMAGE_PREFIX: Final = "page:"
_PAGE_SEGMENT_RE = re.compile(r"^[1-9][0-9]{0,4}$")


def format_page_image_id(doc_id: str, page_one_based: int) -> str:
    """Return ``page:{doc_id}:{page}`` with a 1-based human page number."""
    RagStorage.validate_doc_id(doc_id)
    RagStorage.validate_page_number(page_one_based)
    return f"{_PAGE_IMAGE_PREFIX}{doc_id}:{page_one_based}"


def is_page_image_id(image_id: str) -> bool:
    """Return whether ``image_id`` uses the page render id scheme."""
    return image_id.startswith(_PAGE_IMAGE_PREFIX)


def parse_page_image_id(image_id: str) -> tuple[str, int] | None:
    """Parse ``page:{doc_id}:{page}`` or return ``None`` when not a page id."""
    if not is_page_image_id(image_id):
        return None
    body = image_id[len(_PAGE_IMAGE_PREFIX) :]
    parts = body.split(":")
    if len(parts) != 2:
        return None
    doc_id, page_raw = parts
    try:
        RagStorage.validate_doc_id(doc_id)
    except ValueError:
        return None
    if not _PAGE_SEGMENT_RE.fullmatch(page_raw):
        return None
    page = int(page_raw)
    try:
        RagStorage.validate_page_number(page)
    except ValueError:
        return None
    return doc_id, page


def page_image_id_from_payload_page(
    doc_id: str,
    page_number_zero_based: int | None,
) -> str | None:
    """Build a page image id from Qdrant's 0-based ``page_number`` field."""
    if page_number_zero_based is None:
        return None
    page_one = int(page_number_zero_based) + 1
    try:
        return format_page_image_id(doc_id, page_one)
    except ValueError:
        return None
