"""Tests for page image id parsing and validation."""

from __future__ import annotations

import pytest

from rag.page_ids import (
    format_page_image_id,
    is_page_image_id,
    page_image_id_from_payload_page,
    parse_page_image_id,
)

DOC = "11111111-1111-1111-1111-111111111111"


def test_format_and_parse_roundtrip() -> None:
    """Page ids use 1-based page numbers in the third segment."""
    image_id = format_page_image_id(DOC, 3)
    assert image_id == f"page:{DOC}:3"
    assert is_page_image_id(image_id)
    parsed = parse_page_image_id(image_id)
    assert parsed == (DOC, 3)


def test_parse_rejects_traversal() -> None:
    """Invalid doc ids and pages are rejected."""
    assert parse_page_image_id("page:not-a-uuid:1") is None
    assert parse_page_image_id(f"page:{DOC}:0") is None
    assert parse_page_image_id("abcd1234abcd1234") is None


def test_page_image_id_from_zero_based_payload() -> None:
    """Qdrant ``page_number`` 0 maps to human page 1 in the id."""
    assert page_image_id_from_payload_page(DOC, 0) == f"page:{DOC}:1"
    assert page_image_id_from_payload_page(DOC, None) is None


def test_validate_page_number_bounds() -> None:
    """Formatting enforces positive page numbers."""
    with pytest.raises(ValueError):
        format_page_image_id(DOC, 0)
