"""Unit tests for figure-clip image delivery fallback."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from rag.image_delivery import (
    MISSING_IMAGE_FILE_MESSAGE,
    UNKNOWN_IMAGE_ID_MESSAGE,
    locate_global_image,
    resolve_image_delivery,
)
from rag.storage import ImageMeta, RagStorage

DOC_ID = "0b5019ec-dab5-4307-8161-37a6ef474d04"
FIGURE_IMAGE_ID = "9981e4e524e1b54c"
MISSING_IMAGE_ID = "aaaaaaaaaaaaaaaa"


def _png_bytes(*, color: str = "navy") -> bytes:
    """Return a tiny PNG payload for disk-seeding tests."""
    image = Image.new("RGB", (12, 12), color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _seed_doc(storage: RagStorage, *, images: list[ImageMeta] | None = None) -> None:
    """Write a catalog document with optional ``meta.images`` rows."""
    storage.save_original(DOC_ID, "wiener-secession-1898.pdf", b"%PDF-1.4")
    storage.write_processed(
        doc_id=DOC_ID,
        filename="wiener-secession-1898.pdf",
        mime="application/pdf",
        sha256="abc",
        fulltext="secession",
        chunks=[{"index": 0, "text": "secession", "page": 0}],
        images=list(images or []),
    )


def test_resolve_serves_figure_png_absent_from_meta(rag_storage: RagStorage) -> None:
    """Figure clips on disk must resolve even when ``meta.images`` is empty."""
    png = _png_bytes()
    _seed_doc(rag_storage)
    dest = rag_storage.save_image_bytes(
        DOC_ID, image_id=FIGURE_IMAGE_ID, extension=".png", data=png
    )
    assert rag_storage.find_image(DOC_ID, FIGURE_IMAGE_ID) is None
    path, mime = resolve_image_delivery(
        rag_storage, doc_id=DOC_ID, image_id=FIGURE_IMAGE_ID
    )
    assert path == dest
    assert mime == "image/png"
    assert path.read_bytes() == png


def test_resolve_unknown_id_message(rag_storage: RagStorage) -> None:
    """Missing catalog row and missing file should report an unknown id."""
    _seed_doc(rag_storage)
    with pytest.raises(FileNotFoundError, match=UNKNOWN_IMAGE_ID_MESSAGE):
        resolve_image_delivery(rag_storage, doc_id=DOC_ID, image_id=MISSING_IMAGE_ID)


def test_resolve_missing_on_disk_message(rag_storage: RagStorage) -> None:
    """A catalogued image whose file is gone should report a missing file."""
    _seed_doc(
        rag_storage,
        images=[
            ImageMeta(
                image_id=MISSING_IMAGE_ID,
                sha256="deadbeef",
                file_path=f"processed/{DOC_ID}/images/{MISSING_IMAGE_ID}.png",
                page_number=0,
                width=1,
                height=1,
                mime="image/png",
            )
        ],
    )
    with pytest.raises(FileNotFoundError, match=MISSING_IMAGE_FILE_MESSAGE):
        resolve_image_delivery(rag_storage, doc_id=DOC_ID, image_id=MISSING_IMAGE_ID)


def test_locate_global_image_finds_figure_png(rag_storage: RagStorage) -> None:
    """MCP ``get_image`` lookup should find figure clips without ``meta.images``."""
    png = _png_bytes(color="teal")
    _seed_doc(rag_storage)
    rag_storage.save_image_bytes(
        DOC_ID, image_id=FIGURE_IMAGE_ID, extension=".png", data=png
    )
    located = locate_global_image(rag_storage, FIGURE_IMAGE_ID)
    assert located is not None
    doc_id, path, mime = located
    assert doc_id == DOC_ID
    assert mime == "image/png"
    assert path.read_bytes() == png


def test_locate_image_file_rejects_invalid_id(rag_storage: RagStorage) -> None:
    """On-disk lookup must keep the same image id validation as HTTP routes."""
    with pytest.raises(ValueError, match="Invalid image_id"):
        rag_storage.locate_image_file(DOC_ID, "../secret")
    with pytest.raises(ValueError, match="Invalid doc_id"):
        rag_storage.locate_image_file("not-a-uuid", FIGURE_IMAGE_ID)
    assert rag_storage.locate_image_file(DOC_ID, FIGURE_IMAGE_ID) is None


def test_locate_image_file_finds_png(rag_storage: RagStorage) -> None:
    """``locate_image_file`` should resolve a PNG written by ``save_image_bytes``."""
    _seed_doc(rag_storage)
    png = _png_bytes()
    dest = rag_storage.save_image_bytes(
        DOC_ID, image_id=FIGURE_IMAGE_ID, extension=".png", data=png
    )
    found = rag_storage.locate_image_file(DOC_ID, FIGURE_IMAGE_ID)
    assert found == dest
    assert found is not None
    assert found.is_file()
