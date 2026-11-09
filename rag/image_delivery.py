"""Resolve on-disk paths for extracted images and page renders."""

from __future__ import annotations

from pathlib import Path

from rag.page_ids import parse_page_image_id
from rag.pdf_pages import ensure_page_render
from rag.storage import RagStorage


def resolve_image_delivery(
    storage: RagStorage,
    *,
    doc_id: str,
    image_id: str,
) -> tuple[Path, str]:
    """Return ``(path, mime)`` for an extracted image or page render id."""
    storage.validate_doc_id(doc_id)
    parsed_page = parse_page_image_id(image_id)
    if parsed_page is not None:
        pid, page = parsed_page
        if pid != doc_id:
            raise ValueError("doc_id does not match page image id")
        path = ensure_page_render(storage, doc_id, page)
        return path, "image/jpeg"

    storage.validate_image_id(image_id)
    image_meta = storage.find_image(doc_id, image_id)
    if image_meta is None:
        raise FileNotFoundError("Image not found")
    path = storage.image_file_path(doc_id, image_meta)
    if not path.is_file():
        raise FileNotFoundError("Image file missing on disk")
    return path, image_meta.mime


def locate_global_image(
    storage: RagStorage,
    image_id: str,
) -> tuple[str, Path, str] | None:
    """Locate an image globally (extracted id or ``page:{doc_id}:{page}``)."""
    parsed = parse_page_image_id(image_id)
    if parsed is not None:
        doc_id, page = parsed
        try:
            path, mime = resolve_image_delivery(
                storage, doc_id=doc_id, image_id=image_id
            )
        except (FileNotFoundError, ValueError):
            return None
        return doc_id, path, mime

    located = storage.find_image_global(image_id)
    if located is None:
        return None
    doc_id, image_meta = located
    path = storage.image_file_path(doc_id, image_meta)
    if not path.is_file():
        return None
    return doc_id, path, image_meta.mime
