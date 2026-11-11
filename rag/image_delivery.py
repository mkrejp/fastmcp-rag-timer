"""Resolve on-disk paths for extracted images, figure clips, and page renders."""

from __future__ import annotations

from pathlib import Path

from rag.page_ids import parse_page_image_id
from rag.pdf_pages import ensure_page_render
from rag.storage import RagStorage

UNKNOWN_IMAGE_ID_MESSAGE = "Unknown image id"
MISSING_IMAGE_FILE_MESSAGE = "Image file missing on disk"

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".jpx": "image/jp2",
    ".jb2": "image/jb2",
}


def mime_for_image_path(path: Path) -> str:
    """Return a MIME type from ``path``'s suffix, defaulting to PNG."""
    return _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")


def resolve_image_delivery(
    storage: RagStorage,
    *,
    doc_id: str,
    image_id: str,
) -> tuple[Path, str]:
    """Return ``(path, mime)`` for an extracted image, figure clip, or page render.

    ``pdf_figure`` PNGs live under ``processed/{doc_id}/images/{image_id}.png``
    but are not listed in ``meta.images``. After id validation, fall back to
    that on-disk lookup when catalog metadata has no match.
    """
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
    if image_meta is not None:
        path = storage.image_file_path(doc_id, image_meta)
        if path.is_file():
            return path, image_meta.mime
        fallback = storage.locate_image_file(doc_id, image_id)
        if fallback is not None:
            return fallback, mime_for_image_path(fallback)
        raise FileNotFoundError(MISSING_IMAGE_FILE_MESSAGE)

    located = storage.locate_image_file(doc_id, image_id)
    if located is None:
        raise FileNotFoundError(UNKNOWN_IMAGE_ID_MESSAGE)
    return located, mime_for_image_path(located)


def locate_global_image(
    storage: RagStorage,
    image_id: str,
) -> tuple[str, Path, str] | None:
    """Locate an image globally (extracted id, figure clip, or page render).

    Returns ``None`` when the id is unknown. Raises ``FileNotFoundError`` with
    ``MISSING_IMAGE_FILE_MESSAGE`` when a catalog entry exists but the file is
    gone and no on-disk fallback is found.
    """
    parsed = parse_page_image_id(image_id)
    if parsed is not None:
        doc_id, _page = parsed
        try:
            path, mime = resolve_image_delivery(
                storage, doc_id=doc_id, image_id=image_id
            )
        except (FileNotFoundError, ValueError):
            return None
        return doc_id, path, mime

    storage.validate_image_id(image_id)
    saw_missing_file = False
    for entry in storage.list_documents():
        doc_id = str(entry.get("doc_id", ""))
        if not doc_id:
            continue
        try:
            storage.validate_doc_id(doc_id)
        except ValueError:
            continue
        try:
            path, mime = resolve_image_delivery(
                storage, doc_id=doc_id, image_id=image_id
            )
        except FileNotFoundError as exc:
            if str(exc) == MISSING_IMAGE_FILE_MESSAGE:
                saw_missing_file = True
            continue
        except ValueError:
            continue
        return doc_id, path, mime
    if saw_missing_file:
        raise FileNotFoundError(MISSING_IMAGE_FILE_MESSAGE)
    return None
