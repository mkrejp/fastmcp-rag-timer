"""Persistent filesystem storage under the RAG data root."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

_IMAGE_ID_RE = re.compile(r"^[a-f0-9]{16}$")
_DOC_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_IMAGE_FILE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".jpx",
        ".jb2",
    }
)
# Prefer PNG first: pdf_figure clips are stored as ``{image_id}.png``.
_IMAGE_FILE_LOOKUP_ORDER = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".jpx",
    ".jb2",
)


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def parse_iso_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; naive values are treated as UTC.

    Accepts a trailing ``Z``. Date-only values become midnight UTC.
    """
    text = value.strip()
    if not text:
        raise ValueError("timestamp is empty")
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


_IMAGE_META_KNOWN = frozenset(
    {
        "image_id",
        "sha256",
        "file_path",
        "page_number",
        "width",
        "height",
        "mime",
        "ocr_text",
        "ocr_confidence",
        "lat",
        "lon",
        "geo_refined",
        "geo_refined_history",
        "extra",
    }
)
_DOCUMENT_META_KNOWN = frozenset(
    {
        "doc_id",
        "filename",
        "mime",
        "sha256",
        "uploaded_at",
        "chunk_count",
        "status",
        "original_path",
        "fulltext_path",
        "chunks_path",
        "images",
        "figure_index_version",
        "figures_count",
        "figures_status",
        "lat",
        "lon",
        "geo_refined",
        "geo_refined_history",
        "extra",
    }
)


def _optional_float(value: object) -> float | None:
    """Return ``value`` as float, or ``None`` when unset."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float, str)):
        return float(value)
    return float(cast(Any, value))


def _optional_geo_object(value: object) -> dict[str, Any] | None:
    """Return a mapping for ``geo_refined``, or ``None``."""
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return dict(value)
    return None


def _geo_history(value: object) -> list[dict[str, Any]]:
    """Return a list of previous ``geo_refined`` objects."""
    if not value:
        return []
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _geo_constructor_kwargs(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return lat/lon/geo_refined kwargs shared by metadata dataclasses."""
    return {
        "lat": _optional_float(data.get("lat")),
        "lon": _optional_float(data.get("lon")),
        "geo_refined": _optional_geo_object(data.get("geo_refined")),
        "geo_refined_history": _geo_history(data.get("geo_refined_history")),
    }


def _omit_empty_geo(out: dict[str, Any]) -> dict[str, Any]:
    """Drop unset GPS fields from a serialized metadata payload."""
    if out.get("lat") is None:
        out.pop("lat", None)
    if out.get("lon") is None:
        out.pop("lon", None)
    if out.get("geo_refined") is None:
        out.pop("geo_refined", None)
    if not out.get("geo_refined_history"):
        out.pop("geo_refined_history", None)
    return out


def _unknown_extra(data: Mapping[str, Any], known: frozenset[str]) -> dict[str, Any]:
    """Preserve unrecognized metadata keys (including original GPS aliases)."""
    extra: dict[str, Any] = {}
    nested = data.get("extra")
    if isinstance(nested, dict):
        extra.update(nested)
    for key, value in data.items():
        if key in known or key == "images":
            continue
        extra[key] = value
    return extra


@dataclass
class ImageMeta:
    """Metadata for one extracted PDF image on disk."""

    image_id: str
    sha256: str
    file_path: str
    page_number: int
    width: int
    height: int
    mime: str
    ocr_text: str | None = None
    ocr_confidence: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    lat: float | None = None
    lon: float | None = None
    geo_refined: dict[str, Any] | None = None
    geo_refined_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize image metadata to a plain dictionary."""
        payload = asdict(self)
        extra = payload.pop("extra", {}) or {}
        return _omit_empty_geo({**extra, **payload})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageMeta:
        """Deserialize image metadata from a dictionary."""
        ocr_conf = data.get("ocr_confidence")
        return cls(
            image_id=str(data["image_id"]),
            sha256=str(data["sha256"]),
            file_path=str(data["file_path"]),
            page_number=int(data["page_number"]),
            width=int(data["width"]),
            height=int(data["height"]),
            mime=str(data["mime"]),
            ocr_text=(
                str(data["ocr_text"]) if data.get("ocr_text") is not None else None
            ),
            ocr_confidence=float(ocr_conf) if ocr_conf is not None else None,
            extra=_unknown_extra(data, _IMAGE_META_KNOWN),
            **_geo_constructor_kwargs(data),
        )


@dataclass
class DocumentMeta:
    """Metadata for a stored document and its processed artifacts."""

    doc_id: str
    filename: str
    mime: str
    sha256: str
    uploaded_at: str
    chunk_count: int
    status: str
    original_path: str
    fulltext_path: str
    chunks_path: str
    images: list[ImageMeta] = field(default_factory=list)
    figure_index_version: int = 0
    figures_count: int = 0
    figures_status: str = "pending"
    lat: float | None = None
    lon: float | None = None
    geo_refined: dict[str, Any] | None = None
    geo_refined_history: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a plain dictionary."""
        payload = asdict(self)
        extra = payload.pop("extra", {}) or {}
        payload.pop("images", None)
        out = {**extra, **payload}
        out["images"] = [img.to_dict() for img in self.images]
        return _omit_empty_geo(out)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentMeta:
        """Deserialize metadata from a dictionary."""
        raw_images = data.get("images") or []
        images = [
            ImageMeta.from_dict(item) for item in raw_images if isinstance(item, dict)
        ]
        return cls(
            doc_id=str(data["doc_id"]),
            filename=str(data["filename"]),
            mime=str(data["mime"]),
            sha256=str(data["sha256"]),
            uploaded_at=str(data["uploaded_at"]),
            chunk_count=int(data["chunk_count"]),
            status=str(data["status"]),
            original_path=str(data["original_path"]),
            fulltext_path=str(data["fulltext_path"]),
            chunks_path=str(data["chunks_path"]),
            images=images,
            figure_index_version=int(data.get("figure_index_version") or 0),
            figures_count=int(data.get("figures_count") or 0),
            figures_status=str(data.get("figures_status") or "pending"),
            extra=_unknown_extra(data, _DOCUMENT_META_KNOWN),
            **_geo_constructor_kwargs(data),
        )


class RagStorage:
    """Manage originals, processed artifacts, and the document catalog."""

    def __init__(self, root: str | Path) -> None:
        """Initialize storage rooted at ``root`` (typically ``/data/rag``)."""
        self.root = Path(root)
        self.originals_dir = self.root / "originals"
        self.processed_dir = self.root / "processed"
        self.qdrant_dir = self.root / "qdrant"
        self.versions_dir = self.root / "versions"
        self.index_path = self.root / "index.json"
        self.trails_outbox_path = self.root / "trails_outbox.json"

    def ensure_layout(self) -> None:
        """Create the persistent directory tree if it does not exist."""
        for path in (
            self.root,
            self.originals_dir,
            self.processed_dir,
            self.qdrant_dir,
            self.versions_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_index([])

    def original_dir(self, doc_id: str) -> Path:
        """Return the originals directory for ``doc_id``."""
        return self.originals_dir / doc_id

    def processed_doc_dir(self, doc_id: str) -> Path:
        """Return the processed artifacts directory for ``doc_id``."""
        return self.processed_dir / doc_id

    def versions_doc_dir(self, doc_id: str) -> Path:
        """Return the on-disk version history directory for ``doc_id``."""
        return self.versions_dir / doc_id

    def images_dir(self, doc_id: str) -> Path:
        """Return the extracted images directory for ``doc_id``."""
        return self.processed_doc_dir(doc_id) / "images"

    def pages_dir(self, doc_id: str) -> Path:
        """Return the lazily rendered full-page images directory for ``doc_id``."""
        return self.processed_doc_dir(doc_id) / "pages"

    def page_jpeg_path(self, doc_id: str, page_one_based: int) -> Path:
        """Return the cached JPEG path for a 1-based page number."""
        self.validate_doc_id(doc_id)
        self.validate_page_number(page_one_based)
        dest_dir = self.pages_dir(doc_id)
        dest = (dest_dir / f"{page_one_based}.jpg").resolve()
        if not str(dest).startswith(str(dest_dir.resolve())):
            raise ValueError("Invalid page destination")
        return dest

    @staticmethod
    def validate_doc_id(doc_id: str) -> str:
        """Reject path traversal in document identifiers."""
        if not _DOC_ID_RE.fullmatch(doc_id):
            raise ValueError("Invalid doc_id")
        return doc_id

    @staticmethod
    def validate_image_id(image_id: str) -> str:
        """Reject path traversal in image identifiers."""
        if not _IMAGE_ID_RE.fullmatch(image_id):
            raise ValueError("Invalid image_id")
        return image_id

    @staticmethod
    def validate_page_number(page_one_based: int) -> int:
        """Validate a 1-based PDF page number."""
        if page_one_based < 1 or page_one_based > 9999:
            raise ValueError("Invalid page number")
        return page_one_based

    @staticmethod
    def validate_image_route_id(image_id: str) -> str:
        """Accept extracted image ids or ``page:{doc_id}:{page}`` ids."""
        from rag.page_ids import parse_page_image_id

        if _IMAGE_ID_RE.fullmatch(image_id):
            return image_id
        if parse_page_image_id(image_id) is not None:
            return image_id
        raise ValueError("Invalid image_id")

    def relative_to_root(self, path: Path) -> str:
        """Return ``path`` relative to the RAG data root using forward slashes."""
        rel = path.resolve().relative_to(self.root.resolve())
        return rel.as_posix()

    def resolve_under_root(self, relative_path: str) -> Path:
        """Resolve a catalog-relative path and ensure it stays under ``root``."""
        candidate = (self.root / relative_path).resolve()
        root = self.root.resolve()
        if not str(candidate).startswith(str(root)):
            raise ValueError("Path escapes RAG data root")
        return candidate

    def save_original(self, doc_id: str, filename: str, data: bytes) -> Path:
        """Persist the raw uploaded file and return its path."""
        dest_dir = self.original_dir(doc_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name or "document.bin"
        dest = dest_dir / safe_name
        dest.write_bytes(data)
        return dest

    def write_processed(
        self,
        *,
        doc_id: str,
        filename: str,
        mime: str,
        sha256: str,
        fulltext: str,
        chunks: list[dict[str, Any]],
        images: list[ImageMeta] | None = None,
        status: str = "ready",
        uploaded_at: str | None = None,
    ) -> DocumentMeta:
        """Write processed text, chunks, and metadata for a document."""
        dest_dir = self.processed_doc_dir(doc_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        fulltext_path = dest_dir / "fulltext.txt"
        chunks_path = dest_dir / "chunks.json"
        meta_path = dest_dir / "meta.json"
        original_path = self.original_dir(doc_id) / (
            Path(filename).name or "document.bin"
        )

        fulltext_path.write_text(fulltext, encoding="utf-8")
        chunks_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        meta = DocumentMeta(
            doc_id=doc_id,
            filename=filename,
            mime=mime,
            sha256=sha256,
            uploaded_at=uploaded_at or utc_now_iso(),
            chunk_count=len(chunks),
            status=status,
            original_path=str(original_path),
            fulltext_path=str(fulltext_path),
            chunks_path=str(chunks_path),
            images=list(images or []),
            figures_status="pending",
        )
        meta_path.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.upsert_index_entry(meta)
        return meta

    def write_meta(self, meta: DocumentMeta) -> None:
        """Persist updated ``meta.json`` and refresh the catalog entry."""
        meta_path = self.processed_doc_dir(meta.doc_id) / "meta.json"
        meta_path.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.upsert_index_entry(meta)

    def read_meta(self, doc_id: str) -> DocumentMeta | None:
        """Load document metadata, or ``None`` if missing."""
        meta_path = self.processed_doc_dir(doc_id) / "meta.json"
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return DocumentMeta.from_dict(data)

    def read_fulltext(self, doc_id: str) -> str | None:
        """Load processed full text, or ``None`` if missing."""
        path = self.processed_doc_dir(doc_id) / "fulltext.txt"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def read_chunks(self, doc_id: str) -> list[dict[str, Any]] | None:
        """Load processed chunks, or ``None`` if missing."""
        path = self.processed_doc_dir(doc_id) / "chunks.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Invalid chunks.json for doc_id={doc_id}")
        return data

    def write_chunks(self, doc_id: str, chunks: list[dict[str, Any]]) -> None:
        """Overwrite ``chunks.json`` for ``doc_id`` (does not change Qdrant)."""
        dest_dir = self.processed_doc_dir(doc_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = dest_dir / "chunks.json"
        chunks_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def find_by_sha256(self, sha256: str) -> DocumentMeta | None:
        """Return an existing document with the same content hash, if any."""
        for entry in self.list_documents():
            if entry.get("sha256") == sha256:
                meta = self.read_meta(str(entry["doc_id"]))
                if meta is not None:
                    return meta
        return None

    def list_documents(self, since: str | None = None) -> list[dict[str, Any]]:
        """Return the document catalog from ``index.json``.

        ``since`` is an optional ISO-8601 timestamp. When set, only entries
        whose ``uploaded_at`` is greater than or equal to ``since`` are
        returned. Naive timestamps are treated as UTC.
        """
        if not self.index_path.exists():
            return []
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        cutoff_raw = str(since).strip() if since is not None else ""
        if not cutoff_raw:
            return data
        cutoff = parse_iso_timestamp(cutoff_raw)
        filtered: list[dict[str, Any]] = []
        for entry in data:
            uploaded = str(entry.get("uploaded_at") or "")
            if not uploaded:
                continue
            try:
                uploaded_at = parse_iso_timestamp(uploaded)
            except ValueError:
                continue
            if uploaded_at >= cutoff:
                filtered.append(entry)
        return filtered

    def upsert_index_entry(self, meta: DocumentMeta) -> None:
        """Insert or replace a catalog entry for ``meta``."""
        entries = [e for e in self.list_documents() if e.get("doc_id") != meta.doc_id]
        entries.append(
            {
                "doc_id": meta.doc_id,
                "filename": meta.filename,
                "mime": meta.mime,
                "sha256": meta.sha256,
                "uploaded_at": meta.uploaded_at,
                "chunk_count": meta.chunk_count,
                "status": meta.status,
                "figure_index_version": meta.figure_index_version,
                "figures_count": meta.figures_count,
                "figures_status": meta.figures_status,
            }
        )
        entries.sort(key=lambda e: str(e.get("uploaded_at", "")), reverse=True)
        self._write_index(entries)

    def _write_index(self, entries: list[dict[str, Any]]) -> None:
        """Atomically write the document catalog."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.index_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.index_path)

    def save_image_bytes(
        self,
        doc_id: str,
        *,
        image_id: str,
        extension: str,
        data: bytes,
    ) -> Path:
        """Persist extracted image bytes under ``processed/{doc_id}/images/``."""
        self.validate_doc_id(doc_id)
        self.validate_image_id(image_id)
        ext = extension if extension.startswith(".") else f".{extension}"
        safe_ext = Path(ext).suffix.lower()
        if safe_ext not in _IMAGE_FILE_EXTENSIONS:
            safe_ext = ".png"
        dest_dir = self.images_dir(doc_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = (dest_dir / f"{image_id}{safe_ext}").resolve()
        if not str(dest).startswith(str(dest_dir.resolve())):
            raise ValueError("Invalid image destination")
        dest.write_bytes(data)
        return dest

    def locate_image_file(self, doc_id: str, image_id: str) -> Path | None:
        """Return an on-disk image for ``image_id`` under the document images dir.

        Looks up ``processed/{doc_id}/images/{image_id}{ext}`` with the same
        extensions ``save_image_bytes`` allows. This finds ``pdf_figure`` PNGs
        that are stored on disk but never recorded in ``meta.images``.
        """
        self.validate_doc_id(doc_id)
        self.validate_image_id(image_id)
        dest_dir = self.images_dir(doc_id)
        if not dest_dir.is_dir():
            return None
        dest_dir_resolved = dest_dir.resolve()
        for ext in _IMAGE_FILE_LOOKUP_ORDER:
            dest = (dest_dir / f"{image_id}{ext}").resolve()
            if not str(dest).startswith(str(dest_dir_resolved)):
                continue
            if dest.is_file():
                return dest
        return None

    def find_image(self, doc_id: str, image_id: str) -> ImageMeta | None:
        """Return image metadata for ``doc_id`` + ``image_id``, if present."""
        meta = self.read_meta(doc_id)
        if meta is None:
            return None
        for img in meta.images:
            if img.image_id == image_id:
                return img
        return None

    def find_image_global(self, image_id: str) -> tuple[str, ImageMeta] | None:
        """Locate ``image_id`` across the catalog (first match)."""
        self.validate_image_id(image_id)
        for entry in self.list_documents():
            doc_id = str(entry.get("doc_id", ""))
            found = self.find_image(doc_id, image_id)
            if found is not None:
                return doc_id, found
        return None

    def document_is_pdf(self, doc_id: str) -> bool:
        """Return whether ``doc_id`` refers to an indexed PDF."""
        meta = self.read_meta(doc_id)
        return meta is not None and meta.filename.lower().endswith(".pdf")

    def image_file_path(self, doc_id: str, image_meta: ImageMeta) -> Path:
        """Resolve the on-disk path for ``image_meta`` with traversal checks."""
        return self.resolve_under_root(image_meta.file_path)

    def remove_document_files(self, doc_id: str) -> None:
        """Delete original and processed files for ``doc_id`` and drop the catalog row.

        Processed cleanup includes the extracted ``images/`` directory.
        """
        for path in (
            self.original_dir(doc_id),
            self.processed_doc_dir(doc_id),
            self.versions_doc_dir(doc_id),
        ):
            if path.exists():
                shutil.rmtree(path)
        entries = [e for e in self.list_documents() if e.get("doc_id") != doc_id]
        self._write_index(entries)
