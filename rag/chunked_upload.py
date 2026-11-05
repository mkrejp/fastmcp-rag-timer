"""Server-side staging and reassembly for chunked HTTP uploads."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rag.config import (
    resolve_upload_chunk_size_default,
    resolve_upload_max_bytes,
    resolve_upload_max_chunk_size,
    resolve_upload_ttl_seconds,
)
from rag.storage import utc_now_iso

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_UPLOAD_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def normalize_sha256(value: str) -> str:
    """Return lowercase hex SHA-256 or raise ``ValueError``."""
    lowered = value.strip().lower()
    if not _SHA256_RE.fullmatch(lowered):
        raise ValueError("sha256 must be 64 lowercase hex characters")
    return lowered


def chunk_count_for(size: int, chunk_size: int) -> int:
    """Return how many fixed-size chunks cover ``size`` bytes."""
    if size <= 0:
        raise ValueError("size must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return math.ceil(size / chunk_size)


def expected_chunk_bytes(*, size: int, chunk_size: int, index: int) -> int:
    """Return the expected byte length for chunk ``index``."""
    if index < 0:
        raise ValueError("index must be non-negative")
    start = index * chunk_size
    if start >= size:
        raise ValueError("chunk index out of range")
    return min(chunk_size, size - start)


@dataclass
class ChunkReceipt:
    """Recorded metadata for one uploaded chunk."""

    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkReceipt:
        """Deserialize from a dictionary."""
        return cls(bytes=int(data["bytes"]), sha256=str(data["sha256"]))


@dataclass
class UploadSessionMeta:
    """Persistent metadata for an in-progress chunked upload."""

    upload_id: str
    filename: str
    size: int
    sha256: str
    chunk_size: int
    chunk_count: int
    content_type: str | None
    expires_at: str
    created_at: str
    received: dict[str, ChunkReceipt] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize session metadata."""
        return {
            "upload_id": self.upload_id,
            "filename": self.filename,
            "size": self.size,
            "sha256": self.sha256,
            "chunk_size": self.chunk_size,
            "chunk_count": self.chunk_count,
            "content_type": self.content_type,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
            "received": {k: v.to_dict() for k, v in self.received.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UploadSessionMeta:
        """Deserialize session metadata."""
        raw_received = data.get("received") or {}
        received = {
            str(k): ChunkReceipt.from_dict(v)
            for k, v in raw_received.items()
            if isinstance(v, dict)
        }
        return cls(
            upload_id=str(data["upload_id"]),
            filename=str(data["filename"]),
            size=int(data["size"]),
            sha256=str(data["sha256"]),
            chunk_size=int(data["chunk_size"]),
            chunk_count=int(data["chunk_count"]),
            content_type=(
                str(data["content_type"]) if data.get("content_type") else None
            ),
            expires_at=str(data["expires_at"]),
            created_at=str(data["created_at"]),
            received=received,
        )


class ChunkedUploadError(Exception):
    """Base error for chunked upload operations."""

    def __init__(self, message: str, *, code: str) -> None:
        """Attach a stable machine-readable ``code``."""
        super().__init__(message)
        self.code = code


class UploadNotFoundError(ChunkedUploadError):
    """Raised when ``upload_id`` is unknown."""

    def __init__(self) -> None:
        super().__init__("Upload session not found", code="not_found")


class UploadExpiredError(ChunkedUploadError):
    """Raised when a session is past ``expires_at``."""

    def __init__(self) -> None:
        super().__init__("Upload session expired", code="expired")


class ChunkConflictError(ChunkedUploadError):
    """Raised when the same chunk index arrives with different bytes."""

    def __init__(self) -> None:
        super().__init__(
            "Chunk already stored with different content",
            code="chunk_conflict",
        )


class ChunkedUploadStore:
    """Manage chunked upload staging under ``{rag_root}/uploads/``."""

    def __init__(self, rag_root: str | Path) -> None:
        """Bind staging to the RAG data root."""
        self.rag_root = Path(rag_root)
        self.uploads_dir = self.rag_root / "uploads"

    def ensure_layout(self) -> None:
        """Create the uploads directory if missing."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, upload_id: str) -> Path:
        """Return the staging directory for ``upload_id``."""
        if not _UPLOAD_ID_RE.fullmatch(upload_id):
            raise ValueError("Invalid upload_id")
        return self.uploads_dir / upload_id

    def _meta_path(self, upload_id: str) -> Path:
        """Return the session metadata path."""
        return self._session_dir(upload_id) / "meta.json"

    def _chunks_dir(self, upload_id: str) -> Path:
        """Return the chunk parts directory."""
        return self._session_dir(upload_id) / "chunks"

    def _chunk_path(self, upload_id: str, index: int) -> Path:
        """Return on-disk path for chunk ``index``."""
        return self._chunks_dir(upload_id) / f"{index:08d}.part"

    def _assembled_path(self, upload_id: str, filename: str) -> Path:
        """Return the reassembled file path inside staging."""
        safe_name = Path(filename).name or "document.bin"
        return self._session_dir(upload_id) / safe_name

    def _write_meta(self, meta: UploadSessionMeta) -> None:
        """Atomically persist session metadata."""
        path = self._meta_path(meta.upload_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)

    def _load_meta(self, upload_id: str) -> UploadSessionMeta:
        """Load session metadata or raise ``UploadNotFoundError``."""
        path = self._meta_path(upload_id)
        if not path.is_file():
            raise UploadNotFoundError()
        data = json.loads(path.read_text(encoding="utf-8"))
        return UploadSessionMeta.from_dict(data)

    def _parse_expires(self, meta: UploadSessionMeta) -> datetime:
        """Parse ``expires_at`` as timezone-aware UTC."""
        parsed = datetime.fromisoformat(meta.expires_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _is_expired(self, meta: UploadSessionMeta) -> bool:
        """Return whether the session is past its expiry."""
        return datetime.now(timezone.utc) >= self._parse_expires(meta)

    def _require_active(self, meta: UploadSessionMeta) -> None:
        """Raise ``UploadExpiredError`` and delete staging when expired."""
        if self._is_expired(meta):
            self.delete_session(meta.upload_id)
            raise UploadExpiredError()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        """Compute SHA-256 of a file without loading it entirely at once."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def cleanup_expired(self) -> int:
        """Remove expired upload sessions; return how many were deleted."""
        if not self.uploads_dir.is_dir():
            return 0
        removed = 0
        for child in self.uploads_dir.iterdir():
            if not child.is_dir():
                continue
            upload_id = child.name
            try:
                meta = self._load_meta(upload_id)
            except UploadNotFoundError:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
                continue
            if self._is_expired(meta):
                self.delete_session(upload_id)
                removed += 1
        return removed

    def create_session(
        self,
        *,
        filename: str,
        size: int,
        sha256: str,
        content_type: str | None = None,
        chunk_size: int | None = None,
    ) -> UploadSessionMeta:
        """Initialize a new chunked upload session."""
        self.cleanup_expired()
        self.ensure_layout()

        safe_filename = Path(filename).name
        if not safe_filename:
            raise ValueError("filename is required")

        digest = normalize_sha256(sha256)
        if size <= 0:
            raise ValueError("size must be positive")

        max_bytes = resolve_upload_max_bytes()
        if size > max_bytes:
            raise ValueError(f"size exceeds maximum allowed ({max_bytes} bytes)")

        default_chunk = resolve_upload_chunk_size_default()
        effective_chunk = chunk_size if chunk_size is not None else default_chunk
        max_chunk = resolve_upload_max_chunk_size()
        if effective_chunk <= 0:
            raise ValueError("chunk_size must be positive")
        if effective_chunk > max_chunk:
            raise ValueError(f"chunk_size exceeds maximum allowed ({max_chunk} bytes)")

        ttl = resolve_upload_ttl_seconds()
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=ttl)
        upload_id = str(uuid.uuid4())
        meta = UploadSessionMeta(
            upload_id=upload_id,
            filename=safe_filename,
            size=size,
            sha256=digest,
            chunk_size=effective_chunk,
            chunk_count=chunk_count_for(size, effective_chunk),
            content_type=content_type,
            expires_at=expires.isoformat(),
            created_at=utc_now_iso(),
            received={},
        )
        self._session_dir(upload_id).mkdir(parents=True, exist_ok=True)
        self._chunks_dir(upload_id).mkdir(parents=True, exist_ok=True)
        self._write_meta(meta)
        return meta

    def get_session(self, upload_id: str) -> UploadSessionMeta:
        """Return session metadata after expiry checks."""
        meta = self._load_meta(upload_id)
        self._require_active(meta)
        return meta

    def delete_session(self, upload_id: str) -> bool:
        """Remove staging for ``upload_id``; return whether anything was removed."""
        path = self._session_dir(upload_id)
        if not path.exists():
            return False
        shutil.rmtree(path, ignore_errors=True)
        return True

    def store_chunk(
        self,
        upload_id: str,
        index: int,
        data: bytes,
        *,
        chunk_sha256: str | None = None,
    ) -> UploadSessionMeta:
        """Persist one chunk; idempotent when bytes match an existing part."""
        meta = self._load_meta(upload_id)
        self._require_active(meta)

        if index < 0 or index >= meta.chunk_count:
            raise ValueError("chunk index out of range")

        expected_len = expected_chunk_bytes(
            size=meta.size, chunk_size=meta.chunk_size, index=index
        )
        if len(data) != expected_len:
            raise ValueError(
                f"chunk {index} must be exactly {expected_len} bytes, got {len(data)}"
            )

        computed = hashlib.sha256(data).hexdigest()
        if chunk_sha256 is not None:
            header_digest = normalize_sha256(chunk_sha256)
            if header_digest != computed:
                raise ValueError("X-Chunk-SHA256 does not match request body")

        chunk_path = self._chunk_path(upload_id, index)
        key = str(index)
        if chunk_path.is_file():
            on_disk = self._file_sha256(chunk_path)
            if on_disk == computed:
                if meta.received.get(key) is None:
                    meta.received[key] = ChunkReceipt(bytes=len(data), sha256=computed)
                    self._write_meta(meta)
                return meta
            raise ChunkConflictError()

        tmp_path = chunk_path.with_suffix(".part.tmp")
        tmp_path.write_bytes(data)
        os.replace(tmp_path, chunk_path)

        meta.received[key] = ChunkReceipt(bytes=len(data), sha256=computed)
        self._write_meta(meta)
        return meta

    def assemble_to_file(self, upload_id: str) -> tuple[Path, UploadSessionMeta]:
        """Stream-assemble chunk parts into one file and verify the whole-file hash."""
        meta = self._load_meta(upload_id)
        self._require_active(meta)

        missing = [
            i
            for i in range(meta.chunk_count)
            if not self._chunk_path(upload_id, i).is_file()
        ]
        if missing:
            raise ValueError(f"missing chunk indexes: {missing}")

        out_path = self._assembled_path(upload_id, meta.filename)
        tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
        digest = hashlib.sha256()
        with tmp_out.open("wb") as out_handle:
            for index in range(meta.chunk_count):
                chunk_path = self._chunk_path(upload_id, index)
                with chunk_path.open("rb") as in_handle:
                    while True:
                        block = in_handle.read(1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                        out_handle.write(block)

        if digest.hexdigest() != meta.sha256:
            tmp_out.unlink(missing_ok=True)
            raise ValueError("assembled file sha256 does not match init sha256")

        os.replace(tmp_out, out_path)
        return out_path, meta

    def received_summary(self, meta: UploadSessionMeta) -> dict[str, Any]:
        """Shape progress fields for HTTP status responses."""
        indexes = sorted(int(k) for k in meta.received.keys())
        bytes_received = sum(r.bytes for r in meta.received.values())
        return {
            "upload_id": meta.upload_id,
            "filename": meta.filename,
            "size": meta.size,
            "chunk_size": meta.chunk_size,
            "chunk_count": meta.chunk_count,
            "received_indexes": indexes,
            "bytes_received": bytes_received,
            "expires_at": meta.expires_at,
            "complete": len(indexes) == meta.chunk_count
            and bytes_received == meta.size,
        }
