"""Persistent filesystem storage under the RAG data root."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentMeta:
        """Deserialize metadata from a dictionary."""
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
        )


class RagStorage:
    """Manage originals, processed artifacts, and the document catalog."""

    def __init__(self, root: str | Path) -> None:
        """Initialize storage rooted at ``root`` (typically ``/data/rag``)."""
        self.root = Path(root)
        self.originals_dir = self.root / "originals"
        self.processed_dir = self.root / "processed"
        self.qdrant_dir = self.root / "qdrant"
        self.index_path = self.root / "index.json"

    def ensure_layout(self) -> None:
        """Create the persistent directory tree if it does not exist."""
        for path in (
            self.root,
            self.originals_dir,
            self.processed_dir,
            self.qdrant_dir,
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
        )
        meta_path.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.upsert_index_entry(meta)
        return meta

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

    def find_by_sha256(self, sha256: str) -> DocumentMeta | None:
        """Return an existing document with the same content hash, if any."""
        for entry in self.list_documents():
            if entry.get("sha256") == sha256:
                meta = self.read_meta(str(entry["doc_id"]))
                if meta is not None:
                    return meta
        return None

    def list_documents(self) -> list[dict[str, Any]]:
        """Return the document catalog from ``index.json``."""
        if not self.index_path.exists():
            return []
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return data

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

    def remove_document_files(self, doc_id: str) -> None:
        """Delete original and processed files for ``doc_id`` and drop the catalog row."""
        for path in (self.original_dir(doc_id), self.processed_doc_dir(doc_id)):
            if path.exists():
                shutil.rmtree(path)
        entries = [e for e in self.list_documents() if e.get("doc_id") != doc_id]
        self._write_index(entries)
