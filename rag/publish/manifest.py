"""Release manifest schema for incremental writer→reader publishes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def new_release_id() -> str:
    """Return a unique release id (UTC stamp + short uuid)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


@dataclass
class FileChange:
    """One file add/update in a release."""

    path: str
    sha256: str
    size: int
    doc_id: str | None = None


@dataclass
class ReleaseManifest:
    """Diff-only release descriptor (not a full volume dump)."""

    release_id: str
    parent_release_id: str | None
    created_at: str
    writer_host: str | None = None
    files_added: list[FileChange] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    catalog_doc_ids_upserted: list[str] = field(default_factory=list)
    catalog_doc_ids_deleted: list[str] = field(default_factory=list)
    qdrant_point_ids_upserted: list[str] = field(default_factory=list)
    qdrant_point_ids_deleted: list[str] = field(default_factory=list)
    full_baseline: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReleaseManifest:
        """Deserialize from JSON."""
        added = [
            FileChange(**item) if isinstance(item, dict) else item
            for item in data.get("files_added") or []
        ]
        return cls(
            release_id=str(data["release_id"]),
            parent_release_id=data.get("parent_release_id"),
            created_at=str(data.get("created_at") or ""),
            writer_host=data.get("writer_host"),
            files_added=list(added),
            files_deleted=[str(p) for p in data.get("files_deleted") or []],
            catalog_doc_ids_upserted=[
                str(i) for i in data.get("catalog_doc_ids_upserted") or []
            ],
            catalog_doc_ids_deleted=[
                str(i) for i in data.get("catalog_doc_ids_deleted") or []
            ],
            qdrant_point_ids_upserted=[
                str(i) for i in data.get("qdrant_point_ids_upserted") or []
            ],
            qdrant_point_ids_deleted=[
                str(i) for i in data.get("qdrant_point_ids_deleted") or []
            ],
            full_baseline=bool(data.get("full_baseline")),
        )

    def write_json(self, path: Path) -> None:
        """Write this manifest to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def load_manifest(path: Path) -> ReleaseManifest:
    """Load a manifest JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    return ReleaseManifest.from_dict(payload)
