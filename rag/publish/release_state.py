"""Per-node release ACK stored under the RAG data root."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.logging_setup import get_logger
from rag.storage import utc_now_iso

RELEASE_STATE_NAME = "release.json"

logger = get_logger("rag.publish.release_state")


def release_state_path(rag_root: Path | str) -> Path:
    """Return the path of ``release.json`` under ``rag_root``."""
    return Path(rag_root) / RELEASE_STATE_NAME


def read_release_state(rag_root: Path | str) -> dict[str, Any] | None:
    """Load ``release.json`` when present; return ``None`` if missing/invalid."""
    path = release_state_path(rag_root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def write_release_state(
    rag_root: Path | str,
    *,
    release_id: str,
    parent_release_id: str | None = None,
    source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write ``release.json`` after a successful apply or publish."""
    root = Path(rag_root)
    root.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "release_id": release_id,
        "parent_release_id": parent_release_id,
        "applied_at": utc_now_iso(),
        "source": source,
    }
    if extra:
        body.update(extra)
    path = release_state_path(root)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return body
