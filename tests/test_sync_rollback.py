"""Tests for soft-rollback snapshot retention (R2)."""

from __future__ import annotations

import json
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

from rag.config import DEFAULT_SYNC_ROLLBACK_KEEP, resolve_sync_rollback_keep
from rag.publish.sync_constants import ROLLBACK_DIR_NAME, SYNC_DIR_NAME
from rag.publish.sync_rollback import (
    META_NAME,
    list_rollback_snapshots,
    prune_rollback_snapshots,
    rollback_root,
)


def _write_fake_snapshot(rag_root: Path, release_id: str, *, created_at: str) -> Path:
    """Create a minimal rollback dir with meta only (no Qdrant)."""
    dest = rollback_root(rag_root, release_id)
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        "release_id": release_id,
        "created_at": created_at,
        "doc_ids": [],
        "files_copied": [],
        "point_ids": [],
        "points_count": 0,
        "had_release_json": False,
    }
    (dest / META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (dest / "marker.txt").write_text(release_id, encoding="utf-8")
    return dest


def test_resolve_sync_rollback_keep_default(monkeypatch: MonkeyPatch) -> None:
    """Unset env → keep exactly one snapshot."""
    monkeypatch.delenv("RAG_SYNC_ROLLBACK_KEEP", raising=False)
    assert resolve_sync_rollback_keep() == DEFAULT_SYNC_ROLLBACK_KEEP == 1


def test_resolve_sync_rollback_keep_configurable(monkeypatch: MonkeyPatch) -> None:
    """RAG_SYNC_ROLLBACK_KEEP accepts 0 / N; invalid falls back to default."""
    monkeypatch.setenv("RAG_SYNC_ROLLBACK_KEEP", "3")
    assert resolve_sync_rollback_keep() == 3
    monkeypatch.setenv("RAG_SYNC_ROLLBACK_KEEP", "0")
    assert resolve_sync_rollback_keep() == 0
    monkeypatch.setenv("RAG_SYNC_ROLLBACK_KEEP", "nope")
    assert resolve_sync_rollback_keep() == 1
    monkeypatch.setenv("RAG_SYNC_ROLLBACK_KEEP", "-2")
    assert resolve_sync_rollback_keep() == 1
    monkeypatch.setenv("RAG_SYNC_ROLLBACK_KEEP", "999")
    assert resolve_sync_rollback_keep() == 100


def test_prune_keeps_only_newest_by_default(tmp_path: Path) -> None:
    """R2: after prune with keep=1 only the latest release snapshot remains."""
    rag_root = tmp_path / "rag"
    rag_root.mkdir()
    _write_fake_snapshot(rag_root, "rel-old", created_at="2026-11-18T20:00:00Z")
    _write_fake_snapshot(rag_root, "rel-mid", created_at="2026-11-18T21:00:00Z")
    newest = _write_fake_snapshot(
        rag_root, "rel-new", created_at="2026-11-18T22:00:00Z"
    )

    result = prune_rollback_snapshots(rag_root, keep=1)
    assert result["keep"] == 1
    assert result["kept"] == ["rel-new"]
    assert set(result["pruned"]) == {"rel-old", "rel-mid"}
    assert newest.is_dir()
    assert not (rag_root / SYNC_DIR_NAME / ROLLBACK_DIR_NAME / "rel-old").exists()
    assert not (rag_root / SYNC_DIR_NAME / ROLLBACK_DIR_NAME / "rel-mid").exists()
    remaining = list_rollback_snapshots(rag_root)
    assert len(remaining) == 1
    assert remaining[0]["release_id"] == "rel-new"


def test_prune_respects_keep_three(tmp_path: Path) -> None:
    """Configurable keep retains the newest N snapshots."""
    rag_root = tmp_path / "rag"
    rag_root.mkdir()
    for idx, stamp in enumerate(
        (
            "2026-11-18T10:00:00Z",
            "2026-11-18T11:00:00Z",
            "2026-11-18T12:00:00Z",
            "2026-11-18T13:00:00Z",
        ),
        start=1,
    ):
        _write_fake_snapshot(rag_root, f"rel-{idx}", created_at=stamp)

    result = prune_rollback_snapshots(rag_root, keep=3)
    assert result["kept"] == ["rel-4", "rel-3", "rel-2"]
    assert result["pruned"] == ["rel-1"]
    assert {s["release_id"] for s in list_rollback_snapshots(rag_root)} == {
        "rel-2",
        "rel-3",
        "rel-4",
    }


def test_prune_keep_zero_removes_all(tmp_path: Path) -> None:
    """keep=0 deletes every rollback snapshot."""
    rag_root = tmp_path / "rag"
    rag_root.mkdir()
    _write_fake_snapshot(rag_root, "rel-a", created_at="2026-11-18T10:00:00Z")
    _write_fake_snapshot(rag_root, "rel-b", created_at="2026-11-18T11:00:00Z")
    result = prune_rollback_snapshots(rag_root, keep=0)
    assert result["kept"] == []
    assert set(result["pruned"]) == {"rel-a", "rel-b"}
    assert list_rollback_snapshots(rag_root) == []


def test_prune_uses_env_default(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """When keep is omitted, env RAG_SYNC_ROLLBACK_KEEP drives retention."""
    monkeypatch.setenv("RAG_SYNC_ROLLBACK_KEEP", "1")
    rag_root = tmp_path / "rag"
    rag_root.mkdir()
    _write_fake_snapshot(rag_root, "older", created_at="2026-11-18T09:00:00Z")
    _write_fake_snapshot(rag_root, "newer", created_at="2026-11-18T10:00:00Z")
    result = prune_rollback_snapshots(rag_root)
    assert result["keep"] == 1
    assert result["kept"] == ["newer"]
    assert result["pruned"] == ["older"]
