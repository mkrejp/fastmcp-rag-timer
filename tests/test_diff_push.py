"""Tests for ingester-side live diff push helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from _pytest.monkeypatch import MonkeyPatch

from rag.publish.diff_push import maybe_push_after_ingest, sync_push_enabled
from rag.storage import RagStorage


def test_sync_push_disabled_without_env(monkeypatch: MonkeyPatch) -> None:
    """Push is off when URL or token is missing."""
    monkeypatch.delenv("RAG_SYNC_URL", raising=False)
    monkeypatch.delenv("RAG_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("RAG_SYNC_PUSH_ENABLED", raising=False)
    monkeypatch.delenv("RAG_SYNC_PUSH_ON_INGEST", raising=False)
    assert sync_push_enabled() is False


def test_sync_push_default_off_even_with_url_token(monkeypatch: MonkeyPatch) -> None:
    """URL + token alone do not enable post-ingest push (opt-in required)."""
    monkeypatch.setenv("RAG_SYNC_URL", "https://reader.example.com")
    monkeypatch.setenv("RAG_SYNC_TOKEN", "secret")
    monkeypatch.delenv("RAG_SYNC_PUSH_ENABLED", raising=False)
    monkeypatch.delenv("RAG_SYNC_PUSH_ON_INGEST", raising=False)
    assert sync_push_enabled() is False
    monkeypatch.setenv("RAG_SYNC_PUSH_ENABLED", "1")
    assert sync_push_enabled() is True
    monkeypatch.setenv("RAG_SYNC_PUSH_ENABLED", "0")
    monkeypatch.setenv("RAG_SYNC_PUSH_ON_INGEST", "1")
    assert sync_push_enabled() is True


def test_nightly_offline_skips_push(monkeypatch: MonkeyPatch) -> None:
    """RAG_SYNC_NIGHTLY_OFFLINE=1 disables push even when opted in."""
    monkeypatch.setenv("RAG_SYNC_URL", "https://reader.example.com")
    monkeypatch.setenv("RAG_SYNC_TOKEN", "secret")
    monkeypatch.setenv("RAG_SYNC_PUSH_ENABLED", "1")
    monkeypatch.setenv("RAG_SYNC_NIGHTLY_OFFLINE", "1")
    assert sync_push_enabled() is False
    monkeypatch.delenv("RAG_SYNC_NIGHTLY_OFFLINE", raising=False)
    assert sync_push_enabled() is True


def test_maybe_push_after_ingest_noop(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Post-ingest hook returns None when sync is not configured / not opted in."""
    monkeypatch.delenv("RAG_SYNC_URL", raising=False)
    monkeypatch.delenv("RAG_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("RAG_SYNC_PUSH_ENABLED", raising=False)
    storage = RagStorage(str(tmp_path / "rag"))
    storage.ensure_layout()
    assert maybe_push_after_ingest(storage, MagicMock()) is None
