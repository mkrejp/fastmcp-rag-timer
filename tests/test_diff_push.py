"""Tests for ingester-side live diff push helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from _pytest.monkeypatch import MonkeyPatch

from rag.publish.diff import (
    catalog_fingerprint,
    document_content_fingerprint,
)
from rag.publish.diff_push import (
    DiffPushClient,
    build_and_push,
    maybe_push_after_ingest,
    sync_push_enabled,
)
from rag.publish.manifest import FileChange, ReleaseManifest
from rag.storage import RagStorage, utc_now_iso


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


def test_build_and_push_empty_diff_on_catalog_parity(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """V2: catalog parity must not begin a release (cron safety net)."""
    monkeypatch.delenv("RAG_SYNC_NIGHTLY_OFFLINE", raising=False)
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_id = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
    storage.save_original(doc_id, "e.txt", b"echo")
    storage.write_processed(
        doc_id=doc_id,
        filename="e.txt",
        mime="text/plain",
        sha256="eee",
        fulltext="echo",
        chunks=[{"index": 0, "page": 0, "text": "echo"}],
        images=[],
        status="ready",
    )
    writer_fp = catalog_fingerprint(storage)

    client = MagicMock()
    client.get_watermark.return_value = {"release_id": "rel-parent"}
    client.fetch_catalog.return_value = (writer_fp, writer_fp, {})
    monkeypatch.setattr(
        "rag.publish.diff_push.locality_links_catalog_fingerprint",
        lambda _q: {},
    )

    result = build_and_push(storage, qdrant=MagicMock(), client=client)
    assert result["pushed"] is False
    assert result["reason"] == "empty_diff"
    assert result["files_added"] == 0
    assert result["bytes_added"] == 0
    client.begin.assert_not_called()
    client.put_chunk.assert_not_called()
    client.commit.assert_not_called()


def test_dry_run_http_only_uses_get(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Dry-run must not invoke POST/PUT/PATCH/DELETE on the reader sync API."""
    monkeypatch.delenv("RAG_SYNC_NIGHTLY_OFFLINE", raising=False)
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    storage.save_original(doc_id, "a.txt", b"a")
    storage.write_processed(
        doc_id=doc_id,
        filename="a.txt",
        mime="text/plain",
        sha256="aaa",
        fulltext="a",
        chunks=[{"index": 0, "page": 0, "text": "a"}],
        images=[],
        status="ready",
    )
    meta = storage.read_meta(doc_id)
    assert meta is not None
    content_fp = document_content_fingerprint(meta)
    methods: list[str] = []

    def _track_request_json(method: str, url: str, **kwargs: object) -> object:
        methods.append(method.upper())
        if "watermark" in url:
            return {"release_id": "rel-parent"}
        if "catalog" in url:
            return {
                "documents": {doc_id: "reader-stale"},
                "documents_content": {doc_id: content_fp},
                "locality_links": {},
            }
        raise AssertionError(f"unexpected URL in dry-run: {url}")

    monkeypatch.setattr("rag.publish.diff_push._request_json", _track_request_json)
    monkeypatch.setattr(
        "rag.publish.diff_push.locality_links_catalog_fingerprint",
        lambda _q: {},
    )
    client = DiffPushClient("https://reader.example.com", token="t")
    result = build_and_push(storage, qdrant=MagicMock(), client=client, dry_run=True)
    assert result["dry_run"] is True
    assert methods
    assert set(methods) == {"GET"}


def test_build_and_push_dry_run_is_read_only(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """CLI dry-run must only call reader GETs (no begin/chunk/commit)."""
    monkeypatch.delenv("RAG_SYNC_NIGHTLY_OFFLINE", raising=False)
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
    storage.save_original(doc_id, "f.txt", b"foo")
    storage.write_processed(
        doc_id=doc_id,
        filename="f.txt",
        mime="text/plain",
        sha256="fff",
        fulltext="foo",
        chunks=[{"index": 0, "page": 0, "text": "foo"}],
        images=[],
        status="ready",
    )
    meta = storage.read_meta(doc_id)
    assert meta is not None
    content_fp = document_content_fingerprint(meta)
    reader_fp = {doc_id: "stale-fingerprint"}

    client = MagicMock()
    client.get_watermark.return_value = {"release_id": "rel-parent"}
    client.fetch_catalog.return_value = (reader_fp, {doc_id: content_fp}, {})
    monkeypatch.setattr(
        "rag.publish.diff_push.locality_links_catalog_fingerprint",
        lambda _q: {},
    )
    export_mock = MagicMock()
    monkeypatch.setattr(
        "rag.publish.diff_push.export_points_for_docs",
        export_mock,
    )

    result = build_and_push(storage, qdrant=MagicMock(), client=client, dry_run=True)
    assert result["dry_run"] is True
    assert result["pushed"] is False
    assert result["reason"] == "dry_run_preview"
    assert doc_id in result["sample_doc_ids_upserted"]
    assert result["docs_upserted_metadata_only"] == 1
    assert result["docs_upserted_full"] == 0
    client.get_watermark.assert_called_once()
    client.fetch_catalog.assert_called_once()
    client.begin.assert_not_called()
    client.put_chunk.assert_not_called()
    client.commit.assert_not_called()
    client.abort.assert_not_called()
    export_mock.assert_not_called()


def test_build_and_push_ignores_index_json_only_manifest(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """V2 belt: index.json-only file lists still count as empty_diff."""
    monkeypatch.delenv("RAG_SYNC_NIGHTLY_OFFLINE", raising=False)
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    (root / "index.json").write_text("[]", encoding="utf-8")

    index_only = ReleaseManifest(
        release_id="rel-index-only",
        parent_release_id="rel-parent",
        created_at=utc_now_iso(),
        files_added=[
            FileChange(path="index.json", sha256="abc", size=3962, doc_id=None)
        ],
    )

    client = MagicMock()
    client.get_watermark.return_value = {"release_id": "rel-parent"}
    client.fetch_catalog.return_value = ({"doc": "fp"}, {"doc": "fp"}, {})
    monkeypatch.setattr(
        "rag.publish.diff_push.locality_links_catalog_fingerprint",
        lambda _q: {},
    )

    monkeypatch.setattr(
        "rag.publish.diff_push.build_file_diff",
        lambda *args, **kwargs: index_only,
    )

    result = build_and_push(storage, qdrant=MagicMock(), client=client)
    assert result["pushed"] is False
    assert result["reason"] == "empty_diff"
    client.begin.assert_not_called()
