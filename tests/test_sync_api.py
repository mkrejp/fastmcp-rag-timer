"""Tests for reader live diff-push sync API (Option A)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import pytest
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from rag.config import COLLECTION_NAME
from rag.embeddings import EMBEDDING_DIM
from rag.image_embed import IMAGE_EMBEDDING_DIM
from rag.ingest import stable_point_id
from rag.publish.diff import build_file_diff, catalog_fingerprint
from rag.publish.manifest import ReleaseManifest
from rag.publish.qdrant_sync import (
    export_points_for_docs,
    open_qdrant_for_sync,
    scroll_points_for_doc,
)
from rag.publish.release_state import read_release_state, write_release_state
from rag.publish.sync_chunk import pack_blob_chunks, pack_points_chunks
from rag.storage import RagStorage
from rag.vectorstore import IMAGE_VECTOR_NAME, TEXT_VECTOR_NAME
from tests.conftest import patched_embeddings

SYNC_TOKEN = "test-sync-token-option-a"
SYNC_HEADERS = {"Authorization": f"Bearer {SYNC_TOKEN}"}


def _upload_release_chunks(
    client: TestClient,
    *,
    writer_root: Path,
    manifest: ReleaseManifest,
    records: Sequence[Mapping[str, Any]],
    expect_first_blob_unique: bool = True,
) -> None:
    """PUT blob + points chunks for ``manifest`` (shared by apply/dry-run tests)."""
    blob_files = [
        (change.path, (writer_root / change.path).read_bytes(), change.sha256)
        for change in manifest.files_added
    ]
    for index, chunk in enumerate(pack_blob_chunks(manifest.release_id, blob_files)):
        put = client.put(
            f"/sync/v1/push/chunk?release_id={manifest.release_id}",
            headers=SYNC_HEADERS,
            content=chunk.encode(),
        )
        assert put.status_code == 200, put.text
        if expect_first_blob_unique and index == 0:
            assert put.json()["duplicate"] is False
    for chunk in pack_points_chunks(manifest.release_id, records):
        put = client.put(
            f"/sync/v1/push/chunk?release_id={manifest.release_id}",
            headers=SYNC_HEADERS,
            content=chunk.encode(),
        )
        assert put.status_code == 200, put.text


def _upsert_chunk(
    client: QdrantClient,
    *,
    doc_id: str,
    text: str,
    index: int = 0,
) -> str:
    """Insert one knowledge point and return its id."""
    point_id = stable_point_id(doc_id, index)
    base = float(len(text) % 97) / 97.0
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector={
                    TEXT_VECTOR_NAME: [base] * EMBEDDING_DIM,
                    IMAGE_VECTOR_NAME: [0.0] * IMAGE_EMBEDDING_DIM,
                },
                payload={
                    "doc_id": doc_id,
                    "chunk_index": index,
                    "text": text,
                    "source_type": "chunk",
                },
            )
        ],
        wait=True,
    )
    return point_id


@pytest.fixture()
def reader_sync_client(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> Iterator[TestClient]:
    """Reader TestClient with sync token configured."""
    monkeypatch.setenv("APP_MODE", "reader")
    monkeypatch.setenv("EMBEDDING_PROVIDER_QUERY", "novita")
    monkeypatch.setenv("RAG_SYNC_TOKEN", SYNC_TOKEN)
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as test_client:
            yield test_client


def test_sync_requires_token(reader_sync_client: TestClient) -> None:
    """Missing bearer yields 401."""
    response = reader_sync_client.get("/sync/v1/watermark")
    assert response.status_code == 401


def test_watermark_and_catalog_empty(reader_sync_client: TestClient) -> None:
    """Fresh reader reports empty watermark and catalog."""
    wm = reader_sync_client.get("/sync/v1/watermark", headers=SYNC_HEADERS)
    assert wm.status_code == 200
    assert wm.json()["present"] is False
    assert wm.json()["release_id"] is None

    cat = reader_sync_client.get("/sync/v1/catalog", headers=SYNC_HEADERS)
    assert cat.status_code == 200
    assert cat.json()["count"] == 0
    assert cat.json()["documents"] == {}


def test_begin_rejects_parent_mismatch(
    reader_sync_client: TestClient, rag_root: Path
) -> None:
    """Begin fails when parent_release_id does not match watermark."""
    write_release_state(rag_root, release_id="already-there", parent_release_id=None)
    body = {
        "manifest": {
            "release_id": "new-one",
            "parent_release_id": "wrong-parent",
            "created_at": "2026-01-01T00:00:00+00:00",
            "files_added": [],
            "files_deleted": [],
            "catalog_doc_ids_upserted": [],
            "catalog_doc_ids_deleted": [],
        }
    }
    response = reader_sync_client.post(
        "/sync/v1/push/begin", headers=SYNC_HEADERS, json=body
    )
    assert response.status_code == 409
    assert "parent_release_id mismatch" in response.json()["detail"]


def test_live_push_commit_applies_files_and_points(
    reader_sync_client: TestClient,
    rag_root: Path,
    tmp_path: Path,
) -> None:
    """Full begin→chunk→commit path applies blobs + vectors in-process."""
    writer = tmp_path / "writer"
    w = RagStorage(str(writer))
    w.ensure_layout()
    doc_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    content = b"live sync hello"
    w.save_original(doc_id, "note.txt", content)
    digest = hashlib.sha256(content).hexdigest()
    w.write_processed(
        doc_id=doc_id,
        filename="note.txt",
        mime="text/plain",
        sha256=digest,
        fulltext="live sync hello",
        chunks=[{"index": 0, "page": 0, "text": "live sync hello"}],
        images=[],
        status="ready",
    )

    w_client = open_qdrant_for_sync(w.qdrant_dir)
    try:
        _upsert_chunk(w_client, doc_id=doc_id, text="live sync hello")
        records, point_ids = export_points_for_docs(w_client, [doc_id])
    finally:
        w_client.close()

    reader_fp = catalog_fingerprint(RagStorage(str(rag_root)))
    manifest = build_file_diff(
        w, parent_release_id=None, reader_catalog=reader_fp or None
    )
    manifest.qdrant_point_ids_upserted = point_ids

    begin = reader_sync_client.post(
        "/sync/v1/push/begin",
        headers=SYNC_HEADERS,
        json={"manifest": manifest.to_dict()},
    )
    assert begin.status_code == 200, begin.text
    assert begin.json()["status"] == "receiving"

    _upload_release_chunks(
        reader_sync_client,
        writer_root=writer,
        manifest=manifest,
        records=records,
    )

    # Idempotent re-PUT of the first blob chunk.
    blob_files = [
        (change.path, (writer / change.path).read_bytes(), change.sha256)
        for change in manifest.files_added
    ]
    first_blob = pack_blob_chunks(manifest.release_id, blob_files)[0]
    again = reader_sync_client.put(
        f"/sync/v1/push/chunk?release_id={manifest.release_id}",
        headers=SYNC_HEADERS,
        content=first_blob.encode(),
    )
    assert again.status_code == 200
    assert again.json()["duplicate"] is True

    commit = reader_sync_client.post(
        "/sync/v1/push/commit",
        headers=SYNC_HEADERS,
        json={"release_id": manifest.release_id, "source": "unit-test"},
    )
    assert commit.status_code == 200, commit.text
    body = commit.json()
    assert body["status"] == "committed"
    assert body["commit"]["points_upserted"] == 1

    # Reader catalog + fulltext + vectors updated; watermark advanced.
    r = RagStorage(str(rag_root))
    assert r.read_fulltext(doc_id) == "live sync hello"
    state = read_release_state(rag_root)
    assert state is not None
    assert state["release_id"] == manifest.release_id

    points = scroll_points_for_doc(
        reader_sync_client.app.state.qdrant,
        collection_name=COLLECTION_NAME,
        doc_id=doc_id,
    )
    assert len(points) == 1
    assert points[0].payload["text"] == "live sync hello"

    # Duplicate commit is idempotent.
    commit2 = reader_sync_client.post(
        "/sync/v1/push/commit",
        headers=SYNC_HEADERS,
        json={"release_id": manifest.release_id},
    )
    assert commit2.status_code == 200
    assert commit2.json()["duplicate"] is True


def test_sync_routes_inactive_on_ingester(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> None:
    """Non-reader modes hide sync with 404 even with a valid token."""
    monkeypatch.setenv("APP_MODE", "ingester")
    monkeypatch.setenv("RAG_SYNC_TOKEN", SYNC_TOKEN)
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as client:
            response = client.get("/sync/v1/watermark", headers=SYNC_HEADERS)
            assert response.status_code == 404


def test_sync_token_unset_returns_404(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> None:
    """Without RAG_SYNC_TOKEN the sync surface is fully inactive (404)."""
    monkeypatch.setenv("APP_MODE", "reader")
    monkeypatch.setenv("EMBEDDING_PROVIDER_QUERY", "novita")
    monkeypatch.delenv("RAG_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("RAG_SYNC_TOKEN_PREVIOUS", raising=False)
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as client:
            assert client.get("/sync/v1/watermark").status_code == 404
            assert (
                client.get(
                    "/sync/v1/watermark",
                    headers={"Authorization": "Bearer anything"},
                ).status_code
                == 404
            )


def test_previous_sync_token_accepted(
    reader_sync_client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    """RAG_SYNC_TOKEN_PREVIOUS is accepted during rotation."""
    monkeypatch.setenv("RAG_SYNC_TOKEN", "current-token")
    monkeypatch.setenv("RAG_SYNC_TOKEN_PREVIOUS", "previous-token")
    # Fixture already set SYNC_TOKEN; override takes effect on next request.
    ok = reader_sync_client.get(
        "/sync/v1/watermark",
        headers={"Authorization": "Bearer previous-token"},
    )
    assert ok.status_code == 200
    cur = reader_sync_client.get(
        "/sync/v1/watermark",
        headers={"Authorization": "Bearer current-token"},
    )
    assert cur.status_code == 200
    bad = reader_sync_client.get(
        "/sync/v1/watermark",
        headers={"Authorization": "Bearer wrong"},
    )
    assert bad.status_code == 401


def test_dry_run_commit_does_not_advance_watermark(
    reader_sync_client: TestClient,
    rag_root: Path,
    tmp_path: Path,
) -> None:
    """dry_run assembles/validates but leaves release.json and catalog untouched."""
    writer = tmp_path / "writer-dry"
    w = RagStorage(str(writer))
    w.ensure_layout()
    doc_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    content = b"dry run payload"
    w.save_original(doc_id, "dry.txt", content)
    digest = hashlib.sha256(content).hexdigest()
    w.write_processed(
        doc_id=doc_id,
        filename="dry.txt",
        mime="text/plain",
        sha256=digest,
        fulltext="dry run payload",
        chunks=[{"index": 0, "page": 0, "text": "dry run payload"}],
        images=[],
        status="ready",
    )
    w_client = open_qdrant_for_sync(w.qdrant_dir)
    try:
        _upsert_chunk(w_client, doc_id=doc_id, text="dry run payload")
        records, point_ids = export_points_for_docs(w_client, [doc_id])
    finally:
        w_client.close()

    before = read_release_state(rag_root)
    manifest = build_file_diff(w, parent_release_id=None, reader_catalog={})
    manifest.qdrant_point_ids_upserted = point_ids

    begin = reader_sync_client.post(
        "/sync/v1/push/begin",
        headers=SYNC_HEADERS,
        json={"manifest": manifest.to_dict(), "dry_run": True},
    )
    assert begin.status_code == 200, begin.text
    assert begin.json()["dry_run"] is True

    _upload_release_chunks(
        reader_sync_client,
        writer_root=writer,
        manifest=manifest,
        records=records,
        expect_first_blob_unique=False,
    )

    validate = reader_sync_client.post(
        "/sync/v1/push/validate",
        headers=SYNC_HEADERS,
        json={"release_id": manifest.release_id},
    )
    assert validate.status_code == 200, validate.text
    assert validate.json()["validate"]["ok"] is True

    commit = reader_sync_client.post(
        "/sync/v1/push/commit",
        headers=SYNC_HEADERS,
        json={"release_id": manifest.release_id},
    )
    assert commit.status_code == 200, commit.text
    assert commit.json()["status"] == "dry_run_ok"

    after = read_release_state(rag_root)
    assert after == before
    assert RagStorage(str(rag_root)).read_fulltext(doc_id) is None


def test_first_push_parent_null_ok(
    reader_sync_client: TestClient,
) -> None:
    """Null watermark accepts parent_release_id=null on first begin."""
    body: dict[str, Any] = {
        "manifest": {
            "release_id": "first-baseline-rel",
            "parent_release_id": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "files_added": [],
            "files_deleted": [],
            "catalog_doc_ids_upserted": [],
            "catalog_doc_ids_deleted": [],
        }
    }
    response = reader_sync_client.post(
        "/sync/v1/push/begin", headers=SYNC_HEADERS, json=body
    )
    assert response.status_code == 200, response.text
    assert response.json()["parent_release_id"] is None


def test_reader_health_exposes_sync_fields(reader_sync_client: TestClient) -> None:
    """Health includes sync/catalog/qdrant additive fields."""
    response = reader_sync_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["app_mode"] == "reader"
    assert body["writes_enabled"] is False
    assert "release_id" in body
    assert body["catalog_doc_count"] == 0
    assert isinstance(body["qdrant_points"], dict)
    assert body["sync_enabled"] is True
    assert body["sync_push_on_ingest"] is False


def test_second_begin_different_release_conflicts(
    reader_sync_client: TestClient,
) -> None:
    """Global push mutex: second begin with another release_id returns 409."""
    first: dict[str, Any] = {
        "manifest": {
            "release_id": "active-rel-1",
            "parent_release_id": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "files_added": [],
            "files_deleted": [],
            "catalog_doc_ids_upserted": [],
            "catalog_doc_ids_deleted": [],
        }
    }
    ok = reader_sync_client.post(
        "/sync/v1/push/begin", headers=SYNC_HEADERS, json=first
    )
    assert ok.status_code == 200, ok.text

    second: dict[str, Any] = {
        "manifest": {
            "release_id": "active-rel-2",
            "parent_release_id": None,
            "created_at": "2026-01-01T00:00:01+00:00",
            "files_added": [],
            "files_deleted": [],
            "catalog_doc_ids_upserted": [],
            "catalog_doc_ids_deleted": [],
        }
    }
    conflict = reader_sync_client.post(
        "/sync/v1/push/begin", headers=SYNC_HEADERS, json=second
    )
    assert conflict.status_code == 409, conflict.text
    detail = conflict.json()["detail"]
    assert detail["active_release_id"] == "active-rel-1"

    # Same release_id remains idempotent.
    resume = reader_sync_client.post(
        "/sync/v1/push/begin", headers=SYNC_HEADERS, json=first
    )
    assert resume.status_code == 200, resume.text


def test_rollback_restores_watermark_for_next_begin(
    reader_sync_client: TestClient,
    rag_root: Path,
    tmp_path: Path,
) -> None:
    """After commit+rollback, release.json matches pre-commit parent."""
    write_release_state(rag_root, release_id="prior-wm", parent_release_id=None)
    writer = tmp_path / "writer-rb"
    w = RagStorage(str(writer))
    w.ensure_layout()
    doc_id = "dddddddd-dddd-dddd-dddd-dddddddddddd"
    content = b"rollback payload"
    w.save_original(doc_id, "rb.txt", content)
    digest = hashlib.sha256(content).hexdigest()
    w.write_processed(
        doc_id=doc_id,
        filename="rb.txt",
        mime="text/plain",
        sha256=digest,
        fulltext="rollback payload",
        chunks=[{"index": 0, "page": 0, "text": "rollback payload"}],
        images=[],
        status="ready",
    )
    w_client = open_qdrant_for_sync(w.qdrant_dir)
    try:
        _upsert_chunk(w_client, doc_id=doc_id, text="rollback payload")
        records, point_ids = export_points_for_docs(w_client, [doc_id])
    finally:
        w_client.close()

    # Parent must match current watermark.
    manifest = build_file_diff(
        w,
        parent_release_id="prior-wm",
        reader_catalog=catalog_fingerprint(RagStorage(str(rag_root))),
    )
    manifest.qdrant_point_ids_upserted = point_ids
    begin = reader_sync_client.post(
        "/sync/v1/push/begin",
        headers=SYNC_HEADERS,
        json={"manifest": manifest.to_dict()},
    )
    assert begin.status_code == 200, begin.text
    _upload_release_chunks(
        reader_sync_client,
        writer_root=writer,
        manifest=manifest,
        records=records,
        expect_first_blob_unique=False,
    )
    commit = reader_sync_client.post(
        "/sync/v1/push/commit",
        headers=SYNC_HEADERS,
        json={"release_id": manifest.release_id},
    )
    assert commit.status_code == 200, commit.text
    assert read_release_state(rag_root)["release_id"] == manifest.release_id

    rollback = reader_sync_client.post(
        f"/sync/v1/rollback/{manifest.release_id}",
        headers=SYNC_HEADERS,
    )
    assert rollback.status_code == 200, rollback.text
    assert rollback.json()["restored_release_id"] == "prior-wm"
    state = read_release_state(rag_root)
    assert state is not None
    assert state["release_id"] == "prior-wm"

    # Next begin with parent=prior-wm succeeds (empty catalog on reader after restore).
    follow: dict[str, Any] = {
        "manifest": {
            "release_id": "after-rollback-rel",
            "parent_release_id": "prior-wm",
            "created_at": "2026-01-02T00:00:00+00:00",
            "files_added": [],
            "files_deleted": [],
            "catalog_doc_ids_upserted": [],
            "catalog_doc_ids_deleted": [],
        }
    }
    again = reader_sync_client.post(
        "/sync/v1/push/begin", headers=SYNC_HEADERS, json=follow
    )
    assert again.status_code == 200, again.text


def test_metadata_fingerprint_push_updates_reader_reliability(
    reader_sync_client: TestClient,
    rag_root: Path,
    tmp_path: Path,
) -> None:
    """Metadata-only reliability change produces a push that updates the reader."""
    from qdrant_client.models import PointIdsList

    from rag.publish.diff import document_catalog_fingerprint

    # Seed reader with a document at reliability=1 (reuse app Qdrant client).
    r = RagStorage(str(rag_root))
    doc_id = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
    content = b"meta seed"
    digest = hashlib.sha256(content).hexdigest()
    r.save_original(doc_id, "m.txt", content)
    r.write_processed(
        doc_id=doc_id,
        filename="m.txt",
        mime="text/plain",
        sha256=digest,
        fulltext="meta seed",
        chunks=[{"index": 0, "page": 0, "text": "meta seed"}],
        images=[],
        status="ready",
    )
    r_meta = r.read_meta(doc_id)
    assert r_meta is not None
    r_meta.extra["reliability"] = 1
    r.write_meta(r_meta)
    write_release_state(rag_root, release_id="seed-wm", parent_release_id=None)
    app_qdrant = reader_sync_client.app.state.qdrant
    _upsert_chunk(app_qdrant, doc_id=doc_id, text="meta seed")
    points = scroll_points_for_doc(
        app_qdrant, collection_name=COLLECTION_NAME, doc_id=doc_id
    )
    app_qdrant.set_payload(
        collection_name=COLLECTION_NAME,
        payload={"reliability": 1},
        points=PointIdsList(points=[points[0].id]),
        wait=True,
    )

    # Writer has same content sha but reliability=3.
    writer = tmp_path / "writer-meta"
    w = RagStorage(str(writer))
    w.ensure_layout()
    w.save_original(doc_id, "m.txt", content)
    w.write_processed(
        doc_id=doc_id,
        filename="m.txt",
        mime="text/plain",
        sha256=digest,
        fulltext="meta seed",
        chunks=[{"index": 0, "page": 0, "text": "meta seed"}],
        images=[],
        status="ready",
    )
    w_meta = w.read_meta(doc_id)
    assert w_meta is not None
    w_meta.extra["reliability"] = 3
    w.write_meta(w_meta)
    w_client = open_qdrant_for_sync(w.qdrant_dir)
    try:
        _upsert_chunk(w_client, doc_id=doc_id, text="meta seed")
        w_points = scroll_points_for_doc(
            w_client, collection_name=COLLECTION_NAME, doc_id=doc_id
        )
        w_client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"reliability": 3},
            points=PointIdsList(points=[w_points[0].id]),
            wait=True,
        )
        records, point_ids = export_points_for_docs(w_client, [doc_id])
    finally:
        w_client.close()

    assert document_catalog_fingerprint(w_meta) != document_catalog_fingerprint(r_meta)
    reader_fp = catalog_fingerprint(r)
    manifest = build_file_diff(w, parent_release_id="seed-wm", reader_catalog=reader_fp)
    assert doc_id in manifest.catalog_doc_ids_upserted
    manifest.qdrant_point_ids_upserted = point_ids

    begin = reader_sync_client.post(
        "/sync/v1/push/begin",
        headers=SYNC_HEADERS,
        json={"manifest": manifest.to_dict()},
    )
    assert begin.status_code == 200, begin.text
    _upload_release_chunks(
        reader_sync_client,
        writer_root=writer,
        manifest=manifest,
        records=records,
        expect_first_blob_unique=False,
    )
    commit = reader_sync_client.post(
        "/sync/v1/push/commit",
        headers=SYNC_HEADERS,
        json={"release_id": manifest.release_id},
    )
    assert commit.status_code == 200, commit.text

    updated = RagStorage(str(rag_root)).read_meta(doc_id)
    assert updated is not None
    assert updated.extra.get("reliability") == 3
    points = scroll_points_for_doc(
        reader_sync_client.app.state.qdrant,
        collection_name=COLLECTION_NAME,
        doc_id=doc_id,
    )
    assert points[0].payload.get("reliability") == 3


def test_apply_qdrant_with_existing_client(tmp_path: Path) -> None:
    """In-process apply reuses the open client and does not close it."""
    from rag.publish.qdrant_sync import (
        apply_qdrant_release_with_client,
        write_points_jsonl,
    )

    root = tmp_path / "reader"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    client = open_qdrant_for_sync(storage.qdrant_dir)
    try:
        doc_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        _upsert_chunk(client, doc_id=doc_id, text="before")
        records, _ = export_points_for_docs(client, [doc_id])
        records[0]["payload"]["text"] = "after"
        bundle = tmp_path / "points.jsonl"
        write_points_jsonl(bundle, records)
        stats = apply_qdrant_release_with_client(
            client,
            upsert_doc_ids=[doc_id],
            delete_doc_ids=[],
            points_jsonl=bundle,
        )
        assert stats["points_upserted"] == 1
        # Client must still work after apply (not closed).
        points = scroll_points_for_doc(
            client, collection_name=COLLECTION_NAME, doc_id=doc_id
        )
        assert points[0].payload["text"] == "after"
    finally:
        client.close()
