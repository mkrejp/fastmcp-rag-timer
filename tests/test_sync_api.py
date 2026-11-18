"""Tests for reader live diff-push sync API (Option A)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Iterator

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

    blob_files = [
        (change.path, (writer / change.path).read_bytes(), change.sha256)
        for change in manifest.files_added
    ]
    for chunk in pack_blob_chunks(manifest.release_id, blob_files):
        put = reader_sync_client.put(
            f"/sync/v1/push/chunk?release_id={manifest.release_id}",
            headers=SYNC_HEADERS,
            content=chunk.encode(),
        )
        assert put.status_code == 200, put.text
        assert put.json()["duplicate"] is False

    # Idempotent re-PUT of the same chunk.
    first_blob = pack_blob_chunks(manifest.release_id, blob_files)[0]
    again = reader_sync_client.put(
        f"/sync/v1/push/chunk?release_id={manifest.release_id}",
        headers=SYNC_HEADERS,
        content=first_blob.encode(),
    )
    assert again.status_code == 200
    assert again.json()["duplicate"] is True

    for chunk in pack_points_chunks(manifest.release_id, records):
        put = reader_sync_client.put(
            f"/sync/v1/push/chunk?release_id={manifest.release_id}",
            headers=SYNC_HEADERS,
            content=chunk.encode(),
        )
        assert put.status_code == 200, put.text

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


def test_sync_routes_forbidden_on_ingester(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> None:
    """Non-reader modes reject sync even with a valid token."""
    monkeypatch.setenv("APP_MODE", "ingester")
    monkeypatch.setenv("RAG_SYNC_TOKEN", SYNC_TOKEN)
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as client:
            response = client.get("/sync/v1/watermark", headers=SYNC_HEADERS)
            assert response.status_code == 403


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
