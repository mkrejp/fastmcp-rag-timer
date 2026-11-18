"""Unit tests for in-process Qdrant/file apply primitives (diff-push PR1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from rag.config import COLLECTION_NAME
from rag.embeddings import EMBEDDING_DIM
from rag.image_embed import IMAGE_EMBEDDING_DIM
from rag.ingest import stable_point_id
from rag.publish.apply import (
    apply_file_release,
    restore_touched_docs_snapshot,
    snapshot_touched_docs,
)
from rag.publish.diff import build_file_diff
from rag.publish.qdrant_sync import (
    POINTS_REL_PATH,
    apply_point_records,
    apply_qdrant_release,
    apply_qdrant_release_with_client,
    delete_points_by_doc_id,
    export_points_for_docs,
    open_qdrant_for_sync,
    scroll_points_for_doc,
    write_points_jsonl,
)
from rag.publish.sync_constants import (
    CHUNK_HARD_MAX_BYTES,
    CHUNK_SOFT_MAX_BYTES,
    CHUNK_TARGET_BYTES,
)
from rag.storage import RagStorage
from rag.vectorstore import IMAGE_VECTOR_NAME, TEXT_VECTOR_NAME


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


def test_sync_chunk_constants() -> None:
    """Chunk budgets match the agreed 32 / 64 / 96 MiB ladder."""
    assert CHUNK_TARGET_BYTES == 32 * 1024 * 1024
    assert CHUNK_SOFT_MAX_BYTES == 64 * 1024 * 1024
    assert CHUNK_HARD_MAX_BYTES == 96 * 1024 * 1024
    assert CHUNK_TARGET_BYTES < CHUNK_SOFT_MAX_BYTES < CHUNK_HARD_MAX_BYTES


def test_apply_with_existing_client_no_second_open(tmp_path: Path) -> None:
    """In-process apply mutates points on the provided client without reopen."""
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    client = open_qdrant_for_sync(storage.qdrant_dir)
    try:
        _upsert_chunk(client, doc_id=doc_id, text="before")
        records, _ids = export_points_for_docs(client, [doc_id])
        assert len(records) == 1
        records[0]["payload"]["text"] = "after"

        stats = apply_qdrant_release_with_client(
            client,
            upsert_doc_ids=[doc_id],
            delete_doc_ids=[],
            records=records,
        )
        assert stats["points_upserted"] == 1
        assert stats["points_deleted"] == 1

        points = scroll_points_for_doc(
            client, collection_name=COLLECTION_NAME, doc_id=doc_id
        )
        assert len(points) == 1
        assert points[0].payload["text"] == "after"
    finally:
        client.close()


def test_apply_point_records_and_delete_by_doc_id(tmp_path: Path) -> None:
    """Helpers upsert then delete points by doc_id on an open client."""
    storage = RagStorage(str(tmp_path / "rag"))
    storage.ensure_layout()
    doc_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    client = open_qdrant_for_sync(storage.qdrant_dir)
    try:
        point_id = _upsert_chunk(client, doc_id=doc_id, text="keep")
        records, _ = export_points_for_docs(client, [doc_id])
        records[0]["payload"]["text"] = "mutated"
        written = apply_point_records(client, records)
        assert written == 1
        points = scroll_points_for_doc(
            client, collection_name=COLLECTION_NAME, doc_id=doc_id
        )
        assert points[0].payload["text"] == "mutated"
        assert str(points[0].id) == point_id

        removed = delete_points_by_doc_id(client, [doc_id])
        assert removed == 1
        assert (
            scroll_points_for_doc(
                client, collection_name=COLLECTION_NAME, doc_id=doc_id
            )
            == []
        )
    finally:
        client.close()


def test_cli_wrapper_still_opens_and_closes(tmp_path: Path) -> None:
    """``apply_qdrant_release`` remains usable as an open+close offline path."""
    storage = RagStorage(str(tmp_path / "rag"))
    storage.ensure_layout()
    doc_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    client = open_qdrant_for_sync(storage.qdrant_dir)
    try:
        _upsert_chunk(client, doc_id=doc_id, text="seed")
        records, _ = export_points_for_docs(client, [doc_id])
    finally:
        client.close()

    bundle = tmp_path / "points.jsonl"
    records[0]["payload"]["text"] = "via-wrapper"
    write_points_jsonl(bundle, records)
    stats = apply_qdrant_release(
        storage.qdrant_dir,
        upsert_doc_ids=[doc_id],
        delete_doc_ids=[],
        points_jsonl=bundle,
    )
    assert stats["points_upserted"] == 1

    verify = open_qdrant_for_sync(storage.qdrant_dir)
    try:
        points = scroll_points_for_doc(
            verify, collection_name=COLLECTION_NAME, doc_id=doc_id
        )
    finally:
        verify.close()
    assert points[0].payload["text"] == "via-wrapper"


def test_apply_file_release_with_existing_client(tmp_path: Path) -> None:
    """File apply can sync Qdrant through a caller-owned client."""
    writer = tmp_path / "writer"
    reader = tmp_path / "reader"
    release = tmp_path / "release"
    w = RagStorage(str(writer))
    w.ensure_layout()
    r = RagStorage(str(reader))
    r.ensure_layout()

    doc_id = "dddddddd-dddd-dddd-dddd-dddddddddddd"
    w.save_original(doc_id, "d.txt", b"delta")
    w.write_processed(
        doc_id=doc_id,
        filename="d.txt",
        mime="text/plain",
        sha256="ddd",
        fulltext="delta",
        chunks=[{"index": 0, "page": 0, "text": "delta"}],
        images=[],
        status="ready",
    )
    w_client = open_qdrant_for_sync(w.qdrant_dir)
    try:
        _upsert_chunk(w_client, doc_id=doc_id, text="delta")
        records, point_ids = export_points_for_docs(w_client, [doc_id])
    finally:
        w_client.close()

    manifest = build_file_diff(w, parent_release_id=None, reader_catalog=None)
    manifest.qdrant_point_ids_upserted = point_ids
    release.mkdir(parents=True)
    manifest.write_json(release / "manifest.json")
    write_points_jsonl(release / POINTS_REL_PATH, records)
    blob_root = release / "blobs"
    for change in manifest.files_added:
        src = writer / change.path
        dest = blob_root / change.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())

    r_client = open_qdrant_for_sync(r.qdrant_dir)
    try:
        state = apply_file_release(
            reader,
            manifest=manifest,
            blob_root=blob_root,
            points_jsonl=release / POINTS_REL_PATH,
            qdrant_client=r_client,
        )
        assert state["release_id"] == manifest.release_id
        assert state["points_upserted"] == 1
        points = scroll_points_for_doc(
            r_client, collection_name=COLLECTION_NAME, doc_id=doc_id
        )
        assert len(points) == 1
        assert points[0].payload["text"] == "delta"
    finally:
        r_client.close()
    assert r.read_fulltext(doc_id) == "delta"


def test_rollback_snapshot_round_trip(tmp_path: Path) -> None:
    """Snapshot captures files+points; restore undoes a later mutation."""
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_id = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
    storage.save_original(doc_id, "e.txt", b"original-bytes")
    storage.write_processed(
        doc_id=doc_id,
        filename="e.txt",
        mime="text/plain",
        sha256="eee",
        fulltext="original-text",
        chunks=[{"index": 0, "page": 0, "text": "original-text"}],
        images=[],
        status="ready",
    )

    client = open_qdrant_for_sync(storage.qdrant_dir)
    try:
        _upsert_chunk(client, doc_id=doc_id, text="original-text")
        snap_meta = snapshot_touched_docs(
            root,
            client=client,
            doc_ids=[doc_id],
            snapshot_dir=tmp_path / "rollback" / "snap1",
        )
        assert snap_meta["points_exported"] == 1
        assert (tmp_path / "rollback" / "snap1" / "qdrant" / "points.jsonl").is_file()
        assert (
            tmp_path / "rollback" / "snap1" / "files" / "originals" / doc_id
        ).is_dir()

        # Mutate live state after snapshot.
        (storage.processed_doc_dir(doc_id) / "fulltext.txt").write_text(
            "mutated", encoding="utf-8"
        )
        records, _ = export_points_for_docs(client, [doc_id])
        records[0]["payload"]["text"] = "mutated"
        apply_point_records(client, records)
        assert storage.read_fulltext(doc_id) == "mutated"

        restored: dict[str, Any] = restore_touched_docs_snapshot(
            root,
            client=client,
            snapshot_dir=tmp_path / "rollback" / "snap1",
        )
        assert restored["points_upserted"] == 1
        assert storage.read_fulltext(doc_id) == "original-text"
        points = scroll_points_for_doc(
            client, collection_name=COLLECTION_NAME, doc_id=doc_id
        )
        assert points[0].payload["text"] == "original-text"
    finally:
        client.close()
