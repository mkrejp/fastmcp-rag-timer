"""Tests for Qdrant point export/import in incremental publish."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from rag.config import COLLECTION_NAME
from rag.embeddings import EMBEDDING_DIM
from rag.image_embed import IMAGE_EMBEDDING_DIM
from rag.ingest import stable_point_id
from rag.publish.apply import apply_release_dir
from rag.publish.diff import build_file_diff
from rag.publish.qdrant_sync import (
    POINTS_REL_PATH,
    apply_qdrant_release,
    apply_qdrant_release_with_client,
    export_points_for_docs,
    open_qdrant_for_sync,
    scroll_points_for_doc,
    write_points_jsonl,
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


def test_export_apply_round_trip(tmp_path: Path) -> None:
    """Exported points land on a fresh reader Qdrant with the same payload."""
    writer_root = tmp_path / "writer"
    reader_root = tmp_path / "reader"
    w_storage = RagStorage(str(writer_root))
    w_storage.ensure_layout()
    r_storage = RagStorage(str(reader_root))
    r_storage.ensure_layout()

    doc_id = "dddddddd-dddd-dddd-dddd-dddddddddddd"
    w_client = open_qdrant_for_sync(w_storage.qdrant_dir)
    try:
        point_id = _upsert_chunk(w_client, doc_id=doc_id, text="hello sync")
        records, ids = export_points_for_docs(w_client, [doc_id])
    finally:
        w_client.close()

    assert point_id in ids
    assert len(records) == 1
    assert records[0]["payload"]["text"] == "hello sync"

    bundle = tmp_path / "points.jsonl"
    write_points_jsonl(bundle, records)
    stats = apply_qdrant_release(
        r_storage.qdrant_dir,
        upsert_doc_ids=[doc_id],
        delete_doc_ids=[],
        points_jsonl=bundle,
    )
    assert stats["points_upserted"] == 1

    r_client = open_qdrant_for_sync(r_storage.qdrant_dir)
    try:
        points = scroll_points_for_doc(
            r_client, collection_name=COLLECTION_NAME, doc_id=doc_id
        )
    finally:
        r_client.close()
    assert len(points) == 1
    assert points[0].payload["text"] == "hello sync"
    assert str(points[0].id) == point_id


def test_apply_release_dir_syncs_qdrant(tmp_path: Path) -> None:
    """Full release apply copies files and upserts Qdrant points."""
    writer = tmp_path / "writer"
    reader = tmp_path / "reader"
    release = tmp_path / "release"
    w = RagStorage(str(writer))
    w.ensure_layout()
    r = RagStorage(str(reader))
    r.ensure_layout()

    doc_id = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
    w.save_original(doc_id, "e.txt", b"echo")
    w.write_processed(
        doc_id=doc_id,
        filename="e.txt",
        mime="text/plain",
        sha256="eee",
        fulltext="echo",
        chunks=[{"index": 0, "page": 0, "text": "echo"}],
        images=[],
        status="ready",
    )
    w_client = open_qdrant_for_sync(w.qdrant_dir)
    try:
        _upsert_chunk(w_client, doc_id=doc_id, text="echo")
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

    state = apply_release_dir(reader, release)
    assert state["release_id"] == manifest.release_id
    assert r.read_fulltext(doc_id) == "echo"

    r_client = open_qdrant_for_sync(r.qdrant_dir)
    try:
        points = scroll_points_for_doc(
            r_client, collection_name=COLLECTION_NAME, doc_id=doc_id
        )
    finally:
        r_client.close()
    assert len(points) == 1
    assert points[0].payload["text"] == "echo"


def test_delete_doc_removes_reader_points(tmp_path: Path) -> None:
    """Deleted catalog docs purge matching reader points even without JSONL."""
    reader = tmp_path / "reader"
    r = RagStorage(str(reader))
    r.ensure_layout()
    doc_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
    client = open_qdrant_for_sync(r.qdrant_dir)
    try:
        _upsert_chunk(client, doc_id=doc_id, text="gone")
    finally:
        client.close()

    stats = apply_qdrant_release(
        r.qdrant_dir,
        upsert_doc_ids=[],
        delete_doc_ids=[doc_id],
        points_jsonl=None,
    )
    assert stats["points_deleted"] == 1
    client = open_qdrant_for_sync(r.qdrant_dir)
    try:
        left = scroll_points_for_doc(
            client, collection_name=COLLECTION_NAME, doc_id=doc_id
        )
    finally:
        client.close()
    assert left == []


def test_metadata_only_apply_uses_set_payload_not_upsert(tmp_path: Path) -> None:
    """Metadata-only sync merges payloads without vector upsert or delete."""
    reader = tmp_path / "reader"
    r = RagStorage(str(reader))
    r.ensure_layout()
    doc_id = "99999999-9999-9999-9999-999999999999"
    client = open_qdrant_for_sync(r.qdrant_dir)
    try:
        _upsert_chunk(client, doc_id=doc_id, text="before")
        points = scroll_points_for_doc(
            client, collection_name=COLLECTION_NAME, doc_id=doc_id
        )
        point_id = str(points[0].id)
        payload_records = [
            {
                "collection": COLLECTION_NAME,
                "id": point_id,
                "vector": None,
                "payload": {
                    "doc_id": doc_id,
                    "text": "before",
                    "reliability": 3,
                },
            }
        ]
        with patch.object(client, "upsert") as mock_upsert:
            stats = apply_qdrant_release_with_client(
                client,
                upsert_doc_ids=[doc_id],
                delete_doc_ids=[],
                metadata_only_doc_ids=[doc_id],
                records=payload_records,
            )
        mock_upsert.assert_not_called()
        assert stats["points_upserted"] == 0
        assert stats["points_payload_merged"] == 1
        updated = scroll_points_for_doc(
            client, collection_name=COLLECTION_NAME, doc_id=doc_id
        )
        assert updated[0].payload.get("reliability") == 3
    finally:
        client.close()
