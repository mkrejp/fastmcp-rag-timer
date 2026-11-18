"""Tests for incremental publish file diffs."""

from __future__ import annotations

from pathlib import Path

from rag.publish.apply import apply_file_release
from rag.publish.diff import (
    build_file_diff,
    catalog_fingerprint,
    document_catalog_fingerprint,
    summarize_diff,
)
from rag.publish.manifest import FileChange
from rag.storage import RagStorage


def test_baseline_then_incremental_diff(tmp_path: Path) -> None:
    """Empty reader catalog yields baseline; second pass only changed docs."""
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    doc_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    storage.save_original(doc_a, "a.txt", b"alpha")
    storage.write_processed(
        doc_id=doc_a,
        filename="a.txt",
        mime="text/plain",
        sha256="aaa",
        fulltext="alpha",
        chunks=[{"index": 0, "page": 0, "text": "alpha"}],
        images=[],
        status="ready",
    )

    baseline = build_file_diff(storage, parent_release_id=None, reader_catalog=None)
    assert baseline.full_baseline is True
    assert doc_a in baseline.catalog_doc_ids_upserted
    assert any(
        f.path == "index.json" or f.path.startswith("originals/")
        for f in baseline.files_added
    )
    summary = summarize_diff(baseline)
    assert summary["files_added"] >= 1

    writer_fp = catalog_fingerprint(storage)
    # Reader already has doc-a at same fingerprint → empty upsert set.
    empty = build_file_diff(
        storage,
        parent_release_id=baseline.release_id,
        reader_catalog=writer_fp,
    )
    assert empty.full_baseline is False
    assert empty.catalog_doc_ids_upserted == []
    assert empty.catalog_doc_ids_deleted == []

    storage.save_original(doc_b, "b.txt", b"beta")
    storage.write_processed(
        doc_id=doc_b,
        filename="b.txt",
        mime="text/plain",
        sha256="bbb",
        fulltext="beta",
        chunks=[{"index": 0, "page": 0, "text": "beta"}],
        images=[],
        status="ready",
    )

    delta = build_file_diff(
        storage,
        parent_release_id=baseline.release_id,
        reader_catalog={doc_a: writer_fp[doc_a]},
    )
    assert delta.catalog_doc_ids_upserted == [doc_b]
    assert all((f.doc_id == doc_b) or f.path == "index.json" for f in delta.files_added)


def test_metadata_only_change_produces_catalog_diff(tmp_path: Path) -> None:
    """Reliability / lifecycle edits change the composite fingerprint."""
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    storage.save_original(doc_id, "c.txt", b"content")
    storage.write_processed(
        doc_id=doc_id,
        filename="c.txt",
        mime="text/plain",
        sha256="ccc",
        fulltext="content",
        chunks=[{"index": 0, "page": 0, "text": "content"}],
        images=[],
        status="ready",
    )
    meta = storage.read_meta(doc_id)
    assert meta is not None
    before = document_catalog_fingerprint(meta)
    reader_fp = {doc_id: before}

    empty = build_file_diff(storage, parent_release_id="r0", reader_catalog=reader_fp)
    assert empty.catalog_doc_ids_upserted == []

    meta.extra["reliability"] = 3
    storage.write_meta(meta)
    updated = storage.read_meta(doc_id)
    assert updated is not None
    after = document_catalog_fingerprint(updated)
    assert after != before

    delta = build_file_diff(storage, parent_release_id="r0", reader_catalog=reader_fp)
    assert delta.catalog_doc_ids_upserted == [doc_id]
    assert any(f.path.startswith(f"processed/{doc_id}/") for f in delta.files_added)


def test_apply_file_release_copies_blobs(tmp_path: Path) -> None:
    """Apply copies staged blobs and writes release.json."""
    rag_root = tmp_path / "reader"
    blob_root = tmp_path / "blobs"
    (blob_root / "originals" / "x").mkdir(parents=True)
    payload = b"hello"
    (blob_root / "originals" / "x" / "f.txt").write_bytes(payload)

    from rag.publish.manifest import ReleaseManifest, new_release_id
    from rag.storage import utc_now_iso

    manifest = ReleaseManifest(
        release_id=new_release_id(),
        parent_release_id=None,
        created_at=utc_now_iso(),
        files_added=[
            FileChange(
                path="originals/x/f.txt",
                sha256="unused",
                size=len(payload),
                doc_id="x",
            )
        ],
        full_baseline=True,
    )
    state = apply_file_release(rag_root, manifest=manifest, blob_root=blob_root)
    assert state["release_id"] == manifest.release_id
    assert (rag_root / "originals" / "x" / "f.txt").read_bytes() == payload
    assert (rag_root / "release.json").is_file()
