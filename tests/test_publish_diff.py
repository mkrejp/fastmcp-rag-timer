"""Tests for incremental publish file diffs."""

from __future__ import annotations

from pathlib import Path

from rag.publish.apply import apply_file_release
from rag.publish.diff import build_file_diff, summarize_diff
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

    baseline = build_file_diff(
        storage, parent_release_id=None, reader_catalog=None
    )
    assert baseline.full_baseline is True
    assert doc_a in baseline.catalog_doc_ids_upserted
    assert any(
        f.path == "index.json" or f.path.startswith("originals/")
        for f in baseline.files_added
    )
    summary = summarize_diff(baseline)
    assert summary["files_added"] >= 1

    # Reader already has doc-a at same sha → empty upsert set.
    empty = build_file_diff(
        storage,
        parent_release_id=baseline.release_id,
        reader_catalog={doc_a: "aaa"},
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
        reader_catalog={doc_a: "aaa"},
    )
    assert delta.catalog_doc_ids_upserted == [doc_b]
    assert all(
        (f.doc_id == doc_b) or f.path == "index.json" for f in delta.files_added
    )


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
