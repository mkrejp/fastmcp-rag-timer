"""Tests for incremental publish file diffs."""

from __future__ import annotations

from pathlib import Path

from rag.publish.apply import apply_file_release
from rag.publish.diff import (
    build_file_diff,
    catalog_fingerprint,
    document_catalog_fingerprint,
    document_content_fingerprint,
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
    # V2: catalog parity must not list index.json (or any file) as a change.
    assert empty.files_added == []
    assert empty.files_deleted == []
    assert summarize_diff(empty)["files_added"] == 0
    assert summarize_diff(empty)["bytes_added"] == 0

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


def test_v2_catalog_parity_excludes_index_json(tmp_path: Path) -> None:
    """V2 regression: parity with index.json on disk must yield an empty diff.

    Incremental ``build_file_diff`` used to always append ``index.json``, so
    ``files_added=1`` / ``bytes_added>0`` even when catalogs matched — which
    made a no-change cron open a new release.
    """
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_id = "dddddddd-dddd-dddd-dddd-dddddddddddd"
    storage.save_original(doc_id, "d.txt", b"delta")
    storage.write_processed(
        doc_id=doc_id,
        filename="d.txt",
        mime="text/plain",
        sha256="ddd",
        fulltext="delta",
        chunks=[{"index": 0, "page": 0, "text": "delta"}],
        images=[],
        status="ready",
    )
    assert (root / "index.json").is_file()
    reader_fp = catalog_fingerprint(storage)

    parity = build_file_diff(
        storage,
        parent_release_id="rel-parity",
        reader_catalog=reader_fp,
    )
    assert parity.full_baseline is False
    assert parity.catalog_doc_ids_upserted == []
    assert parity.catalog_doc_ids_deleted == []
    assert parity.files_added == []
    assert parity.files_deleted == []
    assert not any(f.path == "index.json" for f in parity.files_added)
    summary = summarize_diff(parity)
    assert summary["files_added"] == 0
    assert summary["bytes_added"] == 0
    assert summary["docs_upserted"] == 0
    assert summary["docs_deleted"] == 0


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


def test_tag_only_change_produces_catalog_diff(tmp_path: Path) -> None:
    """Tags live in catalog extra and must change the composite fingerprint."""
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_id = "11111111-1111-1111-1111-111111111111"
    storage.save_original(doc_id, "t.txt", b"tagged")
    storage.write_processed(
        doc_id=doc_id,
        filename="t.txt",
        mime="text/plain",
        sha256="tag-sha",
        fulltext="tagged",
        chunks=[{"index": 0, "page": 0, "text": "tagged"}],
        images=[],
        status="ready",
    )
    meta = storage.read_meta(doc_id)
    assert meta is not None
    reader_fp = {doc_id: document_catalog_fingerprint(meta)}

    meta.extra["tags"] = ["secese"]
    storage.write_meta(meta)
    delta = build_file_diff(storage, parent_release_id="r0", reader_catalog=reader_fp)
    assert delta.catalog_doc_ids_upserted == [doc_id]


def test_geo_refined_change_produces_catalog_diff(tmp_path: Path) -> None:
    """``geo_refined`` on meta.json is included in the catalog fingerprint."""
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_id = "22222222-2222-2222-2222-222222222222"
    storage.save_original(doc_id, "g.txt", b"geo")
    storage.write_processed(
        doc_id=doc_id,
        filename="g.txt",
        mime="text/plain",
        sha256="geo-sha",
        fulltext="geo",
        chunks=[{"index": 0, "page": 0, "text": "geo"}],
        images=[],
        status="ready",
    )
    meta = storage.read_meta(doc_id)
    assert meta is not None
    reader_fp = {doc_id: document_catalog_fingerprint(meta)}

    meta.geo_refined = {
        "lat": 50.08,
        "lon": 14.42,
        "precision": "building",
        "status": "refined",
        "confidence_level": "high",
        "needs_refinement": False,
        "verified": True,
    }
    storage.write_meta(meta)
    delta = build_file_diff(storage, parent_release_id="r0", reader_catalog=reader_fp)
    assert delta.catalog_doc_ids_upserted == [doc_id]


def test_metadata_only_when_reader_content_matches(tmp_path: Path) -> None:
    """Same content digest → metadata_only list (no originals in file diff)."""
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_id = "33333333-3333-3333-3333-333333333333"
    storage.save_original(doc_id, "m.txt", b"meta")
    storage.write_processed(
        doc_id=doc_id,
        filename="m.txt",
        mime="text/plain",
        sha256="meta-sha",
        fulltext="meta",
        chunks=[{"index": 0, "page": 0, "text": "meta"}],
        images=[],
        status="ready",
    )
    meta = storage.read_meta(doc_id)
    assert meta is not None
    reader_fp = {doc_id: document_catalog_fingerprint(meta)}
    reader_content = {doc_id: document_content_fingerprint(meta)}

    meta.extra["reliability"] = 2
    storage.write_meta(meta)

    delta = build_file_diff(
        storage,
        parent_release_id="r0",
        reader_catalog=reader_fp,
        reader_content_catalog=reader_content,
    )
    assert doc_id in delta.catalog_doc_ids_metadata_only
    paths = {change.path for change in delta.files_added}
    assert f"processed/{doc_id}/meta.json" in paths
    assert not any(p.startswith(f"originals/{doc_id}") for p in paths)


def test_unchanged_doc_produces_empty_diff(tmp_path: Path) -> None:
    """Identical catalog fingerprint must not open a release diff."""
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_id = "44444444-4444-4444-4444-444444444444"
    storage.save_original(doc_id, "u.txt", b"same")
    storage.write_processed(
        doc_id=doc_id,
        filename="u.txt",
        mime="text/plain",
        sha256="same-sha",
        fulltext="same",
        chunks=[{"index": 0, "page": 0, "text": "same"}],
        images=[],
        status="ready",
    )
    meta = storage.read_meta(doc_id)
    assert meta is not None
    meta.extra["tags"] = ["praha"]
    storage.write_meta(meta)
    fp = catalog_fingerprint(storage)
    empty = build_file_diff(storage, parent_release_id="r0", reader_catalog=fp)
    assert empty.catalog_doc_ids_upserted == []
    assert empty.files_added == []


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
