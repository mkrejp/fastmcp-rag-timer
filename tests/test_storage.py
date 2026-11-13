"""Tests for persistent RAG filesystem storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from rag.storage import RagStorage, parse_iso_timestamp

if TYPE_CHECKING:
    pass


def test_ensure_layout_creates_tree(rag_root: Path) -> None:
    """Creating storage should materialize originals, processed, qdrant, index."""
    storage = RagStorage(rag_root)
    storage.ensure_layout()

    assert (rag_root / "originals").is_dir()
    assert (rag_root / "processed").is_dir()
    assert (rag_root / "qdrant").is_dir()
    assert (rag_root / "versions").is_dir()
    assert (rag_root / "index.json").is_file()
    assert storage.list_documents() == []


def test_write_and_read_processed_artifacts(rag_storage: RagStorage) -> None:
    """Processed fulltext, chunks, and meta should round-trip on disk."""
    doc_id = "11111111-1111-1111-1111-111111111111"
    rag_storage.save_original(doc_id, "note.txt", b"hello world")
    meta = rag_storage.write_processed(
        doc_id=doc_id,
        filename="note.txt",
        mime="text/plain",
        sha256="abc",
        fulltext="hello world",
        chunks=[{"index": 0, "text": "hello world", "page": 0}],
    )

    assert meta.chunk_count == 1
    assert rag_storage.read_fulltext(doc_id) == "hello world"
    chunks = rag_storage.read_chunks(doc_id)
    assert chunks is not None
    assert chunks[0]["text"] == "hello world"
    loaded = rag_storage.read_meta(doc_id)
    assert loaded is not None
    assert loaded.filename == "note.txt"
    catalog = rag_storage.list_documents()
    assert len(catalog) == 1
    assert catalog[0]["doc_id"] == doc_id


def test_meta_preserves_original_lat_lon_and_extra(rag_storage: RagStorage) -> None:
    """Original GPS fields on meta.json must survive a geo_refined write round-trip."""
    doc_id = "33333333-3333-3333-3333-333333333333"
    rag_storage.save_original(doc_id, "akon.txt", b"akon")
    rag_storage.write_processed(
        doc_id=doc_id,
        filename="akon.txt",
        mime="text/plain",
        sha256="gps",
        fulltext="akon",
        chunks=[{"index": 0, "text": "akon", "page": 0, "lat": 50.0, "lon": 14.4}],
    )
    meta = rag_storage.read_meta(doc_id)
    assert meta is not None
    meta.lat = 50.0755
    meta.lon = 14.4378
    meta.extra["akon_id"] = "AK110_057"
    meta.geo_refined = {
        "lat": 50.366,
        "lon": 12.938,
        "precision": "building",
        "precision_m": 25,
        "confidence": 0.8,
        "confidence_level": "high",
        "status": "zpresneno",
        "needs_refinement": False,
        "verified": False,
        "method": "wikidata_p625",
    }
    rag_storage.write_meta(meta)
    loaded = rag_storage.read_meta(doc_id)
    assert loaded is not None
    assert loaded.lat == pytest.approx(50.0755)
    assert loaded.lon == pytest.approx(14.4378)
    assert loaded.extra.get("akon_id") == "AK110_057"
    assert loaded.geo_refined is not None
    assert loaded.geo_refined["lat"] == pytest.approx(50.366)
    raw = json.loads(
        (rag_storage.processed_doc_dir(doc_id) / "meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw["lat"] == pytest.approx(50.0755)
    assert raw["akon_id"] == "AK110_057"


def test_find_by_sha256(rag_storage: RagStorage) -> None:
    """Catalog lookup by content hash should return the matching document."""
    doc_id = "22222222-2222-2222-2222-222222222222"
    rag_storage.save_original(doc_id, "a.txt", b"same")
    rag_storage.write_processed(
        doc_id=doc_id,
        filename="a.txt",
        mime="text/plain",
        sha256="deadbeef",
        fulltext="same",
        chunks=[{"index": 0, "text": "same", "page": 0}],
    )
    found = rag_storage.find_by_sha256("deadbeef")
    assert found is not None
    assert found.doc_id == doc_id
    assert rag_storage.find_by_sha256("missing") is None


def test_list_documents_since_filters_uploaded_at(rag_storage: RagStorage) -> None:
    """``since`` should keep catalog rows with uploaded_at >= the cutoff."""
    older = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    newer = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    rag_storage.save_original(older, "old.txt", b"old")
    rag_storage.write_processed(
        doc_id=older,
        filename="old.txt",
        mime="text/plain",
        sha256="old",
        fulltext="old",
        chunks=[{"index": 0, "text": "old", "page": 0}],
        uploaded_at="2026-01-01T00:00:00+00:00",
    )
    rag_storage.save_original(newer, "new.txt", b"new")
    rag_storage.write_processed(
        doc_id=newer,
        filename="new.txt",
        mime="text/plain",
        sha256="new",
        fulltext="new",
        chunks=[{"index": 0, "text": "new", "page": 0}],
        uploaded_at="2026-06-01T12:00:00Z",
    )
    assert len(rag_storage.list_documents()) == 2
    since_june = rag_storage.list_documents(since="2026-06-01T00:00:00Z")
    assert [row["doc_id"] for row in since_june] == [newer]
    inclusive = rag_storage.list_documents(since="2026-06-01T12:00:00+00:00")
    assert [row["doc_id"] for row in inclusive] == [newer]
    none_match = rag_storage.list_documents(since="2026-12-01T00:00:00Z")
    assert none_match == []
    assert rag_storage.list_documents(since="") == rag_storage.list_documents()
    with pytest.raises(ValueError, match="invalid ISO timestamp"):
        rag_storage.list_documents(since="not-a-date")
    parsed = parse_iso_timestamp("2026-06-01T12:00:00Z")
    assert parsed.tzinfo is not None
    naive = parse_iso_timestamp("2026-06-01T12:00:00")
    assert naive.tzinfo is not None
