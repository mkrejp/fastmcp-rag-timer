"""Tests for live sync chunk pack/unpack and reassembly."""

from __future__ import annotations

import hashlib

import pytest

from rag.publish.sync_chunk import (
    assemble_blob_files,
    assemble_points_jsonl,
    decode_chunk,
    pack_blob_chunks,
    pack_points_chunks,
)
from rag.publish.sync_constants import (
    CHUNK_HARD_MAX_BYTES,
    CHUNK_TARGET_BYTES,
    POINTS_STAGE,
)


def test_chunk_constants_match_approved_budget() -> None:
    """Marek-approved 32 MiB target with soft/hard ceilings."""
    assert CHUNK_TARGET_BYTES == 32 * 1024 * 1024
    assert CHUNK_HARD_MAX_BYTES == 96 * 1024 * 1024


def test_pack_many_small_files_into_one_chunk(tmp_path) -> None:
    """Small files share a single chunk under the target budget."""
    files = [(f"originals/d{i}/f.txt", f"hello-{i}".encode(), "") for i in range(5)]
    files = [(path, data, hashlib.sha256(data).hexdigest()) for path, data, _ in files]
    chunks = pack_blob_chunks("rel-1", files, target_bytes=1024, soft_max_bytes=2048)
    assert len(chunks) == 1
    encoded = chunks[0].encode()
    decoded = decode_chunk(encoded)
    assert decoded.stage == "blobs"
    assert decoded.seq == 0
    assert len(decoded.entries) == 5

    written = assemble_blob_files([decoded], tmp_path / "out")
    assert len(written) == 5
    assert (tmp_path / "out" / "originals/d0/f.txt").read_text() == "hello-0"


def test_oversized_blob_split_across_parts(tmp_path) -> None:
    """A single blob larger than target is split into part chunks."""
    data = b"x" * 2500
    digest = hashlib.sha256(data).hexdigest()
    chunks = pack_blob_chunks(
        "rel-2",
        [("originals/big/a.bin", data, digest)],
        target_bytes=1000,
        soft_max_bytes=1500,
    )
    assert len(chunks) >= 3
    for chunk in chunks:
        for entry in chunk.entries:
            assert entry.parts == len(chunks)
            assert entry.file_sha256 == digest

    written = assemble_blob_files(chunks, tmp_path / "out")
    assert written == ["originals/big/a.bin"]
    assert (tmp_path / "out" / "originals/big/a.bin").read_bytes() == data


def test_hard_max_rejects_huge_wire_body() -> None:
    """Decode rejects bodies over the hard max."""
    with pytest.raises(ValueError, match="hard max"):
        decode_chunk(b"x" * (CHUNK_HARD_MAX_BYTES + 1))


def test_pack_points_jsonl_round_trip(tmp_path) -> None:
    """Point records survive JSONL chunk packing."""
    records = [
        {
            "collection": "c",
            "id": "1",
            "vector": {"t": [0.1]},
            "payload": {"doc_id": "a"},
        },
        {
            "collection": "c",
            "id": "2",
            "vector": {"t": [0.2]},
            "payload": {"doc_id": "b"},
        },
    ]
    chunks = pack_points_chunks("rel-3", records, target_bytes=10_000)
    assert len(chunks) == 1
    assert chunks[0].stage == POINTS_STAGE
    path = assemble_points_jsonl(chunks, tmp_path / "points.jsonl")
    assert path is not None
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
