"""Tests for chunked HTTP upload init, chunks, and complete."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from tests.conftest import AUTH_HEADERS, patched_embeddings


def _sha256(data: bytes) -> str:
    """Return hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _split(data: bytes, chunk_size: int) -> list[bytes]:
    """Split ``data`` into fixed-size chunks."""
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


@pytest.fixture()
def client(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> Iterator[TestClient]:
    """Build a TestClient with temp storage and mocked embeddings."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app, headers=AUTH_HEADERS) as test_client:
            yield test_client


def test_chunked_upload_happy_path_two_chunks(client: TestClient, rag_root: Path) -> None:
    """Init → two chunks → complete should ingest and return doc_id."""
    payload = b"chunked upload " * 4000
    digest = _sha256(payload)
    chunk_size = len(payload) // 2 + 1
    parts = _split(payload, chunk_size)
    assert len(parts) >= 2

    init = client.post(
        "/upload/init",
        json={
            "filename": "notes.txt",
            "size": len(payload),
            "sha256": digest,
            "content_type": "text/plain",
            "chunk_size": chunk_size,
        },
    )
    assert init.status_code == 200
    body = init.json()
    upload_id = body["upload_id"]
    assert body["chunk_count"] == len(parts)
    assert body["chunk_size"] == chunk_size

    for index, part in enumerate(parts):
        put = client.put(
            f"/upload/{upload_id}/chunks/{index}",
            content=part,
            headers={"X-Chunk-SHA256": _sha256(part)},
        )
        assert put.status_code == 200
        assert put.json()["received_indexes"] == list(range(index + 1))

    status = client.get(f"/upload/{upload_id}")
    assert status.status_code == 200
    assert status.json()["complete"] is True

    complete = client.post(f"/upload/{upload_id}/complete")
    assert complete.status_code == 200
    result = complete.json()
    assert result["status"] == "Success"
    assert result["doc_id"]
    assert result["sha256"] == digest

    assert not (rag_root / "uploads" / upload_id).exists()
    assert client.get(f"/upload/{upload_id}").status_code == 404


def test_chunked_upload_idempotent_chunk(client: TestClient) -> None:
    """Re-uploading the same chunk bytes should succeed without conflict."""
    payload = b"repeatable chunk test"
    digest = _sha256(payload)
    init = client.post(
        "/upload/init",
        json={
            "filename": "repeat.txt",
            "size": len(payload),
            "sha256": digest,
            "chunk_size": len(payload),
        },
    )
    upload_id = init.json()["upload_id"]
    headers = {"X-Chunk-SHA256": _sha256(payload)}
    first = client.put(f"/upload/{upload_id}/chunks/0", content=payload, headers=headers)
    second = client.put(f"/upload/{upload_id}/chunks/0", content=payload, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200


def test_chunked_upload_bad_whole_file_sha256(client: TestClient) -> None:
    """Complete should fail when init sha256 does not match assembled bytes."""
    payload = b"wrong digest example"
    init = client.post(
        "/upload/init",
        json={
            "filename": "bad.txt",
            "size": len(payload),
            "sha256": "0" * 64,
            "chunk_size": len(payload),
        },
    )
    upload_id = init.json()["upload_id"]
    assert (
        client.put(f"/upload/{upload_id}/chunks/0", content=payload).status_code == 200
    )
    complete = client.post(f"/upload/{upload_id}/complete")
    assert complete.status_code == 400
    assert "sha256" in complete.json()["detail"].lower()
    assert client.get(f"/upload/{upload_id}").status_code == 200


def test_chunked_upload_missing_chunk(client: TestClient) -> None:
    """Complete with a missing chunk index should return 400."""
    payload = b"alpha beta gamma"
    chunk_size = 5
    parts = _split(payload, chunk_size)
    assert len(parts) >= 2
    init = client.post(
        "/upload/init",
        json={
            "filename": "missing.txt",
            "size": len(payload),
            "sha256": _sha256(payload),
            "chunk_size": chunk_size,
        },
    )
    upload_id = init.json()["upload_id"]
    assert client.put(f"/upload/{upload_id}/chunks/0", content=parts[0]).status_code == 200
    complete = client.post(f"/upload/{upload_id}/complete")
    assert complete.status_code == 400
    assert "missing" in complete.json()["detail"].lower()


def test_chunked_upload_bad_chunk_sha256_header(client: TestClient) -> None:
    """X-Chunk-SHA256 mismatch should reject the chunk."""
    payload = b"header check"
    init = client.post(
        "/upload/init",
        json={
            "filename": "hdr.txt",
            "size": len(payload),
            "sha256": _sha256(payload),
            "chunk_size": len(payload),
        },
    )
    upload_id = init.json()["upload_id"]
    response = client.put(
        f"/upload/{upload_id}/chunks/0",
        content=payload,
        headers={"X-Chunk-SHA256": "f" * 64},
    )
    assert response.status_code == 400


def test_chunked_upload_delete(client: TestClient, rag_root: Path) -> None:
    """DELETE should remove staging for an upload session."""
    payload = b"to delete"
    init = client.post(
        "/upload/init",
        json={
            "filename": "del.txt",
            "size": len(payload),
            "sha256": _sha256(payload),
            "chunk_size": len(payload),
        },
    )
    upload_id = init.json()["upload_id"]
    assert (rag_root / "uploads" / upload_id).is_dir()
    deleted = client.delete(f"/upload/{upload_id}")
    assert deleted.status_code == 204
    assert not (rag_root / "uploads" / upload_id).exists()
