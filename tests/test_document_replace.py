"""Tests for in-place document content replacement and versioning."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.config import LOCALITY_LINKS_COLLECTION
from rag.doc_locks import doc_lock
from rag.document_delete import DocumentNotFoundError, delete_document
from rag.document_patch import patch_document, patch_document_metadata
from rag.document_replace import (
    DocumentReplaceError,
    DocumentVersionNotFoundError,
    list_document_versions,
    replace_document_content,
    restore_document_version,
)
from rag.geo_index import refine_geo
from rag.ingest import content_sha256, ingest_bytes, stable_point_id
from rag.storage import RagStorage
from rag.vectorstore import denormalize_knowledge_localities, upsert_locality_links
from tests.conftest import AUTH_HEADERS, patched_embeddings
from tests.fixtures.pdf_factory import build_image_rich_pdf

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from qdrant_client import QdrantClient

COLLECTION = "client_domain_knowledge"


def _scroll_doc(
    qdrant_client: QdrantClient, doc_id: str, *, with_vectors: bool = False
) -> list[Any]:
    """Return all knowledge points for ``doc_id``."""
    points, _offset = qdrant_client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=64,
        with_payload=True,
        with_vectors=with_vectors,
    )
    return list(points)


def _chunks(qdrant_client: QdrantClient, doc_id: str) -> list[Any]:
    """Return text-chunk points for ``doc_id``."""
    return [
        point
        for point in _scroll_doc(qdrant_client, doc_id)
        if (point.payload or {}).get("source_type") == "pdf_chunk"
    ]


def _ingest_note(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
    *,
    filename: str = "note.txt",
    data: bytes = b"Original indexed sentence about a facade.",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Ingest a small text document and return its ``doc_id``."""
    result = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename=filename,
        data=data,
        embed_fn=fake_embed,
        metadata=metadata,
    )
    return str(result["doc_id"])


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


def test_replace_rewrites_chunks_and_archives_version(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Replace should keep doc_id, re-embed, and persist a version snapshot."""
    doc_id = _ingest_note(rag_storage, qdrant_client, fake_embed)
    old_ids = {point.id for point in _chunks(qdrant_client, doc_id)}
    assert old_ids

    result = replace_document_content(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        content="Replacement text about a different facade detail.",
        reason="correct OCR",
        author="tester@example.test",
        embed_fn=fake_embed,
    )
    assert result["status"] == "replaced"
    assert result["doc_id"] == doc_id
    assert result["version"] == 1
    assert result["chunks_indexed"] >= 1

    new_points = _chunks(qdrant_client, doc_id)
    assert new_points
    new_ids = {point.id for point in new_points}
    assert new_ids.isdisjoint(old_ids)
    texts = " ".join(str((p.payload or {}).get("text") or "") for p in new_points)
    assert "Replacement text" in texts
    assert "Original indexed" not in texts

    meta = rag_storage.read_meta(doc_id)
    assert meta is not None
    assert meta.doc_id == doc_id
    assert rag_storage.read_fulltext(doc_id) == (
        "Replacement text about a different facade detail."
    )
    assert meta.extra["content_hash"] == result["content_hash"]
    assert meta.extra["updated_at"]

    listed = list_document_versions(storage=rag_storage, doc_id=doc_id)
    assert listed["next_version"] == 2
    assert len(listed["versions"]) == 1
    assert listed["versions"][0]["reason"] == "correct OCR"
    assert listed["versions"][0]["author"] == "tester@example.test"
    assert "text" not in listed["versions"][0]
    with_text = list_document_versions(
        storage=rag_storage, doc_id=doc_id, include_text=True
    )
    assert "Original indexed sentence" in with_text["versions"][0]["text"]
    version_dir = rag_storage.versions_doc_dir(doc_id) / "v0001"
    assert (version_dir / "fulltext.txt").is_file()
    assert (version_dir / "manifest.json").is_file()


def test_replace_identical_content_is_noop(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Identical text should not create a version or rewrite points."""
    original = "Original indexed sentence about a facade."
    doc_id = _ingest_note(
        rag_storage, qdrant_client, fake_embed, data=original.encode("utf-8")
    )
    before = {point.id for point in _chunks(qdrant_client, doc_id)}
    result = replace_document_content(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        content=original,
        reason="retry",
        embed_fn=fake_embed,
    )
    assert result["status"] == "unchanged"
    assert result["content_hash"] == content_sha256(original.encode("utf-8"))
    assert {point.id for point in _chunks(qdrant_client, doc_id)} == before
    listed = list_document_versions(storage=rag_storage, doc_id=doc_id)
    assert listed["versions"] == []


def test_replace_dry_run_does_not_write(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """dry_run should report hashes and counts without mutating storage."""
    original = "Original indexed sentence about a facade."
    doc_id = _ingest_note(
        rag_storage, qdrant_client, fake_embed, data=original.encode("utf-8")
    )
    before_ids = {point.id for point in _chunks(qdrant_client, doc_id)}
    result = replace_document_content(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        content="A brand new replacement paragraph for dry run.",
        reason="preview",
        dry_run=True,
        embed_fn=fake_embed,
    )
    assert result["status"] == "dry_run"
    assert result["next_version"] == 1
    assert result["old_chunk_count"] >= 1
    assert result["new_chunk_count"] >= 1
    assert result["old_content_hash"] != result["new_content_hash"]
    assert rag_storage.read_fulltext(doc_id) == original
    assert {point.id for point in _chunks(qdrant_client, doc_id)} == before_ids
    assert list_document_versions(storage=rag_storage, doc_id=doc_id)["versions"] == []


def test_replace_rollback_on_embed_failure(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Embed failure should keep old points and not leave a version."""
    original = "Original indexed sentence about a facade."
    doc_id = _ingest_note(
        rag_storage, qdrant_client, fake_embed, data=original.encode("utf-8")
    )
    before = _scroll_doc(qdrant_client, doc_id, with_vectors=True)
    before_ids = {point.id for point in before}

    def boom(_text: str) -> list[float]:
        """Simulate ingest embedding failure."""
        raise RuntimeError("embed failed")

    with pytest.raises(RuntimeError, match="embed failed"):
        replace_document_content(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            doc_id=doc_id,
            content="This replacement must not land after embed failure.",
            reason="broken embed",
            embed_fn=boom,
        )
    after = _scroll_doc(qdrant_client, doc_id)
    assert {point.id for point in after} == before_ids
    assert rag_storage.read_fulltext(doc_id) == original
    assert list_document_versions(storage=rag_storage, doc_id=doc_id)["versions"] == []


def test_replace_rollback_on_upsert_failure(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> None:
    """Upsert failure should delete any new points and keep the old index."""
    original = "Original indexed sentence about a facade."
    doc_id = _ingest_note(
        rag_storage, qdrant_client, fake_embed, data=original.encode("utf-8")
    )
    before_ids = {point.id for point in _chunks(qdrant_client, doc_id)}
    real_upsert = qdrant_client.upsert

    def boom(*args: Any, **kwargs: Any) -> None:
        """Fail after Qdrant would have accepted the call."""
        raise RuntimeError("upsert failed")

    monkeypatch.setattr(qdrant_client, "upsert", boom)
    with pytest.raises(RuntimeError, match="upsert failed"):
        replace_document_content(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            doc_id=doc_id,
            content="New text that must roll back when upsert fails.",
            reason="broken upsert",
            embed_fn=fake_embed,
        )
    monkeypatch.setattr(qdrant_client, "upsert", real_upsert)
    after_ids = {point.id for point in _chunks(qdrant_client, doc_id)}
    assert after_ids == before_ids
    assert rag_storage.read_fulltext(doc_id) == original
    assert not rag_storage.versions_doc_dir(doc_id).exists() or (
        list_document_versions(storage=rag_storage, doc_id=doc_id)["versions"] == []
    )


def test_restore_archives_current_then_reapplies_version(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Restore should snapshot live text and re-index the chosen version."""
    original = "Original indexed sentence about a facade."
    doc_id = _ingest_note(
        rag_storage, qdrant_client, fake_embed, data=original.encode("utf-8")
    )
    replace_document_content(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        content="Second body after the first correction.",
        reason="first edit",
        embed_fn=fake_embed,
    )
    restored = restore_document_version(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        version=1,
        reason="roll back the correction",
        author="tester@example.test",
        embed_fn=fake_embed,
    )
    assert restored["status"] == "restored"
    assert restored["restored_version"] == 1
    assert rag_storage.read_fulltext(doc_id) == original
    listed = list_document_versions(
        storage=rag_storage, doc_id=doc_id, include_text=True
    )
    assert [item["version"] for item in listed["versions"]] == [1, 2]
    assert "Second body" in listed["versions"][1]["text"]
    texts = " ".join(
        str((p.payload or {}).get("text") or "") for p in _chunks(qdrant_client, doc_id)
    )
    assert "Original indexed sentence" in texts


def test_restore_missing_version_raises(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Unknown version numbers should raise ``DocumentVersionNotFoundError``."""
    doc_id = _ingest_note(rag_storage, qdrant_client, fake_embed)
    with pytest.raises(DocumentVersionNotFoundError):
        restore_document_version(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            doc_id=doc_id,
            version=9,
            reason="missing",
            embed_fn=fake_embed,
        )


def test_replace_preserves_verification_metadata(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Verification fields, tags, obec_kod, gps, and qg_* stay on new points."""
    doc_id = _ingest_note(
        rag_storage,
        qdrant_client,
        fake_embed,
        metadata={
            "reliability": 2,
            "verification_status": "overeno",
            "obec_kod": 554782,
            "gps": "wikidata",
            "qg_trida": "B",
            "qg_flagy": ["wiki"],
            "qg_verze": "2026-11-12:abcd",
            "content_hash": "oldhash",
            "verification": [
                {
                    "agent": "TesterVerifier",
                    "check": "gps",
                    "tool": "qgate.py",
                    "date": "2026-11-12T12:00:00+01:00",
                    "result": "potvrzeno-wiki",
                    "source_url": "https://example.test/wiki",
                    "quote": "Souřadnice u náměstí.",
                    "independent": False,
                    "source_key": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                }
            ],
        },
    )
    patch_document(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        extra={"tags": ["secese", "praha"]},
    )
    replace_document_content(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        content="Corrected sentence that must keep verification metadata.",
        reason="fix text",
        embed_fn=fake_embed,
    )
    payload = _chunks(qdrant_client, doc_id)[0].payload or {}
    assert payload["reliability"] == 2
    assert payload["verification_status"] == "overeno"
    assert payload["obec_kod"] == 554782
    assert payload["gps"] == "wikidata"
    assert payload["qg_trida"] == "B"
    assert payload["qg_flagy"] == ["wiki"]
    assert payload["qg_verze"] == "2026-11-12:abcd"
    assert payload["verification"][0]["check"] == "gps"
    tags = payload.get("tags") or (payload.get("extra") or {}).get("tags")
    assert tags == ["secese", "praha"]
    assert payload["content_hash"] != "oldhash"
    assert payload["updated_at"]
    meta = rag_storage.read_meta(doc_id)
    assert meta is not None
    assert meta.extra["reliability"] == 2
    assert meta.extra["tags"] == ["secese", "praha"]
    assert meta.extra["content_hash"] == payload["content_hash"]


def test_replace_keeps_pdf_image_points(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """PDF text replace must not drop ``pdf_image`` points or image files."""
    pdf_bytes = build_image_rich_pdf()
    with patch(
        "rag.ingest.ocr_image_file",
        return_value=("indexed ocr phrase long enough", 90.0),
    ):
        ingested = ingest_bytes(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            filename="fixture.pdf",
            data=pdf_bytes,
            embed_fn=fake_embed,
            metadata={"reliability": 3, "qg_trida": "A"},
        )
    doc_id = str(ingested["doc_id"])
    before = _scroll_doc(qdrant_client, doc_id)
    image_ids = {
        point.id
        for point in before
        if (point.payload or {}).get("source_type") == "pdf_image"
    }
    assert image_ids
    replace_document_content(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        content="Replaced PDF fulltext while images stay indexed.",
        reason="fix extract",
        embed_fn=fake_embed,
    )
    after = _scroll_doc(qdrant_client, doc_id)
    after_images = [
        point
        for point in after
        if (point.payload or {}).get("source_type") == "pdf_image"
    ]
    assert {point.id for point in after_images} == image_ids
    for point in after_images:
        payload = point.payload or {}
        assert payload["content_hash"]
        assert payload["updated_at"]
        assert payload.get("reliability") == 3
    assert rag_storage.images_dir(doc_id).is_dir()
    chunk_texts = " ".join(
        str((p.payload or {}).get("text") or "")
        for p in after
        if (p.payload or {}).get("source_type") == "pdf_chunk"
    )
    assert "Replaced PDF fulltext" in chunk_texts


def test_replace_unknown_doc_raises(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Missing ids should raise ``DocumentNotFoundError``."""
    missing = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(DocumentNotFoundError):
        replace_document_content(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            doc_id=missing,
            content="nope",
            reason="missing",
            embed_fn=fake_embed,
        )


def test_replace_requires_reason(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Empty reason should be rejected before any write."""
    doc_id = _ingest_note(rag_storage, qdrant_client, fake_embed)
    with pytest.raises(DocumentReplaceError, match="reason"):
        replace_document_content(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            doc_id=doc_id,
            content="changed",
            reason="   ",
            embed_fn=fake_embed,
        )


def test_delete_document_removes_versions(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Deleting a document should drop its version history directory."""
    doc_id = _ingest_note(rag_storage, qdrant_client, fake_embed)
    replace_document_content(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        content="Edited before delete.",
        reason="edit",
        embed_fn=fake_embed,
    )
    assert rag_storage.versions_doc_dir(doc_id).is_dir()
    delete_document(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
    )
    assert not rag_storage.versions_doc_dir(doc_id).exists()


def test_http_replace_versions_restore_and_dry_run(client: TestClient) -> None:
    """HTTP PUT/GET/POST should replace, list versions, restore, and dry-run."""
    upload = client.post(
        "/upload",
        files={"file": ("note.txt", b"First body of the stored note.", "text/plain")},
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]

    missing_reason = client.put(
        f"/documents/{doc_id}/content",
        json={"content": "No reason here."},
    )
    assert missing_reason.status_code == 400

    dry = client.put(
        f"/documents/{doc_id}/content",
        json={
            "content": "Previewed replacement for the stored note.",
            "reason": "preview",
            "dry_run": True,
        },
    )
    assert dry.status_code == 200
    assert dry.json()["status"] == "dry_run"
    assert dry.json()["next_version"] == 1
    assert client.get(f"/documents/{doc_id}/text").json()["text"] == (
        "First body of the stored note."
    )

    replaced = client.put(
        f"/documents/{doc_id}/content",
        json={
            "content": "Second body of the stored note after edit.",
            "reason": "fix wording",
        },
    )
    assert replaced.status_code == 200
    assert replaced.json()["status"] == "replaced"
    assert replaced.json()["version"] == 1
    assert "Second body" in client.get(f"/documents/{doc_id}/text").json()["text"]

    identical = client.put(
        f"/documents/{doc_id}/content",
        json={
            "content": "Second body of the stored note after edit.",
            "reason": "retry",
        },
    )
    assert identical.status_code == 200
    assert identical.json()["status"] == "unchanged"

    versions = client.get(f"/documents/{doc_id}/versions")
    assert versions.status_code == 200
    body = versions.json()
    assert len(body["versions"]) == 1
    assert "text" not in body["versions"][0]
    with_text = client.get(
        f"/documents/{doc_id}/versions", params={"include_text": True}
    )
    assert "First body" in with_text.json()["versions"][0]["text"]

    restored = client.post(
        f"/documents/{doc_id}/versions/1/restore",
        json={"reason": "undo edit"},
    )
    assert restored.status_code == 200
    assert restored.json()["status"] == "restored"
    assert "First body" in client.get(f"/documents/{doc_id}/text").json()["text"]

    unknown = client.put(
        "/documents/00000000-0000-0000-0000-000000000000/content",
        json={"content": "x", "reason": "gone"},
    )
    assert unknown.status_code == 404


def test_http_replace_requires_bearer(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """PUT content, list versions, and restore should reject missing tokens."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as unauth:
            put = unauth.put(
                "/documents/00000000-0000-0000-0000-000000000000/content",
                json={"content": "x", "reason": "nope"},
            )
            listed = unauth.get(
                "/documents/00000000-0000-0000-0000-000000000000/versions"
            )
            restore = unauth.post(
                "/documents/00000000-0000-0000-0000-000000000000/versions/1/restore",
                json={"reason": "nope"},
            )
            assert put.status_code == 401
            assert listed.status_code == 401
            assert restore.status_code == 401


def test_openapi_documents_content_and_versions(client: TestClient) -> None:
    """OpenAPI should advertise replace, versions list, and restore routes."""
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    payload = spec.json()
    paths = payload["paths"]
    assert "put" in paths["/documents/{doc_id}/content"]
    assert "get" in paths["/documents/{doc_id}/versions"]
    assert "post" in paths["/documents/{doc_id}/versions/{version}/restore"]
    put_op = paths["/documents/{doc_id}/content"]["put"]
    assert "ReplaceDocumentContentRequest" in payload["components"]["schemas"]
    assert put_op["requestBody"]
    restore_schema = payload["components"]["schemas"]["RestoreDocumentVersionRequest"]
    assert "reason" in restore_schema.get("properties", {})


def _two_chunk_text() -> str:
    """Return text long enough to produce at least two ingest chunks."""
    return ("Praha secese facade. " * 80) + "\n" + ("Brno funkcionalismus. " * 80)


def _sample_geo() -> dict[str, object]:
    """Return a valid ``geo_refined`` object for refine-after-replace tests."""
    return {
        "lat": 49.1951,
        "lon": 16.6080,
        "precision": "building",
        "precision_m": 15.0,
        "confidence": 0.86,
        "confidence_level": "high",
        "status": "zpresneno",
        "sources": [
            {
                "type": "wikidata_p625",
                "url": "https://www.wikidata.org/wiki/Q123",
                "distance_m": 12.4,
            }
        ],
        "method": "wikidata_p625",
        "distance_m": 6120.0,
        "verified": False,
        "refined_by": "gps-review",
        "note": "Deutsches Haus",
    }


def test_replace_drops_differing_per_chunk_fields(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Per-chunk keys that differ must not smear from the first old chunk."""
    doc_id = _ingest_note(
        rag_storage,
        qdrant_client,
        fake_embed,
        data=_two_chunk_text().encode("utf-8"),
        metadata={"reliability": 2, "obec_kod": 554782},
    )
    points = _chunks(qdrant_client, doc_id)
    assert len(points) >= 2
    for index, point in enumerate(points):
        qdrant_client.set_payload(
            collection_name=COLLECTION,
            payload={
                "chunk_note": f"note-{index}",
                "shared_flag": "keep-me",
            },
            points=[point.id],
            wait=True,
        )

    result = replace_document_content(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        content="Short replacement that must not inherit per-chunk notes.",
        reason="fix smear",
        embed_fn=fake_embed,
    )
    assert result["status"] == "replaced"
    assert "chunk_note" in result["dropped_chunk_fields"]
    assert "shared_flag" not in result["dropped_chunk_fields"]
    new_points = _chunks(qdrant_client, doc_id)
    assert new_points
    for point in new_points:
        payload = point.payload or {}
        assert payload.get("chunk_note") is None
        assert payload.get("shared_flag") == "keep-me"
        assert payload.get("reliability") == 2
        assert payload.get("obec_kod") == 554782
        assert payload.get("content_hash") == result["content_hash"]


def test_replace_and_metadata_patch_share_doc_lock(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Content replace and metadata PATCH should wait on the same per-doc lock."""
    doc_id = _ingest_note(rag_storage, qdrant_client, fake_embed)
    started = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    finished = threading.Event()

    def holder() -> None:
        """Hold the document lock until ``release`` is set."""
        with doc_lock(doc_id):
            started.set()
            if not release.wait(timeout=5):
                errors.append(TimeoutError("holder was not released"))

    def do_replace() -> None:
        """Replace while the lock is held by another thread."""
        try:
            replace_document_content(
                storage=rag_storage,
                qdrant=qdrant_client,
                collection_name=COLLECTION,
                doc_id=doc_id,
                content="Replacement that must wait for the shared lock.",
                reason="lock test",
                embed_fn=fake_embed,
            )
        except BaseException as exc:  # noqa: BLE001 - collect for the assertion
            errors.append(exc)
        finally:
            finished.set()

    hold_thread = threading.Thread(target=holder)
    hold_thread.start()
    assert started.wait(timeout=5)
    worker = threading.Thread(target=do_replace)
    worker.start()
    assert not finished.wait(timeout=0.4)
    release.set()
    assert finished.wait(timeout=10)
    hold_thread.join(timeout=5)
    worker.join(timeout=5)
    assert errors == []
    assert rag_storage.read_fulltext(doc_id) == (
        "Replacement that must wait for the shared lock."
    )

    started_meta = threading.Event()
    release_meta = threading.Event()
    finished_meta = threading.Event()

    def hold_again() -> None:
        """Hold the lock while metadata PATCH is attempted."""
        with doc_lock(doc_id):
            started_meta.set()
            if not release_meta.wait(timeout=5):
                errors.append(TimeoutError("metadata holder was not released"))

    def do_patch() -> None:
        """Patch verification metadata while the lock is held."""
        try:
            patch_document_metadata(
                storage=rag_storage,
                qdrant=qdrant_client,
                collection_name=COLLECTION,
                doc_id=doc_id,
                metadata={"reliability": 3},
            )
        except BaseException as exc:  # noqa: BLE001 - collect for the assertion
            errors.append(exc)
        finally:
            finished_meta.set()

    hold_meta = threading.Thread(target=hold_again)
    hold_meta.start()
    assert started_meta.wait(timeout=5)
    patch_thread = threading.Thread(target=do_patch)
    patch_thread.start()
    assert not finished_meta.wait(timeout=0.4)
    release_meta.set()
    assert finished_meta.wait(timeout=10)
    hold_meta.join(timeout=5)
    patch_thread.join(timeout=5)
    assert errors == []
    meta = rag_storage.read_meta(doc_id)
    assert meta is not None
    assert meta.extra["reliability"] == 3


def test_replace_cleanup_pending_when_old_delete_fails(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    """Old-point delete failure after commit should return leftover ids, not 500."""
    original = "Original indexed sentence about a facade."
    doc_id = _ingest_note(
        rag_storage, qdrant_client, fake_embed, data=original.encode("utf-8")
    )
    old_ids = {str(point.id) for point in _chunks(qdrant_client, doc_id)}
    assert old_ids

    def boom(*_args: Any, **_kwargs: Any) -> None:
        """Simulate Qdrant delete failure for leftover cleanup."""
        raise RuntimeError("delete failed")

    monkeypatch.setattr(qdrant_client, "delete", boom)
    caplog.set_level("ERROR", logger="rag.document_replace")
    result = replace_document_content(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        content="New body that stays live when old delete fails.",
        reason="cleanup pending",
        embed_fn=fake_embed,
    )
    assert result["status"] == "replaced"
    assert result["cleanup_pending"] is True
    assert result["leftover_old_point_count"] == len(old_ids)
    assert set(result["leftover_old_point_ids"]) == old_ids
    assert result["chunks_removed"] == 0
    after_ids = {str(point.id) for point in _chunks(qdrant_client, doc_id)}
    assert old_ids <= after_ids
    assert len(after_ids) > len(old_ids)
    assert rag_storage.read_fulltext(doc_id) == (
        "New body that stays live when old delete fails."
    )
    assert any("Leftover old chunk points" in rec.message for rec in caplog.records)


def test_replace_reports_locality_links_without_deleting_them(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Replace must count locality_links and leave those rows in place."""
    doc_id = _ingest_note(rag_storage, qdrant_client, fake_embed)
    upsert_locality_links(
        qdrant_client,
        [
            {
                "obec_id": "554782",
                "name": "Praha",
                "doc_id": doc_id,
                "chunk_index": 0,
                "matched_text": "Praha",
                "char_start": 0,
                "char_end": 5,
                "source_type": "pdf_chunk",
                "method": "exact",
            }
        ],
    )
    before = int(
        qdrant_client.count(
            collection_name=LOCALITY_LINKS_COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            exact=True,
        ).count
    )
    assert before >= 1
    result = replace_document_content(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        content="Replacement that must keep locality_links rows.",
        reason="keep links",
        embed_fn=fake_embed,
    )
    assert result["affected_locality_links"] == before
    after = int(
        qdrant_client.count(
            collection_name=LOCALITY_LINKS_COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            exact=True,
        ).count
    )
    assert after == before
    ingest_id = stable_point_id(doc_id, 0)
    assert not qdrant_client.retrieve(
        collection_name=COLLECTION, ids=[ingest_id], with_payload=False
    )
    updated = denormalize_knowledge_localities(
        qdrant_client,
        [{"doc_id": doc_id, "chunk_index": 0, "obec_id": "554782"}],
    )
    assert updated == 1
    payload = _chunks(qdrant_client, doc_id)[0].payload or {}
    assert "554782" in (payload.get("localities") or [])
    refined = refine_geo(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        doc_id=doc_id,
        chunk_index=0,
        geo_refined=_sample_geo(),
        dry_run=False,
    )
    assert refined["written"] is True
    live = _chunks(qdrant_client, doc_id)[0].payload or {}
    assert live["geo_refined"]["lat"] == pytest.approx(49.1951)
