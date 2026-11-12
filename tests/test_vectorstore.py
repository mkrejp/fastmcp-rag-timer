"""Tests for Qdrant named-vector initialization."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from rag.config import LOCALITY_LINKS_COLLECTION
from rag.embeddings import EMBEDDING_DIM, FASTEMBED_MODEL
from rag.image_embed import IMAGE_EMBEDDING_DIM
from rag.ingest import stable_point_id
from rag.payloads import (
    LocalityLinkNotFoundError,
    LocalityLinkPayloadError,
    chunk_text_sha256,
)
from rag.vectorstore import (
    BACKEND_MARKER_NAME,
    IMAGE_VECTOR_NAME,
    MARKER_SCHEMA_VERSION,
    TEXT_VECTOR_NAME,
    backfill_chunk_payload_fields,
    build_named_vector,
    ensure_locality_links_collection,
    ensure_payload_indexes,
    knowledge_locality_filter,
    list_locality_links,
    locality_link_stats,
    open_qdrant,
    search_similar,
    upsert_locality_links,
    verify_locality_link,
)


def _marker_payload(qdrant_dir: Path) -> dict[str, object]:
    """Read the embedding backend marker written next to ``qdrant_dir``."""
    path = qdrant_dir.parent / BACKEND_MARKER_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def _named_collection(client: QdrantClient) -> object:
    """Return collection info after asserting named text_vector size."""
    info = client.get_collection("client_domain_knowledge")
    vectors = info.config.params.vectors
    assert isinstance(vectors, dict)
    assert vectors[TEXT_VECTOR_NAME].size == EMBEDDING_DIM
    return info


def test_open_qdrant_creates_named_vectors(tmp_path: Path) -> None:
    """A new collection exposes text_vector and image_vector with expected sizes."""
    qdrant_dir = tmp_path / "qdrant"
    client = open_qdrant(qdrant_dir)
    try:
        info = _named_collection(client)
        vectors = info.config.params.vectors
        assert vectors[IMAGE_VECTOR_NAME].size == IMAGE_EMBEDDING_DIM
        marker = _marker_payload(qdrant_dir)
        assert marker["schema_version"] == MARKER_SCHEMA_VERSION
        assert marker["text_dim"] == EMBEDDING_DIM
        assert marker["image_dim"] == IMAGE_EMBEDDING_DIM
    finally:
        client.close()


def test_open_qdrant_recreates_legacy_single_vector_collection(tmp_path: Path) -> None:
    """A persisted single-vector collection is replaced with named vectors."""
    qdrant_dir = tmp_path / "qdrant"
    old_client = QdrantClient(path=str(qdrant_dir))
    old_client.create_collection(
        collection_name="client_domain_knowledge",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    old_client.close()

    client = open_qdrant(qdrant_dir)
    try:
        _named_collection(client)
    finally:
        client.close()


def test_open_qdrant_keeps_matching_named_collection(tmp_path: Path) -> None:
    """An existing named-vector collection with the expected schema is kept."""
    qdrant_dir = tmp_path / "qdrant"
    old_client = QdrantClient(path=str(qdrant_dir))
    old_client.create_collection(
        collection_name="client_domain_knowledge",
        vectors_config={
            TEXT_VECTOR_NAME: VectorParams(
                size=EMBEDDING_DIM, distance=Distance.COSINE
            ),
            IMAGE_VECTOR_NAME: VectorParams(
                size=IMAGE_EMBEDDING_DIM, distance=Distance.COSINE
            ),
        },
    )
    old_client.close()

    client = open_qdrant(qdrant_dir)
    try:
        _named_collection(client)
    finally:
        client.close()


def test_open_qdrant_allows_ingest_provider_switch_without_recreate(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Switching ingest novita→fastembed keeps the named collection (same BGE-M3 dim)."""
    monkeypatch.setenv("EMBEDDING_PROVIDER_QUERY", "fastembed")
    qdrant_dir = tmp_path / "qdrant"
    first = open_qdrant(qdrant_dir)
    first.close()
    assert _marker_payload(qdrant_dir)["ingest_provider"] == "novita"

    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    client = open_qdrant(qdrant_dir)
    try:
        info = client.get_collection("client_domain_knowledge")
        vectors = info.config.params.vectors
        assert isinstance(vectors, dict)
        assert _marker_payload(qdrant_dir)["ingest_provider"] == "fastembed"
        assert _marker_payload(qdrant_dir)["ingest_model"] == FASTEMBED_MODEL
        assert info.points_count == 0
    finally:
        client.close()


def test_ensure_payload_indexes_creates_verification_filters() -> None:
    """Startup should index source_key, verification_status, and reliability."""
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value.payload_schema = {}
    ensure_payload_indexes(client, collection_name="client_domain_knowledge")
    created = {
        (call.kwargs["field_name"], call.kwargs["field_schema"])
        for call in client.create_payload_index.call_args_list
    }
    assert ("source_key", PayloadSchemaType.KEYWORD) in created
    assert ("verification_status", PayloadSchemaType.KEYWORD) in created
    assert ("reliability", PayloadSchemaType.INTEGER) in created
    client.delete_collection.assert_not_called()
    client.upsert.assert_not_called()
    client.delete.assert_not_called()


def test_ensure_payload_indexes_skips_existing() -> None:
    """Existing payload indexes must not be recreated or cause deletes."""
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value.payload_schema = {
        "source_key": MagicMock(data_type=PayloadSchemaType.KEYWORD),
        "verification_status": MagicMock(data_type=PayloadSchemaType.KEYWORD),
        "reliability": MagicMock(data_type=PayloadSchemaType.INTEGER),
        "doc_id": MagicMock(data_type=PayloadSchemaType.KEYWORD),
    }
    ensure_payload_indexes(client, collection_name="client_domain_knowledge")
    created = {
        call.kwargs["field_name"] for call in client.create_payload_index.call_args_list
    }
    assert "source_key" not in created
    assert "verification_status" not in created
    assert "reliability" not in created
    assert "doc_id" not in created
    client.delete_collection.assert_not_called()


def _sample_link(**overrides: object) -> dict[str, object]:
    """Return a valid locality-link mapping for vector-store tests."""
    payload: dict[str, object] = {
        "obec_id": "554782",
        "name": "Praha",
        "doc_id": "doc-1",
        "chunk_index": 0,
        "matched_text": "Praha",
        "char_start": 12,
        "char_end": 17,
        "source_type": "pdf_chunk",
        "method": "exact",
    }
    payload.update(overrides)
    return payload


def test_open_qdrant_creates_payload_only_locality_links(tmp_path: Path) -> None:
    """Startup should create locality_links with no vectors, without embeddings."""
    qdrant_dir = tmp_path / "qdrant"
    client = open_qdrant(qdrant_dir)
    try:
        assert client.collection_exists(LOCALITY_LINKS_COLLECTION)
        info = client.get_collection(LOCALITY_LINKS_COLLECTION)
        assert info.config.params.vectors == {}
        ensure_locality_links_collection(client)
        assert client.get_collection(LOCALITY_LINKS_COLLECTION).points_count == 0
    finally:
        client.close()


def test_open_qdrant_does_not_delete_existing_locality_links(
    tmp_path: Path,
) -> None:
    """Recreating the knowledge collection must not drop locality_links points."""
    qdrant_dir = tmp_path / "qdrant"
    first = open_qdrant(qdrant_dir)
    upsert_locality_links(first, [_sample_link()])
    first.close()

    old_client = QdrantClient(path=str(qdrant_dir))
    old_client.delete_collection("client_domain_knowledge")
    old_client.create_collection(
        collection_name="client_domain_knowledge",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    old_client.close()

    client = open_qdrant(qdrant_dir)
    try:
        listed = list_locality_links(client)
        assert listed["returned"] == 1
        assert listed["links"][0]["obec_id"] == "554782"
        knowledge = client.get_collection("client_domain_knowledge")
        vectors = knowledge.config.params.vectors
        assert isinstance(vectors, dict)
        assert vectors[TEXT_VECTOR_NAME].size == EMBEDDING_DIM
    finally:
        client.close()


def test_upsert_locality_links_is_idempotent_on_unique_key(tmp_path: Path) -> None:
    """The same unique key should update in place rather than insert a second point."""
    qdrant_dir = tmp_path / "qdrant"
    client = open_qdrant(qdrant_dir)
    try:
        first = upsert_locality_links(client, [_sample_link(notes="one")])
        second = upsert_locality_links(
            client, [_sample_link(notes="two", method="lemma")]
        )
        assert first["inserted"] == 1
        assert first["updated"] == 0
        assert second["inserted"] == 0
        assert second["updated"] == 1
        assert first["ids"] == second["ids"]
        listed = list_locality_links(client, doc_id="doc-1")
        assert listed["returned"] == 1
        assert listed["links"][0]["notes"] == "two"
        assert listed["links"][0]["method"] == "lemma"
        other = upsert_locality_links(
            client, [_sample_link(obec_id="582786", name="Brno", char_start=40)]
        )
        assert other["inserted"] == 1
        all_links = list_locality_links(client, limit=10)
        assert all_links["returned"] == 2
        page = list_locality_links(client, limit=1, offset=0)
        assert page["returned"] == 1
        assert page["has_more"] is True
        rest = list_locality_links(client, limit=1, offset=1)
        assert rest["returned"] == 1
        assert rest["has_more"] is False
        filtered = list_locality_links(client, obec_id="582786")
        assert filtered["returned"] == 1
        assert filtered["links"][0]["name"] == "Brno"
        upsert_locality_links(
            client,
            [_sample_link(verification_status="verified", char_start=99)],
        )
        verified = list_locality_links(client, status="verified")
        assert verified["returned"] == 1
        assert verified["links"][0]["char_start"] == 99
        unverified = list_locality_links(client, status="unverified", doc_id="doc-1")
        assert unverified["returned"] == 2
    finally:
        client.close()


def test_verify_and_stats_and_cast_obce_filters(tmp_path: Path) -> None:
    """verify_link, stats, and 005 locality_level/cast_obce filters should work."""
    qdrant_dir = tmp_path / "qdrant"
    client = open_qdrant(qdrant_dir)
    try:
        praha = upsert_locality_links(client, [_sample_link()])
        karlin = upsert_locality_links(
            client,
            [
                _sample_link(
                    locality_level="cast_obce",
                    cast_obce="400637",
                    cast_obce_name="Karlín",
                    matched_text="Karlín",
                    ambiguous=True,
                )
            ],
        )
        brno = upsert_locality_links(
            client,
            [_sample_link(obec_id="582786", name="Brno", char_start=40)],
        )
        assert praha["ids"][0] != karlin["ids"][0]
        verified = verify_locality_link(
            client,
            link_id=praha["ids"][0],
            status="verified",
            verified_by="reviewer",
            notes="looks good",
        )
        assert verified["link"]["verification_status"] == "verified"
        assert verified["link"]["verified"] is True
        assert verified["link"]["verified_by"] == "reviewer"
        assert verified["link"]["verified_at"]
        verify_locality_link(client, link_id=brno["ids"][0], status="rejected")
        missing = "00000000-0000-0000-0000-000000000000"
        with pytest.raises(LocalityLinkNotFoundError, match="not found"):
            verify_locality_link(client, link_id=missing, status="verified")
        with pytest.raises(LocalityLinkPayloadError, match="status"):
            verify_locality_link(client, link_id=praha["ids"][0], status="maybe")

        by_level = list_locality_links(client, locality_level="cast_obce")
        assert by_level["returned"] == 1
        assert by_level["links"][0]["cast_obce_kod"] == "400637"
        by_cast = list_locality_links(client, cast_obce="400637")
        assert by_cast["returned"] == 1
        assert by_cast["links"][0]["cast_obce_name"] == "Karlín"

        stats = locality_link_stats(client)
        assert stats["total"] == 3
        assert stats["by_status"]["verified"] == 1
        assert stats["by_status"]["rejected"] == 1
        assert stats["by_status"]["unverified"] == 1
        assert stats["by_locality_level"]["obec"] == 2
        assert stats["by_locality_level"]["cast_obce"] == 1
        assert stats["ambiguous"]["true"] == 1
        assert stats["docs_covered"] == 1
        assert stats["top_obce"][0]["obec_id"] == "554782"
        assert stats["top_obce"][0]["count"] == 2
    finally:
        client.close()


def test_backfill_chunk_payload_fields_is_payload_only(tmp_path: Path) -> None:
    """Existing points missing new fields should be patched without re-embedding."""
    qdrant_dir = tmp_path / "qdrant"
    client = open_qdrant(qdrant_dir)
    try:
        text = "legacy chunk without locality fields"
        original_vector = [0.25] * EMBEDDING_DIM
        point_id = "11111111-1111-4111-8111-111111111111"
        client.upsert(
            collection_name="client_domain_knowledge",
            points=[
                PointStruct(
                    id=point_id,
                    vector=build_named_vector(text_vector=original_vector),
                    payload={
                        "doc_id": "legacy-doc",
                        "text": text,
                        "sha256": "doc-hash",
                        "source_type": "pdf_chunk",
                        "chunk_index": 0,
                    },
                )
            ],
            wait=True,
        )
        before = client.retrieve(
            collection_name="client_domain_knowledge",
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
        first = backfill_chunk_payload_fields(client)
        assert first["scanned"] == 1
        assert first["updated"] == 1
        after = client.retrieve(
            collection_name="client_domain_knowledge",
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
        payload = after[0].payload or {}
        assert payload["localities"] == []
        assert payload["cast_obce"] == []
        assert payload["chunk_sha256"] == chunk_text_sha256(text)
        assert payload["sha256"] == "doc-hash"
        assert after[0].vector == before[0].vector
        second = backfill_chunk_payload_fields(client)
        assert second["updated"] == 0
        count = client.count("client_domain_knowledge")
        assert count.count == 1
    finally:
        client.close()


def test_link_locality_denormalizes_knowledge_payload(tmp_path: Path) -> None:
    """link upsert should merge obec_id onto the matching chunk via set_payload."""
    qdrant_dir = tmp_path / "qdrant"
    client = open_qdrant(qdrant_dir)
    try:
        doc_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        point_id = stable_point_id(doc_id, 0)
        text = "Praha timer mention"
        original_vector = [0.1] * EMBEDDING_DIM
        client.upsert(
            collection_name="client_domain_knowledge",
            points=[
                PointStruct(
                    id=point_id,
                    vector=build_named_vector(text_vector=original_vector),
                    payload={
                        "doc_id": doc_id,
                        "text": text,
                        "sha256": "doc-hash",
                        "chunk_sha256": chunk_text_sha256(text),
                        "source_type": "pdf_chunk",
                        "chunk_index": 0,
                        "localities": [],
                        "cast_obce": [],
                    },
                )
            ],
            wait=True,
        )
        before = client.retrieve(
            collection_name="client_domain_knowledge",
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
        result = upsert_locality_links(
            client,
            [
                _sample_link(
                    doc_id=doc_id,
                    locality_level="cast_obce",
                    cast_obce="400637",
                    cast_obce_name="Karlín",
                    matched_text="Karlín",
                )
            ],
        )
        assert result["chunks_updated"] == 1
        after = client.retrieve(
            collection_name="client_domain_knowledge",
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
        payload = after[0].payload or {}
        assert payload["localities"] == ["554782"]
        assert payload["cast_obce"] == ["400637"]
        assert after[0].vector == before[0].vector
        hits = search_similar(
            client,
            query_vector=original_vector,
            limit=3,
            query_filter=knowledge_locality_filter(obec_id="554782"),
        )
        assert [str(hit.id) for hit in hits] == [point_id]
        none = search_similar(
            client,
            query_vector=original_vector,
            limit=3,
            query_filter=knowledge_locality_filter(obec_id="582786"),
        )
        assert none == []
        part = search_similar(
            client,
            query_vector=original_vector,
            limit=3,
            query_filter=knowledge_locality_filter(cast_obce="400637"),
        )
        assert [str(hit.id) for hit in part] == [point_id]
        count = client.count("client_domain_knowledge")
        assert count.count == 1
    finally:
        client.close()
