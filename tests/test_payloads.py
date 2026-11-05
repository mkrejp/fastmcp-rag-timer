"""Tests for Qdrant payload builders."""

from __future__ import annotations

import uuid

import pytest

from rag.payloads import (
    LocalityLinkPayloadError,
    apply_locality_link_verification,
    build_chunk_payload,
    build_image_payload,
    build_locality_link_payload,
    chunk_text_sha256,
    knowledge_point_id_for_link,
    stable_locality_link_id,
)


def _sample_link(**overrides: object) -> dict[str, object]:
    """Return a valid locality-link mapping with optional field overrides."""
    payload: dict[str, object] = {
        "obec_id": "554782",
        "name": "Praha",
        "okres": "Praha",
        "kraj": "Hlavní město Praha",
        "lau_id": "CZ0100554782",
        "doc_id": "doc-1",
        "chunk_index": 0,
        "matched_text": "Praha",
        "char_start": 12,
        "char_end": 17,
        "source_type": "pdf_chunk",
        "method": "exact",
        "chunk_sha256": "abc123",
    }
    payload.update(overrides)
    return payload


def test_chunk_payload_includes_sha256_and_empty_localities() -> None:
    """New chunk payloads should hash text and start with empty locality lists."""
    payload = build_chunk_payload(
        doc_id="d",
        parent_document="f.pdf",
        page_number=1,
        doc_sha256="doc-hash",
        text="hello chunk",
        chunk_index=0,
    )
    assert payload["sha256"] == "doc-hash"
    assert payload["chunk_sha256"] == chunk_text_sha256("hello chunk")
    assert payload["chunk_sha256"] != payload["sha256"]
    assert payload["localities"] == []
    assert payload["cast_obce"] == []
    with_links = build_chunk_payload(
        doc_id="d",
        parent_document="f.pdf",
        page_number=1,
        doc_sha256="doc-hash",
        text="hello chunk",
        chunk_index=0,
        localities=["554782"],
        cast_obce=["400637"],
    )
    assert with_links["localities"] == ["554782"]
    assert with_links["cast_obce"] == ["400637"]


def test_image_payload_hashes_content_summary() -> None:
    """Image payloads should hash content_summary, not the document hash."""
    payload = build_image_payload(
        doc_id="d",
        parent_document="f.pdf",
        page_number=1,
        file_path="processed/d/images/ab.png",
        image_id="abcd1234abcd1234",
        image_sha256="image-hash",
        content_summary="OCR text",
    )
    assert payload["sha256"] == "image-hash"
    assert payload["chunk_sha256"] == chunk_text_sha256("OCR text")
    assert payload["localities"] == []
    assert payload["cast_obce"] == []


def test_knowledge_point_id_for_link_matches_ingest_formula() -> None:
    """Link denormalization should reuse ingest's uuid5(doc_id:chunk_index)."""
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, "doc-1:0"))
    assert (
        knowledge_point_id_for_link({"doc_id": "doc-1", "chunk_index": 0})
        == expected
    )
    assert (
        knowledge_point_id_for_link(
            {"doc_id": "doc-1", "chunk_index": 0, "point_id": "explicit-id"}
        )
        == "explicit-id"
    )
    assert knowledge_point_id_for_link({"doc_id": "doc-1"}) is None


def test_stable_locality_link_id_is_uuid5_and_stable() -> None:
    """The unique key should hash to the same UUID every time."""
    first = stable_locality_link_id(
        doc_id="doc-1", chunk_index=0, obec_id="554782", char_start=12
    )
    second = stable_locality_link_id(
        doc_id="doc-1", chunk_index=0, obec_id="554782", char_start=12
    )
    assert first == second
    uuid.UUID(first)
    other = stable_locality_link_id(
        doc_id="doc-1", chunk_index=0, obec_id="554782", char_start=13
    )
    assert other != first


def test_build_locality_link_payload_defaults_and_required_fields() -> None:
    """Builder should fill PostGIS-compatible defaults and require core fields."""
    link_id, payload = build_locality_link_payload(
        _sample_link(), now="2026-11-05T12:00:00+00:00"
    )
    assert payload["link_id"] == link_id
    assert payload["link_type"] == "mentions_locality"
    assert payload["source_dataset"] == "rag-timer"
    assert payload["verification_status"] == "unverified"
    assert payload["verified"] is False
    assert payload["ambiguous"] is False
    assert payload["candidates"] == []
    assert payload["chunk_index"] == 0
    assert payload["lau_id"] == "CZ0100554782"
    assert payload["locality_level"] == "obec"
    assert payload["cast_obce"] is None
    assert payload["cast_obce_kod"] is None
    assert payload["created_at"] == "2026-11-05T12:00:00+00:00"
    assert payload["updated_at"] == "2026-11-05T12:00:00+00:00"
    with pytest.raises(LocalityLinkPayloadError, match="obec_id"):
        build_locality_link_payload(_sample_link(obec_id=""))
    with pytest.raises(LocalityLinkPayloadError, match="source_type"):
        build_locality_link_payload(_sample_link(source_type="web_page"))
    with pytest.raises(LocalityLinkPayloadError, match="method"):
        build_locality_link_payload(_sample_link(method="regex"))


def test_build_locality_link_payload_preserves_created_at_on_update() -> None:
    """Rebuilding an existing point should keep created_at and refresh updated_at."""
    _link_id, first = build_locality_link_payload(
        _sample_link(), now="2026-11-05T12:00:00+00:00"
    )
    _link_id, second = build_locality_link_payload(
        _sample_link(notes="reviewed"),
        existing=first,
        now="2026-11-05T12:05:00+00:00",
    )
    assert second["created_at"] == "2026-11-05T12:00:00+00:00"
    assert second["updated_at"] == "2026-11-05T12:05:00+00:00"
    assert second["notes"] == "reviewed"


def test_build_locality_link_payload_verified_follows_status() -> None:
    """verification_status=verified should force verified true."""
    _link_id, payload = build_locality_link_payload(
        _sample_link(verification_status="verified", verified=False)
    )
    assert payload["verified"] is True
    assert payload["verification_status"] == "verified"


def test_obec_level_link_id_matches_pre_cast_obce_formula() -> None:
    """Obec-level ids must stay stable so existing points are not duplicated."""
    expected = str(
        uuid.uuid5(uuid.NAMESPACE_URL, "locality_link:doc-1:0:554782:12")
    )
    assert (
        stable_locality_link_id(
            doc_id="doc-1", chunk_index=0, obec_id="554782", char_start=12
        )
        == expected
    )


def test_cast_obce_payload_and_distinct_link_id() -> None:
    """Karlín (část obce) should store 005 fields and a different point id."""
    link_id, payload = build_locality_link_payload(
        _sample_link(
            locality_level="cast_obce",
            cast_obce="400637",
            cast_obce_name="Karlín",
            matched_text="Karlín",
        )
    )
    obec_id, obec_payload = build_locality_link_payload(_sample_link())
    assert payload["locality_level"] == "cast_obce"
    assert payload["cast_obce"] == "400637"
    assert payload["cast_obce_kod"] == "400637"
    assert payload["cast_obce_name"] == "Karlín"
    assert link_id != obec_id
    with pytest.raises(LocalityLinkPayloadError, match="cast_obce_kod"):
        build_locality_link_payload(
            _sample_link(locality_level="cast_obce")
        )
    with pytest.raises(LocalityLinkPayloadError, match="empty"):
        build_locality_link_payload(
            _sample_link(locality_level="obec", cast_obce_kod="400637")
        )


def test_apply_locality_link_verification_state_machine() -> None:
    """verified/rejected/unverified should set verified, verified_at, and notes."""
    _link_id, payload = build_locality_link_payload(
        _sample_link(), now="2026-11-05T12:00:00+00:00"
    )
    verified = apply_locality_link_verification(
        payload,
        status="verified",
        verified_by="agent",
        notes="ok",
        now="2026-11-05T12:10:00+00:00",
    )
    assert verified["verification_status"] == "verified"
    assert verified["verified"] is True
    assert verified["verified_by"] == "agent"
    assert verified["verified_at"] == "2026-11-05T12:10:00+00:00"
    assert verified["notes"] == "ok"
    assert verified["created_at"] == payload["created_at"]
    rejected = apply_locality_link_verification(
        verified, status="rejected", now="2026-11-05T12:11:00+00:00"
    )
    assert rejected["verification_status"] == "rejected"
    assert rejected["verified"] is False
    assert rejected["verified_at"] == "2026-11-05T12:11:00+00:00"
    assert rejected["verified_by"] == "agent"
    reset = apply_locality_link_verification(
        rejected, status="unverified", now="2026-11-05T12:12:00+00:00"
    )
    assert reset["verification_status"] == "unverified"
    assert reset["verified"] is False
    assert reset["verified_at"] is None
    with pytest.raises(LocalityLinkPayloadError, match="status"):
        apply_locality_link_verification(payload, status="maybe")
