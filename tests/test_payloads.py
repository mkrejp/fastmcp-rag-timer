"""Tests for Qdrant payload builders."""

from __future__ import annotations

import uuid

import pytest

from rag.payloads import (
    LocalityLinkPayloadError,
    build_locality_link_payload,
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
