"""Unit tests for verification metadata validation."""

from __future__ import annotations

import pytest

from rag.verification_meta import (
    MAX_CORRECTION_NOTE_LEN,
    MAX_OVERENI_SOUHRN_LEN,
    MAX_QUOTE_LEN,
    DocumentMetadataError,
    figure_payload_fields,
    is_hidden_doc_status,
    normalize_overeni_souhrn,
    normalize_verification_metadata,
    split_verification_deletes,
    supersedes_ids,
    validate_merged_lifecycle_fields,
)


def _item(**overrides: object) -> dict[str, object]:
    """Return a valid verification[] row with optional overrides."""
    item: dict[str, object] = {
        "agent": "qgate",
        "check": "gps",
        "tool": "qgate.py",
        "date": "2026-11-12T12:00:00+01:00",
        "result": "overeno",
        "source_url": "https://example.test/zdroj",
        "quote": "Citace z tisku.",
        "independent": True,
        "source_key": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    }
    item.update(overrides)
    return item


def test_defaults_reliability_one() -> None:
    """Absent reliability should default to 1 when requested."""
    out, truncated = normalize_verification_metadata(None, apply_defaults=True)
    assert out["reliability"] == 1
    assert out["verification_status"] == "nejiste"
    assert truncated == []


def test_overeni_souhrn_is_truncated() -> None:
    """``overeni_souhrn`` longer than ~300 chars should be truncated."""
    long_text = "ž" * (MAX_OVERENI_SOUHRN_LEN + 40)
    clipped: list[str] = []
    assert len(normalize_overeni_souhrn(long_text, truncated=clipped)) == (
        MAX_OVERENI_SOUHRN_LEN
    )
    assert clipped == ["overeni_souhrn"]
    out, truncated = normalize_verification_metadata({"overeni_souhrn": long_text})
    assert out["overeni_souhrn"] == "ž" * MAX_OVERENI_SOUHRN_LEN
    assert truncated == ["overeni_souhrn"]


def test_quote_fields_report_truncated_paths() -> None:
    """Long quote / verified_quote values should list their field paths."""
    long_quote = "q" * (MAX_QUOTE_LEN + 10)
    long_verified = "v" * (MAX_QUOTE_LEN + 3)
    out, truncated = normalize_verification_metadata(
        {
            "verified_quote": long_verified,
            "verification": [_item(quote=long_quote)],
        }
    )
    assert out["verified_quote"] == "v" * MAX_QUOTE_LEN
    assert out["verification"][0]["quote"] == "q" * MAX_QUOTE_LEN
    assert truncated == ["verified_quote", "verification[0].quote"]


def test_verification_item_new_fields_round_trip() -> None:
    """Optional method / compared / found / threshold / evidence_type are kept."""
    out, truncated = normalize_verification_metadata(
        {
            "reliability": 2,
            "source_key": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "verification": [
                _item(
                    method="haversine",
                    compared_value=49.2,
                    found_value="49.21,16.61",
                    threshold=100,
                    evidence_type="gps",
                )
            ],
        }
    )
    assert truncated == []
    item = out["verification"][0]
    assert item["method"] == "haversine"
    assert item["compared_value"] == 49.2
    assert item["found_value"] == "49.21,16.61"
    assert item["threshold"] == 100
    assert item["evidence_type"] == "gps"


def test_overeno_without_quote_is_downgraded() -> None:
    """``overeno`` without source_url+quote becomes ``nejiste``."""
    out, truncated = normalize_verification_metadata(
        {"verification": [_item(source_url="", quote="")]}
    )
    assert out["verification"][0]["result"] == "nejiste"
    assert truncated == []


def test_source_axis_does_not_use_filename_key() -> None:
    """Reliability axis ``source`` is stored as ``source_axis``."""
    out, truncated = normalize_verification_metadata({"source": "primary"})
    assert out["source_axis"] == "primary"
    assert "source" not in out
    assert truncated == []


def test_unknown_field_rejected() -> None:
    """Unknown top-level keys should fail."""
    with pytest.raises(DocumentMetadataError, match="unknown metadata"):
        normalize_verification_metadata({"doc_id": "nope"})


def test_split_deletes_whitelist_and_rollback_rule() -> None:
    """Null deletes whitelist keys; source_key/content_hash need verification:null."""
    remaining, deleted = split_verification_deletes(
        {
            "reliability": 2,
            "verified_by": None,
            "overeni_souhrn": None,
            "qg_trida": None,
            "qg_flagy": None,
            "qg_verze": None,
        }
    )
    assert remaining == {"reliability": 2}
    assert deleted == [
        "verified_by",
        "overeni_souhrn",
        "qg_trida",
        "qg_flagy",
        "qg_verze",
    ]

    remaining, deleted = split_verification_deletes(
        {
            "verification": None,
            "source_key": None,
            "content_hash": None,
        }
    )
    assert remaining == {}
    assert deleted == ["verification", "source_key", "content_hash"]

    with pytest.raises(DocumentMetadataError, match="source_key"):
        split_verification_deletes({"source_key": None})
    with pytest.raises(DocumentMetadataError, match="content_hash"):
        split_verification_deletes({"content_hash": None, "reliability": 1})
    for field in ("tags", "obec_kod", "gps_presnost", "extrakce_model", "fakt_ref"):
        with pytest.raises(DocumentMetadataError, match=field):
            split_verification_deletes({field: None})

    remaining, deleted = split_verification_deletes(
        {"doc_status": None, "correction_note": None, "reliability": 2}
    )
    assert remaining == {"reliability": 2}
    assert deleted == ["doc_status", "correction_note"]
    with pytest.raises(DocumentMetadataError, match="supersede"):
        split_verification_deletes({"superseded_by": None})
    with pytest.raises(DocumentMetadataError, match="supersede"):
        split_verification_deletes({"supersedes": None})
    with pytest.raises(DocumentMetadataError, match="doc_status"):
        split_verification_deletes({"status": None})


def test_figure_payload_omits_colliding_keys() -> None:
    """Figures must not receive document verification[] or status."""
    fields, truncated = normalize_verification_metadata(
        {
            "reliability": 1,
            "verification_status": "nejiste",
            "overeni_souhrn": "nejisté",
            "verification": [_item()],
        }
    )
    assert truncated == []
    figure = figure_payload_fields(fields)
    assert figure["reliability"] == 1
    assert figure["overeni_souhrn"] == "nejisté"
    assert "verification" not in figure
    assert "verification_status" not in figure


def test_lifecycle_fields_are_figure_safe() -> None:
    """doc_status and correction_note should be written onto figure points."""
    out, truncated = normalize_verification_metadata(
        {
            "doc_status": "retracted",
            "correction_note": "wrong year",
            "reliability": 2,
        }
    )
    assert truncated == []
    figure = figure_payload_fields(out)
    assert figure["doc_status"] == "retracted"
    assert figure["correction_note"] == "wrong year"


def test_correction_note_is_truncated() -> None:
    """correction_note longer than 500 chars should clip and report truncated."""
    long_note = "n" * (MAX_CORRECTION_NOTE_LEN + 9)
    out, truncated = normalize_verification_metadata({"correction_note": long_note})
    assert out["correction_note"] == "n" * MAX_CORRECTION_NOTE_LEN
    assert truncated == ["correction_note"]


def test_doc_status_enum_and_status_alias_rejected() -> None:
    """Only active|superseded|retracted are valid; status is not the field."""
    out, truncated = normalize_verification_metadata({"doc_status": "active"})
    assert out["doc_status"] == "active"
    assert truncated == []
    with pytest.raises(DocumentMetadataError, match="doc_status"):
        normalize_verification_metadata({"doc_status": "archived"})
    with pytest.raises(DocumentMetadataError, match="doc_status"):
        normalize_verification_metadata({"status": "superseded"})
    with pytest.raises(DocumentMetadataError, match="supersede"):
        normalize_verification_metadata({"superseded_by": "abc"})


def test_merged_lifecycle_invariants() -> None:
    """superseded needs a link; retracted needs a note; hidden statuses work."""
    validate_merged_lifecycle_fields(
        {"doc_status": "superseded", "superseded_by": "doc-b"}
    )
    validate_merged_lifecycle_fields(
        {"doc_status": "retracted", "correction_note": "bad year"}
    )
    with pytest.raises(DocumentMetadataError, match="superseded_by"):
        validate_merged_lifecycle_fields({"doc_status": "superseded"})
    with pytest.raises(DocumentMetadataError, match="correction_note"):
        validate_merged_lifecycle_fields({"doc_status": "retracted"})
    with pytest.raises(DocumentMetadataError, match="unsupersede"):
        validate_merged_lifecycle_fields(
            {"doc_status": "active", "superseded_by": "doc-b"}
        )
    with pytest.raises(DocumentMetadataError, match="unsupersede"):
        validate_merged_lifecycle_fields(
            {"superseded_by": "doc-b"}, deleted=["doc_status"]
        )
    assert is_hidden_doc_status("superseded") is True
    assert is_hidden_doc_status("retracted") is True
    assert is_hidden_doc_status("active") is False
    assert is_hidden_doc_status(None) is False
    assert is_hidden_doc_status("") is False


def test_supersedes_ids_dedupes_and_rejects_bad_types() -> None:
    """supersedes may be a string or list; other types are invalid."""
    assert supersedes_ids(None) == []
    assert supersedes_ids(["a", "a", " b "]) == ["a", "b"]
    assert supersedes_ids("solo") == ["solo"]
    with pytest.raises(DocumentMetadataError, match="list of doc_ids"):
        supersedes_ids(123)
