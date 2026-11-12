"""Unit tests for verification metadata validation."""

from __future__ import annotations

import pytest

from rag.verification_meta import (
    MAX_OVERENI_SOUHRN_LEN,
    MAX_QUOTE_LEN,
    DocumentMetadataError,
    figure_payload_fields,
    normalize_overeni_souhrn,
    normalize_verification_metadata,
    split_verification_deletes,
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
        }
    )
    assert remaining == {"reliability": 2}
    assert deleted == ["verified_by", "overeni_souhrn"]

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
