"""Unit tests for verification metadata validation."""

from __future__ import annotations

import pytest

from rag.verification_meta import (
    MAX_OVERENI_SOUHRN_LEN,
    DocumentMetadataError,
    figure_payload_fields,
    normalize_overeni_souhrn,
    normalize_verification_metadata,
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
    out = normalize_verification_metadata(None, apply_defaults=True)
    assert out["reliability"] == 1
    assert out["verification_status"] == "nejiste"


def test_overeni_souhrn_is_truncated() -> None:
    """``overeni_souhrn`` longer than ~300 chars should be truncated."""
    long_text = "ž" * (MAX_OVERENI_SOUHRN_LEN + 40)
    assert len(normalize_overeni_souhrn(long_text)) == MAX_OVERENI_SOUHRN_LEN
    out = normalize_verification_metadata({"overeni_souhrn": long_text})
    assert out["overeni_souhrn"] == "ž" * MAX_OVERENI_SOUHRN_LEN


def test_verification_item_new_fields_round_trip() -> None:
    """Optional method / compared / found / threshold / evidence_type are kept."""
    out = normalize_verification_metadata(
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
    item = out["verification"][0]
    assert item["method"] == "haversine"
    assert item["compared_value"] == 49.2
    assert item["found_value"] == "49.21,16.61"
    assert item["threshold"] == 100
    assert item["evidence_type"] == "gps"


def test_overeno_without_quote_is_downgraded() -> None:
    """``overeno`` without source_url+quote becomes ``nejiste``."""
    out = normalize_verification_metadata(
        {"verification": [_item(source_url="", quote="")]}
    )
    assert out["verification"][0]["result"] == "nejiste"


def test_source_axis_does_not_use_filename_key() -> None:
    """Reliability axis ``source`` is stored as ``source_axis``."""
    out = normalize_verification_metadata({"source": "primary"})
    assert out["source_axis"] == "primary"
    assert "source" not in out


def test_unknown_field_rejected() -> None:
    """Unknown top-level keys should fail."""
    with pytest.raises(DocumentMetadataError, match="unknown metadata"):
        normalize_verification_metadata({"doc_id": "nope"})


def test_figure_payload_omits_colliding_keys() -> None:
    """Figures must not receive document verification[] or status."""
    fields = normalize_verification_metadata(
        {
            "reliability": 1,
            "verification_status": "nejiste",
            "overeni_souhrn": "nejisté",
            "verification": [_item()],
        }
    )
    figure = figure_payload_fields(fields)
    assert figure["reliability"] == 1
    assert figure["overeni_souhrn"] == "nejisté"
    assert "verification" not in figure
    assert "verification_status" not in figure
