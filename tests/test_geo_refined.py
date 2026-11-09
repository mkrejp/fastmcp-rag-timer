"""Unit tests for ``geo_refined`` schema validation and prefer-refined helpers."""

from __future__ import annotations

import pytest

from rag.geo_index import MAX_GEO_BATCH_SIZE, coerce_geo_batch_items
from rag.geo_refined import (
    GeoRefinedPayloadError,
    apply_geo_refined,
    attach_preferred_geo,
    geo_fingerprint,
    original_lat_lon,
    parse_geo_refined,
    preferred_geo,
)


def _sample_geo(**overrides: object) -> dict[str, object]:
    """Return a valid ``geo_refined`` mapping with optional field overrides."""
    payload: dict[str, object] = {
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
    payload.update(overrides)
    return payload


def test_parse_geo_refined_accepts_valid_payload() -> None:
    """A complete object should normalize derived fields and keep lat/lon."""
    parsed = parse_geo_refined(_sample_geo())
    assert parsed["lat"] == pytest.approx(49.1951)
    assert parsed["lon"] == pytest.approx(16.6080)
    assert parsed["precision"] == "building"
    assert parsed["needs_refinement"] is False
    assert parsed["verified"] is False
    assert parsed["refined_at"]


def test_parse_geo_refined_derives_needs_refinement() -> None:
    """Medium/low confidence and rozpor default ``needs_refinement`` to true."""
    medium = parse_geo_refined(
        _sample_geo(confidence_level="medium", confidence=0.55)
    )
    assert medium["needs_refinement"] is True
    rozpor = parse_geo_refined(
        _sample_geo(status="rozpor", confidence_level="high", confidence=0.4)
    )
    assert rozpor["needs_refinement"] is True
    low = parse_geo_refined(
        _sample_geo(confidence_level="low", confidence=0.2, status="priblizne")
    )
    assert low["needs_refinement"] is True


def test_parse_geo_refined_rejects_out_of_range_coordinates() -> None:
    """Lat/lon outside WGS-84 ranges should raise."""
    with pytest.raises(GeoRefinedPayloadError, match="lat"):
        parse_geo_refined(_sample_geo(lat=91))
    with pytest.raises(GeoRefinedPayloadError, match="lon"):
        parse_geo_refined(_sample_geo(lon=181))


def test_parse_geo_refined_rejects_unknown_enums_and_confidence() -> None:
    """Unknown precision/status/source type and confidence > 1 should raise."""
    with pytest.raises(GeoRefinedPayloadError, match="precision"):
        parse_geo_refined(_sample_geo(precision="city"))
    with pytest.raises(GeoRefinedPayloadError, match="status"):
        parse_geo_refined(_sample_geo(status="ok"))
    with pytest.raises(GeoRefinedPayloadError, match="confidence"):
        parse_geo_refined(_sample_geo(confidence=1.2))
    with pytest.raises(GeoRefinedPayloadError, match="type"):
        parse_geo_refined(
            _sample_geo(sources=[{"type": "google", "url": "https://example.com"}])
        )


def test_apply_geo_refined_is_idempotent_and_keeps_history() -> None:
    """Repeating the same refinement must not grow history; a change must."""
    original = {"lat": 49.2, "lon": 16.6, "text": "Brno"}
    first, changed = apply_geo_refined(original, _sample_geo(), now="2026-01-01T00:00:00+00:00")
    assert changed is True
    assert first["lat"] == 49.2
    assert first["lon"] == 16.6
    assert first["geo_refined"]["lat"] == pytest.approx(49.1951)
    assert first["geo_refined_history"] == []

    again, changed_again = apply_geo_refined(
        first, _sample_geo(), now="2026-01-02T00:00:00+00:00"
    )
    assert changed_again is False
    assert again["geo_refined_history"] == []
    assert geo_fingerprint(again["geo_refined"]) == geo_fingerprint(first["geo_refined"])

    second, changed_second = apply_geo_refined(
        first,
        _sample_geo(lat=49.1952, note="slight move"),
        now="2026-01-03T00:00:00+00:00",
    )
    assert changed_second is True
    assert len(second["geo_refined_history"]) == 1
    assert second["geo_refined_history"][0]["lat"] == pytest.approx(49.1951)
    assert second["lat"] == 49.2
    assert second["lon"] == 16.6


def test_preferred_geo_prefers_refined_then_original() -> None:
    """Preferred lat/lon should come from geo_refined when present."""
    payload = {
        "lat": 49.2,
        "lon": 16.6,
        "geo_refined": parse_geo_refined(_sample_geo()),
    }
    geo = preferred_geo(payload)
    assert geo["geo_source"] == "refined"
    assert geo["lat"] == pytest.approx(49.1951)
    assert geo["geo_original"] == {"lat": 49.2, "lon": 16.6}

    original_only = preferred_geo({"lat": 50.0, "lon": 14.4})
    assert original_only["geo_source"] == "original"
    assert original_only["lat"] == pytest.approx(50.0)
    assert original_only["geo_refined"] is None

    attached = attach_preferred_geo({"lat": 49.2, "lon": 16.6, "doc_id": "d"})
    assert attached["geo_source"] == "original"
    assert attached["doc_id"] == "d"


def test_original_lat_lon_reads_geom_point_not_geo_refined() -> None:
    """GeoJSON Point geom should supply original coords; geo_refined should not."""
    geom = original_lat_lon(
        {"geom": {"type": "Point", "coordinates": [14.4378, 50.0755]}}
    )
    assert geom == (pytest.approx(50.0755), pytest.approx(14.4378))
    refined_only = original_lat_lon(
        {"geo_refined": parse_geo_refined(_sample_geo())}
    )
    assert refined_only is None


def test_coerce_geo_batch_items_enforces_max_and_shape() -> None:
    """Batch helper should accept list/object/JSON and reject oversize batches."""
    one = coerce_geo_batch_items({"doc_id": "a", "geo_refined": _sample_geo()})
    assert len(one) == 1
    encoded = coerce_geo_batch_items(
        '[{"doc_id": "a", "geo_refined": {"lat": 1}}]'
    )
    assert encoded[0]["doc_id"] == "a"
    with pytest.raises(GeoRefinedPayloadError, match="at least one"):
        coerce_geo_batch_items([])
    too_many = [{"doc_id": str(i), "geo_refined": {}} for i in range(MAX_GEO_BATCH_SIZE + 1)]
    with pytest.raises(GeoRefinedPayloadError, match="at most"):
        coerce_geo_batch_items(too_many)
