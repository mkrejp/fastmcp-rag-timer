"""Tests for generic GeoJSON-to-RAG text conversion."""

from __future__ import annotations

import json

from rag.geojson import geojson_to_records
from rag.processing import is_supported_filename, process_document


def test_supported_json_filenames() -> None:
    """JSON and GeoJSON filenames should be accepted."""
    assert is_supported_filename("kraje.json")
    assert is_supported_filename("obce.geojson")
    assert not is_supported_filename("image.png")


def test_czech_style_feature_collection_skips_coordinates() -> None:
    """Top-level name/id/codes are indexed; polygon rings are not."""
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "nationalCode": "19",
                "localId": "AU.2.19",
                "name": "Hlavní město Praha",
                "id": "CZ0100000000",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[14.0, 50.0], [14.1, 50.0], [14.1, 50.1], [14.0, 50.0]]],
                },
            },
            {
                "type": "Feature",
                "nationalCode": "27",
                "localId": "AU.2.27",
                "name": "Středočeský kraj",
                "id": "CZ0200000000",
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[[[13.0, 49.0], [13.2, 49.0], [13.2, 49.2], [13.0, 49.0]]]],
                },
            },
        ],
    }
    records = geojson_to_records(payload)
    assert len(records) == 2
    assert "Hlavní město Praha" in records[0]
    assert "CZ0100000000" in records[0]
    assert "nationalCode: 19" in records[0]
    assert "geometry_type: Polygon" in records[0]
    assert "14.0" not in records[0]
    assert "coordinates" not in records[0]
    assert "Středočeský kraj" in records[1]
    assert "geometry_type: MultiPolygon" in records[1]


def test_standard_feature_uses_properties() -> None:
    """RFC 7946 features should flatten the properties object."""
    payload = {
        "type": "Feature",
        "properties": {"NAME": "Brno", "NUTS": "CZ064"},
        "geometry": {"type": "Point", "coordinates": [16.6, 49.2]},
    }
    records = geojson_to_records(payload)
    assert len(records) == 1
    assert "NAME: Brno" in records[0]
    assert "NUTS: CZ064" in records[0]
    assert "geometry_type: Point" in records[0]
    assert "16.6" not in records[0]


def test_feature_list_without_collection_wrapper() -> None:
    """A bare list of features should still produce one record each."""
    payload = [
        {
            "type": "Feature",
            "properties": {"name": "obec A"},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1]]]},
        },
        {
            "type": "Feature",
            "properties": {"name": "obec B"},
            "geometry": {"type": "Polygon", "coordinates": [[[2, 2], [3, 3]]]},
        },
    ]
    records = geojson_to_records(payload)
    assert [r for r in records if "obec A" in r]
    assert [r for r in records if "obec B" in r]


def test_process_document_indexes_geojson_without_rings() -> None:
    """process_document should emit chunks that mention names, not coordinates."""
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "name": "Benešov",
                "id": "CZ0201000000",
                "geometry": {"type": "Polygon", "coordinates": [[[14.4, 49.8]]]},
            }
        ],
    }
    fulltext, chunks = process_document(
        "okresy.json", json.dumps(payload).encode("utf-8")
    )
    assert "Benešov" in fulltext
    assert chunks
    joined = " ".join(str(chunk["text"]) for chunk in chunks)
    assert "14.4" not in joined
