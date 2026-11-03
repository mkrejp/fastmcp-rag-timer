"""Turn GeoJSON (and similar JSON) into RAG text records.

Coordinate arrays are omitted so large files such as Czech ``obce.json`` /
``okresy.json`` / ``kraje.json`` index names and attributes, not polygons.
"""

from __future__ import annotations

from typing import Any

GEOJSON_FEATURE = "Feature"
GEOJSON_FEATURE_COLLECTION = "FeatureCollection"
GEOJSON_GEOMETRY_TYPES = {
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
    "GeometryCollection",
}
SKIP_KEYS = {"coordinates", "geometry", "geometries", "features", "bbox", "type", "crs"}


def is_geojson_like(payload: Any) -> bool:
    """Return whether ``payload`` looks like GeoJSON or a list of features."""
    if _is_feature_collection(payload) or _is_feature(payload) or _is_geometry(payload):
        return True
    if isinstance(payload, list) and payload:
        return all(isinstance(item, dict) for item in payload) and any(
            _is_feature(item) or _is_geometry(item) or "properties" in item or "geometry" in item
            for item in payload
        )
    return False


def geojson_to_records(payload: Any) -> list[str]:
    """Return indexable text records for any similar GeoJSON payload."""
    if _is_feature_collection(payload):
        records = [_feature_record(item) for item in payload.get("features", []) if isinstance(item, dict)]
        return [record for record in records if record]
    if _is_feature(payload):
        record = _feature_record(payload)
        return [record] if record else []
    if _is_geometry(payload):
        record = _geometry_record(payload)
        return [record] if record else []
    if isinstance(payload, list):
        records: list[str] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if _is_feature(item) or "geometry" in item or "properties" in item:
                record = _feature_record(item)
            else:
                record = _object_record(item)
            if record:
                records.append(record)
        return records
    if isinstance(payload, dict):
        record = _object_record(payload)
        return [record] if record else []
    if payload is None:
        return []
    text = str(payload).strip()
    return [text] if text else []


def _is_feature_collection(payload: Any) -> bool:
    """Return whether ``payload`` is a GeoJSON FeatureCollection."""
    return (
        isinstance(payload, dict)
        and payload.get("type") == GEOJSON_FEATURE_COLLECTION
        and isinstance(payload.get("features"), list)
    )


def _is_feature(payload: Any) -> bool:
    """Return whether ``payload`` is a GeoJSON Feature (or close equivalent)."""
    if not isinstance(payload, dict):
        return False
    if payload.get("type") == GEOJSON_FEATURE:
        return True
    return "geometry" in payload and isinstance(payload.get("geometry"), (dict, type(None)))


def _is_geometry(payload: Any) -> bool:
    """Return whether ``payload`` is a GeoJSON geometry object."""
    return isinstance(payload, dict) and payload.get("type") in GEOJSON_GEOMETRY_TYPES


def _feature_record(feature: dict[str, Any]) -> str:
    """Render one Feature as a single RAG record without coordinates."""
    parts: list[str] = []
    name = _scalar_text(feature.get("name"))
    if name:
        parts.append(f"name: {name}")
    ident = _scalar_text(feature.get("id"))
    if ident:
        parts.append(f"id: {ident}")

    geometry = feature.get("geometry")
    if isinstance(geometry, dict) and isinstance(geometry.get("type"), str):
        parts.append(f"geometry_type: {geometry['type']}")
    elif isinstance(geometry, dict) and geometry.get("type") is None and "geometries" in geometry:
        parts.append("geometry_type: GeometryCollection")

    properties = feature.get("properties")
    if isinstance(properties, dict):
        parts.extend(_kv_parts(properties))

    for key, value in feature.items():
        if key in SKIP_KEYS or key in {"name", "id", "properties"}:
            continue
        formatted = _format_value(value)
        if formatted is not None:
            parts.append(f"{key}: {formatted}")
        elif isinstance(value, dict):
            parts.extend(_kv_parts(value, prefix=key))

    return "; ".join(parts)


def _geometry_record(geometry: dict[str, Any]) -> str:
    """Render a geometry-only object as a short record."""
    geom_type = geometry.get("type")
    if geom_type == "GeometryCollection" and isinstance(geometry.get("geometries"), list):
        inner = [
            item.get("type")
            for item in geometry["geometries"]
            if isinstance(item, dict) and isinstance(item.get("type"), str)
        ]
        if inner:
            return f"geometry_type: GeometryCollection; members: {', '.join(inner)}"
    if isinstance(geom_type, str):
        return f"geometry_type: {geom_type}"
    return ""


def _object_record(obj: dict[str, Any]) -> str:
    """Flatten a generic JSON object, skipping coordinate-like values."""
    parts = _kv_parts(obj)
    return "; ".join(parts)


def _kv_parts(obj: dict[str, Any], *, prefix: str = "") -> list[str]:
    """Return ``key: value`` parts for JSON object fields useful to RAG."""
    parts: list[str] = []
    for key, value in obj.items():
        if key in SKIP_KEYS:
            continue
        label = f"{prefix}.{key}" if prefix else str(key)
        formatted = _format_value(value)
        if formatted is not None:
            parts.append(f"{label}: {formatted}")
        elif isinstance(value, dict):
            parts.extend(_kv_parts(value, prefix=label))
    return parts


def _format_value(value: Any) -> str | None:
    """Stringify a scalar or short list; skip geometries and coordinate arrays."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list):
        if not value or _looks_like_coordinates(value):
            return None
        if all(_is_scalar(item) or item is None for item in value):
            joined = ", ".join(_scalar_text(item) for item in value if _scalar_text(item))
            return joined or None
        return None
    if isinstance(value, dict):
        if _is_geometry(value) or "coordinates" in value:
            geom_type = value.get("type")
            return f"{geom_type}" if isinstance(geom_type, str) else None
        return None
    return str(value)


def _is_scalar(value: Any) -> bool:
    """Return whether ``value`` is a JSON scalar."""
    return isinstance(value, (str, int, float, bool))


def _scalar_text(value: Any) -> str:
    """Return a stripped string for a scalar, or empty."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    return ""


def _looks_like_coordinates(value: list[Any]) -> bool:
    """Return whether a list looks like GeoJSON coordinates rather than tags."""
    first = value[0]
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return True
    if isinstance(first, list):
        return True
    return False
