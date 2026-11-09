"""Schema and helpers for optional ``geo_refined`` GPS metadata.

Original ``lat`` / ``lon`` (and aliases such as ``geom``) are never overwritten.
Refined coordinates live in an optional ``geo_refined`` object plus a
``geo_refined_history`` list so writes stay reversible and idempotent.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GEO_PRECISIONS = frozenset({"municipality", "cast_obce", "street", "building"})
GEO_CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})
GEO_STATUSES = frozenset({"overeno", "zpresneno", "priblizne", "rozpor"})
GEO_SOURCE_TYPES = frozenset(
    {
        "ruian_adresni_misto",
        "wikidata_p625",
        "commons_camera",
        "nominatim",
        "osm_building",
        "manual",
    }
)
GEO_SOURCES = frozenset({"refined", "original"})
ORIGINAL_COORD_KEYS = ("lat", "lon", "latitude", "longitude", "geom")
NEEDS_REFINEMENT_LEVELS = frozenset({"medium", "low"})
EARTH_RADIUS_M = 6_371_000.0

GeoPrecision = Literal["municipality", "cast_obce", "street", "building"]
GeoConfidenceLevel = Literal["high", "medium", "low"]
GeoStatus = Literal["overeno", "zpresneno", "priblizne", "rozpor"]
GeoSourceType = Literal[
    "ruian_adresni_misto",
    "wikidata_p625",
    "commons_camera",
    "nominatim",
    "osm_building",
    "manual",
]
GeoSourceLabel = Literal["refined", "original"]


class GeoRefinedPayloadError(ValueError):
    """Raised when a ``geo_refined`` object is missing or invalid."""


class GeoRefinedNotFoundError(GeoRefinedPayloadError):
    """Raised when the refine target (document, chunk, or image) is missing."""


class GeoRefinedSource(BaseModel):
    """One evidence source that supports a refined coordinate."""

    model_config = ConfigDict(extra="forbid")

    type: GeoSourceType
    url: str | None = None
    distance_m: float | None = None

    @field_validator("url", mode="before")
    @classmethod
    def _blank_url_to_none(cls, value: object) -> object:
        """Treat empty strings as unset."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("distance_m")
    @classmethod
    def _non_negative_source_distance(cls, value: float | None) -> float | None:
        """Reject negative source distances."""
        if value is not None and value < 0:
            raise ValueError("sources[].distance_m must be >= 0")
        return value


class GeoRefined(BaseModel):
    """Validated refined GPS location stored beside original coordinates."""

    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    precision: GeoPrecision
    precision_m: float = Field(..., ge=0)
    confidence: float = Field(..., ge=0, le=1)
    confidence_level: GeoConfidenceLevel
    status: GeoStatus
    sources: list[GeoRefinedSource] = Field(default_factory=list)
    method: str = ""
    distance_m: float | None = Field(default=None, ge=0)
    needs_refinement: bool | None = None
    verified: bool = False
    refined_by: str | None = None
    refined_at: str | None = None
    note: str | None = None

    @field_validator("method", "refined_by", "note", "refined_at", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: object) -> object:
        """Strip optional strings; empty becomes ``None`` except ``method``."""
        if value is None:
            return None
        text = str(value).strip()
        return text

    @field_validator("refined_at")
    @classmethod
    def _validate_refined_at(cls, value: str | None) -> str | None:
        """Require ISO-8601 when ``refined_at`` is provided."""
        if value is None or value == "":
            return None
        _parse_iso_timestamp(value)
        return value

    @model_validator(mode="after")
    def _derive_needs_refinement(self) -> GeoRefined:
        """Default ``needs_refinement`` for medium/low confidence or rozpor."""
        if self.needs_refinement is None:
            self.needs_refinement = (
                self.confidence_level in NEEDS_REFINEMENT_LEVELS
                or self.status == "rozpor"
            )
        if not self.method:
            self.method = ""
        if self.refined_by == "":
            self.refined_by = None
        if self.note == "":
            self.note = None
        return self

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict, dropping unset optional fields."""
        data = self.model_dump(mode="json")
        if data.get("distance_m") is None:
            data.pop("distance_m", None)
        if not data.get("refined_by"):
            data.pop("refined_by", None)
        if not data.get("refined_at"):
            data.pop("refined_at", None)
        if not data.get("note"):
            data.pop("note", None)
        return data


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form (seconds)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; naive values are treated as UTC."""
    text = value.strip()
    if not text:
        raise GeoRefinedPayloadError("refined_at is empty")
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise GeoRefinedPayloadError(f"invalid ISO timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _validation_message(exc: Exception) -> str:
    """Flatten a Pydantic validation error into a short tool-visible string."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        parts: list[str] = []
        for item in errors():
            loc = ".".join(str(part) for part in item.get("loc") or ())
            msg = str(item.get("msg") or "invalid")
            parts.append(f"{loc}: {msg}" if loc else msg)
        if parts:
            return "; ".join(parts)
    return str(exc)


def parse_geo_refined(
    raw: object,
    *,
    now: str | None = None,
    original: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Validate ``raw`` and return a normalized ``geo_refined`` payload dict."""
    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GeoRefinedPayloadError("geo_refined must be a JSON object") from exc
    if not isinstance(parsed, Mapping):
        raise GeoRefinedPayloadError("geo_refined must be a JSON object")
    try:
        model = GeoRefined.model_validate(dict(parsed))
    except Exception as exc:  # noqa: BLE001 - surface as payload error
        raise GeoRefinedPayloadError(_validation_message(exc)) from exc
    timestamp = now or utc_now_iso()
    if not model.refined_at:
        model.refined_at = timestamp
    if model.distance_m is None and original is not None:
        model.distance_m = round(
            haversine_m(original[0], original[1], model.lat, model.lon),
            3,
        )
    elif model.distance_m is None:
        model.distance_m = 0.0
    return model.to_payload()


def coerce_geo_refined_argument(raw: object) -> Mapping[str, Any]:
    """Accept a mapping or JSON object string for MCP ``geo_refined``."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GeoRefinedPayloadError("geo_refined must be a JSON object") from exc
        if not isinstance(parsed, dict):
            raise GeoRefinedPayloadError("geo_refined must be a JSON object")
        return parsed
    if isinstance(raw, Mapping):
        return raw
    raise GeoRefinedPayloadError("geo_refined must be a JSON object")


def coerce_filter_argument(raw: object) -> dict[str, Any]:
    """Accept a mapping, JSON string, or empty value for list filters."""
    if raw in (None, "", {}):
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GeoRefinedPayloadError("filter must be a JSON object") from exc
        if not isinstance(parsed, dict):
            raise GeoRefinedPayloadError("filter must be a JSON object")
        return parsed
    if isinstance(raw, Mapping):
        return dict(raw)
    raise GeoRefinedPayloadError("filter must be a JSON object")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in metres between two WGS-84 points."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    hav = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(hav)))


def _as_float(value: object) -> float | None:
    """Return ``value`` as float, or ``None`` when missing/invalid."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def original_lat_lon(payload: Mapping[str, Any] | None) -> tuple[float, float] | None:
    """Extract original coordinates without reading ``geo_refined``.

    Prefers top-level ``lat``/``lon``, then ``latitude``/``longitude``, then a
    GeoJSON ``geom`` Point (``coordinates`` are lon, lat).
    """
    if not payload:
        return None
    lat = _as_float(payload.get("lat"))
    lon = _as_float(payload.get("lon"))
    if lat is None:
        lat = _as_float(payload.get("latitude"))
    if lon is None:
        lon = _as_float(payload.get("longitude"))
    if lat is not None and lon is not None:
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
        return None
    geom = payload.get("geom")
    if isinstance(geom, Mapping) and str(geom.get("type") or "") == "Point":
        coords = geom.get("coordinates")
        if isinstance(coords, Sequence) and len(coords) >= 2:
            lon = _as_float(coords[0])
            lat = _as_float(coords[1])
            if (
                lat is not None
                and lon is not None
                and -90 <= lat <= 90
                and -180 <= lon <= 180
            ):
                return lat, lon
    return None


def geo_fingerprint(geo: Mapping[str, Any] | None) -> str:
    """Stable comparison key that ignores ``refined_at``."""
    if not geo:
        return ""
    payload = {key: value for key, value in geo.items() if key != "refined_at"}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def apply_geo_refined(
    payload: Mapping[str, Any],
    geo_refined: Mapping[str, Any],
    *,
    now: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Return ``(updated_copy, changed)`` with history, never touching original coords.

    Idempotent: when the new object matches the current one (ignoring
    ``refined_at``), the payload is unchanged and ``changed`` is false.
    """
    original = original_lat_lon(payload)
    normalized = parse_geo_refined(geo_refined, now=now, original=original)
    updated = dict(payload)
    current = updated.get("geo_refined")
    current_map = current if isinstance(current, Mapping) else None
    if geo_fingerprint(current_map) == geo_fingerprint(normalized):
        return updated, False
    history = list(updated.get("geo_refined_history") or [])
    if current_map:
        history.append(dict(current_map))
    updated["geo_refined"] = normalized
    updated["geo_refined_history"] = history
    return updated, True


def geo_patch(updated: Mapping[str, Any]) -> dict[str, Any]:
    """Payload keys that may be written via ``set_payload`` (never lat/lon)."""
    patch: dict[str, Any] = {
        "geo_refined": updated.get("geo_refined"),
        "geo_refined_history": list(updated.get("geo_refined_history") or []),
    }
    for key in ORIGINAL_COORD_KEYS:
        if key in patch:
            raise GeoRefinedPayloadError(f"refusing to overwrite original {key}")
    return patch


def preferred_geo(
    payload: Mapping[str, Any] | None,
    *,
    fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return preferred coordinates plus original/refined extras.

    Prefers ``geo_refined`` on ``payload``, then ``fallback`` (typically
    document metadata). Original coordinates stay in ``geo_original``.
    """
    payload = payload or {}
    fallback = fallback or {}
    refined_raw = payload.get("geo_refined")
    if not isinstance(refined_raw, Mapping):
        refined_raw = fallback.get("geo_refined")
    refined = dict(refined_raw) if isinstance(refined_raw, Mapping) else None
    original = original_lat_lon(payload) or original_lat_lon(fallback)
    geo_original = (
        {"lat": original[0], "lon": original[1]} if original is not None else None
    )
    if refined is not None:
        lat = _as_float(refined.get("lat"))
        lon = _as_float(refined.get("lon"))
        if lat is not None and lon is not None:
            return {
                "lat": lat,
                "lon": lon,
                "geo_source": "refined",
                "geo_refined": refined,
                "geo_original": geo_original,
            }
    if original is not None:
        return {
            "lat": original[0],
            "lon": original[1],
            "geo_source": "original",
            "geo_refined": refined,
            "geo_original": geo_original,
        }
    return {
        "lat": None,
        "lon": None,
        "geo_source": None,
        "geo_refined": refined,
        "geo_original": None,
    }


def attach_preferred_geo(
    record: Mapping[str, Any],
    *,
    fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy ``record`` and overlay preferred lat/lon plus ``geo_source``."""
    out = dict(record)
    geo = preferred_geo(out, fallback=fallback)
    out["lat"] = geo["lat"]
    out["lon"] = geo["lon"]
    out["geo_source"] = geo["geo_source"]
    out["geo_refined"] = geo["geo_refined"]
    out["geo_original"] = geo["geo_original"]
    return out


def matches_geo_filter(
    payload: Mapping[str, Any] | None,
    *,
    precision: str | None = None,
    status: str | None = None,
    confidence_level: str | None = None,
    needs_refinement: bool | None = None,
    verified: bool | None = None,
    fallback: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether ``payload`` (or ``fallback``) matches refinement filters."""
    refined = None
    if payload and isinstance(payload.get("geo_refined"), Mapping):
        refined = payload.get("geo_refined")
    elif fallback and isinstance(fallback.get("geo_refined"), Mapping):
        refined = fallback.get("geo_refined")
    if not isinstance(refined, Mapping):
        return False
    if precision and str(refined.get("precision") or "") != precision:
        return False
    if status and str(refined.get("status") or "") != status:
        return False
    if (
        confidence_level
        and str(refined.get("confidence_level") or "") != confidence_level
    ):
        return False
    if needs_refinement is not None and bool(refined.get("needs_refinement")) != bool(
        needs_refinement
    ):
        return False
    if verified is not None and bool(refined.get("verified")) != bool(verified):
        return False
    return True


def validate_precision(value: str | None) -> str | None:
    """Return a normalized precision filter, or raise on unknown values."""
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text not in GEO_PRECISIONS:
        allowed = ", ".join(sorted(GEO_PRECISIONS))
        raise GeoRefinedPayloadError(f"precision must be one of: {allowed}")
    return text


def validate_status(value: str | None) -> str | None:
    """Return a normalized status filter, or raise on unknown values."""
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text not in GEO_STATUSES:
        allowed = ", ".join(sorted(GEO_STATUSES))
        raise GeoRefinedPayloadError(f"status must be one of: {allowed}")
    return text


def validate_confidence_level(value: str | None) -> str | None:
    """Return a normalized confidence_level filter, or raise on unknown values."""
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text not in GEO_CONFIDENCE_LEVELS:
        allowed = ", ".join(sorted(GEO_CONFIDENCE_LEVELS))
        raise GeoRefinedPayloadError(f"confidence_level must be one of: {allowed}")
    return text
