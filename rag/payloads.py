"""Qdrant payload shapes for text chunks, PDF image OCR hits, and locality links."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

LOCALITY_LINK_TYPE_DEFAULT = "mentions_locality"
LOCALITY_SOURCE_DATASET_DEFAULT = "rag-timer"
LOCALITY_SOURCE_TYPES = frozenset({"pdf_chunk", "pdf_image", "text_chunk"})
LOCALITY_METHODS = frozenset({"exact", "lemma", "fuzzy", "llm", "manual"})
LOCALITY_VERIFICATION_STATUSES = frozenset(
    {"unverified", "verified", "rejected"}
)
LOCALITY_CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})
LOCALITY_LEVELS = frozenset({"obec", "cast_obce"})
LOCALITY_LEVEL_DEFAULT = "obec"


class LocalityLinkPayloadError(ValueError):
    """Raised when a locality-link payload is missing or invalid."""


class LocalityLinkNotFoundError(LocalityLinkPayloadError):
    """Raised when a locality_links point id does not exist."""


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form (seconds)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_locality_link_id(
    *,
    doc_id: str,
    chunk_index: int,
    obec_id: str,
    char_start: int,
    cast_obce_kod: str | None = None,
) -> str:
    """Build a stable UUID matching migration 005 uniqueness.

    Obec-level ids stay ``(doc_id, chunk_index, obec_id, char_start)`` so
    existing points keep their ids. Část obce ids also include ``cast_obce_kod``.
    """
    if cast_obce_kod:
        key = (
            f"locality_link:{doc_id}:{chunk_index}:{obec_id}:"
            f"{cast_obce_kod}:{char_start}"
        )
    else:
        key = f"locality_link:{doc_id}:{chunk_index}:{obec_id}:{char_start}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _require_text(
    raw: Mapping[str, Any],
    key: str,
    *,
    strip: bool = True,
) -> str:
    """Return a required non-empty string field from ``raw``."""
    value = raw.get(key)
    if value is None:
        raise LocalityLinkPayloadError(f"missing required field: {key}")
    text = str(value)
    if strip:
        text = text.strip()
    if not text.strip():
        raise LocalityLinkPayloadError(f"missing required field: {key}")
    return text


def _optional_text(raw: Mapping[str, Any], key: str, default: str = "") -> str:
    """Return an optional string field, defaulting to ``default``."""
    value = raw.get(key, default)
    if value is None:
        return default
    return str(value)


def _require_int(raw: Mapping[str, Any], key: str) -> int:
    """Return a required integer field from ``raw``."""
    if key not in raw or raw[key] is None or raw[key] == "":
        raise LocalityLinkPayloadError(f"missing required field: {key}")
    try:
        return int(raw[key])
    except (TypeError, ValueError) as exc:
        raise LocalityLinkPayloadError(f"{key} must be an integer") from exc


def _optional_int(raw: Mapping[str, Any], key: str) -> int | None:
    """Return an optional integer field, or ``None`` when unset."""
    if key not in raw or raw[key] is None or raw[key] == "":
        return None
    try:
        return int(raw[key])
    except (TypeError, ValueError) as exc:
        raise LocalityLinkPayloadError(f"{key} must be an integer") from exc


def _optional_float(raw: Mapping[str, Any], key: str) -> float | None:
    """Return an optional float field, or ``None`` when unset."""
    if key not in raw or raw[key] is None or raw[key] == "":
        return None
    try:
        return float(raw[key])
    except (TypeError, ValueError) as exc:
        raise LocalityLinkPayloadError(f"{key} must be a number") from exc


def _optional_bool(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    """Return an optional boolean field, defaulting to ``default``."""
    if key not in raw or raw[key] is None or raw[key] == "":
        return default
    value = raw[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise LocalityLinkPayloadError(f"{key} must be a boolean")


def _parse_locality_level_and_cast(
    raw: Mapping[str, Any],
) -> tuple[str, str | None, str | None]:
    """Return ``(locality_level, cast_obce_kod, cast_obce_name)`` from ``raw``.

    Accepts migration-005 columns and a ``cast_obce`` alias (kod string or
    ``{kod, name}`` / ``{cast_obce_kod, cast_obce_name}``).
    """
    kod_from_alias: str | None = None
    name_from_alias: str | None = None
    cast_field = raw.get("cast_obce")
    if isinstance(cast_field, str) and cast_field.strip():
        kod_from_alias = cast_field.strip()
    elif isinstance(cast_field, Mapping):
        kod_from_alias = (
            str(
                cast_field.get("kod")
                or cast_field.get("cast_obce_kod")
                or ""
            ).strip()
            or None
        )
        name_from_alias = (
            str(
                cast_field.get("name")
                or cast_field.get("cast_obce_name")
                or ""
            ).strip()
            or None
        )
    elif cast_field not in (None, ""):
        raise LocalityLinkPayloadError(
            "cast_obce must be a kod string or object with kod/name"
        )

    kod = _optional_text(raw, "cast_obce_kod").strip() or kod_from_alias
    name = _optional_text(raw, "cast_obce_name").strip() or name_from_alias
    level_raw = _optional_text(raw, "locality_level").strip()
    if level_raw:
        level = level_raw
    elif kod:
        level = "cast_obce"
    else:
        level = LOCALITY_LEVEL_DEFAULT
    if level not in LOCALITY_LEVELS:
        allowed = ", ".join(sorted(LOCALITY_LEVELS))
        raise LocalityLinkPayloadError(
            f"locality_level must be one of: {allowed}"
        )
    if level == "obec":
        if kod:
            raise LocalityLinkPayloadError(
                "cast_obce_kod must be empty when locality_level is obec"
            )
        return LOCALITY_LEVEL_DEFAULT, None, None
    if not kod:
        raise LocalityLinkPayloadError(
            "cast_obce_kod is required when locality_level is cast_obce"
        )
    return "cast_obce", kod, name or None


def apply_locality_link_verification(
    payload: Mapping[str, Any],
    *,
    status: str,
    verified_by: str | None = None,
    notes: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Return a copy of ``payload`` with verification fields updated.

    ``verified`` is true only for ``verified``. ``verified_at`` is set on
    verified/rejected and cleared on unverified. ``verified_by`` and ``notes``
    are written only when the caller passes them (including empty to clear).
    """
    status_norm = str(status).strip()
    if status_norm not in LOCALITY_VERIFICATION_STATUSES:
        allowed = ", ".join(sorted(LOCALITY_VERIFICATION_STATUSES))
        raise LocalityLinkPayloadError(f"status must be one of: {allowed}")
    timestamp = now or _utc_now_iso()
    updated = dict(payload)
    updated["verification_status"] = status_norm
    updated["verified"] = status_norm == "verified"
    if status_norm == "unverified":
        updated["verified_at"] = None
    else:
        updated["verified_at"] = timestamp
    if verified_by is not None:
        updated["verified_by"] = str(verified_by).strip() or None
    if notes is not None:
        updated["notes"] = str(notes).strip() or None
    updated["updated_at"] = timestamp
    return updated


def build_locality_link_payload(
    raw: Mapping[str, Any],
    *,
    existing: Mapping[str, Any] | None = None,
    now: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build ``(link_id, payload)`` for a ``locality_links`` Qdrant point.

    Point id is a stable UUID derived from
    ``(doc_id, chunk_index, obec_id, char_start)`` plus ``cast_obce_kod``
    when ``locality_level`` is ``cast_obce``. ``created_at`` is kept when
    ``existing`` already has one; ``updated_at`` is always ``now``.
    """
    timestamp = now or _utc_now_iso()
    obec_id = _require_text(raw, "obec_id")
    name = _require_text(raw, "name")
    doc_id = _require_text(raw, "doc_id")
    chunk_index = _require_int(raw, "chunk_index")
    matched_text = _require_text(raw, "matched_text", strip=False)
    char_start = _require_int(raw, "char_start")
    char_end = _require_int(raw, "char_end")
    source_type = _require_text(raw, "source_type")
    method = _require_text(raw, "method")
    if source_type not in LOCALITY_SOURCE_TYPES:
        allowed = ", ".join(sorted(LOCALITY_SOURCE_TYPES))
        raise LocalityLinkPayloadError(
            f"source_type must be one of: {allowed}"
        )
    if method not in LOCALITY_METHODS:
        allowed = ", ".join(sorted(LOCALITY_METHODS))
        raise LocalityLinkPayloadError(f"method must be one of: {allowed}")

    verification_status = (
        _optional_text(raw, "verification_status", "unverified").strip()
        or "unverified"
    )
    if verification_status not in LOCALITY_VERIFICATION_STATUSES:
        allowed = ", ".join(sorted(LOCALITY_VERIFICATION_STATUSES))
        raise LocalityLinkPayloadError(
            f"verification_status must be one of: {allowed}"
        )
    verified_default = verification_status == "verified"
    verified = _optional_bool(raw, "verified", verified_default)
    if verification_status == "verified":
        verified = True
    elif verification_status == "rejected":
        verified = False

    confidence_level_raw = _optional_text(raw, "confidence_level", "").strip()
    confidence_level: str | None
    if confidence_level_raw:
        if confidence_level_raw not in LOCALITY_CONFIDENCE_LEVELS:
            allowed = ", ".join(sorted(LOCALITY_CONFIDENCE_LEVELS))
            raise LocalityLinkPayloadError(
                f"confidence_level must be one of: {allowed}"
            )
        confidence_level = confidence_level_raw
    else:
        confidence_level = None

    candidates = raw.get("candidates", [])
    if candidates is None:
        candidates = []
    if not isinstance(candidates, list):
        raise LocalityLinkPayloadError("candidates must be a JSON array")

    geom = raw.get("geom")
    if geom is not None and not isinstance(geom, (dict, list)):
        raise LocalityLinkPayloadError("geom must be an object when provided")

    locality_level, cast_obce_kod, cast_obce_name = _parse_locality_level_and_cast(
        raw
    )
    link_id = stable_locality_link_id(
        doc_id=doc_id,
        chunk_index=chunk_index,
        obec_id=obec_id,
        char_start=char_start,
        cast_obce_kod=cast_obce_kod,
    )
    created_at = timestamp
    if existing:
        previous = existing.get("created_at")
        if previous:
            created_at = str(previous)

    payload: dict[str, Any] = {
        "link_id": link_id,
        "link_type": (
            _optional_text(raw, "link_type", LOCALITY_LINK_TYPE_DEFAULT).strip()
            or LOCALITY_LINK_TYPE_DEFAULT
        ),
        "obec_id": obec_id,
        "name": name,
        "okres": _optional_text(raw, "okres"),
        "kraj": _optional_text(raw, "kraj"),
        "lau_id": _optional_text(raw, "lau_id") or None,
        "locality_level": locality_level,
        "cast_obce": cast_obce_kod,
        "cast_obce_kod": cast_obce_kod,
        "cast_obce_name": cast_obce_name,
        "source_dataset": (
            _optional_text(
                raw, "source_dataset", LOCALITY_SOURCE_DATASET_DEFAULT
            ).strip()
            or LOCALITY_SOURCE_DATASET_DEFAULT
        ),
        "doc_id": doc_id,
        "parent_document": _optional_text(raw, "parent_document") or None,
        "chunk_index": chunk_index,
        "page_number": _optional_int(raw, "page_number"),
        "point_id": _optional_text(raw, "point_id") or None,
        "source_type": source_type,
        "chunk_sha256": _optional_text(raw, "chunk_sha256"),
        "matched_text": matched_text,
        "char_start": char_start,
        "char_end": char_end,
        "method": method,
        "ambiguous": _optional_bool(raw, "ambiguous", False),
        "candidates": candidates,
        "verified": verified,
        "verification_status": verification_status,
        "verified_by": _optional_text(raw, "verified_by") or None,
        "verified_at": _optional_text(raw, "verified_at") or None,
        "confidence": _optional_float(raw, "confidence"),
        "confidence_level": confidence_level,
        "notes": _optional_text(raw, "notes") or None,
        "agent_version": _optional_text(raw, "agent_version") or None,
        "created_at": created_at,
        "updated_at": timestamp,
        "geom": geom,
    }
    return link_id, payload


def build_chunk_payload(
    *,
    doc_id: str,
    parent_document: str,
    page_number: int,
    doc_sha256: str,
    text: str,
    chunk_index: int,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Build the payload for a ``pdf_chunk`` (or non-PDF text chunk) point."""
    payload: dict[str, Any] = {
        "source_type": "pdf_chunk",
        "doc_id": doc_id,
        "parent_document": parent_document,
        "page_number": page_number,
        "sha256": doc_sha256,
        "content_summary": text,
        "text": text,
        "chunk_index": chunk_index,
        "source": parent_document,
    }
    if source_url:
        payload["source_url"] = source_url
    return payload


def build_image_payload(
    *,
    doc_id: str,
    parent_document: str,
    page_number: int,
    file_path: str,
    image_id: str,
    image_sha256: str,
    content_summary: str,
) -> dict[str, Any]:
    """Build the payload for a ``pdf_image`` point (OCR text embedded as vector)."""
    return {
        "source_type": "pdf_image",
        "doc_id": doc_id,
        "parent_document": parent_document,
        "page_number": page_number,
        "file_path": file_path,
        "image_id": image_id,
        "sha256": image_sha256,
        "content_summary": content_summary,
        "text": content_summary,
        "source": parent_document,
    }
