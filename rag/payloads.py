"""Qdrant payload shapes for text chunks, PDF image OCR hits, and locality links."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

LOCALITY_LINK_TYPE_DEFAULT = "mentions_locality"
LOCALITY_SOURCE_DATASET_DEFAULT = "rag-timer"
LOCALITY_SOURCE_TYPES = frozenset(
    {"pdf_chunk", "pdf_image", "pdf_figure", "text_chunk"}
)
FIGURE_VERIFICATION_STATUSES = frozenset(
    {"unverified", "auto_checked", "verified", "rejected"}
)
LOCALITY_METHODS = frozenset({"exact", "lemma", "fuzzy", "llm", "manual"})
LOCALITY_VERIFICATION_STATUSES = frozenset({"unverified", "verified", "rejected"})
LOCALITY_CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})
LOCALITY_LEVELS = frozenset({"obec", "cast_obce"})
LOCALITY_LEVEL_DEFAULT = "obec"


class ImageRecordPayloadError(ValueError):
    """Raised when a pdf_figure payload is missing or invalid."""


class ImageRecordNotFoundError(ImageRecordPayloadError):
    """Raised when a pdf_figure point id does not exist."""


class LocalityLinkPayloadError(ValueError):
    """Raised when a locality-link payload is missing or invalid."""


class LocalityLinkNotFoundError(LocalityLinkPayloadError):
    """Raised when a locality_links point id does not exist."""


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form (seconds)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def chunk_text_sha256(text: str) -> str:
    """Return SHA-256 of UTF-8 chunk text (distinct from the source document hash)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def knowledge_point_id_for_link(payload: Mapping[str, Any]) -> str | None:
    """Return the knowledge-collection point UUID matching a locality link.

    Prefer an explicit ``point_id`` on the link. Otherwise reconstruct the
    ingest-time text-chunk UUID from ``doc_id`` + ``chunk_index``. Image
    points use a different formula and are not reconstructed here.
    """
    explicit = payload.get("point_id")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    doc_id = payload.get("doc_id")
    chunk_index = payload.get("chunk_index")
    if not doc_id or chunk_index is None:
        return None
    try:
        index = int(chunk_index)
    except (TypeError, ValueError):
        return None
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{str(doc_id)}:{index}"))


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
            str(cast_field.get("kod") or cast_field.get("cast_obce_kod") or "").strip()
            or None
        )
        name_from_alias = (
            str(
                cast_field.get("name") or cast_field.get("cast_obce_name") or ""
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
        raise LocalityLinkPayloadError(f"locality_level must be one of: {allowed}")
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
        raise LocalityLinkPayloadError(f"source_type must be one of: {allowed}")
    if method not in LOCALITY_METHODS:
        allowed = ", ".join(sorted(LOCALITY_METHODS))
        raise LocalityLinkPayloadError(f"method must be one of: {allowed}")

    verification_status = (
        _optional_text(raw, "verification_status", "unverified").strip() or "unverified"
    )
    if verification_status not in LOCALITY_VERIFICATION_STATUSES:
        allowed = ", ".join(sorted(LOCALITY_VERIFICATION_STATUSES))
        raise LocalityLinkPayloadError(f"verification_status must be one of: {allowed}")
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

    locality_level, cast_obce_kod, cast_obce_name = _parse_locality_level_and_cast(raw)
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
    localities: Sequence[str] | None = None,
    cast_obce: Sequence[str] | None = None,
    extra_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the payload for a ``pdf_chunk`` (or non-PDF text chunk) point.

    ``sha256`` is the source-document hash. ``chunk_sha256`` hashes this
    chunk's UTF-8 text. ``localities`` / ``cast_obce`` are denormalized
    obec_id / cast_obce kod lists (empty until ``link_locality``).
    """
    payload: dict[str, Any] = {
        "source_type": "pdf_chunk",
        "doc_id": doc_id,
        "parent_document": parent_document,
        "page_number": page_number,
        "sha256": doc_sha256,
        "chunk_sha256": chunk_text_sha256(text),
        "content_summary": text,
        "text": text,
        "chunk_index": chunk_index,
        "source": parent_document,
        "localities": list(localities) if localities else [],
        "cast_obce": list(cast_obce) if cast_obce else [],
    }
    if source_url:
        payload["source_url"] = source_url
    if extra_payload:
        payload.update(dict(extra_payload))
    return payload


def stable_figure_point_id(doc_id: str, figure_id: str) -> str:
    """Return a stable Qdrant point id for a ``pdf_figure`` record."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:fig:{figure_id}"))


def build_figure_content_summary(
    *,
    page_number: int,
    caption: str | None,
    summary: str,
    ocr_text: str | None,
    figure_label: str | None = None,
) -> str:
    """Compose the searchable text body for a figure record.

    Caption-less photos still get a page label so ``text`` / embeddings are
    never empty when PyMuPDF indexes a large placement as ``pdf_figure``.
    """
    page_label = page_number + 1
    prefix = figure_label or f"Obr. {page_label}"
    parts: list[str] = []
    if caption:
        parts.append(f"{prefix}, str. {page_label}: {caption}.")
    if summary:
        parts.append(summary)
    if ocr_text:
        parts.append(f"Text v obrázku: {ocr_text}")
    joined = " ".join(part for part in parts if part).strip()
    if joined:
        return joined
    return f"{prefix}, str. {page_label}."


def apply_image_record_verification(
    payload: Mapping[str, Any],
    *,
    status: str,
    verified_by: str | None = None,
    notes: str | None = None,
    caption: str | None = None,
    summary: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Return a copy of ``payload`` with figure verification fields updated."""
    status_norm = str(status).strip()
    if status_norm not in FIGURE_VERIFICATION_STATUSES:
        allowed = ", ".join(sorted(FIGURE_VERIFICATION_STATUSES))
        raise ImageRecordPayloadError(f"status must be one of: {allowed}")
    timestamp = now or _utc_now_iso()
    updated = dict(payload)
    updated["verification_status"] = status_norm
    updated["verified"] = status_norm == "verified"
    verification = dict(updated.get("verification") or {})
    if verified_by is not None:
        verification["verified_by"] = str(verified_by).strip() or None
    if notes is not None:
        verification["notes"] = str(notes).strip() or None
    if status_norm in ("verified", "rejected"):
        verification["verified_at"] = timestamp
    elif status_norm == "unverified":
        verification["verified_at"] = None
    overrides = dict(verification.get("human_overrides") or {})
    if caption is not None:
        overrides["caption"] = caption or None
    if summary is not None:
        overrides["summary"] = summary or None
    if overrides:
        verification["human_overrides"] = overrides
    updated["verification"] = verification
    if caption is not None and caption:
        updated["caption"] = caption
    if summary is not None and summary:
        updated["summary"] = summary
        updated["summary_method"] = "manual"
    updated["updated_at"] = timestamp
    page_number = int(updated.get("page_number") or 0)
    content = build_figure_content_summary(
        page_number=page_number,
        caption=str(updated.get("caption") or "") or None,
        summary=str(updated.get("summary") or ""),
        ocr_text=str(updated.get("ocr_text") or "") or None,
        figure_label=str(updated.get("figure_label") or "") or None,
    )
    updated["content_summary"] = content
    updated["text"] = content
    updated["chunk_sha256"] = chunk_text_sha256(content)
    return updated


def build_figure_payload(
    *,
    doc_id: str,
    parent_document: str,
    doc_sha256: str,
    figure_id: str,
    page_number: int,
    bbox: Sequence[float],
    page_size: Sequence[float],
    area_ratio: float,
    placements: int,
    tiled: bool,
    image_id: str,
    image_sha256: str,
    file_path: str,
    render: Mapping[str, Any],
    figure_label: str | None,
    figure_number: str | None,
    caption: str | None,
    caption_source: str,
    ocr_text: str,
    ocr_text_raw: str,
    ocr_confidence: float | None,
    ocr_word_count: int,
    ocr_source: str,
    related_text_refs: Sequence[Mapping[str, Any]],
    summary: str,
    summary_sentences: Sequence[Mapping[str, Any]],
    summary_method: str,
    summary_model: str | None,
    verified: bool,
    verification_status: str,
    confidence: float,
    verification: Mapping[str, Any],
    figure_index_version: int,
    localities: Sequence[str] | None = None,
    cast_obce: Sequence[str] | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Build the payload for a ``pdf_figure`` point."""
    timestamp = updated_at or _utc_now_iso()
    content_summary = build_figure_content_summary(
        page_number=page_number,
        caption=caption,
        summary=summary,
        ocr_text=ocr_text or None,
        figure_label=figure_label,
    )
    return {
        "source_type": "pdf_figure",
        "figure_id": figure_id,
        "doc_id": doc_id,
        "parent_document": parent_document,
        "sha256": doc_sha256,
        "page_number": page_number,
        "bbox": [float(v) for v in bbox],
        "page_size": [float(v) for v in page_size],
        "area_ratio": float(area_ratio),
        "placements": int(placements),
        "tiled": bool(tiled),
        "image_id": image_id,
        "image_sha256": image_sha256,
        "file_path": file_path,
        "render": dict(render),
        "figure_label": figure_label,
        "figure_number": figure_number,
        "caption": caption,
        "caption_source": caption_source,
        "ocr_text": ocr_text,
        "ocr_text_raw": ocr_text_raw,
        "ocr_confidence": ocr_confidence,
        "ocr_word_count": int(ocr_word_count),
        "ocr_source": ocr_source,
        "related_text_refs": [dict(ref) for ref in related_text_refs],
        "summary": summary,
        "summary_sentences": [dict(item) for item in summary_sentences],
        "summary_method": summary_method,
        "summary_model": summary_model,
        "content_summary": content_summary,
        "text": content_summary,
        "chunk_sha256": chunk_text_sha256(content_summary),
        "verified": bool(verified),
        "verification_status": verification_status,
        "confidence": float(confidence),
        "verification": dict(verification),
        "localities": list(localities) if localities else [],
        "cast_obce": list(cast_obce) if cast_obce else [],
        "figure_index_version": int(figure_index_version),
        "source": parent_document,
        "created_at": created_at or timestamp,
        "updated_at": timestamp,
    }


def build_image_payload(
    *,
    doc_id: str,
    parent_document: str,
    page_number: int,
    file_path: str,
    image_id: str,
    image_sha256: str,
    content_summary: str,
    localities: Sequence[str] | None = None,
    cast_obce: Sequence[str] | None = None,
    extra_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the payload for a ``pdf_image`` point (OCR text embedded as vector).

    ``chunk_sha256`` hashes ``content_summary`` (the text used for the
    image vector). Locality lists stay empty until ``link_locality``.
    """
    payload: dict[str, Any] = {
        "source_type": "pdf_image",
        "doc_id": doc_id,
        "parent_document": parent_document,
        "page_number": page_number,
        "file_path": file_path,
        "image_id": image_id,
        "sha256": image_sha256,
        "chunk_sha256": chunk_text_sha256(content_summary),
        "content_summary": content_summary,
        "text": content_summary,
        "source": parent_document,
        "localities": list(localities) if localities else [],
        "cast_obce": list(cast_obce) if cast_obce else [],
    }
    if extra_payload:
        payload.update(dict(extra_payload))
    return payload
