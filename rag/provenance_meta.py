"""Shared provenance block (ADR media-server-provenance §2.1) for ingest and API."""

from __future__ import annotations

import re
from typing import Any, Mapping

from rag.verification_meta import DocumentMetadataError

LICENSE_ID_VALUES = frozenset(
    {
        "CC-BY-SA-4.0",
        "CC-BY-4.0",
        "CC0-1.0",
        "PD",
        "Copyright",
        "Permission",
        "Own",
        "Unknown",
    }
)
LICENSE_STATUS_VALUES = frozenset(
    {
        "intake",
        "license_check",
        "metadata_complete",
        "verified",
        "published",
        "quarantine",
        "takedown",
        "restricted",
    }
)
CONTENT_TYPE_VALUES = frozenset(
    {"image", "document", "page_scan", "magazine", "book", "text"}
)
PD_STATUS_VALUES = frozenset({"pd_eu", "pd_us", "pd_both", "not_pd", "unknown"})

PROVENANCE_KEYS = frozenset(
    {
        "content_type",
        "source_url",
        "source_name",
        "source_library",
        "author",
        "title",
        "license_id",
        "license_url",
        "attribution_text",
        "retrieved_at",
        "modifications",
        "verified_by",
        "license_status",
        "sha256",
        "publication_year",
        "author_death_year",
        "pd_status",
        "quote",
        "permission_evidence_url",
        "media_asset_id",
    }
)

LEGACY_PROVENANCE_ALIASES = frozenset(
    {
        "licence",
        "obrazek_licence",
        "autor",
        "license",
    }
)

DELETABLE_PROVENANCE_FIELDS = PROVENANCE_KEYS

_MEDIA_ASSET_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

HIDDEN_LICENSE_STATUSES = frozenset({"quarantine", "takedown", "restricted"})
NON_HOSTABLE_LICENSE_IDS = frozenset({"Copyright", "Unknown"})


class ProvenanceMetadataError(DocumentMetadataError):
    """Raised when provenance metadata is missing or invalid."""


def split_provenance_from_metadata(
    raw: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split ingest/PATCH metadata into verification vs provenance dicts."""
    if raw is None:
        return {}, {}
    if not isinstance(raw, Mapping):
        raise ProvenanceMetadataError("metadata must be an object")
    rest: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key)
        if name in PROVENANCE_KEYS:
            provenance[name] = value
        else:
            rest[name] = value
    return rest, provenance


def split_metadata_patch(
    raw: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str], dict[str, Any], list[str]]:
    """Split PATCH body into verification sets/deletes and provenance sets/deletes."""
    from rag.verification_meta import (
        DELETABLE_VERIFICATION_FIELDS,
        ROLLBACK_ONLY_DELETE_FIELDS,
        STATUS_ALIAS_MESSAGE,
        _reject_lifecycle_link_fields,
    )

    if raw is None:
        return {}, [], {}, []
    if not isinstance(raw, Mapping):
        raise ProvenanceMetadataError("metadata must be an object")

    if "status" in raw:
        raise ProvenanceMetadataError(STATUS_ALIAS_MESSAGE)
    _reject_lifecycle_link_fields(raw.keys())
    verification_rollback = "verification" in raw and raw["verification"] is None
    ver_remaining: dict[str, Any] = {}
    ver_deleted: list[str] = []
    prov_remaining: dict[str, Any] = {}
    prov_deleted: list[str] = []
    for key, value in raw.items():
        name = str(key)
        if value is None:
            if name not in DELETABLE_VERIFICATION_FIELDS:
                if name in DELETABLE_PROVENANCE_FIELDS:
                    prov_deleted.append(name)
                else:
                    raise ProvenanceMetadataError(f"{name} must not be null")
                continue
            if name in ROLLBACK_ONLY_DELETE_FIELDS and not verification_rollback:
                raise ProvenanceMetadataError(
                    f"{name} may be null only when verification is also null"
                )
            ver_deleted.append(name)
            continue
        if name in PROVENANCE_KEYS:
            prov_remaining[name] = value
        else:
            ver_remaining[name] = value
    return ver_remaining, ver_deleted, prov_remaining, prov_deleted


def _require_str(value: object, *, path: str, allow_empty: bool = False) -> str:
    """Return a string field, optionally allowing empty."""
    if not isinstance(value, str):
        raise ProvenanceMetadataError(f"{path} must be a string")
    if not allow_empty and not value.strip():
        raise ProvenanceMetadataError(f"{path} must be a non-empty string")
    return value


def _enum(value: object, allowed: frozenset[str], *, path: str) -> str:
    """Return ``value`` if it is one of ``allowed``."""
    text = _require_str(value, path=path)
    if text not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ProvenanceMetadataError(f"{path} must be one of: {choices}")
    return text


def _optional_str(raw: Mapping[str, Any], key: str) -> str | None:
    """Return an optional string or None when unset."""
    if key not in raw or raw[key] is None or raw[key] == "":
        return None
    return _require_str(raw[key], path=key, allow_empty=True).strip() or None


def _optional_int(raw: Mapping[str, Any], key: str) -> int | None:
    """Return an optional integer or None when unset."""
    if key not in raw or raw[key] is None or raw[key] == "":
        return None
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ProvenanceMetadataError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ProvenanceMetadataError(f"{key} must be an integer") from exc


def normalize_provenance_metadata(
    raw: Mapping[str, Any] | None,
    *,
    apply_restricted_defaults: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Validate and return provenance fields for catalog/Qdrant payloads.

    Unknown keys and legacy aliases (``licence``, ``autor``, …) are rejected.
    ``zdroj`` is verification-only and must not appear here.
    """
    if raw is None:
        data: dict[str, Any] = {}
    else:
        if not isinstance(raw, Mapping):
            raise ProvenanceMetadataError("metadata must be an object")
        data = dict(raw)

    for legacy in LEGACY_PROVENANCE_ALIASES:
        if legacy in data:
            raise ProvenanceMetadataError(
                f"legacy provenance field {legacy!r} is not accepted; "
                "use the ADR snake_case names (license_id, author, …)"
            )
    if "zdroj" in data:
        raise ProvenanceMetadataError(
            "zdroj belongs in verification metadata, not provenance"
        )

    unknown = [key for key in data if key not in PROVENANCE_KEYS]
    if unknown:
        raise ProvenanceMetadataError(
            f"unknown provenance fields: {', '.join(sorted(str(k) for k in unknown))}"
        )
    if not data and not apply_restricted_defaults:
        return {}, []

    out: dict[str, Any] = {}
    truncated: list[str] = []

    if "content_type" in data:
        out["content_type"] = _enum(
            data["content_type"], CONTENT_TYPE_VALUES, path="content_type"
        )
    for key in (
        "source_url",
        "source_name",
        "source_library",
        "author",
        "title",
        "license_url",
        "attribution_text",
        "retrieved_at",
        "modifications",
        "verified_by",
        "quote",
        "permission_evidence_url",
    ):
        if key in data:
            out[key] = _require_str(data[key], path=key, allow_empty=True)

    if "license_id" in data:
        out["license_id"] = _enum(
            data["license_id"], LICENSE_ID_VALUES, path="license_id"
        )
    if "license_status" in data:
        out["license_status"] = _enum(
            data["license_status"], LICENSE_STATUS_VALUES, path="license_status"
        )
    if "pd_status" in data:
        out["pd_status"] = _enum(data["pd_status"], PD_STATUS_VALUES, path="pd_status")
    if "sha256" in data:
        digest = _require_str(data["sha256"], path="sha256")
        if len(digest) != 64 or not all(
            c in "0123456789abcdef" for c in digest.lower()
        ):
            raise ProvenanceMetadataError("sha256 must be a 64-char hex digest")
        out["sha256"] = digest.lower()
    if "publication_year" in data:
        out["publication_year"] = _optional_int(data, "publication_year")
    if "author_death_year" in data:
        out["author_death_year"] = _optional_int(data, "author_death_year")
    if "media_asset_id" in data:
        asset_id = _optional_str(data, "media_asset_id")
        if asset_id is not None and not _MEDIA_ASSET_ID_RE.match(asset_id):
            raise ProvenanceMetadataError(
                "media_asset_id must be a UUID string when set"
            )
        out["media_asset_id"] = asset_id

    if apply_restricted_defaults and "license_id" not in out:
        out.setdefault("license_id", "Unknown")
        out.setdefault("license_status", "restricted")

    if (
        out.get("license_id") == "Permission"
        and not (out.get("permission_evidence_url") or "").strip()
    ):
        raise ProvenanceMetadataError(
            "permission_evidence_url is required when license_id is Permission"
        )

    return out, truncated


def document_provenance_from_extra(extra: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return provenance keys copied from document ``extra``."""
    if not extra:
        return {}
    return {key: extra[key] for key in PROVENANCE_KEYS if key in extra}


def inherit_provenance_for_extracted(
    parent: Mapping[str, Any],
    *,
    image_sha256: str,
    content_type: str = "image",
    modifications: str = "extracted from PDF",
) -> dict[str, Any]:
    """Copy parent provenance onto an extracted image/figure payload."""
    inherited = document_provenance_from_extra(parent)
    if not inherited:
        inherited = {"license_id": "Unknown", "license_status": "restricted"}
    inherited["content_type"] = content_type
    inherited["sha256"] = image_sha256
    if modifications:
        prior = (inherited.get("modifications") or "").strip()
        if prior and prior.lower() not in {"none", ""}:
            inherited["modifications"] = f"{prior}; {modifications}"
        else:
            inherited["modifications"] = modifications
    return inherited


def attribution_allows_public_urls(record: Mapping[str, Any]) -> bool:
    """Return whether binary URLs may be exposed (fail-closed without attribution)."""
    text = str(record.get("attribution_text") or "").strip()
    if not text:
        return False
    license_status = str(record.get("license_status") or "").strip()
    if license_status in HIDDEN_LICENSE_STATUSES:
        return False
    license_id = str(record.get("license_id") or "").strip()
    if license_id in NON_HOSTABLE_LICENSE_IDS and license_status != "published":
        return False
    return True


def build_image_delivery_urls(
    record: Mapping[str, Any],
    *,
    doc_id: str,
    image_id: str,
    public_base_url: str | None,
    media_public_base_url: str | None,
) -> dict[str, str | None]:
    """Build thumb/large/rag URLs per ADR (media when ``media_asset_id``, else RAG)."""
    from rag.config import resolve_public_base_url

    base_public: str | None
    if public_base_url is not None:
        base_public = public_base_url
    else:
        base_public = resolve_public_base_url()
    allowed = attribution_allows_public_urls(record)
    if not allowed:
        return {"thumb_url": None, "large_url": None, "url": None}

    asset_id = str(record.get("media_asset_id") or "").strip()
    media_base = (media_public_base_url or "").strip().rstrip("/")
    if not media_base:
        from rag.config import resolve_media_public_base_url

        media_base = resolve_media_public_base_url() or ""

    if asset_id and media_base:
        thumb = f"{media_base}/v1/assets/{asset_id}/thumb"
        large = f"{media_base}/v1/assets/{asset_id}/large"
        return {"thumb_url": thumb, "large_url": large, "url": thumb}

    if base_public and doc_id and image_id:
        rag_url = f"{base_public.rstrip('/')}/images/{doc_id}/{image_id}"
        return {"thumb_url": None, "large_url": None, "url": rag_url}

    return {"thumb_url": None, "large_url": None, "url": None}


def attach_image_api_fields(
    item: dict[str, Any],
    *,
    doc_id: str,
    image_id: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge provenance block and delivery URLs onto a catalog/API image dict."""
    merged = dict(item)
    block = dict(provenance or {})
    for key in PROVENANCE_KEYS:
        if key in block:
            merged[key] = block[key]
    urls = build_image_delivery_urls(
        merged,
        doc_id=doc_id,
        image_id=image_id,
        public_base_url=None,
        media_public_base_url=None,
    )
    merged.update(urls)
    title = str(merged.get("title") or "").strip()
    attribution = str(merged.get("attribution_text") or "").strip()
    if title and attribution:
        merged["alt"] = f"{title} — {attribution}"
    elif attribution:
        merged["alt"] = attribution
    elif title:
        merged["alt"] = title
    else:
        merged["alt"] = ""
    return merged


def merge_ingest_metadata(
    raw: Mapping[str, Any] | None,
    *,
    apply_verification_defaults: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize combined verification + provenance metadata for ingest/PATCH."""
    from rag.verification_meta import normalize_verification_metadata

    ver_raw, prov_raw = split_provenance_from_metadata(raw)
    if ver_raw or apply_verification_defaults:
        ver_fields, truncated_v = normalize_verification_metadata(
            ver_raw if ver_raw else None,
            apply_defaults=apply_verification_defaults,
        )
    else:
        ver_fields, truncated_v = {}, []
    if prov_raw:
        prov_fields, truncated_p = normalize_provenance_metadata(prov_raw)
    else:
        prov_fields, truncated_p = {}, []
    combined = {**ver_fields, **prov_fields}
    if not combined and not apply_verification_defaults:
        raise ProvenanceMetadataError("at least one metadata field is required")
    return combined, truncated_v + truncated_p


def figure_extra_from_document(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Return provenance fields safe to merge onto figure/image Qdrant payloads."""
    from rag.verification_meta import figure_payload_fields

    return figure_payload_fields(meta)
