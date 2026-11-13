"""Verification / reliability payload fields for ingest and metadata PATCH.

Flat Qdrant keys follow RAGloader.md. ``source`` already means the filename on
knowledge points, so the reliability axis is stored as ``source_axis``.
Document ``verification_status`` (six Czech values) is not written onto
``pdf_figure`` points, which keep ``unverified`` / ``verified`` / ….
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

DEFAULT_RELIABILITY = 1
DEFAULT_VERIFICATION_STATUS = "nejiste"
MAX_QUOTE_LEN = 500
MAX_OVERENI_SOUHRN_LEN = 300
MAX_CORRECTION_NOTE_LEN = 500
MAX_VERIFICATION_ITEMS = 20
DOC_STATUS_VALUES = frozenset({"active", "superseded", "retracted"})
HIDDEN_DOC_STATUSES = frozenset({"superseded", "retracted"})
LIFECYCLE_LINK_FIELDS = frozenset({"superseded_by", "supersedes"})
LIFECYCLE_PATCH_FIELDS = frozenset({"doc_status", "correction_note"})
SUPERSEDE_ENDPOINT = "POST /documents/{doc_id}/supersede"
UNSUPERSEDE_ENDPOINT = "POST /documents/{doc_id}/unsupersede"
LIFECYCLE_LINK_PATCH_MESSAGE = (
    "superseded_by and supersedes cannot be set or deleted via "
    "PATCH /documents/{doc_id}/metadata; use "
    f"{SUPERSEDE_ENDPOINT} or {UNSUPERSEDE_ENDPOINT}"
)
STATUS_ALIAS_MESSAGE = (
    "unknown metadata fields: status (document lifecycle is doc_status; "
    "catalog status is the ingest state)"
)

VERIFICATION_RESULTS = frozenset(
    {
        "overeno",
        "potvrzeno-wiki",
        "nejiste",
        "rozpor",
        "zamitnuto",
        "nelze",
    }
)
VERIFICATION_CHECKS = frozenset(
    {
        "licence",
        "dedup",
        "duplicita",
        "qgate",
        "gps",
        "zdroj",
        "ocr",
        "datace",
        "autor",
        "text_vs_zdroj",
        "obec",
        "slepa_kolej",
        "audit",
    }
)
EVIDENCE_SOURCE_TYPES = frozenset(
    {"primary", "approved", "secondary", "machine", "human"}
)
SOURCE_AXIS_VALUES = frozenset({"primary", "secondary", "machine"})
DATING_VALUES = frozenset({"dolozena", "odhad", "chybi"})
GPS_AXIS_VALUES = frozenset({"presna", "wikidata", "obec_centroid", "chybna"})
GPS_BAND_VALUES = frozenset({"lt100", "100_500", "gt500"})
DRUH_TEXTU_VALUES = frozenset(
    {
        "clanek",
        "reprodukce",
        "inzerat",
        "katalog_vystavovatel",
        "seznam",
        "jine",
    }
)
OVERENI_VALUES = frozenset({"neovereno", "zdroj_ok", "dvojite", "udu"})
QG_TRIDA_VALUES = frozenset({"A+", "A", "B", "C", "D"})
UROVEN_TO_RELIABILITY = {"A": 3, "B": 2, "C": 1, "D": 1}

VERIFICATION_ITEM_REQUIRED = (
    "agent",
    "check",
    "tool",
    "date",
    "result",
    "source_url",
    "quote",
    "independent",
    "source_key",
)
VERIFICATION_ITEM_OPTIONAL = frozenset(
    {
        "pozn",
        "source_type",
        "source_id",
        "derived_from",
        "distance_m",
        "gps_band",
        "dating",
        "ref",
        "fakt_ref",
        "method",
        "compared_value",
        "found_value",
        "threshold",
        "evidence_type",
    }
)

# Filename ``source`` stays on the point; axis is ``source_axis``.
_FLAT_STRING_ENUMS: dict[str, frozenset[str]] = {
    "verification_status": VERIFICATION_RESULTS,
    "source_axis": SOURCE_AXIS_VALUES,
    "dating": DATING_VALUES,
    "gps": GPS_AXIS_VALUES,
    "druh_textu": DRUH_TEXTU_VALUES,
    "overeni": OVERENI_VALUES,
    "qg_trida": QG_TRIDA_VALUES,
    "doc_status": DOC_STATUS_VALUES,
}
_FLAT_STRINGS = frozenset(
    {
        "source_key",
        "content_hash",
        "last_verified_at",
        "verified_by",
        "verified_at",
        "verified_source_url",
        "verified_quote",
        "qg_verze",
        "relevance_duvod",
        "overil",
        "overeno_dne",
        "gps_presnost",
        "fakt_ref",
        "overeni_souhrn",
        "correction_note",
    }
)
_QG_LIST_OR_STRING = frozenset({"qg_duvody", "qg_flagy"})

FLAT_PAYLOAD_KEYS = (
    frozenset(_FLAT_STRING_ENUMS)
    | _FLAT_STRINGS
    | _QG_LIST_OR_STRING
    | {
        "reliability",
        "qg_skore",
        "qg_hard_ok",
        "relevance",
        "obec_kod",
        "conflict",
        "agreement",
        "ocr_conf",
        "zdroj",
        "verification",
        "gps_presnost_m",
        "gps_stav",
        "doc_status",
    }
)
FIGURE_EXCLUDED_KEYS = frozenset(
    {
        "verification",
        "verification_status",
        "verified",
        "source",
        "source_url",
    }
)

# PATCH /documents/{id}/metadata: JSON null deletes these payload keys.
DELETABLE_VERIFICATION_FIELDS = frozenset(
    {
        "verification",
        "verification_status",
        "verified_by",
        "verified_at",
        "verified_source_url",
        "verified_quote",
        "reliability",
        "source_axis",
        "source_key",
        "content_hash",
        "overeni_souhrn",
        "qg_trida",
        "qg_flagy",
        "qg_verze",
        "doc_status",
        "correction_note",
    }
)
# Only allowed as null in the same request as ``verification: null``.
ROLLBACK_ONLY_DELETE_FIELDS = frozenset({"source_key", "content_hash"})


class DocumentMetadataError(ValueError):
    """Raised when verification metadata is missing, unknown, or invalid."""


def validate_merged_lifecycle_fields(
    merged: Mapping[str, Any],
    *,
    deleted: Sequence[str] | None = None,
) -> None:
    """Reject lifecycle combinations that would be internally inconsistent.

    ``doc_status=superseded`` requires ``superseded_by`` (set only via the
    supersede endpoint). ``doc_status=retracted`` requires a non-empty
    ``correction_note``. Clearing ``doc_status`` while a supersession link
    remains must go through unlink.
    """
    removed = {str(name) for name in (deleted or [])}
    status = merged.get("doc_status")
    note = merged.get("correction_note")
    superseded_by = str(merged.get("superseded_by") or "").strip()
    if status == "superseded" and not superseded_by:
        raise DocumentMetadataError(
            "doc_status=superseded requires superseded_by; " f"use {SUPERSEDE_ENDPOINT}"
        )
    if status == "retracted":
        if not (isinstance(note, str) and note.strip()):
            raise DocumentMetadataError(
                "doc_status=retracted requires a non-empty correction_note"
            )
    if status == "active" and superseded_by:
        raise DocumentMetadataError(
            "cannot set doc_status=active while superseded_by is set; "
            f"use {UNSUPERSEDE_ENDPOINT}"
        )
    if "doc_status" in removed and superseded_by:
        raise DocumentMetadataError(
            "cannot clear doc_status while superseded_by is set; "
            f"use {UNSUPERSEDE_ENDPOINT}"
        )
    if "correction_note" in removed and status == "retracted":
        raise DocumentMetadataError(
            "cannot clear correction_note while doc_status=retracted"
        )


def supersedes_ids(value: object) -> list[str]:
    """Return de-duplicated document IDs stored in ``supersedes``."""
    if value is None or value == "":
        raw: list[object] = []
    elif isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = list(value)
    else:
        raise DocumentMetadataError("supersedes must be a list of doc_ids")
    out: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def is_hidden_doc_status(value: object) -> bool:
    """Return whether ``value`` should be excluded from search/list defaults.

    Missing, empty, and ``active`` count as visible. ``superseded`` and
    ``retracted`` are hidden unless the caller passes ``include_superseded``.
    """
    text = str(value or "").strip()
    return text in HIDDEN_DOC_STATUSES


def _reject_lifecycle_link_fields(keys: Iterable[object]) -> None:
    """Reject direct writes of ``superseded_by`` / ``supersedes``."""
    found = [str(key) for key in keys if str(key) in LIFECYCLE_LINK_FIELDS]
    if found:
        raise DocumentMetadataError(LIFECYCLE_LINK_PATCH_MESSAGE)


def _jsonable(value: object, *, path: str) -> None:
    """Reject values that cannot round-trip through JSON."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _jsonable(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DocumentMetadataError(f"{path} keys must be strings")
            _jsonable(item, path=f"{path}.{key}")
        return
    raise DocumentMetadataError(f"{path} must be JSON-serializable")


def _clip(text: str, limit: int, *, path: str, truncated: list[str]) -> str:
    """Return ``text`` clipped to ``limit`` characters, recording ``path`` if cut."""
    if len(text) <= limit:
        return text
    truncated.append(path)
    return text[:limit]


def _require_str(value: object, *, path: str, allow_empty: bool = False) -> str:
    """Return a string field, optionally allowing empty."""
    if not isinstance(value, str):
        raise DocumentMetadataError(f"{path} must be a string")
    if not allow_empty and not value.strip():
        raise DocumentMetadataError(f"{path} must be a non-empty string")
    return value


def _enum(value: object, allowed: frozenset[str], *, path: str) -> str:
    """Return ``value`` if it is one of ``allowed``."""
    text = _require_str(value, path=path)
    if text not in allowed:
        choices = ", ".join(sorted(allowed))
        raise DocumentMetadataError(f"{path} must be one of: {choices}")
    return text


def _as_bool(value: object, *, path: str) -> bool:
    """Parse a boolean (including ano/ne and 0/1)."""
    if isinstance(value, bool):
        return value
    if (
        isinstance(value, (int, float))
        and value in (0, 1)
        and not isinstance(value, bool)
    ):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "ano"}:
            return True
        if lowered in {"false", "0", "no", "ne"}:
            return False
    raise DocumentMetadataError(f"{path} must be a boolean")


def _as_int(value: object, *, path: str) -> int:
    """Parse an integer (bools rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DocumentMetadataError(f"{path} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise DocumentMetadataError(f"{path} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DocumentMetadataError(f"{path} must be an integer") from exc


def _as_number(value: object, *, path: str) -> float:
    """Parse a JSON number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DocumentMetadataError(f"{path} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise DocumentMetadataError(f"{path} must be a number") from exc


def _list_or_string(value: object, *, path: str) -> list[str] | str:
    """Return a string or list of strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise DocumentMetadataError(f"{path}[{index}] must be a string")
            out.append(item)
        return out
    raise DocumentMetadataError(f"{path} must be a string or list of strings")


def _normalize_quote(
    value: object,
    *,
    path: str,
    truncated: list[str],
    allow_empty: bool = True,
) -> str:
    """Return a quote/citace string truncated to ``MAX_QUOTE_LEN``."""
    if value is None:
        return ""
    text = _require_str(value, path=path, allow_empty=allow_empty)
    return _clip(text, MAX_QUOTE_LEN, path=path, truncated=truncated)


def normalize_overeni_souhrn(value: object, truncated: list[str] | None = None) -> str:
    """Validate and truncate the flat ``overeni_souhrn`` string (~300 chars)."""
    sink = truncated if truncated is not None else []
    text = _require_str(value, path="overeni_souhrn", allow_empty=True)
    return _clip(
        text.strip(),
        MAX_OVERENI_SOUHRN_LEN,
        path="overeni_souhrn",
        truncated=sink,
    )


def _normalize_verification_item(
    raw: Mapping[str, Any],
    *,
    index: int,
    default_source_key: str | None,
    truncated: list[str],
) -> dict[str, Any]:
    """Return one ``verification[]`` object after alias and enum checks."""
    if not isinstance(raw, Mapping):
        raise DocumentMetadataError(f"verification[{index}] must be an object")
    item: dict[str, Any] = dict(raw)
    if "quote" not in item and "citace" in item:
        item["quote"] = item.pop("citace")
    elif "citace" in item:
        item.pop("citace", None)

    unknown = set(item) - set(VERIFICATION_ITEM_REQUIRED) - VERIFICATION_ITEM_OPTIONAL
    if unknown:
        raise DocumentMetadataError(
            f"verification[{index}] unknown fields: {', '.join(sorted(unknown))}"
        )

    if not item.get("source_key") and default_source_key:
        item["source_key"] = default_source_key

    missing = [key for key in VERIFICATION_ITEM_REQUIRED if key not in item]
    if missing:
        raise DocumentMetadataError(
            f"verification[{index}] missing required fields: {', '.join(missing)}"
        )

    out: dict[str, Any] = {
        "agent": _require_str(item["agent"], path=f"verification[{index}].agent"),
        "check": _enum(
            item["check"], VERIFICATION_CHECKS, path=f"verification[{index}].check"
        ),
        "tool": _require_str(item["tool"], path=f"verification[{index}].tool"),
        "date": _require_str(item["date"], path=f"verification[{index}].date"),
        "result": _enum(
            item["result"],
            VERIFICATION_RESULTS,
            path=f"verification[{index}].result",
        ),
        "source_url": _require_str(
            item["source_url"],
            path=f"verification[{index}].source_url",
            allow_empty=True,
        ),
        "quote": _normalize_quote(
            item.get("quote"),
            path=f"verification[{index}].quote",
            truncated=truncated,
        ),
        "independent": _as_bool(
            item["independent"], path=f"verification[{index}].independent"
        ),
        "source_key": _require_str(
            item["source_key"], path=f"verification[{index}].source_key"
        ),
    }
    if out["result"] == "overeno" and not (
        out["source_url"].strip() and out["quote"].strip()
    ):
        out["result"] = "nejiste"

    if "pozn" in item:
        out["pozn"] = _require_str(
            item["pozn"], path=f"verification[{index}].pozn", allow_empty=True
        )
    if "source_type" in item:
        out["source_type"] = _enum(
            item["source_type"],
            EVIDENCE_SOURCE_TYPES,
            path=f"verification[{index}].source_type",
        )
    if "source_id" in item:
        out["source_id"] = _require_str(
            item["source_id"],
            path=f"verification[{index}].source_id",
            allow_empty=True,
        )
    if "derived_from" in item:
        out["derived_from"] = _require_str(
            item["derived_from"],
            path=f"verification[{index}].derived_from",
            allow_empty=True,
        )
    if "distance_m" in item:
        out["distance_m"] = _as_number(
            item["distance_m"], path=f"verification[{index}].distance_m"
        )
    if "gps_band" in item:
        out["gps_band"] = _enum(
            item["gps_band"],
            GPS_BAND_VALUES,
            path=f"verification[{index}].gps_band",
        )
    if "dating" in item:
        out["dating"] = _enum(
            item["dating"], DATING_VALUES, path=f"verification[{index}].dating"
        )
    if "ref" in item:
        out["ref"] = _require_str(
            item["ref"], path=f"verification[{index}].ref", allow_empty=True
        )
    if "fakt_ref" in item:
        out["fakt_ref"] = _require_str(
            item["fakt_ref"],
            path=f"verification[{index}].fakt_ref",
            allow_empty=True,
        )
    if "method" in item:
        out["method"] = _require_str(
            item["method"], path=f"verification[{index}].method", allow_empty=True
        )
    if "compared_value" in item:
        _jsonable(item["compared_value"], path=f"verification[{index}].compared_value")
        out["compared_value"] = item["compared_value"]
    if "found_value" in item:
        _jsonable(item["found_value"], path=f"verification[{index}].found_value")
        out["found_value"] = item["found_value"]
    if "threshold" in item:
        _jsonable(item["threshold"], path=f"verification[{index}].threshold")
        out["threshold"] = item["threshold"]
    if "evidence_type" in item:
        out["evidence_type"] = _require_str(
            item["evidence_type"],
            path=f"verification[{index}].evidence_type",
            allow_empty=True,
        )
    return out


def split_verification_deletes(
    raw: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Split PATCH metadata into fields to set and whitelist keys to delete.

    JSON ``null`` on a whitelisted key means delete it. ``source_key`` and
    ``content_hash`` may be nulled only when the same body also has
    ``verification: null`` (full verification rollback). ``qg_trida``,
    ``qg_flagy``, and ``qg_verze`` delete without that requirement. Any
    other null (including ``tags``, ``obec_kod``, …) raises
    ``DocumentMetadataError``.
    """
    if raw is None:
        return {}, []
    if not isinstance(raw, Mapping):
        raise DocumentMetadataError("metadata must be an object")
    if "status" in raw:
        raise DocumentMetadataError(STATUS_ALIAS_MESSAGE)
    _reject_lifecycle_link_fields(raw.keys())
    verification_rollback = "verification" in raw and raw["verification"] is None
    remaining: dict[str, Any] = {}
    deleted: list[str] = []
    for key, value in raw.items():
        if value is not None:
            remaining[str(key)] = value
            continue
        name = str(key)
        if name not in DELETABLE_VERIFICATION_FIELDS:
            raise DocumentMetadataError(f"{name} must not be null")
        if name in ROLLBACK_ONLY_DELETE_FIELDS and not verification_rollback:
            raise DocumentMetadataError(
                f"{name} may be null only when verification is also null"
            )
        deleted.append(name)
    return remaining, deleted


def normalize_verification_metadata(
    raw: Mapping[str, Any] | None,
    *,
    apply_defaults: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Return cleaned payload fields and paths of values that were truncated.

    Unknown keys are rejected. ``apply_defaults`` writes ``reliability=1``
    (and ``verification_status=nejiste``) when those keys are absent.
    Truncation keeps the stored value clipped; callers must surface
    ``truncated`` so the cut is not silent.
    """
    if raw is None:
        data: dict[str, Any] = {}
    else:
        if not isinstance(raw, Mapping):
            raise DocumentMetadataError("metadata must be an object")
        data = dict(raw)

    if "status" in data:
        raise DocumentMetadataError(STATUS_ALIAS_MESSAGE)
    _reject_lifecycle_link_fields(data.keys())

    if "source" in data:
        axis = data.pop("source")
        if "source_axis" not in data:
            data["source_axis"] = axis
    if "verified_quote" not in data and "citace" in data:
        data["verified_quote"] = data.pop("citace")
    else:
        data.pop("citace", None)
    if "reliability" not in data and "uroven" in data:
        alias = str(data.pop("uroven")).strip().upper()
        if alias not in UROVEN_TO_RELIABILITY:
            raise DocumentMetadataError("uroven must be one of: A, B, C, D")
        data["reliability"] = UROVEN_TO_RELIABILITY[alias]
    else:
        data.pop("uroven", None)

    if "overil" in data and "verified_by" not in data:
        data["verified_by"] = data["overil"]
    if "overeno_dne" in data and "verified_at" not in data:
        data["verified_at"] = data["overeno_dne"]

    unknown = [
        key
        for key in data
        if key not in FLAT_PAYLOAD_KEYS
        and not str(key).startswith("qg_body_")
        and key not in {"overil", "overeno_dne"}
    ]
    if unknown:
        raise DocumentMetadataError(
            f"unknown metadata fields: {', '.join(sorted(str(k) for k in unknown))}"
        )
    if "doc_id" in data:
        raise DocumentMetadataError("doc_id cannot be changed")

    out: dict[str, Any] = {}
    truncated: list[str] = []
    for key, value in data.items():
        if key in {"overil", "overeno_dne"}:
            # Stored via verified_by / verified_at; keep aliases too when sent.
            if isinstance(value, str):
                out[key] = value
            continue
        if value is None:
            raise DocumentMetadataError(f"{key} must not be null")
        if key == "reliability":
            number = _as_int(value, path=key)
            if number < 0 or number > 4:
                raise DocumentMetadataError("reliability must be an integer 0–4")
            out[key] = number
            continue
        if key in _FLAT_STRING_ENUMS:
            out[key] = _enum(value, _FLAT_STRING_ENUMS[key], path=key)
            continue
        if key == "overeni_souhrn":
            out[key] = normalize_overeni_souhrn(value, truncated=truncated)
            continue
        if key == "correction_note":
            text = _require_str(value, path=key, allow_empty=True)
            out[key] = _clip(
                text.strip(),
                MAX_CORRECTION_NOTE_LEN,
                path=key,
                truncated=truncated,
            )
            continue
        if key == "verified_quote":
            out[key] = _normalize_quote(value, path=key, truncated=truncated)
            continue
        if key in _FLAT_STRINGS:
            out[key] = _require_str(value, path=key, allow_empty=True)
            continue
        if key in _QG_LIST_OR_STRING:
            out[key] = _list_or_string(value, path=key)
            continue
        if key == "qg_skore":
            out[key] = _as_number(value, path=key)
            continue
        if key == "qg_hard_ok":
            out[key] = _as_bool(value, path=key)
            continue
        if key == "relevance":
            number = _as_int(value, path=key)
            if number < 1 or number > 5:
                raise DocumentMetadataError("relevance must be an integer 1–5")
            out[key] = number
            continue
        if key == "obec_kod":
            if isinstance(value, str) and not value.strip().isdigit():
                out[key] = _require_str(value, path=key)
            else:
                out[key] = _as_int(value, path=key)
            continue
        if key == "conflict":
            out[key] = _as_bool(value, path=key)
            continue
        if key == "agreement":
            number = _as_int(value, path=key)
            if number < 0:
                raise DocumentMetadataError("agreement must be >= 0")
            out[key] = number
            continue
        if key in {"ocr_conf", "gps_presnost_m"}:
            out[key] = _as_number(value, path=key)
            continue
        if key == "gps_stav":
            out[key] = _require_str(value, path=key, allow_empty=True)
            continue
        if key == "zdroj":
            _jsonable(value, path=key)
            if not isinstance(value, (str, dict)):
                raise DocumentMetadataError("zdroj must be a string or object")
            out[key] = value
            continue
        if key == "verification":
            if not isinstance(value, list):
                raise DocumentMetadataError("verification must be a list")
            if len(value) > MAX_VERIFICATION_ITEMS:
                raise DocumentMetadataError(
                    f"verification may contain at most {MAX_VERIFICATION_ITEMS} items"
                )
            default_key = None
            source_key = data.get("source_key")
            if isinstance(source_key, str) and source_key.strip():
                default_key = source_key.strip()
            out[key] = [
                _normalize_verification_item(
                    item,
                    index=index,
                    default_source_key=default_key,
                    truncated=truncated,
                )
                for index, item in enumerate(value)
            ]
            continue
        if str(key).startswith("qg_body_"):
            _jsonable(value, path=key)
            out[key] = value
            continue
        raise DocumentMetadataError(f"unknown metadata field: {key}")

    if apply_defaults and "reliability" not in out:
        out["reliability"] = DEFAULT_RELIABILITY
    if (
        apply_defaults
        and "verification_status" not in out
        and "verification" not in out
    ):
        out["verification_status"] = DEFAULT_VERIFICATION_STATUS
    if not out:
        raise DocumentMetadataError("at least one metadata field is required")
    return out, truncated


def chunk_payload_fields(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields written onto text/image knowledge points."""
    return dict(meta)


def figure_payload_fields(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Return document-level fields that are safe on ``pdf_figure`` points."""
    return {
        key: value for key, value in meta.items() if key not in FIGURE_EXCLUDED_KEYS
    }
