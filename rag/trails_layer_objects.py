"""Forward curated ``layer_objects`` to TrailsAssemblage (parallel to Diff Push).

Metadata-only: reads ``DocumentMeta`` / image ``geo_refined`` from disk.
Never scrolls Qdrant vectors. Opt-in via ``TRAILS_LAYER_OBJECTS_FORWARD=1``
plus ``TRAILS_API_URL`` / ``TRAILS_API_TOKEN``. Readers must not call this.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import requests

from rag.app_mode import is_reader_mode
from rag.config import (
    DEFAULT_TRAILS_API_TIMEOUT_SECONDS,
    TRAILS_OBJECTS_BATCH_MAX,
    resolve_trails_api_token,
    resolve_trails_api_url,
    trails_layer_objects_forwarding_enabled,
)
from rag.logging_setup import get_logger
from rag.storage import DocumentMeta, ImageMeta, RagStorage
from rag.verification_meta import HIDDEN_DOC_STATUSES

logger = get_logger("rag.trails_layer_objects")

_RETRY_DELAYS = (0.25, 0.75, 2.0)
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_OUTBOX_MAX_DELAY_SECONDS = 300
_HTTP_TIMEOUT = DEFAULT_TRAILS_API_TIMEOUT_SECONDS
_objects_outbox_thread_lock = threading.Lock()
_ack_lock = threading.Lock()


@contextmanager
def objects_outbox_lock(path: Path) -> Iterator[None]:
    """Exclusive lock for ``trails_objects_outbox.json`` (cross-process + threads).

    Uses ``fcntl.flock`` on a sibling ``.lock`` file so concurrent processes
    serialize read-modify-write, plus an in-process threading lock so nested
    same-thread callers still coalesce safely. Callers must keep the atomic
    temp+rename write path.
    """
    outbox = Path(path)
    outbox.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(outbox) + ".lock")
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with _objects_outbox_thread_lock:
            yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


TRAILS_OBJECTS_DISABLED: dict[str, Any] = {"enabled": False}

_GEO_OK_STATUS = frozenset({"overeno", "zpresneno"})
_GEO_OK_PRECISION = frozenset({"building", "street"})
_MIN_CONFIDENCE = 0.8
SECESE_GATE_TERMS: tuple[str, ...] = (
    "secese",
    "art nouveau",
    "art-nouveau",
    "artnouveau",
    "jugendstil",
    "sezession",
)
_ACK_SEP = "\x1f"

PR_STAV_BUDOVA = "budova"
PR_STAV_FASADA = "fasáda"


class TrailsLayerObjectsError(RuntimeError):
    """Raised when a Trails layer_objects HTTP call fails after retries."""


def _utc_now() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return _utc_now().isoformat(timespec="seconds")


def _parse_iso(value: str) -> datetime | None:
    """Parse an ISO timestamp; ``None`` if invalid."""
    text = (value or "").strip()
    if not text:
        return None
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _safe_error_text(exc: BaseException) -> str:
    """Format an exception without request headers or tokens."""
    text = f"{type(exc).__name__}: {exc}"
    token = resolve_trails_api_token()
    if token and token in text:
        text = text.replace(token, "***")
    return text[:500]


def _safe_response_snippet(response: requests.Response) -> str:
    """Return a short response body snippet with any token redacted."""
    try:
        text = (response.text or "")[:300]
    except Exception:  # noqa: BLE001
        text = ""
    token = resolve_trails_api_token()
    if token and token in text:
        text = text.replace(token, "***")
    return text


def _http_request(
    method: str,
    url: str,
    *,
    json_body: Mapping[str, Any] | None,
    token: str,
) -> Any:
    """POST Trails with retry/backoff. Never logs ``token``."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    last_error: str | None = None
    delays = (0.0,) + _RETRY_DELAYS
    for attempt, delay in enumerate(delays):
        if delay:
            time.sleep(delay)
        try:
            response = requests.request(
                method,
                url,
                json=dict(json_body) if json_body is not None else None,
                headers=headers,
                timeout=_HTTP_TIMEOUT,
            )
        except requests.RequestException as exc:
            last_error = _safe_error_text(exc)
            logger.warning(
                "Trails objects %s %s attempt %s network error: %s",
                method,
                url,
                attempt + 1,
                last_error,
            )
            continue
        if response.status_code in _RETRY_STATUSES:
            last_error = f"Trails HTTP {response.status_code}"
            logger.warning(
                "Trails objects %s %s attempt %s status %s",
                method,
                url,
                attempt + 1,
                response.status_code,
            )
            continue
        if response.status_code >= 400:
            snippet = _safe_response_snippet(response)
            error = f"Trails HTTP {response.status_code}"
            if snippet:
                error = f"{error}: {snippet}"
            raise TrailsLayerObjectsError(error)
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise TrailsLayerObjectsError("Trails returned non-JSON") from exc
    raise TrailsLayerObjectsError(last_error or "Trails objects request failed")


def _fold(text: str) -> str:
    """Strip diacritics and casefold for gate term matching."""
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return stripped.casefold()


_TAG_FIELD_KEYS = (
    "tags",
    "tag",
    "labels",
    "categories",
    "category",
    "kind",
    "type",
)


def _iter_tag_strings(extra: Mapping[str, Any] | None) -> list[str]:
    """Collect string tags / labels from ``extra`` or a top-level meta mapping."""
    if not isinstance(extra, Mapping):
        return []
    out: list[str] = []
    for key in _TAG_FIELD_KEYS:
        value = extra.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    out.append(item)
    return out


def _extra_title(extra: Mapping[str, Any] | None) -> str | None:
    """Return a human title from document or image ``extra``."""
    if not isinstance(extra, Mapping):
        return None
    for key in ("title", "name", "caption", "label", "hlavni_prvek"):
        value = extra.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _text_matches_secese_terms(text: str) -> bool:
    """Return True when ``text`` contains a :data:`SECESE_GATE_TERMS` substring."""
    folded = _fold(text)
    if not folded:
        return False
    compact = folded.replace(" ", "").replace("_", "").replace("-", "")
    words = folded.replace("-", " ").replace("_", " ").split()
    for needle in SECESE_GATE_TERMS:
        nfold = _fold(needle)
        ncompact = nfold.replace(" ", "").replace("_", "").replace("-", "")
        if ncompact in compact or nfold in folded:
            return True
        if nfold == "secese":
            for word in words:
                if word.startswith("seces") and len(word) >= 5:
                    return True
    return False


def has_secese_signal(
    *extras: Mapping[str, Any] | None,
    meta_root: Mapping[str, Any] | None = None,
    title: str | None = None,
    filename: str | None = None,
) -> bool:
    """Return True when tags, ``title``, or ``filename`` mention secese / art nouveau."""
    blobs: list[str] = []
    for extra in extras:
        blobs.extend(_iter_tag_strings(extra))
    if meta_root is not None:
        blobs.extend(_iter_tag_strings(meta_root))
    if blobs:
        tag_blob = " ".join(_fold(item) for item in blobs)
        if _text_matches_secese_terms(tag_blob):
            return True
    if title and title.strip() and _text_matches_secese_terms(title):
        return True
    if filename and str(filename).strip():
        name = Path(str(filename).strip()).name
        if _text_matches_secese_terms(name):
            return True
    return False


def geo_qualifies(geo: Mapping[str, Any] | None) -> bool:
    """Return True when ``geo_refined`` meets v1 status/confidence/precision."""
    if not isinstance(geo, Mapping):
        return False
    status = str(geo.get("status") or "").strip()
    if status not in _GEO_OK_STATUS:
        return False
    precision = str(geo.get("precision") or "").strip()
    if precision not in _GEO_OK_PRECISION:
        return False
    raw_confidence = geo.get("confidence")
    if raw_confidence is None or raw_confidence == "":
        return False
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return False
    return confidence >= _MIN_CONFIDENCE


def _doc_status(meta: DocumentMeta) -> str:
    """Return lifecycle ``doc_status`` from meta/extra (default active)."""
    extra = meta.extra if isinstance(meta.extra, dict) else {}
    raw = extra.get("doc_status") or getattr(meta, "doc_status", None)
    text = str(raw or "active").strip().lower()
    return text or "active"


def _doc_hidden(meta: DocumentMeta) -> bool:
    """Return True when the document must not appear as a curated object."""
    return _doc_status(meta) in HIDDEN_DOC_STATUSES


def _label_for(meta: DocumentMeta, image: ImageMeta | None = None) -> str:
    """Best-effort ``hlavni_prvek`` label."""
    if image is not None:
        img_extra = image.extra if isinstance(image.extra, dict) else {}
        for key in ("title", "name", "caption", "label"):
            value = img_extra.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    extra = meta.extra if isinstance(meta.extra, dict) else {}
    for key in ("title", "name", "label", "hlavni_prvek"):
        value = extra.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    stem = Path(str(meta.filename or "")).stem.strip()
    if stem:
        return stem
    return str(meta.doc_id)


def ack_key(rag_doc_id: str, rag_image_id: str | None = None) -> str:
    """Stable ack / outbox identity for one Trails object."""
    doc = str(rag_doc_id).strip()
    image = str(rag_image_id or "").strip()
    return f"{doc}{_ACK_SEP}{image}" if image else doc


def object_fingerprint(body: Mapping[str, Any]) -> str:
    """Hash contract fields used for skip / idempotency."""
    payload = {
        "op": body.get("op"),
        "rag_doc_id": body.get("rag_doc_id"),
        "rag_image_id": body.get("rag_image_id"),
        "lon": body.get("lon"),
        "lat": body.get("lat"),
        "hlavni_prvek": body.get("hlavni_prvek"),
        "pr_stav_nazev": body.get("pr_stav_nazev"),
        "geo_status": body.get("geo_status"),
        "geo_confidence": body.get("geo_confidence"),
        "geo_precision": body.get("geo_precision"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _upsert_body(
    *,
    meta: DocumentMeta,
    geo: Mapping[str, Any],
    pr_stav_nazev: str,
    rag_image_id: str | None,
) -> dict[str, Any]:
    """Build one Trails upsert object (fingerprint filled by caller)."""
    return {
        "op": "upsert",
        "rag_doc_id": str(meta.doc_id),
        "rag_image_id": rag_image_id,
        "lon": float(geo["lon"]),
        "lat": float(geo["lat"]),
        "hlavni_prvek": _label_for(
            meta,
            (
                next((img for img in meta.images if img.image_id == rag_image_id), None)
                if rag_image_id
                else None
            ),
        ),
        "pr_stav_nazev": pr_stav_nazev,
        "geo_status": str(geo.get("status") or ""),
        "geo_confidence": float(geo["confidence"]),
        "geo_precision": str(geo.get("precision") or ""),
    }


def _absent_body(rag_doc_id: str, rag_image_id: str | None) -> dict[str, Any]:
    """Build one Trails soft-absent object."""
    return {
        "op": "absent",
        "rag_doc_id": str(rag_doc_id),
        "rag_image_id": rag_image_id,
        "lon": None,
        "lat": None,
        "hlavni_prvek": None,
        "pr_stav_nazev": None,
        "geo_status": None,
        "geo_confidence": None,
        "geo_precision": None,
    }


def build_layer_object_ops(
    meta: DocumentMeta | None,
    *,
    previously_acked: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return upsert/absent bodies for one document (metadata only).

    Facades = one object per image with qualifying ``geo_refined``.
    Buildings = document-level qualifying ``geo_refined`` with secese signal.
    """
    acked = dict(previously_acked or {})
    wanted: dict[str, dict[str, Any]] = {}
    if meta is not None and not _doc_hidden(meta):
        doc_extra = meta.extra if isinstance(meta.extra, dict) else {}
        doc_root = meta.to_dict()
        secese_doc = has_secese_signal(
            doc_extra,
            meta_root=doc_root,
            title=_extra_title(doc_extra),
            filename=meta.filename,
        )
        if geo_qualifies(meta.geo_refined) and secese_doc:
            body = _upsert_body(
                meta=meta,
                geo=meta.geo_refined or {},
                pr_stav_nazev=PR_STAV_BUDOVA,
                rag_image_id=None,
            )
            body["fingerprint"] = object_fingerprint(body)
            wanted[ack_key(meta.doc_id)] = body
        for image in meta.images:
            img_extra = image.extra if isinstance(image.extra, dict) else {}
            if not (
                geo_qualifies(image.geo_refined)
                and has_secese_signal(
                    img_extra,
                    doc_extra,
                    meta_root={**doc_root, **image.to_dict()},
                    title=_extra_title(img_extra) or _extra_title(doc_extra),
                    filename=image.file_path,
                )
            ):
                continue
            body = _upsert_body(
                meta=meta,
                geo=image.geo_refined or {},
                pr_stav_nazev=PR_STAV_FASADA,
                rag_image_id=str(image.image_id),
            )
            body["fingerprint"] = object_fingerprint(body)
            wanted[ack_key(meta.doc_id, image.image_id)] = body

    doc_id = str(meta.doc_id) if meta is not None else None
    ops: list[dict[str, Any]] = []
    # Soft-absent prior keys that no longer qualify (or whole doc gone/hidden).
    for key, _old_fp in acked.items():
        if doc_id is not None and not (
            key == doc_id or key.startswith(f"{doc_id}{_ACK_SEP}")
        ):
            continue
        if key in wanted:
            continue
        if _ACK_SEP in key:
            rag_doc_id, rag_image_id = key.split(_ACK_SEP, 1)
        else:
            rag_doc_id, rag_image_id = key, None
        if doc_id is None:
            # Caller must pass doc_id via key scan; skip unrelated.
            continue
        body = _absent_body(rag_doc_id, rag_image_id or None)
        body["fingerprint"] = object_fingerprint(body)
        ops.append(body)
    ops.extend(wanted.values())
    return ops


def _read_json_file(path: Path) -> Any:
    """Load JSON from ``path`` or return ``None``."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Trails objects file unreadable at %s", path)
        return None


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def read_acks(path: Path) -> dict[str, str]:
    """Return ack_key → fingerprint map."""
    data = _read_json_file(path)
    if not isinstance(data, dict):
        return {}
    raw = data.get("acks") if "acks" in data else data
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, str) and key and value:
            out[key] = value
    return out


def write_acks(path: Path, acks: Mapping[str, str]) -> None:
    """Persist ack fingerprints."""
    with _ack_lock:
        _write_json_file(path, {"acks": dict(acks)})


def _update_acks_after_success(
    path: Path,
    bodies: Sequence[Mapping[str, Any]],
) -> None:
    """Apply successful upsert/absent ops to the ack file."""
    with _ack_lock:
        acks = read_acks(path)
        for body in bodies:
            doc_id = str(body.get("rag_doc_id") or "").strip()
            if not doc_id:
                continue
            image_id = body.get("rag_image_id")
            image = str(image_id).strip() if image_id not in (None, "") else ""
            key = ack_key(doc_id, image or None)
            op = str(body.get("op") or "").strip()
            fingerprint = str(body.get("fingerprint") or "").strip()
            if op == "absent":
                acks.pop(key, None)
            elif op == "upsert" and fingerprint:
                acks[key] = fingerprint
        _write_json_file(path, {"acks": acks})


def _read_objects_outbox(path: Path) -> list[dict[str, Any]]:
    """Load object outbox items."""
    data = _read_json_file(path)
    if data is None:
        return []
    if isinstance(data, dict):
        items = data.get("items")
    else:
        items = data
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _write_objects_outbox(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    """Persist object outbox items."""
    _write_json_file(path, {"items": [dict(item) for item in items]})


def enqueue_objects_outbox(
    path: Path,
    bodies: Sequence[Mapping[str, Any]],
    *,
    error: str | None = None,
) -> int:
    """Queue Trails object ops. Coalesce by ack key (last body wins)."""
    wanted = [dict(body) for body in bodies if isinstance(body, Mapping)]
    if not wanted:
        return 0
    fresh_keys = {
        ack_key(
            str(body.get("rag_doc_id") or ""),
            str(body.get("rag_image_id") or "") or None,
        )
        for body in wanted
    }
    by_key = {
        ack_key(
            str(body.get("rag_doc_id") or ""),
            str(body.get("rag_image_id") or "") or None,
        ): dict(body)
        for body in wanted
    }
    with objects_outbox_lock(path):
        items = _read_objects_outbox(path)
        cleaned: list[dict[str, Any]] = []
        for item in items:
            kept = []
            for body in item.get("objects") or []:
                if not isinstance(body, Mapping):
                    continue
                key = ack_key(
                    str(body.get("rag_doc_id") or ""),
                    str(body.get("rag_image_id") or "") or None,
                )
                if key in fresh_keys:
                    continue
                kept.append(dict(body))
            if kept:
                updated = dict(item)
                updated["objects"] = kept
                cleaned.append(updated)
        cleaned.append(
            {
                "id": str(uuid.uuid4()),
                "objects": list(by_key.values()),
                "created_at": _utc_now_iso(),
                "attempts": 0,
                "next_attempt_at": _utc_now_iso(),
                "last_error": error,
            }
        )
        _write_objects_outbox(path, cleaned)
    logger.warning(
        "Queued %s Trails layer_objects ops (%s)",
        len(wanted),
        error or "retry",
    )
    return len(wanted)


def _next_attempt_at(attempts: int) -> str:
    """Exponential backoff cap for outbox retries."""
    delay = min(2 ** max(attempts, 0), _OUTBOX_MAX_DELAY_SECONDS)
    return (_utc_now() + timedelta(seconds=delay)).isoformat(timespec="seconds")


def _due_items(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split outbox into due vs future items."""
    now = _utc_now()
    due: list[dict[str, Any]] = []
    later: list[dict[str, Any]] = []
    for item in items:
        when = _parse_iso(str(item.get("next_attempt_at") or ""))
        if when is None or when <= now:
            due.append(item)
        else:
            later.append(item)
    return due, later


def _filter_unchanged(
    bodies: Sequence[Mapping[str, Any]],
    acks: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Drop upsert/absent ops whose fingerprint matches the ack file."""
    out: list[dict[str, Any]] = []
    for body in bodies:
        key = ack_key(
            str(body.get("rag_doc_id") or ""),
            str(body.get("rag_image_id") or "") or None,
        )
        fingerprint = str(body.get("fingerprint") or "")
        if fingerprint and acks.get(key) == fingerprint and body.get("op") == "upsert":
            continue
        if (
            body.get("op") == "absent"
            and key not in acks
            and fingerprint
            and acks.get(key) is None
        ):
            # Already absent / never acked — skip.
            continue
        if body.get("op") == "absent" and key not in acks:
            continue
        out.append(dict(body))
    return out


# TrailsAssemblage #44 docs/catalog-rag-objects.md per-item statuses.
_ACK_OK_STATUSES = frozenset(
    {
        "inserted",
        "updated",
        "unchanged",
        "marked_absent",
        "already_absent",
        "not_found",
    }
)


def _body_object_key(body: Mapping[str, Any]) -> str:
    """Ack / coalesce key for one catalog object body."""
    return ack_key(
        str(body.get("rag_doc_id") or ""),
        str(body.get("rag_image_id") or "") or None,
    )


def coalesce_outbox_items(
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge outbox rows; last body per object key wins (fingerprint order)."""
    by_key: dict[str, dict[str, Any]] = {}
    max_attempts = 0
    last_error: str | None = None
    next_attempt_at: str | None = None
    for item in items:
        if not isinstance(item, Mapping):
            continue
        try:
            max_attempts = max(max_attempts, int(item.get("attempts") or 0))
        except (TypeError, ValueError):
            pass
        err = item.get("last_error")
        if isinstance(err, str) and err.strip():
            last_error = err.strip()
        nat = item.get("next_attempt_at")
        if isinstance(nat, str) and nat.strip():
            next_attempt_at = nat.strip()
        for body in item.get("objects") or []:
            if isinstance(body, Mapping) and body.get("rag_doc_id"):
                by_key[_body_object_key(body)] = dict(body)
    if not by_key:
        return []
    return [
        {
            "id": str(uuid.uuid4()),
            "objects": list(by_key.values()),
            "created_at": _utc_now_iso(),
            "attempts": max_attempts,
            "next_attempt_at": next_attempt_at or _utc_now_iso(),
            "last_error": last_error,
        }
    ]


def classify_catalog_response(
    sent: Sequence[Mapping[str, Any]],
    response: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a Trails batch response into acked / retry / invalid bodies.

    Ack only ``items[].status`` in :data:`_ACK_OK_STATUSES`. ``invalid`` is
    logged and dropped (not acked, not requeued — retry will not help).
    Missing or unknown statuses are retried via outbox.
    """
    acked: list[dict[str, Any]] = []
    retry: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    raw_items = response.get("items") if isinstance(response, Mapping) else None
    if not isinstance(raw_items, list):
        return [], [dict(body) for body in sent], []
    for idx, body in enumerate(sent):
        entry = raw_items[idx] if idx < len(raw_items) else None
        if not isinstance(entry, Mapping):
            retry.append(dict(body))
            continue
        status = str(entry.get("status") or "").strip()
        if status in _ACK_OK_STATUSES:
            acked.append(dict(body))
            continue
        if status == "invalid":
            invalid.append(dict(body))
            detail = entry.get("detail")
            logger.warning(
                "Trails catalog object invalid rag_doc_id=%s rag_image_id=%s detail=%s",
                body.get("rag_doc_id"),
                body.get("rag_image_id"),
                detail,
            )
            continue
        retry.append(dict(body))
        logger.warning(
            "Trails catalog object unexpected status=%s rag_doc_id=%s; will retry",
            status or "(missing)",
            body.get("rag_doc_id"),
        )
    return acked, retry, invalid


def _post_objects(bodies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """POST ``/catalog/objects`` with ``{items:[...]}`` (Trails #44 contract).

    Returns acked / retry / invalid partitions plus status counts. Raises on
    transport or non-2xx after retries (caller enqueues the whole batch).
    """
    base = resolve_trails_api_url()
    token = resolve_trails_api_token()
    if not base or not token:
        raise TrailsLayerObjectsError(
            "Trails layer_objects forwarding is not configured"
        )
    auth_token: str = token
    url = f"{base}/catalog/objects"
    acked_all: list[dict[str, Any]] = []
    retry_all: list[dict[str, Any]] = []
    invalid_all: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    def _flush(batch: list[Mapping[str, Any]]) -> None:
        result = _http_request(
            "POST",
            url,
            json_body={"items": [dict(item) for item in batch]},
            token=auth_token,
        )
        payload = result if isinstance(result, dict) else {}
        acked, retry, invalid = classify_catalog_response(batch, payload)
        acked_all.extend(acked)
        retry_all.extend(retry)
        invalid_all.extend(invalid)
        for entry in payload.get("items") or []:
            if not isinstance(entry, Mapping):
                continue
            status = str(entry.get("status") or "").strip() or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1

    batch: list[Mapping[str, Any]] = []
    for body in bodies:
        batch.append(body)
        if len(batch) < TRAILS_OBJECTS_BATCH_MAX:
            continue
        _flush(batch)
        batch = []
    if batch:
        _flush(batch)
    return {
        "acked": acked_all,
        "retry": retry_all,
        "invalid": invalid_all,
        "statuses": status_counts,
        "forwarded": len(acked_all),
        "queued": len(retry_all),
        "dropped_invalid": len(invalid_all),
    }


def drain_objects_outbox(
    *,
    outbox_path: Path,
    ack_path: Path,
) -> dict[str, int]:
    """Attempt due object outbox items with concurrency-safe RMW merge.

    Claims due items under the lock (leaves non-due on disk), POSTs without
    holding the lock, then re-reads and coalesces with anything enqueued
    during the HTTP call (last fingerprint per object key wins).
    """
    if not trails_layer_objects_forwarding_enabled() or is_reader_mode():
        return {"drained": 0, "remaining": 0}
    with objects_outbox_lock(outbox_path):
        items = _read_objects_outbox(outbox_path)
        due, later = _due_items(items)
        _write_objects_outbox(outbox_path, later)
    drained = 0
    still_due: list[dict[str, Any]] = []
    for item in due:
        bodies = [
            dict(body)
            for body in (item.get("objects") or [])
            if isinstance(body, Mapping)
        ]
        if not bodies:
            drained += 1
            continue
        try:
            outcome = _post_objects(bodies)
            acked = outcome.get("acked") or []
            if acked:
                _update_acks_after_success(ack_path, acked)
            retry_bodies = list(outcome.get("retry") or [])
            if retry_bodies:
                attempts = int(item.get("attempts") or 0) + 1
                still_due.append(
                    {
                        "id": str(item.get("id") or uuid.uuid4()),
                        "objects": retry_bodies,
                        "created_at": item.get("created_at") or _utc_now_iso(),
                        "attempts": attempts,
                        "next_attempt_at": _next_attempt_at(attempts),
                        "last_error": "retry_after_partial_or_unknown_status",
                    }
                )
            else:
                drained += 1
        except Exception as exc:  # noqa: BLE001
            error = _safe_error_text(exc)
            attempts = int(item.get("attempts") or 0) + 1
            failed = dict(item)
            failed["attempts"] = attempts
            failed["last_error"] = error
            failed["next_attempt_at"] = _next_attempt_at(attempts)
            still_due.append(failed)
            logger.warning(
                "Trails objects outbox item %s failed attempt %s: %s",
                item.get("id"),
                attempts,
                error,
            )
    with objects_outbox_lock(outbox_path):
        current = _read_objects_outbox(outbox_path)
        remaining_items = coalesce_outbox_items(list(current) + still_due)
        _write_objects_outbox(outbox_path, remaining_items)
    remaining_ops = 0
    for item in remaining_items:
        remaining_ops += len(item.get("objects") or [])
    return {
        "drained": drained,
        "remaining": remaining_ops,
        "items": len(remaining_items),
    }


def _collect_ops_for_docs(
    storage: RagStorage,
    doc_ids: Sequence[str],
    *,
    ack_path: Path,
) -> list[dict[str, Any]]:
    """Build fingerprint-filtered ops for ``doc_ids`` (storage metadata only)."""
    acks = read_acks(ack_path)
    ops: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        trimmed = str(doc_id or "").strip()
        if not trimmed:
            continue
        meta = storage.read_meta(trimmed)
        doc_acks = {
            key: value
            for key, value in acks.items()
            if key == trimmed or key.startswith(f"{trimmed}{_ACK_SEP}")
        }
        if meta is None:
            # Document removed from disk: soft-absent every prior ack key.
            for key in doc_acks:
                if _ACK_SEP in key:
                    rag_doc_id, rag_image_id = key.split(_ACK_SEP, 1)
                else:
                    rag_doc_id, rag_image_id = key, None
                body = _absent_body(rag_doc_id, rag_image_id or None)
                body["fingerprint"] = object_fingerprint(body)
                ops.append(body)
            continue
        ops.extend(build_layer_object_ops(meta, previously_acked=doc_acks))
    return _filter_unchanged(ops, acks)


def maybe_forward_layer_objects(
    storage: RagStorage,
    *,
    doc_ids: Sequence[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Best-effort Trails ``layer_objects`` forward (never raises).

    Parallel to Diff Push: callers must not await push result. Readers and
    disabled flags return ``enabled=False``.
    """
    if is_reader_mode():
        return {**TRAILS_OBJECTS_DISABLED, "reason": "reader_mode"}
    if not trails_layer_objects_forwarding_enabled():
        return dict(TRAILS_OBJECTS_DISABLED)
    outbox_path = storage.trails_objects_outbox_path
    ack_path = storage.trails_objects_ack_path
    drain: dict[str, Any] = {"drained": 0, "remaining": 0}
    try:
        drain = drain_objects_outbox(outbox_path=outbox_path, ack_path=ack_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Trails objects outbox drain skipped: %s", _safe_error_text(exc))
    ids = [str(doc_id).strip() for doc_id in (doc_ids or []) if str(doc_id).strip()]
    if not ids:
        return {
            "enabled": True,
            "forwarded": 0,
            "skipped": 0,
            "queued": 0,
            "dry_run": bool(dry_run),
            "drain": drain,
        }
    try:
        ops = _collect_ops_for_docs(storage, ids, ack_path=ack_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Trails layer_objects build failed: %s", _safe_error_text(exc))
        return {
            "enabled": True,
            "ok": False,
            "error": _safe_error_text(exc),
            "drain": drain,
        }
    if dry_run:
        return {
            "enabled": True,
            "dry_run": True,
            "would_forward": len(ops),
            "ops": ops,
            "drain": drain,
        }
    if not ops:
        return {
            "enabled": True,
            "forwarded": 0,
            "skipped": len(ids),
            "queued": 0,
            "drain": drain,
        }
    try:
        posted = _post_objects(ops)
        acked = list(posted.get("acked") or [])
        retry_bodies = list(posted.get("retry") or [])
        if acked:
            _update_acks_after_success(ack_path, acked)
        queued = 0
        if retry_bodies:
            queued = enqueue_objects_outbox(
                outbox_path,
                retry_bodies,
                error="retry_after_partial_or_unknown_status",
            )
        return {
            "enabled": True,
            "ok": not retry_bodies,
            "forwarded": len(acked),
            "queued": queued,
            "dropped_invalid": int(posted.get("dropped_invalid") or 0),
            "statuses": posted.get("statuses") or {},
            "drain": drain,
        }
    except Exception as exc:  # noqa: BLE001
        error = _safe_error_text(exc)
        queued = enqueue_objects_outbox(outbox_path, ops, error=error)
        logger.warning("Trails layer_objects forward failed; queued %s ops", queued)
        return {
            "enabled": True,
            "ok": False,
            "queued": queued,
            "error": error,
            "drain": drain,
        }


def sync_layer_objects_to_trails(
    storage: RagStorage,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backfill all catalog docs to Trails layer_objects (idempotent)."""
    if is_reader_mode():
        return {
            "dry_run": bool(dry_run),
            "enabled": False,
            "status": "disabled",
            "reason": "reader_mode",
        }
    docs = storage.list_documents()
    doc_ids = [
        str(item.get("doc_id") or "").strip()
        for item in docs
        if isinstance(item, dict) and item.get("doc_id")
    ]
    if dry_run:
        # Compute ops even when the forward flag is off so operators can preview.
        acks = read_acks(storage.trails_objects_ack_path)
        would: list[dict[str, Any]] = []
        for doc_id in doc_ids:
            meta = storage.read_meta(doc_id)
            doc_acks = {
                key: value
                for key, value in acks.items()
                if key == doc_id or key.startswith(f"{doc_id}{_ACK_SEP}")
            }
            if meta is None:
                continue
            would.extend(build_layer_object_ops(meta, previously_acked=doc_acks))
        would = _filter_unchanged(would, acks)
        return {
            "dry_run": True,
            "enabled": trails_layer_objects_forwarding_enabled(),
            "scanned": len(doc_ids),
            "would_forward": len(would),
            "ops": would,
            "status": "dry_run",
        }
    if not trails_layer_objects_forwarding_enabled():
        return {
            "dry_run": False,
            "enabled": False,
            "status": "disabled",
            "error": (
                "TRAILS_LAYER_OBJECTS_FORWARD=1 and TRAILS_API_URL/TOKEN required"
            ),
            "scanned": len(doc_ids),
        }
    return {
        **maybe_forward_layer_objects(storage, doc_ids=doc_ids, dry_run=False),
        "scanned": len(doc_ids),
        "status": "ok",
        "dry_run": False,
    }


def maybe_drain_objects_on_startup(storage: RagStorage) -> None:
    """Best-effort objects outbox drain during app startup."""
    if is_reader_mode() or not trails_layer_objects_forwarding_enabled():
        logger.info("Trails layer_objects forwarding disabled")
        return
    try:
        result = drain_objects_outbox(
            outbox_path=storage.trails_objects_outbox_path,
            ack_path=storage.trails_objects_ack_path,
        )
        logger.info(
            "Trails objects outbox drain on startup drained=%s remaining=%s",
            result.get("drained"),
            result.get("remaining"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Trails objects outbox drain on startup failed: %s",
            _safe_error_text(exc),
        )


def doc_ids_from_mutate_result(result: Mapping[str, Any] | None) -> list[str]:
    """Extract document ids from a mutator response for targeted forward."""
    if not isinstance(result, Mapping):
        return []
    found: list[str] = []
    for key in (
        "doc_id",
        "old_doc_id",
        "new_doc_id",
        "superseded_doc_id",
        "supersedes",
        "superseded_by",
    ):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            found.append(value.strip())
    raw_ids = result.get("doc_ids")
    if isinstance(raw_ids, (list, tuple)):
        for item in raw_ids:
            if isinstance(item, str) and item.strip():
                found.append(item.strip())
    figure = result.get("figure")
    if isinstance(figure, Mapping):
        fig_doc = figure.get("doc_id")
        if isinstance(fig_doc, str) and fig_doc.strip():
            found.append(fig_doc.strip())
    # Dedupe preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in found:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out
