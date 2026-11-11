"""Forward locality_links to TrailsAssemblage PostGIS over HTTP.

Enabled only when ``TRAILS_API_URL`` and ``TRAILS_API_TOKEN`` are both set.
The token is sent as ``Authorization: Bearer`` and is never written to logs.
HTTP failures are retried with backoff, then persisted to a JSON outbox on
the RAG data volume and drained on later calls and startup. MCP tools must
not fail because Trails is down.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests
from qdrant_client import QdrantClient

from rag.accounting import get_global_store
from rag.config import (
    DEFAULT_TRAILS_API_TIMEOUT_SECONDS,
    LOCALITY_LINKS_COLLECTION,
    TRAILS_LINKS_BATCH_MAX,
    resolve_trails_api_token,
    resolve_trails_api_url,
    trails_forwarding_enabled,
)
from rag.logging_setup import get_logger
from rag.payloads import chunk_text_sha256

logger = get_logger("rag.trails_sync")

_RETRY_DELAYS = (0.25, 0.75, 2.0)
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_OUTBOX_MAX_DELAY_SECONDS = 300
_HTTP_TIMEOUT = DEFAULT_TRAILS_API_TIMEOUT_SECONDS
_outbox_lock = threading.Lock()

TRAILS_DISABLED: dict[str, Any] = {"enabled": False}


class TrailsSyncError(RuntimeError):
    """Raised when a Trails HTTP call fails after retries."""


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


def _record_outbound(*, ok: bool, error: str | None = None) -> None:
    """Record a Trails HTTP attempt when accounting is configured."""
    store = get_global_store()
    if store is None:
        return
    store.record_outbound(target="trails_api", ok=ok, error=error)


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
    """POST/PATCH Trails with retry/backoff. Never logs ``token``."""
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
                "Trails %s %s attempt %s network error: %s",
                method,
                url,
                attempt + 1,
                last_error,
            )
            _record_outbound(ok=False, error=last_error)
            continue
        if response.status_code in _RETRY_STATUSES:
            last_error = f"Trails HTTP {response.status_code}"
            logger.warning(
                "Trails %s %s attempt %s status %s",
                method,
                url,
                attempt + 1,
                response.status_code,
            )
            _record_outbound(ok=False, error=last_error)
            continue
        if response.status_code >= 400:
            snippet = _safe_response_snippet(response)
            error = f"Trails HTTP {response.status_code}"
            if snippet:
                error = f"{error}: {snippet}"
            _record_outbound(ok=False, error=error[:200])
            raise TrailsSyncError(error)
        _record_outbound(ok=True)
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise TrailsSyncError("Trails returned non-JSON") from exc
    raise TrailsSyncError(last_error or "Trails request failed")


def _geom_for_trails(value: object) -> dict[str, Any] | None:
    """Return a GeoJSON Point dict Trails accepts, or ``None``."""
    if not isinstance(value, Mapping):
        return None
    if value.get("type") != "Point":
        return None
    coords = value.get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        return None
    try:
        lon = float(coords[0])
        lat = float(coords[1])
    except (TypeError, ValueError):
        return None
    out: list[float] = [lon, lat]
    if len(coords) >= 3:
        try:
            out.append(float(coords[2]))
        except (TypeError, ValueError):
            pass
    return {"type": "Point", "coordinates": out}


def _cast_obce_kod(payload: Mapping[str, Any]) -> str | None:
    """Return Trails ``cast_obce_kod`` from payload aliases."""
    kod = payload.get("cast_obce_kod")
    if kod in (None, ""):
        alias = payload.get("cast_obce")
        if isinstance(alias, Mapping):
            kod = alias.get("kod") or alias.get("cast_obce_kod")
        else:
            kod = alias
    text = str(kod).strip() if kod not in (None, "") else ""
    return text or None


def locality_payload_to_trails_upsert(
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Shape a Qdrant locality_links payload into Trails POST item (no link_id)."""
    try:
        obec_id = str(payload.get("obec_id") or "").strip()
        name = str(payload.get("name") or "").strip()
        doc_id = str(payload.get("doc_id") or "").strip()
        matched = str(payload.get("matched_text") or "")
        source_type = str(payload.get("source_type") or "").strip()
        method = str(payload.get("method") or "").strip()
        chunk_index = int(payload["chunk_index"])
        char_start = int(payload["char_start"])
        char_end = int(payload["char_end"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all((obec_id, name, doc_id, matched.strip(), source_type, method)):
        return None
    chunk_sha = str(payload.get("chunk_sha256") or "").strip()
    if not chunk_sha:
        chunk_sha = chunk_text_sha256(matched)
    level = str(payload.get("locality_level") or "obec").strip() or "obec"
    kod = _cast_obce_kod(payload)
    cast_name = str(payload.get("cast_obce_name") or "").strip() or None
    if level != "cast_obce":
        level = "obec"
        kod = None
        cast_name = None
    elif not kod:
        return None
    page = payload.get("page_number")
    try:
        page_number = int(page) if page not in (None, "") else None
    except (TypeError, ValueError):
        page_number = None
    confidence = payload.get("confidence")
    try:
        confidence_f = float(confidence) if confidence not in (None, "") else None
    except (TypeError, ValueError):
        confidence_f = None
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    body: dict[str, Any] = {
        "obec_id": obec_id,
        "name": name,
        "okres": str(payload.get("okres") or ""),
        "kraj": str(payload.get("kraj") or ""),
        "source_dataset": str(payload.get("source_dataset") or "rag-timer"),
        "doc_id": doc_id,
        "parent_document": str(payload.get("parent_document") or "").strip()
        or None,
        "chunk_index": chunk_index,
        "page_number": page_number,
        "point_id": str(payload.get("point_id") or "").strip() or None,
        "source_type": source_type,
        "chunk_sha256": chunk_sha,
        "matched_text": matched,
        "char_start": char_start,
        "char_end": char_end,
        "method": method,
        "ambiguous": bool(payload.get("ambiguous")),
        "candidates": candidates,
        "confidence": confidence_f,
        "confidence_level": str(payload.get("confidence_level") or "").strip()
        or None,
        "notes": str(payload.get("notes") or "").strip() or None,
        "agent_version": str(payload.get("agent_version") or "").strip()
        or None,
        "locality_level": level,
        "cast_obce_kod": kod,
        "cast_obce_name": cast_name,
        "link_type": str(payload.get("link_type") or "mentions_locality"),
    }
    geom = _geom_for_trails(payload.get("geom"))
    if geom is not None:
        body["geom"] = geom
    return body


def _read_outbox(path: Path) -> list[dict[str, Any]]:
    """Load outbox items from ``path``."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Trails outbox unreadable at %s", path)
        return []
    if isinstance(data, dict):
        items = data.get("items")
    else:
        items = data
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _write_outbox(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    """Atomically persist outbox items."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    payload = {"items": [dict(item) for item in items]}
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def enqueue_trails_outbox(
    path: Path,
    link_ids: Sequence[str],
    *,
    error: str | None = None,
) -> int:
    """Queue local link ids for later Trails forwarding. Returns queued count."""
    wanted = [str(link_id).strip() for link_id in link_ids if str(link_id).strip()]
    if not wanted:
        return 0
    with _outbox_lock:
        items = _read_outbox(path)
        pending: set[str] = set()
        for item in items:
            for link_id in item.get("link_ids") or []:
                pending.add(str(link_id))
        new_ids = [link_id for link_id in wanted if link_id not in pending]
        if new_ids:
            items.append(
                {
                    "id": str(uuid.uuid4()),
                    "link_ids": new_ids,
                    "created_at": _utc_now_iso(),
                    "attempts": 0,
                    "next_attempt_at": _utc_now_iso(),
                    "last_error": error,
                }
            )
        elif error and items:
            items[-1]["last_error"] = error
        _write_outbox(path, items)
    if new_ids:
        logger.warning(
            "Queued %s locality_links for Trails outbox (%s pending)",
            len(new_ids),
            error or "retry",
        )
    return len(new_ids)


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


def _retrieve_link_payloads(
    client: QdrantClient,
    link_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Return Qdrant locality_links payloads keyed by point id."""
    if not link_ids or not client.collection_exists(LOCALITY_LINKS_COLLECTION):
        return {}
    found = client.retrieve(
        collection_name=LOCALITY_LINKS_COLLECTION,
        ids=list(link_ids),
        with_payload=True,
        with_vectors=False,
    )
    out: dict[str, dict[str, Any]] = {}
    for point in found:
        payload = dict(point.payload or {})
        payload.setdefault("link_id", str(point.id))
        out[str(point.id)] = payload
    return out


def _set_trails_link_ids(
    client: QdrantClient,
    mapping: Mapping[str, str],
) -> None:
    """Store Trails-generated ids on local points (payload-only)."""
    if not mapping or not client.collection_exists(LOCALITY_LINKS_COLLECTION):
        return
    for local_id, trails_id in mapping.items():
        if not trails_id:
            continue
        try:
            client.set_payload(
                collection_name=LOCALITY_LINKS_COLLECTION,
                payload={"trails_link_id": str(trails_id)},
                points=[local_id],
                wait=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to store trails_link_id on %s: %s",
                local_id,
                _safe_error_text(exc),
            )


def _post_links(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """POST /locality-links batch. ``items`` are Trails upsert bodies."""
    base = resolve_trails_api_url()
    token = resolve_trails_api_token()
    if not base or not token:
        raise TrailsSyncError("Trails forwarding is not configured")
    url = f"{base}/locality-links"
    inserted = 0
    updated = 0
    ids: list[str] = []
    batch: list[Mapping[str, Any]] = []
    for body in items:
        batch.append(body)
        if len(batch) < TRAILS_LINKS_BATCH_MAX:
            continue
        result = _http_request(
            "POST", url, json_body={"links": list(batch)}, token=token
        )
        inserted += int(result.get("inserted") or 0)
        updated += int(result.get("updated") or 0)
        ids.extend(str(item) for item in (result.get("ids") or []))
        batch = []
    if batch:
        result = _http_request(
            "POST", url, json_body={"links": list(batch)}, token=token
        )
        inserted += int(result.get("inserted") or 0)
        updated += int(result.get("updated") or 0)
        ids.extend(str(item) for item in (result.get("ids") or []))
    return {"inserted": inserted, "updated": updated, "ids": ids}


def _patch_verify(
    trails_link_id: str,
    *,
    status: str,
    verified_by: str | None,
    notes: str | None,
) -> dict[str, Any]:
    """PATCH /locality-links/{id}/verify."""
    base = resolve_trails_api_url()
    token = resolve_trails_api_token()
    if not base or not token:
        raise TrailsSyncError("Trails forwarding is not configured")
    url = f"{base}/locality-links/{trails_link_id}/verify"
    body: dict[str, Any] = {"status": status}
    if verified_by is not None:
        body["verified_by"] = verified_by
    if notes is not None:
        body["notes"] = notes
    result = _http_request("PATCH", url, json_body=body, token=token)
    return result if isinstance(result, dict) else {}


def _needs_verify(payload: Mapping[str, Any]) -> bool:
    """Return whether Trails should receive a verify PATCH for this payload."""
    status = str(payload.get("verification_status") or "unverified").strip()
    return status in {"verified", "rejected"}


def _forward_payloads(
    client: QdrantClient,
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """POST a batch then PATCH verification for non-unverified rows."""
    prepared: list[tuple[str, Mapping[str, Any], dict[str, Any]]] = []
    skipped = 0
    for payload in payloads:
        body = locality_payload_to_trails_upsert(payload)
        if body is None:
            skipped += 1
            continue
        prepared.append((str(payload.get("link_id") or ""), payload, body))
    posted: dict[str, Any] = {"inserted": 0, "updated": 0, "ids": []}
    verified = 0
    if prepared:
        posted = _post_links([body for _lid, _payload, body in prepared])
        remote_ids = [str(item) for item in posted.get("ids") or []]
        mapping = {
            local_id: remote_id
            for (local_id, _payload, _body), remote_id in zip(
                prepared, remote_ids, strict=False
            )
            if local_id and remote_id
        }
        _set_trails_link_ids(client, mapping)
        for (_local_id, payload, _body), remote_id in zip(
            prepared, remote_ids, strict=False
        ):
            if not remote_id or not _needs_verify(payload):
                continue
            status = str(payload.get("verification_status") or "").strip()
            verified_by = payload.get("verified_by")
            notes = payload.get("notes")
            _patch_verify(
                remote_id,
                status=status,
                verified_by=str(verified_by) if verified_by else None,
                notes=str(notes) if notes is not None else None,
            )
            verified += 1
    return {
        "forwarded": len(prepared),
        "skipped": skipped,
        "inserted": posted.get("inserted", 0),
        "updated": posted.get("updated", 0),
        "verified": verified,
        "ids": posted.get("ids") or [],
    }


def forward_locality_links(
    client: QdrantClient,
    link_ids: Sequence[str],
    *,
    outbox_path: Path,
) -> dict[str, Any]:
    """Forward existing Qdrant locality_links points to Trails.

    Returns a status dict. Never raises to callers: failures are queued.
    """
    if not trails_forwarding_enabled():
        return dict(TRAILS_DISABLED)
    ids = [str(link_id).strip() for link_id in link_ids if str(link_id).strip()]
    drain = {"drained": 0, "remaining": 0}
    try:
        drain = drain_trails_outbox(client, outbox_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Trails outbox drain skipped: %s", _safe_error_text(exc)
        )
    if not ids:
        return {
            "enabled": True,
            "forwarded": 0,
            "queued": 0,
            "drain": drain,
        }
    payloads_by_id = _retrieve_link_payloads(client, ids)
    ordered = [payloads_by_id[link_id] for link_id in ids if link_id in payloads_by_id]
    missing = [link_id for link_id in ids if link_id not in payloads_by_id]
    try:
        result = _forward_payloads(client, ordered)
        result["enabled"] = True
        result["queued"] = 0
        result["missing"] = missing
        result["drain"] = drain
        return result
    except Exception as exc:  # noqa: BLE001
        error = _safe_error_text(exc)
        queued = enqueue_trails_outbox(outbox_path, ids, error=error)
        logger.warning("Trails forward failed; queued %s links", queued)
        return {
            "enabled": True,
            "ok": False,
            "queued": queued,
            "error": error,
            "drain": drain,
        }


def drain_trails_outbox(client: QdrantClient, outbox_path: Path) -> dict[str, int]:
    """Attempt due outbox items. Returns drained/remaining counts."""
    if not trails_forwarding_enabled():
        return {"drained": 0, "remaining": 0}
    with _outbox_lock:
        items = _read_outbox(outbox_path)
        due, later = _due_items(items)
    drained = 0
    still_due: list[dict[str, Any]] = []
    for item in due:
        link_ids = [
            str(link_id).strip()
            for link_id in (item.get("link_ids") or [])
            if str(link_id).strip()
        ]
        payloads_by_id = _retrieve_link_payloads(client, link_ids)
        ordered = [
            payloads_by_id[link_id]
            for link_id in link_ids
            if link_id in payloads_by_id
        ]
        if not ordered:
            drained += 1
            continue
        try:
            _forward_payloads(client, ordered)
            drained += 1
        except Exception as exc:  # noqa: BLE001
            error = _safe_error_text(exc)
            attempts = int(item.get("attempts") or 0) + 1
            item["attempts"] = attempts
            item["last_error"] = error
            item["next_attempt_at"] = _next_attempt_at(attempts)
            still_due.append(item)
            logger.warning(
                "Trails outbox item %s failed attempt %s: %s",
                item.get("id"),
                attempts,
                error,
            )
    remaining_items = later + still_due
    with _outbox_lock:
        _write_outbox(outbox_path, remaining_items)
    remaining_ids = 0
    for item in remaining_items:
        remaining_ids += len(item.get("link_ids") or [])
    return {
        "drained": drained,
        "remaining": remaining_ids,
        "items": len(remaining_items),
    }


def _iter_all_link_payloads(client: QdrantClient) -> list[dict[str, Any]]:
    """Scroll every locality_links payload."""
    if not client.collection_exists(LOCALITY_LINKS_COLLECTION):
        return []
    payloads: list[dict[str, Any]] = []
    next_offset: Any = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=LOCALITY_LINKS_COLLECTION,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        for point in batch:
            payload = dict(point.payload or {})
            payload.setdefault("link_id", str(point.id))
            payloads.append(payload)
        if next_offset is None:
            break
    return payloads


def sync_links_to_trails(
    client: QdrantClient,
    *,
    outbox_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backfill all locality_links points to Trails (idempotent)."""
    payloads = _iter_all_link_payloads(client)
    would_upsert = 0
    would_verify = 0
    skipped = 0
    for payload in payloads:
        body = locality_payload_to_trails_upsert(payload)
        if body is None:
            skipped += 1
            continue
        would_upsert += 1
        if _needs_verify(payload):
            would_verify += 1
    report: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "enabled": trails_forwarding_enabled(),
        "scanned": len(payloads),
        "would_upsert": would_upsert,
        "would_verify": would_verify,
        "skipped": skipped,
    }
    if dry_run:
        return report
    if not trails_forwarding_enabled():
        report["status"] = "disabled"
        report["error"] = (
            "TRAILS_API_URL and TRAILS_API_TOKEN must both be set"
        )
        return report
    drain = drain_trails_outbox(client, outbox_path)
    try:
        forwarded = _forward_payloads(client, payloads)
    except Exception as exc:  # noqa: BLE001
        error = _safe_error_text(exc)
        ids = [
            str(payload.get("link_id") or "")
            for payload in payloads
            if payload.get("link_id")
        ]
        queued = enqueue_trails_outbox(outbox_path, ids, error=error)
        report["status"] = "queued"
        report["queued"] = queued
        report["error"] = error
        report["drain"] = drain
        return report
    report["status"] = "ok"
    report["inserted"] = forwarded.get("inserted", 0)
    report["updated"] = forwarded.get("updated", 0)
    report["verified"] = forwarded.get("verified", 0)
    report["forwarded"] = forwarded.get("forwarded", 0)
    report["drain"] = drain
    return report


def maybe_drain_on_startup(client: QdrantClient, outbox_path: Path) -> None:
    """Best-effort outbox drain during app startup."""
    if not trails_forwarding_enabled():
        logger.info("Trails locality-link forwarding disabled (env unset)")
        return
    try:
        result = drain_trails_outbox(client, outbox_path)
        logger.info(
            "Trails outbox drain on startup drained=%s remaining=%s",
            result.get("drained"),
            result.get("remaining"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Trails outbox drain on startup failed: %s",
            _safe_error_text(exc),
        )
