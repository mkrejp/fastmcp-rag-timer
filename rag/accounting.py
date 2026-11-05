"""Per-day token and communications accounting under ``/data/accnt``."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

EMBEDDING_SOURCES = ("ingest_http", "ingest_mcp", "search_mcp", "other")
MCP_TOOLS = (
    "list_documents",
    "get_document_text",
    "get_document_chunks",
    "ingest_text",
    "search_domain_knowledge",
)
HTTP_ROUTES = (
    "upload",
    "upload_pdf",
    "upload_media",
    "ingest_json",
    "documents_list",
    "documents_get",
    "documents_text",
    "documents_delete",
    "health",
)
OUTBOUND_TARGETS = ("novita_embeddings", "remote_fetch")

_accounting_source: ContextVar[str] = ContextVar("accounting_source", default="other")
_global_store: AccountingStore | None = None
_global_lock = threading.Lock()


def utc_today() -> str:
    """Return today's UTC date as ``YYYY-MM-DD``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_accounting_source() -> str:
    """Return the current embedding attribution source from context."""
    return _accounting_source.get()


@contextmanager
def accounting_source(source: str) -> Iterator[None]:
    """Temporarily set the embedding attribution source for nested calls."""
    token = _accounting_source.set(source)
    try:
        yield
    finally:
        _accounting_source.reset(token)


def set_global_store(store: AccountingStore | None) -> None:
    """Register or clear the process-global accounting store."""
    global _global_store
    with _global_lock:
        _global_store = store


def get_global_store() -> AccountingStore | None:
    """Return the process-global accounting store, if configured."""
    return _global_store


def _empty_embed_counters() -> dict[str, int]:
    """Return zeroed embedding counters."""
    return {
        "calls": 0,
        "errors": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
        "input_chars": 0,
        "tokens_from_api": 0,
        "tokens_estimated": 0,
    }


def _empty_simple_counters(
    *, with_chars: bool = False, with_bytes: bool = False
) -> dict[str, int]:
    """Return zeroed call/error counters with optional char/byte fields."""
    counters: dict[str, int] = {"calls": 0, "errors": 0}
    if with_chars:
        counters["request_chars"] = 0
        counters["response_chars"] = 0
    if with_bytes:
        counters["bytes_in"] = 0
    return counters


def empty_day_rollup(date: str) -> dict[str, Any]:
    """Build an empty daily rollup document for ``date``."""
    return {
        "date": date,
        "updated_at": utc_now_iso(),
        "embeddings": {
            **_empty_embed_counters(),
            "by_source": {name: _empty_embed_counters() for name in EMBEDDING_SOURCES},
        },
        "mcp": {
            **_empty_simple_counters(with_chars=True),
            "by_tool": {
                name: _empty_simple_counters(with_chars=True) for name in MCP_TOOLS
            },
        },
        "http": {name: _empty_simple_counters(with_bytes=True) for name in HTTP_ROUTES},
        "outbound": {
            "novita_embeddings": {"calls": 0, "errors": 0, "latency_ms_sum": 0},
            "remote_fetch": {"calls": 0, "errors": 0, "bytes_in": 0},
        },
    }


def _add_ints(target: dict[str, Any], key: str, amount: int) -> None:
    """Increment an integer field on ``target``."""
    target[key] = int(target.get(key) or 0) + int(amount)


def _merge_embed_counters(target: dict[str, Any], **kwargs: int) -> None:
    """Add embedding counter deltas onto ``target``."""
    for key, value in kwargs.items():
        if value:
            _add_ints(target, key, value)


class AccountingStore:
    """Durable per-day rollups and append-only event logs under a root directory."""

    def __init__(self, root: str | Path) -> None:
        """Initialize the store rooted at ``root`` (default ``/data/accnt``)."""
        self.root = Path(root)
        self.events_dir = self.root / "events"
        self._lock = threading.Lock()

    def ensure_layout(self) -> None:
        """Create the accounting root and events directory if missing."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def day_path(self, date: str) -> Path:
        """Return the rollup JSON path for ``date``."""
        return self.root / f"{date}.json"

    def events_path(self, date: str) -> Path:
        """Return the JSONL events path for ``date``."""
        return self.events_dir / f"{date}.jsonl"

    def list_days(self) -> list[str]:
        """Return available UTC day keys, newest first."""
        if not self.root.exists():
            return []
        days = sorted(
            (path.stem for path in self.root.glob("????-??-??.json") if path.is_file()),
            reverse=True,
        )
        return days

    def read_day(self, date: str) -> dict[str, Any]:
        """Read one day's rollup, or an empty document if missing."""
        return self._load_unlocked(date)

    def summarize(self, days: int = 30) -> dict[str, Any]:
        """Return newest ``days`` rollups plus an aggregate ``totals`` object."""
        available = self.list_days()
        selected = available[: max(0, days)]
        day_docs = [self.read_day(day) for day in selected]
        totals = empty_day_rollup("totals")
        totals.pop("date", None)
        for doc in day_docs:
            self._accumulate(totals, doc)
        totals["updated_at"] = utc_now_iso()
        totals["day_count"] = len(day_docs)
        return {
            "accnt_root": str(self.root),
            "days_requested": days,
            "days": day_docs,
            "totals": totals,
        }

    def read_events(
        self,
        date: str,
        *,
        limit: int = 200,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the newest ``limit`` JSONL events for ``date``.

        When ``kind`` is set (for example ``embedding``), only matching events
        are returned. Filtering scans from the end of the file so the newest
        matching events are preferred even when other kinds dominate the log.
        """
        path = self.events_path(date)
        if not path.exists() or limit <= 0:
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        events: list[dict[str, Any]] = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if kind is not None and payload.get("kind") != kind:
                continue
            events.append(payload)
            if len(events) >= limit:
                break
        return events

    def record_embedding(
        self,
        *,
        source: str | None = None,
        ok: bool,
        input_chars: int,
        prompt_tokens: int = 0,
        total_tokens: int = 0,
        tokens_from_api: bool = False,
        latency_ms: int = 0,
        model: str | None = None,
        error: str | None = None,
        count_outbound: bool = True,
        provider: str | None = None,
    ) -> None:
        """Record one embedding attempt into today's rollup and event log."""
        resolved_source = source or get_accounting_source()
        if resolved_source not in EMBEDDING_SOURCES:
            resolved_source = "other"
        date = utc_today()
        with self._lock:
            day = self._load_unlocked(date)
            emb = day["embeddings"]
            by_source = emb["by_source"][resolved_source]
            deltas = {
                "calls": 1,
                "errors": 0 if ok else 1,
                "prompt_tokens": prompt_tokens if ok else 0,
                "total_tokens": total_tokens if ok else 0,
                "input_chars": input_chars,
                "tokens_from_api": 1 if (ok and tokens_from_api) else 0,
                "tokens_estimated": 1 if (ok and not tokens_from_api) else 0,
            }
            _merge_embed_counters(emb, **deltas)
            _merge_embed_counters(by_source, **deltas)
            if count_outbound:
                outbound = day["outbound"]["novita_embeddings"]
                _add_ints(outbound, "calls", 1)
                if not ok:
                    _add_ints(outbound, "errors", 1)
                _add_ints(outbound, "latency_ms_sum", max(0, latency_ms))
            self._save_unlocked(date, day)
            event: dict[str, Any] = {
                "kind": "embedding",
                "ts": utc_now_iso(),
                "source": resolved_source,
                "ok": ok,
                "chars": input_chars,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "tokens_from_api": tokens_from_api,
                "latency_ms": latency_ms,
                "model": model,
                "error": error,
            }
            if provider:
                event["provider"] = provider
            self._append_event_unlocked(date, event)

    def record_mcp(
        self,
        *,
        tool: str,
        ok: bool,
        request_chars: int = 0,
        response_chars: int = 0,
        error: str | None = None,
    ) -> None:
        """Record one inbound MCP tool invocation."""
        date = utc_today()
        tool_name = tool if tool in MCP_TOOLS else tool
        with self._lock:
            day = self._load_unlocked(date)
            mcp = day["mcp"]
            _add_ints(mcp, "calls", 1)
            if not ok:
                _add_ints(mcp, "errors", 1)
            _add_ints(mcp, "request_chars", request_chars)
            _add_ints(mcp, "response_chars", response_chars)
            by_tool = mcp["by_tool"].setdefault(
                tool_name, _empty_simple_counters(with_chars=True)
            )
            _add_ints(by_tool, "calls", 1)
            if not ok:
                _add_ints(by_tool, "errors", 1)
            _add_ints(by_tool, "request_chars", request_chars)
            _add_ints(by_tool, "response_chars", response_chars)
            self._save_unlocked(date, day)
            self._append_event_unlocked(
                date,
                {
                    "kind": "mcp",
                    "ts": utc_now_iso(),
                    "tool": tool_name,
                    "ok": ok,
                    "request_chars": request_chars,
                    "response_chars": response_chars,
                    "error": error,
                },
            )

    def record_http(
        self,
        *,
        route: str,
        ok: bool,
        bytes_in: int = 0,
        error: str | None = None,
    ) -> None:
        """Record one inbound HTTP route invocation."""
        date = utc_today()
        with self._lock:
            day = self._load_unlocked(date)
            bucket = day["http"].setdefault(
                route, _empty_simple_counters(with_bytes=True)
            )
            _add_ints(bucket, "calls", 1)
            if not ok:
                _add_ints(bucket, "errors", 1)
            _add_ints(bucket, "bytes_in", max(0, bytes_in))
            self._save_unlocked(date, day)
            self._append_event_unlocked(
                date,
                {
                    "kind": "http",
                    "ts": utc_now_iso(),
                    "route": route,
                    "ok": ok,
                    "bytes_in": bytes_in,
                    "error": error,
                },
            )

    def record_outbound(
        self,
        *,
        target: str,
        ok: bool,
        bytes_in: int = 0,
        latency_ms: int = 0,
        error: str | None = None,
    ) -> None:
        """Record one outbound connection (non-embedding)."""
        if target not in OUTBOUND_TARGETS:
            target = "remote_fetch"
        date = utc_today()
        with self._lock:
            day = self._load_unlocked(date)
            bucket = day["outbound"].setdefault(
                target,
                {"calls": 0, "errors": 0, "bytes_in": 0, "latency_ms_sum": 0},
            )
            _add_ints(bucket, "calls", 1)
            if not ok:
                _add_ints(bucket, "errors", 1)
            if "bytes_in" in bucket:
                _add_ints(bucket, "bytes_in", max(0, bytes_in))
            if "latency_ms_sum" in bucket:
                _add_ints(bucket, "latency_ms_sum", max(0, latency_ms))
            self._save_unlocked(date, day)
            self._append_event_unlocked(
                date,
                {
                    "kind": "outbound",
                    "ts": utc_now_iso(),
                    "target": target,
                    "ok": ok,
                    "bytes_in": bytes_in,
                    "latency_ms": latency_ms,
                    "error": error,
                },
            )

    def _load_unlocked(self, date: str) -> dict[str, Any]:
        """Load and normalize a day document without acquiring the lock."""
        path = self.day_path(date)
        if not path.exists():
            return empty_day_rollup(date)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return empty_day_rollup(date)
        if not isinstance(data, dict):
            return empty_day_rollup(date)
        return self._normalize_day(data, date)

    def _save_unlocked(self, date: str, day: dict[str, Any]) -> None:
        """Atomically write ``day`` for ``date``."""
        self.ensure_layout()
        day = self._normalize_day(day, date)
        day["updated_at"] = utc_now_iso()
        path = self.day_path(date)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(day, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)

    def _append_event_unlocked(self, date: str, event: dict[str, Any]) -> None:
        """Append one JSONL event for ``date``."""
        self.ensure_layout()
        path = self.events_path(date)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _normalize_day(self, data: dict[str, Any], date: str) -> dict[str, Any]:
        """Fill missing keys on a loaded day document."""
        base = empty_day_rollup(date)
        base["updated_at"] = str(data.get("updated_at") or base["updated_at"])
        emb_raw = data.get("embeddings")
        emb: dict[str, Any] = emb_raw if isinstance(emb_raw, dict) else {}
        for key in _empty_embed_counters():
            base["embeddings"][key] = int(emb.get(key) or 0)
        by_source_raw = emb.get("by_source")
        by_source: dict[str, Any] = (
            by_source_raw if isinstance(by_source_raw, dict) else {}
        )
        for name in EMBEDDING_SOURCES:
            src_raw = by_source.get(name)
            src: dict[str, Any] = src_raw if isinstance(src_raw, dict) else {}
            for key in _empty_embed_counters():
                base["embeddings"]["by_source"][name][key] = int(src.get(key) or 0)
        mcp_raw = data.get("mcp")
        mcp: dict[str, Any] = mcp_raw if isinstance(mcp_raw, dict) else {}
        for key in ("calls", "errors", "request_chars", "response_chars"):
            base["mcp"][key] = int(mcp.get(key) or 0)
        by_tool_raw = mcp.get("by_tool")
        by_tool: dict[str, Any] = by_tool_raw if isinstance(by_tool_raw, dict) else {}
        tool_names = set(MCP_TOOLS) | set(by_tool.keys())
        for name in tool_names:
            tool_raw = by_tool.get(name)
            tool: dict[str, Any] = tool_raw if isinstance(tool_raw, dict) else {}
            bucket = base["mcp"]["by_tool"].setdefault(
                name, _empty_simple_counters(with_chars=True)
            )
            for key in ("calls", "errors", "request_chars", "response_chars"):
                bucket[key] = int(tool.get(key) or 0)
        http_raw = data.get("http")
        http: dict[str, Any] = http_raw if isinstance(http_raw, dict) else {}
        route_names = set(HTTP_ROUTES) | set(http.keys())
        for name in route_names:
            route_raw = http.get(name)
            route: dict[str, Any] = route_raw if isinstance(route_raw, dict) else {}
            bucket = base["http"].setdefault(
                name, _empty_simple_counters(with_bytes=True)
            )
            for key in ("calls", "errors", "bytes_in"):
                bucket[key] = int(route.get(key) or 0)
        outbound_raw = data.get("outbound")
        outbound: dict[str, Any] = (
            outbound_raw if isinstance(outbound_raw, dict) else {}
        )
        for name in OUTBOUND_TARGETS:
            target_raw = outbound.get(name)
            target: dict[str, Any] = target_raw if isinstance(target_raw, dict) else {}
            bucket = base["outbound"][name]
            for key in bucket:
                bucket[key] = int(target.get(key) or 0)
        return base

    def _accumulate(self, totals: dict[str, Any], day: dict[str, Any]) -> None:
        """Add ``day`` counters into ``totals``."""
        for key in _empty_embed_counters():
            _add_ints(totals["embeddings"], key, int(day["embeddings"].get(key) or 0))
        for name in EMBEDDING_SOURCES:
            src = day["embeddings"]["by_source"][name]
            target = totals["embeddings"]["by_source"][name]
            for key in _empty_embed_counters():
                _add_ints(target, key, int(src.get(key) or 0))
        for key in ("calls", "errors", "request_chars", "response_chars"):
            _add_ints(totals["mcp"], key, int(day["mcp"].get(key) or 0))
        for name, tool in day["mcp"]["by_tool"].items():
            bucket = totals["mcp"]["by_tool"].setdefault(
                name, _empty_simple_counters(with_chars=True)
            )
            for key in ("calls", "errors", "request_chars", "response_chars"):
                _add_ints(bucket, key, int(tool.get(key) or 0))
        for name, route in day["http"].items():
            bucket = totals["http"].setdefault(
                name, _empty_simple_counters(with_bytes=True)
            )
            for key in ("calls", "errors", "bytes_in"):
                _add_ints(bucket, key, int(route.get(key) or 0))
        for name in OUTBOUND_TARGETS:
            src = day["outbound"][name]
            target = totals["outbound"][name]
            for key in target:
                _add_ints(target, key, int(src.get(key) or 0))
