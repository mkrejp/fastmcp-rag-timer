"""Process role: monolith, read-only reader, or write-focused ingester."""

from __future__ import annotations

import os
from typing import Literal, cast

AppMode = Literal["all", "reader", "ingester"]

DEFAULT_APP_MODE: AppMode = "all"
SUPPORTED_APP_MODES: tuple[AppMode, ...] = ("all", "reader", "ingester")


def resolve_app_mode() -> AppMode:
    """Return ``APP_MODE`` / ``RAG_APP_MODE`` (default ``all``).

    ``reader`` is ČTENÍ-only (writes enforced off, low-RAM query defaults).
    ``ingester`` is ZÁPIS-capable (upload/ingest/publish). ``all`` keeps the
    historical single-process behaviour for local/dev and existing deploys.
    Either mode may run on Railway or a VPS; the mode is not tied to a host.
    """
    raw = (
        os.getenv("APP_MODE", "").strip() or os.getenv("RAG_APP_MODE", "").strip()
    ).lower()
    if not raw:
        return DEFAULT_APP_MODE
    if raw in {"ro", "readonly", "read", "cteni"}:
        return "reader"
    if raw in {"rw", "write", "writer", "ingest", "zapis"}:
        return "ingester"
    if raw not in SUPPORTED_APP_MODES:
        supported = ", ".join(SUPPORTED_APP_MODES)
        raise ValueError(f"Unknown APP_MODE={raw!r}. Use one of: {supported}.")
    return cast(AppMode, raw)


def is_reader_mode() -> bool:
    """Return whether this process is read-only."""
    return resolve_app_mode() == "reader"


def is_ingester_mode() -> bool:
    """Return whether this process is the write/ingest role."""
    return resolve_app_mode() == "ingester"


def mcp_include_write_tools() -> bool:
    """Return whether mutating MCP tools should be registered."""
    return not is_reader_mode()


def app_mode_status() -> dict[str, str | bool]:
    """Public health/admin summary for the process role."""
    mode = resolve_app_mode()
    return {
        "app_mode": mode,
        "writes_enabled": mode != "reader",
        "mcp_write_tools": mcp_include_write_tools(),
    }
