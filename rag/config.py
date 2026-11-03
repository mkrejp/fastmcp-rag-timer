"""Runtime configuration for the RAG service."""

from __future__ import annotations

import os

DEFAULT_RAG_DATA_ROOT = "/data/rag"
COLLECTION_NAME = "client_domain_knowledge"
HOST = "0.0.0.0"
PORT = 8080
DEFAULT_MAX_REMOTE_JSON_BYTES = 128 * 1024 * 1024
DEFAULT_REMOTE_FETCH_TIMEOUT_SECONDS = 120
MAX_REMOTE_REDIRECTS = 5


def resolve_rag_data_root(override: str | None = None) -> str:
    """Resolve the persistent RAG data root directory."""
    if override:
        return override
    return os.getenv("RAG_DATA_ROOT", DEFAULT_RAG_DATA_ROOT)


def should_skip_default_app() -> bool:
    """Return whether module-level app construction should be skipped."""
    return os.getenv("RAG_SKIP_DEFAULT_APP") == "1"


def resolve_api_token() -> str | None:
    """Return the inbound HTTP bearer token, if configured."""
    token = os.getenv("RAG_API_TOKEN")
    if token is None:
        return None
    stripped = token.strip()
    return stripped or None


def resolve_max_remote_json_bytes() -> int:
    """Return the maximum allowed remote JSON download size in bytes."""
    raw = os.getenv("RAG_MAX_REMOTE_JSON_BYTES")
    if not raw:
        return DEFAULT_MAX_REMOTE_JSON_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_REMOTE_JSON_BYTES
    return value if value > 0 else DEFAULT_MAX_REMOTE_JSON_BYTES


def resolve_remote_fetch_timeout() -> int:
    """Return the remote JSON fetch timeout in seconds."""
    raw = os.getenv("RAG_REMOTE_FETCH_TIMEOUT")
    if not raw:
        return DEFAULT_REMOTE_FETCH_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_REMOTE_FETCH_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_REMOTE_FETCH_TIMEOUT_SECONDS
