"""Runtime configuration for the RAG service."""

from __future__ import annotations

import os

DEFAULT_RAG_DATA_ROOT = "/data/rag"
COLLECTION_NAME = "client_domain_knowledge"
HOST = "0.0.0.0"
PORT = 8080


def resolve_rag_data_root(override: str | None = None) -> str:
    """Resolve the persistent RAG data root directory."""
    if override:
        return override
    return os.getenv("RAG_DATA_ROOT", DEFAULT_RAG_DATA_ROOT)


def should_skip_default_app() -> bool:
    """Return whether module-level app construction should be skipped."""
    return os.getenv("RAG_SKIP_DEFAULT_APP") == "1"
