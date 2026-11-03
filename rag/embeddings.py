"""Novita BGE-M3 embedding helpers for the RAG pipeline."""

from __future__ import annotations

import os
from typing import Any

import requests

EMBEDDING_MODEL = "baai/bge-m3"
EMBEDDING_URL = "https://api.novita.ai/openai/v1/embeddings"
EMBEDDING_DIM = 1024


def resolve_api_key() -> str | None:
    """Return the Novita token, with legacy environment names as fallbacks."""
    return (
        os.getenv("BGE_M3_NOVITA_API_KEY")
        or os.getenv("NOVITA_API_KEY")
        or os.getenv("OPENAI_TOKEN")
        or os.getenv("OPENAI_API_KEY")
    )


def get_embedding(text: str, *, api_key: str | None = None) -> list[float]:
    """Embed ``text`` with Novita's OpenAI-compatible BGE-M3 endpoint."""
    key = api_key if api_key is not None else resolve_api_key()
    if not key:
        raise ValueError("Missing Novita credentials. Set BGE_M3_NOVITA_API_KEY.")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {"input": text, "model": EMBEDDING_MODEL}
    response = requests.post(EMBEDDING_URL, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    embedding = response.json()["data"][0]["embedding"]
    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(
            f"Novita model {EMBEDDING_MODEL} returned {len(embedding)} dimensions; "
            f"expected {EMBEDDING_DIM}."
        )
    return list(embedding)
