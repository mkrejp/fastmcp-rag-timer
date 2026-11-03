"""OpenAI embedding helpers for the RAG pipeline."""

from __future__ import annotations

import os
from typing import Any

import requests

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
EMBEDDING_DIM = 1536


def resolve_api_key() -> str | None:
    """Return the OpenAI API token from known environment variables."""
    return os.getenv("OPENAI_TOKEN") or os.getenv("OPENAI_API_KEY")


def get_embedding(text: str, *, api_key: str | None = None) -> list[float]:
    """Embed ``text`` with OpenAI ``text-embedding-3-small``.

    Prefers ``OPENAI_TOKEN``, then falls back to ``OPENAI_API_KEY``.
    """
    key = api_key if api_key is not None else resolve_api_key()
    if not key:
        raise ValueError(
            "Missing OpenAI credentials. Set OPENAI_TOKEN or OPENAI_API_KEY."
        )
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {"input": text, "model": EMBEDDING_MODEL}
    response = requests.post(EMBEDDING_URL, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    body = response.json()
    return list(body["data"][0]["embedding"])
