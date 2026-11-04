"""Novita BGE-M3 embedding helpers for the RAG pipeline."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from rag.accounting import get_global_store

EMBEDDING_MODEL = "baai/bge-m3"
EMBEDDING_URL = "https://api.novita.ai/openai/v1/embeddings"
EMBEDDING_DIM = 1024


def resolve_api_key() -> str | None:
    """Return the embedding API token from environment variables.

    Checks ``BGE_M3_NOVITA_API_KEY``, then ``NOVITA_API_KEY``, ``OPENAI_TOKEN``,
    and ``OPENAI_API_KEY``.
    """
    return (
        os.getenv("BGE_M3_NOVITA_API_KEY")
        or os.getenv("NOVITA_API_KEY")
        or os.getenv("OPENAI_TOKEN")
        or os.getenv("OPENAI_API_KEY")
    )


def _parse_usage(payload: dict[str, Any]) -> tuple[int, int, bool]:
    """Extract prompt/total tokens from an OpenAI-shaped embeddings response."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, False
    prompt = usage.get("prompt_tokens")
    total = usage.get("total_tokens")
    if prompt is None and total is None:
        return 0, 0, False
    prompt_tokens = int(prompt or 0)
    total_tokens = int(total if total is not None else prompt_tokens)
    return prompt_tokens, total_tokens, True


def _record_embedding(
    *,
    ok: bool,
    input_chars: int,
    prompt_tokens: int = 0,
    total_tokens: int = 0,
    tokens_from_api: bool = False,
    latency_ms: int = 0,
    error: str | None = None,
) -> None:
    """Record embedding usage when a global accounting store is configured."""
    store = get_global_store()
    if store is None:
        return
    store.record_embedding(
        ok=ok,
        input_chars=input_chars,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
        tokens_from_api=tokens_from_api,
        latency_ms=latency_ms,
        model=EMBEDDING_MODEL,
        error=error,
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
    input_chars = len(text)
    started = time.perf_counter()
    try:
        response = requests.post(
            EMBEDDING_URL, json=payload, headers=headers, timeout=60
        )
        response.raise_for_status()
        body = response.json()
        embedding = body["data"][0]["embedding"]
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Novita model {EMBEDDING_MODEL} returned {len(embedding)} dimensions; "
                f"expected {EMBEDDING_DIM}."
            )
        prompt_tokens, total_tokens, tokens_from_api = _parse_usage(body)
        latency_ms = int((time.perf_counter() - started) * 1000)
        _record_embedding(
            ok=True,
            input_chars=input_chars,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            tokens_from_api=tokens_from_api,
            latency_ms=latency_ms,
        )
        return list(embedding)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _record_embedding(
            ok=False,
            input_chars=input_chars,
            latency_ms=latency_ms,
            error=str(exc),
        )
        raise
