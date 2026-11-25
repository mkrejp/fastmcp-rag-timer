"""OpenRouter client for Cloud Agent / ingester development.

This is **not** the production embedding or rerank path. Indexed text stays on
prepaid Novita or local FastEmbed BGE-M3. OpenRouter is an OpenAI-compatible
chat API keyed by ``OPENROUTER_API_KEY``. Never log the key.
"""

from __future__ import annotations

import os
from typing import Any

import requests

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_NAME = "OPENROUTER_API_KEY"
_MODELS_TIMEOUT_SECONDS = 30


def resolve_openrouter_api_key() -> str | None:
    """Return ``OPENROUTER_API_KEY`` if set.

    Empty strings are treated as unset. The value is never logged.
    """
    raw = os.getenv(OPENROUTER_KEY_NAME, "")
    stripped = raw.strip() if raw else ""
    return stripped or None


def openrouter_configured() -> bool:
    """Return whether an OpenRouter API key is present (no secret value)."""
    return bool(resolve_openrouter_api_key())


def resolve_openrouter_model() -> str | None:
    """Return optional ``OPENROUTER_MODEL`` catalog id, if set."""
    raw = os.getenv("OPENROUTER_MODEL", "").strip()
    return raw or None


def openrouter_status() -> dict[str, str | bool | None]:
    """Public health flags for OpenRouter (no secrets)."""
    return {
        "configured": openrouter_configured(),
        "base_url": OPENROUTER_BASE_URL,
        "model": resolve_openrouter_model(),
    }


def list_openrouter_models() -> list[str]:
    """List OpenRouter model ids via ``GET /api/v1/models``.

    Raises:
        RuntimeError: If ``OPENROUTER_API_KEY`` is unset.
        requests.RequestException: On HTTP or network failure.
    """
    key = resolve_openrouter_api_key()
    if not key:
        raise RuntimeError(
            "Missing OpenRouter credentials. Set OPENROUTER_API_KEY "
            "as a runtime secret."
        )
    response = requests.get(
        f"{OPENROUTER_BASE_URL}/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=_MODELS_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload: Any = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict):
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id.strip():
                ids.append(model_id.strip())
    return ids
