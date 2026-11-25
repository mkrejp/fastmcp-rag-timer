"""Hosted NVIDIA NIM client for Cloud Agent / ingester development.

This is **not** the production embedding or rerank path. Indexed text stays on
prepaid Novita or local FastEmbed BGE-M3. NIM is an OpenAI-compatible chat API
keyed by ``NVIDIA_NIM_API`` (Railway ingester runtime and Cursor environment
secret). Never log the key.
"""

from __future__ import annotations

import os
from typing import Any

import requests

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_NIM_KEY_NAMES = ("NVIDIA_NIM_API", "NVIDIA_API_KEY")
_MODELS_TIMEOUT_SECONDS = 30


def resolve_nvidia_nim_api_key() -> str | None:
    """Return the hosted NIM key, preferring ``NVIDIA_NIM_API``.

    Falls back to ``NVIDIA_API_KEY`` (NVIDIA catalog name). Empty strings are
    treated as unset. The value is never logged.
    """
    for name in NVIDIA_NIM_KEY_NAMES:
        raw = os.getenv(name, "")
        stripped = raw.strip() if raw else ""
        if stripped:
            return stripped
    return None


def nvidia_nim_configured() -> bool:
    """Return whether a NIM API key is present (no secret value)."""
    return bool(resolve_nvidia_nim_api_key())


def resolve_nvidia_nim_model() -> str | None:
    """Return optional ``NVIDIA_NIM_MODEL`` catalog id, if set."""
    raw = os.getenv("NVIDIA_NIM_MODEL", "").strip()
    return raw or None


def nvidia_nim_status() -> dict[str, str | bool | None]:
    """Public health flags for hosted NIM (no secrets)."""
    return {
        "configured": nvidia_nim_configured(),
        "base_url": NVIDIA_NIM_BASE_URL,
        "model": resolve_nvidia_nim_model(),
    }


def list_nim_models() -> list[str]:
    """List hosted NIM model ids via ``GET /v1/models``.

    Raises:
        RuntimeError: If ``NVIDIA_NIM_API`` / ``NVIDIA_API_KEY`` is unset.
        requests.RequestException: On HTTP or network failure.
    """
    key = resolve_nvidia_nim_api_key()
    if not key:
        raise RuntimeError(
            "Missing NVIDIA NIM credentials. Set NVIDIA_NIM_API "
            "(or NVIDIA_API_KEY) as a runtime secret."
        )
    response = requests.get(
        f"{NVIDIA_NIM_BASE_URL}/models",
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
