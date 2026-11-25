"""OpenRouter client for Cloud Agent / ingester development.

This is **not** the production embedding or rerank path. Indexed text stays on
prepaid Novita or local FastEmbed BGE-M3. OpenRouter is an OpenAI-compatible
chat API keyed by ``OPENROUTER_API_KEY``. BYOK for ``nvidia`` and ``novita``
uses ``OPENROUTER_MANAGEMENT_KEY``. Never log keys.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import requests

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_NAME = "OPENROUTER_API_KEY"
OPENROUTER_MANAGEMENT_KEY_NAME = "OPENROUTER_MANAGEMENT_KEY"
OPENROUTER_NVIDIA_PROVIDER = "nvidia"
OPENROUTER_NOVITA_PROVIDER = "novita"
OPENROUTER_BYOK_PROVIDERS = (
    OPENROUTER_NVIDIA_PROVIDER,
    OPENROUTER_NOVITA_PROVIDER,
)
_MODELS_TIMEOUT_SECONDS = 30
_BYOK_TIMEOUT_SECONDS = 30

ByokSyncStatus = Literal[
    "created",
    "exists",
    "skipped_missing_key",
    "skipped_no_management",
]


def resolve_openrouter_api_key() -> str | None:
    """Return ``OPENROUTER_API_KEY`` if set.

    Empty strings are treated as unset. The value is never logged.
    """
    raw = os.getenv(OPENROUTER_KEY_NAME, "")
    stripped = raw.strip() if raw else ""
    return stripped or None


def resolve_openrouter_management_key() -> str | None:
    """Return ``OPENROUTER_MANAGEMENT_KEY`` if set.

    Empty strings are treated as unset. The value is never logged. This key
    cannot call chat completions; it is only for BYOK and other admin APIs.
    """
    raw = os.getenv(OPENROUTER_MANAGEMENT_KEY_NAME, "")
    stripped = raw.strip() if raw else ""
    return stripped or None


def openrouter_configured() -> bool:
    """Return whether an OpenRouter inference API key is present (no secret)."""
    return bool(resolve_openrouter_api_key())


def openrouter_management_configured() -> bool:
    """Return whether an OpenRouter management key is present (no secret)."""
    return bool(resolve_openrouter_management_key())


def resolve_openrouter_model() -> str | None:
    """Return optional ``OPENROUTER_MODEL`` catalog ID, if set."""
    raw = os.getenv("OPENROUTER_MODEL", "").strip()
    return raw or None


def openrouter_status() -> dict[str, str | bool | None]:
    """Public health flags for OpenRouter (no secrets)."""
    return {
        "configured": openrouter_configured(),
        "management_configured": openrouter_management_configured(),
        "base_url": OPENROUTER_BASE_URL,
        "model": resolve_openrouter_model(),
        "nvidia_provider": OPENROUTER_NVIDIA_PROVIDER,
        "novita_provider": OPENROUTER_NOVITA_PROVIDER,
    }


def list_openrouter_models() -> list[str]:
    """List OpenRouter model IDs via ``GET /api/v1/models``.

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


def _first_env(*names: str) -> str | None:
    """Return the first non-empty environment value among ``names``."""
    for name in names:
        raw = os.getenv(name, "")
        stripped = raw.strip() if raw else ""
        if stripped:
            return stripped
    return None


def resolve_openrouter_nvidia_byok_key() -> str | None:
    """Return the NVIDIA key to attach as OpenRouter BYOK (never logged)."""
    return _first_env("NVIDIA_NIM_API", "NVIDIA_NIM_APU", "NVIDIA_API_KEY")


def resolve_openrouter_novita_byok_key() -> str | None:
    """Return the prepaid Novita key to attach as OpenRouter BYOK.

    Does not fall back to ``OPENAI_API_KEY``. Never logged.
    """
    return _first_env("BGE_M3_NOVITA_API_KEY", "NOVITA_API_KEY")


def _management_headers() -> dict[str, str]:
    """Return Bearer headers for the OpenRouter management API."""
    key = resolve_openrouter_management_key()
    if not key:
        raise RuntimeError(
            "Missing OpenRouter management credentials. Set "
            "OPENROUTER_MANAGEMENT_KEY as a runtime secret."
        )
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def list_openrouter_byok_providers() -> list[str]:
    """List enabled BYOK provider slugs (no keys, no labels).

    Raises:
        RuntimeError: If ``OPENROUTER_MANAGEMENT_KEY`` is unset or OpenRouter
            rejects the management key.
    """
    response = requests.get(
        f"{OPENROUTER_BASE_URL}/byok",
        headers=_management_headers(),
        timeout=_BYOK_TIMEOUT_SECONDS,
    )
    if response.status_code in {401, 403}:
        raise RuntimeError(
            "OpenRouter BYOK list requires a management key "
            "(OPENROUTER_MANAGEMENT_KEY), not OPENROUTER_API_KEY."
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenRouter BYOK list failed with HTTP {response.status_code}."
        )
    payload: Any = response.json()
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return []
    providers: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("disabled") is True:
            continue
        provider = item.get("provider")
        if isinstance(provider, str) and provider.strip():
            providers.append(provider.strip())
    return providers


def ensure_openrouter_byok_credential(
    provider: str, *, key: str, name: str
) -> Literal["created", "exists"]:
    """Create a BYOK credential for ``provider`` if one is not already enabled.

    The raw provider key is sent to OpenRouter and never logged or returned.

    Args:
        provider: OpenRouter provider slug (``nvidia`` or ``novita``).
        key: Upstream provider API key.
        name: Workspace label for the credential.

    Returns:
        ``exists`` if an enabled credential for this provider is already
        registered, otherwise ``created``.

    Raises:
        RuntimeError: If the management key is missing, the provider slug is
            unsupported, or OpenRouter returns an error.
    """
    slug = provider.strip().lower()
    if slug not in OPENROUTER_BYOK_PROVIDERS:
        raise RuntimeError(
            f"Unsupported OpenRouter BYOK provider {slug!r}. "
            "Use nvidia or novita."
        )
    stripped_key = key.strip()
    if not stripped_key:
        raise RuntimeError(f"Missing provider key for OpenRouter BYOK {slug}.")
    existing = list_openrouter_byok_providers()
    if slug in existing:
        return "exists"
    response = requests.post(
        f"{OPENROUTER_BASE_URL}/byok",
        headers=_management_headers(),
        json={"provider": slug, "key": stripped_key, "name": name},
        timeout=_BYOK_TIMEOUT_SECONDS,
    )
    if response.status_code in {401, 403}:
        raise RuntimeError(
            "OpenRouter BYOK create requires a management key "
            "(OPENROUTER_MANAGEMENT_KEY), not OPENROUTER_API_KEY."
        )
    if response.status_code in {200, 201}:
        return "created"
    if response.status_code in {409, 422}:
        return "exists"
    raise RuntimeError(
        f"OpenRouter BYOK create for {slug} failed with HTTP "
        f"{response.status_code}."
    )


def sync_openrouter_nvidia_novita_byok() -> dict[str, ByokSyncStatus]:
    """Attach NVIDIA and Novita keys as OpenRouter BYOK when secrets exist.

    Production embeddings and rerank still call Novita directly. This only
    registers development provider credentials on OpenRouter.

    Returns:
        Per-provider status without secret values.
    """
    if not resolve_openrouter_management_key():
        return {
            OPENROUTER_NVIDIA_PROVIDER: "skipped_no_management",
            OPENROUTER_NOVITA_PROVIDER: "skipped_no_management",
        }
    result: dict[str, ByokSyncStatus] = {}
    nvidia_key = resolve_openrouter_nvidia_byok_key()
    if nvidia_key:
        result[OPENROUTER_NVIDIA_PROVIDER] = ensure_openrouter_byok_credential(
            OPENROUTER_NVIDIA_PROVIDER,
            key=nvidia_key,
            name="NVIDIA NIM (Cursor dev)",
        )
    else:
        result[OPENROUTER_NVIDIA_PROVIDER] = "skipped_missing_key"
    novita_key = resolve_openrouter_novita_byok_key()
    if novita_key:
        result[OPENROUTER_NOVITA_PROVIDER] = ensure_openrouter_byok_credential(
            OPENROUTER_NOVITA_PROVIDER,
            key=novita_key,
            name="Novita prepaid (Cursor dev)",
        )
    else:
        result[OPENROUTER_NOVITA_PROVIDER] = "skipped_missing_key"
    return result
