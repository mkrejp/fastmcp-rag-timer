"""Vercel AI Gateway chat for the optional reader helper (fail-closed).

Chat tokens go through Gateway virtual model ``vmc/cheap`` (Novita
``deepseek/deepseek-v4.1-flash`` BYOK). Novita spend is prepaid Model API
credits, not the monthly Coding Plan. This client never retries a 403
against another provider and ignores embedding ``OPENAI_*`` keys.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Mapping, Sequence

import requests

from rag.accounting import get_global_store, today_outbound_calls
from rag.logging_setup import get_logger
from rag.novita_rerank import classify_upstream_error

DEFAULT_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
ALLOWED_CHAT_MODELS = frozenset({"vmc/cheap"})
DEFAULT_CHAT_MODEL = "vmc/cheap"
DEFAULT_MAX_TOKENS = 700
DEFAULT_DAILY_CHAT_CALLS = 300
CHAT_TIMEOUT_SECONDS = 15

logger = get_logger("rag.gateway_chat")

_THINK_BLOCK_RE = re.compile(
    r"<(?:redacted_thinking|reasoning)>.*?</(?:redacted_thinking|reasoning)>",
    re.IGNORECASE | re.DOTALL,
)


def _load_json_object(text: str) -> dict[str, Any]:
    """Parse ``text`` as a JSON object, tolerating leading non-JSON prose."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        payload, _end = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(payload, dict):
        raise ValueError("helper JSON must be an object")
    return payload


def resolve_gateway_api_key() -> str | None:
    """Return the AI Gateway key. Never logs the value.

    ``AI_GATEWAY_API_KEY`` wins; ``VERCEL_AI_GATEWAY_API_KEY`` is fallback.
    Embedding ``OPENAI_API_KEY`` / Novita keys are **not** used here.
    """
    for name in ("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_API_KEY"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def resolve_chat_model() -> str:
    """Return the pinned Gateway virtual model for the reader helper."""
    override = os.getenv("AI_GATEWAY_CHAT_MODEL", "").strip()
    if override and override not in ALLOWED_CHAT_MODELS:
        return ""
    if override:
        return override
    return DEFAULT_CHAT_MODEL


def resolve_chat_max_tokens() -> int:
    """Return the helper completion cap (enough for a short cited essay)."""
    raw = os.getenv("CHAT_HELPER_MAX_TOKENS", "").strip()
    if not raw:
        return DEFAULT_MAX_TOKENS
    try:
        return max(32, min(int(raw), 1536))
    except ValueError:
        return DEFAULT_MAX_TOKENS


def chat_configured() -> bool:
    """Return whether an explicit Gateway key is present."""
    return bool(resolve_gateway_api_key()) and bool(resolve_chat_model())


def chat_enabled() -> bool:
    """Return whether the helper may call Gateway chat (off by default)."""
    value = os.getenv("CHAT_HELPER_CHAT", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def daily_chat_call_limit() -> int:
    """Return the daily Gateway chat call cap."""
    raw = os.getenv("CHAT_HELPER_DAILY_CHAT_CALLS", "").strip()
    if not raw:
        return DEFAULT_DAILY_CHAT_CALLS
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_DAILY_CHAT_CALLS


def chat_status() -> dict[str, str | bool | int]:
    """Public health flags for Gateway chat (no secrets)."""
    return {
        "model": resolve_chat_model() or DEFAULT_CHAT_MODEL,
        "configured": chat_configured(),
        "enabled": chat_enabled(),
        "max_tokens": resolve_chat_max_tokens(),
        "daily_call_limit": daily_chat_call_limit(),
        "novita_billing": "prepaid_credits",
    }


def _record_outbound(
    *,
    ok: bool,
    latency_ms: int,
    error: str | None = None,
    prompt_tokens: int = 0,
    total_tokens: int = 0,
) -> None:
    """Record one Gateway chat call when accounting is configured."""
    store = get_global_store()
    if store is None:
        return
    store.record_outbound(
        target="ai_gateway_chat",
        ok=ok,
        latency_ms=latency_ms,
        error=error,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
    )


def _parse_usage(payload: Mapping[str, Any]) -> tuple[int, int]:
    """Extract prompt/total tokens from a chat completion body."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0, 0
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    total = usage.get("total_tokens")
    try:
        prompt_tokens = int(prompt)
        if total is None:
            total_tokens = prompt_tokens + int(completion)
        else:
            total_tokens = int(total)
        return prompt_tokens, total_tokens
    except (TypeError, ValueError):
        return 0, 0


def complete_chat(
    messages: Sequence[Mapping[str, str]],
    *,
    temperature: float = 0.1,
) -> str:
    """Call Gateway ``/chat/completions`` and return the assistant text.

    Pins ``vmc/cheap`` and ``providerOptions.gateway.only = ["novita"]``.
    Raises ``RuntimeError`` with a coarse code on failure.
    """
    model = resolve_chat_model()
    if not model:
        raise RuntimeError("skipped_bad_model")
    key = resolve_gateway_api_key()
    if not key:
        raise RuntimeError("skipped_no_key")
    if today_outbound_calls("ai_gateway_chat") >= daily_chat_call_limit():
        raise RuntimeError("skipped_budget")
    url = f"{DEFAULT_GATEWAY_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": [dict(item) for item in messages],
        "temperature": temperature,
        "max_tokens": resolve_chat_max_tokens(),
        "response_format": {"type": "json_object"},
        "providerOptions": {"gateway": {"only": ["novita"]}},
        "reasoning_effort": "none",
    }
    started = time.perf_counter()
    try:
        response = requests.post(
            url, json=body, headers=headers, timeout=CHAT_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("non-object JSON")
        prompt_tokens, total_tokens = _parse_usage(payload)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("bad choice")
        message = first.get("message")
        if not isinstance(message, dict):
            raise ValueError("bad message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            _record_outbound(
                ok=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error="failed_empty",
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
            )
            raise RuntimeError("failed_empty")
        _record_outbound(
            ok=True,
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )
        return content.strip()
    except RuntimeError:
        raise
    except Exception as exc:
        code = classify_upstream_error(exc)
        latency_ms = int((time.perf_counter() - started) * 1000)
        _record_outbound(ok=False, latency_ms=latency_ms, error=code)
        logger.warning("Gateway chat helper failed: %s", code)
        raise RuntimeError(code) from exc


def parse_helper_json(raw: str) -> dict[str, Any]:
    """Parse an LLM JSON object, including fenced markdown payloads."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    text = _THINK_BLOCK_RE.sub("", text).strip()
    return _load_json_object(text)
