"""Novita prepaid BGE rerank for reader chat-helper ranking.

Uses the same prepaid Model API balance as ``baai/bge-m3`` embeddings
(``BGE_M3_NOVITA_API_KEY`` / ``NOVITA_API_KEY`` only). This is **not**
the monthly Novita Coding Plan. Embedding ``OPENAI_*`` keys are ignored.
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping, Sequence

import requests

from rag.accounting import get_global_store, today_outbound_calls
from rag.logging_setup import get_logger

NOVITA_RERANK_MODEL = "baai/bge-reranker-v2-m3"
NOVITA_RERANK_URL = "https://api.novita.ai/openai/v1/rerank"
NOVITA_BILLING = "prepaid_credits"
MAX_RERANK_DOCUMENTS = 32
RERANK_TIMEOUT_SECONDS = 5
RERANK_QUERY_CHARS = 512
DEFAULT_DAILY_RERANK_CALLS = 3000

logger = get_logger("rag.novita_rerank")


def resolve_novita_api_key() -> str | None:
    """Return the prepaid Novita key. Ignores ``OPENAI_*`` embedding fallbacks."""
    for name in ("BGE_M3_NOVITA_API_KEY", "NOVITA_API_KEY"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def resolve_rerank_model() -> str:
    """Return the pinned Novita rerank model id."""
    return NOVITA_RERANK_MODEL


def rerank_configured() -> bool:
    """Return whether a prepaid Novita API key is present for rerank."""
    return bool(resolve_novita_api_key())


def rerank_enabled() -> bool:
    """Return whether the helper should call the prepaid reranker."""
    value = os.getenv("CHAT_HELPER_RERANK", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def daily_rerank_call_limit() -> int:
    """Return the daily Novita rerank call cap."""
    raw = os.getenv("CHAT_HELPER_DAILY_RERANK_CALLS", "").strip()
    if not raw:
        return DEFAULT_DAILY_RERANK_CALLS
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_DAILY_RERANK_CALLS


def rerank_status() -> dict[str, str | bool | int]:
    """Public health flags for the prepaid reranker (no secrets)."""
    return {
        "billing": NOVITA_BILLING,
        "model": resolve_rerank_model(),
        "configured": rerank_configured(),
        "enabled": rerank_enabled(),
        "daily_call_limit": daily_rerank_call_limit(),
    }


def classify_upstream_error(exc: BaseException) -> str:
    """Return a coarse error code that does not leak URLs or bodies."""
    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        status = int(getattr(response, "status_code", 0) or 0)
        if 400 <= status < 500:
            return "http_4xx"
        return "http_5xx"
    if isinstance(exc, (ValueError, TypeError)):
        return "bad_json"
    return "upstream_error"


def _record_outbound(
    *,
    ok: bool,
    latency_ms: int,
    error: str | None = None,
    prompt_tokens: int = 0,
    total_tokens: int = 0,
) -> None:
    """Record one Novita rerank call when accounting is configured."""
    store = get_global_store()
    if store is None:
        return
    store.record_outbound(
        target="novita_rerank",
        ok=ok,
        latency_ms=latency_ms,
        error=error,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
    )


def _parse_results(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize Novita / OpenAI-shaped rerank results."""
    raw = payload.get("results")
    if not isinstance(raw, list):
        raw = payload.get("data")
    if not isinstance(raw, list):
        return []
    parsed: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw_index = item.get("index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            continue
        score_raw = item.get("relevance_score", item.get("score"))
        score: float | None
        try:
            score = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score = None
        parsed.append({"index": raw_index, "score": score})
    return parsed


def _parse_usage(payload: Mapping[str, Any]) -> tuple[int, int]:
    """Extract prompt/total tokens from a rerank response."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0, 0
    prompt = usage.get("prompt_tokens") or usage.get("total_tokens") or 0
    total = usage.get("total_tokens") or prompt or 0
    try:
        return int(prompt), int(total)
    except (TypeError, ValueError):
        return 0, 0


def rerank_texts(
    query: str,
    documents: Sequence[str],
    *,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Rerank ``documents`` against ``query`` on prepaid Novita BGE rerank.

    Returns a list of ``{"index", "score"}`` in reranker order. Empty
    documents or a missing key return ``[]`` without an HTTP call.
    Raises ``RuntimeError`` with a coarse code on HTTP failure so callers
    can keep vector order.
    """
    texts = [str(item) for item in documents][:MAX_RERANK_DOCUMENTS]
    if not query.strip() or not texts:
        return []
    if not rerank_enabled():
        return []
    key = resolve_novita_api_key()
    if not key:
        return []
    limit = daily_rerank_call_limit()
    if today_outbound_calls("novita_rerank") >= limit:
        raise RuntimeError("skipped_budget")
    fetch_n = len(texts) if top_n is None else max(1, min(int(top_n), len(texts)))
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": NOVITA_RERANK_MODEL,
        "query": query.strip()[:RERANK_QUERY_CHARS],
        "documents": texts,
        "top_n": fetch_n,
    }
    started = time.perf_counter()
    try:
        response = requests.post(
            NOVITA_RERANK_URL,
            json=body,
            headers=headers,
            timeout=RERANK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("non-object JSON")
        results = _parse_results(payload)
        prompt_tokens, total_tokens = _parse_usage(payload)
        _record_outbound(
            ok=True,
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )
        return results
    except RuntimeError:
        raise
    except Exception as exc:
        code = classify_upstream_error(exc)
        latency_ms = int((time.perf_counter() - started) * 1000)
        _record_outbound(ok=False, latency_ms=latency_ms, error=code)
        logger.warning("Novita prepaid rerank failed: %s", code)
        raise RuntimeError(code) from exc
