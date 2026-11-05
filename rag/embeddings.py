"""Embedding helpers with a Novita HTTP backend and optional local FastEmbed."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import requests

from rag.accounting import get_global_store
from rag.config import resolve_rag_data_root
from rag.logging_setup import get_logger

EmbeddingProvider = Literal["novita", "fastembed"]
QueryEmbeddingProvider = Literal["fastembed"]

DEFAULT_EMBEDDING_PROVIDER: EmbeddingProvider = "novita"
SUPPORTED_EMBEDDING_PROVIDERS: tuple[EmbeddingProvider, ...] = ("novita", "fastembed")
BGE_M3_MODEL_FAMILY = "bge-m3"

NOVITA_EMBEDDING_MODEL = "baai/bge-m3"
NOVITA_EMBEDDING_URL = "https://api.novita.ai/openai/v1/embeddings"
NOVITA_EMBEDDING_DIM = 1024

# FastEmbed does not ship BGE-M3 as a built-in dense model (qdrant/fastembed#107).
# Register the official Hugging Face ONNX export: same 1024-dim multilingual model
# as Novita. Vectors are not guaranteed identical across runtimes; re-ingest after
# switching providers even though the collection size stays 1024.
FASTEMBED_MODEL = "BAAI/bge-m3"
FASTEMBED_DIM = 1024
FASTEMBED_HF_SOURCE = "BAAI/bge-m3"
FASTEMBED_ONNX_FILE = "onnx/model.onnx"
FASTEMBED_ONNX_DATA = "onnx/model.onnx_data"
FASTEMBED_SIZE_GB = 2.27

# Backward-compatible aliases for the historical Novita BGE-M3 constants.
EMBEDDING_MODEL = NOVITA_EMBEDDING_MODEL
EMBEDDING_URL = NOVITA_EMBEDDING_URL
EMBEDDING_DIM = NOVITA_EMBEDDING_DIM
TEXT_EMBEDDING_DIM = NOVITA_EMBEDDING_DIM

logger = get_logger("rag.embeddings")

_fastembed_lock = threading.Lock()
_fastembed_model: Any = None
_fastembed_registered = False


def _parse_provider_env(name: str) -> EmbeddingProvider | None:
    """Parse one provider env var, or return ``None`` when unset."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    if raw not in SUPPORTED_EMBEDDING_PROVIDERS:
        supported = ", ".join(SUPPORTED_EMBEDDING_PROVIDERS)
        raise ValueError(f"Unknown {name}={raw!r}. Use one of: {supported}.")
    return cast(EmbeddingProvider, raw)


def resolve_embedding_provider() -> EmbeddingProvider:
    """Return the default embedding backend from ``EMBEDDING_PROVIDER``.

    Unset or empty values keep the Novita HTTP API so production is unchanged.
    """
    parsed = _parse_provider_env("EMBEDDING_PROVIDER")
    return parsed if parsed is not None else DEFAULT_EMBEDDING_PROVIDER


def resolve_embedding_provider_ingest() -> EmbeddingProvider:
    """Return the ingest/reindex embedding backend (process-wide)."""
    parsed = _parse_provider_env("EMBEDDING_PROVIDER_INGEST")
    return parsed if parsed is not None else resolve_embedding_provider()


def resolve_embedding_provider_query_unvalidated() -> EmbeddingProvider:
    """Return the query provider before startup validation."""
    for name in ("EMBEDDING_PROVIDER_QUERY", "EMBEDDING_PROVIDER_SEARCH"):
        parsed = _parse_provider_env(name)
        if parsed is not None:
            return parsed
    return resolve_embedding_provider()


def validate_embedding_configuration() -> None:
    """Refuse startup when search would use a remote provider or text spaces mismatch."""
    ingest = resolve_embedding_provider_ingest()
    query = resolve_embedding_provider_query_unvalidated()
    if query != "fastembed":
        raise ValueError(
            "Search paths must use local fastembed (BAAI/bge-m3, 1024-dim). "
            f"Resolved EMBEDDING_PROVIDER_QUERY/SEARCH={query!r}. "
            "Set EMBEDDING_PROVIDER_QUERY=fastembed (ingest may stay novita)."
        )
    ingest_dim = provider_text_dim(ingest)
    query_dim = provider_text_dim("fastembed")
    if ingest_dim != query_dim:
        raise ValueError(
            f"Ingest provider {ingest!r} uses {ingest_dim}-dim text vectors but "
            f"query fastembed uses {query_dim}-dim. Both must be BGE-M3 1024."
        )
    if ingest not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise ValueError(f"Unsupported ingest provider: {ingest!r}")


def resolve_embedding_provider_query() -> QueryEmbeddingProvider:
    """Return the query embedding backend (always local fastembed after validation)."""
    validate_embedding_configuration()
    return "fastembed"


def provider_text_dim(provider: EmbeddingProvider) -> int:
    """Return the BGE-M3 text dimension for ``provider``."""
    if provider == "fastembed":
        return FASTEMBED_DIM
    return NOVITA_EMBEDDING_DIM


def provider_text_model(provider: EmbeddingProvider) -> str:
    """Return the model id used for text embeddings by ``provider``."""
    if provider == "fastembed":
        return FASTEMBED_MODEL
    return NOVITA_EMBEDDING_MODEL


def embedding_provider_status() -> dict[str, str | int | None]:
    """Summarize ingest vs query providers for health/admin (no secrets)."""
    ingest = resolve_embedding_provider_ingest()
    query = resolve_embedding_provider_query_unvalidated()
    return {
        "default_provider": resolve_embedding_provider(),
        "ingest_provider": ingest,
        "ingest_model": provider_text_model(ingest),
        "ingest_dim": provider_text_dim(ingest),
        "query_provider": query,
        "query_model": provider_text_model("fastembed"),
        "query_dim": provider_text_dim("fastembed"),
        "text_model_family": BGE_M3_MODEL_FAMILY,
    }


def embedding_model() -> str:
    """Return the ingest-side text model id (Qdrant ``text_vector`` family)."""
    return provider_text_model(resolve_embedding_provider_ingest())


def embedding_url() -> str | None:
    """Return the Novita embeddings URL when ingest uses Novita."""
    if resolve_embedding_provider_ingest() == "fastembed":
        return None
    return NOVITA_EMBEDDING_URL


def embedding_dim() -> int:
    """Return the ``text_vector`` size (always BGE-M3 1024)."""
    return TEXT_EMBEDDING_DIM


def resolve_fastembed_cache_dir() -> str:
    """Return the FastEmbed model cache directory.

    ``FASTEMBED_CACHE_PATH`` wins; otherwise models live under
    ``{RAG_DATA_ROOT}/fastembed`` so a Railway volume keeps the ~2.3GB ONNX
    files across deploys.
    """
    override = os.getenv("FASTEMBED_CACHE_PATH", "").strip()
    if override:
        return override
    return str(Path(resolve_rag_data_root()) / "fastembed")


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


def reset_fastembed_runtime() -> None:
    """Drop the cached FastEmbed model. Used by tests."""
    global _fastembed_model, _fastembed_registered
    with _fastembed_lock:
        _fastembed_model = None
        _fastembed_registered = False


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
    model: str | None = None,
    count_outbound: bool = True,
    provider: str | None = None,
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
        model=model or embedding_model(),
        error=error,
        count_outbound=count_outbound,
        provider=provider or resolve_embedding_provider(),
    )


def _as_float_list(vector: Any) -> list[float]:
    """Coerce a FastEmbed/numpy vector into a Python list of floats."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _ensure_bge_m3_registered() -> None:
    """Register BGE-M3 on FastEmbed if the installed package does not ship it."""
    global _fastembed_registered
    if _fastembed_registered:
        return
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    names = {
        str(item.get("model", "")).lower()
        for item in TextEmbedding.list_supported_models()
    }
    if FASTEMBED_MODEL.lower() not in names:
        TextEmbedding.add_custom_model(
            model=FASTEMBED_MODEL,
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf=FASTEMBED_HF_SOURCE),
            dim=FASTEMBED_DIM,
            model_file=FASTEMBED_ONNX_FILE,
            additional_files=[FASTEMBED_ONNX_DATA],
            description=(
                "BGE-M3 dense embeddings, 1024-dim, multilingual. "
                "Registered from the Hugging Face ONNX export because FastEmbed "
                "does not include this model in its built-in dense catalog."
            ),
            license="mit",
            size_in_gb=FASTEMBED_SIZE_GB,
        )
        logger.info(
            "Registered FastEmbed custom model %s (%s-dim) from %s",
            FASTEMBED_MODEL,
            FASTEMBED_DIM,
            FASTEMBED_HF_SOURCE,
        )
    _fastembed_registered = True


def _load_fastembed_model() -> Any:
    """Import FastEmbed and construct the BGE-M3 text embedding model."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "EMBEDDING_PROVIDER=fastembed requires the fastembed package. "
            "Install it or leave EMBEDDING_PROVIDER unset to keep Novita."
        ) from exc
    _ensure_bge_m3_registered()
    cache_dir = resolve_fastembed_cache_dir()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Loading FastEmbed model %s cache_dir=%s (first call may download ~%.1fGB)",
        FASTEMBED_MODEL,
        cache_dir,
        FASTEMBED_SIZE_GB,
    )
    return TextEmbedding(model_name=FASTEMBED_MODEL, cache_dir=cache_dir)


def _get_fastembed_model() -> Any:
    """Return the process-wide FastEmbed model, loading it on first use."""
    global _fastembed_model
    if _fastembed_model is not None:
        return _fastembed_model
    with _fastembed_lock:
        if _fastembed_model is None:
            _fastembed_model = _load_fastembed_model()
        return _fastembed_model


def _get_novita_embedding(
    text: str, *, api_key: str | None = None, expected_dim: int | None = None
) -> list[float]:
    """Embed ``text`` with Novita's OpenAI-compatible BGE-M3 endpoint."""
    key = api_key if api_key is not None else resolve_api_key()
    if not key:
        raise ValueError("Missing Novita credentials. Set BGE_M3_NOVITA_API_KEY.")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {"input": text, "model": NOVITA_EMBEDDING_MODEL}
    input_chars = len(text)
    started = time.perf_counter()
    try:
        response = requests.post(
            NOVITA_EMBEDDING_URL, json=payload, headers=headers, timeout=60
        )
        response.raise_for_status()
        body = response.json()
        embedding = body["data"][0]["embedding"]
        expected = expected_dim if expected_dim is not None else TEXT_EMBEDDING_DIM
        if len(embedding) != expected:
            raise ValueError(
                f"Novita model {NOVITA_EMBEDDING_MODEL} returned "
                f"{len(embedding)} dimensions; expected {expected}."
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
            model=NOVITA_EMBEDDING_MODEL,
            count_outbound=True,
            provider="novita",
        )
        return list(embedding)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _record_embedding(
            ok=False,
            input_chars=input_chars,
            latency_ms=latency_ms,
            error=str(exc),
            model=NOVITA_EMBEDDING_MODEL,
            count_outbound=True,
            provider="novita",
        )
        raise


def _get_fastembed_embeddings(
    texts: Sequence[str], *, expected_dim: int | None = None
) -> list[list[float]]:
    """Embed ``texts`` with the locally loaded BGE-M3 FastEmbed model."""
    expected = expected_dim if expected_dim is not None else TEXT_EMBEDDING_DIM
    input_chars_list = [len(text) for text in texts]
    started = time.perf_counter()
    try:
        model = _get_fastembed_model()
        raw_vectors = list(model.embed(list(texts)))
        vectors = [_as_float_list(item) for item in raw_vectors]
        if len(vectors) != len(texts):
            raise ValueError(
                f"FastEmbed model {FASTEMBED_MODEL} returned {len(vectors)} "
                f"vectors for {len(texts)} texts."
            )
        for vector in vectors:
            if len(vector) != expected:
                raise ValueError(
                    f"FastEmbed model {FASTEMBED_MODEL} returned "
                    f"{len(vector)} dimensions; expected {expected}."
                )
        latency_ms = int((time.perf_counter() - started) * 1000)
        per_call = latency_ms // max(len(texts), 1)
        remainder = latency_ms - per_call * len(texts)
        for index, chars in enumerate(input_chars_list):
            extra = remainder if index == 0 else 0
            _record_embedding(
                ok=True,
                input_chars=chars,
                latency_ms=per_call + extra,
                model=FASTEMBED_MODEL,
                count_outbound=False,
                provider="fastembed",
            )
        return vectors
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        for chars in input_chars_list:
            _record_embedding(
                ok=False,
                input_chars=chars,
                latency_ms=latency_ms,
                error=str(exc),
                model=FASTEMBED_MODEL,
                count_outbound=False,
                provider="fastembed",
            )
        raise


def get_ingest_embeddings(
    texts: Sequence[str], *, api_key: str | None = None
) -> list[list[float]]:
    """Embed ``texts`` on the ingest path (``EMBEDDING_PROVIDER_INGEST``)."""
    if not texts:
        return []
    provider = resolve_embedding_provider_ingest()
    expected = provider_text_dim(provider)
    if provider == "fastembed":
        return _get_fastembed_embeddings(texts, expected_dim=expected)
    return [
        _get_novita_embedding(text, api_key=api_key, expected_dim=expected)
        for text in texts
    ]


def get_ingest_embedding(text: str, *, api_key: str | None = None) -> list[float]:
    """Embed one string on the ingest path."""
    return get_ingest_embeddings([text], api_key=api_key)[0]


def get_query_embeddings(texts: Sequence[str]) -> list[list[float]]:
    """Embed ``texts`` on the search path (local fastembed only; no outbound tokens)."""
    if not texts:
        return []
    validate_embedding_configuration()
    expected = provider_text_dim("fastembed")
    return _get_fastembed_embeddings(texts, expected_dim=expected)


def get_query_embedding(text: str) -> list[float]:
    """Embed one query string with local fastembed (MCP/HTTP search)."""
    return get_query_embeddings([text])[0]


def get_embeddings(
    texts: Sequence[str], *, api_key: str | None = None
) -> list[list[float]]:
    """Embed ``texts`` with the ingest provider (backward-compatible alias)."""
    return get_ingest_embeddings(texts, api_key=api_key)


def get_embedding(text: str, *, api_key: str | None = None) -> list[float]:
    """Embed ``text`` with the ingest provider (backward-compatible alias)."""
    return get_ingest_embedding(text, api_key=api_key)
