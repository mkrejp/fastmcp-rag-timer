"""Local CLIP image/text embeddings for the ``image_vector`` Qdrant field."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Sequence

from rag.config import resolve_rag_data_root
from rag.logging_setup import get_logger

# Paired FastEmbed CLIP ViT-B/32 models (512-dim, shared semantic space).
CLIP_VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"
CLIP_TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"
IMAGE_EMBEDDING_DIM = 512

logger = get_logger("rag.image_embed")

_vision_lock = threading.Lock()
_text_lock = threading.Lock()
_vision_model: Any = None
_text_model: Any = None


def image_embedding_enabled() -> bool:
    """Return whether visual embedding and ``search_images`` are enabled."""
    raw = os.getenv("IMAGE_EMBEDDING_ENABLED", "false").strip().lower()
    return raw in ("1", "true", "yes", "on")


def resolve_image_embed_cache_dir() -> str:
    """Return the CLIP ONNX cache directory under ``RAG_DATA_ROOT``."""
    override = os.getenv("IMAGE_EMBEDDING_CACHE_PATH", "").strip()
    if override:
        return override
    return str(Path(resolve_rag_data_root()) / "clip")


def reset_image_embed_runtime() -> None:
    """Drop cached CLIP models (tests)."""
    global _vision_model, _text_model
    with _vision_lock:
        _vision_model = None
    with _text_lock:
        _text_model = None


def _as_float_list(vector: Any) -> list[float]:
    """Coerce a FastEmbed/numpy vector into a Python list of floats."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _load_vision_model() -> Any:
    """Import FastEmbed and construct the CLIP vision model."""
    try:
        from fastembed import ImageEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "IMAGE_EMBEDDING_ENABLED requires the fastembed package with "
            "ImageEmbedding support."
        ) from exc
    cache_dir = resolve_image_embed_cache_dir()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Loading CLIP vision model %s cache_dir=%s (lazy; adds RAM beside BGE-M3)",
        CLIP_VISION_MODEL,
        cache_dir,
    )
    return ImageEmbedding(model_name=CLIP_VISION_MODEL, cache_dir=cache_dir)


def _load_text_model() -> Any:
    """Import FastEmbed and construct the CLIP text tower for query embedding."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "IMAGE_EMBEDDING_ENABLED requires the fastembed package with "
            "TextEmbedding support."
        ) from exc
    cache_dir = resolve_image_embed_cache_dir()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Loading CLIP text model %s cache_dir=%s",
        CLIP_TEXT_MODEL,
        cache_dir,
    )
    return TextEmbedding(model_name=CLIP_TEXT_MODEL, cache_dir=cache_dir)


def _get_vision_model() -> Any:
    """Return the process-wide CLIP vision model."""
    global _vision_model
    if _vision_model is not None:
        return _vision_model
    with _vision_lock:
        if _vision_model is None:
            _vision_model = _load_vision_model()
        return _vision_model


def _get_text_model() -> Any:
    """Return the process-wide CLIP text model."""
    global _text_model
    if _text_model is not None:
        return _text_model
    with _text_lock:
        if _text_model is None:
            _text_model = _load_text_model()
        return _text_model


def embed_image_path(path: Path) -> list[float]:
    """Embed one on-disk image into ``image_vector`` space."""
    if not image_embedding_enabled():
        raise RuntimeError(
            "IMAGE_EMBEDDING_ENABLED is false; enable it to embed or search images."
        )
    model = _get_vision_model()
    vectors = list(model.embed([str(path)]))
    if not vectors:
        raise ValueError(f"CLIP vision returned no vector for {path}")
    vector = _as_float_list(vectors[0])
    if len(vector) != IMAGE_EMBEDDING_DIM:
        raise ValueError(
            f"CLIP vision model {CLIP_VISION_MODEL} returned {len(vector)} "
            f"dimensions; expected {IMAGE_EMBEDDING_DIM}."
        )
    return vector


def embed_image_paths(paths: Sequence[Path]) -> list[list[float]]:
    """Embed multiple image files in one CLIP vision batch."""
    if not paths:
        return []
    if not image_embedding_enabled():
        raise RuntimeError(
            "IMAGE_EMBEDDING_ENABLED is false; enable it to embed or search images."
        )
    model = _get_vision_model()
    raw = list(model.embed([str(path) for path in paths]))
    vectors = [_as_float_list(item) for item in raw]
    if len(vectors) != len(paths):
        raise ValueError(
            f"CLIP vision returned {len(vectors)} vectors for {len(paths)} images."
        )
    for vector in vectors:
        if len(vector) != IMAGE_EMBEDDING_DIM:
            raise ValueError(
                f"CLIP vision model {CLIP_VISION_MODEL} returned "
                f"{len(vector)} dimensions; expected {IMAGE_EMBEDDING_DIM}."
            )
    return vectors


def embed_image_query(text: str) -> list[float]:
    """Embed a text query with the CLIP text tower (``search_images``)."""
    if not image_embedding_enabled():
        raise RuntimeError(
            "IMAGE_EMBEDDING_ENABLED is false; enable it to run search_images."
        )
    model = _get_text_model()
    vectors = list(model.embed([text]))
    if not vectors:
        raise ValueError("CLIP text tower returned no vector")
    vector = _as_float_list(vectors[0])
    if len(vector) != IMAGE_EMBEDDING_DIM:
        raise ValueError(
            f"CLIP text model {CLIP_TEXT_MODEL} returned {len(vector)} "
            f"dimensions; expected {IMAGE_EMBEDDING_DIM}."
        )
    return vector
