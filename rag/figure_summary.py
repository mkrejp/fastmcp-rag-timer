"""Extractive and optional LLM summaries for ``pdf_figure`` records."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from rag.config import (
    FIGURE_INDEX_VERSION,
    resolve_figure_llm_max_calls_per_doc,
    resolve_figure_llm_max_input_tokens,
    resolve_figure_summary_llm,
    resolve_figure_summary_llm_model,
    resolve_figure_summary_max_chars,
    resolve_figure_summary_top_k,
    resolve_rag_data_root,
)
from rag.logging_setup import get_logger

logger = get_logger("rag.figure_summary")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _normalize_space(text: str) -> str:
    """Collapse whitespace for dedup and overlap checks."""
    return re.sub(r"\s+", " ", text.strip())


def _token_overlap(a: str, b: str) -> float:
    """Return Jaccard-like token overlap between two strings."""
    ta = {t.lower() for t in re.findall(r"\w+", a) if len(t) > 2}
    tb = {t.lower() for t in re.findall(r"\w+", b) if len(t) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like spans."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [part.strip() for part in parts if part.strip()]


def build_extractive_summary(
    *,
    caption: str | None,
    related_text_refs: Sequence[Mapping[str, Any]],
    span_texts: Sequence[str],
    ocr_text: str | None,
) -> tuple[str, list[dict[str, Any]], str]:
    """Build caption + top-K sentences with ref indices (extractive, free).

    ``span_texts`` aligns with ``related_text_refs`` by index.
    Returns ``(summary, summary_sentences, summary_method)``.
    """
    scored: list[tuple[float, str, int]] = []
    kind_weights = {
        "explicit_ref": 1.0,
        "caption": 0.95,
        "same_page": 0.6,
        "adjacent_page": 0.5,
    }
    for ref_index, text in enumerate(span_texts):
        ref = related_text_refs[ref_index] if ref_index < len(related_text_refs) else {}
        kind = str(ref.get("kind") or "same_page")
        base = kind_weights.get(kind, 0.4)
        score = base * float(ref.get("score") or 1.0)
        for sentence in _split_sentences(text):
            if caption and _normalize_space(sentence) == _normalize_space(caption):
                continue
            if ocr_text and _token_overlap(sentence, ocr_text) > 0.6:
                continue
            scored.append((score, sentence, ref_index))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_k = resolve_figure_summary_top_k()
    chosen: list[tuple[str, int]] = []
    seen: set[str] = set()
    for _score, sentence, ref_index in scored:
        norm = _normalize_space(sentence)
        if norm in seen:
            continue
        if caption and _token_overlap(sentence, caption) > 0.85:
            continue
        seen.add(norm)
        chosen.append((sentence, ref_index))
        if len(chosen) >= top_k:
            break

    parts: list[str] = []
    sentences_out: list[dict[str, Any]] = []
    if caption:
        parts.append(caption)
        cap_ref = next(
            (
                i
                for i, ref in enumerate(related_text_refs)
                if ref.get("kind") == "caption"
            ),
            None,
        )
        if cap_ref is not None:
            sentences_out.append({"text": caption, "ref": cap_ref})
    for sentence, ref_index in chosen:
        parts.append(sentence)
        sentences_out.append({"text": sentence, "ref": ref_index})

    summary = " ".join(parts).strip()
    max_chars = resolve_figure_summary_max_chars()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return summary, sentences_out, "extractive"


def _cache_path(image_sha256: str, input_hash: str) -> Path:
    """Return the on-disk LLM cache file for a figure summary."""
    root = Path(resolve_rag_data_root()) / "figure_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{image_sha256}_{input_hash}.json"


def _input_hash(spans: Sequence[str]) -> str:
    """Hash summary input spans for cache lookup."""
    joined = "\n".join(spans)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _llm_summary_novita(
    spans: Sequence[str],
    *,
    related_text_refs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    """Call Novita OpenAI-compatible chat for figure sentences JSON."""
    import os

    api_key = os.getenv("NOVITA_API_KEY", "").strip()
    if not api_key:
        return None
    model = (
        resolve_figure_summary_llm_model()
        or os.getenv("NOVITA_EMBEDDING_MODEL", "").strip()
    )
    if not model:
        model = "meta-llama/Meta-Llama-3-8B-Instruct"
    base = "https://api.novita.ai/openai/v1"
    import requests

    budget = resolve_figure_llm_max_input_tokens()
    body_text = "\n".join(spans)[: budget * 4]
    prompt = (
        'Vrať JSON pole objektů {"text": string, "ref": integer} — věty '
        "shrnující obrázek pouze z dodaného textu. Každá věta musí mít platný "
        "ref index do seznamu zdrojů (0-based)."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": body_text},
        ],
        "temperature": 0.2,
    }
    try:
        response = requests.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            return None
        out: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            ref = item.get("ref")
            if not text or ref is None:
                continue
            try:
                ref_i = int(ref)
            except (TypeError, ValueError):
                continue
            if ref_i < 0 or ref_i >= len(related_text_refs):
                continue
            source = spans[ref_i] if ref_i < len(spans) else ""
            if _token_overlap(text, source) < 0.5:
                continue
            out.append({"text": text, "ref": ref_i})
        return out or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Figure LLM summary failed: %s", exc)
        return None


class FigureLlmBudget:
    """Per-document LLM call counter for figure summaries."""

    def __init__(self) -> None:
        """Initialize with configured per-doc cap."""
        self.remaining = resolve_figure_llm_max_calls_per_doc()

    def consume(self) -> bool:
        """Return whether a call is allowed and decrement the budget."""
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


def build_figure_summary(
    *,
    caption: str | None,
    related_text_refs: Sequence[Mapping[str, Any]],
    span_texts: Sequence[str],
    ocr_text: str | None,
    image_sha256: str,
    llm_budget: FigureLlmBudget | None = None,
) -> tuple[str, list[dict[str, Any]], str, str | None]:
    """Return summary text, sentences, method, and optional model id."""
    extractive_summary, extractive_sentences, _ = build_extractive_summary(
        caption=caption,
        related_text_refs=related_text_refs,
        span_texts=span_texts,
        ocr_text=ocr_text,
    )
    mode = resolve_figure_summary_llm()
    if mode != "novita" or not llm_budget or not llm_budget.consume():
        return extractive_summary, extractive_sentences, "extractive", None

    input_hash = _input_hash(span_texts)
    cache_file = _cache_path(image_sha256, input_hash)
    if cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            sentences = cached.get("summary_sentences") or []
            summary = str(cached.get("summary") or extractive_summary)
            return summary, sentences, "llm", cached.get("summary_model")
        except (OSError, json.JSONDecodeError):
            pass

    llm_sentences = _llm_summary_novita(span_texts, related_text_refs=related_text_refs)
    if not llm_sentences:
        return extractive_summary, extractive_sentences, "extractive", None
    summary = " ".join(str(s["text"]) for s in llm_sentences).strip()
    model = resolve_figure_summary_llm_model() or None
    try:
        cache_file.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "summary_sentences": llm_sentences,
                    "summary_model": model,
                    "figure_index_version": FIGURE_INDEX_VERSION,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass
    return summary, llm_sentences, "llm", model
