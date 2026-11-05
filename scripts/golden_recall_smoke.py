#!/usr/bin/env python3
"""Optional smoke check for Novita-ingest vs fastembed-query recall (phase D).

Run after re-ingesting a small golden set with ``EMBEDDING_PROVIDER_INGEST=novita``
and ``EMBEDDING_PROVIDER_QUERY=fastembed``. Expect high but not identical overlap
because Novita HTTP and local ONNX BGE-M3 can drift slightly.

Usage (requires live Novita key for ingest side only)::

    export EMBEDDING_PROVIDER_INGEST=novita
    export EMBEDDING_PROVIDER_QUERY=fastembed
    export BGE_M3_NOVITA_API_KEY=...
    python scripts/golden_recall_smoke.py "phrase from golden doc" "another phrase"
"""

from __future__ import annotations

import sys

from rag.embeddings import get_ingest_embedding, get_query_embedding


def main(argv: list[str]) -> int:
    """Print cosine similarity between ingest and query embeddings for each phrase."""
    if len(argv) < 2:
        print("Provide at least one query phrase.", file=sys.stderr)
        return 2
    for phrase in argv[1:]:
        ingest_vec = get_ingest_embedding(phrase)
        query_vec = get_query_embedding(phrase)
        if len(ingest_vec) != len(query_vec):
            print(f"{phrase!r}: dimension mismatch")
            continue
        dot = sum(a * b for a, b in zip(ingest_vec, query_vec, strict=True))
        norm_a = sum(a * a for a in ingest_vec) ** 0.5
        norm_b = sum(b * b for b in query_vec) ** 0.5
        cosine = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
        print(f"{phrase!r}: cosine(ingest, query)={cosine:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
