"""Entrypoint for the lightweight RAG engine."""

from __future__ import annotations

from rag.app import create_app
from rag.config import HOST, PORT, should_skip_default_app

# Skip eager app construction under pytest so tests can open isolated Qdrant paths.
# Production uses a single module-level ``app`` so uvicorn does not open Qdrant twice.
if should_skip_default_app():
    app = None  # type: ignore[assignment]
else:
    app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app or create_app(),
        host=HOST,
        port=PORT,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
