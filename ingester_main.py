"""Entrypoint for the write/ingest RAG node (APP_MODE=ingester)."""

from __future__ import annotations

import os

from rag.config import HOST, PORT, should_skip_default_app

os.environ.setdefault("APP_MODE", "ingester")

from rag.app import create_app  # noqa: E402

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
        timeout_graceful_shutdown=15,
    )
