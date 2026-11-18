"""Entrypoint for the read-only RAG node (APP_MODE=reader)."""

from __future__ import annotations

import os

from rag.config import HOST, PORT, should_skip_default_app

# Force reader role before app construction (overridable only by explicit env set earlier).
os.environ.setdefault("APP_MODE", "reader")

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
