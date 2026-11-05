"""ASGI helpers for cleaner process shutdown (MCP SSE, embedded Qdrant)."""

from __future__ import annotations

import asyncio
import logging

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_log = logging.getLogger("rag.asgi_shutdown")


class CompleteResponseOnExit:
    """Finish an HTTP response if the inner app returns mid-stream (common on deploy).

    Streamable MCP clients keep ``GET /mcp/`` SSE connections open. When uvicorn
    receives SIGTERM, the MCP transport may terminate without sending the final
    ASGI body frame, which uvicorn logs as::

        ASGI callable returned without completing response.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = False
        complete = False

        async def send_wrapper(message: Message) -> None:
            nonlocal started, complete
            if message["type"] == "http.response.start":
                started = True
            elif message["type"] == "http.response.body":
                complete = not message.get("more_body", False)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except asyncio.CancelledError:
            await self._finish_if_needed(started, complete, send_wrapper)
            raise
        finally:
            await self._finish_if_needed(started, complete, send_wrapper)

    @staticmethod
    async def _finish_if_needed(
        started: bool, complete: bool, send: Send
    ) -> None:
        if not started or complete:
            return
        try:
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                }
            )
        except Exception:  # noqa: BLE001 - connection may already be gone
            _log.debug("Could not finalize HTTP response during shutdown", exc_info=True)
