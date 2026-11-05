"""Tests for ASGI shutdown helpers."""

from __future__ import annotations

import pytest
from starlette.types import ASGIApp, Receive, Scope, Send


async def _incomplete_app(scope: Scope, receive: Receive, send: Send) -> None:
    """Simulate an SSE handler that starts a response then returns."""
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )


@pytest.mark.anyio
async def test_complete_response_on_exit_finishes_body() -> None:
    """Incomplete responses should receive a final empty body frame."""
    from rag.asgi_shutdown import CompleteResponseOnExit

    messages: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    scope: Scope = {
        "type": "http",
        "method": "GET",
        "path": "/mcp/",
        "headers": [],
    }

    async def receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    wrapped: ASGIApp = CompleteResponseOnExit(_incomplete_app)
    await wrapped(scope, receive, send)

    assert messages[0]["type"] == "http.response.start"
    assert messages[-1]["type"] == "http.response.body"
    assert messages[-1].get("more_body") is False
