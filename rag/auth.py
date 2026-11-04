"""Inbound bearer-token authentication for selected HTTP routes."""

from __future__ import annotations

import hmac
import logging

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from rag.config import resolve_api_token, resolve_mcp_token

_log = logging.getLogger("rag.auth")

_bearer = HTTPBearer(auto_error=False)


def require_api_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """Require ``Authorization: Bearer`` matching ``RAG_API_TOKEN``."""
    expected = resolve_api_token()
    if expected is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG_API_TOKEN is not configured",
        )
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not credentials.credentials
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    provided = credentials.credentials.encode("utf-8")
    wanted = expected.encode("utf-8")
    if not hmac.compare_digest(provided, wanted):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return expected


def bearer_matches(header_value: str | None, expected: str) -> bool:
    """Constant-time check of an ``Authorization: Bearer <token>`` header."""
    if not header_value:
        return False
    scheme, _, token = header_value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return False
    return hmac.compare_digest(token.strip().encode("utf-8"), expected.encode("utf-8"))


class McpBearerAuth:
    """ASGI wrapper requiring the bearer token on every ``/mcp`` request (fail closed)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        expected = resolve_mcp_token()
        if expected is None:
            _log.warning("MCP_API_TOKEN/RAG_API_TOKEN not set; rejecting /mcp request")
            response = JSONResponse(
                {"detail": "MCP token is not configured"}, status_code=503
            )
            await response(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        if not bearer_matches(headers.get("authorization"), expected):
            response = JSONResponse(
                {"detail": "Missing or invalid bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def require_embed_access(request: Request) -> None:
    """Allow embedding REST routes for a signed-in Google user or a valid bearer token."""
    if "session" in request.scope and isinstance(
        request.session.get("google_user"), dict
    ):
        return
    expected = resolve_mcp_token()
    if expected is None:
        _log.warning("MCP_API_TOKEN/RAG_API_TOKEN not set; rejecting embedding request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API token is not configured",
        )
    if not bearer_matches(request.headers.get("authorization"), expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
