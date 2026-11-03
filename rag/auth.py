"""Simple shared-token auth for mutating RAG operations."""

from __future__ import annotations

import hmac
import os
from typing import Annotated

from fastapi import Header, HTTPException, status


def resolve_api_token() -> str | None:
    """Return the configured write-access token, if any."""
    token = os.getenv("RAG_API_TOKEN")
    if token is None:
        return None
    token = token.strip()
    return token or None


def token_configured() -> bool:
    """Return whether write auth is enabled via ``RAG_API_TOKEN``."""
    return resolve_api_token() is not None


def verify_write_token(provided: str | None) -> None:
    """Validate a caller-supplied token for write access.

    When ``RAG_API_TOKEN`` is unset, writes are allowed (local/dev).
    When set, ``provided`` must match exactly.
    """
    expected = resolve_api_token()
    if expected is None:
        return
    if not provided or not hmac.compare_digest(provided, expected):
        raise PermissionError("Invalid or missing RAG write token")


def extract_bearer_token(authorization: str | None) -> str | None:
    """Parse a Bearer token from an Authorization header value."""
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


async def require_write_access(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """FastAPI dependency that gates mutating HTTP endpoints."""
    provided = x_api_key or extract_bearer_token(authorization)
    try:
        verify_write_token(provided)
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
