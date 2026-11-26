"""Inbound bearer-token authentication for /ingest-json, /mcp, and embedding REST."""

from __future__ import annotations

import hmac
import json
import logging
from collections.abc import Mapping
from typing import Any, Literal

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag.config import resolve_api_token, resolve_mcp_read_token, resolve_mcp_token
from rag.node_info import MCP_WRITE_TOOLS

_log = logging.getLogger("rag.auth")

_bearer = HTTPBearer(auto_error=False)

McpAuthRole = Literal["write", "read"]

# JSON-RPC server-error range; paired with HTTP 403 for read-token write calls.
JSONRPC_FORBIDDEN = -32003
_READ_TOKEN_WRITE_TOOL_MESSAGE = "Read-only token cannot call write tools"


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


def mcp_auth_configured() -> bool:
    """Return whether any ``/mcp`` bearer (write or read-only) is configured."""
    return resolve_mcp_token() is not None or resolve_mcp_read_token() is not None


def classify_mcp_bearer(header_value: str | None) -> McpAuthRole | None:
    """Classify a Bearer header as write, read-only, or unmatched.

    Write token (``MCP_API_TOKEN`` / ``RAG_API_TOKEN``) wins when both match.
    Both configured tokens are compared with ``hmac.compare_digest``. Callers
    must never log ``header_value``.
    """
    write_token = resolve_mcp_token()
    read_token = resolve_mcp_read_token()
    write_ok = False
    if write_token is not None:
        write_ok = bearer_matches(header_value, write_token)
    read_ok = False
    if read_token is not None:
        read_ok = bearer_matches(header_value, read_token)
    if write_ok:
        return "write"
    if read_ok:
        return "read"
    return None


def _iter_rpc_objects(payload: object) -> list[dict[str, Any]]:
    """Return JSON-RPC objects from a single request or a batch array."""
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _rpc_tool_name(item: dict[str, Any]) -> str | None:
    """Return ``params.name`` from a JSON-RPC tools/call object."""
    params = item.get("params")
    if not isinstance(params, dict):
        return None
    name = params.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _forbidden_write_call(payload: object) -> dict[str, Any] | None:
    """Return the first tools/call of a mutating MCP tool, if any."""
    for item in _iter_rpc_objects(payload):
        if item.get("method") != "tools/call":
            continue
        name = _rpc_tool_name(item)
        if name in MCP_WRITE_TOOLS:
            return item
    return None


def _is_tools_list(payload: object) -> bool:
    """Return whether the JSON-RPC payload lists tools."""
    return any(
        item.get("method") == "tools/list" for item in _iter_rpc_objects(payload)
    )


def _filter_tools_list_payload(payload: object) -> object:
    """Drop mutating tools from a JSON-RPC tools/list result (single or batch)."""
    if isinstance(payload, list):
        return [_filter_tools_list_payload(item) for item in payload]
    if not isinstance(payload, dict):
        return payload
    result = payload.get("result")
    if not isinstance(result, dict):
        return payload
    tools = result.get("tools")
    if not isinstance(tools, list):
        return payload
    filtered = [
        tool
        for tool in tools
        if not (isinstance(tool, dict) and tool.get("name") in MCP_WRITE_TOOLS)
    ]
    new_result = dict(result)
    new_result["tools"] = filtered
    new_payload = dict(payload)
    new_payload["result"] = new_result
    return new_payload


def _rewrite_sse_tools_list(text: str) -> str:
    """Filter ``data:`` JSON-RPC lines in an SSE tools/list body."""
    rewritten: list[str] = []
    for line in text.splitlines(keepends=True):
        if not line.startswith("data:"):
            rewritten.append(line)
            continue
        newline = ""
        content = line[5:]
        if content.endswith("\r\n"):
            newline = "\r\n"
            content = content[:-2]
        elif content.endswith("\n"):
            newline = "\n"
            content = content[:-1]
        stripped = content.lstrip(" ")
        if not stripped.strip():
            rewritten.append(line)
            continue
        try:
            parsed: object = json.loads(stripped)
        except json.JSONDecodeError:
            rewritten.append(line)
            continue
        filtered = _filter_tools_list_payload(parsed)
        rewritten.append(
            "data: "
            + json.dumps(filtered, ensure_ascii=False, separators=(",", ":"))
            + newline
        )
    return "".join(rewritten)


def _rewrite_mcp_list_body(body: bytes, content_type: str) -> bytes:
    """Filter write tools out of a tools/list HTTP body (JSON or SSE)."""
    if not body:
        return body
    if "text/event-stream" in content_type:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return body
        return _rewrite_sse_tools_list(text).encode("utf-8")
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body
    filtered = _filter_tools_list_payload(parsed)
    return json.dumps(filtered, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _parse_json_body(body: bytes) -> object | None:
    """Parse a JSON request body, or ``None`` when empty/invalid."""
    if not body or not body.strip():
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


async def _read_http_body(receive: Receive) -> tuple[bytes, list[Message]]:
    """Read the ASGI request body; return leftover non-request messages."""
    chunks: list[bytes] = []
    extras: list[Message] = []
    while True:
        message = await receive()
        msg_type = message.get("type")
        if msg_type == "http.request":
            chunks.append(message.get("body") or b"")
            if not message.get("more_body", False):
                break
            continue
        extras.append(message)
        if msg_type == "http.disconnect":
            break
    return b"".join(chunks), extras


def _replay_receive(body: bytes, extras: list[Message], receive: Receive) -> Receive:
    """Replay a buffered request body, then continue with the original receive."""
    queued: list[Message] = [
        {"type": "http.request", "body": body, "more_body": False},
        *extras,
    ]
    index = 0

    async def replay() -> Message:
        nonlocal index
        if index < len(queued):
            message = queued[index]
            index += 1
            return message
        return await receive()

    return replay


def _jsonrpc_forbidden(request_id: Any) -> JSONResponse:
    """Return HTTP 403 with a JSON-RPC error body (never includes tokens)."""
    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": JSONRPC_FORBIDDEN,
                "message": _READ_TOKEN_WRITE_TOOL_MESSAGE,
            },
        },
        status_code=status.HTTP_403_FORBIDDEN,
    )


class _BufferedResponse:
    """Capture an ASGI HTTP response so the body can be rewritten."""

    def __init__(self, send: Send) -> None:
        self._send = send
        self.status = 200
        self.headers: list[tuple[bytes, bytes]] = []
        self.body = bytearray()
        self.started = False
        self.complete = False

    async def __call__(self, message: Message) -> None:
        """Buffer start/body frames; forward any other ASGI messages."""
        msg_type = message.get("type")
        if msg_type == "http.response.start":
            self.started = True
            self.status = int(message["status"])
            self.headers = list(message.get("headers") or [])
            return
        if msg_type == "http.response.body":
            self.body.extend(message.get("body") or b"")
            if not message.get("more_body", False):
                self.complete = True
            return
        await self._send(message)

    async def flush_filtered_tools_list(self) -> None:
        """Send the buffered response with write tools removed from tools/list."""
        if not self.started:
            return
        content_type = ""
        for key, value in self.headers:
            if key.lower() == b"content-type":
                content_type = value.decode("latin-1")
                break
        new_body = _rewrite_mcp_list_body(bytes(self.body), content_type)
        headers = [
            (key, value)
            for key, value in self.headers
            if key.lower() != b"content-length"
        ]
        headers.append((b"content-length", str(len(new_body)).encode("ascii")))
        await self._send(
            {
                "type": "http.response.start",
                "status": self.status,
                "headers": headers,
            }
        )
        await self._send(
            {
                "type": "http.response.body",
                "body": new_body,
                "more_body": False,
            }
        )


def _decode_request_headers(scope: Scope) -> dict[str, str]:
    """Lower-case ASGI header names for bearer checks."""
    return {
        k.decode("latin-1").lower(): v.decode("latin-1")
        for k, v in scope.get("headers", [])
    }


def _mcp_bearer_json_response(
    headers: Mapping[str, str],
    *,
    unconfigured_detail: str,
) -> JSONResponse | None:
    """Return a bearer auth error response, or ``None`` when the header is valid."""
    if not mcp_auth_configured():
        return JSONResponse({"detail": unconfigured_detail}, status_code=503)
    if classify_mcp_bearer(headers.get("authorization")) is None:
        return JSONResponse(
            {"detail": "Missing or invalid bearer token"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return None


class _AsgiHttpMiddleware:
    """Shared ASGI boilerplate for HTTP-only middleware wrappers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def _skip_non_http(self, scope: Scope, receive: Receive, send: Send) -> bool:
        """Forward non-HTTP scopes. Return True when the request was forwarded."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return True
        return False


class McpBearerAuth(_AsgiHttpMiddleware):
    """ASGI wrapper requiring the bearer token on every ``/mcp`` request (fail closed).

    Accepts ``MCP_API_TOKEN`` (falling back to ``RAG_API_TOKEN``) for full access.
    When ``MCP_READ_TOKEN`` is set, that token is also accepted for read-only MCP:
    ``tools/list`` omits write tools and ``tools/call`` of a write tool is HTTP 403
    with a JSON-RPC error. ``MCP_READ_TOKEN`` is not used by embedding REST routes.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if await self._skip_non_http(scope, receive, send):
            return
        headers = _decode_request_headers(scope)
        denied = _mcp_bearer_json_response(
            headers, unconfigured_detail="MCP token is not configured"
        )
        if denied is not None:
            if denied.status_code == 503:
                _log.warning(
                    "MCP_API_TOKEN/RAG_API_TOKEN/MCP_READ_TOKEN not set; "
                    "rejecting /mcp request"
                )
            await denied(scope, receive, send)
            return
        role = classify_mcp_bearer(headers.get("authorization"))
        assert role is not None
        if role == "write":
            await self.app(scope, receive, send)
            return
        await self._enforce_read_only(scope, receive, send)

    async def _enforce_read_only(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Allow read MCP methods; reject write tools/call; filter tools/list."""
        method = (scope.get("method") or "GET").upper()
        if method not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        body, extras = await _read_http_body(receive)
        replay = _replay_receive(body, extras, receive)
        payload = _parse_json_body(body)
        if payload is not None:
            forbidden = _forbidden_write_call(payload)
            if forbidden is not None:
                response = _jsonrpc_forbidden(forbidden.get("id"))
                await response(scope, receive, send)
                return
            if _is_tools_list(payload):
                buffered = _BufferedResponse(send)
                await self.app(scope, replay, buffered)
                await buffered.flush_filtered_tools_list()
                return
        await self.app(scope, replay, send)


def _http_path(scope: Scope) -> str:
    """Normalized request path without trailing slash (except ``/``)."""
    path = scope.get("path") or ""
    if path != "/" and path.endswith("/"):
        return path.rstrip("/")
    return path


def _requires_http_read_auth(method: str, path: str) -> bool:
    """Return whether the HTTP request needs a read/write bearer token."""
    if path in {"/openapi.json", "/docs", "/redoc"} or path.startswith("/docs/"):
        return True
    if method.upper() != "GET":
        return False
    return path == "/documents" or path.startswith("/documents/")


class HttpReadAuthMiddleware(_AsgiHttpMiddleware):
    """ASGI guard for catalog GET routes and FastAPI OpenAPI surfaces."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if await self._skip_non_http(scope, receive, send):
            return
        method = (scope.get("method") or "GET").upper()
        path = _http_path(scope)
        if not _requires_http_read_auth(method, path):
            await self.app(scope, receive, send)
            return
        headers = _decode_request_headers(scope)
        denied = _mcp_bearer_json_response(
            headers, unconfigured_detail="Read token is not configured"
        )
        if denied is not None:
            if denied.status_code == 503:
                _log.warning(
                    "MCP_API_TOKEN/RAG_API_TOKEN/MCP_READ_TOKEN not set; "
                    "rejecting %s %s",
                    method,
                    path,
                )
            await denied(scope, receive, send)
            return
        await self.app(scope, receive, send)


def require_read_access(request: Request) -> None:
    """Allow catalog and OpenAPI HTTP reads for a valid write or read-only bearer.

    Accepts ``MCP_API_TOKEN`` / ``RAG_API_TOKEN`` or ``MCP_READ_TOKEN`` (same
    classification as ``/mcp``). Fails closed when no bearer env is configured.
    """
    if not mcp_auth_configured():
        _log.warning(
            "MCP_API_TOKEN/RAG_API_TOKEN/MCP_READ_TOKEN not set; "
            "rejecting protected read request"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Read token is not configured",
        )
    if classify_mcp_bearer(request.headers.get("authorization")) is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_embed_access(request: Request) -> None:
    """Allow embedding REST routes for a signed-in Google user or a valid bearer token.

    Only ``MCP_API_TOKEN`` / ``RAG_API_TOKEN`` are accepted. ``MCP_READ_TOKEN``
    never authorizes these HTTP routes (uploads, deletes, metadata, images).
    """
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
