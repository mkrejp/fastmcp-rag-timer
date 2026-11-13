"""Public RAG node identity, catalog, and optional write-role enforcement."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from html import escape
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from fastapi import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from rag.config import resolve_public_base_url
from rag.logging_setup import get_logger

logger = get_logger("rag.node_info")

ROLE_WRITE = "write"
ROLE_READ = "read"
ROLE_STOP = "stop"
ROLE_ORDER = (ROLE_WRITE, ROLE_READ, ROLE_STOP)
ROLE_LABELS = {
    ROLE_WRITE: "ZÁPIS",
    ROLE_READ: "ČTENÍ",
    ROLE_STOP: "STOP",
}

DEFAULT_NODE_00001_NAME = "rag-timer-00001"
DEFAULT_NODE_00002_NAME = "rag-timer-00002"
DEFAULT_NODE_00001_URL = "https://rag-timer-00001-production.up.railway.app"
DEFAULT_NODE_00002_URL = "https://rag-artnouveau.up.railway.app"

# Railway injects these; only these names are ever copied into public payloads.
RAILWAY_PUBLIC_ENV_KEYS = (
    "RAILWAY_SERVICE_NAME",
    "RAILWAY_REPLICA_REGION",
    "RAILWAY_ENVIRONMENT_NAME",
    "RAILWAY_PUBLIC_DOMAIN",
)

MCP_WRITE_TOOLS = frozenset(
    {
        "ingest_text",
        "update_document",
        "update_document_metadata",
        "supersede_document",
        "unsupersede_document",
        "link_locality",
        "verify_link",
        "sync_links_to_trails",
        "verify_image_record",
        "reindex_figures",
        "refine_geo",
        "refine_geo_batch",
    }
)

_ROLE_ALIASES: dict[str, frozenset[str]] = {
    "write": frozenset({ROLE_WRITE}),
    "zapis": frozenset({ROLE_WRITE}),
    "zapisu": frozenset({ROLE_WRITE}),
    "w": frozenset({ROLE_WRITE}),
    "read": frozenset({ROLE_READ}),
    "cteni": frozenset({ROLE_READ}),
    "cte": frozenset({ROLE_READ}),
    "ro": frozenset({ROLE_READ}),
    "readonly": frozenset({ROLE_READ}),
    "replica": frozenset({ROLE_READ}),
    "stop": frozenset({ROLE_STOP}),
    "stopped": frozenset({ROLE_STOP}),
    "none": frozenset({ROLE_STOP}),
    "off": frozenset({ROLE_STOP}),
    "disabled": frozenset({ROLE_STOP}),
    "rw": frozenset({ROLE_WRITE, ROLE_READ}),
    "readwrite": frozenset({ROLE_WRITE, ROLE_READ}),
    "both": frozenset({ROLE_WRITE, ROLE_READ}),
    "primary": frozenset({ROLE_WRITE, ROLE_READ}),
    "all": frozenset({ROLE_WRITE, ROLE_READ}),
}

_NODE_ID_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")


@dataclass(frozen=True)
class NodeSpec:
    """One configured RAG node (name, public URL, roles)."""

    name: str
    url: str
    roles: frozenset[str]


def _strip_accents(value: str) -> str:
    """Return ``value`` without combining diacritics."""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_role_token(token: str) -> str:
    """Normalize a role fragment for alias lookup."""
    return _strip_accents(token).strip().lower()


def parse_roles(
    raw: object, *, default: frozenset[str] | None = None
) -> frozenset[str]:
    """Parse a role string, list, or JSON value into canonical role codes.

    Accepts Czech labels (``ZÁPIS``, ``ČTENÍ``, ``STOP``), English aliases
    (``write``, ``read``, ``replica``, ``primary``), and combinations joined
    by ``+``, ``,``, ``/``, ``|``, or whitespace.
    """
    fallback = default if default is not None else frozenset({ROLE_READ})
    tokens: list[str] = []
    if raw is None:
        return fallback
    if isinstance(raw, (list, tuple, set, frozenset)):
        tokens = [str(item) for item in raw]
    else:
        text = str(raw).strip()
        if not text:
            return fallback
        tokens = re.split(r"[\s,+/|]+", text)
    roles: set[str] = set()
    for token in tokens:
        key = _normalize_role_token(token)
        if not key:
            continue
        mapped = _ROLE_ALIASES.get(key)
        if mapped is None and key in {ROLE_WRITE, ROLE_READ, ROLE_STOP}:
            mapped = frozenset({key})
        if mapped:
            roles.update(mapped)
    if not roles:
        return fallback
    if ROLE_STOP in roles:
        return frozenset({ROLE_STOP})
    return frozenset(roles)


def format_role_labels(roles: Iterable[str]) -> list[str]:
    """Return Czech role labels in display order."""
    role_set = frozenset(roles)
    if ROLE_STOP in role_set:
        return [ROLE_LABELS[ROLE_STOP]]
    return [ROLE_LABELS[role] for role in ROLE_ORDER if role in role_set]


def format_role_text(roles: Iterable[str]) -> str:
    """Return a Czech display string such as ``ZÁPIS + ČTENÍ``."""
    return " + ".join(format_role_labels(roles))


def writes_allowed_for(roles: Iterable[str]) -> bool:
    """Return whether the role set permits mutating operations."""
    role_set = frozenset(roles)
    return ROLE_WRITE in role_set and ROLE_STOP not in role_set


def enforce_node_role() -> bool:
    """Return whether write endpoints should honor ``RAG_NODE_ROLE``.

    ``RAG_ENFORCE_NODE_ROLE`` is off by default so current behaviour is unchanged.
    """
    raw = os.getenv("RAG_ENFORCE_NODE_ROLE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _railway_public_env() -> dict[str, str]:
    """Return the allow-listed Railway-provided variables that are set."""
    values: dict[str, str] = {}
    for key in RAILWAY_PUBLIC_ENV_KEYS:
        value = os.getenv(key, "").strip()
        if value:
            values[key] = value
    return values


def _safe_http_url(url: str) -> str | None:
    """Return ``url`` when it is an ``http(s)`` URL with a host, else ``None``."""
    text = url.strip()
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return text.rstrip("/")
    return None


def _host_from_url(url: str) -> str:
    """Return a lowercase host:port (default ports stripped) for matching."""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.hostname or "").lower()
    port = parsed.port
    if not host:
        return ""
    if port in {None, 80, 443}:
        return host
    return f"{host}:{port}"


def _urls_match(left: str, right: str) -> bool:
    """Return whether two public URLs refer to the same host."""
    left_host = _host_from_url(left)
    right_host = _host_from_url(right)
    return bool(left_host) and left_host == right_host


def _node_id_token(name: str) -> str | None:
    """Return a trailing five-digit node id such as ``00001``, if present."""
    matches = _NODE_ID_RE.findall(name)
    return matches[-1] if matches else None


def default_nodes() -> tuple[NodeSpec, NodeSpec]:
    """Return the built-in primary (write+read) and replica (read) nodes."""
    return (
        NodeSpec(
            name=DEFAULT_NODE_00001_NAME,
            url=DEFAULT_NODE_00001_URL,
            roles=frozenset({ROLE_WRITE, ROLE_READ}),
        ),
        NodeSpec(
            name=DEFAULT_NODE_00002_NAME,
            url=DEFAULT_NODE_00002_URL,
            roles=frozenset({ROLE_READ}),
        ),
    )


def _default_roles_for_name(name: str) -> frozenset[str]:
    """Guess a role from a node name when the config omits one."""
    token = _node_id_token(name)
    if token == "00001":
        return frozenset({ROLE_WRITE, ROLE_READ})
    if token == "00002":
        return frozenset({ROLE_READ})
    return frozenset({ROLE_READ})


def _spec_from_mapping(item: Mapping[str, Any]) -> NodeSpec | None:
    """Build a ``NodeSpec`` from a JSON object; ignore unknown keys."""
    name = str(item.get("name") or item.get("id") or "").strip()
    url_raw = str(item.get("url") or item.get("address") or "").strip()
    url = _safe_http_url(url_raw)
    if not name and url:
        name = _host_from_url(url) or url
    if not name or not url:
        return None
    role_value: object
    if "role" in item:
        role_value = item["role"]
    elif "roles" in item:
        role_value = item["roles"]
    else:
        role_value = None
    roles = parse_roles(role_value, default=_default_roles_for_name(name))
    return NodeSpec(name=name, url=url, roles=roles)


def _spec_from_pipe(entry: str) -> NodeSpec | None:
    """Parse ``name|url|role`` (role optional)."""
    parts = [part.strip() for part in entry.split("|")]
    if len(parts) < 2:
        return None
    name, url_raw = parts[0], parts[1]
    url = _safe_http_url(url_raw)
    if not name or not url:
        return None
    role_raw = parts[2] if len(parts) >= 3 else None
    roles = parse_roles(role_raw, default=_default_roles_for_name(name))
    return NodeSpec(name=name, url=url, roles=roles)


def _specs_from_simple_list(raw: str) -> list[NodeSpec]:
    """Parse newline, semicolon, or comma separated ``name|url|role`` entries."""
    chunks = re.split(r"[\n;]+", raw)
    if len(chunks) == 1 and "|" in raw and raw.count("|") >= 2:
        # Single-line comma-separated entries: a|url|role,b|url|role
        chunks = re.split(r"\s*,\s*", raw)
    specs: list[NodeSpec] = []
    for chunk in chunks:
        text = chunk.strip().strip(",")
        if not text:
            continue
        spec = _spec_from_pipe(text)
        if spec is not None:
            specs.append(spec)
    return specs


def parse_rag_nodes(raw: str | None = None) -> tuple[NodeSpec, ...]:
    """Parse ``RAG_NODES`` JSON or a simple list; fall back to built-in defaults."""
    text = os.getenv("RAG_NODES", "") if raw is None else raw
    text = text.strip()
    if not text:
        return default_nodes()
    try:
        loaded: object = json.loads(text)
    except json.JSONDecodeError:
        simple_specs = _specs_from_simple_list(text)
        if simple_specs:
            return tuple(simple_specs)
        logger.warning("Invalid RAG_NODES value; using built-in node defaults")
        return default_nodes()

    specs: list[NodeSpec] = []
    if isinstance(loaded, list):
        for item in loaded:
            if isinstance(item, Mapping):
                spec = _spec_from_mapping(item)
            elif isinstance(item, str):
                spec = _spec_from_pipe(item)
            else:
                spec = None
            if spec is not None:
                specs.append(spec)
    elif isinstance(loaded, Mapping):
        if "name" in loaded or "url" in loaded:
            spec = _spec_from_mapping(loaded)
            if spec is not None:
                specs.append(spec)
        else:
            for key, value in loaded.items():
                if isinstance(value, Mapping):
                    merged = {"name": str(key), **value}
                    spec = _spec_from_mapping(merged)
                elif isinstance(value, str):
                    spec = _spec_from_pipe(f"{key}|{value}")
                else:
                    spec = None
                if spec is not None:
                    specs.append(spec)
    if specs:
        return tuple(specs)
    logger.warning("RAG_NODES parsed to an empty catalog; using built-in defaults")
    return default_nodes()


def resolve_node_role_override() -> frozenset[str] | None:
    """Return ``RAG_NODE_ROLE`` as canonical roles, or ``None`` if unset."""
    raw = os.getenv("RAG_NODE_ROLE", "").strip()
    if not raw:
        return None
    parsed = parse_roles(raw, default=frozenset())
    return parsed or None


def resolve_node_name_override() -> str | None:
    """Return ``RAG_NODE_NAME`` when set."""
    value = os.getenv("RAG_NODE_NAME", "").strip()
    return value or None


def _origin_from_request(request: Request | None) -> str | None:
    """Return PUBLIC_BASE_URL, else the request origin, else Railway public domain."""
    configured = resolve_public_base_url()
    if configured:
        return configured
    if request is not None:
        host = request.headers.get("host", "").strip()
        if host:
            scheme = request.url.scheme or "https"
            forwarded = (
                request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
            )
            if forwarded:
                scheme = forwarded
            return f"{scheme}://{host}".rstrip("/")
        base = str(request.base_url).rstrip("/")
        if base:
            return base
    public_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip().rstrip("/")
    if public_domain:
        if "://" in public_domain:
            return public_domain
        return f"https://{public_domain}"
    return None


def _match_node(
    nodes: tuple[NodeSpec, ...],
    *,
    name: str | None,
    url: str | None,
    railway_service_name: str | None,
) -> NodeSpec | None:
    """Identify the catalog entry for this process, if any."""
    candidates = [value for value in (name, railway_service_name) if value]
    for node in nodes:
        for candidate in candidates:
            if candidate.lower() == node.name.lower():
                return node
            left_id = _node_id_token(candidate)
            right_id = _node_id_token(node.name)
            if left_id and left_id == right_id:
                return node
    if url:
        for node in nodes:
            if _urls_match(url, node.url):
                return node
    return None


def current_roles(
    nodes: tuple[NodeSpec, ...] | None = None,
    *,
    request: Request | None = None,
) -> frozenset[str]:
    """Return the effective role set for this process."""
    override = resolve_node_role_override()
    if override is not None:
        return override
    catalog = nodes if nodes is not None else parse_rag_nodes()
    railway = _railway_public_env()
    matched = _match_node(
        catalog,
        name=resolve_node_name_override(),
        url=_origin_from_request(request),
        railway_service_name=railway.get("RAILWAY_SERVICE_NAME"),
    )
    if matched is not None:
        return matched.roles
    return frozenset({ROLE_WRITE, ROLE_READ})


def node_info_payload(request: Request | None = None) -> dict[str, Any]:
    """Build the public JSON document for ``GET /node-info`` and the login page.

    Only node names, public URLs, Czech roles, and allow-listed Railway
    identity fields are included. Other environment values are never copied.
    """
    nodes = parse_rag_nodes()
    railway = _railway_public_env()
    origin = _origin_from_request(request)
    override_name = resolve_node_name_override()
    override_roles = resolve_node_role_override()
    matched = _match_node(
        nodes,
        name=override_name,
        url=origin,
        railway_service_name=railway.get("RAILWAY_SERVICE_NAME"),
    )
    this_roles = (
        override_roles
        if override_roles is not None
        else (
            matched.roles if matched is not None else frozenset({ROLE_WRITE, ROLE_READ})
        )
    )
    this_name = (
        override_name
        or (matched.name if matched is not None else None)
        or railway.get("RAILWAY_SERVICE_NAME")
        or (_host_from_url(origin) if origin else None)
        or "local"
    )
    this_url = origin or (matched.url if matched is not None else "")
    node_rows: list[dict[str, Any]] = []
    for node in nodes:
        is_this = matched is not None and node.name == matched.name
        roles = this_roles if is_this else node.roles
        node_rows.append(
            {
                "name": node.name,
                "url": node.url,
                "roles": [role for role in ROLE_ORDER if role in roles],
                "role_labels": format_role_labels(roles),
                "role": format_role_text(roles),
                "is_this": is_this,
            }
        )
    this_body: dict[str, Any] = {
        "name": this_name,
        "url": this_url,
        "roles": [role for role in ROLE_ORDER if role in this_roles],
        "role_labels": format_role_labels(this_roles),
        "role": format_role_text(this_roles),
    }
    service_name = railway.get("RAILWAY_SERVICE_NAME")
    region = railway.get("RAILWAY_REPLICA_REGION")
    environment = railway.get("RAILWAY_ENVIRONMENT_NAME")
    if service_name:
        this_body["railway_service_name"] = service_name
    if region:
        this_body["railway_replica_region"] = region
    if environment:
        this_body["railway_environment_name"] = environment
    return {
        "this": this_body,
        "nodes": node_rows,
        "enforce_node_role": enforce_node_role(),
        "writes_allowed": writes_allowed_for(this_roles),
    }


def write_nodes_from_payload(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return catalog nodes whose role includes write."""
    found: list[dict[str, str]] = []
    for node in payload.get("nodes") or []:
        if not isinstance(node, Mapping):
            continue
        roles = node.get("roles") or []
        if ROLE_WRITE in roles:
            name = str(node.get("name") or "")
            url = str(node.get("url") or "")
            if name and url:
                found.append({"name": name, "url": url})
    return found


def writes_blocked_detail(request: Request | None = None) -> dict[str, Any] | None:
    """Return a 403 body when enforcement is on and this node cannot write."""
    if not enforce_node_role():
        return None
    payload = node_info_payload(request)
    if payload["writes_allowed"]:
        return None
    this = payload["this"]
    role_text = str(this.get("role") or "STOP")
    return {
        "detail": f"Writes are disabled on this node ({role_text}).",
        "role": role_text,
        "role_labels": this.get("role_labels") or [],
        "write_nodes": write_nodes_from_payload(payload),
    }


def mcp_write_blocked_message(tool: str, request: Request | None = None) -> str | None:
    """Return an MCP error string when ``tool`` is a blocked write, else ``None``."""
    if tool not in MCP_WRITE_TOOLS:
        return None
    blocked = writes_blocked_detail(request)
    if blocked is None:
        return None
    return f"Error {tool}: {blocked['detail']}"


def is_http_write_request(method: str, path: str) -> bool:
    """Return whether an HTTP method+path mutates RAG data."""
    upper = method.upper()
    normalized = path.rstrip("/") or "/"
    if upper in {"POST", "PUT", "DELETE"} and normalized.startswith("/upload"):
        return True
    if upper == "POST" and normalized in {"/ingest-json"}:
        return True
    if upper == "DELETE" and normalized.startswith("/documents/"):
        return True
    if upper == "PATCH" and normalized.startswith("/documents/"):
        return True
    if upper == "POST" and normalized.startswith("/documents/"):
        return True
    return False


class NodeWriteGuard:
    """ASGI wrapper that optionally rejects writes on ČTENÍ / STOP nodes.

    Inactive unless ``RAG_ENFORCE_NODE_ROLE`` is enabled, so production
    behaviour stays unchanged until the flag is turned on.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app`` with optional write-role enforcement."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject HTTP write routes when this node is not allowed to write."""
        if scope["type"] != "http" or not enforce_node_role():
            await self.app(scope, receive, send)
            return
        method = str(scope.get("method") or "GET")
        path = str(scope.get("path") or "/")
        if not is_http_write_request(method, path):
            await self.app(scope, receive, send)
            return
        blocked = writes_blocked_detail()
        if blocked is None:
            await self.app(scope, receive, send)
            return
        response = JSONResponse(blocked, status_code=403)
        await response(scope, receive, send)


def _dt_dd(term: str, value_html: str) -> str:
    """Return a definition-list pair with a Czech term."""
    return (
        f"<div class='node-info-row'><dt>{escape(term)}</dt>"
        f"<dd>{value_html}</dd></div>"
    )


def _role_badges(labels: list[str]) -> str:
    """Return HTML badges for Czech role labels."""
    parts: list[str] = []
    css_by_label = {
        ROLE_LABELS[ROLE_WRITE]: "role-write",
        ROLE_LABELS[ROLE_READ]: "role-read",
        ROLE_LABELS[ROLE_STOP]: "role-stop",
    }
    for label in labels:
        css = css_by_label.get(label, "role-read")
        parts.append(f"<span class='role-badge {css}'>{escape(label)}</span>")
    return " ".join(parts) if parts else escape("—")


def _anchor(url: str, label: str | None = None) -> str:
    """Return a safe http(s) link, or escaped plain text if the URL is not http(s)."""
    safe = _safe_http_url(url)
    text = escape(label or url)
    if not safe:
        return text
    return f"<a href='{escape(safe, quote=True)}'>{text}</a>"


def render_login_node_info_html(request: Request) -> str:
    """Return the Czech information block for the Google sign-in page."""
    payload = node_info_payload(request)
    this = payload["this"]
    rows = [
        _dt_dd("Uzel", escape(str(this.get("name") or "—"))),
        _dt_dd("Veřejná adresa", _anchor(str(this.get("url") or ""))),
    ]
    service = this.get("railway_service_name")
    region = this.get("railway_replica_region")
    environment = this.get("railway_environment_name")
    if service:
        rows.append(_dt_dd("Služba Railway", escape(str(service))))
    if region:
        rows.append(_dt_dd("Region", escape(str(region))))
    if environment:
        rows.append(_dt_dd("Prostředí", escape(str(environment))))
    labels = [str(item) for item in (this.get("role_labels") or [])]
    rows.append(_dt_dd("Role", _role_badges(labels)))

    node_items: list[str] = []
    for node in payload["nodes"]:
        marker = (
            " <span class='this-node'>(tento uzel)</span>"
            if node.get("is_this")
            else ""
        )
        role_html = _role_badges(
            [str(item) for item in (node.get("role_labels") or [])]
        )
        node_items.append(
            "<li>"
            f"{_anchor(str(node.get('url') or ''), str(node.get('name') or node.get('url') or ''))}"
            f"{marker}"
            f"<span class='node-role'> — {role_html}</span>"
            "</li>"
        )
    nodes_html = "".join(node_items) or "<li>Žádné uzly nejsou nakonfigurovány.</li>"
    return f"""
    <section class="node-info" lang="cs" aria-labelledby="node-info-heading">
      <h2 id="node-info-heading">Informace o této instanci</h2>
      <dl class="node-info-dl">
        {''.join(rows)}
      </dl>
      <h3>Všechny uzly</h3>
      <ul class="node-list">
        {nodes_html}
      </ul>
    </section>
    """
