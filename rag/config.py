"""Runtime configuration for the RAG service."""

from __future__ import annotations

import os
import secrets

DEFAULT_RAG_DATA_ROOT = "/data/rag"
DEFAULT_ACCNT_DATA_ROOT = "/data/accnt"
COLLECTION_NAME = "client_domain_knowledge"
LOCALITY_LINKS_COLLECTION = "locality_links"
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8080"))

DEFAULT_MCP_ALLOWED_HOSTS = (
    "rag-timer-00001-production.up.railway.app",
    "localhost",
    "localhost:*",
    "127.0.0.1",
    "127.0.0.1:*",
)


def mcp_allowed_hosts() -> list[str]:
    """Hosts accepted by MCP DNS-rebinding protection (``MCP_ALLOWED_HOSTS``, comma-separated)."""
    raw = os.getenv("MCP_ALLOWED_HOSTS", "")
    hosts = [h.strip() for h in raw.split(",") if h.strip()]
    return hosts or list(DEFAULT_MCP_ALLOWED_HOSTS)


def mcp_dns_rebinding_protection() -> bool:
    """Set ``MCP_DISABLE_DNS_REBINDING_PROTECTION=1`` to accept any Host header."""
    return os.getenv("MCP_DISABLE_DNS_REBINDING_PROTECTION", "").lower() not in (
        "1",
        "true",
        "yes",
    )


DEFAULT_MAX_REMOTE_JSON_BYTES = 128 * 1024 * 1024
DEFAULT_REMOTE_FETCH_TIMEOUT_SECONDS = 120
DEFAULT_UPLOAD_CHUNK_SIZE = 52_428_800  # 50 MiB
DEFAULT_UPLOAD_MAX_CHUNK_SIZE = 52_428_800
DEFAULT_UPLOAD_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_UPLOAD_TTL_SECONDS = 86_400
MAX_REMOTE_REDIRECTS = 5
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_SCOPES = "openid email profile"


def resolve_rag_data_root(override: str | None = None) -> str:
    """Resolve the persistent RAG data root directory."""
    if override:
        return override
    return os.getenv("RAG_DATA_ROOT", DEFAULT_RAG_DATA_ROOT)


def resolve_accnt_data_root(override: str | None = None) -> str:
    """Resolve the persistent per-day accounting root directory."""
    if override:
        return override
    return os.getenv("ACCNT_DATA_ROOT", DEFAULT_ACCNT_DATA_ROOT)


def should_skip_default_app() -> bool:
    """Return whether module-level app construction should be skipped."""
    return os.getenv("RAG_SKIP_DEFAULT_APP") == "1"


def resolve_api_token() -> str | None:
    """Return the inbound HTTP bearer token, if configured."""
    token = os.getenv("RAG_API_TOKEN")
    if token is None:
        return None
    stripped = token.strip()
    return stripped or None


def resolve_max_remote_json_bytes() -> int:
    """Return the maximum allowed remote JSON download size in bytes."""
    raw = os.getenv("RAG_MAX_REMOTE_JSON_BYTES")
    if not raw:
        return DEFAULT_MAX_REMOTE_JSON_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_REMOTE_JSON_BYTES
    return value if value > 0 else DEFAULT_MAX_REMOTE_JSON_BYTES


def resolve_remote_fetch_timeout() -> int:
    """Return the remote JSON fetch timeout in seconds."""
    raw = os.getenv("RAG_REMOTE_FETCH_TIMEOUT")
    if not raw:
        return DEFAULT_REMOTE_FETCH_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_REMOTE_FETCH_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_REMOTE_FETCH_TIMEOUT_SECONDS


def resolve_session_secret() -> str:
    """Return the cookie session secret, generating an ephemeral one if unset."""
    configured = os.getenv("SESSION_SECRET", "").strip()
    if configured:
        return configured
    # Ephemeral secret: sessions reset on process restart.
    return secrets.token_urlsafe(32)


def resolve_public_base_url() -> str | None:
    """Return the public base URL used for OAuth redirect URIs, if set."""
    value = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    return value or None


def resolve_google_client_id() -> str | None:
    """Return the Google OAuth client id."""
    value = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    return value or None


def resolve_google_client_secret() -> str | None:
    """Return the Google OAuth client secret."""
    value = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    return value or None


def resolve_google_allowed_emails() -> set[str]:
    """Return the optional allowlist of Google account emails."""
    raw = os.getenv("GOOGLE_ALLOWED_EMAILS", "")
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def resolve_google_allowed_domains() -> set[str]:
    """Return the optional allowlist of Google account email domains."""
    raw = os.getenv("GOOGLE_ALLOWED_DOMAINS", "")
    return {part.strip().lower().lstrip("@") for part in raw.split(",") if part.strip()}


def google_oauth_configured() -> bool:
    """Return whether Google OAuth client credentials are present."""
    return bool(resolve_google_client_id() and resolve_google_client_secret())


def resolve_ocr_enabled() -> bool:
    """Return whether OCR runs during PDF ingest (default true)."""
    raw = os.getenv("OCR_ENABLED", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def resolve_ocr_langs() -> str:
    """Return Tesseract language string (default ``ces+eng``)."""
    raw = os.getenv("OCR_LANGS", "ces+eng").strip()
    return raw or "ces+eng"


def resolve_ocr_min_confidence() -> int:
    """Minimum per-word OCR confidence to keep (default 60)."""
    raw = os.getenv("OCR_MIN_CONFIDENCE", "60").strip()
    try:
        value = int(raw)
    except ValueError:
        return 60
    return max(0, min(100, value))


def resolve_ocr_min_chars() -> int:
    """Minimum cleaned OCR length required to index an image point (default 20)."""
    raw = os.getenv("OCR_MIN_CHARS", "20").strip()
    try:
        value = int(raw)
    except ValueError:
        return 20
    return max(0, value)


def _positive_int_env(name: str, default: int) -> int:
    """Parse a positive integer environment variable."""
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def resolve_upload_chunk_size_default() -> int:
    """Default chunk size for ``POST /upload/init`` (``RAG_UPLOAD_CHUNK_SIZE``)."""
    return _positive_int_env("RAG_UPLOAD_CHUNK_SIZE", DEFAULT_UPLOAD_CHUNK_SIZE)


def resolve_upload_max_chunk_size() -> int:
    """Maximum allowed per-chunk body size (``RAG_UPLOAD_MAX_CHUNK_SIZE``)."""
    return _positive_int_env("RAG_UPLOAD_MAX_CHUNK_SIZE", DEFAULT_UPLOAD_MAX_CHUNK_SIZE)


def resolve_upload_max_bytes() -> int:
    """Maximum whole-file size for chunked uploads (``RAG_UPLOAD_MAX_BYTES``)."""
    return _positive_int_env("RAG_UPLOAD_MAX_BYTES", DEFAULT_UPLOAD_MAX_BYTES)


def resolve_upload_ttl_seconds() -> int:
    """Staging TTL for chunked uploads in seconds (``RAG_UPLOAD_TTL_SECONDS``)."""
    return _positive_int_env("RAG_UPLOAD_TTL_SECONDS", DEFAULT_UPLOAD_TTL_SECONDS)


def resolve_mcp_token() -> str | None:
    """Bearer token for ``/mcp`` and embedding REST routes.

    ``MCP_API_TOKEN`` wins; falls back to ``RAG_API_TOKEN``. ``None`` means fail closed.
    """
    for name in ("MCP_API_TOKEN", "RAG_API_TOKEN"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None
