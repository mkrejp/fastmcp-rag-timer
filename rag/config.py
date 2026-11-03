"""Runtime configuration for the RAG service."""

from __future__ import annotations

import os
import secrets

DEFAULT_RAG_DATA_ROOT = "/data/rag"
COLLECTION_NAME = "client_domain_knowledge"
HOST = "0.0.0.0"
PORT = 8080
DEFAULT_MAX_REMOTE_JSON_BYTES = 128 * 1024 * 1024
DEFAULT_REMOTE_FETCH_TIMEOUT_SECONDS = 120
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
