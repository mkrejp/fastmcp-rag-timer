"""Google OAuth login for the RAG web console."""

from __future__ import annotations

import secrets
from typing import Any
from urllib.parse import urlencode

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from rag.config import (
    GOOGLE_AUTH_URL,
    GOOGLE_SCOPES,
    GOOGLE_TOKEN_URL,
    GOOGLE_USERINFO_URL,
    google_oauth_configured,
    resolve_google_allowed_domains,
    resolve_google_allowed_emails,
    resolve_google_client_id,
    resolve_google_client_secret,
    resolve_public_base_url,
)
from rag.logging_setup import get_logger

router = APIRouter(tags=["auth"])
logger = get_logger("rag.auth.google")

SESSION_USER_KEY = "google_user"
SESSION_STATE_KEY = "oauth_state"


def _base_url(request: Request) -> str:
    """Return the public origin for OAuth redirects."""
    configured = resolve_public_base_url()
    if configured:
        return configured
    return str(request.base_url).rstrip("/")


def _redirect_uri(request: Request) -> str:
    """Return the Google OAuth callback URI."""
    return f"{_base_url(request)}/auth/callback"


def current_user(request: Request) -> dict[str, Any] | None:
    """Return the signed-in Google user from the session, if any."""
    user = request.session.get(SESSION_USER_KEY)
    return user if isinstance(user, dict) else None


def require_google_user(request: Request) -> dict[str, Any]:
    """Require a signed-in Google user for UI/admin routes."""
    if not google_oauth_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
        )
    user = current_user(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in with Google required",
        )
    return user


def _email_allowed(email: str) -> bool:
    """Return whether ``email`` passes optional allowlists."""
    normalized = email.strip().lower()
    allowed_emails = resolve_google_allowed_emails()
    allowed_domains = resolve_google_allowed_domains()
    if not allowed_emails and not allowed_domains:
        return True
    if normalized in allowed_emails:
        return True
    domain = normalized.split("@")[-1] if "@" in normalized else ""
    return domain in allowed_domains


@router.get("/auth/login", response_class=HTMLResponse, response_model=None)
def login_page(request: Request) -> HTMLResponse | RedirectResponse:
    """Render the Google sign-in page for the web console."""
    user = current_user(request)
    if user is not None:
        return RedirectResponse(url="/ui/", status_code=302)
    configured = google_oauth_configured()
    error = request.query_params.get("error", "")
    message = {
        "access_denied": "Google sign-in was cancelled.",
        "not_configured": "Google OAuth is not configured on this service.",
        "not_allowed": "That Google account is not allowed to access this console.",
        "state": "OAuth state mismatch. Try signing in again.",
        "token": "Could not exchange the Google authorization code.",
        "userinfo": "Could not load your Google profile.",
    }.get(error, error)
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sign in · RAG console</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="/ui/styles.css" />
</head>
<body class="auth-body">
  <div class="atmosphere" aria-hidden="true"></div>
  <main class="auth-card">
    <p class="brand">fastmcp-rag-timer</p>
    <h1>RAG console</h1>
    <p class="lede">Sign in with Google to browse the data volume, indexed RAG content, MCP status, and logs.</p>
    {"<p class='error'>" + message + "</p>" if message else ""}
    {"<a class='google-btn' href='/auth/google'>Continue with Google</a>" if configured else "<p class='error'>Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to enable sign-in.</p>"}
  </main>
</body>
</html>"""
    return HTMLResponse(body)


@router.get("/auth/google")
def start_google_login(request: Request) -> RedirectResponse:
    """Begin the Google OAuth authorization code flow."""
    if not google_oauth_configured():
        return RedirectResponse(url="/auth/login?error=not_configured", status_code=302)
    state = secrets.token_urlsafe(24)
    request.session[SESSION_STATE_KEY] = state
    params = {
        "client_id": resolve_google_client_id(),
        "redirect_uri": _redirect_uri(request),
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "access_type": "online",
        "include_granted_scopes": "true",
        "prompt": "select_account",
        "state": state,
    }
    logger.info("Starting Google OAuth login redirect_uri=%s", _redirect_uri(request))
    return RedirectResponse(url=f"{GOOGLE_AUTH_URL}?{urlencode(params)}", status_code=302)


@router.get("/auth/callback")
def google_callback(request: Request) -> RedirectResponse:
    """Exchange the Google auth code and establish a session."""
    if not google_oauth_configured():
        return RedirectResponse(url="/auth/login?error=not_configured", status_code=302)

    error = request.query_params.get("error")
    if error:
        logger.warning("Google OAuth error: %s", error)
        return RedirectResponse(url="/auth/login?error=access_denied", status_code=302)

    state = request.query_params.get("state", "")
    expected = request.session.pop(SESSION_STATE_KEY, None)
    if not state or not expected or state != expected:
        logger.warning("Google OAuth state mismatch")
        return RedirectResponse(url="/auth/login?error=state", status_code=302)

    code = request.query_params.get("code", "")
    if not code:
        return RedirectResponse(url="/auth/login?error=token", status_code=302)

    try:
        token_response = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": resolve_google_client_id(),
                "client_secret": resolve_google_client_secret(),
                "redirect_uri": _redirect_uri(request),
                "grant_type": "authorization_code",
            },
            timeout=30,
        )
        token_response.raise_for_status()
        tokens = token_response.json()
        access_token = tokens.get("access_token")
        if not access_token:
            raise ValueError("missing access_token")
        profile_response = requests.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )
        profile_response.raise_for_status()
        profile = profile_response.json()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Google OAuth token/userinfo failed: %s", exc)
        return RedirectResponse(url="/auth/login?error=token", status_code=302)

    email = str(profile.get("email") or "").strip().lower()
    if not email or not profile.get("email_verified", True):
        logger.warning("Google profile missing verified email")
        return RedirectResponse(url="/auth/login?error=userinfo", status_code=302)
    if not _email_allowed(email):
        logger.warning("Google account not allowlisted: %s", email)
        return RedirectResponse(url="/auth/login?error=not_allowed", status_code=302)

    user = {
        "email": email,
        "name": profile.get("name") or email,
        "picture": profile.get("picture"),
        "sub": profile.get("sub"),
    }
    request.session[SESSION_USER_KEY] = user
    logger.info("Google sign-in succeeded for %s", email)
    return RedirectResponse(url="/ui/", status_code=302)


@router.get("/auth/logout")
def logout(request: Request) -> RedirectResponse:
    """Clear the signed-in Google session."""
    email = (current_user(request) or {}).get("email")
    request.session.clear()
    logger.info("Signed out %s", email or "anonymous")
    return RedirectResponse(url="/auth/login", status_code=302)


@router.get("/auth/me")
def auth_me(user: dict[str, Any] = Depends(require_google_user)) -> dict[str, Any]:
    """Return the current Google user for the web console."""
    return {"authenticated": True, "user": user, "oauth": "google"}


@router.get("/auth/status")
def auth_status(request: Request) -> dict[str, Any]:
    """Return whether Google OAuth is configured and whether a user is signed in."""
    user = current_user(request)
    return {
        "configured": google_oauth_configured(),
        "authenticated": user is not None,
        "user": user,
        "login_url": "/auth/login",
        "google_url": "/auth/google",
    }
