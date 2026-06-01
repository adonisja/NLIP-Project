"""GitHub OAuth for the Angel Filter.

Replaces the previous bcrypt + JSON-file login. The flow is:
    1. User clicks "Sign in with GitHub" -> GET /auth/github/login
    2. We generate a random `state`, stash it in the session, and
       redirect to GitHub's authorize URL.
    3. GitHub redirects back to /auth/github/callback with `code` and
       the same `state`.
    4. We verify `state`, exchange `code` for an access token at
       GitHub's token endpoint, then call api.github.com/user with that
       token to learn the GitHub username.
    5. If the username is in ANGEL_ALLOWED_USERS we set
       request.session["user"]; otherwise the login is rejected.

Allowlist:
    Set ANGEL_ALLOWED_USERS=alice,bob,charlie (comma-separated GitHub
    usernames). Adding or removing a teammate is a config change on the
    cloud platform, not a code change. Case-insensitive — GitHub
    usernames are case-insensitive too.

Env vars required:
    GITHUB_CLIENT_ID            (public, from GitHub OAuth App settings)
    GITHUB_CLIENT_SECRET        (secret, from GitHub OAuth App settings)
    GITHUB_OAUTH_CALLBACK_URL   (must exactly match the URL you set as
                                 "Authorization callback URL" in GitHub)
    ANGEL_ALLOWED_USERS         (comma-separated GitHub usernames)

GitHub OAuth App registration:
    https://github.com/settings/developers -> OAuth Apps -> New OAuth App
    Homepage URL:                  http://localhost:8000  (or prod URL)
    Authorization callback URL:    http://localhost:8000/auth/github/callback
    Register a separate app for production and dev so each can have its
    own callback URL.

This module exposes the helpers that server.py composes into routes:
    * current_user(request)              -> Optional[str]
    * require_session                    -> FastAPI dependency
    * new_oauth_state()                  -> random URL-safe state token
    * build_authorize_url(state)         -> URL to redirect to GitHub
    * exchange_code_for_username(code)   -> verified GitHub username or
                                            None (allowlist check inside)
"""

from __future__ import annotations

import logging
import os
import secrets
import urllib.parse
from typing import Optional

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"


def _allowed_users() -> set[str]:
    """Re-read the allowlist on every check so env-var updates take effect
    without rebuilding any module-level state. Cheap — it's a string split."""
    raw = os.getenv("ANGEL_ALLOWED_USERS", "")
    return {u.strip().lower() for u in raw.split(",") if u.strip()}


def current_user(request: Request) -> Optional[str]:
    """Return the logged-in GitHub username from the session, or None.

    Re-validates the session against the current ANGEL_ALLOWED_USERS on
    every call. This means removing someone from the allowlist (and
    restarting the server, since env vars are read at process start)
    kicks them out on their very next request — no waiting for the
    session cookie to expire.
    """
    if not hasattr(request, "session"):
        return None
    user = request.session.get("user")
    if not user:
        return None
    if user.lower() not in _allowed_users():
        # Was on the list when they logged in, no longer is. Drop the
        # session so the next request hits /login.
        logger.info(
            "Session invalidated for %r (no longer in ANGEL_ALLOWED_USERS)",
            user,
        )
        request.session.clear()
        return None
    return user


def require_session(request: Request) -> str:
    """FastAPI dependency: ensure the request has a valid session.

    Returns the username so route handlers can use it. Raises 401 if no
    session is present; the SPA can handle that by redirecting to /login.
    """
    user = current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Cookie"},
        )
    return user


def new_oauth_state() -> str:
    """Random CSRF token for the OAuth round-trip. Stored in the session
    before redirect and compared on callback."""
    return secrets.token_urlsafe(32)


def build_authorize_url(state: str) -> str:
    """Construct the GitHub authorize URL we send the browser to."""
    client_id = os.getenv("GITHUB_CLIENT_ID")
    callback = os.getenv("GITHUB_OAUTH_CALLBACK_URL")
    if not client_id or not callback:
        raise RuntimeError(
            "GitHub OAuth is misconfigured: set GITHUB_CLIENT_ID and "
            "GITHUB_OAUTH_CALLBACK_URL env vars."
        )
    params = {
        "client_id": client_id,
        "redirect_uri": callback,
        "scope": "read:user",  # only the public profile — that's all we need
        "state": state,
        "allow_signup": "false",  # never let unknown users create accounts mid-flow
    }
    return f"{GITHUB_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


async def exchange_code_for_username(code: str) -> Optional[str]:
    """Run code -> access_token -> /user and return the GitHub username.

    Returns the username string only if (a) GitHub accepts the code,
    (b) the /user call succeeds, AND (c) the username is in the
    allowlist. Returns None on any failure or rejection — the route
    handler turns that into a 403.
    """
    import httpx  # lazy: keeps the module importable without httpx

    client_id = os.getenv("GITHUB_CLIENT_ID")
    client_secret = os.getenv("GITHUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "GitHub OAuth secrets missing: set GITHUB_CLIENT_ID and "
            "GITHUB_CLIENT_SECRET."
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            token_resp = await client.post(
                GITHUB_TOKEN_URL,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                },
                headers={"Accept": "application/json"},
            )
            token_resp.raise_for_status()
            token_data = token_resp.json()
            access_token = token_data.get("access_token")
            if not access_token:
                logger.warning(
                    "GitHub token exchange returned no access_token: %s",
                    token_data,
                )
                return None

            user_resp = await client.get(
                GITHUB_USER_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github+json",
                },
            )
            user_resp.raise_for_status()
            login = user_resp.json().get("login")
    except httpx.HTTPError as exc:
        logger.warning("GitHub OAuth network error: %s", exc)
        return None

    if not login:
        return None

    allowlist = _allowed_users()
    if not allowlist:
        # Fail closed: if the allowlist is empty, no one gets in. Better
        # than the alternative of letting anyone with GitHub in.
        logger.error(
            "ANGEL_ALLOWED_USERS is empty; rejecting %r. Configure the "
            "allowlist before deploying.",
            login,
        )
        return None
    if login.lower() not in allowlist:
        logger.warning(
            "GitHub login rejected for username %r (not on allowlist)",
            login,
        )
        return None
    return login
