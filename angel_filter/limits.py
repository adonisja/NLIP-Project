"""Cost protection: per-client rate limit + global daily budget cap.

Every query fans out to paid providers, so anything that can reach /query or
/nlip/ can spend real money. There is no login (newcomers are designing their
own — see docs/KNOWN_ISSUES.md), so these limits are the only protection.

Layer 1 — per-client sliding window:
    Each client gets ANGEL_USER_RATE_LIMIT requests per
    ANGEL_USER_RATE_WINDOW seconds. Excess requests get HTTP 429 with
    Retry-After. Defaults: 30 requests per 60 seconds. A client is
    identified by IP address (see client_key), which is a soft identity:
    people behind one NAT share a bucket, and an attacker with many IPs
    gets many buckets. Swap in a user ID once there is real auth.

Layer 2 — daily global budget:
    A single process-wide counter tracks total queries served today (UTC).
    Once ANGEL_DAILY_QUERY_LIMIT is hit, every subsequent query returns 429
    until midnight UTC, regardless of who is asking. Default: 500/day. This
    is the credit-protection backstop, and the one that holds even when
    layer 1 is evaded.

Both counters live in the shared store (state.py). With ANGEL_REDIS_URL set
they are shared by every worker and instance and survive restarts; without
it they are per-process and reset on restart.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from angel_filter.state import PREFIX, get_store

logger = logging.getLogger(__name__)

# --- Configurable thresholds (env vars) -------------------------------------
USER_RATE_LIMIT = int(os.getenv("ANGEL_USER_RATE_LIMIT", "30"))
USER_RATE_WINDOW = int(os.getenv("ANGEL_USER_RATE_WINDOW", "60"))
DAILY_QUERY_LIMIT = int(os.getenv("ANGEL_DAILY_QUERY_LIMIT", "500"))


# --- Client identity ----------------------------------------------------------

def _trusted_proxy_hops() -> int:
    """How many reverse proxies sit in front of the app (ANGEL_TRUSTED_PROXY_HOPS).

    Read per call, like the other env-driven switches, so tests and config
    changes take effect without re-importing.
    """
    try:
        return max(0, int(os.getenv("ANGEL_TRUSTED_PROXY_HOPS", "0")))
    except ValueError:
        return 0


def client_key(request: Request) -> str:
    """Identify the caller for rate limiting: their IP address.

    Behind a reverse proxy (Render, nginx) every request arrives from the
    proxy's address, so the real client IP has to come from X-Forwarded-For.
    That header is client-controlled up to the point a trusted proxy appends
    to it, so only the entry added by our own proxies can be believed: with N
    trusted hops, that is the Nth entry from the right. With 0 (the default,
    and correct for local runs) the header is ignored entirely — otherwise
    anyone could send a fresh fake IP per request and never be limited.
    """
    hops = _trusted_proxy_hops()
    if hops:
        forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(forwarded) >= hops:
            return forwarded[-hops]
    return request.client.host if request.client else "unknown"


# --- Per-client sliding window ------------------------------------------------

async def _enforce_client_rate(key: str) -> None:
    retry_after = await get_store().window_hit(
        PREFIX + "rate:" + key, USER_RATE_WINDOW, USER_RATE_LIMIT,
    )
    if retry_after is None:
        return
    retry_after = int(retry_after) + 1
    logger.warning(
        "Rate limit hit for client %r (%d/%ds)",
        key, USER_RATE_LIMIT, USER_RATE_WINDOW,
    )
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            f"Rate limit exceeded ({USER_RATE_LIMIT} requests per "
            f"{USER_RATE_WINDOW}s). Try again in {retry_after}s."
        ),
        headers={"Retry-After": str(retry_after)},
    )


# --- Global daily budget ----------------------------------------------------

# A day's counter outlives the day by a margin, then expires on its own.
_DAILY_KEY_TTL = 2 * 24 * 60 * 60


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _daily_key(day: str) -> str:
    return PREFIX + "daily:" + day


async def _enforce_daily_budget() -> None:
    store = get_store()
    key = _daily_key(_today_utc())
    # Increment first and undo on rejection: with several workers, a separate
    # read-then-write would let two of them both take the last slot.
    count = await store.increment(key, 1, _DAILY_KEY_TTL)
    if count > DAILY_QUERY_LIMIT:
        await store.increment(key, -1, _DAILY_KEY_TTL)
        logger.warning("Daily query budget exhausted: %d/%d", count - 1, DAILY_QUERY_LIMIT)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily query budget exhausted ({DAILY_QUERY_LIMIT}). "
                f"Resets at 00:00 UTC."
            ),
            headers={"Retry-After": "3600"},
        )


async def daily_budget_status() -> dict[str, int | str]:
    """Read-only view of today's counter."""
    day = _today_utc()
    return {
        "date": day,
        "count": int(await get_store().get(_daily_key(day)) or 0),
        "limit": DAILY_QUERY_LIMIT,
    }


# --- Enforcement ----------------------------------------------------------------

async def check_query_limits(key: str) -> None:
    """Rate limit + daily cap. Raises HTTPException(429) when either is hit.

    Order matters: the client rate check runs first. If it fails, we do
    NOT increment the daily counter — a client spamming the rate limit
    should not also burn the global budget.
    """
    await _enforce_client_rate(key)
    await _enforce_daily_budget()


async def enforce_query_limits(request: Request) -> str:
    """FastAPI dependency for POST /query. Returns the client key."""
    key = client_key(request)
    await check_query_limits(key)
    return key


def nlip_limit_middleware(nlip_path: str):
    """Build an HTTP middleware that applies the same limits to the NLIP route.

    nlip_server registers POST /nlip/ itself, so we cannot attach a FastAPI
    dependency to it without forking the library. A middleware sees the
    request first instead. Only the exact registered path is matched: a POST
    to "/nlip" is redirected to "/nlip/" and must not be counted twice.
    """
    async def middleware(request: Request, call_next):
        if request.method == "POST" and request.url.path == nlip_path:
            try:
                await check_query_limits(client_key(request))
            except HTTPException as exc:
                return JSONResponse(
                    {"detail": exc.detail},
                    status_code=exc.status_code,
                    headers=exc.headers,
                )
        return await call_next(request)

    return middleware
