"""Cost protection: per-user rate limit + global daily budget cap.

These two layers exist because authentication alone does not stop a
*logged-in* user from running /query in a tight loop and draining
upstream API credits. On a public cloud deployment, anyone with a valid
session — a legitimate teammate, a judge whose laptop is unattended,
or someone who guessed a weak password — could do this in seconds.

Layer 1 — per-user sliding window:
    Each username gets ANGEL_USER_RATE_LIMIT requests per
    ANGEL_USER_RATE_WINDOW seconds. Excess requests get HTTP 429 with
    Retry-After. Defaults: 30 requests per 60 seconds.

Layer 2 — daily global budget:
    A single process-wide counter tracks total /query calls served
    today (UTC). Once ANGEL_DAILY_QUERY_LIMIT is hit, every subsequent
    /query returns 429 until midnight UTC, regardless of who is asking.
    Default: 500 queries/day. This is the credit-protection backstop.

Both counters live in process memory, which is fine for a single
instance. If you scale horizontally (multiple workers / containers),
move them to Redis or each instance enforces its own quota independently.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status

from angel_filter.auth import require_session

logger = logging.getLogger(__name__)

# --- Configurable thresholds (env vars) -------------------------------------
USER_RATE_LIMIT = int(os.getenv("ANGEL_USER_RATE_LIMIT", "30"))
USER_RATE_WINDOW = int(os.getenv("ANGEL_USER_RATE_WINDOW", "60"))
DAILY_QUERY_LIMIT = int(os.getenv("ANGEL_DAILY_QUERY_LIMIT", "500"))

# --- Per-user sliding window ------------------------------------------------
_user_hits: dict[str, deque[float]] = {}
_user_lock = asyncio.Lock()


async def _enforce_user_rate(username: str) -> None:
    now = time.monotonic()
    async with _user_lock:
        hits = _user_hits.setdefault(username, deque())
        # Drop timestamps that fell out of the window.
        while hits and now - hits[0] >= USER_RATE_WINDOW:
            hits.popleft()
        if len(hits) >= USER_RATE_LIMIT:
            retry_after = max(1, int(USER_RATE_WINDOW - (now - hits[0])) + 1)
            logger.warning(
                "Rate limit hit for user %r (%d/%ds)",
                username, USER_RATE_LIMIT, USER_RATE_WINDOW,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded ({USER_RATE_LIMIT} requests per "
                    f"{USER_RATE_WINDOW}s). Try again in {retry_after}s."
                ),
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)


# --- Global daily budget ----------------------------------------------------
_daily_count = 0
_daily_date: str | None = None
_daily_lock = asyncio.Lock()


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


async def _enforce_daily_budget() -> None:
    global _daily_count, _daily_date
    async with _daily_lock:
        today = _today_utc()
        if _daily_date != today:
            _daily_date = today
            _daily_count = 0
            logger.info("Daily budget counter reset for %s", today)
        if _daily_count >= DAILY_QUERY_LIMIT:
            logger.warning(
                "Daily query budget exhausted: %d/%d",
                _daily_count, DAILY_QUERY_LIMIT,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Daily query budget exhausted ({DAILY_QUERY_LIMIT}). "
                    f"Resets at 00:00 UTC."
                ),
                headers={"Retry-After": "3600"},
            )
        _daily_count += 1


def daily_budget_status() -> dict[str, int | str | None]:
    """Read-only view of the global counter — useful for /health."""
    return {
        "date": _daily_date,
        "count": _daily_count,
        "limit": DAILY_QUERY_LIMIT,
    }


# --- FastAPI dependency -----------------------------------------------------


async def enforce_query_limits(user: str = Depends(require_session)) -> str:
    """Auth + rate limit + daily cap in a single dependency.

    Order matters: the user rate check runs first. If it fails, we do
    NOT increment the daily counter — a user spamming the rate limit
    should not also burn the global budget.
    """
    await _enforce_user_rate(user)
    await _enforce_daily_budget()
    return user
