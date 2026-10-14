"""Shared state: where the cache, conversations, limits and geocodes live.

Two backends behind one async interface:

  - MemoryStore (default): plain dicts in this process. Lost on restart and
    not shared between workers — fine for local development and tests.
  - RedisStore: used when ANGEL_REDIS_URL is set (e.g. a free Upstash
    database, `rediss://default:<password>@<host>:6379`). Survives restarts
    and is shared by every worker and instance, so the rate limits and the
    daily spending cap actually hold.

Callers never touch a backend directly; they call get_store() at the moment
they need it, so tests can swap the store with use_store().

If Redis becomes unreachable, RedisStore serves each operation from an
in-process MemoryStore instead and logs a warning. A Redis outage therefore
degrades the app to exactly its old in-memory behaviour; it never fails a
search, and the limits still apply per process.

Values are JSON-serialisable (dicts, lists, strings, numbers, None).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import deque
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Every key is namespaced, so the database can be shared with other apps.
PREFIX = "angel:"


class MemoryStore:
    """In-process store. `clock` is injectable so tests can move time."""

    backend = "memory"

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._values: dict[str, tuple[Any, float | None]] = {}
        self._windows: dict[str, deque[float]] = {}

    def _live(self, key: str) -> Any | None:
        item = self._values.get(key)
        if item is None:
            return None
        value, expires_at = item
        if expires_at is not None and self._clock() >= expires_at:
            del self._values[key]
            return None
        return value

    async def get(self, key: str) -> Any | None:
        return self._live(key)

    async def get_many(self, keys: list[str]) -> list[Any | None]:
        return [self._live(k) for k in keys]

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        expires_at = self._clock() + ttl if ttl else None
        # Round-trip through JSON so both backends hand back equal values.
        self._values[key] = (json.loads(json.dumps(value)), expires_at)

    async def delete_prefix(self, prefix: str) -> None:
        for key in [k for k in self._values if k.startswith(prefix)]:
            del self._values[key]

    async def count_prefix(self, prefix: str) -> int:
        return sum(1 for k in list(self._values) if k.startswith(prefix) and self._live(k) is not None)

    async def push_recent(self, key: str, value: str, max_len: int) -> None:
        """Move `value` to the front of a capped most-recent-first list."""
        items = [v for v in (self._live(key) or []) if v != value]
        self._values[key] = ([value, *items][:max_len], None)

    async def recent(self, key: str) -> list[str]:
        return list(self._live(key) or [])

    async def increment(self, key: str, amount: int, ttl: float) -> int:
        value = int(self._live(key) or 0) + amount
        item = self._values.get(key)
        expires_at = item[1] if item and item[1] else self._clock() + ttl
        self._values[key] = (value, expires_at)
        return value

    async def window_hit(self, key: str, window: float, limit: int) -> float | None:
        """Record a hit in a sliding window. None if allowed, else seconds to wait."""
        now = self._clock()
        hits = self._windows.setdefault(key, deque())
        while hits and now - hits[0] >= window:
            hits.popleft()
        if len(hits) >= limit:
            return max(1.0, window - (now - hits[0]))
        hits.append(now)
        return None

    async def ping(self) -> bool:
        return True


class RedisStore:
    """Redis-backed store, with an in-process fallback for outages."""

    backend = "redis"

    def __init__(self, url: str, client: Any = None) -> None:
        if client is None:
            import redis.asyncio as redis   # lazy: only needed when configured
            client = redis.from_url(
                url, decode_responses=True,
                socket_timeout=5, socket_connect_timeout=5,
            )
        self._redis = client
        self._fallback = MemoryStore()
        self._degraded_since: float | None = None

    async def _run(self, op: str, primary, fallback):
        try:
            result = await primary()
        except Exception as exc:   # redis errors, timeouts, DNS: all degrade
            if self._degraded_since is None:
                logger.warning("Redis unavailable (%s: %s); using in-process state", op, exc)
                self._degraded_since = time.time()
            return await fallback()
        if self._degraded_since is not None:
            logger.warning("Redis reachable again; shared state restored")
            self._degraded_since = None
        return result

    async def get(self, key: str) -> Any | None:
        async def primary():
            raw = await self._redis.get(key)
            return None if raw is None else json.loads(raw)
        return await self._run("get", primary, lambda: self._fallback.get(key))

    async def get_many(self, keys: list[str]) -> list[Any | None]:
        async def primary():
            if not keys:
                return []
            return [None if raw is None else json.loads(raw) for raw in await self._redis.mget(keys)]
        return await self._run("get_many", primary, lambda: self._fallback.get_many(keys))

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        async def primary():
            await self._redis.set(key, json.dumps(value), px=int(ttl * 1000) if ttl else None)
        await self._run("set", primary, lambda: self._fallback.set(key, value, ttl))

    async def delete_prefix(self, prefix: str) -> None:
        async def primary():
            batch = []
            async for key in self._redis.scan_iter(match=prefix + "*", count=500):
                batch.append(key)
                if len(batch) >= 500:
                    await self._redis.delete(*batch)
                    batch = []
            if batch:
                await self._redis.delete(*batch)
        await self._run("delete_prefix", primary, lambda: self._fallback.delete_prefix(prefix))

    async def count_prefix(self, prefix: str) -> int:
        async def primary():
            return sum([1 async for _ in self._redis.scan_iter(match=prefix + "*", count=500)])
        return await self._run("count_prefix", primary, lambda: self._fallback.count_prefix(prefix))

    async def push_recent(self, key: str, value: str, max_len: int) -> None:
        async def primary():
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.lrem(key, 0, value)
                pipe.lpush(key, value)
                pipe.ltrim(key, 0, max_len - 1)
                await pipe.execute()
        await self._run("push_recent", primary, lambda: self._fallback.push_recent(key, value, max_len))

    async def recent(self, key: str) -> list[str]:
        return await self._run(
            "recent", lambda: self._redis.lrange(key, 0, -1), lambda: self._fallback.recent(key),
        )

    async def increment(self, key: str, amount: int, ttl: float) -> int:
        async def primary():
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.incrby(key, amount)
                pipe.ttl(key)
                value, remaining = await pipe.execute()
            # Set the expiry only when the counter has none (it was just
            # created), so a day's counter dies with the day instead of being
            # pushed back on every increment. Plain EXPIRE, not EXPIRE NX,
            # which older and hosted Redis versions may not support.
            if remaining < 0:
                await self._redis.expire(key, int(ttl))
            return int(value)
        return await self._run("increment", primary, lambda: self._fallback.increment(key, amount, ttl))

    async def window_hit(self, key: str, window: float, limit: int) -> float | None:
        async def primary():
            now = time.time()
            member = f"{now}:{uuid.uuid4().hex}"
            # Add first, then count, all in one transaction, so two workers
            # can't both see room for the last slot. An over-limit hit is
            # removed again straight away.
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(key, "-inf", now - window)
                pipe.zadd(key, {member: now})
                pipe.zcard(key)
                pipe.pexpire(key, int(window * 1000) + 1000)
                _, _, count, _ = await pipe.execute()
            if count <= limit:
                return None
            await self._redis.zrem(key, member)
            oldest = await self._redis.zrange(key, 0, 0, withscores=True)
            oldest_at = oldest[0][1] if oldest else now
            return max(1.0, window - (now - oldest_at))
        return await self._run(
            "window_hit", primary, lambda: self._fallback.window_hit(key, window, limit),
        )

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:
            return False


_store: MemoryStore | RedisStore | None = None


def get_store() -> MemoryStore | RedisStore:
    """The process-wide store, created on first use from ANGEL_REDIS_URL."""
    global _store
    if _store is None:
        url = os.getenv("ANGEL_REDIS_URL", "").strip()
        if url:
            _store = RedisStore(url)
            logger.info("Shared state: Redis")
        else:
            _store = MemoryStore()
            logger.info("Shared state: in-process memory (set ANGEL_REDIS_URL to persist)")
    return _store


def use_store(store: MemoryStore | RedisStore | None) -> None:
    """Replace the process-wide store. For tests; None re-reads the env."""
    global _store
    _store = store
