"""Shared state: where the cache, conversations, limits and geocodes live.

Two backends behind one async interface:

  - MemoryStore (default): plain dicts in this process. Lost on restart and
    not shared between workers — fine for local development and tests.
  - PostgresStore: used when ANGEL_DATABASE_URL is set, e.g. a free Supabase
    project (see docs/SETUP.md, "Shared state"). Survives restarts and is
    shared by every worker and instance, so the rate limits and the daily
    spending cap actually hold.

Callers never touch a backend directly; they call get_store() at the moment
they need it, so tests can swap the store with use_store().

If the database becomes unreachable (or a free Supabase project has paused),
PostgresStore serves each operation from an in-process MemoryStore instead and
logs a warning, then retries the database after a short cool-down. An outage
therefore degrades the app to exactly its old in-memory behaviour; it never
fails a search, and the limits still apply per process.

Values are JSON-serialisable (dicts, lists, strings, numbers, None).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
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


class _FallbackStore:
    """Runs each operation on a real backend, or on in-process memory if it fails.

    After a failure the backend is skipped for RETRY_AFTER seconds, so an
    outage costs one timeout per cool-down rather than one per operation.
    Pass fallback=False (tests do) to raise instead, so a broken backend
    can't hide behind the in-memory copy.
    """

    RETRY_AFTER = 30.0

    def __init__(self, fallback: bool = True) -> None:
        self._fallback = MemoryStore() if fallback else None
        self._down_until = 0.0
        self._degraded = False

    async def _run(self, op: str, primary, fallback):
        if self._fallback is not None and time.monotonic() < self._down_until:
            return await fallback()
        try:
            result = await primary()
        except Exception as exc:   # connection, timeout, paused project: all degrade
            if self._fallback is None:
                raise
            self._down_until = time.monotonic() + self.RETRY_AFTER
            if not self._degraded:
                logger.warning("%s unavailable (%s: %s); using in-process state",
                               self.backend, op, exc)
                self._degraded = True
            return await fallback()
        if self._degraded:
            logger.warning("%s reachable again; shared state restored", self.backend)
            self._degraded = False
        return result


_SCHEMA = "angel_state"

# The tables live in their own schema, not `public`: Supabase exposes `public`
# through its Data API by default, and this state (query history,
# conversations) must not be readable with a project's public API key. Row
# level security is enabled as a second barrier; the app connects as the
# tables' owner, which RLS doesn't restrict.
_MIGRATION = f"""
CREATE SCHEMA IF NOT EXISTS {_SCHEMA};
CREATE TABLE IF NOT EXISTS {_SCHEMA}.kv (
    key        text PRIMARY KEY,
    value      jsonb NOT NULL,
    expires_at timestamptz
);
CREATE INDEX IF NOT EXISTS kv_expires_at ON {_SCHEMA}.kv (expires_at)
    WHERE expires_at IS NOT NULL;
CREATE TABLE IF NOT EXISTS {_SCHEMA}.hits (
    key text        NOT NULL,
    at  timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS hits_key_at ON {_SCHEMA}.hits (key, at);
ALTER TABLE {_SCHEMA}.kv   ENABLE ROW LEVEL SECURITY;
ALTER TABLE {_SCHEMA}.hits ENABLE ROW LEVEL SECURITY;
"""

# A live row: no expiry, or not expired yet.
_LIVE = "(expires_at IS NULL OR expires_at > now())"


def _ttl_expr(param: str) -> str:
    """SQL for 'now + ttl seconds', or NULL when the ttl parameter is NULL."""
    return f"CASE WHEN {param}::float8 IS NULL THEN NULL ELSE now() + {param}::float8 * interval '1 second' END"


class PostgresStore(_FallbackStore):
    """Postgres-backed store (Supabase, Neon, or any Postgres 12+)."""

    backend = "postgres"

    # Postgres has no key expiry, so expired rows are deleted by the app, at
    # most this often per process.
    CLEANUP_EVERY = 600.0

    def __init__(self, url: str, fallback: bool = True) -> None:
        super().__init__(fallback)
        self._url = url
        # One pool per event loop. The server runs a single loop; tests and
        # TestClient create several, and an asyncpg pool can't cross loops.
        self._pools: dict[int, Any] = {}
        self._migrated = False
        self._last_cleanup = 0.0

    async def _pool(self):
        loop_id = id(asyncio.get_running_loop())
        pool = self._pools.get(loop_id)
        if pool is None:
            import asyncpg   # lazy: only needed when configured

            async def init(conn):
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog",
                )

            pool = await asyncpg.create_pool(
                self._url, min_size=1, max_size=5, init=init,
                # Supabase's transaction pooler (port 6543) can't hold prepared
                # statements; with the cache off, either pooler works.
                statement_cache_size=0,
                timeout=5, command_timeout=5,
            )
            self._pools[loop_id] = pool
        if not self._migrated:
            async with pool.acquire() as conn, conn.transaction():
                # Serialise first-run setup across workers starting together.
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext('angel_state.migrate'))")
                await conn.execute(_MIGRATION)
            self._migrated = True
        return pool

    async def _maybe_cleanup(self, pool) -> None:
        now = time.monotonic()
        if now - self._last_cleanup < self.CLEANUP_EVERY:
            return
        self._last_cleanup = now
        await pool.execute(f"DELETE FROM {_SCHEMA}.kv WHERE expires_at <= now()")
        await pool.execute(f"DELETE FROM {_SCHEMA}.hits WHERE at < now() - interval '1 day'")

    async def get(self, key: str) -> Any | None:
        async def primary():
            pool = await self._pool()
            return await pool.fetchval(
                f"SELECT value FROM {_SCHEMA}.kv WHERE key = $1 AND {_LIVE}", key,
            )
        return await self._run("get", primary, lambda: self._fallback.get(key))

    async def get_many(self, keys: list[str]) -> list[Any | None]:
        async def primary():
            if not keys:
                return []
            pool = await self._pool()
            rows = await pool.fetch(
                f"SELECT key, value FROM {_SCHEMA}.kv WHERE key = ANY($1::text[]) AND {_LIVE}", keys,
            )
            found = {r["key"]: r["value"] for r in rows}
            return [found.get(k) for k in keys]
        return await self._run("get_many", primary, lambda: self._fallback.get_many(keys))

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        async def primary():
            pool = await self._pool()
            await pool.execute(
                f"""INSERT INTO {_SCHEMA}.kv (key, value, expires_at)
                    VALUES ($1, $2::jsonb, {_ttl_expr('$3')})
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, expires_at = EXCLUDED.expires_at""",
                key, value, ttl,
            )
            await self._maybe_cleanup(pool)
        await self._run("set", primary, lambda: self._fallback.set(key, value, ttl))

    async def delete_prefix(self, prefix: str) -> None:
        async def primary():
            pool = await self._pool()
            await pool.execute(f"DELETE FROM {_SCHEMA}.kv WHERE starts_with(key, $1)", prefix)
            await pool.execute(f"DELETE FROM {_SCHEMA}.hits WHERE starts_with(key, $1)", prefix)
        await self._run("delete_prefix", primary, lambda: self._fallback.delete_prefix(prefix))

    async def count_prefix(self, prefix: str) -> int:
        async def primary():
            pool = await self._pool()
            return await pool.fetchval(
                f"SELECT count(*) FROM {_SCHEMA}.kv WHERE starts_with(key, $1) AND {_LIVE}", prefix,
            )
        return await self._run("count_prefix", primary, lambda: self._fallback.count_prefix(prefix))

    async def push_recent(self, key: str, value: str, max_len: int) -> None:
        async def primary():
            pool = await self._pool()
            async with pool.acquire() as conn, conn.transaction():
                # Lock this one key, so concurrent pushes can't drop each other.
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", key)
                current = await conn.fetchval(
                    f"SELECT value FROM {_SCHEMA}.kv WHERE key = $1", key,
                ) or []
                items = [value, *[v for v in current if v != value]][:max_len]
                await conn.execute(
                    f"""INSERT INTO {_SCHEMA}.kv (key, value, expires_at) VALUES ($1, $2::jsonb, NULL)
                        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
                    key, items,
                )
        await self._run("push_recent", primary, lambda: self._fallback.push_recent(key, value, max_len))

    async def recent(self, key: str) -> list[str]:
        async def primary():
            pool = await self._pool()
            return list(await pool.fetchval(
                f"SELECT value FROM {_SCHEMA}.kv WHERE key = $1", key,
            ) or [])
        return await self._run("recent", primary, lambda: self._fallback.recent(key))

    async def increment(self, key: str, amount: int, ttl: float) -> int:
        async def primary():
            pool = await self._pool()
            # One atomic statement. An expired counter restarts from `amount`
            # with a fresh expiry; a live one keeps the expiry it was created
            # with, so a day's counter dies with the day.
            return await pool.fetchval(
                f"""INSERT INTO {_SCHEMA}.kv AS kv (key, value, expires_at)
                    VALUES ($1, to_jsonb($2::bigint), {_ttl_expr('$3')})
                    ON CONFLICT (key) DO UPDATE SET
                      value = CASE WHEN kv.expires_at IS NOT NULL AND kv.expires_at <= now()
                                   THEN to_jsonb($2::bigint)
                                   ELSE to_jsonb((kv.value #>> '{{}}')::bigint + $2::bigint) END,
                      expires_at = CASE WHEN kv.expires_at IS NULL OR kv.expires_at <= now()
                                        THEN EXCLUDED.expires_at ELSE kv.expires_at END
                    RETURNING (value #>> '{{}}')::bigint""",
                key, amount, ttl,
            )
        return await self._run("increment", primary, lambda: self._fallback.increment(key, amount, ttl))

    async def window_hit(self, key: str, window: float, limit: int) -> float | None:
        async def primary():
            pool = await self._pool()
            async with pool.acquire() as conn, conn.transaction():
                # Serialise hits on this key across workers: count and insert
                # must not interleave, or two could take the last slot.
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", key)
                await conn.execute(
                    f"DELETE FROM {_SCHEMA}.hits WHERE key = $1 AND at <= now() - $2::float8 * interval '1 second'",
                    key, window,
                )
                row = await conn.fetchrow(
                    f"""SELECT count(*) AS n,
                               extract(epoch FROM min(at) + $2::float8 * interval '1 second' - now()) AS wait
                        FROM {_SCHEMA}.hits WHERE key = $1""",
                    key, window,
                )
                if row["n"] >= limit:
                    return max(1.0, float(row["wait"] or window))
                await conn.execute(f"INSERT INTO {_SCHEMA}.hits (key, at) VALUES ($1, now())", key)
            return None
        return await self._run(
            "window_hit", primary, lambda: self._fallback.window_hit(key, window, limit),
        )

    async def ping(self) -> bool:
        try:
            pool = await self._pool()
            return await pool.fetchval("SELECT 1") == 1
        except Exception:
            return False


_store: MemoryStore | PostgresStore | None = None


def get_store() -> MemoryStore | PostgresStore:
    """The process-wide store, created on first use from ANGEL_DATABASE_URL."""
    global _store
    if _store is None:
        url = os.getenv("ANGEL_DATABASE_URL", "").strip()
        if url:
            _store = PostgresStore(url)
            logger.info("Shared state: Postgres")
        else:
            _store = MemoryStore()
            logger.info("Shared state: in-process memory (set ANGEL_DATABASE_URL to persist)")
    return _store


def use_store(store: MemoryStore | PostgresStore | None) -> None:
    """Replace the process-wide store. For tests; None re-reads the env."""
    global _store
    _store = store


async def _heartbeat() -> None:
    """Write one row, so a free Supabase project sees activity and doesn't pause.

    Run by .github/workflows/keepalive.yml: `python -m angel_filter.state heartbeat`.
    """
    url = os.getenv("ANGEL_DATABASE_URL", "").strip()
    if not url:
        raise SystemExit("ANGEL_DATABASE_URL is not set")
    store = PostgresStore(url, fallback=False)
    await store.set(PREFIX + "heartbeat", {"at": time.time()})
    print("heartbeat written")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["heartbeat"]:
        asyncio.run(_heartbeat())
    else:
        raise SystemExit("usage: python -m angel_filter.state heartbeat")
