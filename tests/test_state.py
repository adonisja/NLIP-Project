"""The shared state store: one behaviour, two backends.

Every behaviour test runs against MemoryStore and, when a test database is
available (ANGEL_TEST_DATABASE_URL; CI provides one), against PostgresStore, so
the two can't drift. The Postgres-only tests cover what the database is for:
state shared between workers, state that survives a restart, nothing exposed
through Supabase's public API, and degrading to in-process state when the
database is down.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from angel_filter import state
from angel_filter.state import MemoryStore, PostgresStore

DB_URL = os.getenv("ANGEL_TEST_DATABASE_URL")
needs_db = pytest.mark.skipif(not DB_URL, reason="set ANGEL_TEST_DATABASE_URL to run Postgres tests")


def _pg() -> PostgresStore:
    """A store on the test database. Fallback off: errors must fail the test."""
    return PostgresStore(DB_URL, fallback=False)


@pytest.fixture
def clean_db():
    if DB_URL:
        from tests.conftest import reset_postgres
        reset_postgres(DB_URL)


@pytest.fixture(params=["memory", pytest.param("postgres", marks=needs_db)])
def store(request, clean_db):
    return MemoryStore() if request.param == "memory" else _pg()


# --- Behaviour both backends must share ------------------------------------------

@pytest.mark.asyncio
async def test_values_round_trip_as_json(store):
    value = {"query": "ramen", "payload": {"results": [{"score": 0.5, "sponsored": None}]}}
    await store.set("angel:k", value)
    assert await store.get("angel:k") == value
    assert await store.get("angel:missing") is None


@pytest.mark.asyncio
async def test_set_overwrites(store):
    await store.set("angel:k", 1)
    await store.set("angel:k", [2])
    assert await store.get("angel:k") == [2]


@pytest.mark.asyncio
async def test_get_many_keeps_order_and_gaps(store):
    await store.set("angel:a", 1)
    await store.set("angel:c", 3)
    assert await store.get_many(["angel:a", "angel:b", "angel:c"]) == [1, None, 3]
    assert await store.get_many([]) == []


@pytest.mark.asyncio
async def test_delete_and_count_by_prefix(store):
    await store.set("angel:cache:1", 1)
    await store.set("angel:cache:2", 2)
    await store.set("angel:other", 3)
    assert await store.count_prefix("angel:cache:") == 2
    await store.delete_prefix("angel:cache:")
    assert await store.count_prefix("angel:cache:") == 0
    assert await store.get("angel:other") == 3


@pytest.mark.asyncio
async def test_prefixes_are_literal(store):
    """'_' and '%' are LIKE wildcards; a prefix must not match by pattern."""
    await store.set("angel:a_b", 1)
    await store.set("angel:axb", 2)
    assert await store.count_prefix("angel:a_") == 1


@pytest.mark.asyncio
async def test_recent_list_is_deduplicated_and_capped(store):
    for v in ["a", "b", "c", "a", "d"]:
        await store.push_recent("angel:recent", v, max_len=3)
    assert await store.recent("angel:recent") == ["d", "a", "c"]


@pytest.mark.asyncio
async def test_increment_counts_and_can_undo(store):
    assert await store.increment("angel:n", 1, ttl=60) == 1
    assert await store.increment("angel:n", 1, ttl=60) == 2
    assert await store.increment("angel:n", -1, ttl=60) == 1
    assert await store.get("angel:n") == 1


@pytest.mark.asyncio
async def test_sliding_window_allows_up_to_the_limit(store):
    results = [await store.window_hit("angel:rate:x", window=60, limit=2) for _ in range(3)]
    assert results[:2] == [None, None]
    assert results[2] is not None and 1 <= results[2] <= 60
    assert await store.window_hit("angel:rate:y", window=60, limit=2) is None


@pytest.mark.asyncio
async def test_entries_expire(store):
    await store.set("angel:k", "v", ttl=0.2)
    await store.increment("angel:n", 5, ttl=0.2)
    await asyncio.sleep(0.3)
    assert await store.get("angel:k") is None
    assert await store.count_prefix("angel:") == 0
    # An expired counter starts again from zero.
    assert await store.increment("angel:n", 1, ttl=60) == 1


@pytest.mark.asyncio
async def test_memory_ttl_follows_the_injected_clock():
    clock = {"now": 1000.0}
    store = MemoryStore(clock=lambda: clock["now"])
    await store.set("angel:k", "v", ttl=10)
    clock["now"] += 11
    assert await store.get("angel:k") is None


# --- What the database is for ------------------------------------------------------

@needs_db
@pytest.mark.asyncio
async def test_expired_rows_are_deleted_not_just_hidden(clean_db):
    """Postgres has no key expiry; without cleanup the tables grow forever."""
    import asyncpg

    store = _pg()
    store.CLEANUP_EVERY = 0
    await store.set("angel:old", "v", ttl=0.1)
    await asyncio.sleep(0.2)
    await store.set("angel:trigger", "v")   # writes run the cleanup

    conn = await asyncpg.connect(DB_URL)
    try:
        assert await conn.fetchval("SELECT count(*) FROM angel_state.kv WHERE key = 'angel:old'") == 0
    finally:
        await conn.close()


@needs_db
@pytest.mark.asyncio
async def test_two_workers_share_one_rate_limit(clean_db):
    """Per-process limits let N workers allow N times the limit. A shared database doesn't."""
    worker_a, worker_b = _pg(), _pg()
    assert await worker_a.window_hit("angel:rate:ip", window=60, limit=2) is None
    assert await worker_b.window_hit("angel:rate:ip", window=60, limit=2) is None
    assert await worker_a.window_hit("angel:rate:ip", window=60, limit=2) is not None


@needs_db
@pytest.mark.asyncio
async def test_concurrent_hits_cannot_exceed_the_limit(clean_db):
    """Twenty simultaneous requests from separate workers; exactly five get through."""
    workers = [_pg() for _ in range(4)]
    results = await asyncio.gather(*(
        workers[i % 4].window_hit("angel:rate:burst", window=60, limit=5) for i in range(20)
    ))
    assert sum(r is None for r in results) == 5


@needs_db
@pytest.mark.asyncio
async def test_a_conversation_survives_a_restart(clean_db):
    from angel_filter.constraints import QueryConstraints
    from angel_filter.conversation import ConversationStore, Turn

    before = ConversationStore(store=_pg())
    await before.record("tok", Turn(query="ramen", constraints=QueryConstraints(budget=20), top_price=18.0))

    after_restart = ConversationStore(store=_pg())   # fresh process, same database
    conversation = await after_restart.get("tok")
    assert conversation.latest.query == "ramen"
    assert conversation.latest.top_price == 18.0


@needs_db
@pytest.mark.asyncio
async def test_the_daily_cap_survives_a_restart(clean_db, monkeypatch):
    from angel_filter import limits

    monkeypatch.setattr(limits, "DAILY_QUERY_LIMIT", 2)
    state.use_store(_pg())
    await limits.check_query_limits("1.1.1.1")
    await limits.check_query_limits("2.2.2.2")

    state.use_store(_pg())   # the restart
    with pytest.raises(Exception) as exc:
        await limits.check_query_limits("3.3.3.3")
    assert getattr(exc.value, "status_code", None) == 429


@needs_db
@pytest.mark.asyncio
async def test_state_is_not_exposed_through_supabase_public_api(clean_db):
    """Tables in `public` are served by Supabase's Data API; ours must not be."""
    import asyncpg

    await _pg().set("angel:k", 1)   # creates the schema
    conn = await asyncpg.connect(DB_URL)
    try:
        rows = await conn.fetch("""
            SELECT n.nspname, c.relname, c.relrowsecurity
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname IN ('kv', 'hits') AND c.relkind = 'r'""")
    finally:
        await conn.close()
    assert {(r["nspname"], r["relname"]) for r in rows} == {("angel_state", "kv"), ("angel_state", "hits")}
    assert all(r["relrowsecurity"] for r in rows), "row level security is off"


# --- Outages ---------------------------------------------------------------------------

_UNREACHABLE = "postgresql://nobody:x@127.0.0.1:1/none"


@pytest.mark.asyncio
async def test_an_unreachable_database_degrades_to_in_process_state():
    store = PostgresStore(_UNREACHABLE)

    await store.set("angel:k", {"v": 1})
    assert await store.get("angel:k") == {"v": 1}
    assert await store.increment("angel:n", 1, ttl=60) == 1
    assert await store.window_hit("angel:rate:x", window=60, limit=1) is None
    assert await store.window_hit("angel:rate:x", window=60, limit=1) is not None
    await store.push_recent("angel:r", "a", 3)
    assert await store.recent("angel:r") == ["a"]
    await store.delete_prefix("angel:")
    assert await store.count_prefix("angel:") == 0
    assert await store.ping() is False


@pytest.mark.asyncio
async def test_without_fallback_an_outage_raises():
    """Tests run this way, so a broken backend can't hide behind memory."""
    with pytest.raises(Exception):
        await PostgresStore(_UNREACHABLE, fallback=False).get("angel:k")


def test_the_store_follows_angel_database_url(monkeypatch):
    monkeypatch.setenv("ANGEL_DATABASE_URL", "postgresql://localhost/angel")
    state.use_store(None)
    assert state.get_store().backend == "postgres"   # creating the store doesn't connect

    monkeypatch.delenv("ANGEL_DATABASE_URL")
    state.use_store(None)
    assert state.get_store().backend == "memory"
