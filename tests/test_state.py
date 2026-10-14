"""The shared state store: one behaviour, two backends.

Every behaviour test runs against MemoryStore and against RedisStore on
fakeredis (an in-process Redis), so the two can't drift. The Redis-only tests
cover what the move to Redis is for: state shared between workers, state that
survives a restart, and degrading to in-process state when Redis is down.
"""

from __future__ import annotations

import pytest

from angel_filter import state
from angel_filter.state import MemoryStore, RedisStore

fakeredis = pytest.importorskip("fakeredis")


def _redis_store(server=None) -> RedisStore:
    server = server or fakeredis.FakeServer()
    return RedisStore("redis://fake", client=fakeredis.FakeAsyncRedis(server=server, decode_responses=True))


@pytest.fixture(params=["memory", "redis"])
def store(request):
    return MemoryStore() if request.param == "memory" else _redis_store()


# --- Behaviour both backends must share ------------------------------------------

@pytest.mark.asyncio
async def test_values_round_trip_as_json(store):
    value = {"query": "ramen", "payload": {"results": [{"score": 0.5, "sponsored": None}]}}
    await store.set("angel:k", value)
    assert await store.get("angel:k") == value
    assert await store.get("angel:missing") is None


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
    # A rejected hit must not take a slot: another key is unaffected, and the
    # same key is still exactly at its limit.
    assert await store.window_hit("angel:rate:y", window=60, limit=2) is None


@pytest.mark.asyncio
async def test_memory_ttl_expires_entries():
    clock = {"now": 1000.0}
    store = MemoryStore(clock=lambda: clock["now"])
    await store.set("angel:k", "v", ttl=10)
    clock["now"] += 11
    assert await store.get("angel:k") is None


@pytest.mark.asyncio
async def test_redis_keys_carry_a_ttl():
    """In Redis the key itself must expire; nothing else ever cleans it up."""
    server = fakeredis.FakeServer()
    store = _redis_store(server)
    raw = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)

    await store.set("angel:k", "v", ttl=10)
    await store.increment("angel:n", 1, ttl=100)
    await store.increment("angel:n", 1, ttl=100)
    assert 0 < await raw.pttl("angel:k") <= 10_000
    assert 0 < await raw.ttl("angel:n") <= 100


# --- What Redis is for ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_workers_share_one_rate_limit():
    """Per-process limits let N workers allow N times the limit. Redis doesn't."""
    server = fakeredis.FakeServer()
    worker_a, worker_b = _redis_store(server), _redis_store(server)

    assert await worker_a.window_hit("angel:rate:ip", window=60, limit=2) is None
    assert await worker_b.window_hit("angel:rate:ip", window=60, limit=2) is None
    assert await worker_a.window_hit("angel:rate:ip", window=60, limit=2) is not None


@pytest.mark.asyncio
async def test_a_conversation_survives_a_restart():
    from angel_filter.constraints import QueryConstraints
    from angel_filter.conversation import ConversationStore, Turn

    server = fakeredis.FakeServer()
    before = ConversationStore(store=_redis_store(server))
    await before.record("tok", Turn(query="ramen", constraints=QueryConstraints(budget=20), top_price=18.0))

    after_restart = ConversationStore(store=_redis_store(server))   # fresh process, same Redis
    conversation = await after_restart.get("tok")
    assert conversation.latest.query == "ramen"
    assert conversation.latest.top_price == 18.0


@pytest.mark.asyncio
async def test_the_daily_cap_survives_a_restart(monkeypatch):
    from angel_filter import limits

    server = fakeredis.FakeServer()
    monkeypatch.setattr(limits, "DAILY_QUERY_LIMIT", 2)

    state.use_store(_redis_store(server))
    await limits.check_query_limits("1.1.1.1")
    await limits.check_query_limits("2.2.2.2")

    state.use_store(_redis_store(server))   # the restart
    with pytest.raises(Exception) as exc:
        await limits.check_query_limits("3.3.3.3")
    assert getattr(exc.value, "status_code", None) == 429


class _DeadRedis:
    """Every command fails, like an unreachable server."""

    def __getattr__(self, name):
        async def fail(*args, **kwargs):
            raise ConnectionError("redis is down")
        return fail

    def pipeline(self, *args, **kwargs):
        raise ConnectionError("redis is down")

    def scan_iter(self, *args, **kwargs):
        raise ConnectionError("redis is down")


@pytest.mark.asyncio
async def test_an_unreachable_redis_degrades_to_in_process_state():
    store = RedisStore("redis://down", client=_DeadRedis())

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


def test_the_store_follows_angel_redis_url(monkeypatch):
    monkeypatch.setenv("ANGEL_REDIS_URL", "redis://localhost:6379/0")
    state.use_store(None)
    assert state.get_store().backend == "redis"   # building the client doesn't connect

    monkeypatch.delenv("ANGEL_REDIS_URL")
    state.use_store(None)
    assert state.get_store().backend == "memory"
