"""Shared pytest setup.

`angel_filter.server` builds the orchestrator at import time
(`ORCHESTRATOR = _build_orchestrator()`), which raises RuntimeError if no
provider is configured. Any test that imports the server module therefore
needs at least one provider present in the environment *before collection*.

CI runs with no API keys and no .env, so we register the one provider that
needs neither — Ollama — by setting OLLAMA_MODEL. This only makes
_build_orchestrator register OllamaProvider; it opens no connection at import,
and tests that touch the orchestrator mock it, so Ollama is never actually
called. Locally a developer's real .env is left untouched (setdefault).

This keeps the suite deterministic and network-free per the project's testing
rule, regardless of what is or isn't set on the machine running it.
"""

import os

# setdefault: don't clobber a real value a developer may have set locally.
os.environ.setdefault("OLLAMA_MODEL", "llama3.2")

import pytest


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    """Give every test an empty in-memory store (see angel_filter/state.py).

    The query cache, conversations, rate-limit counters and geocode caches all
    live in that store. Shared across tests, a query matching an earlier test's
    key was served the stored payload and never reached its orchestrator spy,
    and enough requests across the suite would trip the rate limit in an
    unrelated test. Tests must not depend on execution order.

    ANGEL_REDIS_URL is cleared so a developer's real Redis is never touched.
    With ANGEL_TEST_STORE=redis, every test runs on an in-process fake Redis
    instead, so the whole app is exercised on the Redis backend (CI does both).
    """
    from angel_filter import state

    monkeypatch.delenv("ANGEL_REDIS_URL", raising=False)
    if os.getenv("ANGEL_TEST_STORE") == "redis":
        import fakeredis

        client = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)
        state.use_store(state.RedisStore("redis://fake", client=client))
    else:
        state.use_store(state.MemoryStore())
    yield
    state.use_store(None)
