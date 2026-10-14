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


def reset_postgres(url: str) -> None:
    """Empty the state tables in a test database (created on first use)."""
    import asyncio

    import asyncpg

    async def run():
        conn = await asyncpg.connect(url)
        try:
            await conn.execute("""
                DO $$ BEGIN
                  IF EXISTS (SELECT FROM pg_namespace WHERE nspname = 'angel_state') THEN
                    TRUNCATE angel_state.kv, angel_state.hits;
                  END IF;
                END $$;""")
        finally:
            await conn.close()

    asyncio.run(run())


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    """Give every test an empty state store (see angel_filter/state.py).

    The query cache, conversations, rate-limit counters and geocode caches all
    live in that store. Shared across tests, a query matching an earlier test's
    key was served the stored payload and never reached its orchestrator spy,
    and enough requests across the suite would trip the rate limit in an
    unrelated test. Tests must not depend on execution order.

    ANGEL_DATABASE_URL is cleared so a developer's real database is never
    touched. With ANGEL_TEST_STORE=postgres, every test instead runs on the
    Postgres backend against ANGEL_TEST_DATABASE_URL (a throwaway database; CI
    uses a service container), with the in-memory fallback disabled so a
    backend error fails the test instead of hiding.
    """
    from angel_filter import state

    monkeypatch.delenv("ANGEL_DATABASE_URL", raising=False)
    if os.getenv("ANGEL_TEST_STORE") == "postgres":
        url = os.environ["ANGEL_TEST_DATABASE_URL"]
        reset_postgres(url)
        state.use_store(state.PostgresStore(url, fallback=False))
    else:
        state.use_store(state.MemoryStore())
    yield
    state.use_store(None)
