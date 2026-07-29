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
def _clear_query_cache():
    """Empty the process-wide query cache around every test.

    CACHE is module-level state shared by both transports. Once the NLIP path
    started caching, a test whose query matched an earlier test's key was served
    the stored payload and never reached its orchestrator spy — the assertion
    then failed on an empty kwargs dict, passing in isolation and failing in the
    suite. Tests must not depend on execution order, so this resets it.

    QueryCache exposes no public clear(); /cache/clear reaches into the same two
    attributes, so this mirrors it rather than inventing an API.
    """
    from angel_filter.cache import CACHE
    from angel_filter import geocode

    def _reset():
        CACHE._store.clear()
        CACHE._history.clear()
        # The venue-coordinate and locality caches are process-wide and
        # deliberately outlive a query, so without this a test that geocodes
        # "Joe's Pizza" makes the next one's lookup count zero.
        geocode._coord_cache.clear()
        geocode._locality_cache.clear()

    _reset()
    yield
    _reset()
