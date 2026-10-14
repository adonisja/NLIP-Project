"""Embedding failures must degrade a search, never fail it.

The ranker probes its embedding backend once. Before this fix the answer was
kept forever, so a backend that went away afterwards made rank() raise on every
search, after the providers had already been paid for, until a restart.
"""

from __future__ import annotations

import pytest

from angel_filter import ranker as ranker_mod
from angel_filter.constraints import QueryConstraints
from angel_filter.orchestrator import Orchestrator
from angel_filter.providers.base import ProviderResult
from angel_filter.providers.mock import MockProvider
from angel_filter.ranker import Ranker


class _BrokenOllama:
    def __init__(self) -> None:
        self.calls = 0

    async def embeddings(self, **kwargs):
        self.calls += 1
        raise ConnectionError("ollama went away")


class _WorkingOllama:
    async def embeddings(self, model, prompt):
        return {"embedding": [1.0, float(len(prompt) % 7), 0.5]}


def _results() -> list[ProviderResult]:
    return [
        ProviderResult("Joe's Pizza", "cheap lunch pizza", provider="a", price=12, rating=4.5),
        ProviderResult("Taco Spot", "lunch tacos", provider="b", price=9, rating=4.2),
    ]


def _ranker_believing_ollama_is_up(client) -> Ranker:
    r = Ranker()
    r._ollama_client = client
    r._ollama_available = True        # the probe succeeded earlier
    r._openai_available = False       # pinned off, so the fallback is keywords
    return r


@pytest.mark.asyncio
async def test_a_failing_backend_falls_back_to_keywords():
    r = _ranker_believing_ollama_is_up(_BrokenOllama())
    ranked = await r.rank("cheap lunch", _results(), constraints=QueryConstraints())
    assert len(ranked) == 2
    assert all("keyword fallback" in x.rationale for x in ranked)


@pytest.mark.asyncio
async def test_the_whole_pipeline_survives_an_embedding_failure():
    """The bug as users saw it: providers answered, then the search failed."""
    r = _ranker_believing_ollama_is_up(_BrokenOllama())
    response = await Orchestrator([MockProvider()], ranker=r).handle_query("lunch under $15")
    assert response.ranked


@pytest.mark.asyncio
async def test_the_next_search_checks_the_backend_again():
    """A stale True is forgotten, so recovery doesn't need a restart."""
    broken = _BrokenOllama()
    r = _ranker_believing_ollama_is_up(broken)
    await r.rank("cheap lunch", _results())
    assert r._ollama_available is not True

    r._ollama_client = _WorkingOllama()
    ranked = await r.rank("cheap lunch", _results())
    assert r._ollama_available is True
    assert not any("keyword fallback" in x.rationale for x in ranked)


@pytest.mark.asyncio
async def test_a_failed_probe_is_retried_after_the_recheck_interval(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(ranker_mod.time, "monotonic", lambda: clock["now"])
    broken = _BrokenOllama()
    r = Ranker()
    r._ollama_client = broken

    assert await r._has_ollama() is False
    assert await r._has_ollama() is False
    assert broken.calls == 1, "probed again inside the recheck interval"

    clock["now"] += ranker_mod.BACKEND_RECHECK_SECONDS + 1
    r._ollama_client = _WorkingOllama()
    assert await r._has_ollama() is True


@pytest.mark.asyncio
async def test_a_hand_set_false_is_never_reprobed(monkeypatch):
    """How the rest of the suite pins keyword mode; it must stay pinned."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(ranker_mod.time, "monotonic", lambda: clock["now"])
    broken = _BrokenOllama()
    r = Ranker()
    r._ollama_client = broken
    r._ollama_available = False
    clock["now"] += 10 * ranker_mod.BACKEND_RECHECK_SECONDS
    assert await r._has_ollama() is False
    assert broken.calls == 0


def test_the_ollama_client_has_a_timeout():
    pytest.importorskip("ollama")
    client = Ranker()._ollama()
    timeout = client._client.timeout
    assert timeout.read is not None and timeout.read <= ranker_mod.OLLAMA_EMBED_TIMEOUT


def test_the_ollama_client_uses_ollama_url(monkeypatch):
    """The provider always honoured OLLAMA_URL; the ranker's embeddings didn't."""
    pytest.importorskip("ollama")
    monkeypatch.setenv("OLLAMA_URL", "http://gpu-box:11434")
    assert str(Ranker()._ollama()._client.base_url).startswith("http://gpu-box:11434")


def test_without_ollama_url_the_client_uses_the_local_default(monkeypatch):
    pytest.importorskip("ollama")
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert "11434" in str(Ranker()._ollama()._client.base_url)
