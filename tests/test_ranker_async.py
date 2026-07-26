"""Tests that the Ollama embedding path is genuinely non-blocking.

The methods here are `async def`, but that alone proves nothing — the earlier
version was `async def` too while calling the *synchronous* `ollama.embeddings`,
which blocks the single event-loop thread until the model responds and freezes
every other in-flight request (CLAUDE.md: "Do not introduce blocking I/O into
these paths").

These tests exercise the real Ranker methods against a fake AsyncClient whose
`embeddings` coroutine sleeps, so we can observe two things a blocking
implementation could never do:

  1. yielding — a separate task makes progress *while* an embedding is in
     flight, i.e. the coroutine hands control back to the loop mid-call.
  2. concurrency — N embeddings complete in ~1x the per-call delay, not Nx,
     because _embed_all_ollama gathers them instead of awaiting in sequence.

No real Ollama, no network. The fake is injected by setting `_ollama_client`,
which `Ranker._ollama()` returns as-is when already populated.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from angel_filter.providers.base import ProviderResult
from angel_filter.ranker import QueryIntent, Ranker


class FakeAsyncOllama:
    """Stand-in for ollama.AsyncClient with a controllable per-call delay.

    Each embeddings() call sleeps `delay` seconds (a real await, so the loop is
    free to run other tasks) and returns a deterministic vector. `concurrent`
    tracks the peak number of overlapping calls, which is how we prove gather
    actually overlaps them.
    """

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls = 0
        self._in_flight = 0
        self.max_in_flight = 0

    async def embeddings(self, model: str, prompt: str):
        self.calls += 1
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self._in_flight -= 1
        # A trivial but non-degenerate vector so cosine math stays well-defined.
        return {"embedding": [float(len(prompt)), 1.0, 0.0]}


def _ranker_with_fake(delay: float = 0.05) -> tuple[Ranker, FakeAsyncOllama]:
    r = Ranker()
    fake = FakeAsyncOllama(delay=delay)
    r._ollama_client = fake          # _ollama() returns this instead of importing
    r._ollama_available = True       # skip the live probe
    return r, fake


def _results(n: int) -> list[ProviderResult]:
    return [
        ProviderResult(title=f"Place {i}", snippet="lunch", provider=f"p{i}")
        for i in range(n)
    ]


# --- Yielding ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embedding_call_yields_to_the_event_loop():
    """A background task must make progress while an embedding is in flight.

    If the embedding blocked the thread, the ticker could not increment until
    the embedding returned. That it ticks several times during a single call is
    direct evidence the coroutine yields.
    """
    ranker, _ = _ranker_with_fake(delay=0.1)

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    bg = asyncio.create_task(ticker())
    try:
        await ranker._embed_query("some preference text", backend="ollama")
    finally:
        bg.cancel()

    assert ticks > 5, (
        f"background task ticked only {ticks} times during a 0.1s embedding — "
        "the call is blocking the event loop instead of yielding"
    )


# --- Concurrency ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_all_runs_concurrently_not_sequentially():
    """Embedding N results must take ~1x the per-call delay, not Nx.

    Sequential awaits would sum to N * delay; gather overlaps them so wall time
    stays near a single delay.
    """
    ranker, fake = _ranker_with_fake(delay=0.1)
    results = _results(8)

    start = time.perf_counter()
    vecs = await ranker._embed_all_ollama(results)
    elapsed = time.perf_counter() - start

    assert fake.calls == 8
    assert len(vecs) == 8
    # 8 sequential calls would be ~0.8s. Concurrent should finish near 0.1s;
    # allow generous headroom for scheduling so the test isn't flaky.
    assert elapsed < 0.4, (
        f"8 embeddings took {elapsed:.3f}s — expected ~0.1s if concurrent, "
        "~0.8s if sequential"
    )


@pytest.mark.asyncio
async def test_embed_all_actually_overlaps_calls():
    """Stronger than timing: prove multiple calls were in flight at once."""
    ranker, fake = _ranker_with_fake(delay=0.05)

    await ranker._embed_all_ollama(_results(5))

    assert fake.max_in_flight > 1, (
        f"peak concurrency was {fake.max_in_flight} — calls ran one at a time"
    )


@pytest.mark.asyncio
async def test_embed_all_preserves_result_order():
    """gather returns results in submission order, so index i maps to result i.

    Regression guard: if this were built with as_completed or a dict populated
    out of order, embeddings could be misattributed to the wrong result and
    every downstream score would be silently wrong.
    """
    ranker, _ = _ranker_with_fake(delay=0.01)
    # Titles of distinct lengths -> distinct first vector component (len(prompt)).
    results = [
        ProviderResult(title="A", snippet="", provider="p"),          # "A. " -> len 3
        ProviderResult(title="BBBB", snippet="", provider="p"),       # "BBBB. " -> len 6
        ProviderResult(title="CCCCCCCCC", snippet="", provider="p"),  # longer
    ]
    vecs = await ranker._embed_all_ollama(results)

    lengths = [len(f"{r.title}. {r.snippet}") for r in results]
    for i, expected_len in enumerate(lengths):
        assert vecs[i][0] == float(expected_len), (
            f"vector {i} has component {vecs[i][0]}, expected {expected_len} — "
            "embeddings were reordered relative to their results"
        )


# --- Wiring: the whole rank() path uses the async client -----------------------

@pytest.mark.asyncio
async def test_rank_end_to_end_on_the_async_ollama_path():
    """rank() must drive the async client and return scored results.

    Every other test in the suite forces _ollama_available = False, so this is
    the only coverage that runs rank() through the real Ollama branch.
    """
    ranker, fake = _ranker_with_fake(delay=0.01)

    ranked = await ranker.rank(
        "cheap lunch nearby",
        _results(4),
        top_k=3,
        intent=QueryIntent.GENERAL,
    )

    assert len(ranked) == 3
    # results embeddings (4) + preference embedding (1)
    assert fake.calls == 5
    assert all(0.0 <= r.score <= 1.0 for r in ranked)


@pytest.mark.asyncio
async def test_client_is_created_once_and_cached():
    """_ollama() must reuse one AsyncClient, not build a new one per call."""
    ranker, fake = _ranker_with_fake()
    assert ranker._ollama() is fake
    assert ranker._ollama() is fake  # second call returns the same instance
