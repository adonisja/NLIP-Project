"""Test the orchestrator end-to-end with the mock provider.

These tests do not require network or Ollama — they prove the fan-out and
ranking pipeline works as a standalone unit. Run with:
    poetry run pytest
"""

import asyncio

import pytest

from angel_filter.orchestrator import Orchestrator
from angel_filter.providers import MockProvider


@pytest.mark.asyncio
async def test_orchestrator_returns_ranked_results():
    orch = Orchestrator(providers=[MockProvider()])
    response = await orch.handle_query(
        user_query="cheap toilet paper",
        user_preference="low price, not premium brand",
        top_k=5,
    )

    assert "mock" in response.providers_used
    assert response.providers_failed == []
    assert len(response.ranked) > 0


@pytest.mark.asyncio
async def test_sponsored_results_are_penalized():
    """The sponsored item has a lower score than equally-matching organic ones."""
    orch = Orchestrator(providers=[MockProvider()])
    response = await orch.handle_query(
        user_query="toilet paper",
        top_k=10,
    )

    sponsored = [r for r in response.ranked if r.result.sponsored]
    organic = [r for r in response.ranked if not r.result.sponsored]

    assert sponsored, "expected at least one sponsored item in the canned data"
    assert organic, "expected at least one organic item in the canned data"

    # The penalty should cost the sponsored item the top spot against a
    # comparably-relevant organic result. We don't assert strict ordering —
    # the ranker is allowed to still keep a strongly-matching sponsored item
    # reasonably high — but the top result must not be sponsored when an
    # organic alternative exists.
    assert not response.ranked[0].result.sponsored, (
        f"top-ranked result was sponsored: {response.ranked[0]}"
    )


@pytest.mark.asyncio
async def test_orchestrator_tolerates_provider_failures():
    """If one provider blows up, the others still return results."""

    from angel_filter.providers.base import BaseProvider, ProviderError

    class BrokenProvider(BaseProvider):
        name = "broken"

        async def query(self, user_query: str, max_results: int = 10):
            raise ProviderError("simulated outage")

    orch = Orchestrator(providers=[MockProvider(), BrokenProvider()])
    response = await orch.handle_query(user_query="anything")

    assert "broken" in response.providers_failed
    assert "mock" in response.providers_used
    assert len(response.ranked) > 0
