"""/metrics must count searches on every route, not only POST /query.

The counters were updated inline in the /query handlers. The UI searches
through /nlip/, so real use never reached the metrics at all.
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

import angel_filter.server as server
from angel_filter.constraints import QueryConstraints
from angel_filter.orchestrator import OrchestratorResponse
from angel_filter.providers.base import ProviderResult
from angel_filter.ranker import QueryIntent, RankedResult


def _sample(name: str, labels: dict | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


class _Spy:
    providers: list = []

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def handle_query(self, user_query, **kwargs):
        if self.fail:
            raise RuntimeError("every provider failed")
        ranked = [
            RankedResult(
                result=ProviderResult(f"Venue {i}", "s", provider="mock", sponsored=(i == 0)),
                score=0.5, rationale="r", axis_scores={}, consensus_count=1,
            )
            for i in range(2)
        ]
        return OrchestratorResponse(
            ranked=ranked, providers_used=["mock"], providers_failed=[],
            intent=QueryIntent.GENERAL, constraints=QueryConstraints(),
        )


@pytest.mark.asyncio
@pytest.mark.skipif(not server._NLIP_AVAILABLE, reason="NLIP libraries not available")
async def test_an_nlip_search_is_counted(monkeypatch):
    from nlip_sdk.nlip import NLIP_Factory

    monkeypatch.setattr(server, "ORCHESTRATOR", _Spy())
    ok = _sample("angel_filter_queries_total", {"status": "success"})
    sponsored = _sample("angel_filter_sponsored_penalties_total")
    timed = _sample("angel_filter_query_duration_seconds_count")

    await server.AngelFilterSession().execute(NLIP_Factory.create_text("lunch near me"))

    assert _sample("angel_filter_queries_total", {"status": "success"}) == ok + 1
    assert _sample("angel_filter_sponsored_penalties_total") == sponsored + 1
    assert _sample("angel_filter_query_duration_seconds_count") == timed + 1


@pytest.mark.asyncio
async def test_a_failed_search_is_counted_as_an_error(monkeypatch):
    monkeypatch.setattr(server, "ORCHESTRATOR", _Spy(fail=True))
    errors = _sample("angel_filter_queries_total", {"status": "error"})

    with pytest.raises(RuntimeError):
        await server._run_search(user_query="lunch")

    assert _sample("angel_filter_queries_total", {"status": "error"}) == errors + 1
