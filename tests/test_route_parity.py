"""Tests that both transports expose the same routes, and that NLIP caches.

server.py declares its routes twice — once under `if _NLIP_AVAILABLE:` and once
in the fallback `else:`. Nothing kept the two lists in sync, and they drifted:
`/history` and `/cache/clear` existed only in the fallback, so on the path that
actually runs they 404'd. The UI's "Recent queries" button was dead and a stale
cache could not be cleared.

Worse, the NLIP session never touched CACHE at all, so the README's "3-hour TTL,
10 query history" and the "run any query twice → cache hit" demo were both false
in NLIP mode.

These pin the parity itself rather than the two specific paths, so the next route
added to one branch and forgotten in the other fails here instead of in a demo.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("nlip_sdk.nlip")
import angel_filter.server as server

_SRC = Path(server.__file__).read_text()


def _branch_routes() -> tuple[set[str], set[str]]:
    """Parse the route decorators out of each branch of the if/else."""
    lines = _SRC.splitlines()
    nlip_at = next(i for i, l in enumerate(lines) if l.startswith("if _NLIP_AVAILABLE:"))
    else_at = next(i for i, l in enumerate(lines) if i > nlip_at and l.startswith("else:"))

    def grab(a: int, b: int) -> set[str]:
        found = set()
        for line in lines[a:b]:
            m = re.match(r'\s*@app\.(get|post|put|delete)\("([^"]+)"\)', line)
            if m:
                found.add(f"{m.group(1).upper()} {m.group(2)}")
        return found

    return grab(nlip_at, else_at), grab(else_at, len(lines))


def test_both_branches_declare_the_same_routes():
    """The regression guard: a route added to one branch only fails here.

    Reported as a symmetric diff so the message names exactly which paths are
    missing from which branch.
    """
    nlip, fallback = _branch_routes()
    assert nlip == fallback, (
        f"missing from NLIP branch: {sorted(fallback - nlip)}; "
        f"missing from fallback branch: {sorted(nlip - fallback)}"
    )


@pytest.mark.parametrize("path", ["/health", "/metrics", "/history", "/cache/clear", "/query"])
def test_route_is_registered_exactly_once(path):
    """Zero means it 404s; two means one silently shadows the other.

    Both failure modes have already happened in this file — /history had zero,
    /health had two.
    """
    hits = [r for r in server.app.routes if getattr(r, "path", None) == path]
    assert len(hits) == 1, f"{path} has {len(hits)} registrations"


@pytest.mark.parametrize("path", ["/history", "/cache/clear", "/metrics"])
def test_diagnostic_routes_require_a_session(path):
    """These expose other users' queries and let a caller wipe shared state.

    The fallback copies predate the auth layer and have no dependency at all;
    the NLIP versions must not repeat that.
    """
    route = next(r for r in server.app.routes if getattr(r, "path", None) == path)
    names = [d.call.__name__ for d in route.dependant.dependencies]
    assert any("session" in n or "limits" in n for n in names), (
        f"{path} has no auth dependency (found {names})"
    )


# --- NLIP caching --------------------------------------------------------------

@pytest.mark.asyncio
async def test_nlip_session_caches_and_reuses_the_payload(monkeypatch):
    """Second identical query must not re-run the providers."""
    from nlip_sdk.nlip import NLIP_Factory
    from angel_filter.constraints import QueryConstraints
    from angel_filter.orchestrator import OrchestratorResponse
    from angel_filter.ranker import QueryIntent

    calls = {"n": 0}

    class _Spy:
        async def handle_query(self, user_query, **kw):
            calls["n"] += 1
            return OrchestratorResponse(
                ranked=[], providers_used=["mock"], providers_failed=[],
                intent=QueryIntent.GENERAL, constraints=QueryConstraints(),
            )

    monkeypatch.setattr(server, "ORCHESTRATOR", _Spy())
    session = server.AngelFilterSession()

    await session.execute(NLIP_Factory.create_text("same question"))
    await session.execute(NLIP_Factory.create_text("same question"))

    assert calls["n"] == 1, "second identical query re-ran the providers"


@pytest.mark.asyncio
async def test_different_priorities_do_not_share_a_cache_entry(monkeypatch):
    """Ranking by price and by rating are different requests.

    The cache key folds the priority in; without that the second query would be
    served the first one's ordering.
    """
    from nlip_sdk.nlip import NLIP_Factory
    from angel_filter.constraints import QueryConstraints
    from angel_filter.orchestrator import OrchestratorResponse
    from angel_filter.ranker import QueryIntent

    calls = {"n": 0}

    class _Spy:
        async def handle_query(self, user_query, **kw):
            calls["n"] += 1
            return OrchestratorResponse(
                ranked=[], providers_used=["mock"], providers_failed=[],
                intent=QueryIntent.GENERAL, constraints=QueryConstraints(),
            )

    monkeypatch.setattr(server, "ORCHESTRATOR", _Spy())
    session = server.AngelFilterSession()

    for axis in ("price", "rating"):
        m = NLIP_Factory.create_text("lunch")
        m.add_text(axis, label="priority")
        await session.execute(m)

    assert calls["n"] == 2, "different priorities collapsed into one cache entry"


@pytest.mark.asyncio
async def test_cached_reply_reads_identically_to_a_fresh_one(monkeypatch):
    """A cache hit must not degrade the human-readable summary.

    Both go through _format_reply_from_payload for exactly this reason — a
    separate code path for the cached case is how the two would drift.
    """
    from nlip_sdk.nlip import NLIP_Factory
    from angel_filter.constraints import QueryConstraints
    from angel_filter.orchestrator import OrchestratorResponse
    from angel_filter.providers.base import ProviderResult
    from angel_filter.ranker import QueryIntent, RankedResult

    ranked = [RankedResult(
        result=ProviderResult(title="Taco Spot", snippet="s", provider="mock", sponsored=True),
        score=0.7, rationale="why", axis_scores={}, consensus_count=1,
    )]

    class _Spy:
        async def handle_query(self, user_query, **kw):
            return OrchestratorResponse(
                ranked=ranked, providers_used=["mock"], providers_failed=[],
                intent=QueryIntent.GENERAL, constraints=QueryConstraints(),
            )

    monkeypatch.setattr(server, "ORCHESTRATOR", _Spy())
    session = server.AngelFilterSession()

    fresh = await session.execute(NLIP_Factory.create_text("tacos"))
    cached = await session.execute(NLIP_Factory.create_text("tacos"))

    assert fresh.content == cached.content
    assert "[SPONSORED]" in cached.content
