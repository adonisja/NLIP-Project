"""Tests for the axis priority override on the REST /query path.

The picker shipped on the NLIP path only, so /query silently ignored a
`priority` field while /nlip/ honoured it — the two transports ranked the same
request differently. CLAUDE.md keeps the fallback FastAPI server as demo
insurance, so that divergence mattered.

These pin the parity: one shared parser (`_parse_priority`) behind both
transports, and — the subtler half — the cache key includes the priority, so a
Price query cannot be served a Rating query's cached ranking.

The handlers themselves sit behind auth and rate-limit dependencies, so these
exercise the units the handlers call rather than booting an HTTP client. That
matches how the rest of the suite is written and keeps the run network-free.
"""

from __future__ import annotations

import pytest

from angel_filter.ranker import QueryIntent
from angel_filter.server import _cache_pref, _parse_priority


# --- The shared parser ---------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("price",    QueryIntent.PRICE),
    ("distance", QueryIntent.DISTANCE),
    ("rating",   QueryIntent.RATING),
    ("general",  QueryIntent.GENERAL),
])
def test_parse_priority_maps_each_axis(value, expected):
    assert _parse_priority(value) is expected


@pytest.mark.parametrize("value", ["PRICE", "Distance", "  rating  "])
def test_parse_priority_normalises_case_and_whitespace(value):
    """A hand-rolled REST client won't necessarily send the UI's exact casing."""
    assert _parse_priority(value) is not None


@pytest.mark.parametrize("value", [None, "", "banana", "cheapest", "P1"])
def test_parse_priority_degrades_to_none(value):
    """Absent or unrecognised means auto-detect — never a 500.

    A REST client that sends a typo should get the inferred ranking, the same
    as one that sends nothing.
    """
    assert _parse_priority(value) is None


def test_rest_and_nlip_share_one_parser():
    """The NLIP extractor must delegate here, not keep a second copy.

    Two independent implementations would drift — that is exactly how /query
    came to ignore a field /nlip/ honoured.
    """
    nlip = pytest.importorskip("nlip_sdk.nlip")
    import angel_filter.server as server
    if not server._NLIP_AVAILABLE:
        pytest.skip("NLIP libraries not available")

    m = nlip.NLIP_Factory.create_text("lunch")
    m.add_text("PRICE", label="priority")
    assert server._extract_priority(m) is _parse_priority("PRICE")


# --- The cache key -------------------------------------------------------------
# The bug this guards: /query caches on (query, preference). Ranking by price
# and by rating produces different orderings for the same query, so if priority
# is not part of the key the second request is served the first one's ranking.

def test_cache_key_separates_priorities():
    keys = {_cache_pref(None, None, None, p) for p in ["price", "distance", "rating"]}
    assert len(keys) == 3, f"priorities collapsed to the same cache key: {keys}"


def test_cache_key_auto_differs_from_explicit():
    """Auto-detect and an explicit axis are different requests.

    Even when detection would infer the same intent, the caller asked for
    different things and a later detection change must not leak across them.
    """
    assert _cache_pref(None, None, None, None) != _cache_pref(None, None, None, "price")


@pytest.mark.parametrize("equivalent", ["PRICE", "  price  ", "price"])
def test_cache_key_normalises_before_keying(equivalent):
    """Casing must not fragment the cache into near-duplicate entries."""
    assert _cache_pref(None, None, None, equivalent) == _cache_pref(None, None, None, "price")


def test_unrecognised_priority_shares_the_auto_cache_key():
    """Garbage means auto, so it must key like auto — not mint a junk entry."""
    assert _cache_pref(None, None, None, "banana") == _cache_pref(None, None, None, None)


def test_cache_key_still_separates_location():
    """Priority folding must not clobber the existing location component."""
    a = _cache_pref("quiet", 40.768, -73.982, "price")
    b = _cache_pref("quiet", 34.052, -118.243, "price")
    assert a != b


def test_cache_key_is_backward_compatible_without_priority():
    """The default argument keeps pre-existing callers keying exactly as before."""
    assert _cache_pref("quiet", 40.768, -73.982) == _cache_pref("quiet", 40.768, -73.982, None)


# --- The request model ---------------------------------------------------------

def test_query_model_accepts_priority_and_defaults_to_none():
    """`priority` is optional: existing clients that omit it keep working."""
    import angel_filter.server as server

    # QueryIn is defined inside whichever branch built `app`, so it isn't
    # importable by name. Reach it through the route's body field instead —
    # that also proves the *live* model is the one carrying the field.
    route = next(r for r in server.app.routes if getattr(r, "path", "") == "/query")
    model = route.body_field.type_

    assert model(query="lunch").priority is None
    assert model(query="lunch", priority="price").priority == "price"


# --- The handler actually uses it ----------------------------------------------
# Parser and cache-key tests both pass even if the handler drops the field on the
# floor — which is precisely the bug being fixed. These drive the real endpoint.

def _reset_cache(cache) -> None:
    """Empty the shared query cache between tests.

    QueryCache exposes no public clear(); the /cache/clear endpoint reaches into
    the same two attributes, so this mirrors it rather than inventing an API.
    """
    cache._store.clear()
    cache._history.clear()


@pytest.fixture
def rest_client(monkeypatch):
    """A TestClient with auth stubbed and the orchestrator spied.

    /query sits behind enforce_query_limits (session + rate limit + daily cap).
    Overriding that one dependency is enough to reach the handler without
    standing up an OAuth session.
    """
    from fastapi.testclient import TestClient
    import angel_filter.server as server
    from angel_filter.limits import enforce_query_limits
    from angel_filter.constraints import QueryConstraints
    from angel_filter.orchestrator import OrchestratorResponse

    seen: dict = {}

    class _Spy:
        async def handle_query(self, user_query, **kwargs):
            seen.update(kwargs)
            return OrchestratorResponse(
                ranked=[], providers_used=["mock"], providers_failed=[],
                intent=kwargs.get("intent") or QueryIntent.GENERAL,
                constraints=QueryConstraints(),
            )

    monkeypatch.setattr(server, "ORCHESTRATOR", _Spy())
    _reset_cache(server.CACHE)
    server.app.dependency_overrides[enforce_query_limits] = lambda: "tester"
    try:
        yield TestClient(server.app), seen
    finally:
        server.app.dependency_overrides.clear()
        _reset_cache(server.CACHE)


@pytest.mark.parametrize("sent,expected", [
    ("price",    QueryIntent.PRICE),
    ("distance", QueryIntent.DISTANCE),
    ("rating",   QueryIntent.RATING),
])
def test_post_query_forwards_priority_to_the_orchestrator(rest_client, sent, expected):
    client, seen = rest_client
    r = client.post("/query", json={"query": "lunch", "priority": sent})
    assert r.status_code == 200
    assert seen["intent"] is expected


def test_post_query_without_priority_leaves_detection_alone(rest_client):
    """Omitting the field must send intent=None so detect_intent() still runs."""
    client, seen = rest_client
    r = client.post("/query", json={"query": "lunch"})
    assert r.status_code == 200
    assert seen["intent"] is None


def test_post_query_with_garbage_priority_does_not_500(rest_client):
    """A bad value degrades to auto rather than failing the request."""
    client, seen = rest_client
    r = client.post("/query", json={"query": "lunch", "priority": "banana"})
    assert r.status_code == 200
    assert seen["intent"] is None
