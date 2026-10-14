"""Tests for cost protection now that there is no login.

Before the GitHub login was removed, it gated only the UI page and POST /query.
POST /nlip/ — the route the UI actually sends searches to — had neither a
session check nor a rate limit, so it was the one unmetered way to trigger a
paid fan-out. With no login at all, the per-client limit and the daily cap are
the only protection, so these pin that both query routes are metered and that
the client identity cannot be trivially spoofed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import angel_filter.server as server
from angel_filter import limits
from angel_filter.constraints import QueryConstraints
from angel_filter.orchestrator import OrchestratorResponse
from angel_filter.ranker import QueryIntent


def _request(client_host: str = "10.0.0.1", forwarded: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request({
        "type": "http", "method": "POST", "path": "/query",
        "headers": headers, "client": (client_host, 1234),
    })


# --- client_key ----------------------------------------------------------------

def test_forwarded_header_is_ignored_by_default(monkeypatch):
    """Locally there is no proxy, so X-Forwarded-For is whatever the caller says.

    Trusting it would let anyone send a fresh fake IP per request and never be
    limited.
    """
    monkeypatch.delenv("ANGEL_TRUSTED_PROXY_HOPS", raising=False)
    assert limits.client_key(_request("10.0.0.1", "1.2.3.4")) == "10.0.0.1"


def test_one_trusted_hop_uses_the_entry_the_proxy_added(monkeypatch):
    """The proxy appends the address it saw; anything to its left is client-supplied."""
    monkeypatch.setenv("ANGEL_TRUSTED_PROXY_HOPS", "1")
    req = _request("10.0.0.1", "6.6.6.6, 203.0.113.7")   # 6.6.6.6 is a spoof
    assert limits.client_key(req) == "203.0.113.7"


def test_two_trusted_hops_skip_the_inner_proxy(monkeypatch):
    monkeypatch.setenv("ANGEL_TRUSTED_PROXY_HOPS", "2")
    req = _request("10.0.0.1", "6.6.6.6, 203.0.113.7, 172.16.0.9")
    assert limits.client_key(req) == "203.0.113.7"


def test_short_forwarded_header_falls_back_to_the_socket(monkeypatch):
    monkeypatch.setenv("ANGEL_TRUSTED_PROXY_HOPS", "2")
    assert limits.client_key(_request("10.0.0.1", "203.0.113.7")) == "10.0.0.1"


def test_bad_hops_value_trusts_nothing(monkeypatch):
    monkeypatch.setenv("ANGEL_TRUSTED_PROXY_HOPS", "lots")
    assert limits.client_key(_request("10.0.0.1", "1.2.3.4")) == "10.0.0.1"


# --- Enforcement through the real app --------------------------------------------

class _Spy:
    providers: list = []   # /health lists ORCHESTRATOR.providers

    def __init__(self) -> None:
        self.calls = 0

    async def handle_query(self, user_query, **kwargs):
        self.calls += 1
        return OrchestratorResponse(
            ranked=[], providers_used=["mock"], providers_failed=[],
            intent=QueryIntent.GENERAL, constraints=QueryConstraints(),
        )


@pytest.fixture
def spy(monkeypatch):
    s = _Spy()
    monkeypatch.setattr(server, "ORCHESTRATOR", s)
    monkeypatch.setattr(limits, "USER_RATE_LIMIT", 2)
    monkeypatch.setattr(limits, "DAILY_QUERY_LIMIT", 100)
    return s


def test_query_route_returns_429_past_the_client_limit(spy):
    with TestClient(server.app) as client:
        codes = [client.post("/query", json={"query": f"lunch {i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]
    assert spy.calls == 2, "the rejected request still reached the providers"


def test_rate_limited_requests_do_not_burn_the_daily_budget(spy):
    with TestClient(server.app) as client:
        for i in range(5):
            client.post("/query", json={"query": f"lunch {i}"})
    assert limits.daily_budget_status()["count"] == 2


def test_daily_cap_applies_across_clients(spy, monkeypatch):
    """The backstop holds even when every request comes from a new address."""
    monkeypatch.setattr(limits, "DAILY_QUERY_LIMIT", 2)
    monkeypatch.setenv("ANGEL_TRUSTED_PROXY_HOPS", "1")
    with TestClient(server.app) as client:
        codes = [
            client.post("/query", json={"query": f"q{i}"},
                        headers={"x-forwarded-for": f"203.0.113.{i}"}).status_code
            for i in range(3)
        ]
    assert codes == [200, 200, 429]


# --- The NLIP route ------------------------------------------------------------

nlip_only = pytest.mark.skipif(not server._NLIP_AVAILABLE, reason="NLIP libraries not available")


@nlip_only
def test_nlip_route_is_rate_limited(spy):
    """Also guards the path itself: the middleware matches server._NLIP_PATH
    exactly, so if upstream moved its route these posts would 404 rather than
    pass, instead of the limit silently ceasing to apply."""
    from nlip_sdk.nlip import NLIP_Factory

    with TestClient(server.app) as client:
        codes = [
            client.post(server._NLIP_PATH,
                        json=NLIP_Factory.create_text(f"lunch {i}").model_dump()).status_code
            for i in range(3)
        ]
    assert codes[:2] == [200, 200]
    assert codes[2] == 429
    assert spy.calls == 2, "the rejected NLIP request still reached the providers"


@nlip_only
def test_other_routes_are_not_counted(spy):
    """Only searches spend money; health checks and page loads must not use up the limit."""
    with TestClient(server.app) as client:
        for _ in range(5):
            client.get("/health")
            client.get("/")
        assert client.post("/query", json={"query": "lunch"}).status_code == 200


def test_no_login_redirect():
    """The UI is served directly; there is no /login to bounce to any more."""
    with TestClient(server.app) as client:
        r = client.get("/", follow_redirects=False)
    assert r.status_code == 200
    assert client.get("/login").status_code == 404
