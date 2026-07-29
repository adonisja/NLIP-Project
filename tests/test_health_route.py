"""Tests for /health ownership in NLIP mode.

nlip_server.setup_server() mounts its own health router, and FastAPI resolves a
request against the first route registered for a path. Our richer handler was
declared afterwards, so it never ran: /health returned a bare
{"status": "healthy"} and could not say which providers had loaded — the one
thing you want from it while debugging a demo.

The fix drops the upstream /health entry from our app's route table before
declaring ours. These pin both halves of that: we own /health, and we did not
trample the sibling probes upstream also registers.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nlip_sdk.nlip")
import angel_filter.server as server

pytestmark = pytest.mark.skipif(
    not server._NLIP_AVAILABLE, reason="NLIP libraries not available"
)


def _routes_for(path: str):
    return [r for r in server.app.routes if getattr(r, "path", None) == path]


def test_health_is_registered_exactly_once():
    """Two registrations means the shadowing is back, whichever one wins."""
    assert len(_routes_for("/health")) == 1


def test_health_belongs_to_us_not_upstream():
    (route,) = _routes_for("/health")
    assert route.endpoint.__module__ == "angel_filter.server"


def test_health_returns_the_diagnostic_payload():
    """The point of owning the route: mode, uptime, and the provider list.

    A bare {"status": "healthy"} passes a liveness check but cannot tell you a
    provider silently failed to configure.
    """
    body = server._health_response(mode="nlip", nlip_available=True)
    assert body["ok"] is True
    assert body["mode"] == "nlip"
    assert body["nlip_available"] is True
    assert isinstance(body["providers"], list)
    assert "uptime_seconds" in body


@pytest.mark.parametrize("path", ["/health/live", "/health/ready"])
def test_upstream_probe_routes_survive(path):
    """We reclaimed one path, not the whole health router.

    Deployment platforms may be pointed at these; dropping them would break
    liveness/readiness checks with no error at startup.
    """
    routes = _routes_for(path)
    assert len(routes) == 1
    assert routes[0].endpoint.__module__.startswith("nlip_server")


# --- The helper ----------------------------------------------------------------

def test_drop_route_reports_what_it_removed():
    """The count is the guard: a silent zero would mean nothing was shadowed.

    If upstream renames or moves its health route, _drop_route returns 0 and
    this contract makes that visible rather than letting the override quietly
    become a no-op.
    """
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/thing")
    async def thing():
        return {}

    assert server._drop_route(app, "/thing") == 1
    assert server._drop_route(app, "/thing") == 0
    assert not [r for r in app.routes if getattr(r, "path", None) == "/thing"]


def test_drop_route_leaves_other_paths_alone():
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/keep")
    async def keep():
        return {}

    @app.get("/drop")
    async def drop():
        return {}

    server._drop_route(app, "/drop")
    assert [r for r in app.routes if getattr(r, "path", None) == "/keep"]
