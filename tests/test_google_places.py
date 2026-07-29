"""Tests for the Google Places provider — the only source of real P2 distance.

No live network: a fake httpx transport returns canned Nearby Search payloads,
so these are deterministic and key-free (CLAUDE.md: tests must not require
network or a real API). Two things get exercised:

  - the provider's contract (needs a key, needs user coordinates, maps the
    Places response into ProviderResult with a computed distance)
  - the haversine helper against known distances

The provider constructs its own httpx.AsyncClient, so we monkeypatch
httpx.AsyncClient to bind a MockTransport that serves our canned response.
"""

from __future__ import annotations

import functools
import json

import httpx
import pytest

from angel_filter.constraints import QueryConstraints
from angel_filter.providers.base import ProviderError
from angel_filter.providers.google_places import (
    GooglePlacesProvider,
    haversine_miles,
)


# --- Canned Places responses --------------------------------------------------

# User origin used across the distance tests.
_USER_LAT, _USER_LNG = 40.7680, -73.9819  # Columbus Circle, NYC

# Shapes below are Places API (New) — places.googleapis.com/v1/places:searchText.
# The legacy maps.googleapis.com endpoints return REQUEST_DENIED on any project
# created after Google retired them, so the provider had to migrate. Field names
# differ throughout: "places" not "results", displayName.text not name,
# location.latitude not geometry.location.lat, and priceLevel is an enum string
# rather than a 0-4 integer. Verified against the live API before canning.
_NEARBY_OK = {
    "places": [
        {
            "displayName": {"text": "Joe's Pizza"},
            "formattedAddress": "7 Carmine St",
            "rating": 4.7,
            "priceLevel": "PRICE_LEVEL_INEXPENSIVE",
            "location": {"latitude": 40.7305, "longitude": -74.0026},
        },
        {
            "displayName": {"text": "Corner Bistro"},
            "formattedAddress": "331 W 4th St",
            "rating": 4.2,
            "priceLevel": "PRICE_LEVEL_EXPENSIVE",
            # ~same block as the user -> should be the closest
            "location": {"latitude": 40.7682, "longitude": -73.9820},
        },
        {
            # No priceLevel, no rating, no location -> graceful degradation
            "displayName": {"text": "Mystery Spot"},
            "formattedAddress": "Somewhere",
        },
    ],
}

# A miss is an empty body, not a status string.
_ZERO_RESULTS: dict = {}


def _patch_transport(monkeypatch, payload: dict, *, capture: dict | None = None):
    """Make the provider's internal httpx.AsyncClient serve `payload`.

    If `capture` is given, the outgoing request is recorded into it so tests can
    assert on what the provider sent. The new API is a POST with a JSON body and
    an API key in a header, so we capture those rather than query params.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["url"] = str(request.url)
            capture["headers"] = dict(request.headers)
            try:
                capture["body"] = json.loads(request.content or b"{}")
            except ValueError:
                capture["body"] = {}
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs.pop("timeout", None)  # MockTransport ignores it anyway
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)


def _provider() -> GooglePlacesProvider:
    return GooglePlacesProvider(api_key="test-key")


# --- Contract -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_api_key_raises(monkeypatch):
    # api_key=None falls back to os.getenv, so clear the env to stay
    # deterministic on machines where the key happens to be set.
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    p = GooglePlacesProvider(api_key=None)
    with pytest.raises(ProviderError, match="GOOGLE_PLACES_API_KEY"):
        await p.query("pizza", constraints=QueryConstraints(user_lat=1.0, user_lng=2.0))


@pytest.mark.asyncio
async def test_missing_user_location_raises():
    """No origin means no distance to compute — the provider must skip itself."""
    p = _provider()
    with pytest.raises(ProviderError, match="no user location"):
        await p.query("pizza", constraints=QueryConstraints())  # no lat/lng


@pytest.mark.asyncio
async def test_missing_user_location_with_no_constraints_raises():
    p = _provider()
    with pytest.raises(ProviderError, match="no user location"):
        await p.query("pizza", constraints=None)


# --- Parsing + distance -------------------------------------------------------

@pytest.mark.asyncio
async def test_parses_places_into_results_with_distance(monkeypatch):
    _patch_transport(monkeypatch, _NEARBY_OK)
    p = _provider()

    results = await p.query(
        "pizza",
        constraints=QueryConstraints(user_lat=_USER_LAT, user_lng=_USER_LNG),
    )

    assert [r.title for r in results] == ["Joe's Pizza", "Corner Bistro", "Mystery Spot"]
    assert all(r.provider == "google_places" for r in results)

    joes, bistro, mystery = results
    # Corner Bistro is essentially at the user's location; Joe's is ~2.6 mi away.
    assert bistro.distance is not None and bistro.distance < 0.1
    assert joes.distance is not None and joes.distance > 2.0
    # price_level 1 -> $10, level 3 -> $50; ratings pass through.
    assert joes.price == 10.0 and joes.rating == 4.7
    assert bistro.price == 50.0 and bistro.rating == 4.2
    # Mystery Spot has no geometry/price/rating -> all None, no crash.
    assert mystery.distance is None
    assert mystery.price is None
    assert mystery.rating is None


@pytest.mark.asyncio
async def test_sends_user_location_and_radius(monkeypatch):
    """max_distance (miles) must become the Google radius (meters), capped 50km."""
    capture: dict = {}
    _patch_transport(monkeypatch, _NEARBY_OK, capture=capture)
    p = _provider()

    await p.query(
        "pizza",
        constraints=QueryConstraints(
            user_lat=_USER_LAT, user_lng=_USER_LNG, max_distance=2.0
        ),
    )

    circle = capture["body"]["locationBias"]["circle"]
    assert circle["center"] == {"latitude": _USER_LAT, "longitude": _USER_LNG}
    # 2 miles ≈ 3218 meters
    assert circle["radius"] == float(int(2.0 * 1609.34))
    # searchNearby has no free-text field, so the query would be silently
    # dropped there. searchText carries it — pin that it actually does.
    assert capture["body"]["textQuery"] == "pizza"
    # The new API authenticates by header, not a `key` query param.
    assert capture["headers"]["x-goog-api-key"] == "test-key"
    assert "places.location" in capture["headers"]["x-goog-fieldmask"]


@pytest.mark.asyncio
async def test_radius_capped_at_50km(monkeypatch):
    capture: dict = {}
    _patch_transport(monkeypatch, _NEARBY_OK, capture=capture)
    p = _provider()

    await p.query(
        "pizza",
        constraints=QueryConstraints(
            user_lat=_USER_LAT, user_lng=_USER_LNG, max_distance=100.0  # 160km
        ),
    )
    assert capture["body"]["locationBias"]["circle"]["radius"] == 50000.0


@pytest.mark.asyncio
async def test_null_location_degrades_to_no_distance(monkeypatch):
    """An explicit "location": null must not crash the whole provider.

    .get("location", {}) only defaults on a missing key; a present-but-null
    location would raise AttributeError without the `or {}` guard. Such a result
    should degrade to distance=None, not take down the batch.
    """
    payload = {
        "places": [
            {"displayName": {"text": "Null Loc"}, "location": None, "rating": 4.0},
            {"displayName": {"text": "No Loc Key"}},
            {"displayName": {"text": "Good"},
             "location": {"latitude": 40.7682, "longitude": -73.9820}},
        ],
    }
    _patch_transport(monkeypatch, payload)
    results = await _provider().query(
        "pizza", constraints=QueryConstraints(user_lat=_USER_LAT, user_lng=_USER_LNG)
    )
    by_title = {r.title: r for r in results}
    assert by_title["Null Loc"].distance is None
    assert by_title["No Loc Key"].distance is None
    assert by_title["Good"].distance is not None  # the valid one still computes


@pytest.mark.asyncio
async def test_zero_results_returns_empty_not_error(monkeypatch):
    _patch_transport(monkeypatch, _ZERO_RESULTS)
    p = _provider()
    results = await p.query(
        "pizza", constraints=QueryConstraints(user_lat=_USER_LAT, user_lng=_USER_LNG)
    )
    assert results == []


@pytest.mark.asyncio
async def test_http_error_becomes_a_provider_error(monkeypatch):
    """The new API signals failures with HTTP codes, not a status field.

    A bad key returns 403 rather than a 200 carrying REQUEST_DENIED, so the
    provider must surface it as ProviderError for the orchestrator to isolate.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "bad key"}})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)

    with pytest.raises(ProviderError, match="Google Places request failed"):
        await _provider().query(
            "pizza", constraints=QueryConstraints(user_lat=_USER_LAT, user_lng=_USER_LNG)
        )


# --- Haversine ----------------------------------------------------------------

def test_haversine_known_distances():
    # Same point.
    assert haversine_miles(40.0, -73.0, 40.0, -73.0) == pytest.approx(0.0, abs=1e-6)
    # One degree of latitude ≈ 69 miles.
    assert haversine_miles(40.0, -73.0, 41.0, -73.0) == pytest.approx(69.0, abs=0.5)
    # Columbus Circle -> Times Square ≈ 0.7 mi.
    assert haversine_miles(40.7681, -73.9819, 40.7580, -73.9855) == pytest.approx(0.72, abs=0.1)


def test_haversine_is_symmetric():
    a = haversine_miles(40.7, -73.9, 34.0, -118.2)
    b = haversine_miles(34.0, -118.2, 40.7, -73.9)
    assert a == pytest.approx(b, abs=1e-9)
