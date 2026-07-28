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

_NEARBY_OK = {
    "status": "OK",
    "results": [
        {
            "name": "Joe's Pizza",
            "vicinity": "7 Carmine St",
            "rating": 4.7,
            "price_level": 1,
            "geometry": {"location": {"lat": 40.7305, "lng": -74.0026}},
        },
        {
            "name": "Corner Bistro",
            "vicinity": "331 W 4th St",
            "rating": 4.2,
            "price_level": 3,
            # ~same block as the user -> should be the closest
            "geometry": {"location": {"lat": 40.7682, "lng": -73.9820}},
        },
        {
            # No price_level, no rating, no geometry -> graceful degradation
            "name": "Mystery Spot",
            "vicinity": "Somewhere",
        },
    ],
}

_ZERO_RESULTS = {"status": "ZERO_RESULTS", "results": []}
_REQUEST_DENIED = {"status": "REQUEST_DENIED", "error_message": "bad key"}


def _patch_transport(monkeypatch, payload: dict, *, capture: dict | None = None):
    """Make the provider's internal httpx.AsyncClient serve `payload`.

    If `capture` is given, the outgoing request's query params are recorded
    into it so tests can assert on what the provider sent.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["url"] = str(request.url)
            capture["params"] = dict(request.url.params)
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
async def test_missing_api_key_raises():
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

    assert capture["params"]["location"] == f"{_USER_LAT},{_USER_LNG}"
    # 2 miles ≈ 3218 meters
    assert capture["params"]["radius"] == str(int(2.0 * 1609.34))
    assert capture["params"]["keyword"] == "pizza"


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
    assert capture["params"]["radius"] == "50000"


@pytest.mark.asyncio
async def test_zero_results_returns_empty_not_error(monkeypatch):
    _patch_transport(monkeypatch, _ZERO_RESULTS)
    p = _provider()
    results = await p.query(
        "pizza", constraints=QueryConstraints(user_lat=_USER_LAT, user_lng=_USER_LNG)
    )
    assert results == []


@pytest.mark.asyncio
async def test_api_status_error_raises(monkeypatch):
    """Places encodes failures in a status field, not the HTTP code."""
    _patch_transport(monkeypatch, _REQUEST_DENIED)
    p = _provider()
    with pytest.raises(ProviderError, match="REQUEST_DENIED"):
        await p.query(
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
