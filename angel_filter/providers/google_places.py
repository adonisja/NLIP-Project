"""Google Places provider — the only source of real distance data.

Unlike the AI providers (OpenAI, Gemini, Ollama, WatsonX), which have no
location context and are deliberately forbidden from inventing a distance
(see prompt.py), this provider returns actual venues with coordinates. It
computes each venue's distance from the user's supplied origin so the ranker's
P2 axis has real data to score instead of a neutral 0.5 placeholder.

Requires GOOGLE_PLACES_API_KEY. The user's origin arrives on
QueryConstraints.user_lat / user_lng (from the request, not the query text) —
without it there is nothing to measure distance from, so the provider raises
ProviderError and the orchestrator drops it from the results, exactly as it
would for any other provider that can't fulfil a request.

Uses the Places Nearby Search endpoint:
  https://developers.google.com/maps/documentation/places/web-service/search-nearby

Note for the PR: this calls a Google API under its Terms of Service and billing
applies. It is an official API (not scraping), keyed per deployment.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

from angel_filter.constraints import QueryConstraints
from angel_filter.providers.base import BaseProvider, ProviderError, ProviderResult

logger = logging.getLogger(__name__)

# Places API (New). The legacy maps.googleapis.com/maps/api/place/* endpoints
# return REQUEST_DENIED on any project created after Google retired them, so a
# contributor with a fresh key could not use this provider at all.
# searchText, not searchNearby: the new API's nearby endpoint filters by place
# *type* and has no free-text field, so it would silently drop the user's query
# and return generic nearby venues. searchText keeps the query, which is what
# the legacy `keyword` parameter did.
_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
_TIMEOUT = 15
_EARTH_RADIUS_MI = 3958.7613  # mean earth radius in miles

# The new API returns priceLevel as an enum string rather than the legacy 0-4
# integer. Map it to a rough per-person dollar figure so the P1 price axis has
# something to score — deliberately coarse, since Places exposes no real prices.
_PRICE_LEVEL_TO_USD: dict[str, float] = {
    "PRICE_LEVEL_FREE": 0.0,
    "PRICE_LEVEL_INEXPENSIVE": 10.0,
    "PRICE_LEVEL_MODERATE": 25.0,
    "PRICE_LEVEL_EXPENSIVE": 50.0,
    "PRICE_LEVEL_VERY_EXPENSIVE": 100.0,
}


def haversine_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two lat/lng points, in miles."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * _EARTH_RADIUS_MI * math.asin(math.sqrt(a))


class GooglePlacesProvider(BaseProvider):
    name = "google_places"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("GOOGLE_PLACES_API_KEY")

    async def query(
        self,
        user_query: str,
        max_results: int = 10,
        constraints: QueryConstraints | None = None,
    ) -> list[ProviderResult]:
        import httpx

        if not self._api_key:
            raise ProviderError("GOOGLE_PLACES_API_KEY is not set")

        c = constraints or QueryConstraints()
        if c.user_lat is None or c.user_lng is None:
            # No origin means no distance to compute. Skip rather than return
            # location-less results that would just duplicate the AI providers.
            raise ProviderError("no user location supplied; skipping location provider")

        # Radius in meters. Prefer the user's max_distance if they set one,
        # capped to Google's 50km limit; otherwise a sensible walking-ish default.
        if c.max_distance is not None:
            radius_m = min(int(c.max_distance * 1609.34), 50_000)
        else:
            radius_m = 5_000

        body = {
            "textQuery": user_query,
            # locationBias, not locationRestriction: searchText's restriction
            # field accepts only a rectangle, and sending a circle there is a
            # 400 ("Unknown name 'circle' at 'location_restriction'"). Bias takes
            # the circle and is the right shape for "near me" anyway — results
            # outside the radius are demoted rather than excluded, and the P2
            # axis already scores distance properly.
            "locationBias": {
                "circle": {
                    "center": {"latitude": c.user_lat, "longitude": c.user_lng},
                    "radius": float(radius_m),
                }
            },
            "maxResultCount": min(max_results, 20),
        }
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._api_key,
            # The new API bills by requested fields, so ask only for what the
            # three axes need.
            "X-Goog-FieldMask": (
                "places.displayName,places.location,places.rating,"
                "places.priceLevel,places.formattedAddress"
            ),
        }

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(_SEARCH_URL, json=body, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            # Unlike the legacy API, failures come back as HTTP status codes,
            # which raise_for_status turns into the exception caught here.
            raise ProviderError(f"Google Places request failed: {exc}") from exc

        return _parse_results(data, c.user_lat, c.user_lng, max_results)


def _parse_results(
    data: dict[str, Any],
    user_lat: float,
    user_lng: float,
    max_results: int,
) -> list[ProviderResult]:
    results: list[ProviderResult] = []
    # Places API (New) returns "places"; the legacy API returned "results".
    for i, place in enumerate((data.get("places") or [])[:max_results]):
        name = str(((place.get("displayName") or {}).get("text") or "")).strip()
        if not name:
            continue

        # Guard against explicit nulls: the API may send "location": null, and
        # .get(k, {}) only defaults on a *missing* key, not a present-but-null
        # one. `or {}` degrades those to no coords (distance=None) instead of
        # crashing the whole provider.
        loc = place.get("location") or {}
        lat, lng = loc.get("latitude"), loc.get("longitude")
        distance = (
            round(haversine_miles(user_lat, user_lng, lat, lng), 2)
            if lat is not None and lng is not None
            else None
        )

        # priceLevel is now an enum string ("PRICE_LEVEL_MODERATE"), not 0-4.
        price = _PRICE_LEVEL_TO_USD.get(place.get("priceLevel"))

        rating = place.get("rating")
        rating = float(rating) if isinstance(rating, (int, float)) else None

        vicinity = str(place.get("formattedAddress", "")).strip()

        results.append(ProviderResult(
            title=name,
            snippet=vicinity or name,
            url=None,
            provider="google_places",
            rank_in_provider=i,
            price=price,
            distance=distance,
            rating=rating,
            sponsored=False,
            raw=place,
        ))
    return results
