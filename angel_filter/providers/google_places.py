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

_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
_TIMEOUT = 15
_EARTH_RADIUS_MI = 3958.7613  # mean earth radius in miles

# Places returns price_level as an integer 0-4. Map it to a rough per-person
# dollar figure so the P1 price axis has something to score. These are
# deliberately coarse — Places does not expose actual prices.
_PRICE_LEVEL_TO_USD: dict[int, float] = {0: 0.0, 1: 10.0, 2: 25.0, 3: 50.0, 4: 100.0}


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

        params = {
            "location": f"{c.user_lat},{c.user_lng}",
            "radius": str(radius_m),
            "keyword": user_query,
            "key": self._api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_NEARBY_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise ProviderError(f"Google Places request failed: {exc}") from exc

        # Places encodes API-level failures in a status field, not the HTTP code.
        status = data.get("status", "UNKNOWN")
        if status not in ("OK", "ZERO_RESULTS"):
            raise ProviderError(
                f"Google Places returned status {status}: {data.get('error_message', '')}"
            )

        return _parse_results(data, c.user_lat, c.user_lng, max_results)


def _parse_results(
    data: dict[str, Any],
    user_lat: float,
    user_lng: float,
    max_results: int,
) -> list[ProviderResult]:
    results: list[ProviderResult] = []
    for i, place in enumerate(data.get("results", [])[:max_results]):
        name = str(place.get("name", "")).strip()
        if not name:
            continue

        # Guard against explicit nulls: the API may send "geometry": null or
        # "location": null, and .get(k, {}) only defaults on a *missing* key,
        # not a present-but-null one. `or {}` degrades those to no coords
        # (distance=None) instead of crashing the whole provider.
        loc = (place.get("geometry") or {}).get("location") or {}
        lat, lng = loc.get("lat"), loc.get("lng")
        distance = (
            round(haversine_miles(user_lat, user_lng, lat, lng), 2)
            if lat is not None and lng is not None
            else None
        )

        price_level = place.get("price_level")
        price = _PRICE_LEVEL_TO_USD.get(price_level) if price_level is not None else None

        rating = place.get("rating")
        rating = float(rating) if isinstance(rating, (int, float)) else None

        vicinity = str(place.get("vicinity", "")).strip()

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
