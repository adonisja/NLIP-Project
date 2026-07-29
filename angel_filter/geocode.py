"""Resolve real coordinates for results that named a venue but gave no distance.

Most providers cannot supply distance. The AI providers are forbidden from
guessing it — prompt.py tells the model not to and the adapters hardcode
`distance=None`, because a model with no location context would fabricate the
number. Brave returns web pages, which have no coordinates at all. So before
this, only Google Places populated the P2 axis and every other result was
scored on price and rating alone.

This closes that gap without inventing anything: after fan-out, any result that
named a place gets looked up through the Places Text Search API, and its real
coordinates are haversined against the user's. A result we cannot resolve keeps
`distance=None` and stays honestly unscored on P2 — the ranker already
renormalises the axis weights over whichever axes have data.

Requires GOOGLE_PLACES_API_KEY and a user location. Without either, enrichment
is a no-op and the pipeline behaves exactly as it did before.
"""

from __future__ import annotations

import asyncio
import logging
import os

from angel_filter.providers.base import ProviderResult
from angel_filter.providers.google_places import haversine_miles

logger = logging.getLogger(__name__)

_TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
_TIMEOUT = 10

# Cap concurrent lookups. The fan-out can produce ~40 results across providers
# and firing that many geocode requests at once invites rate limiting for no
# latency benefit — they overlap fine at this width.
_MAX_CONCURRENCY = 8

# Skip titles that clearly are not venues. A web-search result like "The 10
# Best Tacos in Brooklyn" would geocode to *something*, and that something
# would be wrong — a listicle is not a place you can walk to.
_NON_VENUE_MARKERS = (
    "best ", "top ", "guide", "review", "blog", " vs ", "how to",
    "what to", "where to", "near me", "list of", "things to",
)


def looks_like_a_venue(title: str) -> bool:
    """Cheap filter for titles worth geocoding.

    Deliberately conservative: a false negative costs one unscored axis, while a
    false positive puts a plausible-looking but wrong distance into the ranking.
    Given the project's stance that missing data beats invented data, we skip
    when unsure.
    """
    if not title or not title.strip():
        return False
    t = title.strip()
    if len(t) > 80:  # headlines and article titles, not venue names
        return False
    low = t.lower()
    if any(marker in low for marker in _NON_VENUE_MARKERS):
        return False
    if low.startswith(("http://", "https://", "www.")):
        return False
    # A digit-led title is usually a ranked listicle ("5 Great Diners").
    if t[0].isdigit():
        return False
    return True


async def _lookup_one(client, title: str, lat: float, lng: float, api_key: str):
    """Geocode a single title. Returns (lat, lng) or None.

    Never raises: enrichment is best-effort, and one bad lookup must not fail
    the query. The caller keeps distance=None for anything unresolved.
    """
    try:
        resp = await client.get(
            _TEXT_SEARCH_URL,
            params={
                "query": title,
                # Bias toward the user so "Joe's Pizza" resolves to the nearby
                # one rather than a same-named place in another city.
                "location": f"{lat},{lng}",
                "radius": "50000",
                "key": api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Geocode failed for %r: %s", title, exc)
        return None

    if data.get("status") not in ("OK", "ZERO_RESULTS"):
        logger.debug("Geocode status %s for %r", data.get("status"), title)
        return None

    results = data.get("results") or []
    if not results:
        return None
    loc = (results[0].get("geometry") or {}).get("location") or {}
    if not isinstance(loc.get("lat"), (int, float)) or not isinstance(loc.get("lng"), (int, float)):
        return None
    return float(loc["lat"]), float(loc["lng"])


async def enrich_distances(
    results: list[ProviderResult],
    user_lat: float | None,
    user_lng: float | None,
    api_key: str | None = None,
) -> int:
    """Fill in `distance` on results that named a venue but reported none.

    Mutates `results` in place and returns how many were enriched. A no-op
    without an API key or a user location, so the pipeline degrades to its
    previous behaviour rather than erroring.

    Results are deduplicated by title before lookup: providers frequently name
    the same venue, and geocoding it once per mention would multiply the API
    calls for an identical answer.
    """
    api_key = api_key or os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key or user_lat is None or user_lng is None:
        return 0

    pending = [
        r for r in results
        if r.distance is None and looks_like_a_venue(r.title)
    ]
    if not pending:
        return 0

    # One lookup per distinct title, shared across every result naming it.
    by_title: dict[str, list[ProviderResult]] = {}
    for r in pending:
        by_title.setdefault(r.title.strip(), []).append(r)

    import httpx

    sem = asyncio.Semaphore(_MAX_CONCURRENCY)

    async def resolve(client, title: str):
        async with sem:
            return title, await _lookup_one(client, title, user_lat, user_lng, api_key)

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            pairs = await asyncio.gather(
                *(resolve(client, t) for t in by_title),
                return_exceptions=True,
            )
    except Exception as exc:
        logger.warning("Distance enrichment aborted: %s", exc)
        return 0

    enriched = 0
    for pair in pairs:
        # return_exceptions=True: a task that blew up arrives as the exception.
        if isinstance(pair, BaseException) or pair is None:
            continue
        title, coords = pair
        if coords is None:
            continue
        miles = round(haversine_miles(user_lat, user_lng, coords[0], coords[1]), 2)
        for r in by_title[title]:
            r.distance = miles
            enriched += 1

    if enriched:
        logger.info(
            "Distance enrichment: resolved %d/%d titles, filled %d results",
            sum(1 for p in pairs if not isinstance(p, BaseException) and p and p[1]),
            len(by_title), enriched,
        )
    return enriched
