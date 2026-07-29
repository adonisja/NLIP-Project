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

# Places API (New). The legacy maps.googleapis.com/maps/api/place/* endpoints
# return REQUEST_DENIED on any project created after Google retired them
# ("You're calling a legacy API, which is not enabled for your project"), so a
# contributor with a fresh key cannot use them at all.
_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
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


# Words that carry no identifying signal, so they should not be what makes two
# names "agree" — every other venue is a cafe or a restaurant.
_GENERIC_WORDS = frozenset({
    "the", "a", "an", "of", "and", "at", "in", "on", "for", "to",
    "cafe", "café", "coffee", "restaurant", "bar", "grill", "kitchen",
    "house", "shop", "co", "inc", "llc", "nyc", "new", "york",
    "pizza", "pizzeria", "deli", "delicatessen", "bakery", "diner",
    "food", "eatery", "bistro", "tacos", "taco", "sushi", "noodle",
})


def _significant_words(name: str) -> set[str]:
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in name.lower())
    return {w for w in cleaned.split() if w and w not in _GENERIC_WORDS}


def _names_agree(asked: str, matched: str) -> bool:
    """Did Text Search return the place we actually asked for?

    Must tolerate real variation — "Joe's Pizza" legitimately matches "Joe's
    Pizza Broadway" — while rejecting the unrelated best-guess that comes back
    for a name that does not exist.

    The test is on *distinctive* words: generic ones like "cafe" or "pizza" are
    stripped first, so "Fake Cafe" and "Kawaii Coffee" share nothing, while
    "Joe's Pizza" and "Joe's Pizza Broadway" still share "joe's". When the asked
    name is entirely generic there is nothing to verify against, and we accept —
    the venue-shape filter has already screened the obvious junk.
    """
    a, m = _significant_words(asked), _significant_words(matched)
    if not a:
        return True
    return bool(a & m)


_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"

# Cache resolved localities: a user's coordinates barely move between queries,
# and the answer ("Manhattan, NY") is stable. Keyed on coordinates rounded to
# ~110m, the same precision the query cache uses.
_locality_cache: dict[tuple[float, float], str | None] = {}


def _locality_from_address(address: str) -> str | None:
    """Trim a street address down to the part a model should reason about.

    "1464 Atlantic Ave, Brooklyn, NY 11216, USA" -> "Brooklyn, NY". The house
    number and ZIP add nothing to "suggest lunch nearby" and invite the model to
    treat them as a precise target.
    """
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if not parts:
        return None
    # Drop a leading street line (starts with a house number) and a trailing
    # country. What remains is locality plus state/ZIP.
    if parts and parts[0][:1].isdigit():
        parts = parts[1:]
    if parts and parts[-1].upper() in {"USA", "US", "UNITED STATES"}:
        parts = parts[:-1]
    if not parts:
        return None
    # Strip a ZIP off the state component: "NY 11216" -> "NY".
    tail = parts[-1].split()
    if len(tail) > 1 and tail[-1].replace("-", "").isdigit():
        parts[-1] = " ".join(tail[:-1])
    return ", ".join(parts[:2]) or None


async def describe_location(
    lat: float | None,
    lng: float | None,
    api_key: str | None = None,
) -> str | None:
    """Human-readable locality for the user's coordinates, or None.

    Uses the Places API rather than the Geocoding API: they are separate
    products in Google Cloud, and requiring a second one to be enabled would be
    another setup step for every contributor. Nearest-place lookup at a 200m
    radius yields the same answer for this purpose.

    Best-effort — any failure returns None and the caller simply omits location
    from the prompt, which is the pre-existing behaviour.
    """
    api_key = api_key or os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key or lat is None or lng is None:
        return None

    key = (round(lat, 3), round(lng, 3))
    if key in _locality_cache:
        return _locality_cache[key]

    import httpx

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                _NEARBY_URL,
                json={
                    "locationRestriction": {
                        "circle": {
                            "center": {"latitude": lat, "longitude": lng},
                            "radius": 200.0,
                        }
                    },
                    "maxResultCount": 1,
                },
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": api_key,
                    "X-Goog-FieldMask": "places.formattedAddress",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.debug("Reverse lookup failed for (%s, %s): %s", lat, lng, exc)
        _locality_cache[key] = None
        return None

    places = data.get("places") or []
    locality = _locality_from_address(places[0].get("formattedAddress", "")) if places else None
    _locality_cache[key] = locality
    if locality:
        logger.info("User location resolved to %r", locality)
    return locality


async def _lookup_one(client, title: str, lat: float, lng: float, api_key: str):
    """Geocode a single title. Returns (lat, lng) or None.

    Never raises: enrichment is best-effort, and one bad lookup must not fail
    the query. The caller keeps distance=None for anything unresolved.
    """
    try:
        resp = await client.post(
            _TEXT_SEARCH_URL,
            json={
                "textQuery": title,
                # Bias toward the user so "Joe's Pizza" resolves to the nearby
                # one rather than a same-named place in another city.
                "locationBias": {
                    "circle": {
                        "center": {"latitude": lat, "longitude": lng},
                        "radius": 50000.0,
                    }
                },
                "maxResultCount": 1,
            },
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                # The new API bills by which fields you ask for, so request only
                # the location — this is a geocode, not a details lookup.
                # displayName too, not just location: Text Search always returns
                # its best guess, so we must check *what* it matched.
                "X-Goog-FieldMask": "places.location,places.displayName",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Geocode failed for %r: %s", title, exc)
        return None

    # No match is an empty body, not an error status — the new API signals
    # failures with the HTTP code, which raise_for_status already caught.
    places = data.get("places") or []
    if not places:
        return None

    # Text Search never says "not found" — it returns its closest guess. Asking
    # for a venue that does not exist yields a real, unrelated place, and taking
    # its coordinates would attach a plausible distance to a hallucinated name.
    # Observed live: "Totally Fake Nonexistent Cafe XYZQ" matched "Kawaii Coffee".
    # So confirm the match actually resembles what we asked for.
    matched = ((places[0].get("displayName") or {}).get("text") or "")
    if not _names_agree(title, matched):
        logger.debug("Geocode rejected: asked %r, got %r", title, matched)
        return None

    loc = places[0].get("location") or {}
    lat_v, lng_v = loc.get("latitude"), loc.get("longitude")
    if not isinstance(lat_v, (int, float)) or not isinstance(lng_v, (int, float)):
        return None
    return float(lat_v), float(lng_v)


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
