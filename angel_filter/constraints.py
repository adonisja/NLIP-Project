"""Constraint extraction — parse explicit user constraints from a query string.

Turns natural language like "lunch under $15, within 5 miles, at least 4 stars"
into structured values the ranker can use to compute real axis gaps instead of
neutral proxies.

Extracted values feed directly into the P1/P2/P3 gap calculation:
    price_gap    = candidate.price    - budget          (negative = under budget = good)
    distance_gap = candidate.distance - max_distance    (negative = closer = good)
    rating_gap   = min_rating         - candidate.rating (negative = meets threshold = good)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QueryConstraints:
    """Structured constraints parsed from the user's query.

    None means the user didn't specify that constraint — the ranker will
    use neutral scoring for that axis rather than a real gap.

    user_lat / user_lng are the origin a location-aware provider (e.g. Google
    Places) measures distance from. They come from the request, not from
    parsing the query text — the user's coordinates aren't something you can
    extract from "lunch nearby". Providers with no notion of location ignore
    them; providers that need them and don't get them return no distance.
    """
    budget: float | None = None        # P1 — maximum price the user will pay
    max_distance: float | None = None  # P2 — maximum distance in miles
    min_rating: float | None = None    # P3 — minimum acceptable star rating
    user_lat: float | None = None      # request-supplied origin latitude
    user_lng: float | None = None      # request-supplied origin longitude
    # Human-readable form of the above ("Manhattan, NY"), resolved once per
    # query and injected into AI provider prompts. Models reason about named
    # areas, not decimal coordinates — without it they suggest venues from
    # anywhere. Never parsed from the query text; derived from the coordinates.
    user_locality: str | None = None
    # Prior conversation turns, prepended to AI provider prompts. Only set for
    # a multi-turn follow-up the deterministic refinement parser did not match,
    # so the common case costs no extra prompt tokens.
    context_prefix: str = ""


# --- Spelled-out numbers ------------------------------------------------------
# "under twenty bucks", "four and a half stars", "half a mile". These are
# rewritten to digits before the patterns below run, so each pattern only has
# to understand one way of writing a number.

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_UNIT_WORDS = "|".join(sorted(_UNITS, key=len, reverse=True))
_TEN_WORDS = "|".join(_TENS)

_NUMBER_WORDS_RE = re.compile(
    rf"""
    (?:
    \b(?:(?P<a>a|one)\s+hundred(?:\s+and)?\s*)?          # "a hundred (and)"
    (?:
        (?P<tens>{_TEN_WORDS})(?:[\s-]+(?P<unit_after_ten>{_UNIT_WORDS}))?
      | (?P<unit>{_UNIT_WORDS})
    )
    (?P<half>\s+and\s+a\s+half)?\b
    | \b(?P<hundred_only>a|one)\s+hundred\b
    )
    # Only a number that a unit follows. "lunch for at least two people" must
    # not become a two-star minimum through the rating pattern's "at least N".
    (?=\s*-?\s*(?:dollars?|bucks?|miles?|mi\b|km\b|kilometers?|\+?\s*stars?))
    """,
    re.IGNORECASE | re.VERBOSE,
)

# "half a mile" / "half mile" carry no number word for the pattern above.
_HALF_MILE_RE = re.compile(r"\bhalf\s+(?:a\s+)?mile\b", re.IGNORECASE)


def _number_words_to_digits(text: str) -> str:
    def convert(m: re.Match) -> str:
        if m.group("hundred_only"):
            return "100"
        value = 100 if m.group("a") else 0
        if m.group("tens"):
            value += _TENS[m.group("tens").lower()]
            if m.group("unit_after_ten"):
                value += _UNITS[m.group("unit_after_ten").lower()]
        else:
            value += _UNITS[m.group("unit").lower()]
        number = f"{value}.5" if m.group("half") else str(value)
        return number

    text = _HALF_MILE_RE.sub("0.5 miles", text)
    return _NUMBER_WORDS_RE.sub(convert, text)


# --- Regex patterns -----------------------------------------------------------

# Price: "$15", "under $20", "budget $10", "15 dollars", "less than $30"
_PRICE_RE = re.compile(
    r"""
    (?:under|below|budget|max|less\s+than|within|around|about|up\s+to)?\s*
    \$\s*(\d+(?:\.\d+)?)
    |
    (\d+(?:\.\d+)?)\s*(?:dollars?|bucks?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Distance: "within 5 miles", "5 mile radius", "under 2 km", "nearby" (→ 1 mi default)
_DISTANCE_RE = re.compile(
    r"""
    (?:within|under|less\s+than|in\s+a|in)?\s*
    (\d+(?:\.\d+)?)\s*(?:mile(?:s)?|mi|km|kilometer(?:s)?)
    |
    \b(nearby|walking\s+distance|close\s+by)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Rating: "4 stars", "at least 4.5 stars", "rated above 3", "4+ stars"
# The number MUST be followed by a star keyword OR preceded by an explicit
# rating prefix — bare numbers are not captured to avoid false positives from
# price/distance figures (e.g. "0.5 miles" must not become a rating).
_RATING_RE = re.compile(
    r"""
    (?:at\s+least|above|over|minimum|min|rated\s+(?:at\s+least|above|over)?)
    \s*(\d+(?:\.\d+)?)
    |
    (\d+(?:\.\d+)?)\s*\+?\s*stars?
    |
    (\d+(?:\.\d+)?)\s+star\s+rating
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_constraints(query: str) -> QueryConstraints:
    """Parse a raw user query and return whatever constraints can be found.

    Unrecognised constraints are left as None — the ranker handles None
    gracefully by falling back to neutral axis scoring.
    """
    query        = _number_words_to_digits(query)
    budget       = _extract_price(query)
    max_distance = _extract_distance(query)
    min_rating   = _extract_rating(query)
    return QueryConstraints(
        budget=budget,
        max_distance=max_distance,
        min_rating=min_rating,
    )


def _extract_price(text: str) -> float | None:
    m = _PRICE_RE.search(text)
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _extract_distance(text: str) -> float | None:
    m = _DISTANCE_RE.search(text)
    if not m:
        return None
    # "nearby" / "walking distance" → treat as 1 mile
    if m.group(2):
        return 1.0
    raw = m.group(1)
    try:
        val = float(raw)
        # Rough km → miles conversion if the unit was km
        unit = m.group(0).lower()
        if "km" in unit or "kilo" in unit:
            val = val * 0.621371
        return round(val, 2)
    except (TypeError, ValueError):
        return None


def _extract_rating(text: str) -> float | None:
    m = _RATING_RE.search(text)
    if not m:
        return None
    raw = m.group(1) or m.group(2) or m.group(3)
    try:
        val = float(raw)
        if 0.0 <= val <= 5.0:
            return val
        return None
    except (TypeError, ValueError):
        return None
