"""Tests for post-hoc distance enrichment.

Only Google Places could populate the P2 axis: the AI providers are forbidden
from reporting distance (prompt.py tells the model not to, the adapters hardcode
None) because a model with no location context would fabricate it, and Brave
returns web pages that have no coordinates. Every other result was therefore
scored on price and rating alone.

enrich_distances resolves real coordinates for results that named a venue and
measures them. The contract these pin is that it never invents: an unresolvable
title keeps distance=None and stays honestly unscored, which is the same policy
the ranker's axis renormalisation already implements.

No network — a fake httpx client returns canned Places payloads.
"""

from __future__ import annotations

import asyncio

import pytest

from angel_filter.geocode import _names_agree, enrich_distances, looks_like_a_venue
from angel_filter.providers.base import ProviderResult

# Times Square-ish, and a point about 1.15 miles north.
USER_LAT, USER_LNG = 40.7580, -73.9855
NEAR_LAT, NEAR_LNG = 40.7736, -73.9866


def _result(title, provider="openai", distance=None):
    return ProviderResult(
        title=title, snippet="s", provider=provider, distance=distance,
    )


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    """Stands in for httpx.AsyncClient; serves canned Places API (New) payloads.

    Shape mirrors places.googleapis.com/v1/places:searchText — a POST with a JSON
    body, and a response whose misses are an empty `places` list rather than a
    status string. Verified against the live API before being canned here.
    """

    def __init__(self, by_query=None, fail_on=()):
        self.by_query = by_query or {}
        self.fail_on = set(fail_on)
        self.calls: list[str] = []
        self.headers_seen: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        q = (json or {}).get("textQuery", "")
        self.calls.append(q)
        self.headers_seen.append(headers or {})
        if q in self.fail_on:
            raise RuntimeError("boom")
        entry = self.by_query.get(q)
        if entry is None:
            return _FakeResponse({})  # no match: empty body
        # An entry may carry the name Google matched, which differs from the
        # query whenever Text Search falls back to its best guess.
        coords, matched_name = (entry, q) if len(entry) == 2 else (entry[:2], entry[2])
        return _FakeResponse({
            "places": [{
                "location": {"latitude": coords[0], "longitude": coords[1]},
                "displayName": {"text": matched_name},
            }]
        })


@pytest.fixture
def fake_httpx(monkeypatch):
    """Patch the httpx module geocode imports lazily inside the function."""
    import httpx

    holder = {}

    def install(client):
        holder["client"] = client
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)
        return client

    return install


# --- The venue filter ----------------------------------------------------------

@pytest.mark.parametrize("title", [
    "Joe's Pizza", "The Green Bowl", "Taco Haven", "Pho Saigon Noodle House",
])
def test_venue_names_are_geocodable(title):
    assert looks_like_a_venue(title) is True


@pytest.mark.parametrize("title", [
    "The 10 Best Tacos in Brooklyn",   # listicle
    "5 Great Diners Near You",         # digit-led
    "Best pizza near me",              # search-phrase
    "How to make ramen",               # article
    "Top rated lunch spots guide",     # guide
    "https://example.com/food",        # bare URL
    "",                                # empty
    "   ",                             # whitespace
    "A" * 120,                         # headline-length
])
def test_non_venues_are_skipped(title):
    """A listicle would geocode to *something*, and that something is wrong.

    Skipping costs one unscored axis; guessing puts a plausible but false
    distance into the ranking. The project's stance is that missing beats
    invented, so the filter errs toward skipping.
    """
    assert looks_like_a_venue(title) is False


# --- Enrichment ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolves_and_measures_distance(fake_httpx):
    results = [_result("Joe's Pizza")]
    fake_httpx(_FakeClient({"Joe's Pizza": (NEAR_LAT, NEAR_LNG)}))

    n = await enrich_distances(results, USER_LAT, USER_LNG, api_key="k")

    assert n == 1
    assert results[0].distance == pytest.approx(1.08, abs=0.15)


@pytest.mark.asyncio
async def test_unresolvable_title_stays_none(fake_httpx):
    """The honesty guarantee: no coordinates means no distance, not a guess."""
    results = [_result("Nonexistent Diner")]
    fake_httpx(_FakeClient({}))  # ZERO_RESULTS

    n = await enrich_distances(results, USER_LAT, USER_LNG, api_key="k")

    assert n == 0
    assert results[0].distance is None


@pytest.mark.asyncio
async def test_existing_distance_is_never_overwritten(fake_httpx):
    """Google Places already measured this one — do not re-resolve it."""
    results = [_result("Joe's Pizza", provider="google_places", distance=0.3)]
    client = fake_httpx(_FakeClient({"Joe's Pizza": (NEAR_LAT, NEAR_LNG)}))

    await enrich_distances(results, USER_LAT, USER_LNG, api_key="k")

    assert results[0].distance == 0.3
    assert client.calls == [], "re-geocoded a result that already had distance"


@pytest.mark.asyncio
async def test_duplicate_titles_cost_one_lookup(fake_httpx):
    """Providers name the same venue constantly; geocoding each mention is waste."""
    results = [
        _result("Joe's Pizza", provider="openai"),
        _result("Joe's Pizza", provider="gemini"),
        _result("Joe's Pizza", provider="ollama"),
    ]
    client = fake_httpx(_FakeClient({"Joe's Pizza": (NEAR_LAT, NEAR_LNG)}))

    n = await enrich_distances(results, USER_LAT, USER_LNG, api_key="k")

    assert client.calls == ["Joe's Pizza"], f"expected 1 lookup, got {client.calls}"
    assert n == 3, "all three mentions should receive the resolved distance"
    assert {r.distance for r in results} == {results[0].distance}


@pytest.mark.asyncio
async def test_a_failing_lookup_does_not_sink_the_others(fake_httpx):
    """One bad geocode must not cost the whole batch its distances."""
    results = [_result("Bad Place"), _result("Good Place")]
    fake_httpx(_FakeClient({"Good Place": (NEAR_LAT, NEAR_LNG)}, fail_on={"Bad Place"}))

    n = await enrich_distances(results, USER_LAT, USER_LNG, api_key="k")

    assert n == 1
    assert results[0].distance is None
    assert results[1].distance is not None


@pytest.mark.asyncio
async def test_listicles_are_not_geocoded(fake_httpx):
    """The filter runs before the API call, so junk titles cost nothing."""
    results = [_result("The 10 Best Tacos in Brooklyn")]
    client = fake_httpx(_FakeClient({}))

    n = await enrich_distances(results, USER_LAT, USER_LNG, api_key="k")

    assert n == 0
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("lat,lng,key", [
    (None, None, "k"),            # no location
    (USER_LAT, USER_LNG, None),   # no API key
    (USER_LAT, None, "k"),        # half a location
])
async def test_noop_without_prerequisites(lat, lng, key, monkeypatch):
    """Degrades to the pre-enrichment pipeline rather than erroring.

    This is also what keeps the test suite offline: with no key configured,
    enrichment never reaches the network.
    """
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    results = [_result("Joe's Pizza")]

    n = await enrich_distances(results, lat, lng, api_key=key)

    assert n == 0
    assert results[0].distance is None


# --- Rejecting Google's best-guess fallback ------------------------------------
# Text Search never reports "not found" — it returns the closest thing it has.
# Asking for a venue that does not exist yields a real but unrelated place, and
# taking its coordinates would attach a plausible distance to a hallucinated
# name. Observed against the live API: "Totally Fake Nonexistent Cafe XYZQ"
# matched "Kawaii Coffee - Vietnamese Coffee". AI providers do invent venue
# names, so this is the failure mode that would actually bite.

@pytest.mark.parametrize("asked,matched", [
    ("Joe's Pizza", "Joe's Pizza Broadway"),          # real suffix variation
    ("Katz's Delicatessen", "Katz's Delicatessen"),   # exact
    ("Los Tacos No. 1", "Los Tacos No 1"),            # punctuation drift
])
def test_name_agreement_accepts_real_variation(asked, matched):
    assert _names_agree(asked, matched) is True


@pytest.mark.parametrize("asked,matched", [
    ("Totally Fake Nonexistent Cafe XYZQ", "Kawaii Coffee - Vietnamese Coffee"),
    ("Imaginary Ramen Palace", "Blackstone Coffee Roaster"),
    ("Zzzq Bogus Diner", "Katz's Delicatessen"),
])
def test_name_agreement_rejects_unrelated_matches(asked, matched):
    assert _names_agree(asked, matched) is False


def test_generic_words_alone_do_not_constitute_agreement():
    """Two unrelated venues both being a "cafe" must not count as a match.

    Without stripping generic words, "Fake Cafe" and "Kawaii Cafe" would agree
    on "cafe" and the fabricated name would acquire real coordinates.
    """
    assert _names_agree("Fake Cafe", "Kawaii Cafe") is False


@pytest.mark.asyncio
async def test_hallucinated_venue_stays_unscored(fake_httpx):
    """End to end: a name Google best-guesses to something else gets no distance."""
    results = [_result("Totally Fake Nonexistent Cafe XYZQ")]
    fake_httpx(_FakeClient({
        # Google returns real coordinates — for an entirely different place.
        "Totally Fake Nonexistent Cafe XYZQ": (NEAR_LAT, NEAR_LNG, "Kawaii Coffee"),
    }))

    n = await enrich_distances(results, USER_LAT, USER_LNG, api_key="k")

    assert n == 0
    assert results[0].distance is None


@pytest.mark.asyncio
async def test_real_venue_with_suffix_still_resolves(fake_httpx):
    """The check must not be so strict that legitimate matches are lost."""
    results = [_result("Joe's Pizza")]
    fake_httpx(_FakeClient({"Joe's Pizza": (NEAR_LAT, NEAR_LNG, "Joe's Pizza Broadway")}))

    n = await enrich_distances(results, USER_LAT, USER_LNG, api_key="k")

    assert n == 1
    assert results[0].distance is not None


@pytest.mark.asyncio
async def test_enrichment_feeds_the_p2_axis(fake_httpx):
    """End of the wire: an enriched result is scored on distance.

    Before this the axis mask reported P2 unscored for every non-Places result,
    so the axis was dropped from the weighting entirely.
    """
    from angel_filter.constraints import QueryConstraints
    from angel_filter.ranker import QueryIntent, _assemble_score, _compute_gap_scores

    r = _result("Joe's Pizza")
    fake_httpx(_FakeClient({"Joe's Pizza": (NEAR_LAT, NEAR_LNG)}))
    await enrich_distances([r], USER_LAT, USER_LNG, api_key="k")

    c = QueryConstraints(max_distance=2.0)
    ranked = _assemble_score(r, 0.5, _compute_gap_scores(r, c), 1, QueryIntent.DISTANCE, "why")

    assert ranked.axis_scored["P2_distance"] is True


# --- Locality for prompts ------------------------------------------------------
# The AI providers previously received no location at all, so they returned
# invented placeholder names ("The Green Bowl", "Taco Haven") that geocoded to
# nothing. Verified live: the same prompt with "Manhattan, NY" attached returns
# real venues (Joe's Pizza, Los Tacos No. 1) instead.

@pytest.mark.parametrize("address,expected", [
    ("1464 Atlantic Ave, Brooklyn, NY 11216, USA", "Brooklyn, NY"),
    ("Manhattan, NY 10036, USA", "Manhattan, NY"),
    ("350 5th Ave, New York, NY 10118, USA", "New York, NY"),
])
def test_locality_trims_street_and_zip(address, expected):
    """A house number and ZIP add nothing to "suggest lunch nearby".

    Worse, they invite the model to treat the address as a precise target
    rather than an area to search.
    """
    from angel_filter.geocode import _locality_from_address
    assert _locality_from_address(address) == expected


@pytest.mark.parametrize("address", ["", "   ", "USA"])
def test_locality_degrades_to_none(address):
    from angel_filter.geocode import _locality_from_address
    assert _locality_from_address(address) is None


@pytest.mark.asyncio
async def test_describe_location_noop_without_prerequisites(monkeypatch):
    """No key or no coordinates means no locality — and no network call."""
    from angel_filter.geocode import describe_location
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    assert await describe_location(40.758, -73.9855, api_key=None) is None
    assert await describe_location(None, None, api_key="k") is None


def test_prompt_includes_locality_when_known():
    """The whole point: the model is told where the user is."""
    from angel_filter.constraints import QueryConstraints
    from angel_filter.prompt import build_prompt

    p = build_prompt("lunch", 5, QueryConstraints(user_locality="Manhattan, NY"))
    assert "Manhattan, NY" in p


def test_prompt_omits_locality_when_unknown():
    """Without coordinates the prompt must look exactly as it did before."""
    from angel_filter.constraints import QueryConstraints
    from angel_filter.prompt import build_prompt

    p = build_prompt("lunch", 5, QueryConstraints())
    assert "User is in or near" not in p


def test_prompt_still_forbids_inventing_distance():
    """Knowing the area must not become licence to guess distances.

    The model proposes local venues; geocode.py measures them. If this rule
    ever drops, fabricated distance_miles values reach the P2 axis.
    """
    from angel_filter.constraints import QueryConstraints
    from angel_filter.prompt import build_prompt

    p = build_prompt("lunch", 5, QueryConstraints(user_locality="Manhattan, NY"))
    assert "Do NOT include distance_miles" in p


@pytest.mark.asyncio
async def test_orchestrator_resolves_locality_into_constraints(monkeypatch):
    """The wiring, not just the pieces.

    The prompt tests pass even if the orchestrator never calls
    describe_location — the field simply stays None and the prompt omits it
    silently. Only driving handle_query catches that.
    """
    import angel_filter.orchestrator as orch
    from angel_filter.providers.base import BaseProvider

    async def fake_describe(lat, lng, api_key=None):
        return "Manhattan, NY"

    monkeypatch.setattr(orch, "describe_location", fake_describe)

    seen = {}

    class _P(BaseProvider):
        name = "mock"

        async def query(self, q, max_results=10, constraints=None):
            seen["locality"] = constraints.user_locality
            return []

    class _R:
        async def rank(self, *a, **k):
            return []

    await orch.Orchestrator(providers=[_P()], ranker=_R()).handle_query(
        "lunch", user_lat=40.758, user_lng=-73.9855
    )

    assert seen["locality"] == "Manhattan, NY", (
        "provider prompts never received the user's locality"
    )


@pytest.mark.asyncio
async def test_orchestrator_skips_locality_without_coordinates(monkeypatch):
    """No coordinates means no lookup at all — not a wasted API call."""
    import angel_filter.orchestrator as orch
    from angel_filter.providers.base import BaseProvider

    calls = {"n": 0}

    async def fake_describe(lat, lng, api_key=None):
        calls["n"] += 1
        return "Somewhere"

    monkeypatch.setattr(orch, "describe_location", fake_describe)

    class _P(BaseProvider):
        name = "mock"

        async def query(self, q, max_results=10, constraints=None):
            return []

    class _R:
        async def rank(self, *a, **k):
            return []

    await orch.Orchestrator(providers=[_P()], ranker=_R()).handle_query("lunch")

    assert calls["n"] == 0


# --- Cost guards ---------------------------------------------------------------
# A single uncached query fired 17 geocode lookups: the fan-out returns ~40
# results, the user sees 5, and we were geocoding everything in between. Three
# layers now bound that — a shortlist, a process-wide venue cache, and a hard
# ceiling with an off switch.

def _bulk(n, *, strong_upto=0, query_words="lunch spots near me"):
    out = []
    for i in range(n):
        strong = i < strong_upto
        out.append(ProviderResult(
            title=f"{'Great' if strong else 'Random'} Place {i}",
            snippet=query_words if strong else "entirely unrelated filler",
            provider=f"p{i % 4}",
            price=12.0 if strong else 90.0,
            rating=4.8 if strong else 2.9,
        ))
    return out


def test_shortlist_keeps_the_plausible_candidates():
    """Spend lookups on results that could actually place, not all 40."""
    from angel_filter.constraints import QueryConstraints
    from angel_filter.geocode import shortlist_for_enrichment

    results = _bulk(40, strong_upto=12)
    picked = shortlist_for_enrichment(
        results, "lunch spots near me", QueryConstraints(budget=20.0), limit=12
    )

    assert len(picked) == 12
    assert all(r.title.startswith("Great") for r in picked), (
        "shortlist dropped strong candidates in favour of weak ones"
    )


def test_shortlist_is_a_noop_below_the_limit():
    """Small result sets are not worth filtering."""
    from angel_filter.constraints import QueryConstraints
    from angel_filter.geocode import shortlist_for_enrichment

    results = _bulk(5)
    assert shortlist_for_enrichment(results, "lunch", QueryConstraints(), limit=12) == results


def test_shortlist_always_keeps_already_measured_results():
    """A Google Places result costs no lookup, so excluding it saves nothing.

    Dropping it would also discard a real measurement in favour of geocoding
    something else — strictly worse on both axes.
    """
    from angel_filter.constraints import QueryConstraints
    from angel_filter.geocode import shortlist_for_enrichment

    # Every other result is a strong match for the query; the measured one is a
    # deliberately terrible one. On merit it loses the shortlist outright — so
    # if it survives, it survives *because* it already has a distance.
    results = _bulk(30, strong_upto=30, query_words="lunch spots near me")
    measured = ProviderResult(
        title="Zzz Irrelevant", snippet="nothing to do with the query",
        provider="google_places", price=999.0, rating=1.0, distance=0.4,
    )
    results.append(measured)

    picked = shortlist_for_enrichment(
        results, "lunch spots near me", QueryConstraints(budget=20.0), limit=5
    )
    assert measured in picked, "dropped a result that already had a real measurement"


def test_shortlist_leaves_headroom_for_distance_to_reorder():
    """The shortlist must be well above top_k, not equal to it.

    Distance carries 60% of the axis score on a distance query. A shortlist the
    size of top_k would fix the winner before the deciding axis was measured.
    """
    from angel_filter.geocode import _shortlist_size

    assert _shortlist_size() >= 10


@pytest.mark.asyncio
async def test_second_query_reuses_cached_coordinates(fake_httpx):
    """The same venue across queries must cost one lookup, not one per query."""
    client = fake_httpx(_FakeClient({"Joe's Pizza": (NEAR_LAT, NEAR_LNG)}))

    first = [_result("Joe's Pizza")]
    await enrich_distances(first, USER_LAT, USER_LNG, api_key="k")
    second = [_result("Joe's Pizza", provider="gemini")]
    await enrich_distances(second, USER_LAT, USER_LNG, api_key="k")

    assert client.calls == ["Joe's Pizza"], f"re-geocoded a known venue: {client.calls}"
    assert second[0].distance is not None, "cached coordinates were not applied"


@pytest.mark.asyncio
async def test_unresolvable_names_are_cached_too(fake_httpx):
    """A name that resolves to nothing must not be retried every query.

    Otherwise a hallucinated venue that appears in every run costs a lookup
    every run, forever.
    """
    client = fake_httpx(_FakeClient({}))

    await enrich_distances([_result("Nowhere Cafe")], USER_LAT, USER_LNG, api_key="k")
    await enrich_distances([_result("Nowhere Cafe")], USER_LAT, USER_LNG, api_key="k")

    assert client.calls == ["Nowhere Cafe"]


@pytest.mark.asyncio
async def test_lookup_ceiling_bounds_a_pathological_run(fake_httpx, monkeypatch):
    """Backstop: many distinct venues cannot silently cost 40 calls."""
    monkeypatch.setenv("ANGEL_GEOCODE_MAX_LOOKUPS", "3")
    results = [_result(f"Distinct Venue {i}") for i in range(10)]
    client = fake_httpx(_FakeClient({f"Distinct Venue {i}": (NEAR_LAT, NEAR_LNG) for i in range(10)}))

    await enrich_distances(results, USER_LAT, USER_LNG, api_key="k")

    assert len(client.calls) == 3
    # Titles past the ceiling stay honestly unscored rather than guessed.
    assert sum(1 for r in results if r.distance is None) == 7


@pytest.mark.asyncio
async def test_env_flag_disables_enrichment_entirely(fake_httpx, monkeypatch):
    """The off switch: no calls, no distances, no error."""
    monkeypatch.setenv("ANGEL_GEOCODE_ENABLED", "false")
    results = [_result("Joe's Pizza")]
    client = fake_httpx(_FakeClient({"Joe's Pizza": (NEAR_LAT, NEAR_LNG)}))

    n = await enrich_distances(results, USER_LAT, USER_LNG, api_key="k")

    assert n == 0
    assert client.calls == []
    assert results[0].distance is None


@pytest.mark.asyncio
async def test_orchestrator_enriches_only_the_shortlist(monkeypatch):
    """Wiring: handle_query must pass a shortlist, not the whole fan-out.

    Testing shortlist_for_enrichment alone proves nothing about whether the
    orchestrator actually calls it — the same wiring gap that hid the locality
    bug and the REST priority bug.
    """
    import angel_filter.orchestrator as orch
    from angel_filter.providers.base import BaseProvider

    seen = {}

    async def spy_enrich(results, lat, lng, api_key=None):
        seen["n"] = len(results)
        return 0

    monkeypatch.setattr(orch, "enrich_distances", spy_enrich)
    monkeypatch.setenv("ANGEL_GEOCODE_SHORTLIST", "12")

    bulk = _bulk(40, strong_upto=12)

    class _P(BaseProvider):
        name = "mock"

        async def query(self, q, max_results=10, constraints=None):
            return bulk

    class _R:
        async def rank(self, *a, **k):
            return []

    await orch.Orchestrator(providers=[_P()], ranker=_R()).handle_query(
        "lunch spots near me", user_lat=40.758, user_lng=-73.9855
    )

    assert seen["n"] < len(bulk), (
        f"enriched all {len(bulk)} results; shortlist not applied"
    )
    assert seen["n"] <= 13
