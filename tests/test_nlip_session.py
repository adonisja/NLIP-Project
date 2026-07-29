"""Tests for the NLIP session's message handling.

The NLIP layer is the project's namesake, but almost nothing exercised it. In
particular execute() used to read the query with str(msg.content), which only
sees a message's top-level content and ignores its submessages — so multipart
or structured NLIP messages (which the protocol explicitly allows) had their
real query dropped or replaced with a stringified dict. These tests pin the
correct behaviour: the query is pulled with the SDK's extract_text().

execute() calls the module-level ORCHESTRATOR, so we patch it with a spy that
records the query string it was handed. That isolates the message-extraction
logic — what this fix changed — with no network and no live providers.
"""

from __future__ import annotations

import pytest

# Skip the whole module if the NLIP libraries aren't importable in this env —
# the fix only exists on the NLIP path.
nlip = pytest.importorskip("nlip_sdk.nlip")
import angel_filter.server as server

pytestmark = pytest.mark.skipif(
    not server._NLIP_AVAILABLE, reason="NLIP libraries not available"
)

from nlip_sdk.nlip import NLIP_Factory


def _make_response(ranked=None):
    from angel_filter.orchestrator import OrchestratorResponse
    from angel_filter.constraints import QueryConstraints
    from angel_filter.ranker import QueryIntent
    return OrchestratorResponse(
        ranked=ranked or [], providers_used=["mock"], providers_failed=[],
        intent=QueryIntent.GENERAL, constraints=QueryConstraints(),
    )


def _ranked(title, score, sponsored):
    from angel_filter.ranker import RankedResult
    from angel_filter.providers.base import ProviderResult
    return RankedResult(
        result=ProviderResult(title=title, snippet="s", provider="mock", sponsored=sponsored),
        score=score,
        rationale="why",
        axis_scores={"P1_price": 0.9, "P2_distance": 0.5, "P3_rating": 0.8},
        consensus_count=1,
    )


class _SpyOrchestrator:
    """Records everything execute() hands the orchestrator; returns a response."""

    def __init__(self, ranked=None):
        self.seen_query: str | None = None
        self.seen_kwargs: dict = {}
        self._ranked = ranked

    async def handle_query(self, user_query, **kwargs):
        self.seen_query = user_query
        self.seen_kwargs = kwargs
        return _make_response(self._ranked)


@pytest.fixture
def spy(monkeypatch):
    s = _SpyOrchestrator()
    monkeypatch.setattr(server, "ORCHESTRATOR", s)
    return s


async def _run(msg):
    return await server.AngelFilterSession().execute(msg)


# --- The bug the fix addresses -------------------------------------------------

@pytest.mark.asyncio
async def test_simple_text_query_extracted(spy):
    """The happy path str(msg.content) also handled — must keep working."""
    await _run(NLIP_Factory.create_text("cheap lunch nearby"))
    assert spy.seen_query == "cheap lunch nearby"


@pytest.mark.asyncio
async def test_multipart_text_is_fully_extracted(spy):
    """Text split across submessages must be joined, not truncated.

    str(msg.content) returned only the first part; extract_text() joins all
    text parts. This is the core regression the fix targets.
    """
    msg = NLIP_Factory.create_text("find me lunch")
    msg.add_text("under $15 within 1 mile")
    await _run(msg)
    assert spy.seen_query == "find me lunch under $15 within 1 mile"


@pytest.mark.asyncio
async def test_dict_content_does_not_leak_into_query(spy):
    """A message with dict top-level content + a text submessage.

    str(msg.content) produced "{'intent': 'search'}" as the query; extract_text
    must pull the actual text submessage instead.
    """
    msg = NLIP_Factory.create_json({"intent": "search"})
    msg.add_text("vegan tacos")
    await _run(msg)
    assert spy.seen_query == "vegan tacos"
    assert "intent" not in (spy.seen_query or "")


# --- Empty / no-text handling --------------------------------------------------

@pytest.mark.asyncio
async def test_message_with_no_text_short_circuits(spy):
    """A message carrying no text must not fan out to providers with ''."""
    msg = NLIP_Factory.create_json({"intent": "search"})  # no text part at all
    reply = await _run(msg)
    assert spy.seen_query is None, "orchestrator should not be called with empty text"
    assert "No text query" in reply.extract_text()


@pytest.mark.asyncio
async def test_whitespace_only_query_short_circuits(spy):
    reply = await _run(NLIP_Factory.create_text("   "))
    assert spy.seen_query is None
    assert "No text query" in reply.extract_text()


# --- Structured (multipart) reply ---------------------------------------------

from nlip_sdk.nlip import AllowedFormats


@pytest.mark.asyncio
async def test_reply_carries_both_text_and_json(monkeypatch):
    """The reply is multipart: a human summary AND a machine-readable payload."""
    ranked = [_ranked("Joe's Pizza", 0.82, False), _ranked("SponsorCo", 0.55, True)]
    monkeypatch.setattr(server, "ORCHESTRATOR", _SpyOrchestrator(ranked))

    reply = await _run(NLIP_Factory.create_text("lunch"))

    # Text part: readable summary.
    assert isinstance(reply.extract_text(), str) and reply.extract_text() != ""

    # JSON part: the full structured ranking, as a dict (not a stringified blob).
    payloads = reply.extract_field_list(AllowedFormats.structured, "JSON")
    assert len(payloads) == 1
    data = payloads[0]
    assert isinstance(data, dict)
    assert [r["title"] for r in data["results"]] == ["Joe's Pizza", "SponsorCo"]


@pytest.mark.asyncio
async def test_sponsored_flag_is_machine_readable_in_json(monkeypatch):
    """The sponsored penalty — the project thesis — must survive as a boolean.

    In the old text-only reply it was only the word '[SPONSORED]' inside a
    sentence; an agent couldn't reliably act on it. The JSON submessage carries
    it as a real field.
    """
    ranked = [_ranked("Joe's Pizza", 0.82, False), _ranked("SponsorCo", 0.55, True)]
    monkeypatch.setattr(server, "ORCHESTRATOR", _SpyOrchestrator(ranked))

    reply = await _run(NLIP_Factory.create_text("lunch"))
    data = reply.extract_field_list(AllowedFormats.structured, "JSON")[0]

    by_title = {r["title"]: r for r in data["results"]}
    assert by_title["SponsorCo"]["sponsored"] is True
    assert by_title["Joe's Pizza"]["sponsored"] is False
    # And the axis breakdown is present per result, not flattened away.
    assert set(by_title["Joe's Pizza"]["axis_scores"]) == {"P1_price", "P2_distance", "P3_rating"}


@pytest.mark.asyncio
async def test_text_only_client_still_gets_a_readable_summary(spy):
    """A consumer that only reads text is unaffected by the added JSON part."""
    reply = await _run(NLIP_Factory.create_text("pizza"))
    assert isinstance(reply.extract_text(), str)
    assert reply.extract_text() != ""


# --- Multi-input extraction (query + preference + location over NLIP) ----------
# The demo now sends its three inputs as typed NLIP parts, not a REST body:
#   query      -> top-level text        preference -> text submessage (labeled)
#   location   -> GPS submessage (labeled, JSON content)
# These prove execute() pulls each out correctly and hands them to the
# orchestrator — the plumbing that puts the protocol on the demo's critical path.
import json as _json

from angel_filter.ranker import QueryIntent
from angel_filter.server import (
    _extract_query, _extract_preference, _extract_location, _extract_priority,
)


def _full_demo_message():
    m = NLIP_Factory.create_text("cheap lunch nearby")
    m.add_text("casual, low price", label="preference")
    m.add_location_gps(_json.dumps({"lat": 40.768, "lng": -73.982}), label="user_location")
    return m


@pytest.mark.asyncio
async def test_execute_threads_all_three_inputs_to_orchestrator(monkeypatch):
    spy = _SpyOrchestrator()
    monkeypatch.setattr(server, "ORCHESTRATOR", spy)

    await _run(_full_demo_message())

    assert spy.seen_query == "cheap lunch nearby"
    assert spy.seen_kwargs["user_preference"] == "casual, low price"
    assert spy.seen_kwargs["user_lat"] == 40.768
    assert spy.seen_kwargs["user_lng"] == -73.982


def test_query_excludes_the_labeled_preference():
    """The query must be the unlabeled text only — not merged with preference.

    extract_text() would join them; _extract_query reads unlabeled parts so the
    preference stays a separate field.
    """
    assert _extract_query(_full_demo_message()) == "cheap lunch nearby"
    assert _extract_preference(_full_demo_message()) == "casual, low price"


def test_query_still_joins_unlabeled_submessages():
    """A client that splits its query across unlabeled text parts still works."""
    m = NLIP_Factory.create_text("find lunch")
    m.add_text("under $15")  # unlabeled -> part of the query
    assert _extract_query(m) == "find lunch under $15"


def test_simple_text_message_has_no_preference_or_location():
    """The #13 simple-text path: no submessages, must not crash on lookups."""
    m = NLIP_Factory.create_text("just pizza")
    assert _extract_query(m) == "just pizza"
    assert _extract_preference(m) is None
    assert _extract_location(m) == (None, None)


@pytest.mark.parametrize("bad_content", [
    "not-json",
    _json.dumps({"lat": 40.7}),          # missing lng
    _json.dumps({"lat": "x", "lng": 1}), # non-numeric
    _json.dumps([1, 2]),                 # wrong type
])
def test_malformed_location_degrades_to_none(bad_content):
    """A bad location payload must never raise — location is an optional signal."""
    m = NLIP_Factory.create_text("pizza")
    m.add_location_gps(bad_content, label="user_location")
    assert _extract_location(m) == (None, None)


@pytest.mark.asyncio
async def test_location_only_no_preference(monkeypatch):
    """Location without a preference: lat/lng flow, preference is None."""
    spy = _SpyOrchestrator()
    monkeypatch.setattr(server, "ORCHESTRATOR", spy)
    m = NLIP_Factory.create_text("lunch nearby")
    m.add_location_gps(_json.dumps({"lat": 1.5, "lng": 2.5}), label="user_location")

    await _run(m)

    assert spy.seen_kwargs["user_preference"] is None
    assert (spy.seen_kwargs["user_lat"], spy.seen_kwargs["user_lng"]) == (1.5, 2.5)


# --- Axis priority picker ------------------------------------------------------
# The UI's picker lets the user name the axis that matters instead of relying on
# detect_intent() inferring it from query keywords. It rides the protocol as a
# text submessage labeled "priority". The contract these tests pin: a recognised
# value becomes a QueryIntent, and *anything else* — absent, misspelled, wrong
# label — yields None, which handle_query() reads as "fall back to detection".
# That asymmetry is the safety property: a bad client loses the override, not
# the query.

def _priority_message(value, label="priority"):
    m = NLIP_Factory.create_text("lunch spots")
    m.add_text(value, label=label)
    return m


@pytest.mark.parametrize("value,expected", [
    ("price",    QueryIntent.PRICE),
    ("distance", QueryIntent.DISTANCE),
    ("rating",   QueryIntent.RATING),
    ("general",  QueryIntent.GENERAL),
])
def test_priority_submessage_maps_to_intent(value, expected):
    assert _extract_priority(_priority_message(value)) is expected


@pytest.mark.parametrize("value", ["PRICE", "Distance", "  rating  "])
def test_priority_is_case_and_whitespace_insensitive(value):
    """The picker sends lowercase, but a hand-rolled agent client may not."""
    assert _extract_priority(_priority_message(value)) is not None


@pytest.mark.parametrize("bad", ["banana", "", "cheapest", "P1"])
def test_unrecognised_priority_degrades_to_detection(bad):
    """An unknown axis must not raise — the query still runs, just auto-detected.

    Returning None (rather than raising) is what keeps a misbehaving client on
    the old inference path instead of handing it a 500.
    """
    assert _extract_priority(_priority_message(bad)) is None


def test_priority_ignores_other_labels():
    """A preference that happens to say 'price' must not become the override.

    The two are separate inputs: preference feeds semantic similarity, priority
    sets the axis weighting. Reading one as the other would silently couple them.
    """
    m = NLIP_Factory.create_text("lunch")
    m.add_text("price", label="preference")
    assert _extract_priority(m) is None


def test_simple_text_message_has_no_priority():
    """No submessages at all: the auto path, and no crash on the lookup."""
    assert _extract_priority(NLIP_Factory.create_text("just pizza")) is None


@pytest.mark.asyncio
async def test_priority_reaches_the_orchestrator(monkeypatch):
    """End of the wire: a picked axis arrives as handle_query(intent=...)."""
    spy = _SpyOrchestrator()
    monkeypatch.setattr(server, "ORCHESTRATOR", spy)

    await _run(_priority_message("distance"))

    assert spy.seen_kwargs["intent"] is QueryIntent.DISTANCE


@pytest.mark.asyncio
async def test_auto_sends_no_intent_so_detection_still_runs(monkeypatch):
    """The picker's Auto option omits the submessage entirely.

    handle_query() must receive intent=None so detect_intent() decides — this is
    the guarantee that the picker is additive and the pre-picker behaviour is
    unchanged for anyone who ignores it.
    """
    spy = _SpyOrchestrator()
    monkeypatch.setattr(server, "ORCHESTRATOR", spy)

    await _run(NLIP_Factory.create_text("cheap lunch nearby"))

    assert spy.seen_kwargs["intent"] is None
