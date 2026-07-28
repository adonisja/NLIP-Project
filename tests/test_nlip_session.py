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
    """Records the query execute() extracts; returns a configurable response."""

    def __init__(self, ranked=None):
        self.seen_query: str | None = None
        self._ranked = ranked

    async def handle_query(self, user_query, **kwargs):
        self.seen_query = user_query
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
