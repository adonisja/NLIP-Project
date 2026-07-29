"""Tests for multi-turn conversations over the NLIP conversation token.

NLIP already carries a conversation token — `correlated_execute` mints one and
echoes the client's back — but nothing was stored against it, so every turn
started cold and "cheaper than that" had no "that" to refer to.

Two properties matter most and are pinned hardest here:

  1. A refinement anchors on what the previous turn actually *returned*, not on
     the constraint it ran under. Being shown an $18 result and asking for
     cheaper means cheaper than $18, even if the budget was $50.
  2. Conversations are isolated by token. One user's follow-up must never
     inherit another's context, and a client that sends no token at all must
     still get a plain search rather than an error.

No network: these drive the store and the refinement parser directly, and the
session-level tests use a spy orchestrator.
"""

from __future__ import annotations

import time

import pytest

from angel_filter.constraints import QueryConstraints
from angel_filter.conversation import (
    Conversation,
    ConversationStore,
    Turn,
    apply_refinement,
    build_context_prefix,
    effective_query,
    looks_like_a_refinement,
)


def _turn(query="lunch under $20", budget=20.0, price=18.0, distance=None, rating=None):
    return Turn(
        query=query,
        constraints=QueryConstraints(budget=budget),
        top_title="Joe's Pizza",
        top_price=price,
        top_distance=distance,
        top_rating=rating,
    )


# --- Recognising a refinement --------------------------------------------------

@pytest.mark.parametrize("text", [
    "cheaper", "cheaper than that", "something closer", "closer",
    "better rated", "too expensive", "too far", "higher rated",
])
def test_short_comparatives_are_refinements(text):
    assert looks_like_a_refinement(text) is True


@pytest.mark.parametrize("text", [
    "cheaper pizza in Brooklyn with outdoor seating and a full bar",
    "find me a closer alternative to the place downtown near the park",
])
def test_a_full_sentence_is_a_new_search(text):
    """Length is the discriminator.

    A long query containing "cheaper" describes a new search that happens to use
    the word, not an adjustment to the previous one. Treating it as a refinement
    would silently discard everything else the user just asked for.
    """
    assert looks_like_a_refinement(text) is False


@pytest.mark.parametrize("text", ["", "   ", "lunch", "pizza near me"])
def test_non_comparatives_are_not_refinements(text):
    assert looks_like_a_refinement(text) is False


# --- Anchoring -----------------------------------------------------------------

def test_cheaper_anchors_on_what_was_actually_shown():
    """The point of storing the turn: anchor on the result, not the constraint.

    Budget was $50 but the winner cost $18. "Cheaper" means cheaper than $18 —
    anchoring on the $50 nobody reached would barely move the results.
    """
    previous = _turn(budget=50.0, price=18.0)
    c, notes = apply_refinement("cheaper", previous)

    assert c.budget is not None and c.budget < 18.0
    assert any("budget" in n for n in notes)


def test_cheaper_falls_back_to_the_budget_when_price_is_unknown():
    """Not every provider reports price; the constraint is the next best anchor."""
    previous = _turn(budget=20.0, price=None)
    c, _ = apply_refinement("cheaper", previous)

    assert c.budget is not None and c.budget < 20.0


def test_closer_tightens_distance():
    previous = _turn(distance=2.0)
    c, notes = apply_refinement("closer", previous)

    assert c.max_distance is not None and c.max_distance < 2.0
    assert any("distance" in n for n in notes)


def test_distance_never_tightens_to_zero():
    """Repeated "closer" must not converge on an unsatisfiable 0 miles."""
    previous = _turn(distance=0.05)
    c, _ = apply_refinement("closer", previous)

    assert c.max_distance is not None and c.max_distance >= 0.1


def test_better_rated_raises_the_floor():
    previous = _turn(rating=4.2)
    c, notes = apply_refinement("better rated", previous)

    assert c.min_rating is not None and c.min_rating > 4.2
    assert any("rating" in n for n in notes)


def test_rating_is_capped_at_five():
    """"Better" than 4.9 must not produce an unsatisfiable 5.4."""
    previous = _turn(rating=4.9)
    c, _ = apply_refinement("better rated", previous)

    assert c.min_rating is not None and c.min_rating <= 5.0


def test_refinement_preserves_untouched_axes():
    """Asking for cheaper must not silently drop a distance constraint."""
    previous = Turn(
        query="lunch", constraints=QueryConstraints(budget=20.0, max_distance=1.0),
        top_price=18.0,
    )
    c, _ = apply_refinement("cheaper", previous)

    assert c.max_distance == 1.0


def test_refinement_reuses_the_previous_subject():
    """"cheaper" carries the adjustment but no subject to search for."""
    previous = _turn(query="ramen near the office")
    assert effective_query("cheaper", previous) == "ramen near the office"


# --- The store -----------------------------------------------------------------

def test_turns_are_isolated_by_token():
    """One conversation's history must never leak into another's."""
    store = ConversationStore()
    store.record("token-a", _turn(query="sushi"))

    assert store.get("token-b") is None
    assert store.get("token-a").latest.query == "sushi"


def test_no_token_means_no_memory():
    """A stateless client gets a plain search, not an error."""
    store = ConversationStore()
    store.record(None, _turn())

    assert store.get(None) is None


def test_history_is_bounded():
    """A long conversation must not grow without limit."""
    from angel_filter.conversation import MAX_TURNS

    store = ConversationStore()
    for i in range(MAX_TURNS + 5):
        store.record("t", _turn(query=f"q{i}"))

    assert len(store.get("t").turns) == MAX_TURNS
    assert store.get("t").latest.query == f"q{MAX_TURNS + 4}"


def test_stale_conversations_are_pruned():
    store = ConversationStore(ttl_seconds=1)
    store.record("old", _turn())
    store._store["old"].last_seen = time.time() - 10

    assert store.get("old") is None


def test_context_prefix_summarises_recent_turns():
    """The model fallback needs the subject and what it produced."""
    conv = Conversation()
    conv.add(_turn(query="ramen"))
    conv.add(_turn(query="cheaper ramen"))

    prefix = build_context_prefix(conv)
    assert "ramen" in prefix
    assert "Joe's Pizza" in prefix


def test_context_prefix_is_empty_for_a_fresh_conversation():
    """A first turn must cost no extra prompt tokens."""
    assert build_context_prefix(Conversation()) == ""


# --- Session wiring ------------------------------------------------------------
# The tests above pass even if execute() never reads the token or never records
# a turn — the same wiring gap that hid the locality bug, the REST priority bug,
# and the shortlist bug. These drive the real session.

nlip = pytest.importorskip("nlip_sdk.nlip")
import angel_filter.server as server  # noqa: E402

pytestmark = pytest.mark.skipif(
    not server._NLIP_AVAILABLE, reason="NLIP libraries not available"
)


class _SpyOrchestrator:
    """Records the constraints each turn ran under; returns a plausible winner.

    The winner's price tracks the budget it ran under, the way a real run does —
    a tighter budget genuinely surfaces a cheaper venue. A spy that returned a
    fixed price would make consecutive refinements anchor on the same number and
    produce identical results, which reads as a product bug but is not one.
    """

    def __init__(self, price=18.0):
        self.calls: list[dict] = []
        self._price = price

    async def handle_query(self, user_query, **kwargs):
        from angel_filter.orchestrator import OrchestratorResponse
        from angel_filter.providers.base import ProviderResult
        from angel_filter.ranker import QueryIntent, RankedResult

        self.calls.append({"query": user_query, **kwargs})
        c = kwargs.get("override_constraints") or QueryConstraints(budget=20.0)
        price = min(self._price, c.budget * 0.9) if c.budget else self._price
        return OrchestratorResponse(
            ranked=[RankedResult(
                result=ProviderResult(
                    title="Joe's Pizza", snippet="s", provider="mock",
                    price=round(price, 2), rating=4.4,
                ),
                score=0.8, rationale="why", axis_scores={}, consensus_count=1,
            )],
            providers_used=["mock"], providers_failed=[],
            intent=QueryIntent.GENERAL, constraints=c,
        )


@pytest.fixture
def spy_session(monkeypatch):
    from angel_filter.conversation import CONVERSATIONS

    spy = _SpyOrchestrator()
    monkeypatch.setattr(server, "ORCHESTRATOR", spy)
    CONVERSATIONS._store.clear()
    yield server.AngelFilterSession(), spy
    CONVERSATIONS._store.clear()


def _msg(text, token=None):
    m = nlip.NLIP_Factory.create_text(text)
    if token:
        m.add_conversation_token(token)
    return m


@pytest.mark.asyncio
async def test_follow_up_tightens_the_budget(spy_session):
    """End to end: turn 2 must run under bounds derived from turn 1's winner."""
    session, spy = spy_session

    await session.execute(_msg("lunch under $20", token="c1"))
    await session.execute(_msg("cheaper than that", token="c1"))

    assert len(spy.calls) == 2
    override = spy.calls[1]["override_constraints"]
    assert override is not None, "second turn ran with no refinement applied"
    assert override.budget < 18.0, "budget did not tighten below the winner's price"


@pytest.mark.asyncio
async def test_follow_up_reuses_the_previous_subject(spy_session):
    """"cheaper than that" alone is not searchable; turn 1 supplies the subject."""
    session, spy = spy_session

    await session.execute(_msg("ramen near the office", token="c1"))
    await session.execute(_msg("cheaper", token="c1"))

    assert spy.calls[1]["query"] == "ramen near the office"


@pytest.mark.asyncio
async def test_a_different_token_does_not_inherit_context(spy_session):
    """The isolation guarantee — one user's history is not another's."""
    session, spy = spy_session

    await session.execute(_msg("sushi under $40", token="conv-a"))
    await session.execute(_msg("cheaper than that", token="conv-b"))

    assert spy.calls[1]["override_constraints"] is None
    assert spy.calls[1]["query"] == "cheaper than that"


@pytest.mark.asyncio
async def test_no_token_still_answers(spy_session):
    """A stateless NLIP client must get a plain search, not an error."""
    session, spy = spy_session

    reply = await session.execute(_msg("cheaper than that"))

    assert reply.content
    assert spy.calls[0]["override_constraints"] is None


@pytest.mark.asyncio
async def test_first_turn_sends_no_conversation_context(spy_session):
    """A fresh conversation must not pay for prompt context it does not have."""
    session, spy = spy_session

    await session.execute(_msg("lunch under $20", token="c1"))

    assert spy.calls[0]["context_prefix"] == ""


@pytest.mark.asyncio
async def test_unrecognised_follow_up_falls_back_to_prompt_context(spy_session):
    """The LLM fallback: a phrasing the delta parser misses still gets history.

    "what about vegetarian options" is a genuine follow-up but matches no
    comparative pattern, so the models receive the previous turns instead.
    """
    session, spy = spy_session

    await session.execute(_msg("lunch under $20", token="c1"))
    await session.execute(_msg("what about vegetarian options", token="c1"))

    assert spy.calls[1]["override_constraints"] is None
    assert "lunch under $20" in spy.calls[1]["context_prefix"]


@pytest.mark.asyncio
async def test_refinement_is_announced_in_the_reply(spy_session):
    """A refined result set must say why it changed, not silently differ."""
    session, _ = spy_session

    await session.execute(_msg("lunch under $20", token="c1"))
    reply = await session.execute(_msg("cheaper than that", token="c1"))

    assert "Refined" in reply.content


@pytest.mark.asyncio
async def test_refinements_compound_across_turns(spy_session):
    """Three turns of "cheaper" must keep tightening, not reset each time."""
    session, spy = spy_session

    await session.execute(_msg("lunch under $20", token="c1"))
    await session.execute(_msg("cheaper", token="c1"))
    await session.execute(_msg("cheaper", token="c1"))

    first = spy.calls[1]["override_constraints"].budget
    second = spy.calls[2]["override_constraints"].budget
    assert second < first, f"second refinement did not tighten further ({second} vs {first})"


# --- The turn_mode flag --------------------------------------------------------
# The UI states outright whether a query continues the thread. Inferring it from
# phrasing is wrong in both directions: a genuinely new search containing
# "cheaper" reads as a refinement, and a follow-up phrased unusually reads as
# new. The flag removes the guess; its absence keeps the old heuristic so an
# agent client that never sends the label is unaffected.

def _msg_mode(text, token=None, mode=None):
    m = nlip.NLIP_Factory.create_text(text)
    if token:
        m.add_conversation_token(token)
    if mode:
        m.add_text(mode, label="turn_mode")
    return m


@pytest.mark.parametrize("value,expected", [
    ("continue", "continue"), ("new", "new"),
    ("CONTINUE", "continue"), ("  new  ", "new"),
])
def test_turn_mode_is_read_from_the_message(value, expected):
    assert server._extract_turn_mode(_msg_mode("q", mode=value)) == expected


@pytest.mark.parametrize("value", ["maybe", "", "resume"])
def test_unrecognised_turn_mode_degrades_to_none(value):
    """An unknown value falls back to the heuristic, never raises."""
    assert server._extract_turn_mode(_msg_mode("q", mode=value)) is None


def test_absent_turn_mode_is_none():
    """Agent clients that know nothing about this label keep working."""
    assert server._extract_turn_mode(nlip.NLIP_Factory.create_text("q")) is None


@pytest.mark.asyncio
async def test_explicit_new_ignores_prior_turns(spy_session):
    """The flag's real work.

    "cheaper" after a previous turn would normally refine. Saying "new" must
    override that — the user told us this is a fresh search, and honouring the
    wording instead would silently apply bounds they did not ask for.
    """
    session, spy = spy_session

    await session.execute(_msg_mode("lunch under $20", token="c1", mode="new"))
    await session.execute(_msg_mode("cheaper", token="c1", mode="new"))

    assert spy.calls[1]["override_constraints"] is None
    assert spy.calls[1]["query"] == "cheaper"
    assert spy.calls[1]["context_prefix"] == ""


@pytest.mark.asyncio
async def test_explicit_continue_refines(spy_session):
    session, spy = spy_session

    await session.execute(_msg_mode("lunch under $20", token="c1", mode="new"))
    await session.execute(_msg_mode("cheaper", token="c1", mode="continue"))

    assert spy.calls[1]["override_constraints"] is not None
    assert spy.calls[1]["query"] == "lunch under $20"


@pytest.mark.asyncio
async def test_continue_without_a_recognised_delta_sends_context(spy_session):
    """Stated follow-up, unrecognised wording — the model fallback path."""
    session, spy = spy_session

    await session.execute(_msg_mode("lunch under $20", token="c1", mode="new"))
    await session.execute(_msg_mode("any vegetarian ones", token="c1", mode="continue"))

    assert spy.calls[1]["override_constraints"] is None
    assert "lunch under $20" in spy.calls[1]["context_prefix"]
