"""Tests for axis weighting when providers return incomplete data.

Not every provider fills every axis. AI providers (OpenAI, Gemini, Ollama,
WatsonX) return price and rating but never distance — prompt.py forbids it
because they have no location context and would fabricate one. Brave returns
no structured fields at all.

That makes "missing" the common case in production, not an edge case, and it
drives two failure modes _axis_bonus has to avoid at once:

  1. Scoring missing data as a neutral 0.5 makes a bare link with no fields
     look merely mediocre instead of unrankable.
  2. Renormalising the missing weight away entirely rewards a provider for
     staying silent on the very axis the user asked about.

These tests pin both ends.
"""

from __future__ import annotations

import pytest

from angel_filter.constraints import QueryConstraints
from angel_filter.providers.base import ProviderResult
from angel_filter.ranker import (
    QueryIntent,
    _axis_bonus,
    _axis_scored_mask,
    _compute_gap_scores,
)


def _axis(r: ProviderResult, c: QueryConstraints, intent: QueryIntent) -> float:
    """Score one result the way the ranker's scoring loops do."""
    return _axis_bonus(_compute_gap_scores(r, c), intent, _axis_scored_mask(r))


def _result(title: str = "x", **kwargs) -> ProviderResult:
    return ProviderResult(title=title, snippet="", provider="test", **kwargs)


# --- The mask ------------------------------------------------------------------

def test_mask_reports_only_populated_axes():
    ai_shaped = _result(price=12.0, rating=4.5)          # typical AI provider
    assert _axis_scored_mask(ai_shaped) == {
        "P1_price": True, "P2_distance": False, "P3_rating": True,
    }


def test_mask_reports_nothing_for_a_bare_search_result():
    brave_shaped = _result()                              # typical Brave result
    assert not any(_axis_scored_mask(brave_shaped).values())


def test_zero_values_are_data_not_absence():
    """A free item and a 0.0 rating are real values, not missing fields."""
    mask = _axis_scored_mask(_result(price=0.0, distance=0.0, rating=0.0))
    assert all(mask.values()), "0.0 was treated as missing — check `is not None`"


# --- Failure mode 1: missing data must not read as mediocre --------------------

def test_result_with_no_data_cannot_approach_a_perfect_match():
    """A bare link must not land within a sponsored-penalty's distance of a
    result that satisfies every constraint.

    Before renormalisation, neutral-0.5 padding put these ~0.11 apart after
    W_AXIS weighting — closer than the 0.20 sponsored penalty, meaning an
    empty result was "cheaper" to rank than an ad was to demote.
    """
    c = QueryConstraints(budget=15.0, min_rating=4.0)
    perfect = _result(price=12.0, rating=4.8)
    bare = _result()

    spread = _axis(perfect, c, QueryIntent.PRICE) - _axis(bare, c, QueryIntent.PRICE)
    assert spread > 0.35, (
        f"a result with no data scored within {spread:.3f} of a perfect match"
    )


def test_partial_data_is_judged_on_what_was_disclosed():
    """Price+rating results are ranked against each other on those axes.

    Both lack distance, so neither is advantaged by it — the better price and
    rating must win on a PRICE query.
    """
    c = QueryConstraints(budget=15.0, min_rating=4.0)
    better = _result(price=9.0, rating=4.8)
    worse = _result(price=14.9, rating=4.0)

    assert _axis(better, c, QueryIntent.PRICE) > _axis(worse, c, QueryIntent.PRICE)


# --- Failure mode 2: silence must not beat disclosure on the intent axis --------

@pytest.mark.parametrize("distance", [0.05, 0.3, 0.9])
def test_real_distance_beats_silence_on_a_distance_query(distance):
    """A result that IS nearby must outrank one that never said where it is.

    Pure renormalisation broke this: the silent result's 60% distance weight
    was redistributed onto price and rating, so a 0.05 mi result with mediocre
    price lost to an AI result that dodged the question entirely.
    """
    c = QueryConstraints(max_distance=1.0)
    nearby = _result(distance=distance, price=20.0, rating=3.5)   # close, otherwise weak
    silent = _result(price=10.0, rating=4.9)                       # no distance, strong

    assert _axis(nearby, c, QueryIntent.DISTANCE) > _axis(silent, c, QueryIntent.DISTANCE), (
        f"a result {distance} mi away lost to one with no distance data at all"
    )


def test_disclosed_but_bad_still_loses_to_silence():
    """Silence is penalised; genuinely violating the constraint is penalised more.

    A result 3 mi away on a "within 1 mile" query must rank below one that
    simply has no distance — otherwise we would be punishing providers for
    reporting honestly.
    """
    c = QueryConstraints(max_distance=1.0)
    far = _result(distance=3.0, price=20.0, rating=3.5)
    silent = _result(price=10.0, rating=4.9)

    assert _axis(far, c, QueryIntent.DISTANCE) < _axis(silent, c, QueryIntent.DISTANCE)


def test_silence_on_a_non_intent_axis_is_not_penalised():
    """Missing distance should not hurt on a PRICE query — nobody asked.

    The intent-axis guard applies only to the axis the user cares about; the
    other two renormalise away cleanly.
    """
    c = QueryConstraints(budget=15.0)
    with_distance = _result(price=10.0, distance=0.5, rating=4.5)
    without = _result(price=10.0, rating=4.5)

    # Distance 0.5 mi with no max_distance scores 0.9 on P2 — above neutral.
    # If the missing axis were dragging `without` down, it would score lower.
    assert _axis(without, c, QueryIntent.PRICE) >= _axis(with_distance, c, QueryIntent.PRICE)


# --- Boundary cases ------------------------------------------------------------

def test_no_data_at_all_returns_neutral():
    assert _axis(_result(), QueryConstraints(), QueryIntent.GENERAL) == 0.5
    assert _axis(_result(), QueryConstraints(), QueryIntent.PRICE) == 0.5


def test_general_intent_renormalises_freely():
    """GENERAL has no dominant axis, so no axis gets the silence guard."""
    r = _result(price=10.0, rating=4.5)
    # Both present axes score high; without a guarded axis the result should
    # score high too rather than being pulled toward 0.5.
    assert _axis(r, QueryConstraints(), QueryIntent.GENERAL) > 0.8


def test_full_data_is_unchanged_by_the_mask():
    """When every axis has data, masking must be a no-op.

    This is the regression guard for the whole change: existing behaviour on
    complete results (the mock provider, and anything from a future provider
    that fills all three) must not shift.
    """
    c = QueryConstraints(budget=15.0, min_rating=4.0)
    full = _result(price=12.0, distance=0.5, rating=4.5)
    scores = _compute_gap_scores(full, c)

    for intent in QueryIntent:
        assert _axis_bonus(scores, intent, _axis_scored_mask(full)) == pytest.approx(
            _axis_bonus(scores, intent)
        ), f"masking changed the score for {intent.value} on a fully-populated result"


def test_mask_defaults_to_scoring_everything():
    """`scored=None` preserves the pre-change behaviour for callers without a
    ProviderResult to inspect."""
    scores = {"P1_price": 0.9, "P2_distance": 0.1, "P3_rating": 0.5}
    assert _axis_bonus(scores, QueryIntent.PRICE) == pytest.approx(
        0.60 * 0.9 + 0.20 * 0.1 + 0.20 * 0.5
    )


# --- Wiring --------------------------------------------------------------------
# The tests above call _axis_bonus directly, so they pass even if a scoring
# loop forgets to hand it the mask. These drive the real ranker end-to-end.
# Both scoring loops need this: the mask has to be threaded through
# _score_with_embeddings AND _score_with_keywords, and past changes to this
# file have repeatedly updated one and missed the other.

@pytest.mark.asyncio
async def test_keyword_scoring_loop_passes_the_mask():
    """rank() on the keyword path must judge partial results on disclosed axes."""
    from angel_filter.ranker import Ranker

    ranker = Ranker()
    ranker._ollama_available = False
    ranker._openai_available = False

    # Both titles are the same word, so keyword similarity is identical and the
    # axis score is the only thing that can separate them. (An earlier version
    # of this test used different titles; similarity then dominated and the
    # assertion passed with or without the mask.)
    ranked = await ranker.rank(
        "lunch",
        [
            ProviderResult(title="lunch", snippet="", provider="a", price=9.0, rating=4.8),
            ProviderResult(title="lunch", snippet="", provider="b"),
        ],
        top_k=5,
        intent=QueryIntent.PRICE,
        constraints=QueryConstraints(budget=15.0, min_rating=4.0),
    )

    # Same title, so identify them by which one carried price data.
    strong = next(r for r in ranked if r.result.price is not None)
    bare = next(r for r in ranked if r.result.price is None)

    # Measured: 0.168 with the mask, 0.134 without it. The threshold sits
    # between those so dropping the mask fails here rather than passing quietly.
    spread = strong.score - bare.score
    assert spread > 0.15, (
        f"keyword loop scored a data-less result within {spread:.3f} of a strong "
        "one — check that _axis_scored_mask is passed to _axis_bonus"
    )


@pytest.mark.asyncio
async def test_embedding_scoring_loop_passes_the_mask():
    """Same guarantee on the embedding path, using the stubbed ranker."""
    from tests.test_ranker_embeddings import StubRanker

    # Identical vectors for both results -> identical cosine similarity, so the
    # axis score is the only thing that can separate them.
    vectors = {"lunch": [1.0, 0.0], "Cheap": [1.0, 0.0], "Bare": [1.0, 0.0]}
    ranker = StubRanker(vectors)

    ranked = await ranker.rank(
        "lunch",
        [
            ProviderResult(title="Cheap", snippet="", provider="a", price=9.0, rating=4.8),
            ProviderResult(title="Bare", snippet="", provider="b"),
        ],
        top_k=5,
        intent=QueryIntent.PRICE,
        constraints=QueryConstraints(budget=15.0, min_rating=4.0),
    )

    by_title = {r.result.title: r for r in ranked}
    # Same 0.168-vs-0.134 boundary as the keyword-loop test above.
    spread = by_title["Cheap"].score - by_title["Bare"].score
    assert spread > 0.15, (
        f"embedding loop scored a data-less result within {spread:.3f} of a strong "
        "one — check that _axis_scored_mask is passed to _axis_bonus"
    )
