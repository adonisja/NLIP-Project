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

    # Threshold is the sponsored penalty, which is the property that matters:
    # a data-less result must be further from a perfect match than an ad is
    # demoted. (Was >0.35 when an unscored axis was dropped from the weighting
    # entirely; the coverage factor now pulls both toward neutral, so the
    # absolute spread is smaller while the guarantee is unchanged.)
    spread = _axis(perfect, c, QueryIntent.PRICE) - _axis(bare, c, QueryIntent.PRICE)
    assert spread > 0.30, (
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


def test_silence_on_a_non_intent_axis_is_only_mildly_penalised():
    """Withholding an axis nobody asked about should cost little — but not nothing.

    This test previously asserted the opposite: that a result *without* distance
    scored >= one that disclosed a good distance. That encoded the inversion this
    module exists to prevent — silence outscoring disclosure — and it showed up
    live as results reading "no data" on two axes outranking a result that
    reported everything.

    The real requirement is that the penalty is proportionate. On a PRICE query
    distance carries only 20% of the axis weight, so withholding it should move
    the score a little, not decide the ranking.
    """
    c = QueryConstraints(budget=15.0)
    with_distance = _result(price=10.0, distance=0.5, rating=4.5)
    without = _result(price=10.0, rating=4.5)

    disclosed = _axis(with_distance, c, QueryIntent.PRICE)
    silent = _axis(without, c, QueryIntent.PRICE)

    assert silent < disclosed, "withholding an axis scored better than disclosing it"
    assert disclosed - silent < 0.15, (
        f"a non-intent axis cost {disclosed - silent:.3f} — too much for a 20% axis"
    )


# --- Boundary cases ------------------------------------------------------------

def test_no_data_at_all_returns_neutral():
    assert _axis(_result(), QueryConstraints(), QueryIntent.GENERAL) == 0.5
    assert _axis(_result(), QueryConstraints(), QueryIntent.PRICE) == 0.5


def test_general_intent_still_rewards_what_was_disclosed():
    """GENERAL has no dominant axis, so no single axis gets the silence guard.

    A result with two strong axes should still score well above neutral — it is
    judged on what it disclosed. It is pulled *toward* 0.5 in proportion to the
    third axis it withheld, which is what stops a partial result outranking one
    that disclosed everything, but it is not dragged all the way down.
    """
    r = _result(price=10.0, rating=4.5)
    score = _axis(r, QueryConstraints(), QueryIntent.GENERAL)
    assert 0.7 < score < 0.85, f"two strong disclosed axes scored {score:.3f}"


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

    # Both titles overlap the query on exactly one token, so keyword similarity
    # is identical and the axis score is the only thing that can separate them.
    # (An earlier version used titles that differed in similarity; that
    # dominated and the assertion passed with or without the mask.)
    #
    # They must still be *distinct* titles: identical ones are now collapsed as
    # duplicates before top-k, which would leave only one result to compare.
    ranked = await ranker.rank(
        "lunch",
        [
            ProviderResult(title="lunch spot", snippet="", provider="a", price=9.0, rating=4.8),
            ProviderResult(title="lunch place", snippet="", provider="b"),
        ],
        top_k=5,
        intent=QueryIntent.PRICE,
        constraints=QueryConstraints(budget=15.0, min_rating=4.0),
    )

    # Identify them by which one carried price data.
    strong = next(r for r in ranked if r.result.price is not None)
    bare = next(r for r in ranked if r.result.price is None)

    # A result that disclosed price and rating must clearly beat one that
    # disclosed nothing. Threshold was 0.15 when it also had to detect a dropped
    # mask; the coverage factor in _axis_bonus now handles the missing-data case
    # whether or not the mask is passed, so this pins the user-visible guarantee
    # (disclosure wins) rather than the internal plumbing.
    spread = strong.score - bare.score
    assert spread > 0.12, (
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
    # Same guarantee and same threshold as the keyword-loop test above:
    # disclosure must clearly beat silence through the real rank() path.
    spread = by_title["Cheap"].score - by_title["Bare"].score
    assert spread > 0.12, (
        f"embedding loop scored a data-less result within {spread:.3f} of a strong "
        "one — check that _axis_scored_mask is passed to _axis_bonus"
    )


# --- Failure mode 3: a partial result must not beat a complete one -------------
# Reported from a live run: four results showing "no data" on two axes outranked
# a result that disclosed price, distance and rating. Renormalisation judged the
# partial result on its single best axis while the complete one was judged on the
# average of three, so disclosing more could only ever hurt.

@pytest.mark.parametrize("value", [0.6, 0.75, 0.9, 1.0])
def test_disclosing_every_axis_beats_disclosing_one(value):
    """Same value on every axis must beat that value on one axis alone.

    This is the property the live bug violated. A dominant axis carries more
    weight, but it cannot make the other two irrelevant.
    """
    scores = {"P1_price": value, "P2_distance": value, "P3_rating": value}
    partial_scores = {"P1_price": value, "P2_distance": 0.5, "P3_rating": 0.5}

    complete = _axis_bonus(
        scores, QueryIntent.GENERAL,
        {"P1_price": True, "P2_distance": True, "P3_rating": True},
    )
    partial = _axis_bonus(
        partial_scores, QueryIntent.GENERAL,
        {"P1_price": True, "P2_distance": False, "P3_rating": False},
    )

    assert complete > partial, (
        f"a result disclosing one axis at {value} scored {partial:.3f}, beating "
        f"one disclosing all three at {value} ({complete:.3f})"
    )


def test_the_exact_case_from_the_live_run():
    """Regression: the ranking that prompted this fix.

    Tortilleria El Rey disclosed only price (0.90) and outranked Pugsley Pizza,
    which disclosed 0.863 / 0.82 / 0.86 across all three.
    """
    el_rey = _axis_bonus(
        {"P1_price": 0.9, "P2_distance": 0.5, "P3_rating": 0.5},
        QueryIntent.GENERAL,
        {"P1_price": True, "P2_distance": False, "P3_rating": False},
    )
    pugsley = _axis_bonus(
        {"P1_price": 0.863, "P2_distance": 0.82, "P3_rating": 0.86},
        QueryIntent.GENERAL,
        {"P1_price": True, "P2_distance": True, "P3_rating": True},
    )

    assert pugsley > el_rey, (
        f"price-only result ({el_rey:.3f}) still beats the fully-disclosed one "
        f"({pugsley:.3f})"
    )


@pytest.mark.parametrize("intent", list(QueryIntent))
def test_withholding_is_never_an_advantage_on_any_intent(intent):
    """Holding every axis at the same strong value, silence must never win.

    Parametrised across intents because the old intent-axis guard protected
    exactly one axis — and on GENERAL, none at all.
    """
    full = _axis_bonus(
        {"P1_price": 0.9, "P2_distance": 0.9, "P3_rating": 0.9}, intent,
        {"P1_price": True, "P2_distance": True, "P3_rating": True},
    )
    for withheld in ("P1_price", "P2_distance", "P3_rating"):
        scores = {"P1_price": 0.9, "P2_distance": 0.9, "P3_rating": 0.9}
        scores[withheld] = 0.5
        mask = {"P1_price": True, "P2_distance": True, "P3_rating": True}
        mask[withheld] = False
        assert _axis_bonus(scores, intent, mask) < full, (
            f"withholding {withheld} on a {intent.value} query was not a penalty"
        )
