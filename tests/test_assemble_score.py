"""Tests for _assemble_score, the single owner of the final-score formula.

Both scoring paths (embedding and keyword) delegate here after computing their
own similarity and rationale. This formula previously lived as two identical
copies that drifted apart; these tests pin its behaviour directly so a change
to the arithmetic has to survive an assertion, not just a code review.
"""

from __future__ import annotations

import pytest

from angel_filter.constraints import QueryConstraints
from angel_filter.providers.base import ProviderResult
from angel_filter.ranker import (
    W_AXIS,
    W_CONSENSUS,
    W_SIMILARITY,
    SPONSORED_PENALTY,
    QueryIntent,
    _assemble_score,
    _axis_bonus,
    _axis_scored_mask,
    _compute_gap_scores,
)


def _result(**kwargs) -> ProviderResult:
    return ProviderResult(title="x", snippet="", provider="p", **kwargs)


def _expected_axis(r: ProviderResult, axis_scores, intent) -> float:
    return _axis_bonus(axis_scores, intent, _axis_scored_mask(r))


def test_formula_matches_the_documented_weights():
    """final = W_SIMILARITY*sim + W_AXIS*axis + W_CONSENSUS*c_factor - penalty."""
    r = _result(price=10.0, rating=4.5)  # no distance -> P2 unscored
    axis_scores = {"P1_price": 0.8, "P2_distance": 0.5, "P3_rating": 0.9}
    similarity = 0.6

    result = _assemble_score(r, similarity, axis_scores, 1, QueryIntent.GENERAL, "why")

    axis = _expected_axis(r, axis_scores, QueryIntent.GENERAL)
    expected = W_SIMILARITY * similarity + W_AXIS * axis + W_CONSENSUS * 0.0 - 0.0
    assert result.score == pytest.approx(round(expected, 4))


def test_sponsored_penalty_is_subtracted():
    """A sponsored result scores exactly SPONSORED_PENALTY below its twin."""
    axis_scores = {"P1_price": 0.5, "P2_distance": 0.5, "P3_rating": 0.5}
    organic = _assemble_score(
        _result(sponsored=False), 0.7, axis_scores, 1, QueryIntent.GENERAL, "w"
    )
    ad = _assemble_score(
        _result(sponsored=True), 0.7, axis_scores, 1, QueryIntent.GENERAL, "w"
    )
    assert organic.score - ad.score == pytest.approx(SPONSORED_PENALTY, abs=1e-9)


@pytest.mark.parametrize("count,expected_factor", [(1, 0.0), (2, 0.5), (3, 1.0), (5, 1.0)])
def test_consensus_factor_scales_and_caps(count, expected_factor):
    """c_factor is (extras capped at 2) / 2, contributing W_CONSENSUS at most."""
    axis_scores = {"P1_price": 0.5, "P2_distance": 0.5, "P3_rating": 0.5}
    base = _assemble_score(_result(), 0.5, axis_scores, 1, QueryIntent.GENERAL, "w")
    got = _assemble_score(_result(), 0.5, axis_scores, count, QueryIntent.GENERAL, "w")
    assert got.score - base.score == pytest.approx(W_CONSENSUS * expected_factor, abs=1e-9)


def test_carries_rationale_and_metadata_through():
    """The helper passes rationale, axis_scores, and consensus_count untouched."""
    axis_scores = {"P1_price": 0.5, "P2_distance": 0.5, "P3_rating": 0.5}
    result = _assemble_score(
        _result(), 0.5, axis_scores, 3, QueryIntent.PRICE, "the rationale"
    )
    assert result.rationale == "the rationale"
    assert result.axis_scores is axis_scores
    assert result.consensus_count == 3


def test_score_is_rounded_to_four_places():
    axis_scores = {"P1_price": 0.333, "P2_distance": 0.5, "P3_rating": 0.777}
    result = _assemble_score(
        _result(price=1.0, rating=3.0), 0.123456, axis_scores, 1, QueryIntent.GENERAL, "w"
    )
    assert result.score == round(result.score, 4)


# --- Score breakdown -----------------------------------------------------------
# _assemble_score computed all four terms and discarded them, leaving the UI with
# a bare 0.82 and no way to show what drove it — including the sponsored penalty,
# which is the project's thesis.

def test_breakdown_terms_sum_to_the_score():
    """The published arithmetic must actually reconstruct the score.

    If these drift, the UI shows a breakdown that does not add up to the number
    printed beside it.
    """
    r = ProviderResult(title="x", snippet="s", provider="p", price=10.0, rating=4.5)
    ranked = _assemble_score(r, 0.7, _compute_gap_scores(r, QueryConstraints(budget=15.0)),
                             2, QueryIntent.PRICE, "why")
    b = ranked.score_breakdown

    total = (b["similarity_weighted"] + b["axis_weighted"]
             + b["consensus_weighted"] - b["sponsored_penalty"])
    assert total == pytest.approx(ranked.score, abs=0.001)


def test_breakdown_reports_the_sponsored_penalty():
    """The penalty must be visible as its own term, not folded into the total."""
    r = ProviderResult(title="x", snippet="s", provider="p", price=10.0,
                       rating=4.5, sponsored=True)
    ranked = _assemble_score(r, 0.7, _compute_gap_scores(r, QueryConstraints(budget=15.0)),
                             1, QueryIntent.PRICE, "why")

    assert ranked.score_breakdown["sponsored_penalty"] == pytest.approx(SPONSORED_PENALTY)


def test_breakdown_penalty_is_zero_for_organic_results():
    r = ProviderResult(title="x", snippet="s", provider="p", price=10.0, rating=4.5)
    ranked = _assemble_score(r, 0.7, _compute_gap_scores(r, QueryConstraints(budget=15.0)),
                             1, QueryIntent.PRICE, "why")

    assert ranked.score_breakdown["sponsored_penalty"] == 0.0


def test_breakdown_survives_serialisation():
    """The UI cannot show the arithmetic if the API does not ship it."""
    import angel_filter.server as server
    from angel_filter.orchestrator import OrchestratorResponse

    r = ProviderResult(title="x", snippet="s", provider="p", price=10.0, rating=4.5)
    ranked = _assemble_score(r, 0.7, _compute_gap_scores(r, QueryConstraints(budget=15.0)),
                             1, QueryIntent.PRICE, "why")
    payload = server._serialize_response(OrchestratorResponse(
        ranked=[ranked], providers_used=["p"], providers_failed=[],
        intent=QueryIntent.PRICE, constraints=QueryConstraints(),
    ))

    assert payload["results"][0]["score_breakdown"] == ranked.score_breakdown
