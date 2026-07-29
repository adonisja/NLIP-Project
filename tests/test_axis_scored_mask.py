"""Tests for exposing which axes a provider actually disclosed.

The ranker stores a neutral 0.5 in `axis_scores` for any axis a provider did not
report, and separately tracks whether the axis was real so `_axis_bonus` can
renormalise over populated axes only. That mask never left the ranker: the API
shipped `axis_scores` alone, so every consumer saw 0.5 with no way to tell a
measured-mediocre result from an undisclosed one.

That is the exact distinction the README's "missing data is not mediocre data"
policy rests on, and the demo UI was plotting the placeholder as though it were
a measurement. These pin the mask onto RankedResult and through serialisation.
"""

from __future__ import annotations

import pytest

from angel_filter.constraints import QueryConstraints
from angel_filter.providers.base import ProviderResult
from angel_filter.ranker import QueryIntent, _assemble_score, _compute_gap_scores


def _result(**kw) -> ProviderResult:
    base = dict(title="Somewhere", snippet="s", provider="mock")
    base.update(kw)
    return ProviderResult(**base)


def _rank(r: ProviderResult, c: QueryConstraints | None = None):
    c = c or QueryConstraints()
    return _assemble_score(
        r, 0.5, _compute_gap_scores(r, c), 1, QueryIntent.GENERAL, "why"
    )


def test_fully_disclosed_result_marks_every_axis_scored():
    ranked = _rank(_result(price=12.0, distance=0.4, rating=4.5))
    assert ranked.axis_scored == {
        "P1_price": True, "P2_distance": True, "P3_rating": True,
    }


def test_missing_distance_is_marked_unscored():
    """The common case: AI providers return price and rating, never distance."""
    ranked = _rank(_result(price=12.0, rating=4.5))
    assert ranked.axis_scored["P2_distance"] is False
    assert ranked.axis_scored["P1_price"] is True
    assert ranked.axis_scored["P3_rating"] is True


def test_bare_search_result_marks_all_axes_unscored():
    """Brave returns no structured fields at all."""
    ranked = _rank(_result())
    assert set(ranked.axis_scored.values()) == {False}


def test_placeholder_score_is_indistinguishable_without_the_mask():
    """Why the mask has to exist at all.

    An undisclosed axis and a genuinely mid-scoring one can both read 0.5 in
    axis_scores. Only axis_scored separates them — so if this ever collapses,
    the UI is back to drawing silence as mediocrity.
    """
    undisclosed = _rank(_result(price=12.0))
    assert undisclosed.axis_scores["P2_distance"] == 0.5
    assert undisclosed.axis_scored["P2_distance"] is False


def test_mask_does_not_change_the_score():
    """Carrying the mask is an exposure change, not a scoring change.

    _axis_bonus already consumed the same mask internally; this guards against
    the refactor accidentally altering the weighting it feeds.
    """
    r = _result(price=12.0, rating=4.5)
    c = QueryConstraints(budget=15.0)
    assert _rank(r, c).score == pytest.approx(
        _assemble_score(r, 0.5, _compute_gap_scores(r, c), 1, QueryIntent.GENERAL, "why").score
    )


# --- Serialisation -------------------------------------------------------------

def test_serialized_response_carries_the_mask():
    """The UI cannot honour the distinction if the API never ships it."""
    import angel_filter.server as server
    from angel_filter.orchestrator import OrchestratorResponse

    ranked = _rank(_result(price=12.0, rating=4.5))
    payload = server._serialize_response(
        OrchestratorResponse(
            ranked=[ranked], providers_used=["mock"], providers_failed=[],
            intent=QueryIntent.GENERAL, constraints=QueryConstraints(),
        )
    )

    entry = payload["results"][0]
    assert entry["axis_scored"] == ranked.axis_scored
    assert entry["axis_scored"]["P2_distance"] is False


def test_serialized_mask_keys_match_the_score_keys():
    """Both dicts must describe the same three axes, or the UI lookup misses."""
    import angel_filter.server as server
    from angel_filter.orchestrator import OrchestratorResponse

    ranked = _rank(_result(price=1.0, distance=2.0, rating=3.0))
    payload = server._serialize_response(
        OrchestratorResponse(
            ranked=[ranked], providers_used=["mock"], providers_failed=[],
            intent=QueryIntent.GENERAL, constraints=QueryConstraints(),
        )
    )
    entry = payload["results"][0]
    assert entry["axis_scored"].keys() == entry["axis_scores"].keys()
