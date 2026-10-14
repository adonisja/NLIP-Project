"""Out-of-range provider values must not escape the 0-1 scoring contract.

The AI providers parse price and rating out of free-form model output. A model
that answered "rating": "92%" produced a rating of 92, which scored 18.4 on the
P3 axis and gave the result a final score of 4.46 — beyond anything the
sponsored penalty or any other term could correct.
"""

from __future__ import annotations

import math

import pytest

from angel_filter.constraints import QueryConstraints
from angel_filter.providers.base import ProviderResult
from angel_filter.providers.openai_provider import _parse_results
from angel_filter.ranker import QueryIntent, Ranker, _compute_gap_scores


@pytest.mark.parametrize("rating", [92.0, 9.0, 5.01, -1.0, math.nan, math.inf])
def test_out_of_range_rating_is_treated_as_missing(rating):
    assert ProviderResult("X", "", rating=rating).rating is None


@pytest.mark.parametrize("field", ["price", "distance"])
@pytest.mark.parametrize("value", [-5.0, math.nan, math.inf])
def test_invalid_price_or_distance_is_treated_as_missing(field, value):
    assert getattr(ProviderResult("X", "", **{field: value}), field) is None


def test_booleans_are_not_numbers():
    """bool subclasses int in Python, so True would otherwise pass as 1.0."""
    assert ProviderResult("X", "", rating=True).rating is None


@pytest.mark.parametrize("rating", [0.0, 3.7, 5.0])
def test_valid_values_survive(rating):
    r = ProviderResult("X", "", price=0.0, distance=0.0, rating=rating)
    assert (r.price, r.distance, r.rating) == (0.0, 0.0, rating)


def test_provider_parsing_drops_a_percentage_rating():
    """The real path: model JSON → provider parser → ProviderResult."""
    results = _parse_results(
        {"candidates": [{"name": "Odd Diner", "price": "$12", "rating": "92%"}]}, 5,
    )
    assert results[0].rating is None
    assert results[0].price == 12.0


def test_axis_scores_stay_within_0_1_even_if_set_after_construction():
    r = ProviderResult("X", "")
    r.rating, r.price, r.distance = 92.0, -50.0, -3.0   # bypasses __post_init__
    scores = _compute_gap_scores(r, QueryConstraints())
    assert all(0.0 <= v <= 1.0 for v in scores.values()), scores


@pytest.mark.asyncio
async def test_a_bad_rating_cannot_outrank_an_honest_result():
    ranker = Ranker()
    ranker._ollama_available = False
    ranker._openai_available = False
    results = [
        ProviderResult("Good Pizza", "lunch pizza", provider="a", price=12, rating=4.8),
        ProviderResult("Odd Diner", "lunch pizza", provider="b", price=12, rating=92),
    ]
    ranked = await ranker.rank(
        "lunch pizza", results, intent=QueryIntent.RATING, constraints=QueryConstraints(),
    )
    assert ranked[0].result.title == "Good Pizza"
    assert all(0.0 <= r.score <= 1.0 for r in ranked)
