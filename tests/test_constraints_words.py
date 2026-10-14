"""Spelled-out numbers in constraints ("under twenty bucks")."""

from __future__ import annotations

import pytest

from angel_filter.constraints import extract_constraints


@pytest.mark.parametrize("query, field, expected", [
    ("lunch under twenty bucks", "budget", 20.0),
    ("dinner for twenty-five dollars", "budget", 25.0),
    ("no more than a hundred dollars", "budget", 100.0),
    ("one hundred and twenty dollars max", "budget", 120.0),
    ("within two miles", "max_distance", 2.0),
    ("half a mile away", "max_distance", 0.5),
    ("rated four stars", "min_rating", 4.0),
    ("at least four and a half stars", "min_rating", 4.5),
])
def test_spelled_out_numbers(query, field, expected):
    assert getattr(extract_constraints(query), field) == expected


@pytest.mark.parametrize("query", [
    "lunch for at least two people",
    "one of the best taco spots",
    "something open over one hour from now",
])
def test_number_words_without_a_unit_are_not_constraints(query):
    c = extract_constraints(query)
    assert (c.budget, c.max_distance, c.min_rating) == (None, None, None)


def test_digits_still_work_alongside_words():
    c = extract_constraints("under $15 within two miles rated 4 stars")
    assert (c.budget, c.max_distance, c.min_rating) == (15.0, 2.0, 4.0)
