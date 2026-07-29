"""Tests for the embedding scoring path — the branch every other test skips.

tests/test_orchestrator.py forces `_ollama_available = False` everywhere, so
`_score_with_embeddings` and `_build_fuzzy_consensus` never run there. That
blind spot is where the consensus double-weighting and the title-collision
bugs both lived.

StubRanker below overrides the three embedding seams with a canned vector
table, so the real scoring path runs with no network, no Ollama, and fully
deterministic numbers.
"""

from __future__ import annotations

import pytest

from angel_filter.constraints import QueryConstraints
from angel_filter.providers.base import ProviderResult
from angel_filter.ranker import (
    W_CONSENSUS,
    QueryIntent,
    Ranker,
    _build_fuzzy_consensus,
    _cosine,
)


# --- Stub ranker --------------------------------------------------------------

class StubRanker(Ranker):
    """Ranker with a canned embedding table instead of a live backend.

    Overrides exactly the three seams that touch the network:
      _has_ollama      -> always True, so rank() takes the embedding branch
      _embed_all_ollama-> canned vector per result title
      _embed_query     -> canned vector for the query text
    Everything else — scoring, clustering, weighting — is the real code.
    """

    def __init__(self, vectors: dict[str, list[float]]):
        super().__init__()
        self._vectors = vectors
        self._ollama_available = True

    async def _has_ollama(self) -> bool:
        return True

    async def _embed_all_ollama(self, results):
        return {i: self._vectors[r.title] for i, r in enumerate(results)}

    async def _embed_query(self, text: str, backend: str = "ollama"):
        return self._vectors[text]


# --- Fixtures -----------------------------------------------------------------

# 3-element vectors are plenty to test clustering geometry. "Joe's Pizza" and
# "Joe Pizza" sit close together (cosine ~0.99, above FUZZY_THRESHOLD=0.75);
# "Terry Yaki" is orthogonal to both so it must not cluster with them.
_VECTORS = {
    "pizza":        [1.0, 0.0, 0.0],
    "Joe's Pizza":  [1.0, 0.1, 0.0],
    "Joe Pizza":    [1.0, 0.15, 0.0],
    "Terry Yaki":   [0.0, 0.0, 1.0],
}


def _result(title: str, provider: str, **kwargs) -> ProviderResult:
    return ProviderResult(title=title, snippet="", provider=provider, **kwargs)


# --- Consensus weighting ------------------------------------------------------

@pytest.mark.asyncio
async def test_consensus_reaches_documented_weight():
    """Two agreeing providers must contribute the full W_CONSENSUS.

    Regression test for the double-weighting bug: c_bonus was pre-multiplied
    by CONSENSUS_BONUS and then multiplied by W_CONSENSUS again, capping the
    real contribution at 0.0225 instead of the documented 0.15.
    """
    ranker = StubRanker(_VECTORS)

    # Same venue from three providers -> 2 extras -> full consensus weight.
    agreed = await ranker.rank("pizza", [
        _result("Joe's Pizza", "openai"),
        _result("Joe's Pizza", "gemini"),
        _result("Joe's Pizza", "brave"),
    ], top_k=5)

    # Same venue from one provider -> no consensus.
    alone = await ranker.rank("pizza", [
        _result("Joe's Pizza", "openai"),
    ], top_k=5)

    assert agreed[0].consensus_count == 3
    assert alone[0].consensus_count == 1

    delta = agreed[0].score - alone[0].score
    assert delta == pytest.approx(W_CONSENSUS, abs=1e-3), (
        f"consensus contributed {delta:.4f}, expected {W_CONSENSUS} "
        "— check for double-weighting in _score_with_embeddings"
    )


@pytest.mark.asyncio
async def test_consensus_caps_at_two_extra_providers():
    """A fourth agreeing provider must not add more score than the third."""
    ranker = StubRanker(_VECTORS)

    three = await ranker.rank("pizza", [
        _result("Joe's Pizza", p) for p in ("openai", "gemini", "brave")
    ], top_k=5)
    four = await ranker.rank("pizza", [
        _result("Joe's Pizza", p) for p in ("openai", "gemini", "brave", "watsonx")
    ], top_k=5)

    assert four[0].score == pytest.approx(three[0].score, abs=1e-4), (
        "consensus is not capped at 2 extra providers — a gang-up of weak "
        "results could outrank a strong one"
    )


# --- Fuzzy clustering ---------------------------------------------------------

@pytest.mark.asyncio
async def test_fuzzy_clustering_groups_near_identical_titles():
    """'Joe's Pizza' and 'Joe Pizza' from different providers must cluster."""
    ranker = StubRanker(_VECTORS)

    ranked = await ranker.rank("pizza", [
        _result("Joe's Pizza", "openai"),
        _result("Joe Pizza", "gemini"),
        _result("Terry Yaki", "brave"),
    ], top_k=5)

    by_title = {r.result.title: r for r in ranked}
    assert by_title["Joe's Pizza"].consensus_count == 2
    assert by_title["Joe Pizza"].consensus_count == 2
    assert by_title["Terry Yaki"].consensus_count == 1, (
        "an unrelated venue was pulled into the pizza cluster"
    )


def test_same_provider_results_do_not_cluster():
    """Two near-identical titles from ONE provider is not consensus.

    Guards the `results[i].provider == results[j].provider` skip — without it
    a single chatty provider could manufacture its own agreement.
    """
    results = [
        _result("Joe's Pizza", "openai"),
        _result("Joe Pizza", "openai"),
    ]
    embeddings = {0: _VECTORS["Joe's Pizza"], 1: _VECTORS["Joe Pizza"]}

    counts = _build_fuzzy_consensus(results, embeddings, threshold=0.75)

    assert counts[0] == 1
    assert counts[1] == 1


def test_consensus_is_keyed_by_index_not_title():
    """Distinct venues sharing a normalised title must not overwrite each other.

    Regression test for the collision bug: counts were keyed by _normalise(title),
    so two different results with the same normalised title clobbered one another
    and the union-find clustering was discarded at lookup time.
    """
    # Both normalise to "joes pizza" but they are separate cluster members.
    results = [
        _result("Joe's Pizza", "openai"),
        _result("JOES PIZZA", "gemini"),
        _result("Terry Yaki", "brave"),
    ]
    embeddings = {
        0: _VECTORS["Joe's Pizza"],
        1: _VECTORS["Joe Pizza"],
        2: _VECTORS["Terry Yaki"],
    }

    counts = _build_fuzzy_consensus(results, embeddings, threshold=0.75)

    assert set(counts.keys()) == {0, 1, 2}, "consensus map must be index-keyed"
    assert counts[2] == 1


# --- Scoring path sanity ------------------------------------------------------

@pytest.mark.asyncio
async def test_sponsored_penalty_applies_on_embedding_path():
    """The project thesis must hold on the embedding path, not just keywords."""
    ranker = StubRanker(_VECTORS)

    ranked = await ranker.rank("pizza", [
        _result("Joe's Pizza", "openai", sponsored=True),
        _result("Joe Pizza", "gemini", sponsored=False),
    ], top_k=5)

    assert ranked[0].result.title == "Joe Pizza", (
        "sponsored result outranked an equally-similar organic one"
    )
    assert "sponsored" in ranked[1].rationale.lower()


@pytest.mark.asyncio
async def test_rating_intent_separates_by_rating_under_constraint():
    """With min_rating set, a 4.8 must clearly outscore a 4.0 on P3.

    Regression test for the P3 compression bug: the old gap formula squeezed
    every above-threshold rating into [0.75, 0.80], making rating queries
    effectively unranked once the user said "at least 4 stars".
    """
    ranker = StubRanker(_VECTORS)

    ranked = await ranker.rank(
        "pizza",
        [
            _result("Joe's Pizza", "openai", rating=4.8),
            _result("Terry Yaki", "gemini", rating=4.0),
        ],
        top_k=5,
        intent=QueryIntent.RATING,
        constraints=QueryConstraints(min_rating=4.0),
    )

    by_title = {r.result.title: r for r in ranked}
    spread = (
        by_title["Joe's Pizza"].axis_scores["P3_rating"]
        - by_title["Terry Yaki"].axis_scores["P3_rating"]
    )
    assert spread > 0.25, (
        f"P3 spread between 4.8 and 4.0 was only {spread:.3f} — rating axis "
        "is too compressed to rank on"
    )


def test_stub_vectors_straddle_the_fuzzy_threshold():
    """Guard the fixtures themselves: the canned geometry must be meaningful.

    If someone edits _VECTORS and accidentally makes everything similar (or
    nothing similar), the clustering tests above would pass or fail for
    reasons unrelated to the clustering code.
    """
    close = _cosine(_VECTORS["Joe's Pizza"], _VECTORS["Joe Pizza"])
    far = _cosine(_VECTORS["Joe's Pizza"], _VECTORS["Terry Yaki"])

    assert close >= 0.75, f"fixture titles no longer cluster (cosine {close:.3f})"
    assert far < 0.75, f"unrelated fixture titles now cluster (cosine {far:.3f})"


# --- Duplicate collapsing -------------------------------------------------------
# Consensus clustering refuses to group results from the same provider, which is
# correct for counting agreement and wrong for deduplicating output — nothing
# collapsed duplicates, so a venue every provider named took one slot per
# mention. Observed live: "Shake Shack" held three of five. The consensus bonus
# compounds it, since agreeing copies all score identically and land adjacent.

def _r(title, provider, **kw):
    return ProviderResult(title=title, snippet=kw.pop("snippet", "s"), provider=provider, **kw)


@pytest.mark.asyncio
async def test_same_venue_from_many_providers_takes_one_slot():
    from angel_filter.ranker import Ranker

    ranker = Ranker()
    ranker._ollama_available = False
    ranker._openai_available = False

    ranked = await ranker.rank(
        "lunch",
        [
            _r("Shake Shack", "openai", price=12.0, rating=4.5),
            _r("Shake Shack", "gemini", price=12.0, rating=4.5),
            _r("Shake Shack", "ollama", price=12.0, rating=4.5),
            _r("Joe's Pizza", "openai", price=10.0, rating=4.4),
            _r("Katz's Deli", "brave", price=20.0, rating=4.6),
        ],
        top_k=5,
        constraints=QueryConstraints(),
    )

    titles = [r.result.title for r in ranked]
    assert len(titles) == len(set(titles)), f"duplicate venues in output: {titles}"
    assert len(titles) == 3


@pytest.mark.asyncio
async def test_dedup_keeps_the_earned_consensus_count():
    """Collapsing must not discard the agreement the copies represent.

    The surviving entry stands for all of them, so it keeps the consensus bonus
    — otherwise deduplicating would silently delete the project's own
    multi-provider-agreement signal.
    """
    from angel_filter.ranker import Ranker

    ranker = Ranker()
    ranker._ollama_available = False
    ranker._openai_available = False

    ranked = await ranker.rank(
        "lunch",
        [
            _r("Shake Shack", "openai", price=12.0, rating=4.5),
            _r("Shake Shack", "gemini", price=12.0, rating=4.5),
            _r("Shake Shack", "ollama", price=12.0, rating=4.5),
        ],
        top_k=5,
        constraints=QueryConstraints(),
    )

    assert len(ranked) == 1
    assert ranked[0].consensus_count == 3


@pytest.mark.asyncio
async def test_dedup_keeps_the_best_scoring_copy():
    """Copies can differ — one provider may report price where another does not.

    Dedup runs after the sort, so the copy retained is the one that scored
    highest, not whichever happened to arrive first.
    """
    from angel_filter.ranker import Ranker

    ranker = Ranker()
    ranker._ollama_available = False
    ranker._openai_available = False

    ranked = await ranker.rank(
        "lunch",
        [
            _r("Bar Bao", "openai"),                                # no axis data
            _r("Bar Bao", "gemini", price=9.0, rating=4.9),         # strong
        ],
        top_k=5,
        constraints=QueryConstraints(budget=15.0, min_rating=4.0),
    )

    assert len(ranked) == 1
    assert ranked[0].result.price == 9.0, "kept the weaker copy"


@pytest.mark.asyncio
async def test_dedup_matches_on_normalised_title():
    """Punctuation and casing differences are the same venue."""
    from angel_filter.ranker import Ranker

    ranker = Ranker()
    ranker._ollama_available = False
    ranker._openai_available = False

    ranked = await ranker.rank(
        "pizza",
        [
            _r("Joe's Pizza", "openai", price=10.0),
            _r("joes pizza", "gemini", price=10.0),
        ],
        top_k=5,
        constraints=QueryConstraints(),
    )

    assert len(ranked) == 1


@pytest.mark.asyncio
async def test_distinct_venues_are_all_kept():
    """Dedup must not over-collapse — different places stay separate."""
    from angel_filter.ranker import Ranker

    ranker = Ranker()
    ranker._ollama_available = False
    ranker._openai_available = False

    ranked = await ranker.rank(
        "lunch",
        [
            _r("Shake Shack", "openai", price=12.0),
            _r("Joe's Pizza", "openai", price=10.0),
            _r("Katz's Deli", "brave", price=20.0),
            _r("Los Tacos", "gemini", price=9.0),
        ],
        top_k=5,
        constraints=QueryConstraints(),
    )

    assert len(ranked) == 4
