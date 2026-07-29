"""Ranker — the brain of the Angel Filter.

Scoring has four layers, applied in order:

  1. Semantic similarity — cosine similarity between the user preference
     embedding and the result text (via Ollama). Falls back to keyword
     overlap when Ollama is offline.

  2. Real-gap axis scoring — compute actual deltas between the candidate's
     structured fields and the user's extracted constraints:
       P1 price_gap    = candidate.price    - budget       (negative = under budget)
       P2 distance_gap = candidate.distance - max_distance (negative = closer)
       P3 rating_gap   = min_rating         - candidate.rating (negative = meets threshold)
     Gaps are normalised to 0-1 and weighted by the detected intent axis.

  3. Fuzzy consensus — candidates mentioned by multiple providers are boosted.
     With an embedding backend, matching clusters on embedding distance so
     "Joe's Pizza" and "Joe Pizza" group together; the keyword fallback
     matches on normalised title instead. Capped at 2 extra providers.

  4. Sponsored penalty — explicit deduction for any ad-flagged result.

Final score:
    score = (W_SIMILARITY * similarity)     # similarity  in 0-1
            + (W_AXIS      * axis_score)    # axis_score  in 0-1
            + (W_CONSENSUS * c_factor)      # c_factor    in 0-1
            - (SPONSORED_PENALTY if sponsored)

Every term is weight x (value in 0-1), and the three weights sum to 1.0, so
an unsponsored result scores in 0-1. Keep that invariant when tuning: folding
a weight into its own term double-applies it and silently shrinks the signal.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from enum import Enum

from angel_filter.constraints import QueryConstraints
from angel_filter.providers.base import ProviderResult

logger = logging.getLogger(__name__)

# --- Tunable weights ----------------------------------------------------------
# Weights sum to 1.0 across the three scoring components so no single
# signal can dominate. Adjust these to shift ranking behaviour.
W_SIMILARITY: float  = 0.50   # semantic similarity contribution
W_AXIS: float        = 0.35   # axis score contribution (split across all 3 axes)
W_CONSENSUS: float   = 0.15   # consensus contribution (capped)

SPONSORED_PENALTY: float = 0.20   # raised — ads should be clearly demoted
FUZZY_THRESHOLD: float   = 0.75   # embedding similarity above which two titles cluster
DEFAULT_EMBED_MODEL: str = "nomic-embed-text"


# --- Query intent -------------------------------------------------------------

class QueryIntent(str, Enum):
    PRICE    = "price"
    DISTANCE = "distance"
    RATING   = "rating"
    GENERAL  = "general"


_PRICE_KEYWORDS    = {"price", "cheap", "cheapest", "cost", "budget", "affordable",
                      "inexpensive", "low", "deal", "discount", "free", "save"}
_DISTANCE_KEYWORDS = {"near", "nearest", "close", "closest", "nearby", "distance",
                      "walking", "local", "around", "location", "convenient"}
_RATING_KEYWORDS   = {"best", "top", "rated", "rating", "review", "reviews",
                      "trusted", "quality", "popular", "recommended", "highest"}


def detect_intent(query: str) -> QueryIntent:
    tokens        = {t.lower().strip(".,!?;:'\"") for t in query.split()}
    price_hits    = len(tokens & _PRICE_KEYWORDS)
    distance_hits = len(tokens & _DISTANCE_KEYWORDS)
    rating_hits   = len(tokens & _RATING_KEYWORDS)
    best = max(price_hits, distance_hits, rating_hits)
    if best == 0:
        return QueryIntent.GENERAL
    if price_hits == best:
        return QueryIntent.PRICE
    if distance_hits == best:
        return QueryIntent.DISTANCE
    return QueryIntent.RATING


# --- Result dataclass ---------------------------------------------------------

@dataclass
class RankedResult:
    result: ProviderResult
    score: float
    rationale: str
    axis_scores: dict[str, float] = field(default_factory=dict)
    consensus_count: int = 0
    # Which axes the provider actually supplied data for. axis_scores stores a
    # neutral 0.5 placeholder for the others, so without this a consumer cannot
    # tell "measured as mediocre" from "never disclosed" — the distinction the
    # whole missing-data policy rests on. Scoring already computes this; carrying
    # it on the result lets the UI and API consumers see it too.
    axis_scored: dict[str, bool] = field(default_factory=dict)


# --- Ranker -------------------------------------------------------------------

OPENAI_EMBED_MODEL = "text-embedding-3-small"


class Ranker:
    def __init__(self, embed_model: str = DEFAULT_EMBED_MODEL):
        self.embed_model = embed_model
        self._ollama_available: bool | None = None
        self._openai_available: bool | None = None
        self._ollama_client = None  # lazily-created ollama.AsyncClient

    def _ollama(self):
        """Return a cached ollama.AsyncClient, importing lazily.

        Using the async client (not the module-level sync `ollama.embeddings`)
        is what keeps the embedding calls off the event-loop thread. The sync
        functions block until the model responds, which would freeze the whole
        FastAPI server for every other request in flight — see the note in
        _embed_all_ollama. Subclasses that stub embeddings never call this.
        """
        if self._ollama_client is None:
            import ollama
            self._ollama_client = ollama.AsyncClient()
        return self._ollama_client

    async def rank(
        self,
        user_preference: str,
        results: list[ProviderResult],
        top_k: int = 5,
        intent: QueryIntent = QueryIntent.GENERAL,
        constraints: QueryConstraints | None = None,
    ) -> list[RankedResult]:
        if not results:
            return []

        constraints = constraints or QueryConstraints()

        results = _apply_hard_constraints(results, constraints)
        if not results:
            return []

        if await self._has_ollama():
            logger.info("Embedding backend: Ollama (%s)", self.embed_model)
            embeddings = await self._embed_all_ollama(results)
            consensus  = _build_fuzzy_consensus(results, embeddings, FUZZY_THRESHOLD)
            scored     = await self._score_with_embeddings(
                user_preference, results, intent, constraints, consensus,
                embeddings, backend="ollama",
            )
        elif await self._has_openai():
            logger.info("Embedding backend: OpenAI (%s)", OPENAI_EMBED_MODEL)
            embeddings = await self._embed_all_openai(results)
            consensus  = _build_fuzzy_consensus(results, embeddings, FUZZY_THRESHOLD)
            scored     = await self._score_with_embeddings(
                user_preference, results, intent, constraints, consensus,
                embeddings, backend="openai",
            )
        else:
            logger.warning("No embedding backend available — using keyword-overlap fallback.")
            consensus = _build_token_consensus(results)
            scored    = _score_with_keywords(
                user_preference, results, intent, constraints, consensus
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        scored = _collapse_duplicates(scored)
        return scored[:top_k]

    # -- private ---------------------------------------------------------------

    async def _has_ollama(self) -> bool:
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            await self._ollama().embeddings(model=self.embed_model, prompt="ping")
            self._ollama_available = True
        except Exception as exc:
            logger.info("Ollama probe failed: %s", exc)
            self._ollama_available = False
        return self._ollama_available

    async def _has_openai(self) -> bool:
        if self._openai_available is not None:
            return self._openai_available
        import os
        if not os.getenv("OPENAI_API_KEY"):
            self._openai_available = False
            return False
        try:
            await self._openai_embed("ping")
            self._openai_available = True
        except Exception as exc:
            logger.info("OpenAI embedding probe failed: %s", exc)
            self._openai_available = False
        return self._openai_available

    async def _openai_embed(self, text: str) -> list[float]:
        import httpx, os
        api_key = os.getenv("OPENAI_API_KEY")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": OPENAI_EMBED_MODEL, "input": text},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    async def _embed_all_ollama(
        self, results: list[ProviderResult]
    ) -> dict[int, list[float]]:
        """Embed every result concurrently via the async Ollama client.

        Two properties matter here:
          - async client: each call awaits on the network instead of blocking
            the event-loop thread, so other requests keep being served while
            Ollama is thinking.
          - gather: the calls run concurrently rather than one-after-another.
            Ollama has no batch-embeddings endpoint (unlike OpenAI, see
            _embed_all_openai), so N results was N sequential round-trips;
            gather collapses that to roughly one round-trip of wall time.
        """
        client = self._ollama()
        texts = [f"{r.title}. {r.snippet}" for r in results]
        responses = await asyncio.gather(
            *(client.embeddings(model=self.embed_model, prompt=t) for t in texts)
        )
        return {i: resp["embedding"] for i, resp in enumerate(responses)}

    async def _embed_all_openai(
        self, results: list[ProviderResult]
    ) -> dict[int, list[float]]:
        import httpx, os
        # Batch all texts in a single API call — much faster than one call per result
        texts = [f"{r.title}. {r.snippet}" for r in results]
        api_key = os.getenv("OPENAI_API_KEY")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": OPENAI_EMBED_MODEL, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            # API returns embeddings in the same order as input
            return {i: item["embedding"] for i, item in enumerate(data)}

    async def _embed_query(self, text: str, backend: str = "ollama") -> list[float]:
        """Embed the user's preference text with whichever backend is active.

        Split out from _score_with_embeddings so tests can stub the embedding
        source without reaching into scoring logic. See StubRanker in
        tests/test_ranker_embeddings.py.
        """
        if backend == "ollama":
            resp = await self._ollama().embeddings(model=self.embed_model, prompt=text)
            return resp["embedding"]
        return await self._openai_embed(text)

    async def _score_with_embeddings(
        self,
        user_preference: str,
        results: list[ProviderResult],
        intent: QueryIntent,
        constraints: QueryConstraints,
        consensus: dict[int, int],
        embeddings: dict[int, list[float]],
        backend: str = "ollama",
    ) -> list[RankedResult]:
        pref_vec = await self._embed_query(user_preference, backend)

        scored: list[RankedResult] = []
        for i, r in enumerate(results):
            similarity = _cosine(pref_vec, embeddings[i])
            axis_scores = _compute_gap_scores(r, constraints)
            c_count = consensus.get(i, 1)
            rationale = _explain(
                similarity, axis_scores, intent, c_count, constraints, r.sponsored
            )
            scored.append(_assemble_score(
                r, similarity, axis_scores, c_count, intent, rationale,
            ))
        return scored


# --- Keyword fallback ---------------------------------------------------------

def _score_with_keywords(
    user_preference: str,
    results: list[ProviderResult],
    intent: QueryIntent,
    constraints: QueryConstraints,
    consensus: dict[int, int],
) -> list[RankedResult]:
    pref_tokens = _tokens(user_preference)
    scored: list[RankedResult] = []
    for i, r in enumerate(results):
        haystack = _tokens(f"{r.title} {r.snippet}")
        overlap = len(pref_tokens & haystack)
        similarity = overlap / max(len(pref_tokens), 1)
        axis_scores = _compute_gap_scores(r, constraints)
        c_count = consensus.get(i, 1)
        rationale = (
            f"[keyword fallback] {overlap} terms matched"
            + (f", {intent.value} axis" if intent != QueryIntent.GENERAL else "")
            + (f", {c_count} providers agreed" if c_count > 1 else "")
            + (" — sponsored penalty applied" if r.sponsored else "")
        )
        scored.append(_assemble_score(
            r, similarity, axis_scores, c_count, intent, rationale,
        ))
    return scored


# --- Shared score assembly ----------------------------------------------------

def _assemble_score(
    r: ProviderResult,
    similarity: float,
    axis_scores: dict[str, float],
    consensus_count: int,
    intent: QueryIntent,
    rationale: str,
) -> RankedResult:
    """Combine the four scoring components into a RankedResult.

    The embedding and keyword paths differ only in how they compute
    `similarity` (cosine vs. token overlap) and `rationale`. Everything after
    that — axis weighting, consensus factor, sponsored penalty, and the final
    formula — is identical, and lived as two byte-for-byte copies that
    repeatedly drifted when one was edited and the other missed. This is the
    single owner of that arithmetic; both paths pass their already-computed
    similarity and rationale in.

    Final formula (see the module docstring): every term is weight x (0-1
    value), the three weights sum to 1.0, and the sponsored penalty is
    subtracted last so an ad is demoted regardless of how well it matches.
    """
    scored = _axis_scored_mask(r)
    axis_bonus = _axis_bonus(axis_scores, intent, scored)
    c_factor = min(consensus_count - 1, 2) / 2
    penalty = SPONSORED_PENALTY if r.sponsored else 0.0

    final_score = (
        W_SIMILARITY * similarity
        + W_AXIS * axis_bonus
        + W_CONSENSUS * c_factor
        - penalty
    )

    return RankedResult(
        result=r,
        score=round(final_score, 4),
        rationale=rationale,
        axis_scores=axis_scores,
        consensus_count=consensus_count,
        axis_scored=scored,
    )


# --- Hard constraint filtering ------------------------------------------------

def _apply_hard_constraints(
    results: list[ProviderResult],
    c: QueryConstraints,
) -> list[ProviderResult]:
    """Remove results that clearly violate hard constraints.

    Only filters when the result has data for that axis AND the violation
    is significant (>25% over budget, below min rating). Results with no
    data for an axis pass through — we don't penalize missing data.
    """
    filtered = []
    for r in results:
        # Budget: reject if price is more than 25% over budget
        if c.budget is not None and r.price is not None:
            if r.price > c.budget * 1.25:
                logger.debug("Hard filter: %s ($%.2f) exceeds budget $%.2f", r.title, r.price, c.budget)
                continue
        # Rating: reject if rating is more than 0.5 stars below minimum
        if c.min_rating is not None and r.rating is not None:
            if r.rating < c.min_rating - 0.5:
                logger.debug("Hard filter: %s (%.1f★) below min rating %.1f★", r.title, r.rating, c.min_rating)
                continue
        filtered.append(r)

    # Never return empty — if everything got filtered, return all results
    # (better to show something than nothing)
    return filtered if filtered else results


# --- Real-gap axis scoring ----------------------------------------------------

def _compute_gap_scores(r: ProviderResult, c: QueryConstraints) -> dict[str, float]:
    """Compute normalised 0-1 scores for each P axis using real constraint gaps.

    Gap convention: negative gap = candidate meets or beats the constraint.
    We map gap → score so that meeting the constraint gives 1.0 and badly
    missing it gives 0.0.

    When no constraint is set for an axis but the candidate has data, we score
    the raw value on an absolute scale. When the candidate has NO data for an
    axis, the score is a neutral 0.5 placeholder — but _axis_scored_mask()
    reports that axis as unscored so _axis_bonus() can drop it from the
    weighted average instead of averaging the placeholder in. See the note in
    _axis_bonus about why "missing" must not mean "mediocre".
    """

    # P1 — Price: lower is better
    if c.budget is not None and r.price is not None:
        gap = r.price - c.budget          # negative = under budget
        # Map: gap=-budget (free) → 1.0, gap=0 → 0.75, gap=budget → 0.0
        p1 = max(0.0, min(1.0, 0.75 - (gap / max(c.budget, 1.0)) * 0.75))
    elif r.price is not None:
        # No budget set — score by absolute price (lower = better, $100 ceiling)
        p1 = max(0.0, 1.0 - (r.price / 100.0))
    else:
        p1 = 0.5

    # P2 — Distance: closer is better
    if c.max_distance is not None and r.distance is not None:
        gap = r.distance - c.max_distance  # negative = within range
        p2 = max(0.0, min(1.0, 0.75 - (gap / max(c.max_distance, 0.1)) * 0.75))
    elif r.distance is not None:
        # No distance constraint — score by absolute distance (5 mile ceiling)
        p2 = max(0.0, 1.0 - (r.distance / 5.0))
    else:
        p2 = 0.5

    # P3 — Rating: higher is better
    if c.min_rating is not None and r.rating is not None:
        gap = c.min_rating - r.rating      # negative = meets or exceeds threshold
        if gap <= 0:
            headroom = max(5.0 - c.min_rating, 0.1)
            p3 = 0.6 + 0.4 * min(-gap / headroom, 1.0)
        else:
            p3 = max(0.0, 0.6 - gap * 1.2)
    elif r.rating is not None:
        p3 = r.rating / 5.0
    else:
        p3 = 0.5

    return {
        "P1_price":    round(p1, 3),
        "P2_distance": round(p2, 3),
        "P3_rating":   round(p3, 3),
    }


def _axis_scored_mask(r: ProviderResult) -> dict[str, bool]:
    """Report which axes the candidate actually has data for.

    An axis is scoreable only when the provider supplied a value. This is not
    a detail: AI providers (OpenAI, Gemini, Ollama, WatsonX) never return
    distance — they have no location context and prompt.py deliberately forbids
    them from inventing one — and search providers like Brave return no
    structured fields at all.
    """
    return {
        "P1_price":    r.price is not None,
        "P2_distance": r.distance is not None,
        "P3_rating":   r.rating is not None,
    }


# Intent → per-axis weight. Weights are renormalised over whichever axes have
# data, so these are ratios rather than absolute shares.
_INTENT_AXIS_WEIGHTS: dict[QueryIntent, dict[str, float]] = {
    QueryIntent.PRICE:    {"P1_price": 0.60, "P2_distance": 0.20, "P3_rating": 0.20},
    QueryIntent.DISTANCE: {"P1_price": 0.20, "P2_distance": 0.60, "P3_rating": 0.20},
    QueryIntent.RATING:   {"P1_price": 0.20, "P2_distance": 0.20, "P3_rating": 0.60},
    QueryIntent.GENERAL:  {"P1_price": 1 / 3, "P2_distance": 1 / 3, "P3_rating": 1 / 3},
}

# The axis each intent is "about". GENERAL has no dominant axis, so it is
# absent here and renormalises freely across whatever data exists.
_INTENT_AXIS: dict[QueryIntent, str] = {
    QueryIntent.PRICE:    "P1_price",
    QueryIntent.DISTANCE: "P2_distance",
    QueryIntent.RATING:   "P3_rating",
}


def _axis_bonus(
    axis_scores: dict[str, float],
    intent: QueryIntent,
    scored: dict[str, bool] | None = None,
) -> float:
    """Weighted axis score over the axes that actually have data.

    Intent shifts the weights so the dominant axis gets more influence, but
    the other axes still count. This handles "cheap AND nearby" queries
    correctly instead of winner-take-all on a single axis.

    Base weights per intent (dominant / secondary / tertiary):
      PRICE:    60% / 20% / 20%
      DISTANCE: 60% / 20% / 20%
      RATING:   60% / 20% / 20%
      GENERAL:  33% / 33% / 33%

    `scored` marks which axes have real data. Weights are renormalised over
    only those axes, so a result with price and rating but no distance is
    judged on price and rating alone rather than being pulled toward the
    neutral 0.5 placeholder.

    Why this matters: treating "no data" as a mediocre 0.5 lets a bare search
    link with no structured fields score 0.50 on every axis, landing within
    ~0.11 of a result that perfectly satisfies every constraint — less than
    the 0.20 sponsored penalty. Missing data must not be scoreable as
    mediocrity, or the axis system rewards providers for telling us nothing.

    Renormalisation alone would over-reward silence on the axis the user
    actually asked about: on a DISTANCE query, a result with no distance would
    have its 60% redistributed onto price and rating, letting it beat a result
    that really is 0.05 mi away. So when the intent axis itself is unscored we
    substitute a neutral 0.5 for it and keep its weight in the denominator.
    The result is judged on what it disclosed, but it cannot win the axis it
    stayed silent on.

    When no axis has data, there is nothing to rank on and we return the
    neutral 0.5. `scored=None` scores all three axes, preserving the old
    behaviour for callers that have no ProviderResult to inspect.
    """
    weights = _INTENT_AXIS_WEIGHTS[intent]

    if scored is None:
        scored = {axis: True for axis in weights}

    if not any(scored.get(axis) for axis in weights):
        return 0.5

    intent_axis = _INTENT_AXIS.get(intent)

    total_weight = 0.0
    weighted_sum = 0.0
    for axis, w in weights.items():
        if scored.get(axis):
            weighted_sum += w * axis_scores[axis]
            total_weight += w
        elif axis == intent_axis:
            # Unscored intent axis: neutral value, weight retained so the
            # silence is not converted into credit on the other axes.
            weighted_sum += w * 0.5
            total_weight += w

    return weighted_sum / total_weight if total_weight else 0.5


# --- Fuzzy consensus clustering -----------------------------------------------

def _build_fuzzy_consensus(
    results: list[ProviderResult],
    embeddings: dict[int, list[float]],
    threshold: float,
) -> dict[int, int]:
    """Cluster results by embedding similarity, then count providers per cluster.

    Two results are in the same cluster if their embeddings are above
    `threshold` similar AND they come from different providers. The cluster
    representative is the normalised title of the first member seen.
    """
    n = len(results)
    # Union-find cluster assignment
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if results[i].provider == results[j].provider:
                continue  # don't cluster results from the same provider
            if _cosine(embeddings[i], embeddings[j]) >= threshold:
                union(i, j)

    # Count distinct providers per cluster
    cluster_providers: dict[int, set[str]] = {}
    for i, r in enumerate(results):
        root = find(i)
        cluster_providers.setdefault(root, set()).add(r.provider)

    # Map each normalised title to the provider count of its cluster
    counts: dict[int, int] = {}
    for i, r in enumerate(results):
        counts[i] = len(cluster_providers[find(i)])
    return counts


def _collapse_duplicates(scored: list[RankedResult]) -> list[RankedResult]:
    """Keep one entry per real-world venue, best-scoring first.

    Consensus clustering deliberately refuses to group results from the same
    provider, so one provider cannot manufacture its own agreement. That is the
    right rule for *counting* providers and the wrong one for deduplicating
    output: nothing collapsed the duplicates, so a venue every provider named
    took a slot per mention. Observed live — "Shake Shack" held three of five.

    The consensus bonus makes it worse rather than better: agreeing providers
    all score identically, so the copies land adjacent at the top and crowd out
    the rest of the ranking.

    Runs after scoring so the surviving copy keeps the consensus count it
    earned, and after the sort so the copy we keep is the best-scoring one.
    Matching is on the normalised title, which is the same key the keyword
    consensus path already trusts for identity.
    """
    seen: set[str] = set()
    unique: list[RankedResult] = []
    for r in scored:
        key = _normalise(r.result.title)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(r)
    dropped = len(scored) - len(unique)
    if dropped:
        logger.info("Collapsed %d duplicate result(s) before top-k", dropped)
    return unique


def _build_token_consensus(results: list[ProviderResult]) -> dict[int, int]:
    """Fallback consensus: simple normalised-title exact match across providers."""
    providers_by_title: dict[str,set[str]] = {}
    for r in results:
        providers_by_title.setdefault(_normalise(r.title), set()).add(r.provider)
    return {i: len(providers_by_title[_normalise(r.title)]) for i, r in enumerate(results)}


# --- Maths & helpers ----------------------------------------------------------

def _tokens(text: str) -> set[str]:
    return {t.lower().strip(".,!?;:") for t in text.split() if len(t) > 2}


def _normalise(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum() or c.isspace()).strip()


def _cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _explain(
    similarity: float,
    axis_scores: dict[str, float],
    intent: QueryIntent,
    consensus_count: int,
    constraints: QueryConstraints,
    sponsored: bool | None,
) -> str:
    tag  = "strong match" if similarity > 0.7 else "partial match" if similarity > 0.4 else "weak match"
    base = f"{tag} (similarity {similarity:.2f})"

    if intent == QueryIntent.PRICE and constraints.budget is not None:
        base += f", P1 price score {axis_scores['P1_price']:.2f} (budget ${constraints.budget})"
    elif intent == QueryIntent.DISTANCE and constraints.max_distance is not None:
        base += f", P2 distance score {axis_scores['P2_distance']:.2f} (within {constraints.max_distance} mi)"
    elif intent == QueryIntent.RATING and constraints.min_rating is not None:
        base += f", P3 rating score {axis_scores['P3_rating']:.2f} (min {constraints.min_rating}★)"
    elif intent != QueryIntent.GENERAL:
        axis_key = {"price": "P1_price", "distance": "P2_distance", "rating": "P3_rating"}[intent.value]
        base += f", {intent.value} axis {axis_scores[axis_key]:.2f}"

    if consensus_count > 1:
        base += f", {consensus_count} providers agreed"
    if sponsored:
        base += f" — sponsored, penalty {SPONSORED_PENALTY} applied"

    return base
