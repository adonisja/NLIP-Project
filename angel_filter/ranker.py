"""Ranker — the brain of the Angel Filter.

Takes the user's stated preferences and a flat list of provider results, and
re-orders them by how well each result matches the user's request. Uses
embeddings from a local Ollama model so nothing leaves the user's machine.

Scoring is:
    final_score = cosine_similarity(user_pref, result_text)
                  - SPONSORED_PENALTY if result.sponsored

SPONSORED_PENALTY is the whole point of the project — it's why we're not just
re-ranking by relevance, we're explicitly de-weighting ad content.

If Ollama is unavailable (no model pulled, daemon not running) the ranker
falls back to a keyword-overlap baseline so the demo still runs. The fallback
is clearly marked in the response so it's never mistaken for the real thing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from angel_filter.providers.base import ProviderResult

logger = logging.getLogger(__name__)

SPONSORED_PENALTY: float = 0.15
DEFAULT_EMBED_MODEL: str = "nomic-embed-text"  # small, fast, widely available on Ollama


@dataclass
class RankedResult:
    result: ProviderResult
    score: float
    rationale: str  # short human-readable explanation; the UI will show this


class Ranker:
    def __init__(self, embed_model: str = DEFAULT_EMBED_MODEL):
        self.embed_model = embed_model
        self._ollama_available: bool | None = None  # lazily probed

    async def rank(
        self,
        user_preference: str,
        results: list[ProviderResult],
        top_k: int = 5,
    ) -> list[RankedResult]:
        """Return the top_k results, ordered best-first."""
        if not results:
            return []

        if await self._has_ollama():
            scored = await self._score_with_embeddings(user_preference, results)
        else:
            logger.warning("Ollama unavailable; using keyword-overlap fallback.")
            scored = _score_with_keywords(user_preference, results)

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    # -- private -------------------------------------------------------------

    async def _has_ollama(self) -> bool:
        """Probe Ollama once and cache the result for the lifetime of this ranker."""
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            import ollama  # lazy import so the module loads even if ollama isn't installed
            # ollama.embeddings is a thin sync wrapper; calling it as a smoke test.
            ollama.embeddings(model=self.embed_model, prompt="ping")
            self._ollama_available = True
        except Exception as exc:  # ImportError, ConnectionError, ResponseError, etc.
            logger.info("Ollama probe failed: %s", exc)
            self._ollama_available = False
        return self._ollama_available

    async def _score_with_embeddings(
        self,
        user_preference: str,
        results: list[ProviderResult],
    ) -> list[RankedResult]:
        import ollama

        pref_vec = ollama.embeddings(
            model=self.embed_model,
            prompt=user_preference,
        )["embedding"]

        scored: list[RankedResult] = []
        for r in results:
            result_text = f"{r.title}. {r.snippet}"
            res_vec = ollama.embeddings(
                model=self.embed_model,
                prompt=result_text,
            )["embedding"]
            similarity = _cosine(pref_vec, res_vec)
            score = similarity - (SPONSORED_PENALTY if r.sponsored else 0.0)
            scored.append(RankedResult(
                result=r,
                score=score,
                rationale=_explain(similarity, r.sponsored),
            ))
        return scored


def _score_with_keywords(
    user_preference: str,
    results: list[ProviderResult],
) -> list[RankedResult]:
    """Very simple fallback: token overlap between preference and title+snippet."""
    pref_tokens = _tokens(user_preference)
    scored: list[RankedResult] = []
    for r in results:
        haystack = _tokens(f"{r.title} {r.snippet}")
        overlap = len(pref_tokens & haystack)
        base_score = overlap / max(len(pref_tokens), 1)
        score = base_score - (SPONSORED_PENALTY if r.sponsored else 0.0)
        scored.append(RankedResult(
            result=r,
            score=score,
            rationale=(
                f"[keyword fallback] {overlap} preference terms matched"
                + (" — sponsored penalty applied" if r.sponsored else "")
            ),
        ))
    return scored


def _tokens(text: str) -> set[str]:
    return {t.lower().strip(".,!?;:") for t in text.split() if len(t) > 2}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _explain(similarity: float, sponsored: bool | None) -> str:
    tag = "strong match" if similarity > 0.7 else "partial match" if similarity > 0.4 else "weak match"
    base = f"{tag} (similarity {similarity:.2f})"
    if sponsored:
        base += f" — sponsored, penalty {SPONSORED_PENALTY} applied"
    return base
