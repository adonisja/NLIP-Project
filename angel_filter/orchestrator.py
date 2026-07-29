"""Orchestrator — runs every registered provider in parallel, collects results,
then hands the combined pile to the ranker.

Responsible for three things the ranker doesn't own:
  1. Fan-out — fire all providers simultaneously, isolate failures.
  2. Intent detection — classify the query as price / distance / rating / general.
  3. Constraint extraction — parse explicit values ($15, 5 miles, 4 stars) from
     the query so the ranker can compute real axis gaps instead of neutral proxies.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from angel_filter.constraints import QueryConstraints, extract_constraints
from angel_filter.geocode import (
    describe_location,
    enrich_distances,
    shortlist_for_enrichment,
)
from angel_filter.providers.base import BaseProvider, ProviderError, ProviderResult
from angel_filter.ranker import QueryIntent, RankedResult, Ranker, detect_intent

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResponse:
    ranked: list[RankedResult]
    providers_used: list[str]
    providers_failed: list[str]
    intent: QueryIntent
    constraints: QueryConstraints
    axis_scores: dict[str, float] | None = None  # P1/P2/P3 of the top result


class Orchestrator:
    def __init__(self, providers: list[BaseProvider], ranker: Ranker | None = None):
        if not providers:
            raise ValueError("Orchestrator needs at least one provider.")
        self.providers = providers
        self.ranker = ranker or Ranker()

    async def handle_query(
        self,
        user_query: str,
        user_preference: str | None = None,
        top_k: int = 5,
        user_lat: float | None = None,
        user_lng: float | None = None,
        intent: QueryIntent | None = None,
        override_constraints: QueryConstraints | None = None,
        context_prefix: str = "",
    ) -> OrchestratorResponse:
        """Run the full pipeline: extract constraints → detect intent → fan out → rank.

        `intent` overrides keyword detection when the caller states a priority
        explicitly (the UI's axis picker). Left as None, the intent is inferred
        from the query text exactly as before.

        `override_constraints` replaces the ones parsed from the text. A
        multi-turn refinement ("cheaper than that") computes its bounds from the
        previous turn rather than from this turn's words, which carry the
        adjustment but no numbers to parse.

        `context_prefix` is prior conversation prepended to provider prompts,
        used only for follow-ups the refinement parser did not recognise.
        """

        # Combine query + preference so signals in either field are captured
        full_text   = f"{user_query} {user_preference or ''}".strip()
        if intent is None:
            intent = detect_intent(full_text)
        constraints = override_constraints or extract_constraints(full_text)
        constraints.context_prefix = context_prefix
        # Location comes from the request, not the query text — attach it so
        # location-aware providers (Google Places) can measure distance.
        constraints.user_lat = user_lat
        constraints.user_lng = user_lng

        # Resolve the coordinates to a neighbourhood *before* fan-out so the AI
        # providers can put it in their prompts. They previously got no location
        # at all and suggested venues from anywhere, which is why so many failed
        # to geocode near the user. Best-effort: None just omits it, as before.
        if user_lat is not None and user_lng is not None:
            try:
                constraints.user_locality = await describe_location(user_lat, user_lng)
            except Exception:
                logger.exception("Locality lookup failed; querying without it")

        logger.info(
            "Intent: %s | Constraints: budget=%s, distance=%s, rating=%s | Locality: %s",
            intent.value,
            constraints.budget,
            constraints.max_distance,
            constraints.min_rating,
            constraints.user_locality or "unknown",
        )

        # Fan out to all providers in parallel, passing constraints so AI
        # providers can inject them directly into their prompts
        tasks = [self._safe_query(p, user_query, constraints) for p in self.providers]
        per_provider = await asyncio.gather(*tasks)

        all_results: list[ProviderResult] = []
        used: list[str] = []
        failed: list[str] = []
        for provider, outcome in zip(self.providers, per_provider):
            if outcome is None:
                failed.append(provider.name)
            else:
                used.append(provider.name)
                all_results.extend(outcome)

        if not all_results:
            return OrchestratorResponse(
                ranked=[],
                providers_used=used,
                providers_failed=failed,
                intent=intent,
                constraints=constraints,
            )

        # Most providers cannot report distance — the AI ones are forbidden from
        # guessing it and search results have no coordinates — so before this the
        # P2 axis was populated only by Google Places. Resolve real coordinates
        # for results that named a venue, then measure. Best-effort: anything
        # unresolved keeps distance=None and stays honestly unscored on P2.
        if constraints.user_lat is not None and constraints.user_lng is not None:
            try:
                # Geocoding every result would pay for ~35 answers nobody reads:
                # the fan-out returns ~40 and the user sees top_k. Spend the
                # calls on the candidates that could plausibly place, then let
                # the ranker score the full set — a result outside the shortlist
                # simply keeps distance=None, exactly as if no provider knew it.
                candidates = shortlist_for_enrichment(
                    all_results, user_query, constraints
                )
                await enrich_distances(
                    candidates, constraints.user_lat, constraints.user_lng
                )
            except Exception:
                # Enrichment is an enhancement, never a reason to fail a query.
                logger.exception("Distance enrichment failed; continuing unenriched")

        ranked = await self.ranker.rank(
            user_preference or user_query,
            all_results,
            top_k=top_k,
            intent=intent,
            constraints=constraints,
        )

        return OrchestratorResponse(
            ranked=ranked,
            providers_used=used,
            providers_failed=failed,
            intent=intent,
            constraints=constraints,
            axis_scores=ranked[0].axis_scores if ranked else None,
        )

    async def _safe_query(
        self,
        provider: BaseProvider,
        user_query: str,
        constraints: QueryConstraints | None = None,
    ) -> list[ProviderResult] | None:
        try:
            return await provider.query(user_query, constraints=constraints)
        except ProviderError as exc:
            logger.warning("Provider %s failed: %s", provider.name, exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Provider %s raised unexpected error: %s", provider.name, exc)
            return None
