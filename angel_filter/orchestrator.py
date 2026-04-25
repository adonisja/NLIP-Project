"""Orchestrator — runs every registered provider in parallel, collects results,
then hands the combined pile to the ranker.

This is the function the NLIP handler calls. Keeping the orchestration logic
separate from the NLIP plumbing makes it trivially unit-testable (no server
required) and easy for the team to extend: to add a provider, just register
it here; to change how results combine, edit the ranker.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from angel_filter.providers.base import BaseProvider, ProviderError, ProviderResult
from angel_filter.ranker import RankedResult, Ranker

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResponse:
    ranked: list[RankedResult]
    providers_used: list[str]
    providers_failed: list[str]


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
    ) -> OrchestratorResponse:
        """Run the full pipeline: fan out, collect, rank.

        Args:
            user_query: what the user typed (goes to the providers).
            user_preference: optional clarifier like "I care about price, not speed."
                If omitted we rank against the query text itself.
            top_k: how many ranked results to return.
        """
        # Fan out to all providers in parallel. A slow provider does not block
        # the fast ones, and a failing one does not poison the rest.
        tasks = [self._safe_query(p, user_query) for p in self.providers]
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
            return OrchestratorResponse(ranked=[], providers_used=used, providers_failed=failed)

        ranked = await self.ranker.rank(
            user_preference or user_query,
            all_results,
            top_k=top_k,
        )
        return OrchestratorResponse(
            ranked=ranked,
            providers_used=used,
            providers_failed=failed,
        )

    async def _safe_query(
        self,
        provider: BaseProvider,
        user_query: str,
    ) -> list[ProviderResult] | None:
        try:
            return await provider.query(user_query)
        except ProviderError as exc:
            logger.warning("Provider %s failed: %s", provider.name, exc)
            return None
        except Exception as exc:  # noqa: BLE001 — we really do want to absorb everything here
            logger.exception("Provider %s raised unexpected error: %s", provider.name, exc)
            return None
