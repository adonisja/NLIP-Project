"""Mock provider — returns canned results for deterministic testing.

Use this when you want to demo the ranking logic without depending on live
provider APIs, and in unit tests. Canned results are intentionally a mix of
sponsored-looking and genuinely useful entries so the ranker has something
meaningful to do.
"""

from angel_filter.providers.base import BaseProvider, ProviderResult


class MockProvider(BaseProvider):
    name = "mock"

    def __init__(self, canned_results: list[ProviderResult] | None = None):
        self.canned_results = canned_results or _default_shopping_results()

    async def query(self, user_query: str, max_results: int = 10) -> list[ProviderResult]:
        return self.canned_results[:max_results]


def _default_shopping_results() -> list[ProviderResult]:
    """Toilet-paper-style demo set matching Instructor Z's example."""
    return [
        ProviderResult(
            title="SponsorCo Ultra-Plush 24-pack",
            snippet="Our #1 featured partner. Premium quilted 3-ply.",
            url="https://example.com/sponsorco",
            provider="mock",
            rank_in_provider=0,
            price=39.99,
            sponsored=True,
        ),
        ProviderResult(
            title="StoreBrand Basic 30-pack",
            snippet="2-ply, no frills, strong reviews for everyday use.",
            url="https://example.com/storebrand",
            provider="mock",
            rank_in_provider=1,
            price=18.49,
            sponsored=False,
        ),
        ProviderResult(
            title="EcoSoft Bamboo 12-pack",
            snippet="Plastic-free, compostable packaging, 3-ply.",
            url="https://example.com/ecosoft",
            provider="mock",
            rank_in_provider=2,
            price=22.00,
            sponsored=False,
        ),
        ProviderResult(
            title="MegaDeal 48-pack warehouse",
            snippet="Bulk value, 1-ply, best price per sheet.",
            url="https://example.com/megadeal",
            provider="mock",
            rank_in_provider=3,
            price=29.99,
            sponsored=False,
        ),
    ]
