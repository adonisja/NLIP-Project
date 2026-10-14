"""Sponsored-content detection.

No live provider reports ads, so before this the sponsored penalty — the
project's thesis — only ever fired on test data. These pin what each rule
catches and, just as importantly, what it must leave alone: a false positive
penalises an organic result.
"""

from __future__ import annotations

import pytest

from angel_filter.orchestrator import Orchestrator
from angel_filter.providers.base import BaseProvider, ProviderResult
from angel_filter.ranker import Ranker
from angel_filter.sponsorship import flag_sponsored, sponsorship_reason


def _r(title="Joe's Pizza", snippet="Classic slices.", url=None) -> ProviderResult:
    return ProviderResult(title=title, snippet=snippet, url=url, provider="brave", sponsored=False)


@pytest.mark.parametrize("url, expected", [
    ("https://adclick.g.doubleclick.net/pcs/click?xai=1", "doubleclick.net"),
    ("https://www.googleadservices.com/pagead/aclk?sa=L", "googleadservices.com"),
    ("https://www.bing.com/aclk?ld=e8", "bing.com/aclk"),
    ("https://trc.taboola.com/redirect?x=1", "taboola.com"),
    ("https://joes.example/menu?gclid=abc123", "gclid"),
    ("https://joes.example/menu?msclkid=abc", "msclkid"),
    ("https://joes.example/?utm_source=google&utm_medium=cpc", "utm_medium=cpc"),
    ("https://joes.example/?utm_medium=Paid_Social", "utm_medium=paid_social"),
    ("https://shop.example/item?cjevent=9f2", "cjevent"),
    ("https://shop.example/item?irclickid=x1", "irclickid"),
    ("https://www.amazon.com/dp/B000?tag=bestlunch-20", "Amazon affiliate tag"),
])
def test_url_rules(url, expected):
    reason = sponsorship_reason(_r(url=url))
    assert reason and expected in reason


@pytest.mark.parametrize("title, snippet, expected", [
    ("Sponsored · Taco Haven", "Great tacos.", "sponsored"),
    ("Ad: Burger Barn", "Burgers.", "ad"),
    ("[Ad] Burger Barn", "Burgers.", "ad"),
    ("Ad - Burger Barn", "Burgers.", "ad"),
    ("Promoted — Taco Haven", "Tacos.", "promoted"),
    ("Best lunch in SoHo", "Sponsored content from Burger Barn.", "Sponsored content"),
    ("Best lunch in SoHo", "In paid partnership with Taco Haven.", "paid partnership"),
    ("Taco Haven review", "Loved it #ad", "#ad"),
    ("10 best pizza spots", "This post contains affiliate links.", "contains affiliate links"),
    ("10 best pizza spots", "We may earn a commission when you buy.", "earn a commission"),
])
def test_text_rules(title, snippet, expected):
    reason = sponsorship_reason(_r(title=title, snippet=snippet))
    assert reason and expected.lower() in reason.lower()


@pytest.mark.parametrize("title, snippet, url", [
    # Event sponsorship is ordinary copy, not a paid placement.
    ("Smorgasburg", "A food festival sponsored by local businesses.", None),
    # Words that merely start with "ad".
    ("Adams Diner", "Advertised specials daily.", None),
    ("Adriatic Grill", "Address: 12 Main St.", None),
    ("Taco Haven", "Great for an #adventure.", None),
    ("Ad-free lunch spots", "No ads, just food.", None),
    ("Sponsored-free zone", "Independent reviews.", None),
    # Ordinary tracking that says nothing about payment.
    ("Joe's Pizza", "Slices.", "https://joes.example/?utm_source=newsletter&utm_medium=email"),
    ("Joe's Pizza", "Slices.", "https://joes.example/?ref=homepage"),
    # "tag" is only an affiliate signal on Amazon.
    ("Joe's Pizza", "Slices.", "https://joes.example/menu?tag=lunch"),
    ("Amazon Go", "Grab and go.", "https://www.amazon.com/b?node=16008589011"),
    # Organic Bing and Google pages.
    ("Joe's Pizza", "Slices.", "https://www.bing.com/maps?q=joes+pizza"),
    ("Joe's Pizza", "Slices.", "https://www.google.com/maps/place/Joes+Pizza"),
    # A malformed URL must not raise.
    ("Joe's Pizza", "Slices.", "http://[::1"),
])
def test_organic_results_are_not_flagged(title, snippet, url):
    assert sponsorship_reason(_r(title=title, snippet=snippet, url=url)) is None


def test_flagging_records_the_reason_and_trusts_existing_flags():
    provider_flagged = ProviderResult("Promo Place", "x", provider="p", sponsored=True)
    detected = _r(url="https://joes.example/?gclid=1")
    organic = _r()

    assert flag_sponsored([provider_flagged, detected, organic]) == 1
    assert provider_flagged.sponsored and provider_flagged.sponsored_reason is None
    assert detected.sponsored and "gclid" in detected.sponsored_reason
    assert organic.sponsored is False


class _AdProvider(BaseProvider):
    name = "brave"

    async def query(self, user_query, max_results=10, constraints=None):
        return [
            ProviderResult("Burger Barn", "lunch burgers", provider="brave",
                           url="https://burgerbarn.example/?utm_medium=cpc", sponsored=False),
            ProviderResult("Joe's Diner", "lunch burgers", provider="brave",
                           url="https://joesdiner.example/", sponsored=False),
        ]


@pytest.mark.asyncio
async def test_detected_ads_are_penalised_end_to_end():
    """Two otherwise identical results: the paid click ranks below the organic one."""
    ranker = Ranker()
    ranker._ollama_available = False
    ranker._openai_available = False
    response = await Orchestrator([_AdProvider()], ranker=ranker).handle_query("lunch burgers")

    titles = [r.result.title for r in response.ranked]
    assert titles == ["Joe's Diner", "Burger Barn"]
    ad = response.ranked[1]
    assert ad.result.sponsored and ad.score_breakdown["sponsored_penalty"] == 0.2
    assert "utm_medium=cpc" in ad.result.sponsored_reason


def test_the_reason_reaches_api_clients():
    pytest.importorskip("nlip_sdk.nlip")
    import angel_filter.server as server
    from angel_filter.constraints import QueryConstraints
    from angel_filter.orchestrator import OrchestratorResponse
    from angel_filter.ranker import QueryIntent, RankedResult

    r = _r(url="https://x.example/?gclid=1")
    flag_sponsored([r])
    payload = server._serialize_response(OrchestratorResponse(
        ranked=[RankedResult(result=r, score=0.3, rationale="why", axis_scores={}, consensus_count=1)],
        providers_used=["brave"], providers_failed=[],
        intent=QueryIntent.GENERAL, constraints=QueryConstraints(),
    ))
    assert "gclid" in payload["results"][0]["sponsored_reason"]
    assert "[SPONSORED: ad-click tracking (gclid)]" in server._format_reply_from_payload(payload)
