"""DuckDuckGo provider using the public Instant Answer API.

This is the only provider wired up for the Friday demo because it needs no API
key. Google and Bing/Copilot integrations come next; they follow the same
pattern as this file.
"""

from angel_filter.providers.base import BaseProvider, ProviderError, ProviderResult


class DuckDuckGoProvider(BaseProvider):
    """Queries DuckDuckGo's Instant Answer API.

    Docs: https://duckduckgo.com/api
    Caveats:
      - The Instant Answer API is lightweight and only returns structured data
        for queries that match their known topics. For arbitrary queries the
        results list can be empty.
      - For the MVP demo we still get enough hits to prove the fan-out pattern.
      - For richer results we will eventually need a scraping layer or a paid
        API — that is explicitly a May-1st-or-later task, not Friday's demo.
    """

    name = "duckduckgo"
    BASE_URL = "https://api.duckduckgo.com/"

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s

    async def query(self, user_query: str, max_results: int = 10) -> list[ProviderResult]:
        import httpx  # lazy so the rest of the package imports without httpx installed
        params = {
            "q": user_query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # httpx.HTTPError, ValueError(JSON decode), connection errors
            raise ProviderError(f"DuckDuckGo call failed: {exc}") from exc

        results: list[ProviderResult] = []

        # The abstract / main answer, if present, is the most relevant result.
        if data.get("AbstractText"):
            results.append(ProviderResult(
                title=data.get("Heading", user_query),
                snippet=data["AbstractText"],
                url=data.get("AbstractURL") or None,
                provider=self.name,
                rank_in_provider=0,
                raw=data,
            ))

        # RelatedTopics is a mixed list of results and categorized sub-lists.
        for i, topic in enumerate(data.get("RelatedTopics", [])[:max_results]):
            if "Text" in topic:
                results.append(ProviderResult(
                    title=_first_line(topic["Text"]),
                    snippet=topic["Text"],
                    url=topic.get("FirstURL") or None,
                    provider=self.name,
                    rank_in_provider=i + 1,
                    raw=topic,
                ))

        return results[:max_results]


def _first_line(text: str) -> str:
    """Use the first sentence/line as a title when the provider gives us a blob."""
    for sep in (" - ", ". ", "\n"):
        if sep in text:
            return text.split(sep, 1)[0]
    return text[:80]
