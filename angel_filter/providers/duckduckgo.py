"""DuckDuckGo provider using the public Instant Answer API.

This provider needs no API key. It returns search-style results rather than a
hosted AI model answer, but it uses the same ProviderResult shape so it can run
beside Claude, OpenAI, Gemini, and the mock provider.
"""

from angel_filter.providers.base import BaseProvider, ProviderError, ProviderResult


class DuckDuckGoProvider(BaseProvider):
    """Queries DuckDuckGo's Instant Answer API."""

    name = "duckduckgo"
    base_url = "https://api.duckduckgo.com/"

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s

    async def query(self, user_query: str, max_results: int = 10) -> list[ProviderResult]:
        import httpx

        params = {
            "q": user_query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            raise ProviderError(f"DuckDuckGo call failed: {exc}") from exc

        results: list[ProviderResult] = []

        if data.get("AbstractText"):
            results.append(
                ProviderResult(
                    title=data.get("Heading", user_query),
                    snippet=data["AbstractText"],
                    url=data.get("AbstractURL") or None,
                    provider=self.name,
                    rank_in_provider=0,
                    sponsored=False,
                    raw=data,
                )
            )

        for i, topic in enumerate(data.get("RelatedTopics", [])[:max_results]):
            if "Text" not in topic:
                continue
            results.append(
                ProviderResult(
                    title=_first_line(topic["Text"]),
                    snippet=topic["Text"],
                    url=topic.get("FirstURL") or None,
                    provider=self.name,
                    rank_in_provider=i + 1,
                    sponsored=False,
                    raw=topic,
                )
            )

        return results[:max_results]


def _first_line(text: str) -> str:
    for sep in (" - ", ". ", "\n"):
        if sep in text:
            return text.split(sep, 1)[0]
    return text[:80]
