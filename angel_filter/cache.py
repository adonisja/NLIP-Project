"""Query result cache, TTL-based, kept in the shared store (see state.py).

Keyed by (query, preference) normalized to lowercase stripped strings.
Entries expire after TTL_SECONDS: in Redis the key itself expires, and in
memory an expired entry is dropped the next time it is read.
"""

from __future__ import annotations

import hashlib
from typing import Any

from angel_filter.state import PREFIX, get_store

TTL_SECONDS = 3 * 60 * 60  # 3 hours
MAX_HISTORY = 10

_ENTRY = PREFIX + "cache:"
_HISTORY = PREFIX + "cache-history"


class QueryCache:
    """Async facade over the store. Holds no state of its own."""

    def _key(self, query: str, preference: str | None) -> str:
        combined = f"{query.strip().lower()}|{(preference or '').strip().lower()}"
        return _ENTRY + hashlib.md5(combined.encode()).hexdigest()

    async def get(self, query: str, preference: str | None) -> dict[str, Any] | None:
        entry = await get_store().get(self._key(query, preference))
        return entry["payload"] if entry else None

    async def set(self, query: str, preference: str | None, payload: dict[str, Any]) -> None:
        key = self._key(query, preference)
        store = get_store()
        # The query text travels with the entry so /history can show it.
        await store.set(key, {"query": query.strip(), "payload": payload}, ttl=TTL_SECONDS)
        await store.push_recent(_HISTORY, key, MAX_HISTORY)

    async def history(self) -> list[str]:
        """The MAX_HISTORY most recent query strings still cached, freshest first."""
        store = get_store()
        keys = await store.recent(_HISTORY)
        entries = await store.get_many(keys)
        return [e["query"] for e in entries if e]

    async def stats(self) -> dict[str, int]:
        store = get_store()
        return {
            "entries": await store.count_prefix(_ENTRY),
            "history": len(await self.history()),
        }

    async def clear(self) -> None:
        store = get_store()
        await store.delete_prefix(_ENTRY)
        await store.delete_prefix(_HISTORY)


# Module-level singleton shared across the app
CACHE = QueryCache()
