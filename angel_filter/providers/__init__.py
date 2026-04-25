"""Provider adapters.

Each adapter takes a user query and returns a list of ProviderResult objects.
All adapters implement the same async interface so they can be fanned out in parallel.
"""

from angel_filter.providers.base import BaseProvider, ProviderResult
from angel_filter.providers.duckduckgo import DuckDuckGoProvider
from angel_filter.providers.mock import MockProvider

__all__ = [
    "BaseProvider",
    "ProviderResult",
    "DuckDuckGoProvider",
    "MockProvider",
]
