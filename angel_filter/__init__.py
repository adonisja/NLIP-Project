"""Angel Filter — a local proxy agent that re-ranks responses from multiple
AI / search providers against user preferences using a local LLM via Ollama.

Implements the NLIP (Natural Language Interaction Protocol) standards 430 & 431.
"""

# Load .env at package import so every submodule (server, auth, limits, ...)
# sees the same environment regardless of how it's invoked. The file is
# searched at the repo root (one directory above this package). We use an
# explicit path so `poetry run` from inside angel_filter/ behaves the same
# as running from the repo root. Silently skipped if python-dotenv isn't
# installed yet — useful for very-first-clone bootstrap.
try:
    from pathlib import Path as _Path
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(_Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    # python-dotenv not installed yet. Real env vars still work; users
    # who want .env-file behavior should run `poetry install`.
    pass

__version__ = "0.1.0"
