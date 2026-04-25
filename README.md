# Angel Filter

A local proxy agent that queries multiple AI / search providers, ranks their
responses against what the user actually cares about, and penalizes sponsored
content. Uses the [NLIP protocol](https://github.com/nlip-project) for
communication and a local LLM via [Ollama](https://ollama.com) for ranking, so
user queries never leave the machine.

> CUNY capstone project — target demo May 1, final demo May 15.

## Status (as of this scaffold)

| Component | State | Owner |
|---|---|---|
| NLIP server skeleton (`NLIPApplication` / `NLIPSession`) | Wired, pending `poetry install` on a real machine | SWE team |
| FastAPI fallback server (runs without NLIP) | **Working** | — |
| Provider adapter: DuckDuckGo | **Working** (no API key needed) | — |
| Provider adapter: Mock (canned data for demos) | **Working** | — |
| Provider adapter: Google | Not started — next task | SWE team |
| Provider adapter: Bing / Copilot | Not started — next task | SWE team |
| Orchestrator (parallel fan-out, failure isolation) | **Working** | — |
| Ranker (Ollama embeddings + sponsored penalty) | **Working**, with keyword-overlap fallback when Ollama is offline | — |
| Static demo frontend | **Working** (`static/index.html`) | UX teammate to redesign |
| Tests | 3 passing (`tests/test_orchestrator.py`) | — |

"Working" means end-to-end tested locally against the mock provider and against the live DuckDuckGo API.

## Architecture

```
    user (browser)
          │
          │  POST /query  (or NLIP message once nlip_server is installed)
          ▼
    ┌──────────────────────────┐
    │   FastAPI / NLIP server  │       angel_filter/server.py
    └────────────┬─────────────┘
                 │
                 ▼
    ┌──────────────────────────┐
    │       Orchestrator       │       angel_filter/orchestrator.py
    │   (fans out in parallel) │
    └──────┬───────┬───────────┘
           │       │
     ┌─────▼──┐ ┌──▼────┐  ┌───────┐   angel_filter/providers/*.py
     │ DDG    │ │ Mock  │  │ Google│   (more adapters plug in here)
     └─────┬──┘ └──┬────┘  └───┬───┘
           │       │           │
           └───┬───┴───────────┘
               │  normalized ProviderResult list
               ▼
    ┌──────────────────────────┐
    │         Ranker           │       angel_filter/ranker.py
    │  Ollama embeddings +     │
    │  sponsored-content       │
    │  penalty                 │
    └────────────┬─────────────┘
                 │  RankedResult list
                 ▼
             user sees
```

## Setup

Prereqs: Python 3.10+, [Poetry](https://python-poetry.org/), and (optional but
recommended) [Ollama](https://ollama.com) with a pulled embedding model.

```bash
# 1. Clone the repo
git clone <this-repo>
cd angel_filter

# 2. Install deps (includes the three NLIP libraries per Mentor D)
poetry install

# 3. (Optional) pull an embedding model for ranking
ollama pull nomic-embed-text

# 4. Run the server
poetry run python -m angel_filter.server
#    or, for auto-reload during development:
poetry run fastapi dev angel_filter/server.py
```

Then open <http://localhost:8000> in a browser.

### Running without Ollama

The ranker falls back to a keyword-overlap scorer and clearly tags each
ranking explanation with `[keyword fallback]`. The demo still shows the
sponsored-content penalty in action — just without the semantic muscle.

### Running without the NLIP libraries installed

If `poetry install` can't resolve the NLIP packages (they're Git-based and
occasionally in flux), `server.py` detects the missing imports and exposes
a plain FastAPI endpoint at `POST /query` that does the same pipeline work.
The static frontend talks to this endpoint. See `docs/DEV_FALLBACK.md`.

## Using the three NLIP libraries

Per Mentor D's direction the project depends on:

- [`nlip_sdk`](https://github.com/nlip-project/nlip_sdk) — message format, factory helpers
- [`nlip_server`](https://github.com/nlip-project/nlip_server) — `NLIPApplication`, `NLIPSession`, `start_server` (FastAPI-based)
- [`nlip_web`](https://github.com/nlip-project/nlip_web) — reference implementation of a multi-store product search using the above; worth reading for patterns but not a dependency

Our server code lives in `angel_filter/server.py` and follows the echo.py
pattern from `nlip_server`: subclass the two base classes, pass the app to
`start_server`. When `nlip_sdk` and `nlip_server` are importable, the module
uses them; when they aren't, it falls back to a plain FastAPI app.

## Running the tests

```bash
poetry run pytest
```

Three tests live in `tests/test_orchestrator.py`:

1. End-to-end pipeline returns ranked results.
2. Sponsored items are penalized (do not end up #1).
3. One failing provider does not break the pipeline for the others.

None of the tests require network or Ollama — they use the mock provider and
the keyword-fallback ranker, making them fast and deterministic.

## What to build next (after Friday)

Ordered roughly by priority for the May 1 integration milestone:

1. **Real Google adapter.** Either via their Custom Search API (paid past
   the free quota) or via scraping. Model it on `providers/duckduckgo.py`.
2. **Real Bing / Microsoft Copilot adapter.** Same shape; the provider
   registry in `server.py` just needs one more entry.
3. **Clarifying-question flow.** When the user doesn't supply a preference,
   have the proxy ask one question before ranking. Owned by the prompt
   engineering sub-team — they write the prompt, SWEs wire it up.
4. **Session state.** Right now every query is stateless. `NLIPSession` gives
   us per-connection state for free — wire it up so clarifier answers stick.
5. **Real UI.** The current `static/index.html` is a placeholder — UX owner
   redesigns it using the results of usability testing.
6. **Swap the sponsored-content detector from "the provider told us" to
   real heuristics** once we have a real provider (keyword patterns, URL
   inspection, position-on-page signals).

## Project layout

```
angel_filter/
  angel_filter/
    __init__.py
    server.py             # NLIP server + FastAPI fallback
    orchestrator.py       # parallel fan-out + ranker call
    ranker.py             # Ollama embeddings + sponsored penalty
    providers/
      __init__.py
      base.py             # BaseProvider, ProviderResult, ProviderError
      duckduckgo.py       # real provider, no API key required
      mock.py             # canned data for offline demos and tests
  static/
    index.html            # demo UI
  tests/
    test_orchestrator.py
  pyproject.toml
  README.md
```

## License

Apache-2.0 (matches the upstream NLIP projects).
