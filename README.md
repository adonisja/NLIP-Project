# Angel Filter

## Contributing

All changes go through pull requests — no direct commits to `main`, including from project owners.

1. Create a branch from `main`:
   ```bash
   git checkout main && git pull origin main
   git checkout -b your-name/short-description
   ```
2. Make your changes, commit, and push:
   ```bash
   git push -u origin your-name/short-description
   ```
3. Open a pull request on GitHub targeting `main`. Add a brief description of what changed and why.
4. Get at least one teammate review before merging.

---

A local proxy agent that queries multiple AI / search providers, ranks their
responses against what the user actually cares about, and penalizes sponsored
content. Uses the [NLIP protocol](https://github.com/nlip-project) for
communication and a local LLM via [Ollama](https://ollama.com) for ranking, so
user queries never leave the machine.

> CUNY capstone project — target demo May 1, final demo May 15.

## Status (as of April 27, 2026)

| Component | State | Owner |
|---|---|---|
| NLIP server (`NLIPApplication` / `NLIPSession`) | **Working** | SWE team |
| FastAPI fallback server (runs without NLIP) | **Working** | — |
| Provider adapter: DuckDuckGo | **Working** (no API key needed) | — |
| Provider adapter: Mock (canned data for demos) | **Working** | — |
| Provider adapter: Google | Not started | Teammate J |
| Provider adapter: Bing / Copilot | Not started | Teammate J |
| Orchestrator (parallel fan-out, failure isolation) | **Working** | — |
| Ranker (Ollama embeddings + sponsored penalty) | **Working**, with keyword-overlap fallback when Ollama is offline | — |
| `GET /health` (uptime + provider list) | **Working** | — |
| `GET /metrics` (Prometheus) | **Working** | — |
| `GET /docs` (Swagger UI) | **Working** | — |
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

## What to build next

Ordered by priority for the May 1 integration milestone:

1. **Real Google adapter** — model on `providers/duckduckgo.py`; needs `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` from env. Owner: Teammate J.
2. **Real Bing / Copilot adapter** — same shape; one line added to `_build_orchestrator()`. Owner: Teammate J.
3. **Swap mock data** — `mock.py` currently returns toilet paper products; swap canned results to match the demo vertical (flights or shopping) before May 1. Owner: Teammate J.
4. **Clarifying-question flow** — prompt engineering sub-team writes the policy; SWEs wire session state in `NLIPSession`. Coordinate by May 4.
5. **Session state** — `NLIPSession` already supports per-connection state; wire up so clarifier answers persist across turns.
6. **UI redesign** — `static/index.html` is functional but a placeholder; UX owner (Teammate Ma) redesigns using the stable `/query` response schema.
7. **Real sponsored-content detection** — the penalty is live; what's missing is reliable detection from real providers (URL patterns, position signals, keyword heuristics). Owner: Prompt Engineering.

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
