# Angel Filter — Instructions for Claude Code

## What this project is

A Python/FastAPI proxy that sends a query to several AI and search providers
in parallel, ranks the combined results against the user's constraints and
preferences, and penalizes sponsored results. It speaks NLIP. It began as a
CUNY capstone (spring 2026) and has been handed to new maintainers; the
original author (@adonisja) is their mentor.

- `README.md`: overview, architecture diagram, quickstart, reading order
- `docs/DESIGN.md`: how scoring works and why. Read it before touching `ranker.py`.
- `docs/KNOWN_ISSUES.md`: unfinished work
- `CONTRIBUTING.md`: workflow and conventions

## Tech stack

- Python 3.12, Poetry (the only supported install; it fetches the NLIP libraries)
- FastAPI, via `nlip_server`, which wraps it
- `nlip_sdk` and `nlip_server` from github.com/nlip-project
- Ollama for local embeddings (`nomic-embed-text`) and as a provider
- Shared state in `state.py`: in-process memory, or Postgres (Supabase) when
  `ANGEL_DATABASE_URL` is set. Cache, conversations, limits and geocodes all go through it; never
  keep request-spanning state in a module-level dict.
- Pytest + pytest-asyncio

## Code conventions

- Type hints everywhere. Use `from __future__ import annotations` in modules
  that reference their own types.
- Async-first: every provider's `query()` and every orchestrator path is
  async. Do not introduce blocking I/O into these paths.
- No hardcoded secrets. Read keys from env and document new variables in
  `.env.example` and `docs/SETUP.md`.
- Prefer lazy imports for optional heavy dependencies (see how `ollama` and
  `httpx` are imported inside functions).
- Both branches of `server.py` (NLIP and fallback FastAPI) must expose the
  same routes; `tests/test_route_parity.py` enforces it.
- There is no login (see `docs/KNOWN_ISSUES.md` §1). `limits.py` is the only
  cost protection: any new route that triggers provider calls must go through
  `enforce_query_limits` (or the NLIP middleware) and have a test proving it.

## When adding a new provider

1. Create `angel_filter/providers/<name>.py`, subclass `BaseProvider`, set
   `name`, implement `async query()`.
2. Normalize into `ProviderResult`; never leak provider-specific shapes. Leave
   fields `None` when the provider doesn't really know them. Never invent a
   price, distance, or rating.
3. Raise `ProviderError` on recoverable failures.
4. Export from `providers/__init__.py`; register in `_build_orchestrator()` in
   `server.py` behind an env-variable check.
5. Add a test with canned responses.

Reference implementations: `providers/brave.py` (search API),
`providers/openai_provider.py` (LLM).

## When changing the ranker

    score = 0.50 × similarity + 0.35 × axis_score + 0.15 × consensus − (0.20 if sponsored)

All in `_assemble_score` in `ranker.py`. The weights sum to 1.0; keep it so.

The sponsored penalty is the **thesis of the project**. Do not remove, reduce,
or bypass it without discussion. The missing-data rules in `_axis_bonus` are
deliberate (see `docs/DESIGN.md` §4); if `test_axis_scoring.py` fails after a
change, the change is probably wrong. If you change the embedding model,
update `DEFAULT_EMBED_MODEL` and the docs.

## Testing

Run: `poetry run pytest`

Tests must not require live network, API keys, Ollama, or a database (each
test gets a fresh in-memory store). CI also runs the suite on a throwaway
Postgres with `ANGEL_TEST_STORE=postgres`; Postgres-only tests skip unless
`ANGEL_TEST_DATABASE_URL` is set. Use `MockProvider`
or canned responses, and set `ranker._ollama_available = False` for
deterministic runs.

## Running

    poetry run uvicorn angel_filter.server:app --reload --port 8005

Port 8005 is the convention throughout the docs.

## What NOT to do

- Do not add dependencies without `poetry add` (updating `pyproject.toml`)
  and a note in `docs/SETUP.md`.
- Do not write scraping code for Google/Bing without checking their ToS
  first and raising the question in the PR description.
- Do not modify or fork the NLIP libraries — use them as published; override
  on our side if needed (see the `/health` fix in `docs/DESIGN.md`).
- Do not delete the fallback FastAPI path without team agreement (see
  `docs/KNOWN_ISSUES.md`).
