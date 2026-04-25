# Angel Filter — Instructions for Claude Code

## What this project is

A Python/FastAPI proxy agent that queries multiple AI/search providers in
parallel, ranks their responses against user preferences using a local LLM
via Ollama, and penalizes sponsored results. CUNY capstone project.

Read `README.md` for the full architecture and current status table.

## Tech stack

- Python 3.10+, Poetry for dependency management
- FastAPI for the server (via `nlip_server`, which wraps FastAPI)
- `nlip_sdk`, `nlip_server`, and reference `nlip_web` from github.com/nlip-project
- Ollama for local embeddings (`nomic-embed-text` default)
- Pytest + pytest-asyncio for tests

## Code conventions

- Type hints everywhere. Use `from __future__ import annotations` in modules
  that reference their own types.
- Async-first: every provider's `query()` and every orchestrator path is
  async. Do not introduce blocking I/O into these paths.
- No hardcoded secrets. If a provider needs an API key, read it from env.
- Prefer lazy imports for optional heavy dependencies (see how `ollama` and
  `httpx` are imported inside functions rather than at module top).
- Keep the fallback FastAPI server in `server.py` working — it's our demo
  insurance if the NLIP libraries fail to install.

## When adding a new provider

1. Create `angel_filter/providers/<name>.py`.
2. Subclass `BaseProvider`, set `name`, implement `query()`.
3. Normalize the provider's native response into `ProviderResult` objects —
   never leak provider-specific shapes past this boundary.
4. Raise `ProviderError` on recoverable failures; the orchestrator handles it.
5. Register the new provider in `_build_orchestrator()` in `server.py`.
6. Add a test in `tests/` that uses canned responses (not the live API).

Use `providers/duckduckgo.py` as the reference implementation.

## When changing the ranker

The ranker's scoring formula is:

    final_score = cosine_similarity(user_pref, result_text) - (SPONSORED_PENALTY if sponsored else 0)

The sponsored penalty is the **thesis of the project** — do not remove or
bypass it without discussion. If you change the embedding model, update
`DEFAULT_EMBED_MODEL` in `ranker.py` and update the README.

## Team ownership (per project plan)

- Software: me, Teammate I (NLIP layer), Teammate J (providers + frontend)
- Prompt engineering: Teammate My, Teammate E
- UX: Teammate Ma
- PM: Teammate P

When I ask you to implement prompt logic or UI, flag that those live in
other owners' lanes and suggest we scope the change accordingly.

## Testing

Run: `poetry run pytest`

Tests must not require live network or Ollama. Use `MockProvider` and set
`ranker._ollama_available = False` for deterministic runs.

## What NOT to do

- Do not delete the fallback FastAPI path in `server.py` yet. It goes once
  `poetry install` is reliable across all teammates' machines.
- Do not add new top-level dependencies without updating `pyproject.toml`
  AND the README's setup section.
- Do not write scraping code for Google/Bing without checking their ToS
  first and raising the question in the PR description.
- Do not modify the NLIP protocol expectations — our job is to use the
  upstream libraries, not fork them.

## Deadlines

- May 1: integrated version ready (Dinesh returns)
- May 15: final demo to 15-org audience
