# Contributing

## Workflow

Every change goes through a pull request, including changes from whoever owns
the repository.

```bash
git checkout main && git pull
git checkout -b your-name/short-description
# ...make changes, run the tests...
git push -u origin your-name/short-description
```

Open a PR against `main` that says what changed and why, and get at least one
review from a teammate before merging. CI runs the test suite on every PR.

When you set up your own repository, turn on branch protection for `main`
(Settings → Branches): require a PR, one approval, and a passing CI check.
None of that carries over when you clone.

## Code conventions

- **Type hints everywhere.** Add `from __future__ import annotations` in
  modules that refer to their own types.
- **Async all the way.** Every provider's `query()` and everything the
  orchestrator calls is async. Don't add blocking I/O to those paths: use
  `httpx.AsyncClient`, not `requests`.
- **No secrets in code.** Read keys from the environment, and add new
  variables to [.env.example](.env.example) and [docs/SETUP.md](docs/SETUP.md).
- **Import heavy optional dependencies inside functions** (as with `ollama`
  and `httpx`), so the app still starts without them.
- **New dependencies** go in `pyproject.toml` via `poetry add`, and get a
  mention in the setup docs.

## Tests

```bash
poetry run pytest
```

Tests must not need the network, API keys, or Ollama. Use `MockProvider` or
canned responses, and set `ranker._ollama_available = False` for
deterministic ranking. `tests/conftest.py` already sets `OLLAMA_MODEL` so the
server module can be imported, and clears the shared caches between tests.

Any bug fix should come with a test that fails without the fix.

## Adding a provider

1. Create `angel_filter/providers/<name>.py`. Use
   [brave.py](angel_filter/providers/brave.py) as the model for a search API,
   or [openai_provider.py](angel_filter/providers/openai_provider.py) for an
   LLM.
2. Subclass `BaseProvider`, set `name`, and implement `async query()`.
3. Convert everything into `ProviderResult`. No provider-specific shapes
   should get past this point. **Leave a field `None` if the provider doesn't
   really know it. Never guess.** The ranker treats missing data specially
   (see [docs/DESIGN.md](docs/DESIGN.md#4-missing-data-is-not-average-data)),
   and a made-up distance or rating is worse than none.
4. Set `sponsored` from real information if the provider has it. See
   [KNOWN_ISSUES.md #2](docs/KNOWN_ISSUES.md#2-no-live-result-is-ever-flagged-as-sponsored).
5. Raise `ProviderError` on recoverable failures. The orchestrator logs it and
   carries on with the other providers.
6. Export it from `angel_filter/providers/__init__.py`, and register it in
   `_build_orchestrator()` in [server.py](angel_filter/server.py), behind a
   check for its env variable.
7. Add a test with canned responses (see
   [tests/test_google_places.py](tests/test_google_places.py)).

## Updating the NLIP libraries

`pyproject.toml` follows the `main` branch of `nlip-sdk` and `nlip-server`,
but `poetry.lock` pins exact commits, so nothing changes until you choose to.
To update:

```bash
poetry update nlip-sdk nlip-server
poetry run pytest
poetry export --without-hashes -f requirements.txt -o requirements.txt
```

Read what changed upstream first. The export step keeps deployments in step
(CI fails if `requirements.txt` and `poetry.lock` disagree).

## Changing the ranker

Read [docs/DESIGN.md](docs/DESIGN.md) first. In particular:

- The **sponsored penalty is the thesis of the project**. Don't remove,
  reduce, or route around it without team agreement.
- Keep the score weights adding up to 1.0.
- If you change the embedding model, update `DEFAULT_EMBED_MODEL` in
  `ranker.py` and the docs.

## Don't

- **Fork or patch the NLIP libraries.** Our job is to use `nlip_sdk` and
  `nlip_server` as they are. If you need to override something (see the
  `/health` fix in DESIGN.md), do it on our side.
- **Write scraping code** for Google, Bing, or similar without checking their
  terms of service first and raising it in the PR description. Use official
  APIs.
- **Commit `.env`** or anything else containing keys.
