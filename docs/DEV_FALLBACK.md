# Dev Fallback Mode

`angel_filter/server.py` tries to import the NLIP libraries at startup:

```python
from nlip_server.server import NLIPApplication, NLIPSession, start_server
from nlip_sdk import nlip
```

If either import fails — most commonly because `poetry install` hasn't
resolved the Git-based NLIP dependencies on a teammate's machine yet — the
module logs a warning and falls back to a plain FastAPI app. This is
intentional: we don't want Friday's demo (or anyone's first five minutes with
the repo) gated on a dependency install problem.

## What the fallback exposes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serves `static/index.html`, the demo UI |
| `/query` | POST | Runs the full pipeline: fan out → rank → return JSON |
| `/health` | GET | Returns `{ok: true, mode: "fallback", nlip_available: false}` |

### POST /query

**Request body** (`application/json`):
```json
{ "query": "cheapest 30-pack toilet paper", "preference": "low price not premium brand" }
```

`preference` is optional. If omitted, results are ranked against the `query`
text itself.

**Response body**:
```json
{
  "providers_used": ["duckduckgo", "mock"],
  "providers_failed": [],
  "results": [
    {
      "title": "...",
      "snippet": "...",
      "url": "...",
      "provider": "mock",
      "score": 0.73,
      "rationale": "strong match (similarity 0.88) — sponsored, penalty 0.15 applied",
      "sponsored": true
    }
  ]
}
```

## Why keep this mode in production code

Once every teammate has NLIP reliably installed we can delete the fallback
path. Until then it serves three purposes:

1. **New-contributor onboarding** — `git clone` + `pip install fastapi httpx
   pydantic uvicorn` + `uvicorn angel_filter.server:app` is enough to see
   something work.
2. **Demo insurance** — if the NLIP libraries break upstream the day before
   our demo, we don't panic.
3. **Frontend development** — the UX owner can iterate on `static/index.html`
   without needing any of the NLIP or Ollama setup.

Once `nlip_sdk` and `nlip_server` stabilize (tracking their `main` branches
is a known risk), plan is to pin them to a release tag in `pyproject.toml`
and remove the fallback.
