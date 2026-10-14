# Setup

The [README quickstart](../README.md#quickstart) gets you running with Ollama
alone. This page covers everything else: Windows, each provider, deployment,
and what to do when things break.

---

## Requirements

| Tool | Version | Notes |
|---|---|---|
| Python | 3.12 | `.python-version` pins 3.12.3; anything ≥ 3.10 should work |
| Poetry | 1.8+ | The only supported way to install. It pulls the NLIP libraries from GitHub. |
| Ollama | latest | Needed for local embeddings and the Ollama provider |
| Git | any | Poetry needs it to fetch the NLIP libraries |

### Mac

```bash
brew install python@3.12 git pipx
pipx install poetry
```

Install Ollama from <https://ollama.com/download>.

### Windows

1. Install [Python 3.12](https://www.python.org/downloads/) and tick **"Add
   python.exe to PATH"**.
2. Install [Git for Windows](https://git-scm.com/download/win) and
   [Ollama for Windows](https://ollama.com/download).
3. Install Poetry in PowerShell: `py -m pip install --user pipx`, then
   `py -m pipx install poetry`.
4. Download Plotly with PowerShell instead of `curl`:
   ```powershell
   Invoke-WebRequest -Uri "https://cdn.plot.ly/plotly-2.32.0.min.js" -OutFile "static\plotly.min.js"
   ```
5. Use `copy .env.example .env` instead of `cp`.

Everything else in the quickstart is the same. Run the server with
`poetry run uvicorn ...` (or `./start.sh` from Git Bash).

---

## Environment variables

The package loads `.env` from the repo root automatically at import. Restart
the server after you change it: `--reload` only watches `.py` files.
[.env.example](../.env.example) lists every variable with comments.

### Providers

A provider turns on only when its variables are set. **At least one is
required**, or the server refuses to start with `No providers configured`.

| Provider | Variables | Where to get it | Cost |
|---|---|---|---|
| Ollama | `OLLAMA_MODEL` (e.g. `llama3.2`), optional `OLLAMA_URL` | Runs locally | Free |
| Gemini | `GEMINI_API_KEY`, optional `GEMINI_MODEL` | [aistudio.google.com](https://aistudio.google.com) | Free tier |
| OpenAI | `OPENAI_API_KEY`, optional `OPENAI_MODEL` | [platform.openai.com](https://platform.openai.com) | Paid |
| WatsonX | `WATSONX_API_KEY` + `WATSONX_PROJECT_ID`, optional `WATSONX_REGION`, `WATSONX_MODEL` | [cloud.ibm.com](https://cloud.ibm.com) | Free tier |
| Brave Search | `BRAVE_API_KEY` | [api.search.brave.com](https://api.search.brave.com) | Free tier |
| Google Places | `GOOGLE_PLACES_API_KEY` | [console.cloud.google.com](https://console.cloud.google.com) | Billed per call; has a monthly free credit |

Notes:

- **Ollama isn't enabled just because it's installed.** You have to set
  `OLLAMA_MODEL`. The code default is `llama3.1:8b`, so set it to whichever
  model you pulled.
- **Google Places does two jobs.** It's a provider in its own right, and it
  also powers *distance enrichment*: geocoding the venues that other providers
  name. Both need the user's location. The UI sends it through its location
  presets or the browser's geolocation. Without a location both are skipped,
  and distance goes unscored.
- **Google Places is the one that can cost real money.** Put a budget alert on
  the Google Cloud project. The cost controls below limit each query, not the
  monthly total.
- **Keys are per person.** Keys from the original team don't carry over. Get
  your own, and never commit `.env` (it's in `.gitignore`).

### Embeddings (used by the ranker, not a provider)

The ranker compares text using embeddings, and tries these in order:

1. **Ollama `nomic-embed-text`**, if Ollama is running on `localhost:11434`.
2. **OpenAI `text-embedding-3-small`**, if `OPENAI_API_KEY` is set.
3. **Keyword overlap**, which always works but scores noticeably worse.

Both the ranker and the Ollama provider use `OLLAMA_URL`, so a remote Ollama
serves embeddings and generation alike.

### Authentication

There is none. The original GitHub login was removed so the new team can
design its own (see [KNOWN_ISSUES.md](KNOWN_ISSUES.md#1-design-and-build-login-and-access-control)).
Anyone who can reach the server can search, which is fine on `localhost` and
needs thought before you deploy.

### Limits and cost controls

| Variable | Default | Meaning |
|---|---|---|
| `ANGEL_USER_RATE_LIMIT` / `ANGEL_USER_RATE_WINDOW` | 30 / 60 s | Searches per client IP per window |
| `ANGEL_DAILY_QUERY_LIMIT` | 500 | Searches across all clients per UTC day, then HTTP 429 |
| `ANGEL_TRUSTED_PROXY_HOPS` | 0 | Reverse proxies in front of the app. Controls whether the client IP is read from `X-Forwarded-For`. Leave 0 locally. |
| `ANGEL_GEOCODE_ENABLED` | `true` | Turns distance enrichment on or off |
| `ANGEL_GEOCODE_SHORTLIST` | 12 | How many results are considered for geocoding |
| `ANGEL_GEOCODE_MAX_LOOKUPS` | 15 | Hard cap on paid Places lookups per query |

Both `/query` and `/nlip/` count against these. With no login, they're the
only thing limiting spend on a public deployment, and the daily cap is the one
that holds even against a caller with many IP addresses. Where the counters
live is the next section.

### Shared state (Redis)

The query cache, conversations, rate-limit counters, daily count, and geocode
caches live in one store ([state.py](../angel_filter/state.py)):

| `ANGEL_REDIS_URL` | Where state lives | Use it for |
|---|---|---|
| unset (default) | This process's memory. Lost on restart, not shared between workers. | Local development and tests. Nothing to set up. |
| a Redis URL | Redis. Survives restarts, shared by every worker and instance. | Any deployment, so the limits and the daily cap actually hold. |

**Free option: [Upstash](https://upstash.com).** Its free Redis keeps data on
disk, allows 256 MB and 500K commands a month, and is archived (with a backup)
only after 30 days with no activity. A search uses roughly 15–25 commands, so
the default 200-searches-a-day cap stays well inside the free limit.

1. Create a free account at <https://console.upstash.com> and create a Redis
   database in the region nearest your server.
2. Copy its **TCP** connection URL. It looks like
   `rediss://default:<password>@<name>.upstash.io:6379` (note `rediss`, with
   TLS). Not the REST URL.
3. Set it as `ANGEL_REDIS_URL` (in `.env` locally, in the dashboard on Render)
   and restart.
4. Check `/health`: it should report `"state": {"backend": "redis",
   "reachable": true}`.

Any other Redis works the same way, including paid Render Key Value or a local
one (`docker run -p 6379:6379 redis:7-alpine`, then
`ANGEL_REDIS_URL=redis://localhost:6379/0`). Render's *free* Key Value doesn't
keep data across its restarts, so it doesn't help here.

**If Redis becomes unreachable**, the app logs a warning and keeps running on
in-process state, exactly as if `ANGEL_REDIS_URL` were unset, until Redis is
back. `/health` then shows `"reachable": false`.

Geocoded venue coordinates expire after 30 days, because Google's Maps
Platform terms limit how long Places content may be cached. Check the current
terms before changing that.

---

## Running

```bash
poetry run uvicorn angel_filter.server:app --reload --port 8005
```

Port 8005 is the convention throughout the docs. Any free port works.

| URL | What it is |
|---|---|
| `/` | Demo UI |
| `/health` | Mode, uptime, active providers |
| `/docs` | Swagger UI |
| `/metrics` | Prometheus metrics |
| `/nlip/` | The NLIP endpoint (the UI posts here) |

## Testing

```bash
poetry run pytest
```

The suite runs fully offline, with no keys, network, or Ollama needed. CI
([.github/workflows/ci.yml](../.github/workflows/ci.yml)) runs the same command
on every push and PR to `main`.

---

## Deploying your own copy

[render.yaml](../render.yaml) is a Render Blueprint. Connect your repository in
the Render dashboard and it creates a web service. Render installs from
`requirements.txt`, which is generated from `poetry.lock` (CI checks they
match), so it gets the same versions CI tests, NLIP libraries included.

1. **Think about cost first.** There is no login, so anyone with the URL can
   search, and every search spends your provider credit. `render.yaml` sets a
   daily cap of 200 searches and 20 per client per minute. Lower them if your
   budget is small, and put spending alerts on every provider account. Doing
   the [login starter task](KNOWN_ISSUES.md#1-design-and-build-login-and-access-control)
   before deploying publicly is worth considering.
2. **Set the secrets** in the Render dashboard: the provider keys marked
   `sync: false` in `render.yaml`. Set only the ones you use.
3. **Set `OPENAI_API_KEY` for embeddings.** Render can't run Ollama, and
   without an embedding backend the ranker falls back to keyword overlap.
4. **Don't set `OLLAMA_MODEL`** unless `OLLAMA_URL` points at an Ollama server
   you host somewhere reachable. Otherwise it enables a provider that fails on
   every query.
5. **Check `/health` after the deploy.** It must report `"mode": "nlip"`. The
   UI only talks to `/nlip/`, so in fallback mode it can't search.
6. **Check that rate limiting sees real client IPs.** `render.yaml` sets
   `ANGEL_TRUSTED_PROXY_HOPS=1`, which assumes Render adds exactly one entry
   to `X-Forwarded-For`. To confirm, search a few times past the per-minute
   limit from two different networks (e.g. home wifi and a phone hotspot).
   Each should get its own allowance. If one network's searches block the
   other's, the hop count is wrong for your setup. The daily cap still
   protects you either way.

**Set `ANGEL_REDIS_URL`** (see [Shared state](#shared-state-redis)). The
free tier sleeps when idle and restarts often. Without Redis, each restart
wipes the cache, the conversations, and the rate-limit and daily counters,
which means the daily spending cap resets too.

---

## Troubleshooting

### Server

| Symptom | Cause |
|---|---|
| `RuntimeError: No providers configured` on startup | No provider variables are set. Set `OLLAMA_MODEL` at minimum. |
| Searching in the UI fails with `HTTP 404`, or `/health` shows `"mode": "fallback"` | The NLIP libraries didn't import. Run `poetry install` again and make sure you started the server through `poetry run`. |
| A provider is missing from `/health` | Its variable is missing or misspelled in `.env`, or you didn't restart after editing. |
| Every result has "no data" for distance | No user location was sent, or `GOOGLE_PLACES_API_KEY` isn't set |
| The 3D chart says "visualization unavailable" | `static/plotly.min.js` hasn't been downloaded |
| Searches fail with "Rate limit exceeded" or "Daily query budget exhausted" | You hit `ANGEL_USER_RATE_LIMIT` or `ANGEL_DAILY_QUERY_LIMIT`. Raise them in `.env` for local work, then restart. |
| Tests fail with `'ModelField' object has no attribute 'type_'` | You're running them outside Poetry against a newer FastAPI. Use `poetry run pytest`. |
