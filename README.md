# Angel Filter

A multi-provider AI proxy that fans queries out to multiple AI and search
providers simultaneously, ranks results using semantic embeddings and a
three-axis scoring system (price · distance · rating), and penalizes
sponsored content — putting the user's interests ahead of advertiser dollars.

CUNY capstone project — final demo May 15, 2026.

---

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

## Status (as of July 28, 2026)

| Component | State |
|---|---|
| NLIP server (`NLIP_Application` / `NLIP_Session`) | **Working** — the active path; the UI posts to `/nlip/` |
| NLIP — structured replies (text + JSON submessages) | **Working** — ranking returned as machine-readable JSON, not just prose |
| FastAPI fallback server | **Working** — used only if the NLIP libraries fail to import |
| GitHub OAuth login + allowlist | **Working** — needs your own OAuth App (see [Authentication](#authentication)) |
| Per-user rate limit + daily query cap | **Working** |
| Provider: OpenAI (`gpt-4o-mini`) | **Working** — needs `OPENAI_API_KEY` |
| Provider: Gemini (`gemini-2.5-flash`) | **Working** — needs `GEMINI_API_KEY` |
| Provider: Ollama (`llama3.2`) | **Working** — runs locally, no key needed |
| Provider: WatsonX (`granite-13b-instruct-v2`) | **Working** — needs `WATSONX_API_KEY` + `WATSONX_PROJECT_ID` |
| Provider: Brave Search | **Ready** — needs `BRAVE_API_KEY` |
| Provider: Google Places (real distance) | **Working** — needs `GOOGLE_PLACES_API_KEY` + user lat/lng |
| Distance enrichment (geocodes other providers' venues) | **Working** — same key; fills P2 for non-Places results |
| Enrichment cost controls (shortlist · venue cache · ceiling) | **Working** — see [Distance enrichment cost](#distance-enrichment-cost) |
| Provider: Mock (canned lunch data for tests) | **Working** (tests only, not in server build) |
| Orchestrator (parallel fan-out, failure isolation) | **Working** |
| Constraint extraction (`$15`, `within 1 mile`, `4 stars`) | **Working** |
| Intent detection (price / distance / rating / general) | **Working** |
| Demo UI — axis priority picker (overrides detection) | **Working** |
| Ranker — semantic similarity (Ollama embeddings) | **Working** |
| Ranker — three-axis gap scoring (P1/P2/P3) | **Working** |
| Ranker — multi-intent axis weighting | **Working** |
| Ranker — missing-data axis renormalization | **Working** |
| Ranker — hard constraint filtering | **Working** |
| Ranker — fuzzy consensus clustering | **Working** |
| Ranker — duplicate collapsing (one slot per venue) | **Working** |
| Ranker — sponsored content penalty | **Working** |
| Query result cache (3-hour TTL, 10 query history) | **Working** — shared by both the NLIP and REST paths |
| `GET /health` | **Working** |
| `GET /metrics` (Prometheus) | **Working** |
| `GET /history` (recent queries) | **Working** — requires a session |
| `POST /cache/clear` | **Working** — requires a session |
| Demo UI — ranked results with score bars | **Working** |
| Demo UI — 3D scoring space (Plotly) | **Working** |
| Demo UI — radar chart (top 3 comparison) | **Working** |
| Demo UI — provider breakdown panel | **Working** |
| Demo UI — query history dropdown | **Working** |
| Demo UI — browser geolocation (sends `lat`/`lng` for distance) | **Working** — best-effort; degrades if the user declines |
| Tests | **217 passing** |

---

## Architecture

```
                       user (browser)
                             │
              GitHub OAuth ──┤  no session? → /login → GitHub → allowlist
              auth.py        │                        (ANGEL_ALLOWED_USERS)
                             ▼
    ┌────────────────────────────────────────────────┐
    │  POST /nlip/          NLIP_Session.execute()   │  server.py
    │                                                │
    │  Multipart NLIP_Message in, by label:          │
    │    (unlabeled text) → query                    │
    │    "preference"     → similarity target        │
    │    "user_location"  → lat/lng                  │
    │    "priority"       → axis override            │
    │                                                │
    │  rate limit + daily cap ─── limits.py          │
    │  query cache (3h TTL) ───── cache.py           │
    │    on a hit the stored payload is returned     │
    │    here — everything below is skipped          │
    └───────────────────┬────────────────────────────┘
                        │  cache miss
                        ▼
    ┌────────────────────────────────────────────────┐
    │  Orchestrator                                  │  orchestrator.py
    │   1. extract_constraints()  ($15, 1mi, 4★)     │  constraints.py
    │   2. detect_intent()  — unless "priority" set  │
    │   3. describe_location()  coords → "Manhattan, │  geocode.py
    │      NY", injected into the AI prompts         │
    │   4. asyncio.gather() over every provider      │
    │      a failing provider is isolated, not fatal │
    └─┬────────┬────────┬────────┬───────┬─────────┬─┘
      │        │        │        │       │         │
   OpenAI   Gemini   Ollama  WatsonX   Brave   Google Places    providers/*.py
      │        │        │        │       │         │
      └────────┴───┬────┴────────┘       │         └── measures distance
                   │                     │             itself
      price + rating, never distance     └── no structured fields
      (told the neighbourhood, not the                  │
       exact position — we do the                       │
       measuring)   │                                   │
                    │                                   │
       ┌────────────┴───────────────────────────────────┘
       │  ProviderResult[]  (normalized at this boundary)
       ▼
    ┌────────────────────────────────────────────────┐
    │  Distance enrichment                           │  geocode.py
    │   for results with no distance, geocode the    │
    │   venue name and haversine vs. the user;       │
    │   a name that resolves to something else is    │
    │   rejected, and unresolved stays None          │
    └───────────────────┬────────────────────────────┘
                        ▼
    ┌────────────────────────────────────────────────┐
    │  Ranker                                        │  ranker.py
    │   1. hard constraint filter (>25% over budget, │
    │      >0.5★ under minimum → dropped)            │
    │   2. embed: Ollama → OpenAI → keyword overlap  │
    │   3. fuzzy consensus clusters (cos ≥ 0.75)     │
    │   4. per-result scoring:                       │
    │        0.50 × similarity                       │
    │      + 0.35 × axis (P1/P2/P3, intent-weighted, │
    │                renormalized over real axes)    │
    │      + 0.15 × consensus (capped at 2)          │
    │      − 0.20 if sponsored                       │
    └───────────────────┬────────────────────────────┘
                        │  RankedResult[] → _serialize_response()
                        ▼  text summary + structured JSON submessage
    ┌────────────────────────────────────────────────┐
    │  Demo UI                                       │  static/index.html
    │   ranked cards · P1/P2/P3 chips · score bars   │
    │   3D scoring space · radar · provider panel    │
    │   priority picker · query history              │
    └────────────────────────────────────────────────┘

  Also on the server:  GET /health · GET /metrics (Prometheus)
                       GET /history · POST /cache/clear     (session required)

  POST /query is the same pipeline over plain REST — the fallback FastAPI
  app in server.py, used only if the NLIP libraries fail to import.
```

---

## Embedding backends

Semantic similarity scoring requires an embedding model. The ranker tries
three backends in order, using whichever is available:

| Priority | Backend | When it's used |
|---|---|---|
| 1 | **Ollama** (`nomic-embed-text`) | Local development — Ollama running on `localhost:11434` |
| 2 | **OpenAI** (`text-embedding-3-small`) | Cloud deployment — Ollama not available, `OPENAI_API_KEY` is set |
| 3 | **Keyword overlap** | Last resort — no embedding backend available, scores are weaker |

> **For Render deployment:** Ollama cannot run on Render's free tier. Set
> `OPENAI_API_KEY` in Render's environment variables and the ranker will
> automatically use OpenAI embeddings instead. All scoring, consensus
> clustering, and axis weighting remain fully active — only the embedding
> source changes.

---

## How scoring works

Every result is scored across four layers:

### 1. Semantic similarity (weight: 50%)
The user's query and each result's title + snippet are embedded using Ollama
(`nomic-embed-text`). Cosine similarity between the query vector and each
result vector produces a 0–1 score. Falls back to keyword overlap when Ollama
is offline.

### 2. Three-axis gap scoring (weight: 35%)

Explicit constraints are extracted from the query and injected into provider
prompts and the ranker:

| Axis | Constraint example | Gap math |
|---|---|---|
| P1 Price | `under $15` | `candidate.price - budget` (negative = under budget) |
| P2 Distance | `within 1 mile` | `candidate.distance - max_distance` (negative = closer) |
| P3 Rating | `rated 4 stars` | `min_rating - candidate.rating` (negative = meets threshold) |

Each gap maps to a 0–1 score, scaled so that meeting a constraint scores well
and beating it scores better. With `rated at least 4 stars`, for example, a
4.0★ result scores 0.60 on P3 and a 5.0★ result scores 1.00 — the axis stays
useful for ranking *within* the set of results that satisfy the constraint.

Intent detection (price / distance / rating / general) shifts the axis weights
— a price query gives P1 60% of the axis score, with P2 and P3 splitting the
remaining 40%. All three axes always contribute — no winner-take-all.

**The user can override the inferred intent.** The demo UI's priority picker
(Auto · Price · Distance · Rating) sends the chosen axis as an NLIP submessage
labeled `priority`; the server maps it to a `QueryIntent` and passes it to
`handle_query(intent=...)`, skipping keyword detection. **Auto** sends nothing
and keeps the inferred behaviour, so the picker is purely additive. An
unrecognised value logs a warning and falls back to detection rather than
erroring. Agent clients get the same control by attaching that submessage.

The REST `/query` endpoint accepts the same override as an optional `priority`
field (`"price"` / `"distance"` / `"rating"`; omit it for auto), so both
transports rank a request identically. The priority is part of the cache key —
ranking by price and by rating produce different orderings, so they must not
share a cached entry.

Hard constraint filtering removes results that are more than 25% over budget
or more than 0.5★ below the minimum rating before scoring begins. The P3 scale
is anchored to that same 0.5★ cutoff: a result exactly at the filter boundary
scores 0.0, so the two thresholds agree by construction.

**Missing data is not mediocre data.** Providers disclose different fields: AI
providers (OpenAI, Gemini, Ollama, WatsonX) return price and rating but never
distance — they have no location context and would fabricate it — and Brave
returns no structured fields at all — it is a *web* search API, so its results
are pages, not places, and carry no coordinates to read.

**The models are told where you are.** The browser's coordinates are reverse-
resolved to a neighbourhood ("Manhattan, NY") and injected into every AI
provider's prompt. Without it they were being asked for "lunch nearby" with no
idea where "near" is, and returned invented placeholders — *The Green Bowl*,
*Taco Haven*, *Bistro Bites* — that geocoded to nothing. With it, the same
prompt returns *Joe's Pizza*, *Los Tacos No. 1*, *The Halal Guys*. Naming the
area is not licence to invent distances: the prompt still forbids
`distance_miles`, because the model knows the neighbourhood, not your position.

**Distance is measured, never guessed.** Two things populate the P2 axis. The
**Google Places** provider returns nearby venues with coordinates and computes
each one's distance directly. Then, after fan-out, `geocode.py` resolves
coordinates for results from *every other* provider that named a venue but
reported no distance, and haversines them against the user — so an OpenAI or
Brave result can now score on P2 instead of sitting it out. A title that cannot
be resolved keeps `distance=None` and stays honestly unscored; a conservative
filter skips titles that look like articles rather than places ("The 10 Best
Tacos in Brooklyn" would geocode to *something*, and that something would be
wrong). Enrichment needs `GOOGLE_PLACES_API_KEY` and a user location; without
either it is a no-op.

#### Distance enrichment cost

Every geocode is a billed Places call, and the fan-out returns ~40 results while
the user sees 5 — so left unbounded a single query cost **17 lookups**, most of
them for results nobody would ever read. Three layers bound it:

| Layer | Env var | What it does |
|---|---|---|
| **Shortlist** | `ANGEL_GEOCODE_SHORTLIST` (12) | Preranks on the signals already free — keyword overlap plus whichever of price and rating the provider disclosed — and only geocodes the top slice. Results already carrying a distance are always kept, since they cost nothing. |
| **Venue cache** | — | Venue name → coordinates, held for the process lifetime. The same places recur constantly across queries and users, so repeat runs converge toward zero. Unresolvable names are cached too, so a hallucinated venue is not retried forever. |
| **Ceiling** | `ANGEL_GEOCODE_MAX_LOOKUPS` (15) | Hard cap per query as a backstop. Titles past it keep `distance=None` and stay honestly unscored. |

`ANGEL_GEOCODE_ENABLED=false` turns enrichment off entirely; P2 then reverts to
Google Places only, exactly as before the feature existed.

Measured on the same repeated query, clearing the *query* cache each run so the
full pipeline executes every time: **17 → 6 → 4 → 2** lookups.

The shortlist is deliberately much larger than `top_k`. Distance carries 60% of
the axis score on a distance query, so it has to be able to reorder the final
ranking — a shortlist the size of `top_k` would fix the winner before the
deciding axis was ever measured.

Scoring an absent axis as a neutral 0.5 would let a bare search result
with no data land within 0.11 of a result that satisfies every constraint,
which is less than the 0.20 sponsored penalty.

So the axis weights are **renormalised over whichever axes actually have
data**: a result with price and rating but no distance is judged on price and
rating alone. One exception guards the obvious abuse — if the axis the user
asked about is the missing one, it keeps its weight at a neutral 0.5 rather
than being redistributed. Otherwise a result that never said where it is could
win a "nearest" query by being cheap. A result that *does* disclose a bad
value still ranks below one that disclosed nothing, so honest reporting is
never punished.

**The API says which axes were real.** `axis_scores` stores a neutral `0.5` for
an undisclosed axis, which is indistinguishable from a genuinely mid-scoring
one, so every result also carries an `axis_scored` map:

```json
"axis_scores": {"P1_price": 0.93, "P2_distance": 0.5, "P3_rating": 0.84},
"axis_scored": {"P1_price": true, "P2_distance": false, "P3_rating": true}
```

The demo UI honours it: undisclosed axes read **"no data"** on hover and in the
winner's axis chips, and a result missing any axis is drawn hollow in the 3D
plot — its position on that axis is an assumption, not a measurement. Without
this the charts plotted the `0.5` placeholder as though it were measured, which
is precisely the equivalence this section exists to reject.

**One slot per venue.** Consensus clustering deliberately refuses to group
results from the same provider, so nobody can manufacture their own agreement —
correct for *counting* providers, wrong for deduplicating output. Nothing
collapsed the copies, so a venue every provider named took one slot per mention
("Shake Shack" once held three of five), and the consensus bonus made it worse
by scoring the copies identically high. Duplicates are now collapsed after
scoring and before `top_k`, so the surviving entry keeps the consensus count it
earned and the best-scoring copy is the one kept.

### 3. Fuzzy consensus bonus (weight: 15%, capped at 2 providers)
Results mentioned by multiple providers are boosted. Matching uses embedding
cosine similarity ≥ 0.75 so "Joe's Pizza" and "Joe Pizza" cluster together;
the keyword fallback matches on normalised title instead. Results from the
same provider never cluster with each other — one chatty provider cannot
manufacture its own consensus.

The bonus scales linearly with the number of *additional* providers that
agree, capped at 2: one extra provider earns half the 15%, two or more earn
all of it. The cap stops a mediocre result from winning just because every
provider mentioned it.

### 4. Sponsored penalty (flat −0.20)
Any result flagged as sponsored receives a flat score deduction regardless of
how well it matches the query. This is the thesis of the project.

**Final score formula:**
```
score = 0.50 × similarity        # 0–1 cosine (or keyword overlap)
      + 0.35 × axis_score        # 0–1 intent-weighted P1/P2/P3
      + 0.15 × consensus_factor  # 0–1: extra_providers capped at 2, ÷ 2
      - 0.20 (if sponsored)
```

Each term is a weight times a 0–1 value, and the three weights sum to 1.0, so
an unsponsored result always scores in 0–1. Preserve that invariant when
tuning — folding a weight into its own term applies it twice and silently
shrinks that signal.

---

## Setup

Choose your operating system below. You need at least one API key to run the server — Gemini has a free tier and is the easiest to get started with.

---

### Mac

#### Requirements
- macOS 11 or later
- [Homebrew](https://brew.sh) (package manager)
- Python 3.12
- [Ollama](https://ollama.com) (local AI — free, no key needed)
- Git
- At least one API key (see [API Keys](#api-keys) below)

#### Step-by-step

**1. Install Homebrew** (skip if already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**2. Install Python 3.12 and Git**
```bash
brew install python@3.12 git
```

Verify:
```bash
python3.12 --version   # should print Python 3.12.x
git --version
```

**3. Install Ollama**

Download from **https://ollama.com/download** and run the installer.

Then pull the two models Angel Filter needs:
```bash
ollama pull nomic-embed-text   # embedding model — used for ranking
ollama pull llama3.2           # generation model — used as a provider
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```
You should see a JSON list of installed models.

**4. Clone the repo**
```bash
git clone https://github.com/adonisja/NLIP-Project
cd NLIP-Project
```

**5. Install Python dependencies**
```bash
pip3.12 install fastapi "uvicorn[standard]" httpx prometheus-client ollama python-dotenv
```

**6. Download Plotly** (required for the 3D visualization)
```bash
curl -o static/plotly.min.js https://cdn.plot.ly/plotly-2.32.0.min.js
```

**7. Set up your API keys**

Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Then open `.env` in any text editor and fill in your keys (see [API Keys](#api-keys) below).

**8. Start the server**
```bash
./start.sh
```

Open **http://localhost:8005** in your browser.

---

### Windows

#### Requirements
- Windows 10 or 11
- [Python 3.12](https://www.python.org/downloads/) (check "Add to PATH" during install)
- [Ollama for Windows](https://ollama.com/download)
- [Git for Windows](https://git-scm.com/download/win)
- At least one API key (see [API Keys](#api-keys) below)

#### Step-by-step

**1. Install Python 3.12**

Download from **https://www.python.org/downloads/release/python-3120/**

During installation, check **"Add python.exe to PATH"** — this is important.

Verify in a new terminal (Command Prompt or PowerShell):
```
python --version   # should print Python 3.12.x
pip --version
```

**2. Install Git**

Download from **https://git-scm.com/download/win** and run the installer with default settings.

**3. Install Ollama**

Download from **https://ollama.com/download** and run the installer.

Open a new terminal and pull the two models:
```
ollama pull nomic-embed-text
ollama pull llama3.2
```

Verify Ollama is running:
```
curl http://localhost:11434/api/tags
```

**4. Clone the repo**
```
git clone https://github.com/adonisja/NLIP-Project
cd NLIP-Project
```

**5. Install Python dependencies**
```
pip install fastapi "uvicorn[standard]" httpx prometheus-client ollama python-dotenv
```

**6. Download Plotly**

In PowerShell:
```powershell
Invoke-WebRequest -Uri "https://cdn.plot.ly/plotly-2.32.0.min.js" -OutFile "static\plotly.min.js"
```

**7. Set up your API keys**

Copy the example env file:
```
copy .env.example .env
```
Open `.env` in Notepad or VS Code and fill in your keys.

**8. Start the server**

On Windows, `start.sh` won't work directly. Run this instead:
```
python -m uvicorn angel_filter.server:app --reload --port 8005
```

Or if you have Git Bash installed:
```bash
./start.sh
```

Open **http://localhost:8005** in your browser.

---

### API Keys

You need **at least one** of the following. The server auto-detects which keys are present and enables those providers.

| Provider | Key name | Where to get it | Cost |
|---|---|---|---|
| Gemini | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | Free tier available |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) | Free trial credits |
| WatsonX | `WATSONX_API_KEY` + `WATSONX_PROJECT_ID` | [cloud.ibm.com](https://cloud.ibm.com) | Free tier available |
| Brave Search | `BRAVE_API_KEY` | [api.search.brave.com](https://api.search.brave.com) | 2,000 free queries/month |
| Google Places | `GOOGLE_PLACES_API_KEY` | [console.cloud.google.com](https://console.cloud.google.com) | Billing applies (has free monthly credit) |
| Ollama | *(no key needed)* | Runs locally after install | Free |

> Google Places is the only provider that returns real distance. It also needs
> the user's coordinates, sent as `lat` / `lng` in the `POST /query` body — the
> browser's geolocation prompt supplies these (frontend). Without coordinates
> the provider is skipped and distance ranking falls back to neutral.

Create a `.env` file in the project root with your keys:

```
# Required — at least one AI provider
GEMINI_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here

# WatsonX (needs both values)
WATSONX_API_KEY=your-key-here
WATSONX_PROJECT_ID=your-project-id-here
WATSONX_REGION=us-east
WATSONX_MODEL=ibm/granite-13b-instruct-v2

# Ollama (no key — just set the model name)
OLLAMA_MODEL=llama3.2:latest

# Optional
BRAVE_API_KEY=your-key-here
GOOGLE_PLACES_API_KEY=your-key-here   # real distance; needs lat/lng per request
                                      # also powers distance enrichment, which
                                      # geocodes venues the other providers name
```

> **Never commit your `.env` file.** It is already listed in `.gitignore`.
> Each contributor creates their own `.env` locally.

---

### Authentication

The app is behind GitHub OAuth. Visiting `/` without a session redirects to
`/login`, so **you cannot use the app until this is configured** — a missing or
mismatched OAuth setup looks like a login page you can never get past.

**Every contributor needs their own OAuth App for local development.** An OAuth
App has exactly one callback URL field, and the shared one is already pointed at
the deployed site — so you cannot reuse the team's credentials for `localhost`
without breaking production.

**1. Create an OAuth App**

Go to **https://github.com/settings/applications/new** and fill in:

| Field | Value |
|---|---|
| Application name | `Angel Filter (local dev)` |
| Homepage URL | `http://localhost:8005` |
| Authorization callback URL | `http://localhost:8005/auth/github/callback` |

The callback URL must match **exactly** — `http` not `https`, `localhost` not
`127.0.0.1`, port `8005`, no trailing slash. A mismatch produces GitHub's
"The redirect_uri is not associated with this application" page.

**2. Generate a client secret**

On the app's page, click **Generate a new client secret**. It is shown only
once — copy it immediately.

**3. Add the values to `.env`**

```
GITHUB_CLIENT_ID=your-client-id
GITHUB_CLIENT_SECRET=your-client-secret
GITHUB_OAUTH_CALLBACK_URL=http://localhost:8005/auth/github/callback

# Comma-separated GitHub usernames allowed to sign in — add yours
ANGEL_ALLOWED_USERS=adonisja,your-github-username

# Random string used to sign session cookies
ANGEL_SESSION_SECRET=any-long-random-string

# false for local http; true only when serving over https
ANGEL_COOKIE_SECURE=false
```

`.env` is read at startup, so **restart the server after changing it** —
`--reload` watches `.py` files only.

**Troubleshooting.** Every failure redirects back to `/login?error=...`, and the
value names the stage that failed:

| Landing page | Cause |
|---|---|
| GitHub "Be careful!" page | Callback URL doesn't match the one registered on the app |
| `?error=exchange_failed` | Client secret wrong or truncated |
| `?error=forbidden` | Auth worked, but your username isn't in `ANGEL_ALLOWED_USERS` |
| `?error=state_mismatch` | Session cookie lost — check `ANGEL_SESSION_SECRET` is set and `ANGEL_COOKIE_SECURE=false` on http |

**Rate limits.** Signed-in users get `ANGEL_USER_RATE_LIMIT` requests per
`ANGEL_USER_RATE_WINDOW` seconds (default 30/60s), and the deployment stops
serving after `ANGEL_DAILY_QUERY_LIMIT` queries per UTC day (default 500) to cap
provider spend. Exceeding either returns HTTP 429.

---

### Verifying your setup

After starting the server, check that providers loaded correctly:

```bash
curl http://localhost:8005/health
```

You should see:
```json
{
  "ok": true,
  "mode": "nlip",
  "nlip_available": true,
  "uptime_seconds": 5.1,
  "providers": ["brave", "openai", "gemini", "ollama"]
}
```

`mode` is `"nlip"` when the NLIP libraries import (the default) and `"fallback"`
otherwise. If `providers` is empty or missing one you expected, check the
corresponding key in `.env`.

> **Implementation note.** `nlip_server.setup_server()` mounts its own health
> router, whose `/health` returns a bare `{"status": "healthy"}`. FastAPI
> resolves against the first route registered for a path, so simply declaring
> ours afterwards was never reached — `/health` couldn't report which providers
> had loaded. We now drop that one route from our app's table before declaring
> ours. Upstream's `/health/live` and `/health/ready` are untouched, and the
> library itself is not forked.

Run the test suite (no network or API keys needed):
```bash
# Mac
python3.12 -m pytest tests/ -v

# Windows
python -m pytest tests/ -v
```

All 217 tests should pass.

---

## Running the tests

```bash
python3.12 -m pytest tests/ -v
```

217 tests covering:
- End-to-end pipeline with all providers
- Sponsored penalty applied and visible in scores
- Provider failure isolation
- Budget constraint filtering (`$15` pushes `$28` bistro out)
- Distance intent favors nearest result
- Rating intent favors highest-rated result
- Axis scores present and in 0–1 range on all results
- Consensus bonus applied when two providers agree
- Intent detection for all four intent types (8 parametrized cases)
- Constraint extraction from natural language (7 parametrized cases)
- Consensus bonus reaches its documented 15% weight and caps at 2 providers
- Fuzzy clustering groups near-identical titles, and refuses to cluster two
  results from the same provider
- Sponsored penalty applies on the embedding path, not just the keyword path
- P3 rating axis stays discriminating under a `min_rating` constraint
- Axis weights renormalize over populated axes only, and a result cannot win
  the intent axis by omitting it
- Both scoring loops pass the populated-axis mask (wiring regression guard)
- Ollama embedding calls yield to the event loop instead of blocking it, and
  run concurrently (N results ≈ one round-trip, not N)
- Both scoring paths share one final-score formula (`_assemble_score`); the
  weights, consensus cap, and sponsored penalty are pinned directly
- REST and NLIP share one priority parser, `POST /query` actually forwards the
  override to the orchestrator, and the cache key separates priorities so a
  price ranking is never served from a rating query's entry
- `axis_scored` marks which axes a provider actually disclosed and survives
  serialisation, so the UI can distinguish a placeholder from a measurement
- `/health` is ours in NLIP mode (upstream's router no longer shadows it) while
  its `/health/live` and `/health/ready` probes still respond
- Both branches of the `_NLIP_AVAILABLE` if/else declare the same routes, each
  registered exactly once, with the diagnostic ones behind a session
- The NLIP session caches: a repeated query reuses the stored payload instead of
  re-running every provider, and a different priority keys separately
- Google Places maps venues to real distances (haversine); user coordinates
  flow request → constraints → provider → a discriminating P2 axis
- The user's locality reaches provider prompts through the orchestrator (the
  wiring, not just the prompt builder), and is skipped entirely without coords
- Post-hoc enrichment resolves coordinates for venues other providers named,
  never overwrites a distance a provider already measured, deduplicates repeat
  titles to one lookup, survives a failing lookup, skips article-like titles,
  and leaves anything unresolvable at None rather than defaulting it
- The NLIP session reads query, preference, and location as separate typed
  submessages (a labeled preference never leaks into the query text), replies
  with both a human-readable summary and a machine-readable JSON submessage
  carrying the sponsored flag, and degrades to `None` on malformed location

No tests require network or Ollama. `test_orchestrator.py` uses the mock
provider and the keyword-fallback ranker; `test_ranker_embeddings.py` uses a
`StubRanker` that overrides the embedding seams with a canned vector table;
`test_ranker_async.py` injects a fake async client whose calls sleep, so the
non-blocking behaviour can be observed without a real Ollama. All run
deterministically and offline.

---

## Demo queries

| Query | What it demonstrates |
|---|---|
| `lunch under $15` | Budget constraint + price intent |
| `best rated lunch spots near me` | Rating + distance intent together |
| `Find me the top 3 lunch spots under $15, within 1 mile, rated at least 4 stars` | All three axes, hard filter, constraint injection |
| Same query, switching the priority picker | User override beats inferred intent — the winner changes per axis |
| Run any query twice | Cache hit — instant response, "from cache" badge |

---

## Project layout

```
angel_filter/
  server.py             # NLIP session + fallback FastAPI server + provider wiring
  orchestrator.py       # parallel fan-out + ranker call
  ranker.py             # scoring: similarity + axis + consensus + penalty
  constraints.py        # natural language constraint extraction
  prompt.py             # shared prompt builder for AI providers
  cache.py              # in-memory query cache (3-hour TTL)
  geocode.py            # post-hoc distance: resolve venue coords, haversine
  auth.py               # GitHub OAuth flow + username allowlist
  limits.py             # per-user rate limit + daily query cap
  providers/
    base.py             # BaseProvider, ProviderResult, ProviderError
    openai_provider.py  # OpenAI gpt-4o-mini
    gemini.py           # Google Gemini
    ollama_provider.py  # Local Ollama (llama3.2)
    watsonx.py          # IBM WatsonX
    brave.py            # Brave Search API
    google_places.py    # Google Places — real distance (needs user lat/lng)
    mock.py             # canned lunch data (tests only)
static/
  index.html            # demo UI (results + 3D plot + radar chart + priority picker)
  login.html            # GitHub sign-in page
  plotly.min.js         # Plotly served locally (gitignored, download once)
tests/
  test_geocode.py            # 53 tests — enrichment, locality, venue filter, cost guards
  test_nlip_session.py       # 32 tests — NLIP multipart handling + priority picker
  test_orchestrator.py       # 25 tests — pipeline, intent, constraints, geo distance
  test_rest_priority.py      # 27 tests — REST /query priority parity + cache keying
  test_axis_scored_mask.py   # 7 tests — which axes were measured vs. placeholder
  test_health_route.py       # 7 tests — /health ownership vs. nlip_server's router
  test_route_parity.py       # 12 tests — both branches expose the same routes; NLIP caching
  test_axis_scoring.py       # 16 tests — axis weighting with incomplete data
  test_google_places.py      # 11 tests — Google Places provider + haversine
  test_ranker_embeddings.py  # 8 tests — embedding scoring path (stubbed)
  test_assemble_score.py     # 8 tests — the shared final-score formula
  test_ranker_async.py       # 6 tests — non-blocking, concurrent Ollama embeddings
start.sh                # starts server on port 8005, loads .env
pyproject.toml
README.md
```

---

## License

Apache-2.0 (matches the upstream NLIP projects).
