# Angel Filter

A search proxy that works for the user, not the advertiser.

You ask one natural-language question — *"lunch under $15 within 1 mile, rated
at least 4 stars"*. Angel Filter sends it to several AI and search providers at
once, merges what comes back, and ranks it against what you actually asked for.
Anything sponsored takes a flat score penalty. That penalty is the point of the
project: a paid placement has to be clearly better to beat an organic result.

It speaks [NLIP](https://github.com/nlip-project), the Natural Language
Interaction Protocol, so other agents can query it as well as people using the
web UI.

---

## Project status

Angel Filter began as a CUNY capstone project (spring 2026) and was demoed on
May 15, 2026. It is now being handed to new students to carry forward.

- **Mentor:** Discord `akkeemt`, email <tyrellakkeem@gmail.com>,
  GitHub [@adonisja](https://github.com/adonisja). Wrote much of the current
  code. Ask about design history or why something is the way it is.
- **What works:** the full pipeline end to end: six live providers, ranking,
  real distances, sponsored-result detection, multi-turn follow-ups, and the
  demo UI. State (cache, conversations, rate limits, the daily cap) persists
  in Postgres when a database is configured; Supabase's free tier is the
  documented option.
- **There is no login.** The original GitHub login was removed so you can
  design your own. That's the first starter task in
  [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md). Until then, anyone who can
  reach your server can search, so per-client rate limits and a daily query
  cap are the only cost protection.
- **What doesn't yet:** see [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md). Read
  it before you start. Sponsored results are detected by conservative rules
  ([sponsorship.py](angel_filter/sponsorship.py)), and nobody has yet measured
  how often they fire on real searches.

You will be running your **own copy**: your own repository, API keys,
database and deployment. Nothing from the original team's accounts carries
over. **Start with [docs/HANDOFF.md](docs/HANDOFF.md)**, a checklist that
takes you from an empty account to a working deployment.

---

## How it works

```
                       user (browser)
                             │
                             ▼
    ┌────────────────────────────────────────────────┐
    │  POST /nlip/          NLIP_Session.execute()   │  server.py
    │                                                │
    │  Multipart NLIP_Message in, by label:          │
    │    (unlabeled text) → query                    │
    │    "preference"     → similarity target        │
    │    "user_location"  → lat/lng                  │
    │    "priority"       → axis override            │
    │    "turn_mode"      → continue / new           │
    │    (token submsg)   → conversation identity    │
    │                                                │
    │  multi-turn memory ──────── conversation.py    │
    │  rate limit + daily cap ─── limits.py          │
    │    (per client IP; no login)                   │
    │  query cache (3h TTL) ───── cache.py           │
    │    all three persist via state.py: in memory,  │
    │    or Postgres (Supabase) if configured        │
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
                   │                                    │
       ┌───────────┴────────────────────────────────────┘
       │  ProviderResult[]  (normalized at this boundary)
       ▼
    ┌────────────────────────────────────────────────┐
    │  Sponsored detection                           │  sponsorship.py
    │   ad-network links, ad-click IDs, affiliate    │
    │   tags, "Sponsored" disclosures → flagged,     │
    │   with the reason recorded                     │
    └───────────────────┬────────────────────────────┘
                        ▼
    ┌────────────────────────────────────────────────┐
    │  Distance enrichment                           │  geocode.py
    │   geocode venues that have no distance and     │
    │   measure them against the user's location     │
    └───────────────────┬────────────────────────────┘
                        ▼
    ┌────────────────────────────────────────────────┐
    │  Ranker                                        │  ranker.py
    │   1. hard constraint filter                    │
    │   2. embed: Ollama → OpenAI → keyword overlap  │
    │   3. group results several providers agree on  │
    │   4. score each result, collapse duplicates    │
    └───────────────────┬────────────────────────────┘
                        │  text summary + structured JSON submessage
                        ▼
    ┌────────────────────────────────────────────────┐
    │  Demo UI                                       │  static/index.html
    │   ranked cards · score breakdown · 3D plot ·   │
    │   radar chart · priority picker · history      │
    └────────────────────────────────────────────────┘

  Also on the server:  GET /health · GET /metrics (Prometheus)
                       GET /history · POST /cache/clear

  POST /query runs the same pipeline over plain REST. If the NLIP libraries
  fail to import, it's the only query route left (see docs/DEV_FALLBACK.md).
```

Every result gets one score:

```
score = 0.50 × similarity    how well the text matches what you asked for
      + 0.35 × axis score    price (P1), distance (P2), rating (P3) vs. your limits
      + 0.15 × agreement     other providers named the same place (capped at 2)
      − 0.20 if sponsored
```

The three weights add up to 1.0, so every unsponsored result scores between 0
and 1. [docs/DESIGN.md](docs/DESIGN.md) explains each term and the decisions
behind it. Read it before changing the ranker.

---

## Quickstart

This is the smallest setup that works: local Ollama as the only provider, so
you need no paid API keys. [docs/SETUP.md](docs/SETUP.md) has Windows
instructions, the other providers, and deployment.

**You need:** Python 3.12, [Poetry](https://python-poetry.org/docs/#installation),
[Ollama](https://ollama.com/download), and Git.

```bash
# 1. Get the code (your own copy)
git clone <your-repo-url> angel-filter
cd angel-filter

# 2. Install dependencies, including the NLIP libraries
poetry install

# 3. Pull the local models
ollama pull nomic-embed-text   # embeddings, used by the ranker
ollama pull llama3.2           # generation, used as a provider

# 4. Plotly for the 3D chart (not committed; the UI works without it)
curl -o static/plotly.min.js https://cdn.plot.ly/plotly-2.32.0.min.js

# 5. Configure (the defaults already enable Ollama)
cp .env.example .env

# 6. Run
poetry run uvicorn angel_filter.server:app --reload --port 8005
```

Open <http://localhost:8005> and try `lunch under $15`.

**Check it:**

```bash
curl http://localhost:8005/health   # expect "mode": "nlip" and "providers": ["ollama"]
poetry run pytest                   # offline: no network, keys, or Ollama needed
```

If `mode` says `"fallback"`, the NLIP libraries didn't install, and the UI
can't search. See
[docs/SETUP.md](docs/SETUP.md#troubleshooting).

**Optional: a database.** Without one, state lives in memory and resets when
the server restarts, which is fine for local work. To keep it, create a free
Supabase project and follow
[docs/SETUP.md → Shared state](docs/SETUP.md#shared-state-supabase). The
short version is to add its Session pooler URL to `.env` as
`ANGEL_DATABASE_URL`, then run:

```bash
poetry run python scripts/check_database.py --set-password   # hidden prompt, tests the connection
```

Restart the server; `/health` should show `"state": {"backend": "postgres",
"reachable": true}`.

---

## Where to start reading

Read in this order. Each file builds on the one before it.

1. [angel_filter/providers/base.py](angel_filter/providers/base.py): the
   `ProviderResult` shape every provider returns. Everything downstream depends
   on it.
2. [angel_filter/providers/openai_provider.py](angel_filter/providers/openai_provider.py)
   and [prompt.py](angel_filter/prompt.py): a typical AI provider, and the
   shared prompt that asks for price and rating.
3. [angel_filter/orchestrator.py](angel_filter/orchestrator.py): the path a
   single query takes: constraints, intent, fan-out, enrichment, ranking.
4. [angel_filter/ranker.py](angel_filter/ranker.py): scoring. Start at
   `_assemble_score`, then `_axis_bonus`.
5. [angel_filter/server.py](angel_filter/server.py): the NLIP session
   (`AngelFilterSession.execute`) and the HTTP routes.
6. [tests/test_orchestrator.py](tests/test_orchestrator.py): the whole
   pipeline running on canned data. It's the easiest way to watch the system
   work.

---

## Repository layout

```
angel_filter/
  server.py             NLIP session, HTTP routes, fallback app, provider wiring
  orchestrator.py       parallel fan-out, then enrichment, then ranking
  sponsorship.py        infers sponsored results (ad links, affiliate tags, disclosures)
  ranker.py             scoring: similarity + axes + agreement − sponsored
  constraints.py        pulls "$15", "1 mile", "4 stars" out of the query
  prompt.py             prompt shared by the AI providers
  geocode.py            distance: neighbourhood lookup, venue geocoding
  conversation.py       multi-turn memory, keyed by the NLIP conversation token
  state.py              shared state: in memory, or Postgres (Supabase) when ANGEL_DATABASE_URL is set
  cache.py              query cache (3-hour TTL)
  limits.py             per-client rate limit + daily query cap
  providers/            one file per provider, plus base.py and mock.py
static/
  index.html            demo UI
supabase/
  config.toml           Supabase CLI config (local only; production ignores it)
  migrations/           schema, applied by Supabase's GitHub integration on merge
scripts/
  check_database.py     diagnose the database connection; set its password safely
tests/                  pytest suite, fully offline
docs/
  HANDOFF.md            start here: the takeover checklist
  SETUP.md              full setup, all providers, deployment, troubleshooting
  DESIGN.md             how scoring works and why
  KNOWN_ISSUES.md       unfinished work, written as starter tasks
  DEV_FALLBACK.md       the REST fallback server
  history/              the original team's status report (April 2026)
CONTRIBUTING.md         workflow, adding a provider, testing rules
CLAUDE.md               instructions for Claude Code, if you use it
```

---

## Glossary

| Term | Meaning |
|---|---|
| **NLIP** | Natural Language Interaction Protocol. Messages are multipart: a main text part plus labeled submessages. We use the upstream `nlip_sdk` / `nlip_server` libraries unmodified. |
| **Provider** | One source of results (OpenAI, Gemini, Brave, …). Each one is a `BaseProvider` subclass that returns `ProviderResult`s. |
| **Constraints** | Hard limits parsed from the query: budget, max distance, min rating. |
| **Intent** | Which axis the user cares about most: price, distance, rating, or general. Guessed from keywords unless the user sets **priority**. |
| **P1 / P2 / P3** | The three axes: price, distance, rating. |
| **Axis score** | How well a result does on P1–P3 against the constraints, weighted by intent. |
| **Agreement (consensus)** | A bonus when different providers name the same place. |
| **Enrichment** | Geocoding venues that other providers named, so they can be scored on distance. |
| **Conversation token** | NLIP's ID for a conversation. We store turns against it so "cheaper than that" works. |
| **Turn mode** | The client's choice: `continue` a conversation or start a `new` one. |
| **Fallback mode** | Plain FastAPI `POST /query`, used when the NLIP libraries can't be imported. |
| **Sponsored detection** | Rules in `sponsorship.py` that flag paid placements (ad links, affiliate tags, disclosures). Each flag records a `sponsored_reason`. |
| **State store** | Where the cache, conversations and limits live (`state.py`): process memory, or Postgres when `ANGEL_DATABASE_URL` is set. |
| **Session pooler** | Supabase's IPv4 connection endpoint. Use it instead of the Direct connection, which needs IPv6. |

---

## License

[MIT](LICENSE). The NLIP libraries this project depends on are separately
licensed under Apache-2.0 by their authors.
