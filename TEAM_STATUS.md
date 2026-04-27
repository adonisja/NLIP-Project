# Angel Filter : Team Status Report
**Date:** April 27, 2026   
**Next milestone:** May 1 : integrated demo (Dinesh returns)

---

## 1. What's Been Built

### Sub-task A: Proxy Core ✅ Complete

The FastAPI server is standing and handles the full pipeline end-to-end.

**`angel_filter/server.py`** : dual-mode server:
- When NLIP libraries are available (they are), runs as a proper NLIP application using `NLIP_Application` / `NLIP_Session` from `nlip_server`, receiving NLIP messages at `POST /nlip/`
- Also exposes `POST /query` and `GET /` for the demo UI : same pipeline, REST-accessible without needing the NLIP client
- `/health`, `/docs` (Swagger UI), and `/metrics` (Prometheus) all live

**`angel_filter/orchestrator.py`** : the fan-out engine:
- Fires all registered providers simultaneously via `asyncio.gather`
- A provider that times out or throws does not kill the others : each failure is caught, logged, and reported in `providers_failed`
- Returns a flat ranked list regardless of which providers succeeded

**`angel_filter/providers/base.py`** : the contract every provider must follow:
- `BaseProvider` : subclass this, set `name`, implement `async query()`
- `ProviderResult` : normalized shape every provider must return (title, snippet, url, provider, sponsored, price, raw)
- `ProviderError` : raise this on recoverable failures

### Sub-task D: Ollama Integration ✅ Complete

**`angel_filter/ranker.py`** : the scoring engine:
- Embeds the user's preference and each result using `nomic-embed-text` via Ollama
- Scores via cosine similarity
- Subtracts `SPONSORED_PENALTY = 0.15` from any sponsored result : this is the thesis of the project
- Automatically falls back to keyword overlap if Ollama is unreachable, so the demo never dies

**Ollama status:** `nomic-embed-text` is pulled and confirmed working. Real similarity scores are live (e.g. `partial match (similarity 0.47)`).

### Additional Providers ✅ Complete

**`angel_filter/providers/duckduckgo.py`** : hits the real DDG Instant Answer API, no key needed, returns normalized `ProviderResult` objects. Use this as the exact pattern for Google and Bing.

**`angel_filter/providers/mock.py`** : canned shopping results used by all tests. Anyone can develop and test without network or Ollama.

### Observability ✅ Complete

- **`GET /health`** — returns `ok`, `mode` (nlip or fallback), `nlip_available`, `uptime_seconds`, and the list of active `providers`. Wired on both the NLIP path and the fallback path via a shared `_health_response()` helper in `server.py`.
- **`GET /docs`** — Swagger UI, auto-generated from route definitions. Use this to explore and test the API interactively.
- **`GET /metrics`** — Prometheus-format metrics endpoint, tracking:
  - `angel_filter_queries_total` — success/error counts (labelled by status)
  - `angel_filter_query_duration_seconds` — per-request latency histogram
  - `angel_filter_sponsored_penalties_total` — how many results were penalized (good demo talking point)
  - `angel_filter_start_timestamp_seconds` — server start Unix timestamp (subtract from `now()` in Grafana to get uptime)

### Git Workflow ✅ Enforced

All changes go through pull requests — no direct commits to `main`, including from project owners. See the Contributing section at the top of `README.md` for the branch + PR workflow.

### Tests ✅ 3/3 Passing

Tests live in `tests/test_orchestrator.py`. They require no network or Ollama : fast and deterministic.

1. Full pipeline returns ranked results
2. Sponsored item does not end up ranked #1
3. One broken provider does not kill the others

---

## 2. Known Issues & Things That Need More Work

### DuckDuckGo Returns Sparse Results ⚠️
The DDG Instant Answer API is lightweight : it returns structured data only for queries that match known topics (famous people, places, definitions). For travel or shopping queries like "cheapest flight JFK to LAX" it returns nothing, so only `MockProvider`'s canned data shows up in demo results. This is expected behavior from the API, not a bug in our code.

**Impact:** The demo currently shows toilet paper mock data regardless of what you search.  
**Fix:** Needs real Google or Bing provider (Teammate J's lane), or a temporary swap of `MockProvider`'s canned data to match the demo vertical.

### MockProvider Has Wrong Demo Data ⚠️
`mock.py` returns toilet paper products. Until a real provider is wired in, the demo tells an incoherent story for flight or shopping queries.  
**Fix:** Quick swap of canned data in `mock.py` to flight or shopping results : 10 minutes of work. Should happen before May 1.

### NLIP Message Extraction Needs Verification ⚠️
The NLIP session in `server.py` extracts the query from an incoming message using `msg.content`. This works for text messages, but the exact field behavior for compound or multi-part NLIP messages (per standard 430/431) hasn't been stress-tested.  
**Owner:** Teammate I : needs a test that sends a real NLIP-formatted request to `POST /nlip/` and validates the response shape.

### Ollama Latency is High 🔍
In testing, a single `/query` request took ~1.59 seconds (visible in `/metrics`). Most of that is Ollama making one embedding call per result. With 4 mock results + 1 user preference that's 5 sequential embedding calls.  
**Impact:** Acceptable for demo, but will feel slow with 10+ real results.  
**Future fix:** Batch embedding calls, or cache embeddings for identical result text. Not urgent for May 1.

### No Session State 🔍
Each query is stateless : the proxy does not remember clarifier answers between turns. `NLIPSession` supports per-connection state but we haven't wired it up yet.  
**Impact:** The clarifying-question flow (Prompt Engineering sub-task C) can't persist answers yet.  
**Fix:** Wire `AngelFilterSession` state in `server.py` once the prompt team defines the clarifier policy. Coordinate before May 4.

### Sponsored Detection is Mocked 🔍
Right now `sponsored=True` only gets set if the provider explicitly tells us. For the mock data, it's hardcoded. For real providers (Google, Bing), we don't yet have heuristics to detect sponsored/ad results automatically.  
**Impact:** The penalty works correctly once `sponsored` is set : the detection logic is what's missing.  
**Future fix:** Position-on-page signals, URL patterns, brand-heavy language detection. Owned by Prompt Engineering sub-task D.

---

## 3. Project Layout

```
NLIP_project/
  angel_filter/           ← Python package (import as angel_filter.*)
    server.py             ← start here : NLIP app + REST fallback
    orchestrator.py       ← parallel fan-out logic
    ranker.py             ← Ollama embeddings + sponsored penalty
    providers/
      base.py             ← BaseProvider, ProviderResult, ProviderError
      duckduckgo.py       ← live provider (reference impl for new ones)
      mock.py             ← canned data for tests and offline demo
  static/
    index.html            ← demo UI (UX teammate to redesign)
  tests/
    test_orchestrator.py  ← 3 passing tests
  docs/
    DEV_FALLBACK.md       ← how to run without NLIP installed
  pyproject.toml          ← Poetry deps including nlip_sdk + nlip_server
  refs/                   ← upstream NLIP repos for reading, not editing
    nlip_sdk/             ← NLIP_Message, NLIP_Factory
    nlip_server/          ← NLIP_Application, NLIP_Session, setup_server
    nlip_web/             ← reference implementation
```

---

## 4. How to Run

```bash
# First time only
poetry install
ollama pull nomic-embed-text

# Every time : two terminals
ollama serve                               # terminal 1 (skip if Ollama app is running)
poetry run python -m angel_filter.server   # terminal 2

# Then open:
# http://localhost:8000        → demo UI
# http://localhost:8000/docs   → Swagger : interactive API explorer
# http://localhost:8000/metrics → Prometheus metrics
```

**Run tests (no network or Ollama needed):**
```bash
poetry run pytest
```

---

## 5. Next Steps by Role

### Teammate I : NLIP Layer (due Apr 28)

- Read `refs/nlip_sdk/nlip_sdk/nlip.py` : specifically `NLIP_Message`, `NLIP_Factory`, and `NLIP_Factory.create_text()`
- Read `refs/nlip_server/nlip_server/server.py` : `NLIP_Application`, `NLIP_Session`, `setup_server`
- Review the wiring in `angel_filter/server.py` lines 49–91 : it's working but needs your eyes to confirm `msg.content` extraction and `NLIP_Factory.create_text()` reply encoding match the 430/431 spec
- Write a test that sends a real NLIP-formatted request to `POST /nlip/` and validates the response
- Write the one-page team summary of how NLIP works

### Teammate J : Provider Integrations + Web Client

**Providers (due May 2):**
- `providers/duckduckgo.py` is your reference : copy it, rename, implement `query()` returning `ProviderResult` objects
- Google: Custom Search JSON API : needs `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` from env (no hardcoded keys : see pyproject convention)
- Bing: Bing Web Search API : needs `BING_API_KEY` from env
- Register each new provider in `_build_orchestrator()` in `server.py` : one line each
- Add a test in `tests/` using canned responses, no live network
- **Also:** swap `mock.py` canned data from toilet paper to flight or shopping results before May 1 so the demo makes sense

**Web client (due May 6, after providers stable):**
- `static/index.html` is functional : it POSTs to `/query` and renders ranked results with score/sponsored badges
- Wire Teammate Ma's designs into HTML/JS : the `/query` response schema is stable and won't change

### Teammate My + Teammate E : Prompt Engineering

The proxy has a clean seam for your work. No code changes needed to start.

- **System prompt (Teammate My, due Apr 30):** There is currently no prompt layer : the ranker scores by embedding similarity only. The natural place to add one is a new function in `ranker.py`. Coordinate with SWEs before touching that file. Commit prompts as versioned files in a `prompts/` directory.
- **Voting/combining logic (Teammate E, due May 3):** Results from all providers are currently pooled and ranked together. Weighted voting or LLM-as-judge would slot into `orchestrator.py` between the fan-out and the ranker call. Define the interface, SWEs will wire it.
- **Clarifying-question policy (shared, due May 4):** The session is currently stateless. Coordinate with SWEs on session state before building the clarifier flow : we need to agree on the data shape before either side implements.
- **Sponsored detection heuristics (shared, due May 6):** The penalty already works : what's missing is reliable detection. Position-on-page signals, URL patterns, brand-heavy language. This is prompt/heuristic work that feeds a `sponsored: bool` flag on `ProviderResult`.

### Teammate Ma : UX (due May 6)

The demo UI is at `static/index.html`. It already works end-to-end and renders:
- Ranked results with `#1`, `#2` position badges
- Score chips showing the similarity score
- Green "organic" / orange "sponsored · penalized" badges
- Rationale text explaining why each result ranked where it did

The API shape it depends on is **stable**:

```
POST /query
Body:  { "query": "...", "preference": "..." }

Response:
{
  "providers_used": ["duckduckgo", "mock"],
  "providers_failed": [],
  "results": [
    {
      "title": "...",
      "snippet": "...",
      "url": "...",
      "provider": "...",
      "score": 0.479,
      "rationale": "partial match (similarity 0.48)",
      "sponsored": false
    }
  ]
}
```

Hit `http://localhost:8000/docs` to explore the full API interactively before designing.

### Teammate P : PM

**Two blockers to resolve before Apr 28:**

1. **API key decision** : Teammate J needs Google and/or Bing API keys to build the real providers. DuckDuckGo is live but returns thin results for the demo vertical. This decision needs to happen now or Teammate J is blocked until May.
2. **Demo vertical** : the mock data currently shows toilet paper. Before May 1 someone needs to swap it to match whichever vertical (flights or shopping) we're demoing. Quick fix, but needs a decision on which vertical first.

---

## 6. Alignment with Project Plan

| Plan Item | Status |
|---|---|
| Proxy skeleton accepts NLIP requests | ✅ Done : `POST /nlip/` live |
| FastAPI fallback for offline dev | ✅ Done : `POST /query` live |
| DuckDuckGo provider end-to-end | ✅ Done |
| Local LLM ranking from mock data | ✅ Done : Ollama + nomic-embed-text live |
| Sponsored penalty | ✅ Done : `SPONSORED_PENALTY = 0.15` |
| `/health` endpoint (uptime + provider list, both server paths) | ✅ Done |
| Prometheus metrics (queries, latency, sponsored hits, start timestamp) | ✅ Done |
| Swagger UI (`/docs`) | ✅ Done |
| Branch + PR workflow documented in README | ✅ Done |
| 3 passing tests | ✅ Done |
| Google provider | ❌ Not started : Teammate J |
| Bing/Copilot provider | ❌ Not started : Teammate J |
| Real provider-based sponsored detection | ❌ Not started : Prompt Engineering |
| System prompt layer | ❌ Not started : Prompt Engineering |
| Clarifying-question flow | ❌ Not started : Prompt Engineering + SWEs |
| Session state | ❌ Not started : SWEs after clarifier policy defined |
| UI redesign | ❌ Not started : Teammate Ma |
| NLIP spec verification + team summary | ❌ Not started : Teammate I |
