# Known issues and starter tasks

What's unfinished as of the handoff (October 2026), roughly in priority
order. Each item says what's wrong, where to look, and a way in. None of them
is fully specified, so talk to your mentor before starting the bigger ones.

Bugs (B-numbered) are being fixed before the handoff, and each entry is
removed when its fix merges. Numbered items are starter tasks for the new
team.

---

# Bugs being fixed before handoff

## B3. An embedding failure fails the whole search

The ranker checks once whether Ollama or OpenAI embeddings are available and
keeps that answer forever. If the backend fails afterwards, `rank()` raises
and the whole search fails, after the providers have already been paid for.
This keeps happening on every search until a restart. On Render, an OpenAI
rate-limit error on the embeddings call would do the same. The Ollama client
is also created with no timeout, so a stalled Ollama hangs every search.

**Fix:** if embedding fails, fall back to keyword scoring for that search and
check the backend again next time. Give the Ollama client a timeout.

## B4. `/metrics` doesn't count searches made through `/nlip/`

The query counter, latency histogram, and sponsored-penalty counter are only
updated in the `/query` handlers. The UI searches through `/nlip/`, so the
metrics show close to zero for real use.

**Fix:** record the same metrics in the NLIP session.

---

# Starter tasks

## 1. Design and build login and access control

**Why it matters:** right now anyone who can reach the server can use it.
Every search spends provider credit, and some routes expose or change shared
state.

The original team used GitHub OAuth with a username allowlist. It was
removed at handoff on purpose, so you can decide what access this app
*should* have rather than inheriting a choice made for a class demo. To see
how it worked, run `git log --diff-filter=D -- angel_filter/auth.py` and check
out the commit before the one shown.

What protects the app today is [limits.py](../angel_filter/limits.py): a rate
limit per client IP, and a daily cap on searches across all callers. That
limits spending; it doesn't control *who* can do what.

**Questions to answer first** (write the answers in your PR description or a
short design doc, then review them with your mentor before writing code):

1. **Who should be able to use it?** Anyone? Anyone with an account? Only
   invited people? Different answers fit a public demo, a class project, and a
   tool for a partner organisation.
2. **What needs protecting, and from whom?**
   - `POST /query` and `POST /nlip/`: these spend money.
   - `GET /history`: currently returns *every* caller's recent queries, which
     is a privacy problem as soon as there's more than one user.
   - `POST /cache/clear`: anyone can wipe the shared cache and force fresh,
     paid fan-outs.
   - `GET /metrics`: harmless, but usually not public.
3. **How do non-browser clients authenticate?** NLIP exists so *agents* can
   call the service, and agents don't log in through a browser. Cookies,
   API keys, and bearer tokens each fit some clients and not others.
4. **What becomes per-user?** Rate limits (swap `client_key` for a user ID),
   query history, and conversation tokens. Today any caller who learns a
   conversation token can continue that conversation.

**Things to know before building:**
- `nlip_server` registers `/nlip/` itself, so you can't add a FastAPI
  dependency to it. `nlip_limit_middleware` in `limits.py` shows the pattern
  that works: a middleware that checks the request before it reaches the
  route. Don't fork or patch the NLIP libraries.
- Both `server.py` branches (NLIP and fallback) must stay in sync
  (`test_route_parity.py`).
- `itsdangerous` is still a dependency. Starlette's `SessionMiddleware` needs
  it if you go with cookie sessions; remove it with `poetry remove` if you
  don't.

**Done means:** a test for every protected route showing an unauthenticated
request is rejected, *including `/nlip/`*; limits keyed by user once users
exist; `.env.example`, `docs/SETUP.md`, and `render.yaml` updated with the new
configuration.

## 2. No live result is ever flagged as sponsored

**Why it matters:** the sponsored penalty is the project's thesis, and right
now it only ever fires on test data.

Every live provider hardcodes `sponsored=False` when building its results. See
the `ProviderResult(...)` call at the end of each file in
[angel_filter/providers/](../angel_filter/providers/). The penalty itself
works: `_assemble_score` in [ranker.py](../angel_filter/ranker.py) applies it
whenever the flag is set, and the UI already has a callout showing how far a
sponsored result dropped. Nothing upstream sets the flag.

**Ways in:**
- A provider whose API actually returns ads or sponsored listings, mapped
  straight onto the flag.
- Heuristics that detect sponsorship: affiliate or tracking parameters in URLs
  (`utm_`, `ref=`, `aff`), known ad and affiliate domains, promotional
  language, "sponsored" or "ad" markers in snippets.
- Asking the AI providers to report whether a recommendation is a paid
  placement. This is weak on its own, since models rarely know.

Keep detection in a separate step (e.g. a `sponsorship.py` run in the
orchestrator after fan-out) rather than scattering it across providers, so it
can be tested with canned results. Getting the rate of false positives right
matters: flagging organic results would undermine the project's claim.

## 3. All state is in memory

The query cache, conversation memory, per-client rate limits, the daily query
counter, and the venue geocode cache are module-level Python objects. As a
result:

- a restart (including Render's free tier waking from sleep) wipes all of it,
  including the daily spending count
- running more than one worker gives each worker its own limits and cache, so
  the caps no longer hold

The query cache also never deletes expired entries: a stale entry is only
skipped when read, so memory grows until a restart (the daily cap bounds how
fast).

SQLite is enough for one instance; Redis if you scale out. The venue cache
matters most for cost: it's what brings Places lookups down to nearly zero on
repeat queries.

## 4. The ranker's Ollama client ignores `OLLAMA_URL`

The Ollama *provider* reads `OLLAMA_URL`, but the *ranker's* embedding client
(`Ranker._ollama` in `ranker.py`) calls `ollama.AsyncClient()` with no host,
so embeddings always go to `localhost:11434` (or the `ollama` library's own
`OLLAMA_HOST` variable). Pass the configured URL through so a remote Ollama
works for both.

## 5. Validate location input

`_extract_location` in `server.py` (NLIP) and the `lat`/`lng` fields of
`POST /query` accept any number. NaN, infinity, and out-of-range coordinates
(latitude 500) get through, and so does `true`, because Python treats a
boolean as an integer. Bad coordinates then go to Google Places (which
rejects them, failing that provider) and into distance math. Reject anything
outside −90…90 / −180…180 or not finite, and treat it the same as no location.
A good first issue: small, self-contained, easy to test.

## 6. Say when nothing met the user's limits

When every result breaks a hard limit (all over budget, say),
`_apply_hard_constraints` in `ranker.py` returns all of them unfiltered, and
the UI shows them as if they qualified. Showing something is reasonable, but
the reply should say "nothing met your budget, here are the closest". That
means a flag on the response, a line in the NLIP text reply, and a notice in
the UI.

## 7. Smaller cleanups

- **Brave returns no structured fields.** Its results only ever compete on
  text similarity. Parsing prices or ratings out of snippets would help, or
  drop Brave for a places-oriented API.
- **The DuckDuckGo provider is unused.** `providers/duckduckgo.py` is exported
  but never registered in `server.py`, and its Instant Answer API returns
  almost nothing for this kind of query. Delete it, or register it on purpose.
- **Should the fallback server stay?** It was kept as insurance while NLIP
  installs were unreliable. They now install cleanly through Poetry and in
  deployments, and the UI can't search in fallback mode anyway, so it could
  go. It's a team decision. Until then, keep both branches in sync.
- **The NLIP libraries track `main` in `pyproject.toml`.** `poetry.lock` pins
  the exact commits, so nothing moves until someone runs `poetry lock` or
  `poetry update`. When you do, read the upstream changes, run the tests, and
  regenerate `requirements.txt` (the command is in its header).
- **`start.sh`** calls `python3.12` directly instead of `poetry run`, so it
  only works if you also pip-installed everything. Update it or delete it.
- **Port defaults disagree.** Running `python -m angel_filter.server` defaults
  to port 8000; everything else uses 8005.
- **Constraint parsing is regex-based.** It handles "$15", "less than 15
  dollars", "within 1 mile", "walking distance", and "4.5+ stars", but not
  spelled-out numbers ("under twenty bucks" yields no budget). See
  [constraints.py](../angel_filter/constraints.py).
