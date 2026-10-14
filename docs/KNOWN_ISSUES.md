# Known issues and starter tasks

What's unfinished as of the handoff (October 2026), roughly in priority
order. Each item says what's wrong, where to look, and a way in. None of them
is fully specified, so talk to your mentor before starting the bigger ones.

---

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

## 2. Measure and extend sponsored detection

**Why it matters:** the sponsored penalty is the project's thesis, and
detection is what decides whether it fires on real searches.

No live provider reports ads, so [sponsorship.py](../angel_filter/sponsorship.py)
infers them after the fan-out: links through ad networks, ad-click IDs and
paid UTM tags, affiliate parameters, and disclosure text ("Sponsored",
"paid partnership", "#ad"). The rules are deliberately conservative, and
every flag records its reason, which the reply and the UI show.

What nobody knows yet is **how often it fires on real searches**. Brave
returns organic web results and the AI providers don't see ads, so on
today's providers it may rarely trigger. Ways in:

- **Measure it.** Log how many results each rule flags over a few hundred real
  searches, and hand-check a sample of flagged *and* unflagged results for
  false positives and misses.
- **Add a provider that surfaces paid listings**, so the penalty has real ads
  to act on. Check its terms of service first, and raise them in the PR.
- **Add rules** where the measurements show misses: more ad and affiliate
  networks, or listicle signals such as "best of" pages that are mostly
  affiliate links. Each rule needs a test for what it must *not* match, next
  to the existing ones in `tests/test_sponsorship.py`.

## 3. Validate location input

`_extract_location` in `server.py` (NLIP) and the `lat`/`lng` fields of
`POST /query` accept any number. NaN, infinity, and out-of-range coordinates
(latitude 500) get through, and so does `true`, because Python treats a
boolean as an integer. Bad coordinates then go to Google Places (which
rejects them, failing that provider) and into distance math. Reject anything
outside −90…90 / −180…180 or not finite, and treat it the same as no location.
A good first issue: small, self-contained, easy to test.

## 4. Say when nothing met the user's limits

When every result breaks a hard limit (all over budget, say),
`_apply_hard_constraints` in `ranker.py` returns all of them unfiltered, and
the UI shows them as if they qualified. Showing something is reasonable, but
the reply should say "nothing met your budget, here are the closest". That
means a flag on the response, a line in the NLIP text reply, and a notice in
the UI.

## 5. Smaller cleanups

- **Brave returns pages, not places.** Brave's web search results have no
  price, rating, or coordinates, so they only compete on text similarity (and
  distance when enrichment can geocode the title). Places data would need
  Brave's separate local search offering or a different places API; check the
  plan and terms before building on either.
- **Should the fallback server stay?** It was kept as insurance while NLIP
  installs were unreliable. They now install cleanly through Poetry and in
  deployments, and the UI can't search in fallback mode anyway, so it could
  go. It's a team decision. Until then, keep both branches in sync.
