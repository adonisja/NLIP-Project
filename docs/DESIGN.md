# Design

How Angel Filter ranks results, and why it works the way it does. Several
choices here look odd until you know what they prevent. Each section ends with
a **Before you change this** note so you don't undo one by accident.

---

## 1. The score

Every result that survives filtering gets one number:

```
score = 0.50 × similarity        0–1
      + 0.35 × axis_score        0–1
      + 0.15 × consensus_factor  0–1
      − 0.20 if sponsored
```

Both scoring paths (embeddings and the keyword fallback) compute this in one
function, `_assemble_score` in [ranker.py](../angel_filter/ranker.py). The
weights are constants at the top of that file (`W_SIMILARITY`, `W_AXIS`,
`W_CONSENSUS`, `SPONSORED_PENALTY`).

**Before you change this:** the three weights add up to 1.0, and each term is a
weight times a 0–1 value, so an unsponsored result always scores between 0 and
1. Keep it that way when tuning. Folding a weight into its own term as well
applies it twice and quietly shrinks that signal. `test_assemble_score.py`
pins the constants.

---

## 2. Similarity (50%)

The **preference** (if the user gave one, otherwise the query) and each
result's title and snippet are embedded. The cosine similarity between them is
the score.

Embedding backends, in order: Ollama `nomic-embed-text` (local), then OpenAI
`text-embedding-3-small` (for cloud deploys where Ollama can't run), then
keyword overlap as a last resort. Ollama embedding calls run concurrently on
the async client, so ranking N results costs about one round trip, not N.

**Before you change this:** if you change the embedding model, update
`DEFAULT_EMBED_MODEL` in `ranker.py` and the docs. Keep every call async. A
blocking call here stalls the whole server.

---

## 3. The three axes (35%)

### Constraints and gap scores

[constraints.py](../angel_filter/constraints.py) pulls explicit limits out of
the query with regular expressions:

| Axis | Example | Gap |
|---|---|---|
| P1 price | `under $15` | `price − budget` (negative means under budget) |
| P2 distance | `within 1 mile` | `distance − max_distance` (negative means closer) |
| P3 rating | `rated 4 stars` | `min_rating − rating` (negative means it meets the bar) |

Each gap maps to a 0–1 score (`_compute_gap_scores`). Meeting a limit scores
well, and beating it scores better, so the axis still separates results that
all qualify. With a minimum of 4 stars, a 4.0★ result scores 0.60 on P3 and a
5.0★ result scores 1.00.

### Hard filters

Before scoring, `_apply_hard_constraints` drops results that are **more than
25% over budget** or **more than 0.5★ below the minimum rating**. A result
with no data for an axis passes through. If the filter would remove
everything, nothing is removed, because showing something beats showing
nothing.

The P3 scale is anchored to the same 0.5★ cutoff: a result sitting exactly on
the filter boundary scores 0.0. The two thresholds agree by construction.

### Intent and priority

`detect_intent` looks for keywords ("cheap", "nearest", "best rated", …) to
decide which axis matters most. That axis gets 60% of the axis score and the
other two get 20% each. A general query splits evenly. No axis is ever
dropped to zero.

The user can override the guess. The UI's priority picker (Auto, Price,
Distance, Rating) sends a `priority` submessage over NLIP, or a `priority`
field on REST `/query`. Auto sends nothing. An unrecognised value logs a
warning and falls back to detection. Priority is part of the cache key,
because a price ranking and a rating ranking of the same query are different
results.

---

## 4. Missing data is not average data

This is the subtlest part of the ranker, and it's all in `_axis_bonus`.

Providers report different fields. The AI providers give price and rating but
never distance: they don't know where the user is, and would make it up. Brave
is a web search API, so its results are pages, not places, and have no price,
distance, or rating at all.

**Why not score a missing axis as 0.5?** Because then a bare search result
with no data at all lands within 0.11 of a result that meets every limit. That
gap is smaller than the 0.20 sponsored penalty, which would make the penalty
meaningless.

What happens instead:

1. **Weights are renormalised over the axes that have data.** A result with
   price and rating but no distance is judged on price and rating.
2. **The axis the user asked about can't be dodged.** If it's missing, it
   keeps its weight at a neutral 0.5 instead of being shared out. Otherwise a
   result that never said where it is could win a "nearest" query by being
   cheap.
3. **Leaving data out is never an advantage.** Renormalising alone would judge
   a partial result on its one best axis, while a complete result is judged on
   its average. A venue that reported only a strong price (0.90) would beat
   one reporting a good price, distance *and* rating (0.85 average). So an
   incomplete result is pulled toward neutral in proportion to how much it
   left out. It still gets credit for what it reported, but it can't outrank
   a result reporting the same values on every axis.
4. **Breaking a limit is worse than saying nothing.** Staying silent is
   penalised, but reporting a value that actually breaks a limit is penalised
   more. A venue 3 miles away on a "within 1 mile" query ranks below one that
   gave no distance at all.

**The API says which axes were real.** `axis_scores` holds a 0.5 placeholder
for a missing axis, which looks identical to a genuine 0.5. So every result
also carries `axis_scored`:

```json
"axis_scores": {"P1_price": 0.93, "P2_distance": 0.5, "P3_rating": 0.84},
"axis_scored": {"P1_price": true, "P2_distance": false, "P3_rating": true}
```

The UI uses it to show "no data" and to draw incomplete results hollow in the
3D plot.

**Before you change this:** `test_axis_scoring.py` and
`test_axis_scored_mask.py` encode every rule above. If a change makes one of
them fail, that's the design objecting, not a flaky test.

---

## 5. Distance: measured, never guessed

Distance (P2) comes from exactly two places, both in
[geocode.py](../angel_filter/geocode.py) and
[providers/google_places.py](../angel_filter/providers/google_places.py):

1. **The Google Places provider** returns nearby venues with coordinates and
   computes their distance directly.
2. **Enrichment.** After the fan-out, venues that other providers named but
   gave no distance for are geocoded through Places and measured against the
   user (haversine distance). A match whose name doesn't agree with what was
   asked for is rejected. Titles that look like articles ("The 10 Best Tacos
   in Brooklyn") are skipped, because they would geocode to *something*, and
   that something would be wrong. Anything that can't be resolved stays
   `None`.

**The AI providers are told the neighbourhood, not the position.** The user's
coordinates are turned into a place name ("Manhattan, NY") and added to every
AI prompt. Without this, the models were asked for "lunch nearby" with no idea
where nearby was, and invented places (*The Green Bowl*, *Taco Haven*) that
geocoded to nothing. With it, they return real ones (*Joe's Pizza*, *Los Tacos
No. 1*). The prompt still forbids `distance_miles`, because the model knows the
area, not the user's exact location.

### Enrichment cost

Every geocode is a billed Places call. A fan-out returns around 40 results and
the user sees 5, so unbounded it cost 17 lookups per query. Three layers limit
it:

| Layer | Default | What it does |
|---|---|---|
| Shortlist | 12 | Pre-ranks on the signals that cost nothing (keyword overlap, reported price and rating) and geocodes only the top 12. Results that already have a distance always stay. |
| Venue cache | — | Venue name → coordinates, kept for the life of the process. Names that couldn't be resolved are cached too, so an invented venue isn't looked up again. |
| Ceiling | 15 | Hard cap per query. |

Measured on a repeated query (clearing the query cache each time so the whole
pipeline runs): 17 → 6 → 4 → 2 lookups.

**Before you change this:** the shortlist is deliberately much larger than
`top_k`. On a distance query, distance carries 60% of the axis score, so it
has to be able to reorder the final ranking. A shortlist the size of `top_k`
would lock in the winner before distance was ever measured.

---

## 6. Agreement between providers (15%)

Results that several providers mention get a bonus. With embeddings, titles
cluster together when their cosine similarity is ≥ 0.75, so "Joe's Pizza" and
"Joe Pizza" match. The keyword fallback matches on a normalised title instead.
**Results from the same provider never cluster together**, so one chatty
provider can't manufacture its own agreement.

The bonus scales with the number of *additional* providers that agree, capped
at 2: one extra provider earns half the 15%, two or more earn all of it. The
cap stops a mediocre result from winning just because everyone mentioned it.

### Duplicates

Clustering refuses same-provider matches, which is right for *counting*
agreement but means copies of a venue aren't merged. Without an extra step, a
venue every provider named took one slot per mention (one venue once held
three of the top five). `_collapse_duplicates` runs after scoring and before
`top_k`: it keeps the best-scoring copy, which keeps the agreement count it
earned.

---

## 7. The sponsored penalty (−0.20)

A flat deduction for any result with `sponsored=True`, no matter how well it
matches. **This is the thesis of the project.** A paid placement has to be
clearly better than the organic alternatives to reach the top.

**Current gap:** the penalty works, but no live provider ever sets
`sponsored=True`. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md#2-no-live-result-is-ever-flagged-as-sponsored).

**Before you change this:** don't remove, reduce, or route around the penalty
without the team (and your mentor) agreeing. Several other numbers are sized
against it. For example, section 4 exists so that missing data can't erase
0.20 worth of difference.

---

## 8. Multi-turn conversations

NLIP messages carry a **conversation token**. The browser creates it, keeps it
in `sessionStorage` (so a new tab is a new conversation), and sends it with
every turn. `nlip_server` builds a new session object for every request, so
memory can't live on the session. It lives in
[conversation.py](../angel_filter/conversation.py), keyed by the token, and
expires after 30 minutes.

```
turn 1   "lunch spots under $20"     → budget $20, top result $16
turn 2   "cheaper than that"         → budget tightened, cheaper results
```

**The client decides whether a turn is a follow-up.** After a query, the UI
shows **New Search** and **Continue from Previous**, and sends the choice as a
`turn_mode` submessage. The server never guesses from wording, because
guessing goes wrong both ways: a new search containing "cheaper" looks like a
follow-up, and an unusually phrased follow-up looks like a new search. New
Search also throws away the token. If a client sends no `turn_mode`, the
server falls back to a phrase check. If it sends no token, it gets an ordinary
one-shot search, which is what keeps plain text-only NLIP clients working.

**Follow-ups are resolved in two ways, in order:**

1. **Fixed adjustments.** "cheaper", "closer", and "better rated" tighten the
   previous turn's limits. There's no model call, so they're predictable and
   testable offline. A follow-up must be short: a full sentence that happens
   to contain "cheaper" is a new search.
2. **Model fallback.** Anything else ("what about vegetarian options") goes to
   the providers with the recent turns prepended. It costs prompt tokens,
   which is why it only runs when the fixed adjustments don't match.

**A follow-up is measured from what the previous turn returned.** If the user
saw an $18 result under a $50 budget, "cheaper" means cheaper than $18.
Anchoring on the $50 nobody reached would barely change the results. The reply
also says what changed (`Refined: budget tightened to $8.00`), so the result
set never changes without explanation.

---

## 9. Server structure

### NLIP and the fallback app

[server.py](../angel_filter/server.py) tries to import `nlip_sdk` and
`nlip_server`. If that works, it builds the NLIP app (`POST /nlip/`) and adds
the HTTP routes. If not, it builds a plain FastAPI app with the same routes
plus `POST /query`. Both branches must expose the same routes, and
`test_route_parity.py` checks that. See [DEV_FALLBACK.md](DEV_FALLBACK.md).

The NLIP reply has two parts: a text summary for people, and a JSON
submessage holding the full ranking (scores, axes, sponsored flags) for
agents. The input is read by submessage label, not flattened into one string,
so a preference never leaks into the query text.

We use the upstream NLIP libraries as they are. **Don't fork or patch them.**

### The /health override

`nlip_server.setup_server()` adds its own `/health`, which returns a bare
`{"status": "healthy"}`. FastAPI uses the first matching route, so declaring
ours afterwards would never be reached. `_drop_route` removes upstream's
`/health` before we declare ours. Upstream's `/health/live` and
`/health/ready` are left alone.

### Cost limits (there is no login)

Every search spends provider credit, and there is no login (designing one is
[starter task #1](KNOWN_ISSUES.md#1-design-and-build-login-and-access-control)).
[limits.py](../angel_filter/limits.py) applies two limits to both query
routes: a sliding-window rate limit per client IP, and a daily cap across all
callers. `/query` gets them through a FastAPI dependency. `/nlip/` belongs to
`nlip_server`, so the same check runs as middleware matched to that exact
path. A rejected request never reaches the providers and doesn't count
against the daily cap.

The client IP comes from the socket unless `ANGEL_TRUSTED_PROXY_HOPS` says
how many reverse proxies to trust. Then it's read from that position in
`X-Forwarded-For`, counting from the right, because the entries to the left
were written by the caller and can't be believed.

**Before you change this:** any new route that triggers provider calls has to
go through these limits. `tests/test_limits.py` covers both routes.

### State

The query cache, conversation memory, rate limiter, daily counter, and venue
coordinate cache are all plain in-memory Python objects. That's fine for one
process and a demo. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for what it means
beyond that.
