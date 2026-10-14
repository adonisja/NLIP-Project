# Fallback mode

At startup, [angel_filter/server.py](../angel_filter/server.py) tries to
import the NLIP libraries:

```python
from nlip_server.server import NLIP_Application, NLIP_Session, setup_server
from nlip_sdk.nlip import NLIP_Factory, NLIP_Message
```

If that fails, usually because dependencies were installed piecemeal with
`pip` instead of through `poetry install` or `requirements.txt`, it logs a
warning and builds a plain FastAPI app instead.
`/health` then reports `"mode": "fallback"`.

## What's different

Both modes run the same pipeline and expose the same routes (`/`,
`/health`, `/metrics`, `/history`, `/cache/clear`, `/query`).
`tests/test_route_parity.py` keeps them in sync. The only difference is that fallback mode has **no `/nlip/` endpoint**. That matters
more than it sounds: the demo UI sends every search to `/nlip/`, so **the UI
can't search in fallback mode**. Only direct `POST /query` calls (curl,
Swagger at `/docs`) work, and there are no multi-turn follow-ups.

## POST /query

Counts against the rate limits and the daily cap (see [SETUP.md](SETUP.md#limits-and-cost-controls)).

```json
{
  "query": "lunch under $15 within 1 mile",
  "preference": "quiet, vegetarian options",
  "lat": 40.7359,
  "lng": -73.9911,
  "priority": "price"
}
```

Only `query` is required. `preference` is what similarity is measured against
(it defaults to the query). `lat`/`lng` enable distance scoring. `priority`
(`"price"`, `"distance"`, `"rating"`) overrides the guessed intent; leave it
out for auto.

The response carries the same ranking payload as the JSON submessage in an
NLIP reply: `providers_used`, `providers_failed`, the parsed `constraints`,
and `results`. Each result has its `score`, `axis_scores` / `axis_scored`,
`sponsored` flag, and a `rationale`. `/docs` (Swagger) shows the exact shape.

## Should it stay?

It was kept as insurance while NLIP installs were unreliable. NLIP now
installs cleanly through Poetry and in deployments (`requirements.txt` is
generated from `poetry.lock`, and CI checks that a pip install runs in NLIP
mode). Since the UI moved to `/nlip/`, the fallback can't keep the demo
working on its own anyway. Removing it is a reasonable team decision.
