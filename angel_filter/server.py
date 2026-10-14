"""NLIP server for the Angel Filter.

Follows the pattern documented in nlip_server's README: subclass NLIPApplication
and NLIPSession, then pass them to the server startup helper.

Reference: https://github.com/nlip-project/nlip_server — see echo.py for the
minimum viable example this is modeled on.

Run locally with:
    poetry run fastapi dev angel_filter/server.py

If the nlip_server imports below fail, it means the NLIP packages are not
installed yet (poetry install has not been run, or the repos weren't
accessible). The fallback FastAPI app at the bottom of this file lets you
still run the proxy for local testing — see docs/DEV_FALLBACK.md.
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import Depends, FastAPI, HTTPException, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from angel_filter.cache import CACHE
from angel_filter.conversation import (
    CONVERSATIONS,
    Turn,
    apply_refinement,
    build_context_prefix,
    effective_query,
    looks_like_a_refinement,
)
from angel_filter.limits import enforce_query_limits, nlip_limit_middleware
from angel_filter.constraints import QueryConstraints
from angel_filter.orchestrator import Orchestrator
from angel_filter.ranker import QueryIntent
from angel_filter.state import get_store
from angel_filter.providers import BraveProvider, GeminiProvider, GooglePlacesProvider, OllamaProvider, OpenAIProvider, WatsonXProvider

logger = logging.getLogger(__name__)

# The route nlip_server registers for message exchange (its router is mounted
# under the "/nlip" prefix). Rate limiting is applied to this exact path.
_NLIP_PATH = "/nlip/"

# --- Prometheus metrics -------------------------------------------------------
QUERY_COUNT = Counter(
    "angel_filter_queries_total",
    "Total number of /query requests",
    ["status"],  # "success" or "error"
)
QUERY_LATENCY = Histogram(
    "angel_filter_query_duration_seconds",
    "Time spent processing a /query request",
)
SPONSORED_PENALTY_COUNT = Counter(
    "angel_filter_sponsored_penalties_total",
    "Number of results that had the sponsored penalty applied",
)
UPTIME_GAUGE = Gauge(
    "angel_filter_start_timestamp_seconds",
    "Unix timestamp when the server process started",
)
_START_TIME = time.time()
UPTIME_GAUGE.set(_START_TIME)


# --- Build the orchestrator once at import time ---
def _build_orchestrator() -> Orchestrator:
    providers = []
    # Add WatsonX if API key and project ID are present
    if os.getenv("WATSONX_API_KEY") and os.getenv("WATSONX_PROJECT_ID"):
        providers.append(WatsonXProvider())
        logger.info("WatsonXProvider enabled.")
    else:
        logger.info("WATSONX_API_KEY or WATSONX_PROJECT_ID not set — WatsonXProvider skipped.")

    # Add Brave if API key is present
    if os.getenv("BRAVE_API_KEY"):
        providers.append(BraveProvider())
        logger.info("BraveProvider enabled.")
    else:
        logger.info("BRAVE_API_KEY not set — BraveProvider skipped.")

    # Add Google Places if API key is present — the only provider that returns
    # real distance data (it also needs user lat/lng on each request).
    if os.getenv("GOOGLE_PLACES_API_KEY"):
        providers.append(GooglePlacesProvider())
        logger.info("GooglePlacesProvider enabled.")
    else:
        logger.info("GOOGLE_PLACES_API_KEY not set — GooglePlacesProvider skipped.")

    # Add OpenAI if API key is present
    if os.getenv("OPENAI_API_KEY"):
        providers.append(OpenAIProvider())
        logger.info("OpenAIProvider enabled.")
    else:
        logger.info("OPENAI_API_KEY not set — OpenAIProvider skipped.")

    # Add Gemini if API key is present
    if os.getenv("GEMINI_API_KEY"):
        providers.append(GeminiProvider())
        logger.info("GeminiProvider enabled.")
    else:
        logger.info("GEMINI_API_KEY not set — GeminiProvider skipped.")

    # Add Ollama if configured
    if os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_MODEL"):
        providers.append(OllamaProvider())
        logger.info("OllamaProvider enabled.")

    if not providers:
        raise RuntimeError(
            "No providers configured. Set at least one of: "
            "OPENAI_API_KEY, GEMINI_API_KEY, WATSONX_API_KEY, BRAVE_API_KEY, OLLAMA_MODEL"
        )

    return Orchestrator(providers=providers)


ORCHESTRATOR = _build_orchestrator()


# --- Shared health helper -----------------------------------------------------

def _drop_route(app, path: str) -> int:
    """Remove every route registered for `path`, returning how many were dropped.

    Used to take back a path an upstream router already claimed. FastAPI resolves
    a request against the first matching route, so re-declaring the path is not
    enough — the earlier registration keeps winning silently. Removing it is the
    only way to override without forking the library.

    Returns the count so callers can assert the removal actually happened; a
    silent zero would mean an upstream rename had quietly restored the shadowing.
    """
    matching = [r for r in app.routes if getattr(r, "path", None) == path]
    for route in matching:
        app.routes.remove(route)
    return len(matching)


def _constraints_from_payload(payload: dict) -> QueryConstraints:
    """Rebuild the constraints a turn ran under, from its serialised response.

    A cache hit has no OrchestratorResponse to read, so the stored payload is
    the only record of what bounds produced that ranking — and the next
    refinement adjusts exactly those.
    """
    c = payload.get("constraints") or {}
    return QueryConstraints(
        budget=c.get("budget"),
        max_distance=c.get("max_distance"),
        min_rating=c.get("min_rating"),
    )


async def _health_response(mode: str, nlip_available: bool) -> dict:
    store = get_store()
    state = {"backend": store.backend}
    if store.backend != "memory":
        # False means state is running on the in-process fallback right now
        # (database unreachable, or a free Supabase project has paused).
        state["reachable"] = await store.ping()
    return {
        "ok": True,
        "mode": mode,
        "nlip_available": nlip_available,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "providers": [p.name for p in ORCHESTRATOR.providers],
        "state": state,
    }


async def _run_search(**kwargs):
    """Run one fan-out through the orchestrator, recording it in /metrics.

    Every route that searches goes through here. The metrics were once
    recorded inline in the /query handlers only, so the NLIP session — the
    path the UI actually uses — never showed up in them at all.
    """
    with QUERY_LATENCY.time():
        try:
            response = await ORCHESTRATOR.handle_query(**kwargs)
        except Exception:
            QUERY_COUNT.labels(status="error").inc()
            raise
    QUERY_COUNT.labels(status="success").inc()
    sponsored = sum(1 for r in response.ranked if r.result.sponsored)
    if sponsored:
        SPONSORED_PENALTY_COUNT.inc(sponsored)
    return response


async def _rest_search(body) -> dict:
    """POST /query, shared by both server modes: cache, search, serialise.

    The two /query handlers were separate copies and had drifted: only the
    fallback one used the cache, so in NLIP mode every REST call paid for a
    fresh fan-out. `body` is the mode's QueryIn model.
    """
    # Fold location into the cache's preference component so two users at
    # different coordinates don't get served each other's distance results.
    cache_pref = _cache_pref(body.preference, body.lat, body.lng, body.priority)
    cached = await CACHE.get(body.query, cache_pref)
    if cached:
        logger.info("Cache hit for query: %r", body.query)
        return cached

    try:
        response = await _run_search(
            user_query=body.query,
            user_preference=body.preference,
            user_lat=body.lat,
            user_lng=body.lng,
            intent=_parse_priority(body.priority),
        )
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Query failed — providers may be temporarily unavailable. Please try again. ({type(exc).__name__})",
        )

    payload = {**_serialize_response(response), "cached": False}
    await CACHE.set(body.query, cache_pref, {**payload, "cached": True})
    return payload


# --- NLIP integration ---------------------------------------------------------
# Per nlip_server's README, we subclass NLIPApplication and NLIPSession, and
# start the server via its helper. The exact import paths below mirror their
# echo.py example; if an upstream refactor changes them, fix here in one place.

try:
    from nlip_server.server import NLIP_Application, NLIP_Session, setup_server
    from nlip_sdk.nlip import NLIP_Factory, NLIP_Message

    _NLIP_AVAILABLE = True
except ImportError as exc:
    logger.warning("NLIP libraries not importable (%s); server.py will expose "
                   "a plain FastAPI fallback instead. Run `poetry install` once "
                   "dependencies resolve.", exc)
    _NLIP_AVAILABLE = False


if _NLIP_AVAILABLE:

    class AngelFilterSession(NLIP_Session):
        """One session = one user's ongoing conversation with the proxy."""

        async def start(self) -> None:
            logger.info("AngelFilterSession started.")

        async def stop(self) -> None:
            logger.info("AngelFilterSession stopped.")

        async def execute(self, msg: NLIP_Message) -> NLIP_Message:
            # An NLIP_Message is multipart: a top-level content field plus a list
            # of typed submessages. We read three distinct inputs by type/label
            # rather than flattening everything into one string (which is what
            # extract_text() does — it would merge the query with the preference):
            #   query      = the unlabeled text parts (robust to a client that
            #                sends text as a submessage, not just top-level)
            #   preference = a text submessage labeled "preference"
            #   location   = a GPS submessage labeled "user_location"
            # This is how the demo carries lat/lng through the protocol instead
            # of a REST side-channel.
            user_query = _extract_query(msg)
            preference = _extract_preference(msg)
            user_lat, user_lng = _extract_location(msg)
            priority = _extract_priority(msg)
            logger.info(
                "NLIP query: %r | preference=%r | loc=(%s,%s) | priority=%s",
                user_query, preference, user_lat, user_lng,
                priority.value if priority else "auto",
            )
            if not user_query:
                return NLIP_Factory.create_text(
                    "No text query found in the NLIP message."
                )

            # --- Multi-turn -------------------------------------------------
            # The conversation token is NLIP's own; correlated_execute echoes it
            # back to the client, so a follow-up arrives carrying the same one.
            # Memory cannot live on `self` — nlip_server builds a new session
            # object per request — so it is keyed by that token in CONVERSATIONS.
            token = msg.extract_conversation_token()
            turn_mode = _extract_turn_mode(msg)
            conversation = await CONVERSATIONS.get(token)
            previous = conversation.latest if conversation else None

            # An explicit "new" ends the thread outright: the user told us this
            # is a fresh search, so prior turns must not colour it even if the
            # wording looks like a refinement.
            if turn_mode == "new":
                previous = None
                conversation = None

            search_query = user_query
            refinement: QueryConstraints | None = None
            notes: list[str] = []
            context_prefix = ""

            if previous is not None:
                # Reaching here means the client sent a token matching a live
                # conversation and did not say "new", so this turn belongs to
                # the thread. The flag's real work is the "new" case above,
                # which ends the thread outright — here we only need to know
                # whether there is a recognisable adjustment to apply.
                if looks_like_a_refinement(user_query):
                    # "cheaper than that" carries the adjustment but not the
                    # subject; the previous turn supplies what we are searching
                    # for, and the deltas supply the new bounds.
                    refinement, notes = apply_refinement(user_query, previous)
                    search_query = effective_query(user_query, previous)
                    logger.info(
                        "Refinement %r on %r -> %s",
                        user_query, previous.query, "; ".join(notes) or "no change",
                    )
                else:
                    # A follow-up with no recognisable delta ("what about
                    # vegetarian options"). Hand the models the recent history
                    # so it still resolves — the fallback path, paid for in
                    # tokens only when needed.
                    context_prefix = build_context_prefix(conversation)

            # Cache on the same (query, composed-preference) key the REST path
            # uses, so the two transports share entries instead of each paying
            # for a fan-out the other already did. Until this was added the NLIP
            # path — the active one — never touched the cache at all: /history
            # was always empty and repeating a query re-queried every provider.
            #
            # A refinement folds its adjusted bounds into the key: the same
            # subject under a tighter budget is a different search and must not
            # be served the previous turn's ranking.
            cache_pref = _cache_pref(
                preference, user_lat, user_lng,
                priority.value if priority else None,
            )
            if refinement is not None:
                cache_pref = f"{cache_pref}|r={refinement.budget},{refinement.max_distance},{refinement.min_rating}"
            # Likewise a follow-up answered with prior turns in the prompt: "what
            # about vegetarian options" means something different in every
            # conversation. Without this, one conversation's answer was served
            # to another that shared the same wording and location.
            if context_prefix:
                cache_pref = f"{cache_pref}|ctx={context_prefix}"

            payload = await CACHE.get(search_query, cache_pref)
            if payload is None:
                response = await _run_search(
                    user_query=search_query,
                    user_preference=preference,
                    user_lat=user_lat,
                    user_lng=user_lng,
                    intent=priority,
                    override_constraints=refinement,
                    context_prefix=context_prefix,
                )
                payload = _serialize_response(response)
                await CACHE.set(search_query, cache_pref, {**payload, "cached": True})
                summary = _format_reply(response)
            else:
                logger.info("Cache hit for NLIP query: %r", search_query)
                summary = _format_reply_from_payload(payload)

            # Say what changed, so a refined result set does not just silently
            # differ from the previous one.
            if notes:
                summary = f"Refined: {', '.join(notes)}.\n{summary}"

            # Remember this turn so the *next* follow-up has an anchor. The top
            # result's actual values are what "cheaper than that" refers to.
            top = (payload.get("results") or [None])[0]
            await CONVERSATIONS.record(token, Turn(
                query=search_query,
                constraints=refinement or _constraints_from_payload(payload),
                top_title=top.get("title") if top else None,
                top_price=(top.get("price") if top else None),
                top_distance=(top.get("distance") if top else None),
                top_rating=(top.get("rating") if top else None),
            ))

            # Multipart reply: a human-readable text summary AND a structured
            # JSON submessage carrying the full ranking (scores, axis breakdown,
            # sponsored flags). A text-only client reads extract_text(); an agent
            # reads the JSON via extract_field_list("structured", "JSON") and gets
            # machine-readable data — including the sponsored flag, which is the
            # project's thesis — instead of having to parse it out of prose.
            reply = NLIP_Factory.create_text(summary)
            reply.add_json(payload)
            return reply


    class AngelFilterApplication(NLIP_Application):
        """The NLIP application — spawns one session per client connection."""

        async def startup(self) -> None:
            logger.info("AngelFilterApplication starting up.")

        async def shutdown(self) -> None:
            logger.info("AngelFilterApplication shutting down.")

        def create_session(self) -> AngelFilterSession:
            return AngelFilterSession()


    app = setup_server(AngelFilterApplication())

    # nlip_server owns the /nlip/ route, so the rate limit and daily cap that
    # /query gets through a dependency are applied here as middleware instead.
    # Without this, the route the UI actually uses would be unmetered.
    app.middleware("http")(nlip_limit_middleware(_NLIP_PATH))

    # Mount the demo UI routes onto the NLIP app so /query and / work for the
    # frontend regardless of whether NLIP is the active transport.
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pathlib import Path
    from pydantic import BaseModel

    _STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    class QueryIn(BaseModel):
        query: str
        preference: str | None = None
        lat: float | None = None   # user origin latitude (for distance-aware providers)
        lng: float | None = None   # user origin longitude
        priority: str | None = None  # "price"|"distance"|"rating"; None = auto-detect

    @app.get("/")
    async def index():
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"msg": "Angel Filter — POST to /query or /nlip/"}

    # setup_server() mounts nlip_server's own health router, whose GET /health
    # returns a bare {"status": "healthy"}. FastAPI matches the first route
    # registered for a path, so simply declaring ours below would never be
    # reached — which is what happened: our richer payload was dead code in NLIP
    # mode and /health could not tell you which providers had loaded.
    #
    # Drop only the conflicting /health entry from *our* app's route table.
    # Upstream's /health/live and /health/ready stay, and the library itself is
    # untouched — we are consumers of nlip_server, not a fork of it.
    _drop_route(app, "/health")

    @app.get("/health")
    async def health():
        return await _health_response(mode="nlip", nlip_available=True)

    @app.post("/query")
    async def query(body: QueryIn, _client: str = Depends(enforce_query_limits)):
        return await _rest_search(body)

    # With no login, these are open to anyone who can reach the server:
    # /history shows every caller's recent queries and /cache/clear wipes the
    # shared cache. Deciding who may call them is part of the auth starter task
    # in docs/KNOWN_ISSUES.md.
    @app.get("/history")
    async def history():
        return {"queries": await CACHE.history(), "cache_stats": await CACHE.stats()}

    @app.post("/cache/clear")
    async def cache_clear():
        await CACHE.clear()
        return {"ok": True, "message": "Cache cleared."}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


else:
    # --- Fallback: plain FastAPI so the demo still runs without NLIP installed -
    # This path exists so Friday's demo is not held hostage by a dependency
    # install problem. It exposes a single POST /query endpoint that does the
    # same thing the NLIP session would do. Remove once NLIP is reliably
    # installable on every teammate's machine.
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from pathlib import Path
    from pydantic import BaseModel

    app = FastAPI(
        title="Angel Filter",
        description="A local proxy that re-ranks multi-provider AI results and penalizes sponsored content.",
        version="0.1.0",
    )

    _STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

    if _STATIC_DIR.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    class QueryIn(BaseModel):
        query: str
        preference: str | None = None
        lat: float | None = None   # user origin latitude (for distance-aware providers)
        lng: float | None = None   # user origin longitude
        priority: str | None = None  # "price"|"distance"|"rating"; None = auto-detect

    @app.get("/")
    async def index():
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"msg": "Angel Filter is running (fallback mode). POST to /query."}

    @app.get("/health")
    async def health():
        return await _health_response(mode="fallback", nlip_available=False)

    @app.post("/query")
    async def query(body: QueryIn, _client: str = Depends(enforce_query_limits)):
        return await _rest_search(body)

    @app.get("/history")
    async def history():
        return {"queries": await CACHE.history(), "cache_stats": await CACHE.stats()}

    @app.post("/cache/clear")
    async def cache_clear():
        await CACHE.clear()
        return {"ok": True, "message": "Cache cleared."}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- NLIP message extraction --------------------------------------------------
# These read the three demo inputs out of an NLIP_Message by type and label.
# They take the message object directly (duck-typed) so they need no NLIP import
# and stay unit-testable without booting the server.

_PREFERENCE_LABEL = "preference"
_LOCATION_LABEL = "user_location"
_PRIORITY_LABEL = "priority"
_TURN_MODE_LABEL = "turn_mode"


def _extract_query(msg) -> str:
    """The user's query: the unlabeled text parts, joined.

    Reading unlabeled text (rather than extract_text(), which would also pull in
    the labeled preference) keeps the query separate from the other fields while
    staying robust to a client that puts its text in a submessage instead of the
    top-level content.
    """
    parts: list[str] = []
    # Top-level content, if it's plain text with no label.
    if isinstance(msg.content, str) and getattr(msg, "label", None) is None:
        parts.append(msg.content)
    for sub in (msg.submessages or []):
        fmt = str(getattr(sub, "format", "")).lower()
        if fmt.endswith("text") and getattr(sub, "label", None) is None:
            if isinstance(sub.content, str):
                parts.append(sub.content)
    return " ".join(p.strip() for p in parts if p and p.strip()).strip()


def _labeled(msg, label):
    """find_labeled_submessage, guarded against a None submessages list.

    The SDK's find_labeled_submessage iterates msg.submessages directly, which
    raises when it's None (a message with no submessages — e.g. a plain text
    query). Guard so the simple-text client path never crashes here.
    """
    if not msg.submessages:
        return None
    return msg.find_labeled_submessage(label)


def _extract_preference(msg) -> str | None:
    """A text submessage labeled 'preference', or None."""
    sub = _labeled(msg, _PREFERENCE_LABEL)
    if sub is not None and isinstance(sub.content, str) and sub.content.strip():
        return sub.content.strip()
    return None


def _extract_turn_mode(msg) -> str | None:
    """Whether the client says this turn continues the conversation.

    Returns "continue", "new", or None when the client did not say. An explicit
    signal beats inferring from phrasing, which is wrong in both directions: a
    genuinely new search containing "cheaper" reads as a refinement, and a
    follow-up phrased unusually reads as new. None keeps the phrase-based
    heuristic, so an agent client that knows nothing about this label behaves
    exactly as before.
    """
    sub = _labeled(msg, _TURN_MODE_LABEL)
    if sub is None or not isinstance(sub.content, str):
        return None
    value = sub.content.strip().lower()
    return value if value in ("continue", "new") else None


def _extract_priority(msg) -> QueryIntent | None:
    """A text submessage labeled 'priority' naming one axis, or None.

    None means "no explicit choice" — the caller falls back to keyword
    detection. An unrecognised value is treated the same way rather than
    raising: a client sending garbage gets the old auto behaviour, not a 500.
    """
    sub = _labeled(msg, _PRIORITY_LABEL)
    if sub is None or not isinstance(sub.content, str):
        return None
    return _parse_priority(sub.content)


def _extract_location(msg) -> tuple[float | None, float | None]:
    """A GPS submessage labeled 'user_location' -> (lat, lng), or (None, None).

    The content is a JSON string like {"lat": .., "lng": ..}. Malformed or
    partial payloads degrade to (None, None) rather than raising — location is
    an optional signal, and a bad one must not fail the whole query.
    """
    sub = _labeled(msg, _LOCATION_LABEL)
    if sub is None or not isinstance(sub.content, str):
        return None, None
    try:
        import json
        data = json.loads(sub.content)
        lat, lng = data.get("lat"), data.get("lng")
        if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
            return float(lat), float(lng)
    except (ValueError, TypeError, AttributeError):
        logger.info("Ignoring malformed NLIP location payload: %r", sub.content)
    return None, None


# --- Helpers ------------------------------------------------------------------

def _serialize_response(response) -> dict:
    """Canonical structured form of an OrchestratorResponse.

    Single source of truth for the response shape shared by all three
    consumers: both /query handlers (JSON over HTTP) and the NLIP session's
    structured submessage. Keeping it in one place stops the per-result fields
    — score, axis_scores, sponsored, consensus_count — from drifting between
    the paths, which they had already started to (two near-identical inline
    copies existed before this).
    """
    return {
        "providers_used": response.providers_used,
        "providers_failed": response.providers_failed,
        "intent": response.intent.value,
        "constraints": {
            "budget": response.constraints.budget,
            "max_distance": response.constraints.max_distance,
            "min_rating": response.constraints.min_rating,
        },
        "results": [
            {
                "title": r.result.title,
                "snippet": r.result.snippet,
                "url": r.result.url,
                "provider": r.result.provider,
                "score": round(r.score, 3),
                "rationale": r.rationale,
                "sponsored": r.result.sponsored,
                "sponsored_reason": r.result.sponsored_reason,
                "consensus_count": r.consensus_count,
                # The raw values behind the axis scores. A follow-up like
                # "cheaper than that" has to anchor on what the top result
                # actually cost, not on its normalised 0-1 P1 score.
                "price": r.result.price,
                "distance": r.result.distance,
                "rating": r.result.rating,
                "axis_scores": r.axis_scores,
                "score_breakdown": r.score_breakdown,
                # Which axes the provider actually disclosed. axis_scores holds
                # a 0.5 placeholder for the rest, so a consumer that ignores
                # this cannot distinguish "mediocre" from "unknown".
                "axis_scored": r.axis_scored,
            }
            for r in response.ranked
        ],
    }


def _parse_priority(value: str | None) -> QueryIntent | None:
    """Map a priority string to a QueryIntent, or None to auto-detect.

    Shared by the REST handlers and (via _extract_priority) the NLIP session so
    both paths treat the same input identically. Unrecognised values degrade to
    None rather than raising: a client that sends garbage loses the override,
    not the query.
    """
    if not value:
        return None
    try:
        return QueryIntent(value.strip().lower())
    except ValueError:
        logger.warning("Unrecognised priority %r — falling back to detection", value)
        return None


def _cache_pref(
    preference: str | None,
    lat: float | None,
    lng: float | None,
    priority: str | None = None,
) -> str:
    """Compose the cache's preference key so location is part of the identity.

    The cache keys on (query, preference); results now depend on the user's
    coordinates too, so we fold them in. Coordinates are rounded to ~3 decimal
    places (~110m) so trivially different GPS readings still hit the cache.

    Priority is folded in for the same reason: the same query ranked by price
    and by rating produces different orderings, so they must not share a cache
    entry. Normalised through _parse_priority so "PRICE" and "price" — and every
    unrecognised value, which all mean auto — collapse to one key.
    """
    base = preference or ""
    if lat is not None and lng is not None:
        base = f"{base}|@{round(lat, 3)},{round(lng, 3)}"
    intent = _parse_priority(priority)
    if intent is not None:
        base = f"{base}|p={intent.value}"
    return base


def _format_reply_from_payload(payload: dict) -> str:
    """Human-readable summary built from the serialised response.

    Works off the payload rather than an OrchestratorResponse so a cache hit —
    which only has the stored dict — produces byte-identical text to a fresh
    query. Duplicating this formatting for the cached case is how the two would
    drift.
    """
    used = payload.get("providers_used", [])
    failed = payload.get("providers_failed", [])
    results = payload.get("results", [])
    if not results:
        return "No results from any provider. Providers tried: " + ", ".join(used + failed)
    lines = [f"Ranked {len(results)} results (providers used: {', '.join(used)}):"]
    for i, r in enumerate(results, start=1):
        tag = ""
        if r.get("sponsored"):
            reason = r.get("sponsored_reason")
            tag = f" [SPONSORED: {reason}]" if reason else " [SPONSORED]"
        lines.append(f"{i}. {r.get('title')}{tag} — {r.get('provider')} — {r.get('rationale')}")
        if r.get("url"):
            lines.append(f"   {r['url']}")
    if failed:
        lines.append("(failed: " + ", ".join(failed) + ")")
    return "\n".join(lines)


def _format_reply(response) -> str:
    """Summary for a fresh OrchestratorResponse — one hop to the shared formatter."""
    return _format_reply_from_payload(_serialize_response(response))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8005"))
    uvicorn.run(app, host="0.0.0.0", port=port)
