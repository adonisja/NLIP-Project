"""Multi-turn memory keyed by the NLIP conversation token.

This is where NLIP's session model earns its place. The protocol already
carries a conversation token — `NLIP_Session.correlated_execute` mints one,
echoes the client's back, and the SDK moves it as a `format: "token"`
submessage — but nothing was stored against it, so every turn started cold.
"cheaper than that" had no "that" to refer to.

A new session object is constructed per request (see nlip_server's route: it
calls `create_session()` on every call), so memory cannot live on the session
instance. It lives here, keyed by token, and is pruned by age.

Two ways a follow-up gets resolved, in order:

  1. Deterministic deltas. "cheaper", "closer", "better rated" are adjustments
     to the previous turn's constraints, anchored on what the previous winner
     actually was. No model call, inspectable, and testable offline.
  2. Model fallback. Anything the delta parser does not recognise is handed to
     the providers with the previous turn's context prepended, so vaguer
     phrasings still work. Costs tokens, so it is the fallback rather than the
     default.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from angel_filter.constraints import QueryConstraints

# How long a conversation stays resolvable. Long enough for a demo or a genuine
# back-and-forth, short enough that abandoned tokens do not accumulate.
TURN_TTL_SECONDS = 30 * 60

# Keep the last few turns per conversation. Only the most recent is needed to
# resolve a delta, but a short history is what the model fallback sends.
MAX_TURNS = 6

# How hard an unquantified refinement bites. "cheaper" with no number means
# "meaningfully less than what you just showed me", not "a penny less".
_REFINE_FACTOR = 0.8

# Distance tightens harder: halving is what "closer" usually means to someone
# looking at a map, and distances are small enough that 0.8 barely moves.
_DISTANCE_FACTOR = 0.5

_CHEAPER = re.compile(
    r"\b(cheaper|less expensive|lower price|more affordable|budget|"
    r"under that|below that|too expensive|too pricey|cheaper than)\b", re.I
)
_CLOSER = re.compile(
    r"\b(closer|nearer|nearby|too far|shorter walk|walking distance|"
    r"closer than)\b", re.I
)
_BETTER = re.compile(
    r"\b(better rated|higher rated|better reviews|better reviewed|"
    r"more highly rated|top rated|better than that)\b", re.I
)

# A follow-up is only a refinement if it is *short*. "cheaper pizza in Brooklyn
# with outdoor seating" is a new search that happens to contain the word
# cheaper, not an adjustment to the previous one.
_MAX_REFINEMENT_WORDS = 6


@dataclass
class Turn:
    """One completed exchange, enough to anchor the next refinement."""

    query: str
    constraints: QueryConstraints
    top_title: str | None = None
    top_price: float | None = None
    top_distance: float | None = None
    top_rating: float | None = None
    at: float = field(default_factory=time.time)


@dataclass
class Conversation:
    turns: list[Turn] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        del self.turns[:-MAX_TURNS]
        self.last_seen = turn.at

    @property
    def latest(self) -> Turn | None:
        return self.turns[-1] if self.turns else None


class ConversationStore:
    """In-process conversation memory, keyed by NLIP conversation token.

    In-process deliberately: the query cache already works this way, a single
    Render instance serves the demo, and a durable store would be the wrong
    thing to add before the protocol shape has been reviewed. A restart loses
    conversations, which degrades to every turn being a fresh search.
    """

    def __init__(self, ttl_seconds: int = TURN_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._store: dict[str, Conversation] = {}

    def get(self, token: str | None) -> Conversation | None:
        if not token:
            return None
        self._prune()
        return self._store.get(token)

    def record(self, token: str | None, turn: Turn) -> None:
        if not token:
            return
        self._prune()
        self._store.setdefault(token, Conversation()).add(turn)

    def _prune(self) -> None:
        cutoff = time.time() - self._ttl
        stale = [k for k, c in self._store.items() if c.last_seen < cutoff]
        for k in stale:
            del self._store[k]

    def stats(self) -> dict[str, int]:
        self._prune()
        return {
            "conversations": len(self._store),
            "turns": sum(len(c.turns) for c in self._store.values()),
        }


CONVERSATIONS = ConversationStore()


def looks_like_a_refinement(query: str) -> bool:
    """Is this a follow-up adjusting the last search, or a new one?

    Length is the discriminator. A refinement is short by nature ("cheaper",
    "something closer"); once the user writes a full sentence they have
    described a new search, even if it contains a comparative word.
    """
    if not query or not query.strip():
        return False
    if len(query.split()) > _MAX_REFINEMENT_WORDS:
        return False
    return bool(_CHEAPER.search(query) or _CLOSER.search(query) or _BETTER.search(query))


def apply_refinement(
    query: str,
    previous: Turn,
) -> tuple[QueryConstraints, list[str]]:
    """Build this turn's constraints by adjusting the previous turn's.

    Returns the new constraints and a human-readable list of what changed, so
    the reply can say *why* the results moved rather than silently shifting.

    Anchoring is on what the previous turn actually returned, not on the
    previous constraint. Asking for "cheaper" after being shown an $18 result
    means cheaper than $18 — anchoring on a $50 budget nobody hit would change
    nothing.
    """
    c = QueryConstraints(
        budget=previous.constraints.budget,
        max_distance=previous.constraints.max_distance,
        min_rating=previous.constraints.min_rating,
    )
    notes: list[str] = []

    if _CHEAPER.search(query):
        anchor = previous.top_price if previous.top_price is not None else c.budget
        if anchor is not None:
            c.budget = round(anchor * _REFINE_FACTOR, 2)
            notes.append(f"budget tightened to ${c.budget:.2f}")

    if _CLOSER.search(query):
        anchor = previous.top_distance if previous.top_distance is not None else c.max_distance
        if anchor is not None:
            c.max_distance = round(max(anchor * _DISTANCE_FACTOR, 0.1), 2)
            notes.append(f"distance tightened to {c.max_distance} mi")

    if _BETTER.search(query):
        anchor = previous.top_rating if previous.top_rating is not None else c.min_rating
        if anchor is not None:
            # Ratings cap at 5, and asking for "better" than 4.9 should not
            # produce an unsatisfiable 5.4.
            c.min_rating = round(min(anchor + 0.2, 5.0), 2)
            notes.append(f"minimum rating raised to {c.min_rating}★")

    return c, notes


def effective_query(query: str, previous: Turn) -> str:
    """The search text to actually run for a refinement.

    "cheaper" alone is not a searchable query — it carries the adjustment but
    none of the subject. The previous turn's query supplies the subject; the
    numeric part of the refinement is already in the constraints.
    """
    return previous.query


def build_context_prefix(conversation: Conversation, limit: int = 3) -> str:
    """Prior turns as prompt context, for follow-ups the delta parser missed.

    Only used on the fallback path — a refinement the patterns recognise never
    reaches the models as history, so the common case costs no extra tokens.
    """
    recent = conversation.turns[-limit:]
    if not recent:
        return ""
    lines = ["Earlier in this conversation:"]
    for t in recent:
        if t.top_title:
            lines.append(f'- Asked "{t.query}" — top result was {t.top_title}')
        else:
            lines.append(f'- Asked "{t.query}"')
    return "\n".join(lines)
