"""Page feeds: what happened, in order, resumable from where you stopped.

Extracted from :mod:`mc_llm_jobs`, which has run MiniMax's external requests
this way since it was written: a monotonic cursor, a bounded ring per
subscriber, a TTL so a browser that walked away cannot pin a log for ever, and
reads that take everything past a cursor rather than everything. Nothing here
is a new idea; what is new is that conversations use it too.

Why a cursor and not a callback
-------------------------------
A browser's connection drops. It always eventually drops -- a laptop lid, a
phone's radio, a proxy's idle timeout -- and what happens next decides whether
the window is trustworthy. With a callback the answer is "some events were
missed and nobody knows which"; with a cursor it is "resume after 4,117", and
the only case left is the one where 4,117 has already been trimmed out of the
ring, which is detectable and answered with a fresh snapshot rather than with a
guess.

Monotonic across one epoch, and the epoch changes when the process restarts.
A cursor from the old process is therefore never mistaken for a position in the
new one's stream -- it is simply from an epoch that no longer exists, and the
page is told to reload.

What is *not* here
------------------
No polling loop, and nothing that reads a conversation. This module carries
records of things other modules did. A feed that read the store to work out
what to say would be a second source of truth about a conversation, and the
whole architecture has exactly one.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

MAX_EVENTS = 2000
"""Events kept per page feed before the oldest are dropped.

Two thousand is several minutes of a streaming reply at the coalescing rate
below, and a subscriber that falls that far behind has a connection problem
rather than a backlog problem -- it is told its cursor was trimmed and asks for
a snapshot, which is cheaper than either of us pretending otherwise.
"""

MAX_AGE = 600.0
"""How long an event stays in a ring. Ten minutes.

The other half of the bound, and the one that matters for an idle page: a
window left open overnight must not be holding yesterday's tokens.
"""

FEED_TTL = 1800.0
"""How long a feed nobody reads stays alive. Thirty minutes.

Long enough to survive a laptop asleep in a bag for the length of a meeting;
short enough that a browser that will never come back is forgotten the same
afternoon.
"""

PATCH_INTERVAL = 0.05
"""The floor between two ``reply_patch`` events for one operation. 20/s.

Tokens arrive faster than a person can read and far faster than a DOM wants to
be rewritten. Coalescing is done at the publisher rather than the subscriber
because a slow subscriber is not the only reader -- the ring would otherwise
fill with a thousand one-word events, and the terminal text would push the
start of the reply out of it.

The terminal event is never coalesced: whatever else is dropped, the finished
reply is published the moment it exists.
"""


@dataclass
class Event:
    """One thing that happened, addressed to whoever is listening."""

    kind: str
    stream_cursor: int = 0
    character: str = ""
    thread_id: str = ""
    conversation_revision: object = None
    operation_id: str = ""
    operation_seq: int | None = None
    payload: dict = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def describe(self) -> dict:
        import mc_llm_conversation_service as service

        return {"protocol_version": service.PROTOCOL_VERSION,
                "server_epoch": service.SERVER_EPOCH,
                "stream_cursor": self.stream_cursor,
                "kind": self.kind,
                "conversation": {"character": self.character, "thread_id": self.thread_id}
                if self.thread_id or self.character else None,
                "conversation_revision": self.conversation_revision,
                "operation_id": self.operation_id or None,
                "operation_seq": self.operation_seq,
                "payload": dict(self.payload)}


_guard = threading.Condition()
_sequence = 0
_feeds: dict[str, "Feed"] = {}
_last_patch: dict[str, float] = {}


class Feed:
    """One page's window onto the stream. Created by :func:`subscribe`.

    ``generation`` is what makes a resubscribe safe. A page that reconnects
    gets a new generation, and anything still in flight from the old one is
    discarded by the client rather than merged into a view that has already
    moved on -- which is the difference between a reconnect and two readers
    fighting over one transcript.
    """

    def __init__(self, page_id: str = "", ttl: float = FEED_TTL):
        self.identifier = uuid.uuid4().hex[:12]
        self.page_id = str(page_id or "")
        self.generation = uuid.uuid4().hex[:8]
        self.ttl = float(ttl)
        self.created = time.time()
        self.touched = time.time()
        self.closed = False
        self.delivered = 0
        self.cursor = 0
        self._events: list[Event] = []

    # -- lifecycle -------------------------------------------------------- #

    @property
    def expired(self) -> bool:
        return time.time() - self.touched > self.ttl

    def close(self) -> None:
        with _guard:
            self.closed = True
            _feeds.pop(self.identifier, None)
            _guard.notify_all()

    def describe(self) -> dict:
        return {"feed": self.identifier, "page": self.page_id or None,
                "subscription_generation": self.generation, "cursor": self.cursor,
                "created": self.created, "idle": round(time.time() - self.touched, 3),
                "ttl": self.ttl, "closed": self.closed, "delivered": self.delivered,
                "pending": len(self._events)}

    # -- writing ---------------------------------------------------------- #

    def _accept_locked(self, event: Event) -> None:
        self._events.append(event)
        self._trim_locked()

    def _trim_locked(self) -> None:
        if len(self._events) > MAX_EVENTS:
            del self._events[:len(self._events) - MAX_EVENTS]
        floor = time.time() - MAX_AGE
        while self._events and self._events[0].at < floor:
            self._events.pop(0)

    # -- reading ---------------------------------------------------------- #

    def since(self, cursor: int) -> tuple[list, bool]:
        """``(events, gap)`` for everything past ``cursor``.

        ``gap`` says the cursor is older than anything still held, so the
        caller has provably missed something. It is not an error and is never
        papered over: the page asks for a snapshot and starts again from a
        position both sides agree on.
        """
        with _guard:
            self.touched = time.time()
            self._trim_locked()
            kept = [event for event in self._events if event.stream_cursor > cursor]
            gap = bool(self._events) and cursor < (self._events[0].stream_cursor - 1)
            if kept:
                self.cursor = kept[-1].stream_cursor
                self.delivered += len(kept)
            return [event.describe() for event in kept], gap

    def poll(self) -> list:
        """Everything not yet delivered to this feed, right now."""
        events, _ = self.since(self.cursor)
        return events

    def events(self, timeout: float | None = None, idle: float = 1.0):
        """Yield events as they happen, ending when the feed does.

        Nothing is yielded while the lock is held. A consumer writing each event
        to a socket would otherwise hold the publishing lock for the length of a
        network write, and one slow browser would stall the thread streaming a
        reply into every other one.
        """
        started = time.monotonic()
        while True:
            with _guard:
                self.touched = time.time()
                closed = self.closed
            for event in self.poll():
                yield event
            if closed:
                return
            if timeout is not None and time.monotonic() - started >= timeout:
                return
            with _guard:
                if self.closed:
                    return
                _guard.wait(timeout=idle)


# --------------------------------------------------------------------------- #
# Publishing and subscribing
# --------------------------------------------------------------------------- #


def cursor() -> int:
    """The stream position right now. What a snapshot is taken *at*."""
    with _guard:
        return _sequence


def publish(kind: str, key=None, **payload) -> int:
    """Record one event for every open page. Returns its cursor.

    Coalescing happens here and only for ``reply_patch``, whose payload is the
    cumulative text rather than a delta -- so dropping one is genuinely
    lossless, which is exactly why it carries the whole text instead of the
    difference.
    """
    global _sequence

    import mc_llm_conversation_service as service

    operation_id = str(payload.pop("operation_id", "") or "")
    if kind == service.REPLY_PATCH and not payload.pop("force", False):
        now = time.monotonic()
        with _guard:
            last = _last_patch.get(operation_id, 0.0)
            if now - last < PATCH_INTERVAL:
                return _sequence
            _last_patch[operation_id] = now
    else:
        payload.pop("force", None)

    revision = payload.pop("revision", None)
    operation_seq = payload.pop("operation_seq", None)

    with _guard:
        _sequence += 1
        event = Event(kind=kind, stream_cursor=_sequence,
                      character=str(getattr(key, "character", "") or ""),
                      thread_id=str(getattr(key, "thread_id", "") or ""),
                      conversation_revision=revision, operation_id=operation_id,
                      operation_seq=operation_seq, payload=dict(payload))
        for feed in list(_feeds.values()):
            if feed.closed:
                continue
            feed._accept_locked(event)
        _guard.notify_all()
        return _sequence


def subscribe(page_id: str = "", ttl: float = FEED_TTL) -> Feed:
    """Open a feed. The snapshot taken alongside it must use :func:`cursor`.

    Registration happens under the publishing lock, which is the race the
    specification's 8.4 names: an event that lands between "read the
    conversation" and "start listening" would otherwise be delivered to nobody,
    and the page would show a transcript one message behind until something
    else happened to arrive.
    """
    feed = Feed(page_id=page_id, ttl=ttl)
    with _guard:
        _expire_locked()
        feed.cursor = _sequence
        _feeds[feed.identifier] = feed
    return feed


def find(identifier: str) -> Feed | None:
    with _guard:
        found = _feeds.get(str(identifier or ""))
        return None if found is None or found.closed else found


def feeds() -> list:
    with _guard:
        return [feed.describe() for feed in _feeds.values()]


def _expire_locked() -> None:
    for identifier in [key for key, feed in _feeds.items() if feed.expired]:
        found = _feeds.pop(identifier, None)
        if found is not None:
            found.closed = True


def expire() -> int:
    """Close feeds nobody has read for :data:`FEED_TTL`. Returns how many."""
    with _guard:
        before = len(_feeds)
        _expire_locked()
        return before - len(_feeds)


def forget_operation(operation_id: str) -> None:
    with _guard:
        _last_patch.pop(str(operation_id or ""), None)


def reset() -> None:
    """Drop every feed and the sequence. For the tests, and for a UI reload."""
    global _sequence

    with _guard:
        for feed in list(_feeds.values()):
            feed.closed = True
        _feeds.clear()
        _last_patch.clear()
        _sequence = 0
        _guard.notify_all()
