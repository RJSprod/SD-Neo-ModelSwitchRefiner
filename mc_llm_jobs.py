"""The external request queue: one FIFO, one worker, one record per request.

Another extension in this WebUI wants MiniMax H3 prompts written for it. What
it must *not* be able to do is take the card out from under a job that is
already running, and what it must be able to do is find out what happened
afterwards. Those two sentences are the whole module.

What this is not
----------------
It is not a second scheduler. Nothing here decides who gets the GPU --
:class:`mc_llm_sessions._Gpu` still does, exactly as it does for the panel, and
an external request waits for an image generation on the same card for the same
reasons and with the same messages. This queue sits *above* that: it decides
which external request is offered to the existing machinery next, and it
guarantees only ever offering one at a time.

The distinction matters because the alternative was tempting and wrong. A
request that arrived while a panel run was streaming could have called
``mc_llm_sessions.minimax`` on its own thread and let the workload lock sort it
out -- that is, after all, what the lock is for. But the lock is a race, not a
line: three requests arriving during one long Ref2VA run would have been three
threads spinning in :meth:`_Gpu.acquire`, finishing in whatever order the GIL
happened to wake them, with no id, no position, and nothing to cancel. The
queue is what turns that into something a caller can be told about.

One worker, and why not more
----------------------------
Two external requests never run at once, even on a machine with two cards.
llama-server is one process per role and the prompt cache is per process, so a
second concurrent enhancement would not be twice as fast -- it would be two
requests taking turns inside the server instead of taking turns outside it,
with the difference that this module could no longer say which one was running.
Serialising here is not a limitation being worked around; it is the property
the whole traceability story rests on.

Cancellation, and what "in progress is not interrupted" means
-------------------------------------------------------------
A caller may cancel a request that is *theirs*, whether it is queued or
running. Cancelling a queued one drops it from the line and never touches the
card. Cancelling a running one sets the same :class:`mc_llm_sessions
.Cancellation` the panel's Stop button sets, which llama.cpp honours between
tokens and which leaves residency exactly as it found it.

What no external request can do is displace another. Arriving does not preempt,
a higher position is not purchasable, and nothing here cancels a job on the
caller's behalf to make room. A queue that could be jumped would be a queue
that could interrupt work in progress, which is the one thing this was asked
not to do.

Feeds
-----
A caller may subscribe to a request and read events as they happen. The events
are deliberately SSE-shaped -- ``seq``, ``event``, ``data`` -- because the
caller is an extension with a browser panel of its own, and the cheapest thing
it can do with this feed is forward it to that panel as ``text/event-stream``
without translating anything. ``seq`` is a cursor, so a caller whose connection
dropped resumes from where it stopped rather than from the beginning or from
nothing.

Feeds have a lifecycle because the alternative is a memory leak with good
manners: a caller that opens a feed and goes away must not pin a job's event
log forever. A feed expires when nobody has read it for :data:`FEED_TTL`
seconds, and a feed bound to one request closes itself once it has delivered
that request's terminal event.

Nothing here is written to disk. Job records live in this process's RAM and are
forgotten on the schedule in :func:`_reap`; the finished prompt is saved to the
MiniMax history only because that is what the panel does with one, and only
when the caller asked for it.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

# -- states ----------------------------------------------------------------- #

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

TERMINAL = (DONE, FAILED, CANCELLED)

# -- event names ------------------------------------------------------------ #
#
# One name per thing a caller can act on, and no name for anything else. These
# are the strings that reach the other extension's code, so they are a contract:
# adding to this tuple is a compatible change and renaming anything in it is not.

EV_QUEUED = "queued"
EV_POSITION = "position"
EV_STARTED = "started"
EV_STATUS = "status"
EV_CAPTION = "caption"
EV_CHUNK = "chunk"
EV_DONE = "done"
EV_FAILED = "failed"
EV_CANCELLED = "cancelled"

EVENTS = (EV_QUEUED, EV_POSITION, EV_STARTED, EV_STATUS, EV_CAPTION, EV_CHUNK,
          EV_DONE, EV_FAILED, EV_CANCELLED)

# -- limits ----------------------------------------------------------------- #

MAX_QUEUED = 32
"""How many requests may be waiting before submission is refused.

A cap rather than an unbounded deque, because the failure it prevents is a
caller in a retry loop filling this process's RAM with requests nobody will
read the answers to. Thirty-two is far more than a human queue and far less
than a runaway one.
"""

MAX_EVENTS = 600
"""Events kept per job, oldest ``chunk`` first when trimming.

A Ref2VA prompt is two thousand tokens and llama.cpp hands them over a few
characters at a time, so an untrimmed log would be several thousand entries of
which all but a handful are one word each. Only ``chunk`` events are ever
dropped -- every status, caption and terminal event survives, and the terminal
``done`` carries the complete prompt -- so a subscriber that fell behind loses
some of the typewriter effect and none of the answer.
"""

JOB_RETENTION = 900.0
"""How long a finished job stays readable. Fifteen minutes.

Long enough that a caller which crashed and restarted can still collect a
result it never saw; short enough that a busy day does not accumulate a
transcript of itself.
"""

MAX_REMEMBERED = 200
"""How many finished jobs to keep regardless of age."""

FEED_TTL = 300.0
"""How long a feed survives with nobody reading it. Five minutes.

Read-idle rather than absolute: a feed being actively consumed is never
expired, however long its job takes, and a feed nobody is listening to stops
holding anything open five minutes later.
"""

WAKE_SECONDS = 0.5
"""How often the worker looks around while it has nothing to do."""


class Rejected(Exception):
    """A request that was never queued, with the reason a caller can act on."""

    def __init__(self, reason: str, code: str = "rejected"):
        super().__init__(reason)
        self.reason, self.code = reason, code


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #


@dataclass
class Job:
    """One external request, from arrival to forgotten.

    **No image bytes live here.** The data URL a caller supplied is held in
    :attr:`_image` until the run is over and then dropped, and it is never part
    of :meth:`describe`. The same rule the Krea history follows and for the same
    reason: a record somebody may read, log or hand to a UI should not contain a
    base64 copy of somebody's photograph. What is kept is which slots were
    filled and which one was used, because that is the part a caller needs to
    understand the answer it got.
    """

    identifier: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    kind: str = "minimax"
    origin: str = ""
    """Who asked. A free-text label the caller supplies, for the console and the
    banner the panel shows -- never authorisation, and never trusted for
    anything."""

    state: str = QUEUED
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0

    variant: str = ""
    prompt: str = ""
    seed: int = 0
    system: str | None = None
    """The caller's replacement for this variant's instructions, or ``None``.

    Held rather than merged in advance, because which of the four vendored
    system prompts a request would otherwise have used depends on whether the
    image survived preparation -- a question this record answers before the run
    and the enhancer answers during it."""
    images: tuple = ()
    """Which slots the caller filled, in the caller's own words."""
    image_used: str = ""
    """Which one was captioned. Empty when the request had no image."""
    image_ignored: tuple = ()
    remember: bool = True

    announced_position: int = 0
    """The position this job was last *told* it was at.

    Kept so that :func:`_announce_positions_locked` can be called from every
    place the line can change -- a job starting, a job finishing, a job being
    cancelled out of the middle -- without a subscriber receiving three
    identical "you are second" events for one shift it already knows about."""

    result: str = ""
    caption: str = ""
    error: str = ""
    cancel_reason: str = ""
    cancelling: bool = False

    _image: str | None = None
    _cancel: object = None
    _events: list = field(default_factory=list)
    _seq: int = 0
    _dropped: int = 0

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    @property
    def system_override(self) -> bool:
        return self.system is not None

    def describe(self, *, prompt: bool = True) -> dict:
        """The job as a caller sees it. Plain data, safe to serialise.

        ``prompt=False`` leaves the written prompt out, which is what the queue
        listing wants: a caller polling for position should not be handed
        thirty kilobytes of other people's prompts on every poll.
        """
        found = {
            "id": self.identifier,
            "kind": self.kind,
            "state": self.state,
            "origin": self.origin,
            "variant": self.variant,
            "seed": self.seed,
            "created": self.created,
            "started": self.started or None,
            "finished": self.finished or None,
            "elapsed": self.elapsed,
            "queued_for": self.queued_for,
            "position": position_of(self.identifier),
            "cancelling": self.cancelling,
            "system_override": self.system_override,
            "images": list(self.images),
            "image_used": self.image_used,
            "image_ignored": list(self.image_ignored),
            "events": self._seq,
            "dropped_events": self._dropped,
        }
        if self.state == FAILED:
            found["error"] = self.error
        if self.state == CANCELLED:
            found["reason"] = self.cancel_reason
        if prompt:
            found["prompt"] = self.result
            found["caption"] = self.caption
            found["request"] = self.prompt
        return found

    @property
    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        return round((self.finished or time.time()) - self.started, 3)

    @property
    def queued_for(self) -> float:
        return round((self.started or time.time()) - self.created, 3)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

_lock = threading.RLock()
_wake = threading.Condition(_lock)
"""Signalled whenever the worker has something to do or a feed has something to
read. One condition for both, because both are woken by the same events and two
would only have made it possible to forget one."""

_jobs: dict = {}
_pending: collections.deque = collections.deque()
_running: str = ""
_feeds: dict = {}
_stream: int = 0
"""A process-wide event counter, beside each event's per-job one.

Two cursors because there are two things worth subscribing to and they need
different orderings. A feed on one request wants that request's events in that
request's order, which is what ``seq`` gives it and what survives the record
being read on its own. A feed on the whole queue wants every job's events
interleaved in the order they actually happened, which no per-job counter can
express -- job A's ``seq`` 4 and job B's ``seq`` 4 are not the same moment.
"""

_worker: threading.Thread | None = None
_paused = False
"""Set only by the tests, which drive :func:`drain_once` themselves rather than
racing a background thread."""


# --------------------------------------------------------------------------- #
# Submitting
# --------------------------------------------------------------------------- #


def submit(job: Job) -> Job:
    """Put one request at the back of the line. Never preempts anything.

    Returns the same job, now carrying its position. Raises :class:`Rejected`
    when the queue is full, which is a refusal a caller can retry rather than an
    error it has to interpret.
    """
    with _wake:
        waiting = len(_pending)
        if waiting >= MAX_QUEUED:
            raise Rejected(
                f"The MiniMax queue is full ({waiting} requests waiting). Try again once "
                f"some of them have finished.", "queue_full")
        _reap_locked()
        _jobs[job.identifier] = job
        _pending.append(job.identifier)
        _emit_locked(job, EV_QUEUED, {"position": len(_pending), "waiting": len(_pending),
                                      "running": bool(_running)})
        _wake.notify_all()
    logger.info("Model Chain: MiniMax request %s queued at position %d%s",
                job.identifier, position_of(job.identifier),
                f" for {job.origin}" if job.origin else "")
    _ensure_worker()
    return job


def _ensure_worker() -> None:
    """Start the drain thread the first time anything is submitted.

    Lazily, so an installation that never uses the external API never grows a
    thread for it, and idempotently, so a UI reload does not grow a second one.
    """
    global _worker
    with _lock:
        if _paused:
            return
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(target=_drain, name="mc-llm-jobs", daemon=True)
        _worker.start()


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def job(identifier: str) -> Job | None:
    with _lock:
        return _jobs.get(str(identifier or ""))


def position_of(identifier: str) -> int:
    """1 for the next request to run, 0 for one that is running or finished."""
    with _lock:
        try:
            return _pending.index(str(identifier or "")) + 1
        except ValueError:
            return 0


def running() -> Job | None:
    with _lock:
        return _jobs.get(_running) if _running else None


def active() -> bool:
    """Whether anything external is running or waiting. The panel's gate."""
    with _lock:
        return bool(_running) or bool(_pending)


def snapshot(*, limit: int = 20) -> dict:
    """Everything a queue view needs, in one consistent read.

    One function rather than three, because a banner assembled from separate
    calls to :func:`running`, :func:`position_of` and a listing can describe a
    state that never existed -- a job counted as running by the first call and
    as finished by the third.
    """
    with _lock:
        _reap_locked()
        current = _jobs.get(_running) if _running else None
        waiting = [_jobs[key] for key in _pending if key in _jobs]
        recent = sorted((found for found in _jobs.values() if found.terminal),
                        key=lambda found: found.finished, reverse=True)[:limit]
        return {
            "active": bool(current) or bool(waiting),
            "running": current.describe(prompt=False) if current else None,
            "waiting": len(waiting),
            "queue": [found.describe(prompt=False) for found in waiting],
            "recent": [found.describe(prompt=False) for found in recent],
            "capacity": MAX_QUEUED,
        }


# --------------------------------------------------------------------------- #
# Cancelling
# --------------------------------------------------------------------------- #


def cancel(identifier: str, reason: str = "") -> dict:
    """Stop one request, wherever it is in its life.

    Queued is the easy half: it leaves the line and never touches the card.
    Running sets the same cancellation the panel's Stop sets, and returns before
    the run has actually stopped -- llama.cpp honours it between tokens, so
    "cancelling" is a truthful answer and "cancelled" would not be. The terminal
    event still arrives on the feed, which is where a caller that needs to know
    it really stopped should be looking.
    """
    key = str(identifier or "")
    with _wake:
        found = _jobs.get(key)
        if found is None:
            return {"ok": False, "error": "There is no request with that id.",
                    "code": "unknown_job"}
        if found.terminal:
            return {"ok": False, "state": found.state, "code": "already_finished",
                    "error": f"That request has already {found.state}."}
        if key in _pending:
            _pending.remove(key)
            _finish_locked(found, CANCELLED, reason=reason or "cancelled before it started")
            # Everything behind it has just moved up, and a caller watching its
            # own position should hear about it from the queue rather than by
            # noticing the answer arrived early.
            _announce_positions_locked()
            _wake.notify_all()
            logger.info("Model Chain: MiniMax request %s cancelled while queued", key)
            return {"ok": True, "state": CANCELLED, "was": QUEUED}
        found.cancelling = True
        found.cancel_reason = reason or "cancelled"
        _wake.notify_all()
    if found._cancel is not None:
        found._cancel.cancel()
    logger.info("Model Chain: MiniMax request %s is being cancelled while running", key)
    return {"ok": True, "state": "cancelling", "was": RUNNING}


def cancel_all(reason: str = "") -> dict:
    """Cancel the running request and everything waiting behind it.

    What the panel's banner offers, and the only bulk operation there is. It
    takes the queue as it stands at one instant rather than looping until empty,
    so a caller submitting while this runs is not starved by it.
    """
    with _lock:
        keys = list(_pending) + ([_running] if _running else [])
    stopped = [key for key in keys if cancel(key, reason).get("ok")]
    return {"ok": True, "cancelled": len(stopped), "ids": stopped}


def forget(identifier: str) -> bool:
    """Drop a finished job's record early, at the caller's request."""
    with _wake:
        found = _jobs.get(str(identifier or ""))
        if found is None or not found.terminal:
            return False
        _jobs.pop(found.identifier, None)
        _close_feeds_for_locked(found.identifier)
        return True


# --------------------------------------------------------------------------- #
# Events and feeds
# --------------------------------------------------------------------------- #


def _emit_locked(found: Job, name: str, data: dict | None = None) -> dict:
    """Append one event to a job's log. Caller holds ``_wake``."""
    global _stream

    found._seq += 1
    _stream += 1
    event = {"seq": found._seq, "stream": _stream, "id": found.identifier,
             "event": name, "time": time.time(), "data": dict(data or {})}
    found._events.append(event)
    _trim_locked(found)
    _wake.notify_all()
    return event


def _trim_locked(found: Job) -> None:
    """Keep the log bounded by dropping the oldest ``chunk`` events only.

    Never a plain "drop the first entry": the first entry is ``queued``, and a
    log that had shed its own beginning would leave a late subscriber unable to
    tell a job that was still waiting from one that had already started.
    """
    if len(found._events) <= MAX_EVENTS:
        return
    kept, over = [], len(found._events) - MAX_EVENTS
    for event in found._events:
        if over > 0 and event["event"] == EV_CHUNK:
            over -= 1
            found._dropped += 1
            continue
        kept.append(event)
    found._events = kept


class Feed:
    """A caller's view of one request, or of the queue, as events arrive.

    Cursor-based on purpose. Every event carries a ``seq`` that only goes up, so
    a caller that reconnects passes back the last one it saw and gets exactly
    what it missed -- no duplicates to filter and no gap to notice later. A feed
    that has never read anything starts at 0 and therefore replays the job from
    ``queued``, which is what a caller subscribing a moment after submitting
    actually wants.

    Bound to one job, the feed closes itself after delivering that job's
    terminal event: there will never be another, and a connection held open for
    a finished request is a connection held open for nothing. A queue-wide feed
    (``job_id=None``) has no such moment and closes on :meth:`close` or on its
    idle TTL.
    """

    def __init__(self, job_id: str = "", ttl: float = FEED_TTL, cursor: int = 0):
        self.identifier = uuid.uuid4().hex[:12]
        self.job_id = str(job_id or "")
        self.ttl = float(ttl)
        self.cursor = int(cursor)
        self.created = time.time()
        self.touched = time.time()
        self.closed = False
        self.delivered = 0

    # -- lifecycle ------------------------------------------------------- #

    @property
    def expired(self) -> bool:
        return time.time() - self.touched > self.ttl

    def close(self) -> None:
        with _wake:
            self.closed = True
            _feeds.pop(self.identifier, None)
            _wake.notify_all()

    def describe(self) -> dict:
        return {"feed": self.identifier, "job": self.job_id or None,
                "cursor": self.cursor, "created": self.created,
                "idle": round(time.time() - self.touched, 3),
                "ttl": self.ttl, "closed": self.closed, "delivered": self.delivered}

    # -- reading --------------------------------------------------------- #

    def poll(self) -> list:
        """Everything since the cursor, right now, without waiting."""
        with _wake:
            self.touched = time.time()
            return self._collect_locked()

    def events(self, timeout: float | None = None, idle: float = 1.0):
        """Yield events as they happen. Ends when the feed does.

        ``idle`` is how long a single wait may block, not a timeout: it is what
        lets a caller forwarding this to an HTTP response notice its client has
        gone away, and what lets a closed feed stop promptly rather than at the
        next event. ``timeout`` is the real limit -- ``None`` means "until this
        feed closes", which for a job-bound feed is a bounded promise because
        every job reaches a terminal event.

        Nothing is yielded while the lock is held. A consumer that writes each
        event to a socket would otherwise be holding the queue's own lock for
        the duration of a network write, and one slow reader would stall the
        worker thread behind it.
        """
        started = time.monotonic()
        while True:
            with _wake:
                self.touched = time.time()
                batch = self._collect_locked()
                closed = self.closed
            for event in batch:
                yield event
            if closed:
                return
            if batch:
                continue
            if timeout is not None and time.monotonic() - started >= timeout:
                return
            with _wake:
                if self.closed:
                    return
                _wake.wait(timeout=idle)

    def _collect_locked(self) -> list:
        """Events past the cursor, advancing it. Closes on a bound terminal.

        Which counter the cursor means depends on what is being watched, and
        that is the whole difference between the two kinds of feed -- see
        :data:`_stream`.
        """
        if self.closed:
            return []
        counter = "seq" if self.job_id else "stream"
        batch = sorted((event
                        for found in self._sources_locked()
                        for event in found._events
                        if event[counter] > self.cursor),
                       key=lambda event: event[counter])
        if batch:
            self.cursor = batch[-1][counter]
        self.delivered += len(batch)
        if batch and self.job_id and batch[-1]["event"] in (EV_DONE, EV_FAILED, EV_CANCELLED):
            self.closed = True
            _feeds.pop(self.identifier, None)
        return batch

    def _sources_locked(self) -> list:
        if self.job_id:
            found = _jobs.get(self.job_id)
            return [found] if found is not None else []
        return list(_jobs.values())


def subscribe(job_id: str = "", *, ttl: float = FEED_TTL, cursor: int = 0) -> Feed:
    """Open a feed. ``job_id`` empty watches the whole queue.

    Raises :class:`Rejected` for a job that does not exist, rather than handing
    back a feed that will never produce anything: a caller that mistyped an id
    should find out now and not after a five-minute timeout.
    """
    key = str(job_id or "")
    with _wake:
        if key and key not in _jobs:
            raise Rejected("There is no request with that id.", "unknown_job")
        feed = Feed(key, ttl=ttl, cursor=cursor)
        _feeds[feed.identifier] = feed
        return feed


def feeds() -> list:
    """Every open feed, for a caller that wants to see its own housekeeping."""
    with _lock:
        _expire_feeds_locked()
        return [feed.describe() for feed in _feeds.values()]


def _expire_feeds_locked() -> None:
    for key in [key for key, feed in _feeds.items() if feed.expired or feed.closed]:
        feed = _feeds.pop(key, None)
        if feed is not None:
            feed.closed = True


def _close_feeds_for_locked(job_id: str) -> None:
    for key in [key for key, feed in _feeds.items() if feed.job_id == job_id]:
        feed = _feeds.pop(key, None)
        if feed is not None:
            feed.closed = True


# --------------------------------------------------------------------------- #
# The worker
# --------------------------------------------------------------------------- #


def _drain() -> None:
    """Take one request at a time, forever. Never dies on a bad job."""
    while True:
        try:
            if not drain_once(wait=WAKE_SECONDS):
                with _wake:
                    _reap_locked()
                    _expire_feeds_locked()
        except Exception:
            logger.warning("Model Chain: the MiniMax request queue hit an unexpected "
                           "error; it is still running", exc_info=True)
            time.sleep(WAKE_SECONDS)


def drain_once(wait: float = 0.0) -> bool:
    """Run the next request, if there is one. Returns whether it ran one.

    Separate from :func:`_drain` so the tests can step the queue deterministically
    instead of sleeping and hoping.
    """
    global _running
    with _wake:
        while not _pending:
            if wait <= 0:
                return False
            if not _wake.wait(timeout=wait):
                return False
        found = _jobs.get(_pending.popleft())
        if found is None or found.terminal:
            return True
        _running = found.identifier
        found.state = RUNNING
        found.started = time.time()
        _emit_locked(found, EV_STARTED, {"variant": found.variant, "seed": found.seed})
        _announce_positions_locked()
    try:
        _run(found)
    finally:
        with _wake:
            _running = ""
            if not found.terminal:
                # A path out of :func:`_run` nobody predicted. Recorded as a
                # failure rather than left RUNNING forever, because a job stuck
                # in RUNNING blocks the panel's gate for the life of the process.
                _finish_locked(found, FAILED,
                               error="The request ended without saying how.")
            found._image = None
            _announce_positions_locked()
            _wake.notify_all()
    return True


def _announce_positions_locked() -> None:
    """Tell everything still waiting where it is now, if it has moved.

    Emitted rather than left to be polled, because the number a waiting caller
    most wants is the one that changes when it is not looking. Only on a real
    change, so this is safe to call from every place the line can shift.
    """
    for index, key in enumerate(_pending, start=1):
        found = _jobs.get(key)
        if found is None or found.announced_position == index:
            continue
        found.announced_position = index
        _emit_locked(found, EV_POSITION, {"position": index, "waiting": len(_pending)})


def _run(found: Job) -> None:
    """One request, through exactly the machinery the panel uses.

    ``mc_llm_sessions`` is imported here and read off the module rather than
    bound at import time, for two reasons: this module must be importable in a
    host that has not finished setting up the LLM side, and the tests replace
    ``minimax`` with a generator of their own.
    """
    import mc_llm_sessions as sessions

    cancel_token = sessions.Cancellation()
    with _wake:
        found._cancel = cancel_token
        if found.cancelling:
            cancel_token.cancel()

    text, caption = "", ""
    try:
        for event in sessions.minimax(found.prompt, found.variant, found._image,
                                      found.seed, cancel_token,
                                      system=found.system):
            if event.kind == sessions.CHUNK:
                text += event.text or ""
                with _wake:
                    _emit_locked(found, EV_CHUNK, {"text": event.text or ""})
            elif event.kind == sessions.CAPTION:
                caption = event.text or ""
                with _wake:
                    found.caption = caption
                    _emit_locked(found, EV_CAPTION, {"text": caption})
            elif event.kind == sessions.STATUS:
                with _wake:
                    _emit_locked(found, EV_STATUS, {"text": event.text or ""})
            elif event.kind == sessions.DONE:
                with _wake:
                    found.result = event.text or ""
                    _finish_locked(found, DONE)
                _remember(found)
                return
            elif event.kind == sessions.CANCELLED:
                with _wake:
                    _finish_locked(found, CANCELLED,
                                   reason=found.cancel_reason or "cancelled")
                return
            elif event.kind == sessions.FAILED:
                with _wake:
                    _finish_locked(found, FAILED, error=event.text or "The run failed.")
                return
    except Exception as exc:
        logger.debug("Model Chain: MiniMax request %s failed", found.identifier,
                     exc_info=True)
        with _wake:
            _finish_locked(found, FAILED, error=str(exc) or exc.__class__.__name__)
        return
    # The generator ended without a terminal event, which the three modes never
    # do -- but a partial answer that was actually produced is worth more than
    # an error about the shape of a generator.
    with _wake:
        found.result = text
        if text.strip():
            _finish_locked(found, DONE)
        else:
            _finish_locked(found, FAILED, error="The run produced no prompt.")
    if found.state == DONE:
        _remember(found)


def _finish_locked(found: Job, state: str, error: str = "", reason: str = "") -> None:
    found.state = state
    found.finished = time.time()
    found.cancelling = False
    found._cancel = None
    if error:
        found.error = error
    if reason:
        found.cancel_reason = reason
    if state == DONE:
        payload = {"prompt": found.result, "caption": found.caption,
                   "seed": found.seed, "variant": found.variant,
                   "elapsed": found.elapsed}
    elif state == FAILED:
        payload = {"error": found.error}
    else:
        payload = {"reason": found.cancel_reason}
    _emit_locked(found, {DONE: EV_DONE, FAILED: EV_FAILED,
                         CANCELLED: EV_CANCELLED}[state], payload)
    logger.info("Model Chain: MiniMax request %s %s after %.1fs%s",
                found.identifier, state, found.elapsed,
                f" — {found.error}" if state == FAILED else "")


def _remember(found: Job) -> None:
    """File the finished prompt in the MiniMax history, as the panel would.

    "As if the user had gone to LLM Studio and asked for it" is the whole brief,
    and a prompt that never appeared in Saved prompts would be the one visible
    place that sentence stopped being true. A caller that is driving this in
    bulk turns it off, which is why it is a flag rather than an assumption.
    """
    if not found.remember:
        return
    try:
        import mc_llm_state

        mc_llm_state.save_minimax_session(mc_llm_state.MinimaxSession(
            variant=found.variant, prompt=found.prompt, caption=found.caption,
            result=found.result, seed=int(found.seed),
            image_name=found.image_used))
    except Exception:
        logger.debug("Model Chain: could not save the MiniMax session for request %s",
                     found.identifier, exc_info=True)


# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #


def _reap_locked() -> None:
    """Forget finished jobs that are old or surplus. Caller holds ``_lock``."""
    now = time.time()
    finished = sorted((found for found in _jobs.values() if found.terminal),
                      key=lambda found: found.finished)
    stale = [found for found in finished if now - found.finished > JOB_RETENTION]
    surplus = finished[:max(0, len(finished) - MAX_REMEMBERED)]
    for found in {found.identifier: found for found in stale + surplus}.values():
        _jobs.pop(found.identifier, None)
        _close_feeds_for_locked(found.identifier)
    _expire_feeds_locked()


def reset() -> None:
    """Forget everything. For the tests, and for a UI reload that wants a clean slate."""
    global _running, _stream
    with _wake:
        for feed in list(_feeds.values()):
            feed.closed = True
        _feeds.clear()
        _pending.clear()
        _jobs.clear()
        _running = ""
        _stream = 0
        _wake.notify_all()


def pause_worker(paused: bool = True) -> None:
    """Stop starting the background drain thread. The tests step it by hand."""
    global _paused
    with _lock:
        _paused = bool(paused)
