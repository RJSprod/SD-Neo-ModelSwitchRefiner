"""Whose turn it is on each card.

A graphics card can run one heavy thing at a time well, and until now that was
never a question anybody here had to answer out loud: the image model had the
card it generates on, the language model lived in what it left over, and WanGP
had a card of its own. A VibeVoice render is none of those. It is an eighteen-
gigabyte model that runs for minutes, on whichever card the user picks, and it
has to share that card with the thing that already lives there without either
of them ever being cut off halfway. So this module answers the question the
others never needed: *whose turn is it on this card?*

The rules
---------
They are the user's, stated for one physical card, and every function below is
one of them made concrete (``docs/23-voice-box.md`` section 4.1):

1. A running job is never cut off. An image job -- one Generate press, a whole
   batch -- a WanGP task, an LLM turn and a VibeVoice render all run to their end.
2. A VibeVoice request goes **next**, ahead of every job on its card that has
   not started.
3. All VibeVoice requests go first: consecutive ones on a card run back to back
   before any waiting job.
4. A render runs to its end in one generation.
5. One render per card; the two cards may each run one.
6. Between requests the guest stays **warm** on its card while nothing else
   waits for it, and is evicted -- only while idle -- when an image job, WanGP
   or the language model needs the card, or when system RAM runs low while the
   image model is parked for it.
7. The image model that made room comes back, from system RAM or from its
   recipe, never from a temporary file.
8. Never the pagefile: a request that would overflow VRAM or RAM is refused with
   a warning that names the shortfall.

Guests
------
VibeVoice is a *guest* here, not a caller: it is registered with this module by
``scripts/model_chain.py`` and handed a :class:`Turn` for each request. The
dependency runs one way on purpose. Voice modules may not import the memory
side at any depth (invariant I-3, ``tests/test_voice_independence.py``), and
this module imports the broker, the memory side and ``mc_wangp`` -- so the
engine is registered from outside and never imports this. A guest is any object
with three methods, each about one card, named by UUID::

    resident_bytes(card_uuid) -> int     # VRAM it holds there right now
    rendering(card_uuid) -> bool         # whether it is working there now
    evict(card_uuid, reason) -> int      # unload while idle; bytes freed

The three kinds of card
-----------------------
*The image model's card.* Every txt2img and img2img generation calls
:func:`image_gate` first thing in ``before_process``, and while a turn is queued
or running there the gate holds the job. A job the gate is holding is inside
Forge's ``queue_lock`` -- every GPU call the host makes is -- so nothing else the
host runs can start either. With no job to hold, the turn takes that lock itself,
but only without waiting for it: queueing for a FIFO lock would put VibeVoice
behind whatever was already waiting, which is rule 2 broken. The image model is
parked for the turn (:func:`image_return_mode`) and comes back afterwards.

*WanGP's card.* Mini Paint NEO feeds WanGP one Clipboard job at a time, so the
turn asks Mini Paint for the card through a lease (``mc_wangp.lease_*``): Mini
Paint stops submitting, lets the running job finish, has its bridge hold WanGP
between tasks, and reports the card held. The lease is renewed on every tick, so
a Forge that dies cannot leave WanGP held.

*Any other card.* Nothing to hold but the language model.

On every card, a language-model turn already running there finishes first, and
an idle llama-server there is stopped before the guest loads.

Threads
-------
One driver thread per card with work on it, which ends when the card has none.
The gate and the language model's hooks run on their own threads and only read
state and wait, except for one thing: evicting a warm guest, which is what the
thread that needs the card does itself, synchronously, before it proceeds.
Nothing here holds :data:`_lock` across a wait.
"""

from __future__ import annotations

import collections
import itertools
import logging
import threading
import time

import mc_broker

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_GB = 1024**3


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

OPT_IMAGE_RETURN = "model_chain_turn_image_return"

RETURN_AUTOMATIC = "automatic"
RETURN_RAM = "ram"
RETURN_RECIPE = "recipe"

RETURN_MODES = (
    (RETURN_AUTOMATIC, "Automatic — in system RAM when it fits above the margin, "
                       "otherwise from its recipe"),
    (RETURN_RAM, "From system RAM — the same model moved back, LoRAs and all; "
                 "VibeVoice is refused when RAM cannot hold it"),
    (RETURN_RECIPE, "From its recipe — dropped, and reloaded from its own files "
                    "afterwards; uses no RAM while VibeVoice runs"),
)
"""How the image model makes room for a VibeVoice turn on its card, and comes back.

The stored constants are the machine's; the labels are the user's and may be
reworded, which :func:`mc_broker.resolve` tolerates as long as the words before
the dash stay put.
"""

OPT_KEEP_WARM = "model_chain_turn_keep_warm"

WARM_EVERY = "every"
WARM_NOT_IMAGE = "not-image"
WARM_OFF = "off"

WARM_MODES = (
    (WARM_EVERY, "On every card — VibeVoice stays loaded until an image job, WanGP or "
                 "the LLM needs the card"),
    (WARM_NOT_IMAGE, "Not on the image model's card — there the image model comes "
                     "straight back"),
    (WARM_OFF, "Off — VibeVoice unloads after every request"),
)

RAM_MARGIN_BYTES = 4 * _GB
"""System RAM a turn leaves free, above everything it needs.

Twice the host floor (:data:`mc_memory.RAM_RESERVE_BYTES`), because this one is
about the pagefile rather than about a crash: Windows starts trimming working
sets -- paging parked weights out -- well before available memory reaches zero,
and a parked eighteen-gigabyte checkpoint paged out is exactly what the user
ruled out. Logged every time it decides something, so the number can be
checked against a real machine.
"""

VRAM_MARGIN_BYTES = 1 * _GB
"""VRAM a turn leaves free above the guest's own estimate: its activations and
the allocator's granularity, which no estimate made before loading can know."""

TICK_SECONDS = 0.25
"""How often a card's driver looks again. A turn waits on other work, never on
arithmetic, so a quarter of a second is well below anything a person notices."""

WARM_WATCH_SECONDS = 1.0
"""How often a warm stay is checked: its lease renewed (Mini Paint expires one in
twenty seconds), its guest still loaded, and the RAM under a parked model."""

GRANT_IDLE_SECONDS = 120.0
"""How long a granted turn may show nothing -- no memory, no work -- before its
guest is taken to have gone. Longer than any model load takes to begin."""

UNLOAD_WAIT_SECONDS = 10.0
"""How long a warm eviction waits for the guest to report its memory gone before
the card is handed back anyway. WanGP sizes itself against what the card reports
free and has no fallback, so it is not told the card is free while the guest is
still on it -- but a guest that never answers must not hold WanGP for ever."""

GATE_NOTICE = "Waiting for VibeVoice on {card} — this generation starts when it is done"

LEASE_OWNER = "ModelSwitchRefiner"
"""Who asks Mini Paint for WanGP's card. One owner, so a second request while a
lease lives is answered with that lease rather than refused (the contract)."""


def _option(name: str, default: str) -> str:
    return mc_broker.option(name, default)


def image_return_mode() -> str:
    """The stored value of *Bring the image model back after VibeVoice*."""
    return mc_broker.resolve(_option(OPT_IMAGE_RETURN, RETURN_AUTOMATIC), RETURN_MODES,
                             RETURN_AUTOMATIC)


def keep_warm_mode() -> str:
    """The stored value of *Keep VibeVoice warm between requests*."""
    return mc_broker.resolve(_option(OPT_KEEP_WARM, WARM_EVERY), WARM_MODES, WARM_EVERY)


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #

QUEUED = "queued"
"""Behind another VibeVoice request on the same card (rule 5)."""
CLEARING = "clearing"
"""The card's gates are closed to work that has not started, and this waits for
the work that has (rule 1)."""
MAKING_ROOM = "making room"
"""The image model is being parked, an idle llama-server stopped, WanGP flushed
if it has to be, and the card measured."""
GRANTED = "granted"
"""The guest may use the card."""
DONE = "done"
BLOCKED = "blocked"
"""Refused before anything was loaded; :attr:`Turn.warning` says why."""
CANCELLED = "cancelled"

FINAL = (DONE, BLOCKED, CANCELLED)
_HOLDING = (MAKING_ROOM, GRANTED)
"""Phases in which the card is the guest's, as far as a new language-model turn
is concerned. See :func:`llm_wait_reason` for why queued and clearing are not."""


# --------------------------------------------------------------------------- #
# Guests
# --------------------------------------------------------------------------- #

_guests: dict[str, object] = {}


def register_guest(name: str, guest) -> None:
    """Make ``guest`` known under ``name``. Called by ``scripts/model_chain.py``."""
    with _lock:
        _guests[str(name)] = guest


def unregister_guest(name: str) -> None:
    with _lock:
        _guests.pop(str(name), None)


def _guest(name: str):
    with _lock:
        return _guests.get(str(name))


def _resident(guest, uuid: str) -> int:
    try:
        return max(int(guest.resident_bytes(uuid) or 0), 0)
    except Exception:
        logger.debug("Model Chain: a guest could not say what it holds", exc_info=True)
        return 0


def _rendering(guest, uuid: str) -> bool:
    try:
        return bool(guest.rendering(uuid))
    except Exception:
        # Unanswerable is answered yes: the cost of being wrong that way is a
        # warm guest that stays a little longer, and the cost of the other is a
        # render cut off in the middle, which rule 1 exists to prevent.
        logger.debug("Model Chain: a guest could not say whether it is working",
                     exc_info=True)
        return True


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #


class Card:
    """One physical card, by the identity no ordering can move."""

    __slots__ = ("uuid", "index", "name")

    def __init__(self, uuid: str, index: int | None = None, name: str = ""):
        self.uuid = str(uuid or "")
        self.index = index
        self.name = str(name or "")

    @property
    def domain(self) -> mc_broker.ExecutionDomain:
        return mc_broker.cuda_execution(self.index, uuid=self.uuid, name=self.name)

    def describe(self) -> str:
        if self.name:
            return self.name
        return f"GPU {self.index}" if self.index is not None else "the card"


def card(uuid: str) -> Card:
    """The card with ``uuid``, its physical index and name filled in where known."""
    import mc_memory

    key = mc_memory._uuid_key(uuid)
    index = None
    name = ""
    try:
        topology = mc_memory._cards()
        index = topology.get("by_uuid", {}).get(key)
        if index is not None:
            name = topology.get("names", {}).get(index, "")
    except Exception:
        logger.debug("Model Chain: could not resolve a card's index", exc_info=True)
    return Card(uuid, index, name)


def _same_card(uuid: str, other: str) -> bool:
    import mc_memory

    mine, theirs = mc_memory._uuid_key(uuid), mc_memory._uuid_key(other)
    return bool(mine) and mine == theirs


def is_image_card(target: Card) -> bool:
    """Whether Forge generates on ``target``. Unanswerable is no: see below.

    No, because the gate is the only thing this changes: an answer of yes on the
    wrong card would hold image jobs for a render on another card, which costs
    the user their generations for nothing, and an answer of no on the right
    card still leaves the image model's own reclaim path (``_victim_order``) as
    the backstop that evicts a warm guest before a pass runs short.
    """
    try:
        return _same_card(target.uuid, mc_broker.image_device_uuid())
    except Exception:
        return False


def is_wangp_card(target: Card) -> bool:
    """Whether WanGP, as Mini Paint NEO runs it, is set up on ``target``."""
    try:
        import mc_wangp

        found = mc_wangp.presence()
        return bool(found.available and found.uuid and _same_card(target.uuid, found.uuid))
    except Exception:
        return False


def _wangp_running() -> bool:
    try:
        import mc_wangp

        found = mc_wangp.presence(fresh=True)
        return bool(found.available and found.running)
    except Exception:
        return False


def _free_on(target: Card, image_card: bool) -> int:
    """The driver's free figure for ``target``, without touching it needlessly.

    The image card is asked through torch, because Forge already has a context
    there. Any other card is asked through nvidia-smi: a torch reading from this
    process creates a CUDA context on the card it reads -- hundreds of megabytes
    of the very card a turn is trying to empty (handoff 28).
    """
    import mc_memory

    if target.index is None:
        return 0
    if image_card:
        return mc_memory.device_free_vram_bytes(target.index)
    return mc_memory.physical_free_vram_bytes(target.index, fresh=True)


# --------------------------------------------------------------------------- #
# Turns
# --------------------------------------------------------------------------- #

_ids = itertools.count(1)


class Turn:
    """One request for one card, from :func:`request` until it is finished.

    The guest waits on it (:meth:`wait`), uses the card while it is
    :data:`GRANTED`, and hands it back with :meth:`finish`. Everything else --
    what it is waiting for, and why it was refused -- is on the object for a
    status line to read.
    """

    def __init__(self, guest: str, target: Card, *, need_vram: int, need_ram: int,
                 label: str):
        self.id = next(_ids)
        self.guest = str(guest)
        self.card = target
        self.need_vram = max(int(need_vram), 0)
        self.need_ram = max(int(need_ram), 0)
        self.label = label or guest
        self.phase = QUEUED
        self.reason = ""
        """What this turn is waiting for right now, as a sentence."""
        self.warning = ""
        """Why it was refused, as a sentence with the numbers in it."""
        self.created = time.monotonic()
        self.granted_at: float | None = None
        self._keep_warm: bool | None = None
        self._said_unheld = False
        self._finished = False
        self._cancelled = False
        self._changed = threading.Condition()

    def _set(self, phase: str, reason: str = "", warning: str = "") -> None:
        with self._changed:
            self.phase = phase
            self.reason = reason
            if warning:
                self.warning = warning
            self._changed.notify_all()

    def _say(self, reason: str) -> None:
        with self._changed:
            self.reason = reason

    def wait(self, timeout: float | None = None, cancelled=None) -> str:
        """Block until granted, refused or cancelled. Returns the phase.

        ``cancelled`` is an optional ``threading.Event``: a request whose caller
        gave up is withdrawn rather than left in the queue, because a queued turn
        holds every image job on its card.
        """
        deadline = None if timeout is None else time.monotonic() + max(float(timeout), 0.0)
        with self._changed:
            while self.phase not in (GRANTED,) + FINAL:
                if cancelled is not None and cancelled.is_set():
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return self.phase
                self._changed.wait(0.25 if remaining is None else min(remaining, 0.25))
            else:
                return self.phase
        # The caller gave up. Withdrawn now, concluded by the card's driver on its
        # next step -- and reported as withdrawn at once, because a caller that
        # cancelled has no further use for "still clearing".
        self.cancel()
        return CANCELLED

    @property
    def granted(self) -> bool:
        return self.phase == GRANTED

    def finish(self, keep_warm: bool | None = None) -> None:
        """Hand the card back. ``keep_warm`` None follows the setting."""
        with self._changed:
            self._keep_warm = keep_warm
            self._finished = True
        _wake(self.card.uuid)

    def cancel(self) -> None:
        """Withdraw a request that has not been granted. A granted one finishes."""
        with self._changed:
            if self.phase == GRANTED:
                self._finished = True
            else:
                self._cancelled = True
        _wake(self.card.uuid)

    def describe(self) -> str:
        what = self.warning if self.phase == BLOCKED else self.reason
        return f"{self.label} on {self.card.describe()}: {self.phase}" + (
            f" — {what}" if what else "")


# --------------------------------------------------------------------------- #
# Per-card state
# --------------------------------------------------------------------------- #


class _CardState:
    def __init__(self, target: Card):
        self.card = target
        self.queue: collections.deque[Turn] = collections.deque()
        self.active: Turn | None = None
        self.warm: str = ""
        """The guest resident and idle here, or empty."""
        self.parked = ""
        """How the image model made room: ``ram``, ``recipe`` or empty."""
        self.lease = ""
        """Mini Paint's lease id while this card is WanGP's and held for us."""
        self.lease_held = False
        self.flushing = ""
        """The flush asked of WanGP for the turn making room here: ``soft``, ``hard`` or empty."""
        self.flush_said = ""
        """What Mini Paint said when that flush settled -- its outcome, or why it was refused."""
        self.queue_lock = None
        """Forge's queue lock, while this turn is holding it."""
        self.parked_job = None
        """The image job the gate is holding right now, by its job key."""
        self.passed: collections.deque = collections.deque(maxlen=32)
        """Job keys the gate has already let through (it is called twice a job)."""
        self.thread: threading.Thread | None = None
        self.woken = threading.Event()

    @property
    def busy(self) -> bool:
        return bool(self.queue or self.active is not None)

    def holding(self) -> bool:
        return self.active is not None and self.active.phase in _HOLDING


_lock = threading.RLock()
_cards: dict[str, _CardState] = {}
_sleep = time.sleep


def _state(uuid: str, create: bool = False) -> _CardState | None:
    import mc_memory

    key = mc_memory._uuid_key(uuid)
    with _lock:
        found = _cards.get(key)
        if found is None and create:
            found = _cards[key] = _CardState(card(uuid))
        return found


def _wake(uuid: str) -> None:
    found = _state(uuid)
    if found is not None:
        found.woken.set()


def request(guest: str, card_uuid: str, *, need_vram: int, need_ram: int = 0,
            label: str = "") -> Turn:
    """Ask for ``card_uuid`` on behalf of the registered guest ``guest``.

    Returns at once; the caller waits on the :class:`Turn`. ``need_vram`` is the
    guest's own estimate of what it will hold at its peak on the card, and
    ``need_ram`` what its process needs in system RAM beyond what it already
    has; both are what the pagefile rule is checked against.
    """
    if _guest(guest) is None:
        turn = Turn(guest, card(card_uuid), need_vram=need_vram, need_ram=need_ram,
                    label=label)
        turn._set(BLOCKED, warning=f"{label or guest} is not registered on this WebUI")
        return turn
    target = _state(card_uuid, create=True)
    turn = Turn(guest, target.card, need_vram=need_vram, need_ram=need_ram, label=label)
    with _lock:
        target.queue.append(turn)
    logger.info("Model Chain: %s asked for %s — next on the card", turn.label,
                target.card.describe())
    _ensure_driver(target)
    return turn


def _ensure_driver(target: _CardState) -> None:
    with _lock:
        if target.thread is not None and target.thread.is_alive():
            target.woken.set()
            return
        target.thread = threading.Thread(target=_drive, args=(target,),
                                         name=f"mc-turns-{target.card.uuid[-6:]}",
                                         daemon=True)
        target.thread.start()


def _drive(target: _CardState) -> None:
    """The card's driver: one tick at a time until it has nothing to do."""
    while True:
        with _lock:
            if not (target.busy or target.warm):
                target.thread = None
                return
        try:
            tick(target)
        except Exception:
            logger.warning("Model Chain: a turn on %s failed a step; retrying",
                           target.card.describe(), exc_info=True)
        target.woken.wait(TICK_SECONDS if target.busy else WARM_WATCH_SECONDS)
        target.woken.clear()


# --------------------------------------------------------------------------- #
# One step
# --------------------------------------------------------------------------- #


def tick(target: _CardState) -> None:
    """Advance ``target`` by one step. Public so a test can drive it by hand."""
    with _lock:
        active = target.active
        if active is None and target.queue:
            active = target.active = target.queue.popleft()
            if target.warm == active.guest:
                # The warm guest is this turn's now. It stops being warm the moment
                # it has a request again, or an image job arriving in the gap
                # before it starts working could evict it from under its own turn.
                target.warm = ""
            active._set(CLEARING, "waiting for the work already on the card")
    if active is None:
        if target.warm:
            _watch_warm(target)
        return

    if active._cancelled and active.phase not in (GRANTED,) + FINAL:
        _conclude(target, active, CANCELLED, keep_warm=None)
        return

    if active.phase == CLEARING:
        verdict, words = _clear(target, active)
        if verdict == "wait":
            active._say(words)
        elif verdict == "block":
            _refuse(target, active, words)
        else:
            active._set(MAKING_ROOM, "making room")
        return

    if active.phase == MAKING_ROOM:
        verdict, words = _make_room(target, active)
        if verdict == "wait":
            active._say(words)
            return
        if verdict == "block":
            _refuse(target, active, words)
            return
        _declare(target, active.guest, busy=True)
        logger.info("Model Chain: %s has %s", active.label, target.card.describe())
        active.granted_at = time.monotonic()
        active._set(GRANTED, "")
        return

    if active.phase == GRANTED:
        record = _renew(target)
        if (record is not None and target.lease_held and not active._said_unheld
                and str(record.get("phase") or "") != "held"):
            # A task that slipped past the hold, or WanGP started from its own tab
            # with a bridge that cannot hold it. Nothing can be done about it under
            # a render (rule 1), and it is worth one line in the log when WanGP
            # then runs short.
            active._said_unheld = True
            logger.warning("Model Chain: Mini Paint says WanGP's card is no longer held (%s) "
                           "while %s renders on it; the render goes on",
                           record.get("reason") or record.get("phase"), active.label)
        if active._finished:
            _conclude(target, active, DONE, keep_warm=active._keep_warm)
            return
        if _abandoned(target, active):
            logger.warning("Model Chain: %s was granted %s %.0f s ago and has neither loaded "
                           "nor started; giving the card back", active.label,
                           target.card.describe(), GRANT_IDLE_SECONDS)
            _conclude(target, active, DONE, keep_warm=False)


def _abandoned(target: _CardState, turn: Turn) -> bool:
    """Whether a granted turn's guest has gone without handing the card back.

    A guest finishes its turn in a ``finally``, so this is for the process that
    died or the caller that was killed: nothing working, nothing resident, for
    longer than any load takes to begin. Without it, a card whose guest crashed
    would hold every image job on it until the WebUI restarted.
    """
    found = _guest(turn.guest)
    if found is None:
        return True
    if turn.granted_at is None or time.monotonic() - turn.granted_at < GRANT_IDLE_SECONDS:
        return False
    return not _rendering(found, target.card.uuid) and _resident(found, target.card.uuid) <= 0


# --------------------------------------------------------------------------- #
# Clearing: the work that has started finishes
# --------------------------------------------------------------------------- #


def _clear(target: _CardState, turn: Turn) -> tuple[str, str]:
    """``("wait", why)``, ``("block", warning)`` or ``("go", "")``."""
    running = mc_broker.conflicting_llm(target.card.domain)
    if running is not None:
        return "wait", f"waiting for {running.label} to finish"

    if is_image_card(target.card):
        if not _image_card_quiet(target):
            return "wait", "waiting for the image generation to finish"
        if target.parked_job is None and target.queue_lock is None:
            lock = _host_queue_lock()
            if lock is not None:
                if not lock.acquire(False):
                    # Somebody took it between the check and here. Either it is a
                    # generation, which the gate will hold, or host work the gate
                    # cannot see, which is waited for like any other.
                    return "wait", "waiting for the WebUI's current job"
                target.queue_lock = lock

    if is_wangp_card(target.card):
        return _clear_wangp(target, turn)
    return "go", ""


def _image_card_quiet(target: _CardState) -> bool:
    """Whether nothing is running on the host except a job the gate is holding.

    Three signals, because none of them is enough alone: ``shared.state`` is
    cleared in the ``finally`` of any wrapped call and can read idle mid-job; the
    progress module's current task is None for an API call; and a job the gate
    is holding reads busy on both, correctly, while being exactly the job this
    turn is allowed to go ahead of.
    """
    if target.parked_job is not None:
        return True
    if mc_broker.host_busy():
        return False
    try:
        from modules import progress

        return getattr(progress, "current_task", None) is None
    except Exception:
        return True


def _host_queue_lock():
    try:
        from modules import call_queue

        return getattr(call_queue, "queue_lock", None)
    except Exception:
        return None


def _clear_wangp(target: _CardState, turn: Turn) -> tuple[str, str]:
    import mc_wangp

    if not target.lease:
        record = mc_wangp.lease_request(owner=LEASE_OWNER,
                                        purpose=turn.label, need_bytes=turn.need_vram)
        if record is None:
            # Mini Paint says WanGP lives on this card and offers no lease: one
            # older than the lease, or one whose lease just failed. Nothing can
            # hold WanGP, so the turn goes only while WanGP is not running --
            # and whatever WanGP holds still counts as not free when the card
            # is measured.
            if _wangp_running():
                return "block", ("WanGP is running on this card, and this Mini Paint NEO "
                                 "cannot hold it between jobs. Update Mini Paint NEO (bridge "
                                 "1.12.0), stop WanGP, or render on the other card")
            return "go", ""
        target.lease = str(record.get("lease") or "")
    else:
        record = mc_wangp.lease_state(target.lease)
        if record is None:
            # No answer is not an answer: the lease is asked for again on the
            # next step. Mini Paint gives the same owner its live lease back, and
            # one it has lost is simply a new request.
            with _lock:
                target.lease, target.lease_held = "", False
            return "wait", "waiting for Mini Paint to answer"

    phase = str(record.get("phase") or "")
    if phase in ("pending", "holding"):
        if record.get("bridge_hold") == "unsupported":
            # WanGP is up and cannot be held -- a bridge or a build without the
            # hold, or one Mini Paint cannot drive right now. Mini Paint keeps
            # such a lease holding until it is released, so waiting on it would
            # last exactly as long as WanGP stayed up.
            reason = str(record.get("reason") or "WanGP cannot be held between its jobs")
            _drop_lease(target)
            return "block", f"WanGP's card could not be held: {reason}"
        return "wait", str(record.get("reason") or "waiting for WanGP to finish its job")
    if phase == "held":
        if record.get("bridge_hold") == "unsupported" and _wangp_running():
            _drop_lease(target)
            return "block", ("WanGP is running, and its bridge is too old to hold its own "
                             "queue while VibeVoice renders. Install bridge 1.12.0 from "
                             "Mini Paint's WanGP setup, or render on the other card")
        target.lease_held = True
        return "go", ""
    # Refused, expired or released before it was ever held.
    reason = str(record.get("reason") or "Mini Paint did not hold WanGP")
    target.lease = ""
    if _wangp_running():
        return "block", f"WanGP's card could not be held: {reason}"
    return "go", ""


# --------------------------------------------------------------------------- #
# Making room
# --------------------------------------------------------------------------- #


def _make_room(target: _CardState, turn: Turn) -> tuple[str, str]:
    """Park, stop, flush and measure. ``("go", "")``, ``("wait", why)`` or ``("block", warning)``."""
    import mc_memory

    running = mc_broker.conflicting_llm(target.card.domain)
    if running is not None:
        # One that began between the clearing step and this one. A turn still
        # clearing is not waited for by a language-model turn the host's own job
        # is blocked on (llm_wait_reason), so one can start in that gap -- and
        # it is waited for here, before anything on the card is stopped under it.
        return "wait", f"waiting for {running.label} to finish"

    guest = _guest(turn.guest)
    uuid = target.card.uuid
    image_card = is_image_card(target.card)
    already = _resident(guest, uuid) if guest is not None else 0

    with _lock:
        other = target.warm if target.warm and target.warm != turn.guest else ""
        target.warm = "" if other else target.warm
    if other:
        _evict(target, other, f"{turn.label}'s turn")

    _stop_idle_llm(target)

    if image_card and not target.parked:
        verdict, words = _park_image_model(target, turn)
        if verdict == "block":
            return verdict, words

    free_ram = mc_memory.free_ram_bytes()
    if free_ram > 0 and free_ram < turn.need_ram + RAM_MARGIN_BYTES:
        return "block", (
            f"{turn.label} needs {turn.need_ram / _GB:.1f} GB of system RAM and "
            f"{RAM_MARGIN_BYTES / _GB:.0f} GB left free for Windows, and "
            f"{free_ram / _GB:.1f} GB is available. Nothing was loaded, so nothing "
            f"was paged")

    if image_card:
        mc_memory.release_cached_vram()
    needed = max(turn.need_vram - already, 0) + VRAM_MARGIN_BYTES
    if target.flushing:
        verdict = _flush_settled(target)
        if verdict is not None:
            return verdict
    free = _free_on(target.card, image_card)
    if free < needed and target.lease_held and target.flushing != "hard":
        return _ask_flush(target, "hard" if target.flushing == "soft" else "soft")
    if free and free < needed:
        return "block", (
            f"{turn.label} needs {turn.need_vram / _GB:.1f} GB on "
            f"{target.card.describe()} plus {VRAM_MARGIN_BYTES / _GB:.0f} GB of margin, "
            f"and {free / _GB:.1f} GB can be freed there"
            + (" with the image model out of the way" if target.parked else "")
            + (f". WanGP said: {target.flush_said}" if target.flush_said else ""))
    with _lock:
        target.flushing, target.flush_said = "", ""
    return "go", ""


def _stop_idle_llm(target: _CardState) -> None:
    """Stop our own idle llama-servers on this card. A running turn was waited for."""
    if target.card.index is None:
        return
    if mc_broker.held_bytes(mc_broker.FAMILY_LLM, card=target.card.index) <= 0:
        return
    mc_broker._release(mc_broker.FAMILY_LLM, 1 << 50,
                       f"VibeVoice's turn on {target.card.describe()}",
                       sweep=True, card=target.card.index)


def _park_image_model(target: _CardState, turn: Turn) -> tuple[str, str]:
    """Move the image model out of the turn's way, the way the setting says."""
    import mc_memory

    resident = mc_memory.resident_vram_bytes()
    if resident <= 0:
        return "go", ""
    mode = image_return_mode()
    free_ram = mc_memory.free_ram_bytes()
    fits = free_ram <= 0 or free_ram >= resident + turn.need_ram + RAM_MARGIN_BYTES
    if mode != RETURN_RECIPE and fits:
        target.parked = mc_memory.park_image_model(reason=f"{turn.label}'s turn")
        return "go", ""
    if mode == RETURN_RAM:
        return "block", (
            f"Bringing the image model back from system RAM needs "
            f"{resident / _GB:.1f} GB of RAM for it, {turn.need_ram / _GB:.1f} GB for "
            f"{turn.label} and {RAM_MARGIN_BYTES / _GB:.0f} GB left free for Windows; "
            f"{free_ram / _GB:.1f} GB is available. Nothing was moved. Choose "
            f"Automatic or From its recipe in Settings → Model Chain to let it make "
            f"room from its files instead")
    if not mc_memory.can_drop_image_model():
        # Dropping would fall back to parking in RAM -- the one thing RAM was just
        # measured too short for, or the one the user chose against.
        return "block", (
            f"This Forge cannot unload its model to reload it from its files, and system "
            f"RAM has {free_ram / _GB:.1f} GB available for a {resident / _GB:.1f} GB image "
            f"model. Nothing was moved; render on the other card, or free system RAM")
    if mode == RETURN_AUTOMATIC:
        logger.info("Model Chain: system RAM has %.1f GB free and parking the image model "
                    "needs %.1f GB plus a %.0f GB margin — it will be reloaded from its "
                    "files instead", free_ram / _GB, (resident + turn.need_ram) / _GB,
                    RAM_MARGIN_BYTES / _GB)
    target.parked = mc_memory.drop_image_model(reason=f"{turn.label}'s turn")
    return "go", ""


def _return_image_model(reason: str) -> None:
    try:
        import mc_arm

        mc_arm.arm_later(reason=reason)
    except Exception:
        logger.warning("Model Chain: could not bring the image model back; the next "
                       "generation will load it", exc_info=True)


def _ask_flush(target: _CardState, level: str) -> tuple[str, str]:
    """Ask Mini Paint for WanGP's VRAM, ``soft`` first and ``hard`` only if still short.

    The asking is all this does. A flush is Mini Paint's executor's to perform,
    on its own next pass, and until it has the lease says ``holding``; ``held``
    again is how the turn learns it is done (:func:`_flush_settled`). Measuring
    the card straight after asking reads a WanGP that has not started letting
    go -- and would refuse a render the flush was about to make room for.
    """
    import mc_wangp

    record = mc_wangp.lease_flush(target.lease, level)
    with _lock:
        target.flushing = level
    logger.info("Model Chain: asked WanGP for a %s flush on %s — %s", level,
                target.card.describe(),
                (record or {}).get("reason") or (record or {}).get("phase") or "no answer")
    return "wait", f"waiting for WanGP to move its weights off the card ({level} flush)"


def _flush_settled(target: _CardState) -> tuple[str, str] | None:
    """None once the flush asked for has settled; otherwise the verdict to return.

    Settled is the lease ``held`` again, whether the flush was done or refused
    (a WanGP run paused between tasks refuses a hard one); either way the card
    is measured next, and a refusal's reason goes into the warning if it is
    still short. A lease that stopped being held in the meantime ends the turn:
    WanGP may be running again, and the room that was being made is not the
    turn's to take.
    """
    import mc_wangp

    record = mc_wangp.lease_state(target.lease)
    if record is None:
        return "wait", "waiting for Mini Paint to answer"
    phase = str(record.get("phase") or "")
    reason = str(record.get("reason") or "")
    if phase in ("pending", "holding"):
        return "wait", reason or "waiting for WanGP to move its weights off the card"
    if phase != "held":
        return "block", ("WanGP's card stopped being held while its weights were being moved "
                         f"off it: {reason or phase}")
    with _lock:
        level, target.flush_said = target.flushing, reason
    logger.info("Model Chain: WanGP's %s flush has settled — %s", level,
                reason or "Mini Paint said nothing more")
    return None


# --------------------------------------------------------------------------- #
# After a turn
# --------------------------------------------------------------------------- #


def _keep_warm_here(target: _CardState, asked: bool | None) -> bool:
    if asked is False:
        return False
    mode = keep_warm_mode()
    if mode == WARM_OFF:
        return False
    if mode == WARM_NOT_IMAGE and is_image_card(target.card):
        return False
    if not target.lease_held and is_wangp_card(target.card):
        # A warm stay on WanGP's card lasts only as long as a lease: the lease is
        # the one thing that says when WanGP wants its card back. Without one,
        # the next job WanGP started would find eighteen gigabytes it was never
        # told about.
        return False
    return True


def _conclude(target: _CardState, turn: Turn, phase: str, *, keep_warm,
              warning: str = "") -> None:
    """End ``turn`` and decide what the card does next.

    Every ending comes through here -- finished, refused and withdrawn -- because
    each of them can leave the guest loaded: a refused turn for a guest that was
    already warm is still a warm guest, and one that is simply forgotten is an
    eighteen-gigabyte model nothing is tracking.
    """
    with _lock:
        target.active = None
        more = bool(target.queue)
        # Read with the same hold that frees the card: the gate clears this the
        # moment it sees the card free, and a job it was holding loads its own
        # model -- a warm-up started beside that load is two loads of one model.
        held_job = target.parked_job is not None
    guest = _guest(turn.guest)
    resident = _resident(guest, target.card.uuid) if guest is not None else 0
    turn._set(phase, "", warning)
    if more:
        # Rule 3: the next VibeVoice request goes before anything that waited.
        # Whatever this card is holding stays held.
        if resident > 0:
            with _lock:
                target.warm = turn.guest
            _declare(target, turn.guest, busy=False)
        return
    asked = keep_warm if phase == DONE else None
    if resident > 0 and _keep_warm_here(target, asked):
        with _lock:
            target.warm = turn.guest
        _declare(target, turn.guest, busy=False)
        _release_queue_lock(target)
        logger.info("Model Chain: %s stays warm on %s until something else needs the card",
                    turn.label, target.card.describe())
        return
    if resident > 0:
        _evict(target, turn.guest, "its request finished")
    _hand_back(target, return_image=not held_job)


def _refuse(target: _CardState, turn: Turn, warning: str) -> None:
    logger.warning("Model Chain: %s was refused on %s — %s", turn.label,
                   target.card.describe(), warning)
    _conclude(target, turn, BLOCKED, keep_warm=None, warning=warning)


def _hand_back(target: _CardState, *, return_image: bool) -> None:
    """Release everything the card was holding for guests, and bring back what moved.

    Reachable from the card's driver and from any thread that ends a warm stay,
    so each thing held is taken out of the state under the lock before it is
    released, and whichever thread gets there second finds nothing to release.
    """
    with _lock:
        parked, target.parked = target.parked, ""
    _release_queue_lock(target)
    _drop_lease(target)
    if parked and return_image:
        _return_image_model("VibeVoice gave the card back")


def _release_queue_lock(target: _CardState) -> None:
    with _lock:
        lock, target.queue_lock = target.queue_lock, None
    if lock is not None:
        try:
            lock.release()
        except Exception:
            logger.warning("Model Chain: could not give the WebUI's queue back", exc_info=True)


def _drop_lease(target: _CardState) -> None:
    with _lock:
        lease, target.lease = target.lease, ""
        target.lease_held = False
        target.flushing, target.flush_said = "", ""
    if lease:
        try:
            import mc_wangp

            mc_wangp.lease_release(lease, reason="VibeVoice gave the card back")
        except Exception:
            logger.warning("Model Chain: could not release WanGP's card", exc_info=True)


def _renew(target: _CardState) -> dict | None:
    if not target.lease:
        return None
    try:
        import mc_wangp

        return mc_wangp.lease_state(target.lease)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Warm residency
# --------------------------------------------------------------------------- #


def _declare(target: _CardState, guest: str, *, busy: bool) -> None:
    """Tell the broker what the guest holds here, and whether it may be evicted."""
    found = _guest(guest)
    size = _resident(found, target.card.uuid) if found is not None else 0
    key = _residency_key(guest, target.card)
    if size <= 0 and not busy:
        mc_broker.retire(key)
        return
    mc_broker.declare(mc_broker.FAMILY_VOICE, key, f"{guest} on {target.card.describe()}",
                      size, rank=mc_broker.RANK_ACTIVE if busy else mc_broker.RANK_HOT,
                      card=target.card.index)


def _residency_key(guest: str, target: Card) -> str:
    return f"voice:{guest}:{target.uuid}"


def _evict(target: _CardState, guest: str, reason: str) -> int:
    """Unload ``guest`` from ``target`` and wait until it says the memory is gone.

    The wait is the point, not a courtesy. Whoever needed the card goes next --
    an image pass, WanGP, a llama-server -- and each of them sizes itself against
    what the card reports free; handing the card over while the guest's process
    is still letting go of eighteen gigabytes is how the thing that went next
    runs out of memory. Bounded by :data:`UNLOAD_WAIT_SECONDS`.
    """
    found = _guest(guest)
    freed = 0
    if found is not None:
        try:
            freed = max(int(found.evict(target.card.uuid, reason) or 0), 0)
        except Exception:
            logger.warning("Model Chain: %s could not be unloaded from %s", guest,
                           target.card.describe(), exc_info=True)
        deadline = time.monotonic() + UNLOAD_WAIT_SECONDS
        while _resident(found, target.card.uuid) > 0:
            if time.monotonic() >= deadline:
                logger.warning("Model Chain: %s still reports memory on %s after %.0f s; "
                               "handing the card back anyway", guest,
                               target.card.describe(), UNLOAD_WAIT_SECONDS)
                break
            _sleep(0.1)
    mc_broker.retire(_residency_key(guest, target.card))
    logger.info("Model Chain: unloaded %s from %s (%.1f GB) — %s", guest,
                target.card.describe(), freed / _GB, reason)
    return freed


def end_warm(target: _CardState, reason: str, *, return_image: bool = True) -> int:
    """Evict the warm guest on ``target`` if it is idle. Returns bytes freed.

    Never a guest that is working: warm means idle, and a guest that says it is
    rendering here keeps the card (rule 1) -- the caller waits for its turn like
    everybody else.
    """
    with _lock:
        guest = target.warm
        busy = target.busy
    if not guest or busy:
        # A card with a request queued or running is not warm, whatever is
        # resident on it: that guest is next, and rule 3 keeps it there.
        return 0
    found = _guest(guest)
    if found is not None and _rendering(found, target.card.uuid):
        return 0
    with _lock:
        if target.warm != guest or target.busy:
            return 0
        target.warm = ""
    freed = _evict(target, guest, reason)
    if not target.busy:
        _hand_back(target, return_image=return_image)
    return freed


def _watch_warm(target: _CardState) -> None:
    """Keep a warm stay honest: its lease, its guest, and the RAM beneath it."""
    import mc_memory

    name = target.warm
    guest = _guest(name)
    if guest is None or _resident(guest, target.card.uuid) <= 0:
        # Unloaded by somebody else (the engine's own Unload, a crash): nothing
        # is warm any more, and what was parked for it can come back.
        with _lock:
            if target.warm != name or target.busy:
                return
            target.warm = ""
        mc_broker.retire(_residency_key(name, target.card))
        _hand_back(target, return_image=True)
        return
    if target.lease:
        record = _renew(target)
        phase = str((record or {}).get("phase") or "")
        if record is not None and phase != "held":
            end_warm(target, "WanGP's card is no longer held for it")
            return
        if record is not None and record.get("wanted"):
            end_warm(target, "WanGP needs its card")
            return
        if (record is not None and record.get("bridge_hold") == "unsupported"
                and _wangp_running()):
            # A bridge too old to hold WanGP's own queue held nothing but the
            # Clipboard, which was enough while WanGP was down. It is up now, and
            # its Generate button is not held by anything.
            end_warm(target, "WanGP started, and its bridge cannot hold it")
            return
    if target.parked == "ram":
        free_ram = mc_memory.free_ram_bytes()
        if 0 < free_ram < RAM_MARGIN_BYTES:
            logger.warning("Model Chain: system RAM is down to %.1f GB with the image model "
                           "parked in it — ending VibeVoice's warm stay so the model can go "
                           "back to the card", free_ram / _GB)
            end_warm(target, "system RAM is running low")


# --------------------------------------------------------------------------- #
# The image gate
# --------------------------------------------------------------------------- #


def _job_key(p=None):
    """What identifies the host job this call belongs to.

    The progress module's task id when there is one -- it is the same for every
    ``process_images`` a script makes inside one press, so an X/Y/Z plot is one
    job to the gate, as rule 1 needs. An API call has none, and the host's own
    start stamp stands in for it.
    """
    try:
        from modules import progress

        task = getattr(progress, "current_task", None)
        if task:
            return ("task", task)
    except Exception:
        pass
    try:
        from modules import shared

        state = shared.state
        stamp = getattr(state, "time_start", None)
        if stamp is not None:
            return ("state", getattr(state, "job", ""), stamp)
    except Exception:
        pass
    return ("p", id(p))


def _interrupted() -> bool:
    try:
        from modules import shared

        state = shared.state
        return bool(getattr(state, "interrupted", False) or getattr(state, "skipped", False)
                    or getattr(state, "stopping_generation", False))
    except Exception:
        return False


def _set_textinfo(text: str | None) -> None:
    try:
        from modules import shared

        shared.state.textinfo = text
    except Exception:
        pass


def image_gate(p=None) -> bool:
    """Hold an image generation while VibeVoice has, or is next for, its card.

    Called first thing in ``before_process`` for txt2img and img2img, from Model
    Chain's own script and from the gate script that covers img2img; the second
    call in a job returns at once. Returns True when it waited or evicted.

    Interrupt and Skip do not end the wait. They are honoured -- the host ends
    the job as soon as it continues -- but the job does not continue until the
    card is free, because a job let go early still loads its checkpoint and, for
    img2img, encodes its input on the card, on top of a render rule 1 says may
    not be cut off. The progress line says so.

    Never sets ``state.job`` or ``job_count``: a hook that does blocks every
    later request that waits for the host to go idle, including its own.
    """
    import mc_memory

    if not _cards:
        # No guest has ever asked for a card this session: the gate is one
        # dictionary check on every generation, and nothing else.
        return False
    uuid = ""
    try:
        uuid = mc_broker.image_device_uuid()
    except Exception:
        return False
    target = _state(uuid) if uuid else None
    if target is None:
        return False
    job = _job_key(p)
    with _lock:
        if job in target.passed:
            return False
        waiting = target.busy
        ahead = (target.active or (target.queue[0] if target.queue else None))
        label = ahead.label if ahead is not None else "VibeVoice"
    waited = False
    if waiting:
        waited = True
        logger.info("Model Chain: holding an image generation — %s is next on %s",
                    label, target.card.describe())
        while True:
            with _lock:
                if not target.busy:
                    target.parked_job = None
                    break
                target.parked_job = job
            text = GATE_NOTICE.format(card=target.card.describe())
            if _interrupted():
                text = "Interrupted — this generation ends as soon as VibeVoice is done"
            _set_textinfo(text)
            _sleep(TICK_SECONDS)
        _set_textinfo(None)
    if target.warm:
        # The image job needs the card: the warm guest leaves, and the job loads
        # its own model the way it always does -- from RAM or from its recipe,
        # whichever parked it -- so nothing is warmed here to be moved twice.
        if end_warm(target, "an image generation needs the card", return_image=False):
            waited = True
    with _lock:
        target.passed.append(job)
    if waited:
        mc_memory.ensure_model_loadable()
    # A turn that ended with nobody waiting brings the image model back on a
    # thread of its own (mc_arm). A job that arrives while that load is still
    # under way waits for it here rather than starting a second load of the
    # same model beside it -- which on img2img, where Model Chain's own join
    # never runs, nothing else would prevent.
    mc_memory.join_preload()
    return waited


def image_card_held() -> bool:
    """Whether a turn holds, or is about to hold, the image model's card.

    Asked by Model Chain's own background warming -- the post-generation preload
    and the warm-up -- which must not move weights onto a card a guest is about
    to use.
    """
    if not _cards:
        return False
    try:
        uuid = mc_broker.image_device_uuid()
    except Exception:
        return False
    target = _state(uuid) if uuid else None
    return bool(target is not None and (target.busy or target.warm))


# --------------------------------------------------------------------------- #
# The language model's side
# --------------------------------------------------------------------------- #


def llm_wait_reason(domain, *, inside_host_job: bool = False) -> str:
    """Why a language-model turn on ``domain`` must wait, or ``""`` when it need not.

    A turn that holds the card (making room, or granted) is waited for by every
    language-model turn on it. A turn that is queued or clearing is waited for
    too -- rule 3 puts VibeVoice first -- *except* by a turn the host's own job
    is blocked on, such as Krea's writer inside ``before_process``: that turn is
    part of the job the clearing turn is waiting for, and making it wait would be
    a deadlock with a progress bar on it.
    """
    if domain is None:
        return ""
    with _lock:
        states = list(_cards.values())
    for target in states:
        active = target.active
        if active is None and not target.queue:
            continue
        if not domain.conflicts_with(target.card.domain):
            continue
        if target.holding() or not inside_host_job:
            label = active.label if active is not None else target.queue[0].label
            return f"{label} on {target.card.describe()}"
    return ""


# --------------------------------------------------------------------------- #
# The broker's view
# --------------------------------------------------------------------------- #


class _VoiceReclaimer:
    """What the broker calls when an image pass or an LLM needs a guest's VRAM.

    It can only ever free a warm guest: a working one declared itself active,
    which the broker never asks for, and this checks again because a stale
    register is cheaper to distrust than to believe.
    """

    def release(self, needed_bytes: int, reason: str = "", *, card=mc_broker.ANY_CARD) -> int:
        # The image model is not brought back from here. The room was asked for
        # by somebody about to use it -- an image pass loads its own model, and a
        # language model is about to start in it -- so a warm-up now would be a
        # third tenant racing the second for the space the first just left.
        freed = 0
        for target in _states_on(card):
            if freed >= needed_bytes:
                break
            freed += end_warm(target, reason or "another workload needs the card",
                              return_image=False)
        return freed

    def resident_bytes(self, *, card=mc_broker.ANY_CARD) -> int:
        total = 0
        for target in _states_on(card):
            for name in {target.warm, getattr(target.active, "guest", "")} - {""}:
                found = _guest(name)
                if found is not None:
                    total += _resident(found, target.card.uuid)
        return total

    def describe(self) -> str:
        return "VibeVoice"


def _states_on(card) -> list[_CardState]:
    with _lock:
        states = list(_cards.values())
    if isinstance(card, type(mc_broker.ANY_CARD)):
        return states
    try:
        index = int(card)
    except (TypeError, ValueError):
        return []
    return [s for s in states if s.card.index is not None and int(s.card.index) == index]


mc_broker.register_reclaimer(mc_broker.FAMILY_VOICE, _VoiceReclaimer())


# --------------------------------------------------------------------------- #
# Reading it back
# --------------------------------------------------------------------------- #


def snapshot() -> list[dict]:
    """Every card with turns or a warm guest, for a status line or a panel."""
    with _lock:
        states = list(_cards.values())
    found = []
    for target in states:
        if not (target.busy or target.warm):
            continue
        found.append({
            "card": target.card.describe(),
            "uuid": target.card.uuid,
            "active": target.active.describe() if target.active is not None else "",
            "queued": [turn.describe() for turn in target.queue],
            "warm": target.warm,
            "parked": target.parked,
            "lease": bool(target.lease),
        })
    return found


def forget() -> None:
    """Drop every card's state. For tests; a live WebUI never needs this.

    Each card is emptied as well as forgotten, and its driver woken: a driver
    holds its card's state, not the table, and would otherwise go on ticking a
    card nobody could see -- into whatever test came next.
    """
    with _lock:
        states = list(_cards.values())
        _cards.clear()
        _guests.clear()
        for target in states:
            target.queue.clear()
            target.active = None
            target.warm = ""
            target.parked_job = None
    for target in states:
        _release_queue_lock(target)
        target.woken.set()
