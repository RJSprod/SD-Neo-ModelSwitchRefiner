"""WanGP on the other card: what the language model yields to it, and how it knows.

The machine this was written for has two cards. Forge and its image model are
on one; WanGP -- a video generator, started and owned by the Mini Paint NEO
extension in this same process -- is on the other; and the language model can
be configured onto either. Every rule in :mod:`mc_broker` is about the first
card: the image model keeps its VRAM and the LLM takes what is spare. Nothing
knew the second card had an owner. From a user's log, llama-server was placed
on WanGP's card in the room a generation had not yet taken, put twenty-one of
sixty-five layers there and the rest in system RAM, and then ran the processor
flat out for seven minutes -- on the card WanGP was rendering with and the
cores WanGP was feeding it from.

The rule this module adds is the user's, stated once:

    WanGP's VRAM is priority one and is never taken from it. The language
    model may use what WanGP is not using, is stopped when WanGP grows into
    the room kept for it, and is never a reason WanGP runs short. A card WanGP
    is not running on is the language model's to claim in full, and a language
    model on some other card costs WanGP nothing.

Three mechanisms carry it, and they are deliberately small:

**A ceiling on WanGP's card.** While WanGP is running there, a placement on
that card may spend at most the card less WanGP's footprint less a reserve.
The footprint is the larger of what WanGP holds *now* and the most it has been
seen holding this session -- a video model is loaded and unloaded around every
generation, so the idle reading is the one number that must not be sized
against -- and the reserve is room left free above that for the next
allocation. Everything else about placement is unchanged: the same ladder
shrinks the context, moves the experts and drops blocks against the smaller
figure, and lands in system RAM when nothing is spare.

**A watch that stops the server when it is squeezed.** Nothing in WanGP asks
this extension for memory; it allocates, and either gets it or fails. So while
a llama-server of ours holds VRAM on WanGP's card, a thread reads that card
every couple of seconds and stops the server the moment WanGP has eaten into
the reserve, the moment WanGP's known peak leaves no room for what the server
holds, or the moment WanGP starts generating on a card whose needs have not
been measured yet. Stopping is the only surrender a process has, and it is
the right one here: the weights stay warm in the page cache, and the next
request places again against what WanGP now leaves.

**Fewer processor threads while WanGP is up.** A placement with weights in
system RAM runs its matrix multiplies on the processor, and llama.cpp takes
every core for them by default. WanGP needs the processor too -- to encode a
prompt, to move tensors, to decode a frame -- and a server that has every core
starves it whichever card either is on. So a start that leaves anything on the
processor is capped at half the physical cores while WanGP is running, and says
so.

How it knows
------------
Mini Paint NEO owns the WanGP child and answers for it in-process:
``minipaint_neo.wangp.presence.report()``, one plain dict with the card's UUID,
whether the child is up and -- when its bridge has said -- whether it is
generating. That package is looked up by name and only if it has already been
imported by the host; nothing here adds a path, imports an extension that is
not loaded, or holds a reference across calls. A Mini Paint too old to have the
report is read through its runtime snapshot instead, without the generating
flag; no Mini Paint at all, or the setting turned off, is the whole of this
module answering "not applicable" and every placement behaving exactly as it
did before.

A WanGP started by hand, outside Mini Paint, is not seen by any of this. Its
VRAM still counts -- it is not free, so the ladder places around it as it
always has -- but nothing here knows it has priority.

Two namespaces, again
---------------------
Mini Paint records WanGP's card as a UUID, this extension records the LLM's
card as a UUID beside an nvidia-smi index, and Forge numbers cards a third way.
The comparison is by UUID whenever both sides have one and by physical index
otherwise, through the same topology :mod:`mc_memory` keeps for the image
card; a card that cannot be matched either way is *not* WanGP's, because the
cost of that mistake is the old behaviour on one card, and the cost of the
opposite is a language model shrunk for a WanGP it never shared a card with.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass

import mc_broker
import mc_llm_context

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_GB = 1024**3

OPT_MODE = "model_chain_wangp_mode"
OPT_RESERVE_GB = "model_chain_wangp_reserve_gb"
OPT_THREADS = "model_chain_wangp_threads"

MODE_SHARE = "share"
MODE_OFF = "off"

MODES = (
    (MODE_SHARE, "On WanGP's terms — on the card WanGP is running on, the LLM keeps to what "
                 "WanGP has never needed and stops when WanGP grows into the reserve; "
                 "while WanGP is up, a placement that touches the processor runs on fewer "
                 "threads"),
    (MODE_OFF, "Off — WanGP is not considered; the LLM places against free VRAM as before"),
)
"""Whether WanGP is given priority on its card. See the module docstring."""

DEFAULT_RESERVE_GB = 4.0
"""VRAM kept free on WanGP's card above the most WanGP has been seen holding.

Room for the next allocation rather than for the model: a generation grows in
steps -- weights, then activations, then a decoder -- and the watch below
reads the card every couple of seconds, so what the reserve buys is the time
between WanGP starting to grow and the server being gone. Four gigabytes is a
few seconds of a fast load and more than any single step of an idle-to-busy
transition observed while this was written.
"""

PRESENCE_TTL = 1.0
"""How long one reading of Mini Paint's report is reused.

The negotiation ladder asks the same question many times in one decision, and
a report that changed inside a second describes a WanGP that is starting or
stopping -- which the watch below sees on its next tick anyway.
"""

WATCH_SECONDS = 2.0
"""How often the watch reads WanGP's card while a server of ours is on it."""

PACKAGE = "minipaint_neo"
PRESENCE_MODULE = "minipaint_neo.wangp.presence"
RUNTIME_MODULE = "minipaint_neo.wangp.runtime"
CONFIG_MODULE = "minipaint_neo.wangp.config"
CONTROL_MODULE = "minipaint_neo.wangp.control"

THREADS_FLAG = "--threads"
THREADS_BATCH_FLAG = "--threads-batch"

KEPT = "kept"
RELEASED = "released"
STOP = "stop"
"""What one tick of the watch did. See :meth:`Watch.tick`."""


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def mode() -> str:
    return mc_broker.resolve(mc_broker.option(OPT_MODE, MODE_SHARE), MODES, MODE_SHARE)


def enabled() -> bool:
    return mode() == MODE_SHARE


def reserve_bytes() -> int:
    """What stays free on WanGP's card above its peak, from the setting."""
    try:
        gigabytes = float(mc_broker.option(OPT_RESERVE_GB, DEFAULT_RESERVE_GB))
    except (TypeError, ValueError):
        gigabytes = DEFAULT_RESERVE_GB
    return int(max(gigabytes, 0.0) * _GB)


def thread_count() -> int:
    """Processor threads a start may use while WanGP is up: the setting, or half the cores."""
    try:
        wanted = int(float(mc_broker.option(OPT_THREADS, 0) or 0))
    except (TypeError, ValueError):
        wanted = 0
    if wanted > 0:
        return wanted
    return max(_physical_cores() // 2, 1)


def _physical_cores() -> int:
    """Physical cores, which is what llama.cpp sizes its own default from."""
    try:
        import psutil

        found = psutil.cpu_count(logical=False)
        if found:
            return int(found)
    except Exception:
        logger.debug("Model Chain: could not count the physical cores", exc_info=True)
    logical = os.cpu_count() or 2
    return max(int(logical) // 2, 1)


# --------------------------------------------------------------------------- #
# Presence: what Mini Paint says about WanGP
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Presence:
    """WanGP as Mini Paint reports it, translated into this extension's terms.

    ``card`` is the *physical* index of the card the UUID names, or None when
    the topology cannot place it -- in which case the UUID alone still settles
    a comparison against a configuration that recorded one.
    """

    available: bool = False
    """Mini Paint NEO's WanGP integration is loaded in this process."""
    configured: bool = False
    running: bool = False
    generating: bool | None = None
    """True or False when WanGP's bridge has said; None when nothing has."""
    uuid: str = ""
    card: int | None = None
    state: str = ""
    instance: str = ""
    source: str = ""

    @property
    def where(self) -> str:
        return f"GPU {self.card}" if self.card is not None else "its card"


_seams: dict = {"source": None}
"""Test seam: a callable standing in for Mini Paint's ``presence.report()``."""

_module: dict = {"looked": False, "report": None}
"""The report function, resolved once per session. See :func:`_report`."""

_cache: list = [0.0, None]
_cache_lock = threading.Lock()


def use_source(source) -> None:
    """Answer presence questions from ``source()`` instead of Mini Paint. For tests.

    ``None`` puts the real lookup back. Whatever it returns is read exactly as
    ``report()`` would be, so a test can hand over the dict a real Mini Paint
    would, or ``None`` for a Mini Paint that is not there.
    """
    _seams["source"] = source
    forget_presence()


def forget_presence() -> None:
    """Drop the cached reading, so the next question is asked afresh."""
    with _cache_lock:
        _cache[0], _cache[1] = 0.0, None
    _module["looked"], _module["report"] = False, None


def _report() -> dict | None:
    """Mini Paint's report, or None when there is no Mini Paint to ask.

    By name and only once the host has imported the package: an extension that
    is not loaded is not looked for on disk, and one that is loaded is reached
    through the module system exactly as its own code reaches it. A Mini Paint
    without ``presence`` -- one older than this change -- is read through the
    runtime snapshot it has always had, with no generating flag.
    """
    source = _seams["source"]
    if source is not None:
        return source()
    if PACKAGE not in sys.modules:
        return None
    if not _module["looked"]:
        _module["looked"] = True
        try:
            _module["report"] = importlib.import_module(PRESENCE_MODULE).report
        except Exception:
            logger.debug("Model Chain: Mini Paint has no WanGP presence report; reading its "
                         "runtime instead", exc_info=True)
            _module["report"] = _fallback_report
    try:
        return _module["report"]()
    except Exception:
        logger.debug("Model Chain: could not read WanGP's presence", exc_info=True)
        return None


def _fallback_report() -> dict | None:
    """What an older Mini Paint can still be asked, in the report's own shape."""
    runtime = importlib.import_module(RUNTIME_MODULE)
    snapshot = dict(runtime.snapshot() or {})
    uuid, configured = "", False
    try:
        saved = importlib.import_module(CONFIG_MODULE).load()
        if saved is not None:
            configured = bool(getattr(saved, "initialized", False))
            uuid = str((getattr(saved, "gpu", None) or {}).get("uuid") or "")
    except Exception:
        logger.debug("Model Chain: could not read Mini Paint's WanGP setup", exc_info=True)
    generating = None
    try:
        hello = importlib.import_module(CONTROL_MODULE).last_hello()
        if hello is not None and isinstance(hello.get("generation_running"), bool):
            generating = bool(hello["generation_running"])
    except Exception:
        generating = None
    return {
        "version": 0, "available": True, "configured": configured, "gpu_uuid": uuid,
        "state": str(snapshot.get("state") or ""), "running": bool(snapshot.get("running")),
        "instance_id": str(snapshot.get("instance_id") or ""), "generating": generating,
        "source": "runtime",
    }


def _tristate(value) -> bool | None:
    return value if isinstance(value, bool) else None


def card_index(uuid: str) -> int | None:
    """The physical index of the card ``uuid`` names, or None when it cannot be placed."""
    try:
        import mc_memory

        key = mc_memory._uuid_key(uuid)
        if not key:
            return None
        found = mc_memory._cards()["by_uuid"].get(key)
    except Exception:
        logger.debug("Model Chain: could not place WanGP's card in the topology", exc_info=True)
        return None
    return int(found) if found is not None else None


def _read_presence() -> Presence:
    raw = _report()
    if not isinstance(raw, dict):
        return Presence()
    uuid = str(raw.get("gpu_uuid") or "")
    return Presence(
        available=True,
        configured=bool(raw.get("configured")),
        running=bool(raw.get("running")),
        generating=_tristate(raw.get("generating")),
        uuid=uuid,
        card=card_index(uuid),
        state=str(raw.get("state") or ""),
        instance=str(raw.get("instance_id") or ""),
        source=str(raw.get("source") or "presence"),
    )


def presence(fresh: bool = False) -> Presence:
    """WanGP as Mini Paint reports it right now. Never raises.

    Cached for :data:`PRESENCE_TTL`; ``fresh`` asks again regardless, which is
    what the watch does, because the watch is the one caller for whom a
    second-old answer is the wrong one.
    """
    now = time.monotonic()
    with _cache_lock:
        when, found = _cache
        if not fresh and found is not None and now - when < PRESENCE_TTL:
            return found
    try:
        found = _read_presence()
    except Exception:
        logger.debug("Model Chain: could not read WanGP's presence", exc_info=True)
        found = Presence()
    with _cache_lock:
        _cache[0], _cache[1] = now, found
    return found


def is_wangps_card(card, configuration=None, found: Presence | None = None) -> bool:
    """Whether ``card`` -- the LLM's, physical index -- is the card WanGP is running on.

    UUIDs first, from ``configuration.gpu_uuid`` and Mini Paint's record, then
    physical indices, which are the same namespace on both sides: the LLM's
    is nvidia-smi's from setup, WanGP's is translated from its UUID through the
    topology. Unanswerable is *no* -- see the module docstring for which way
    that caution points and why.
    """
    if not enabled():
        return False
    found = found if found is not None else presence()
    if not found.running:
        return False
    try:
        import mc_memory

        mine = mc_memory._uuid_key(getattr(configuration, "gpu_uuid", "") or "")
        theirs = mc_memory._uuid_key(found.uuid)
    except Exception:
        mine, theirs = "", ""
    if mine and theirs:
        return mine == theirs
    if found.card is None or card is None:
        return False
    try:
        return int(card) == int(found.card)
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# The ceiling: what WanGP leaves on its card
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Observation:
    """One reading of WanGP's card, in the terms the ceiling is decided in."""

    card: int
    free: int
    total: int
    ours: int
    """What llama-servers of ours hold on the card, from the register."""
    others: int
    """What everything else holds: WanGP, the driver, the desktop."""
    peak: int
    """The most ``others`` has been this session while WanGP was running."""
    cap: int
    """What ours may hold in total: the card, less the peak, less the reserve."""


_peaks: dict = {}
"""Per WanGP card (by UUID key), the most it has been seen holding."""

_learned: set = set()
"""Cards whose WanGP has been watched through a generation, or has squeezed us.

Until a card is here its peak is a reading of an idle WanGP, and a generation
starting there is met by giving the whole card back rather than by trusting a
number that has never been tested. See :func:`_squeeze`.
"""

_said: dict = {}
"""The last ceiling reported per card, so the console hears a change and not a loop."""

_state_lock = threading.Lock()


def forget() -> None:
    """Drop the learned peaks and the cached presence. For tests and a settings reset."""
    with _state_lock:
        _peaks.clear()
        _learned.clear()
        _said.clear()
    forget_presence()


def _key(uuid: str) -> str:
    try:
        import mc_memory

        return mc_memory._uuid_key(uuid) or "?"
    except Exception:
        return "?"


def _note_peak(uuid: str, others: int) -> int:
    key = _key(uuid)
    with _state_lock:
        peak = max(int(_peaks.get(key, 0)), int(others))
        _peaks[key] = peak
        return peak


def _learn(uuid: str) -> None:
    with _state_lock:
        _learned.add(_key(uuid))


def learned(uuid: str) -> bool:
    with _state_lock:
        return _key(uuid) in _learned


def _reading(card: int) -> tuple[int, int]:
    """``(free, total)`` on physical card ``card`` as the driver reports, or zeros."""
    try:
        free = int(mc_broker.device_free_vram_bytes(card))
    except Exception:
        free = 0
    try:
        import mc_memory

        total = int(mc_memory.physical_total_vram_bytes(card))
    except Exception:
        total = 0
    return max(free, 0), max(total, 0)


def observe(card, found: Presence | None = None) -> Observation | None:
    """Read WanGP's card and file its peak. None when the card cannot be read.

    ``ours`` is what the register and the LLM reclaimer say is ours on this
    card, so a server of ours already there is not mistaken for WanGP growing
    -- and, the other way round, a server that is mid-start and undeclared
    would be; which is why the watch reads only after a start has declared.
    """
    if card is None:
        return None
    found = found if found is not None else presence()
    try:
        index = int(card)
    except (TypeError, ValueError):
        return None
    free, total = _reading(index)
    if total <= 0:
        return None
    ours = max(int(mc_broker.held_bytes(mc_broker.FAMILY_LLM, card=index)), 0)
    others = max(total - free - ours, 0)
    peak = _note_peak(found.uuid, others)
    cap = max(total - max(peak, others) - reserve_bytes(), 0)
    return Observation(index, free, total, ours, others, peak, cap)


def cap_bytes(card, configuration=None) -> int:
    """What the LLM may hold on ``card`` while WanGP runs there, or -1 when it does not.

    -1 is "not applicable" -- the setting is off, Mini Paint is not here, WanGP
    is not running, or this is not its card -- and every caller treats it as
    "no ceiling from here", which is the behaviour every placement had before.
    """
    if card is None or not enabled():
        return -1
    found = presence()
    if not is_wangps_card(card, configuration, found):
        return -1
    seen = observe(card, found)
    if seen is None:
        return -1
    _say_ceiling(seen, found)
    return seen.cap


def _say_ceiling(seen: Observation, found: Presence) -> None:
    """One console line per change of ceiling, not one per rung of the ladder."""
    step = 256 * 1024 * 1024
    with _state_lock:
        last = _said.get(seen.card)
        if last is not None and abs(last - seen.cap) < step:
            return
        _said[seen.card] = seen.cap
    logger.info("Model Chain: WanGP is running on GPU %d%s — it holds %.1f GB there (peak "
                "%.1f GB this session) and %.1f GB stays free for it, so the LLM may hold "
                "up to %.1f GB of the card's %.1f GB",
                seen.card, " and generating" if found.generating else "",
                seen.others / _GB, seen.peak / _GB, reserve_bytes() / _GB, seen.cap / _GB,
                seen.total / _GB)


# --------------------------------------------------------------------------- #
# The watch: stopping the server when WanGP squeezes it
# --------------------------------------------------------------------------- #


def _squeeze(seen: Observation, found: Presence, began: bool) -> str:
    """Why the server on this card has to go now, or "" when it may stay.

    Two reasons, checked in the order they are urgent:

    1. The server holds more than the ceiling now allows. Two ways that
       happens, said in two ways because they are found in two ways: WanGP
       has grown into the reserve right now -- the room kept for its next
       allocation is being used for one, and the server is what is in the way
       -- or WanGP's *known* peak leaves less than the server holds, because
       the server was placed while WanGP was smaller, or not running, and the
       card has since been seen needing more. One comparison covers both: the
       ceiling is the card less the larger of the two footprints less the
       reserve, so a live squeeze and a remembered one are the same
       arithmetic.
    2. WanGP has just gone from idle to generating and this card's needs have
       never been measured. Its idle footprint is the only peak on record and
       is not a number to bet a generation on, so the whole card is given
       back; once a generation has been watched through, or has squeezed us
       once, the peak is trusted and a server that fits under it stays. A
       server placed while a generation was already running was sized
       against that generation and is not sent away by this rule.
    """
    if seen.ours <= 0:
        return ""
    reserve = reserve_bytes()
    if seen.ours > seen.cap:
        if seen.free < reserve:
            return (f"WanGP has grown into the {reserve / _GB:.1f} GB kept for it on GPU "
                    f"{seen.card} ({seen.free / _GB:.1f} GB free)")
        return (f"WanGP's peak on GPU {seen.card} ({seen.peak / _GB:.1f} GB) leaves the LLM "
                f"{seen.cap / _GB:.1f} GB there and it holds {seen.ours / _GB:.1f} GB")
    if began and not learned(found.uuid):
        return (f"WanGP has started generating on GPU {seen.card} and how much it needs "
                f"there has not been measured yet, so the LLM gives it the whole card")
    return ""


def _our_cards() -> list:
    """``(card, configuration)`` for every running server of ours holding VRAM on a card."""
    reclaimer = mc_broker._reclaimer(mc_broker.FAMILY_LLM)
    running = getattr(reclaimer, "running", None)
    if not callable(running):
        return []
    found = []
    try:
        servers = list(running())
    except Exception:
        logger.debug("Model Chain: could not list the running llama-servers", exc_info=True)
        return []
    for server in servers:
        try:
            settings = server.configuration()
            if not getattr(settings, "uses_cuda_compute", False):
                continue
            card = getattr(server, "_card", None)
            if card is None:
                card = int(getattr(settings, "gpu_index"))
            if int(server.resident_bytes(card=card) or 0) <= 0:
                continue
        except Exception:
            logger.debug("Model Chain: could not ask a llama-server which card it holds",
                         exc_info=True)
            continue
        found.append((int(card), settings))
    return found


def _release(seen: Observation, reason: str) -> int:
    """Stop every server of ours on the card, for WanGP. Returns bytes freed."""
    released = mc_broker._release(mc_broker.FAMILY_LLM, seen.ours, reason, sweep=True,
                                  card=seen.card)
    if released.freed > 0:
        mc_broker.note(mc_broker.FAMILY_LLM,
                       f"stopped llama-server on GPU {seen.card} for WanGP — {reason}; the "
                       f"weights stay warm in the system page cache, and the next request "
                       f"places again against what WanGP now leaves")
    else:
        logger.info("Model Chain: llama-server on GPU %d could not be stopped for WanGP — %s",
                    seen.card, reason)
    return int(released.freed)


class Watch:
    """Reads WanGP's card while a server of ours is on it, and stops that server when it must.

    One thread, started when a server is handed out and ending itself when
    nothing of ours holds VRAM on any card. :meth:`tick` is the whole of the
    decision and takes no thread, so a test drives it directly.
    """

    def __init__(self, sleep=time.sleep) -> None:
        self._sleep = sleep
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_generating: bool | None = None
        self.releases = 0
        self.ticks = 0

    @property
    def watching(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def arm(self) -> bool:
        """Start the watch if there is a WanGP to watch for. Returns whether one is up."""
        if not enabled():
            return False
        if not presence().available:
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._thread = threading.Thread(target=self._run, name="mc-wangp-watch",
                                            daemon=True)
            self._thread.start()
        return True

    def _run(self) -> None:
        try:
            while True:
                try:
                    outcome = self.tick()
                except Exception:
                    logger.debug("Model Chain: the WanGP watch failed a tick", exc_info=True)
                    outcome = KEPT
                if outcome == STOP:
                    return
                self._sleep(WATCH_SECONDS)
        finally:
            with self._lock:
                self._thread = None

    def tick(self) -> str:
        """One reading of every card of ours that is WanGP's. See :func:`_squeeze`."""
        self.ticks += 1
        if not enabled():
            return STOP
        found = presence(fresh=True)
        if not found.available:
            return STOP
        ours = _our_cards()
        if not ours:
            # Nothing of ours holds a card, so there is nothing to protect
            # WanGP from. The next start arms the watch again.
            self._last_generating = None
            return STOP
        if not found.running:
            # WanGP is off. Its card is the LLM's to keep; the watch stays up
            # only so that WanGP starting is noticed on the next tick.
            self._last_generating = None
            return KEPT
        # Idle to generating, and only that: a card first seen with a
        # generation already on it was placed against that generation's
        # footprint, which is the measurement rule 3 is waiting for.
        began = found.generating is True and self._last_generating is False
        ended = found.generating is False and self._last_generating is True
        self._last_generating = found.generating
        if ended:
            # A generation was watched from start to finish, and its peak is
            # on record: from here on that number is trusted.
            _learn(found.uuid)
        outcome = KEPT
        for card, settings in ours:
            if not is_wangps_card(card, settings, found):
                continue
            seen = observe(card, found)
            if seen is None:
                continue
            reason = _squeeze(seen, found, began)
            if not reason:
                continue
            if _release(seen, reason) > 0:
                self.releases += 1
                _learn(found.uuid)
                outcome = RELEASED
        return outcome


_watch = Watch()


def watch() -> bool:
    """Arm the watch for a server that has just been handed out. Never raises."""
    try:
        return _watch.arm()
    except Exception:
        logger.debug("Model Chain: could not start the WanGP watch", exc_info=True)
        return False


# --------------------------------------------------------------------------- #
# Threads: leaving WanGP the processor
# --------------------------------------------------------------------------- #


def touches_the_processor(configuration, placement) -> bool:
    """Whether this placement runs any of the model's arithmetic on the processor.

    A full offload does not, whatever the card; everything else does -- a
    processor placement outright, a partial offload for the blocks it left
    behind, and an expert spill for the experts it moved, which are most of
    the weights. An Intel GPU placement is unified memory and is left alone.
    """
    if getattr(placement, "uma", False):
        return False
    if not getattr(placement, "on_gpu", True):
        return True
    if int(getattr(placement, "gpu_layers", mc_llm_context.ALL_LAYERS)) != mc_llm_context.ALL_LAYERS:
        return True
    return int(getattr(placement, "cpu_expert_layers",
                       mc_llm_context.NO_EXPERTS)) != mc_llm_context.NO_EXPERTS


def thread_flags(configuration, placement, supports=None) -> list[str]:
    """``--threads N --threads-batch N`` for a start that will share the processor with WanGP.

    Only while WanGP is running -- on any card, because the processor is the
    machine's -- only for a placement that touches the processor at all, and
    only in the spelling this build advertises. ``supports`` is asked per
    flag; by default it is :func:`mc_llm_runtime.runtime_supports` for the
    configured build.
    """
    if not enabled():
        return []
    found = presence()
    if not found.running:
        return []
    if not touches_the_processor(configuration, placement):
        return []
    count = thread_count()
    if count <= 0:
        return []
    if supports is None:
        import mc_llm_runtime

        def supports(flag, configuration=configuration):
            return mc_llm_runtime.runtime_supports(flag, configuration)

    flags: list[str] = []
    for flag in (THREADS_FLAG, THREADS_BATCH_FLAG):
        try:
            if supports(flag):
                flags += [flag, str(count)]
        except Exception:
            logger.debug("Model Chain: could not ask whether the build takes %s", flag,
                         exc_info=True)
    if flags:
        logger.info("Model Chain: WanGP is running, so this placement — which runs part of "
                    "the model on the processor — is held to %d threads to leave WanGP the "
                    "rest of the cores", count)
    return flags


# --------------------------------------------------------------------------- #
# Saying so
# --------------------------------------------------------------------------- #


def holds(card) -> bool:
    """Whether WanGP is the owner of VRAM on ``card`` that neither family accounts for."""
    if card is None or not enabled():
        return False
    found = presence()
    return is_wangps_card(card, None, found)


def stray_explanation(card) -> str:
    """The stray sentence for a card WanGP is running on, or "" for any other."""
    if not holds(card):
        return ""
    return ("of the card is held by WanGP, which has priority there: the LLM keeps to what "
            "WanGP leaves and is stopped when WanGP needs the room")


def _llm_has_business_on(card, found: Presence) -> bool:
    """Whether a server of ours holds VRAM on ``card``, or is configured to be placed there.

    The panel reads WanGP's card only then. The first driver reading of a card
    from this process costs a CUDA context on it -- hundreds of megabytes of
    the very card this module exists to leave alone -- and a placement on that
    card pays it anyway, so the panel adds nothing; a machine whose language
    model never goes near WanGP's card must not pay it for a status line.
    """
    if card is None:
        return False
    try:
        if mc_broker.held_bytes(mc_broker.FAMILY_LLM, card=int(card)) > 0:
            return True
    except Exception:
        logger.debug("Model Chain: could not ask what the LLM holds on WanGP's card", exc_info=True)
    try:
        import mc_llm_runtime

        configuration = mc_llm_runtime.config()
        return is_wangps_card(mc_llm_runtime.card_of(configuration), configuration, found)
    except Exception:
        logger.debug("Model Chain: could not read the LLM's configured card", exc_info=True)
        return False


def describe() -> str:
    """One sentence for the residency panel, or "" when there is nothing to say."""
    if not enabled():
        return "WanGP: not considered — the setting is off"
    found = presence()
    if not found.available:
        return ""
    if not found.configured:
        return "WanGP: not set up in Mini Paint NEO"
    if not found.running:
        return "WanGP: not running — its card is the LLM's to use in full"
    busy = (" and generating" if found.generating is True
            else " and idle" if found.generating is False else "")
    if not _llm_has_business_on(found.card, found):
        return (f"WanGP: running on {found.where}{busy} — the LLM is neither on its card nor "
                f"configured for it, so nothing here reads that card")
    seen = observe(found.card, found) if found.card is not None else None
    if seen is None:
        return f"WanGP: running on {found.where}{busy} — what it holds could not be read"
    return (f"WanGP: running on {found.where}{busy} — holds {seen.others / _GB:.1f} GB "
            f"(peak {seen.peak / _GB:.1f} GB this session); {reserve_bytes() / _GB:.1f} GB "
            f"stays free for it, so the LLM may hold up to {seen.cap / _GB:.1f} GB there"
            + (f", and holds {seen.ours / _GB:.1f} GB" if seen.ours > 0 else ""))
