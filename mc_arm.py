"""Whether the pipeline can run without loading anything, and getting it there.

A warm run and a cold one are not the same product. From a user's log, five
consecutive Creative-plus-Spatial jobs on unchanged settings:

    81.7s   cold — llama-server loaded, and loaded a conservative placement
    27.3s   warmer — re-placed onto the whole card, prompt cache still empty
     3.6s   warm
     4.1s   warm
     5.1s   warm

Twenty times, and none of it is the model being faster. It is the same model
not being loaded again. Everything expensive happens once and then stops
happening, which makes "has it happened yet" a question worth being able to
answer and worth being able to answer *early*.

What this module is
-------------------
Two functions over machinery that already exists. :func:`readiness` measures --
it starts nothing and moves nothing, so a status panel may call it freely.
:func:`arm` does the loading, by asking the same entry points a generation
would have asked a moment later: ``mc_memory``'s preload for the image side,
``Runtime.client`` for the language model.

It is deliberately not a scheduler and not a cache. Nothing here decides where
a model goes, how large it may be, or what has to move for it -- those are
:mod:`mc_plan`, :mod:`mc_llm_runtime` and :mod:`mc_broker`, and arming is only
the decision to ask them *now* rather than when somebody is watching a progress
bar.

Why "armed"
-----------
The word is the user's, and it is a better word than "warm" for what this
reports, because it is about the whole pipeline rather than one model: an
installation with a resident checkpoint and no llama-server is warm in one
place and cold in the other, and the number that matters is when the next
Generate finishes.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import mc_broker

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_GB = 1024**3

COLD = "cold"
PARTIAL = "partly armed"
ARMED = "armed"

_ORDER = (COLD, PARTIAL, ARMED)

OPT_WARM_UP = "model_chain_warm_up"

WARM_OFF = "off"
WARM_BEFORE = "before"
WARM_STARTUP = "startup"

WARM_MODES = (
    (WARM_OFF, "Off — the first generation loads what it needs, as it needs it"),
    (WARM_BEFORE, "Before a generation — load everything first, then run; never a cold run"),
    (WARM_STARTUP, "At startup and before a generation — also load in the background when "
                   "the WebUI comes up, so the first Generate is already warm"),
)
"""When to load what a generation is going to need anyway.

None of these change *what* is loaded or where it goes. They change *when*, and
the only thing that costs is the moment somebody spends waiting for it: work
done at startup is work nobody is watching, and work done before a generation
is work that would otherwise have happened halfway through one.
"""


def mode() -> str:
    return mc_broker.resolve(mc_broker.option(OPT_WARM_UP, WARM_OFF), WARM_MODES, WARM_OFF)


# --------------------------------------------------------------------------- #
# Measuring
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Part:
    """One thing that has to be loaded, and whether it is."""

    name: str
    state: str
    detail: str = ""

    @property
    def armed(self) -> bool:
        return self.state == ARMED


@dataclass(frozen=True)
class Readiness:
    """The pipeline's state, part by part."""

    parts: tuple[Part, ...] = ()

    @property
    def state(self) -> str:
        """The worst part's state, because the slowest part is what is waited for."""
        if not self.parts:
            return ARMED
        return min((part.state for part in self.parts), key=_ORDER.index)

    @property
    def armed(self) -> bool:
        return self.state == ARMED

    @property
    def cold(self) -> tuple[Part, ...]:
        return tuple(part for part in self.parts if not part.armed)

    def describe(self) -> str:
        if not self.parts:
            return "nothing is configured to warm up"
        if self.armed:
            return "armed — " + "; ".join(part.name.lower() for part in self.parts)
        return f"{self.state} — " + "; ".join(
            f"{part.name.lower()} is {part.state}" for part in self.cold)

    def explain(self) -> str:
        """The same, plus why each cold part is cold.

        For the line a warm-up *ends* on, which is the one somebody reads when
        it did not work. "warm-up finished in 0.0s — cold — image model is
        cold" is a true sentence that answers nothing, and the answer was
        already measured a line earlier: every part carries the reason it is in
        the state it is in.
        """
        if self.armed or not self.parts:
            return self.describe()
        return f"{self.state} — " + "; ".join(
            f"{part.name.lower()} is {part.state}"
            + (f" ({part.detail})" if part.detail else "")
            for part in self.cold)


def readiness() -> Readiness:
    """What is loaded right now. Starts nothing, moves nothing, never raises.

    Safe to call from a status panel on every refresh, which is the whole
    reason it is separate from :func:`arm`: a readout that loaded a model to
    find out whether a model was loaded would be a very expensive way to draw a
    label, and would make the answer true by asking the question.
    """
    return Readiness(tuple(part for part in (_image_part(), _llm_part()) if part is not None))


def _image_part() -> Part | None:
    """Stage 1, measured against the host's live view. See ``mc_memory``."""
    try:
        import mc_memory

        state, said = mc_memory.stage_1_readiness()
    except Exception:
        logger.debug("Model Chain: could not read the image side's readiness", exc_info=True)
        return None
    # mc_memory's own three words, mapped onto this module's three. They mean
    # the same things and are deliberately not shared constants: that module
    # answers about one checkpoint and this one about a pipeline.
    mapped = {"warm": ARMED, "partially warm": PARTIAL}.get(str(state), COLD)
    # The console line it returns is a whole sentence with its own prefix; what
    # is wanted here is the half after the dash, if there is one.
    _, _, detail = str(said).partition("— ")
    return Part("Image model", mapped, detail.strip().rstrip(".") or str(said))


def _llm_part() -> Part | None:
    """Whether a llama-server is up, and whether it is placed as well as it can be.

    Running is most of it -- a model load was sixteen to eighteen seconds in
    the log this was written from -- but not all of it. The *first* start of a
    session is placed conservatively, because the runtime reserve has not been
    calibrated yet, and in that log it put thirty-seven of sixty-five layers on
    a card that turned out to hold all of them. A server running degraded is
    not cold, and it is not ready either.
    """
    try:
        import mc_llm_runtime

        if not mc_llm_runtime.config().configured:
            return None
        running = mc_llm_runtime.registry.running()
    except Exception:
        logger.debug("Model Chain: could not read the language model's readiness",
                     exc_info=True)
        return None
    if not running:
        return Part("Language model", COLD, "no llama-server is running")
    degraded = [found for found in running if _degraded(found)]
    if degraded:
        return Part("Language model", PARTIAL,
                    "running, with part of the model in system RAM")
    return Part("Language model", ARMED,
                f"{len(running)} llama-server{'' if len(running) == 1 else 's'} running")


def _degraded(found) -> bool:
    try:
        placement = found.placement()
    except Exception:
        return False
    if placement is None or not placement.on_gpu:
        return False
    import mc_llm_context

    return placement.gpu_layers != mc_llm_context.ALL_LAYERS


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

_arming = threading.Lock()


def arm(width: int = 0, height: int = 0, *, reason: str = "") -> Readiness:
    """Load what a generation would have loaded, now. Returns the state after.

    Synchronous, because the case it exists for is "never a cold run" and a
    warm-up that returned before it had warmed anything would be a slower cold
    run with a reassuring message on it.

    Never raises. Every part is attempted independently and a part that cannot
    be loaded leaves the pipeline exactly as cold as it was -- which the
    generation that follows already knows how to handle, because that is what
    it did before this module existed.

    The lock is not for correctness; both halves are safe to call twice. It is
    so that a startup warm-up and a Generate pressed two seconds later do not
    both pay for the same model load.
    """
    if not _arming.acquire(blocking=False):
        logger.info("Model Chain: a warm-up is already running; waiting for it")
        with _arming:
            return readiness()
    started = time.monotonic()
    try:
        before = readiness()
        if before.armed:
            return before
        logger.info("Model Chain: warming up%s — %s",
                    f" for {reason}" if reason else "", before.describe())
        # Image first. The order is not arbitrary and it is not free: a
        # warm-up that starts llama-server, spends twenty seconds on it and
        # then reports the image model cold has spent the user's whole wait on
        # the half they were not waiting for. Generate is what somebody is
        # sitting in front of.
        #
        # Nothing is lost on the language side by going second. Its VRAM
        # allowance is the plan's remainder, and ``mc_plan.usable_vram_bytes``
        # adds the image family's own residency back before dividing -- so
        # llama-server is placed at exactly the same size whether the
        # checkpoint has reached the card yet or not.
        _arm_image(width, height)
        _arm_llm()
        after = readiness()
    finally:
        _arming.release()
    logger.info("Model Chain: warm-up finished in %.1fs — %s",
                time.monotonic() - started, after.explain())
    return after


def _arm_llm() -> None:
    """Start the language model, so a generation does not have to.

    Asking for a client is what starts a server, which is the same entry point
    every mode uses and is deliberately not a second way in: a warm-up that
    started llama-server by some private path would be a placement nothing else
    had negotiated.
    """
    try:
        import mc_llm_runtime

        configuration = mc_llm_runtime.config()
        if not configuration.configured:
            return
        mc_llm_runtime.registry.for_role().client()
    except Exception:
        logger.info("Model Chain: the language model could not be warmed up; the next "
                    "request will start it", exc_info=True)


def _arm_image(width: int = 0, height: int = 0) -> None:
    """Get Stage 1's weights into VRAM, and wait for them.

    ``mc_memory`` owns every part of this: the load, the budget, the eviction
    and the circuit breaker. This only asks, and then waits -- because an
    asynchronous warm-up is not a warm-up as far as the generation behind it is
    concerned.

    Two things are asked for that the background pass after a generation does
    not ask for, and both are the difference between this warming the image
    model and this doing nothing at all:

    ``allow_disk_load``
        because "never a cold run" has to include the first run of a session,
        which is the coldest one there is.

    ``force``
        because the preload setting is consent to a background thread after
        every generation, and that is a different question from the one a user
        answers by turning this on. Requiring both -- one of them off by
        default and named after the other half of the extension -- is how a
        warm-up came to spend twenty seconds on a language model and report the
        image model cold. The circuit breaker is untouched: a machine where
        this does not work is still a machine where it does not work.
    """
    try:
        import mc_memory

        if mc_memory.preload_async(width, height, allow_disk_load=True, force=True):
            mc_memory.join_preload()
            return
        # Nothing started. Either it was already warm -- in which case the line
        # this warm-up ends with says so -- or it has been retired, which is
        # worth saying out loud rather than leaving as an unexplained 0.0s.
        retired = mc_memory.preload_disabled_reason()
        if retired is not None:
            logger.info("Model Chain: the image model was not warmed up — %s", retired)
    except Exception:
        logger.info("Model Chain: the image model could not be warmed up; the next "
                    "generation will load it", exc_info=True)


def arm_later(width: int = 0, height: int = 0, *, reason: str = "") -> None:
    """Arm on a background thread. For startup, where nobody is waiting."""
    thread = threading.Thread(target=arm, args=(width, height), kwargs={"reason": reason},
                              name="mc-arm", daemon=True)
    thread.start()


def on_app_started(*_args) -> None:
    """Warm up when the WebUI comes up, if that is what the setting says.

    The most valuable moment to do it and the cheapest: the model load happens
    while somebody is still finding their prompt, and the first Generate of the
    session is a warm one. Off by default, because starting a llama-server is
    not something a WebUI should do because it launched.
    """
    if mode() != WARM_STARTUP:
        return
    arm_later(reason="the WebUI starting")
