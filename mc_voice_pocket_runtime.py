"""The PocketTTS worker process: starting it, speaking through it, and the drain.

The parent half of the third engine. It owns a subprocess, a pipe, a handshake
it refuses to accept a bad answer to, five ways of making sure the process dies
with this one, and -- uniquely among the three engines -- a state in which
playback has already stopped and the engine has not finished yet.

Why Pocket's Stop is not Sopro's Stop
-------------------------------------
Kokoro's and Sopro's synthesis are generators the worker stops pulling from, so
abandoning one abandons the work. Released PocketTTS 3.0.2 is not like that:

    ``generate_audio_stream()`` runs its own generation and decoder threads,
    and abandoning the generator leaves the generation thread running for the
    remainder of the input. Upstream's own change to add cooperative
    cancellation is open rather than merged, and it says in as many words that
    draining the stream to completion was the correct Python-API behaviour
    before it. The model is documented as not thread-safe, so starting the next
    generation while the old one is alive is incorrect (section 5.10).

So this runtime does not claim to cancel. It claims what it can do:

    STOP WHAT I AM HEARING NOW.

Playback goes silent immediately -- that is :meth:`mc_voice_turn.VoiceTurn.cancel`
and it happens before anything here is reached. Then the turn stops accepting
text, its not-yet-started units are dropped, the one native call already inside
Pocket is allowed to finish, and everything it produces is consumed and thrown
away until the worker says the lane is free. Only then may a new Pocket
inference start (I-PKT-11, I-PKT-13, section 21).

The consuming is the part that is easy to get wrong. "Muted" is not "stop
reading": Pocket 3.0.2's internal queues are ordinary unbounded ``queue.Queue``
instances, so an outer reader that stopped reading would not stop Pocket
generating -- it would only make the pipe fill and the drain take *longer*
(I-PKT-12, section 49.3). Every frame is read; the PCM in it is dropped at the
earliest safe point, which is :meth:`VoiceTurn.offer_audio` returning False
without blocking.

Only the unit already executing may drain. A turn with unit 2 inside Pocket,
unit 3 queued and unit 4 not yet committed loses 3 and 4 outright: draining a
whole queued assistant answer would turn a bounded compatibility policy into a
long lockout (section 21.3, section 49.4).

Nothing here is a local copy of upstream's unmerged cancellation. When Kyutai
merges it and this project deliberately adopts a reviewed release,
:data:`INTERRUPT_MODE` becomes ``cooperative``, the worker sets an Event instead
of running to the end, and neither the command below nor the browser's Stop
changes at all (I-PKT-14, section 21.7).

What this module does not do
----------------------------
It never holds a lock while it waits for native inference. It never imports
Torch, PocketTTS, Sopro or sherpa-onnx. It never sends the worker a URL, a
repository id or a credential. And it never reports a path, a PID, an
interpreter or a voice name in anything a browser can read (section 18,
section 36).
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field

import mc_voice_paths as paths
from pocket_worker import worker as protocol

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""


class PocketRuntimeError(RuntimeError):
    """A Pocket request that could not be served. Never fatal to Conversation."""


CONTAINMENT = {"windows": "job", "linux": "pdeathsig"}
"""Which parent-death mechanism each supported platform uses.

A platform absent from here has no tested way of guaranteeing that a hard kill
of the WebUI takes the speech process with it, and this runtime refuses to start
one there rather than starting one it cannot promise to end (section 34).
"""

INTERRUPT_MODE = "drain_unit"
"""What Stop means on this engine, as the capability the shared turn reads.

``drain_unit`` for released PocketTTS 3.0.2, and it is a statement about
upstream rather than a preference: see the module docstring. The *effective*
mode comes from the worker's handshake once one is resident, so that adopting a
merged upstream cancellation is a change to the worker and this constant, and
not a change to Voice Chat.
"""

SUPPORTED_INTERRUPT_MODES = ("drain_unit", "cooperative")
"""What this build's parent-side state machine implements.

A worker declaring anything else is refused at the handshake. That is not
pedantry: the waiting state the browser draws is cleared by an authoritative
worker report, and a mode this parent has no report for would be a waiting
state nothing ever clears (I-PKT-13).
"""

HANDSHAKE_TIMEOUT = 600.0
"""How long a cold PocketTTS load may take before the start is called failed.

Ten minutes, because a first load reads the model off a disk that may be slow
and builds an interpreter's import graph around Torch. A number rather than
forever: a start that will never finish should end as a sentence in Settings
rather than as a Voice panel that spins.
"""

REQUEST_TIMEOUT = 300.0
PREPARE_TIMEOUT = 900.0
"""Preparing a voice is a fifteen-second recording through an encoder and an
export; it is minutes on a slow CPU and it is not a synthesis."""

CATALOG_TIMEOUT = 30.0

STOP_GRACE = 3.0
TERMINATE_GRACE = 1.5

DRAIN_GRACE_HARD = 60.0
"""How long an abandoned Pocket unit may take to finish before the parent stops
believing it will.

A failsafe for abnormal non-quiescence and *not* the normal implementation of
Stop (section 21.8). The normal implementation is: the unit finishes, the worker
says so, and the lane is free -- which for a bounded safe text unit is seconds.
Sixty is generous by design, because terminating a worker throws away a warmed
model and every cached voice state with it, and doing that to a unit that was
about to finish would be a cure worse than the wait.

The real number belongs to GATE P-3, which measures stop-to-ready against real
unit sizes on the release machine. This is the ceiling that keeps a hang from
being permanent, not a measurement.
"""

MAX_TEXT_BYTES = 200_000

CRASH_WINDOW = 60.0
CRASH_LIMIT = 3
"""Three failed starts in a minute stops the fourth. A worker that cannot load
its model will not load it on the tenth attempt either, and a respawn loop
during an image generation is a machine that gets slower for no reason."""


@dataclass
class Handshake:
    """What the worker said it is, once, at ``init``.

    Held rather than re-asked, because these are facts about a process rather
    than about a request -- and because a status route that asked the worker
    would be a status route that could block behind a solver step.
    """

    protocol: int = 0
    engine: str = ""
    backend: str = ""
    pocket_version: str = ""
    upstream_build_id: str = ""
    torch_version: str = ""
    sample_rate: int = 0
    provider: str = ""
    device: str = ""
    containment: str = ""
    model_id: str = ""
    model_fingerprint: str = ""
    quantization: str = ""
    sampler_steps: int = 0
    streaming: bool = False
    interrupt_mode: str = ""
    thread_policy: str = ""
    voice_state_schema: int = 0
    voices: int = 0
    defaults: dict = field(default_factory=dict)


_state_lock = threading.RLock()
_write_lock = threading.Lock()
_start_lock = threading.Lock()

_process = None
_reader: "threading.Thread | None" = None
_handshake: "Handshake | None" = None
_closing = False
_session = ""
_next_id = 0
_generation = 0
"""Incremented every time a worker is started and every time one is discarded,
so a frame already in the pipe cannot be delivered to a request or a turn
belonging to its successor."""

_pending: dict = {}
_turns: dict = {}

_draining: dict = {}
"""Turns whose playback has stopped and whose native unit has not finished.

The whole of the parent's drain bookkeeping, and the reason it is a second map
rather than a flag on ``_turns``: a turn in here is *not* a turn that can
receive audio. It is a record that exists so the dispatcher recognises the
frames still arriving for it, consumes them, throws the PCM away, and knows when
the worker says the lane is free (I-PKT-12, I-PKT-13).

Bounded by construction: only a unit already executing may drain, and there is
one inference lane, so this holds at most one entry.
"""

_lane_free = threading.Event()
_lane_free.set()
"""Whether a new Pocket inference may start.

Cleared when a turn is interrupted and set when the worker reports its native
call returned -- never on a timer. A waiting state that cleared itself would be
a waiting state that let a second generation start while the first was still
alive, which is the one thing a model documented as not thread-safe must never
be asked to do.
"""

_busy = 0
_preparing = 0
_loading = False
_last_error = ""
_failures: list = []
_job_handle = None

_exit_registered = False
_exit_lock = threading.Lock()


def declared_interrupt_mode() -> str:
    """What Stop means on this engine right now.

    The resident worker's answer where there is one, and this build's declared
    default otherwise -- because the adapter is asked for its capabilities long
    before any worker starts, and answering "unknown" would leave the browser
    with nothing to draw.
    """
    with _state_lock:
        if _handshake is not None and _handshake.interrupt_mode:
            return _handshake.interrupt_mode
    return INTERRUPT_MODE


def _busier(step: int) -> None:
    global _busy

    with _state_lock:
        _busy = max(0, _busy + int(step))


def _preparing_by(step: int) -> None:
    global _preparing

    with _state_lock:
        _preparing = max(0, _preparing + int(step))


def status() -> dict:
    """What the panel and the playback control read. Starts nothing.

    ``busy`` is "the one inference lane is occupied" and ``draining`` is "it is
    occupied by work nobody is listening to". Two fields rather than one because
    the sentences are different: the first is why a new turn waits, and the
    second is why the browser says *Voice finishing…* rather than *Speaking*
    (section 24, section 29.3).
    """
    with _state_lock:
        running = _process is not None and _process.poll() is None
        draining = bool(_draining)
        return {
            "loaded": bool(running),
            "busy": bool(_busy or draining or not _lane_free.is_set()),
            "draining": draining,
            "preparing": bool(_preparing),
            "interrupt_mode": (_handshake.interrupt_mode if _handshake and running
                               else INTERRUPT_MODE),
            "error": _last_error,
        }


def engine() -> dict:
    """The live engine state the Pocket panel and the Voice overlay draw.

    Deliberately nothing section 18 forbids: no pid, no command line, no
    filesystem path, no interpreter, no spoken text and no voice name.
    """
    with _state_lock:
        running = _process is not None and _process.poll() is None
        if _loading:
            state = "loading"
        elif _closing:
            state = "stopping"
        elif not running:
            state = "error" if _last_error else "unloaded"
        elif _draining:
            state = "draining"
        elif _preparing:
            state = "preparing"
        elif _busy:
            state = "speaking"
        else:
            state = "idle"
        found = {
            "loaded": bool(running),
            "state": state,
            "backend": "pocket",
            "error": _last_error if state == "error" else "",
        }
        if running and _handshake is not None:
            found.update({
                "device": _handshake.device,
                "provider": _handshake.provider,
                "quantization": _handshake.quantization,
                "sampler_steps": _handshake.sampler_steps,
                "sample_rate": _handshake.sample_rate,
                "voices": _handshake.voices,
                "streaming": _handshake.streaming,
                "interrupt_mode": _handshake.interrupt_mode,
                "thread_policy": _handshake.thread_policy,
                "fingerprint": _handshake.model_fingerprint,
                "defaults": dict(_handshake.defaults),
            })
        return found


def defaults() -> dict:
    """The selected model's own generation defaults, or an empty mapping.

    Read from the handshake rather than repeated in the UI, so a model revision
    that changes its recommended temperature changes what "model default" means
    everywhere at once (I-PKT-25). Empty when no worker has ever started, which
    the panel shows as "the model's own" rather than as a number it made up.
    """
    with _state_lock:
        return dict(_handshake.defaults) if _handshake else {}


def supported_platform() -> bool:
    """Whether this platform has a tested containment mechanism for Pocket."""
    import mc_voice_models as models

    system, _machine, _python = models.current_platform()
    return system in CONTAINMENT


def _expected_containment() -> str:
    """The parent-death mechanism this platform is supposed to use, or ``""``.

    Empty for a platform with no tested mechanism, which makes every answer a
    child could give the wrong one -- the right outcome, since Pocket refuses to
    install there in the first place.
    """
    import mc_voice_models as models

    try:
        system, _machine, _python = models.current_platform()
    except Exception:
        logger.debug("Model Chain: could not identify this platform for PocketTTS",
                     exc_info=True)
        return ""
    return CONTAINMENT.get(system, "")


def _declared_sample_rate() -> int:
    """What the installed Pocket bundle says its model produces, or ``0``.

    Zero for a manifest that does not say, and a comparison against zero is not
    made: a build that has not recorded a rate has nothing to check against, and
    inventing one would be checking the worker against a guess.
    """
    try:
        import mc_voice_pocket as pocket

        return int(pocket.bundle().sample_rate or 0)
    except Exception:
        logger.debug("Model Chain: the PocketTTS bundle declares no sample rate",
                     exc_info=True)
        return 0


def config_line() -> str:
    """One line naming how this worker is configured, for the turn summary.

    Numbers and enumerations only. It exists so a shared log comparing two runs
    can say whether they were the same configuration, which a real-time factor
    on its own cannot.
    """
    with _state_lock:
        found = _handshake
    if found is None:
        return "pocket: not started"
    return (f"pocket: {found.quantization}, {found.sampler_steps} step(s), "
            f"{found.sample_rate} Hz, interrupt={found.interrupt_mode}, "
            f"{found.thread_policy}")


# --------------------------------------------------------------------------- #
# Requests that are not a turn
# --------------------------------------------------------------------------- #


def prepare_voice(root: str, voice_id: str, wav_bytes: bytes, seconds: float = 0.0,
                  audition: str = "") -> dict:
    """Turn a recording into a voice state, write it, read it back, audition it.

    All four in one request, because the four together are what proves a clone
    exists: a preparation that returned without an exception says nothing about
    whether the file it wrote can be read after a restart, and the audition is
    synthesised from the state that was read back off the disk rather than from
    whatever was still in memory (section 26.2, T-PKT-CLONE-4).

    ``root`` is a directory this process built inside Pocket's own tree. The
    worker writes into it and nowhere else.
    """
    _preparing_by(1)
    try:
        header = {"op": "prepare", "root": str(root), "voice_id": str(voice_id),
                  "seconds": float(seconds or 0.0), "audition": str(audition or "")}
        reply, body = _request(header, bytes(wav_bytes or b""), PREPARE_TIMEOUT)
        return {"sample_rate": int(reply.get("sample_rate") or 0),
                "audition_ms": int(reply.get("audition_ms") or 0),
                "state_bytes": int(reply.get("state_bytes") or 0),
                "audio": body}
    finally:
        _preparing_by(-1)


def refresh_catalog(voices: dict, forget=()) -> int:
    """Tell a running worker which voices exist now. Starts nothing.

    A no-op when no worker is resident, which is the ordinary case after a
    delete: the catalogue is part of the config the next start is given, so
    there is nothing to correct.
    """
    with _state_lock:
        if _process is None or _process.poll() is not None:
            return 0
    try:
        reply, _body = _request({"op": "catalog", "voices": dict(voices or {}),
                                 "forget": list(forget or ())}, b"", CATALOG_TIMEOUT)
    except PocketRuntimeError:
        logger.debug("Model Chain: the PocketTTS catalogue could not be refreshed",
                     exc_info=True)
        return 0
    return int(reply.get("voices") or 0)


def synthesize(text: str, voice_id: str = "", profile=None) -> bytes:
    """One complete utterance as a WAV. What Test and the /tts route use.

    The same worker, the same voice state and the same delivery a reply would
    use (section 32) -- an audition down a different path would be an audition
    that could pass for a voice which cannot actually be spoken.

    It waits for the lane like any other inference, so a Test pressed while a
    drain is finishing waits for the drain rather than overlapping it.
    """
    wanted = str(text or "")
    if len(wanted.encode("utf-8")) > MAX_TEXT_BYTES:
        raise PocketRuntimeError("That text is too long to read aloud.")
    _await_lane("that audition")
    _busier(1)
    try:
        header = {"op": "tts", "voice_id": str(voice_id or "")}
        header.update(_delivery(profile))
        _reply, body = _request(header, wanted.encode("utf-8"), REQUEST_TIMEOUT)
        return body
    finally:
        _busier(-1)


def warm(voice_id: str = "", profile=None) -> str:
    """Ask the worker to have this voice's state loaded before it is needed."""
    try:
        header = {"op": "warm", "voice_id": str(voice_id or "")}
        header.update(_delivery(profile))
        reply, _body = _request(header, b"", REQUEST_TIMEOUT)
        return str(reply.get("state") or "")
    except PocketRuntimeError:
        logger.debug("Model Chain: PocketTTS could not warm a voice", exc_info=True)
        return ""


def _delivery(profile) -> dict:
    """A resolved profile as the worker's frame header carries it.

    Through the profile module, so "what does +3 semitones mean" is answered
    once in the process that has the settings. A ``None`` generation field is
    omitted rather than sent as null: the worker reads an absent key as the
    model's own (I-PKT-25).
    """
    try:
        import mc_voice_pocket_profile as profiles

        return profiles.request(profile)
    except Exception:
        logger.debug("Model Chain: could not build the PocketTTS delivery header",
                     exc_info=True)
        return {}


def prepare() -> bool:
    """Start the worker now if it is not running. Returns whether it already was.

    Called from the turn thread while the reply is still being written, so a
    cold load overlaps the model's own generation rather than queueing behind
    it. A failure here is not fatal: :func:`begin_turn` meets the same failure a
    moment later and reports it through the path that exists for it.
    """
    with _state_lock:
        if _process is not None and _process.poll() is None:
            return True
    ensure_started()
    return False


# --------------------------------------------------------------------------- #
# One speaking turn
# --------------------------------------------------------------------------- #


def begin_turn(turn, voice_id: str = "", profile=None) -> int:
    """Open a turn on the worker, and wait for the lane if something is draining.

    The wait is the whole of I-PKT-13 on the parent side. Pocket has one
    inference lane and its model is documented as not thread-safe, so a new
    generation may not start while an abandoned one is still alive -- and
    "may not" has to be enforced by something rather than hoped for. This is
    that something, and it wakes on the turn's own cancellation so a Stop
    pressed while a new turn is waiting does not have to wait for the old one
    too.
    """
    _await_lane("that reply", turn)
    ensure_started()
    with _state_lock:
        _turns[turn.id] = turn
    _busier(1)
    try:
        header = {"op": "tts_begin", "turn": turn.id, "voice_id": str(voice_id or "")}
        header.update(_delivery(profile))
        _write(header, b"")
    except _WorkerGone:
        _release_turn(turn)
        raise PocketRuntimeError("PocketTTS stopped before it could start speaking.") \
            from None
    return _await_rate(turn)


def _await_lane(what: str, turn=None) -> None:
    """Block until no abandoned Pocket unit is still running.

    Bounded, and the bound is a failsafe rather than a policy: if the worker has
    not reported its native call complete within :data:`DRAIN_GRACE_HARD` then
    something is wrong that waiting will not fix, and the parent stops and
    restarts the worker rather than starting a second generation inside a
    process where the first may still be running (section 21.8).
    """
    if _lane_free.is_set():
        return
    deadline = time.monotonic() + DRAIN_GRACE_HARD
    while not _lane_free.wait(0.05):
        if turn is not None and getattr(turn, "cancelled", None) is not None \
                and turn.cancelled.is_set():
            # The turn that was waiting has itself been superseded or stopped.
            # Nothing to start, and nothing to wait for.
            return
        if time.monotonic() > deadline:
            logger.warning("Model Chain: PocketTTS did not finish an interrupted unit "
                           "within %.0f seconds, so its worker was restarted before %s",
                           DRAIN_GRACE_HARD, what)
            stop("PocketTTS did not become quiescent after an interruption")
            return


def _await_rate(turn) -> int:
    """The sample rate for this turn, once the worker has said what it is.

    The worker answers ``tts_ready`` before the first block, so this is a short
    wait rather than a synthesis-long one. It wakes on cancellation, because a
    turn stopped between ``begin`` and ``ready`` must not hold this thread.
    """
    deadline = time.monotonic() + REQUEST_TIMEOUT
    while not turn.cancelled.is_set():
        if turn.sample_rate:
            return int(turn.sample_rate)
        with _state_lock:
            found = _handshake
        if found is not None and found.sample_rate:
            return int(found.sample_rate)
        if time.monotonic() > deadline:
            break
        time.sleep(0.02)
    with _state_lock:
        return int(_handshake.sample_rate) if _handshake else 0


def send_segment(turn, text: str) -> None:
    """One committed safe unit, to be synthesised as one native call.

    One unit per call in V1, deliberately (I-PKT-29, section 20.1). Pocket's own
    long-text splitter exists and upstream calls it "very simplistic"; the
    segmenter that decides what incomplete assistant text is safe to speak is
    this repository's and stays this repository's. How many safe units are
    coalesced into one native call is a separate, *measured* policy -- GATE P-2B
    -- and until it has been measured the answer is one, because the size of the
    unit in flight is exactly what a Stop has to wait for (section 21.3).
    """
    wanted = str(text or "")
    if not wanted:
        return
    if len(wanted.encode("utf-8")) > MAX_TEXT_BYTES:
        raise PocketRuntimeError("That reply is too long to read aloud.")
    with _state_lock:
        if turn.id in _draining:
            # Refused rather than queued. A unit sent to an interrupted turn is
            # a unit that would extend a drain nobody is listening to.
            return
    try:
        _write({"op": "tts_text", "turn": turn.id}, wanted.encode("utf-8"))
    except _WorkerGone:
        turn.audio_failed("PocketTTS stopped while it was speaking.")
        _release_turn(turn)


def finish_turn(turn) -> None:
    try:
        _write({"op": "tts_end", "turn": turn.id}, b"")
    except _WorkerGone:
        turn.audio_failed("PocketTTS stopped while it was speaking.")
        _release_turn(turn)


def cancel_turn(turn) -> None:
    """The shared name, kept, and on this engine it means :func:`interrupt_turn`.

    Present so that a caller written before interruption was a capability still
    reaches the right behaviour. It does not mean cancellation here and does not
    pretend to.
    """
    interrupt_turn(turn)


def interrupt_turn(turn) -> None:
    """Stop this turn being heard, and let its in-flight unit finish silently.

    Returns immediately. Stop must never wait for native compute -- the browser
    is already silent by the time this is reached, and a Stop that blocked for
    the length of a solver would be a Stop that felt broken.

    What it arranges, in order:

        1  the turn moves out of ``_turns`` and into ``_draining``, so no later
           frame for it can reach playback and every frame for it is still
           read;
        2  the lane is marked occupied, so nothing new may start;
        3  ``tts_interrupt`` goes down the pipe, which tells the worker to drop
           this turn's not-yet-started units and to stop offering the ones the
           current call is still producing;
        4  the turn is told it is draining, so the surface can say so.

    The record is cleared and the lane released only by an authoritative report
    from the worker -- ``tts_interrupted`` with ``state="complete"``, or
    ``tts_done``, or the pipe ending. Never by a timer (I-PKT-13).
    """
    identifier = getattr(turn, "id", "")
    if getattr(turn, "synthesis_done", False):
        # The reply finished between the turn thread deciding to interrupt and
        # this call. Nothing is in the model, nothing more will arrive for this
        # turn, and a drain record for it would be a record the worker has no
        # reason to ever close -- so the lane would stay held until the failsafe
        # rather than until the next sentence. Narrow, and worth a line: it is
        # a Stop pressed on the last word.
        with _state_lock:
            _turns.pop(identifier, None)
        return
    with _state_lock:
        known = _turns.pop(identifier, None)
        if known is None and identifier not in _draining:
            # A turn this runtime never opened, or one already drained. Both are
            # ordinary: Stop is idempotent and the browser sends what it has.
            return
        _draining[identifier] = {"turn": turn, "since": time.monotonic()}
        _lane_free.clear()
    if known is not None:
        _busier(-1)
    try:
        # Before the write, and that ordering is the whole of it. The worker
        # answers ``state="complete"`` synchronously when nothing was inside
        # the model -- a Stop between units, or on a turn the lane had not
        # picked up yet -- so the reader thread can run :func:`_finish_drain`
        # before this line would otherwise have been reached. ``interrupted()``
        # returns early on a turn that was never marked draining, and this call
        # would then set it draining *after* the report that clears it, leaving
        # a "Voice finishing..." nothing would ever take down.
        turn.interrupting()
    except Exception:
        logger.debug("Model Chain: could not mark a PocketTTS turn as draining",
                     exc_info=True)
    try:
        _write({"op": "tts_interrupt", "turn": identifier}, b"")
    except _WorkerGone:
        # The worker is gone, so its lane is not occupied by anything.
        _finish_drain(identifier)
        return
    logger.info("Model Chain: PocketTTS was interrupted — playback is silent and the unit "
                "already inside the model is being drained")


def _finish_drain(identifier: str, header: dict = None) -> None:
    """The worker says its lane is free. The only place that clears the wait.

    ``header`` is the frame that said so, and it carries how big the abandoned
    unit was. Read here rather than dropped, because the release envelope is
    measured in exactly those two numbers (section 43): "a Stop cost 4.2
    seconds" is only a finding if something also recorded that the unit was
    seventy-eight characters.
    """
    with _state_lock:
        record = _draining.pop(str(identifier or ""), None)
        free = not _draining
    if free:
        _lane_free.set()
    if record is None:
        return
    turn = record.get("turn")
    found = dict(header or {})
    try:
        if turn is not None:
            if found.get("chars") is not None or found.get("audio_ms") is not None:
                turn.interrupting(chars=found.get("chars"), audio_ms=found.get("audio_ms"))
            turn.interrupted()
    except Exception:
        logger.debug("Model Chain: could not clear a PocketTTS drain state", exc_info=True)
    logger.info("Model Chain: PocketTTS finished its interrupted unit after %.1f s and is "
                "ready again", max(0.0, time.monotonic() - float(record.get("since") or 0)))


def _release_turn(turn) -> None:
    with _state_lock:
        _turns.pop(getattr(turn, "id", ""), None)
    _busier(-1)


# --------------------------------------------------------------------------- #
# One round trip
# --------------------------------------------------------------------------- #


def _request(header: dict, payload: bytes, timeout: float):
    """One round trip, with exactly one restart if the worker had died.

    "Exactly one" is the whole policy. A worker that vanished between two
    auditions is an ordinary thing; making the user press the button again for
    it would be unkind. A worker that vanishes again on the retry is a broken
    installation, and the honest answer to that is an error rather than a third
    process.

    Note what this does *not* do: hold a lock while it waits. Every wait in this
    module is on a queue or an event that the caller owns, which is what lets
    Stop, a status poll and an engine switch all reach a runtime that is inside
    a native call.
    """
    for attempt in (1, 2):
        with _state_lock:
            if _closing:
                raise PocketRuntimeError("Voice Chat is shutting down.")
        ensure_started()
        try:
            return _exchange(header, payload, timeout)
        except _WorkerGone:
            stop("the PocketTTS worker stopped unexpectedly")
            if attempt == 2:
                raise PocketRuntimeError("PocketTTS stopped unexpectedly. Try again.") \
                    from None
    raise PocketRuntimeError("PocketTTS stopped unexpectedly. Try again.")


class _WorkerGone(Exception):
    """The pipe ended or the worker did not answer in time."""


def _exchange(header: dict, payload: bytes, timeout: float):
    """Write one frame and wait for the reply that carries its id.

    The waiting is on a queue this call owns, so two callers cannot receive one
    another's answers even in principle.
    """
    global _next_id

    with _state_lock:
        _next_id += 1
        request_id = _next_id
        answers: queue.Queue = queue.Queue(maxsize=1)
        _pending[request_id] = answers
    try:
        message = dict(header)
        message["id"] = request_id
        _write(message, payload)
        try:
            found = answers.get(timeout=timeout)
        except queue.Empty:
            raise _WorkerGone("the PocketTTS worker did not answer in time") from None
        if found is None:
            raise _WorkerGone("the PocketTTS worker stopped")
        reply, body = found
        if not reply.get("ok", True):
            raise PocketRuntimeError(_readable(str(reply.get("error") or "")))
        return reply, body
    finally:
        with _state_lock:
            _pending.pop(request_id, None)


def _write(header: dict, payload: bytes) -> None:
    with _state_lock:
        started = _process
    if started is None or started.poll() is not None or started.stdin is None:
        raise _WorkerGone("the PocketTTS worker is not running")
    try:
        with _write_lock:
            protocol.write_frame(started.stdin, header, payload)
    except Exception as exc:
        raise _WorkerGone(str(exc)) from None


def _readable(reason: str) -> str:
    """A worker's own words as something a user can act on.

    The worker sends an exception *class* for anything it did not raise itself,
    because a raw ``repr`` of a library exception can carry a path, a tensor
    shape or the text that was being spoken (I-PKT-27). So this turns the
    handful it does send into sentences and leaves everything else as a general
    failure rather than showing somebody ``RuntimeError``.
    """
    text = str(reason or "").strip()
    known = {
        "": "PocketTTS could not complete that request.",
        "RuntimeError": "PocketTTS could not complete that request.",
        "OSError": "PocketTTS could not read one of its own files.",
        "FileNotFoundError": "One of PocketTTS's files is missing. Reinstall it in "
                             "Settings → Voice Chat.",
        "MemoryError": "There was not enough memory to run PocketTTS.",
    }
    if text in known:
        return known[text]
    # A ValueError is this feature's own refusal and is already a sentence.
    return text if " " in text else known[""]


# --------------------------------------------------------------------------- #
# Starting
# --------------------------------------------------------------------------- #


def ensure_started() -> None:
    """Start the worker if it is not running. Idempotent.

    Every failure path is written so that a process which has been started is
    stopped before its handle is dropped: ownership begins at ``Popen``, not at
    the moment the handshake succeeds.

    Refuses outright when Pocket is not the selected engine. One active TTS
    worker at a time is the product rule, and a stale request from a page drawn
    before somebody switched must not be able to load a Torch runtime behind
    their back.
    """
    global _process, _reader, _handshake, _session, _generation, _loading, _last_error

    with _state_lock:
        if _process is not None and _process.poll() is None:
            return

    import mc_voice_engines as engines
    import mc_voice_pocket as pocket

    if engines.active() != engines.POCKET:
        raise PocketRuntimeError(
            "PocketTTS is not the selected text-to-speech engine, so it was not started.")

    with _start_lock:
        with _state_lock:
            if _process is not None and _process.poll() is None:
                return
            if _process is not None:
                _discard("a previous PocketTTS worker had already exited")
        _guard_crash_loop()

        state = pocket.status()
        if not state.ready:
            raise PocketRuntimeError(
                "PocketTTS is not installed. Install it in Settings → Voice Chat.")
        if not supported_platform():
            raise PocketRuntimeError(
                "PocketTTS has no tested process-containment mechanism on this platform, "
                "so it will not start a speech process here.")
        interpreter = pocket.runtime_python()
        if interpreter is None:
            raise PocketRuntimeError("The PocketTTS runtime is not installed.")

        session = uuid.uuid4().hex[:12]
        command = [str(interpreter), str(paths.pocket_worker_script()), protocol.MARKER,
                   "--parent-pid", str(os.getpid()), "--session", session]
        environ = dict(os.environ)
        # Pocket's own environment *last*, so an HF_TOKEN or a CUDA variable
        # inherited from the WebUI cannot survive into the worker: the mapping
        # below sets the GPU variables empty and carries no credential at all
        # (I-PKT-7, I-PKT-21).
        environ.update(pocket.worker_environment())
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            environ.pop(name, None)

        started = None
        with _state_lock:
            _loading = True
            _last_error = ""
            _session = session
        try:
            started = subprocess.Popen(  # noqa: S603 - a path this module built
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=environ, cwd=str(paths.extension_root()),
                bufsize=0, close_fds=True)
            _die_with_us(started)
            with _state_lock:
                _process = started
                _generation += 1
                mine = _generation
                _pending.clear()
                _turns.clear()
                _draining.clear()
            _lane_free.set()
            _reader = threading.Thread(target=_read_frames, args=(started, mine),
                                       name="mc-pocket-reader", daemon=True)
            _reader.start()
            threading.Thread(target=_drain_stderr, args=(started,),
                             name="mc-pocket-stderr", daemon=True).start()
            _handshake = _handshake_with(state)
        except Exception as exc:
            # Whatever went wrong, the process this function started is this
            # function's to end. Nothing below this line may leave a handle.
            _note_failure()
            with _state_lock:
                if started is not None:
                    _process = started
                    _discard("the PocketTTS worker failed to start")
                _handshake = None
                _last_error = str(exc) if isinstance(exc, PocketRuntimeError) else ""
            if isinstance(exc, PocketRuntimeError):
                raise
            logger.warning("Model Chain: the PocketTTS worker could not be started",
                           exc_info=True)
            if isinstance(exc, _WorkerGone):
                raise PocketRuntimeError(
                    "PocketTTS did not finish starting, so it was stopped. Try again.") \
                    from None
            raise PocketRuntimeError(
                "PocketTTS could not be started. Check Settings → Voice Chat.") from None
        finally:
            with _state_lock:
                _loading = False

        stop_on_exit()
        logger.info("Model Chain: PocketTTS ready — %s, %s, %d step(s), %d Hz, "
                    "interrupt=%s, %s", _handshake.pocket_version, _handshake.quantization,
                    _handshake.sampler_steps, _handshake.sample_rate,
                    _handshake.interrupt_mode, _handshake.thread_policy)


def _handshake_with(state) -> Handshake:
    """Ask the worker what it is, and refuse an answer this build cannot accept.

    Eight refusals, each with its own sentence, because "PocketTTS failed to
    start" is not a diagnosable state (section 18):

        a protocol this build does not speak;
        an engine or backend that is not Pocket's native runtime;
        a provider or device that is not the CPU, which this release supports
            only (I-PKT-7) -- and a worker that found a GPU is a worker whose
            memory this WebUI's image generation was counting on;
        no containment evidence where the parent could not arrange it itself;
        a sample rate that is not the one the installed bundle declares, or is
            outside what the browser's scheduler was built for;
        a model fingerprint that is not the one on disk, which means a saved
            voice state would be loaded into a model it was not prepared for
            (I-PKT-18);
        an interrupt mode this parent has no state machine for, which would be
            a waiting state nothing ever clears (I-PKT-13).
    """
    import mc_voice_pocket as pocket

    reply, _body = _exchange({"op": "init", "parent_pid": os.getpid(),
                              "session": _session, "config": pocket.worker_config()},
                             b"", HANDSHAKE_TIMEOUT)
    found = Handshake(
        protocol=int(reply.get("protocol") or 0),
        engine=str(reply.get("engine") or ""),
        backend=str(reply.get("backend") or ""),
        pocket_version=str(reply.get("pocket_version") or ""),
        upstream_build_id=str(reply.get("upstream_build_id") or ""),
        torch_version=str(reply.get("torch_version") or ""),
        sample_rate=int(reply.get("sample_rate") or 0),
        provider=str(reply.get("provider") or ""),
        device=str(reply.get("device") or ""),
        containment=str(reply.get("containment") or ""),
        model_id=str(reply.get("model_id") or ""),
        model_fingerprint=str(reply.get("model_fingerprint") or ""),
        quantization=str(reply.get("quantization") or ""),
        sampler_steps=int(reply.get("sampler_steps") or 0),
        streaming=bool(reply.get("streaming")),
        interrupt_mode=str(reply.get("interrupt_mode") or ""),
        thread_policy=str(reply.get("thread_policy") or ""),
        voice_state_schema=int(reply.get("voice_state_schema") or 0),
        voices=int(reply.get("voices") or 0),
        defaults=dict(reply.get("defaults") or {}))

    if found.protocol != protocol.PROTOCOL_VERSION:
        raise PocketRuntimeError(
            f"The PocketTTS worker speaks protocol {found.protocol} and this build speaks "
            f"{protocol.PROTOCOL_VERSION}. Reinstall PocketTTS in Settings → Voice Chat.")
    if found.engine != "pocket" or found.backend != pocket.BACKEND:
        raise PocketRuntimeError(
            "The PocketTTS worker did not identify itself as PocketTTS, so it was stopped.")
    if found.provider != "cpu" or found.device != "cpu":
        raise PocketRuntimeError(
            f"The PocketTTS worker reported the device {found.device or found.provider!r} "
            f"rather than the CPU. This release supports PocketTTS on the CPU only, so it "
            f"was stopped.")
    if os.name != "nt" and found.containment != _expected_containment():
        # On Windows the parent arranged the job object itself and verified it
        # with real handles, so the child's own answer is diagnostic. On every
        # other platform the mechanism has to run *inside* the child between
        # fork and the first request, and the child's confirmation is the only
        # evidence there is.
        #
        # Compared against *this* platform's mechanism and not against the set
        # of all of them: a Linux child that answered "job" would have been
        # accepted by a membership test, and "job" on Linux is not a mechanism
        # that exists -- it is a child claiming a containment nothing arranged.
        raise PocketRuntimeError(
            "The PocketTTS worker could not confirm that it will be ended if this WebUI "
            "stops, so it was stopped now instead.")
    declared = _declared_sample_rate()
    if declared and found.sample_rate != declared:
        # Against the rate the installed bundle declares, rather than against a
        # range the browser could play. A model revision that changed its output
        # rate is exactly the case this catches, and it is not audible as a
        # failure -- it is audible as a voice pitched wrong, because every voice
        # state and every delivery ratio downstream was computed for the
        # declared rate (I-PKT-18).
        raise PocketRuntimeError(
            f"The PocketTTS worker reported a sample rate of {found.sample_rate} Hz and "
            f"the installed model declares {declared} Hz, so it was stopped. Reinstall "
            f"PocketTTS in Settings → Voice Chat.")
    if found.sample_rate <= 0 or found.sample_rate > 96000:
        raise PocketRuntimeError(
            f"The PocketTTS worker reported a sample rate of {found.sample_rate} Hz, which "
            f"is not one Voice Chat can play.")
    if state.fingerprint and found.model_fingerprint \
            and found.model_fingerprint != state.fingerprint:
        raise PocketRuntimeError(
            "The PocketTTS worker loaded a different model from the one this build has "
            "recorded, so it was stopped. Reinstall PocketTTS in Settings → Voice Chat.")
    if found.interrupt_mode not in SUPPORTED_INTERRUPT_MODES:
        raise PocketRuntimeError(
            f"The PocketTTS worker declared the interrupt mode "
            f"{found.interrupt_mode or 'nothing'!r}, which this build does not implement. "
            f"It was stopped rather than left in a state Stop could not end.")
    if not found.streaming:
        # Not fatal. Streaming is measured behaviour rather than a method name
        # (I-PKT-9), and a worker that cannot stream still speaks -- it just
        # speaks a unit at a time with the whole unit's latency in front of it,
        # and the turn summary should say so rather than the panel claiming a
        # property nobody observed.
        logger.info("Model Chain: the PocketTTS worker reports that it is not streaming; "
                    "first audio will arrive when each unit is complete")
    return found


def load() -> dict:
    """Start the worker deliberately, from the Voice flyout's Load control."""
    ensure_started()
    return engine()


def unload(reason: str = "unloaded") -> dict:
    """Stop the worker deliberately, and wake everything waiting on it.

    A lifecycle operation, not a Stop: it does not wait for a drain. Whatever
    was being spoken is cancelled, whatever was draining is abandoned with its
    process, and the lane is free because there is no longer a process holding
    it (section 21.6).
    """
    with _state_lock:
        if _process is None:
            return engine()
        _discard(reason)
    return engine()


# --------------------------------------------------------------------------- #
# The one reader
# --------------------------------------------------------------------------- #


def _read_frames(started, generation: int) -> None:
    """The one reader. Ordinary replies to their request, speech to its turn.

    It never stops reading for a turn that has been interrupted, which is the
    parent's half of I-PKT-12: Pocket's internal queues are unbounded, so a
    reader that stopped would not stop Pocket generating -- it would only make
    the pipe fill and the drain take longer.
    """
    stream = started.stdout
    try:
        while True:
            frame = protocol.read_frame(stream)
            if frame is None:
                break
            header, payload = frame
            with _state_lock:
                if generation != _generation:
                    # This worker has been replaced. Everything still in its
                    # pipe belongs to a conversation that is over.
                    break
            operation = str(header.get("op") or "")
            if operation.startswith("tts_") and header.get("turn") is not None:
                _dispatch_turn(operation, header, payload)
                continue
            with _state_lock:
                answers = _pending.get(header.get("id"))
            if answers is not None:
                answers.put((header, payload))
    except Exception:
        logger.debug("Model Chain: the PocketTTS worker's pipe ended", exc_info=True)
    finally:
        _fail_everything(generation)


def _dispatch_turn(operation: str, header: dict, payload: bytes) -> None:
    """One streaming frame to the turn it belongs to, or to the drain, or nowhere.

    Three destinations and the middle one is the whole point. A turn in
    ``_turns`` is being listened to and gets its audio. A turn in ``_draining``
    is not, and its frames are read and dropped here -- at the earliest safe
    point, so nothing downstream can apply backpressure to a unit that has to be
    allowed to finish. A turn in neither has been forgotten entirely, and its
    frames are dropped at this single point rather than at each destination.
    """
    identifier = str(header.get("turn") or "")
    with _state_lock:
        turn = _turns.get(identifier)
        draining = identifier in _draining

    if draining:
        if operation in ("tts_interrupted", "tts_done", "tts_error"):
            state = str(header.get("state") or "")
            if operation == "tts_interrupted" and state == "draining":
                # A heartbeat. The worker is still inside the abandoned call and
                # is saying so, which is what keeps a bounded wait honest rather
                # than a guess.
                return
            _finish_drain(identifier, header)
        # Everything else for a draining turn -- audio above all -- is consumed
        # and discarded. Read, and dropped: I-PKT-12 in one branch.
        return

    if turn is None:
        return

    if operation == "tts_audio":
        turn.offer_audio(payload, int(header.get("sample_rate") or 0))
    elif operation == "tts_ready":
        turn.sample_rate = int(header.get("sample_rate") or 0) or turn.sample_rate
        turn.streaming = str(header.get("streaming") or "") or turn.streaming
    elif operation == "tts_segment":
        note = getattr(turn, "note_segment", None)
        if note is not None:
            # Wrapped, because this is the one reader thread: a destination that
            # cannot take a new field would otherwise end the pipe for every
            # frame after it, and a measurement is never worth the audio it was
            # measuring.
            try:
                note(blocks=int(header.get("blocks") or 0),
                     first_block_ms=int(header.get("first_block_ms") or 0),
                     synth_ms=int(header.get("synth_ms") or 0),
                     audio_ms=int(header.get("audio_ms") or 0),
                     trimmed_ms=int(header.get("trimmed_ms") or 0),
                     streaming=str(header.get("streaming") or ""))
            except Exception:
                logger.debug("Model Chain: a PocketTTS unit's timing could not be recorded",
                             exc_info=True)
    elif operation == "tts_interrupted":
        # The worker acknowledged an interrupt for a turn the parent has not
        # moved yet, which is possible when the worker interrupted itself --
        # a fatal error inside a unit, say. Treat it as the parent's own
        # interruption so the lane bookkeeping stays true.
        with _state_lock:
            _turns.pop(identifier, None)
            _draining[identifier] = {"turn": turn, "since": time.monotonic()}
            _lane_free.clear()
        _busier(-1)
        if str(header.get("state") or "") == "complete":
            _finish_drain(identifier, header)
    elif operation == "tts_done":
        turn.audio_finished()
        _release_turn(turn)
    elif operation == "tts_error":
        turn.audio_failed(_readable(str(header.get("error") or "")))
        _release_turn(turn)


def _fail_everything(generation: int) -> None:
    """The worker's pipe ended. Wake every waiter rather than leaving them.

    The drain records go too, and the lane is released: a lane held by a process
    that no longer exists is a lane nothing would ever free.
    """
    with _state_lock:
        if generation != _generation and _process is not None:
            return
        waiting = list(_pending.values())
        speaking = list(_turns.values())
        draining = [record.get("turn") for record in _draining.values()]
        _turns.clear()
        _draining.clear()
    _lane_free.set()
    for answers in waiting:
        try:
            answers.put(None)
        except Exception:
            pass
    for turn in speaking:
        try:
            turn.audio_failed("PocketTTS stopped while it was speaking.")
        except Exception:
            pass
    for turn in draining:
        try:
            if turn is not None:
                turn.interrupted()
        except Exception:
            pass


def _drain_stderr(started) -> None:
    """Log the worker's own diagnostics, which never contain content.

    Read rather than ignored for a plain reason: a pipe nobody reads fills, and
    a child blocked writing to a full stderr is a child that has stopped
    answering for a reason nothing in the parent can see.
    """
    try:
        for line in iter(started.stderr.readline, b""):
            text = line.decode("utf-8", "replace").strip()
            if text:
                logger.info("Model Chain: %s", text)
    except Exception:
        pass


def _guard_crash_loop() -> None:
    now = time.monotonic()
    recent = [when for when in _failures if now - when < CRASH_WINDOW]
    _failures[:] = recent
    if len(recent) >= CRASH_LIMIT:
        raise PocketRuntimeError(
            "PocketTTS has failed to start several times in a row, so it is not being "
            "started again for now. Check Settings → Voice Chat.")


def _note_failure() -> None:
    _failures.append(time.monotonic())


# --------------------------------------------------------------------------- #
# Stopping
# --------------------------------------------------------------------------- #


def stop(reason: str = "") -> None:
    """Stop the worker if one is running. Idempotent, and never raises.

    A lifecycle operation. It does not wait for a drain and it is not the
    ordinary Stop contract: engine switch, settings restart, unload and the
    drain failsafe all come through here, and every one of them is a decision to
    replace the runtime rather than to stop listening to it (section 21.6).

    Reachable while another thread is inside a native call, which is only true
    because nothing waiting holds ``_state_lock``.

    Including this. ``_state_lock`` is re-entrant, so calling :func:`_discard`
    from inside it would have kept it held across the whole bounded escalation
    -- ask, close, wait, terminate, wait, kill -- which is several seconds on
    the one engine where a worker asked to stop is *expected* not to answer
    promptly. A status poll landing in that window would have blocked on it, so
    an unload would have looked like a hang. The check is under the lock and the
    teardown is outside it; two callers racing is harmless, because
    :func:`_discard` takes the handle under the lock and the loser finds none.
    """
    with _state_lock:
        running = _process is not None
    if not running:
        return
    _discard(reason or "PocketTTS stopped")


def shutdown() -> None:
    """Door A and the body of doors B and C. Idempotent; must never raise.

    Called from Forge's script-unload callback, from ``atexit``, and from the
    chained signal handlers. Everything inside is swallowed: a WebUI that will
    not close because a speech process is thinking is a worse bug than any this
    could be reporting -- and on this engine "thinking" is the ordinary state
    after a Stop, so a shutdown that waited for quiescence would be a shutdown
    that hung on the feature's own workaround.
    """
    global _closing

    try:
        try:
            import mc_voice_turn as turns

            turns.forget_all("shutdown")
        except Exception:
            pass
        try:
            import mc_voice_pocket as pocket

            # A voice built but never kept. Its directory holds a recording of
            # somebody, and nothing in the registry points at it -- so if it is
            # not removed here, nothing ever removes it. The decision was
            # "not yet", and a WebUI that closed on "not yet" has answered it.
            pocket.discard_preview()
        except Exception:
            pass
        with _state_lock:
            _closing = True
            running = _process is not None
        # Outside the lock, for the reason :func:`stop` sets out: the escalation
        # is bounded but not instant, and holding the state lock across it makes
        # every reader of that state wait for a process to die.
        if running:
            _discard("WebUI shutdown")
    except Exception:
        logger.debug("Model Chain: the PocketTTS shutdown hook failed", exc_info=True)
    finally:
        _closing = False


def _discard(reason: str) -> None:
    """End the worker and let go of the handle, in that order.

    The escalation is bounded at every step, because the thing being stopped may
    be inside a generation that will not look at a pipe again: ask, close the
    pipe, wait three seconds, terminate, wait, kill. None of the steps is "wait
    for it" -- which matters more here than on the other two engines, since a
    Pocket worker asked to stop during a drain is a worker that is *expected*
    not to answer promptly.
    """
    global _process, _reader, _handshake, _generation, _busy, _preparing

    with _state_lock:
        started, _process, _reader, _handshake = _process, None, None, None
        _generation += 1
        waiting = list(_pending.values())
        speaking = list(_turns.values())
        draining = [record.get("turn") for record in _draining.values()]
        _pending.clear()
        _turns.clear()
        _draining.clear()
        _busy = 0
        _preparing = 0
    _lane_free.set()
    for turn in speaking:
        try:
            turn.cancel("unloaded")
            turn.drain_audio()
        except Exception:
            pass
    for turn in draining:
        try:
            if turn is not None:
                turn.interrupted()
        except Exception:
            pass
    for answers in waiting:
        try:
            answers.put(None)
        except Exception:
            pass
    if started is None:
        return

    if started.poll() is None:
        try:
            with _write_lock:
                protocol.write_frame(started.stdin, {"op": "shutdown", "id": 0})
        except Exception:
            pass
    try:
        if started.stdin is not None:
            started.stdin.close()
    except Exception:
        pass

    if not _wait(started, STOP_GRACE):
        try:
            started.terminate()
        except Exception:
            pass
        if not _wait(started, TERMINATE_GRACE):
            try:
                started.kill()
            except Exception:
                pass
            _wait(started, TERMINATE_GRACE)

    for stream in (started.stdout, started.stderr):
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass
    logger.info("Model Chain: the PocketTTS worker stopped — %s", reason or "no reason given")


def _wait(started, seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        if started.poll() is not None:
            return True
        time.sleep(0.05)
    return started.poll() is not None


# --------------------------------------------------------------------------- #
# The five doors
# --------------------------------------------------------------------------- #


def stop_on_exit() -> None:
    """Doors B and C, arranged once, the first time a worker starts.

    Registered lazily rather than at import: an installation that never selects
    PocketTTS should not have this module's handlers on its interpreter, and a
    handler chained for a process that never existed is a handler nobody can
    account for.
    """
    global _exit_registered

    with _exit_lock:
        if _exit_registered:
            return
        _exit_registered = True
    import atexit

    atexit.register(_at_exit)
    for name in ("SIGINT", "SIGTERM"):
        _relay_signal(name)


def _at_exit() -> None:
    try:
        shutdown()
    except Exception:
        pass


def _relay_signal(name: str) -> None:
    """Chain a signal handler rather than replacing one.

    Forge installs its own, and a handler that replaced it would be a Ctrl-C
    that stopped a speech process and left the WebUI running.
    """
    number = getattr(signal, name, None)
    if number is None:
        return
    try:
        previous = signal.getsignal(number)
    except (OSError, ValueError):
        return

    def handler(received, frame):
        _at_exit()
        if callable(previous) and previous not in (signal.SIG_IGN, signal.SIG_DFL):
            previous(received, frame)
        elif previous == signal.SIG_DFL:
            signal.signal(received, signal.SIG_DFL)
            signal.raise_signal(received)

    try:
        signal.signal(number, handler)
    except (OSError, ValueError):
        logger.debug("Model Chain: could not chain the %s handler for PocketTTS", name,
                     exc_info=True)


JOB_KILL_ON_CLOSE = 0x00002000
JOB_EXTENDED_LIMIT_INFORMATION = 9


def _die_with_us(started) -> None:
    """Door E on Windows: the job object, arranged before the worker gets work.

    A job of Pocket's own rather than the one :mod:`mc_voice_runtime` or
    :mod:`mc_voice_sopro_runtime` holds, and duplicated for the reason those
    modules give for duplicating it: a neutral helper would mean editing two
    proven shutdown paths in order to add a third engine.

    The handle is held for the life of this process on purpose. What does the
    work is the handle being *closed*, which happens when this process ends
    however it ends -- including the kill that runs no handler at all, and
    including a kill that lands while Pocket is inside a drain.

    On Linux the equivalent has to run *inside* the child, between fork and the
    first request, so it lives in ``pocket_worker/worker.py`` and is reported
    back through the handshake. Nothing is arranged here for that platform, and
    :func:`_handshake_with` refuses to run without the child's confirmation.
    """
    global _job_handle

    if os.name != "nt":
        return
    handle = getattr(started, "_handle", None)
    if handle is None:
        raise PocketRuntimeError("The PocketTTS worker could not be tied to this process.")
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    # Declared rather than left to ctypes' defaults. Without argtypes every
    # argument is marshalled as a C ``int``, and a HANDLE on 64-bit Windows is
    # not one: the check below passes a real handle and reads a BOOL out by
    # pointer, which is exactly the shape that does not survive it.
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                               ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
                                      ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    if _job_handle is None:
        job = kernel.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())

        class _Limits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _Limits),
                        ("IoInfo", ctypes.c_byte * 48),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        information = _Extended()
        information.BasicLimitInformation.LimitFlags = JOB_KILL_ON_CLOSE
        if not kernel.SetInformationJobObject(
                job, JOB_EXTENDED_LIMIT_INFORMATION, ctypes.byref(information),
                ctypes.sizeof(information)):
            raise ctypes.WinError(ctypes.get_last_error())
        _job_handle = job
    if not kernel.AssignProcessToJobObject(_job_handle, wintypes.HANDLE(int(handle))):
        raise ctypes.WinError(ctypes.get_last_error())

    # And then ask the kernel whether it took, here, where both handles are real
    # ones and the question can name *this* job rather than "any job". The
    # child's own answer to the same question cannot: a process can only ask
    # whether it is in *some* job, which on a machine where the WebUI is already
    # inside one is true before this function runs.
    inside = wintypes.BOOL(0)
    if not kernel.IsProcessInJob(wintypes.HANDLE(int(handle)), _job_handle,
                                 ctypes.byref(inside)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not inside.value:
        raise PocketRuntimeError(
            "The PocketTTS worker could not be tied to this process, so it was not "
            "started. A speech process that outlives the WebUI is worse than no speech.")
