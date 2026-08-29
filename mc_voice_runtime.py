"""One Voice Worker, owned properly, and stopped however the WebUI ends.

The repository has already paid for the lesson this module is built around. A
llama-server that worked perfectly while the WebUI ran and was still holding
twenty gigabytes of VRAM after it exited was not a finished feature, and the fix
(commit ``fac7ec8``) needed four doors, not one: the extension-unload callback
nobody actually triggers, ``atexit`` for the ordinary exits, chained SIGINT and
SIGTERM handlers, and -- for the kill that runs no handler at all -- a Windows
job object arranged *before* the kill.

Voice Chat copies the principles and shares no code with that fix. Importing
``mc_llm_runtime`` to reuse one helper would give the speech feature a
dependency on the language-model runtime, and invariant I-3 exists to stop
exactly that kind of quiet coupling. Nothing here imports ``mc_memory``,
``mc_broker`` or ``mc_plan`` either: voice runs *beside* Forge rather than
negotiating with it, so there is no residency to publish, no VRAM to make room
for, and nothing to wait for the broker about.

The five doors
--------------
A  Forge extension unload      ``scripts/model_chain.py`` -> :func:`shutdown`
B  ordinary interpreter exit   ``atexit``, armed by the first start
C  SIGINT / SIGTERM            chained in front of whatever was there
D  pipe EOF                    the worker's own loop ends when its input closes
E  OS parent-death             a Windows job object with KILL_ON_JOB_CLOSE, and
                               ``PR_SET_PDEATHSIG`` inside the child on Linux

D and E are two layers of the same requirement and neither is redundant. EOF is
noticed when the child is *waiting to read*, which it is not while it is inside
native ONNX code -- so the hard-kill claim rests on E. That is why
:data:`CONTAINMENT` is a support contract rather than a bonus: on a platform
listed there, a worker that comes back from its handshake without the expected
containment is stopped and the start fails, because a Voice Chat that works
until somebody kills the WebUI is the bug this module was written to not have.

Warm, lazy, and serialized
--------------------------
Nothing starts because the extension imported. The first valid transcription or
synthesis starts the worker, and after that the models stay resident: reading
four hundred megabytes of ONNX between letting go of the microphone and seeing
the words is the latency this feature would be judged on. Inference is
serialized *inside the worker*, in its one lane, which is a statement about the
worker rather than about Forge: an image generation and a transcription run at
the same time and neither waits for the other.

Two locks, and why it used to be one
------------------------------------
V1 had a single ``RLock`` held around a whole request -- start the worker, send
the frame, wait for the reply. That was correct and it was also the reason
nothing could be cancelled: Stop, ``status()`` and unload all wanted the same
lock the waiting thread was holding, and a thread waiting three seconds for
Kokoro held it for three seconds. Section 15 asks for the responsibilities to
be split, and they are:

    _state_lock   the process handle, the handshake, the lifecycle flags and
                  the two registries. Held for microseconds at a time and never
                  across a wait for anything.

    _write_lock   serializes writes to the worker's stdin, so two frames cannot
                  interleave inside one length prefix.

Nothing waits while holding either. A request registers a queue of its own
under ``_pending``, writes its frame, and then waits on *that queue* -- so
:func:`status`, :func:`cancel_turn`, :func:`unload` and :func:`shutdown` are all
reachable while another thread is inside a three-second synthesis, which is
what gate 1 asks to be true and what release blocker one forbids being false.

One reader, two destinations
----------------------------
The single thread that reads the worker's stdout is the only thing that ever
does. It dispatches an ordinary reply to the queue of the request whose id it
carries -- one queue per request, so two callers cannot receive one another's
answers even in principle -- and a streaming speech frame to the turn whose
opaque id it carries. A frame for a turn nobody is listening to any more is
dropped, which is section 24's late-audio race handled at the one place every
frame passes through.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass

import mc_voice_models as models
import mc_voice_paths as paths
from voice_worker import worker as protocol

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""


class VoiceRuntimeError(RuntimeError):
    """A speech request that could not be served. Never fatal to Conversation."""


CONTAINMENT = {"windows": "job", "linux": "pdeathsig"}
"""The OS parent-death mechanism each fully supported platform must produce.

R2-3. A platform in this table gets the absolute I-7 guarantee and a start that
fails closed without it. A platform not in it is not offered a runtime by the
manifest at all, which is the honest way to say "not supported in V1" -- better
than shipping something that works until the day somebody force-quits.
"""

STT_THREADS = 4
TTS_THREADS = 4
"""Fixed, hidden, and deliberately not adaptive.

Two was the conservative opening bid: speech runs beside an image model, and the
point of a separate process is lost if it takes every core. Four is the
considered one. Synthesis is the stage the first-to-second gap is actually made
of -- one serialized lane, one sentence at a time -- and sherpa's ``num_threads``
is what that lane is given to work with.

What this is *not* is a tuner. Nothing in this repository rotates between two,
four and six, runs an A/B, or picks a number from real-time factors, underrun
counts or CPU load: a production feature that reconfigures itself is a feature
whose logs describe a different program each time somebody reads them. The
number appears in the diagnostics instead, so a shared log says exactly which
configuration produced the run, and moving it again is a deliberate change to
this line supported by those logs.

STT is untouched at four. A transcription is a single burst after the user has
stopped talking, and it was never the stage anybody was waiting through.
"""

HANDSHAKE_TIMEOUT = 300.0
"""Cold, this is four hundred megabytes of ONNX off a spinning disk on a
machine that may also be loading a checkpoint. Generous, and still a number."""

STT_TIMEOUT = 180.0
TTS_TIMEOUT = 300.0
STOP_GRACE = 2.0
TERMINATE_GRACE = 1.0

MAX_TEXT_BYTES = 200_000
"""The ceiling on one synthesis request, stated rather than silently applied.

Section 69: no silent truncation. A reply past this is refused with a sentence
the user can read, because speaking the first half of an answer without saying
so is worse than not speaking it.
"""

CRASH_WINDOW = 60.0
CRASH_LIMIT = 3
"""Three failed starts in a minute stops the fourth. A worker that cannot load
its models will not load them on the tenth attempt either, and a respawn loop
during a generation is a machine that gets slower for no reason at all."""


@dataclass
class Handshake:
    """What the worker said it is, once, before it was believed."""

    protocol_version: int
    runtime_version: str
    provider: str
    parent_death: str
    stt_model_id: str
    tts_model_id: str
    stt_threads: int
    tts_threads: int
    num_speakers: int = 0
    sample_rate: int = 0
    bank_version: str = ""
    streaming: str = ""
    callback_probe_ms: int = 0


_state_lock = threading.RLock()
"""The process handle, the handshake, the lifecycle flags, ``_pending`` and
``_turns``. Never held across a wait for inference, a stream, a queue or a
subprocess -- see the module docstring."""

_write_lock = threading.Lock()
"""One frame at a time down the worker's stdin."""

_start_lock = threading.Lock()
"""Held only by :func:`ensure_started`, so that two first requests arriving
together produce one worker rather than two. Separate from ``_state_lock``
because starting a worker *does* wait -- for a handshake that reads four
hundred megabytes of ONNX -- and nothing that only wants to read the state
should be behind that."""

_process = None
_reader: threading.Thread | None = None
_handshake: Handshake | None = None
_closing = False
_session = ""
_next_id = 0
_generation = 0
"""Incremented every time a worker is discarded.

The reader thread captures it at start and compares before dispatching, so a
frame that was already in the pipe when a worker was replaced cannot be
delivered to a request or a turn belonging to its successor.
"""

_pending: dict = {}
"""``request id -> Queue``. One queue per request; see T-RT-9."""

_turns: dict = {}
"""``turn id -> VoiceTurn``. The streaming destinations."""

_busy = {"stt": 0, "tts": 0}
_loading = False
_last_error = ""
_failures: list[float] = []
_job_handle = None
"""The Windows job object every Voice Worker is put in.

Held for the life of this process on purpose, exactly as ``mc_llm_runtime``
holds its own: what does the work is the handle being *closed*, which happens
when this process ends however it ends -- including the kill that runs no
handler."""


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


def status() -> dict:
    """What the runtime is, without starting anything.

    Never raises and never touches the network: this is read by a status route
    a browser polls, and a status call that could start a four-hundred-megabyte
    model load would make polling it an attack on the user's own machine.

    It also never waits. Section 16 names ``runtime.status()`` among the things
    that must answer while another thread is inside inference, and the reason
    it can is that ``_state_lock`` is not held by that thread -- the waiting
    happens on a per-request queue instead.
    """
    with _state_lock:
        running = _process is not None and _process.poll() is None
        return {
            "running": running,
            "pid": getattr(_process, "pid", None) if running else None,
            "closing": _closing,
            "provider": _handshake.provider if _handshake else "",
            "parent_death": _handshake.parent_death if _handshake else "",
            "stt_model_id": _handshake.stt_model_id if _handshake else "",
            "tts_model_id": _handshake.tts_model_id if _handshake else "",
            "session": _session,
        }


def engine() -> dict:
    """The live engine state the Voice flyout draws, without starting anything.

    Section 30. Installation and residency are different questions and the V1
    flyout could only answer the first: "Text to speech: Installed" says
    nothing about whether four hundred megabytes of ONNX are in RAM right now.
    What is deliberately *not* here is anything section 30 forbids -- no pid,
    no command line, no filesystem path, no speech text.
    """
    with _state_lock:
        running = _process is not None and _process.poll() is None
        if _loading:
            state = "loading"
        elif _closing:
            state = "stopping"
        elif not running:
            state = "error" if _last_error else "unloaded"
        elif _busy["tts"]:
            state = "tts"
        elif _busy["stt"]:
            state = "stt"
        else:
            state = "idle"
        found = {
            "loaded": bool(running),
            "state": state,
            "provider": (_handshake.provider if _handshake and running else ""),
            "error": _last_error if state == "error" else "",
        }
        if running and _handshake is not None:
            found.update({
                "stt_threads": _handshake.stt_threads,
                "tts_threads": _handshake.tts_threads,
                "voices": _handshake.num_speakers,
                "sample_rate": _handshake.sample_rate,
                "voice_bank_version": _handshake.bank_version,
                "streaming": _handshake.streaming,
            })
        return found


def supported_platform() -> bool:
    """Whether this platform is one V1 advertises the I-7 guarantee for."""
    system, _machine, _python = models.current_platform()
    return system in CONTAINMENT


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


def transcribe(wav_bytes: bytes) -> dict:
    """One utterance to text. Bytes in, ``{"text": ..., ...}`` out.

    The bytes are dropped the moment this returns -- there is no cache, no file,
    and no second reference kept for a retry. Section 48: the WAV exists in
    browser memory, in this request's memory, and in the worker's memory, and
    in none of them for longer than the call.
    """
    if not wav_bytes:
        raise VoiceRuntimeError("No audio was received.")
    with _state_lock:
        _busy["stt"] += 1
    try:
        reply, _payload = _request({"op": "stt", "format": "wav"}, wav_bytes, STT_TIMEOUT)
    finally:
        with _state_lock:
            _busy["stt"] = max(0, _busy["stt"] - 1)
    return {"text": str(reply.get("text") or ""),
            "audio_seconds": reply.get("audio_seconds"),
            "elapsed": reply.get("elapsed")}


def synthesize(text: str, sid: int = 0, profile=None) -> bytes:
    """One complete string to a WAV, in memory. Test playback and fallback.

    ``text`` is an immutable snapshot the caller already took; nothing here
    re-reads a conversation, and nothing here writes the audio anywhere. The
    return value is the response body.

    ``sid`` is a numeric speaker the *caller* resolved from the registry.
    Nothing in this module turns a name into a number, and nothing accepts one
    from a browser -- see :func:`mc_voice_registry.resolve`.

    ``profile`` is a delivery in :mod:`mc_voice_profile`'s own spelling --
    speed, pitch in semitones, volume in dB, pacing in milliseconds -- and is
    converted to the worker's multipliers by :func:`mc_voice_profile.request`.
    ``None`` means the default voice's stored delivery, which is what makes an
    audition sound like the reply that voice would speak.
    """
    encoded = (text or "").encode("utf-8")
    if not encoded.strip():
        raise VoiceRuntimeError("There was nothing to read aloud.")
    if len(encoded) > MAX_TEXT_BYTES:
        raise VoiceRuntimeError(
            f"That reply is longer than Voice Chat will read aloud in one go "
            f"({len(encoded)} bytes; the limit is {MAX_TEXT_BYTES}). It was not spoken, and "
            f"nothing was cut short without telling you.")
    with _state_lock:
        _busy["tts"] += 1
    try:
        header = {"op": "tts", "sid": int(sid or 0)}
        header.update(_delivery(profile))
        _reply, payload = _request(header, encoded, TTS_TIMEOUT)
    finally:
        with _state_lock:
            _busy["tts"] = max(0, _busy["tts"] - 1)
    if not payload:
        raise VoiceRuntimeError("The voice runtime produced no audio.")
    return payload


# --------------------------------------------------------------------------- #
# Streaming speech
# --------------------------------------------------------------------------- #


def _delivery(profile) -> dict:
    """A delivery profile as the worker's header carries it. Never raises.

    Read on the path that speaks a reply, so a profile module that could not be
    imported or a settings store that would not answer is a reply spoken in
    Kokoro's own delivery rather than a reply that is not spoken at all.
    """
    try:
        import mc_voice_profile as voice_profile

        # ``None`` means the default voice's stored delivery rather than a
        # neutral one. A turn carries a profile it resolved when it opened; a
        # caller that has none -- an audition, the completed-reply fallback --
        # is asking for "however the default voice speaks", and answering that
        # with Kokoro's own would make those two paths sound different from
        # every reply.
        return voice_profile.request(
            voice_profile.stored() if profile is None else profile)
    except Exception:
        logger.debug("Model Chain: could not read the voice delivery profile", exc_info=True)
        return {"speed": 1.0, "pitch": 1.0, "gain": 1.0, "pause_ms": 0}


def prepare() -> bool:
    """Start the Voice worker without opening a turn. Returns "it was warm".

    The cold-start overlap. :meth:`mc_voice_turn.VoiceTurn._run` used to wait
    for the first committed segment and only then call :func:`begin_turn`, which
    is what ultimately starts the worker -- so on a cold run the four hundred
    megabytes of ONNX were read *after* the model had finished writing its first
    sentence, when they could have been read while it was writing it. Both of
    those are the machine waiting; only one of them is the user waiting.

    What this deliberately does not do is anything a turn does. It registers no
    turn, it does not touch the TTS busy count, it sends no ``tts_begin`` and it
    synthesizes nothing: the worker becomes ready, and the turn that opens on it
    a moment later opens exactly as it would have. So a warmed worker that is
    then cancelled before any text arrives leaves nothing behind but a warm
    worker -- which is the state the first real use would have left anyway.

    Crash-loop guarding, restart limits, containment and the "one worker even if
    two callers arrive together" lock are all :func:`ensure_started`'s, unchanged,
    because this is that function with a question in front of it.

    Raises :class:`VoiceRuntimeError` exactly as :func:`ensure_started` does. The
    caller is a turn pump, and a turn that cannot warm a worker is a reply that
    is not spoken -- never a reply that is not written.
    """
    with _state_lock:
        if _process is not None and _process.poll() is None:
            return True
    ensure_started()
    return False


def begin_turn(turn, sid: int = 0, profile=None) -> int:
    """Open one streaming turn on the worker. Returns its sample rate.

    The turn is registered *before* the frame is written, because the worker
    answers ``tts_ready`` fast enough that a registration afterwards is a real
    race -- and a ``tts_ready`` for a turn the reader does not know about is a
    turn that never learns its own sample rate.

    ``profile`` is resolved once, here, at the moment the turn opens -- for the
    same reason its voice is (section 56): a reply is spoken with the delivery
    that was configured when it started, and changing a slider halfway through
    does not re-pitch the sentence already in the speaker.
    """
    ensure_started()
    with _state_lock:
        _turns[turn.id] = turn
        _busy["tts"] += 1
    try:
        header = {"op": "tts_begin", "turn": turn.id, "sid": int(sid or 0)}
        header.update(_delivery(profile))
        _write(header, b"")
    except _WorkerGone:
        _release_turn(turn)
        raise VoiceRuntimeError("The voice runtime stopped before it could speak.") from None
    rate = _await_rate(turn)
    return rate


def _await_rate(turn) -> int:
    """Wait for ``tts_ready``, or for the turn to end without one."""
    deadline = time.monotonic() + TTS_TIMEOUT
    while time.monotonic() < deadline:
        if turn.sample_rate:
            return turn.sample_rate
        if turn.cancelled.is_set() or turn.finished.is_set():
            return turn.sample_rate or 0
        time.sleep(0.02)
    raise VoiceRuntimeError("The voice runtime did not start speaking in time.")


def send_segment(turn, text: str) -> None:
    """Hand one immutable segment to the worker."""
    encoded = str(text or "").encode("utf-8")
    if not encoded.strip():
        return
    try:
        _write({"op": "tts_text", "turn": turn.id}, encoded)
    except _WorkerGone:
        turn.audio_failed("The voice runtime stopped while it was speaking.")


def finish_turn(turn) -> None:
    try:
        _write({"op": "tts_finish", "turn": turn.id}, b"")
    except _WorkerGone:
        turn.audio_failed("The voice runtime stopped while it was speaking.")


def cancel_turn(turn) -> None:
    """Tell the worker to stop this turn. Never waits, never raises.

    Section 27: this is not ``stop()``. Cancelling a turn is an ordinary thing
    that happens several times in a conversation, and tearing down a
    four-hundred-megabyte process to do it would make Stop cost a model reload.
    """
    try:
        _write({"op": "tts_cancel", "turn": turn.id}, b"")
    except Exception:
        logger.debug("Model Chain: could not send a Voice cancel", exc_info=True)
    finally:
        _release_turn(turn)


def _release_turn(turn) -> None:
    with _state_lock:
        _turns.pop(getattr(turn, "id", ""), None)
        _busy["tts"] = max(0, _busy["tts"] - 1)


# --------------------------------------------------------------------------- #
# One round trip
# --------------------------------------------------------------------------- #


def _request(header: dict, payload: bytes, timeout: float) -> tuple[dict, bytes]:
    """One round trip, with exactly one restart if the worker had died.

    "Exactly one" is the whole policy. A worker that vanished between two
    dictations is an ordinary thing -- a crash, a memory-hungry neighbour, an
    antivirus -- and making the user press the microphone again for it would be
    unkind. A worker that vanishes again on the retry is a broken installation,
    and the honest answer to that is an error rather than a third process.

    Note what this function does *not* do any more: hold a lock while it waits.
    """
    for attempt in (1, 2):
        with _state_lock:
            if _closing:
                raise VoiceRuntimeError("Voice Chat is shutting down.")
        ensure_started()
        try:
            return _exchange(header, payload, timeout)
        except _WorkerGone:
            stop("the voice worker stopped unexpectedly")
            if attempt == 2:
                raise VoiceRuntimeError(
                    "Voice runtime stopped unexpectedly. Try again.") from None
    raise VoiceRuntimeError("Voice runtime stopped unexpectedly. Try again.")


class _WorkerGone(Exception):
    """The pipe ended or the worker did not answer in time."""


def _exchange(header: dict, payload: bytes, timeout: float) -> tuple[dict, bytes]:
    """Write one frame and wait for the reply that carries its id.

    The waiting is on a queue this call owns. Nothing else can be handed an
    answer from it, which is what makes "two ordinary requests cannot receive
    one another's replies" structural rather than a check.
    """
    global _next_id

    with _state_lock:
        _next_id += 1
        request_id = _next_id
        answers: "queue.Queue" = queue.Queue()
        _pending[request_id] = answers

    try:
        outgoing = dict(header)
        outgoing["id"] = request_id
        _write(outgoing, payload)

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _WorkerGone("timed out")
            try:
                item = answers.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                with _state_lock:
                    alive = _process is not None and _process.poll() is None
                if not alive:
                    raise _WorkerGone("worker exited") from None
                continue
            if item is None:
                raise _WorkerGone("input closed") from None
            reply, body = item
            if not reply.get("ok"):
                raise VoiceRuntimeError(_readable(str(reply.get("error") or "")))
            return reply, body
    finally:
        with _state_lock:
            _pending.pop(request_id, None)


def _write(header: dict, payload: bytes) -> None:
    """One frame down the worker's stdin, under the write lock and no other."""
    with _state_lock:
        process = _process
    if process is None or process.poll() is not None:
        raise _WorkerGone("no worker")
    try:
        with _write_lock:
            protocol.write_frame(process.stdin, header, payload)
    except (OSError, ValueError) as exc:
        raise _WorkerGone(str(exc)) from None


def _readable(reason: str) -> str:
    """The worker's fault word, as a sentence for a status line."""
    known = {
        "audio is not mono": "The recording was not mono audio.",
        "audio is not 16-bit": "The recording was not 16-bit audio.",
        "audio is empty": "No audio was received.",
        "audio is too long": "That recording is longer than Voice Chat accepts.",
        "runtime is not initialised": "The voice runtime is not ready yet.",
        "unknown operation": "Voice Chat asked its runtime for something it does not do.",
        "that voice is not in the installed voice bank":
            "That voice is not in the installed voice bank. Choose another voice in "
            "Settings → Voice Chat.",
    }
    return known.get(reason, "The voice runtime could not complete that request.")


# --------------------------------------------------------------------------- #
# Starting
# --------------------------------------------------------------------------- #


def ensure_started() -> None:
    """Start the worker if it is not running. Idempotent.

    Every failure path below is written so that a process which has been started
    is stopped before its handle is dropped. That is the rule the orphaned
    llama-server broke: ownership begins at ``Popen``, not at the moment the
    handshake succeeds.

    Held by :data:`_start_lock` rather than by the state lock, because starting
    a worker waits for a handshake and the state lock is the one thing that must
    never be held across a wait.
    """
    global _process, _reader, _handshake, _session, _generation, _loading, _last_error

    with _state_lock:
        if _process is not None and _process.poll() is None:
            return

    with _start_lock:
        with _state_lock:
            if _process is not None and _process.poll() is None:
                return
            if _process is not None:
                _discard("a previous voice worker had already exited")
        _guard_crash_loop()

        state = models.status()
        if not state.ready:
            raise VoiceRuntimeError(
                "Voice Chat is not set up. Install both models in Settings → Voice Chat.")
        if not supported_platform():
            raise VoiceRuntimeError(
                "Voice Chat has no tested process-containment mechanism on this platform, so "
                "it will not start a speech process here.")

        interpreter = models.runtime_python()
        if interpreter is None:
            raise VoiceRuntimeError("The Voice Chat runtime is not installed.")

        session = uuid.uuid4().hex[:12]
        command = [str(interpreter), str(paths.worker_script()), protocol.MARKER,
                   "--parent-pid", str(os.getpid()), "--session", session]
        environ = dict(os.environ)
        environ.update(models.worker_environment())

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
            _reader = threading.Thread(target=_read_frames, args=(started, mine),
                                       name="mc-voice-reader", daemon=True)
            _reader.start()
            threading.Thread(target=_drain_stderr, args=(started,),
                             name="mc-voice-stderr", daemon=True).start()
            _handshake = _handshake_with(started, state)
        except Exception as exc:
            # Whatever went wrong, the process this function started is this
            # function's to end. Nothing below this line may leave a handle.
            _note_failure()
            with _state_lock:
                if started is not None:
                    _process = started
                    _discard("the voice worker failed to start")
                _handshake = None
                _last_error = str(exc) if isinstance(exc, VoiceRuntimeError) else ""
            if isinstance(exc, VoiceRuntimeError):
                raise
            # Everything else becomes a sentence. A caller of this function is a
            # status line in a chat window, and ``FileNotFoundError: [Errno 2]``
            # against a path inside somebody's home directory is neither useful
            # to them nor theirs to have shown.
            logger.warning("Model Chain: the Voice Chat worker could not be started",
                           exc_info=True)
            if isinstance(exc, _WorkerGone):
                raise VoiceRuntimeError(
                    "The Voice Chat runtime did not finish starting, so it was stopped. "
                    "Try again.") from None
            raise VoiceRuntimeError(
                "The Voice Chat runtime could not be started. Check Settings → Voice "
                "Chat.") from None
        finally:
            with _state_lock:
                _loading = False

        stop_on_exit()
        logger.info("Model Chain: Voice runtime ready — %s provider, STT threads %d, "
                    "TTS threads %d, %d voices, %s streaming (probed in %d ms), "
                    "containment %s, pid %s",
                    _handshake.provider, _handshake.stt_threads, _handshake.tts_threads,
                    _handshake.num_speakers, _handshake.streaming or "no",
                    _handshake.callback_probe_ms, _handshake.parent_death, started.pid)


# --------------------------------------------------------------------------- #
# Load and unload
# --------------------------------------------------------------------------- #


def load() -> dict:
    """The Load button. Start and handshake now rather than on first use.

    Section 31 and section 34: lazy start remains the default and nothing is
    preloaded at WebUI start-up, but a user who knows they are about to talk
    should be able to pay the model load while they are still typing rather
    than in the middle of their first sentence.

    Blocking, on purpose, and therefore never called from the event loop --
    :mod:`mc_voice_api` offloads it. What comes back is the same live engine
    state the flyout polls, so the caller redraws from the truth.
    """
    ensure_started()
    return engine()


def unload(reason: str = "unloaded") -> dict:
    """The Unload button. Frees the worker's RAM and keeps the installation.

    Not an uninstall and not a persistent disable switch: the next dictation or
    reply starts it again. What it does do is cancel the active turn first --
    replacing a speaking process without telling the turn would leave the
    browser waiting on a stream that will never produce another byte.
    """
    try:
        import mc_voice_turn as turns

        turns.forget_all(reason)
    except Exception:
        logger.debug("Model Chain: could not cancel voice turns before unloading",
                     exc_info=True)
    stop(reason)
    return engine()


def _handshake_with(started, state) -> Handshake:
    """Send ``init``, then believe nothing that comes back until it is checked.

    Five things are checked and each one is a different bug that has to fail
    closed: a protocol the parent cannot speak, a provider that is not CPU (I-1
    in the only place it can actually be observed), a containment mechanism this
    platform's support contract requires (R2-3), a pair of model ids that are
    not the pair this installation verified, and a voice bank whose speaker
    count is smaller than the registry believes -- which is the one that would
    otherwise make a custom voice speak silently in somebody else's.
    """
    config = {
        "stt": models.bundle_paths("stt"),
        "tts": _tts_config(),
        "stt_threads": STT_THREADS,
        "tts_threads": TTS_THREADS,
        "bank_version": _bank_version(),
    }
    reply, _payload = _exchange(
        {"op": "init", "parent_pid": os.getpid(), "session": _session, "config": config},
        b"", HANDSHAKE_TIMEOUT)

    found = Handshake(
        protocol_version=int(reply.get("protocol_version") or 0),
        runtime_version=str(reply.get("runtime_version") or ""),
        provider=str(reply.get("provider") or ""),
        parent_death=str(reply.get("parent_death") or ""),
        stt_model_id=str(reply.get("stt_model_id") or ""),
        tts_model_id=str(reply.get("tts_model_id") or ""),
        stt_threads=int(reply.get("stt_threads") or 0),
        tts_threads=int(reply.get("tts_threads") or 0),
        num_speakers=int(reply.get("num_speakers") or 0),
        sample_rate=int(reply.get("sample_rate") or 0),
        bank_version=str(reply.get("bank_version") or ""),
        streaming=str(reply.get("streaming") or ""),
        callback_probe_ms=int(reply.get("callback_probe_ms") or 0),
    )
    if found.protocol_version != protocol.PROTOCOL_VERSION:
        raise VoiceRuntimeError(
            f"The Voice Chat worker speaks protocol {found.protocol_version} and this "
            f"extension speaks {protocol.PROTOCOL_VERSION}. Reinstall the voice runtime.")
    if found.provider != "cpu":
        raise VoiceRuntimeError(
            f"The Voice Chat worker reported provider {found.provider!r}. Voice Chat runs on "
            f"the CPU only and will not use a graphics device.")
    system, _machine, _python = models.current_platform()
    wanted = CONTAINMENT.get(system)
    if wanted and found.parent_death != wanted:
        raise VoiceRuntimeError(
            "The Voice Chat worker could not be tied to this process's lifetime, so it was "
            "not started. A speech process that outlives the WebUI is not something this "
            "feature will leave running.")
    if (found.stt_model_id, found.tts_model_id) != (state.stt_id, state.tts_id):
        raise VoiceRuntimeError("The Voice Chat worker loaded different models from the ones "
                                "this installation verified.")
    _check_bank(found)
    return found


def _tts_config() -> dict:
    """Where the worker should read Kokoro and its voice bank from.

    The bundle is the fallback and the bank is the answer whenever one has been
    built: a custom voice exists only in a Model Chain bank, and an installation
    that has never cloned anything has no bank and runs from the bundle exactly
    as V1 did. Both are paths this process built from a manifest and a data
    root; neither can be influenced from a browser.
    """
    found = models.bundle_paths("tts")
    try:
        import mc_voice_bank as bank

        live = bank.live_paths()
    except Exception:
        logger.debug("Model Chain: could not read the voice bank", exc_info=True)
        live = None
    if live:
        found.update(live)
    return found


def _bank_version() -> str:
    try:
        import mc_voice_bank as bank

        return bank.version()
    except Exception:
        return ""


def _check_bank(found: Handshake) -> None:
    """Refuse a worker whose bank is smaller than the registry expects."""
    try:
        import mc_voice_registry as registry

        wanted = registry.highest_sid()
    except Exception:
        return
    if wanted is None or not found.num_speakers:
        return
    if wanted >= found.num_speakers:
        raise VoiceRuntimeError(
            "The installed voice bank has fewer voices than Voice Chat expects, so it was "
            "not used. Open Settings → Voice Chat to rebuild it.")


def _read_frames(started, generation: int) -> None:
    """The one reader. Ordinary replies to their request, speech to its turn.

    A thread rather than a blocking read in the requesting thread, so a wedged
    worker costs one request its timeout rather than costing the WebUI a thread
    forever -- and so that streaming audio has somewhere to arrive when no
    request is outstanding at all.
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
            # A reply to a request that already timed out is dropped rather than
            # kept: giving one caller another caller's transcript is the one
            # mistake this feature genuinely must not make.
    except Exception:
        logger.debug("Model Chain: the voice worker's pipe ended", exc_info=True)
    finally:
        _fail_everything(generation)


def _dispatch_turn(operation: str, header: dict, payload: bytes) -> None:
    """One streaming frame to the turn it names, or nowhere.

    "Or nowhere" is the point. A frame whose turn is not in ``_turns`` belongs
    to a reply that was cancelled or superseded, and section 24 says what to do
    with it: nothing at all.
    """
    with _state_lock:
        turn = _turns.get(str(header.get("turn") or ""))
    if turn is None:
        return
    if operation == "tts_audio":
        rate = int(header.get("sample_rate") or 0)
        # Blocks while the browser is behind, which is the backpressure that
        # reaches the worker through the pipe. It never blocks past a
        # cancellation -- see VoiceTurn.offer_audio.
        turn.offer_audio(payload, rate)
    elif operation == "tts_ready":
        turn.sample_rate = int(header.get("sample_rate") or 0) or turn.sample_rate
        turn.streaming = str(header.get("streaming") or "") or turn.streaming
    elif operation == "tts_segment_done":
        # Counts and milliseconds, and nothing else. With the worker's
        # ``max_num_sentences=1`` the block count is how many sentence batches
        # sherpa handed back for that segment, which is what tells a callback
        # handshake apart from a callback that actually delivered early.
        note = getattr(turn, "note_segment", None)
        if note is not None:
            note(blocks=int(header.get("blocks") or 0),
                 first_block_ms=int(header.get("first_block_ms") or 0),
                 streaming=str(header.get("streaming") or ""))
    elif operation == "tts_done":
        turn.audio_finished()
        _release_turn(turn)
    elif operation == "tts_cancelled":
        turn.cancel("worker")
        _release_turn(turn)
    elif operation == "tts_error":
        turn.audio_failed(_readable(str(header.get("error") or "")))
        _release_turn(turn)


def _fail_everything(generation: int) -> None:
    """The worker's pipe ended. Wake every waiter rather than leaving them."""
    with _state_lock:
        if generation != _generation and _process is not None:
            return
        waiting = list(_pending.values())
        speaking = list(_turns.values())
        _turns.clear()
    for answers in waiting:
        try:
            answers.put(None)
        except Exception:
            pass
    for turn in speaking:
        try:
            turn.audio_failed("The voice runtime stopped while it was speaking.")
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
                logger.debug("Model Chain: %s", text)
    except Exception:
        pass


def _guard_crash_loop() -> None:
    now = time.monotonic()
    recent = [when for when in _failures if now - when < CRASH_WINDOW]
    _failures[:] = recent
    if len(recent) >= CRASH_LIMIT:
        raise VoiceRuntimeError(
            "The Voice Chat runtime has failed to start several times in a row, so it is not "
            "being started again for now. Check Settings → Voice Chat.")


def _note_failure() -> None:
    _failures.append(time.monotonic())


# --------------------------------------------------------------------------- #
# Stopping
# --------------------------------------------------------------------------- #


def stop(reason: str = "") -> None:
    """Stop the worker if one is running. Idempotent, and never raises.

    Reachable while another thread is inside inference, which is section 16's
    requirement and is only true because nothing waiting holds ``_state_lock``.
    """
    with _state_lock:
        if _process is None:
            return
        _discard(reason or "Voice Chat stopped")


def shutdown() -> None:
    """Door A and the body of doors B and C. Idempotent; must never raise.

    Called from Forge's script-unload callback, from ``atexit``, and from the
    chained signal handlers. Everything inside is swallowed: a WebUI that will
    not close because a speech process is thinking is a worse bug than any this
    could be reporting.
    """
    global _closing

    try:
        try:
            import mc_voice_turn as turns

            turns.forget_all("shutdown")
        except Exception:
            pass
        with _state_lock:
            _closing = True
            if _process is not None:
                _discard("WebUI shutdown")
    except Exception:
        logger.debug("Model Chain: the Voice Chat shutdown hook failed", exc_info=True)
    finally:
        _closing = False


def _discard(reason: str) -> None:
    """End the worker and let go of the handle, in that order.

    The escalation is bounded at every step, because the thing being stopped may
    be inside a native inference call that will not look at a pipe again: ask,
    close the pipe, wait two seconds, terminate, wait one, kill. Nine steps in
    the design intent and nine steps here, and none of them is "wait for it".
    """
    global _process, _reader, _handshake, _generation

    with _state_lock:
        started, _process, _reader, _handshake = _process, None, None, None
        _generation += 1
        waiting = list(_pending.values())
        speaking = list(_turns.values())
        _pending.clear()
        _turns.clear()
        _busy["stt"] = _busy["tts"] = 0
    for turn in speaking:
        try:
            turn.cancel("unloaded")
            turn.drain_audio()
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
    for stream in (started.stdin,):
        try:
            if stream is not None:
                stream.close()
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
    logger.info("Model Chain: Voice worker stopped — %s", reason or "no reason given")


def _wait(started, seconds: float) -> bool:
    try:
        started.wait(timeout=seconds)
        return True
    except Exception:
        return started.poll() is not None


# --------------------------------------------------------------------------- #
# Doors B, C and E
# --------------------------------------------------------------------------- #


_exit_registered = False
_exit_lock = threading.Lock()


def stop_on_exit() -> None:
    """Arm the ordinary exits, once, from the first start.

    Not at import: an installation that never presses the microphone should not
    register a hook to stop something it never started. Registered from
    :func:`ensure_started` instead, which is the first moment there is anything
    to stop.
    """
    global _exit_registered

    with _exit_lock:
        if _exit_registered:
            return
        _exit_registered = True
    import atexit

    atexit.register(_at_exit)
    for name in ("SIGTERM", "SIGINT"):
        _relay_signal(name)


def _at_exit() -> None:
    try:
        shutdown()
    except Exception:
        logger.debug("Model Chain: the Voice Chat exit hook failed", exc_info=True)


def _relay_signal(name: str) -> None:
    """Stop the worker on ``name``, then do whatever was going to happen.

    Chained, never replaced. A handler of ours that swallowed SIGINT would leave
    a terminal whose Ctrl+C does nothing, which is a far more visible bug than
    the one it was added to fix.
    """
    import signal

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
        logger.debug("Model Chain: could not chain the %s handler for Voice Chat", name,
                     exc_info=True)


JOB_KILL_ON_CLOSE = 0x00002000
JOB_EXTENDED_LIMIT_INFORMATION = 9


def _die_with_us(started) -> None:
    """Door E on Windows: the job object, arranged before the worker gets work.

    Duplicated from ``mc_llm_runtime`` rather than shared, deliberately. The
    design intent offers both, and a neutral helper would mean editing the file
    that holds the proven llama-server fix in order to add a speech feature --
    which is how a working shutdown path acquires a regression. This is forty
    lines, it is tested on its own, and it has no opinions about language models.

    On Linux the equivalent has to run *inside* the child, between fork and the
    first request, so it lives in ``voice_worker/worker.py`` and is reported back
    through the handshake. Nothing is arranged here for that platform, and
    :func:`_handshake_with` refuses to run without the child's confirmation.
    """
    global _job_handle

    if os.name != "nt":
        return
    handle = getattr(started, "_handle", None)
    if handle is None:
        raise VoiceRuntimeError("The Voice Chat worker could not be tied to this process.")
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
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
    if not kernel.AssignProcessToJobObject(_job_handle, int(handle)):
        raise ctypes.WinError(ctypes.get_last_error())
