"""One Sopro worker, owned properly, and stopped however the WebUI ends.

The same five doors :mod:`mc_voice_runtime` documents, arranged again for a
second process, because a Torch runtime that outlives the WebUI is the same bug
as an ONNX one and a hundred and forty megabytes worse:

    A  Forge extension unload      ``scripts/model_chain.py`` -> :func:`shutdown`
    B  ordinary interpreter exit   ``atexit``, armed by the first start
    C  SIGINT / SIGTERM            chained in front of whatever was there
    D  pipe EOF                    the worker's own loop ends when stdin closes
    E  OS parent-death             a Windows job object with KILL_ON_JOB_CLOSE,
                                   and ``PR_SET_PDEATHSIG`` inside the child on
                                   Linux

D and E are two layers of the same requirement and neither is redundant. EOF is
noticed when the child is *waiting to read*, which it is not while it is inside
the acoustic solver -- so the hard-kill claim rests on E. :data:`CONTAINMENT` is
therefore a support contract rather than a bonus: on a platform listed there, a
worker that comes back from its handshake without the expected containment is
stopped and the start fails.

Why this is not a mode of the Kokoro runtime
--------------------------------------------
Nothing here imports :mod:`mc_voice_runtime` and nothing there imports this.
They start different interpreters out of different closures, they check
different things in their handshakes, and they have different reasons to refuse
-- and I-6 says their process lifecycles are separate concerns. The forty lines
of job-object code below are duplicated for the same reason ``mc_voice_runtime``
gives for duplicating them from ``mc_llm_runtime``: sharing them would mean
editing a proven shutdown path in order to add an optional engine, which is how
a working shutdown acquires a regression.

Exactly one TTS worker runs at a time (section 19), and that is enforced one
level up in :func:`mc_voice_engines.select`, which stops both before it persists
a new choice. This module only ever starts a worker when Sopro is the selected
engine -- :func:`ensure_started` refuses otherwise, so a stale request from a
page drawn before somebody switched cannot load a hundred and forty megabytes of
Torch behind their back.

Two locks and a per-request queue
---------------------------------
Identical in shape to Kokoro's, and for the identical reason: nothing waits
while holding state. A request registers a queue of its own, writes its frame,
and waits on *that queue*, so :func:`status`, :func:`cancel_turn`,
:func:`unload` and :func:`shutdown` are all reachable while another thread is
three seconds into a paragraph. That is Gate S-4's precondition and it is
structural rather than checked.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field

import mc_voice_paths as paths
from sopro_worker import worker as protocol

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""


class SoproRuntimeError(RuntimeError):
    """A Sopro request that could not be served. Never fatal to Conversation."""


CONTAINMENT = {"windows": "job", "linux": "pdeathsig"}
"""The OS parent-death mechanism each platform must produce before this runs.

A platform in this table gets the absolute guarantee and a start that fails
closed without it. Windows is the only system this release's manifest offers a
Sopro closure for; Linux is listed anyway because a maintainer who adds a Linux
CPU closure must not have to remember to add its containment contract too.
"""

HANDSHAKE_TIMEOUT = 600.0
"""Cold, this is a Torch import plus a model load on a CPU that may also be
rendering an image. Twice Kokoro's, and still a number rather than "forever"."""

REQUEST_TIMEOUT = 300.0
PREPARE_TIMEOUT = 900.0
"""Preparing a voice runs the speaker encoder, the semantic encoder, the
vocoder's mel and then a full production audition, on the CPU. Generous because
the alternative is a clone that fails on a slow machine for no reason but the
clock."""

STOP_GRACE = 3.0
TERMINATE_GRACE = 1.5

MAX_TEXT_BYTES = 200_000
"""The ceiling on one synthesis request, stated rather than silently applied.
A reply past this is refused with a sentence, because speaking the first half of
an answer without saying so is worse than not speaking it."""

CRASH_WINDOW = 60.0
CRASH_LIMIT = 3
"""Three failed starts in a minute stops the fourth. A worker that cannot load
its model will not load it on the tenth attempt either, and a respawn loop
during an image generation is a machine that gets slower for no reason."""


@dataclass
class Handshake:
    """What the worker said it is, once, before it was believed.

    Everything section 20 asks a Sopro handshake to report, including the two
    thread counts -- so that a shared log can never present a Kokoro measured at
    four threads beside a Sopro that silently took every logical core.
    """

    protocol_version: int = 0
    backend: str = ""
    parent_death: str = ""
    device: str = ""
    model_id: str = ""
    fingerprint: str = ""
    sopro_version: str = ""
    torch_version: str = ""
    precision: str = ""
    sample_rate: int = 0
    hop_ratio: int = 0
    style_ctrl_dim: int = 0
    voices: int = 0
    streaming: str = ""
    intraop_threads: int = 0
    interop_threads: int = 0
    omp_num_threads: str = ""
    load_seconds: float = 0.0
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
"""Incremented every time a worker is discarded, so a frame already in the pipe
cannot be delivered to a request or a turn belonging to its successor."""

_pending: dict = {}
_turns: dict = {}
_busy = 0
_preparing = 0
"""How many synthesis requests and how many voice preparations are in flight.

Counters rather than flags because the settings page can audition while a clone
finishes, and a boolean would be cleared by whichever ended first.
"""
_loading = False
_last_error = ""
_failures: list = []
_job_handle = None


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


def status() -> dict:
    """What the runtime is, without starting anything.

    Never raises, never waits and never touches the network: this is read by a
    status route a browser polls, and a status call that could start a model
    load would make polling it an attack on the user's own machine.
    """
    with _state_lock:
        running = _process is not None and _process.poll() is None
        return {
            "running": running,
            "closing": _closing,
            "backend": _handshake.backend if _handshake else "",
            "parent_death": _handshake.parent_death if _handshake else "",
            "model_id": _handshake.model_id if _handshake else "",
            "session": _session,
        }


def _busier(step: int) -> None:
    global _busy

    with _state_lock:
        _busy = max(0, _busy + int(step))


def _preparing_by(step: int) -> None:
    global _preparing

    with _state_lock:
        _preparing = max(0, _preparing + int(step))


def engine() -> dict:
    """The live engine state the Sopro panel and the Voice overlay draw.

    Section 58's states, and deliberately nothing section 30 forbids: no pid, no
    command line, no filesystem path, no spoken text.
    """
    with _state_lock:
        running = _process is not None and _process.poll() is None
        if _loading:
            state = "loading"
        elif _closing:
            state = "stopping"
        elif not running:
            state = "error" if _last_error else "unloaded"
        elif _preparing:
            state = "preparing"
        elif _busy:
            state = "speaking"
        else:
            state = "idle"
        found = {
            "loaded": bool(running),
            "state": state,
            "backend": "sopro",
            "error": _last_error if state == "error" else "",
        }
        if running and _handshake is not None:
            found.update({
                "device": _handshake.device,
                "precision": _handshake.precision,
                "sample_rate": _handshake.sample_rate,
                "intraop_threads": _handshake.intraop_threads,
                "interop_threads": _handshake.interop_threads,
                "voices": _handshake.voices,
                "streaming": _handshake.streaming,
                "fingerprint": _handshake.fingerprint,
                "style_ctrl_dim": _handshake.style_ctrl_dim,
            })
        return found


def defaults() -> dict:
    """The pinned model's own generation defaults, or an empty mapping.

    Read from the handshake rather than repeated in the UI, so a model revision
    that changes its temperature changes what "default" means everywhere at once.
    Empty when no worker has ever started, which the Advanced panel shows as
    "the model's own" rather than as a number it made up.
    """
    with _state_lock:
        return dict(_handshake.defaults) if _handshake else {}


def supported_platform() -> bool:
    """Whether this platform has a tested containment mechanism for Sopro."""
    import mc_voice_models as models

    system, _machine, _python = models.current_platform()
    return system in CONTAINMENT


def config_line() -> str:
    """One stable line naming every fixed choice this Sopro build speaks with.

    Written at start-up so a log somebody shares is self-describing. Section 52:
    a report of "there was a four-second pause" is not actionable until the
    configuration that produced it is on the same page -- which precision, how
    many threads, what chunk size, how many solver steps, and which pinned
    closure. Effective values throughout: what the *worker reported*, never what
    this module asked for.
    """
    found = _handshake
    if found is None:
        return "Sopro TTS config — no worker has started"
    return ("Sopro TTS config — backend=sopro, fingerprint={fingerprint}, sopro={sopro}, "
            "torch={torch}, precision={precision}, device={device}, "
            "intraop_threads={intraop}, interop_threads={interop}, "
            "chunk_frames={chunk}, steps={steps}, sample_rate={rate}".format(
                fingerprint=found.fingerprint[:12] or "unknown", sopro=found.sopro_version,
                torch=found.torch_version, precision=found.precision, device=found.device,
                intraop=found.intraop_threads, interop=found.interop_threads,
                chunk=(found.defaults or {}).get("chunk_frames", "?"),
                steps=(found.defaults or {}).get("steps", "?"), rate=found.sample_rate))


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


def prepare_voice(root: str, voice_id: str, wav_bytes: bytes, seconds: float = 0.0,
                  audition: str = "", profile=None) -> dict:
    """Turn a reference recording into a saved voice, and prove it before saying yes.

    Everything about the transaction is the worker's, because everything about
    it needs a tensor: preparing, writing both assets, reading them back from
    the files that were written, and streaming a short audition through the
    production path. What comes back is the metadata to commit and the WAV to
    play. A failure anywhere leaves the caller with an exception and a directory
    it can remove -- there is no half-registered state to unwind (section 27).
    """
    _preparing_by(1)
    try:
        header = {"op": "prepare", "root": str(root), "voice_id": str(voice_id),
                  "seconds": float(seconds or 0.0),
                  "audition": str(audition or "This is a test of the cloned voice.")}
        header.update(_delivery(profile))
        reply, payload = _request(header, bytes(wav_bytes or b""), PREPARE_TIMEOUT)
    finally:
        _preparing_by(-1)
    return {"metadata": dict(reply.get("metadata") or {}),
            "sample_rate": int(reply.get("sample_rate") or 0),
            "audition_ms": int(reply.get("audition_ms") or 0),
            "audio": payload}


def refresh_catalog(voices: dict, forget=()) -> int:
    """Tell a running worker what exists now. Never starts one.

    A voice-library mutation refreshes the active worker's validated catalogue
    rather than restarting it (section 18), and does nothing at all when no
    worker is running -- the next start reads the catalogue from the registry
    anyway, so there is nothing to keep in step.
    """
    with _state_lock:
        if _process is None or _process.poll() is not None:
            return 0
    try:
        reply, _payload = _request({"op": "catalog", "voices": dict(voices or {}),
                                    "forget": list(forget or ())}, b"", 30.0)
    except SoproRuntimeError:
        logger.debug("Model Chain: could not refresh the Sopro voice catalogue",
                     exc_info=True)
        return 0
    return int(reply.get("voices") or 0)


def synthesize(text: str, voice_id: str, profile=None) -> bytes:
    """One complete string to a WAV, in memory. Auditions and the Test button.

    ``voice_id`` is the backend-qualified stable id, and this module never turns
    a name into anything else: what the worker receives is the same id the
    registry committed, and the mapping from it to tensors happens inside the
    worker's catalogue (I-10).
    """
    encoded = (text or "").encode("utf-8")
    if not encoded.strip():
        raise SoproRuntimeError("There was nothing to read aloud.")
    if len(encoded) > MAX_TEXT_BYTES:
        raise SoproRuntimeError(
            f"That reply is longer than Voice Chat will read aloud in one go "
            f"({len(encoded)} bytes; the limit is {MAX_TEXT_BYTES}). It was not spoken, and "
            f"nothing was cut short without telling you.")
    header = {"op": "tts", "voice_id": str(voice_id or "")}
    header.update(_delivery(profile))
    _busier(1)
    try:
        _reply, payload = _request(header, encoded, REQUEST_TIMEOUT)
    finally:
        _busier(-1)
    if not payload:
        raise SoproRuntimeError("Sopro produced no audio.")
    return payload


def lab_audition(voice_id: str, text: str, deltas=(), blend=None, profile=None) -> dict:
    """One Voice Lab audition. A separate operation, on purpose.

    Section 39 asks for production code to be unable to read Lab state *because
    the types are separate*, rather than because a caller promised not to. This
    is the parent half of that: a different operation, a different worker
    function, a different return shape, and no path from here into the turn
    machinery, the registry, a character or a default.
    """
    header = {"op": "lab", "voice_id": str(voice_id or ""),
              "deltas": [float(value) for value in (deltas or ())]}
    if blend:
        header["blend"] = dict(blend)
    header.update(_delivery(profile))
    reply, payload = _request(header, str(text or "").encode("utf-8"), REQUEST_TIMEOUT)
    return {"audio": payload,
            "sample_rate": int(reply.get("sample_rate") or 0),
            "first_audio_ms": int(reply.get("first_audio_ms") or 0),
            "elapsed_ms": int(reply.get("elapsed_ms") or 0),
            "audio_ms": int(reply.get("audio_ms") or 0),
            "rtf": float(reply.get("rtf") or 0.0),
            "chunks": int(reply.get("chunks") or 0)}


def lab_style(voice_id: str) -> list:
    """The saved ``style_ctrl`` a Lab session's eight sliders start from."""
    reply, _payload = _request({"op": "lab_style", "voice_id": str(voice_id or "")},
                               b"", REQUEST_TIMEOUT)
    return [float(value) for value in (reply.get("style_ctrl") or ())]


# --------------------------------------------------------------------------- #
# Streaming speech
# --------------------------------------------------------------------------- #


def _delivery(profile) -> dict:
    """A delivery profile as the worker's header carries it. Never raises.

    Read on the path that speaks a reply, so a profile module that could not be
    imported or a settings store that would not answer is a reply spoken in
    Sopro's own delivery rather than a reply that is not spoken at all.
    """
    try:
        import mc_voice_sopro_profile as sopro_profile

        return sopro_profile.request(
            sopro_profile.stored() if profile is None else profile)
    except Exception:
        logger.debug("Model Chain: could not read the Sopro delivery profile", exc_info=True)
        return {"speed": 1.0, "pitch": 1.0, "gain": 1.0, "pause_ms": 0}


def prepare() -> bool:
    """Start the Sopro worker without opening a turn. Returns "it was warm".

    The cold-start overlap (section 47). A turn that waited for the first
    committed segment before starting anything would read a hundred and forty
    megabytes of Torch *after* the language model had finished writing its first
    sentence, when it could have read it while the model was writing. Both are
    the machine waiting; only one of them is the user waiting.

    It registers no turn, sends no ``tts_begin`` and synthesizes nothing, so a
    warmed worker that is then cancelled before any text arrives leaves nothing
    behind but a warm worker.
    """
    with _state_lock:
        if _process is not None and _process.poll() is None:
            return True
    ensure_started()
    return False


def warm(voice_id: str, profile=None) -> str:
    """Reconstruct and warm one voice ahead of the text. Never raises.

    Best effort by design: this is an optimisation, and a voice that could not
    be warmed is a first sentence that takes longer, not a reply that goes
    unspoken. What it returns is what the cache did, so the telemetry can tell a
    slow opening caused by the model apart from one caused by a cache miss.
    """
    try:
        header = {"op": "warm", "voice_id": str(voice_id or "")}
        header.update(_delivery(profile))
        reply, _payload = _request(header, b"", REQUEST_TIMEOUT)
        return str(reply.get("cache_state") or "")
    except Exception:
        logger.debug("Model Chain: could not warm a Sopro voice", exc_info=True)
        return ""


def begin_turn(turn, voice_id: str = "", profile=None) -> int:
    """Open one streaming turn on the worker. Returns its sample rate.

    The turn is registered *before* the frame is written, because the worker can
    answer ``tts_ready`` fast enough that a registration afterwards is a real
    race -- and a ``tts_ready`` for a turn the reader does not know about is a
    turn that never learns its own sample rate.

    ``voice_id`` is backend-qualified and is what the shared protocol carries;
    there is no ``sid`` here and there is nothing for one to mean (I-10).
    """
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
        raise SoproRuntimeError("Sopro stopped before it could speak.") from None
    return _await_rate(turn)


def _await_rate(turn) -> int:
    deadline = time.monotonic() + REQUEST_TIMEOUT
    while time.monotonic() < deadline:
        if turn.sample_rate:
            return turn.sample_rate
        if turn.cancelled.is_set() or turn.finished.is_set():
            return turn.sample_rate or 0
        time.sleep(0.02)
    raise SoproRuntimeError("Sopro did not start speaking in time.")


def send_segment(turn, text: str) -> None:
    """Hand one immutable committed unit to the worker."""
    encoded = str(text or "").encode("utf-8")
    if not encoded.strip():
        return
    try:
        _write({"op": "tts_text", "turn": turn.id}, encoded)
    except _WorkerGone:
        turn.audio_failed("Sopro stopped while it was speaking.")


def finish_turn(turn) -> None:
    try:
        _write({"op": "tts_finish", "turn": turn.id}, b"")
    except _WorkerGone:
        turn.audio_failed("Sopro stopped while it was speaking.")


def cancel_turn(turn) -> None:
    """Tell the worker to stop this turn. Never waits, never raises.

    Not :func:`stop`. Cancelling a turn is an ordinary thing that happens
    several times in a conversation, and tearing down a Torch process to do it
    would make Stop cost a model reload.
    """
    try:
        _write({"op": "tts_cancel", "turn": turn.id}, b"")
    except Exception:
        logger.debug("Model Chain: could not send a Sopro cancel", exc_info=True)
    finally:
        _release_turn(turn)


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

    Note what this does *not* do: hold a lock while it waits.
    """
    for attempt in (1, 2):
        with _state_lock:
            if _closing:
                raise SoproRuntimeError("Voice Chat is shutting down.")
        ensure_started()
        try:
            return _exchange(header, payload, timeout)
        except _WorkerGone:
            stop("the Sopro worker stopped unexpectedly")
            if attempt == 2:
                raise SoproRuntimeError("Sopro stopped unexpectedly. Try again.") from None
    raise SoproRuntimeError("Sopro stopped unexpectedly. Try again.")


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
            raise _WorkerGone("the Sopro worker did not answer in time") from None
        if found is None:
            raise _WorkerGone("the Sopro worker stopped")
        reply, body = found
        if not reply.get("ok", True):
            raise SoproRuntimeError(_readable(str(reply.get("error") or "")))
        return reply, body
    finally:
        with _state_lock:
            _pending.pop(request_id, None)


def _write(header: dict, payload: bytes) -> None:
    with _state_lock:
        started = _process
    if started is None or started.poll() is not None or started.stdin is None:
        raise _WorkerGone("the Sopro worker is not running")
    try:
        with _write_lock:
            protocol.write_frame(started.stdin, header, payload)
    except Exception as exc:
        raise _WorkerGone(str(exc)) from None


def _readable(reason: str) -> str:
    """A worker's own words as something a user can act on.

    The worker sends an exception class for anything it did not raise itself, so
    this turns the handful it does send into sentences and leaves everything
    else as a general failure rather than showing somebody ``RuntimeError``.
    """
    text = str(reason or "").strip()
    known = {
        "": "Sopro could not complete that request.",
        "RuntimeError": "Sopro could not complete that request.",
        "OSError": "Sopro could not read one of its own files.",
        "FileNotFoundError": "One of Sopro's files is missing. Reinstall it in Settings → "
                             "Voice Chat.",
        "MemoryError": "There was not enough memory to run Sopro.",
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

    Refuses outright when Sopro is not the selected engine. Section 19 allows
    exactly one active TTS worker, and a stale request from a page drawn before
    somebody switched must not be able to load a Torch runtime behind their back.
    """
    global _process, _reader, _handshake, _session, _generation, _loading, _last_error

    with _state_lock:
        if _process is not None and _process.poll() is None:
            return

    import mc_voice_engines as engines
    import mc_voice_sopro as sopro

    if engines.active() != engines.SOPRO:
        raise SoproRuntimeError(
            "Sopro is not the selected text-to-speech engine, so it was not started.")

    with _start_lock:
        with _state_lock:
            if _process is not None and _process.poll() is None:
                return
            if _process is not None:
                _discard("a previous Sopro worker had already exited")
        _guard_crash_loop()

        state = sopro.status()
        if not state.ready:
            raise SoproRuntimeError(
                "Sopro is not installed. Install it in Settings → Voice Chat.")
        if not supported_platform():
            raise SoproRuntimeError(
                "Sopro has no tested process-containment mechanism on this platform, so it "
                "will not start a speech process here.")
        interpreter = sopro.runtime_python()
        if interpreter is None:
            raise SoproRuntimeError("The Sopro runtime is not installed.")

        session = uuid.uuid4().hex[:12]
        command = [str(interpreter), str(paths.sopro_worker_script()), protocol.MARKER,
                   "--parent-pid", str(os.getpid()), "--session", session]
        environ = dict(os.environ)
        environ.update(sopro.worker_environment())

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
                                       name="mc-sopro-reader", daemon=True)
            _reader.start()
            threading.Thread(target=_drain_stderr, args=(started,),
                             name="mc-sopro-stderr", daemon=True).start()
            _handshake = _handshake_with(state)
        except Exception as exc:
            # Whatever went wrong, the process this function started is this
            # function's to end. Nothing below this line may leave a handle.
            _note_failure()
            with _state_lock:
                if started is not None:
                    _process = started
                    _discard("the Sopro worker failed to start")
                _handshake = None
                _last_error = str(exc) if isinstance(exc, SoproRuntimeError) else ""
            if isinstance(exc, SoproRuntimeError):
                raise
            logger.warning("Model Chain: the Sopro worker could not be started", exc_info=True)
            if isinstance(exc, _WorkerGone):
                raise SoproRuntimeError(
                    "Sopro did not finish starting, so it was stopped. Try again.") from None
            raise SoproRuntimeError(
                "Sopro could not be started. Check Settings → Voice Chat.") from None
        finally:
            with _state_lock:
                _loading = False

        stop_on_exit()
        logger.info("Model Chain: Sopro ready — %s, precision %s, %d intra-op threads, "
                    "%d voices, containment %s, loaded in %.1f s, pid %s",
                    _handshake.device, _handshake.precision, _handshake.intraop_threads,
                    _handshake.voices, _handshake.parent_death, _handshake.load_seconds,
                    started.pid)
        logger.info("Model Chain: %s", config_line())


def _handshake_with(state) -> Handshake:
    """Send ``init``, then believe nothing that comes back until it is checked.

    Five checks and each one is a different bug that has to fail closed:

        a protocol the parent cannot speak;
        a backend that is not Sopro -- the parent refuses a handshake for an
            engine other than the globally selected one (section 18);
        a device that is not the CPU, which is I-9 in the only place it can
            actually be observed;
        a containment mechanism this platform's support contract requires;
        a preparation fingerprint that is not the one this installation
            verified, which is what stops a re-installed model from silently
            speaking every saved voice through conditioning it no longer matches.
    """
    import mc_voice_sopro as sopro

    config = sopro.worker_config()
    reply, _payload = _exchange(
        {"op": "init", "parent_pid": os.getpid(), "session": _session, "config": config},
        b"", HANDSHAKE_TIMEOUT)

    found = Handshake(
        protocol_version=int(reply.get("protocol_version") or 0),
        backend=str(reply.get("backend") or ""),
        parent_death=str(reply.get("parent_death") or ""),
        device=str(reply.get("device") or ""),
        model_id=str(reply.get("model_id") or ""),
        fingerprint=str(reply.get("fingerprint") or ""),
        sopro_version=str(reply.get("sopro_version") or ""),
        torch_version=str(reply.get("torch_version") or ""),
        precision=str(reply.get("precision") or ""),
        sample_rate=int(reply.get("sample_rate") or 0),
        hop_ratio=int(reply.get("hop_ratio") or 0),
        style_ctrl_dim=int(reply.get("style_ctrl_dim") or 0),
        voices=int(reply.get("voices") or 0),
        streaming=str(reply.get("streaming") or ""),
        intraop_threads=int(reply.get("intraop_threads") or 0),
        interop_threads=int(reply.get("interop_threads") or 0),
        omp_num_threads=str(reply.get("omp_num_threads") or ""),
        load_seconds=float(reply.get("load_seconds") or 0.0),
        defaults=dict(reply.get("defaults") or {}),
    )
    if found.protocol_version != protocol.PROTOCOL_VERSION:
        raise SoproRuntimeError(
            f"The Sopro worker speaks protocol {found.protocol_version} and this extension "
            f"speaks {protocol.PROTOCOL_VERSION}. Reinstall Sopro.")
    if found.backend != "sopro":
        raise SoproRuntimeError(
            f"That worker reported backend {found.backend!r} rather than Sopro, so it was "
            f"not used.")
    if not found.device.startswith("cpu"):
        raise SoproRuntimeError(
            f"The Sopro worker reported device {found.device!r}. Sopro runs on the CPU only "
            f"and will not use a graphics device.")
    import mc_voice_models as models

    system, _machine, _python = models.current_platform()
    wanted = CONTAINMENT.get(system)
    if system == "windows":
        # Proved in :func:`_die_with_us`, against this job and this child's own
        # handle, before the worker was given any work. What the worker says is
        # corroboration: it can only ask the weaker question -- am I in *some*
        # job -- and on a real machine it could not always ask it at all. It is
        # logged when it disagrees and it is never a veto, because a veto here
        # is a worker refused for failing to confirm a fact the kernel has
        # already confirmed at the other end.
        if found.parent_death != wanted:
            logger.info("Model Chain: the Sopro worker reported its containment as %r; the "
                        "job object was confirmed at this end, and that is what ends it",
                        found.parent_death or "unknown")
    elif wanted and found.parent_death != wanted:
        raise SoproRuntimeError(
            "The Sopro worker could not be tied to this process's lifetime, so it was not "
            "started. A speech process that outlives the WebUI is not something this feature "
            "will leave running.")
    if state.fingerprint and found.fingerprint and found.fingerprint != state.fingerprint:
        raise SoproRuntimeError(
            "The Sopro worker loaded a different build from the one this installation "
            "verified, so it was stopped.")
    return found


# --------------------------------------------------------------------------- #
# Load and unload
# --------------------------------------------------------------------------- #


def load() -> dict:
    """The Load button. Start and handshake now rather than on first use.

    Blocking, on purpose, and therefore never called from the event loop --
    :mod:`mc_voice_api` offloads it. What comes back is the same live engine
    state the panel polls, so the caller redraws from the truth.
    """
    ensure_started()
    return engine()


def unload(reason: str = "unloaded") -> dict:
    """The Unload button. Frees the worker's RAM and keeps the installation.

    Not an uninstall and not a persistent disable switch: the next reply starts
    it again. What it does do is cancel the active turn first -- replacing a
    speaking process without telling the turn would leave the browser waiting on
    a stream that will never produce another byte.
    """
    try:
        import mc_voice_turn as turns

        turns.forget_all(reason)
    except Exception:
        logger.debug("Model Chain: could not cancel voice turns before unloading Sopro",
                     exc_info=True)
    stop(reason)
    return engine()


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _read_frames(started, generation: int) -> None:
    """The one reader. Ordinary replies to their request, speech to its turn."""
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
        logger.debug("Model Chain: the Sopro worker's pipe ended", exc_info=True)
    finally:
        _fail_everything(generation)


def _dispatch_turn(operation: str, header: dict, payload: bytes) -> None:
    """One streaming frame to the turn it names, or nowhere.

    "Or nowhere" is the point, and it is the central dispatch point section 49
    asks for: a frame whose turn is not in ``_turns`` belongs to a reply that
    was cancelled or superseded, and late audio for it reaches no browser.
    """
    with _state_lock:
        turn = _turns.get(str(header.get("turn") or ""))
    if turn is None:
        return
    if operation == "tts_audio":
        # Blocks while the browser is behind, which is the backpressure that
        # reaches the worker through the pipe. It never blocks past a
        # cancellation -- see VoiceTurn.offer_audio.
        turn.offer_audio(payload, int(header.get("sample_rate") or 0))
    elif operation == "tts_ready":
        turn.sample_rate = int(header.get("sample_rate") or 0) or turn.sample_rate
        turn.streaming = str(header.get("streaming") or "") or turn.streaming
    elif operation == "tts_segment_done":
        note = getattr(turn, "note_segment", None)
        if note is not None:
            # Wrapped, because this is the one reader thread: a destination that
            # cannot take a new field would otherwise end the pipe for every
            # frame after it, and a measurement is never worth the audio it was
            # measuring.
            try:
                rate = int(header.get("sample_rate") or 0) or turn.sample_rate or 0
                samples = int(header.get("samples") or 0)
                note(blocks=int(header.get("chunks") or 0),
                     first_block_ms=int(header.get("first_audio_ms") or 0),
                     synth_ms=int(header.get("segment_ms") or 0),
                     audio_ms=int(samples * 1000 / rate) if rate else 0,
                     streaming="chunk")
            except Exception:
                logger.debug("Model Chain: a Sopro unit's timing could not be recorded",
                             exc_info=True)
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
            turn.audio_failed("Sopro stopped while it was speaking.")
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
                # Info rather than debug. These are exceptional-condition notes,
                # not chatter, and the one that mattered most -- why the worker
                # could not confirm its own containment -- was invisible at the
                # default level in the log of the user it stopped.
                logger.info("Model Chain: %s", text)
    except Exception:
        pass


def _guard_crash_loop() -> None:
    now = time.monotonic()
    recent = [when for when in _failures if now - when < CRASH_WINDOW]
    _failures[:] = recent
    if len(recent) >= CRASH_LIMIT:
        raise SoproRuntimeError(
            "Sopro has failed to start several times in a row, so it is not being started "
            "again for now. Check Settings → Voice Chat.")


def _note_failure() -> None:
    _failures.append(time.monotonic())


# --------------------------------------------------------------------------- #
# Stopping
# --------------------------------------------------------------------------- #


def stop(reason: str = "") -> None:
    """Stop the worker if one is running. Idempotent, and never raises.

    Reachable while another thread is inside the solver, which is only true
    because nothing waiting holds ``_state_lock``.
    """
    with _state_lock:
        if _process is None:
            return
        _discard(reason or "Sopro stopped")


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
        logger.debug("Model Chain: the Sopro shutdown hook failed", exc_info=True)
    finally:
        _closing = False


def _discard(reason: str) -> None:
    """End the worker and let go of the handle, in that order.

    The escalation is bounded at every step, because the thing being stopped may
    be inside a solver step that will not look at a pipe again: ask, close the
    pipe, wait three seconds, terminate, wait, kill. None of the steps is "wait
    for it".
    """
    global _process, _reader, _handshake, _generation, _busy, _preparing

    with _state_lock:
        started, _process, _reader, _handshake = _process, None, None, None
        _generation += 1
        waiting = list(_pending.values())
        speaking = list(_turns.values())
        _pending.clear()
        _turns.clear()
        _busy = 0
        _preparing = 0
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
    logger.info("Model Chain: Sopro worker stopped — %s", reason or "no reason given")


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

    Not at import: an installation that never selects Sopro should not register
    a hook to stop something it never started.
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
        logger.debug("Model Chain: the Sopro exit hook failed", exc_info=True)


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
        logger.debug("Model Chain: could not chain the %s handler for Sopro", name,
                     exc_info=True)


JOB_KILL_ON_CLOSE = 0x00002000
JOB_EXTENDED_LIMIT_INFORMATION = 9


def _die_with_us(started) -> None:
    """Door E on Windows: the job object, arranged before the worker gets work.

    A job of Sopro's own rather than the one :mod:`mc_voice_runtime` holds, and
    duplicated for the reason that module gives for duplicating it in the first
    place: a neutral helper would mean editing a proven shutdown path in order
    to add an optional engine. Forty lines, tested on its own, with no opinions
    about sherpa.

    The handle is held for the life of this process on purpose. What does the
    work is the handle being *closed*, which happens when this process ends
    however it ends -- including the kill that runs no handler at all.

    On Linux the equivalent has to run *inside* the child, between fork and the
    first request, so it lives in ``sopro_worker/worker.py`` and is reported
    back through the handshake. Nothing is arranged here for that platform, and
    :func:`_handshake_with` refuses to run without the child's confirmation.
    """
    global _job_handle

    if os.name != "nt":
        return
    handle = getattr(started, "_handle", None)
    if handle is None:
        raise SoproRuntimeError("The Sopro worker could not be tied to this process.")
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    # Declared rather than left to ctypes' defaults. Without argtypes every
    # argument is marshalled as a C ``int``, and a HANDLE on 64-bit Windows is
    # not one: the calls here happen to survive it because kernel handles are
    # small, and the check below -- which passes a real handle and reads a
    # BOOL out by pointer -- is exactly the shape that does not.
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

    # And then ask the kernel whether it took, here, where both handles are
    # real ones and the question can name *this* job rather than "any job".
    #
    # This is where containment is proved, and it did not used to be. The proof
    # lived in the worker instead, which could only ask the weaker question --
    # am I in some job -- through a pseudo-handle, and which answered "no" when
    # it meant "I could not tell". A worker whose containment was arranged and
    # enforced was then refused for failing to confirm it, on a real machine,
    # every time. The place that made the arrangement is the place that can
    # check it.
    inside = wintypes.BOOL(0)
    if not kernel.IsProcessInJob(wintypes.HANDLE(int(handle)), _job_handle,
                                 ctypes.byref(inside)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not inside.value:
        raise SoproRuntimeError(
            "Windows did not put the Sopro worker in this process's job object, so it was "
            "not started. A speech process that outlives the WebUI is not something this "
            "feature will leave running.")
    logger.info("Model Chain: the Sopro worker is in this process's job object and Windows "
                "will end it if this process is killed")
