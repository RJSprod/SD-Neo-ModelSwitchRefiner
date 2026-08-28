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
the words is the latency this feature would be judged on. Requests are
serialized by :data:`_lock` -- one in flight at a time, matching the worker's
single-threaded loop -- which is a statement about the *worker*, not about
Forge: an image generation and a transcription run at the same time and neither
waits for the other.
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
TTS_THREADS = 2
"""Conservative and hidden, per section 34. Written down once here so a V2
setting has one place to come from, and small on purpose: the point of running
beside Forge rather than inside its scheduler is lost if speech takes every
core from the image that is rendering."""

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


_lock = threading.RLock()
_process = None
_reader: threading.Thread | None = None
_replies: "queue.Queue" = queue.Queue()
_handshake: Handshake | None = None
_closing = False
_session = ""
_next_id = 0
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
    """
    with _lock:
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
    reply, _payload = _request({"op": "stt", "format": "wav"}, wav_bytes, STT_TIMEOUT)
    return {"text": str(reply.get("text") or ""),
            "audio_seconds": reply.get("audio_seconds"),
            "elapsed": reply.get("elapsed")}


def synthesize(text: str) -> bytes:
    """One completed reply to a WAV, in memory.

    ``text`` is an immutable snapshot the caller already took; nothing here
    re-reads a conversation, and nothing here writes the audio anywhere. The
    return value is the response body.
    """
    encoded = (text or "").encode("utf-8")
    if not encoded.strip():
        raise VoiceRuntimeError("There was nothing to read aloud.")
    if len(encoded) > MAX_TEXT_BYTES:
        raise VoiceRuntimeError(
            f"That reply is longer than Voice Chat will read aloud in one go "
            f"({len(encoded)} bytes; the limit is {MAX_TEXT_BYTES}). It was not spoken, and "
            f"nothing was cut short without telling you.")
    _reply, payload = _request({"op": "tts", "voice": "default"}, encoded, TTS_TIMEOUT)
    if not payload:
        raise VoiceRuntimeError("The voice runtime produced no audio.")
    return payload


def _request(header: dict, payload: bytes, timeout: float) -> tuple[dict, bytes]:
    """One round trip, with exactly one restart if the worker had died.

    "Exactly one" is the whole policy. A worker that vanished between two
    dictations is an ordinary thing -- a crash, a memory-hungry neighbour, an
    antivirus -- and making the user press the microphone again for it would be
    unkind. A worker that vanishes again on the retry is a broken installation,
    and the honest answer to that is an error rather than a third process.
    """
    for attempt in (1, 2):
        with _lock:
            if _closing:
                raise VoiceRuntimeError("Voice Chat is shutting down.")
            ensure_started()
            try:
                return _exchange(header, payload, timeout)
            except _WorkerGone:
                _discard("the voice worker stopped unexpectedly")
                if attempt == 2:
                    raise VoiceRuntimeError(
                        "Voice runtime stopped unexpectedly. Try again.") from None
    raise VoiceRuntimeError("Voice runtime stopped unexpectedly. Try again.")


class _WorkerGone(Exception):
    """The pipe ended or the worker did not answer in time."""


def _exchange(header: dict, payload: bytes, timeout: float) -> tuple[dict, bytes]:
    global _next_id

    _next_id += 1
    request_id = _next_id
    outgoing = dict(header)
    outgoing["id"] = request_id
    try:
        protocol.write_frame(_process.stdin, outgoing, payload)
    except (OSError, ValueError) as exc:
        raise _WorkerGone(str(exc)) from None

    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _WorkerGone("timed out")
        try:
            item = _replies.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            if _process is None or _process.poll() is not None:
                raise _WorkerGone("worker exited") from None
            continue
        if item is None:
            raise _WorkerGone("input closed") from None
        reply, body = item
        if reply.get("id") != request_id:
            # A late answer to a request that already timed out. Dropped rather
            # than returned: giving one caller another caller's transcript is
            # the one mistake this feature genuinely must not make.
            continue
        if not reply.get("ok"):
            raise VoiceRuntimeError(_readable(str(reply.get("error") or "")))
        return reply, body


def _readable(reason: str) -> str:
    """The worker's fault word, as a sentence for a status line."""
    known = {
        "audio is not mono": "The recording was not mono audio.",
        "audio is not 16-bit": "The recording was not 16-bit audio.",
        "audio is empty": "No audio was received.",
        "audio is too long": "That recording is longer than Voice Chat accepts.",
        "runtime is not initialised": "The voice runtime is not ready yet.",
        "unknown operation": "Voice Chat asked its runtime for something it does not do.",
    }
    return known.get(reason, "The voice runtime could not complete that request.")


# --------------------------------------------------------------------------- #
# Starting
# --------------------------------------------------------------------------- #


def ensure_started() -> None:
    """Start the worker if it is not running. Idempotent, and holds the lock.

    Every failure path below is written so that a process which has been started
    is stopped before its handle is dropped. That is the rule the orphaned
    llama-server broke: ownership begins at ``Popen``, not at the moment the
    handshake succeeds.
    """
    global _process, _reader, _handshake, _session, _replies

    with _lock:
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

        _session = uuid.uuid4().hex[:12]
        command = [str(interpreter), str(paths.worker_script()), protocol.MARKER,
                   "--parent-pid", str(os.getpid()), "--session", _session]
        environ = dict(os.environ)
        environ.update(models.worker_environment())

        started = None
        try:
            started = subprocess.Popen(  # noqa: S603 - a path this module built
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=environ, cwd=str(paths.extension_root()),
                bufsize=0, close_fds=True)
            _die_with_us(started)
            _process = started
            # A fresh queue per worker, so a late frame from a process that has
            # already been discarded cannot be handed to the next one's caller.
            _replies = queue.Queue()
            _reader = threading.Thread(target=_read_replies, args=(started, _replies),
                                       name="mc-voice-reader", daemon=True)
            _reader.start()
            threading.Thread(target=_drain_stderr, args=(started,),
                             name="mc-voice-stderr", daemon=True).start()
            _handshake = _handshake_with(started, state)
        except Exception as exc:
            # Whatever went wrong, the process this function started is this
            # function's to end. Nothing below this line may leave a handle.
            _note_failure()
            if started is not None:
                _process = started
                _discard("the voice worker failed to start")
            _handshake = None
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

        stop_on_exit()
        logger.info("Model Chain: Voice runtime ready — %s provider, STT threads %d, "
                    "TTS threads %d, containment %s, pid %s",
                    _handshake.provider, _handshake.stt_threads, _handshake.tts_threads,
                    _handshake.parent_death, started.pid)


def _handshake_with(started, state) -> Handshake:
    """Send ``init``, then believe nothing that comes back until it is checked.

    Four things are checked and each one is a different bug that has to fail
    closed: a protocol the parent cannot speak, a provider that is not CPU (I-1
    in the only place it can actually be observed), a containment mechanism this
    platform's support contract requires (R2-3), and a pair of model ids that
    are not the pair this installation verified.
    """
    config = {
        "stt": models.bundle_paths("stt"),
        "tts": models.bundle_paths("tts"),
        "stt_threads": STT_THREADS,
        "tts_threads": TTS_THREADS,
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
    return found


def _read_replies(started, replies) -> None:
    """Move frames off the pipe into :data:`_replies` until it ends.

    A thread rather than a blocking read in the requesting thread, so a wedged
    worker costs one request its timeout rather than costing the WebUI a thread
    forever.
    """
    stream = started.stdout
    try:
        while True:
            frame = protocol.read_frame(stream)
            if frame is None:
                break
            replies.put(frame)
    except Exception:
        logger.debug("Model Chain: the voice worker's pipe ended", exc_info=True)
    finally:
        replies.put(None)


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
    """Stop the worker if one is running. Idempotent, and never raises."""
    with _lock:
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
        with _lock:
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
    global _process, _reader, _handshake

    started, _process, _reader, _handshake = _process, None, None, None
    if started is None:
        return

    if started.poll() is None:
        try:
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
    _replies.put(None)
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
