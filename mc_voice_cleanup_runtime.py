"""Start the cleanup worker when it is wanted, and end it when it is not.

The third of three runtimes and the only one with a clock in it. Kokoro's and
Sopro's stay loaded because a reply arrives without warning and a cold load in
front of one is heard; nothing arrives without warning here. Somebody presses a
button, twenty seconds of audio goes through, and the engine has no further
reason to hold a Torch runtime -- so it does not.

The five doors, again
--------------------
A. the extension's unload callback
B. ``atexit``
C. chained SIGINT/SIGTERM
D. the pipe: the child sees end-of-input when this process dies
E. the OS: a Windows job object with ``KILL_ON_JOB_CLOSE``, and
   ``PR_SET_PDEATHSIG`` on Linux

Plus one this runtime has and the others do not: an idle timer, which is a
courtesy rather than a guarantee. The guarantee is E.

Where containment is proved
---------------------------
At the parent, immediately after the assignment, with
``IsProcessInJob(child, job)`` -- real handles on both sides, naming *this* job
rather than any job. The worker's own answer is corroboration and is logged when
it disagrees, never a veto. That is not the arrangement this feature shipped
with: it asked the child, the child could only ask the weaker question through a
pseudo-handle, and a worker whose containment was arranged and enforced was
refused for failing to confirm it. See ``docs/17-voice-chat-sopro.md``.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import subprocess
import threading
import time
import uuid

import mc_voice_paths as paths

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

START_TIMEOUT = 300.0
"""A cold Torch import plus a model load, on a CPU, from a cold file cache."""

CLEAN_TIMEOUT = 300.0
"""Twenty seconds of audio is seconds of work. This is the ceiling before the
worker is assumed wedged, not an expectation."""

STOP_TIMEOUT = 10.0

CONTAINMENT = {"windows": "job", "linux": "pdeathsig"}

JOB_KILL_ON_CLOSE = 0x00002000
JOB_EXTENDED_LIMIT_INFORMATION = 9

_lock = threading.RLock()
_process = None
_handshake = None
_job_handle = None
_idle_timer = None
_last_used = 0.0
_installed_hooks = False


class CleanupRuntimeError(RuntimeError):
    """The cleanup engine could not be used. Never fatal to a reply."""


def _protocol():
    from cleanup_worker import worker

    return worker


# --------------------------------------------------------------------------- #
# Starting
# --------------------------------------------------------------------------- #


def ensure_started() -> None:
    """Start the worker if it is not running, and prove its containment."""
    import mc_voice_cleanup as cleanup

    with _lock:
        if _process is not None and _process.poll() is None:
            return
        _discard("a previous cleanup worker had already exited")

        state = cleanup.status()
        if not state.ready:
            raise CleanupRuntimeError(
                "Recording cleanup is not installed. Install it in Settings → Voice Chat.")
        system, _machine, _python = _platform()
        if system not in CONTAINMENT:
            raise CleanupRuntimeError(
                "Recording cleanup has no tested process-containment mechanism on this "
                "platform, so it will not start a process here.")
        interpreter = cleanup.runtime_python()
        if interpreter is None:
            raise CleanupRuntimeError("The cleanup runtime is not installed.")

        environ = dict(os.environ)
        environ.update(cleanup.worker_environment())
        command = [str(interpreter), str(paths.cleanup_worker_script()),
                   _protocol().MARKER, "--model", str(paths.cleanup_model_root()),
                   "--parent-pid", str(os.getpid())]
        started = None
        try:
            started = subprocess.Popen(  # noqa: S603 - a path this module built
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=environ,
                cwd=str(paths.extension_root()), bufsize=0, close_fds=True)
            _die_with_us(started)
            globals()["_process"] = started
            threading.Thread(target=_drain_stderr, args=(started,),
                             name="mc-cleanup-stderr", daemon=True).start()
            globals()["_handshake"] = _handshake_with()
        except Exception:
            if started is not None:
                globals()["_process"] = started
                _discard("the cleanup worker failed to start")
            globals()["_handshake"] = None
            raise
        _install_hooks()
        _touch()
        logger.info("Model Chain: the cleanup engine is running — DeepFilterNet %s, "
                    "Torch %s, %s, pid %s", _handshake.get("deepfilternet"),
                    _handshake.get("torch"), _handshake.get("device"), started.pid)


def _platform():
    import mc_voice_models as models

    return models.current_platform()


def _die_with_us(started) -> None:
    """Door E on Windows, and the place containment is proved.

    The handle is held for the life of this process on purpose: what does the
    work is the handle being *closed*, which happens when this process ends
    however it ends, including a kill that runs no handler at all.
    """
    global _job_handle

    if os.name != "nt":
        return
    handle = getattr(started, "_handle", None)
    if handle is None:
        raise CleanupRuntimeError("The cleanup worker could not be tied to this process.")
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    # Declared rather than left to ctypes' defaults: a HANDLE is not a C ``int``
    # on 64-bit Windows, and the check below is exactly the shape that does not
    # survive being treated as one.
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
        if not kernel.SetInformationJobObject(job, JOB_EXTENDED_LIMIT_INFORMATION,
                                              ctypes.byref(information),
                                              ctypes.sizeof(information)):
            raise ctypes.WinError(ctypes.get_last_error())
        _job_handle = job

    if not kernel.AssignProcessToJobObject(_job_handle, wintypes.HANDLE(int(handle))):
        raise ctypes.WinError(ctypes.get_last_error())
    inside = wintypes.BOOL(0)
    if not kernel.IsProcessInJob(wintypes.HANDLE(int(handle)), _job_handle,
                                 ctypes.byref(inside)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not inside.value:
        raise CleanupRuntimeError(
            "Windows did not put the cleanup worker in this process's job object, so it was "
            "not started. A process that outlives the WebUI is not something this feature "
            "will leave running.")
    logger.info("Model Chain: the cleanup worker is in this process's job object and "
                "Windows will end it if this process is killed")


def _handshake_with() -> dict:
    """Start the model and check what came back. Five things, then it is used."""
    reply, _payload = _request({"op": "start", "parent_pid": os.getpid()}, b"",
                               START_TIMEOUT)
    protocol = _protocol()
    if int(reply.get("protocol_version") or 0) != protocol.PROTOCOL_VERSION:
        raise CleanupRuntimeError(
            f"The cleanup worker speaks protocol {reply.get('protocol_version')} and this "
            f"extension speaks {protocol.PROTOCOL_VERSION}. Reinstall it.")
    if str(reply.get("backend") or "") != "deepfilternet":
        raise CleanupRuntimeError(
            f"That worker reported backend {reply.get('backend')!r} rather than "
            f"DeepFilterNet, so it was not used.")
    if not str(reply.get("device") or "").startswith("cpu"):
        raise CleanupRuntimeError(
            f"The cleanup worker reported device {reply.get('device')!r}. It runs on the CPU "
            f"only and will not use a graphics device.")
    system, _machine, _python = _platform()
    wanted = CONTAINMENT.get(system)
    if system == "windows":
        # Proved in `_die_with_us`, against this job and this child's handle.
        # What the worker says is corroboration, logged when it disagrees.
        if reply.get("parent_death") != wanted:
            logger.info("Model Chain: the cleanup worker reported its containment as %r; the "
                        "job object was confirmed at this end, and that is what ends it",
                        reply.get("parent_death") or "unknown")
    elif wanted and reply.get("parent_death") != wanted:
        raise CleanupRuntimeError(
            "The cleanup worker could not be tied to this process's lifetime, so it was not "
            "started.")
    return dict(reply)


# --------------------------------------------------------------------------- #
# Using it
# --------------------------------------------------------------------------- #


def clean(pcm: bytes, rate: int) -> bytes:
    """Mono PCM16 in, mono PCM16 out, at the rate it came in at.

    Starts the worker if it is not running and restarts the idle clock, so the
    engine's lifetime is exactly the span somebody is using it plus a couple of
    quiet minutes.
    """
    if not pcm:
        raise CleanupRuntimeError("There was no audio to clean.")
    ensure_started()
    reply, payload = _request({"op": "clean", "rate": int(rate)}, pcm, CLEAN_TIMEOUT)
    if not reply.get("ok"):
        raise CleanupRuntimeError(str(reply.get("error") or "That recording could not be "
                                                           "cleaned."))
    _touch()
    logger.info("Model Chain: cleaned %.1f s of audio in %d ms",
                len(pcm) / 2.0 / max(1, int(rate)), int(reply.get("elapsed_ms") or 0))
    return payload


def _request(header: dict, payload: bytes, timeout: float):
    """One request, one reply, under this module's lock.

    Serialised rather than pipelined because there is one model and one caller:
    a second cleanup while the first is running would be two requests racing for
    one stdin, and the engine is not the bottleneck anybody is optimising.
    """
    protocol = _protocol()
    with _lock:
        started = _process
        if started is None or started.poll() is not None:
            raise CleanupRuntimeError("The cleanup engine is not running.")
        header = dict(header, id=uuid.uuid4().hex[:12])
        answer = {}

        def run():
            try:
                protocol.write_frame(started.stdin, header, payload)
                found = protocol.read_frame(started.stdout)
                answer["found"] = found
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                answer["error"] = exc

        worker = threading.Thread(target=run, name="mc-cleanup-request", daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            _discard("the cleanup worker stopped answering")
            raise CleanupRuntimeError("The cleanup engine stopped answering and was ended.")
        if answer.get("error") is not None:
            _discard("the cleanup worker could not be spoken to")
            raise CleanupRuntimeError(f"The cleanup engine could not be spoken to "
                                      f"({answer['error'].__class__.__name__}).")
        found = answer.get("found")
        if found is None:
            _discard("the cleanup worker closed its pipe")
            raise CleanupRuntimeError("The cleanup engine stopped before it answered.")
        reply, body = found
        if not reply.get("ok") and header.get("op") == "start":
            raise CleanupRuntimeError(str(reply.get("error") or "The cleanup engine could "
                                                                "not start."))
        return reply, body


# --------------------------------------------------------------------------- #
# Stopping
# --------------------------------------------------------------------------- #


def _touch() -> None:
    """Restart the idle clock. Called after every use."""
    global _idle_timer, _last_used

    _last_used = time.monotonic()
    if _idle_timer is not None:
        return
    _idle_timer = threading.Thread(target=_watch_idle, name="mc-cleanup-idle", daemon=True)
    _idle_timer.start()


def _watch_idle() -> None:
    """End the worker once it has been idle long enough.

    A daemon thread with a coarse tick rather than a timer per request: the
    thing being measured is minutes, and a thread that wakes twice a second to
    look at a float costs nothing next to the Torch runtime it is there to
    release.
    """
    global _idle_timer

    idle = float(getattr(_protocol(), "IDLE_SECONDS", 120.0))
    try:
        while True:
            time.sleep(0.5)
            with _lock:
                if _process is None or _process.poll() is not None:
                    return
                if time.monotonic() - _last_used < idle:
                    continue
                logger.info("Model Chain: the cleanup engine has been idle for %.0f s and "
                            "was stopped", idle)
                _stop_locked("it had nothing to do")
                return
    finally:
        _idle_timer = None


def shutdown() -> None:
    """Doors A, B and C. Idempotent, and must never raise."""
    try:
        with _lock:
            _stop_locked("the WebUI is shutting down")
    except Exception:
        logger.debug("Model Chain: the cleanup engine could not be stopped cleanly",
                     exc_info=True)


def _stop_locked(why: str) -> None:
    started = _process
    if started is None:
        return
    if started.poll() is None:
        try:
            protocol = _protocol()
            protocol.write_frame(started.stdin, {"id": "stop", "op": "stop"})
            started.wait(timeout=STOP_TIMEOUT)
        except Exception:
            logger.debug("Model Chain: the cleanup worker did not stop when asked",
                         exc_info=True)
    _discard(why)


def _discard(why: str) -> None:
    """End the process and forget it. Never leaves a handle behind."""
    global _process, _handshake

    started = _process
    _process = None
    _handshake = None
    if started is None:
        return
    try:
        if started.poll() is None:
            started.kill()
            started.wait(timeout=STOP_TIMEOUT)
    except Exception:
        logger.debug("Model Chain: the cleanup worker could not be killed", exc_info=True)
    for stream in (started.stdin, started.stdout, started.stderr):
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass
    logger.info("Model Chain: cleanup worker stopped — %s", why)


def _drain_stderr(started) -> None:
    """Log the worker's own notes. A pipe nobody reads fills, and a child
    blocked writing to a full stderr is a child that has stopped answering for a
    reason nothing here can see."""
    try:
        for line in iter(started.stderr.readline, b""):
            text = line.decode("utf-8", "replace").strip()
            if text:
                logger.info("Model Chain: %s", text)
    except Exception:
        pass


def _install_hooks() -> None:
    """Doors B and C, once per process."""
    global _installed_hooks

    if _installed_hooks:
        return
    _installed_hooks = True
    atexit.register(shutdown)
    for number in (signal.SIGINT, signal.SIGTERM):
        try:
            previous = signal.getsignal(number)
        except (ValueError, OSError):
            continue

        def chained(signum, frame, previous=previous):
            shutdown()
            if callable(previous):
                previous(signum, frame)
            elif previous == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        try:
            signal.signal(number, chained)
        except (ValueError, OSError):
            # Not the main thread, which is where Forge sometimes imports from.
            # Doors A, D and E are unaffected.
            logger.debug("Model Chain: could not chain signal %s for cleanup", number)


def running() -> bool:
    with _lock:
        return _process is not None and _process.poll() is None


def status() -> dict:
    """What the settings row draws. Reads nothing off disk and starts nothing."""
    with _lock:
        alive = _process is not None and _process.poll() is None
        idle = max(0.0, time.monotonic() - _last_used) if alive else 0.0
        return {
            "running": alive,
            "idle_seconds": round(idle, 1),
            "stops_after": float(getattr(_protocol(), "IDLE_SECONDS", 120.0)),
            "deepfilternet": (_handshake or {}).get("deepfilternet") or "",
            "torch": (_handshake or {}).get("torch") or "",
            "device": (_handshake or {}).get("device") or "",
        }
