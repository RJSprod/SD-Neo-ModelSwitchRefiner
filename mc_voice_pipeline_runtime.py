"""The enhancement worker's lifetime, its one queue, and its five doors.

Starts the process :mod:`mc_voice_pipeline` installed, proves it cannot outlive
this one, hands it verified local model directories, and pumps one turn's PCM
through it. Nothing in this file imports an enhancement model and nothing in it
touches the network.

Where this sits
---------------
Between a speech engine's finalised PCM and :meth:`mc_voice_turn.VoiceTurn.offer_audio`,
and nowhere else::

    Pocket worker  →  parent reader thread
                          │
                          ├── turn is draining ─────────────→ read and dropped
                          │                                    (never reaches here)
                          └── turn is speaking
                                  │
                    pipeline off  ├────────────────────────→ VoiceTurn.offer_audio
                                  │
                    pipeline on   └→ bounded ingress → pump → worker → reader
                                                                          │
                                                                          ↓
                                                             VoiceTurn.offer_audio

The branch above the ingress is the one that matters, and it is not in this
file: :func:`mc_voice_pocket_runtime._dispatch_turn` already drops a draining
turn's audio before any destination sees it. That is what makes I-VP-18 true by
structure rather than by care -- a Pocket unit that must be allowed to finish
after Stop cannot block on this queue, because its frames never arrive at it.

One queue, and why it is not a prebuffer
----------------------------------------
There is exactly one queue here, it is bounded, and it is a *backpressure*
limit rather than a start target (I-VP-17). Nothing waits for it to fill.
The first finalised sample the worker produces is forwarded to the VoiceTurn
immediately, and the browser's own adaptive buffer -- 0.7 s to start, 0.4 s
floor, 2.0 s ceiling -- remains the playback authority it has been since PR
#124. The earlier draft of this feature proposed a fixed 1.5 s server reservoir
in front of that; it is not here, and a future one would undo two PRs of
measured latency work.

What the queue is for is the other direction. If enhancement falls behind
production for long enough to fill two seconds, the pump stops taking blocks,
the Pocket reader thread waits, the pipe behind it fills, and the source slows
down. That is a real-time factor problem being made visible instead of being
hidden in a buffer that grows until something runs out of memory.

The doors
---------
The same five the other workers hold, and the couplings section 12 asks for:
Pocket's residency group loads this and unloads this, a graceful WebUI exit
stops it, a hard kill cannot leave it behind, and switching away from Pocket
takes it with the engine it was polishing.
"""

from __future__ import annotations

import collections
import logging
import os
import subprocess
import threading
import time
import uuid

import mc_voice_paths as paths

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

START_TIMEOUT = 180.0
"""How long the worker gets to answer a frame that may have to read a model.

Used for the handshake, which is quick, and for ``load``, which is not: loading
is where two inference sessions are read off a disk that may be spinning and may
be busy with an image model's weights. One number for both because the pair is
one start sequence, and the slow half is the one worth sizing for."""

REQUEST_TIMEOUT = 30.0
"""How long a request that expects a reply may wait.

Every one of them -- the handshake, the load, opening a turn -- happens while no
audio is flowing. *Ending* a turn deliberately expects no reply at all, because
the only thread that could deliver one is at that moment inside playback
backpressure; see :func:`_finish`.
"""

FLUSH_IDLE = 60.0
"""How long a turn may be finishing with nothing happening at all.

Not how long it may take to finish -- see :func:`_watch_flush` for why those are
different numbers and only one of them can honestly be bounded."""

STOP_GRACE = 3.0
TERMINATE_GRACE = 2.0
PUMP_POLL = 0.02

INGRESS_SECONDS = 2.0
"""How much source audio may be waiting for enhancement before the source waits.

A capacity, not a target, and the difference is the whole of section 11.4.
Nothing anywhere waits for this to reach any level; it is emptied as fast as the
worker will take it. Two seconds is large enough to absorb the scheduling jitter
of a machine also generating an image and small enough that a sustained
real-time factor above 1.0 backpressures the source within a couple of seconds
instead of growing a deficit nobody can see.
"""

CONTAINMENT = {"windows": "job", "linux": "pdeathsig"}
"""The tested parent-death mechanisms, and the only platforms this will start a
process on. An untested containment story for a fourth platform would be a
promise this feature has no way to keep, and section 12.10 says to follow the
repository's refusal policy rather than claim a weaker guarantee."""

JOB_KILL_ON_CLOSE = 0x00002000
JOB_EXTENDED_LIMIT_INFORMATION = 9

_lock = threading.RLock()
"""The state lock, and it is never held across anything that waits.

Held to read or write the handful of module globals below and let go before any
pipe write, any handshake and any inference. That is not tidiness: the reader
thread takes this lock to find the turn a frame belongs to, so a start sequence
that held it while waiting for the worker's first reply would be a start
sequence waiting for a thread that is waiting for it.
"""

_start_lock = threading.Lock()
"""Serialises starting, which the state lock deliberately cannot do because it
is let go in the middle."""

_write_lock = threading.Lock()
_process = None
_handshake = None
_job_handle = None
_generation = 0
_reader = None
_loaded = ()
_turns = {}
_failures = collections.deque(maxlen=8)
_closing = False


class PipelineRuntimeError(RuntimeError):
    """The Voice Pipeline could not be used. Never fatal to a reply.

    Every caller of everything here either falls back to unenhanced speech --
    when nothing has been played yet and the format is still open -- or ends
    that one turn cleanly. A reply that is not polished is a disappointment; a
    reply that does not happen because an optional polish failed is a bug
    (section 15).
    """


def _protocol():
    from pipeline_worker import worker

    return worker


def status() -> dict:
    """What the settings surface and the flyout draw about the running worker.

    Numbers, enums and one PID. No path, no token, no model file name -- the
    same bar every other runtime status in this feature clears (I-VP-26).
    """
    with _lock:
        started = _process
        found = dict(_handshake or {})
        stages = tuple(_loaded)
        turns = len(_turns)
    running = started is not None and started.poll() is None
    return {
        "runtime_state": ("loaded" if running and not turns
                          else "busy" if running else "unloaded"),
        "loaded_stages": list(stages),
        "pid": started.pid if running else 0,
        "protocol_version": int(found.get("protocol_version") or 0),
        "parent_death": str(found.get("parent_death") or ""),
        "restarts": len(_failures),
    }


def running() -> bool:
    with _lock:
        return _process is not None and _process.poll() is None


# --------------------------------------------------------------------------- #
# Starting
# --------------------------------------------------------------------------- #


def ensure_started(stages=()) -> None:
    """Start the worker with exactly these stages, and prove its containment.

    Idempotent when the stages already loaded are the stages asked for. A
    different set is a reconfiguration, which is only allowed between turns and
    is checked at both ends.
    """
    import mc_voice_pipeline as pipeline

    wanted = tuple(sorted(str(name) for name in stages))
    with _lock:
        started = _process
        alive = started is not None and started.poll() is None
        if alive and tuple(sorted(_loaded)) == wanted:
            return

    with _start_lock:
        with _lock:
            if _closing:
                # A shutdown is in flight. Starting a process now would be a
                # process started after the door it was supposed to leave by.
                raise PipelineRuntimeError("The Voice Pipeline is shutting down.")
            started = _process
            alive = started is not None and started.poll() is None
            same = tuple(sorted(_loaded)) == wanted
            speaking = bool(_turns)
        if alive and same:
            return
        if alive:
            if speaking:
                raise PipelineRuntimeError(
                    "The Voice Pipeline's stages cannot change while a reply is being "
                    "spoken.")
            _send_load(wanted)
            return
        if started is not None:
            _discard("a previous Voice Pipeline worker had already exited")

        if not wanted:
            raise PipelineRuntimeError("No Voice Pipeline stage was selected.")
        state = pipeline.status()
        if not state.runtime_ready:
            raise PipelineRuntimeError(state.runtime_message)
        for name in wanted:
            held = state.stage(name)
            if held is None or held.install_state != "installed":
                raise PipelineRuntimeError(
                    f"{held.label if held else name} is not installed.")
        system, _machine, _python = _platform()
        if system not in CONTAINMENT:
            raise PipelineRuntimeError(
                "The Voice Pipeline has no tested process-containment mechanism on this "
                "platform, so it will not start a process here.")
        interpreter = pipeline.runtime_python()
        if interpreter is None:
            raise PipelineRuntimeError("The Voice Pipeline runtime is not installed.")

        environ = dict(os.environ)
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            # Removed rather than merely not added. The parent's own environment
            # may carry one -- somebody who set HF_TOKEN in the shell they start
            # the WebUI from -- and inheriting it would put a credential inside
            # a process this feature promises has none (I-VP-21, section 13.7).
            environ.pop(name, None)
        environ.update(pipeline.worker_environment())
        command = [str(interpreter), str(paths.pipeline_worker_script()),
                   _protocol().MARKER, "--parent-pid", str(os.getpid())]
        started = None
        try:
            started = subprocess.Popen(  # noqa: S603 - a path this module built
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=environ,
                cwd=str(paths.extension_root()), bufsize=0, close_fds=True)
            _die_with_us(started)
            with _lock:
                globals()["_process"] = started
                globals()["_generation"] += 1
                generation = _generation
            threading.Thread(target=_drain_stderr, args=(started,),
                             name="mc-pipeline-stderr", daemon=True).start()
            # The reader starts BEFORE the handshake, not after it. Every reply
            # this module waits for is delivered by that thread, the handshake's
            # included -- start it afterwards and the first frame is a wait on
            # an event nothing is going to set, which is a three-minute pause
            # rather than a refusal.
            reader = threading.Thread(target=_read_frames, args=(started, generation),
                                      name="mc-pipeline-reader", daemon=True)
            globals()["_reader"] = reader
            reader.start()
            globals()["_handshake"] = _handshake_with(started)
            _send_load(wanted)
        except Exception:
            if started is not None:
                with _lock:
                    globals()["_process"] = started
                _discard("the Voice Pipeline worker failed to start")
            with _lock:
                globals()["_handshake"] = None
            raise
        logger.info("Model Chain: the Voice Pipeline is running — %s, pid %s",
                    ", ".join(wanted) or "no stages", started.pid)


def _platform():
    import mc_voice_models as models

    return models.current_platform()


def _die_with_us(started) -> None:
    """Door E on Windows, and the place containment is proved rather than asked.

    The handle is held for the life of this process on purpose: what does the
    work is the handle being *closed*, which happens when this process ends
    however it ends, including a kill that runs no handler at all.
    """
    global _job_handle

    if os.name != "nt":
        return
    handle = getattr(started, "_handle", None)
    if handle is None:
        raise PipelineRuntimeError(
            "The Voice Pipeline worker could not be tied to this process.")
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
        raise PipelineRuntimeError(
            "Windows did not put the Voice Pipeline worker in this process's job object, "
            "so it was not started. A process that outlives the WebUI is not something this "
            "feature will leave running.")
    logger.info("Model Chain: the Voice Pipeline worker is in this process's job object "
                "and Windows will end it if this process is killed")


def _handshake_with(started) -> dict:
    """Start the worker and check what came back. Three refusals, then it is used.

    A protocol this build does not speak, a device that is not the CPU, and no
    containment evidence where the parent could not arrange it itself. What the
    worker is *not* asked about here is its models: it has not been told to load
    any yet, and ``load`` is where that is checked.
    """
    protocol = _protocol()
    reply = _exchange(started, {"op": "start", "parent_pid": os.getpid()}, b"",
                      START_TIMEOUT)
    if int(reply.get("protocol_version") or 0) != protocol.PROTOCOL_VERSION:
        raise PipelineRuntimeError(
            f"The Voice Pipeline worker speaks protocol {reply.get('protocol_version')} "
            f"and this extension speaks {protocol.PROTOCOL_VERSION}. Reinstall it.")
    if not str(reply.get("device") or "").startswith("cpu"):
        raise PipelineRuntimeError(
            f"The Voice Pipeline worker reported device {reply.get('device')!r}. It runs "
            f"on the CPU only and will not use a graphics device.")
    system, _machine, _python = _platform()
    wanted = CONTAINMENT.get(system)
    if system == "windows":
        # Proved in :func:`_die_with_us`, against this job and this child's
        # handle. What the worker says is corroboration, logged when it
        # disagrees rather than trusted over the parent's own check.
        if reply.get("parent_death") != wanted:
            logger.info("Model Chain: the Voice Pipeline worker reported its containment "
                        "as %r; the job object was confirmed at this end, and that is what "
                        "ends it", reply.get("parent_death") or "unknown")
    elif wanted and reply.get("parent_death") != wanted:
        raise PipelineRuntimeError(
            "The Voice Pipeline worker could not be tied to this process's lifetime, so it "
            "was not started.")
    return dict(reply)


def _send_load(stages) -> None:
    """Tell the worker which stages to hold open, with local paths and contracts.

    Paths come from the installed records, never from anything a browser sent
    (section 20.2). The browser names a stage; this turns that name into a
    directory; nothing in between accepts a filesystem path from outside.
    """
    import mc_voice_pipeline as pipeline

    roots = pipeline.stage_paths()
    config = {}
    for name in stages:
        found = dict(pipeline.stage_config(name))
        found["intraop"] = pipeline.threads()
        found["interop"] = pipeline.INTEROP_THREADS
        config[name] = found
    with _lock:
        started = _process
    if started is None:
        raise PipelineRuntimeError("The Voice Pipeline is not running.")
    # Outside the state lock, for the reason the lock's own docstring gives: the
    # reply to this comes back through the reader thread.
    reply = _exchange(started, {"op": "load", "stages": list(stages),
                                "paths": {name: roots[name] for name in stages
                                          if name in roots},
                                "config": config}, b"", START_TIMEOUT)
    if not reply.get("ok"):
        raise PipelineRuntimeError(str(reply.get("error")
                                       or "The Voice Pipeline could not load its models."))
    globals()["_loaded"] = tuple(reply.get("loaded") or stages)


# --------------------------------------------------------------------------- #
# One turn
# --------------------------------------------------------------------------- #


class Handle:
    """One VoiceTurn's ingress, its counters, and its half of the clock.

    Held on the turn itself as ``turn.pipeline`` so that the engine's reader
    thread can find it with a ``getattr`` and no import. That is not
    micro-optimisation: it is what makes the pipeline-off path in
    :func:`mc_voice_pocket_runtime._dispatch_turn` a single attribute read
    rather than a module lookup and a branch on global state (I-VP-03).
    """

    def __init__(self, turn, snapshot, generation: int):
        self.turn = turn
        self.snapshot = snapshot
        self.generation = generation
        self.id = turn.id
        self.input_samples = 0
        self.sent_samples = 0
        self.output_samples = 0
        self.input_packets = 0
        self.output_packets = 0
        self.peak_queue = 0
        self.backpressure = 0.0
        self.opened = time.monotonic()
        self.first_input = 0.0
        self.first_output = 0.0
        self.ended = False
        self.failed = ""
        self.delivering = False
        """Whether the reader thread is inside playback right now.

        Read by the flush watchdog, and it is the difference between the two
        reasons a turn can take a long time to finish. A listener whose buffer
        is full parks the reader inside ``offer_audio`` -- healthy, and the turn
        completes the moment they start listening again. A wedged inference
        parks nothing and simply never answers, and that is the one the
        watchdog is for.
        """

        self.on_finished = None
        """What closes the browser's stream, run exactly once when this turn is
        done however it ends -- flushed, failed, cancelled or unloaded."""

        self.measured = {}
        """What the worker measured for this turn, read out of ``turn_flushed``.

        Kept because the parent cannot compute it: the compute time and the
        real-time factor of the stages are wall-clock inside the other process,
        and a number this side invented from its own timings would be measuring
        the pipe as much as the models (section 17.7).
        """
        self._queue = collections.deque()
        self._queued = 0
        self._limit = max(1, int(INGRESS_SECONDS * (snapshot.input_rate or 24000)))
        self._gate = threading.Condition()
        self._closed = False

    # -- the producer side ------------------------------------------------ #

    def offer(self, pcm: bytes, rate: int) -> bool:
        """One block of finalised source PCM, from the engine's reader thread.

        Blocks while the ingress is full, which is the point: the pipe behind
        this thread fills, the source worker's writer stops, and the engine
        stops producing faster than the pipeline can take it. It never blocks
        past a cancellation -- the wait is a loop around the turn's own event --
        so a Stop pressed while enhancement is behind frees this thread at once.

        Returns False when the block was discarded, which is what a cancelled or
        already-ended turn does with one.
        """
        if self._closed or self.turn.cancelled.is_set():
            return False
        if not pcm:
            return True
        if rate and int(rate) != self.snapshot.input_rate:
            # The rate changed under a turn that has already advertised an
            # output rate to a browser. There is no honest way to continue, so
            # the turn ends rather than being quietly played at the wrong speed
            # (I-VP-07).
            self.fail("the source changed sample rate in the middle of a reply")
            return False
        samples = len(pcm) // 2
        waited = time.monotonic()
        with self._gate:
            while (self._queued + samples > self._limit and self._queued
                   and not self._closed and not self.turn.cancelled.is_set()):
                self._gate.wait(PUMP_POLL)
            if self._closed or self.turn.cancelled.is_set():
                return False
            if not self.first_input:
                self.first_input = time.monotonic()
            self._queue.append(pcm)
            self._queued += samples
            self.input_samples += samples
            self.input_packets += 1
            self.peak_queue = max(self.peak_queue, self._queued)
            self._gate.notify_all()
        held = time.monotonic() - waited
        if held > PUMP_POLL:
            self.backpressure += held
        return True

    # -- the pump side ---------------------------------------------------- #

    def take(self):
        """One queued block for the pump, or ``None`` when there is none yet."""
        with self._gate:
            if not self._queue:
                return None
            pcm = self._queue.popleft()
            self._queued -= len(pcm) // 2
            self._gate.notify_all()
            return pcm

    def wait_for_work(self, seconds: float) -> bool:
        """Sleep until there is a block or this turn is over. False when over."""
        with self._gate:
            if self._queue:
                return True
            if self._closed:
                return False
            self._gate.wait(seconds)
            return not self._closed

    def pending(self) -> int:
        """How many blocks are still waiting to be written to the worker."""
        with self._gate:
            return 0 if self._closed else len(self._queue)

    def close(self, reason: str = "") -> None:
        """Stop accepting, wake anything waiting, and drop what is queued."""
        with self._gate:
            self._closed = True
            self._queue.clear()
            self._queued = 0
            if reason:
                self.failed = self.failed or reason
            self._gate.notify_all()

    def fail(self, reason: str) -> None:
        """End this turn on an enhancement failure, honestly.

        After bytes have been played there is no fallback: splicing unenhanced
        24 kHz into a 48 kHz stream would change the rate mid-response, and
        continuing with the last processed block would be a stutter presented as
        speech. So the turn ends and the next one recovers (section 15.2).
        """
        self.failed = self.failed or reason
        self.close(reason)
        try:
            if self.output_samples:
                self.turn.audio_failed("The Voice Pipeline stopped while it was speaking.")
            self.turn.cancel("pipeline")
        except Exception:
            logger.debug("Model Chain: could not end a turn the Voice Pipeline failed on",
                         exc_info=True)
        # And let go of it. "Failed" means this turn is over, so the handle
        # leaves the register and whatever was waiting to close the stream runs
        # -- on this path as surely as on the one where everything worked.
        _release(self)

    def metrics(self) -> dict:
        """This turn's enhancement numbers, beside the source's own rather than
        instead of them (I-VP-32, section 17.2)."""
        rate = float(self.snapshot.input_rate or 1)
        return {
            "pipeline_stages": ",".join(self.snapshot.stage_ids),
            "pipeline_input_rate": self.snapshot.input_rate,
            "pipeline_output_rate": self.snapshot.output_rate,
            "pipeline_input_sample_count": self.input_samples,
            "pipeline_output_sample_count": self.output_samples,
            "pipeline_input_packet_count": self.input_packets,
            "pipeline_output_packet_count": self.output_packets,
            "pipeline_first_output_ms": (int((self.first_output - self.first_input) * 1000)
                                         if self.first_output and self.first_input else 0),
            "pipeline_ingress_peak_ms": int(1000 * self.peak_queue / rate),
            "pipeline_backpressure_ms": int(self.backpressure * 1000),
            "pipeline_compute_ms": int(self.measured.get("compute_ms") or 0),
            # A thousandth rather than a float, so a telemetry field stays an
            # integer like every other one: 700 is a real-time factor of 0.7,
            # and the target is comfortably under 1000 (section 11.7).
            "pipeline_rtf_milli": int(self.measured.get("rtf_milli") or 0),
            "pipeline_window_count": int(self.measured.get("lava_window_count") or 0),
            "pipeline_correction_count": int(
                self.measured.get("lava_correction_count") or 0)
            + int(self.measured.get("dpdf_correction_count") or 0),
            "dpdfnet_rtf_milli": int(self.measured.get("dpdfnet_rtf_milli") or 0),
            "lavasr_rtf_milli": int(self.measured.get("lavasr_rtf_milli") or 0),
        }


def begin_turn(turn, input_rate: int, snapshot) -> int:
    """Open a pipeline turn, and answer the rate the response will advertise.

    Called from the engine's ``begin_turn`` before any text has been sent and
    long before any audio exists, because the answer goes into the response
    headers and cannot change afterwards (I-VP-07, section 8.5).

    A failure here is not fatal and does not cost the reply. The turn simply has
    no pipeline handle, the engine's own rate is returned, and the reply is
    spoken unenhanced -- which is a legitimate outcome precisely because the
    format has not been committed to yet (section 15.1).
    """
    if snapshot is None or not snapshot.active:
        return int(input_rate)
    try:
        _await_free(turn)
        ensure_started(snapshot.stage_ids)
        with _lock:
            started = _process
            generation = _generation
        if started is None:
            raise PipelineRuntimeError("The Voice Pipeline is not running.")
        reply = _exchange(started, {"op": "turn_begin", "turn": turn.id,
                                    "sample_rate": int(input_rate), "channels": 1,
                                    "sample_format": "pcm16",
                                    "stages": list(snapshot.stage_ids)},
                          b"", REQUEST_TIMEOUT)
        if not reply.get("ok"):
            raise PipelineRuntimeError(str(reply.get("error") or "unknown"))
        advertised = int(reply.get("output_sample_rate") or 0)
        if advertised != snapshot.output_rate:
            # The worker and the snapshot disagree about what comes out. Both
            # numbers are named, because a message that says only "wrong rate"
            # is a message nobody can act on.
            raise PipelineRuntimeError(
                f"The Voice Pipeline offered {advertised} Hz where this build expects "
                f"{snapshot.output_rate} Hz.")
    except Exception as exc:
        logger.info("Model Chain: this reply is being spoken without the Voice Pipeline — "
                    "%s", _readable(exc))
        _note_failure()
        return int(input_rate)

    handle = Handle(turn, snapshot, generation)
    with _lock:
        _turns[turn.id] = handle
    turn.pipeline = handle
    threading.Thread(target=_pump, args=(handle,), name="mc-pipeline-pump",
                     daemon=True).start()
    logger.info("Model Chain: the Voice Pipeline is enhancing this reply — %s",
                snapshot.describe("PocketTTS"))
    return int(snapshot.output_rate)


def _await_free(turn) -> None:
    """Wait for the previous reply's flush to finish before opening a new one.

    The worker holds one turn, because there is one speaking lane upstream. The
    engine frees that lane the moment *it* is done, which is a little before the
    pipeline is: LavaSR is one analysis window behind by construction, so the
    previous reply's ``turn_end`` may still be in flight when the next reply's
    ``turn_begin`` wants to go down the same pipe. Without this wait the new
    turn replaces the old one inside the worker, the old ``turn_end`` is refused
    for a turn that is no longer there, and the previous reply loses its last
    few hundred milliseconds to a race nobody would ever reproduce twice.

    Short by construction -- what it is waiting for is one window's flush, not
    an inference -- and it wakes on the new turn's own cancellation, so a Stop
    pressed while a new reply is waiting does not have to wait for the old one
    too. The bound is a failsafe rather than a policy: if it expires, the turn
    goes ahead unenhanced rather than not at all.
    """
    deadline = time.monotonic() + REQUEST_TIMEOUT
    while time.monotonic() < deadline:
        with _lock:
            busy = bool(_turns)
        if not busy:
            return
        if turn is not None and turn.cancelled.is_set():
            return
        time.sleep(PUMP_POLL)
    logger.info("Model Chain: a Voice Pipeline turn was still finishing when the next "
                "reply began")


def finish_turn(turn, then=None) -> None:
    """The source's audio is complete. Flush the stages, then let the turn end.

    ``then`` is what closes the browser's stream -- ``VoiceTurn.audio_finished``
    -- and it is a callback rather than something the caller runs afterwards
    because the ordering is load-bearing. That method puts the end-of-stream
    sentinel into the playback queue immediately, so anything the pipeline was
    still holding -- one analysis window, up to a few hundred milliseconds --
    would arrive after the stream had already been closed and be the last words
    of the reply, missing.

    With no pipeline on this turn, ``then`` runs here and now on the caller's
    own thread, so the unenhanced path is the call it always was (I-VP-03).
    With one, the flush happens on a thread of its own: the caller is the
    engine's single reader thread, and that thread is also the one that reads a
    cancelled unit's drain, so it is the one thread in this feature that must
    never wait for an inference (I-VP-18).

    ``TURN_END`` means the *turn's* source audio is over. It is never sent at a
    synthesis-unit boundary, however many extra blocks a source worker's trim
    and seam emit when a unit resolves: to this feature those are ordinary
    samples in one continuous stream (I-VP-11, section 7.7).
    """
    handle = getattr(turn, "pipeline", None)
    if handle is None:
        if then is not None:
            then()
        return
    handle.on_finished = then
    threading.Thread(target=_finish, args=(handle,), name="mc-pipeline-flush",
                     daemon=True).start()


def _finish(handle) -> None:
    """Send the end of the turn. The answer arrives on the reader, not here.

    Fire and forget, deliberately, and it is the one place this module does not
    wait for a reply. ``turn_flushed`` comes back down the same pipe as the
    audio, *behind* every remaining ``audio_out`` frame, so it cannot arrive
    before the reply's last sample has been handed to playback -- and waiting for
    it here would mean waiting on the reader thread, which is at that moment
    inside :meth:`mc_voice_turn.VoiceTurn.offer_audio` applying exactly the
    backpressure it is supposed to apply.

    That coupling was real and it was reproducible. A listener whose buffer is
    full -- a phone with the page in the background, which stops the browser
    reading -- parks the reader; the reply this module was waiting for sits
    unread in the pipe behind the audio; the wait expires; and a reply the
    worker had processed perfectly is truncated and reported as an error.
    Removing the wait removes the failure, because the frame was already ordered
    correctly and reading it in order is all that was ever needed.
    """
    try:
        _drain(handle)
        with _lock:
            started = _process
        if started is None:
            raise PipelineRuntimeError("The Voice Pipeline stopped before the reply ended.")
        _write(started, {"op": "turn_end", "turn": handle.id,
                         "final_input_sample_count": handle.sent_samples}, b"")
        _watch_flush(handle)
    except Exception as exc:
        handle.fail(_readable(exc))


def _watch_flush(handle) -> None:
    """Wait for the turn to end, and end it ourselves if nothing is happening.

    Bounded by *inactivity* rather than by elapsed time, because the two long
    waits here are not the same thing. A listener with a full buffer can hold
    the reader inside playback for as long as they are not listening, and that
    is a healthy stream finishing at the speed somebody is hearing it; a worker
    stuck inside an inference that will not return holds nothing and answers
    nothing. So this measures whether anything is *moving* -- a sample
    delivered, or the reader parked in playback -- and only gives up when
    neither has been true for a minute.

    Giving up is not a Stop: the turn is already over at the source, and what
    this prevents is a browser left holding a response nothing will ever close.

    A turn cancelled while this is watching needs no special case, and used to
    have one. Cancelling makes ``offer_audio`` return at once, so the reader
    drains what is left in moments and ``turn_flushed`` arrives behind it and
    ends the loop -- and in the one case where it does not, because the worker
    is wedged as well, leaving early would have been the only path in this
    module that lets go of a turn without letting go of its handle.
    """
    idle = 0.0
    seen = -1
    while not handle.ended and idle < FLUSH_IDLE:
        time.sleep(PUMP_POLL)
        moved = handle.output_samples != seen or handle.delivering
        seen = handle.output_samples
        idle = 0.0 if moved else idle + PUMP_POLL
    if not handle.ended:
        handle.fail("the Voice Pipeline stopped answering at the end of a reply")


def _flushed(handle, reply: dict) -> None:
    """The worker's own account of a finished turn, checked against ours.

    Reached from the reader thread, in pipe order, after every sample of this
    turn has been offered to playback. Two counts have to agree: what the worker
    says it produced and what actually arrived here. A disagreement is the clock
    contract broken, and it ends the turn rather than playing it.
    """
    try:
        if not reply.get("ok"):
            raise PipelineRuntimeError(str(reply.get("error") or "unknown"))
        handle.measured = {name: reply.get(name) for name in
                           ("compute_ms", "rtf_milli", "first_output_ms",
                            "lava_window_count", "lava_correction_count",
                            "dpdf_correction_count", "dpdfnet_compute_ms",
                            "dpdfnet_rtf_milli", "lavasr_compute_ms",
                            "lavasr_rtf_milli",
                            "input_packet_count", "output_packet_count")
                           if reply.get(name) is not None}
        wanted = int(reply.get("final_output_sample_count") or 0)
        if wanted != handle.output_samples:
            raise PipelineRuntimeError(
                f"The Voice Pipeline says it produced {wanted} samples and "
                f"{handle.output_samples} arrived.")
        logger.info("Model Chain: the Voice Pipeline finished a reply — %d source samples "
                    "at %d Hz became %d at %d Hz",
                    handle.sent_samples, handle.snapshot.input_rate,
                    handle.output_samples, handle.snapshot.output_rate)
    except Exception as exc:
        handle.fail(_readable(exc))
    finally:
        _release(handle)


def cancel_turn(turn) -> None:
    """Stop enhancing now, drop what is queued, and emit no tail.

    Returns immediately and never waits for inference. If a model call cannot be
    interrupted, its result is discarded when it returns; the browser is already
    silent and a Stop that waited for a solver would be a Stop that felt broken
    (section 16.3).
    """
    handle = getattr(turn, "pipeline", None)
    if handle is None:
        return
    handle.close("cancelled")
    try:
        with _lock:
            started = _process
        if started is not None:
            _write_control(started, {"op": "turn_cancel", "turn": handle.id})
    except Exception:
        logger.debug("Model Chain: could not tell the Voice Pipeline to stop a turn",
                     exc_info=True)
    _release(handle)


def _release(handle) -> None:
    """Let go of the handle, keeping its numbers on the turn it belonged to.

    The numbers are copied across before the handle is dropped because the turn
    outlives it: a reply that finished cleanly is asked how it went *after* its
    last sample, and a measurement that went out with the handle would be one
    only a failure ever reported.
    """
    with _lock:
        first = not handle.ended
        handle.ended = True
        if _turns.get(handle.id) is handle:
            _turns.pop(handle.id, None)
    if not first:
        return
    try:
        handle.turn.pipeline_metrics = handle.metrics()
    except Exception:
        logger.debug("Model Chain: a Voice Pipeline turn's numbers were not kept",
                     exc_info=True)
    try:
        if getattr(handle.turn, "pipeline", None) is handle:
            handle.turn.pipeline = None
    except Exception:
        pass
    # Last, and on every path. This is what closes the browser's stream, and a
    # turn that ended by failing, by being cancelled or by the worker being
    # unloaded has to close it exactly as surely as one that finished -- or the
    # reply hangs with somebody waiting for audio nothing will send.
    then, handle.on_finished = handle.on_finished, None
    if then is not None:
        try:
            then()
        except Exception:
            logger.debug("Model Chain: could not close a turn the Voice Pipeline had "
                         "finished", exc_info=True)


def _drain(handle) -> None:
    """Wait until the pump has *written* everything this turn queued.

    Written, not merely dequeued, and the distinction is a real race rather than
    pedantry: the pump takes a block off the queue before it writes it, so an
    empty queue alone would let ``turn_end`` overtake the last block of audio
    down the same pipe and the worker would refuse the turn for a count that
    disagreed with what had arrived.

    Bounded, and the bound is a failsafe rather than a policy: what it is
    waiting for is a pipe write, not an inference, so a wait that does not end
    means the worker has stopped reading and the turn is over either way.
    """
    deadline = time.monotonic() + REQUEST_TIMEOUT
    while time.monotonic() < deadline:
        if handle.sent_samples >= handle.input_samples and not handle.pending():
            return
        if handle.turn.cancelled.is_set() or handle.failed:
            return
        time.sleep(PUMP_POLL)


def _pump(handle) -> None:
    """One turn's blocks, from its ingress into the worker's stdin.

    A thread rather than writing from the engine's reader thread, and the reason
    is cancellation. A write into a stalled worker's pipe blocks in a syscall
    that nothing can interrupt; the engine's reader thread is the one thread
    that must stay free, because it is also the thread that reads a cancelled
    Pocket unit's drain (I-VP-18). So the reader thread only ever touches a
    Python condition variable it can be woken out of, and this thread takes the
    risk of the pipe.

    The offset it stamps on each block is this thread's own running total of
    what it has *written*, which is deliberately not the handle's count of what
    was *queued*: the two differ by whatever is in the ingress, and an offset
    taken from the wrong one would accuse the source of skipping samples every
    time the queue was not empty (I-VP-31).
    """
    sent = 0
    while True:
        if handle.ended or handle.turn.cancelled.is_set():
            handle.close()
            return
        pcm = handle.take()
        if pcm is None:
            if not handle.wait_for_work(PUMP_POLL):
                return
            continue
        with _lock:
            started = _process
            generation = _generation
        if started is None or generation != handle.generation:
            handle.fail("the Voice Pipeline worker stopped")
            return
        try:
            _write(started, {"op": "audio", "turn": handle.id,
                             "input_sample_offset": sent}, pcm)
        except Exception as exc:
            handle.fail(_readable(exc))
            return
        sent += len(pcm) // 2
        handle.sent_samples = sent


# --------------------------------------------------------------------------- #
# One round trip
# --------------------------------------------------------------------------- #

_pending = {}


def _exchange(started, header: dict, payload: bytes, timeout: float) -> dict:
    """One request, one reply, matched by id rather than by arrival order.

    By id because the reader thread is also delivering audio for a turn while
    this is waiting: a reply is whatever comes back carrying this request's id,
    and everything else on the pipe belongs to somebody else.
    """
    identifier = uuid.uuid4().hex[:12]
    answers = collections.deque()
    ready = threading.Event()
    with _lock:
        _pending[identifier] = (answers, ready)
    try:
        _write(started, dict(header, id=identifier), payload)
        # Waited for in short steps rather than in one long one, so that a
        # worker which died on its own imports is noticed in a tenth of a second
        # instead of at the far end of a timeout written for a cold model load.
        deadline = time.monotonic() + timeout
        while not ready.wait(0.05):
            if started.poll() is not None:
                raise PipelineRuntimeError(
                    "The Voice Pipeline worker stopped before it answered.")
            if time.monotonic() > deadline:
                raise PipelineRuntimeError("The Voice Pipeline stopped answering.")
        found = answers.popleft() if answers else None
        if found is None:
            raise PipelineRuntimeError("The Voice Pipeline stopped before it answered.")
        if found.get("ok") is False and found.get("op") == "error":
            raise PipelineRuntimeError(str(found.get("error") or "unknown"))
        return found
    finally:
        with _lock:
            _pending.pop(identifier, None)


def _write(started, header: dict, payload: bytes) -> None:
    """One frame, under the one write lock. Serialised because there is one pipe."""
    protocol = _protocol()
    with _write_lock:
        if started.poll() is not None or started.stdin is None:
            raise PipelineRuntimeError("The Voice Pipeline is not running.")
        protocol.write_frame(started.stdin, header, payload)


def _write_control(started, header: dict) -> bool:
    """A frame that must never wait: taken if the pipe is free, dropped if not.

    Stop is the reason this exists. The pump holds the write lock for the length
    of one audio block's write, and against a worker that has stopped reading
    that write does not return -- so a cancel or a shutdown that queued behind
    it would be a Stop that waits for an inference, which is the one thing this
    feature promises never to do.

    Dropping the frame is safe and is not a shortcut. The turn is already closed
    at this end: its queue is discarded, its pump is stopping, and
    :func:`_deliver` refuses anything that arrives for it afterwards. What the
    frame buys is the worker letting go of its own per-turn state a little
    sooner, and the next ``turn_begin`` does that anyway.

    Returns whether it was sent, so a caller that cares can say so in a log.
    """
    if not _write_lock.acquire(timeout=0.25):
        logger.debug("Model Chain: a Voice Pipeline control frame was dropped because the "
                     "pipe was busy")
        return False
    try:
        if started.poll() is not None or started.stdin is None:
            return False
        _protocol().write_frame(started.stdin, header, b"")
        return True
    except Exception:
        return False
    finally:
        _write_lock.release()


def _readable(exc) -> str:
    """A failure as a sentence with nothing in it but this repository's words.

    The same bar :func:`mc_voice_pocket_runtime._readable` holds: a worker's
    exception contributes its class name and a message this file wrote
    contributes itself, and nothing from a library reaches a log through here.
    """
    if isinstance(exc, PipelineRuntimeError):
        return str(exc)
    if isinstance(exc, str):
        return exc
    return f"the Voice Pipeline failed ({exc.__class__.__name__})"


# --------------------------------------------------------------------------- #
# The one reader
# --------------------------------------------------------------------------- #


def _read_frames(started, generation: int) -> None:
    """Every frame the worker sends, to the turn or the waiter it belongs to."""
    protocol = _protocol()
    try:
        while True:
            frame = protocol.read_frame(started.stdout)
            if frame is None:
                break
            header, payload = frame
            if generation != _generation:
                # Read without the state lock, deliberately. This thread is how
                # every reply reaches every waiter, so a lock it takes per frame
                # is a lock that can stop the whole module if anything ever
                # holds it while waiting. An int compare is enough for what this
                # is: noticing that the worker was replaced.
                break
            operation = str(header.get("op") or "")
            if operation == "audio_out":
                _deliver(header, payload)
                continue
            if operation == "turn_flushed":
                # In pipe order, behind this turn's last block of audio. That
                # ordering is the whole reason ending a turn does not wait for a
                # reply -- see :func:`_finish`.
                with _lock:
                    handle = _turns.get(str(header.get("turn") or ""))
                if handle is not None:
                    _flushed(handle, dict(header))
                continue
            identifier = header.get("id")
            if identifier is not None:
                with _lock:
                    waiting = _pending.get(identifier)
                if waiting is not None:
                    waiting[0].append(dict(header))
                    waiting[1].set()
                    continue
            if operation == "error":
                _fail_turn(str(header.get("turn") or ""),
                           str(header.get("error") or "unknown"))
    except Exception:
        logger.debug("Model Chain: the Voice Pipeline worker's pipe ended", exc_info=True)
    finally:
        _fail_everything(generation)


def _deliver(header: dict, payload: bytes) -> None:
    """One block of enhanced PCM, straight to the turn's own playback queue.

    Immediately. Nothing here accumulates a target number of seconds before
    forwarding, and nothing here is allowed to: a finalised sample is one the
    browser may as well have, and the browser's adaptive buffer is what decides
    when to start playing it (I-VP-17, section 11.3).
    """
    identifier = str(header.get("turn") or "")
    with _lock:
        handle = _turns.get(identifier)
    if handle is None or handle.ended:
        # A block for a turn that was cancelled or has already ended. Dropped
        # here, at the one place output arrives, so no stale generation's audio
        # can reach a playback queue (I-VP-31, section 16.4).
        return
    offset = header.get("output_sample_offset")
    if offset is not None and int(offset) != handle.output_samples:
        handle.fail("the enhanced stream skipped or repeated samples")
        return
    rate = int(header.get("output_sample_rate") or 0)
    if rate and rate != handle.snapshot.output_rate:
        handle.fail("the Voice Pipeline changed sample rate inside one reply")
        return
    handle.output_samples += len(payload) // 2
    handle.output_packets += 1
    if not handle.first_output:
        handle.first_output = time.monotonic()
    # Flagged around the offer because it can block, on purpose: this is the
    # backpressure that reaches all the way back to the speech engine. The flag
    # is what lets the flush watchdog tell "waiting for a listener" from "waiting
    # for a worker that has stopped answering".
    handle.delivering = True
    try:
        handle.turn.offer_audio(payload, handle.snapshot.output_rate)
    finally:
        handle.delivering = False


def _fail_turn(identifier: str, reason: str) -> None:
    """A refusal the worker sent for one turn. Ends that turn and nothing else.

    The worker answers a bad frame rather than dying on it, so one turn's
    failure costs that turn and not the residency of two warm models.
    """
    with _lock:
        handle = _turns.get(str(identifier or ""))
    if handle is not None:
        logger.info("Model Chain: the Voice Pipeline refused a reply — %s", reason)
        handle.fail(reason)


def _fail_everything(generation: int) -> None:
    """The worker's pipe ended. Wake every waiter rather than leaving them."""
    with _lock:
        if generation != _generation and _process is not None:
            return
        waiting = list(_pending.values())
        speaking = list(_turns.values())
        _pending.clear()
        _turns.clear()
    for _answers, ready in waiting:
        try:
            ready.set()
        except Exception:
            pass
    for handle in speaking:
        try:
            handle.fail("the Voice Pipeline worker stopped")
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
                logger.info("Model Chain: Voice Pipeline worker — %s", text[:400])
    except Exception:
        logger.debug("Model Chain: the Voice Pipeline worker's stderr ended",
                     exc_info=True)


def _note_failure() -> None:
    """Remember a failure, so a worker that cannot start is not retried forever."""
    _failures.append(time.monotonic())


def _guard_crash_loop() -> bool:
    """Whether the worker has failed too often to keep trying this minute.

    Bounded automatic restart, matching the spirit of the other runtimes'
    recovery: one immediate retry is a hiccup and four in a minute is a
    configuration that does not work, and retrying that forever turns an
    optional polish into a process-spawn loop (section 15.4).
    """
    now = time.monotonic()
    recent = [when for when in _failures if now - when < 60.0]
    return len(recent) < 4


# --------------------------------------------------------------------------- #
# Stopping
# --------------------------------------------------------------------------- #


def stop(reason: str = "") -> None:
    """Stop the worker if one is running. Idempotent, and never raises.

    Reachable while another thread is inside an inference, which is only true
    because nothing waiting holds ``_lock``: the check is under it and the
    teardown is outside it, so a status poll landing during a bounded escalation
    does not wait several seconds for a process to die.
    """
    with _lock:
        running_now = _process is not None
    if not running_now:
        return
    _discard(reason or "the Voice Pipeline stopped")


def shutdown() -> None:
    """Door A and the body of doors B and C. Idempotent; must never raise.

    Called from Forge's script-unload callback, from ``atexit`` and from the
    chained signal handlers that :mod:`mc_voice_pocket_runtime` installs.
    Everything inside is swallowed: a WebUI that will not close because an
    optional audio polish is thinking is a worse bug than any this could be
    reporting.
    """
    global _closing

    try:
        with _lock:
            _closing = True
            running_now = _process is not None
        if running_now:
            _discard("WebUI shutdown")
    except Exception:
        logger.debug("Model Chain: the Voice Pipeline shutdown hook failed", exc_info=True)
    finally:
        _closing = False


def _discard(reason: str) -> None:
    """End the worker and let go of the handle, in that order.

    Bounded at every step, because the thing being stopped may be inside an
    inference that will not look at a pipe again: ask, close the pipe, wait,
    terminate, wait, kill. None of the steps is "wait for it".
    """
    global _process, _reader, _handshake, _generation, _loaded

    with _lock:
        started, _process, _reader, _handshake = _process, None, None, None
        _generation += 1
        _loaded = ()
        speaking = list(_turns.values())
        waiting = list(_pending.values())
        _turns.clear()
        _pending.clear()
    for handle in speaking:
        try:
            # Cancelled, not merely detached. A turn whose handle was taken away
            # while it was still speaking would go back to the unenhanced branch
            # in the engine's reader -- and splice raw 24 kHz PCM into a
            # response whose headers already told the browser 48 kHz (I-VP-27).
            handle.fail("the Voice Pipeline was unloaded")
        except Exception:
            pass
    for _answers, ready in waiting:
        try:
            ready.set()
        except Exception:
            pass
    if started is None:
        return

    if started.poll() is None:
        # Asked politely if the pipe is free, and never waited for if it is not:
        # closing stdin below breaks a pump blocked inside a write, and the
        # escalation after it does not need the worker's cooperation.
        _write_control(started, {"op": "shutdown", "id": 0})
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
    logger.info("Model Chain: the Voice Pipeline worker stopped — %s",
                reason or "no reason given")


def _wait(started, seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        if started.poll() is not None:
            return True
        time.sleep(0.05)
    return started.poll() is not None


# --------------------------------------------------------------------------- #
# Coupling to the engine that feeds it
# --------------------------------------------------------------------------- #


def warm(engine: str = "pocket") -> None:
    """Load the selected stages beside their speech engine, not in front of a reply.

    Called from the engine's own prepare/load, so the enhancement models are
    read off disk while the language model is still writing the sentence they
    will polish. Cold-start overlap rather than a manufactured delay: warm-up
    exists to take the model load out of the first spoken sample, not to hold
    audio back until a buffer is comfortable (section 11.10).

    Never fatal. A pipeline that cannot warm leaves the reply unenhanced, which
    :func:`begin_turn` will discover a moment later through the path that exists
    for it.
    """
    import mc_voice_pipeline as pipeline

    if engine not in pipeline.SUPPORTED_ENGINES:
        return
    try:
        state = pipeline.status()
        snapshot = pipeline.snapshot(0, engine, state)
        if not snapshot.enabled or not snapshot.stage_ids:
            return
        if not _guard_crash_loop():
            return
        ensure_started(snapshot.stage_ids)
    except Exception as exc:
        _note_failure()
        logger.info("Model Chain: the Voice Pipeline did not warm — %s", _readable(exc))


def unload(reason: str = "unloaded") -> None:
    """Follow the speech engine out of residency.

    The other half of I-VP-19, and the half that has to be called from the
    engine rather than decided here: Pocket unloading is the event, and a
    pipeline that timed its own residency would be a pipeline that is sometimes
    resident with nothing to enhance and sometimes cold when a reply arrives.
    """
    stop(reason)
