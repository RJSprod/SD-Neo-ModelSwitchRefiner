"""The VibeVoice worker processes: one per card, proven, rendered through, evicted.

The parent half of the Voice Box's speech guest. It owns one subprocess per
graphics card, a pipe to each, a handshake it refuses to accept a bad answer
to, the same five ways of making sure a process dies with this one that the
Voice Chat runtimes have, and two figures the turns on each card read every
hundred milliseconds: how much of the card the worker holds, and whether it is
rendering.

Why one process per card
------------------------
A VibeVoice render happens on a card, and the turns (``mc_turns``) are kept per
card: the two cards may each run one render, a warm guest on one card is
evicted by an image job on *that* card and not the other, and a worker is
started with exactly one card visible. So the unit of state here is a
:class:`_Process` keyed by the card's UUID reduced to its hex digits --
nvidia-smi's ``GPU-…`` and torch's bare digits are one card -- and every public
function takes the card first.

What the handshake has to prove
-------------------------------
That the worker came up *on the card it was asked for*. ``CUDA_VISIBLE_DEVICES``
is a request, not a guarantee: a UUID mistyped, a driver that ignores the
variable, or a torch too old to report a device UUID would each give a worker
that is about to load eighteen gigabytes onto whichever card it happens to see.
So the worker reports the UUID torch gives its device 0, and this side refuses
a worker whose UUID does not match the card by key, one that reports no UUID
at all ("cannot prove which card it is on"), one that found no CUDA device, one
speaking another protocol or claiming another name, and -- off Windows, where
the parent arranged the job object itself -- one that could not arrange its own
death with this process.

What this module does not do
----------------------------
It never holds a lock while it waits for the worker: every wait is on a queue
the caller owns, which is what lets a status poll, a cancel and an eviction
reach a runtime whose worker is inside a forty-minute generation. It never
imports Torch or vibevoice. It never imports the memory planner, the broker,
the plan, the language model runtime or the turns: it is registered *with* the
turns as a guest by ``scripts/model_chain.py``, and the direction of that
dependency is what keeps invariant I-3 (``tests/test_voice_independence.py``)
true. And it never puts a path, a script or a sample in anything it logs.
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field

import mc_voice_paths as paths
from vibevoice_worker import worker as protocol

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""


class VibeVoiceRuntimeError(RuntimeError):
    """A VibeVoice request that could not be served. Never fatal to the host."""


CONTAINMENT = {"windows": "job", "linux": "pdeathsig"}
"""Which parent-death mechanism each supported platform uses.

A platform absent from here has no tested way of guaranteeing that a hard kill
of the WebUI takes an eighteen-gigabyte GPU process with it, and this runtime
refuses to start one there rather than starting one it cannot promise to end.
"""

HANDSHAKE_TIMEOUT = 600.0
"""Torch import plus CUDA initialisation on a cold machine, generously."""

LOAD_TIMEOUT = 900.0
"""Eighteen gigabytes off a disk that may be slow, into a card."""

RENDER_TIMEOUT = 3600.0
"""An hour. The model's context holds about forty-five minutes of speech, so a
render that has not answered in an hour is a worker that will not answer."""

CANCEL_TIMEOUT = 10.0
STATUS_TIMEOUT = 30.0
UNLOAD_TIMEOUT = 300.0

RESIDENT_REFRESH = 2.0
"""How often :func:`resident_bytes` may ask the worker rather than answer from
its cache. The turns poll it every hundred milliseconds during an eviction and
every second under a warm stay; every reply the worker sends refreshes the
cache anyway, and a render never gets asked."""

STOP_GRACE = 10.0
TERMINATE_GRACE = 5.0
"""How long a worker gets to end on its own after ``shutdown``, and then after
``terminate``, before the next step. Longer than the CPU engines' graces: the
process is letting go of a CUDA context holding eighteen gigabytes, and a
context torn down by the driver after a kill is reclaimed more slowly than one
released by the process."""

CRASH_WINDOW = 60.0
CRASH_LIMIT = 3
"""Three failed starts on one card in a minute stops the fourth. A worker that
cannot load its model will not load it on the tenth attempt either."""


@dataclass
class Handshake:
    """What the worker said it is, once, at ``init``. Held rather than re-asked."""

    protocol: int = 0
    worker: str = ""
    python: str = ""
    torch: str = ""
    cuda: bool = False
    device_name: str = ""
    device_uuid: str = ""
    device_index: int = 0
    total_vram_bytes: int = 0
    free_vram_bytes: int = 0
    containment: str = ""
    vibevoice: str = ""
    transformers: str = ""
    sample_rate: int = 0


@dataclass
class RenderJob:
    """One render as the render service hands it over.

    ``script`` is a list of ``(speaker, text)`` pairs, speakers numbered from
    one as the user numbered them; ``voices`` maps each speaker to
    ``(float32 little-endian mono PCM bytes, rate)``.
    """

    id: str
    script: list = field(default_factory=list)
    voices: dict = field(default_factory=dict)
    cfg_scale: float = 1.3
    steps: int = 10
    seed: "int | None" = None
    max_new_tokens: "int | None" = None


@dataclass
class RenderResult:
    """What one render produced. ``wav`` is a complete mono PCM16 file."""

    wav: bytes
    seconds: float
    sample_rate: int
    render_seconds: float
    peak_bytes: int
    cancelled: bool
    tokens: int
    capped: bool = False


def card_key(card_uuid) -> str:
    """A GPU UUID as the hex digits both of its spellings share.

    nvidia-smi writes ``GPU-6a1f…`` and torch hands back the bare digits; the
    same reduction ``mc_broker.card_identity`` makes, implemented here rather
    than imported because a voice module may not import the broker.
    """
    text = str(card_uuid or "").strip().casefold()
    return "".join(character for character in text if character in "0123456789abcdef")


class _WorkerGone(Exception):
    """The pipe ended, the worker was stopping, or it did not answer in time."""

    def __init__(self, message: str, timed_out: bool = False):
        super().__init__(message)
        self.timed_out = timed_out


class _Process:
    """One worker on one card: its handle, its pipe, and the cached figures.

    ``lock`` guards the fields and is never held across a wait on the pipe;
    ``write_lock`` serialises writes to the worker's stdin. A record is made
    once per start and never reused: a stale reader thread therefore cannot
    deliver a frame to a successor's request, because the successor is a
    different record with a different ``pending`` table.
    """

    def __init__(self, key: str, card_uuid: str, process):
        self.key = key
        self.uuid = str(card_uuid or "")
        self.process = process
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.pending: dict = {}
        self.next_id = 0
        self.handshake: "Handshake | None" = None
        self.loaded = False
        self.rendering = False
        self.job = ""
        self.resident_bytes = 0
        self.peak_bytes = 0
        self.refreshed_at = 0.0
        self.closing = False

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def label(self) -> str:
        """The card as a log line names it: its name, else its UUID."""
        if self.handshake is not None and self.handshake.device_name:
            return self.handshake.device_name
        return self.uuid or self.key or "an unnamed card"


_table_lock = threading.RLock()
_processes: "dict[str, _Process]" = {}
_lifecycle: "dict[str, threading.RLock]" = {}
"""One re-entrant lock per card, held across a start (Popen and handshake) and
across a stop (the bounded escalation). Held across those waits on purpose --
they are lifecycle operations, and two of them on one card must not overlap --
and never taken by anything that only reads state."""

_failures: "dict[str, list]" = {}
_peaks: "dict[str, int]" = {}
_last_error = ""
_closing = False
_job_handle = None
_exit_registered = False
_exit_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Reading state
# --------------------------------------------------------------------------- #


def status(card_uuid: str = "") -> dict:
    """What the page reads. A pure read of cached state: nothing is asked.

    ``running`` is whether the process exists, ``loaded`` whether it has said
    it holds a model, ``rendering`` whether a render call is in flight on that
    card from this side. No path, no pid, no command line.
    """
    wanted = card_key(card_uuid)
    with _table_lock:
        records = list(_processes.values())
        error = _last_error
    cards = {}
    for record in records:
        if wanted and record.key != wanted:
            continue
        with record.lock:
            running = record.alive()
            handshake = record.handshake
            cards[record.key] = {
                "running": running,
                "loaded": bool(record.loaded and running),
                "rendering": bool(record.rendering),
                "job": record.job,
                "resident_bytes": record.resident_bytes if running else 0,
                "peak_bytes": record.peak_bytes,
                "device_name": handshake.device_name if handshake else "",
                "uuid": record.uuid,
            }
    return {"cards": cards, "last_error": error}


def rendering(card_uuid: str) -> bool:
    """Whether a render call is in flight on that card. Never blocks."""
    record = _live(card_key(card_uuid))
    if record is None:
        return False
    with record.lock:
        return bool(record.rendering)


def resident_bytes(card_uuid: str) -> int:
    """How much of the card the worker holds; ``0`` when there is no process.

    Cheap by construction. The figure comes from the last reply the worker
    sent -- every ``load``, ``render``, ``unload`` and ``status`` reply carries
    it -- and a ``status`` request is sent to refresh it at most every
    :data:`RESIDENT_REFRESH` seconds, and never while a render is in flight:
    the worker's lane is inside the model then, and the figure cannot change
    in a way the caller could act on.

    A process that is stopping still answers its last figure until it has
    exited, which is what lets an eviction be measured as "the card has its
    memory back" rather than "the handle was dropped".
    """
    key = card_key(card_uuid)
    record = _record(key)
    if record is None or not record.alive():
        return 0
    now = time.monotonic()
    with record.lock:
        if record.closing or record.rendering or record.handshake is None \
                or now - record.refreshed_at < RESIDENT_REFRESH:
            return record.resident_bytes
        # Claimed before the request is sent, so a hundred pollers in the
        # window make one request rather than a hundred.
        record.refreshed_at = now
    try:
        _exchange(record, {"op": "status"}, b"", STATUS_TIMEOUT)
    except _WorkerGone as exc:
        _lost(record, f"the VibeVoice worker stopped answering ({exc})")
        return 0
    except VibeVoiceRuntimeError:
        logger.debug("Model Chain: the VibeVoice worker refused a status read", exc_info=True)
    with record.lock:
        return record.resident_bytes if record.alive() else 0


def peak_observed(card_uuid: str) -> int:
    """The highest peak any render on that card reported this session."""
    with _table_lock:
        return int(_peaks.get(card_key(card_uuid), 0))


def _record(key: str) -> "_Process | None":
    with _table_lock:
        return _processes.get(key)


def _live(key: str) -> "_Process | None":
    """The card's record when its process is running and not being stopped."""
    with _table_lock:
        record = _processes.get(key)
    if record is None:
        return None
    with record.lock:
        if record.closing:
            return None
    if not record.alive():
        return None
    return record


def _lifecycle_lock(key: str) -> threading.RLock:
    with _table_lock:
        found = _lifecycle.get(key)
        if found is None:
            found = _lifecycle[key] = threading.RLock()
        return found


# --------------------------------------------------------------------------- #
# The module the installer owns, reached lazily
# --------------------------------------------------------------------------- #


def _installer():
    """``mc_voice_vibevoice``: the installer, settings and paths. Deferred so
    this module imports on its own, and so a test can stand it in."""
    import mc_voice_vibevoice as vibevoice

    return vibevoice


def _refusal(vibevoice) -> str:
    """Why a worker cannot start now, in the installer's own words, or ``""``."""
    ask = getattr(vibevoice, "refusal", None)
    if ask is None:
        return ""
    try:
        return str(ask() or "")
    except Exception:
        logger.debug("Model Chain: the VibeVoice installer could not say whether it is "
                     "ready", exc_info=True)
        return ""


def _settings(vibevoice) -> dict:
    try:
        return dict(vibevoice.settings() or {})
    except Exception:
        logger.debug("Model Chain: the VibeVoice settings could not be read", exc_info=True)
        return {}


def _worker_script(vibevoice):
    ask = getattr(vibevoice, "worker_script", None)
    if ask is not None:
        try:
            return ask()
        except Exception:
            logger.debug("Model Chain: the VibeVoice installer could not name the worker "
                         "script", exc_info=True)
    return paths.extension_root() / "vibevoice_worker" / "worker.py"


def _note_peak(vibevoice, identifier: str, peak: int) -> None:
    """The calibration figure, handed to the installer's file. Never raises."""
    ask = getattr(vibevoice, "note_peak", None)
    if ask is None or peak <= 0:
        return
    try:
        ask(identifier, int(peak))
    except Exception:
        logger.debug("Model Chain: the VibeVoice peak could not be recorded", exc_info=True)


def _expected_containment() -> str:
    """The parent-death mechanism this platform is supposed to use, or ``""``."""
    import mc_voice_models as models

    try:
        system, _machine, _python = models.current_platform()
    except Exception:
        logger.debug("Model Chain: could not identify this platform for VibeVoice",
                     exc_info=True)
        return ""
    return CONTAINMENT.get(system, "")


# --------------------------------------------------------------------------- #
# Starting
# --------------------------------------------------------------------------- #


def ensure_started(card_uuid: str) -> Handshake:
    """Start the worker on that card if it is not running. Idempotent per card.

    Every failure path is written so that a process which has been started is
    stopped before its handle is dropped: ownership begins at ``Popen``, not at
    the moment the handshake succeeds.
    """
    global _last_error

    key = card_key(card_uuid)
    if not key:
        raise VibeVoiceRuntimeError("No card was named for VibeVoice.")
    found = _live(key)
    if found is not None and found.handshake is not None:
        return found.handshake
    with _table_lock:
        if _closing:
            raise VibeVoiceRuntimeError("Voice Box is shutting down.")
    vibevoice = _installer()

    with _lifecycle_lock(key):
        found = _live(key)
        if found is not None and found.handshake is not None:
            return found.handshake
        stale = _record(key)
        if stale is not None:
            _discard(stale, "a previous VibeVoice worker had already exited")
        _guard_crash_loop(key)

        reason = _refusal(vibevoice)
        if reason:
            raise VibeVoiceRuntimeError(reason)
        interpreter = vibevoice.runtime_python()
        if interpreter is None:
            raise VibeVoiceRuntimeError(
                "The VibeVoice runtime is not installed. Install it in the Voice Box tab.")
        command = [str(interpreter), str(_worker_script(vibevoice)), protocol.MARKER,
                   "--parent-pid", str(os.getpid())]
        environ = dict(os.environ)
        # The worker's own environment *last*, so a CUDA variable inherited
        # from the WebUI cannot outrank the one card this worker is given; and
        # no credential at all, because the worker is offline by construction.
        environ.update(vibevoice.worker_environment(card_uuid))
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            environ.pop(name, None)

        record = None
        with _table_lock:
            _last_error = ""
        try:
            started = subprocess.Popen(  # noqa: S603 - a path this module built
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=environ, cwd=str(paths.extension_root()),
                bufsize=0, close_fds=True)
            record = _Process(key, card_uuid, started)
            with _table_lock:
                _processes[key] = record
            _die_with_us(started)
            threading.Thread(target=_read_frames, args=(record,),
                             name=f"mc-vibevoice-reader-{key[:8]}", daemon=True).start()
            threading.Thread(target=_drain_stderr, args=(started,),
                             name=f"mc-vibevoice-stderr-{key[:8]}", daemon=True).start()
            handshake = _handshake_with(record, key)
            with record.lock:
                record.handshake = handshake
        except Exception as exc:
            # Whatever went wrong, the process this function started is this
            # function's to end. Nothing below this line may leave a handle.
            _note_failure(key)
            if record is not None:
                _discard(record, "the VibeVoice worker failed to start")
            message = str(exc) if isinstance(exc, VibeVoiceRuntimeError) else ""
            with _table_lock:
                _last_error = message or "VibeVoice could not be started."
            if isinstance(exc, VibeVoiceRuntimeError):
                raise
            logger.warning("Model Chain: the VibeVoice worker could not be started",
                           exc_info=True)
            if isinstance(exc, _WorkerGone):
                raise VibeVoiceRuntimeError(
                    "VibeVoice did not finish starting, so it was stopped. Try again.") \
                    from None
            raise VibeVoiceRuntimeError(
                "VibeVoice could not be started. Check the Voice Box tab.") from None

        stop_on_exit()
        logger.info("Model Chain: VibeVoice worker ready on %s — torch %s, vibevoice %s, "
                    "transformers %s, %.1f of %.1f GiB free, containment %s",
                    handshake.device_name or key, handshake.torch, handshake.vibevoice,
                    handshake.transformers, handshake.free_vram_bytes / float(1 << 30),
                    handshake.total_vram_bytes / float(1 << 30), handshake.containment)
        return handshake


def _handshake_with(record: _Process, key: str) -> Handshake:
    """Ask the worker what it is, and refuse an answer this build cannot accept.

    Seven refusals, each with its own sentence, because "VibeVoice failed to
    start" is not a diagnosable state: a protocol this build does not speak; a
    worker that is not VibeVoice's; no CUDA device; no device UUID, which is a
    worker that cannot prove its card; a UUID that is another card's; no
    containment evidence where the parent could not arrange it itself; and a
    sample rate that is not the one every voice sample was prepared at.
    """
    reply, _body = _exchange(record, {"op": "init", "expect_uuid": key,
                                      "parent_pid": os.getpid()}, b"", HANDSHAKE_TIMEOUT)
    found = Handshake(
        protocol=int(reply.get("protocol") or 0),
        worker=str(reply.get("worker") or ""),
        python=str(reply.get("python") or ""),
        torch=str(reply.get("torch") or ""),
        cuda=bool(reply.get("cuda")),
        device_name=str(reply.get("device_name") or ""),
        device_uuid=str(reply.get("device_uuid") or ""),
        device_index=int(reply.get("device_index") or 0),
        total_vram_bytes=int(reply.get("total_vram_bytes") or 0),
        free_vram_bytes=int(reply.get("free_vram_bytes") or 0),
        containment=str(reply.get("containment") or ""),
        vibevoice=str(reply.get("vibevoice") or ""),
        transformers=str(reply.get("transformers") or ""),
        sample_rate=int(reply.get("sample_rate") or 0))

    if found.protocol != protocol.PROTOCOL_VERSION:
        raise VibeVoiceRuntimeError(
            f"The VibeVoice worker speaks protocol {found.protocol} and this build speaks "
            f"{protocol.PROTOCOL_VERSION}. Reinstall VibeVoice in the Voice Box tab.")
    if found.worker != protocol.WORKER_NAME:
        raise VibeVoiceRuntimeError(
            "The VibeVoice worker did not identify itself as VibeVoice, so it was stopped.")
    if not found.cuda:
        raise VibeVoiceRuntimeError(
            "The VibeVoice worker found no CUDA device, so it was stopped.")
    reported = card_key(found.device_uuid)
    if not reported:
        raise VibeVoiceRuntimeError(
            "The VibeVoice worker cannot prove which card it is on (its torch reports no "
            "device UUID), so it was stopped.")
    if reported != key:
        raise VibeVoiceRuntimeError(
            f"The VibeVoice worker came up on {found.device_name or 'an unnamed card'} "
            f"({found.device_uuid}), not the card asked for, so it was stopped.")
    if os.name != "nt" and found.containment != _expected_containment():
        # On Windows the parent arranged the job object itself and verified it
        # with real handles, so the child's own answer is diagnostic. On every
        # other platform the mechanism has to run *inside* the child between
        # fork and the first request, and the child's confirmation is the only
        # evidence there is.
        raise VibeVoiceRuntimeError(
            "The VibeVoice worker could not confirm that it will be ended if this WebUI "
            "stops, so it was stopped now instead.")
    if found.sample_rate != protocol.SAMPLE_RATE:
        raise VibeVoiceRuntimeError(
            f"The VibeVoice worker reported a sample rate of {found.sample_rate} Hz rather "
            f"than {protocol.SAMPLE_RATE}, so it was stopped.")
    return found


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


def load(card_uuid: str) -> dict:
    """Start the card's worker if need be and have it load the model.

    The directory and the step count come from the installer's settings, the
    precision and attention from this build: bf16 and SDPA (design section 3).
    A worker that already holds the model answers at once.
    """
    key = card_key(card_uuid)
    ensure_started(card_uuid)
    record = _live(key)
    if record is None:
        raise VibeVoiceRuntimeError("VibeVoice is not running on that card.")
    with record.lock:
        if record.rendering:
            raise VibeVoiceRuntimeError("VibeVoice is rendering on that card.")
    vibevoice = _installer()
    settings = _settings(vibevoice)
    model_dir = str(vibevoice.model_dir(str(settings.get("model_id") or "")))
    header = {"op": "load", "model_dir": model_dir, "steps": int(settings.get("steps") or 10),
              "dtype": "bf16", "attention": "sdpa"}
    try:
        reply, _body = _exchange(record, header, b"", LOAD_TIMEOUT)
    except _WorkerGone as exc:
        _lost(record, f"the VibeVoice worker was lost while loading ({exc})")
        raise VibeVoiceRuntimeError(
            "VibeVoice stopped while it was loading its model. Try again.") from None
    with record.lock:
        record.loaded = True
    if not reply.get("already_loaded"):
        logger.info("Model Chain: VibeVoice loaded its model on %s in %.1f s — %.1f GiB of "
                    "weights, %.1f GiB held", record.label(),
                    float(reply.get("load_seconds") or 0.0),
                    int(reply.get("weights_bytes") or 0) / float(1 << 30),
                    int(reply.get("resident_bytes") or 0) / float(1 << 30))
    return {"loaded": True,
            "resident_bytes": int(reply.get("resident_bytes") or 0),
            "weights_bytes": int(reply.get("weights_bytes") or 0),
            "load_seconds": float(reply.get("load_seconds") or 0.0)}


def render(card_uuid: str, job: RenderJob, on_progress=None) -> RenderResult:
    """One script through the card's worker. Marks the card rendering meanwhile.

    ``on_progress`` is called with each progress frame's dict (``{"seconds":
    float}``) on the calling thread. A worker that dies mid-render is an
    error naming that, and the card's state is reset; the render is never
    retried, because a retry would be another long generation nobody asked
    for a second time.
    """
    key = card_key(card_uuid)
    record = _live(key)
    if record is None:
        raise VibeVoiceRuntimeError("VibeVoice is not loaded on that card.")
    with record.lock:
        if record.closing:
            raise VibeVoiceRuntimeError("VibeVoice is stopping on that card.")
        if record.rendering:
            raise VibeVoiceRuntimeError(
                "VibeVoice is already rendering on that card; one render at a time.")
        record.rendering = True
        record.job = str(job.id or "")
    try:
        header, payload = _render_frame(job)
        try:
            reply, body = _exchange(record, header, payload, RENDER_TIMEOUT, on_progress)
        except _WorkerGone as exc:
            if exc.timed_out:
                _lost(record, "the VibeVoice worker did not finish a render in time")
                raise VibeVoiceRuntimeError(
                    "the VibeVoice worker did not finish the render in time, so it was "
                    "stopped") from None
            _lost(record, "the VibeVoice worker exited during a render")
            raise VibeVoiceRuntimeError(
                "the VibeVoice worker exited during the render") from None
        result = RenderResult(
            wav=bytes(body or b""),
            seconds=float(reply.get("seconds") or 0.0),
            sample_rate=int(reply.get("sample_rate") or protocol.SAMPLE_RATE),
            render_seconds=float(reply.get("render_seconds") or 0.0),
            peak_bytes=int(reply.get("peak_bytes") or 0),
            cancelled=bool(reply.get("cancelled")),
            tokens=int(reply.get("tokens") or 0),
            capped=bool(reply.get("capped")))
        with record.lock:
            record.peak_bytes = max(record.peak_bytes, result.peak_bytes)
        with _table_lock:
            _peaks[key] = max(int(_peaks.get(key, 0)), result.peak_bytes)
        vibevoice = _installer()
        _note_peak(vibevoice, str(_settings(vibevoice).get("model_id") or ""),
                   result.peak_bytes)
        logger.info("Model Chain: VibeVoice rendered %.1f s of audio on %s in %.1f s — "
                    "%d token(s), peak %.1f GiB%s%s", result.seconds, record.label(),
                    result.render_seconds, result.tokens,
                    result.peak_bytes / float(1 << 30),
                    ", cancelled" if result.cancelled else "",
                    ", reached its token budget" if result.capped else "")
        return result
    finally:
        with record.lock:
            record.rendering = False
            record.job = ""


def _render_frame(job: RenderJob) -> "tuple[dict, bytes]":
    """The render request as the worker reads it: a header and one PCM payload.

    The voices' float32 PCM is concatenated in ascending speaker order and each
    voice entry says where its samples begin and how many there are.
    """
    script = []
    for entry in job.script or ():
        if isinstance(entry, dict):
            speaker, text = entry.get("speaker"), entry.get("text")
        else:
            speaker, text = entry
        script.append({"speaker": int(speaker), "text": str(text or "")})
    voices = []
    chunks = []
    offset = 0
    for speaker in sorted(job.voices or {}, key=lambda one: int(one)):
        pcm, rate = job.voices[speaker]
        data = bytes(pcm or b"")
        count = len(data) // 4
        data = data[:count * 4]
        voices.append({"speaker": int(speaker), "offset": offset, "count": count,
                       "rate": int(rate or 0)})
        chunks.append(data)
        offset += count
    header = {
        "op": "render",
        "job": str(job.id or ""),
        "script": script,
        "voices": voices,
        "cfg_scale": float(job.cfg_scale),
        "seed": None if job.seed is None else int(job.seed),
        "max_new_tokens": None if not job.max_new_tokens else int(job.max_new_tokens),
        "steps": int(job.steps or 0),
    }
    return header, b"".join(chunks)


def cancel(card_uuid: str, job_id: str) -> bool:
    """Ask the card's worker to stop the render named. Never raises.

    ``True`` when the worker set the flag its generation reads; ``False`` when
    nothing was rendering, the job had already finished, or the worker could
    not be asked.
    """
    record = _live(card_key(card_uuid))
    if record is None:
        return False
    with record.lock:
        if not record.rendering:
            return False
    try:
        reply, _body = _exchange(record, {"op": "cancel", "job": str(job_id or "")}, b"",
                                 CANCEL_TIMEOUT)
    except (_WorkerGone, VibeVoiceRuntimeError):
        logger.debug("Model Chain: a VibeVoice render could not be cancelled", exc_info=True)
        return False
    return bool(reply.get("cancelled"))


def unload(card_uuid: str, reason: str = "") -> dict:
    """Have the worker let go of its model and keep the process. Optional path
    for a manual Unload; an eviction stops the process instead."""
    record = _live(card_key(card_uuid))
    if record is None:
        return status(card_uuid)
    with record.lock:
        if record.rendering:
            raise VibeVoiceRuntimeError("VibeVoice is rendering on that card, so it was "
                                        "not unloaded.")
    try:
        _exchange(record, {"op": "unload"}, b"", UNLOAD_TIMEOUT)
    except _WorkerGone as exc:
        _lost(record, f"the VibeVoice worker was lost while unloading ({exc})")
        raise VibeVoiceRuntimeError("VibeVoice stopped while it was unloading.") from None
    with record.lock:
        record.loaded = False
    logger.info("Model Chain: VibeVoice unloaded its model on %s — %s", record.label(),
                reason or "no reason given")
    return status(card_uuid)


def evict(card_uuid: str, reason: str = "") -> int:
    """Stop the card's worker so the card has its memory back. Returns the bytes it held.

    Never while a render is in flight: a running render is never cut off (rule
    1 of the turns), so an eviction then returns ``0``, logs why, and leaves the
    caller to ask again once the render has ended. The stop is the same bounded
    escalation every stop is; the figure returned is the last one the worker
    reported, and :func:`resident_bytes` keeps answering it until the process
    has actually exited.
    """
    record = _record(card_key(card_uuid))
    if record is None:
        return 0
    if not record.alive():
        _discard(record, reason or "evicted")
        return 0
    with record.lock:
        if record.rendering:
            logger.info("Model Chain: VibeVoice is rendering on %s, so it was not evicted "
                        "(%s)", record.label(), reason or "no reason given")
            return 0
        if record.closing:
            return 0
        record.closing = True
        held = int(record.resident_bytes)
    _discard(record, reason or "evicted")
    return held


# --------------------------------------------------------------------------- #
# The pipe
# --------------------------------------------------------------------------- #


def _exchange(record: _Process, header: dict, payload: bytes, timeout: float,
              on_progress=None):
    """Write one frame and wait for the reply that carries its id.

    The waiting is on a queue this call owns, so two callers cannot receive one
    another's answers even in principle. Progress frames for the same id are
    handed to ``on_progress`` and the wait goes on, against one deadline.
    """
    with record.lock:
        if record.closing:
            raise _WorkerGone("the VibeVoice worker is stopping")
        record.next_id += 1
        request_id = f"{record.key[:6]}-{record.next_id}"
        answers: queue.Queue = queue.Queue()
        record.pending[request_id] = answers
    try:
        message = dict(header)
        message["id"] = request_id
        _write(record, message, payload)
        deadline = time.monotonic() + float(timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _WorkerGone("the VibeVoice worker did not answer in time",
                                  timed_out=True)
            try:
                found = answers.get(timeout=remaining)
            except queue.Empty:
                raise _WorkerGone("the VibeVoice worker did not answer in time",
                                  timed_out=True) from None
            if found is None:
                raise _WorkerGone("the VibeVoice worker stopped")
            reply, body = found
            if "ok" not in reply and "progress" in reply:
                if on_progress is not None:
                    try:
                        on_progress(dict(reply.get("progress") or {}))
                    except Exception:
                        logger.debug("Model Chain: a VibeVoice progress callback failed",
                                     exc_info=True)
                continue
            if not reply.get("ok", True):
                raise VibeVoiceRuntimeError(_readable(str(reply.get("error") or "")))
            return reply, body
    finally:
        with record.lock:
            record.pending.pop(request_id, None)


def _write(record: _Process, header: dict, payload: bytes) -> None:
    started = record.process
    if started is None or started.poll() is not None or started.stdin is None:
        raise _WorkerGone("the VibeVoice worker is not running")
    try:
        with record.write_lock:
            protocol.write_frame(started.stdin, header, payload)
    except Exception as exc:
        raise _WorkerGone(str(exc)) from None


def _readable(reason: str) -> str:
    """A worker's own words as something a user can act on.

    The worker sends an exception *class* for anything it did not raise itself,
    because a raw library message can carry a path or the script. So the
    handful it does send become sentences, a refusal (already a sentence) is
    passed through, and everything else is a general failure rather than a
    bare class name.
    """
    text = str(reason or "").strip()
    known = {
        "": "VibeVoice could not complete that request.",
        "RuntimeError": "VibeVoice could not complete that request.",
        "OSError": "VibeVoice could not read one of its files.",
        "FileNotFoundError": "One of VibeVoice's files is missing. Reinstall it in the "
                             "Voice Box tab.",
        "MemoryError": "There was not enough memory to run VibeVoice.",
        "OutOfMemoryError": "The card ran out of memory during that VibeVoice request.",
        "KeyboardInterrupt": "VibeVoice was interrupted.",
    }
    if text in known:
        return known[text]
    return text if " " in text else known[""]


def _read_frames(record: _Process) -> None:
    """The one reader per worker. Every reply to its request, every figure absorbed."""
    stream = record.process.stdout
    try:
        while True:
            frame = protocol.read_frame(stream)
            if frame is None:
                break
            header, payload = frame
            with record.lock:
                _absorb(record, header)
                answers = record.pending.get(header.get("id"))
            if answers is not None:
                answers.put((header, payload))
    except Exception:
        logger.debug("Model Chain: the VibeVoice worker's pipe ended", exc_info=True)
    finally:
        _fail_everything(record)


def _absorb(record: _Process, header: dict) -> None:
    """Every figure a reply carries, cached. Called under the record's lock."""
    if "resident_bytes" in header:
        try:
            record.resident_bytes = max(0, int(header.get("resident_bytes") or 0))
        except (TypeError, ValueError):
            pass
        record.refreshed_at = time.monotonic()
    if "peak_bytes" in header:
        try:
            record.peak_bytes = max(record.peak_bytes, int(header.get("peak_bytes") or 0))
        except (TypeError, ValueError):
            pass
    if "loaded" in header and "rendering" in header:
        # A status reply. The render flag stays this side's: it says whether a
        # render *call* is in flight, which the worker cannot know.
        record.loaded = bool(header.get("loaded"))


def _fail_everything(record: _Process) -> None:
    """The worker's pipe ended. Wake every waiter rather than leaving them."""
    with record.lock:
        waiting = list(record.pending.values())
        record.pending.clear()
        record.loaded = False
    for answers in waiting:
        try:
            answers.put(None)
        except Exception:
            pass


def _drain_stderr(started) -> None:
    """Log the worker's own diagnostics, which never contain content."""
    try:
        for line in iter(started.stderr.readline, b""):
            text = line.decode("utf-8", "replace").strip()
            if text:
                logger.info("Model Chain: %s", text)
    except Exception:
        pass


def _lost(record: _Process, reason: str) -> None:
    """A worker that stopped answering: stopped for good and its record dropped."""
    global _last_error

    with _table_lock:
        _last_error = f"VibeVoice's worker on {record.label()} stopped unexpectedly."
    _discard(record, reason)


def _guard_crash_loop(key: str) -> None:
    now = time.monotonic()
    with _table_lock:
        recent = [when for when in _failures.get(key, ()) if now - when < CRASH_WINDOW]
        _failures[key] = recent
    if len(recent) >= CRASH_LIMIT:
        raise VibeVoiceRuntimeError(
            "VibeVoice has failed to start several times in a row on that card, so it is "
            "not being started again for now. Check the Voice Box tab.")


def _note_failure(key: str) -> None:
    with _table_lock:
        _failures.setdefault(key, []).append(time.monotonic())


# --------------------------------------------------------------------------- #
# Stopping
# --------------------------------------------------------------------------- #


def stop(card_uuid: str = "", reason: str = "") -> None:
    """Stop that card's worker, or every worker. Idempotent, and never raises.

    A lifecycle operation: it does not wait for a render. Whatever was in
    flight fails with a sentence on its own thread.
    """
    key = card_key(card_uuid)
    with _table_lock:
        records = [record for record in _processes.values() if not key or record.key == key]
    for record in records:
        try:
            _discard(record, reason or "VibeVoice stopped")
        except Exception:
            logger.debug("Model Chain: a VibeVoice worker could not be stopped", exc_info=True)


def shutdown() -> None:
    """Door A and the body of doors B and C. Idempotent; must never raise."""
    global _closing

    try:
        with _table_lock:
            _closing = True
        stop("", "WebUI shutdown")
    except Exception:
        logger.debug("Model Chain: the VibeVoice shutdown hook failed", exc_info=True)
    finally:
        with _table_lock:
            _closing = False


def _discard(record: _Process, reason: str) -> None:
    """End the worker and let go of the record, in that order.

    The escalation is bounded at every step, because the thing being stopped
    may be inside a generation that will not look at a pipe again: ask, close
    the pipe, wait, terminate, wait, kill. The record stays in the table until
    the process has exited, so :func:`resident_bytes` keeps answering the
    figure the card has yet to get back; only then is it dropped.
    """
    with _lifecycle_lock(record.key):
        with _table_lock:
            if _processes.get(record.key) is not record:
                return
        with record.lock:
            record.closing = True
            waiting = list(record.pending.values())
            record.pending.clear()
            record.rendering = False
            record.job = ""
        for answers in waiting:
            try:
                answers.put(None)
            except Exception:
                pass

        started = record.process
        if started.poll() is None:
            try:
                with record.write_lock:
                    protocol.write_frame(started.stdin, {"op": "shutdown", "id": "0"})
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
        with _table_lock:
            if _processes.get(record.key) is record:
                del _processes[record.key]
        with record.lock:
            record.loaded = False
        logger.info("Model Chain: the VibeVoice worker on %s stopped — %s", record.label(),
                    reason or "no reason given")


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
    """Doors B and C, arranged once, the first time a worker starts."""
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
        logger.debug("Model Chain: could not chain the %s handler for VibeVoice", name,
                     exc_info=True)


JOB_KILL_ON_CLOSE = 0x00002000
JOB_EXTENDED_LIMIT_INFORMATION = 9


def _die_with_us(started) -> None:
    """Door E on Windows: the job object, arranged before the worker gets work.

    One job for every VibeVoice worker, held for the life of this process on
    purpose: what does the work is the handle being *closed*, which happens
    when this process ends however it ends -- including the kill that runs no
    handler at all, and including a kill that lands mid-render.

    On Linux the equivalent has to run *inside* the child, between fork and the
    first request, so it lives in ``vibevoice_worker/worker.py`` and is reported
    back through the handshake, which :func:`_handshake_with` insists on.
    """
    global _job_handle

    if os.name != "nt":
        return
    handle = getattr(started, "_handle", None)
    if handle is None:
        raise VibeVoiceRuntimeError("The VibeVoice worker could not be tied to this process.")
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
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

    inside = wintypes.BOOL(0)
    if not kernel.IsProcessInJob(wintypes.HANDLE(int(handle)), _job_handle,
                                 ctypes.byref(inside)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not inside.value:
        raise VibeVoiceRuntimeError(
            "The VibeVoice worker could not be tied to this process, so it was not started. "
            "A GPU process that outlives the WebUI is worse than no speech.")
