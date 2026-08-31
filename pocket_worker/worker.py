"""The PocketTTS sidecar: one contained CPU process, and everything Torch touches.

Run by path, under the isolated PocketTTS interpreter, by
:mod:`mc_voice_pocket_runtime`. It never imports a Forge module, a Model Chain
module, sherpa-onnx or Sopro, and it is one of the two processes in this
repository that imports Torch -- the other being Sopro's, which is a *different*
Torch in a different closure (I-PKT-6).

What lives here and why it is not in the parent
-----------------------------------------------
Everything that needs a tensor, which is a shorter list than it looks:

    loading one PocketTTS model from local paths
    loading a precomputed official voice state
    preparing a local reference recording into a voice state
    exporting that state to safetensors, and reading it back
    streaming a committed text unit into PCM
    the delivery DSP -- speed, pitch, volume, pacing

The parent owns product state, voice identity, turn routing and every decision
about *which* voice; this process owns inference and nothing else. It is handed
paths it did not choose and a voice catalogue it did not build, and it validates
both anyway, because it is the process that would crash.

One lane, and why it is not negotiable
---------------------------------------
Upstream documents ``generate_audio_stream()`` as **not thread-safe** and it
uses separate generation and decoder threads internally. So one model instance
serves one inference at a time, from one lane thread, and there is no
configuration in which that becomes two (I-PKT-8). A turn, an audition and a
clone audition all queue behind the same lane.

Interruption is a drain, and it says so
----------------------------------------
Released PocketTTS 3.0.2 exposes no safe cooperative cancellation for an
abandoned stream. Upstream's own change to add one is open rather than merged,
and it states that draining the stream to completion was the correct Python-API
behaviour before it -- because abandoning the generator leaves the generation
thread running for the remainder of the input, and the model being not
thread-safe makes starting the next generation while the old one is alive
incorrect.

So ``tts_interrupt`` here does four things and does not pretend to do a fifth:

    the turn is marked interrupted, so no more text is accepted for it;
    every queued not-yet-started unit for it is dropped;
    the call already inside the model is **kept being consumed** to its
        ordinary completion, and everything it yields is thrown away here
        rather than written to the pipe;
    when the call returns, ``tts_interrupted`` with ``state="complete"`` tells
        the parent the lane is free.

The third is the one that is easy to get wrong. Pocket's internal latent and
result queues are ordinary unbounded ``queue.Queue`` instances, so a consumer
that stopped pulling would not stop the producer -- it would only leave a
generation thread running with nothing draining it (I-PKT-12, section 5.3).

Nothing here is a local copy of upstream's unmerged cancellation. When it is
merged and this project deliberately adopts a reviewed release,
:data:`INTERRUPT_MODE` becomes ``cooperative``, the interrupt sets upstream's
Event instead of running to the end, and the parent's state machine and the
browser's Stop do not change (I-PKT-14, section 21.7).

Speed is DSP here, and says so
------------------------------
The reviewed PocketTTS generation API has no speaking-rate argument. Pocket
Speed is therefore a pitch-preserving time-domain transformation around the
model output, and a naive resample that transposes the voice is not parity and
does not ship (section 15).

:class:`Stretch` is that transformation -- streaming SOLA, state carried across
the chunk boundaries Pocket streams at -- and :class:`Resampler` is the
independent Pitch control composed on top of it:

    time-scale asked of SOLA   = speed / pitch_ratio
    resample ratio afterwards  = pitch_ratio

At neutral both are 1.0 and both objects are skipped entirely, so an
installation that never moves a slider pays nothing for them.

Those three classes are the same arithmetic as ``sopro_worker``'s, and they are
*deliberately duplicated rather than shared* (section 15, section 49.1). A
shared module would have to be importable from inside two isolated closures,
which is exactly the coupling the separate runtimes exist to prevent. The
duplication is held honest by both being run against the same golden signal
tests: fundamental unmoved across the speed range, duration unmoved across the
pitch range, no click at a chunk boundary, exact scaling below unity, limiting
above it.

Nothing durable is a warmed cache
---------------------------------
The loaded model and the bounded LRU of voice states are worker cache. What is
written to disk is a safetensors voice state under a directory the parent named,
validated by being read back before anything calls it a voice. And a cached base
state stays a *base* state: generation is asked to copy it, never to mutate it,
so a voice warmed this morning does not slowly become the continuation of every
sentence it has spoken since (I-PKT-16).
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import queue
import struct
import sys
import threading
import time
import wave

PROTOCOL_VERSION = 1
"""The PocketTTS protocol's own version, counted from one.

Not shared with :data:`voice_worker.worker.PROTOCOL_VERSION` or
:data:`sopro_worker.worker.PROTOCOL_VERSION`. The three workers speak the same
*framing* and a deliberately overlapping set of turn operations, and they are
still three protocols: Pocket carries ``voice_id`` where Kokoro carries ``sid``,
and it has one operation -- ``tts_interrupt`` -- that means something neither of
the others can mean. One number covering all three would be a number that has to
change when any of them changes.
"""

MARKER = "--model-chain-pocket-worker"
"""On the command line so this process is recognisable in a task manager and by
the stray sweep. Never a voice name, never a path, never spoken text: a command
line is world readable on most systems."""

MAX_HEADER = 1 << 20
MAX_PAYLOAD = 40 << 20
"""A fifteen-second 24 kHz mono PCM16 reference is under a megabyte; the ceiling
is generous because the parent's route in front of it is the real limit, and it
is still a number rather than "whatever arrives"."""

SAMPLE_RATE = 24000
"""What PocketTTS produces. Asserted against the loaded config at start-up
rather than assumed -- a model revision that changed it would otherwise be a
worker emitting frames the browser resamples into a chipmunk."""

INTERRUPT_MODE = "drain_unit"
"""What this worker can promise when a turn is interrupted.

Reported in the handshake and read by the parent, which refuses a mode its own
state machine does not implement. ``drain_unit`` is released 3.0.2's honest
answer; ``cooperative`` is what this becomes when a merged upstream
cancellation is adopted, and the only other thing that changes then is what
:meth:`Worker.interrupt` does with the flag it already sets.
"""

_LENGTH = struct.Struct(">I")


# --------------------------------------------------------------------------- #
# Framing -- byte-identical to the other two workers, by agreement not by import
# --------------------------------------------------------------------------- #


def read_frame(stream) -> "tuple[dict, bytes] | None":
    """One request, or ``None`` at end of input.

    ``None`` is how the parent's death arrives when this process is waiting for
    work, and it is not an error: the loop ends, the model is released, and the
    process exits with 0. Door D of the five.
    """
    header_length = _read_exactly(stream, 4)
    if header_length is None:
        return None
    (size,) = _LENGTH.unpack(header_length)
    if size > MAX_HEADER:
        raise ValueError("header too large")
    raw = _read_exactly(stream, size)
    if raw is None:
        return None
    header = json.loads(raw.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("header is not an object")

    payload_length = _read_exactly(stream, 4)
    if payload_length is None:
        return None
    (size,) = _LENGTH.unpack(payload_length)
    if size > MAX_PAYLOAD:
        raise ValueError("payload too large")
    payload = b"" if size == 0 else _read_exactly(stream, size)
    if payload is None:
        return None
    return header, payload


def write_frame(stream, header: dict, payload: bytes = b"") -> None:
    raw = json.dumps(header).encode("utf-8")
    stream.write(_LENGTH.pack(len(raw)))
    stream.write(raw)
    stream.write(_LENGTH.pack(len(payload)))
    if payload:
        stream.write(payload)
    stream.flush()


def _read_exactly(stream, count: int) -> "bytes | None":
    chunks = []
    remaining = count
    while remaining > 0:
        block = stream.read(remaining)
        if not block:
            return None
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


# --------------------------------------------------------------------------- #
# Audio, in stdlib only
# --------------------------------------------------------------------------- #


def decode_wav(data: bytes) -> "tuple[list, int]":
    """A validated PCM16 mono WAV as float samples in [-1, 1].

    Validated even though the parent's route already checked, for the same
    reason the other two workers validate: this is the process that would crash,
    and a worker that dies on a malformed upload is a worker the next clone has
    to restart.
    """
    import array
    import contextlib

    with contextlib.closing(wave.open(io.BytesIO(data), "rb")) as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        if channels != 1:
            raise ValueError("audio is not mono")
        if width != 2:
            raise ValueError("audio is not 16-bit")
        if frames <= 0:
            raise ValueError("audio is empty")
        raw = handle.readframes(frames)

    samples = array.array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if sys.byteorder == "big":
        samples.byteswap()
    scale = 1.0 / 32768.0
    return [value * scale for value in samples], rate


def encode_wav(pcm: bytes, rate: int) -> bytes:
    """Already-quantised PCM16 as a WAV, in memory.

    Takes bytes rather than floats because everything in this process that has
    audio to hand over has already been through :func:`pcm16` -- the streaming
    path has to be, and an audition that quantised separately would be a second
    implementation of the one step where clipping is decided.
    """
    import contextlib

    buffer = io.BytesIO()
    with contextlib.closing(wave.open(buffer, "wb")) as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(pcm)
    return buffer.getvalue()


def pcm16(samples, gain: float = 1.0) -> bytes:
    """Float samples as little-endian PCM16, with a soft limiter above unity.

    The limiter is not decoration. Volume is a scalar on samples a model
    produced at its own level, and a +6 dB setting on a loud passage would clip
    into distortion that sounds like a broken model rather than like a loud one.
    Below unity there is nothing to limit and the knee is skipped.
    """
    import array

    numbers = array.array("h")
    level = float(gain)
    knee = 0.9
    out = numbers.append
    for value in samples:
        scaled = float(value) * level
        if level > 1.0:
            if scaled > knee:
                scaled = knee + (1.0 - knee) * _tanh((scaled - knee) / (1.0 - knee))
            elif scaled < -knee:
                scaled = -(knee + (1.0 - knee) * _tanh((-scaled - knee) / (1.0 - knee)))
        quantised = int(scaled * 32767.0)
        if quantised > 32767:
            quantised = 32767
        elif quantised < -32768:
            quantised = -32768
        out(quantised)
    if sys.byteorder == "big":
        numbers.byteswap()
    return numbers.tobytes()


def _tanh(value: float) -> float:
    import math

    return math.tanh(value)


def silence(rate: int, milliseconds: int) -> bytes:
    """A gap between committed speech units, as PCM16 zeros."""
    count = int(max(0, int(milliseconds)) * int(rate or 0) / 1000)
    return b"\x00\x00" * count


# --------------------------------------------------------------------------- #
# Dying with the parent
# --------------------------------------------------------------------------- #


def _containment(parent_pid: int) -> str:
    """Ask the OS to end this process when the parent ends, and report what it got.

    On Linux ``PR_SET_PDEATHSIG`` is set to SIGKILL and then the parent pid is
    re-read. The re-read is the whole point: if the parent died between its fork
    and this line, the death signal it would have sent has already not been
    sent, and this process would sit here forever holding a Torch runtime.

    On Windows the job object is the parent's to create, and it has already done
    it by the time this runs -- so this side *checks* rather than claims.
    ``IsProcessInJob`` turns "the parent says it arranged containment" into
    evidence from the kernel. A Windows process that is not in a job says
    ``none``, and the parent, which proved its own job membership with real
    handles, treats this side's answer as corroboration.
    """
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            PR_SET_PDEATHSIG = 1
            if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        except Exception as exc:  # noqa: BLE001 - reported, never raised onward
            _note(f"parent-death containment unavailable: {exc.__class__.__name__}")
            return "pipe"
        if parent_pid and os.getppid() != parent_pid:
            _note("parent went away during start-up")
            raise SystemExit(0)
        return "pdeathsig"
    if os.name == "nt":
        return _in_a_job()
    return "pipe"


def _in_a_job() -> str:
    """Whether the kernel agrees this process is in a job. Three answers.

    ``job`` the kernel says yes; ``none`` the kernel says no; ``unknown`` the
    question could not be put at all. Three rather than two because folding the
    third into the second cost a real user the whole Sopro feature once:
    containment had been arranged and was being enforced, and the worker was
    turned away for failing to confirm it.

    The argument types are declared because a HANDLE is not a C ``int`` on
    64-bit Windows, and ``GetCurrentProcess`` returns the pseudo-handle -1 --
    the one value where getting that wrong matters most.
    """
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
                                            ctypes.POINTER(wintypes.BOOL)]
        kernel32.IsProcessInJob.restype = wintypes.BOOL
        inside = wintypes.BOOL(0)
        if not kernel32.IsProcessInJob(kernel32.GetCurrentProcess(), None,
                                       ctypes.byref(inside)):
            raise OSError(ctypes.get_last_error(), "IsProcessInJob failed")
    except Exception as exc:  # noqa: BLE001 - reported, never raised onward
        _note(f"could not ask whether this process is in a job: "
              f"{exc.__class__.__name__}: {exc}")
        return "unknown"
    if not inside.value:
        _note("the kernel says this process is not in a job object")
        return "none"
    return "job"


# --------------------------------------------------------------------------- #
# The CPU policy PocketTTS chose for itself
# --------------------------------------------------------------------------- #


def thread_policy() -> str:
    """What this build can honestly say about PocketTTS's CPU execution.

    A sentence, and there is deliberately no number in it to set. PocketTTS
    3.0.2 calls ``torch.set_num_threads(1)`` itself and gets its parallelism
    from its own generation and decoder threads, so an ``OMP_NUM_THREADS``
    chosen here and reported as a Pocket thread count would be reporting
    something untrue (section 16.4, section 35). Sopro's worker has an
    intra-op policy and a benchmark override; this one has a sentence, and that
    difference is the honest one rather than an omission.
    """
    return "PocketTTS sets its own CPU thread policy (torch.set_num_threads(1))"


def _note(text: str) -> None:
    """One diagnostic line on stderr, which the parent logs.

    Never spoken text, never a voice name, never a path (I-PKT-27). Everything
    written here is a class name, a count, a duration or a state.
    """
    try:
        sys.stderr.write(f"PocketTTS worker: {text}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _safe(exc: BaseException) -> str:
    """An exception as something that can cross the pipe.

    The class name, and only the class name, for anything this process did not
    raise itself. A library's own message can carry a path, a tensor shape or
    the text that was being spoken, and a worker that forwarded one would put
    conversation content in the parent's log by accident (I-PKT-27).

    A :class:`ValueError` is this file's own refusal -- already a sentence,
    already free of anything private -- and is forwarded as written.
    """
    if isinstance(exc, ValueError):
        return str(exc)
    return exc.__class__.__name__


# --------------------------------------------------------------------------- #
# Delivery, and the DSP that makes Speed mean something
# --------------------------------------------------------------------------- #


def _bounded(value, limits, fallback):
    """A number inside ``limits``, or ``fallback`` for anything that is not one.

    Total, because every caller is a JSON header from another process. None of
    them is a reason for a reply to go unspoken.
    """
    low, high = limits
    if value is None or isinstance(value, bool):
        return fallback
    try:
        found = float(value)
    except (TypeError, ValueError):
        return fallback
    if found != found or found in (float("inf"), float("-inf")):
        return fallback
    return min(max(found, low), high)




class Delivery:
    """One turn's delivery and generation settings, as the header carries them.

    Two groups that are deliberately not mixed. Speed, pitch, gain and pause are
    *this process's arithmetic* on audio the model produced. Temperature is
    PocketTTS's own argument and is passed through untouched. Keeping them in
    one object with two accessors is how the code says which is which
    (I-PKT-23, section 15).

    Sopro's equivalent also carries language, top-p, top-k, a solver step count
    and a chunk size. Pocket's does not, and none of the absences is an
    oversight: the model is the language and it is engine-global; the sampler
    step count is an engine setting rather than a per-turn one; and top-p, top-k
    and a seed are advanced controls this release does not expose because
    nobody has given them a user-facing purpose yet (section 16.5).
    """

    __slots__ = ("speed", "pitch", "gain", "pause_ms", "temperature")

    def __init__(self, speed=1.0, pitch=1.0, gain=1.0, pause_ms=0, temperature=None):
        self.speed = _bounded(speed, (0.5, 2.0), 1.0)
        self.pitch = _bounded(pitch, (0.25, 4.0), 1.0)
        self.gain = _bounded(gain, (0.05, 8.0), 1.0)
        self.pause_ms = int(_bounded(pause_ms, (0, 1200), 0))
        self.temperature = (None if temperature is None
                            else _bounded(temperature, (0.05, 2.0), 0.3))

    @classmethod
    def from_header(cls, header: dict) -> "Delivery":
        found = dict(header or {})
        return cls(speed=found.get("speed", 1.0), pitch=found.get("pitch", 1.0),
                   gain=found.get("gain", 1.0), pause_ms=found.get("pause_ms", 0),
                   temperature=found.get("temperature"))

    @property
    def stretch_rate(self) -> float:
        """The time-scale SOLA is asked for, once Pitch has been accounted for.

        The composition in one line. Resampling by ``pitch`` will shorten the
        audio by ``pitch`` as a side effect of transposing it, so the stretch
        before it has to be ``speed / pitch`` for the finished duration to be
        the one Speed asked for.
        """
        return float(self.speed) / float(self.pitch or 1.0)

    @property
    def shapes(self) -> bool:
        """Whether anything here changes a sample at all. The neutral fast path."""
        return not (abs(self.stretch_rate - 1.0) < 1e-6
                    and abs(self.pitch - 1.0) < 1e-6
                    and abs(self.gain - 1.0) < 1e-6)

    def generation(self, config) -> dict:
        """PocketTTS's own arguments for this turn, with the model's defaults kept.

        ``None`` means "whatever the selected model configuration says", which
        is what an untouched Variation control stores. Upstream accepts
        ``temp=None`` and reads the model's own recommendation from its config,
        so materialising today's number into the request would freeze it and a
        model revision that changed its recommendation would then not change
        anybody's voice (I-PKT-25, section 14).
        """
        found = {}
        if self.temperature is not None:
            found["temperature"] = self.temperature
        return found


NEUTRAL = Delivery()


class Stretch:
    """Streaming SOLA: change duration without changing pitch. Section 15.

    Synchronized overlap-add, which is the classical answer and the one whose
    failure modes are understood. The signal is cut into windows of ``2 * HOP``,
    each window is shifted to the position that best correlates with the audio
    already written, and the overlap is cross-faded with a periodic Hann. Moving
    the *analysis* hop while holding the *synthesis* hop fixed is what changes
    the duration; searching for the best offset is what keeps the waveform's
    periods lined up, which is what stops it sounding like a phaser.

    Streaming is the part that costs something. PocketTTS yields decoded chunks
    as its decoder thread produces them, and a stretcher restarted at each one
    would put a discontinuity at every chunk boundary -- an audible tick several
    times a second. So the input buffer, the fractional read position and the
    pending overlap tail all survive between calls, and :meth:`flush` is what
    empties the last of it at the end of a committed unit.

    Bounded by construction: the buffer holds at most one window plus the search
    radius beyond the current read position, so a caller that pushes a whole
    reply through it cannot make this object grow.
    """

    HOP = 240
    """Ten milliseconds at 24 kHz. Small enough that the search below stays
    cheap, long enough to contain a period of a low male voice (down to about
    100 Hz), which is what SOLA needs to find in order to line anything up."""

    SEARCH = 180
    """How far either side of the nominal position an offset may be taken from.

    Three quarters of a hop. Wide enough to reach a neighbouring pitch period at
    ordinary speaking fundamentals, narrow enough that the correlation is a few
    thousand multiplies rather than a transform.
    """

    def __init__(self, rate: float, numpy_module):
        self._np = numpy_module
        self.rate = float(rate)
        window = 2 * self.HOP
        index = self._np.arange(window, dtype=self._np.float32)
        # Periodic rather than symmetric Hann: at 50% overlap the periodic form
        # sums to exactly one, so the overlap-add is unity-gain and a stretched
        # passage is not quietly louder than an unstretched one.
        self._window = (0.5 - 0.5 * self._np.cos(
            2.0 * self._np.pi * index / float(window))).astype(self._np.float32)
        self._buffer = self._np.zeros(0, dtype=self._np.float32)
        self._origin = 0
        self._filled = 0
        self._read = 0.0
        self._tail = self._np.zeros(self.HOP, dtype=self._np.float32)
        self._primed = False

    def block(self, samples):
        """Take one chunk of input and return whatever output it completed.

        May return nothing at all -- at rates below one, several input chunks
        are needed before a synthesis frame can be placed -- and that is not a
        stall: the caller emits what it gets and the audio arrives on the next
        chunk.
        """
        np = self._np
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if incoming.size:
            self._buffer = np.concatenate([self._buffer, incoming])
            self._filled += int(incoming.size)
        return self._drain(final=False)

    def flush(self):
        """Everything that is left, at the end of one committed unit.

        The buffer is zero-padded *once* so that the last real sample can still
        sit inside a full window, and the loop then stops at the last real
        sample rather than at the end of the padding -- which is the difference
        between finishing a sentence and generating silence forever.

        The pending overlap tail is included. Dropping it would take ten
        milliseconds off the end of every unit, which over a paragraph is the
        difference between speech and clipped speech.
        """
        np = self._np
        found = self._drain(final=True)
        tail, self._tail = self._tail, np.zeros(self.HOP, dtype=np.float32)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._origin = 0
        self._filled = 0
        self._read = 0.0
        self._primed = False
        if tail.size:
            found = np.concatenate([found, tail]) if found.size else tail
        return found

    def _drain(self, final: bool):
        """Place as many synthesis frames as the input on hand allows.

        The two modes differ in one condition and nothing else. Streaming stops
        when the *search window* would reach past what has arrived, because a
        frame chosen without its full search span is a frame chosen badly.
        Flushing stops when the *read position* passes the last real sample,
        because everything after that is padding this method added.
        """
        np = self._np
        window = 2 * self.HOP
        reach = window + self.SEARCH
        if final:
            wanted = int(round(self._read)) - self._origin + reach
            if wanted > self._buffer.size:
                self._buffer = np.concatenate(
                    [self._buffer, np.zeros(wanted - self._buffer.size, dtype=np.float32)])
        produced = []
        while True:
            here = int(round(self._read))
            if final:
                if here >= self._filled:
                    break
                if here - self._origin + reach > self._buffer.size:
                    self._buffer = np.concatenate(
                        [self._buffer,
                         np.zeros(here - self._origin + reach - self._buffer.size,
                                  dtype=np.float32)])
            elif here - self._origin + reach > self._buffer.size:
                break
            offset = self._align(here - self._origin)
            frame = self._buffer[offset:offset + window]
            if frame.size < window:
                break
            shaped = frame * self._window
            produced.append(self._tail + shaped[:self.HOP])
            self._tail = shaped[self.HOP:].copy()
            self._primed = True
            self._read += self.HOP * self.rate
            self._trim()
        if not produced:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(produced)

    def _align(self, centre: int) -> int:
        """The offset in the search span whose start best matches the pending tail.

        Plain cross-correlation of a candidate's rising half against the falling
        half already written, normalised by the candidate's own energy so that a
        loud frame beside a quiet one is not chosen merely for being loud --
        which is what makes this track a pitch period rather than an amplitude.

        The first frame of a unit has nothing to match against, so it is placed
        where it was asked for.
        """
        np = self._np
        last = self._buffer.size - 2 * self.HOP
        if last < 0:
            return 0
        centre = max(0, min(int(centre), last))
        if not self._primed:
            return centre
        low = max(0, centre - self.SEARCH)
        high = min(last, centre + self.SEARCH)
        if high <= low:
            return centre
        reference = self._tail
        if float(np.abs(reference).max()) < 1e-6:
            return centre
        span = self._buffer[low:high + self.HOP]
        if span.size < self.HOP:
            return centre
        # One correlation over the whole search span rather than a loop: the
        # sliding dot products are a single small matrix multiply, cheap enough
        # to run on every ten milliseconds of speech.
        count = high - low + 1
        strides = np.lib.stride_tricks.sliding_window_view(span, self.HOP)[:count]
        scores = strides @ reference
        energy = np.sqrt((strides * strides).sum(axis=1)) + 1e-6
        return int(low + int(np.argmax(scores / energy)))

    def _trim(self) -> None:
        """Forget input the next search can no longer reach. Keeps this bounded."""
        keep = int(round(self._read)) - self.SEARCH - self.HOP
        drop = keep - self._origin
        if drop > self.HOP:
            self._buffer = self._buffer[drop:]
            self._origin += drop


class Resampler:
    """Linear resampling with the read position carried between chunks.

    The Pitch control. Reading the stretched audio at a step of ``ratio``
    multiplies every frequency by ``ratio`` and divides the duration by it --
    which is why :attr:`Delivery.stretch_rate` divides the requested speed by
    the same number before this runs. Formants move with the pitch, so this
    reads as a differently-sized speaker rather than as a transposition, which
    is the same honest description :mod:`mc_voice_profile` gives for Kokoro.

    The fractional position and the last sample of the previous chunk are both
    kept, because starting each chunk at zero would put a step discontinuity --
    a click -- at every chunk boundary.
    """

    def __init__(self, ratio: float, numpy_module):
        self._np = numpy_module
        self.ratio = float(ratio)
        self._position = 0.0
        self._carry = self._np.zeros(0, dtype=self._np.float32)

    def block(self, samples):
        np = self._np
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._carry.size:
            incoming = np.concatenate([self._carry, incoming])
        if incoming.size < 2:
            self._carry = incoming
            return np.zeros(0, dtype=np.float32)
        # How many output samples this chunk can produce without reading past
        # its own last sample. The remainder becomes the carry.
        last = incoming.size - 1
        count = int((last - self._position) / self.ratio) + 1
        if count <= 0:
            self._carry = incoming
            return np.zeros(0, dtype=np.float32)
        positions = self._position + self.ratio * np.arange(count, dtype=np.float64)
        positions = positions[positions <= last]
        if positions.size == 0:
            self._carry = incoming
            return np.zeros(0, dtype=np.float32)
        left = positions.astype(np.int64)
        weight = (positions - left).astype(np.float32)
        right = np.minimum(left + 1, last)
        found = (incoming[left] * (1.0 - weight) + incoming[right] * weight)
        consumed = int(positions[-1])
        self._position = float(positions[-1] + self.ratio - consumed)
        self._carry = incoming[consumed:]
        return found.astype(np.float32)

    def flush(self):
        np = self._np
        tail, self._carry = self._carry, np.zeros(0, dtype=np.float32)
        self._position = 0.0
        if tail.size <= 1:
            return np.zeros(0, dtype=np.float32)
        return tail[1:]


class Shaper:
    """Speed, then pitch, then volume -- one committed unit's worth of state.

    One per unit rather than one per turn, because both stages carry a
    fractional read position and restarting them mid-sentence would put a click
    at the boundary; and not one per *chunk*, because restarting them there
    would put a click several times a second.

    The neutral path is a bypass rather than an identity transform: at speed 1,
    pitch 0 and volume 0 dB, samples go straight to :func:`pcm16` and neither
    object above is constructed at all.
    """

    def __init__(self, delivery: Delivery, numpy_module):
        self.delivery = delivery or NEUTRAL
        self._np = numpy_module
        self._stretch = None
        self._resample = None
        if numpy_module is None or not self.delivery.shapes:
            return
        if abs(self.delivery.stretch_rate - 1.0) > 1e-6:
            self._stretch = Stretch(self.delivery.stretch_rate, numpy_module)
        if abs(self.delivery.pitch - 1.0) > 1e-6:
            self._resample = Resampler(self.delivery.pitch, numpy_module)

    @property
    def active(self) -> bool:
        return self._stretch is not None or self._resample is not None

    def block(self, samples) -> bytes:
        found = samples
        if self._stretch is not None:
            found = self._stretch.block(found)
        if self._resample is not None:
            found = self._resample.block(found)
        return pcm16(found, self.delivery.gain)

    def flush(self) -> bytes:
        np = self._np
        found = None
        if self._stretch is not None:
            found = self._stretch.flush()
        if self._resample is not None:
            if found is not None and getattr(found, "size", 0):
                found = np.concatenate([self._resample.block(found),
                                        self._resample.flush()])
            else:
                found = self._resample.flush()
        if found is None or not getattr(found, "size", 0):
            return b""
        return pcm16(found, self.delivery.gain)


# --------------------------------------------------------------------------- #
# The model, and the voice states it speaks from
# --------------------------------------------------------------------------- #

STATE_SCHEMA = 1
"""The exported voice-state layout this worker writes and will read back."""

VOICE_CACHE = 8
"""How many voice states are kept loaded at once.

Bounded, and not a user setting (section 22). Eight is enough for ordinary
character switching, a safetensors state is small next to the model, and an
unbounded dictionary in a process that runs for a WebUI session is a leak with a
slow fuse. If a measurement ever justifies a different number it becomes a
different constant, not a slider.
"""

MAX_STATE_BYTES = 96 * 1024 * 1024
"""A ceiling on what may be read back as a voice state.

A safetensors file is a length-prefixed JSON header followed by tensors, and
nothing stops a corrupt one declaring a gigabyte. This process allocates what
the header asks for, so the header is checked first.
"""


class Engine:
    """One PocketTTS model, one inference lane, and a bounded state cache.

    Everything Torch-shaped is behind this class, and it is constructed once per
    worker. The parent hands it a config file whose every path names a local
    file the parent already verified: there is no repository id in it, no URL,
    and no credential (I-PKT-20, section 25).

    The upstream API this speaks to
    -------------------------------
    Released PocketTTS 3.0.2 offers, among others, ``TTSModel.load_model``,
    ``get_state_for_audio_prompt`` and ``generate_audio_stream``. This class
    resolves those by name at load time and refuses with a sentence if the
    installed package does not have them, rather than failing later inside a
    turn on an ``AttributeError`` nobody can read. GATE P-0 is the throwaway
    harness that proves the shapes against a real build before any of this is
    called released.
    """

    def __init__(self, config: dict):
        found = dict(config or {})
        self.model_root = str(found.get("model_root") or "")
        self.config_path = str(found.get("config_path") or "")
        self.official_root = str(found.get("official_root") or "")
        self.clones_root = str(found.get("clones_root") or "")
        self.model_id = str(found.get("model_id") or "")
        self.precision = str(found.get("precision") or "full")
        self.sampler_steps = int(found.get("sampler_steps") or 1)
        self.fingerprint = str(found.get("fingerprint") or "")
        self.state_schema = int(found.get("state_schema") or STATE_SCHEMA)
        self.sample_rate = int(found.get("sample_rate") or SAMPLE_RATE)
        self.cloning_ready = bool(found.get("cloning_ready"))
        self.voices = dict(found.get("voices") or {})
        self.model = None
        self.defaults = {}
        self.upstream_build_id = ""
        self._numpy = None
        self._states = collections.OrderedDict()
        self._lock = threading.Lock()

    # -- loading ----------------------------------------------------------- #

    def load(self) -> None:
        """Import PocketTTS and Torch, and load one model from local paths.

        The only place in this process that imports either. Everything about it
        is arranged so that a failure is a sentence rather than a traceback: the
        config is read here, the callables are resolved by name here, and the
        device is asserted here rather than assumed from an environment
        variable somebody could have unset.
        """
        import torch

        try:
            import numpy
        except Exception:  # noqa: BLE001 - an optional accelerator, never required
            numpy = None
        self._numpy = numpy

        from pocket_tts import TTSModel  # noqa: F401 - resolved below by name

        config = self._read_config()
        loader = getattr(TTSModel, "load_model", None)
        if loader is None:
            raise ValueError("This PocketTTS build has no TTSModel.load_model, so Voice "
                             "Chat cannot load a model with it.")
        arguments = {"config": config, "device": "cpu"}
        if self.precision == "int8":
            # Upstream's dynamic INT8 at load time. Named ``quantize`` in 3.0.2;
            # passed only when it was asked for, so a build that does not accept
            # the argument still loads at full precision rather than refusing.
            arguments["quantize"] = True
        self.model = loader(**arguments)
        for name in ("generate_audio_stream",):
            if getattr(self.model, name, None) is None:
                raise ValueError(f"This PocketTTS build has no {name}, so Voice Chat "
                                 f"cannot speak with it.")
        self.upstream_build_id = str(getattr(self.model, "build_id", "")
                                     or config.get("upstream_build_id") or "")
        self.defaults = self._read_defaults(config)
        rate = int(getattr(self.model, "sample_rate", 0) or config.get("sample_rate") or 0)
        if rate:
            if rate != self.sample_rate:
                _note(f"model reports {rate} Hz; the parent expected {self.sample_rate}")
            self.sample_rate = rate
        # Asserted rather than assumed. The environment empties every GPU
        # variable before this process starts, and this is the check that the
        # emptying worked (I-PKT-7).
        if str(self.device()) != "cpu":
            raise ValueError(f"PocketTTS loaded on {self.device()} rather than the CPU.")
        _note(f"loaded {self.model_id} at {self.precision}, {self.sampler_steps} step(s), "
              f"{self.sample_rate} Hz")

    def _read_config(self) -> dict:
        """The local-only config the parent wrote. Never a URL, never a repo id.

        Checked here as well as written there, because this is the process that
        would go to the network if a location ever got into it -- and a worker
        that refused one is a worker whose offline claim is enforced rather than
        asserted (I-PKT-20).
        """
        if not self.config_path or not os.path.isfile(self.config_path):
            raise ValueError("PocketTTS has no local model configuration. Reinstall it in "
                             "Settings → Voice Chat.")
        with open(self.config_path, "r", encoding="utf-8") as handle:
            found = json.load(handle)
        if not isinstance(found, dict):
            raise ValueError("PocketTTS's local model configuration is not readable.")
        for key, value in found.items():
            text = str(value)
            if text.startswith("hf://") or text.startswith("http://") \
                    or text.startswith("https://"):
                raise ValueError(f"PocketTTS's local configuration names a network location "
                                 f"for {key}, which this worker will not resolve.")
        return found

    def _read_defaults(self, config: dict) -> dict:
        """The model's own generation defaults, for the panel to show as "model
        default" rather than as a number this build invented (I-PKT-25)."""
        found = {}
        recommended = config.get("recommended_temperature")
        if recommended is not None:
            found["temperature"] = float(recommended)
        for name in ("temperature",):
            value = getattr(getattr(self.model, "config", None), name, None)
            if value is not None:
                found[name] = float(value)
        found["sampler_steps"] = self.sampler_steps
        found["ref_seconds"] = float(config.get("ideal_reference_seconds") or 0.0)
        return found

    def device(self) -> str:
        found = getattr(self.model, "device", None)
        return str(found) if found is not None else "cpu"

    def generation(self, delivery: Delivery) -> dict:
        """The arguments one generation is asked for.

        The sampler step count is the engine's, not the turn's: it is an engine
        setting because it changes the generation policy, and a turn that
        changed it in its middle would be a turn whose second half was a
        different model configuration from its first (I-PKT-24).
        """
        found = {"sampler_decode_steps": self.sampler_steps}
        found.update((delivery or NEUTRAL).generation(self.defaults))
        return found

    # -- voice states ------------------------------------------------------ #

    def refresh_catalog(self, voices, forget=()) -> int:
        """Replace the catalogue and drop the cached states it no longer names."""
        with self._lock:
            self.voices = dict(voices or {})
            for name in list(forget or ()):
                self._states.pop(str(name), None)
            for name in list(self._states):
                if name not in self.voices:
                    self._states.pop(name, None)
            return len(self.voices)

    def forget(self, voice_id: str) -> None:
        with self._lock:
            self._states.pop(str(voice_id or ""), None)

    def state_for(self, voice_id: str):
        """The model state for one voice, from the bounded cache or from disk.

        Loading a safetensors state is much cheaper than re-encoding reference
        audio, and ordinary conversation must never re-run voice preparation --
        so this is a cache with a bound on it rather than a dictionary
        (section 22).

        What comes back is a *base* state and stays one. Generation is asked to
        copy it (see :meth:`stream`), so a voice used for forty replies is the
        same tensors on the fortieth as on the first, rather than the
        accumulated continuation of the previous thirty-nine (I-PKT-16, GATE
        P-11).
        """
        wanted = str(voice_id or "")
        with self._lock:
            found = self._states.get(wanted)
            if found is not None:
                self._states.move_to_end(wanted)
                return found
            entry = self.voices.get(wanted)
        if not isinstance(entry, dict):
            raise ValueError("That voice is not one PocketTTS has been given.")
        path = str(entry.get("state") or "")
        if not path or not os.path.isfile(path):
            raise ValueError("That voice's prepared data is missing. Rebuild it in "
                             "Settings → Voice Chat.")
        state = self._load_state(path)
        with self._lock:
            self._states[wanted] = state
            self._states.move_to_end(wanted)
            while len(self._states) > VOICE_CACHE:
                self._states.popitem(last=False)
        return state

    def _load_state(self, path: str):
        """One exported voice state, checked before a byte of it is allocated.

        The header is read and bounded first. A safetensors file declares its
        own header length, and a corrupt or hostile one can declare a number
        this process would then try to allocate.
        """
        size = os.path.getsize(path)
        if size <= 8 or size > MAX_STATE_BYTES:
            raise ValueError("That voice's prepared data is not a size PocketTTS can read.")
        with open(path, "rb") as handle:
            head = handle.read(8)
        (declared,) = struct.unpack("<Q", head)
        if declared <= 0 or declared + 8 > size:
            raise ValueError("That voice's prepared data does not have a safetensors "
                             "header.")
        loader = getattr(self.model, "load_state", None) or \
            getattr(self.model, "get_state_for_audio_prompt", None)
        if loader is None:
            raise ValueError("This PocketTTS build cannot load an exported voice state.")
        return loader(path)

    def prepare(self, wav_bytes: bytes, seconds: float = 0.0) -> dict:
        """A reference recording as an exported voice state, written and read back.

        Written and read back, in that order, and the reading is not ceremony
        (section 26.2). A preparation that returned without an exception says
        nothing about whether the file it wrote can be loaded after a restart,
        and the audition the user is about to hear is synthesised from what came
        back off the disk rather than from what stayed in memory.
        """
        if not self.cloning_ready:
            raise ValueError("PocketTTS's voice-cloning weights are not installed, so it "
                             "cannot make a voice from a recording.")
        samples, rate = decode_wav(wav_bytes)
        if rate != self.sample_rate:
            raise ValueError(f"That recording is {rate} Hz and PocketTTS wants "
                             f"{self.sample_rate} Hz.")
        if not samples:
            raise ValueError("That recording is empty.")
        extract = getattr(self.model, "get_state_for_audio_prompt", None)
        if extract is None:
            raise ValueError("This PocketTTS build cannot make a voice from a recording.")
        return {"state": extract(wav_bytes), "seconds": float(seconds or 0.0),
                "sample_rate": rate}

    def export(self, state, path: str) -> int:
        """Write one voice state to safetensors and return its size in bytes."""
        save = getattr(self.model, "export_voice", None) or \
            getattr(self.model, "save_state", None)
        if save is None:
            raise ValueError("This PocketTTS build cannot export a voice state.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        staging = f"{path}.new"
        save(state, staging)
        os.replace(staging, path)
        return int(os.path.getsize(path))

    # -- speaking ---------------------------------------------------------- #

    def stream(self, text: str, voice_id: str, delivery: Delivery, on_audio,
               listening) -> dict:
        """Synthesise one committed unit, and hand its PCM over as it arrives.

        ``listening`` is called before every block and decides whether the block
        is offered onward. It is deliberately *not* a way to stop generating:
        when it goes false this loop keeps pulling from PocketTTS's generator
        and throws away what it gets, because abandoning the generator would
        leave upstream's generation thread running against an unbounded internal
        queue with nobody draining it (I-PKT-12, section 5.3).

        ``copy_state=True`` is passed explicitly rather than relied on as a
        default. It is what keeps a cached base state reusable: without it a
        generation may leave the state it was handed carrying this sentence's
        continuation, and the next reply in that voice would start from the
        middle of the last one (I-PKT-16).
        """
        state = self.state_for(voice_id)
        began = time.monotonic()
        first = 0.0
        blocks = 0
        samples = 0
        shaper = Shaper(delivery, self._numpy)
        stream = self.model.generate_audio_stream(
            state, str(text or ""), copy_state=True, **self.generation(delivery))
        for chunk in stream:
            if first == 0.0:
                first = time.monotonic()
            blocks += 1
            block = self._as_pcm(chunk, shaper)
            samples += len(block) // 2
            if block and listening():
                on_audio(block)
        tail = shaper.flush() if shaper.active else b""
        if tail:
            samples += len(tail) // 2
            if listening():
                on_audio(tail)
        pause = silence(self.sample_rate, (delivery or NEUTRAL).pause_ms)
        if pause and listening():
            on_audio(pause)
        return {
            "blocks": blocks,
            "first_block_ms": int(max(0.0, (first or began) - began) * 1000),
            "synth_ms": int(max(0.0, time.monotonic() - began) * 1000),
            "audio_ms": int(samples * 1000 / (self.sample_rate or 1)),
            "streaming": "chunk" if blocks > 1 else "unit",
        }

    def _as_pcm(self, chunk, shaper: "Shaper") -> bytes:
        """One model chunk as PCM16, through the delivery DSP where it is active."""
        if shaper.active:
            return shaper.block(chunk)
        found = chunk
        if hasattr(found, "detach"):
            found = found.detach()
        if hasattr(found, "cpu"):
            found = found.cpu()
        if hasattr(found, "numpy"):
            found = found.numpy()
        if self._numpy is not None:
            found = self._numpy.asarray(found, dtype=self._numpy.float32).reshape(-1)
        return pcm16(found, (shaper.delivery or NEUTRAL).gain)

    def synthesize(self, text: str, voice_id: str, delivery: Delivery) -> bytes:
        """One complete utterance as PCM16. What an audition and /tts use."""
        blocks = []
        self.stream(text, voice_id, delivery, blocks.append, lambda: True)
        return b"".join(blocks)


# --------------------------------------------------------------------------- #
# One speaking turn, and the single inference lane
# --------------------------------------------------------------------------- #

MAX_PENDING_AUDIO = 24
"""How many PCM frames may be waiting to be written before the lane waits.

Backpressure on the *writer*, which is a different thing from the parent's
playback queue and exists for a different reason: a lane that outran the pipe
would build an unbounded list of audio in this process's memory. It never
applies to a draining unit, because a draining unit writes nothing.
"""

WRITE_TIMEOUT = 0.25
DRAIN_HEARTBEAT = 2.0
"""How often a draining turn says it is still draining.

Not a timeout and not a promise. The parent's wait is cleared by
``state="complete"`` and by nothing else; this is so that a long drain looks
like a long drain in the log rather than like a worker that stopped answering.
"""


class Turn:
    """One reply being spoken, and the only place ``interrupted`` is true.

    ``interrupted`` is set by :meth:`Worker.interrupt` and read in three places:
    the lane skips queued units for the turn, the streaming loop stops offering
    blocks onward, and :meth:`next_segment` stops waiting for text that is never
    coming. What it does *not* do is stop the generator -- see the module
    docstring.
    """

    def __init__(self, identifier: str, voice_id: str, delivery: Delivery = None):
        self.id = str(identifier)
        self.voice_id = str(voice_id or "")
        self.delivery = delivery or NEUTRAL
        self.segments: "queue.Queue" = queue.Queue()
        self.done = False
        self.interrupted = False
        self.speaking = False
        self.units = 0
        self.reported = False
        """Whether ``tts_interrupted state=complete`` has already been sent.

        One report per interruption, because two paths can reach the same
        truth: the command loop answers immediately when nothing was inside the
        model, and the lane answers when the unit it *was* inside returns. The
        parent tolerates a second frame -- its drain record is already gone --
        but a worker that sent one would be a worker whose log read as though
        the lane had been freed twice.
        """
        self.dropped = 0
        """How many committed units were never sent to the model because the
        turn was interrupted first. Reported as a number, because "unit 3 was
        discarded" is the difference between a bounded Stop and a lockout
        (section 21.3)."""

    def finish(self) -> None:
        self.done = True
        self.segments.put(None)

    def stop(self) -> None:
        self.interrupted = True
        self.done = True
        # Wake anything waiting for text, and throw away what is queued: only
        # the unit already inside the model may drain (section 21.3).
        while True:
            try:
                found = self.segments.get_nowait()
            except queue.Empty:
                break
            if found is not None:
                self.dropped += 1
        self.segments.put(None)

    def next_segment(self):
        """The next committed unit, or ``None`` when there will be no more."""
        while True:
            if self.interrupted:
                return None
            try:
                found = self.segments.get(timeout=0.05)
            except queue.Empty:
                if self.done and self.segments.empty():
                    return None
                continue
            if found is None:
                if self.done and self.segments.empty():
                    return None
                continue
            return found


class Worker:
    """The lane, the writer, and the turns. One model, one inference at a time.

    One lane thread rather than a pool, and that is not a simplification: the
    model is documented as not thread-safe, so a second concurrent generation
    would be incorrect rather than merely slow (I-PKT-8).
    """

    def __init__(self, stdout):
        self.stdout = stdout
        self.engine = None
        self._turns = {}
        self._turn_lock = threading.Lock()
        self._jobs: "queue.Queue" = queue.Queue()
        self._outbox: "queue.Queue" = queue.Queue(maxsize=MAX_PENDING_AUDIO)
        self._write_lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        self._lane = None
        self._writer = None

    # -- threads ----------------------------------------------------------- #

    def start_threads(self) -> None:
        self._lane = threading.Thread(target=self._lane_loop, name="pocket-lane",
                                      daemon=True)
        self._lane.start()
        self._writer = threading.Thread(target=self._write_loop, name="pocket-writer",
                                        daemon=True)
        self._writer.start()

    def send(self, header: dict, payload: bytes = b"", audio: bool = False) -> None:
        """Queue one frame for the writer, or write it directly.

        Audio goes through the bounded queue so a fast lane cannot build an
        unbounded backlog in this process. Everything else -- replies,
        acknowledgements, the interrupted report -- is written directly, because
        a control frame stuck behind a queue of audio is a control frame that
        arrives after the thing it was describing.
        """
        if not audio:
            with self._write_lock:
                write_frame(self.stdout, header, payload)
            return
        while True:
            try:
                self._outbox.put((header, payload), timeout=WRITE_TIMEOUT)
                return
            except queue.Full:
                if self._stopping:
                    return
                writer = self._writer
                if writer is not None and not writer.is_alive():
                    # The pipe has gone and nothing is emptying this queue. The
                    # lane must not spin here: it is the thread that has to
                    # reach the end of an abandoned unit, and a lane blocked on
                    # a dead writer is a unit that never finishes (I-PKT-12).
                    self._stopping = True
                    return

    def _write_loop(self) -> None:
        while True:
            try:
                found = self._outbox.get(timeout=0.1)
            except queue.Empty:
                if self._stopping:
                    return
                continue
            if found is None:
                return
            header, payload = found
            try:
                with self._write_lock:
                    write_frame(self.stdout, header, payload)
            except Exception:
                # The pipe has gone. Nothing here can fix that, and the read
                # loop is about to end for the same reason.
                return

    def wake_writers(self) -> None:
        try:
            self._outbox.put_nowait(None)
        except Exception:
            pass

    # -- turns ------------------------------------------------------------- #

    def open_turn(self, identifier: str, voice_id: str, delivery: Delivery = None) -> Turn:
        turn = Turn(identifier, voice_id, delivery)
        with self._turn_lock:
            self._turns[turn.id] = turn
        self._jobs.put(("turn", turn.id))
        return turn

    def turn(self, identifier: str):
        with self._turn_lock:
            return self._turns.get(str(identifier or ""))

    def close_turn(self, identifier: str) -> None:
        with self._turn_lock:
            self._turns.pop(str(identifier or ""), None)

    def stop_all_turns(self) -> None:
        with self._turn_lock:
            found = list(self._turns.values())
        for turn in found:
            turn.stop()

    def interrupt(self, identifier: str) -> None:
        """Stop offering this turn's audio, and drop what it has not started.

        Returns at once. The unit already inside the model is not stopped --
        this build cannot stop one safely and does not claim to -- so what
        happens next is that the lane keeps consuming it, throws the blocks
        away, and reports ``state="complete"`` when the call returns
        (I-PKT-11, I-PKT-12).

        When a merged upstream cancellation is adopted, this is where its Event
        is set, and the only thing that changes for anybody else is that the
        report arrives much sooner (section 21.7).
        """
        turn = self.turn(identifier)
        if turn is None:
            # A turn that has already ended. The parent's Stop is idempotent and
            # the honest answer is that the lane is free.
            self.send({"op": "tts_interrupted", "turn": str(identifier or ""),
                       "state": "complete", "interrupt_mode": INTERRUPT_MODE})
            return
        turn.stop()
        if not turn.speaking:
            # Nothing was inside the model, so there is nothing to drain. Said
            # immediately rather than after the lane notices, because a Play
            # control that waited for a drain that never happened would be a
            # control that stayed disabled for no reason.
            turn.reported = True
            self.send({"op": "tts_interrupted", "turn": turn.id, "state": "complete",
                       "interrupt_mode": INTERRUPT_MODE})
            self.close_turn(turn.id)
            return
        self.send({"op": "tts_interrupted", "turn": turn.id, "state": "draining",
                   "interrupt_mode": INTERRUPT_MODE})

    # -- the one lane ------------------------------------------------------ #

    def _lane_loop(self) -> None:
        """One job at a time, for the life of the process.

        A turn and an audition are the same kind of work here and queue behind
        one another for the same reason: there is one model, it is documented as
        not thread-safe, and a Test pressed while a reply is being spoken has to
        wait rather than corrupt both (I-PKT-8).
        """
        while True:
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                if self._stopping:
                    return
                continue
            if job is None:
                return
            kind, found = job
            if kind == "call":
                # Already wrapped by :func:`_queue`, which turns a failure into
                # the reply somebody is waiting for. This is the belt.
                try:
                    found()
                except Exception as exc:  # noqa: BLE001 - never fatal to the lane
                    _note(f"request failed: {exc.__class__.__name__}")
                continue
            turn = self.turn(found)
            if turn is None:
                continue
            try:
                self.speak(turn)
            except Exception as exc:  # noqa: BLE001 - reported, never fatal to the lane
                _note(f"turn failed: {exc.__class__.__name__}")
                try:
                    self.send({"op": "tts_error", "turn": turn.id, "error": _safe(exc)})
                except Exception:
                    pass
            finally:
                self.close_turn(turn.id)

    def speak(self, turn: Turn) -> None:
        """Every committed unit of one turn, in order, until it ends or is stopped.

        The interrupted path is the interesting one and it is deliberately not a
        ``break``. When ``turn.interrupted`` goes true the loop stops *starting*
        units -- :meth:`Turn.next_segment` has already thrown the queued ones
        away -- but the unit that was inside :meth:`Engine.stream` when it
        happened runs to its ordinary end with ``listening`` false, so its
        blocks are consumed here and never written. When that returns, the lane
        is free and the parent is told so.
        """
        self.send({"op": "tts_ready", "turn": turn.id, "sample_rate": self.engine.sample_rate,
                   "streaming": "chunk", "interrupt_mode": INTERRUPT_MODE})
        interrupted_chars = 0
        interrupted_ms = 0
        while True:
            text = turn.next_segment()
            if text is None:
                break
            turn.units += 1
            turn.speaking = True
            heartbeat = [time.monotonic()]

            def listening(turn=turn, heartbeat=heartbeat):
                if not turn.interrupted:
                    return True
                now = time.monotonic()
                if now - heartbeat[0] >= DRAIN_HEARTBEAT:
                    heartbeat[0] = now
                    self.send({"op": "tts_interrupted", "turn": turn.id,
                               "state": "draining", "interrupt_mode": INTERRUPT_MODE})
                return False

            def offer(block, turn=turn):
                self.send({"op": "tts_audio", "turn": turn.id,
                           "sample_rate": self.engine.sample_rate}, block, audio=True)

            try:
                found = self.engine.stream(text, turn.voice_id, turn.delivery, offer,
                                           listening)
            finally:
                turn.speaking = False
            if turn.interrupted:
                interrupted_chars = len(text)
                interrupted_ms = int(found.get("audio_ms") or 0)
                break
            self.send({"op": "tts_segment", "turn": turn.id, **found})

        if turn.interrupted:
            if turn.reported:
                return
            turn.reported = True
            # The lane is free now, and this is the only frame that says so.
            # Never a timer on the parent's side, so this is the thing the
            # waiting state clears on (I-PKT-13).
            self.send({"op": "tts_interrupted", "turn": turn.id, "state": "complete",
                       "interrupt_mode": INTERRUPT_MODE,
                       "chars": interrupted_chars, "audio_ms": interrupted_ms,
                       "dropped_units": turn.dropped})
            _note(f"interrupted turn drained: unit of {interrupted_chars} chars, "
                  f"{interrupted_ms} ms of audio discarded, {turn.dropped} unit(s) dropped")
            return
        self.send({"op": "tts_done", "turn": turn.id, "units": turn.units})


# --------------------------------------------------------------------------- #
# The command loop
# --------------------------------------------------------------------------- #

DRAIN_GRACE = 2.0
"""How long an already-accepted request may take to answer after stdin closes.

A request this process took off the wire is one somebody is waiting for;
dropping it because stdin happened to end first turns a clone into a timeout on
the other side. Bounded, because the reason stdin ended may be that the parent
died.
"""


def serve(stdin, stdout, engine_factory=None) -> int:
    """Read frames until end of input. Returns the process exit status.

    This is the command loop and it does as little as it possibly can. Its one
    job is to never be busy: everything that could take longer than parsing a
    header is handed to the inference lane, so a ``tts_interrupt`` sent while
    the model is generating is *read* immediately rather than after the
    paragraph.

    That property matters more here than on the other two engines. Pocket's Stop
    already costs the length of the unit in flight; a Stop that also had to wait
    to be read would cost that twice, and the browser would be silent for the
    first half of it wondering why the panel had not changed.
    """
    factory = engine_factory or (lambda config: Engine(config))
    worker = Worker(stdout)
    parent_death = "pipe"

    def reply(request_id, header: dict, payload: bytes = b"") -> None:
        found = dict(header)
        found["id"] = request_id
        worker.send(found, payload)

    try:
        while True:
            try:
                frame = read_frame(stdin)
            except Exception as exc:
                _note(f"malformed frame: {_safe(exc)}")
                return 2
            if frame is None:
                _note("input closed; stopping")
                return 0
            header, payload = frame
            operation = str(header.get("op") or "")
            request_id = header.get("id")

            if operation == "shutdown":
                reply(request_id, {"ok": True})
                _note("stopping on request")
                return 0

            if operation == "init":
                try:
                    parent_death = _containment(int(header.get("parent_pid") or 0))
                    engine = factory(header.get("config") or {})
                    engine.load()
                    worker.engine = engine
                    worker.start_threads()
                    reply(request_id, {
                        "ok": True,
                        "op": "ready",
                        # Every field section 18 asks for, and nothing it
                        # forbids: no model path, no interpreter, no pid, no
                        # token or cache location, no voice name.
                        "protocol": PROTOCOL_VERSION,
                        "engine": "pocket",
                        "backend": "pocket-tts-native",
                        "pocket_version": _package_version("pocket_tts"),
                        "upstream_build_id": engine.upstream_build_id,
                        "torch_version": _package_version("torch"),
                        "sample_rate": engine.sample_rate,
                        "provider": "cpu",
                        "device": engine.device(),
                        "containment": parent_death,
                        "model_id": engine.model_id,
                        "model_fingerprint": engine.fingerprint,
                        "quantization": engine.precision,
                        "sampler_steps": engine.sampler_steps,
                        "streaming": True,
                        "interrupt_mode": INTERRUPT_MODE,
                        "thread_policy": thread_policy(),
                        "voice_state_schema": engine.state_schema,
                        "voices": len(engine.voices),
                        "defaults": engine.defaults,
                    })
                    _note(f"ready — cpu, {engine.precision}, {engine.sampler_steps} step(s), "
                          f"{len(engine.voices)} voices, containment {parent_death}, "
                          f"interrupt {INTERRUPT_MODE}")
                except SystemExit:
                    raise
                except Exception as exc:
                    worker.engine = None
                    reply(request_id, {"ok": False, "error": _safe(exc)})
                    _note(f"could not load PocketTTS: {_safe(exc)}")
                continue

            if worker.engine is None:
                if operation.startswith("tts_"):
                    worker.send({"op": "tts_error", "turn": str(header.get("turn") or ""),
                                 "error": "runtime is not initialised"})
                else:
                    reply(request_id, {"ok": False, "error": "runtime is not initialised"})
                continue

            if operation == "ping":
                # Answered on this thread on purpose. "Is the worker alive while
                # it is speaking" is a question whose answer is worthless if it
                # has to queue behind the speaking.
                reply(request_id, {"ok": True, "speaking": bool(worker._turns)})

            elif operation == "catalog":
                count = worker.engine.refresh_catalog(header.get("voices") or {},
                                                      header.get("forget") or ())
                reply(request_id, {"ok": True, "voices": count})

            elif operation == "prepare":
                _queue(worker, reply, request_id,
                       lambda rid=request_id, head=header, body=payload:
                       _do_prepare(worker, reply, rid, head, body))

            elif operation == "warm":
                _queue(worker, reply, request_id,
                       lambda rid=request_id, head=header:
                       _do_warm(worker, reply, rid, head))

            elif operation == "tts":
                _queue(worker, reply, request_id,
                       lambda rid=request_id, head=header, body=payload:
                       _do_tts(worker, reply, rid, head, body))

            elif operation == "tts_begin":
                worker.open_turn(str(header.get("turn") or ""),
                                 str(header.get("voice_id") or ""),
                                 Delivery.from_header(header))

            elif operation == "tts_text":
                found = worker.turn(str(header.get("turn") or ""))
                if found is not None and not found.interrupted:
                    found.segments.put(payload.decode("utf-8", "replace"))

            elif operation == "tts_end":
                found = worker.turn(str(header.get("turn") or ""))
                if found is not None:
                    found.finish()

            elif operation == "tts_interrupt":
                # Read and acted on immediately, on this thread, while the lane
                # is still inside the model. That is the whole reason this loop
                # does no work of its own.
                worker.interrupt(str(header.get("turn") or ""))

            else:
                reply(request_id, {"ok": False, "error": "unknown operation"})
    finally:
        worker._stopping = True
        worker.stop_all_turns()
        worker.wake_writers()
        writer = worker._writer
        if writer is not None and writer.is_alive():
            writer.join(timeout=DRAIN_GRACE)


def _queue(worker, reply, request_id, call) -> None:
    """Put one request on the lane, and guarantee it is answered.

    The guarantee is the point. A request that reached the lane and then raised
    would otherwise be a request nobody ever answered, and the parent's only
    recourse would be a timeout minutes later -- so the failure is turned into
    a reply here, in the class-name-only form :func:`_safe` produces.
    """

    def run():
        try:
            call()
        except Exception as exc:  # noqa: BLE001 - answered rather than raised
            _note(f"request failed: {_safe(exc)}")
            try:
                reply(request_id, {"ok": False, "error": _safe(exc)})
            except Exception:
                pass

    worker._jobs.put(("call", run))


def _package_version(name: str) -> str:
    """One installed package's version, or an empty string. Never raises."""
    try:
        import importlib.metadata as metadata

        return str(metadata.version(name))
    except Exception:
        try:
            module = __import__(name)
            return str(getattr(module, "__version__", "") or "")
        except Exception:
            return ""


def _do_prepare(worker, reply, request_id, header: dict, payload: bytes) -> None:
    """Prepare one reference into a voice state, and prove it before answering.

    The order is section 26.2's and every step of it is a different failure:
    extract, export, read the exported file *back off the disk*, and synthesise
    the audition from what came back. A preparation that answered after the
    export would be answering about tensors that were still in memory.
    """
    root = str(header.get("root") or "")
    if not root or not os.path.isdir(root):
        reply(request_id, {"ok": False, "error": "that preparation has nowhere to write"})
        return
    began = time.monotonic()
    made = worker.engine.prepare(payload, float(header.get("seconds") or 0.0))
    path = os.path.join(root, "state.safetensors")
    size = worker.engine.export(made["state"], path)

    # Read back, and used. The catalogue entry is temporary and is replaced by
    # the parent's own the next time it refreshes; what matters is that the
    # audition below comes from the file rather than from ``made["state"]``.
    voice_id = str(header.get("voice_id") or "")
    worker.engine.refresh_catalog({**worker.engine.voices,
                                   voice_id: {"kind": "clone", "state": path}})
    audition = str(header.get("audition") or "")
    audio = b""
    try:
        if audition:
            audio = worker.engine.synthesize(audition, voice_id,
                                             Delivery.from_header(header))
    finally:
        # The entry was temporary and the file it names is about to be moved or
        # deleted by the parent. Forgotten here rather than left to go stale,
        # because a catalogue that named a path which no longer exists would
        # answer "rebuild it" to a voice that was fine.
        worker.engine.forget(voice_id)
    reply(request_id, {
        "ok": True,
        "sample_rate": worker.engine.sample_rate,
        "state_bytes": size,
        "prepare_ms": int((time.monotonic() - began) * 1000),
        "audition_ms": int(len(audio) / 2 * 1000 / (worker.engine.sample_rate or 1)),
    }, encode_wav(audio, worker.engine.sample_rate) if audio else b"")


def _do_warm(worker, reply, request_id, header: dict) -> None:
    voice_id = str(header.get("voice_id") or "")
    worker.engine.state_for(voice_id)
    reply(request_id, {"ok": True, "state": "warm"})


def _do_tts(worker, reply, request_id, header: dict, payload: bytes) -> None:
    """One complete utterance as a WAV. The audition and /tts path."""
    text = payload.decode("utf-8", "replace")
    audio = worker.engine.synthesize(text, str(header.get("voice_id") or ""),
                                     Delivery.from_header(header))
    reply(request_id, {"ok": True, "sample_rate": worker.engine.sample_rate,
                       "samples": len(audio) // 2},
          encode_wav(audio, worker.engine.sample_rate))


# --------------------------------------------------------------------------- #
# Running this file directly
# --------------------------------------------------------------------------- #


def selftest() -> int:
    """Prove the staged runtime imports and is on the CPU. One JSON line out.

    Run by the installer against a *staged* interpreter before anything is
    promoted, so the last line of stdout is the whole contract: the installer
    reads it with :func:`json.loads` and refuses the install if ``ok`` is false
    or the device is not the CPU.
    """
    report = {"ok": False, "device": "", "error": ""}
    try:
        import torch

        report["torch_version"] = str(getattr(torch, "__version__", ""))
        report["numpy_version"] = _package_version("numpy")
        import pocket_tts  # noqa: F401 - imported to prove the closure is complete

        report["pocket_version"] = _package_version("pocket_tts")
        report["upstream_build_id"] = str(getattr(pocket_tts, "__build__", "") or "")
        report["thread_policy"] = thread_policy()
        report["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        report["ok"] = report["device"] == "cpu"
        if not report["ok"]:
            report["error"] = "a graphics device is visible to this runtime"
    except Exception as exc:  # noqa: BLE001 - the report is the answer
        report["error"] = f"{exc.__class__.__name__}: {exc}"
    sys.stdout.write(json.dumps(report) + "\n")
    sys.stdout.flush()
    return 0 if report["ok"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(MARKER, action="store_true", dest="marker")
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--session", default="")
    parser.add_argument("--selftest", action="store_true")
    found, _rest = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    if found.selftest:
        return selftest()
    return serve(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
