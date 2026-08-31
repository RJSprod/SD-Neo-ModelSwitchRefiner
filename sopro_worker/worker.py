"""The Sopro V2 sidecar: one contained CPU process, and everything Torch touches.

Run by path, under the isolated Sopro interpreter, by :mod:`mc_voice_sopro_runtime`.
It never imports a Forge module, a Model Chain module or sherpa-onnx, and it is
the only process in this repository that imports Torch (I-6).

What lives here and why it is not in the parent
-----------------------------------------------
Everything that needs a tensor. That is a shorter list than it looks:

    preparing a reference recording into canonical conditioning
    serializing that conditioning, and reading it back
    reconstructing a Sopro ``Reference`` and warming its ``PromptState``
    streaming a committed text unit into PCM
    the delivery DSP -- speed, pitch, volume, pacing
    the Voice Lab's experimental conditioning

The parent owns product state, voice identity, turn routing and every decision
about *which* voice; this process owns inference and nothing else. It is handed
paths it did not choose and a voice catalogue it did not build, and it validates
both anyway, because it is the process that would crash.

Speed is DSP here, and says so
------------------------------
Sopro V2 exposes no supported model-native speaking-rate parameter -- there is
no ``speed=`` on ``stream()`` and nothing in ``GenerationConfig`` that means
one. Section 32 is therefore explicit: Sopro Speed is a pitch-preserving
time-domain transformation around the model output, and a naive resample that
transposes the voice is not parity and does not ship.

:class:`Stretch` is that transformation -- streaming SOLA, state carried across
Sopro's chunk boundaries so there is no click where one chunk ends -- and
:class:`Resampler` is the independent Pitch control composed on top of it. The
composition is stated once, in :class:`Delivery`, and it is the whole of the
"speed at neutral pitch does not transpose, pitch at any speed does not change
the requested duration" requirement:

    time-scale asked of SOLA   = speed / pitch_ratio
    resample ratio afterwards  = pitch_ratio

At neutral both are 1.0 and both objects are skipped entirely, so an
installation that never moves a slider pays nothing for them.

Cancellation is cooperative and is measured as three things
-----------------------------------------------------------
Section 49. ``stream()`` is a generator with no callback polled inside the
model's own operations, so this process cannot promise to stop mid-matmul. What
it does promise is to stop at the next yielded chunk, to wake anything waiting
for queue room, and never to begin another unit. The three quantities the parent
measures -- browser silence, no new worker audio, and compute quiescence -- are
three different numbers on purpose, because only the first two are this
process's to make small.

Nothing durable is a warmed session
-----------------------------------
``PromptState``, ``ChunkedSolveState``, per-layer K/V buffers and
``StreamSession`` are worker cache with a bound on them (I-11). What is written
to disk is ``cond_vec``, ``semantic_tokens``, ``mel`` and a scalar, as
safetensors beside versioned JSON -- non-executable, validated by name, shape
and dtype before a byte of it is allocated (section 15, section 57).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
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
"""The Sopro protocol's own version, counted from one.

Not shared with :data:`voice_worker.worker.PROTOCOL_VERSION`. The two workers
speak the same *framing* and a deliberately overlapping set of turn operations,
and they are still two protocols: Sopro carries ``voice_id`` where Kokoro
carries ``sid``, and it has four operations Kokoro has no meaning for. One
number covering both would be a number that has to change when either changes.
"""

MARKER = "--model-chain-sopro-worker"
"""On the command line so this process is recognisable in a task manager and by
the stray sweep. Never a voice name, never a path, never spoken text: a command
line is world readable on most systems."""

MAX_HEADER = 1 << 20
MAX_PAYLOAD = 40 << 20
"""A 20-second 24 kHz mono PCM16 reference is under a megabyte; the ceiling is
generous because the parent's route in front of it is the real limit, and it is
still a number rather than "whatever arrives"."""

SAMPLE_RATE = 24000
"""What Sopro V2 produces. Asserted against the loaded config at start-up rather
than assumed -- a model revision that changed it would otherwise be a worker
emitting frames the browser resamples into a chipmunk."""

_LENGTH = struct.Struct(">I")


# --------------------------------------------------------------------------- #
# Framing -- byte-identical to voice_worker, by agreement rather than by import
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
    reason Kokoro's worker validates: this is the process that would crash, and
    a worker that dies on a malformed upload is a worker the next clone has to
    restart.
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


DECLICK_MS = 8
"""How long a committed unit takes to reach full level, and to leave it again.

A unit does not end the way a recording ends. Sopro decodes one committed unit
and stops; where the waveform happens to be at the last chunk is where the
samples stop, and the next unit is decoded from a session that has no memory of
this one. Played back the way this feature plays speech -- sample-exact, one
unit after another, no gap -- the join between the two is a step, and a step in
a waveform is a click. It is small, it lands at the end of a sentence, and on a
long reply it happens once per sentence.

Eight milliseconds is chosen to be longer than the step and shorter than a
phoneme. A ramp that long puts the click's energy below about sixty hertz,
where it stops being a tick, and what it takes the edge off is a unit's first
and last eight milliseconds -- which is the quiet either side of a sentence
rather than the sentence. Materially shorter stops removing the click;
materially longer starts eating the consonant.
"""


def _int16(pcm: bytes):
    """PCM16 bytes as a mutable array of samples, in this machine's order."""
    import array

    numbers = array.array("h")
    numbers.frombytes(pcm)
    if sys.byteorder == "big":
        numbers.byteswap()
    return numbers


def _wire(numbers) -> bytes:
    """Samples back as little-endian PCM16. Mutates ``numbers`` on a big-endian host."""
    if sys.byteorder == "big":
        numbers.byteswap()
    return numbers.tobytes()


def _ramp(count: int):
    """A raised cosine from silence to unity, ``count`` samples long.

    Raised rather than linear because a linear ramp has a corner at each end,
    and a corner is a discontinuity in the first derivative -- quieter than a
    step, and still audible on a quiet tail.
    """
    import math

    if count <= 0:
        return []
    return [0.5 - 0.5 * math.cos(math.pi * (index + 0.5) / count)
            for index in range(count)]


class Seam:
    """One committed unit's two edges, ramped so that its joins are not clicks.

    A unit is one decode. Its last sample is wherever the waveform was when the
    unit ended and its first is wherever a fresh session starts, and this
    feature plays units back with no gap between them -- so the join is a step,
    and a step is a click. See :data:`DECLICK_MS`.

    Streaming makes this less trivial than a fade. The *end* of the unit has to
    be ramped down, and nothing knows which block is the last one until the
    synthesis is over -- so a unit's final :data:`DECLICK_MS` are withheld here
    and released by :meth:`flush` when the unit ends. Eight milliseconds of
    added latency once per sentence is not something anybody can hear. A click
    is.

    It works on PCM16 bytes rather than on float samples because every path
    into it has already become bytes by the time it arrives, and a ramp is
    exact either way.

    One per unit, and the *unit* is the right scope: a pause may follow it, a
    cancellation may throw it away, and the next one is synthesised with no
    memory of this one. Ramping per block instead would fade several times a
    second in the middle of a word.
    """

    def __init__(self, rate: int, milliseconds: int = DECLICK_MS):
        self.span = max(0, int(int(rate or 0) * max(0, int(milliseconds)) / 1000))
        self._window = _ramp(self.span)
        self._held = bytearray()
        self._risen = 0

    def block(self, pcm: bytes) -> bytes:
        """Whatever of ``pcm`` is certainly not this unit's final milliseconds."""
        if not self.span:
            return pcm or b""
        if pcm:
            self._held.extend(pcm)
        keep = self.span * 2
        if len(self._held) <= keep:
            return b""
        ready = bytes(self._held[:len(self._held) - keep])
        del self._held[:len(ready)]
        return self._rise(ready)

    def flush(self) -> bytes:
        """The unit's final milliseconds, ramped down into the silence after it."""
        if not self.span:
            return b""
        held, self._held = bytes(self._held), bytearray()
        tail = self._rise(held)
        if not tail:
            return b""
        numbers = _int16(tail)
        count = len(numbers)
        window = self._window if count >= self.span else _ramp(count)
        last = count - 1
        for index in range(count):
            numbers[last - index] = int(numbers[last - index] * window[index])
        return _wire(numbers)

    def _rise(self, pcm: bytes) -> bytes:
        """``pcm`` with whatever of the opening ramp has not been spent yet."""
        if not pcm or self._risen >= self.span:
            return pcm
        numbers = _int16(pcm)
        count = min(len(numbers), self.span - self._risen)
        for index in range(count):
            numbers[index] = int(numbers[index] * self._window[self._risen + index])
        self._risen += count
        return _wire(numbers)

KEEP_LEAD_MS = 60
KEEP_TAIL_MS = 120
"""How much of a unit's own quiet is kept at each end, in milliseconds.

A unit is not the length of what it says. Sopro's decoder settles in at the
start of a unit and runs on past the last phoneme at the end, so every unit
carries quiet at both ends. Voice Chat plays units back with nothing between
them, so those two add up into the gap between one sentence and the next: dead
air nobody asked for, in the middle of a reply.

What is *not* done is removing the gap altogether. Sentences in speech are
separated by a pause, and butting one against the next reads as hurried rather
than as continuous -- so the quiet is cut back to a fixed, small amount rather
than cut out, and "Pause between sentences" adds to that for anybody who wants a
more measured delivery. The point is that the gap becomes the one somebody
chose, the same length every time, instead of whatever padding the model
happened to generate.
"""

QUIET_FLOOR = 130
QUIET_MULTIPLE = 3
QUIET_SHARE = 8
"""What counts as quiet, measured against the unit's own noise floor.

The first version of this asked whether a window was below a *share of the
loudest sample*, and on a real machine it trimmed exactly nothing: every unit
came back ``trimmed_ms=0``. The reason is the voice. A cloned voice reproduces
its reference recording's room tone, so what a listener hears as silence between
two sentences is not silence at all -- it is that room tone, and a rule anchored
to the peak puts the line far below it.

So the line is anchored to the floor instead: three times the quietest ten
milliseconds seen so far in this unit, which is where the room tone lives.
Anchored at the bottom it follows a noisy clone up and a clean model down, and
it never has to be guessed at.

Two bounds keep it honest. It is never below :data:`QUIET_FLOOR` -- about -48
dBFS, under any model's own noise -- because a unit of digital silence would
otherwise put the line at zero. And it is never above an eighth of the loudest
sample so far, which is the guard for a unit that contains no silence at all: if
the quietest thing in it is a soft consonant, three times *that* would call a
whole syllable quiet, and this is what stops it.
"""

SPEECH_FLOOR = 1000
OPEN_MULTIPLE = 4
"""What counts as the unit having started. About -30 dBFS, or over the floor.

Two tests, and both have to pass. Absolute, because a unit has to start on
something that is unmistakably not a noise floor. And relative, because on a
noisy clone the room tone is *also* unmistakably above -30 dBFS -- that is
exactly the case that made the first version trim nothing at the front.

Being wrong here is bounded in both directions: too eager and the lead is kept
rather than trimmed, too shy and the unit's audio waits :data:`MAX_LEAD_HOLD_MS`
and then goes out untrimmed.
"""

MAX_HOLD_MS = 600
MAX_LEAD_HOLD_MS = 400
SCAN_MS = 10
"""How much quiet may be held back, and how finely it is judged.

Trailing quiet has to be held to be trimmed -- nothing knows it is *trailing*
until the unit ends. Held audio is audio the listener does not have yet, so the
hold is bounded: past 600 ms it spills through and only the last 600 ms is ever
in hand.

Before the first word the bound is tighter, and it is a deadline rather than a
spill. A unit whose opening cannot be told from its noise floor gives up waiting
after 400 ms, sends what it has untrimmed, and carries on -- so the cost of not
being able to tell is 400 ms once, at the front of that unit, rather than a
trailing trim that never happens.

Ten milliseconds is the window the level is judged over, because a single sample
says nothing and a whole model chunk is eighty.
"""


class Trim:
    """The quiet a model puts either side of a unit, cut back to a set amount.

    A unit's leading quiet is trimmed to :data:`KEEP_LEAD_MS` and its trailing
    quiet to :data:`KEEP_TAIL_MS`. Quiet *inside* a unit is not touched: a pause
    the model put between two clauses is prosody, and only the two ends are
    padding.

    Trailing quiet cannot be recognised as trailing until the unit ends, so it
    is held here and released by :meth:`flush` -- bounded, because held audio is
    audio the listener does not have yet.

    Runs before :class:`Seam` rather than after it. The seam ramps a unit's
    first and last milliseconds to silence, and trimming after that would cut
    the ramp off and put back the click it exists to remove.

    It reports what it saw as well as what it did -- :attr:`floor_db` and
    :attr:`quiet_ms` alongside :attr:`dropped_ms` -- because the first version
    of this class trimmed nothing on a real machine and the log could not say
    why. A unit that reports a floor of -20 dBFS and half a second of quiet it
    did not cut is a different problem from one that reports no quiet at all.
    """

    def __init__(self, rate: int, lead_ms: int = KEEP_LEAD_MS,
                 tail_ms: int = KEEP_TAIL_MS, floor: int = 0):
        self.rate = max(0, int(rate or 0))
        self.scan = max(1, int(self.rate * SCAN_MS / 1000))
        self.lead = int(self.rate * max(0, int(lead_ms)) / 1000)
        self.tail = int(self.rate * max(0, int(tail_ms)) / 1000)
        self.hold = max(self.tail, int(self.rate * MAX_HOLD_MS / 1000))
        self.lead_hold = max(self.lead, int(self.rate * MAX_LEAD_HOLD_MS / 1000))
        self.dropped = 0
        self.quiet_found = 0
        self.hint = max(0, int(floor or 0))
        """What the last unit in this voice measured its noise floor to be.

        Seeded rather than discovered, and used for one thing only: deciding
        that a unit has *started*. A noise floor is a property of the voice and
        not of the sentence, and the front of a unit has to be judged before
        that unit has had any chance to measure its own -- so the only number
        available there is the one the last unit left behind. Zero when there is
        no last unit, and then the question falls back to :data:`SPEECH_FLOOR`
        alone.
        """
        self.measured = 32767
        """The quietest ten milliseconds in this unit. What gets reported, and
        what seeds the next one."""
        self._pending = _int16(b"")
        self._quiet = _int16(b"")
        self._opened = False
        self._peak = 0

    @property
    def dropped_ms(self) -> int:
        """How much quiet this unit lost, in milliseconds. For the log."""
        return int(self.dropped * 1000 / (self.rate or 1))

    @property
    def quiet_ms(self) -> int:
        """How much quiet this unit had at its two ends, cut or not."""
        return int(self.quiet_found * 1000 / (self.rate or 1))

    @property
    def floor_db(self) -> int:
        """This unit's own noise floor, in whole dBFS. Never below -96."""
        import math

        if self.measured <= 0:
            return -96
        return max(-96, int(round(20.0 * math.log10(self.measured / 32767.0))))

    def block(self, pcm: bytes) -> bytes:
        """Whatever of ``pcm`` is speech, or quiet that is being kept."""
        if not self.rate:
            return pcm or b""
        if pcm:
            self._pending.extend(_int16(pcm))
        out = _int16(b"")
        while len(self._pending) >= self.scan:
            window = self._pending[:self.scan]
            del self._pending[:self.scan]
            self._judge(window, out)
        return _wire(out) if len(out) else b""

    def flush(self) -> bytes:
        """The end of the unit: the kept tail, and nothing behind it."""
        if not self.rate:
            return b""
        out = _int16(b"")
        if len(self._pending):
            window = self._pending[:]
            del self._pending[:]
            self._judge(window, out)
        self.quiet_found += len(self._quiet)
        if self._opened and len(self._quiet) > self.tail:
            # Only when the unit opened. In a unit where nothing was ever told
            # apart from its noise floor, nothing has been shown to be padding,
            # and cutting the end off it would be cutting off something quiet
            # that might have been a word.
            self.dropped += len(self._quiet) - self.tail
            del self._quiet[self.tail:]
        out.extend(self._quiet)
        del self._quiet[:]
        return _wire(out) if len(out) else b""

    def _judge(self, window, out) -> None:
        """One window: speech, quiet before the speech, or quiet after it."""
        peak = max(max(window), -min(window))
        if peak > self._peak:
            self._peak = peak
        if peak < self.measured:
            self.measured = peak
        if not self._opened:
            self._before(window, peak, out)
            return
        if peak > self._quiet_level():
            # Speech. Whatever quiet was being held is a gap *inside* the unit
            # and is passed on untouched -- only the two ends are this class's
            # business.
            out.extend(self._quiet)
            del self._quiet[:]
            out.extend(window)
            return
        self._quiet.extend(window)
        if len(self._quiet) > self.hold:
            spill = len(self._quiet) - self.hold
            out.extend(self._quiet[:spill])
            del self._quiet[:spill]

    def _before(self, window, peak: int, out) -> None:
        """A window from before the first word, and whether it is still one."""
        self._quiet.extend(window)
        if peak > SPEECH_FLOOR and (not self.hint or peak > self.hint * OPEN_MULTIPLE):
            self._opened = True
            keep = self.lead + len(window)
            if len(self._quiet) > keep:
                self.quiet_found += len(self._quiet) - len(window)
                self.dropped += len(self._quiet) - keep
                del self._quiet[:len(self._quiet) - keep]
            out.extend(self._quiet)
            del self._quiet[:]
            return
        if len(self._quiet) <= self.lead_hold:
            return
        if peak <= QUIET_FLOOR or (self.hint and peak <= self.hint * QUIET_MULTIPLE):
            # Still plainly the model's own quiet, however long it goes on for.
            # The oldest of it is dropped rather than held or sent: it is the
            # far end of a lead nobody will hear, and holding it was only ever a
            # way of finding out where the lead ended.
            #
            # "Plainly" is the whole of the condition. Either this is under any
            # model's noise floor in absolute terms, or it is at the floor this
            # voice measured last time. Anything else -- a unit whose every
            # window is quiet, with no previous unit to compare it against -- is
            # not dropped at all: it could as easily be somebody speaking softly
            # with the volume turned down, and throwing that away would be
            # throwing away a sentence.
            self.quiet_found += len(self._quiet) - self.lead_hold
            self.dropped += len(self._quiet) - self.lead_hold
            del self._quiet[:len(self._quiet) - self.lead_hold]
            return
        # The deadline. This unit's opening cannot be told from its own noise,
        # so it goes out as it is -- and it counts as opened, because the
        # alternative is losing the trailing trim as well over a question that
        # was only ever about the front.
        self._opened = True
        out.extend(self._quiet)
        del self._quiet[:]

    def _quiet_level(self) -> int:
        """What counts as quiet once the unit has started."""
        return max(QUIET_FLOOR, min(self.measured * QUIET_MULTIPLE,
                                    self._peak // QUIET_SHARE))


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
    evidence from the kernel, which is what section 20 means by placing the
    worker in containment "before it is trusted". A Windows process that is not
    in a job says ``pipe``, and the parent refuses the handshake.
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
    question could not be put at all.

    The third used to be folded into the second and reported as ``pipe``, and
    the parent refused to start on it. That cost a real user the whole feature:
    containment had been arranged and was being enforced, and the worker was
    turned away for failing to confirm it. The parent proves containment itself
    now, against its own job handle; this is corroboration and says which of the
    three it actually is.

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
# The fixed CPU policy
# --------------------------------------------------------------------------- #

INTRAOP_THREADS = 4
INTEROP_THREADS = 1
"""The released CPU policy for this closure. Fixed, reported, never adaptive.

Four intra-op threads is the same budget Kokoro's synthesis lane has, which is
what makes a same-machine comparison mean anything (section 20): a Kokoro
measured at four threads against a Sopro that silently took every logical core
is not a comparison, it is two different experiments.

I-12 is the part that matters more than the number. Nothing in this repository
rotates between two, four and six, runs an A/B or picks a value from measured
real-time factors. The policy is set once, before the model loads, reported in
the handshake and in every benchmarkable log line, and moving it is a deliberate
edit to these two constants supported by Gate S-1 and S-3 measurements.

Correctness wins over the desired number. :func:`selftest` runs the SDPA
repeatability check Gate S-1 asks for against this exact policy, and an install
whose Torch build does not pass it is refused rather than shipped -- see
:func:`_attention_is_stable`.
"""

OVERRIDE_INTRAOP = "MC_SOPRO_INTRAOP_THREADS"
OVERRIDE_INTEROP = "MC_SOPRO_INTEROP_THREADS"
"""The one way the policy above moves without an edit, and what it costs.

I-12 says the policy is *measured*, fixed, and never auto-tuned from runtime
measurements. Those are three requirements and only the third one forbids
anything here: a benchmark that sweeps thread counts is how the first one is
satisfied, and the Run validation button cannot sweep a constant. So the
count may be set from the environment -- by that tool, or by somebody who has
run it and wants to keep the answer.

What is *not* allowed is an installation quietly running a policy nobody
measured. So an override is loud in three places: a warning line when it is
applied, a ``policy`` field of ``"override"`` in the handshake, and the effective
counts in every line that already carried them. Nothing in this repository reads
these variables to decide anything; they are read once, here, before the model
loads, and never again.
"""


def _asked(name: str, fallback: int, note: bool = True) -> "tuple[int, bool]":
    """A thread count from the environment, bounded, with whether it was used.

    Bounded rather than trusted: a zero or a negative would be handed to
    ``torch.set_num_threads`` as-is, and 4096 is a machine spending all of its
    time in barriers. 64 is well past any CPU this closure will meet and is a
    limit rather than a suggestion.

    ``note`` is off when the parent asks, because the parent reads this to build
    the worker's environment and its complaints belong in the WebUI log rather
    than on a stderr stream nobody is draining yet.
    """
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return fallback, False
    try:
        found = int(raw)
    except ValueError:
        if note:
            _note(f"{name} is not a number, so the released policy was kept")
        return fallback, False
    if found < 1 or found > 64:
        if note:
            _note(f"{name}={found} is outside 1-64, so the released policy was kept")
        return fallback, False
    return found, found != fallback


def effective_policy(note: bool = True) -> "tuple[int, int, bool]":
    """The counts that will actually be applied, and whether they are the
    released ones.

    Shared with the parent because :func:`mc_voice_sopro.worker_environment`
    pins ``OMP_NUM_THREADS`` and friends *before* this process starts, and
    OpenMP has usually sized its pool before any of our code runs. A parent that
    capped OMP at the released four while the child asked Torch for eight would
    produce a benchmark of neither number.
    """
    intraop, intraop_set = _asked(OVERRIDE_INTRAOP, INTRAOP_THREADS, note=note)
    interop, interop_set = _asked(OVERRIDE_INTEROP, INTEROP_THREADS, note=note)
    return intraop, interop, (intraop_set or interop_set)


def _apply_cpu_policy() -> dict:
    """Pin the thread policy before the model is built, and report what took.

    Before, not after: ``torch.set_num_interop_threads`` raises once a parallel
    region has run, and a model load is a parallel region. Reported as what
    Torch says afterwards rather than as what was asked for, because those are
    two claims and only the second one is evidence.
    """
    import torch

    intraop, interop, overridden = effective_policy()
    if overridden:
        _note(f"running at {intraop} intra-op / {interop} inter-op threads rather "
              f"than the released {INTRAOP_THREADS} and {INTEROP_THREADS} — either "
              f"the CPU thread setting or a validation sweep asked for it. Timings "
              f"from this process are not measurements of the shipped policy.")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(name, str(intraop))
    try:
        torch.set_num_threads(intraop)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        _note(f"could not set the intra-op thread count: {exc.__class__.__name__}")
    try:
        torch.set_num_interop_threads(interop)
    except Exception:
        # Already fixed by something that ran a parallel region first. Not an
        # error and not silently ignored either: the effective value is read
        # back below and is what the handshake reports.
        pass
    return {
        "thread_policy": "override" if overridden else "released",
        "intraop_threads": int(torch.get_num_threads()),
        "interop_threads": int(torch.get_num_interop_threads()),
        "omp_num_threads": str(os.environ.get("OMP_NUM_THREADS") or ""),
        "mkl_num_threads": str(os.environ.get("MKL_NUM_THREADS") or ""),
    }


def _attention_is_stable() -> "tuple[bool, str]":
    """Gate S-1's correctness check, against this exact Torch build and policy.

    Sopro's decoder is scaled-dot-product attention, and current Windows CPU
    PyTorch has had backend- and thread-specific SDPA correctness problems. The
    honest response to "some builds are wrong" is neither to assume this one is
    fine nor to refuse every Windows build: it is to *ask this build*, at the
    thread policy it will actually run at, before anything is installed.

    Two questions, because they fail differently. Repeatability -- the same
    input twice at four threads -- catches a race in the parallel path.
    Agreement with a single-threaded reference catches a kernel that is
    consistently wrong rather than intermittently. A build that fails either is
    refused with the number it was out by, so the report says what happened
    rather than "self-test failed".
    """
    import torch

    torch.manual_seed(20260829)
    query = torch.randn(1, 8, 96, 64)
    key = torch.randn(1, 8, 96, 64)
    value = torch.randn(1, 8, 96, 64)
    with torch.no_grad():
        first = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        second = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        repeat = float((first - second).abs().max())
        if repeat > 0.0:
            return False, f"attention is not repeatable at {torch.get_num_threads()} threads "\
                          f"(differs by {repeat:.3e} between two identical calls)"
        asked = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            reference = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        finally:
            torch.set_num_threads(asked)
        drift = float((first - reference).abs().max())
    if drift > 2e-4:
        return False, (f"attention at {asked} threads disagrees with the single-threaded "
                       f"result by {drift:.3e}")
    return True, ""


# --------------------------------------------------------------------------- #
# Delivery
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
    *this process's arithmetic* on audio the model produced. Language,
    temperature, top-p and top-k are Sopro's own arguments and are passed
    through untouched. Section 37 asks for the help text to say which is which;
    keeping them in one object with two accessors is how the code says it.
    """

    __slots__ = ("speed", "pitch", "gain", "pause_ms", "language",
                 "temperature", "top_p", "top_k", "steps", "chunk_frames", "seed")

    def __init__(self, speed=1.0, pitch=1.0, gain=1.0, pause_ms=0, language="",
                 temperature=None, top_p=None, top_k=None, steps=None,
                 chunk_frames=None, seed=None):
        self.speed = _bounded(speed, (0.5, 2.0), 1.0)
        self.pitch = _bounded(pitch, (0.25, 4.0), 1.0)
        self.gain = _bounded(gain, (0.05, 8.0), 1.0)
        self.pause_ms = int(_bounded(pause_ms, (0, 1200), 0))
        self.language = str(language or "").strip().lower() or ""
        self.temperature = None if temperature is None else _bounded(temperature, (0.05, 2.0), 0.8)
        self.top_p = None if top_p is None else _bounded(top_p, (0.05, 1.0), 0.9)
        self.top_k = None if top_k is None else int(_bounded(top_k, (1, 500), 25))
        self.steps = None if steps is None else int(_bounded(steps, (1, 32), 2))
        self.chunk_frames = None if chunk_frames is None else int(
            _bounded(chunk_frames, (4, 512), 64))
        self.seed = None if seed is None else int(seed)

    @classmethod
    def from_header(cls, header: dict) -> "Delivery":
        found = dict(header or {})
        return cls(speed=found.get("speed", 1.0), pitch=found.get("pitch", 1.0),
                   gain=found.get("gain", 1.0), pause_ms=found.get("pause_ms", 0),
                   language=found.get("language", ""),
                   temperature=found.get("temperature"), top_p=found.get("top_p"),
                   top_k=found.get("top_k"), steps=found.get("steps"),
                   chunk_frames=found.get("chunk_frames"), seed=found.get("seed"))

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
        """Sopro's own arguments for this turn, with the model's defaults kept.

        ``None`` means "whatever the pinned model configuration says", which is
        what an untouched Advanced control stores. Materialising today's default
        into the request instead would freeze it, and a model revision that
        changed its temperature would then not change anybody's voice.
        """
        found = {}
        if self.language:
            found["lang"] = self.language
        for name, value in (("temperature", self.temperature), ("top_p", self.top_p),
                            ("top_k", self.top_k), ("steps", self.steps),
                            ("chunk_frames", self.chunk_frames)):
            if value is not None:
                found[name] = value
        return found


NEUTRAL = Delivery()


class Stretch:
    """Streaming SOLA: change duration without changing pitch. Section 32.

    Synchronized overlap-add, which is the classical answer and the one whose
    failure modes are understood. The signal is cut into windows of ``2 * HOP``,
    each window is shifted to the position that best correlates with the audio
    already written, and the overlap is cross-faded with a periodic Hann. Moving
    the *analysis* hop while holding the *synthesis* hop fixed is what changes
    the duration; searching for the best offset is what keeps the waveform's
    periods lined up, which is what stops it sounding like a phaser.

    Streaming is the part that costs something. Sopro yields short chunks and a
    stretcher restarted at each one would put a discontinuity at every chunk
    boundary -- an audible tick several times a second. So the input buffer, the
    fractional read position and the pending overlap tail all survive between
    calls, and :meth:`flush` is what empties the last of it at the end of a
    committed unit.

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
# Canonical voice data
# --------------------------------------------------------------------------- #

PREPARATION_SCHEMA = 1
"""The layout of a saved Sopro voice. Bumped whenever a name, shape or dtype in
:data:`PRODUCTION_TENSORS` changes, and folded into the preparation fingerprint
so a stale asset is refused rather than guessed compatible (section 15)."""

PRODUCTION_TENSORS = ("cond_vec", "semantic_tokens", "mel")
"""Everything Conversation needs to reconstruct the canonical ``Reference``.

Exactly the three fields ``Reference`` carries besides its scalar, and
deliberately not ``prompt_states``: those are the warmed streaming caches I-11
forbids as durable voice data, and they are tens of megabytes of K/V buffers
whose shape depends on the solver steps and the chunk size the worker happened
to be configured with when the voice was made.
"""

LAB_TENSORS = ("id_emb", "style_emb", "style_ctrl")
"""The pre-projection speaker components, saved separately for the Voice Lab.

``prepare_reference`` throws these away -- the public ``Reference`` keeps only
the projected ``cond_vec`` -- so :meth:`Engine.prepare` reproduces the pinned
path and keeps them. Physically separate from the production asset because that
separation is what makes "Conversation reads canonical conditioning, never
experimental state" a property of the filesystem rather than a promise
(section 14).
"""

MAX_TENSOR_BYTES = 96 * 1024 * 1024
"""The ceiling on one tensor read out of a saved voice.

A malformed or hostile safetensors header must not be able to turn into an
arbitrary allocation request. Ninety-six megabytes is far above anything a
twenty-second reference produces and far below anything that matters.
"""

REFERENCE_CACHE = 4
PROMPT_CACHE = 3
LAB_CACHE = 2
"""Bounded, small, fixed. Section 29.

A warmed ``PromptState`` at the reviewed topology is tens of megabytes, so an
unbounded map of every cloned voice is how a Settings page ends up holding a
gigabyte. Three warm states is the default voice, the character being spoken and
one more -- which is the working set of an ordinary conversation.

Lab entries are counted separately and can never satisfy a production lookup,
even when their dimensions happen to match: the key space is disjoint because
the dictionaries are.
"""


def _note(text: str) -> None:
    """One line to stderr, which the parent drains into the log.

    Never anything anybody said. Section 56: no reference audio, no embeddings,
    no reply text, no clone audition text -- numbers, enums and opaque ids only.
    """
    try:
        sys.stderr.write(f"[sopro] {text}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _safe(exc: BaseException) -> str:
    """An exception as something the parent may show. Class only, mostly.

    A third-party library is entitled to put whatever it likes in a message,
    including the input it was given, and this feature's invariant is that the
    input never leaves this process. ``ValueError`` is the exception because it
    is what this file raises for its own refusals.
    """
    if isinstance(exc, ValueError):
        return str(exc)
    return exc.__class__.__name__


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


class Engine:
    """Sopro, loaded once and kept warm, with its caches and its adapter.

    Everything that touches a tensor is a method here, so the command loop below
    contains no inference call at all -- which is what lets a ``tts_cancel``
    sent while the solver is running be *read* immediately rather than after the
    paragraph.
    """

    def __init__(self, config: dict):
        self.config = dict(config or {})
        self.model_root = str(self.config.get("model_root") or "")
        self.model_id = str(self.config.get("model_id") or "")
        self.precision = str(self.config.get("precision") or "full").lower()
        self.steps = int(self.config.get("steps") or 0)
        self.chunk_frames = int(self.config.get("chunk_frames") or 0)
        self.fingerprint = str(self.config.get("fingerprint") or "")
        self.catalog = {}
        self.refresh_catalog(self.config.get("voices") or {})

        self.tts = None
        self.device = "cpu"
        self.sample_rate = 0
        self.hop_ratio = 0
        self.style_ctrl_dim = 0
        self.threads = {}
        self.torch_version = ""
        self.sopro_version = ""
        self.load_seconds = 0.0
        self._numpy = None
        self._references = collections.OrderedDict()
        self._prompts = collections.OrderedDict()
        self._lab = collections.OrderedDict()
        self._quiet_floor = {}
        self._cache_lock = threading.Lock()

    # -- loading ----------------------------------------------------------- #

    def load(self) -> None:
        """Build the model on the CPU, at the fixed policy, from local files only.

        ``from_pretrained`` is handed a *directory*. Sopro's ``resolve_artifacts``
        returns a local path unchanged and only reaches for ``huggingface_hub``
        when it is given a repository id -- which this closure does not contain,
        so the offline guarantee (section 54, Gate S-9) is structural rather than
        a flag somebody could unset.
        """
        import numpy
        import torch
        import sopro

        self._numpy = numpy
        self.threads = _apply_cpu_policy()
        self.torch_version = str(getattr(torch, "__version__", "") or "")
        self.sopro_version = str(getattr(sopro, "__version__", "") or "")

        root = self.model_root
        if not root or not os.path.isdir(root):
            raise ValueError("the Sopro model directory was not found")
        if self.precision not in ("full", "int8"):
            raise ValueError(f"unknown precision {self.precision!r}")

        started = time.monotonic()
        self.tts = sopro.SoproTTS.from_pretrained(
            root, device="cpu", dtype=torch.float32,
            quantization=("int8" if self.precision == "int8" else None))
        self.load_seconds = round(time.monotonic() - started, 3)

        self.device = str(getattr(self.tts, "device", "cpu"))
        if not self.device.startswith("cpu"):
            # I-9, at the only place it can actually be observed. A worker that
            # found a graphics device would be a worker competing with Forge for
            # the VRAM an image is being rendered in.
            raise ValueError(f"Sopro loaded on {self.device} rather than the CPU")
        self.sample_rate = int(getattr(self.tts, "sample_rate", 0) or 0)
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"this Sopro model reports {self.sample_rate} Hz, and this "
                             f"build is built for {SAMPLE_RATE} Hz")
        self.hop_ratio = int(getattr(self.tts, "hop_ratio", 0) or 0)
        self.style_ctrl_dim = int(
            getattr(self.tts.config.speaker_encoder, "style_ctrl_dim", 0) or 0)
        if self.chunk_frames and self.hop_ratio and self.chunk_frames % self.hop_ratio:
            raise ValueError(f"a streaming chunk of {self.chunk_frames} frames is not a "
                             f"multiple of this model's hop ratio ({self.hop_ratio})")

    @property
    def generation(self):
        return self.tts.generation

    def defaults(self) -> dict:
        """The pinned model's own generation defaults, for the Advanced panel.

        Read from the loaded configuration rather than repeated in the UI, so a
        model revision that changes its temperature changes what "default" means
        everywhere at once instead of in one place and not the other.
        """
        found = self.generation
        return {
            "temperature": float(found.temperature),
            "top_p": float(found.top_p),
            "top_k": int(found.top_k),
            "steps": int(found.steps),
            "chunk_frames": int(found.stream_chunk_frames),
            "ref_seconds": float(found.ref_seconds),
            "max_segment_chars": int(found.max_segment_chars),
        }

    # -- the voice catalogue ----------------------------------------------- #

    def refresh_catalog(self, voices) -> int:
        """Replace what this process believes exists. Validated, never trusted.

        The parent builds every path from server-generated identifiers under its
        own data root, and this still checks the shape of what arrives: a
        catalogue is the one structure that reaches this process from a file the
        user could have edited.
        """
        found = {}
        for identifier, entry in dict(voices or {}).items():
            if not isinstance(entry, dict):
                continue
            root = str(entry.get("root") or "")
            if not root or not os.path.isdir(root):
                continue
            found[str(identifier)] = {
                "root": root,
                "fingerprint": str(entry.get("fingerprint") or ""),
                "language": str(entry.get("language") or ""),
            }
        self.catalog = found
        return len(found)

    def forget(self, voice_id: str) -> None:
        """Drop every cached thing belonging to one voice. Section 29.

        Production and Lab together, because a deleted voice whose Lab state
        survived would be a voice the Lab could still audition after the user
        watched it disappear from the library.
        """
        wanted = str(voice_id or "")
        with self._cache_lock:
            for store in (self._references, self._prompts, self._lab):
                for key in [key for key in store if str(key).split("|", 1)[0] == wanted]:
                    store.pop(key, None)

    # -- preparation ------------------------------------------------------- #

    def prepare(self, wav_bytes: bytes, seconds: float = 0.0) -> dict:
        """A reference recording into canonical conditioning plus Lab components.

        This is ``prepare_reference``'s own body with one thing kept that the
        public function throws away. Sopro projects ``id_emb``, ``style_emb``
        and ``style_ctrl`` through ``build_condition`` and returns only the
        result, so a Voice Lab that wants to move ``style_ctrl`` has nothing to
        move. Section 26 authorises reproducing the reviewed internal path to
        retain them, on condition that the coupling is pinned and fingerprinted
        -- which it is: the fingerprint below covers the Sopro artifact digest,
        the model digests and this schema version, so a Sopro whose speaker
        encoder changed shape produces a different fingerprint and every asset
        made under the old one is refused rather than misread.

        Returns tensors in memory. Writing them is the caller's step, so that a
        preparation which succeeds and then fails its production audition leaves
        nothing on disk (section 27).
        """
        import torch
        import torchaudio
        from sopro import audio as audio_ops

        samples, rate = decode_wav(wav_bytes)
        if not samples:
            raise ValueError("that recording contains no audio")
        with torch.no_grad():
            wav = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
            wav = audio_ops.to_mono_resampled(wav, int(rate), self.sample_rate)
            wanted = float(seconds or self.generation.ref_seconds)
            wav = audio_ops.crop_on_pause(wav.to(self.device), wanted,
                                          self.sample_rate).unsqueeze(0)
            wav, level_db = audio_ops.normalize_reference(wav, self.sample_rate)
            wav16 = torchaudio.functional.resample(
                wav, self.sample_rate, int(self.tts.config.speaker_encoder.sample_rate))
            speaker = self.tts.speaker_encoder(wav16)
            cond_vec = self.tts.model.build_condition(
                *(speaker[name].to(dtype=self.tts.model.dtype) for name in LAB_TENSORS))
            tokens = self.tts.semantic_encoder.encode(wav)
            mel = (self.tts.vocoder.mel(wav) - self.tts.mel_mean) / self.tts.mel_std

        production = {"cond_vec": cond_vec.detach().cpu().contiguous(),
                      "semantic_tokens": tokens.detach().cpu().contiguous(),
                      "mel": mel.detach().cpu().contiguous()}
        lab = {name: speaker[name].detach().cpu().contiguous() for name in LAB_TENSORS}
        return {
            "production": production,
            "lab": lab,
            "level_db": float(level_db),
            "metadata": self._metadata(production, lab, float(level_db)),
        }

    def _metadata(self, production: dict, lab: dict, level_db: float) -> dict:
        """The versioned JSON that travels beside the tensors.

        Shapes and dtypes are written down so that reading a voice back is a
        *check* rather than an unpacking. A file whose ``mel`` is suddenly two
        dimensions instead of three is refused here, where the answer is "this
        voice needs preparing again", rather than three calls later inside a
        model where the answer is a stack trace.
        """
        return {
            "schema": PREPARATION_SCHEMA,
            "fingerprint": self.fingerprint,
            "sample_rate": int(self.sample_rate),
            "level_db": float(level_db),
            "hop_ratio": int(self.hop_ratio),
            "style_ctrl_dim": int(self.style_ctrl_dim),
            "sopro_version": self.sopro_version,
            "torch_version": self.torch_version,
            "production": {name: {"shape": list(tensor.shape),
                                  "dtype": str(tensor.dtype).replace("torch.", "")}
                           for name, tensor in production.items()},
            "lab": {name: {"shape": list(tensor.shape),
                           "dtype": str(tensor.dtype).replace("torch.", "")}
                    for name, tensor in lab.items()},
        }

    def save(self, tensors: dict, path: str) -> str:
        """Write conditioning as safetensors and report its digest.

        safetensors rather than ``torch.save`` because a saved voice is a file a
        user keeps, syncs and restores, and ``torch.save`` is a pickle -- which
        makes "restore an old voice" a code-execution decision. Section 15 is
        explicit about it and this is the one line that implements it.
        """
        from safetensors.torch import save_file

        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file({name: value.contiguous() for name, value in tensors.items()}, path)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    def _read_tensors(self, path: str, wanted, expected: dict):
        """One safetensors file, checked against what the metadata promised.

        Every check here is a refusal a malformed file could otherwise turn into
        an allocation or a silent misread: the exact set of names, the declared
        shape, the declared dtype, and a ceiling on the size before the bytes are
        materialised.
        """
        from safetensors.torch import load_file

        if not os.path.isfile(path):
            raise ValueError("that voice's conditioning file is missing")
        size = os.path.getsize(path)
        if size <= 0 or size > MAX_TENSOR_BYTES:
            raise ValueError("that voice's conditioning file is not a size this build reads")
        found = load_file(path, device="cpu")
        if set(found) != set(wanted):
            raise ValueError("that voice's conditioning file does not hold what it should")
        for name in wanted:
            promised = dict((expected or {}).get(name) or {})
            tensor = found[name]
            if promised.get("shape") and list(tensor.shape) != list(promised["shape"]):
                raise ValueError(f"that voice's {name} is not the shape it was saved as")
            if promised.get("dtype") and str(tensor.dtype).replace("torch.", "") != \
                    str(promised["dtype"]):
                raise ValueError(f"that voice's {name} is not the type it was saved as")
            if tensor.numel() * tensor.element_size() > MAX_TENSOR_BYTES:
                raise ValueError(f"that voice's {name} is larger than this build reads")
        return found

    # -- reconstruction and warming ---------------------------------------- #

    def reference(self, voice_id: str):
        """The canonical ``Reference`` for a saved voice, from cache or from disk.

        Reconstructed with ``prompt_states`` empty, always. The warmed state is
        built separately, keyed by the settings it is only valid for, so a
        change of solver steps or chunk size invalidates the warm cache without
        touching the reconstruction (section 29).
        """
        wanted = str(voice_id or "")
        entry = self.catalog.get(wanted)
        if entry is None:
            raise ValueError("that voice is not in this worker's catalogue")
        key = f"{wanted}|{entry['fingerprint']}"
        with self._cache_lock:
            found = self._references.get(key)
            if found is not None:
                self._references.move_to_end(key)
                return found, "hit"

        meta = self._read_metadata(entry["root"])
        if meta.get("fingerprint") and entry["fingerprint"] \
                and meta["fingerprint"] != entry["fingerprint"]:
            raise ValueError("that voice was prepared by a different Sopro build")
        tensors = self._read_tensors(
            os.path.join(entry["root"], "production.safetensors"),
            PRODUCTION_TENSORS, meta.get("production") or {})

        from sopro.model import Reference

        found = Reference(cond_vec=tensors["cond_vec"].to(self.device),
                          semantic_tokens=tensors["semantic_tokens"].to(self.device),
                          mel=tensors["mel"].to(self.device),
                          level_db=float(meta.get("level_db", -19.8)))
        with self._cache_lock:
            self._references[key] = found
            while len(self._references) > REFERENCE_CACHE:
                self._references.popitem(last=False)
        return found, "reconstruct"

    def _read_metadata(self, root: str) -> dict:
        path = os.path.join(root, "production.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                found = json.load(handle)
        except (OSError, ValueError):
            raise ValueError("that voice's description could not be read") from None
        if not isinstance(found, dict):
            raise ValueError("that voice's description is not what this build writes")
        if int(found.get("schema") or 0) != PREPARATION_SCHEMA:
            raise ValueError("that voice was saved in a layout this build does not read")
        return found

    def warm(self, voice_id: str, delivery: Delivery = None) -> str:
        """Build and cache the streaming state for a voice at these settings.

        The cold-start overlap (section 47): called while the language model is
        still writing its opening, so the first committed unit meets a warm
        prompt state rather than paying for one. Returns what the cache did, for
        the telemetry field that distinguishes a slow first sentence caused by
        the model from one caused by a cache miss.
        """
        found = delivery or NEUTRAL
        reference, state = self.reference(voice_id)
        steps = int(found.steps or self.steps or self.generation.steps)
        chunk = int(found.chunk_frames or self.chunk_frames
                    or self.generation.stream_chunk_frames)
        entry = self.catalog.get(str(voice_id)) or {}
        key = f"{voice_id}|{entry.get('fingerprint', '')}|{steps}|{chunk}|{self.precision}"
        with self._cache_lock:
            if key in self._prompts:
                self._prompts.move_to_end(key)
                reference.prompt_states = dict(self._prompts[key])
                return "warm"
        # Built through Sopro's own private helper on purpose. Rebuilding it
        # here would be a second implementation of the streaming cache that
        # would have to be kept in step with the pinned model's topology.
        self.tts._prompt_state(reference, steps, chunk)
        # The Reference is the one in the reference cache, and ``_prompt_state``
        # keeps every state it has ever built on it -- keyed by (steps,
        # chunk_frames), so changing either setting would leave the old one
        # attached to a live object for as long as the voice stayed cached.
        # Trimmed to the current one, which is what makes :data:`PROMPT_CACHE`
        # a real bound rather than a bound on one of two places these live
        # (I-11, section 29).
        wanted = (int(steps), int(chunk))
        reference.prompt_states = {name: value
                                   for name, value in reference.prompt_states.items()
                                   if name == wanted}
        with self._cache_lock:
            self._prompts[key] = dict(reference.prompt_states)
            while len(self._prompts) > PROMPT_CACHE:
                self._prompts.popitem(last=False)
        return state

    # -- speaking ----------------------------------------------------------- #

    def stream(self, text: str, voice_id: str, delivery: Delivery, on_audio,
               reference=None) -> dict:
        """Synthesize one committed unit, handing PCM16 over as it appears.

        ``on_audio(pcm, rate)`` returns True to continue and False to stop, and
        returning False is how cancellation actually happens: the generator is
        abandoned at the chunk boundary, which is the guarantee section 49 says
        is the honest one for a public ``stream()`` with no inner callback.

        The bytes handed over are built here, inside the loop, from the tensor
        Sopro yielded. Converting immediately is what makes "nothing downstream
        holds a reference to model-owned memory" true of everything after this
        point rather than merely intended.
        """
        found = delivery or NEUTRAL
        if found.seed is not None:
            import torch

            torch.manual_seed(int(found.seed))
        if reference is None:
            reference, _state = self.reference(voice_id)
            self.warm(voice_id, found)
        shaper = Shaper(found, self._numpy)
        trim = Trim(self.sample_rate, floor=self._quiet_floor.get(voice_id, 0))
        seam = Seam(self.sample_rate)
        arguments = found.generation(self.generation)
        if self.steps and "steps" not in arguments:
            arguments["steps"] = int(self.steps)
        if self.chunk_frames and "chunk_frames" not in arguments:
            arguments["chunk_frames"] = int(self.chunk_frames)

        started = time.monotonic()
        first = 0.0
        chunks = 0
        produced = 0
        stopped = False
        stream = self.tts.stream(text, ref=reference, **arguments)
        try:
            for chunk in stream:
                block = seam.block(trim.block(self._as_pcm(chunk, shaper)))
                chunks += 1
                if not first:
                    first = time.monotonic() - started
                if block:
                    produced += len(block) // 2
                    if not on_audio(block, self.sample_rate):
                        stopped = True
                        break
        finally:
            # Closing the generator is what returns control to Sopro's own
            # ``finally`` blocks and lets the StreamSession go. Left to the
            # collector it would hold a warmed session for as long as the
            # reference count survived, which on a cancelled turn is exactly
            # when this process should be quiet.
            try:
                stream.close()
            except Exception:
                pass
        # What this unit measured its own noise floor to be, kept for the next
        # unit in the same voice. A noise floor belongs to the voice -- a clone
        # carries its reference recording's room tone -- and the front of a unit
        # has to be judged before that unit has measured anything.
        if len(self._quiet_floor) > REFERENCE_CACHE:
            self._quiet_floor.clear()
        self._quiet_floor[voice_id] = trim.measured
        if not stopped:
            # The shaper's own tail is more of this unit's audio and goes
            # through the trim and the seam like every other block. Then the
            # trim's kept tail, and then ``seam.flush`` -- the milliseconds the
            # seam withheld, ramped down to silence, so the join to whatever
            # comes next (the pause, the next sentence, or the end of the reply)
            # is not a step.
            for tail in (seam.block(trim.block(shaper.flush())) if shaper.active else b"",
                         seam.block(trim.flush()),
                         seam.flush()):
                if not tail:
                    continue
                produced += len(tail) // 2
                on_audio(tail, self.sample_rate)
        return {
            "chars": len(text or ""),
            "first_audio_ms": int(first * 1000),
            "segment_ms": int((time.monotonic() - started) * 1000),
            "samples": produced,
            "chunks": chunks,
            "trimmed_ms": trim.dropped_ms,
            "quiet_ms": trim.quiet_ms,
            "floor_db": trim.floor_db,
            "cancelled": stopped,
        }

    def _as_pcm(self, chunk, shaper: "Shaper") -> bytes:
        """One Sopro tensor as wire PCM16, shaped, with NaN refused not muted.

        A model that produced NaN has produced something no amount of clipping
        makes into speech, and quietly turning it into zeros would be a silent
        second of nothing that nobody could diagnose. Refused here, which ends
        the unit with a reason.
        """
        values = chunk.detach().to("cpu").reshape(-1)
        if self._numpy is not None:
            array = values.numpy()
            if not bool(self._numpy.isfinite(array).all()):
                raise ValueError("the model produced audio that is not a number")
            return shaper.block(array) if shaper.active else pcm16(array, shaper.delivery.gain)
        return pcm16(values.tolist(), shaper.delivery.gain)

    def synthesize(self, text: str, voice_id: str, delivery: Delivery,
                   reference=None) -> bytes:
        """One complete string as a WAV, through the same path Conversation uses.

        Section 36: an audition answers "if I use this voice in Conversation
        with these settings, what will I hear?", and it can only answer that if
        it is the same reconstruction, the same generation arguments and the
        same DSP. So this collects :meth:`stream` rather than calling Sopro
        differently.
        """
        blocks = []
        self.stream(text, voice_id, delivery,
                    lambda pcm, _rate: (blocks.append(pcm), True)[1],
                    reference=reference)
        return encode_wav(b"".join(blocks), self.sample_rate)

    # -- the Voice Lab ------------------------------------------------------ #

    def lab_reference(self, voice_id: str, deltas=None, blend=None):
        """A throwaway ``Reference`` built from copies, for an audition only.

        The Lab's whole isolation guarantee lives in three properties of this
        function, and they are properties rather than promises:

        * every tensor it touches is a ``clone()``, so nothing it does can reach
          the cached production ``Reference``;
        * the result is returned rather than stored, so no production lookup can
          ever be satisfied by it;
        * it writes nothing at all -- there is no path argument and no ``save``
          call below.

        ``deltas`` are bounded offsets on the saved ``style_ctrl``, not a
        replacement of it, so Reset is exactly "all zeros". ``blend`` mixes
        another registered voice's pre-projection components in, at a bounded
        weight, holding this voice's semantic and mel reference context -- which
        is why section 42 calls it Conditioning Blend and not identity transfer.
        """
        import torch
        from sopro.model import Reference

        base, _state = self.reference(voice_id)
        parts = self._lab_components(voice_id)
        pieces = {name: parts[name].clone() for name in LAB_TENSORS}

        if blend:
            other = str(blend.get("voice_id") or "")
            weight = float(_bounded(blend.get("weight"), (0.0, 1.0), 0.0))
            fields = [name for name in LAB_TENSORS if blend.get(name)]
            if other and weight > 0.0 and fields:
                theirs = self._lab_components(other)
                for name in fields:
                    if theirs[name].shape != pieces[name].shape:
                        raise ValueError("those two voices were prepared by different builds")
                    pieces[name] = (pieces[name] * (1.0 - weight)
                                    + theirs[name].to(pieces[name].dtype) * weight)

        if deltas:
            offsets = [float(_bounded(value, (-3.0, 3.0), 0.0)) for value in deltas]
            control = pieces["style_ctrl"]
            width = int(control.shape[-1])
            if len(offsets) > width:
                offsets = offsets[:width]
            addition = torch.zeros(width, dtype=control.dtype)
            for index, value in enumerate(offsets):
                addition[index] = value
            pieces["style_ctrl"] = control + addition.view(
                *([1] * (control.dim() - 1)), width)

        with torch.no_grad():
            cond_vec = self.tts.model.build_condition(
                *(pieces[name].to(device=self.device, dtype=self.tts.model.dtype)
                  for name in LAB_TENSORS))
        return Reference(cond_vec=cond_vec,
                         semantic_tokens=base.semantic_tokens.clone(),
                         mel=base.mel.clone(),
                         level_db=float(base.level_db))

    def _lab_components(self, voice_id: str) -> dict:
        wanted = str(voice_id or "")
        entry = self.catalog.get(wanted)
        if entry is None:
            raise ValueError("that voice is not in this worker's catalogue")
        key = f"{wanted}|{entry['fingerprint']}"
        with self._cache_lock:
            found = self._lab.get(key)
            if found is not None:
                self._lab.move_to_end(key)
                return found
        meta = self._read_metadata(entry["root"])
        found = self._read_tensors(os.path.join(entry["root"], "lab-conditioning.safetensors"),
                                   LAB_TENSORS, meta.get("lab") or {})
        with self._cache_lock:
            self._lab[key] = found
            while len(self._lab) > LAB_CACHE:
                self._lab.popitem(last=False)
        return found

    def lab_style(self, voice_id: str) -> list:
        """The saved ``style_ctrl`` a Lab session's sliders start from.

        Returned as plain numbers so the parent can show what zero means without
        the Lab needing a tensor. Eight of them, at the reviewed topology.
        """
        found = self._lab_components(voice_id)["style_ctrl"].reshape(-1)
        return [round(float(value), 6) for value in found]


# --------------------------------------------------------------------------- #
# Turns
# --------------------------------------------------------------------------- #

MAX_PENDING_AUDIO = 24
"""Audio frames allowed to be waiting for the writer at once.

The flow-control valve, and twice Kokoro's because Sopro's chunks are shorter:
at 64 frames a chunk is a fraction of a second rather than a sentence, so the
same number of frames would be a much smaller buffer. Past this the generator
loop waits, so synthesis slows to whatever the parent is actually consuming;
below it the writer always has work. The wait is on a condition variable that
cancellation notifies, which is what stops backpressure from making a cancel
queue behind its own producer (section 50).
"""

WRITE_TIMEOUT = 0.25
DRAIN_GRACE = 2.0


class Turn:
    """One streaming reply inside this process."""

    def __init__(self, identifier: str, voice_id: str, delivery: Delivery = None):
        self.id = str(identifier or "")
        self.voice_id = str(voice_id or "")
        self.delivery = delivery or NEUTRAL
        self.segments = queue.Queue()
        self.cancelled = threading.Event()
        self.source_done = threading.Event()
        self.started = time.monotonic()
        self.samples = 0

    def finish(self) -> None:
        self.source_done.set()
        self.segments.put(None)

    def stop(self) -> None:
        self.cancelled.set()
        self.source_done.set()
        self.segments.put(None)

    def next_segment(self):
        """The next committed unit, or ``None`` when this turn has no more.

        Waits in short steps so that a cancel arriving on the command loop is
        noticed here within :data:`WRITE_TIMEOUT` rather than whenever the parent
        happens to send more text.
        """
        while not self.cancelled.is_set():
            try:
                found = self.segments.get(timeout=WRITE_TIMEOUT)
            except queue.Empty:
                if self.source_done.is_set():
                    return None
                continue
            if found is None:
                if self.source_done.is_set() and self.segments.empty():
                    return None
                continue
            return found
        return None


class Worker:
    """The command loop, the inference lane and the writer, and what they share.

    Everything mutable in this process lives here and every field is touched
    under a lock or is an ``Event``. There is deliberately no lock that spans a
    call into Torch: a thread inside the solver holds nothing anybody else
    needs, which is what makes cancel, status and shutdown reachable while
    Sopro is three seconds into a paragraph.
    """

    def __init__(self, stdout):
        self._stdout = stdout
        self._frames = queue.Queue()
        self._pending_audio = 0
        self._audio_room = threading.Condition()
        self._jobs = collections.deque()
        self._jobs_ready = threading.Condition()
        self._turns = {}
        self._turn_lock = threading.Lock()
        self._stopping = threading.Event()
        self._working = threading.Event()
        self.engine = None
        self._writer = None
        self._lane = None

    # -- the writer -------------------------------------------------------- #

    def start_threads(self) -> None:
        self._writer = threading.Thread(target=self._write_loop, name="sopro-writer",
                                        daemon=True)
        self._writer.start()
        self._lane = threading.Thread(target=self._lane_loop, name="sopro-lane", daemon=True)
        self._lane.start()

    def send(self, header: dict, payload: bytes = b"", audio: bool = False) -> None:
        """Queue one frame for stdout. Never writes from the calling thread.

        One writer is not a style choice: two threads writing length-prefixed
        frames to the same pipe produce a stream where a header is followed by
        somebody else's payload, and the parent's reader has no way to notice.
        """
        if audio:
            with self._audio_room:
                while (self._pending_audio >= MAX_PENDING_AUDIO
                       and not self._stopping.is_set()):
                    self._audio_room.wait(WRITE_TIMEOUT)
                self._pending_audio += 1
        self._frames.put((header, payload, audio))

    def _write_loop(self) -> None:
        while True:
            item = self._frames.get()
            if item is None:
                return
            header, payload, audio = item
            try:
                write_frame(self._stdout, header, payload)
            except Exception:
                self._stopping.set()
            finally:
                if audio:
                    with self._audio_room:
                        self._pending_audio = max(0, self._pending_audio - 1)
                        self._audio_room.notify_all()

    def finish_pending(self, timeout: float) -> None:
        """Answer what has already been accepted, for up to ``timeout``.

        Run when the parent's pipe closes rather than when it asks to shut down.
        A request this process took off the wire is one somebody is waiting for;
        dropping it because stdin happened to end first turns a clone into a
        timeout on the other side. Bounded, because the reason stdin ended may
        be that the parent died.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline and not self._stopping.is_set():
            with self._jobs_ready:
                if not self._jobs and not self._working.is_set():
                    return
                self._jobs_ready.wait(0.05)

    def wake_writers(self) -> None:
        with self._audio_room:
            self._audio_room.notify_all()

    # -- the inference lane ------------------------------------------------- #

    def submit(self, job, first: bool = False) -> None:
        with self._jobs_ready:
            if first:
                self._jobs.appendleft(job)
            else:
                self._jobs.append(job)
            self._jobs_ready.notify()

    def _lane_loop(self) -> None:
        """One model call at a time, forever.

        Serialized rather than parallel because this process shares a CPU with
        an image model and a language model: two solvers running at once would
        take the cores that make Forge feel responsive, which is a trade nobody
        asked for by installing a text-to-speech engine.
        """
        while not self._stopping.is_set():
            with self._jobs_ready:
                while not self._jobs and not self._stopping.is_set():
                    self._jobs_ready.wait(WRITE_TIMEOUT)
                if self._stopping.is_set():
                    return
                job = self._jobs.popleft()
                self._working.set()
            try:
                job()
            except Exception as exc:
                _note(f"an inference job failed: {_safe(exc)}")
            finally:
                with self._jobs_ready:
                    if not self._jobs:
                        self._working.clear()
                    self._jobs_ready.notify_all()

    # -- turns -------------------------------------------------------------- #

    def turn(self, identifier: str):
        with self._turn_lock:
            return self._turns.get(str(identifier or ""))

    def open_turn(self, identifier: str, voice_id: str, delivery: Delivery = None) -> Turn:
        found = Turn(identifier, voice_id, delivery)
        with self._turn_lock:
            self._turns[found.id] = found
        return found

    def close_turn(self, identifier: str) -> None:
        with self._turn_lock:
            self._turns.pop(str(identifier or ""), None)

    def stop_all_turns(self) -> None:
        with self._turn_lock:
            found = list(self._turns.values())
        for turn in found:
            turn.stop()
        self.wake_writers()

    def speak(self, turn: Turn) -> None:
        """One whole turn, committed unit by committed unit, on the lane."""
        engine = self.engine
        try:
            reference, cache_state = engine.reference(turn.voice_id)
            cache_state = engine.warm(turn.voice_id, turn.delivery) or cache_state
        except Exception as exc:
            self.send({"op": "tts_error", "turn": turn.id, "error": _safe(exc)})
            self.close_turn(turn.id)
            return
        self.send({"op": "tts_ready", "turn": turn.id,
                   "sample_rate": int(engine.sample_rate or 0),
                   "cache_state": cache_state, "streaming": "chunk"})
        sequence = 0
        started = time.monotonic()
        spoken = 0
        gap = silence(int(engine.sample_rate or 0), turn.delivery.pause_ms)
        try:
            while True:
                text = turn.next_segment()
                if text is None:
                    break

                def on_audio(chunk: bytes, rate: int, _turn=turn) -> bool:
                    nonlocal sequence
                    if _turn.cancelled.is_set() or self._stopping.is_set():
                        return False
                    sequence += 1
                    _turn.samples += len(chunk) // 2
                    self.send({"op": "tts_audio", "turn": _turn.id, "seq": sequence,
                               "sample_rate": int(rate or 0)}, chunk, audio=True)
                    return not _turn.cancelled.is_set()

                # Before this unit rather than after the last one, so a turn
                # cancelled mid-reply does not end on a gap and one that
                # finishes does not leave the composer waiting through one.
                if gap and spoken and not turn.cancelled.is_set():
                    on_audio(gap, int(engine.sample_rate or 0))
                metrics = engine.stream(text, turn.voice_id, turn.delivery, on_audio,
                                        reference=reference)
                spoken += 1
                if turn.cancelled.is_set():
                    break
                self.send({"op": "tts_segment_done", "turn": turn.id, "seq": sequence,
                           "chars": metrics["chars"],
                           "first_audio_ms": metrics["first_audio_ms"],
                           "segment_ms": metrics["segment_ms"],
                           "samples": metrics["samples"],
                           "chunks": metrics["chunks"],
                           "trimmed_ms": metrics["trimmed_ms"],
                           "quiet_ms": metrics["quiet_ms"],
                           "floor_db": metrics["floor_db"],
                           "sample_rate": int(engine.sample_rate or 0),
                           "speed_dsp": "active" if abs(turn.delivery.stretch_rate - 1.0) > 1e-6
                                        else "neutral",
                           "pitch_dsp": "active" if abs(turn.delivery.pitch - 1.0) > 1e-6
                                        else "neutral"})
        except Exception as exc:
            self.send({"op": "tts_error", "turn": turn.id, "error": _safe(exc)})
            self.close_turn(turn.id)
            _note(f"speech failed: {_safe(exc)}")
            return
        elapsed = time.monotonic() - started
        if turn.cancelled.is_set():
            self.send({"op": "tts_cancelled", "turn": turn.id, "seq": sequence})
            _note(f"speech cancelled after {turn.samples} samples in {elapsed:.1f} s")
        else:
            self.send({"op": "tts_done", "turn": turn.id, "seq": sequence,
                       "samples": turn.samples})
            _note(f"speech finished — {turn.samples} samples in {elapsed:.1f} s")
        self.close_turn(turn.id)


# --------------------------------------------------------------------------- #
# The command loop
# --------------------------------------------------------------------------- #


def serve(stdin, stdout, engine_factory=None) -> int:
    """Read frames until end of input. Returns the process exit status.

    This is the command loop and it does as little as it possibly can. Its one
    job is to never be busy: everything that could take longer than parsing a
    header is handed to the inference lane, so a ``tts_cancel`` sent while the
    acoustic solver is running is *read* immediately rather than after the
    paragraph. That property is the acceptance criterion for Gate S-4, and it is
    why this function contains no model call at all.
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
                worker.finish_pending(DRAIN_GRACE)
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
                        "protocol_version": PROTOCOL_VERSION,
                        "backend": "sopro",
                        "parent_death": parent_death,
                        "device": engine.device,
                        "model_id": engine.model_id,
                        "fingerprint": engine.fingerprint,
                        "sopro_version": engine.sopro_version,
                        "torch_version": engine.torch_version,
                        "precision": engine.precision,
                        "sample_rate": engine.sample_rate,
                        "hop_ratio": engine.hop_ratio,
                        "style_ctrl_dim": engine.style_ctrl_dim,
                        "voices": len(engine.catalog),
                        "streaming": "chunk",
                        "defaults": engine.defaults(),
                        "load_seconds": engine.load_seconds,
                        **engine.threads,
                    })
                    _note(f"ready — {engine.device}, precision {engine.precision}, "
                          f"{engine.threads.get('intraop_threads')} intra-op / "
                          f"{engine.threads.get('interop_threads')} inter-op threads, "
                          f"{len(engine.catalog)} voices, containment {parent_death}")
                except SystemExit:
                    raise
                except Exception as exc:
                    worker.engine = None
                    reply(request_id, {"ok": False, "error": _safe(exc)})
                    _note(f"could not load Sopro: {_safe(exc)}")
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
                count = worker.engine.refresh_catalog(header.get("voices") or {})
                for identifier in header.get("forget") or ():
                    worker.engine.forget(str(identifier))
                reply(request_id, {"ok": True, "voices": count})

            elif operation == "prepare":
                worker.submit(lambda rid=request_id, head=header, body=payload:
                              _do_prepare(worker, reply, rid, head, body), first=True)

            elif operation == "warm":
                worker.submit(lambda rid=request_id, head=header:
                              _do_warm(worker, reply, rid, head), first=True)

            elif operation == "tts":
                worker.submit(lambda rid=request_id, head=header, body=payload:
                              _do_tts(worker, reply, rid, head, body))

            elif operation == "lab":
                worker.submit(lambda rid=request_id, head=header, body=payload:
                              _do_lab(worker, reply, rid, head, body))

            elif operation == "lab_style":
                worker.submit(lambda rid=request_id, head=header:
                              _do_lab_style(worker, reply, rid, head), first=True)

            elif operation == "tts_begin":
                turn = worker.open_turn(str(header.get("turn") or ""),
                                        str(header.get("voice_id") or ""),
                                        Delivery.from_header(header))
                worker.submit(lambda found=turn: worker.speak(found))

            elif operation == "tts_text":
                found = worker.turn(str(header.get("turn") or ""))
                if found is not None and not found.cancelled.is_set():
                    found.segments.put(payload.decode("utf-8", "replace"))

            elif operation == "tts_finish":
                found = worker.turn(str(header.get("turn") or ""))
                if found is not None:
                    found.finish()

            elif operation == "tts_cancel":
                found = worker.turn(str(header.get("turn") or ""))
                if found is not None:
                    found.stop()
                    worker.wake_writers()
                else:
                    # A turn this process never had, or one it has already
                    # finished. Answered anyway so the parent's bookkeeping
                    # closes: cancelling twice is defined behaviour.
                    worker.send({"op": "tts_cancelled",
                                 "turn": str(header.get("turn") or ""), "seq": 0})

            else:
                reply(request_id, {"ok": False, "error": "unknown operation"})
    finally:
        worker._stopping.set()
        worker.stop_all_turns()
        with worker._jobs_ready:
            worker._jobs_ready.notify_all()
        try:
            worker._frames.put(None)
        except Exception:
            pass
        writer = worker._writer
        if writer is not None and writer.is_alive():
            writer.join(timeout=2.0)


def _do_prepare(worker, reply, request_id, header: dict, payload: bytes) -> None:
    """Prepare one reference into the two assets, and prove it before answering.

    The order is section 27's and every step of it is a different failure:
    prepare, write both assets, reconstruct *from the files that were written*,
    stream a short audition through the production path, and only then say yes.
    A voice that prepared cleanly and cannot be read back is a voice the
    registry must never learn about, and reading it back is the only way to know.
    """
    engine = worker.engine
    root = str(header.get("root") or "")
    try:
        if not root:
            raise ValueError("no destination was given for that voice")
        made = engine.prepare(payload, seconds=float(header.get("seconds") or 0.0))
        production_path = os.path.join(root, "production.safetensors")
        lab_path = os.path.join(root, "lab-conditioning.safetensors")
        metadata = dict(made["metadata"])
        metadata["production_sha256"] = engine.save(made["production"], production_path)
        metadata["lab_sha256"] = engine.save(made["lab"], lab_path)
        with open(os.path.join(root, "production.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        # Reconstructed from disk rather than reused from memory. The tensors
        # above are known good; what is being proved here is that the *files*
        # are, which is the only thing a restart will have.
        identifier = str(header.get("voice_id") or "")
        engine.refresh_catalog({**engine.catalog,
                                identifier: {"root": root,
                                             "fingerprint": engine.fingerprint}})
        engine.forget(identifier)
        delivery = Delivery.from_header(header)
        started = time.monotonic()
        audio = engine.synthesize(str(header.get("audition") or "This is a test."),
                                  identifier, delivery)
        if len(audio) <= 44:
            raise ValueError("that voice was prepared but produced no audio")
        reply(request_id, {"ok": True, "metadata": metadata,
                           "sample_rate": engine.sample_rate,
                           "audition_ms": int((time.monotonic() - started) * 1000)}, audio)
        _note(f"prepared a voice in {metadata['production']['mel']['shape']} mel frames")
    except Exception as exc:
        reply(request_id, {"ok": False, "error": _safe(exc)})
        _note(f"voice preparation failed: {_safe(exc)}")


def _do_warm(worker, reply, request_id, header: dict) -> None:
    try:
        state = worker.engine.warm(str(header.get("voice_id") or ""),
                                   Delivery.from_header(header))
        reply(request_id, {"ok": True, "cache_state": state})
    except Exception as exc:
        reply(request_id, {"ok": False, "error": _safe(exc)})


def _do_tts(worker, reply, request_id, header: dict, payload: bytes) -> None:
    """The completed-audio route: auditions, and the Test button.

    Section 36 -- the same reconstruction, the same generation settings and the
    same DSP as a spoken reply, collected instead of streamed. An audition that
    called Sopro differently would be an audition of something else.
    """
    try:
        text = payload.decode("utf-8", "replace")
        started = time.monotonic()
        audio = worker.engine.synthesize(text, str(header.get("voice_id") or ""),
                                         Delivery.from_header(header))
        elapsed = time.monotonic() - started
        reply(request_id, {"ok": True, "sample_rate": worker.engine.sample_rate,
                           "elapsed": round(elapsed, 3)}, audio)
        _note(f"audition finished — {len(text)} characters in {elapsed:.1f} s")
    except Exception as exc:
        reply(request_id, {"ok": False, "error": _safe(exc)})
        _note(f"request failed: {_safe(exc)}")


def _do_lab(worker, reply, request_id, header: dict, payload: bytes) -> None:
    """One Voice Lab audition. Reads copies, writes nothing, returns a WAV.

    Structurally separate from :func:`_do_tts` rather than a flag on it, which
    is section 39's "production code cannot accidentally read Lab state because
    the types are separate rather than because a caller promises not to". There
    is no path from this function into the production caches, the registry or a
    character.
    """
    try:
        engine = worker.engine
        reference = engine.lab_reference(str(header.get("voice_id") or ""),
                                         deltas=header.get("deltas") or (),
                                         blend=header.get("blend") or None)
        delivery = Delivery.from_header(header)
        started = time.monotonic()
        blocks = []
        metrics = engine.stream(payload.decode("utf-8", "replace"),
                                str(header.get("voice_id") or ""), delivery,
                                lambda pcm, _rate: (blocks.append(pcm), True)[1],
                                reference=reference)
        audio = encode_wav(b"".join(blocks), engine.sample_rate)
        elapsed = time.monotonic() - started
        seconds = metrics["samples"] / float(engine.sample_rate or 1)
        reply(request_id, {
            "ok": True,
            "sample_rate": engine.sample_rate,
            "first_audio_ms": metrics["first_audio_ms"],
            "elapsed_ms": int(elapsed * 1000),
            "audio_ms": int(seconds * 1000),
            "rtf": round(elapsed / seconds, 3) if seconds else 0.0,
            "chunks": metrics["chunks"],
        }, audio)
    except Exception as exc:
        reply(request_id, {"ok": False, "error": _safe(exc)})
        _note(f"lab audition failed: {_safe(exc)}")


def _do_lab_style(worker, reply, request_id, header: dict) -> None:
    try:
        found = worker.engine.lab_style(str(header.get("voice_id") or ""))
        reply(request_id, {"ok": True, "style_ctrl": found})
    except Exception as exc:
        reply(request_id, {"ok": False, "error": _safe(exc)})


# --------------------------------------------------------------------------- #
# The self-test
# --------------------------------------------------------------------------- #


def selftest() -> int:
    """Prove the staged runtime imports, runs on the CPU, and computes correctly.

    Run by the installer against the *staging* environment, before anything is
    promoted, and it is Gate S-0 and the first half of Gate S-1 in one command.
    Four claims, in the order they fail:

        the closure imports at all                    -- a wrong wheel for this
                                                         Python or architecture
        Torch has no visible accelerator              -- I-9, checked rather
                                                         than assumed
        attention is repeatable and correct here      -- the SDPA correctness
                                                         gate section 20 asks for
        Sopro's own module and version are what was pinned

    One JSON line to stdout, because the installer parses it and the console
    shows it. An install that fails any of these is refused with the number it
    was out by rather than "self-test failed", which is the difference between
    one round trip and four.
    """
    report = {"ok": False, "device": "", "torch_version": "", "sopro_version": "",
              "numpy_version": "", "intraop_threads": 0, "interop_threads": 0}
    try:
        import numpy

        report["numpy_version"] = str(getattr(numpy, "__version__", "") or "")
        import torch

        report["torch_version"] = str(getattr(torch, "__version__", "") or "")
        threads = _apply_cpu_policy()
        report.update(threads)
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            report["error"] = ("this Torch build can see a graphics device, and the Sopro "
                               "worker is CPU-only")
            sys.stdout.write(json.dumps(report) + "\n")
            sys.stdout.flush()
            return 1
        report["device"] = "cpu"

        stable, why = _attention_is_stable()
        report["attention_stable"] = stable
        if not stable:
            report["error"] = why
            sys.stdout.write(json.dumps(report) + "\n")
            sys.stdout.flush()
            return 1

        import sopro

        report["sopro_version"] = str(getattr(sopro, "__version__", "") or "")
        # Imported rather than only named: ``sopro`` re-exports from modules
        # that pull in torchaudio, safetensors and sentencepiece, so a closure
        # missing any of them fails here rather than at the first clone.
        from sopro.model import Reference, SoproTTS  # noqa: F401
        from sopro.streaming import PromptState  # noqa: F401
        import sentencepiece  # noqa: F401
        import soundfile  # noqa: F401

        report["ok"] = True
    except Exception as exc:
        # The message, not just the class. These are about package and library
        # names, never about anything anybody said, and "ModuleNotFoundError" on
        # its own does not say *which* module.
        report["error"] = f"{exc.__class__.__name__}: {exc}"
        report["prefix"] = sys.prefix
        report["path"] = [entry for entry in sys.path if entry]
    sys.stdout.write(json.dumps(report) + "\n")
    sys.stdout.flush()
    return 0 if report["ok"] else 1


def benchmark(stdin=None) -> int:
    """Measure this closure on this machine, at whatever policy is in force.

    The half of I-12 that is easy to miss: the released CPU policy has to be a
    *measured* one, and 4 intra-op threads was chosen to match Kokoro's lane
    rather than because Sopro was measured against 6 or 8. This is how it gets
    measured, and :mod:`mc_voice_sopro_bench` is what drives it.

    Two lengths rather than one, because a single real-time factor conflates two
    costs that respond to completely different things. Synthesis time on this
    engine fits ``fixed + rate x audio`` very tightly, and the two halves point
    at different fixes: the fixed cost per unit is amortised by *longer
    segments*, and the marginal rate is what threads, precision and solver steps
    move. Reporting one number would hide which of those a machine actually
    needs.

    Nothing here writes anything, speaks to a browser, or touches product
    state. It reads a voice this installation already has, synthesizes fixed
    text into a counter, and prints one JSON line.
    """
    report = {"ok": False}
    try:
        request = json.loads((stdin or sys.stdin).read() or "{}")
        config = dict(request.get("config") or {})
        voice_id = str(request.get("voice_id") or "")
        repeats = max(1, min(10, int(request.get("repeats") or 3)))
        texts = list(request.get("texts") or [])
        if not texts:
            raise ValueError("the benchmark was given no text to speak")

        engine = Engine(config)
        engine.load()
        report.update(engine.threads)
        report["precision"] = engine.precision
        report["steps"] = engine.steps
        report["chunk_frames"] = engine.chunk_frames
        report["torch_version"] = engine.torch_version
        report["load_seconds"] = engine.load_seconds
        if not voice_id:
            voice_id = next(iter(engine.catalog), "")
        if not voice_id:
            raise ValueError("this installation has no Sopro voice to measure with")
        report["voice_id"] = voice_id

        # Warmed first and measured after, deliberately. A cold prompt state is
        # a real cost and it is section 47's, not the thread policy's; folding
        # it in would make every configuration look worse by the same constant
        # and tell nobody anything.
        engine.warm(voice_id, NEUTRAL)
        runs = []
        for text in texts:
            for index in range(repeats):
                samples = [0]

                def count(pcm, _rate, into=samples):
                    into[0] += len(pcm) // 2
                    return True

                started = time.monotonic()
                found = engine.stream(text, voice_id, NEUTRAL, count)
                runs.append({
                    "chars": len(text),
                    "run": index,
                    "compute_ms": int((time.monotonic() - started) * 1000),
                    "audio_ms": int(samples[0] * 1000 / (engine.sample_rate or 24000)),
                    "first_audio_ms": int(found.get("first_audio_ms") or 0),
                    "chunks": int(found.get("chunks") or 0),
                })
        report["runs"] = runs
        report["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the sweep prints it and moves on
        report["error"] = f"{exc.__class__.__name__}: {exc}"
    sys.stdout.write(json.dumps(report) + "\n")
    sys.stdout.flush()
    return 0 if report.get("ok") else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(MARKER, dest="marker", action="store_true")
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--session", default="")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    known, _unknown = parser.parse_known_args(argv)

    if known.selftest:
        return selftest()
    if known.benchmark:
        return benchmark()

    # Binary on both sides: the protocol is length-prefixed bytes, and a text
    # wrapper would translate newlines on Windows and corrupt every WAV.
    return serve(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
