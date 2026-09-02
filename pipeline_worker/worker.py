"""DPDFNet and LavaSR in a process of their own, over one continuous stream.

Launched by :mod:`mc_voice_pipeline_runtime` on the interpreter
:mod:`mc_voice_pipeline` installed, and never by Forge's. It is the only file in
this repository that imports an enhancement model, and the imports happen after
the handshake so a parent that refuses the handshake never pays for them.

The protocol is byte-identical to the Kokoro, Sopro, Pocket and cleanup workers
-- a big-endian length, a JSON header, a big-endian length, a payload -- by
agreement rather than by import, because they run on different interpreters and
sharing a module between them would be sharing a Python version too.

One stream, not a pile of packets
---------------------------------
This is the part that is easy to get wrong and expensive to get wrong quietly.

The parent hands over PCM in whatever blocks its reader thread happened to
receive. Those blocks are transport framing and nothing else: they are not
sentences, not model windows, not flush points and not reset points (I-VP-10).
The same audio cut into 10 ms packets and into one enormous packet must come out
the same. So every stage here appends to a turn-scoped buffer and takes what its
own algorithm asks for, and no stage has a method that means "the packet ended".

A source synthesis unit is not a boundary either (I-VP-11). PocketTTS emits
extra blocks at the end of a unit because its trim and its seam were holding
audio back, and to this file those are ordinary samples. The only boundaries
that exist here are ``TURN_BEGIN`` and ``TURN_END``.

The clock
---------
Duration is preserved from this worker's input boundary onward, exactly
(I-VP-08). If ``N`` source samples arrive at ``Fs_in`` and the turn's output
rate is ``Fs_out``, the committed output is exactly
``(N * Fs_out + Fs_in // 2) // Fs_in`` samples -- computed cumulatively from the
running total rather than per packet, so rounding cannot accumulate. For today's
Pocket path, 24 kHz in and 48 kHz out, that is exactly ``2 * N``.

What it will not do is restore anything. The source worker's trim, its
internal-gap shortening and its intentional pauses all happened before this
boundary and are authoritative: 300 ms of dead air the source removed is not in
``N`` and does not come back (I-VP-24, section 26.3).

What it never does
------------------
It has no HTTP client and no hub client. It is told local directories rather
than model names, its environment has no credential and no visible graphics
device, and every rate it works at is a number the parent measured and passed
in rather than one this file assumed. It writes nothing except its replies and
its stderr notes, and it never writes what anybody said -- numbers, enums and
durations only.
"""

from __future__ import annotations

import argparse
import array
import contextlib
import json
import math
import os
import struct
import sys
import time

PROTOCOL_VERSION = 1
"""Not shared with the other four workers' versions. They speak the same
framing and different vocabularies, and one number covering all five would make
any worker's protocol change everybody else's reinstall."""

MARKER = "--model-chain-pipeline-worker"

MAX_HEADER = 1 << 20
MAX_PAYLOAD = 16 * 1024 * 1024
"""Sixteen megabytes is about five minutes of 24 kHz PCM16 in one packet, which
is far more than the parent's reader will ever hand over in one block. It is a
number rather than "whatever arrives" (section 20.6)."""

MAX_TURN_SAMPLES = 3 * 60 * 60 * 48000
"""Three hours of audio at the highest rate this speaks. A bound rather than an
expectation: a turn is one reply, and a ledger with no ceiling is a memory leak
with a plausible excuse."""

LAVA_OUTPUT_RATE = 48000

PROVIDER_CPU = "CPUExecutionProvider"
PROVIDER_DIRECTML = "DmlExecutionProvider"
PROVIDER_CUDA = "CUDAExecutionProvider"
"""The two execution providers the pinned ONNX Runtime wheel carries.

The closure installs the DirectML build rather than the CPU-only one, and that
build carries both -- so a stage running on the processor and the same stage
running on a graphics card are one installation and two session options, not two
runtimes. The names are spelled here as well as in :mod:`mc_voice_device`
because this file runs inside the isolated runtime and cannot import that one;
they are checked against each other by a test rather than by an import.
"""

CROSSFADE_MS = 20
"""How long the join between two Lava analysis windows is ramped over.

Not the same thing as the source worker's 8 ms unit seam and not a replacement
for it (I-VP-30, section 10.9). That seam fixes the edges of a spoken unit; this
fixes the edges of an *analysis window*, which exists only inside this file and
which the source has never heard of. Both can be present in one stream and
neither is allowed to do the other's job.
"""

WINDOW_SLACK = 2
"""How many samples one analysis window's output may be out by and still be
right. Converting a rate in two steps rather than one can disagree in the last
place, which is arithmetic; anything larger is a backend."""

FRAMING_LIMIT = 10
"""One over the share of a window a backend may withhold and still be framing.

A tenth, and both sides of that are measured rather than chosen.

*Below* it is what a model's own analysis frame costs. LavaSR's BWE is a Vocos
ISTFT: it emits whole frames, so a window whose length is not a multiple of the
hop comes back a partial frame short -- 528 samples of a 16800-sample window on
the machine this was found on, or 3.1%. It is constant, it is bounded by one
frame, and it lands inside the trailing context this stage discards anyway, so
not one sample of it ever reaches the audio.

*Above* it is what a clock error costs. The rates in play are 16000, 24000 and
48000, so the smallest confusion any backend here can have about its own rate
returns half again or two thirds of what was asked for -- 33% at the very
least, and 50% for the 24-as-16 case the adapter exists to catch.

Three times the largest framing loss and a third of the smallest clock error is
a gap wide enough that neither has to be judged finely. What made this worth
separating is that counting them together refused a working installation: four
windows of a perfectly good model summed to 2112 samples of "correction" in one
second, tripped a tolerance meant for drift, and reported it as "the rate it
works at is not the rate it was told" -- about a stage whose audio was exact.
"""

TOLERATED_CORRECTION = 64
"""How many samples of reconciliation a well-behaved backend may need.

A backend that pads its own input pads its own output, and trimming a handful of
samples off the end of a window is ordinary. Sixty-four is about a millisecond
at 48 kHz -- large enough to absorb that, and small enough that a wrong sample
rate, which costs thousands, cannot hide underneath it.
"""

RESAMPLE_ZEROS = 16
"""Half-width of the band-limited resampling kernel, in zero crossings.

Linear interpolation would have been ten lines shorter and is the wrong tool: a
24 kHz source read at 16 kHz through a straight line folds everything above
8 kHz back down into the speech, and the stage immediately after this one is a
model whose whole job is to decide what the missing top of that speech should
be. Feeding a bandwidth-extension model its own aliasing is how you get a
confident, detailed, wrong high band.
"""

_LENGTH = struct.Struct(">I")


class Refusal(Exception):
    """Something this worker declined, in words this file wrote.

    A distinct type rather than ``ValueError`` for the reason the Pocket worker
    records: ``ValueError`` is the base of much of the numeric stack, so "it is
    a ValueError, therefore this file wrote it" is not true, and a library
    message naming a tensor shape or a file path could otherwise reach the
    parent's log through the one function whose job is to stop that.
    """


def _safe(exc: BaseException) -> str:
    """What the parent is told about a failure: a class name, or our own words.

    Anything this file raised deliberately is forwarded verbatim because this
    file knows what is in it -- no path, no rate it was not given, no sample. An
    exception from anywhere else contributes its class name and nothing else.
    """
    if isinstance(exc, Refusal):
        return str(exc)
    return exc.__class__.__name__


def _note(message: str) -> None:
    """One diagnostic line to stderr, which the parent drains into the log."""
    sys.stderr.write(f"[pipeline] {message}\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------- #
# Framing -- byte-identical to the other four workers, by agreement not by import
# --------------------------------------------------------------------------- #


def _read_exactly(stream, count: int):
    chunks = []
    remaining = count
    while remaining > 0:
        block = stream.read(remaining)
        if not block:
            return None
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def read_frame(stream):
    """One message, or ``None`` at end of input.

    ``None`` is how the parent's death arrives while this process is waiting for
    work, and it is not an error: the loop ends, the models are released, and
    the process exits 0.
    """
    header_length = _read_exactly(stream, 4)
    if header_length is None:
        return None
    (size,) = _LENGTH.unpack(header_length)
    if size > MAX_HEADER:
        raise Refusal("header too large")
    raw = _read_exactly(stream, size)
    if raw is None:
        return None
    header = json.loads(raw.decode("utf-8"))
    if not isinstance(header, dict):
        raise Refusal("header is not an object")

    payload_length = _read_exactly(stream, 4)
    if payload_length is None:
        return None
    (size,) = _LENGTH.unpack(payload_length)
    if size > MAX_PAYLOAD:
        raise Refusal("payload too large")
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


# --------------------------------------------------------------------------- #
# PCM, in stdlib only
# --------------------------------------------------------------------------- #


def to_floats(data: bytes) -> list:
    """Little-endian mono PCM16 as floats in [-1, 1].

    Explicitly little-endian rather than native, because the parent's framing is
    big-endian and a reader who saw ``array("h")`` alone would reasonably wonder
    which one this was. Every PCM byte in this feature is little-endian, from
    the source worker through to the browser's ``DataView``.
    """
    if len(data) % 2:
        raise Refusal("audio payload is not whole samples")
    numbers = array.array("h")
    numbers.frombytes(data)
    if sys.byteorder != "little":
        numbers.byteswap()
    return [value / 32768.0 for value in numbers]


def to_pcm16(samples) -> bytes:
    """Floats back to little-endian mono PCM16, clipped rather than wrapped.

    Clipped, because a value that has gone past 1.0 is a loud sample and
    wrapping it would turn a loud sample into the loudest possible click in the
    opposite direction. A bandwidth-extension model adding energy above what was
    there is exactly how a signal gets past 1.0, so this is a path that is
    expected to be taken rather than a defensive line.
    """
    numbers = array.array("h", bytes(2 * len(samples)))
    for index, value in enumerate(samples):
        if value != value:  # NaN, which no arithmetic below would catch
            value = 0.0
        scaled = int(value * 32767.0)
        if scaled > 32767:
            scaled = 32767
        elif scaled < -32768:
            scaled = -32768
        numbers[index] = scaled
    if sys.byteorder != "little":
        numbers.byteswap()
    return numbers.tobytes()


def deterministic_target(samples_in: int, rate_in: int, rate_out: int) -> int:
    """How many output samples ``samples_in`` source samples must become.

    The whole of the clock contract in one expression (section 8.6). Integer
    arithmetic on the *cumulative* input count, so the answer for a turn does
    not depend on how the turn was packetised and rounding has nothing to
    accumulate into. Half-up at the midpoint, which is a choice rather than a
    truth -- what matters is that it is the same choice everywhere.

    For the current Pocket path this is exactly ``2 * samples_in`` and the
    general form is kept anyway: the rate is a number the parent measured, and a
    file that special-cased doubling would be a file that quietly did the wrong
    thing the first time somebody attached an engine that speaks at 22050.
    """
    if rate_in <= 0 or rate_out <= 0:
        raise Refusal("a sample rate of zero cannot describe a duration")
    return (int(samples_in) * int(rate_out) + int(rate_in) // 2) // int(rate_in)


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


class Resampler:
    """Band-limited rational resampling, deterministic and stateless per call.

    Stateless is safe here and would not be anywhere else in this file: the only
    caller is the Lava adapter, which resamples one analysis window *including
    its context*, so the samples either side of every output position are real
    source rather than an assumed zero. That is what the context is for.

    ``numpy_module`` is optional and only makes it fast. The pure-Python path is
    the same arithmetic and exists so that the clock and window behaviour can be
    tested without a runtime installed, which is most of what there is to test
    about this class.
    """

    def __init__(self, rate_in: int, rate_out: int, numpy_module=None):
        self.rate_in = int(rate_in)
        self.rate_out = int(rate_out)
        self._np = numpy_module
        if self.rate_in <= 0 or self.rate_out <= 0:
            raise Refusal("a resampler needs two real sample rates")
        self.ratio = self.rate_out / float(self.rate_in)
        # Anti-alias to the lower of the two Nyquists when going down, and to
        # the source's own when going up: reconstructing above what was there is
        # the next stage's job and not a resampler's.
        self._cutoff = min(1.0, self.ratio)
        self._half = max(1, int(RESAMPLE_ZEROS / self._cutoff))

    @property
    def transparent(self) -> bool:
        """Whether this resampler is the identity and can be skipped entirely."""
        return self.rate_in == self.rate_out

    def count_for(self, samples: int) -> int:
        return deterministic_target(samples, self.rate_in, self.rate_out)

    def __call__(self, samples, count: int = -1):
        """``samples`` at ``rate_in`` read out at ``rate_out``.

        ``count`` is the exact number of output samples wanted. Passing it is
        how the caller keeps the clock rather than asking a float ratio to keep
        it: the adapter already knows what the answer has to be, and a resampler
        that decided for itself would be a second opinion about duration.
        """
        source = list(samples)
        if count < 0:
            count = self.count_for(len(source))
        if not source or count <= 0:
            return [0.0] * max(0, count)
        if self.transparent:
            if count <= len(source):
                return source[:count]
            return source + [source[-1]] * (count - len(source))
        if self._np is not None:
            return self._fast(source, count)
        return self._plain(source, count)

    def _positions(self, count: int):
        step = self.rate_in / float(self.rate_out)
        return [index * step for index in range(count)]

    def _plain(self, source, count: int):
        last = len(source) - 1
        cutoff = self._cutoff
        half = self._half
        found = []
        for position in self._positions(count):
            centre = int(math.floor(position))
            total = 0.0
            weight = 0.0
            for offset in range(centre - half + 1, centre + half + 1):
                distance = (position - offset) * cutoff
                tap = _sinc(distance) * _blackman(distance, half)
                if tap == 0.0:
                    continue
                index = 0 if offset < 0 else (last if offset > last else offset)
                total += source[index] * tap
                weight += tap
            found.append(total / weight if weight else 0.0)
        return found

    def _fast(self, source, count: int):
        np = self._np
        data = np.asarray(source, dtype=np.float64)
        last = data.size - 1
        step = self.rate_in / float(self.rate_out)
        positions = np.arange(count, dtype=np.float64) * step
        centres = np.floor(positions).astype(np.int64)
        total = np.zeros(count, dtype=np.float64)
        weight = np.zeros(count, dtype=np.float64)
        for offset in range(-self._half + 1, self._half + 1):
            taken = centres + offset
            distance = (positions - taken) * self._cutoff
            tap = np.sinc(distance) * _blackman_array(np, distance, self._half)
            total += data[np.clip(taken, 0, last)] * tap
            weight += tap
        with np.errstate(invalid="ignore", divide="ignore"):
            found = np.where(weight != 0, total / weight, 0.0)
        return [float(value) for value in found]


def _sinc(value: float) -> float:
    if value == 0.0:
        return 1.0
    scaled = math.pi * value
    return math.sin(scaled) / scaled


def _blackman(value: float, half: int) -> float:
    """The Blackman window of the kernel, zero outside it."""
    if abs(value) >= half:
        return 0.0
    phase = math.pi * (value + half) / half
    return 0.42 - 0.5 * math.cos(phase) + 0.08 * math.cos(2.0 * phase)


def _blackman_array(np, values, half: int):
    phase = math.pi * (values + half) / half
    window = 0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2.0 * phase)
    return np.where(np.abs(values) >= half, 0.0, window)


class StreamResampler:
    """Band-limited resampling with the read position carried between chunks.

    :class:`Resampler` is stateless because its only caller hands it a whole
    analysis window with real context on both sides. This one is for a
    *continuous* stream, where there is no window and no context -- only what has
    arrived so far -- and so it has to keep three things across calls: the input
    samples a future output will still read, where in that buffer the next
    output sits, and how many outputs have been produced.

    That state is the whole point. Restarting a resampler at every chunk
    boundary makes the audio depend on how the audio was packetised, which is
    the one thing this feature says it never does (I-VP-10) -- and it is exactly
    what upstream's own per-call resampling does, which is why this exists
    rather than that.

    Positions are computed from the cumulative output index in integer
    arithmetic, so the sample at output index *k* is the same sample whatever
    order the input arrived in.
    """

    def __init__(self, rate_in: int, rate_out: int, numpy_module=None):
        self.rate_in = int(rate_in)
        self.rate_out = int(rate_out)
        if self.rate_in <= 0 or self.rate_out <= 0:
            raise Refusal("a resampler needs two real sample rates")
        self._np = numpy_module
        self._cutoff = min(1.0, self.rate_out / float(self.rate_in))
        self._half = max(1, int(RESAMPLE_ZEROS / self._cutoff))
        self._held = []
        self._base = 0          # absolute input index of ``_held[0]``
        self._received = 0
        self._emitted = 0

    @property
    def transparent(self) -> bool:
        return self.rate_in == self.rate_out

    def feed(self, samples) -> list:
        """Every output sample this input now fully supports, and no more."""
        if self.transparent:
            self._received += len(samples)
            self._emitted += len(samples)
            return list(samples)
        self._held.extend(samples)
        self._received += len(samples)
        found = self._emit(self._received - self._half - 1)
        keep = self._read_from(self._emitted)
        if keep > self._base:
            self._held = self._held[keep - self._base:]
            self._base = keep
        return found

    def flush(self) -> list:
        """The tail, read against the end of the stream rather than against more
        input that is never coming."""
        if self.transparent:
            self._held = []
            return []
        found = self._emit(self._received)
        self._held = []
        return found

    def pending(self) -> int:
        """How many output samples the input received could still become."""
        return max(0, deterministic_target(self._received, self.rate_in, self.rate_out)
                   - self._emitted)

    def _read_from(self, index: int) -> int:
        """The earliest input sample output ``index`` will read."""
        return max(0, (index * self.rate_in) // self.rate_out - self._half)

    def _emit(self, supported: int) -> list:
        wanted = deterministic_target(min(supported, self._received),
                                      self.rate_in, self.rate_out)
        if wanted <= self._emitted:
            return []
        found = []
        last = self._received - 1
        for index in range(self._emitted, wanted):
            number = index * self.rate_in
            centre = number // self.rate_out
            position = centre + (number % self.rate_out) / float(self.rate_out)
            total = 0.0
            weight = 0.0
            for offset in range(centre - self._half + 1, centre + self._half + 1):
                distance = (position - offset) * self._cutoff
                tap = _sinc(distance) * _blackman(distance, self._half)
                if tap == 0.0:
                    continue
                taken = 0 if offset < 0 else (last if offset > last else offset)
                total += self._held[taken - self._base] * tap
                weight += tap
            found.append(total / weight if weight else 0.0)
        self._emitted = wanted
        return found


def crossfade_ramp(count: int) -> list:
    """A raised cosine from 0 to 1, ``count`` samples long.

    Amplitude-complementary rather than equal-power, and that is a measurement
    rather than a preference. The two things being crossfaded here are two model
    outputs *of the same audio*, so they are almost perfectly correlated: under
    an equal-power pair the sum in the middle is ``a·√0.5 + a·√0.5 = 1.41a``,
    which is a 3 dB bump at the cadence of the analysis window -- exactly the
    periodic pumping section 21.11 says to reject. Complementary ramps sum to
    one and leave correlated content alone.
    """
    if count <= 0:
        return []
    if count == 1:
        return [0.5]
    return [0.5 - 0.5 * math.cos(math.pi * index / (count - 1)) for index in range(count)]


# --------------------------------------------------------------------------- #
# The stages
# --------------------------------------------------------------------------- #


class DpdfStage:
    """DPDFNet over one turn, with the packet boundaries taken out.

    The backend is stateful and hop-based: upstream documents roughly one 20 ms
    model window before the first enhanced sample and about 10 ms per hop after
    it. So this stage owes the caller samples it has not produced yet for the
    length of one window, and pays them back at :meth:`flush`. That debt is
    tracked rather than hidden, because "the denoiser is one window behind" and
    "the denoiser lost 20 ms of the reply" are the same arithmetic seen from two
    sides and only one of them is acceptable.

    Rate is preserved (section 8.3, case B): the adapter is handed the real
    source rate, whatever it is, and upstream resamples internally and returns
    output at the caller's rate. Passing 48000 because the chosen network is a
    48 kHz variant would be passing a lie about the caller's clock.
    """

    id = "dpdfnet"

    def __init__(self, backend, rate: int):
        self.backend = backend
        self.compute = 0.0
        """Wall time inside this stage alone.

        Per stage rather than per chain, because "the pipeline is too slow" and
        "LavaSR is too slow" are different findings and only the second one
        tells anybody what to switch off (section 17.2)."""

        self.rate = int(rate)
        self.block = int(getattr(backend, "block_samples", 0) or 0)
        """How much this backend wants at a time, or zero for "anything".

        Zero is the real DPDFNet's answer: upstream's StreamEnhancer keeps its
        own input buffer and takes arbitrary chunk sizes, so blocking here would
        be a second buffer in front of a buffer, adding latency to make a
        library's own job easier. A stand-in that wants fixed blocks says so and
        gets them."""
        self._held = []
        self.taken = 0
        self.given = 0
        self.correction = 0
        """Samples this stage had to invent at the flush to pay its own debt."""

        self.backend_taken = 0
        self.backend_given = 0
        """What the model was fed and what it gave back, cumulatively.

        The difference is the model's own latency and is expected to settle: a
        stage one window behind stays one window behind for the whole reply. A
        difference that keeps growing is a backend losing audio, which the flush
        would otherwise paper over with silence and nobody would hear as
        anything but a slightly quieter ending.
        """

    def feed(self, samples) -> list:
        """Every whole block this stage can now run, and nothing else.

        The residue stays in ``_held`` across the call, which is the whole of
        partition invariance for this stage: what is run depends on how many
        samples have arrived in total, never on where the caller drew the line
        between two of them.
        """
        started = time.monotonic()
        self.taken += len(samples)
        found = []
        if self.block <= 0:
            found = self._prime(list(samples))
        else:
            self._held.extend(samples)
            while len(self._held) >= self.block:
                piece, self._held = self._held[:self.block], self._held[self.block:]
                found.extend(self._prime(piece))
        self.given += len(found)
        self.compute += time.monotonic() - started
        return found

    @property
    def debt(self) -> int:
        """How far behind the model is, in samples it has not given back yet."""
        return self.backend_taken - self.backend_given

    def flush(self) -> list:
        """The debt, paid at the end of the turn, however deep the model is.

        Primed with silence until the model has actually handed back what it is
        holding, rather than called once and the remainder padded. The
        difference is audible and only on some backends: a stage a single block
        behind is paid off by one call, and a stage four blocks behind would
        have had three blocks of digital silence at the end of every reply --
        the last quarter-second of speech replaced by nothing, on a path whose
        sample count is exactly right and whose test therefore passes.

        The padding fed in is inside the inference and never inside the
        duration, which is the distinction section 8.8 draws: the result is cut
        back to exactly what was owed.
        """
        started = time.monotonic()
        owed = self.taken - self.given
        if owed <= 0:
            self._held = []
            return []
        tail, self._held = list(self._held), []
        found = []
        drain = getattr(self.backend, "flush", None)
        if drain is not None:
            # The backend knows how to end its own stream, which is better than
            # being fed silence until it lets go: upstream's flush zero-pads to
            # one window and then trims the result back to the samples that came
            # from real input, which is exactly the distinction section 8.8
            # draws between padding inside an inference and padding inside a
            # duration.
            if tail:
                found.extend(self._prime(tail))
            heard = list(drain())
            self.backend_given += len(heard)
            found.extend(heard)
        else:
            block = self.block or max(1, self.rate // 100)
            if tail:
                found.extend(self._prime(tail + [0.0] * ((-len(tail)) % block)))
            # Bounded by what is owed rather than by a constant: a model cannot
            # be holding more blocks than it has been given, so this cannot run
            # away even against a backend that has stopped answering. The margin
            # is for one whose first flush call returns nothing at all.
            # Twice what the arithmetic needs plus a margin: a backend that
            # resamples internally gives back its debt at *its* rate, not the
            # caller's, so the number of primes needed is the ratio between them
            # and not a constant this loop can know.
            for _ in range((owed // block) * 2 + 8):
                if len(found) >= owed:
                    break
                found.extend(self._prime([0.0] * block))
        if len(found) < owed:
            self.correction += owed - len(found)
            found = found + [0.0] * (owed - len(found))
        found = found[:owed]
        self.given += len(found)
        self.compute += time.monotonic() - started
        return found

    def _prime(self, fed) -> list:
        """One backend call, counted on both sides of the ledger."""
        heard = list(self.backend.enhance(fed))
        self.backend_taken += len(fed)
        self.backend_given += len(heard)
        return heard


class LavaStage:
    """LavaSR over one turn, as a rolling window rather than a packet converter.

    Never ``concat(lava(packet_0), lava(packet_1), …)`` (I-VP-12). Each
    inference sees an analysis window plus context on both sides, the context is
    discarded as unreliable, and consecutive kept regions overlap and are ramped
    into one another. The scheme in full, in source samples:

        window i reads   [i·H − C, i·H + A + C)
        window i keeps   [i·H,     i·H + A)
        window i commits [i·H,     i·H + H)      H = A − O
        window i carries [i·H + H, i·H + A)      into window i+1's head

    so the committed stream advances by one hop per window and every place where
    two windows meet is a ramp rather than a join. Only the overlap itself is
    produced twice -- ``O`` of every ``H`` samples, twenty milliseconds in every
    hop at the current numbers -- which is the point of choosing a crossfade
    over a hard concatenation and not, as an earlier draft of this docstring
    said, a property of every sample in the stream.

    The rate contract is the adapter's and it is explicit (section 8.4, 10.3).
    Source PCM is resampled to ``backend_rate`` -- the rate the selected backend
    actually interprets its tensors at, which Phase 0 measured rather than read
    off a README -- the model returns 48 kHz, and the count committed for each
    hop is the difference of two cumulative deterministic targets. Nothing here
    trusts a float ratio to keep a duration.
    """

    id = "lavasr"

    def __init__(self, backend, rate_in: int, analysis_ms: int, context_ms: int,
                 backend_rate: int, numpy_module=None, rate_out: int = LAVA_OUTPUT_RATE):
        self.backend = backend
        self.rate_in = int(rate_in)
        self.rate_out = int(rate_out)
        self.backend_rate = int(backend_rate)
        if self.backend_rate <= 0:
            raise Refusal("LavaSR was given no measured input rate to work at")
        if analysis_ms <= 0 or context_ms <= 0:
            raise Refusal("LavaSR was given no measured analysis window")

        self.analysis = max(1, self.rate_in * int(analysis_ms) // 1000)
        self.context = max(1, self.rate_in * int(context_ms) // 1000)
        self.overlap = max(1, min(self.analysis // 2,
                                  self.rate_in * CROSSFADE_MS // 1000))
        self.hop = self.analysis - self.overlap
        if self.hop <= 0:
            raise Refusal("LavaSR's analysis window is shorter than its own join")

        self.compute = 0.0
        """Wall time inside this stage alone, for the reason DPDFNet's is."""

        # A frame is a property of the model, not of the window it was handed.
        # So the bound is taken once, from a *full* window, and applied to every
        # window including the short one a flush ends on. Judging each window
        # against its own length would forgive a constant loss on the full ones
        # and refuse the identical loss on the last one, which is how a machine
        # whose model has a larger hop than this one would fail an install for a
        # reason that has nothing to do with its clock.
        self._framing_bound = deterministic_target(
            self.analysis + 2 * self.context, self.rate_in, self.rate_out) // FRAMING_LIMIT

        self._down = Resampler(self.rate_in, self.backend_rate, numpy_module)
        self._ramp = crossfade_ramp(deterministic_target(self.overlap, self.rate_in,
                                                         self.rate_out))
        # Absolute source positions throughout, with ``_base`` naming which one
        # ``_held[0]`` is. Absolute rather than relative because every position
        # here is an argument to :func:`deterministic_target`, and a target
        # computed from an index into a buffer somebody trimmed is a duration
        # that changes when the buffer is trimmed.
        self._held = []            # source samples still needed, from ``_base``
        self._base = 0
        self._received = 0         # source samples handed to this stage
        self._start = 0            # where the next window's kept region begins
        self._tail = []            # last window's carried overlap, at rate_out
        self.consumed = 0          # source samples whose output has been committed
        self.emitted = 0           # output samples committed
        self.windows = 0
        self.correction = 0
        self.framing = 0
        """The largest bounded shortfall the backend has withheld per window.

        Its own analysis frame, kept apart from :attr:`correction` because the
        two mean opposite things about an installation. See
        :data:`FRAMING_LIMIT`. Reported rather than merely tolerated: a model
        that grows a frame between builds should be visible, not silent.
        """

    @property
    def first_output_samples(self) -> int:
        """How many source samples must arrive before anything can be finalised.

        The feature's intrinsic latency, in one place so it can be reported
        rather than guessed at (section 11.6). It is the analysis window plus
        the trailing context, and it is the reason section 10.7 asks for the
        smallest window whose quality is acceptable instead of a round number.
        """
        return self.analysis + self.context

    @property
    def held_samples(self) -> int:
        """How much source this stage is holding, which a bound is kept on.

        One analysis window plus context either side and never more, whatever
        the reply's length: a thirty-minute turn holds the same few hundred
        milliseconds a one-second turn does (I-VP-16). The alternative -- keeping
        the turn's whole source so that positions stay simple -- is a buffer
        that grows for as long as somebody keeps talking, which is a memory leak
        with a good explanation.
        """
        return len(self._held)

    def feed(self, samples) -> list:
        """Run every window that can now be finalised, and commit their hops."""
        started = time.monotonic()
        self._held.extend(samples)
        self._received += len(samples)
        found = []
        while self._received >= self._start + self.analysis + self.context:
            found.extend(self._window(self._start + self.analysis, final=False))
        self._compact()
        self.compute += time.monotonic() - started
        return found

    def _compact(self) -> None:
        """Forget the source that no future window can read.

        The earliest sample any window will still ask for is the context before
        the next kept region. Everything before that has been through both of
        the two inferences it will ever be in, and is gone.
        """
        keep_from = max(0, self._start - self.context)
        if keep_from > self._base:
            self._held = self._held[keep_from - self._base:]
            self._base = keep_from

    def _span(self, start: int, stop: int) -> list:
        """Absolute source range ``[start, stop)``, as samples."""
        return self._held[max(0, start - self._base):max(0, stop - self._base)]

    def flush(self, total_in: int) -> list:
        """Finish the turn: the last partial window, then the exact remainder.

        ``total_in`` is the parent's own count of what it sent, and it is the
        authority. Everything committed before this has been a difference of two
        targets; this makes the last difference reach ``target(total_in)``
        exactly, so a turn cannot end a sample short of its own duration however
        the windows fell (section 8.8).

        A short reply -- "Yes." -- is the case this exists for. It is shorter
        than one analysis window, so no window ever became runnable, and
        everything it has is finalised here in one pass with deterministic
        padding rather than held for a window that will never fill (section
        26.1).
        """
        started = time.monotonic()
        while self._start < self._received:
            self._window(self._received, final=True)
        found = []
        if self._tail:
            found.extend(self._tail)
            self._tail = []
        self.consumed = self._received
        self._held = []

        wanted = deterministic_target(int(total_in), self.rate_in, self.rate_out)
        committed = self.emitted + len(found)
        if committed > wanted:
            # Deterministic context excess, trimmed to target. Normal and small:
            # section 26.17 says a large correction here means the adapter is
            # wrong and must not be normalised away, which is why it is counted.
            self.correction += committed - wanted
            found = found[:len(found) - (committed - wanted)]
        elif committed < wanted:
            self.correction += wanted - committed
            edge = found[-1] if found else 0.0
            found.extend([edge] * (wanted - committed))
        self.emitted += len(found)
        self.compute += time.monotonic() - started
        return found

    def _window(self, keep_end: int, final: bool) -> list:
        """One inference, one crossfade, one hop committed."""
        keep_start = self._start
        read_from = max(0, keep_start - self.context)
        read_to = min(self._received, keep_end + self.context)
        span = self._span(read_from, read_to)

        # Left padding is real context when there is any and zeros only at the
        # very start of a turn, where there genuinely is nothing before the
        # first sample. Never a sample from another turn (I-VP-13).
        lead = self.context - (keep_start - read_from)
        if lead > 0:
            span = [0.0] * lead + span
        trail = self.context - (read_to - keep_end)
        if trail > 0:
            span = span + [0.0] * trail

        heard = self.backend.enhance(self._down(span), self.backend_rate)

        # Where the kept region sits inside this window's own output, in output
        # samples, by the same cumulative arithmetic used everywhere else.
        head = deterministic_target(self.context, self.rate_in, self.rate_out)
        length = (deterministic_target(keep_end, self.rate_in, self.rate_out)
                  - deterministic_target(keep_start, self.rate_in, self.rate_out))
        kept = self._fit(heard, head, length,
                         deterministic_target(len(span), self.rate_in, self.rate_out))
        self.windows += 1

        ramp = self._ramp if len(self._ramp) <= len(kept) else crossfade_ramp(len(kept))
        joined = []
        if self._tail:
            width = min(len(self._tail), len(kept), len(ramp))
            for index in range(width):
                weight = ramp[index]
                joined.append(self._tail[index] * (1.0 - weight) + kept[index] * weight)
            joined.extend(self._tail[width:])
            rest = kept[width:]
        else:
            rest = kept

        if final:
            self._tail = joined + rest
            self._start = keep_end
            return []

        # How much of this window is final, as a difference of two cumulative
        # targets rather than as "everything but one overlap's worth". The two
        # agree whenever the rate ratio is a whole number and disagree by a
        # sample here and there when it is not, and taking the second answer
        # would let the committed stream walk away from the clock a sample at a
        # time -- which the end-of-turn reconciliation would then have to put
        # back all at once.
        whole = joined + rest
        self._start = keep_end - self.overlap
        wanted = (deterministic_target(self._start, self.rate_in, self.rate_out)
                  - self.emitted)
        if wanted <= 0:
            self._tail = whole
            return []
        if wanted >= len(whole):
            wanted = len(whole)
        commit, self._tail = whole[:wanted], whole[wanted:]
        self.consumed = self._start
        self.emitted += len(commit)
        return commit

    def _fit(self, heard, head: int, length: int, whole: int) -> list:
        """The kept region of one window's output, at exactly ``length`` samples.

        ``whole`` is what the backend should have returned for the *entire*
        window it was given, guard regions included, and it is what the length
        is judged against. Judging against the kept part instead would count the
        trailing guard -- which the scheme discards on purpose -- as a fault on
        every single window.

        Both directions are counted, and the over-long one is the one that
        matters. A backend that interprets 24 kHz samples as 16 kHz returns half
        again as much audio for every window; slicing that to length hides it
        behind a perfectly exact duration and leaves the speech time-compressed,
        which is exactly the failure section 10.3 says must not be assumed away.
        Two samples of slack, because converting a rate in two steps and in one
        can disagree in the last place and that is arithmetic rather than a
        fault.
        """
        found = list(heard)
        drift = abs(len(found) - whole)
        if drift > WINDOW_SLACK:
            # Which of the two things a short window is. A frame the model
            # cannot emit is bounded by one frame and is discarded with the
            # trailing context; a clock that disagrees is a third of the window
            # or more and is audible as speech at the wrong speed. Summing them
            # together is what refused a stage whose audio was exact.
            #
            # Against the stage's own full window rather than this one's, so a
            # flush's short final window forgives the same absolute frame the
            # full windows before it did.
            if drift <= self._framing_bound:
                self.framing = max(self.framing, drift)
            else:
                self.correction += drift
        wanted = head + length
        if len(found) < wanted:
            found = found + [found[-1] if found else 0.0] * (wanted - len(found))
        return found[head:wanted]


# --------------------------------------------------------------------------- #
# One turn
# --------------------------------------------------------------------------- #


class Turn:
    """One VoiceTurn's worth of signal state, and none of the models.

    Everything here is built at ``TURN_BEGIN`` and thrown away at ``TURN_END``
    or ``TURN_CANCEL`` (I-VP-13). The heavy objects -- the sessions, the weights
    -- are the worker's and stay warm across turns (I-VP-14), because loading
    them per reply would put the whole cold start in front of every sentence.
    """

    def __init__(self, identifier: str, rate_in: int, stages, cancelled=False):
        self.id = identifier
        self.rate_in = int(rate_in)
        self.stages = tuple(stages)
        self.rate_out = int(rate_in)
        for stage in self.stages:
            if isinstance(stage, LavaStage):
                self.rate_out = stage.rate_out
        self.input_samples = 0
        self.output_samples = 0
        self.input_sequence = 0
        self.output_sequence = 0
        self.cancelled = cancelled
        self.began = time.monotonic()
        self.first_input = 0.0
        self.first_output = 0.0
        self.compute = 0.0

    def feed(self, samples) -> list:
        """Source samples in, finalised output samples out, in stage order."""
        if not self.first_input:
            self.first_input = time.monotonic()
        self.input_samples += len(samples)
        if self.input_samples > MAX_TURN_SAMPLES:
            raise Refusal("that reply is longer than this worker will process")
        started = time.monotonic()
        found = list(samples)
        for stage in self.stages:
            found = stage.feed(found)
            if not found:
                break
        self.compute += time.monotonic() - started
        if found and not self.first_output:
            self.first_output = time.monotonic()
        self.output_samples += len(found)
        return found

    def flush(self) -> list:
        """The end of the turn, through every stage in order, exactly once each.

        Each stage is fed whatever the stage before it had been holding, and
        then finishes. That order is the reason a turn's last few milliseconds
        survive at all: DPDFNet is a window behind by construction, so its debt
        has to become LavaSR's input *before* LavaSR is told the turn is over,
        or the reply loses its final word to an accounting order nobody would
        ever hear as an accounting order.
        """
        started = time.monotonic()
        carried = []
        for stage in self.stages:
            if carried:
                carried = list(stage.feed(carried))
            if isinstance(stage, LavaStage):
                # Told the parent's own total rather than asked to remember one:
                # this stage owns the turn's output duration and the parent is
                # the authority on its input's. Everything upstream preserves
                # sample count, so the two are the same number.
                carried = carried + list(stage.flush(self.input_samples))
            else:
                carried = carried + list(stage.flush())
        self.compute += time.monotonic() - started
        if carried and not self.first_output:
            self.first_output = time.monotonic()
        self.output_samples += len(carried)
        return carried

    def expected_output(self) -> int:
        return deterministic_target(self.input_samples, self.rate_in, self.rate_out)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _wanted_provider(config: dict) -> tuple:
    """The execution provider and adapter number this stage was told to use.

    Defaults to the processor for anything the parent did not say, which is
    every load message written before this existed and every stage whose
    placement has never been touched. A load message that names an unknown
    provider is refused rather than quietly run on the CPU: the parent had a
    reason to send it, and running somewhere else while reporting success is how
    a placement setting becomes a placement setting nobody can trust.
    """
    provider = str(config.get("provider") or PROVIDER_CPU)
    if provider not in (PROVIDER_CPU, PROVIDER_DIRECTML):
        raise Refusal("the parent asked for an execution provider this runtime does "
                      "not carry")
    try:
        adapter = max(0, int(config.get("adapter", 0)))
    except (TypeError, ValueError):
        adapter = 0
    return provider, adapter


def _session_options(onnxruntime, threads, inter, provider: str):
    """Session options for one stage, on the provider it was told to use.

    The two DirectML lines are not tuning. ONNX Runtime's own documentation
    requires the memory pattern optimiser off and sequential execution for the
    DirectML execution provider, because that provider allocates through
    Direct3D and the pattern optimiser assumes it owns the arena; a session
    built without them is a session that either fails to create or produces
    wrong output, and neither is something to discover mid-sentence.

    The thread counts are still set for DirectML even though the work is on the
    card. They cost nothing there and they are what the CPU fallback nodes --
    the operators the provider does not implement -- run on.
    """
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = max(1, int(threads))
    options.inter_op_num_threads = max(1, int(inter))
    options.log_severity_level = 3
    if provider == PROVIDER_DIRECTML:
        options.enable_mem_pattern = False
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    return options


def _provider_argument(provider: str, adapter: int) -> list:
    """The ``providers=`` list for one placement.

    The CPU entry stays on the end of the DirectML list on purpose: DirectML
    does not implement every operator, and without a fallback a model with one
    unsupported node fails to load rather than running that node on the
    processor. What must not be silent is the *whole* session landing there,
    and that is what :func:`_check_provider` refuses.
    """
    if provider == PROVIDER_DIRECTML:
        return [(PROVIDER_DIRECTML, {"device_id": int(adapter)}), PROVIDER_CPU]
    return [PROVIDER_CPU]


def _check_provider(session, provider: str) -> tuple:
    """Refuse a session that did not come back on the provider that was asked for.

    ONNX Runtime does not raise when an execution provider is unavailable. It
    drops it and builds on the next one in the list, and the session it returns
    works -- so a machine with no usable Direct3D 12 card would have run every
    stage on the processor while the settings panel said "graphics card", which
    is the failure this whole check exists to make impossible.

    The test is on the *first* provider rather than on the set. A DirectML
    session legitimately reports the CPU provider as well, as the fallback for
    operators DirectML does not implement; a session that has fallen back
    entirely reports the CPU provider first, and alone.
    """
    got = tuple(session.get_providers())
    if not got or got[0] != provider:
        raise Refusal(f"the enhancement runtime built this stage on "
                      f"{got[0] if got else 'no provider'} rather than the "
                      f"{provider} it was asked for")
    return got


class _Session:
    """One ONNX Runtime session over a local file, on the provider it was given.

    Constructed from a path this process was handed, never from a model name.
    Upstream's convenience loaders take a name and will fetch it; none of them
    is reachable from here, and the environment could not resolve one if they
    were (I-VP-22, section 9.6).
    """

    def __init__(self, path, threads: int, inter: int,
                 provider: str = PROVIDER_CPU, adapter: int = 0):
        import onnxruntime

        options = _session_options(onnxruntime, threads, inter, provider)
        self.session = onnxruntime.InferenceSession(
            str(path), sess_options=options,
            providers=_provider_argument(provider, adapter))
        self.providers = _check_provider(self.session, provider)


def _stage_backend(stage_id: str, root, config: dict, numpy_module):
    """The real backend for one stage, built from local files only.

    Split out so that everything above it can be exercised without a model. The
    adapters take a backend object; what makes one is this function on a machine
    with an installation, and a hand-written stub in a test.
    """
    if stage_id == "dpdfnet":
        return _DpdfBackend(root, config, numpy_module)
    if stage_id == "lavasr":
        return _LavaBackend(root, config, numpy_module)
    raise Refusal(f"unknown stage {stage_id!r}")


@contextlib.contextmanager
def _dpdf_session_budget(threads, inter, provider=PROVIDER_CPU, adapter=0):
    """Build DPDFNet's session with a thread budget instead of upstream's one.

    ``dpdfnet.onnx_backend.create_cpu_session`` hardcodes::

        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

    and ``StreamEnhancer`` reaches it through ``build_runtime_model``, so the
    budget this worker is given never reached the stage that needed it most. On
    a sixteen-thread machine DPDFNet was running on one core, and the measured
    result was a real-time factor of 2.38 -- the enhancement taking two and a
    half seconds of compute per second of speech, holding the source back for
    292 of a 300-second reply and starving playback fourteen times.

    ``build_runtime_model`` looks the constructor up on its own module globals
    when it is called rather than binding it at import, so replacing it around
    the construction is enough and nothing in the vendored package is edited.
    Restored in a ``finally`` because this process may build more than one
    stage, and a patch left in place would be this stage's budget silently
    applied to somebody else's session.

    The same seam carries the execution provider, and it has to: the function
    being replaced is named ``create_cpu_session`` and builds a CPU session
    unconditionally, so a stage placed on a graphics card would otherwise run on
    the processor no matter what the panel said. Both arguments arrive from the
    parent's load message and both are refused rather than adjusted -- a session
    that came back on a provider nobody asked for is not a slower session, it is
    a different answer to "where is this running".
    """
    from dpdfnet import onnx_backend

    import onnxruntime

    original = onnx_backend.create_cpu_session

    def opened(onnx_path):
        options = _session_options(onnxruntime, threads, inter, provider)
        session = onnxruntime.InferenceSession(
            str(onnx_path), sess_options=options,
            providers=_provider_argument(provider, adapter))
        built.extend(_check_provider(session, provider))
        return session

    built = []
    onnx_backend.create_cpu_session = opened
    try:
        # Yielded so the caller can record what the session came back on rather
        # than what it asked for. Those are the same thing when nothing is
        # wrong, and the only interesting case is the one where they are not.
        yield built
    finally:
        onnx_backend.create_cpu_session = original


class _DpdfBackend:
    """Upstream DPDFNet's own StreamEnhancer, over a file this parent verified.

    Upstream's rather than a reimplementation, and the ``onnx_path`` argument is
    why that is safe here: given an explicit path, ``resolve_model`` short-
    circuits every search and every download, so the convenience API that would
    otherwise fetch a model by name cannot reach the network at all (I-VP-22).
    The session upstream builds is CPU-only and single-threaded by its own
    construction. Both of those are replaced here -- see
    :func:`_dpdf_session_budget` for the thread budget and for the execution
    provider, neither of which upstream's constructor has a parameter for.

    The rate handed to :meth:`process` is the caller's real one -- 24000 from
    Pocket today -- and upstream resamples to the network's native rate and back
    internally, returning output at the rate it was given. That is the contract
    section 9.4 describes, read out of upstream 0.6.0's source rather than its
    README.

    ``block_samples`` is zero because StreamEnhancer keeps its own input buffer
    and accepts any chunk size; the recurrent state is per turn and lives in
    :meth:`reset`, while the session and the weights are the worker's and
    outlive every turn (I-VP-13, I-VP-14).
    """

    block_samples = 0

    def __init__(self, root, config: dict, numpy_module):
        from pathlib import Path

        from dpdfnet.stream import StreamEnhancer

        if numpy_module is None:
            raise Refusal("the enhancement runtime has no NumPy")
        self._np = numpy_module
        self.rate = 0
        wanted = str(config.get("model_file") or "")
        if not wanted:
            raise Refusal("no DPDFNet model file was named")
        path = Path(root) / wanted
        if not path.is_file():
            raise Refusal("the DPDFNet model file is not where it was said to be")
        provider, adapter = _wanted_provider(config)
        with _dpdf_session_budget(config.get("intraop", 2), config.get("interop", 1),
                                  provider, adapter) as built:
            self._enhancer = StreamEnhancer(model=str(config.get("model_id") or "dpdfnet2"),
                                            onnx_path=path, verbose=False)
        # What it came back on, not what it was asked for. StreamEnhancer builds
        # exactly one session, so an empty list here would mean upstream stopped
        # going through the constructor this seam replaces -- which is worth
        # refusing rather than reporting the CPU by default.
        if not built:
            raise Refusal("the enhancement runtime built this stage without a session "
                          "this worker could see")
        self.providers = tuple(built)
        self._model_rate = int(config.get("model_sample_rate") or 0) or None
        self._up = None
        self._down = None

    def reset(self, rate: int) -> None:
        self.rate = int(rate)
        self._enhancer.reset()
        native = self._model_rate
        if native and native != self.rate:
            # Resampled here, continuously, rather than by ``process`` per call.
            # Upstream resamples each chunk on its own, so the same audio cut
            # into 10 ms packets and into one packet comes back measurably
            # different -- about a fifth of full scale at the joins, which is a
            # packet boundary you can hear. Doing it once, with the read
            # position carried across calls, is what makes the stream depend on
            # the audio rather than on how it was delivered (I-VP-10).
            self._up = StreamResampler(self.rate, native, self._np)
            self._down = StreamResampler(native, self.rate, self._np)
        else:
            self._up = self._down = None

    def enhance(self, samples):
        if self._up is None:
            found = self._enhancer.process(
                self._np.asarray(samples, dtype=self._np.float32),
                sample_rate=self.rate or None)
            return found.tolist()
        native = self._up.feed(samples)
        if not native:
            return []
        # ``sample_rate=None`` means "the model's own rate", so upstream does no
        # resampling of its own and its per-call boundary behaviour never
        # arises.
        heard = self._enhancer.process(
            self._np.asarray(native, dtype=self._np.float32), sample_rate=None)
        return self._down.feed(heard.tolist())

    # No ``flush``, deliberately, and upstream 0.6.0 is the reason. Its
    # StreamEnhancer.flush() drains by calling
    # ``self.process(pad, sample_rate=self._model_sr)`` -- the *model's* rate --
    # while process() refuses a rate that differs from the one the stream was
    # opened at. So on any caller rate that is not the model's native one, and
    # 24 kHz into a 48 kHz network is exactly that, flush() raises
    # "Sample rate changed from 24000 to 48000" instead of draining.
    #
    # Without this method :class:`DpdfStage` drains the stage itself, by feeding
    # silence at the caller's own rate until the model has given back what it is
    # holding. That is the same thing flush() was doing -- zero-pad to a window,
    # keep what came from real input -- done at a rate the library will accept,
    # and the stage's own ledger is what trims it to the exact count either way.


class _LavaBackend:
    """LavaSR upstream, driven one analysis window at a time.

    PyTorch rather than ONNX, which is the correction to a real failure: this
    class used to open the stage's model file with ONNX Runtime, and LavaSR's
    weights are a PyTorch checkpoint, so a perfectly good installation died on
    ``InvalidProtobuf`` at the self-test. The name of the file was never the
    problem -- the runtime it was handed to was.

    The window scheme, the rates and the crossfade all belong to
    :class:`LavaStage` above. What lives here is only "one tensor in, one tensor
    out", so that a future ONNX export of the same model can replace this class
    without the stage noticing.
    """

    def __init__(self, root, config: dict, numpy_module):
        self._np = numpy_module
        self._device = _torch_device(config)
        try:
            import torch
            from LavaSR.enhancer.linkwitz_merge import FastLRMerge
            from LavaSR.model import LavaEnhance2
        except ImportError as exc:
            raise Refusal(f"the enhancement runtime has no PyTorch LavaSR in it ({exc})")

        self._torch = torch
        # A real directory, so upstream's snapshot_download() branch is never
        # taken: I-VP-21 says this process gets verified local paths and no
        # network, at load time as much as at inference time.
        self._model = LavaEnhance2(model_path=str(root), device=self._device)

        # The refiner upstream builds in load_audio(), which is the entry point
        # this adapter does not use. Its constructor default is a 4 kHz cutoff,
        # and load_audio() overrides it to half the input rate -- 8 kHz for the
        # 16 kHz LavaSR actually reads. Leaving the default would blend the
        # original and the upsampled bands an octave too low and throw away real
        # content between 4 and 8 kHz. Read out of upstream's own usage rather
        # than out of its constructor signature.
        # The same key the stage is built from, so the refiner's cutoff and the
        # rate the audio actually arrives at cannot drift apart. Refused rather
        # than defaulted: a cutoff guessed from a missing measurement is exactly
        # the kind of silent wrongness this stage is careful about elsewhere.
        backend_rate = int(config.get("backend_input_rate") or 0)
        if backend_rate <= 0:
            raise Refusal("LavaSR was given no measured input rate to work at")

        # Read rather than assumed, and it used to be assumed the other way.
        # enhance() was called with denoise=True while the manifest contract
        # said false and the settings panel printed "Internal denoise: off --
        # DPDFNet is the cleanup stage" as a fact. So the panel described a
        # pipeline nobody was running: LavaSR's own denoiser ran on every
        # window, after DPDFNet had already cleaned the same audio, and cost a
        # second inference per window to do it.
        self._denoise = bool(config.get("denoise"))
        self._model.bwe_model.lr_refiner = FastLRMerge(
            device=self._device, cutoff=backend_rate // 2, transition_bins=1024)
        self.providers = (self._device,)

    def reset(self, rate: int) -> None:
        return None

    def enhance(self, samples, rate: int):
        """One analysis window in at ``rate``, 48 kHz out.

        ``rate`` is the backend rate the stage resampled to and is not passed
        on: upstream's enhance() has 16000 written into it, which is the whole
        reason the stage measures a backend rate and converts before calling.
        """
        torch = self._torch
        block = self._np.asarray(samples, dtype=self._np.float32).reshape(1, -1)
        if not block.size:
            return []
        with torch.no_grad():
            wav = torch.from_numpy(block).to(self._device)
            heard = self._model.enhance(wav, enhance=True, denoise=self._denoise,
                                        batch=False)
            return heard.detach().to("cpu").float().reshape(-1).numpy().tolist()


def _torch_device(config: dict) -> str:
    """Where a PyTorch stage runs, as a device string torch understands.

    The same placement the ONNX stages take as an execution provider, said in
    the other library's vocabulary. ``adapter`` is a CUDA ordinal here and it is
    zero, which is a fact rather than a default: the parent masks
    ``CUDA_VISIBLE_DEVICES`` to the one card that was chosen, by UUID, so this
    process sees exactly one CUDA device however many the machine holds. Which
    physical card that is was decided at the other end, in the only namespace
    where it could be decided without guessing -- see
    :func:`mc_voice_pipeline.cuda_mask`.

    The bounds check below therefore guards a real condition rather than a
    theoretical one. It fires when the mask named a card the driver will not
    give this process -- a card that has been removed, or one another tenant has
    locked to a compute mode that excludes us -- and saying so beats running
    somewhere nobody asked for.

    A card that was asked for and is not there is refused rather than quietly
    swapped for the processor, for the reason the ONNX path gives: running
    somewhere else while reporting success is how a placement setting becomes
    one nobody can trust.
    """
    provider = str(config.get("provider") or PROVIDER_CPU)
    if provider == PROVIDER_CPU:
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise Refusal(f"the enhancement runtime has no PyTorch in it ({exc})")
    try:
        adapter = max(0, int(config.get("adapter", 0)))
    except (TypeError, ValueError):
        adapter = 0
    if not torch.cuda.is_available():
        raise Refusal("this stage was placed on a graphics card and the enhancement "
                      "runtime has no CUDA build of PyTorch in it")
    if adapter >= torch.cuda.device_count():
        raise Refusal(f"this stage was placed on graphics card {adapter} and PyTorch "
                      f"can see {torch.cuda.device_count()}")
    return f"cuda:{adapter}"


# --------------------------------------------------------------------------- #
# The worker
# --------------------------------------------------------------------------- #


def containment(parent_pid: int) -> str:
    """Ask the OS to end this process when the parent ends, and say which door.

    The same arrangement the other workers make, in the same place: on Linux
    ``PR_SET_PDEATHSIG`` plus a re-check of the parent, because the parent can
    die between the fork and this call and the signal would then never come. On
    Windows the job object is arranged and proved at the parent, and this only
    reports what it believes.
    """
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
                return "pipe"
        except Exception:
            return "pipe"
        if parent_pid and os.getppid() != parent_pid:
            raise SystemExit(0)
        return "pdeathsig"
    if os.name == "nt":
        return "job"
    return "pipe"


class Worker:
    """The message loop, the loaded stages, and at most one live turn.

    One turn at a time, because there is one speaking lane upstream and a second
    concurrent turn would be a second reply nobody asked for. A frame for a turn
    this worker does not have is dropped rather than answered, which is what
    makes a cancelled turn's late audio a no-op instead of a race (I-VP-31,
    section 16.4).
    """

    def __init__(self, output=None):
        self.output = output if output is not None else sys.stdout.buffer
        self.stages = {}
        self.configs = {}
        self.rate_contract = {}
        self.turn = None
        self.cancelled = set()
        self.numpy = _numpy()
        self.parent_death = "pipe"

    # -- sending ---------------------------------------------------------- #

    def send(self, header: dict, payload: bytes = b"") -> None:
        write_frame(self.output, header, payload)

    # -- loading ---------------------------------------------------------- #

    def load(self, header: dict) -> dict:
        """Build the sessions for the stages the parent asked for, and no others.

        A stage the parent did not name costs nothing here: no file is opened
        and no memory is taken (section 12.4). That is what makes "DPDFNet only"
        a real configuration rather than a full load with one stage skipped.
        """
        wanted = [str(name) for name in (header.get("stages") or ())]
        roots = dict(header.get("paths") or {})
        configs = dict(header.get("config") or {})
        for stage_id in wanted:
            if stage_id not in ("dpdfnet", "lavasr"):
                raise Refusal(f"unknown stage {stage_id!r}")
            if stage_id not in roots:
                raise Refusal(f"no local model directory was given for {stage_id!r}")
        self.stages = {}
        self.configs = {}
        for stage_id in wanted:
            config = dict(configs.get(stage_id) or {})
            self.stages[stage_id] = _stage_backend(stage_id, roots[stage_id], config,
                                                   self.numpy)
            self.configs[stage_id] = config
        return {
            "op": "ready",
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "loaded": sorted(self.stages, key=lambda name: 0 if name == "dpdfnet" else 1),
            "output_rate_policy": {"dpdfnet": "preserve", "lavasr": str(LAVA_OUTPUT_RATE)},
            "lavasr_backend_rate": int((configs.get("lavasr") or {}).get(
                "backend_input_rate") or 0),
            "device": self._device_word(),
            "providers": {name: list(getattr(backend, "providers", ()) or ())
                          for name, backend in self.stages.items()},
            "parent_death": self.parent_death,
        }

    def _device_word(self) -> str:
        """One word for where the loaded stages ended up, for the parent's log.

        ``cpu`` only when every stage is on the processor, which is what this
        field has always meant and what the parent's existing check reads. A
        mixture is reported as ``mixed`` rather than as either half, because a
        field that named one stage's device while another was somewhere else
        would be a field that is wrong in exactly the configuration somebody
        would be looking at it to understand. The per-stage truth is in
        ``providers`` beside it, which is the answer for anyone who needs more
        than a word.
        """
        seen = set()
        for backend in self.stages.values():
            found = tuple(getattr(backend, "providers", ()) or ())
            seen.add("cpu" if not found or found[0] == PROVIDER_CPU else "gpu")
        if not seen or seen == {"cpu"}:
            return "cpu"
        return "gpu" if seen == {"gpu"} else "mixed"

    # -- turns ------------------------------------------------------------ #

    def begin(self, header: dict) -> dict:
        """Freeze one turn onto the stages it named, and say what comes out.

        The output rate is answered here rather than discovered later because
        the parent has to put it in a response header before any audio exists
        (I-VP-07, section 8.5). A turn that could change its own rate halfway
        would be a turn the browser decoded at the wrong speed from the point it
        changed.
        """
        identifier = str(header.get("turn") or "")
        if not identifier:
            raise Refusal("a turn with no id")
        if self.turn is not None and self.turn.id == identifier:
            raise Refusal("that turn has already begun")
        if int(header.get("channels") or 1) != 1:
            raise Refusal("this build enhances mono speech only")
        if str(header.get("sample_format") or "pcm16") != "pcm16":
            raise Refusal("this build reads PCM16 only")
        rate = int(header.get("sample_rate") or 0)
        if rate <= 0:
            raise Refusal("a turn with no sample rate")

        wanted = [str(name) for name in (header.get("stages") or ())]
        for stage_id in wanted:
            if stage_id not in self.stages:
                raise Refusal(f"stage {stage_id!r} is not loaded")

        built = []
        for stage_id in sorted(wanted, key=lambda name: 0 if name == "dpdfnet" else 1):
            backend = self.stages[stage_id]
            config = self.configs.get(stage_id) or {}
            reset = getattr(backend, "reset", None)
            if reset is not None:
                reset(rate)
            if stage_id == "dpdfnet":
                built.append(DpdfStage(backend, rate))
            else:
                built.append(LavaStage(
                    backend, rate,
                    analysis_ms=int(config.get("analysis_ms") or 0),
                    context_ms=int(config.get("context_ms") or 0),
                    backend_rate=int(config.get("backend_input_rate") or 0),
                    numpy_module=self.numpy))

        self.turn = Turn(identifier, rate, built)
        self.cancelled.discard(identifier)
        return {"op": "turn_ready", "turn": identifier, "ok": True,
                "output_sample_rate": self.turn.rate_out,
                "first_output_samples": max(
                    (stage.first_output_samples for stage in built
                     if isinstance(stage, LavaStage)), default=0)}

    def audio(self, header: dict, payload: bytes) -> None:
        """One block of source PCM, or a reason it was not one.

        The offset check is the whole of I-VP-31 at this end. Packet numbers
        alone cannot tell a missing block from a differently-sized one, and a
        pipeline that silently enhanced a stream with a hole in it would produce
        audio that is exactly as long as it should be and wrong in the middle.
        """
        identifier = str(header.get("turn") or "")
        if identifier in self.cancelled:
            return
        turn = self.turn
        if turn is None or turn.id != identifier:
            raise Refusal("audio arrived for a turn that has not begun")
        offset = header.get("input_sample_offset")
        if offset is not None and int(offset) != turn.input_samples:
            raise Refusal("the source stream skipped or repeated samples")
        turn.input_sequence += 1
        found = turn.feed(to_floats(payload))
        if found:
            self._emit(turn, found)

    def end(self, header: dict) -> dict:
        """Finish the turn, prove its duration, and say what it cost."""
        identifier = str(header.get("turn") or "")
        turn = self.turn
        if turn is None or turn.id != identifier:
            raise Refusal("a turn ended that had not begun")
        declared = header.get("final_input_sample_count")
        if declared is not None and int(declared) != turn.input_samples:
            raise Refusal("the source's own count of what it sent disagrees with what "
                          "arrived")
        found = turn.flush()
        if found:
            self._emit(turn, found)
        wanted = turn.expected_output()
        metrics = self._metrics(turn)
        if turn.output_samples != wanted:
            # Never silently. A deficit is audio somebody will not hear and an
            # excess is audio nobody generated; either one is the clock contract
            # broken, and the parent ends the turn rather than playing it.
            self.turn = None
            raise Refusal("the enhanced audio is not the duration the source was")
        self.turn = None
        return {"op": "turn_flushed", "turn": identifier, "ok": True,
                "final_output_sample_count": turn.output_samples,
                "output_sample_rate": turn.rate_out, **metrics}

    def cancel(self, header: dict) -> None:
        """Drop the turn now, and every late frame for it afterwards.

        No flush and no tail. A cancelled turn's remaining analysis window is
        audio the listener has already stopped hearing, and emitting it would be
        speech arriving after silence (section 16.3).
        """
        identifier = str(header.get("turn") or "")
        self.cancelled.add(identifier)
        if len(self.cancelled) > 64:
            self.cancelled = set(list(self.cancelled)[-32:])
        if self.turn is not None and self.turn.id == identifier:
            self.turn = None

    def _emit(self, turn, samples) -> None:
        turn.output_sequence += 1
        offset = turn.output_samples - len(samples)
        self.send({"op": "audio_out", "turn": turn.id,
                   "output_sequence": turn.output_sequence,
                   "output_sample_offset": offset,
                   "output_sample_rate": turn.rate_out},
                  to_pcm16(samples))

    def _metrics(self, turn) -> dict:
        """Numbers, enums and durations. Never a sample and never a word."""
        seconds = turn.input_samples / float(turn.rate_in or 1)
        found = {
            "input_sample_count": turn.input_samples,
            "output_sample_count": turn.output_samples,
            "input_packet_count": turn.input_sequence,
            "output_packet_count": turn.output_sequence,
            "compute_ms": int(turn.compute * 1000),
            "rtf_milli": int(1000 * turn.compute / seconds) if seconds else 0,
            "first_output_ms": int((turn.first_output - turn.first_input) * 1000)
                               if turn.first_output and turn.first_input else 0,
        }
        for stage in turn.stages:
            found[f"{stage.id}_compute_ms"] = int(stage.compute * 1000)
            found[f"{stage.id}_rtf_milli"] = (int(1000 * stage.compute / seconds)
                                              if seconds else 0)
            if isinstance(stage, LavaStage):
                found["lava_window_count"] = stage.windows
                found["lava_correction_count"] = stage.correction
            else:
                found["dpdf_correction_count"] = stage.correction
        return found

    # -- the loop --------------------------------------------------------- #

    def serve(self, stream) -> None:
        """Read frames until the pipe ends. Every failure is reported, not raised.

        A malformed frame ends the loop -- the parent's own framing is the only
        thing that writes into this pipe, so a bad one means the pipe is not
        what it was -- but a refusal inside a turn is answered and the loop goes
        on. The difference matters: a turn that could not be enhanced should
        cost that turn, not the worker's residency and everything warm in it.
        """
        while True:
            try:
                frame = read_frame(stream)
            except (Refusal, ValueError, struct.error) as exc:
                _note(f"malformed frame: {_safe(exc)}")
                return
            if frame is None:
                return
            header, payload = frame
            operation = str(header.get("op") or "")
            if operation == "shutdown":
                return
            try:
                self._dispatch(operation, header, payload)
            except Exception as exc:  # noqa: BLE001 - reported, never fatal to the loop
                identifier = str(header.get("turn") or "")
                if identifier and self.turn is not None and self.turn.id == identifier:
                    self.turn = None
                    self.cancelled.add(identifier)
                self.send({"op": "error", "id": header.get("id"), "turn": identifier,
                           "ok": False, "error": _safe(exc)})

    def _dispatch(self, operation: str, header: dict, payload: bytes) -> None:
        if operation == "start":
            self.parent_death = containment(int(header.get("parent_pid") or 0))
            self.send({"op": "hello", "id": header.get("id"), "ok": True,
                       "protocol_version": PROTOCOL_VERSION,
                       "device": self._device_word(),
                       "backend": FEATURE,
                       "parent_death": self.parent_death,
                       "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
                       "numpy": bool(self.numpy)})
        elif operation == "load":
            self.send({**self.load(header), "id": header.get("id")})
        elif operation == "turn_begin":
            self.send({**self.begin(header), "id": header.get("id")})
        elif operation == "audio":
            self.audio(header, payload)
        elif operation == "turn_end":
            self.send({**self.end(header), "id": header.get("id")})
        elif operation == "turn_cancel":
            self.cancel(header)
        elif operation == "reconfigure":
            # Applied only between turns, which is the parent's rule and is
            # re-checked here: a graph that changed under a turn in flight would
            # change that turn's output rate halfway through it.
            if self.turn is not None:
                raise Refusal("the stage graph may not change during a turn")
            self.send({**self.load(header), "id": header.get("id")})
        elif operation == "ping":
            self.send({"op": "pong", "id": header.get("id"), "ok": True})
        else:
            raise Refusal(f"unknown operation {operation!r}")


FEATURE = "voice_pipeline"


def _numpy():
    """NumPy if the closure has it, else ``None`` and the slow honest path."""
    try:
        import numpy

        return numpy
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# The self-test the installer runs before it promotes anything
# --------------------------------------------------------------------------- #


def selftest(roots: dict, configs: dict) -> dict:
    """Prove the installation before it becomes the installation.

    Section 13.10, and the last two checks are the ones that exist because of
    LavaSR's README. It is not enough that a session loads and returns finite
    numbers: a backend that interprets 24 kHz samples as 16 kHz loads perfectly,
    returns perfectly finite numbers, and plays speech a third too slowly. So
    the test feeds a known duration at the real Pocket rate and refuses an
    installation whose answer is the wrong length.

    Needs no sound hardware and no network.
    """
    started = time.monotonic()
    found = {"ok": False, "provider": "cpu", "stages": {}}
    numpy_module = _numpy()
    rate_in = int(configs.get("test_rate") or 24000)
    seconds = 1.0
    total = int(rate_in * seconds)
    source = [0.2 * math.sin(2.0 * math.pi * 220.0 * index / rate_in)
              for index in range(total)]

    for stage_id in sorted(roots, key=lambda name: 0 if name == "dpdfnet" else 1):
        config = dict((configs.get(stage_id) or {}))
        backend = _stage_backend(stage_id, roots[stage_id], config, numpy_module)
        reset = getattr(backend, "reset", None)
        if reset is not None:
            reset(rate_in)
        if stage_id == "dpdfnet":
            stage = DpdfStage(backend, rate_in)
            # In two halves, and the second half is the measurement. A model's
            # own latency is a debt that settles -- one window behind stays one
            # window behind -- while a backend that loses audio owes more after
            # every block. One pass cannot tell those apart and two can.
            first = list(stage.feed(source[:total // 2]))
            settled = stage.debt
            second = list(stage.feed(source[total // 2:]))
            # Judged against almost nothing, because a settled debt is exactly
            # constant: this stage only ever calls the backend with whole
            # blocks, so a model with a fixed latency returns exactly one block
            # per call once it is warm and owes precisely what it owed before.
            # Any growth at all is audio going missing, however slowly -- a
            # sample a block is half a second lost over a five-minute reply.
            if stage.debt - settled > WINDOW_SLACK:
                raise Refusal(
                    f"{stage_id} fell {stage.debt - settled} samples further behind over "
                    f"half a second, so it is losing audio rather than delaying it")
            heard = first + second + list(stage.flush())
            wanted = total
        else:
            stage = LavaStage(backend, rate_in,
                              analysis_ms=int(config.get("analysis_ms") or 0),
                              context_ms=int(config.get("context_ms") or 0),
                              backend_rate=int(config.get("backend_input_rate") or 0),
                              numpy_module=numpy_module)
            heard = list(stage.feed(source)) + list(stage.flush(total))
            wanted = deterministic_target(total, rate_in, LAVA_OUTPUT_RATE)
        if len(heard) != wanted:
            raise Refusal(f"{stage_id} did not preserve the duration it was given")
        # The committed duration is always exact, because the adapter reconciles
        # it -- so the duration alone can never fail this test and the
        # reconciliation is what has to be read instead. A well-behaved backend
        # needs none of it; one whose clock is wrong needs it on every window,
        # and that is the whole of the Phase-0 rate check (sections 10.3, 26.17).
        if stage.correction > max(TOLERATED_CORRECTION, wanted // 100):
            raise Refusal(
                f"{stage_id} had to be corrected by {stage.correction} samples in one "
                f"second of audio, so the rate it works at is not the rate it was told")
        if any(value != value or value in (float("inf"), float("-inf")) for value in heard):
            raise Refusal(f"{stage_id} returned values that are not numbers")
        found["stages"][stage_id] = {"samples": len(heard),
                                     "correction": stage.correction,
                                     # Reported so the installer can say what a
                                     # model's own frame costs. Silence here
                                     # would make a frame that grew between
                                     # upstream builds indistinguishable from
                                     # one that never existed.
                                     "framing": getattr(stage, "framing", 0)}

    found["ok"] = True
    found["elapsed_ms"] = int((time.monotonic() - started) * 1000)
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(MARKER, action="store_true", dest="marker")
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--stage", action="append", default=[],
                        help="stage_id=/local/path, repeatable")
    parser.add_argument("--config", default="")
    found, _rest = parser.parse_known_args(argv)

    if found.selftest:
        roots = {}
        for entry in found.stage:
            name, _, path = str(entry).partition("=")
            if name and path:
                roots[name] = path
        configs = json.loads(found.config) if found.config else {}
        try:
            print(json.dumps(selftest(roots, configs)))
        except Exception as exc:  # noqa: BLE001 - the installer reads this line
            print(json.dumps({"ok": False, "error": _safe(exc)}))
            return 1
        return 0

    worker = Worker()
    worker.parent_death = containment(int(found.parent_pid or 0))
    worker.serve(sys.stdin.buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
