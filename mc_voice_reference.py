"""Reading a reference recording, once, for every engine that takes one.

Two of the three text-to-speech engines clone a voice from a recording somebody
supplies, and a third would make three. What arrives is whatever a phone, a
recorder or an audio editor produced -- 24-bit, 32-bit float, stereo, 44.1 or
48 kHz, and routinely 16-bit PCM wrapped in ``WAVE_FORMAT_EXTENSIBLE`` -- and
what every engine wants is canonical mono PCM16 at its own rate. That
conversion is the same work whichever engine asked for it.

So it lives here, once. Not because duplication is untidy, but because this
particular duplication would *drift*: the resampler below is band-limited with
a measured 102 dB of alias rejection, and ``tests/test_voice_sopro.py`` proves
that number against a swept tone. A second copy would be a second copy nobody
measured, and the failure it produces is not a crash -- it is every clone
sounding slightly worse than its reference, which is exactly the bug the
windowed-sinc path was written to correct.

What is *not* shared is the policy. How long a reference may be, how loud it
has to be, and what the engine is called in the sentence that refuses it are
:class:`Envelope`, supplied by the engine -- because Sopro documents five to
twenty seconds and Pocket's own window is shorter, and an engine whose
validation lived in another engine's module would be an engine that could not
change its own rules.

This module imports no engine, no runtime and no Torch. It is arithmetic over
bytes, it runs in the WebUI's own process, and NumPy is an optional accelerator
it never requires.
"""

from __future__ import annotations

import array
import logging
import math
import struct
import sys
from dataclasses import dataclass

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""


class ReferenceError(ValueError):
    """A recording that could not be used. Never fatal.

    The default an engine gets if it does not name one of its own. Every engine
    here does name one -- a refusal a user reads should be in that engine's
    voice -- so this is the fallback for a caller that has no error class to
    offer rather than the ordinary path.
    """


@dataclass(frozen=True)
class Envelope:
    """What one engine will accept, and what to call it when it refuses.

    Frozen and passed in rather than read from a module, so that this file has
    no opinion about which engines exist. Adding a fourth engine that clones
    from a recording is a fourth :class:`Envelope` and no change here.
    """

    engine: str
    minimum_seconds: float
    maximum_seconds: float
    maximum_bytes: int
    target_rate: int
    minimum_peak: float


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #



WAVE_PCM, WAVE_FLOAT, WAVE_EXTENSIBLE = 0x0001, 0x0003, 0xFFFE

ENCODING_NAMES = {0x0002: "ADPCM", 0x0006: "A-law", 0x0007: "u-law",
                  0x0011: "IMA ADPCM", 0x0031: "GSM", 0x0055: "MP3-in-WAV"}


def encoding_label(encoding: int, bits: int) -> str:
    """What a refusal should call the thing it is refusing.

    "This engine accepts uncompressed 16-bit PCM" told somebody what was wanted
    and not what they had, which is the half of the sentence that would have let
    them fix it.
    """
    if encoding == WAVE_PCM:
        return f"{bits}-bit PCM"
    if encoding == WAVE_FLOAT:
        return f"{bits}-bit floating point"
    named = ENCODING_NAMES.get(int(encoding))
    return named or f"encoding 0x{int(encoding):04X}"


def clipped(value: float) -> int:
    # NaN compares unequal to itself, and reaches here from a float WAV whose
    # producer wrote one. Silence is the only honest sample to put in its place.
    if value != value:
        return 0
    return max(-32768, min(32767, int(value * 32767.0)))


def decode_pcm16(body: bytes, encoding: int, bits: int):
    """Interleaved samples as signed 16-bit, whatever the file stored them as.

    A reference recording arrives from whatever produced it, and that is rarely
    plain 16-bit PCM. An editor that cleans up a clip writes 24-bit or 32-bit
    float as a matter of course, and a great many writers wrap even ordinary
    16-bit PCM in ``WAVE_FORMAT_EXTENSIBLE``. Refusing all of those -- as this
    did, for one user, on a file that was fine -- was refusing the recording
    somebody actually has over a property the rest of this function is about to
    normalise away. It already downmixes and resamples; narrowing a sample is
    the same kind of work and no more of a judgement call.

    ``None`` for an encoding that is genuinely not decodable here, which after
    this is a compressed one: those need a codec, and a codec is a dependency
    this module does not have and should not grow.
    """
    swap = sys.byteorder == "big"
    if encoding == WAVE_PCM and bits == 8:
        # The one unsigned width the format has, with 128 as silence.
        return array.array("h", [(value - 128) << 8 for value in body])
    if encoding == WAVE_PCM and bits == 16:
        found = array.array("h")
        found.frombytes(body[: len(body) - (len(body) % 2)])
        if swap:
            found.byteswap()
        return found
    if encoding == WAVE_PCM and bits == 24:
        usable = len(body) - (len(body) % 3)
        # The top two bytes of each little-endian triple, which is the 16-bit
        # sample: truncation rather than rounding, because a reference is about
        # to be peak-normalised and a half-LSB is not audible in it.
        return array.array("h", [
            int.from_bytes(body[index + 1:index + 3], "little", signed=True)
            for index in range(0, usable, 3)])
    if encoding == WAVE_PCM and bits == 32:
        found = array.array("i")
        found.frombytes(body[: len(body) - (len(body) % 4)])
        if swap:
            found.byteswap()
        return array.array("h", [value >> 16 for value in found])
    if encoding == WAVE_FLOAT and bits in (32, 64):
        found = array.array("f" if bits == 32 else "d")
        width = bits // 8
        found.frombytes(body[: len(body) - (len(body) % width)])
        if swap:
            found.byteswap()
        # Clamped rather than scaled to the peak: a float WAV is allowed to
        # exceed unity and a reference that did would otherwise wrap around.
        return array.array("h", [clipped(value) for value in found])
    return None


RESAMPLE_ZEROS = 24
RESAMPLE_ROLLOFF = 0.945
RESAMPLE_BETA = 8.6
RESAMPLE_BLOCK = 32768
"""The anti-aliasing filter, and why a resampler needs one at all.

What was here before was linear interpolation with no filter, and the damage it
did is not subtle. Measured on this build: a 15 kHz tone in a 48 kHz recording
came back at 9 kHz -- folded into the middle of the speech band -- at **0.0 dB**,
which is to say at full amplitude, entirely unattenuated. Everything a recorder
captured between 12 and 24 kHz was mirrored back down on top of the voice: mic
self-noise, room hiss, and above all sibilance, which is exactly the band where
"s" and "sh" and "t" live. Almost every recording anybody clones from is 44.1 or
48 kHz, so almost every clone was conditioned on a reference with a mirror image
of its own top end laid over it.

A cloned voice inherits that permanently, because the conditioning is computed
from these samples. It is the difference between a clone that sounds like the
speaker and one that sounds like the speaker through a broken microphone.

The replacement is an ordinary windowed-sinc: a Kaiser window at beta 8.6 over
24 zero crossings, with the cutoff pulled to 94.5% of the new Nyquist to leave a
transition band. Same measurement, same tone: **-102 dB**. A 1 kHz and 3 kHz
speech-band pair comes through at 0.00 dB, so nothing that should survive is
touched.

Numbers, not adjectives: ``tests/test_voice_sopro.py`` measures both.
"""

_KAISER_TABLE = None
_KAISER_POINTS = 4097


def bessel_i0(value: float) -> float:
    """The modified Bessel function of the first kind, order zero, by series.

    Written out rather than imported because this module runs in the WebUI
    process, where SciPy is not a dependency and NumPy is only an optional
    accelerator -- and the window has to be identical on both paths or the two
    would resample differently.
    """
    total, term, step = 1.0, 1.0, 1
    while step < 200:
        term *= (value / (2.0 * step)) ** 2
        total += term
        if term < 1e-12 * total:
            break
        step += 1
    return total


def kaiser_table():
    """The Kaiser window sampled on [-1, 1], built once.

    A table rather than a call per tap: the window is evaluated tens of millions
    of times in a twenty-second reference, and two Bessel series per tap is the
    difference between a resample that is imperceptible and one that is a
    visible pause in the interface.
    """
    global _KAISER_TABLE

    if _KAISER_TABLE is None:
        scale = bessel_i0(RESAMPLE_BETA)
        _KAISER_TABLE = [
            bessel_i0(RESAMPLE_BETA * math.sqrt(max(0.0, 1.0 - position * position)))
            / scale
            for position in (
                -1.0 + 2.0 * index / (_KAISER_POINTS - 1)
                for index in range(_KAISER_POINTS))
        ]
    return _KAISER_TABLE


def resample(samples, rate: int, target: int):
    """Band-limited resampling to ``target``, as PCM16 in an ``array``.

    NumPy when it is importable, which in Forge it always is; the pure-Python
    path exists so that this module never *requires* it, and narrows the filter
    because the same 24 zero crossings would take most of a minute in a loop.
    Even the narrow one is a different universe from no filter at all.
    """
    import array as _array

    if rate == target or not len(samples):
        return samples
    count = int(len(samples) * target / float(rate))
    if count <= 0:
        return _array.array("h")
    try:
        import numpy
    except Exception:  # noqa: BLE001 - an optional accelerator, never required
        return resample_slowly(samples, rate, target, count)
    return resample_with(numpy, samples, rate, target, count)


def filter_shape(rate: int, target: int, zeros: int):
    """Cutoff and half-width for a source-rate filter. Shared by both paths.

    The half-width widens by the decimation factor because the filter has to be
    band-limited to the *new* Nyquist while being applied at the *old* rate: a
    fixed number of taps would be a fixed fraction of the source spectrum, which
    is the wrong thing to hold constant.
    """
    down = max(1.0, rate / float(target))
    return 0.5 * RESAMPLE_ROLLOFF / down, int(math.ceil(zeros * down))


def resample_with(numpy, samples, rate: int, target: int, count: int):
    import array as _array

    source = numpy.asarray(samples, dtype=numpy.float64)
    cutoff, half = filter_shape(rate, target, RESAMPLE_ZEROS)
    taps = numpy.arange(-half, half + 1, dtype=numpy.float64)
    window = numpy.asarray(kaiser_table(), dtype=numpy.float64)
    positions = numpy.linspace(-1.0, 1.0, _KAISER_POINTS)
    padded = numpy.concatenate([numpy.zeros(half), source, numpy.zeros(half + 2)])
    out = numpy.empty(count, dtype=numpy.float64)
    step = rate / float(target)
    # Blocked, because the tap matrix is one row per output sample: twenty
    # seconds at 48 kHz against a 97-tap filter is 93 million doubles in one
    # allocation, and this runs in the WebUI's own process.
    for begin in range(0, count, RESAMPLE_BLOCK):
        end = min(count, begin + RESAMPLE_BLOCK)
        centre = numpy.arange(begin, end, dtype=numpy.float64) * step
        base = numpy.floor(centre).astype(numpy.int64)
        delta = taps[None, :] - (centre - base)[:, None]
        argument = 2.0 * cutoff * delta
        sinc = numpy.where(numpy.abs(argument) < 1e-9, 1.0,
                           numpy.sin(numpy.pi * argument)
                           / (numpy.pi * argument + 1e-30))
        weights = sinc * numpy.interp(numpy.clip(delta / half, -1.0, 1.0),
                                      positions, window)
        # Normalised per output sample so the passband is flat and DC is exact.
        weights /= weights.sum(axis=1, keepdims=True)
        index = base[:, None] + taps[None, :].astype(numpy.int64) + half
        out[begin:end] = (padded[index] * weights).sum(axis=1)
    clipped = numpy.clip(numpy.rint(out), -32768, 32767).astype(numpy.int16)
    return _array.array("h", clipped.tobytes())


def resample_slowly(samples, rate: int, target: int, count: int):
    """The same filter in a loop, narrowed so it finishes this decade."""
    import array as _array

    cutoff, half = filter_shape(rate, target, 8)
    window = kaiser_table()
    last = len(samples) - 1
    out = _array.array("h", bytes(2 * count))
    step = rate / float(target)
    for index in range(count):
        centre = index * step
        base = int(math.floor(centre))
        total, weight_sum = 0.0, 0.0
        for tap in range(-half, half + 1):
            delta = tap - (centre - base)
            argument = 2.0 * cutoff * delta
            if abs(argument) < 1e-9:
                sinc = 1.0
            else:
                sinc = math.sin(math.pi * argument) / (math.pi * argument)
            position = delta / half
            if position < -1.0:
                position = -1.0
            elif position > 1.0:
                position = 1.0
            shaped = window[int((position + 1.0) * 0.5 * (_KAISER_POINTS - 1))]
            weight = sinc * shaped
            at = base + tap
            if 0 <= at <= last:
                total += samples[at] * weight
            weight_sum += weight
        value = int(round(total / weight_sum)) if weight_sum else 0
        out[index] = max(-32768, min(32767, value))
    return out


# --------------------------------------------------------------------------- #
# The one entry point
# --------------------------------------------------------------------------- #


def normalize(data: bytes, envelope: Envelope, error=ReferenceError) -> tuple:
    """A recording as canonical mono PCM16 at ``envelope.target_rate``, or a
    refusal that says why.

    The checks are in the order that produces the most useful sentence: is it a
    WAV at all, is it a size this build accepts, is it an encoding this build
    decodes, is it long enough to condition on, is it short enough, and is there
    actually a voice in it.

    What it decodes is deliberately wider than what any engine wants, because
    the two are different questions. An engine wants mono PCM16 at one rate; a
    person has whatever their recorder or editor produced. This already
    downmixes and resamples, so narrowing a sample is the same kind of work --
    and refusing a good recording for its container was, for one user, the whole
    of "I could not create a voice".

    Its own decoder rather than :mod:`mc_voice_clone`'s, which is a boundary
    rather than duplication for its own sake: that module is Kokoro's Storytime
    path, with Storytime's three-to-a-hundred-and-twenty-second window and
    Storytime's error text.
    """
    if not data:
        raise error("No recording was received.")
    if len(data) > envelope.maximum_bytes:
        raise error("That recording is too large.")
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise error("That file is not a WAV recording.")

    offset, fmt, body = 12, None, b""
    fmt_start, fmt_size = 0, 0
    while offset + 8 <= len(data):
        name = data[offset:offset + 4]
        (size,) = struct.unpack_from("<I", data, offset + 4)
        start = offset + 8
        if start + size > len(data):
            raise error("That recording's header is malformed.")
        if name == b"fmt " and size >= 16:
            fmt = struct.unpack_from("<HHIIHH", data, start)
            fmt_start, fmt_size = start, size
        elif name == b"data":
            body = data[start:start + size]
        offset = start + size + (size % 2)
    if fmt is None or not body:
        raise error("That recording is not a complete WAV.")

    encoding, channels, rate, _bps, _align, bits = fmt
    if encoding == WAVE_EXTENSIBLE:
        # The wrapper a writer reaches for as soon as a file has more than two
        # channels or more than sixteen bits -- and, in practice, for plenty of
        # ordinary 16-bit stereo too. The real format tag is the first two bytes
        # of the SubFormat GUID, and reading it is the difference between
        # accepting a perfectly good recording and refusing it for its wrapper.
        if fmt_size < 40:
            raise error("That recording's header is malformed.")
        (encoding,) = struct.unpack_from("<H", data, fmt_start + 24)
    if channels not in (1, 2):
        raise error(f"{envelope.engine} accepts mono or stereo recordings.")
    if rate < 8000 or rate > 192000:
        raise error("That recording's sample rate is not one Voice Chat can use.")

    samples = decode_pcm16(body, int(encoding), int(bits))
    if samples is None:
        raise error(
            f"That recording is {encoding_label(int(encoding), int(bits))}, which "
            f"{envelope.engine} cannot read here. Save it as uncompressed PCM or "
            f"floating-point WAV.")
    if not len(samples):
        raise error("That recording is not a complete WAV.")

    if channels == 2:
        samples = array.array("h", [(samples[index] + samples[index + 1]) // 2
                                    for index in range(0, len(samples) - 1, 2)])
    seconds = len(samples) / float(rate or 1)
    if seconds < envelope.minimum_seconds:
        raise error(f"That recording is {seconds:.1f} seconds long. {envelope.engine} "
                    f"clones from {envelope.minimum_seconds:.0f} to "
                    f"{envelope.maximum_seconds:.0f} seconds of one clear speaker.")
    if seconds > envelope.maximum_seconds + 0.5:
        raise error(f"That recording is {seconds:.0f} seconds long and {envelope.engine} "
                    f"clones from up to {envelope.maximum_seconds:.0f} seconds. Record a "
                    f"shorter one.")
    peak = max((abs(value) for value in samples), default=0) / 32768.0
    if peak < envelope.minimum_peak:
        raise error("That recording is silent. Check that the right microphone is "
                    "selected and that it is not muted.")

    if rate != envelope.target_rate:
        samples = resample(samples, int(rate), envelope.target_rate)
    if sys.byteorder == "big":
        samples.byteswap()

    raw = samples.tobytes()
    rate_out = int(envelope.target_rate)
    header = (b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVEfmt "
              + struct.pack("<IHHIIHH", 16, 1, 1, rate_out, rate_out * 2, 2, 16)
              + b"data" + struct.pack("<I", len(raw)))
    return header + raw, len(raw) / float(rate_out * 2)
