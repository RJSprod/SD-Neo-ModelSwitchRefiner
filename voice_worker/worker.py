"""The Voice Chat sidecar: speech in, speech out, and nothing else at all.

This file runs in the isolated CPU runtime that ``mc_voice_models`` provisions,
under an interpreter that is not Forge's and cannot see Forge's packages. It
reads framed requests from stdin and writes framed responses to stdout. It has
no HTTP client, opens no socket, listens on no port, writes no file, and never
prints a transcript or a reply.

    uint32 big-endian header length
    that many bytes of UTF-8 JSON
    uint32 big-endian payload length
    that many raw bytes

Protocol 2 -- speech while the reply is still being written
-----------------------------------------------------------
Protocol 1 had one synchronous ``tts``: text in, a finished WAV out, and a
command loop that was blocked inside native ONNX code for as long as it took.
That is the shape section 13 of the design intent asks to be taken apart,
because it makes two things impossible at once -- audio cannot leave before the
last sentence is synthesised, and nothing can be *told* to stop while it is
running.

So this process is now three parts that share state and never share a blocking
call:

    the command loop     always reading stdin, so a cancel or a shutdown is
                         read while inference is running, never after it
    one inference lane   Whisper and Kokoro, serialized -- one worker, one
                         model at a time, which is what a CPU beside Forge can
                         actually afford
    one writer           the only thing that touches stdout, so frames from
                         the lane and frames from the command loop cannot
                         interleave halfway through a length prefix

Operations:

    init        the handshake. The parent sends the verified local model paths,
                the thread caps and the voice bank to load; this replies READY
                with the provider it actually got, which is what the parent
                checks before believing the runtime is CPU-only.
    stt         payload is a PCM16 mono RIFF/WAVE; the header of the reply
                carries the transcript.
    tts         payload is UTF-8 text; the payload of the reply is a WAV. Kept
                for Test playback and for an explicit non-streaming fallback.
    tts_begin   open a turn: an id, a validated numeric speaker and a delivery.
    tts_text    one immutable segment of that turn, already decided by the
                parent's segmenter. Never a raw token.
    tts_finish  no more segments are coming.
    tts_cancel  stop this turn now. Answered by ``tts_cancelled`` rather than
                by an ordinary reply, because the parent does not wait for it.
    shutdown    acknowledge and exit.

Frames out carry ``turn`` wherever they belong to a turn: ``tts_ready``,
``tts_audio``, ``tts_segment_done``, ``tts_done``, ``tts_cancelled`` and
``tts_error``. Section 24 -- an audio block that cannot say which reply it
belongs to is an audio block that can be played over the next one.

Delivery: what the model does, and what this file does
------------------------------------------------------
A turn and a ``tts`` both carry four numbers, and it is worth being exact about
which of them Kokoro has ever heard of.

``speed`` is the model's. It goes straight into ``generate`` and changes how the
model articulates. The other three are arithmetic this file does on the samples
that came back, and :class:`Delivery` is where they are held:

    ``pitch``     a frequency ratio. Synthesis runs at ``speed x pitch`` and
                  :class:`Resampler` then reads the result back at ``pitch``
                  samples per output sample -- which shifts every frequency by
                  the ratio and puts the duration back where ``speed`` asked
                  for it. Formants move too, so this reads as a
                  different-sized speaker rather than as a transposed one.
    ``gain``      a linear scalar folded into the PCM16 conversion, so it costs
                  nothing extra and cannot produce a second pass over the
                  samples.
    ``pause_ms``  silence emitted between segments, which is a property of a
                  streamed turn and does nothing on the single-call ``tts``
                  route -- there are no segment boundaries there to put it at.

All of it is skipped when the numbers are neutral: :attr:`Delivery.shapes` is
false for the default profile, and an installation that never touches these
controls runs the path it ran before they existed.

Cancellation, honestly
----------------------
sherpa's generation callback is handed each finished sentence batch and its
return value decides whether generation continues. That is where this file
cancels: the callback copies the samples, hands them to the writer, looks at
the turn's cancelled flag and returns 0 to stop. Nothing here interrupts a
native instruction mid-flight, and section 29 does not ask for that -- what it
asks for is that the *next* boundary is the last one, which is what returning 0
achieves.

Why a pipe and not a port
-------------------------
A port needs discovery, a local authentication story, a firewall prompt, and
one mistake away from binding to a LAN address. A pipe needs none of that, and
it carries the lifetime contract for free: when the parent disappears its handle
closes, the next read returns EOF, and this process exits. That is the portable
first layer of invariant I-7.

It is only the first layer, and this file says so honestly. EOF is observed at
the moment this process is *waiting to read*, and during a long synthesis it is
not waiting to read -- it is inside native ONNX code. So the operating system is
asked as well: on Linux ``PR_SET_PDEATHSIG`` with SIGKILL, installed before the
first request is served and immediately re-checked against the parent pid we
were told about, which closes the window where the parent dies during our own
start-up and leaves a child whose parent-death signal will now never fire. On
Windows the parent puts this process in a job object with KILL_ON_JOB_CLOSE
before it is given any work, which is the same guarantee from the other side.
:func:`_containment` reports which of those is in force, the parent refuses to
mark the runtime running without one, and ``tests/test_voice_shutdown.py`` kills
a parent mid-inference and asserts nothing survives.

Content and logs
----------------
Invariant I-6. Nothing in this file writes transcript text, reply text, or audio
to stderr, and the diagnostics it does write are model ids, byte counts,
durations and error classes. :func:`_safe` exists because an exception message
from a third-party library is not a string this file gets to assume is
content-free -- so the words that come back to the parent are chosen here.
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

PROTOCOL_VERSION = 2

MARKER = "--model-chain-voice-worker"
"""On the command line so this process is recognisable in a task manager and by
the stray sweep. Never a model name and never chat text: a command line is world
readable on most systems."""

MAX_HEADER = 1 << 20
MAX_PAYLOAD = 8 << 20
"""A 60-second 16 kHz mono PCM16 WAV is about 1.9 MB. Eight is generous and is
still a number rather than "whatever arrives"."""

SAMPLE_RATE = 16000
MAX_SECONDS = 60.0

_LENGTH = struct.Struct(">I")


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #


def read_frame(stream) -> tuple[dict, bytes] | None:
    """One request, or ``None`` at end of input.

    ``None`` is how the parent's death arrives when this process is waiting for
    work, and it is not an error: the loop ends, the models are released, and
    the process exits with 0.
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


def _read_exactly(stream, count: int) -> bytes | None:
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


def decode_wav(data: bytes) -> tuple[list, int]:
    """A validated PCM16 mono WAV as float samples in [-1, 1].

    Validated rather than trusted even though the route in front already
    checked: this is the process that would crash, and a worker that dies on a
    malformed upload is a worker the next dictation has to restart.
    """
    with contextlib_closing(wave.open(io.BytesIO(data), "rb")) as handle:
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
        if frames / float(rate or 1) > MAX_SECONDS + 1.0:
            raise ValueError("audio is too long")
        raw = handle.readframes(frames)

    import array

    samples = array.array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if sys.byteorder == "big":
        samples.byteswap()
    scale = 1.0 / 32768.0
    return [value * scale for value in samples], rate


def encode_wav(samples, rate: int) -> bytes:
    """Float samples as a PCM16 mono WAV, in memory.

    In memory is the requirement, not a preference: invariant I-5 says generated
    speech never becomes a file, so there is no path argument here and no
    tempfile import in this module.
    """
    import array

    clipped = array.array("h")
    for value in samples:
        scaled = int(float(value) * 32767.0)
        if scaled > 32767:
            scaled = 32767
        elif scaled < -32768:
            scaled = -32768
        clipped.append(scaled)
    if sys.byteorder == "big":
        clipped.byteswap()

    buffer = io.BytesIO()
    with contextlib_closing(wave.open(buffer, "wb")) as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(clipped.tobytes())
    return buffer.getvalue()


def contextlib_closing(thing):
    import contextlib

    return contextlib.closing(thing)


# --------------------------------------------------------------------------- #
# Dying with the parent
# --------------------------------------------------------------------------- #


def _containment(parent_pid: int) -> str:
    """Ask the OS to end this process when the parent ends. Reports what it got.

    On Linux ``PR_SET_PDEATHSIG`` is set to SIGKILL and then the parent pid is
    re-read. The re-read is the whole point: if the parent died between its fork
    and this line, the death signal it would have sent has already not been
    sent, and this process would sit here forever holding a gigabyte of Whisper.
    Seeing a different parent than the one we were told about is that race, and
    the answer is to exit immediately.

    On Windows the job object is the parent's to create and it has already done
    it by the time the handshake is answered; the honest word for what this side
    contributed is "job". Everywhere else there is only the pipe, the parent is
    told so, and the parent decides whether that platform is one it is willing
    to run on.
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
        return "job"
    return "pipe"


def _lower_priority() -> None:
    """Be the process that yields when an image is rendering. Never fatal."""
    try:
        if os.name == "nt":
            import ctypes

            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(5)
    except Exception:
        _note("could not lower the voice worker's priority")


# --------------------------------------------------------------------------- #
# The engines
# --------------------------------------------------------------------------- #


def _bounded(value, limits, fallback):
    """``value`` as a float inside ``limits``, or ``fallback``.

    Total on purpose. This is the boundary between a JSON header and a number
    that multiplies a synthesis rate, and every way that value can be wrong --
    absent, a string, a NaN, an infinity, negative -- means the same thing here.
    """
    low, high = limits
    try:
        found = float(value)
    except (TypeError, ValueError):
        return fallback
    if found != found or found in (float("inf"), float("-inf")):
        return fallback
    return min(max(found, low), high)


class Delivery:
    """The four numbers one synthesis is shaped by.

    Built from a frame header, which is the only place they come from, and
    bounded here as well as in the parent -- ``mc_voice_profile`` clamps them
    before they are sent, and this process does not take a caller's word for a
    multiplier it is about to hand to native code.
    """

    __slots__ = ("speed", "pitch", "gain", "pause_ms")

    SPEED = (0.25, 4.0)
    PITCH = (0.5, 2.0)
    GAIN = (0.05, 8.0)
    PAUSE = (0, 5000)

    def __init__(self, speed=1.0, pitch=1.0, gain=1.0, pause_ms=0):
        self.speed = _bounded(speed, self.SPEED, 1.0)
        self.pitch = _bounded(pitch, self.PITCH, 1.0)
        self.gain = _bounded(gain, self.GAIN, 1.0)
        self.pause_ms = int(_bounded(pause_ms, self.PAUSE, 0))

    @classmethod
    def from_header(cls, header: dict) -> "Delivery":
        found = header or {}
        return cls(speed=found.get("speed"), pitch=found.get("pitch"),
                   gain=found.get("gain"), pause_ms=found.get("pause_ms"))

    @property
    def generation_speed(self) -> float:
        """What ``generate`` is actually asked for.

        ``speed x pitch``, because the resampler below divides the duration by
        ``pitch`` again on the way out. Asking Kokoro for the user's speed and
        then resampling would produce the right pitch at the wrong length.
        """
        return _bounded(self.speed * self.pitch, self.SPEED, 1.0)

    @property
    def shapes(self) -> bool:
        """Whether anything at all has to be done to the samples."""
        return self.pitch != 1.0 or self.gain != 1.0

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"Delivery(speed={self.speed}, pitch={self.pitch}, gain={self.gain}, "
                f"pause_ms={self.pause_ms})")


NEUTRAL = Delivery()
"""Kokoro exactly as it comes out of the model. The default everywhere."""


class Resampler:
    """Linear resampling of a stream, with the state a stream needs.

    ``ratio`` is how many input samples advance per output sample, so a ratio
    above one shortens the audio and raises its pitch. Blocks arrive from
    sherpa's callback at whatever granularity it chooses, and an output sample
    routinely needs two input samples that arrived in different blocks -- so the
    unconsumed tail and the fractional read position are carried between calls
    rather than recomputed, which is what stops a click at every block boundary.

    One instance per segment. At most one input sample is left unread when a
    segment ends, which at 24 kHz is forty microseconds of the silence Kokoro
    already puts at the end of a sentence.

    Linear rather than windowed, for the reason ``javascript/voice_chat.js``
    gives about the microphone path: the artefacts a proper filter would remove
    sit far above what a 24 kHz speech signal shifted by a few semitones puts
    there, and a resampling dependency inside a two-wheel runtime is a
    dependency this feature has already decided not to have.
    """

    def __init__(self, ratio: float):
        self.ratio = float(ratio)
        self._tail = []
        self._base = 0.0
        self._position = 0.0

    def feed(self, samples) -> list:
        tail = self._tail
        tail.extend(samples)
        position, base, ratio = self._position, self._base, self.ratio
        limit = base + len(tail) - 1
        out = []
        while position <= limit:
            local = position - base
            low = int(local)
            fraction = local - low
            first = tail[low]
            out.append(first + (tail[low + 1] - first) * fraction if fraction else first)
            position += ratio
        # Everything before the next read position can never be read again.
        # Dropped here rather than at the end of the segment, or a long reply
        # would carry its whole first sentence in this list.
        spent = int(position - base)
        if spent > 0:
            del tail[:spent]
            base += spent
        self._position, self._base = position, base
        return out


class Shaper:
    """One segment's samples, pitched and levelled, as PCM16 bytes.

    Holds the resampler so that the pitch shift is continuous across the blocks
    of one segment, and folds the gain into the conversion so that the level
    costs no pass of its own. A neutral delivery makes this exactly
    :func:`pcm16`, which is the path an installation that has never opened these
    controls takes.
    """

    def __init__(self, delivery: "Delivery" = None):
        self.delivery = delivery or NEUTRAL
        self._resampler = (Resampler(self.delivery.pitch)
                           if self.delivery.pitch != 1.0 else None)

    def shaped(self, samples):
        """The samples at the requested pitch. Level is not applied here.

        Split from :meth:`block` because the streaming path wants PCM16 and the
        completed-audio path wants floats to hand to ``encode_wav``, and
        converting to bytes and back for the second one would be a round trip
        for nothing.
        """
        if self._resampler is None:
            return samples
        return self._resampler.feed(samples)

    def block(self, samples) -> bytes:
        """One block of a streamed segment, pitched and levelled, as PCM16."""
        return pcm16(self.shaped(samples), self.delivery.gain)

    def levelled(self, samples):
        """The samples at the requested pitch *and* level, still as floats."""
        found = self.shaped(samples)
        level = self.delivery.gain
        if level == 1.0:
            return found
        return [min(max(float(value) * level, -1.0), 1.0) for value in found]


class Engines:
    """The recognizer and the synthesiser, loaded once and kept warm.

    Warm because the alternative is reading four hundred megabytes of ONNX off
    a disk between "let go of the microphone" and "see the words", every time.
    The design intent's own arithmetic is that the target machine has 96 GB and
    this is the correct trade.

    The speaker is no longer chosen here. Protocol 1 read one ``speaker_id``
    out of the manifest at start-up and used it for everything, which made
    "which voice" a property of the process rather than of the request --
    section 56. Every synthesis below takes the numeric speaker the parent
    resolved from its registry, and this file validates it against what the
    loaded bank actually contains rather than trusting it.
    """

    def __init__(self, config: dict):
        self.config = config
        self.provider = "cpu"
        self.recognizer = None
        self.tts = None
        self.stt_model_id = str((config.get("stt") or {}).get("id") or "")
        self.tts_model_id = str((config.get("tts") or {}).get("id") or "")
        self.stt_threads = int(config.get("stt_threads") or 4)
        self.tts_threads = int(config.get("tts_threads") or 4)
        self.num_speakers = 0
        self.sample_rate = 0
        self.bank_version = str(config.get("bank_version") or "")
        self.streaming = ""
        self.callback_probe_ms = 0
        """How long the streaming probe below took, in milliseconds.

        Reported rather than only decided on. "Callbacks work here" and "this
        runtime is fast enough for them to help" are different claims, and a
        probe that takes a second on somebody's machine is the first evidence
        that the second one may not follow from the first.
        """

    def load(self) -> None:
        import sherpa_onnx

        stt = self.config.get("stt") or {}
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=_required(stt, "encoder"),
            decoder=_required(stt, "decoder"),
            tokens=_required(stt, "tokens"),
            num_threads=self.stt_threads,
            provider="cpu",
            decoding_method="greedy_search",
            language=str(stt.get("language") or ""),
            task="transcribe",
        )

        tts = self.config.get("tts") or {}
        root = tts.get("root") or ""
        kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=_required(tts, "model"),
            voices=_required(tts, "voices"),
            tokens=_required(tts, "tokens"),
            data_dir=_optional(root, "espeak-ng-data"),
            dict_dir=_optional(root, "dict"),
            lexicon=_lexicons(root))
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=kokoro, provider="cpu", num_threads=self.tts_threads)
        # ``max_num_sentences`` is the granularity of the streaming callback as
        # well as of the batching: one sentence per batch means the first
        # sentence's audio leaves this process while the second is still being
        # computed, which is the whole latency argument for streaming.
        self.tts = sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(model=model_config, max_num_sentences=1))
        self.num_speakers = int(getattr(self.tts, "num_speakers", 0) or 0)
        self.sample_rate = int(getattr(self.tts, "sample_rate", 0) or 0)
        probe_started = time.monotonic()
        self.streaming = "callback" if self._callback_supported() else "segment"
        self.callback_probe_ms = int((time.monotonic() - probe_started) * 1000)

    def _callback_supported(self) -> bool:
        """Whether this runtime can hand samples back *during* a synthesis.

        Tried, not read. An earlier version of this asked whether the word
        "callback" appeared in ``generate``'s signature or docstring, which is
        a question about documentation rather than about what happens when you
        call it -- and it was wrong here in a way that took the whole feature
        down silently.

        sherpa's callback parameter is typed ``py::array_t<float>``, so pybind11
        has to build a NumPy array to deliver each batch to Python. This
        runtime is two unpacked wheels -- sherpa_onnx and its core -- and NumPy
        is not one of them, so that construction raises and takes the synthesis
        with it. The completed-audio path is unaffected, because ``samples``
        comes back as an ordinary list, which is exactly why auditions worked
        while every spoken reply failed.

        The probe is one very short synthesis whose callback returns 0
        immediately, so it stops at the first batch. It costs a fraction of a
        second at worker start and buys a handshake that is true.
        """
        seen = []

        def probe(samples, _progress):
            seen.append(1)
            return 0

        try:
            self.tts.generate("ok", sid=0, speed=1.0, callback=probe)
        except Exception as exc:  # noqa: BLE001 - every failure means the same thing
            _note(f"this runtime cannot stream inside a sentence ({_safe(exc)}); "
                  f"speech will arrive one segment at a time instead")
            return False
        if not seen:
            _note("this runtime accepted a streaming callback and never called it; "
                  "speech will arrive one segment at a time instead")
            return False
        return True

    def speaker(self, sid) -> int:
        """A speaker id this bank really has.

        sherpa answers an out-of-range sid by logging a warning and using
        speaker 0, which for a custom voice would mean a clone the user
        selected being spoken silently in somebody else's voice. Refused here
        instead, with a sentence the parent can show.
        """
        wanted = int(sid or 0)
        if wanted < 0 or (self.num_speakers and wanted >= self.num_speakers):
            raise ValueError("that voice is not in the installed voice bank")
        return wanted

    def transcribe(self, samples, rate: int) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(rate, samples)
        self.recognizer.decode_stream(stream)
        return (stream.result.text or "").strip()

    def synthesize(self, text: str, sid: int = 0, delivery: "Delivery" = None):
        """One complete string, shaped, as float samples and a rate.

        Returns samples rather than bytes because ``encode_wav`` above is what
        this route's caller wants. It goes through the same :class:`Shaper` the
        streaming path uses, so an audition and a spoken reply cannot end up
        pitched differently.

        ``pause_ms`` does nothing here and is not quietly approximated: this
        route is one ``generate`` call with no segment boundaries in it, and the
        pacing control is a gap *between* segments. The only text that comes
        down this path is an audition or a fallback for a single short reply.
        """
        found = delivery or NEUTRAL
        audio = self.tts.generate(text, sid=self.speaker(sid),
                                  speed=found.generation_speed)
        rate = int(audio.sample_rate)
        if not found.shapes:
            return audio.samples, rate
        return Shaper(found).levelled(audio.samples), rate

    def stream(self, text: str, sid: int, delivery, on_audio) -> int:
        """Synthesize ``text``, handing PCM16 to ``on_audio`` as it appears.

        ``on_audio(pcm, rate)`` returns True to continue and False to stop, and
        stopping is how cancellation actually happens -- see the module
        docstring. Returns the number of samples produced.

        The bytes handed over are built here, inside the call, from the array
        sherpa gave the callback. That array is the parent binding's own copy
        and is not sherpa's buffer, but converting immediately is what makes
        that true of everything downstream as well: nothing in this process
        holds a reference to native audio memory after the callback returns.
        """
        speaker = self.speaker(sid)
        found = delivery or NEUTRAL
        rate = int(self.sample_rate or 0)
        produced = [0]
        # One per segment, because the pitch shift carries a fractional read
        # position between blocks and starting a new one mid-sentence would put
        # a click at the boundary.
        shaper = Shaper(found)

        def emit(samples, sample_rate) -> bool:
            block = shaper.block(samples)
            if not block:
                return True
            produced[0] += len(block) // 2
            return bool(on_audio(block, sample_rate))

        if self.streaming == "callback":
            def callback(samples, _progress):
                return 1 if emit(samples, rate or self.sample_rate) else 0

            try:
                audio = self.tts.generate(text, sid=speaker,
                                          speed=found.generation_speed,
                                          callback=callback)
                if not produced[0] and getattr(audio, "samples", None) is not None:
                    # A build that accepts the argument and never calls it. One
                    # honest fallback rather than silence.
                    emit(audio.samples, int(audio.sample_rate))
                return produced[0]
            except Exception as exc:  # noqa: BLE001 - see the two branches below
                if produced[0]:
                    # Part of this segment has already been sent, so re-running
                    # it would say those words twice. A failure after audio is a
                    # real failure and is reported as one.
                    raise
                self.streaming = "segment"
                _note(f"this runtime stopped streaming inside a sentence ({_safe(exc)}); "
                      f"speech will arrive one segment at a time from now on")
                # The half-used shaper goes with the attempt that failed. It
                # carries a read position into audio nobody heard, and reusing
                # it would start the retry a fraction of a sample out.
                shaper = Shaper(found)

        audio = self.tts.generate(text, sid=speaker, speed=found.generation_speed)
        emit(audio.samples, int(audio.sample_rate))
        return produced[0]


_NUMPY = "unasked"
"""Whether this runtime has NumPy, decided once.

This runtime is two unpacked wheels and NumPy is not one of them, so the answer
is normally "no" -- and asking per audio block turned that into an ImportError
raised and swallowed thousands of times a minute. Asked once, remembered, and
the stdlib path below is the one that actually runs here.
"""


def _numpy():
    global _NUMPY

    if _NUMPY == "unasked":
        try:
            import numpy

            _NUMPY = numpy
        except Exception:
            _NUMPY = None
    return _NUMPY


def pcm16(samples, gain: float = 1.0) -> bytes:
    """Float samples in [-1, 1] as little-endian PCM16 bytes, times ``gain``.

    The gain is folded in here rather than applied in a pass of its own, and the
    clamp that was already protecting against a model that overshot is what
    limits it: a volume setting cannot distort, it can only stop getting louder.

    Little-endian on every platform, explicitly, because these bytes are read
    by a browser's ``DataView`` with ``littleEndian`` hard-coded true -- and a
    big-endian host that silently wrote its own order would produce noise that
    nothing in the chain could explain.

    Works on whatever sherpa hands over: a NumPy array where this runtime has
    NumPy, and the plain list ``GeneratedAudio.samples`` is otherwise.
    """
    if samples is None:
        return b""
    level = float(gain or 1.0)
    numpy = _numpy()
    if numpy is not None:
        block = numpy.array(samples, dtype="float32", copy=True)
        if not block.size:
            return b""
        if level != 1.0:
            block *= level
        numpy.clip(block, -1.0, 1.0, out=block)
        return (block * 32767.0).astype("<i2").tobytes()

    import array

    clipped = array.array("h")
    scale = 32767.0 * level
    for value in samples:
        scaled = int(float(value) * scale)
        clipped.append(32767 if scaled > 32767 else (-32768 if scaled < -32768 else scaled))
    if sys.byteorder == "big":
        clipped.byteswap()
    return clipped.tobytes()


def silence(rate: int, milliseconds: int) -> bytes:
    """A block of PCM16 quiet, for the gap between two sentences.

    Built rather than synthesized: asking Kokoro to say nothing costs a
    forward pass and produces a room tone that would change with the speaker.
    Zeroes are the same silence every time, which is what a pacing control is
    asking for.
    """
    frames = int(max(int(rate or 0), 0) * max(int(milliseconds or 0), 0) / 1000.0)
    return b"\x00\x00" * frames if frames > 0 else b""


def _required(section: dict, key: str) -> str:
    value = str(section.get(key) or "")
    if not value or not os.path.isfile(value):
        raise ValueError(f"the installed bundle has no {key}")
    return value


def _optional(root: str, name: str) -> str:
    candidate = os.path.join(root, name) if root else ""
    return candidate if candidate and os.path.isdir(candidate) else ""


def _lexicons(root: str) -> str:
    """Kokoro's lexicons, comma-joined, in the order sherpa-onnx expects.

    Discovered from the installed bundle rather than listed in the manifest,
    because which lexicons a Kokoro release ships is a property of that release
    and the manifest already pins the archive it came out of.
    """
    if not root or not os.path.isdir(root):
        return ""
    found = sorted(name for name in os.listdir(root)
                   if name.startswith("lexicon") and name.endswith(".txt"))
    return ",".join(os.path.join(root, name) for name in found)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def _note(text: str) -> None:
    """One diagnostic line on stderr. Never content -- see the module docstring."""
    try:
        sys.stderr.write(f"voice-worker: {text}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _safe(exc: BaseException) -> str:
    """What the parent is told a failure was.

    Deliberately the exception's *class* and nothing else for anything this file
    did not raise itself. A third-party library is entitled to put whatever it
    likes in a message, including the input it was given, and this feature's
    invariant is that the input never leaves this process.
    """
    if isinstance(exc, ValueError):
        return str(exc)
    return exc.__class__.__name__


MAX_PENDING_AUDIO = 12
"""Audio frames allowed to be waiting for the writer at once.

The flow-control valve, and the only one this process needs. Past it the
sherpa callback waits, so generation slows to whatever the parent is actually
consuming; below it the writer always has work. Twelve frames of one sentence
each is a couple of seconds -- enough that a momentarily busy pipe does not
stall inference, small enough that a browser that stopped listening does not
leave megabytes of speech in this process's memory.

The wait is on a condition variable that cancellation also notifies, which is
section 16's requirement that queue backpressure can never make a cancel wait
behind its own producer.
"""

WRITE_TIMEOUT = 0.25

DRAIN_GRACE = 2.0
"""How long a request already accepted may take once stdin has ended.

Two seconds, and see :meth:`Worker.finish_pending` for why it is not zero and
not thirty.
"""


class Turn:
    """One streaming reply inside this process."""

    def __init__(self, identifier: str, sid: int, delivery: "Delivery" = None):
        self.id = str(identifier or "")
        self.sid = int(sid or 0)
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
        """The next segment, or ``None`` when this turn has no more.

        Waits in short steps so that a cancel arriving on the command loop is
        noticed here within :data:`WRITE_TIMEOUT` rather than whenever the
        parent happens to send more text.
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

    Everything mutable in this process lives here, and every field is touched
    under a lock or is a ``threading.Event``. There is deliberately no lock
    that spans a call into sherpa: the whole point of the protocol 2 refactor
    is that a thread inside native inference holds nothing anybody else needs.
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
        self.engines = None
        self._writer = None
        self._lane = None

    # -- the writer -------------------------------------------------------- #

    def start_threads(self) -> None:
        self._writer = threading.Thread(target=self._write_loop, name="voice-writer",
                                        daemon=True)
        self._writer.start()
        self._lane = threading.Thread(target=self._lane_loop, name="voice-lane", daemon=True)
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
                # The parent is gone or its pipe is closed. Nothing here can
                # report that to anybody, and the read loop is about to see the
                # same thing and end the process.
                self._stopping.set()
            finally:
                if audio:
                    with self._audio_room:
                        self._pending_audio = max(0, self._pending_audio - 1)
                        self._audio_room.notify_all()

    def finish_pending(self, timeout: float) -> None:
        """Answer what has already been accepted, for up to ``timeout``.

        Run when the parent's pipe closes rather than when it asks to shut
        down, and the difference matters. A request this process took off the
        wire is a request somebody is waiting for; dropping it because stdin
        happened to end first turns a transcription into a timeout on the other
        side. A ``shutdown`` command is the opposite -- it means stop now -- so
        it deliberately does not come through here.

        Bounded, because the reason stdin ended may be that the parent died,
        and the parent-death containment this process arranged is not a licence
        to take three minutes about it.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline and not self._stopping.is_set():
            with self._jobs_ready:
                if not self._jobs and not self._working.is_set():
                    return
                self._jobs_ready.wait(0.05)

    def wake_writers(self) -> None:
        """Let go of anything waiting for room. Called on cancel and shutdown."""
        with self._audio_room:
            self._audio_room.notify_all()

    # -- the inference lane ------------------------------------------------ #

    def submit(self, job, first: bool = False) -> None:
        with self._jobs_ready:
            if first:
                self._jobs.appendleft(job)
            else:
                self._jobs.append(job)
            self._jobs_ready.notify()

    def _lane_loop(self) -> None:
        """One model at a time, forever. Section 18's single inference lane.

        Serialized rather than parallel because this process shares a CPU with
        an image model: two ONNX sessions running at once would take the cores
        that make Forge feel responsive, which is the trade section 35 refuses
        to make by default.
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

    # -- turns ------------------------------------------------------------- #

    def turn(self, identifier: str):
        with self._turn_lock:
            return self._turns.get(str(identifier or ""))

    def open_turn(self, identifier: str, sid: int, delivery: "Delivery" = None) -> Turn:
        found = Turn(identifier, sid, delivery)
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
        """One whole turn, segment by segment, on the inference lane."""
        engines = self.engines
        try:
            sid = engines.speaker(turn.sid)
        except Exception as exc:
            self.send({"op": "tts_error", "turn": turn.id, "error": _safe(exc)})
            self.close_turn(turn.id)
            return
        self.send({"op": "tts_ready", "turn": turn.id,
                   "sample_rate": int(engines.sample_rate or 0),
                   "streaming": engines.streaming})
        sequence = 0
        started = time.monotonic()
        spoken = 0
        gap = silence(int(engines.sample_rate or 0), turn.delivery.pause_ms)
        # One segment's shape, in numbers only: how many times sherpa handed
        # audio back and how long the first of them took. With
        # ``max_num_sentences=1`` the block count *is* the sentence count, which
        # is why the parent can report callback granularity without this file
        # ever counting anything about the text.
        block = {"count": 0, "first": 0.0, "at": 0.0}
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
                    if block["at"]:
                        block["count"] += 1
                        if not block["first"]:
                            block["first"] = time.monotonic() - block["at"]
                    _turn.samples += len(chunk) // 2
                    self.send({"op": "tts_audio", "turn": _turn.id, "seq": sequence,
                               "sample_rate": int(rate or 0)}, chunk, audio=True)
                    return not _turn.cancelled.is_set()

                # Before this segment rather than after the last one, so a turn
                # that is cancelled mid-reply does not end on a gap and a turn
                # that finishes does not leave the composer waiting through one.
                if gap and spoken and not turn.cancelled.is_set():
                    on_audio(gap, int(engines.sample_rate or 0))
                # Armed after the pause and not before it: a configured pause is
                # deliberate prosody, and counting it as this segment's first
                # audio would make an intentional gap look like a slow synthesis.
                block.update({"count": 0, "first": 0.0, "at": time.monotonic()})
                engines.stream(text, sid, turn.delivery, on_audio)
                elapsed_segment = time.monotonic() - block["at"]
                block["at"] = 0.0
                spoken += 1
                if turn.cancelled.is_set():
                    break
                self.send({"op": "tts_segment_done", "turn": turn.id, "seq": sequence,
                           "blocks": block["count"],
                           "first_block_ms": int(block["first"] * 1000),
                           "segment_ms": int(elapsed_segment * 1000),
                           "streaming": engines.streaming})
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


def serve(stdin, stdout, engines_factory=None) -> int:
    """Read frames until end of input. Returns the process exit status.

    This is the command loop and it does as little as it possibly can. Its one
    job is to never be busy: everything that could take longer than parsing a
    header is handed to the inference lane, so a ``tts_cancel`` sent while
    Kokoro is three seconds into a paragraph is *read* immediately rather than
    after the paragraph. That property is the acceptance criterion for gate 1
    and it is why this function contains no inference call at all.
    """
    factory = engines_factory or (lambda config: Engines(config))
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
                    _lower_priority()
                    engines = factory(header.get("config") or {})
                    started = time.monotonic()
                    engines.load()
                    worker.engines = engines
                    worker.start_threads()
                    reply(request_id, {
                        "ok": True,
                        "op": "ready",
                        "protocol_version": PROTOCOL_VERSION,
                        "runtime_version": _runtime_version(),
                        "provider": engines.provider,
                        "parent_death": parent_death,
                        "stt_model_id": engines.stt_model_id,
                        "tts_model_id": engines.tts_model_id,
                        "stt_threads": engines.stt_threads,
                        "tts_threads": engines.tts_threads,
                        "num_speakers": engines.num_speakers,
                        "sample_rate": engines.sample_rate,
                        "bank_version": engines.bank_version,
                        "streaming": engines.streaming,
                        "callback_probe_ms": engines.callback_probe_ms,
                        "load_seconds": round(time.monotonic() - started, 3),
                    })
                    _note(f"ready — CPU provider, STT threads {engines.stt_threads}, "
                          f"TTS threads {engines.tts_threads}, {engines.num_speakers} "
                          f"speakers, {engines.streaming} streaming, "
                          f"containment {parent_death}")
                except SystemExit:
                    raise
                except Exception as exc:
                    worker.engines = None
                    reply(request_id, {"ok": False, "error": _safe(exc)})
                    _note(f"could not load the speech models: {_safe(exc)}")
                continue

            if worker.engines is None:
                if operation.startswith("tts_"):
                    worker.send({"op": "tts_error", "turn": str(header.get("turn") or ""),
                                 "error": "runtime is not initialised"})
                else:
                    reply(request_id, {"ok": False, "error": "runtime is not initialised"})
                continue

            if operation == "ping":
                # Answered on this thread on purpose. "Is the worker alive and
                # answering while it is speaking" is a question whose answer is
                # worthless if it has to queue behind the speaking.
                reply(request_id, {"ok": True, "speaking": bool(worker._turns)})

            elif operation == "stt":
                # In front of any queued speech. Section 14: pressing the
                # microphone while a reply is being read aloud means the reply
                # is over, and the parent has already cancelled its turn by the
                # time this arrives.
                worker.submit(lambda rid=request_id, body=payload: _do_stt(worker, reply, rid, body),
                              first=True)

            elif operation == "tts":
                worker.submit(lambda rid=request_id, body=payload, head=header:
                              _do_tts(worker, reply, rid, head, body))

            elif operation == "tts_begin":
                turn = worker.open_turn(str(header.get("turn") or ""),
                                        int(header.get("sid") or 0),
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
        # Waited for, briefly. A frame this process decided to send and then
        # exited before writing is a reply the parent waits its whole timeout
        # for, and the only cost of waiting here is the time it takes to write
        # what is already in memory.
        writer = worker._writer
        if writer is not None and writer.is_alive():
            writer.join(timeout=2.0)


def _do_stt(worker, reply, request_id, payload: bytes) -> None:
    try:
        started = time.monotonic()
        samples, rate = decode_wav(payload)
        text = worker.engines.transcribe(samples, rate)
        elapsed = time.monotonic() - started
        seconds = len(samples) / float(rate or 1)
        reply(request_id, {"ok": True, "text": text, "audio_seconds": round(seconds, 2),
                           "elapsed": round(elapsed, 3)})
        _note(f"stt finished — {seconds:.1f} s audio in {elapsed:.1f} s")
    except Exception as exc:
        reply(request_id, {"ok": False, "error": _safe(exc)})
        _note(f"request failed: {_safe(exc)}")


def _do_tts(worker, reply, request_id, header: dict, payload: bytes) -> None:
    """The completed-reply route, kept for Test playback and as a fallback.

    Section 20: automatic speech uses the streaming turn above, and this stays
    because auditioning a voice in Settings genuinely is a complete short
    string, and because an explicit non-streaming fallback is better than a
    deployment that silently buffers a stream and calls it streaming.
    """
    try:
        text = payload.decode("utf-8", "replace")
        started = time.monotonic()
        samples, rate = worker.engines.synthesize(text, int(header.get("sid") or 0),
                                                  Delivery.from_header(header))
        audio = encode_wav(samples, rate)
        elapsed = time.monotonic() - started
        seconds = len(audio) / float(max(rate, 1) * 2)
        reply(request_id, {"ok": True, "sample_rate": rate, "audio_seconds": round(seconds, 2),
                           "elapsed": round(elapsed, 3)}, audio)
        _note(f"tts finished — {len(text)} characters, {seconds:.1f} s audio in "
              f"{elapsed:.1f} s")
    except Exception as exc:
        reply(request_id, {"ok": False, "error": _safe(exc)})
        _note(f"request failed: {_safe(exc)}")


def _runtime_version() -> str:
    try:
        import sherpa_onnx

        return str(getattr(sherpa_onnx, "__version__", "") or "")
    except Exception:
        return ""


def selftest() -> int:
    """Prove the staged runtime imports and can be asked for CPU. One JSON line.

    Run by the installer against the *staging* environment, before anything is
    promoted. It deliberately builds a real config object rather than only
    importing: an ONNX Runtime whose CPU provider is missing imports perfectly
    well and fails at the first model.

    NumPy is reported and never required. It is what lets sherpa hand a batch of
    samples back *during* a synthesis -- pybind11 has to build a
    ``py::array_t<float>`` to do it -- so a runtime without it speaks a segment
    at a time instead of a sentence at a time. That is slower and it is not
    broken, and an install refused for it would trade a working feature for a
    faster one.
    """
    report = {"ok": False, "provider": "", "runtime_version": "", "numpy_version": ""}
    try:
        import numpy

        report["numpy_version"] = str(getattr(numpy, "__version__", "") or "")
    except Exception as exc:  # noqa: BLE001 - absence is a fact, not a failure
        report["numpy_error"] = f"{exc.__class__.__name__}: {exc}"
    try:
        import sherpa_onnx

        report["runtime_version"] = str(getattr(sherpa_onnx, "__version__", "") or "")
        config = sherpa_onnx.OfflineTtsModelConfig(provider="cpu", num_threads=1)
        report["provider"] = str(getattr(config, "provider", "") or "cpu")
        report["ok"] = report["provider"] == "cpu"
    except Exception as exc:
        # The message, not just the class. This is the one place in this file
        # where an exception's own words are worth repeating: they are about
        # package and library names, never about anything anybody said, and
        # "ModuleNotFoundError" on its own does not say *which* module -- which
        # cost a round trip the first time this failed on somebody's machine.
        report["error"] = f"{exc.__class__.__name__}: {exc}"
        report["prefix"] = sys.prefix
        report["path"] = [entry for entry in sys.path if entry]
    sys.stdout.write(json.dumps(report) + "\n")
    sys.stdout.flush()
    return 0 if report["ok"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(MARKER, dest="marker", action="store_true")
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--session", default="")
    parser.add_argument("--selftest", action="store_true")
    known, _unknown = parser.parse_known_args(argv)

    if known.selftest:
        return selftest()

    # Binary on both sides: the protocol is length-prefixed bytes, and a text
    # wrapper would translate newlines on Windows and corrupt every WAV.
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    return serve(stdin, stdout)


if __name__ == "__main__":
    raise SystemExit(main())
