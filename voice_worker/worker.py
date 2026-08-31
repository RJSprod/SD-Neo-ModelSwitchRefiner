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


MAX_NUM_SENTENCES = 1
"""How many sentences sherpa batches, and therefore how often it calls back.

Named rather than written in place because it is now two things: the batching
policy, and the unit the per-segment block counts below are counted in. A log
line saying ``callback_blocks=4`` means four sentences only while this is one,
so the value is reported in the configuration line rather than assumed by
whoever reads the numbers.
"""


def _lower_priority() -> None:
    """Be the process that yields when an image is rendering. Never fatal.

    The call below is deliberately left exactly as it has always been, including
    its undeclared handle types -- see :func:`_priority`, where the same pattern
    turned out to report a failed read. Declaring them here would not be a
    diagnostic change: it would make this call start working, and a worker that
    began yielding CPU for the first time is a change to how fast speech is
    synthesised. That is a decision to take deliberately, from the corrected
    reading this build now produces, rather than a side effect of tidying a
    neighbouring function.
    """
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


PRIORITY_CLASSES = {
    0x00000020: "normal",
    0x00004000: "below_normal",
    0x00008000: "above_normal",
    0x00000080: "high",
    0x00000100: "realtime",
    0x00000040: "idle",
}
"""Windows priority classes, for reading one back. Named rather than raised:
this file sets ``below_normal`` and nothing here ever asks for more."""


def _priority() -> str:
    """What this process's scheduling priority actually ended up as.

    Observation only, and that is the whole design of it. A worker that read its
    own priority in order to change it would be a scheduler, and a scheduler is
    the thing a speech process running beside an image model must not become --
    raising a Linux priority needs ``CAP_SYS_NICE`` anyway, and asking for it
    would turn a chat feature into something that wants privileges.

    So this is here to answer one question in a shared log: was the run that
    produced these numbers a run at the priority everybody assumes? Never
    raises; an unreadable priority is reported as unknown rather than guessed.

    The Windows half declares its own types, and that is not decoration. A
    process handle is pointer-sized, ctypes defaults an undeclared return value
    and an undeclared argument to ``int``, and the pseudo-handle ``GetCurrentProcess``
    answers with is ``(HANDLE)-1`` -- so a 64-bit host can be handed a truncated
    handle and refuse it. That is what the first shipped version of this
    function did: it reported ``class_0x0``, which is not a priority class at
    all but ``GetPriorityClass`` returning zero for failure. A diagnostic that
    quietly reports a failed read as a value is worse than no diagnostic, so
    ``restype`` and ``argtypes`` are declared and a zero is now named as the
    failure it is.
    """
    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetPriorityClass.restype = ctypes.c_uint
            kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
            found = int(kernel32.GetPriorityClass(kernel32.GetCurrentProcess()))
            if not found:
                return "unreadable"
            return PRIORITY_CLASSES.get(found, f"class_{found:#x}")
        found = os.getpriority(os.PRIO_PROCESS, 0)
        return f"nice{found:+d}"
    except Exception:  # noqa: BLE001 - a priority nobody can read is not a failure
        return "unknown"


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
        self._quiet_floor = {}
        """What each speaker's noise floor last measured, for :class:`Trim`."""
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
            sherpa_onnx.OfflineTtsConfig(model=model_config,
                                         max_num_sentences=MAX_NUM_SENTENCES))
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

    def stream(self, text: str, sid: int, delivery, on_audio) -> dict:
        """Synthesize ``text``, handing PCM16 to ``on_audio`` as it appears.

        ``on_audio(pcm, rate)`` returns True to continue and False to stop, and
        stopping is how cancellation actually happens -- see the module
        docstring.

        Returns ``{"samples", "blocks"}``: how much audio this segment produced
        and how many times sherpa handed some back. The second number is
        counted here rather than by the caller because since :class:`Seam` it is
        no longer the same as the number of times ``on_audio`` was called -- the
        seam withholds a segment's last few milliseconds and releases them at
        the end, so it adds an emission of its own. What the metric is *for* is
        whether callback mode delivered anything early, and that question is
        about sherpa's hand-backs.

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
        handed = [0]
        # One per segment, because the pitch shift carries a fractional read
        # position between blocks and starting a new one mid-sentence would put
        # a click at the boundary.
        shaper = Shaper(found)
        # And one per segment each for the other two things that happen at the
        # *edges* of a segment rather than inside it: the model's own padding,
        # and the click where one segment meets the next. In that order --
        # see :class:`Trim` and :class:`Seam`.
        trim = Trim(rate or self.sample_rate, floor=self._quiet_floor.get(speaker, 0),
                    gap_ms=KEEP_GAP_MS + found.pause_ms)
        seam = Seam(rate or self.sample_rate)

        def emit(samples, sample_rate) -> bool:
            handed[0] += 1
            block = seam.block(trim.block(shaper.block(samples)))
            if not block:
                return True
            produced[0] += len(block) // 2
            return bool(on_audio(block, sample_rate))

        def close(sample_rate) -> None:
            """The trim's kept tail and the seam's ramp. Ends the segment."""
            for block in (seam.block(trim.flush()), seam.flush()):
                if not block:
                    continue
                produced[0] += len(block) // 2
                on_audio(block, sample_rate)
            # What this segment measured its own noise floor to be, kept for the
            # next segment in the same voice: a noise floor belongs to the voice
            # rather than to the sentence, and the front of a segment has to be
            # judged before that segment has measured anything.
            if len(self._quiet_floor) > 16:
                self._quiet_floor.clear()
            self._quiet_floor[speaker] = trim.measured

        if self.streaming == "callback":
            def callback(samples, _progress):
                return 1 if emit(samples, rate or self.sample_rate) else 0

            try:
                audio = self.tts.generate(text, sid=speaker,
                                          speed=found.generation_speed,
                                          callback=callback)
                if not handed[0] and getattr(audio, "samples", None) is not None:
                    # A build that accepts the argument and never calls it. One
                    # honest fallback rather than silence.
                    #
                    # The test is whether sherpa called back, not whether any
                    # bytes reached the parent: the seam withholds a segment's
                    # last few milliseconds, so a very short segment can have
                    # been synthesised in full with nothing sent yet, and
                    # generating it again there would say it twice.
                    emit(audio.samples, int(audio.sample_rate))
                close(rate or self.sample_rate)
                return {"samples": produced[0], "blocks": handed[0],
                        "trimmed_ms": trim.dropped_ms,
                        "quiet_ms": trim.quiet_ms, "gap_ms": trim.gap_ms,
                        "floor_db": trim.floor_db}
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
                # it would start the retry a fraction of a sample out. The trim
                # and the seam go with it for the same reason: between them they
                # are holding the last part of a segment that is about to be
                # said again, and counting quiet nobody heard.
                shaper = Shaper(found)
                trim = Trim(rate or self.sample_rate,
                            floor=self._quiet_floor.get(speaker, 0),
                            gap_ms=KEEP_GAP_MS + found.pause_ms)
                seam = Seam(rate or self.sample_rate)
                handed[0] = 0

        audio = self.tts.generate(text, sid=speaker, speed=found.generation_speed)
        emit(audio.samples, int(audio.sample_rate))
        close(rate or self.sample_rate)
        return {"samples": produced[0], "blocks": handed[0],
                "trimmed_ms": trim.dropped_ms,
                "quiet_ms": trim.quiet_ms, "gap_ms": trim.gap_ms,
                "floor_db": trim.floor_db}


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


DECLICK_MS = 8
"""How long a committed unit takes to reach full level, and to leave it again.

A unit does not end the way a recording ends. Kokoro is handed one sentence
and produces the waveform for it; where that waveform happens to be when the
sentence runs out is where the samples stop, and the next sentence is a fresh
forward pass that starts wherever *it* starts. Played back the way this feature
plays speech -- sample-exact, one unit after another, no gap -- the join between
the two is a step, and a step in a waveform is a click. It is small, it lands
at the end of a sentence, and on a long reply it happens once per sentence.

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

    A unit is one sentence's forward pass. Its last sample is wherever the
    waveform was when the sentence ran out and its first is wherever the next
    pass begins, and this feature plays units back with no gap between them --
    so the join is a step, and a step is a click. See :data:`DECLICK_MS`.

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

A segment is not the length of what it says. Kokoro is handed one sentence and
returns a waveform with the model's own silence around the words at both ends.
Voice Chat plays segments back with nothing between them, so those two add up
into the gap between one sentence and the next: dead air nobody asked for, in
the middle of a reply.

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
QUIET_SHARE = 16
"""What counts as quiet: a share of this unit's own loudest sample.

Two earlier rules missed it, and both misses are worth keeping written down
because they were the same mistake from opposite ends.

The first drew the line at two per cent of the peak but capped it at about -34
dBFS. The second, on the theory that a cloned voice's room tone sits above any
such line, anchored it to the *floor* instead -- three times the quietest ten
milliseconds in the unit. Then the machine reported ``floor_db=-68``: this
voice's quietest moment is exceptionally clean, so three times it is 39 counts,
under the absolute minimum, and the anchor made the line *stricter* rather than
looser. Both rules came back ``quiet_ms=0`` on units carrying about seven
hundred milliseconds of audio that no amount of text accounts for.

What is actually wanted is not "at the noise floor" but "after the last thing
worth hearing", and that is a share of the peak: a sixteenth, about 24 dB down.
Nothing intelligible sits under it for long, everything the model adds after the
end-of-speech token does, and it engages whatever the voice's noise floor turns
out to be -- because every unit has a loudest sample, and the quiet is defined
against that rather than against a level somebody guessed.

:data:`QUIET_FLOOR` remains as an absolute minimum, about -48 dBFS, so that a
unit which never gets loud cannot draw the line under its own speech. What
protects a genuinely quiet ending is not the level but the keep: only the run
*after* the last loud window is trimmed, and :data:`KEEP_TAIL_MS` of it stays.
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

KEEP_GAP_MS = 200
GAP_SPLICE_MS = 8
"""How long a pause *inside* a unit may be, and how it is shortened.

This was the answer in the end, and the first two rounds of this class were
looking in the wrong place. A machine reported it: quiet at the two ends of a
unit measured between 10 and 310 milliseconds, most of it under the amount kept
anyway -- while the longest pause *inside* the same units ran 480, 720, 1050.
The gap a listener hears between two sentences is almost never at a join. A
committed unit is a hundred and something characters, which is two or three
sentences, so most sentence boundaries are inside one.

So a pause is kept, up to this much, and what runs on past it is dead air rather
than delivery. Two hundred milliseconds is chosen to match what a unit boundary
already comes to -- ``KEEP_TAIL_MS`` plus the next unit's ``KEEP_LEAD_MS`` --
because the listener should not be able to hear where this feature's units
begin and end. The delivery's own "Pause between sentences" is added to both, so
that control finally means the same thing wherever the sentence boundary falls.

The cut is spliced rather than butted. What is kept is the beginning of the
pause and the last :data:`GAP_SPLICE_MS` before the next word, crossfaded onto
each other, so the listener hears the pause run straight into the approach to
that word. Cutting to the onset instead would put a step where the two ends
meet, and this file exists because a step is a click.
"""

MAX_LEAD_HOLD_MS = 400
SCAN_MS = 10
"""How long the front of a unit may be waited on, and how finely it is judged.

Only the *beginning* of a pause is ever kept, so everything past it can be
dropped where it is found rather than held. That is what keeps the delay this
class adds down to a fraction of a second whatever the model does: a pause of
any length costs the listener the same short wait.

Before the first word there is nothing to compare against yet, so the bound
there is a deadline instead. A unit whose opening cannot be told from its own
noise gives up waiting after 400 ms, sends what it has untrimmed, and carries
on -- so the cost of not being able to tell is 400 ms once, at the front of that
unit, rather than a trailing trim that never happens.

Ten milliseconds is the window the level is judged over, because a single sample
says nothing and a whole model chunk is eighty.
"""


class Trim:
    """The quiet a model puts either side of a unit, cut back to a set amount.

    A segment's leading quiet is trimmed to :data:`KEEP_LEAD_MS` and its
    trailing quiet to :data:`KEEP_TAIL_MS`. Quiet *inside* a segment is not
    touched: a pause the model put between two clauses is prosody, and only the
    two ends are padding.

    Trailing quiet cannot be recognised as trailing until the segment ends, so
    it is held here and released by :meth:`flush` -- bounded, because held audio
    is audio the listener does not have yet.

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
                 tail_ms: int = KEEP_TAIL_MS, floor: int = 0,
                 gap_ms: int = KEEP_GAP_MS):
        self.rate = max(0, int(rate or 0))
        self.scan = max(1, int(self.rate * SCAN_MS / 1000))
        self.lead = int(self.rate * max(0, int(lead_ms)) / 1000)
        self.tail = int(self.rate * max(0, int(tail_ms)) / 1000)
        self.gap = max(self.tail, int(self.rate * max(0, int(gap_ms)) / 1000))
        self.splice = max(1, int(self.rate * GAP_SPLICE_MS / 1000))
        self.lead_hold = max(self.lead, int(self.rate * MAX_LEAD_HOLD_MS / 1000))
        self.dropped = 0
        self.quiet_found = 0
        self.longest = 0
        """The longest run of quiet *inside* this unit, in samples.

        Measured and never trimmed. A pause the model put between two clauses is
        prosody and belongs to the delivery, so this class does not touch it --
        but a reply whose sentences are separated by half a second of nothing
        has to be able to say whether that half second is at the joins between
        units, where this class can do something about it, or inside them, where
        it cannot. Those are different problems and the log could not previously
        tell them apart.
        """
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
        self._edge = _int16(b"")
        self._opened = False
        self._peak = 0
        self._run = 0

    @property
    def dropped_ms(self) -> int:
        """How much quiet this unit lost, in milliseconds. For the log."""
        return int(self.dropped * 1000 / (self.rate or 1))

    @property
    def quiet_ms(self) -> int:
        """How much quiet this unit had at its two ends, cut or not."""
        return int(self.quiet_found * 1000 / (self.rate or 1))

    @property
    def gap_ms(self) -> int:
        """The longest pause inside this unit. Measured, never trimmed."""
        return int(self.longest * 1000 / (self.rate or 1))

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
        self.quiet_found += self._run if self._opened else len(self._quiet)
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
            # Speech, and with it the end of whatever pause came before it. Its
            # length is kept whether or not it was shortened, because it is the
            # one number that says where a gap somebody can hear actually is.
            if self._run > self.longest:
                self.longest = self._run
            out.extend(self._resume())
            self._run = 0
            out.extend(window)
            return
        self._run += len(window)
        self._hush(window)

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

    def _hush(self, window) -> None:
        """A quiet window, once the unit has started.

        The beginning of the pause is kept and the rest is dropped where it is
        found rather than held, because only the beginning is ever going to be
        sent -- :data:`KEEP_GAP_MS` of it inside the unit, :data:`KEEP_TAIL_MS`
        of it at the end. The last few milliseconds are the exception: they are
        kept in ``_edge`` for the splice, because they are the approach to
        whatever word comes next.
        """
        room = max(0, self.gap - len(self._quiet))
        if room:
            self._quiet.extend(window[:room])
        if len(window) > room:
            self.dropped += len(window) - room
        self._edge.extend(window)
        if len(self._edge) > self.splice:
            del self._edge[:len(self._edge) - self.splice]

    def _resume(self):
        """The pause that has just ended, at the length it is allowed to be.

        Short enough and it is returned untouched -- a pause between two clauses
        is the model's delivery and belongs to the listener. Longer, and what
        comes back is its beginning with the last :data:`GAP_SPLICE_MS` before
        the next word crossfaded onto the end, so the join is a fade rather than
        a step.
        """
        held, self._quiet = self._quiet, _int16(b"")
        edge, self._edge = self._edge, _int16(b"")
        if self._run <= len(held):
            return held
        fade = min(self.splice, len(edge), len(held) // 2)
        if fade <= 0:
            return held
        window = _ramp(fade)
        start = len(held) - fade
        for index in range(fade):
            rising = window[index]
            held[start + index] = int(held[start + index] * (1.0 - rising)
                                      + edge[index] * rising)
        held.extend(edge[fade:])
        return held

    def _quiet_level(self) -> int:
        """What counts as quiet once the unit has started."""
        return max(QUIET_FLOOR, self._peak // QUIET_SHARE)


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
        # One segment's shape, in numbers only: how long the first block took to
        # reach the parent and how much speech the segment came to. How many
        # times sherpa handed audio back is counted where it happens, inside
        # ``Engines.stream``: with :class:`Seam` in the path the number of
        # blocks that leave here is no longer the number that arrived, and it is
        # sherpa's hand-backs that the callback-granularity metric is about.
        # With ``max_num_sentences=1`` that count *is* the sentence count, which
        # is why the parent can report callback granularity without this file
        # ever counting anything about the text.
        block = {"first": 0.0, "at": 0.0, "samples": 0}
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
                        block["samples"] += len(chunk) // 2
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
                block.update({"first": 0.0, "samples": 0,
                              "at": time.monotonic()})
                metrics = engines.stream(text, sid, turn.delivery, on_audio)
                elapsed_segment = time.monotonic() - block["at"]
                block["at"] = 0.0
                spoken += 1
                if turn.cancelled.is_set():
                    break
                self.send({"op": "tts_segment_done", "turn": turn.id, "seq": sequence,
                           "blocks": metrics["blocks"],
                           "first_block_ms": int(block["first"] * 1000),
                           "segment_ms": int(elapsed_segment * 1000),
                           # How much speech this unit actually produced, which
                           # is what turns a synthesis time into a real-time
                           # factor for one unit rather than for a whole turn.
                           "samples": block["samples"],
                           "trimmed_ms": metrics["trimmed_ms"],
                           "quiet_ms": metrics["quiet_ms"],
                           "gap_ms": metrics["gap_ms"],
                           "floor_db": metrics["floor_db"],
                           "sample_rate": int(engines.sample_rate or 0),
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
                        "max_num_sentences": MAX_NUM_SENTENCES,
                        "priority": _priority(),
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
