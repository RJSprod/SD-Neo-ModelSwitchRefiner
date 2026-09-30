"""The VibeVoice sidecar: one contained GPU process per card, and everything Torch touches.

Run by path, under the isolated VibeVoice interpreter, by
:mod:`mc_voice_vibevoice_runtime`, once per card that is asked to render. It
never imports a Forge module, a Model Chain module or another engine's runtime,
and it is the first worker in this repository that is *meant* to hold a
graphics card: the parent starts it with exactly one card visible
(``CUDA_VISIBLE_DEVICES`` set to that card's UUID) and refuses its handshake
unless the card it came up on is the card it was asked for.

What lives here and why it is not in the parent
-----------------------------------------------
Everything that needs a tensor, which is a short list:

    proving which card this process is on, and how much of it is free
    loading one VibeVoice model and its processor from a local directory
    turning a script and its voice samples into the model's prompt
    one generation at a time, cancellable between steps
    the audio it produced, as PCM16 WAV bytes

The parent owns the turns on each card, the settings, the samples and the
outputs; this process owns inference and nothing else. It is handed a directory
it did not choose and audio it did not record, and it validates both anyway,
because it is the process that would crash.

The upstream surface this file binds to
---------------------------------------
Four calls, written down here because they are the whole contract with the
community ``vibevoice`` package (0.0.1, the preserved original):

    ``VibeVoiceProcessor.from_pretrained(model_dir)`` -- a local directory
        holding ``preprocessor_config.json``, whose
        ``language_model_pretrained_name`` names the local tokenizer directory
        the installer wrote (the name must contain ``qwen``: the processor
        picks its tokenizer class by that substring);
    ``VibeVoiceForConditionalGenerationInference.from_pretrained(model_dir,
        torch_dtype=..., device_map="cuda", attn_implementation=...)``, then
        ``model.eval()`` and ``model.set_ddpm_inference_steps(num_steps=...)``;
    ``processor(text=[script], voice_samples=[[array, ...]], padding=True,
        return_tensors="pt", return_attention_mask=True)``;
    ``model.generate(**inputs, max_new_tokens=..., cfg_scale=...,
        tokenizer=processor.tokenizer, generation_config={"do_sample": False},
        verbose=False, stop_check_fn=...)``, whose ``speech_outputs[0]`` is the
        audio at 24 kHz and whose ``stop_check_fn`` is read at the top of every
        generation step.

How the processor numbers speakers, and what that costs the caller
-------------------------------------------------------------------
The processor parses one ``Speaker N: text`` line per ``\\n``-separated line
(``^Speaker\\s+(\\d+)\\s*:\\s*(.*)$``, case-insensitive); a line without the
prefix is dropped with a warning, so an entry's text may not contain a newline.
It then subtracts one from every number when the smallest is above zero, and
binds ``voice_samples[i]`` to the prompt line ``Speaker i`` -- so the sample
for the caller's Speaker 1 has to be the first array, Speaker 2's the second,
and the numbers have to be dense. That is why :func:`script_text` orders the
voice arrays by speaker *number* rather than by first appearance, and why a
script that uses Speakers 1 and 3 is renumbered 1 and 2 before it is sent:
left alone, the processor would give Speaker 3 no voice at all and give a
voice to a Speaker 2 who has no line. When the caller's numbers are already
dense from 1, which is the ordinary case, the renumbering is the identity and
the text reads exactly as the caller numbered it.

One render at a time, and cancellation between steps
----------------------------------------------------
One model instance serves one generation at a time, on one lane thread, and a
second ``render`` while one runs is refused rather than queued. The command
loop itself does no work: ``cancel`` and ``status`` are answered on it while
the lane is inside the model, which is the whole reason the loop stays free.
Cancellation is cooperative and honest about it: ``stop_check_fn`` is read at
the top of every generation step, so a cancel lands within one step and the
reply carries whatever audio was made before it, marked ``cancelled``.

Nothing private crosses the pipe
--------------------------------
An error reply carries either a sentence this file wrote (:class:`Refusal`) or
an exception's class name, never a library's own message: a library message
can name a path, a tensor shape or the text that was being spoken. The
directory the model was loaded from is reported by ``status`` because the
parent asked for it; it is never put in an error, a note or a command line.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import platform
import queue
import struct
import sys
import threading
import time
import wave

PROTOCOL_VERSION = 1
"""The VibeVoice protocol's own version, counted from one.

Not shared with the other four workers. They speak the same *framing* and this
one deliberately overlaps none of their operations: it has ``render`` and
``cancel`` where they have turns, and it is the only one that answers for a
graphics card. One number covering all of them would be a number that has to
change when any of them changes.
"""

WORKER_NAME = "vibevoice"
"""What the handshake calls this process, and what the parent refuses to
proceed without."""

MARKER = "--model-chain-vibevoice-worker"
"""On the command line so this process is recognisable in a task manager and
by the stray sweep. Never a path, never a script line: a command line is world
readable on most systems."""

MAX_HEADER = 1 << 20
MAX_PAYLOAD = 256 << 20
"""A forty-five-minute render at 24 kHz PCM16 is about 130 MB, and the voice
samples of a request are a few megabytes; the ceiling is twice the longest
render rather than "whatever arrives"."""

SAMPLE_RATE = 24000
"""What VibeVoice produces and what its voice samples have to be. Reported in
the handshake so the parent can refuse a build that says otherwise."""

MAX_SPEAKERS = 4
"""How many distinct speakers one script may have. The model's own limit."""

MIN_NEW_TOKENS = 512
TOKENS_PER_TEXT_TOKEN = 8
"""The generation budget when the caller sets no ``max_new_tokens``.

Speech comes out at 7.5 tokens a second and English text goes in at about 1.3
tokens a word, so ordinary narration uses two to four speech tokens per text
token. Eight is generous on purpose, with a floor for a script of a few words:
the failure this guards against on the tight side is a render that stops
mid-sentence and says nothing about it, and the failure on the loose side is a
runaway generation that costs card time until it is cancelled -- the second is
visible and the first is not. Upstream's own guard, ``max_length_times``, caps
the audio at twice the *prompt* length, which a long script read against a
short voice sample reaches before its last line; it is lifted out of the way so
this budget is the one that binds (section 3 of the design).
"""

PROGRESS_INTERVAL = 0.5
"""Seconds between two progress frames. A frame per step would be seven a
second down a pipe whose other end logs them."""

SHUTDOWN_GRACE = 5.0
"""How long ``shutdown`` waits for a lane that is inside a generation step
before letting the process end. The cancel flag is set first, so a step's
worth of work is the most it waits; bounded because the parent's escalation
is the real limit and a WebUI that will not close is the worse bug."""

_LENGTH = struct.Struct(">I")


class Refusal(ValueError):
    """A sentence this file wrote, and the only kind of message that crosses back.

    Every refusal raised in this process is one of these, and :func:`_safe`
    forwards it verbatim because this file knows what is in it: no path, no
    script line, no sample. A plain ``ValueError`` would not do -- ``json``,
    NumPy and Torch all raise subclasses of it whose messages name a document,
    a shape or a file.
    """


# --------------------------------------------------------------------------- #
# Framing -- byte-identical to the other workers, by agreement not by import
# --------------------------------------------------------------------------- #


def read_frame(stream) -> "tuple[dict, bytes] | None":
    """One request, or ``None`` at end of input.

    ``None`` is how the parent's death arrives when this process is waiting for
    work, and it is not an error: the loop ends, the model is released with the
    process, and the exit status is 0.
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
# Dying with the parent
# --------------------------------------------------------------------------- #


def _containment(parent_pid: int) -> str:
    """Ask the OS to end this process when the parent ends, and report what it got.

    On Linux ``PR_SET_PDEATHSIG`` is set to SIGKILL and then the parent pid is
    re-read. The re-read is the whole point: if the parent died between its
    fork and this line, the death signal it would have sent has already not
    been sent, and this process would sit here forever holding a card.

    On Windows the job object is the parent's to create, and it has already
    done it by the time this runs -- so this side *checks* rather than claims.
    ``IsProcessInJob`` turns "the parent says it arranged containment" into
    evidence from the kernel. The parent, which proved its own job membership
    with real handles, treats this side's answer as corroboration.
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
            return "none"
        if parent_pid and os.getppid() != parent_pid:
            _note("parent went away during start-up")
            raise SystemExit(0)
        return "pdeathsig"
    if os.name == "nt":
        return _in_a_job()
    return "none"


def _in_a_job() -> str:
    """Whether the kernel agrees this process is in a job. Three answers.

    ``job`` the kernel says yes; ``none`` the kernel says no; ``unknown`` the
    question could not be put at all. Three rather than two because folding the
    third into the second once cost a real user a whole speech engine:
    containment had been arranged and was being enforced, and the worker was
    turned away for failing to confirm it. On Windows the parent arranged the
    job itself and reads this answer as a diagnostic, never as a veto.

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


def _note(text: str) -> None:
    """One diagnostic line on stderr, which the parent logs.

    Never a script line, never a path. Everything written here is a class
    name, a count, a duration or a state.
    """
    try:
        sys.stderr.write(f"VibeVoice worker: {text}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _safe(exc: BaseException) -> str:
    """An exception as something that can cross the pipe.

    The class name, and only the class name, for anything this process did not
    raise itself. A library's own message can carry a path, a tensor shape or
    the text that was being spoken, and a worker that forwarded one would put
    a script in the parent's log by accident.

    A :class:`Refusal` is this file's own -- already a sentence, already free of
    anything private -- and is forwarded as written. Nothing else is, and the
    test cannot be ``ValueError``: ``json``, NumPy and Torch all raise
    subclasses of it whose messages routinely name a path or a shape.
    """
    if isinstance(exc, Refusal):
        return str(exc)
    return exc.__class__.__name__


# --------------------------------------------------------------------------- #
# The script and the voices, as the processor wants them
# --------------------------------------------------------------------------- #


def _clean(text) -> str:
    """One entry's text as a single line, the way the demo hands it over.

    Whitespace runs -- newlines above all -- collapse to one space, because the
    processor reads the script a line at a time and drops a line that does not
    begin with ``Speaker N:``. The curly apostrophe is straightened because
    upstream's own demo does exactly that before it prompts the model.
    """
    return " ".join(str(text or "").replace("’", "'").split())


def script_text(script) -> "tuple[str, list[int]]":
    """The model's text and the speaker order its voice samples must follow.

    Returns ``(text, speakers)``: ``text`` is one ``Speaker N: ...`` line per
    entry with words, joined by newlines; ``speakers`` is every distinct
    speaker number the caller used, ascending. The voice arrays handed to the
    processor must be in exactly that order, because the processor binds its
    i-th sample to the line numbered i (see the module docstring).

    The numbers in ``text`` are dense from 1 -- the caller's own numbers when
    those are already dense, which is the ordinary case, and the position in
    ``speakers`` plus one otherwise. A script that names Speakers 1 and 3 is
    therefore spoken by Speakers 1 and 2, with Speaker 3's sample second: the
    alternative is the processor giving the third speaker no voice.
    """
    entries = []
    for entry in script or ():
        if not isinstance(entry, dict):
            raise Refusal("a script entry is not an object")
        speaker = entry.get("speaker")
        if isinstance(speaker, bool):
            raise Refusal("a script entry has no speaker number")
        try:
            number = int(speaker)
        except (TypeError, ValueError):
            raise Refusal("a script entry has no speaker number") from None
        if number < 1 or number > MAX_SPEAKERS:
            raise Refusal(f"Speaker {number} is outside 1 to {MAX_SPEAKERS}")
        text = _clean(entry.get("text"))
        if not text:
            continue
        entries.append((number, text))
    if not entries:
        raise Refusal("the script has no words")
    speakers = sorted({number for number, _text in entries})
    numbered = {number: index + 1 for index, number in enumerate(speakers)}
    lines = [f"Speaker {numbered[number]}: {text}" for number, text in entries]
    return "\n".join(lines), speakers


def voice_arrays(voices, payload: bytes, speakers) -> list:
    """One float32 array per speaker in ``speakers``, cut from the payload.

    ``voices`` is the header's list of ``{"speaker", "offset", "count",
    "rate"}`` and ``payload`` the float32 little-endian mono PCM they index
    into, in samples. Every speaker the script uses must have exactly one
    sample at 24 kHz that lies inside the payload; a speaker with none is a
    sentence naming the speaker, which is the one thing about a script that is
    safe to say.
    """
    import numpy

    total = len(payload) // 4
    found = {}
    for entry in voices or ():
        if not isinstance(entry, dict):
            raise Refusal("a voice entry is not an object")
        try:
            speaker = int(entry.get("speaker"))
            offset = int(entry.get("offset") or 0)
            count = int(entry.get("count") or 0)
            rate = int(entry.get("rate") or 0)
        except (TypeError, ValueError):
            raise Refusal("a voice entry is not well formed") from None
        if rate != SAMPLE_RATE:
            raise Refusal(f"a voice sample is at {rate} Hz rather than {SAMPLE_RATE}")
        if offset < 0 or count <= 0 or offset + count > total:
            raise Refusal("a voice sample lies outside the audio that was sent")
        if speaker in found:
            raise Refusal(f"Speaker {speaker} has two voice samples")
        array = numpy.frombuffer(payload, dtype="<f4", count=count,
                                 offset=offset * 4).astype(numpy.float32)
        if not numpy.all(numpy.isfinite(array)):
            raise Refusal("a voice sample contains values that are not numbers")
        found[speaker] = array
    for speaker in speakers:
        if speaker not in found:
            raise Refusal(f"Speaker {speaker} has no voice sample")
    return [found[speaker] for speaker in speakers]


def pcm16(samples) -> bytes:
    """Float samples as little-endian PCM16, clipped to [-1, 1].

    Clipped rather than limited: this is a model's finished output at the level
    it chose, not a volume control, and a sample beyond unity is a rare
    overshoot rather than a setting somebody turned up.
    """
    import numpy

    array = numpy.asarray(samples, dtype=numpy.float32).reshape(-1)
    return (numpy.clip(array, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def encode_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Already-quantised PCM16 as a mono WAV, in memory."""
    import contextlib

    buffer = io.BytesIO()
    with contextlib.closing(wave.open(buffer, "wb")) as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(pcm)
    return buffer.getvalue()


def _float(value, fallback: float) -> float:
    """A finite number, or ``fallback`` for an absent one. Anything else is refused."""
    if value is None or isinstance(value, bool):
        return float(fallback)
    try:
        found = float(value)
    except (TypeError, ValueError):
        raise Refusal("a number in the request is not a number") from None
    if math.isnan(found) or math.isinf(found):
        raise Refusal("a number in the request is not a number")
    return found


def _optional_int(value, what: str) -> "int | None":
    """``None`` for an absent value, a non-negative int otherwise."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise Refusal(f"{what} is not a whole number")
    try:
        found = int(value)
    except (TypeError, ValueError):
        raise Refusal(f"{what} is not a whole number") from None
    if found < 0:
        raise Refusal(f"{what} is negative")
    return found


class RenderRequest:
    """One ``render`` frame, validated, with its voices already cut out.

    Parsed on the command thread rather than the lane, so a malformed request
    is refused at once instead of after the render already running.
    """

    __slots__ = ("job", "text", "speakers", "voices", "cfg_scale", "seed",
                 "max_new_tokens", "steps")

    def __init__(self, job, text, speakers, voices, cfg_scale, seed, max_new_tokens, steps):
        self.job = job
        self.text = text
        self.speakers = speakers
        self.voices = voices
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.steps = steps

    @classmethod
    def parse(cls, header: dict, payload: bytes) -> "RenderRequest":
        text, speakers = script_text(header.get("script"))
        voices = voice_arrays(header.get("voices"), payload, speakers)
        max_new_tokens = _optional_int(header.get("max_new_tokens"), "max_new_tokens")
        if max_new_tokens == 0:
            max_new_tokens = None
        return cls(job=str(header.get("job") or ""), text=text, speakers=speakers,
                   voices=voices, cfg_scale=_float(header.get("cfg_scale"), 1.3),
                   seed=_optional_int(header.get("seed"), "the seed"),
                   max_new_tokens=max_new_tokens,
                   steps=_optional_int(header.get("steps"), "the step count") or 0)


class Progress:
    """The audio streamer's shape, used to count what has been made so far.

    ``generate`` hands its streamer every decoded chunk as it is made and calls
    ``end`` when a sample finishes; it also breaks out of its loop the moment
    any ``finished_flags`` entry is true. This object never sets one early --
    only ``end`` does, which ``generate`` itself calls at the very points it is
    about to stop anyway -- and it counts samples from the chunk's shape rather
    than copying the chunk off the card, so it costs the render nothing.
    """

    def __init__(self, rate: int, report, interval: float = PROGRESS_INTERVAL):
        self.rate = int(rate)
        self.samples = 0
        self.finished_flags = [False]
        self._report = report
        self._interval = float(interval)
        self._last = 0.0

    def put(self, audio_chunks, sample_indices=None) -> None:
        try:
            count = int(audio_chunks.shape[-1])
        except Exception:  # noqa: BLE001 - a shape this build does not have; no progress then
            return
        self.samples += count
        now = time.monotonic()
        if now - self._last < self._interval:
            return
        self._last = now
        if self._report is not None:
            try:
                self._report({"seconds": self.samples / float(self.rate or 1)})
            except Exception:  # noqa: BLE001 - progress is advisory
                pass

    def end(self, sample_indices=None) -> None:
        if sample_indices is None:
            self.finished_flags = [True for _flag in self.finished_flags]
            return
        for index in sample_indices:
            try:
                found = int(index.item()) if hasattr(index, "item") else int(index)
            except Exception:  # noqa: BLE001 - an index this build does not have
                continue
            if 0 <= found < len(self.finished_flags):
                self.finished_flags[found] = True

    @property
    def seconds(self) -> float:
        return self.samples / float(self.rate or 1)


# --------------------------------------------------------------------------- #
# The engine: Torch, transformers and vibevoice, behind one class
# --------------------------------------------------------------------------- #


class Engine:
    """One VibeVoice model on one card, one generation at a time.

    Everything Torch-shaped is behind this class, and it is constructed once
    per worker. Its two import points -- :meth:`_frameworks` and
    :meth:`_upstream` -- are methods so a test can hand it stand-ins and
    exercise the exact call sequence without a card.
    """

    def __init__(self):
        self.torch = None
        self.numpy = None
        self.model = None
        self.processor = None
        self.model_dir = ""
        self.steps = 0
        self.weights_bytes = 0

    # -- imports ----------------------------------------------------------- #

    def _frameworks(self):
        """Torch and NumPy, imported here and nowhere else in this process."""
        if self.torch is None:
            import numpy
            import torch

            self.torch, self.numpy = torch, numpy
        return self.torch, self.numpy

    def _upstream(self):
        """The two vibevoice classes this file speaks to, resolved by name."""
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        return VibeVoiceForConditionalGenerationInference, VibeVoiceProcessor

    # -- the card ---------------------------------------------------------- #

    def probe(self) -> dict:
        """Which card this process is on, and how much of it is free.

        The UUID is ``torch.cuda.get_device_properties(0).uuid`` as a string;
        a Torch too old to have the attribute gives an empty string, and the
        parent refuses to use a worker that cannot prove its card. No CUDA
        device at all is a refusal with the sentence the parent expects.
        """
        torch, _numpy = self._frameworks()
        if not torch.cuda.is_available():
            raise Refusal("no CUDA device is visible to the worker")
        properties = torch.cuda.get_device_properties(0)
        uuid = getattr(properties, "uuid", None)
        free, total = torch.cuda.mem_get_info(0)
        return {
            "torch": str(getattr(torch, "__version__", "") or ""),
            "cuda": True,
            "device_name": str(getattr(properties, "name", "") or ""),
            "device_uuid": "" if uuid is None else str(uuid),
            "device_index": 0,
            "total_vram_bytes": int(total),
            "free_vram_bytes": int(free),
        }

    def resident(self) -> int:
        """What the caching allocator holds on the card right now."""
        if self.torch is None:
            return 0
        try:
            return int(self.torch.cuda.memory_reserved(0))
        except Exception:  # noqa: BLE001 - no context yet, or no card
            return 0

    def peak(self) -> int:
        if self.torch is None:
            return 0
        try:
            return int(self.torch.cuda.max_memory_reserved(0))
        except Exception:  # noqa: BLE001 - no context yet, or no card
            return 0

    # -- loading ----------------------------------------------------------- #

    def load(self, model_dir: str, steps: int, dtype: str = "bf16",
             attention: str = "sdpa") -> dict:
        """Load the processor and the model from one local directory.

        A second ``load`` of the directory already loaded is a no-op reply
        (its step count is applied, which is a number on the instance). A
        different directory replaces what is loaded.
        """
        wanted = str(model_dir or "")
        steps = int(steps or 0)
        if self.model is not None and wanted == self.model_dir:
            if steps and steps != self.steps:
                self.model.set_ddpm_inference_steps(num_steps=steps)
                self.steps = steps
            return {"resident_bytes": self.resident(), "weights_bytes": self.weights_bytes,
                    "load_seconds": 0.0, "already_loaded": True}
        if not wanted or not os.path.isdir(wanted):
            raise Refusal("the model directory does not exist")
        torch, _numpy = self._frameworks()
        precision = {"bf16": torch.bfloat16, "fp16": torch.float16,
                     "fp32": torch.float32}.get(str(dtype or "bf16"))
        if precision is None:
            raise Refusal("that precision is not one this worker loads")
        if self.model is not None:
            self.unload()
        model_class, processor_class = self._upstream()
        began = time.monotonic()
        processor = processor_class.from_pretrained(wanted)
        model = model_class.from_pretrained(
            wanted, torch_dtype=precision, device_map="cuda",
            attn_implementation=str(attention or "sdpa"))
        model.eval()
        model.set_ddpm_inference_steps(num_steps=steps or None)
        weights = 0
        for parameter in model.parameters():
            weights += int(parameter.numel()) * int(parameter.element_size())
        self.model = model
        self.processor = processor
        self.model_dir = wanted
        self.steps = steps
        self.weights_bytes = weights
        elapsed = time.monotonic() - began
        _note(f"loaded in {elapsed:.1f} s — {weights / float(1 << 30):.1f} GiB of weights, "
              f"{steps or 'default'} step(s), {str(dtype or 'bf16')}, {attention or 'sdpa'}")
        return {"resident_bytes": self.resident(), "weights_bytes": weights,
                "load_seconds": elapsed}

    def unload(self) -> dict:
        """Let go of the model and give the card its memory back."""
        self.model = None
        self.processor = None
        self.model_dir = ""
        self.steps = 0
        self.weights_bytes = 0
        gc.collect()
        if self.torch is not None:
            try:
                self.torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - nothing to empty
                pass
        return {"resident_bytes": self.resident()}

    # -- rendering --------------------------------------------------------- #

    def render(self, request: RenderRequest, cancelled: threading.Event,
               on_progress=None) -> "tuple[dict, bytes]":
        """One script through the model, as the demo runs it.

        ``cancelled`` is read at the top of every generation step through
        ``stop_check_fn``; a cancel therefore lands within one step, and the
        reply says so and carries whatever audio there was.
        """
        if self.model is None or self.processor is None:
            raise Refusal("no model is loaded")
        torch, numpy = self._frameworks()
        if request.steps and request.steps != self.steps:
            self.model.set_ddpm_inference_steps(num_steps=int(request.steps))
            self.steps = int(request.steps)

        inputs = self.processor(
            text=[request.text],
            voice_samples=[list(request.voices)],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to("cuda")
        if request.seed is not None:
            torch.manual_seed(int(request.seed))
            torch.cuda.manual_seed_all(int(request.seed))

        prompt_length = int(inputs["input_ids"].shape[-1])
        max_new_tokens = request.max_new_tokens or self._token_budget(request.text, prompt_length)
        # Upstream stops at max_length_times * prompt length as well as at
        # max_new_tokens; the multiplier is set so the token budget is the one
        # that binds (see TOKENS_PER_TEXT_TOKEN).
        times = max(2, int(math.ceil(max_new_tokens / float(max(1, prompt_length)))) + 1)
        progress = Progress(SAMPLE_RATE, on_progress)

        torch.cuda.reset_peak_memory_stats(0)
        began = time.monotonic()
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                cfg_scale=float(request.cfg_scale),
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                stop_check_fn=cancelled.is_set,
                audio_streamer=progress,
                max_length_times=times,
                show_progress_bar=False,
            )
            speech = None
            found = getattr(outputs, "speech_outputs", None)
            if found:
                speech = found[0]
            if speech is None:
                samples = numpy.zeros(0, dtype=numpy.float32)
            else:
                samples = speech.detach().reshape(-1).to(torch.float32).cpu().numpy()
            render_seconds = time.monotonic() - began
            sequences = getattr(outputs, "sequences", None)
            tokens = 0
            if sequences is not None:
                tokens = max(0, int(sequences.shape[-1]) - prompt_length)
            capped = False
            reached = getattr(outputs, "reach_max_step_sample", None)
            if reached is not None:
                try:
                    capped = bool(reached.any().item()) if hasattr(reached, "any") \
                        else bool(any(reached))
                except Exception:  # noqa: BLE001 - a shape this build does not have
                    capped = False
            if not capped and not cancelled.is_set() and tokens >= int(max_new_tokens):
                # Upstream's generate never raises its own flag when the budget
                # runs out: its loop's range ends one step before the check
                # that would. Having used the whole budget without ending on
                # end-of-speech is what reaching it means.
                capped = _last_token(sequences) != \
                    getattr(self.processor.tokenizer, "eos_token_id", None)
            peak = int(torch.cuda.max_memory_reserved(0))
            pcm = pcm16(samples)
            seconds = len(samples) / float(SAMPLE_RATE)
        finally:
            outputs = None
            inputs = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - nothing to empty
                pass
        if capped:
            _note(f"the render reached its token budget ({max_new_tokens}) before the "
                  f"script ended")
        _note(f"rendered {seconds:.1f} s of audio in {render_seconds:.1f} s — {tokens} "
              f"token(s), peak {peak / float(1 << 30):.1f} GiB"
              f"{', cancelled' if cancelled.is_set() else ''}")
        return {
            "seconds": seconds,
            "sample_rate": SAMPLE_RATE,
            "render_seconds": render_seconds,
            "peak_bytes": peak,
            "resident_bytes": self.resident(),
            "cancelled": bool(cancelled.is_set()),
            "tokens": tokens,
            "capped": capped,
        }, encode_wav(pcm, SAMPLE_RATE)

    def _token_budget(self, text: str, prompt_length: int) -> int:
        """How many tokens a render may generate when the caller set no cap.

        Proportional to the script's own token count, floored for a script of
        a few words, and never past the context the model has left after its
        prompt. See :data:`TOKENS_PER_TEXT_TOKEN`.
        """
        context = 0
        try:
            context = int(self.model.config.decoder_config.max_position_embeddings or 0)
        except Exception:  # noqa: BLE001 - a config without the field
            context = 0
        ceiling = max(1, context - int(prompt_length)) if context else 1 << 15
        try:
            text_tokens = len(self.processor.tokenizer.encode(text, add_special_tokens=False))
        except Exception:  # noqa: BLE001 - a tokenizer without the keyword
            text_tokens = len(str(text).split())
        wanted = max(MIN_NEW_TOKENS, TOKENS_PER_TEXT_TOKEN * int(text_tokens))
        return int(min(ceiling, wanted))


# --------------------------------------------------------------------------- #
# The worker: the lane, the writer lock, and the one render
# --------------------------------------------------------------------------- #


class Worker:
    """The lane and the render in flight. One model, one generation at a time."""

    def __init__(self, stdout):
        self.stdout = stdout
        self.engine = None
        self._jobs: "queue.Queue" = queue.Queue()
        self._write_lock = threading.Lock()
        self._state = threading.Lock()
        self._job = ""
        self._cancel: "threading.Event | None" = None
        self._stopping = False
        self._lane = None

    def start_threads(self) -> None:
        if self._lane is not None:
            return
        self._lane = threading.Thread(target=self._lane_loop, name="vibevoice-lane",
                                      daemon=True)
        self._lane.start()

    def send(self, header: dict, payload: bytes = b"") -> None:
        """Write one frame, under the lock the lane and the loop share."""
        with self._write_lock:
            write_frame(self.stdout, header, payload)

    def _lane_loop(self) -> None:
        while True:
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                if self._stopping:
                    return
                continue
            if job is None:
                return
            try:
                job()
            except Exception as exc:  # noqa: BLE001 - never fatal to the lane
                _note(f"request failed: {exc.__class__.__name__}")

    def queue(self, call) -> None:
        self._jobs.put(call)

    # -- the one render ---------------------------------------------------- #

    def begin_render(self, job: str) -> threading.Event:
        with self._state:
            if self._job:
                raise Refusal("one render at a time")
            self._job = str(job or "-")
            self._cancel = threading.Event()
            return self._cancel

    def end_render(self) -> None:
        with self._state:
            self._job = ""
            self._cancel = None

    def cancel(self, job: str) -> bool:
        """Set the flag the running render's ``stop_check_fn`` reads.

        ``True`` when a render is in flight and ``job`` names it (or is
        empty); ``False`` otherwise, which is not an error -- the render it
        meant may simply have finished.
        """
        with self._state:
            if self._cancel is None:
                return False
            if job and job != self._job:
                return False
            self._cancel.set()
            return True

    def rendering(self) -> str:
        with self._state:
            return self._job

    def snapshot(self) -> dict:
        engine = self.engine
        with self._state:
            job = self._job
        return {
            "loaded": bool(engine is not None and engine.model is not None),
            "rendering": bool(job),
            "job": job,
            "resident_bytes": engine.resident() if engine is not None else 0,
            "peak_bytes": engine.peak() if engine is not None else 0,
            "model_dir": engine.model_dir if engine is not None else "",
        }


# --------------------------------------------------------------------------- #
# The command loop
# --------------------------------------------------------------------------- #


def serve(stdin, stdout, engine_factory=None) -> int:
    """Read frames until end of input. Returns the process exit status.

    The loop does as little as it can. ``status`` and ``cancel`` are answered
    on this thread while the lane is inside the model; ``load``, ``unload``
    and ``render`` are handed to the lane, which runs them one at a time.
    """
    factory = engine_factory or (lambda: Engine())
    worker = Worker(stdout)

    def reply(request_id, header: dict, payload: bytes = b"") -> None:
        found = dict(header)
        found["id"] = request_id
        worker.send(found, payload)

    def refuse(request_id, exc: BaseException) -> None:
        reply(request_id, {"ok": False, "error": _safe(exc),
                           "refusal": isinstance(exc, Refusal)})

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
                worker.cancel("")
                reply(request_id, {"ok": True})
                _note("stopping on request")
                return 0

            if operation == "init":
                try:
                    containment = _containment(int(header.get("parent_pid") or 0))
                    engine = factory()
                    card = engine.probe()
                    worker.engine = engine
                    worker.start_threads()
                    found = {
                        "ok": True,
                        "protocol": PROTOCOL_VERSION,
                        "worker": WORKER_NAME,
                        "python": platform.python_version(),
                    }
                    found.update(card)
                    found.update({
                        "containment": containment,
                        "vibevoice": _package_version("vibevoice"),
                        "transformers": _package_version("transformers"),
                        "sample_rate": SAMPLE_RATE,
                    })
                    reply(request_id, found)
                    _note(f"ready — {card.get('device_name') or 'unnamed device'}, "
                          f"{int(card.get('free_vram_bytes') or 0) / float(1 << 30):.1f} of "
                          f"{int(card.get('total_vram_bytes') or 0) / float(1 << 30):.1f} GiB "
                          f"free, containment {containment}")
                except SystemExit:
                    raise
                except Exception as exc:
                    worker.engine = None
                    refuse(request_id, exc)
                    _note(f"could not start: {_safe(exc)}")
                continue

            if worker.engine is None:
                reply(request_id, {"ok": False, "error": "runtime is not initialised",
                                   "refusal": True})
                continue

            if operation == "status":
                # On this thread on purpose: "how much of the card does the
                # worker hold" is asked every hundred milliseconds during an
                # eviction, and an answer queued behind a load is no answer.
                found = {"ok": True}
                found.update(worker.snapshot())
                reply(request_id, found)

            elif operation == "load":
                _queue(worker, reply, request_id,
                       lambda rid=request_id, head=header: _do_load(worker, reply, rid, head))

            elif operation == "unload":
                _queue(worker, reply, request_id,
                       lambda rid=request_id: _do_unload(worker, reply, rid))

            elif operation == "render":
                try:
                    request = RenderRequest.parse(header, payload)
                    cancelled = worker.begin_render(request.job)
                except Exception as exc:  # noqa: BLE001 - answered rather than raised
                    refuse(request_id, exc)
                    continue
                _queue(worker, reply, request_id,
                       lambda rid=request_id, req=request, flag=cancelled:
                       _do_render(worker, reply, rid, req, flag))

            elif operation == "cancel":
                reply(request_id, {"ok": True,
                                   "cancelled": worker.cancel(str(header.get("job") or ""))})

            else:
                reply(request_id, {"ok": False, "error": "unknown operation",
                                   "refusal": True})
    finally:
        worker._stopping = True
        worker.cancel("")
        lane = worker._lane
        if lane is not None and lane.is_alive():
            lane.join(timeout=SHUTDOWN_GRACE)


def _queue(worker: Worker, reply, request_id, call) -> None:
    """Put one request on the lane, and guarantee it is answered.

    A request that reached the lane and then raised would otherwise be a
    request nobody ever answered, and the parent's only recourse a timeout
    minutes later -- so the failure is turned into a reply here, in the
    class-name-only form :func:`_safe` produces.
    """

    def run():
        try:
            call()
        except Exception as exc:  # noqa: BLE001 - answered rather than raised
            _note(f"request failed: {_safe(exc)}")
            try:
                reply(request_id, {"ok": False, "error": _safe(exc),
                                   "refusal": isinstance(exc, Refusal)})
            except Exception:
                pass

    worker.queue(run)


def _do_load(worker: Worker, reply, request_id, header: dict) -> None:
    found = worker.engine.load(str(header.get("model_dir") or ""),
                               int(header.get("steps") or 0),
                               str(header.get("dtype") or "bf16"),
                               str(header.get("attention") or "sdpa"))
    answer = {"ok": True}
    answer.update(found)
    reply(request_id, answer)


def _do_unload(worker: Worker, reply, request_id) -> None:
    answer = {"ok": True}
    answer.update(worker.engine.unload())
    reply(request_id, answer)


def _do_render(worker: Worker, reply, request_id, request: RenderRequest,
               cancelled: threading.Event) -> None:
    """One render on the lane, and the render slot freed whatever happens."""
    try:
        def report(progress: dict) -> None:
            worker.send({"id": request_id, "progress": dict(progress)})

        found, audio = worker.engine.render(request, cancelled, report)
    finally:
        # Freed *before* the reply goes out: a parent that sends its next
        # render the moment it reads this one's reply -- the Voice Box renders
        # a script's sections back to back -- would otherwise race this line
        # and be refused "one render at a time" for a render that had ended.
        worker.end_render()
    answer = {"ok": True}
    answer.update(found)
    reply(request_id, answer, audio)


def _last_token(sequences) -> "int | None":
    """The last token of the first sequence ``generate`` returned, or ``None``."""
    try:
        return int(sequences[0, -1].item())
    except Exception:  # noqa: BLE001 - no sequences, or a shape this build does not have
        return None


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


# --------------------------------------------------------------------------- #
# Running this file directly
# --------------------------------------------------------------------------- #


def selftest() -> int:
    """Prove the staged runtime imports. One JSON line out.

    Run by the installer against a *staged* interpreter before anything is
    promoted, with no model: the packages are imported, the two classes this
    file speaks to are resolved, and the versions are printed. ``cuda`` says
    whether a card is visible to that interpreter; it is reported rather than
    required, because the installer decides what a machine without one means.
    """
    report = {"ok": False, "error": "", "cuda": False}
    try:
        import torch

        report["torch"] = str(getattr(torch, "__version__", ""))
        report["numpy"] = _package_version("numpy")
        report["transformers"] = _package_version("transformers")
        Engine()._upstream()
        report["vibevoice"] = _package_version("vibevoice")
        report["cuda"] = bool(torch.cuda.is_available())
        report["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the report is the answer
        report["error"] = f"{exc.__class__.__name__}: {exc}"
    sys.stdout.write(json.dumps(report) + "\n")
    sys.stdout.flush()
    return 0 if report["ok"] else 1


def _claim_stdout():
    """The pipe to the parent, taken away from everything that prints.

    The protocol runs over this process's standard output, and upstream code
    prints to it: ``generate`` has an unconditional ``print`` for a render that
    reaches its length limit (not reached with the arguments this file passes
    today), and a library's print is not this file's to rule out. A line of
    text in the middle of the frame stream is a header length the parent
    cannot read, and the parent would lose its worker over it. So descriptor 1
    is duplicated for the protocol alone, and descriptor 1 itself -- with
    Python's ``sys.stdout`` -- is pointed at standard error, which the parent
    reads line by line into its log. Anything that prints now prints there,
    whether it prints from Python or from a native library.
    """
    sys.stdout.flush()
    descriptor = os.dup(sys.stdout.fileno())
    if os.name == "nt":
        import msvcrt

        msvcrt.setmode(descriptor, os.O_BINARY)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return os.fdopen(descriptor, "wb")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(MARKER, action="store_true", dest="marker")
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--selftest", action="store_true")
    found, _rest = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    if found.selftest:
        return selftest()
    return serve(sys.stdin.buffer, _claim_stdout())


if __name__ == "__main__":
    raise SystemExit(main())
