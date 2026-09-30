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
    loading one VibeVoice model -- the long-form 7B or the Realtime 0.5B --
        and its processor from a local directory, at full precision or with
        its language model quantised, with or without a LoRA
    turning a script and its voices -- recordings for the 7B, preset voice
        prompts for the 0.5B -- into the model's prompt
    one generation at a time, cancellable between steps
    the audio it produced, as one PCM16 WAV or as frames while it is made

The parent owns the turns on each card, the settings, the samples and the
outputs; this process owns inference and nothing else. It is handed a directory
it did not choose and audio it did not record, and it validates both anyway,
because it is the process that would crash.

The upstream surface this file binds to
---------------------------------------
Two packages share one runtime: the community ``vibevoice`` 0.0.1 wheel (the
preserved original, which has the long-form 7B) with five files of Microsoft's
repository copied over it (which bring the Realtime 0.5B, whose code the wheel
never had). The calls, written down here because they are the whole contract:

    ``VibeVoiceProcessor.from_pretrained(model_dir)`` and
    ``VibeVoiceStreamingProcessor.from_pretrained(model_dir)`` -- a local
        directory holding ``preprocessor_config.json``, whose
        ``language_model_pretrained_name`` names the local tokenizer directory
        the installer wrote (the name must contain ``qwen``: the processor
        picks its tokenizer class by that substring);
    ``VibeVoiceForConditionalGenerationInference.from_pretrained`` and
        ``VibeVoiceStreamingForConditionalGenerationInference.from_pretrained``
        with ``torch_dtype``, ``device_map`` and ``attn_implementation`` (and,
        for a quantised 7B, ``quantization_config``), then ``eval()`` and
        ``set_ddpm_inference_steps(num_steps=...)``; the 0.5B's noise scheduler
        is replaced first, exactly as Microsoft's own web demo does;
    ``processor(text=[script], voice_samples=[[array, ...]], padding=True,
        return_tensors="pt", return_attention_mask=True)`` for the 7B, and
        ``processor.process_input_with_cached_prompt(text=..., cached_prompt=
        preset, ...)`` for the 0.5B, whose voice is a prompt already run
        through the model (a *preset*), never a recording;
    ``model.generate(**inputs, ...)``, whose ``speech_outputs[0]`` is the audio
        at 24 kHz, whose ``stop_check_fn`` is read at the top of every
        generation step, and whose ``audio_streamer`` is handed every decoded
        chunk the moment it exists -- a real
        ``vibevoice.modular.streamer.AudioStreamer`` when the parent asked for
        the audio as it is made;
    ``PeftModel.from_pretrained(module, folder, is_trainable=False)`` for a
        LoRA, and ``transformers.BitsAndBytesConfig`` for 8-bit and NF4.

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

The Realtime model has none of that: it speaks one voice, reads plain text,
and takes its voice from a preset prompt file. Microsoft withholds the code
that makes such a file from a recording, so the 0.5B cannot clone a Voice Box
sample, and a render that asks it to is refused rather than approximated.

What is loaded, and when a load is a reload
-------------------------------------------
A load names an *identity*: the model directory, its kind (``longform`` or
``realtime``), the precision (``bf16``, ``int8``, ``nf4``), the LoRA folder and
its strength, and the device. Asking again for the identity already loaded is
a no-op reply; anything else unloads first, because a quantised model cannot
be turned back into a full one and a LoRA's strength is baked into every layer
it touched. Every refusal about the request itself -- a precision the model
does not take, a LoRA for the Realtime model, 8-bit without CUDA -- is made
*before* anything is unloaded, so a bad request never costs the card the model
it already had.

One render at a time, and cancellation between steps
----------------------------------------------------
One model instance serves one generation at a time, on one lane thread, and a
second ``render`` while one runs is refused rather than queued. The command
loop itself does no work: ``cancel`` and ``status`` are answered on it while
the lane is inside the model, which is the whole reason the loop stays free.
Cancellation is cooperative and honest about it: ``stop_check_fn`` is read at
the top of every generation step, so a cancel lands within one step and the
reply carries whatever audio was made before it, marked ``cancelled``. A
"step" is one speech frame for the 7B; the Realtime model reads the flag once
per window (five text tokens and six speech frames), so it may make up to six
more frames before it stops -- none of which a streamed render sends.

A streamed render runs ``generate`` on a thread of its own, as Microsoft's web
demo does, while the lane reads the streamer and writes one audio frame per
chunk down the pipe; the final reply then carries no WAV, because every sample
has already been sent.

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
import copy
import gc
import io
import json
import math
import os
import platform
import queue
import re
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

Still 1 after the Realtime model, quantisation, LoRAs and streamed frames were
added: every new field is optional with the old meaning as its default, and a
parent that sends none of them gets exactly what it got before.
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
"""How many distinct speakers one script may have. The 7B's own limit."""

KIND_LONGFORM = "longform"
KIND_REALTIME = "realtime"
KINDS = (KIND_LONGFORM, KIND_REALTIME)
"""The two models this worker loads. ``longform`` is the 7B (up to four
speakers, voices cloned from recordings); ``realtime`` is the 0.5B (one
speaker, preset voices only)."""

PRECISIONS = ("bf16", "int8", "nf4")
QUANTISED = ("int8", "nf4")
"""``bf16`` is the model as published. ``int8`` and ``nf4`` quantise the 7B's
language model with bitsandbytes and nothing else (:data:`QUANTISE_SKIP`); the
0.5B is only ever loaded at full precision."""

QUANTISE_SKIP = ("acoustic_tokenizer", "semantic_tokenizer", "prediction_head",
                 "acoustic_connector", "semantic_connector", "lm_head")
"""The modules left in bf16 when the 7B is quantised: everything except the
language model's own layers. The tokenizers and the diffusion head are small
and turn latents into sound, where rounding is audible; the language model is
where the gigabytes are. Matched by name the way transformers matches
``llm_int8_skip_modules``: a module whose dotted path contains one of these."""

DEVICES = ("cuda", "cpu")
"""Where a load may put the model. The parent only ever asks for ``cuda``;
``cpu`` exists so ``tools/smoke_vibevoice_worker.py`` can drive this very file
on a machine without a card. On the CPU the model is float32, because that is
what every CPU kernel the model reaches supports."""

LORA_SCALE_MAX = 2.0
"""The strongest a LoRA may be applied: twice its trained strength. Zero turns
it off without unloading anything the parent would notice."""

LORA_ADAPTER_FOLDERS = ("", "lora", "language_model")
"""Where a LoRA folder may keep its language-model adapter, in the order they
are looked at. The installer normalises a library entry so the adapter is at
its root; the two subfolders are what community training scripts write, and
reading them costs nothing."""

LORA_PARTS = (("diffusion_head", "prediction_head", "diffusion head"),
              ("acoustic_connector", "acoustic_connector", "acoustic connector"),
              ("semantic_connector", "semantic_connector", "semantic connector"))
"""The optional parts of a 7B LoRA beside the language-model adapter: the
subfolder, the ``model.model`` attribute it replaces or wraps, and the words a
refusal names it by."""

FULL_WEIGHT_FILES = ("model.safetensors", "diffusion_head_full.bin", "pytorch_model.bin")
"""A LoRA part that is a whole module's weights rather than an adapter pair."""

LONGFORM_CFG = 1.3
REALTIME_CFG = 1.5
REALTIME_STEPS = 5
"""The guidance and step count each model is published with, used when a
request leaves them to the model."""

PRESET_STEM = re.compile(r"[A-Za-z0-9_-]{1,64}")
"""What a preset voice may be called. The stem becomes a file name inside the
model's own ``voices`` folder, so nothing that could leave that folder -- a
separator, a dot, a drive letter -- is ever let through."""

PRESET_KEYS = ("lm", "tts_lm", "neg_lm", "neg_tts_lm")
"""The four prefilled passes a preset voice holds, as the 0.5B's ``generate``
reads them from ``all_prefilled_outputs``."""

MIN_NEW_TOKENS = 512
TOKENS_PER_TEXT_TOKEN = 8
"""The 7B's generation budget when the caller sets no ``max_new_tokens``.

Speech comes out at 7.5 tokens a second and English text goes in at about 1.3
tokens a word, so ordinary narration uses two to four speech tokens per text
token. Eight is generous on purpose, with a floor for a script of a few words:
the failure this guards against on the tight side is a render that stops
mid-sentence and says nothing about it, and the failure on the loose side is a
runaway generation that costs card time until it is cancelled -- the second is
visible and the first is not. Upstream's own guard, ``max_length_times``, caps
the audio at twice the *prompt* length, which a long script read against a
short voice sample reaches before its last line; it is lifted out of the way so
this budget is the one that binds (section 3 of the design). The Realtime
model is given no budget: its own default is the context it has left, and its
end-of-speech classifier is what stops it.
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


def _straighten(text: str) -> str:
    """Curly double quotes as straight ones, as the Realtime model's demo does.

    Microsoft's file demo straightens ``’``, ``“`` and ``”`` before the text
    reaches the 0.5B's tokenizer; :func:`_clean` already took the apostrophe.
    """
    return str(text or "").replace("“", '"').replace("”", '"')


def _entries(script) -> "list[tuple[int, str]]":
    """The script's entries that have words, as ``(speaker, text)``, validated."""
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
    return entries


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
    entries = _entries(script)
    speakers = sorted({number for number, _text in entries})
    numbered = {number: index + 1 for index, number in enumerate(speakers)}
    lines = [f"Speaker {numbered[number]}: {text}" for number, text in entries]
    return "\n".join(lines), speakers


def plain_text(script) -> str:
    """The script as the Realtime model reads it: prose, one voice, no labels.

    The 0.5B's processor takes a single text and no ``Speaker N:`` lines, so
    the entries are joined with spaces, and its demo's quote straightening is
    applied on top of :func:`_clean`'s.
    """
    return _straighten(" ".join(text for _number, text in _entries(script)))


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


def _split_voices(voices) -> "tuple[list, dict]":
    """The header's voices as ``(sample entries, {speaker: preset stem})``.

    A preset entry is ``{"speaker": n, "preset": "<stem>"}``; everything else
    is a sample entry for :func:`voice_arrays` to check. The stem is checked
    here, before it could ever become part of a file name.
    """
    samples, presets = [], {}
    for entry in voices or ():
        if not isinstance(entry, dict):
            raise Refusal("a voice entry is not an object")
        if "preset" not in entry:
            samples.append(entry)
            continue
        speaker = entry.get("speaker")
        if isinstance(speaker, bool):
            raise Refusal("a voice entry is not well formed")
        try:
            number = int(speaker)
        except (TypeError, ValueError):
            raise Refusal("a voice entry is not well formed") from None
        stem = entry.get("preset")
        if not isinstance(stem, str) or not PRESET_STEM.fullmatch(stem):
            raise Refusal("a preset voice's name may hold only letters, digits, '-' and '_'")
        if number in presets:
            raise Refusal(f"Speaker {number} has two voices")
        presets[number] = stem
    return samples, presets


def pcm16(samples) -> bytes:
    """Float samples as little-endian PCM16, clipped to [-1, 1].

    Clipped rather than limited: this is a model's finished output at the level
    it chose, not a volume control, and a sample beyond unity is a rare
    overshoot rather than a setting somebody turned up. A streamed chunk is
    clipped the same way, so a streamed render and a whole one are the same
    samples.
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
    is refused at once instead of after the render already running. What the
    loaded model makes of it -- a preset for the 7B, a recording for the 0.5B
    -- is the engine's to refuse, because only the engine knows which is loaded.
    """

    __slots__ = ("job", "text", "speakers", "voices", "cfg_scale", "seed",
                 "max_new_tokens", "steps", "presets", "plain", "stream")

    def __init__(self, job, text, speakers, voices, cfg_scale, seed, max_new_tokens, steps,
                 presets=None, plain="", stream=False):
        self.job = job
        self.text = text
        self.speakers = speakers
        self.voices = voices
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.presets = dict(presets or {})
        self.plain = plain
        self.stream = bool(stream)

    @classmethod
    def parse(cls, header: dict, payload: bytes) -> "RenderRequest":
        text, speakers = script_text(header.get("script"))
        samples, presets = _split_voices(header.get("voices"))
        if presets:
            if samples:
                raise Refusal("a render names both preset voices and voice samples")
            for speaker in speakers:
                if speaker not in presets:
                    raise Refusal(f"Speaker {speaker} has no voice")
            voices = []
            presets = {speaker: presets[speaker] for speaker in speakers}
        else:
            voices = voice_arrays(samples, payload, speakers)
        max_new_tokens = _optional_int(header.get("max_new_tokens"), "max_new_tokens")
        if max_new_tokens == 0:
            max_new_tokens = None
        cfg_scale = header.get("cfg_scale")
        return cls(job=str(header.get("job") or ""), text=text, speakers=speakers,
                   voices=voices,
                   cfg_scale=None if cfg_scale is None else _float(cfg_scale, LONGFORM_CFG),
                   seed=_optional_int(header.get("seed"), "the seed"),
                   max_new_tokens=max_new_tokens,
                   steps=_optional_int(header.get("steps"), "the step count") or 0,
                   presets=presets, plain=plain_text(header.get("script")),
                   stream=header.get("stream") is True)


class Progress:
    """The audio streamer's shape, used to count what has been made so far.

    ``generate`` hands its streamer every decoded chunk as it is made and calls
    ``end`` when a sample finishes; the 7B also breaks out of its loop the
    moment any ``finished_flags`` entry is true. This object never sets one
    early -- only ``end`` does, which ``generate`` itself calls at the very
    points it is about to stop anyway -- and it counts samples from the chunk's
    shape rather than copying the chunk off the card, so it costs the render
    nothing. A chunk handed over after the end is not counted, which is what
    the real streamer does with it too: the Realtime model finishes its speech
    window after its end-of-speech decision and keeps none of that audio.
    """

    def __init__(self, rate: int, report, interval: float = PROGRESS_INTERVAL):
        self.rate = int(rate)
        self.samples = 0
        self.chunks = 0
        self.finished_flags = [False]
        self._report = report
        self._interval = float(interval)
        self._last = 0.0

    def put(self, audio_chunks, sample_indices=None) -> None:
        if all(self.finished_flags):
            return
        try:
            count = int(audio_chunks.shape[-1])
        except Exception:  # noqa: BLE001 - a shape this build does not have; no progress then
            return
        self.samples += count
        self.chunks += 1
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
    per worker. Its import points -- :meth:`_frameworks`, :meth:`_upstream`,
    :meth:`_upstream_realtime`, :meth:`_streamer_class`,
    :meth:`_quantisation_class`, :meth:`_peft`, :meth:`_preset_classes` and
    :meth:`_read_state` -- are methods so a test can hand it stand-ins and
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
        self.kind = ""
        self.precision = ""
        self.lora_dir = ""
        self.lora_scale = 1.0
        self.lora_parts = ()
        self.lora_layers = 0
        self.device = ""
        self.voices_dir = ""
        self._presets: dict = {}

    # -- imports ----------------------------------------------------------- #

    def _frameworks(self):
        """Torch and NumPy, imported here and nowhere else in this process."""
        if self.torch is None:
            import numpy
            import torch

            self.torch, self.numpy = torch, numpy
        return self.torch, self.numpy

    def _upstream(self):
        """The two vibevoice classes the 7B is loaded with, resolved by name."""
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        return VibeVoiceForConditionalGenerationInference, VibeVoiceProcessor

    def _upstream_realtime(self):
        """The two classes the 0.5B is loaded with, from Microsoft's overlay.

        The community wheel has no streaming model; the installer copies five
        of Microsoft's files over it. A runtime installed before that has
        neither, and says so as a sentence rather than as an import error.
        """
        try:
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_streaming_processor import (
                VibeVoiceStreamingProcessor,
            )
        except ImportError:
            raise Refusal("the runtime was installed before the Realtime model was added "
                          "— reinstall it") from None
        return VibeVoiceStreamingForConditionalGenerationInference, VibeVoiceStreamingProcessor

    def _streamer_class(self):
        """Upstream's own audio streamer: a queue per sample, ended with ``None``."""
        from vibevoice.modular.streamer import AudioStreamer

        return AudioStreamer

    def _quantisation_class(self):
        """``BitsAndBytesConfig``, once bitsandbytes itself has imported."""
        try:
            import bitsandbytes  # noqa: F401 - imported to prove it loads
            from transformers import BitsAndBytesConfig
        except Exception:  # noqa: BLE001 - any failure is the same missing feature
            raise Refusal("the runtime was installed before quantisation was added — "
                          "reinstall it") from None
        return BitsAndBytesConfig

    def _peft(self):
        """``(LoraConfig, PeftModel, LoraLayer)`` from PEFT."""
        try:
            from peft import LoraConfig, PeftModel
            from peft.tuners.lora import LoraLayer
        except Exception:  # noqa: BLE001 - any failure is the same missing feature
            raise Refusal("the runtime was installed before LoRA support was added — "
                          "reinstall it") from None
        return LoraConfig, PeftModel, LoraLayer

    def _preset_classes(self):
        """The two classes a preset voice file is allowed to contain."""
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        return BaseModelOutputWithPast, DynamicCache

    def _read_state(self, path: str):
        """A whole module's weights from a LoRA part, read without running code."""
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(path, device="cpu")
        torch, _numpy = self._frameworks()
        return torch.load(path, map_location="cpu", weights_only=True)

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

    def _on_card(self) -> bool:
        """Whether the loaded model lives on the card. Everything memory-shaped
        is a CUDA question, and on the CPU the honest answer to it is zero."""
        return self.device != "cpu"

    def resident(self) -> int:
        """What the caching allocator holds on the card right now."""
        if self.torch is None or not self._on_card():
            return 0
        try:
            return int(self.torch.cuda.memory_reserved(0))
        except Exception:  # noqa: BLE001 - no context yet, or no card
            return 0

    def peak(self) -> int:
        if self.torch is None or not self._on_card():
            return 0
        try:
            return int(self.torch.cuda.max_memory_reserved(0))
        except Exception:  # noqa: BLE001 - no context yet, or no card
            return 0

    def _reset_peak(self) -> None:
        if self._on_card():
            self.torch.cuda.reset_peak_memory_stats(0)

    def _empty_cache(self) -> None:
        if self.torch is None or not self._on_card():
            return
        try:
            self.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - nothing to empty
            pass

    def _dtype(self):
        """The model's floating type: bf16 on the card, float32 on the CPU."""
        torch = self.torch
        return torch.bfloat16 if self.device != "cpu" else torch.float32

    def identity(self) -> "tuple | None":
        """What is loaded, as the tuple a load is compared against, or ``None``."""
        if self.model is None:
            return None
        return (self.model_dir, self.kind, self.precision, self.lora_dir,
                float(self.lora_scale), self.device)

    # -- loading ----------------------------------------------------------- #

    def load(self, model_dir: str, steps: int, dtype: str = "bf16",
             attention: str = "sdpa", *, kind: str = KIND_LONGFORM, precision: str = "",
             lora_dir=None, lora_scale=1.0, voices_dir: str = "",
             device: str = "cuda") -> dict:
        """Load the processor and the model from one local directory.

        The identity is ``(model_dir, kind, precision, lora_dir, lora_scale,
        device)``. A second ``load`` of the identity already loaded is a no-op
        reply (its step count and voices folder are applied, which are values on
        the instance). A different identity unloads what is loaded first -- but
        only after every refusal this request could earn has been checked, so a
        request that was going to be refused never costs the card its model.

        ``dtype`` is the phase-one name for the precision and is read only when
        ``precision`` is absent; ``bf16`` was the only value any parent sent.
        """
        wanted = str(model_dir or "")
        steps = int(steps or 0)
        kind = str(kind or KIND_LONGFORM)
        precision = str(precision or dtype or "bf16")
        device = str(device or "cuda")
        lora = str(lora_dir or "")
        scale = _float(lora_scale, 1.0)
        voices = str(voices_dir or "")
        if kind not in KINDS:
            raise Refusal("that kind of model is not one this worker loads")
        if precision not in PRECISIONS:
            raise Refusal("that precision is not one this worker loads")
        if device not in DEVICES:
            raise Refusal("that device is not one this worker uses")
        if not 0.0 <= scale <= LORA_SCALE_MAX:
            raise Refusal(f"a LoRA's strength is from 0 to {LORA_SCALE_MAX:g}")
        if kind == KIND_REALTIME and precision != "bf16":
            raise Refusal("the Realtime model loads at full precision only")
        if kind == KIND_REALTIME and lora:
            raise Refusal("the Realtime model does not take a LoRA")
        if precision in QUANTISED and device != "cuda":
            raise Refusal("8-bit and 4-bit loading need a CUDA device")
        if not lora:
            scale = 1.0  # a strength with no LoRA to apply it to is not part of what is loaded

        identity = (wanted, kind, precision, lora, scale, device)
        if self.model is not None and identity == self.identity():
            if steps and steps != self.steps:
                self.model.set_ddpm_inference_steps(num_steps=steps)
                self.steps = steps
            if voices != self.voices_dir:
                self.voices_dir = voices
                self._presets = {}
            answer = self._loaded_reply()
            answer.update({"load_seconds": 0.0, "already_loaded": True})
            return answer

        if not wanted or not os.path.isdir(wanted):
            raise Refusal("the model directory does not exist")
        if lora and not os.path.isdir(lora):
            raise Refusal("the LoRA folder does not exist")
        torch, _numpy = self._frameworks()
        if device == "cuda" and not torch.cuda.is_available():
            raise Refusal("no CUDA device is visible to the worker")
        quantisation = self._quantisation(precision) if precision in QUANTISED else None
        peft = self._peft() if lora else None
        if kind == KIND_REALTIME:
            model_class, processor_class = self._upstream_realtime()
        else:
            model_class, processor_class = self._upstream()

        if self.model is not None:
            self.unload()
        began = time.monotonic()
        processor = processor_class.from_pretrained(wanted)
        arguments = {"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
                     "device_map": device, "attn_implementation": str(attention or "sdpa")}
        if quantisation is not None:
            arguments["quantization_config"] = quantisation
        model = model_class.from_pretrained(wanted, **arguments)
        model.eval()
        if kind == KIND_REALTIME:
            self._realtime_scheduler(model)
            steps = steps or REALTIME_STEPS
            model.set_ddpm_inference_steps(num_steps=steps)
        else:
            model.set_ddpm_inference_steps(num_steps=steps or None)
        parts, layers = (), 0
        if lora:
            try:
                parts, layers = self._attach_lora(model, lora, scale, peft)
            except BaseException:
                # A model half-joined to a LoRA is not the identity that was
                # asked for, so nothing stays loaded: the refusal says why.
                model = processor = None
                gc.collect()
                self._empty_cache_for(device)
                raise
        weights = 0
        for parameter in model.parameters():
            weights += int(parameter.numel()) * int(parameter.element_size())
        self.model = model
        self.processor = processor
        self.model_dir = wanted
        self.steps = steps
        self.weights_bytes = weights
        self.kind = kind
        self.precision = precision
        self.lora_dir = lora
        self.lora_scale = scale
        self.lora_parts = tuple(parts)
        self.lora_layers = int(layers)
        self.device = device
        self.voices_dir = voices
        self._presets = {}
        elapsed = time.monotonic() - began
        _note(f"loaded the {kind} model in {elapsed:.1f} s — "
              f"{weights / float(1 << 30):.1f} GiB of weights, {steps or 'default'} step(s), "
              f"{precision}, {attention or 'sdpa'}, {device}"
              + (f", LoRA on {layers} layer(s) at {scale:g}"
                 f"{' with ' + ', '.join(parts[1:]) if len(parts) > 1 else ''}" if lora else ""))
        answer = self._loaded_reply()
        answer["load_seconds"] = elapsed
        return answer

    def _loaded_reply(self) -> dict:
        return {"resident_bytes": self.resident(), "weights_bytes": self.weights_bytes,
                "kind": self.kind, "precision": self.precision, "lora": bool(self.lora_dir),
                "device": self.device}

    def _empty_cache_for(self, device: str) -> None:
        if device != "cpu" and self.torch is not None:
            try:
                self.torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - nothing to empty
                pass

    def _realtime_scheduler(self, model) -> None:
        """The 0.5B's noise scheduler, replaced as Microsoft's web demo replaces it.

        The checkpoint's own configuration, with the SDE variant of DPM-Solver++
        and the cosine schedule the demo uses for five steps.
        """
        scheduler = model.model.noise_scheduler
        model.model.noise_scheduler = scheduler.from_config(
            scheduler.config, algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2")

    def _quantisation(self, precision: str):
        """The ``quantization_config`` for 8-bit or NF4: the language model only."""
        config_class = self._quantisation_class()
        skip = list(QUANTISE_SKIP)
        if precision == "int8":
            return config_class(load_in_8bit=True, llm_int8_skip_modules=skip)
        return config_class(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=self.torch.bfloat16,
                            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=skip)

    # -- LoRA -------------------------------------------------------------- #

    def _attach_lora(self, model, folder: str, scale: float, peft) -> "tuple[tuple, int]":
        """Join a LoRA folder to a freshly loaded 7B. Returns (parts, layers).

        The language-model adapter is required; the diffusion head and the two
        connectors are optional, each either a whole module's weights or an
        adapter pair of its own. Every PEFT LoRA layer's ``scaling`` is then
        multiplied by ``scale``, which is what "strength" means for a LoRA.
        """
        config_class, peft_model, layer_class = peft
        adapter = None
        for sub in LORA_ADAPTER_FOLDERS:
            candidate = os.path.join(folder, sub) if sub else folder
            if os.path.isfile(os.path.join(candidate, "adapter_config.json")):
                adapter = candidate
                break
        if adapter is None:
            raise Refusal("the LoRA folder holds no language-model adapter")
        model.model.language_model = self._adapter(model.model.language_model, adapter,
                                                    "language-model", config_class,
                                                    peft_model)
        parts = ["llm"]
        for name, attribute, label in LORA_PARTS:
            part = os.path.join(folder, name)
            if not os.path.isdir(part):
                continue
            target = getattr(model.model, attribute)
            if os.path.isfile(os.path.join(part, "adapter_config.json")):
                setattr(model.model, attribute,
                        self._adapter(target, part, label, config_class, peft_model))
            else:
                self._whole_weights(target, part, attribute, label)
            parts.append(name)
        layers = 0
        for module in model.modules():
            if isinstance(module, layer_class):
                scaling = getattr(module, "scaling", None)
                if isinstance(scaling, dict):
                    for adapter_name in list(scaling):
                        scaling[adapter_name] = scaling[adapter_name] * float(scale)
                    layers += 1
        return tuple(parts), layers

    def _adapter(self, module, folder: str, label: str, config_class, peft_model):
        """One adapter pair joined to one module with PEFT's plain wrapper.

        The configuration is read and its task cleared before PEFT sees it: a
        community adapter saved as a causal-LM task would otherwise be joined
        with PEFT's causal-LM wrapper, which hands a ``labels`` argument to a
        module that is the bare decoder, not a model with a head.
        """
        try:
            with open(os.path.join(folder, "adapter_config.json"), encoding="utf-8") as handle:
                declared = json.load(handle)
        except Exception:  # noqa: BLE001 - any failure is the same unreadable file
            raise Refusal(f"the LoRA's {label} adapter has no readable configuration") from None
        if not isinstance(declared, dict) or str(declared.get("peft_type") or "").upper() \
                != "LORA":
            raise Refusal(f"the LoRA's {label} adapter is not a LoRA")
        try:
            config = config_class.from_pretrained(folder)
            config.task_type = None
            config.inference_mode = True
            return peft_model.from_pretrained(module, folder, config=config,
                                              is_trainable=False)
        except Refusal:
            raise
        except Exception:  # noqa: BLE001 - a library's reason may name a path
            raise Refusal(f"the LoRA's {label} adapter does not fit this model") from None

    def _whole_weights(self, module, folder: str, attribute: str, label: str) -> None:
        """A whole module's weights from a LoRA part, loaded strictly."""
        for name in FULL_WEIGHT_FILES:
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            try:
                state = self._read_state(path)
            except Exception:  # noqa: BLE001 - a library's reason may name a path
                raise Refusal(f"the LoRA's {label} could not be read") from None
            if not isinstance(state, dict):
                raise Refusal(f"the LoRA's {label} does not fit this model")
            state = _without_prefix(state, set(module.state_dict()),
                                    (f"model.{attribute}.", f"{attribute}."))
            try:
                module.load_state_dict(state, strict=True)
            except Exception:  # noqa: BLE001 - a library's reason may name a shape
                raise Refusal(f"the LoRA's {label} does not fit this model") from None
            return
        raise Refusal(f"the LoRA's {label} folder holds no weights this worker reads")

    # -- unloading --------------------------------------------------------- #

    def unload(self) -> dict:
        """Let go of the model and give the card its memory back."""
        on_card = self._on_card()
        self.model = None
        self.processor = None
        self.model_dir = ""
        self.steps = 0
        self.weights_bytes = 0
        self.kind = ""
        self.precision = ""
        self.lora_dir = ""
        self.lora_scale = 1.0
        self.lora_parts = ()
        self.lora_layers = 0
        self.voices_dir = ""
        self._presets = {}
        gc.collect()
        if self.torch is not None and on_card:
            try:
                self.torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - nothing to empty
                pass
        self.device = ""
        return {"resident_bytes": self.resident()}

    # -- preset voices ----------------------------------------------------- #

    def preset(self, stem: str):
        """A preset voice prompt, loaded once per stem and kept while the model is.

        ``<voices_dir>/<stem>.pt``, read the one way Microsoft's demos read it:
        ``weights_only`` with exactly the two classes such a file holds allowed,
        so a file that tried to carry code would be refused by Torch itself.
        """
        found = self._presets.get(stem)
        if found is not None:
            return found
        if not isinstance(stem, str) or not PRESET_STEM.fullmatch(stem):
            raise Refusal("a preset voice's name may hold only letters, digits, '-' and '_'")
        path = os.path.join(self.voices_dir, stem + ".pt") if self.voices_dir else ""
        if not path or not os.path.isfile(path):
            raise Refusal("that preset voice is not installed")
        found = self._checked_preset(self._read_preset(path))
        self._presets[stem] = found
        return found

    def _read_preset(self, path: str):
        torch, _numpy = self._frameworks()
        output_class, cache_class = self._preset_classes()
        try:
            with torch.serialization.safe_globals([output_class, cache_class]):
                return torch.load(path, map_location=self.device or "cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - a refused unpickle is a refused file
            raise Refusal("that preset voice file is not one the Realtime model reads") from None

    def _checked_preset(self, found):
        """The four prefilled passes, each with its hidden state and its cache,
        in the model's own floating type (a no-op for the published files,
        which are bf16 like the model on a card)."""
        if not isinstance(found, dict) or any(key not in found for key in PRESET_KEYS):
            raise Refusal("that preset voice file is not one the Realtime model reads")
        dtype = self._dtype()
        for key in PRESET_KEYS:
            entry = found[key]
            hidden = getattr(entry, "last_hidden_state", None)
            cache = getattr(entry, "past_key_values", None)
            if hidden is None or cache is None:
                raise Refusal("that preset voice file is not one the Realtime model reads")
            if getattr(hidden, "dtype", dtype) != dtype:
                entry.last_hidden_state = hidden.to(dtype)
            for name in ("key_cache", "value_cache"):
                tensors = getattr(cache, name, None)
                if isinstance(tensors, list) and any(getattr(one, "dtype", dtype) != dtype
                                                     for one in tensors):
                    setattr(cache, name, [one.to(dtype) for one in tensors])
        return found

    # -- rendering --------------------------------------------------------- #

    def render(self, request: RenderRequest, cancelled: threading.Event,
               on_progress=None, on_audio=None) -> "tuple[dict, bytes]":
        """One script through the model, as the demos run it.

        ``cancelled`` is read at the top of every generation step through
        ``stop_check_fn``; a cancel therefore lands within one step, and the
        reply says so and carries whatever audio there was.

        A request with ``stream`` hands every decoded chunk to
        ``on_audio(seq, pcm16, samples)`` while the model is still generating
        and returns no WAV; ``first_audio_ms`` in its reply is how long after
        this call began the first chunk was ready.
        """
        if self.model is None or self.processor is None:
            raise Refusal("no model is loaded")
        entered = time.monotonic()
        torch, numpy = self._frameworks()
        realtime = self.kind == KIND_REALTIME
        if realtime:
            if not request.presets:
                raise Refusal("the Realtime model speaks with its preset voices; it cannot "
                              "clone a recording")
            if len(request.speakers) > 1:
                raise Refusal("the Realtime model speaks with one voice at a time")
        elif request.presets:
            raise Refusal("preset voices belong to the Realtime model; the 7B speaks with "
                          "voice samples")
        streamed = bool(request.stream)
        if streamed and on_audio is None:
            raise Refusal("a streamed render has nowhere to send its audio")
        if request.steps and request.steps != self.steps:
            self.model.set_ddpm_inference_steps(num_steps=int(request.steps))
            self.steps = int(request.steps)

        if realtime:
            call, prompt_length = self._realtime_call(request, cancelled)
        else:
            call, prompt_length = self._longform_call(request, cancelled)
        budget = call.get("max_new_tokens")
        if request.seed is not None:
            torch.manual_seed(int(request.seed))
            if self._on_card():
                torch.cuda.manual_seed_all(int(request.seed))

        self._reset_peak()
        began = time.monotonic()
        frames = streamed_samples = first_audio_ms = 0
        progress = None
        try:
            if streamed:
                outputs, frames, streamed_samples, first_audio_ms = self._streamed(
                    call, cancelled, on_audio, entered)
            else:
                progress = Progress(SAMPLE_RATE, on_progress)
                outputs = self.model.generate(**call, audio_streamer=progress)
            samples = None
            if not streamed:
                speech = None
                found = getattr(outputs, "speech_outputs", None)
                if found:
                    speech = found[0]
                if speech is None:
                    samples = numpy.zeros(0, dtype=numpy.float32)
                else:
                    samples = speech.detach().reshape(-1).to(torch.float32).cpu().numpy()
            render_seconds = time.monotonic() - began
            tokens = 0
            if realtime:
                tokens = frames if streamed else (progress.chunks if progress else 0)
            else:
                sequences = getattr(outputs, "sequences", None)
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
            if not capped and not realtime and budget and not cancelled.is_set() \
                    and tokens >= int(budget):
                # The 7B's generate never raises its own flag for this: its
                # loop's range ends one step before the check that would.
                # Having used the whole budget without ending on
                # end-of-speech is what reaching the budget means.
                capped = _last_token(getattr(outputs, "sequences", None)) != \
                    getattr(self.processor.tokenizer, "eos_token_id", None)
            peak = self.peak()
            if streamed:
                pcm = b""
                seconds = streamed_samples / float(SAMPLE_RATE)
            else:
                pcm = pcm16(samples)
                seconds = len(samples) / float(SAMPLE_RATE)
        finally:
            outputs = None
            call = None
            gc.collect()
            self._empty_cache()
        if capped:
            _note(f"the render reached its token budget ({budget or 'the context'}) before "
                  f"the script ended")
        _note(f"rendered {seconds:.1f} s of audio in {render_seconds:.1f} s — {tokens} "
              f"token(s), peak {peak / float(1 << 30):.1f} GiB"
              + (f", streamed in {frames} frame(s), the first after {first_audio_ms} ms"
                 if streamed else "")
              + (", cancelled" if cancelled.is_set() else ""))
        reply = {
            "seconds": seconds,
            "sample_rate": SAMPLE_RATE,
            "render_seconds": render_seconds,
            "peak_bytes": peak,
            "resident_bytes": self.resident(),
            "cancelled": bool(cancelled.is_set()),
            "tokens": tokens,
            "capped": capped,
        }
        if streamed:
            reply.update({"streamed": True, "first_audio_ms": int(first_audio_ms),
                          "frames": int(frames)})
            return reply, b""
        return reply, encode_wav(pcm, SAMPLE_RATE)

    def _on_device(self, inputs) -> dict:
        """The processor's output as keyword arguments, its tensors on the model's device."""
        torch = self.torch
        found = {}
        for key, value in list(inputs.items()):
            found[key] = value.to(self.device) if torch.is_tensor(value) else value
        return found

    def _longform_call(self, request: RenderRequest, cancelled) -> "tuple[dict, int]":
        """The 7B's ``generate`` arguments: the phase-one call, unchanged."""
        inputs = self.processor(
            text=[request.text],
            voice_samples=[list(request.voices)],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        call = self._on_device(inputs)
        prompt_length = int(call["input_ids"].shape[-1])
        max_new_tokens = request.max_new_tokens or self._token_budget(request.text, prompt_length)
        # Upstream stops at max_length_times * prompt length as well as at
        # max_new_tokens; the multiplier is set so the token budget is the one
        # that binds (see TOKENS_PER_TEXT_TOKEN).
        times = max(2, int(math.ceil(max_new_tokens / float(max(1, prompt_length)))) + 1)
        call.update({
            "max_new_tokens": max_new_tokens,
            "cfg_scale": float(LONGFORM_CFG if request.cfg_scale is None else request.cfg_scale),
            "tokenizer": self.processor.tokenizer,
            "generation_config": {"do_sample": False},
            "verbose": False,
            "stop_check_fn": cancelled.is_set,
            "max_length_times": times,
            "show_progress_bar": False,
        })
        return call, prompt_length

    def _realtime_call(self, request: RenderRequest, cancelled) -> "tuple[dict, int]":
        """The 0.5B's ``generate`` arguments, as Microsoft's web demo passes them.

        The preset is deep-copied for every render because ``generate`` grows
        the caches it is handed; the cached original stays the voice's prompt.
        """
        prefilled = self.preset(request.presets[request.speakers[0]])
        inputs = self.processor.process_input_with_cached_prompt(
            text=request.plain,
            cached_prompt=prefilled,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        call = self._on_device(inputs)
        prompt_length = int(call["tts_lm_input_ids"].shape[-1]) \
            if "tts_lm_input_ids" in call else 0
        call.update({
            "max_new_tokens": request.max_new_tokens,
            "cfg_scale": float(REALTIME_CFG if request.cfg_scale is None else request.cfg_scale),
            "tokenizer": self.processor.tokenizer,
            "generation_config": {"do_sample": False},
            "stop_check_fn": cancelled.is_set,
            "verbose": False,
            "refresh_negative": True,
            "all_prefilled_outputs": copy.deepcopy(prefilled),
            "show_progress_bar": False,
        })
        return call, prompt_length

    def _streamed(self, call: dict, cancelled, on_audio, entered: float):
        """``generate`` on its own thread, its chunks read here as they arrive.

        Returns ``(outputs, frames, samples, first_audio_ms)``. Built the way
        Microsoft's web demo streams: a real ``AudioStreamer`` for one sample,
        no timeout, ``None`` as its end; the generation thread ends the stream
        whatever happens to it, so this loop always finishes, and an exception
        in the generation is raised here once the thread has stopped. If
        sending a frame fails -- the parent went away -- the cancel flag is set
        so the generation stops at its next step rather than running on for
        nobody.
        """
        streamer = self._streamer_class()(batch_size=1, stop_signal=None, timeout=None)
        box = {}

        def generate() -> None:
            try:
                box["outputs"] = self.model.generate(**call, audio_streamer=streamer)
            except BaseException as exc:  # noqa: BLE001 - carried back to the lane
                box["error"] = exc
            finally:
                try:
                    streamer.end()
                except Exception:  # noqa: BLE001 - already ended
                    pass

        thread = threading.Thread(target=generate, name="vibevoice-generate", daemon=True)
        thread.start()
        frames = samples = first_audio_ms = 0
        try:
            for chunk in streamer.get_stream(0):
                found = self._chunk_samples(chunk)
                if found.size == 0 or cancelled.is_set():
                    # After a cancel nothing more is sent: the Realtime model
                    # looks for one only once per window of six speech frames,
                    # and audio made after the listener said stop is not audio
                    # anybody wants. The stream is still read to its end.
                    continue
                frames += 1
                if frames == 1:
                    first_audio_ms = max(1, int(round((time.monotonic() - entered) * 1000.0)))
                on_audio(frames, pcm16(found), int(found.size))
                samples += int(found.size)
        except BaseException:
            cancelled.set()
            raise
        finally:
            thread.join()
        if "error" in box:
            raise box["error"]
        return box.get("outputs"), frames, samples, first_audio_ms

    def _chunk_samples(self, chunk):
        """One streamed chunk as a flat float32 array (NumPy has no bf16)."""
        torch, numpy = self._frameworks()
        if torch.is_tensor(chunk):
            return chunk.detach().reshape(-1).to(torch.float32).cpu().numpy()
        return numpy.asarray(chunk, dtype=numpy.float32).reshape(-1)

    def _token_budget(self, text: str, prompt_length: int) -> int:
        """How many tokens a 7B render may generate when the caller set no cap.

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


def _last_token(sequences) -> "int | None":
    """The last token of the first sequence ``generate`` returned, or ``None``."""
    try:
        return int(sequences[0, -1].item())
    except Exception:  # noqa: BLE001 - no sequences, or a shape this build does not have
        return None


def _without_prefix(state: dict, expected: set, prefixes) -> dict:
    """A state dict saved with its module's path in front of every key, without it.

    Only when none of its keys is one the module has and every key starts with
    the same one of ``prefixes``; anything else is returned unchanged for a
    strict load to judge.
    """
    keys = set(state)
    if not keys or keys & expected:
        return state
    for prefix in prefixes:
        if all(key.startswith(prefix) for key in keys):
            return {key[len(prefix):]: value for key, value in state.items()}
    return state


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
        loaded = bool(engine is not None and engine.model is not None)
        return {
            "loaded": loaded,
            "rendering": bool(job),
            "job": job,
            "resident_bytes": engine.resident() if engine is not None else 0,
            "peak_bytes": engine.peak() if engine is not None else 0,
            "model_dir": engine.model_dir if engine is not None else "",
            "kind": str(getattr(engine, "kind", "") or "") if loaded else "",
            "precision": str(getattr(engine, "precision", "") or "") if loaded else "",
            "lora": bool(getattr(engine, "lora_dir", "")) if loaded else False,
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


def load_identity(header: dict) -> dict:
    """The optional fields of a ``load`` beyond phase one, as keyword arguments.

    Absent fields are the phase-one meaning: the long-form model, the
    precision ``dtype`` names, no LoRA, the card.
    """
    lora_dir = header.get("lora_dir")
    if lora_dir is not None and not isinstance(lora_dir, str):
        raise Refusal("the LoRA folder is not a path")
    voices_dir = header.get("voices_dir")
    if voices_dir is not None and not isinstance(voices_dir, str):
        raise Refusal("the voices folder is not a path")
    return {
        "kind": str(header.get("kind") or KIND_LONGFORM),
        "precision": str(header.get("precision") or ""),
        "lora_dir": lora_dir or None,
        "lora_scale": _float(header.get("lora_scale"), 1.0),
        "voices_dir": voices_dir or "",
        "device": str(header.get("device") or "cuda"),
    }


def _do_load(worker: Worker, reply, request_id, header: dict) -> None:
    identity = load_identity(header)
    found = worker.engine.load(str(header.get("model_dir") or ""),
                               int(header.get("steps") or 0),
                               str(header.get("dtype") or "bf16"),
                               str(header.get("attention") or "sdpa"),
                               **identity)
    answer = {"ok": True}
    answer.update(found)
    reply(request_id, answer)


def _do_unload(worker: Worker, reply, request_id) -> None:
    answer = {"ok": True}
    answer.update(worker.engine.unload())
    reply(request_id, answer)


def _do_render(worker: Worker, reply, request_id, request: RenderRequest,
               cancelled: threading.Event) -> None:
    """One render on the lane, and the render slot freed whatever happens.

    A streamed render's frames carry the request's own id, ``audio`` instead
    of ``ok``, and the chunk as PCM16 little-endian mono; the reply that ends
    the render follows the last of them down the same pipe, so the parent has
    every frame before it has the reply.
    """
    try:
        def report(progress: dict) -> None:
            worker.send({"id": request_id, "progress": dict(progress)})

        if request.stream:
            def audio(seq: int, pcm: bytes, samples: int) -> None:
                worker.send({"id": request_id,
                             "audio": {"seq": int(seq), "samples": int(samples),
                                       "rate": SAMPLE_RATE}}, pcm)

            found, body = worker.engine.render(request, cancelled, report, on_audio=audio)
        else:
            found, body = worker.engine.render(request, cancelled, report)
    finally:
        # Freed *before* the reply goes out: a parent that sends its next
        # render the moment it reads this one's reply -- the Voice Box renders
        # a script's sections back to back -- would otherwise race this line
        # and be refused "one render at a time" for a render that had ended.
        worker.end_render()
    answer = {"ok": True}
    answer.update(found)
    reply(request_id, answer, body)


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
    promoted, with no model: the packages are imported, the classes this file
    speaks to are resolved -- the 7B's, the Realtime model's from the overlay,
    and the audio streamer -- and the versions are printed. ``cuda`` says
    whether a card is visible to that interpreter; it is reported rather than
    required, because the installer decides what a machine without one means.
    PEFT and bitsandbytes are reported the same way (``lora`` and
    ``quantisation``): each is one feature, and a runtime without it still
    renders at full precision without a LoRA.
    """
    report = {"ok": False, "error": "", "cuda": False}
    try:
        import torch

        report["torch"] = str(getattr(torch, "__version__", ""))
        report["numpy"] = _package_version("numpy")
        report["transformers"] = _package_version("transformers")
        engine = Engine()
        engine._upstream()
        engine._upstream_realtime()
        engine._streamer_class()
        report["vibevoice"] = _package_version("vibevoice")
        report["realtime"] = True
        report["cuda"] = bool(torch.cuda.is_available())
        for key, probe, package in (("lora", engine._peft, "peft"),
                                    ("quantisation", engine._quantisation_class,
                                     "bitsandbytes")):
            try:
                probe()
                report[key] = True
                report[package] = _package_version(package)
            except Refusal:
                report[key] = False
                report[package] = ""
        report["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the report is the answer
        report["error"] = f"{exc.__class__.__name__}: {exc}"
    sys.stdout.write(json.dumps(report) + "\n")
    sys.stdout.flush()
    return 0 if report["ok"] else 1


def _claim_stdout():
    """The pipe to the parent, taken away from everything that prints.

    The protocol runs over this process's standard output, and upstream code
    prints to it: the Realtime model's ``generate`` ends every render that
    reached its length limit with an unconditional ``print``, and the 7B's has
    one too. A line of text in the middle of the frame stream is a header
    length the parent cannot read, and the parent would lose its worker over
    it. So descriptor 1 is duplicated for the protocol alone, and descriptor 1
    itself -- with Python's ``sys.stdout`` -- is pointed at standard error,
    which the parent reads line by line into its log. Anything that prints
    now prints there, whether it prints from Python or from a native library.
    """
    sys.stdout.flush()
    descriptor = os.dup(sys.stdout.fileno())
    if os.name == "nt":
        import msvcrt

        msvcrt.setmode(descriptor, os.O_BINARY)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return os.fdopen(descriptor, "wb")


def main(argv=None, engine_factory=None) -> int:
    """Run as the worker, or ``--selftest``.

    ``engine_factory`` is the engine the loop serves with, the real
    :class:`Engine` when absent; ``tools/smoke_vibevoice_worker.py`` passes one
    that answers ``init`` on a machine without a card, so the whole process --
    this function's own claim on standard output included -- can be driven
    over real pipes.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(MARKER, action="store_true", dest="marker")
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--selftest", action="store_true")
    found, _rest = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    if found.selftest:
        return selftest()
    return serve(sys.stdin.buffer, _claim_stdout(), engine_factory=engine_factory)


if __name__ == "__main__":
    raise SystemExit(main())
