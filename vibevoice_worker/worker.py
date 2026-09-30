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
    the solver and the attention a render asks for
    one generation at a time, of one to four takes, cancellable between steps
    the audio it produced, as the model's own 32-bit float samples

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
        verbose=False, stop_check_fn=...)``, whose ``speech_outputs[k]`` is
        take k's audio at 24 kHz and whose ``stop_check_fn`` is read at the top
        of every generation step.

And three things the demo itself reaches into, the same way it does:

    ``model.model.noise_scheduler``, replaced by
        ``type(scheduler).from_config(scheduler.config, algorithm_type=...)``
        -- how upstream's Gradio demo puts its SDE solver in (:data:`SOLVERS`);
    ``model.model.language_model.config._attn_implementation``, which
        transformers 4.51 reads at every forward pass, so the attention a
        render asks for is one attribute rather than a second load
        (:data:`ATTENTION`);
    for a batch of takes only, the three places ``generate`` draws random
        numbers, each given the take's own generator (:class:`TakeRandomness`).

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

Takes, and why each draws its own random numbers
------------------------------------------------
A render of several takes is one ``generate`` over a batch of identical
prompts: a card that spends its time reading the language model's weights
reads them once for all four. What makes the takes differ is randomness, and
``generate`` draws it from Torch's one global generator in three places -- the
voice prompt's encoding (its tokenizer samples a latent around its mean), the
diffusion head's starting noise (and the SDE solver's noise at every step), and
the token choice when sampling is on. Shared, the takes would come out as
whatever the batch happened to draw. So a batch gives take k a generator of its
own seeded ``seed + k``, and draws for it exactly what a render of that seed
alone draws, in the same order and the same shapes: take k of a batch is the
take the seed ``seed + k`` makes by itself, give or take the rounding of a
batched matrix product. A single take is left to the global generator, exactly
as before, so a seed recorded before batches existed makes the same render.

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
import functools
import gc
import importlib.util
import json
import math
import os
import platform
import queue
import struct
import sys
import threading
import time

PROTOCOL_VERSION = 2
"""The VibeVoice protocol's own version, counted from one.

Not shared with the other four workers. They speak the same *framing* and this
one deliberately overlaps none of their operations: it has ``render`` and
``cancel`` where they have turns, and it is the only one that answers for a
graphics card. One number covering all of them would be a number that has to
change when any of them changes.

Two: a render names its solver, its attention and its number of takes, and
is answered with each take's 32-bit float samples rather than one PCM16 WAV.
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
"""What this process accepts in a request: the voice samples of a render are a
few megabytes, so a quarter of a gigabyte is a garbled length, not a request."""

MAX_REPLY_PAYLOAD = 2 << 30
"""What the parent accepts from its own worker in one reply. The 7B's context
ends near forty-five minutes of speech, which is about 260 MB of 32-bit float
at 24 kHz, and a batch is four takes of it: a little over a gigabyte, so the
ceiling is two rather than "whatever arrives"."""

SAMPLE_RATE = 24000
"""What VibeVoice produces and what its voice samples have to be. Reported in
the handshake so the parent can refuse a build that says otherwise."""

MAX_SPEAKERS = 4
"""How many distinct speakers one script may have. The model's own limit."""

SAMPLING_TEMPERATURE = 0.95
SAMPLING_TOP_P = 0.95
"""What a request that asks to sample without saying how is given."""

SOLVERS = {
    "dpmpp_2m": {"name": "DPM++ 2M", "algorithm_type": "dpmsolver++", "solver_order": 2},
    "dpmpp_2m_sde": {"name": "DPM++ 2M SDE", "algorithm_type": "sde-dpmsolver++",
                     "solver_order": 2},
    "dpmpp_3m": {"name": "DPM++ 3M", "algorithm_type": "dpmsolver++", "solver_order": 3},
    "dpmpp_1m": {"name": "DPM++ 1M", "algorithm_type": "dpmsolver++", "solver_order": 1},
    "dpmpp_1m_sde": {"name": "DPM++ 1M SDE", "algorithm_type": "sde-dpmsolver++",
                     "solver_order": 1},
}
"""Every solver VibeVoice's own scheduler runs, by the id a request names it with.

The model builds ``vibevoice.schedule.dpm_solver.DPMSolverMultistepScheduler``
and walks its timesteps itself, feeding the diffusion head's prediction to
``step`` -- no input scaling, noise of unit variance -- so a solver here is that
class configured, never another class. Its algorithms that are not deprecated
are DPM-Solver++ and its SDE variant, at orders one to three, except that the
third-order update takes no noise: there is no third-order SDE to offer.

``dpmpp_2m`` is what the model is built with and what upstream's
``inference_from_file.py`` renders with; ``dpmpp_2m_sde`` is upstream's Gradio
demo, which swaps it in the moment it loads ("Use SDE solver by default") and
spells the cosine schedule ``squaredcos_cap_v2``, which is the same betas.
First order is DDIM, in the ODE case. The order of this table is the order
they are listed in.

Every one keeps the model's own spacing of the steps, evenly along the
timesteps. The class also offers Karras sigmas and Lu's uniform log-SNR, and
neither survives VibeVoice's cosine noise schedule: the top of that schedule is
so steep that several of their noise levels land on timestep 999, the class
takes its first step at the second copy, and the render runs off the end of
its noise levels on the last step -- at 45 and 42 of the step counts from 1 to
50. A spacing the model's own scheduler cannot finish is not offered.
"""

ATTENTION = {"sdpa": "SDPA", "eager": "Eager", "flash_attention_2": "Flash attention 2"}
"""The language model's attention implementations VibeVoice declares it supports.

Switched per render on the language model's configuration, which transformers
4.51 reads at every forward pass. SDPA is PyTorch's fused kernels (on Windows
without its flash kernel); Eager is plain matrix products with the softmax in
32-bit float; Flash attention 2 is what upstream loads on a CUDA card, and needs
the ``flash-attn`` package, which this runtime does not install. Flex attention
is not listed: VibeVoice does not declare it, and it needs Triton.
"""

SOLVER_DEFAULT = "dpmpp_2m"
ATTENTION_DEFAULT = "sdpa"

MAX_TAKES = 4
"""How many takes one render may make, each at the next seed."""

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


def read_frame(stream, max_payload: int = MAX_PAYLOAD) -> "tuple[dict, bytes] | None":
    """One request, or ``None`` at end of input.

    ``None`` is how the parent's death arrives when this process is waiting for
    work, and it is not an error: the loop ends, the model is released with the
    process, and the exit status is 0. ``max_payload`` is this process's own
    ceiling by default; the parent reads its worker's replies with
    :data:`MAX_REPLY_PAYLOAD`.
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
    if size > max_payload:
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


def float32_bytes(samples) -> bytes:
    """The model's samples as little-endian 32-bit float, exactly as it made them.

    Not clipped and not quantised: the parent encodes the file, and a sample a
    hair beyond unity is its to clip once, at the end. A value that is not a
    number -- which a model should never produce -- is silence rather than a
    file no decoder can play.
    """
    import numpy

    array = numpy.asarray(samples, dtype=numpy.float32).reshape(-1)
    return numpy.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0).astype("<f4").tobytes()


def flash_attention_available() -> bool:
    """Whether the ``flash-attn`` package is importable here, asked without importing it."""
    try:
        return importlib.util.find_spec("flash_attn") is not None
    except (ImportError, ValueError):
        return False


def attention_available() -> list:
    """The ids of :data:`ATTENTION` this runtime can run, in the table's order."""
    return [name for name in ATTENTION
            if name != "flash_attention_2" or flash_attention_available()]


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
                 "max_new_tokens", "steps", "sampling", "temperature", "top_p",
                 "solver", "attention", "takes")

    def __init__(self, job, text, speakers, voices, cfg_scale, seed, max_new_tokens, steps,
                 sampling=False, temperature=None, top_p=None, solver=SOLVER_DEFAULT,
                 attention=ATTENTION_DEFAULT, takes=1):
        self.job = job
        self.text = text
        self.speakers = speakers
        self.voices = voices
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.sampling = bool(sampling)
        self.temperature = temperature
        self.top_p = top_p
        self.solver = solver
        self.attention = attention
        self.takes = int(takes)

    @property
    def seeds(self) -> "list[int] | None":
        """Each take's seed, the next after the last; ``None`` for a render left unseeded."""
        if self.seed is None:
            return None
        return [int(self.seed) + index for index in range(self.takes)]

    @classmethod
    def parse(cls, header: dict, payload: bytes) -> "RenderRequest":
        text, speakers = script_text(header.get("script"))
        voices = voice_arrays(header.get("voices"), payload, speakers)
        max_new_tokens = _optional_int(header.get("max_new_tokens"), "max_new_tokens")
        if max_new_tokens == 0:
            max_new_tokens = None
        # Sampling only when asked for in so many words: anything but ``true``
        # is the model's own greedy choice, which is what upstream ships.
        sampling = header.get("sampling") is True
        temperature = top_p = None
        if sampling:
            temperature = _float(header.get("temperature"), SAMPLING_TEMPERATURE)
            top_p = _float(header.get("top_p"), SAMPLING_TOP_P)
            if not 0.0 < temperature <= 2.0:
                raise Refusal("the temperature must be above 0 and at most 2")
            if not 0.0 < top_p <= 1.0:
                raise Refusal("top-p must be above 0 and at most 1")
        solver = str(header.get("solver") or SOLVER_DEFAULT)
        if solver not in SOLVERS:
            raise Refusal("that solver is not one VibeVoice's scheduler has")
        attention = str(header.get("attention") or ATTENTION_DEFAULT)
        if attention not in ATTENTION:
            raise Refusal("that attention is not one VibeVoice supports")
        takes = _optional_int(header.get("takes"), "the number of takes") or 1
        if takes > MAX_TAKES:
            raise Refusal(f"a render makes 1 to {MAX_TAKES} takes")
        seed = _optional_int(header.get("seed"), "the seed")
        if takes > 1 and seed is None:
            # Take k is the render seed + k makes: without a seed there is no k.
            raise Refusal("a render of several takes needs a seed")
        return cls(job=str(header.get("job") or ""), text=text, speakers=speakers,
                   voices=voices, cfg_scale=_float(header.get("cfg_scale"), 1.3),
                   seed=seed, max_new_tokens=max_new_tokens,
                   steps=_optional_int(header.get("steps"), "the step count") or 0,
                   sampling=sampling, temperature=temperature, top_p=top_p,
                   solver=solver, attention=attention, takes=takes)


class Progress:
    """The audio streamer's shape, used to count what each take has made so far.

    ``generate`` hands its streamer every decoded chunk as it is made, with the
    takes it belongs to, and calls ``end`` when a take finishes; it also breaks
    out of its loop the moment *any* ``finished_flags`` entry is true, for every
    take at once. So a take that ends raises no flag here: its three companions
    would stop with it, mid-sentence. Only ``end`` with no takes named -- which
    ``generate`` calls when the whole render stops -- raises them. Samples are
    counted from the chunk's shape rather than copied off the card, so this
    costs the render nothing; the progress reported is the furthest take's.
    """

    def __init__(self, rate: int, report, takes: int = 1, interval: float = PROGRESS_INTERVAL):
        self.rate = int(rate)
        self.counts = [0] * max(1, int(takes))
        self.finished_flags = [False] * len(self.counts)
        self._report = report
        self._interval = float(interval)
        self._last = 0.0

    def put(self, audio_chunks, sample_indices=None) -> None:
        try:
            count = int(audio_chunks.shape[-1])
        except Exception:  # noqa: BLE001 - a shape this build does not have; no progress then
            return
        for index in _indices(sample_indices, len(self.counts)):
            self.counts[index] += count
        now = time.monotonic()
        if now - self._last < self._interval:
            return
        self._last = now
        if self._report is not None:
            try:
                self._report({"seconds": self.seconds})
            except Exception:  # noqa: BLE001 - progress is advisory
                pass

    def end(self, sample_indices=None) -> None:
        if sample_indices is None:
            self.finished_flags = [True for _flag in self.finished_flags]

    @property
    def samples(self) -> int:
        return max(self.counts)

    @property
    def seconds(self) -> float:
        return self.samples / float(self.rate or 1)


def _indices(sample_indices, count: int) -> list:
    """The takes a chunk belongs to, as ints; every take when none are named."""
    if sample_indices is None:
        return list(range(count))
    try:
        found = sample_indices.tolist() if hasattr(sample_indices, "tolist") \
            else list(sample_indices)
        found = [int(index.item()) if hasattr(index, "item") else int(index) for index in found]
    except Exception:  # noqa: BLE001 - an index this build does not have
        return []
    return [index for index in found if 0 <= index < count]


# --------------------------------------------------------------------------- #
# A batch's takes, each with its own random numbers
# --------------------------------------------------------------------------- #


class TakeRandomness:
    """For the length of one ``generate``: each take's random numbers from its own seed.

    ``generate`` draws from Torch's global generators in three places, and for
    a batch each is given the take's own (see "Takes" in the module docstring):

    the voice prompt -- the acoustic tokenizer's ``encode`` answers with an
        object whose ``sample`` adds noise around the mean; the object this
        tokenizer returns is given a ``sample`` that draws each take's clips
        from that take's card generator, in the shapes a single take draws;
    the diffusion head -- ``sample_speech_tokens`` is replaced by
        :meth:`diffuse`, upstream's own loop with the starting noise drawn per
        take on the host (where upstream draws it) and the SDE solver's noise
        per take on the card, handed to ``step`` as ``variance_noise``; which
        takes are speaking is ``generate``'s own ``diffusion_indices``, read
        from its frame because upstream passes only their conditions;
    the token choice -- ``torch.multinomial``, which ``generate`` calls for
        every take at once when sampling, is answered a take at a time.

    Everything is put back on exit, whatever happened. Only ever used for a
    batch: a single take keeps the global generators, as it always has.
    """

    def __init__(self, torch, model, seeds, clips: int, device: str = "cuda"):
        self.torch = torch
        self.model = model
        self.seeds = [int(seed) for seed in seeds]
        self.clips = int(clips)
        self.host = [torch.Generator().manual_seed(seed) for seed in self.seeds]
        # A card has a generator of its own beside the host's, and a render
        # seeds both; a machine with no card draws everything from the host's.
        self.card = (self.host if str(device) == "cpu" else
                     [torch.Generator(device=device).manual_seed(seed) for seed in self.seeds])
        self._multinomial = None
        self._tokenizer = None

    def __enter__(self):
        torch = self.torch
        tokenizer = self.model.model.acoustic_tokenizer
        encode = tokenizer.encode

        def encoded(*args, **kwargs):
            found = encode(*args, **kwargs)
            found.sample = functools.partial(self.prompt, found)
            return found

        tokenizer.encode = encoded
        self._tokenizer = tokenizer
        self.model.sample_speech_tokens = self.diffuse
        self._multinomial = torch.multinomial
        torch.multinomial = self.choose
        return self

    def __exit__(self, *_exc) -> bool:
        if self._multinomial is not None:
            self.torch.multinomial = self._multinomial
        for owner, name in ((self._tokenizer, "encode"), (self.model, "sample_speech_tokens")):
            if owner is not None and name in vars(owner):
                delattr(owner, name)
        return False

    # -- the voice prompt -------------------------------------------------- #

    def prompt(self, found, dist_type: str = "fix"):
        """``VibeVoiceTokenizerEncoderOutput.sample``, a take's clips at a time.

        The same arithmetic in the same order as upstream's, on each take's
        slice of the batch: ``torch.empty_like(...).normal_(generator=...)`` is
        what ``torch.randn_like`` is, so the noise keeps the mean's memory
        layout (a transposed view) and lands where a single take's would.
        """
        torch = self.torch
        mean = found.mean
        takes = len(self.seeds)
        if dist_type not in ("fix", "gaussian"):
            return found.mean, found.std
        if int(mean.size(0)) != self.clips * takes:
            raise Refusal("the voice prompt did not come out as one set of samples per take")
        parts, spreads = [], []
        for index in range(takes):
            part = mean[index * self.clips:(index + 1) * self.clips]
            generator = self.card[index]
            if dist_type == "fix":
                noise = torch.empty_like(part).normal_(generator=generator)
                parts.append(part + found.std * noise)
                continue
            value = found.std / 0.8
            spread = torch.randn(part.size(0), device=part.device, dtype=part.dtype,
                                 generator=generator) * value
            while spread.dim() < part.dim():
                spread = spread.unsqueeze(-1)
            noise = torch.empty_like(part).normal_(generator=generator)
            parts.append(part + spread * noise)
            spreads.append(spread)
        return torch.cat(parts, dim=0), (torch.cat(spreads, dim=0) if spreads else found.std)

    # -- the diffusion head ------------------------------------------------ #

    def diffuse(self, condition, neg_condition, cfg_scale: float = 3.0):
        """Upstream's ``sample_speech_tokens``, with each take's noise its own.

        Line for line the loop upstream runs, against whichever solver the
        model holds. A single take of upstream's draws ``(2, width)`` of
        starting noise on the host and ``(2, width)`` of SDE noise on the card
        at every step, and uses the first row; each take here draws exactly
        that from its own generators, its first row among the conditioned half
        of the batch and its second in the half the classifier-free guidance
        doubles it into.
        """
        frame = sys._getframe(1)
        rows = frame.f_locals.get("diffusion_indices")
        del frame
        torch = self.torch
        if rows is None:
            raise Refusal("this VibeVoice build does not say which takes are speaking")
        takes = [int(row) for row in rows.tolist()]
        if len(takes) != int(condition.shape[0]):
            raise Refusal("the speaking takes and their conditions do not match")
        model = self.model
        scheduler = model.model.noise_scheduler
        with torch.no_grad():
            scheduler.set_timesteps(model.ddpm_inference_steps)
            condition = torch.cat([condition, neg_condition], dim=0).to(
                model.model.prediction_head.device)
            width = model.config.acoustic_vae_dim
            speech = self._pairs([torch.randn(2, width, generator=self.host[take])
                                  for take in takes]).to(condition)
            stochastic = str(getattr(scheduler.config, "algorithm_type", "")).startswith("sde")
            for step in scheduler.timesteps:
                half = speech[: len(speech) // 2]
                combined = torch.cat([half, half], dim=0)
                eps = model.model.prediction_head(combined, step.repeat(combined.shape[0]).to(combined),
                                                  condition=condition)
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
                noise = None
                if stochastic:
                    noise = self._pairs([torch.randn((2, width), generator=self.card[take],
                                                     device=eps.device, dtype=torch.float32)
                                         for take in takes])
                speech = scheduler.step(eps, step, speech, variance_noise=noise).prev_sample
            return speech[: len(speech) // 2]

    def _pairs(self, drawn: list):
        """Each take's two rows, the first rows in take order and then the second."""
        return self.torch.stack([pair[0] for pair in drawn] + [pair[1] for pair in drawn])

    # -- the token choice -------------------------------------------------- #

    def choose(self, probabilities, num_samples, replacement=False, *, generator=None, out=None):
        """``torch.multinomial``, each take's row drawn with that take's generator."""
        original = self._multinomial
        if generator is None and out is None and getattr(probabilities, "dim", None) is not None \
                and probabilities.dim() == 2 and int(probabilities.shape[0]) == len(self.card):
            return self.torch.cat([
                original(probabilities[index:index + 1], num_samples, replacement,
                         generator=self.card[index])
                for index in range(len(self.card))], dim=0)
        if out is not None:
            return original(probabilities, num_samples, replacement, generator=generator, out=out)
        return original(probabilities, num_samples, replacement, generator=generator)


class _NoHooks:
    """What a single take renders under: nothing replaced."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False


# --------------------------------------------------------------------------- #
# The engine: Torch, transformers and vibevoice, behind one class
# --------------------------------------------------------------------------- #


class Engine:
    """One VibeVoice model on one card, one generation at a time.

    Everything Torch-shaped is behind this class, and it is constructed once
    per worker. Its two import points -- :meth:`_frameworks` and
    :meth:`_upstream` -- are methods so a test can hand it stand-ins and
    exercise the exact call sequence without a card. ``device`` is the card
    (``cuda``, the one card this process can see) everywhere but a check that
    runs upstream's own code on a machine without one.
    """

    def __init__(self, device: str = "cuda"):
        self.device = str(device or "cuda")
        self.torch = None
        self.numpy = None
        self.model = None
        self.processor = None
        self.model_dir = ""
        self.steps = 0
        self.weights_bytes = 0
        self.scheduler = None
        """The scheduler the model was built with, which every other solver is made from."""
        self.solver = SOLVER_DEFAULT
        self.attention = ATTENTION_DEFAULT

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
            wanted, torch_dtype=precision, device_map=self.device,
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
        self.scheduler = getattr(getattr(model, "model", None), "noise_scheduler", None)
        self.solver = SOLVER_DEFAULT
        self.attention = str(attention or ATTENTION_DEFAULT)
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
        self.scheduler = None
        self.solver = SOLVER_DEFAULT
        self.attention = ATTENTION_DEFAULT
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
        """One script through the model, as the demo runs it, as one to four takes.

        ``cancelled`` is read at the top of every generation step through
        ``stop_check_fn``; a cancel therefore lands within one step, and the
        reply says so and carries whatever audio there was. The reply's
        ``takes`` says, take by take, how many samples of the payload are that
        take's: the payload is every take's 32-bit float samples, in order.
        """
        if self.model is None or self.processor is None:
            raise Refusal("no model is loaded")
        torch, numpy = self._frameworks()
        if request.steps and request.steps != self.steps:
            self.model.set_ddpm_inference_steps(num_steps=int(request.steps))
            self.steps = int(request.steps)
        self._use_solver(request.solver)
        self._use_attention(request.attention)
        takes = max(1, int(request.takes or 1))

        inputs = self.processor(
            text=[request.text] * takes,
            voice_samples=[list(request.voices) for _take in range(takes)],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(self.device)
        seeds = request.seeds
        if seeds:
            # The first take's seed on the global generators whatever the
            # batch: a single take draws from them, and a batch's hooks leave
            # nothing unseeded behind them.
            torch.manual_seed(seeds[0])
            torch.cuda.manual_seed_all(seeds[0])
        hooks = (TakeRandomness(torch, self.model, seeds, len(request.voices), self.device)
                 if takes > 1 else _NoHooks())

        prompt_length = int(inputs["input_ids"].shape[-1])
        max_new_tokens = request.max_new_tokens or self._token_budget(request.text, prompt_length)
        # Upstream stops at max_length_times * prompt length as well as at
        # max_new_tokens; the multiplier is set so the token budget is the one
        # that binds (see TOKENS_PER_TEXT_TOKEN).
        times = max(2, int(math.ceil(max_new_tokens / float(max(1, prompt_length)))) + 1)
        progress = Progress(SAMPLE_RATE, on_progress, takes)

        on_card = self.device == "cuda"
        if on_card:
            torch.cuda.reset_peak_memory_stats(0)
        began = time.monotonic()
        made = []
        try:
            with hooks:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    cfg_scale=float(request.cfg_scale),
                    tokenizer=self.processor.tokenizer,
                    generation_config=_generation(request),
                    verbose=False,
                    stop_check_fn=cancelled.is_set,
                    audio_streamer=progress,
                    max_length_times=times,
                    show_progress_bar=False,
                )
            found = list(getattr(outputs, "speech_outputs", None) or ())
            sequences = getattr(outputs, "sequences", None)
            reached = getattr(outputs, "reach_max_step_sample", None)
            ran = max(0, int(sequences.shape[-1]) - prompt_length) if sequences is not None else 0
            eos = getattr(self.processor.tokenizer, "eos_token_id", None)
            for take in range(takes):
                speech = found[take] if take < len(found) else None
                if speech is None:
                    samples = numpy.zeros(0, dtype=numpy.float32)
                else:
                    samples = speech.detach().reshape(-1).to(torch.float32).cpu().numpy()
                # A single take ran until it ended; a batch runs until its last
                # take has, so each take's own count stops at its end of speech.
                tokens = ran if takes == 1 else _tokens_of(sequences, take, prompt_length,
                                                           eos, ran)
                capped = _flag(reached, take)
                if not capped and not cancelled.is_set() and ran >= int(max_new_tokens):
                    # Upstream's generate never raises its own flag when the
                    # budget runs out: its loop's range ends one step before
                    # the check that would. A take that used the whole budget
                    # without ending on end-of-speech is what reaching it means.
                    capped = _last_token(sequences, take) != eos
                made.append((samples, tokens, capped))
            render_seconds = time.monotonic() - began
            peak = int(torch.cuda.max_memory_reserved(0)) if on_card else 0
        finally:
            outputs = None
            inputs = None
            gc.collect()
            if on_card:
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001 - nothing to empty
                    pass
        answers = []
        for index, (samples, tokens, capped) in enumerate(made):
            answers.append({"seed": None if seeds is None else seeds[index],
                            "samples": int(len(samples)),
                            "seconds": len(samples) / float(SAMPLE_RATE),
                            "tokens": int(tokens), "capped": bool(capped)})
        if any(answer["capped"] for answer in answers):
            _note(f"a take reached its token budget ({max_new_tokens}) before the script ended")
        longest = max(answer["seconds"] for answer in answers)
        _note(f"rendered {len(answers)} take(s), the longest {longest:.1f} s of audio, in "
              f"{render_seconds:.1f} s — {sum(a['tokens'] for a in answers)} token(s), peak "
              f"{peak / float(1 << 30):.1f} GiB, {SOLVERS[request.solver]['name']}, "
              f"{ATTENTION[request.attention]}"
              f"{', cancelled' if cancelled.is_set() else ''}")
        return {
            "format": "f32le",
            "sample_rate": SAMPLE_RATE,
            "takes": answers,
            "seconds": answers[0]["seconds"],
            "render_seconds": render_seconds,
            "peak_bytes": peak,
            "resident_bytes": self.resident(),
            "cancelled": bool(cancelled.is_set()),
            "tokens": sum(answer["tokens"] for answer in answers),
            "capped": any(answer["capped"] for answer in answers),
        }, b"".join(float32_bytes(samples) for samples, _tokens, _capped in made)

    def _use_solver(self, solver: str) -> None:
        """Put the solver a render asks for on the model, made from the model's own.

        The scheduler the model was built with is kept and handed back when it
        is already the solver asked for, so the default render uses the very
        object upstream's own render uses; any other solver is that
        scheduler's configuration with the algorithm and order changed, as
        upstream's demo does it. ``step`` keeps history between calls and
        ``set_timesteps`` clears it, and the model calls ``set_timesteps``
        before every speech frame.
        """
        wanted = str(solver or SOLVER_DEFAULT)
        if wanted == self.solver:
            return
        base = self.scheduler
        if base is None:
            raise Refusal("no model is loaded")
        choice = {"algorithm_type": SOLVERS[wanted]["algorithm_type"],
                  "solver_order": SOLVERS[wanted]["solver_order"]}
        config = base.config
        if all(getattr(config, key, None) == value for key, value in choice.items()):
            chosen = base
        else:
            chosen = type(base).from_config(config, **choice)
        self.model.model.noise_scheduler = chosen
        self.solver = wanted

    def _use_attention(self, attention: str) -> None:
        """Have the language model attend the way a render asks. One attribute, no reload."""
        wanted = str(attention or ATTENTION_DEFAULT)
        if wanted == self.attention:
            return
        if wanted == "flash_attention_2" and not flash_attention_available():
            raise Refusal("Flash attention 2 needs the flash-attn package, which VibeVoice's "
                          "runtime does not have")
        self.model.model.language_model.config._attn_implementation = wanted
        self.attention = wanted

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
                        "attention": attention_available(),
                        "max_takes": MAX_TAKES,
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


def _generation(request: "RenderRequest") -> dict:
    """The generation config ``generate`` is handed.

    Greedy unless the request asks to sample. The model's language part only
    chooses among a handful of control tokens -- keep speaking, end a stretch
    of speech, stop -- while the voice itself comes from the diffusion head,
    so temperature and top-p change the pacing from take to take, and a
    seeded render samples the same way every time.
    """
    if not request.sampling:
        return {"do_sample": False}
    return {"do_sample": True, "temperature": float(request.temperature),
            "top_p": float(request.top_p)}


def _last_token(sequences, take: int = 0) -> "int | None":
    """The last token of a take's sequence as ``generate`` returned it, or ``None``."""
    try:
        return int(sequences[take, -1].item())
    except Exception:  # noqa: BLE001 - no sequences, or a shape this build does not have
        return None


def _tokens_of(sequences, take: int, prompt_length: int, eos, ran: int) -> int:
    """How many tokens a take generated, its end-of-speech included.

    ``generate`` runs until every take has ended and pads a take that ended
    early with end-of-speech, so a take's count stops at its first one; a take
    that never ended used every step the render ran.
    """
    if sequences is None:
        return 0
    try:
        made = sequences[take, prompt_length:].tolist()
    except Exception:  # noqa: BLE001 - a shape this build does not have
        return ran
    if eos is not None and eos in made:
        return made.index(eos) + 1
    return len(made)


def _flag(reached, take: int) -> bool:
    """Upstream's own per-take "reached its length" flag, when it raised one."""
    if reached is None:
        return False
    try:
        return bool(reached[take].item()) if hasattr(reached[take], "item") \
            else bool(reached[take])
    except Exception:  # noqa: BLE001 - a shape this build does not have
        return False


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
        report["attention"] = attention_available()
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
