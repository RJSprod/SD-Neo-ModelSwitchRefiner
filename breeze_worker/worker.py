"""The Breeze TTS 2 sidecar: one contained GPU process per card, and everything Torch touches.

Run by path, under the isolated Breeze interpreter, by
:mod:`mc_voice_breeze_runtime`, once per card that is asked to render. It never
imports a Forge module, a Model Chain module or another engine's runtime. The
parent starts it with exactly one card visible (``CUDA_VISIBLE_DEVICES`` set to
that card's UUID) and refuses its handshake unless the card it came up on is
the card it was asked for -- the same rule VibeVoice's worker lives by.

What lives here and why it is not in the parent
-----------------------------------------------
Everything that needs a tensor:

    proving which card this process is on, and how much of it is free
    loading Breeze TTS 2, its tokenizer and its audio codec from a directory
    one speaker's reference audio turned into the codec's frames, once a render
    a section rendered line by line, each line in its speaker's voice
    a line too long for one call split at its sentence ends
    the takes, one after another, take k at seed + k
    the audio, as the codec's own 32-bit float samples

The upstream surface this file binds to
---------------------------------------
Breeze's inference code is not a wheel: the installer puts the repository's
``breeze_infer`` and ``models`` packages, at a pinned commit, in a folder of
the runtime and names it in ``MODEL_CHAIN_BREEZE_SOURCE``. Five calls, which
are what ``infer.py`` itself makes:

    ``breeze_infer.runtime.load_runtime(Path(dir), device=..., attn_implementation=...)``
        -> ``(tokenizer, model, audio_tokenizer)``, then
        ``update_generation_config_for_breeze(model)``;
    ``models.fast_streaming.FastBreezeStreamingRuntime(model, audio_tokenizer,
        FastStreamingConfig(...), tokenizer=tokenizer)``;
    ``breeze_infer.templates.select_template_name(request)``,
        ``get_template(name)`` and ``prepare_inputs(tokenizer, audio_tokenizer,
        model, [request], template, guidance_scale=..., guidance_scale_ref=None,
        guidance_scale_ins=None)``, after ``set_all_seeds(seed)``;
    ``runtime.iter_audio_chunks(inputs, request_id=..., seed=...,
        token_observer=...)``, whose chunks' ``audio`` is the codec's float32
        output at 24 kHz and which is closed, not abandoned, when a render is
        cancelled (the API does the same, to release the codec's per-request
        state).

And two things it reaches into, said here because they are not public:

    ``breeze_infer.templates._encode_prompt_audio(audio_tokenizer, path)``, the
        one place a reference becomes codec frames. Upstream reads a file
        there; this process has the sample as an array already, encodes it
        once per render exactly as ``breeze_infer.audio.encode_prompt_audio``
        does, and answers the templates' look-ups from that, so a section of
        twenty lines encodes each speaker's reference once and writes no
        temporary file (:meth:`Engine._references`);
    ``model.text_encoder``'s configurations' ``_attn_implementation``, which
        the text encoder reads at every forward pass (:meth:`Engine._use_attention`).

Why Flash attention 2 is the text encoder's alone
-------------------------------------------------
Transformers 4.57 reads any attention mask it hands Flash attention 2 as a
two-dimensional padding mask. Breeze's decode loops -- the backbone's static
cache and the depth decoder -- pass a four-dimensional additive mask, which
under Flash attention 2 would be unpadded into nonsense. The text encoder
(``T5Gemma2TextEncoder``) is different: with Flash attention 2 it builds a
padding mask and runs its own varlen kernel, and Flash attention 2 is the text
encoder's own preference in Breeze's code when nothing else is said. So
"Flash attention 2" here is the text encoder on Flash attention 2 and the
decoders on eager, exactly as both of upstream's entry points run them; "SDPA"
is every part on SDPA; "Eager" is every part on eager.

CFG, and the clone that cannot take it
--------------------------------------
Upstream's ``prepare_inputs`` refuses a guidance scale other than 1 for a
template with no negative prompt, and a plain voice clone (``ref_clone_tata``)
has none: only a line that carries an instruction -- a designed voice, or a
cloned voice under a direction -- has a negative to guide against. So a line
with no instruction is rendered at 1.0 whatever the configuration says, and
costs nothing extra; the configuration's CFG is every directed line's.

Nothing private crosses the pipe
--------------------------------
An error reply carries either a sentence this file wrote (:class:`Refusal`) or
an exception's class name, never a library's own message: a library message
can name a path, a tensor shape or the text being spoken. The words of a
script and of a transcript are never written to stderr either.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
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

PROTOCOL_VERSION = 1
"""Breeze's protocol's own version, counted from one.

Not shared with the other workers: they speak the same *framing*, and this one
answers ``render`` with one entry per take exactly as VibeVoice's protocol 2
does, but a number covering several workers is a number that has to change
when any of them changes."""

WORKER_NAME = "breeze"
"""What the handshake calls this process, and what the parent refuses to
proceed without."""

MARKER = "--model-chain-breeze-worker"
"""On the command line so this process is recognisable in a task manager and
by the stray sweep. Never a path, never a script line."""

MAX_HEADER = 1 << 20
MAX_PAYLOAD = 256 << 20
"""What this process accepts in a request: four voice samples of a minute each
are about 23 MB of float32, so a quarter of a gigabyte is a garbled length."""

MAX_REPLY_PAYLOAD = 2 << 30
"""What the parent accepts from its own worker in one reply."""

SAMPLE_RATE = 24000
"""What Breeze's codec produces and what every voice sample is prepared at."""

MAX_SPEAKERS = 4
MAX_TAKES = 4
"""The Voice Box's own limits, held here too because this process is the one
that would do the work."""

FRAME_RATE = 12.5
"""The codec's frames per second of audio: one backbone token is 80 ms."""

MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
"""Upstream's own limits and penalty (``infer.py``, ``breeze_infer/api.py``):
one call is at most 1,500 new frames -- two minutes -- in a 2,048-token
sequence, and the backbone's repeated tokens are penalised by 1.1."""

CFG_DEFAULT = 4.0
CFG_MAX = 8.0
"""What a directed or designed line is guided by when the request says
nothing, and the most it may ask for. Four is the authors' own advice."""

SAMPLING_TEMPERATURE = 0.9
SAMPLING_TOP_P = 1.0
"""What upstream samples with (``update_generation_config_for_breeze``): a
request that asks to sample without saying how gets exactly these."""

GAP_DEFAULT_MS = 150
GAP_MAX_MS = 2000
"""The silence between two lines of different speakers. Breeze renders each
line alone, so where VibeVoice paces a conversation itself, the join here is a
gap; none is put between two lines of one speaker, nor between the pieces of
one line."""

ATTENTION = {"flash_attention_2": "Flash attention 2", "sdpa": "SDPA", "eager": "Eager"}
"""The attention a render may ask for. See the module docstring for what Flash
attention 2 covers; ``eager`` is upstream's own choice."""

ATTENTION_DEFAULT = "flash_attention_2"

SOURCE_VARIABLE = "MODEL_CHAIN_BREEZE_SOURCE"
"""The environment variable naming the folder Breeze's own code is in."""

COMMIT_FILENAME = "COMMIT"
"""Written by the installer beside the code: the upstream commit it came from."""

PROGRESS_INTERVAL = 0.5
"""Seconds between two progress frames."""

SHUTDOWN_GRACE = 5.0

SPEAKER_TAG = "S0"
"""What every request calls its speaker, as upstream's CLI and API do. Each
line is one voice in one call, so there is never a second."""

FRAMES_PER_WORD = 7.0
FRAMES_PER_CJK = 5.0
"""How many codec frames a word or a CJK character is budgeted when a line is
measured against what one call allows. Generous on purpose -- a slow reader is
two words or two and a half characters a second -- because a piece that runs
out of room stops mid-sentence, and a line split once too often only costs a
seam nobody hears."""

HEADROOM = 0.85
"""The share of a call's room a piece may be estimated to fill."""

PROMPT_OVERHEAD = 8
"""Tokens a request's template adds around its texts (speaker tags, markers)."""

PIECE_TOKENS = 512
"""The most text tokens a piece of a line is reckoned to carry into its prompt.
A piece cut to a minute and a half of speech is about two hundred."""

_LENGTH = struct.Struct(">I")


class Refusal(ValueError):
    """A sentence this file wrote, and the only kind of message that crosses back."""


# --------------------------------------------------------------------------- #
# Framing -- byte-identical to the other workers, by agreement not by import
# --------------------------------------------------------------------------- #


def read_frame(stream, max_payload: int = MAX_PAYLOAD) -> "tuple[dict, bytes] | None":
    """One request, or ``None`` at end of input."""
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

    The same mechanism VibeVoice's worker uses: ``PR_SET_PDEATHSIG`` on Linux,
    with the parent re-read afterwards; on Windows the parent has already put
    this process in a job object, and this side only asks the kernel whether
    it is in one.
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
    """Whether the kernel agrees this process is in a job: ``job``, ``none`` or ``unknown``."""
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
    """One diagnostic line on stderr, which the parent logs. Never content, never a path."""
    try:
        sys.stderr.write(f"Breeze worker: {text}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _safe(exc: BaseException) -> str:
    """An exception as something that can cross the pipe: a :class:`Refusal`'s
    sentence, or a class name and nothing else."""
    if isinstance(exc, Refusal):
        return str(exc)
    return exc.__class__.__name__


# --------------------------------------------------------------------------- #
# Text: cleaning, measuring and splitting a line
# --------------------------------------------------------------------------- #


def _clean(text) -> str:
    """One line of text with its whitespace runs collapsed. Never empty-checked here."""
    return " ".join(str(text or "").split())


_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]")
_WORD = re.compile(r"[^\s぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]+")
_SENTENCE = re.compile(r"[^.!?。！？…]*(?:[.!?。！？…]+[\"'”’)\]」』]*|$)")
_CLAUSE = re.compile(r"[^,;:，；：、]*(?:[,;:，；：、]+|$)")


def estimate_frames(text: str) -> int:
    """The codec frames ``text`` is budgeted, spoken slowly (see :data:`FRAMES_PER_WORD`)."""
    words = str(text or "")
    cjk = len(_CJK.findall(words))
    latin = len(_WORD.findall(words))
    return int(math.ceil(cjk * FRAMES_PER_CJK + latin * FRAMES_PER_WORD))


def _parts(text: str, pattern) -> list:
    return [part.strip() for part in pattern.findall(text) if part and part.strip()]


def _joined(left: str, right: str) -> str:
    """Two pieces of a line put back together the way they were apart."""
    if not left:
        return right
    if not right:
        return left
    if _CJK.search(left[-1:]) or _CJK.search(right[:1]):
        return left + right
    return f"{left} {right}"


def split_text(text: str, limit: int) -> list:
    """``text`` as pieces whose :func:`estimate_frames` each fit ``limit``.

    The whole line when it fits. Otherwise sentences, packed greedily in
    order; a sentence that does not fit on its own is split at its clauses,
    and a clause that does not fit at its words (or, for CJK, its characters).
    Nothing is dropped and nothing is reordered: joined back, the pieces are
    the line.
    """
    words = _clean(text)
    if not words:
        return []
    limit = max(int(limit), 1)
    if estimate_frames(words) <= limit:
        return [words]
    units = []
    for sentence in _parts(words, _SENTENCE) or [words]:
        if estimate_frames(sentence) <= limit:
            units.append(sentence)
            continue
        for clause in _parts(sentence, _CLAUSE) or [sentence]:
            if estimate_frames(clause) <= limit:
                units.append(clause)
                continue
            units.extend(_by_tokens(clause, limit))
    pieces: list = []
    current = ""
    for unit in units:
        candidate = _joined(current, unit)
        if current and estimate_frames(candidate) > limit:
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _by_tokens(text: str, limit: int) -> list:
    """A clause too long on its own, cut at word (and CJK character) boundaries."""
    tokens = re.findall(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]|\S+",
                        text)
    pieces, current = [], ""
    for token in tokens:
        candidate = _joined(current, token)
        if current and estimate_frames(candidate) > limit:
            pieces.append(current)
            current = token
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #


def _float(value, fallback: float) -> float:
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


class VoiceEntry:
    """One speaker's voice as the request gave it: samples, transcript, or a description."""

    __slots__ = ("speaker", "samples", "transcript", "description", "instruction")

    def __init__(self, speaker, samples, transcript="", description="", instruction=""):
        self.speaker = int(speaker)
        self.samples = samples
        self.transcript = transcript
        self.description = description
        self.instruction = instruction


def voice_entries(voices, payload: bytes, speakers) -> dict:
    """``{speaker: VoiceEntry}`` for every speaker the lines use, cut from the payload.

    A voice is either a sample (``count`` above zero, at 24 kHz, with the
    sample's exact transcript) or a description and no sample. Anything else
    is a sentence naming the speaker, the one thing about a request that is
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
        if speaker in found:
            raise Refusal(f"Speaker {speaker} has two voices")
        transcript = _clean(entry.get("transcript"))
        description = _clean(entry.get("description"))
        instruction = _clean(entry.get("instruction"))
        samples = None
        if count > 0:
            if rate != SAMPLE_RATE:
                raise Refusal(f"a voice sample is at {rate} Hz rather than {SAMPLE_RATE}")
            if offset < 0 or offset + count > total:
                raise Refusal("a voice sample lies outside the audio that was sent")
            if not transcript:
                raise Refusal(f"Speaker {speaker}'s sample has no transcript")
            samples = numpy.frombuffer(payload, dtype="<f4", count=count,
                                       offset=offset * 4).astype(numpy.float32)
            if not numpy.all(numpy.isfinite(samples)):
                raise Refusal("a voice sample contains values that are not numbers")
        elif not description:
            raise Refusal(f"Speaker {speaker} has neither a sample nor a description")
        found[speaker] = VoiceEntry(speaker, samples, transcript, description, instruction)
    for speaker in speakers:
        if speaker not in found:
            raise Refusal(f"Speaker {speaker} has no voice")
    return found


class RenderRequest:
    """One ``render`` frame, validated, with its voices already cut out."""

    __slots__ = ("job", "lines", "voices", "seed", "takes", "cfg_scale", "attention",
                 "sampling", "temperature", "top_p", "max_new_tokens", "gap_ms")

    def __init__(self, job, lines, voices, seed=None, takes=1, cfg_scale=CFG_DEFAULT,
                 attention=ATTENTION_DEFAULT, sampling=True, temperature=SAMPLING_TEMPERATURE,
                 top_p=SAMPLING_TOP_P, max_new_tokens=None, gap_ms=GAP_DEFAULT_MS):
        self.job = job
        self.lines = lines
        self.voices = voices
        self.seed = seed
        self.takes = int(takes)
        self.cfg_scale = cfg_scale
        self.attention = attention
        self.sampling = bool(sampling)
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.gap_ms = gap_ms

    @property
    def seeds(self) -> "list[int] | None":
        if self.seed is None:
            return None
        return [int(self.seed) + index for index in range(self.takes)]

    @classmethod
    def parse(cls, header: dict, payload: bytes) -> "RenderRequest":
        lines = []
        for entry in header.get("lines") or ():
            if not isinstance(entry, dict):
                raise Refusal("a line is not an object")
            speaker = entry.get("speaker")
            if isinstance(speaker, bool):
                raise Refusal("a line has no speaker number")
            try:
                number = int(speaker)
            except (TypeError, ValueError):
                raise Refusal("a line has no speaker number") from None
            if number < 1 or number > MAX_SPEAKERS:
                raise Refusal(f"Speaker {number} is outside 1 to {MAX_SPEAKERS}")
            text = _clean(entry.get("text"))
            if not text:
                continue
            lines.append((number, text, _clean(entry.get("instruction"))))
        if not lines:
            raise Refusal("the section has no words")
        speakers = sorted({number for number, _text, _direction in lines})
        voices = voice_entries(header.get("voices"), payload, speakers)
        takes = _optional_int(header.get("takes"), "the number of takes") or 1
        if takes > MAX_TAKES:
            raise Refusal(f"a render makes 1 to {MAX_TAKES} takes")
        seed = _optional_int(header.get("seed"), "the seed")
        if takes > 1 and seed is None:
            raise Refusal("a render of several takes needs a seed")
        cfg_scale = _float(header.get("cfg_scale"), CFG_DEFAULT)
        if not 1.0 <= cfg_scale <= CFG_MAX:
            raise Refusal(f"the CFG scale must be from 1 to {CFG_MAX:g}")
        attention = str(header.get("attention") or ATTENTION_DEFAULT)
        if attention not in ATTENTION:
            raise Refusal("that attention is not one Breeze runs")
        sampling = header.get("sampling") is not False
        temperature = top_p = None
        if sampling:
            temperature = _float(header.get("temperature"), SAMPLING_TEMPERATURE)
            top_p = _float(header.get("top_p"), SAMPLING_TOP_P)
            if not 0.0 < temperature <= 2.0:
                raise Refusal("the temperature must be above 0 and at most 2")
            if not 0.0 < top_p <= 1.0:
                raise Refusal("top-p must be above 0 and at most 1")
        max_new_tokens = _optional_int(header.get("max_new_tokens"), "max_new_tokens")
        if not max_new_tokens:
            max_new_tokens = None
        elif max_new_tokens > MAX_NEW_TOKENS:
            raise Refusal(f"max_new_tokens is at most {MAX_NEW_TOKENS}")
        gap_ms = _optional_int(header.get("line_gap_ms"), "the line gap")
        if gap_ms is None:
            gap_ms = GAP_DEFAULT_MS
        if gap_ms > GAP_MAX_MS:
            raise Refusal(f"the line gap is at most {GAP_MAX_MS} ms")
        return cls(job=str(header.get("job") or ""), lines=lines, voices=voices, seed=seed,
                   takes=takes, cfg_scale=cfg_scale, attention=attention, sampling=sampling,
                   temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
                   gap_ms=gap_ms)


def float32_bytes(samples) -> bytes:
    """Samples as little-endian 32-bit float, exactly as made; a non-number is silence."""
    import numpy

    array = numpy.asarray(samples, dtype=numpy.float32).reshape(-1)
    return numpy.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0).astype("<f4").tobytes()


def flash_attention_available() -> bool:
    """Whether the ``flash-attn`` package is importable here, asked without importing it."""
    try:
        return importlib.util.find_spec("flash_attn") is not None
    except (ImportError, ValueError):
        return False


def compile_available() -> bool:
    """Whether ``torch.compile`` has a backend here: Triton, which Windows lacks by default.

    Upstream's fast path compiles the depth decoder and the codec's activations;
    without Triton those two stages stay eager and the CUDA-graph stages are
    the whole fast path.
    """
    try:
        return importlib.util.find_spec("triton") is not None
    except (ImportError, ValueError):
        return False


def attention_available() -> list:
    """The ids of :data:`ATTENTION` this runtime can run, in the table's order."""
    return [name for name in ATTENTION
            if name != "flash_attention_2" or flash_attention_available()]


# --------------------------------------------------------------------------- #
# The engine: Torch, transformers, qwen_tts and Breeze's code, behind one class
# --------------------------------------------------------------------------- #


def _source_folder() -> str:
    return str(os.environ.get(SOURCE_VARIABLE) or "")


def _bootstrap_source() -> str:
    """Put Breeze's own code on the path, once. The folder, or ``""``."""
    folder = _source_folder()
    if folder and os.path.isdir(folder) and folder not in sys.path:
        sys.path.insert(0, folder)
    return folder


def _commit() -> str:
    folder = _source_folder()
    if not folder:
        return ""
    try:
        with open(os.path.join(folder, COMMIT_FILENAME), "r", encoding="utf-8") as handle:
            return handle.read().strip()[:64]
    except OSError:
        return ""


class Upstream:
    """Breeze's own entry points, resolved once. A test hands :class:`Engine` another."""

    def __init__(self):
        _bootstrap_source()
        from pathlib import Path

        from breeze_infer import templates
        from breeze_infer.runtime import (
            load_runtime,
            set_all_seeds,
            update_generation_config_for_breeze,
        )
        from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

        self.Path = Path
        self.templates = templates
        self.load_runtime = load_runtime
        self.set_all_seeds = set_all_seeds
        self.update_generation_config_for_breeze = update_generation_config_for_breeze
        self.FastBreezeStreamingRuntime = FastBreezeStreamingRuntime
        self.FastStreamingConfig = FastStreamingConfig


class Engine:
    """Breeze TTS 2 on one card, one render at a time.

    Everything Torch-shaped is behind this class, constructed once per worker.
    Its two import points -- :meth:`_frameworks` and :meth:`_upstream` -- are
    methods so a test can hand it stand-ins and exercise the exact call
    sequence without a card. ``device`` is the card (``cuda``, the one card
    this process can see) everywhere but a test.
    """

    def __init__(self, device: str = "cuda", upstream=None):
        self.device = str(device or "cuda")
        self.torch = None
        self.numpy = None
        self._given = upstream
        self.up = None
        self.tokenizer = None
        self.model = None
        self.audio_tokenizer = None
        self.runtime = None
        self.model_dir = ""
        self.fast_path = False
        self.backbone = ""
        self.attention = ""
        self.weights_bytes = 0
        self.compiled = False

    # -- imports ----------------------------------------------------------- #

    def _frameworks(self):
        if self.torch is None:
            import numpy
            import torch

            self.torch, self.numpy = torch, numpy
        return self.torch, self.numpy

    def _upstream(self):
        if self.up is None:
            self.up = self._given if self._given is not None else Upstream()
        return self.up

    # -- the card ---------------------------------------------------------- #

    def probe(self) -> dict:
        """Which card this process is on, and how much of it is free."""
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
        return self.device.startswith("cuda")

    def resident(self) -> int:
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

    # -- loading ----------------------------------------------------------- #

    @staticmethod
    def _backbone_of(attention: str) -> str:
        """What the decoders run on for an attention choice: SDPA, or eager."""
        return "sdpa" if attention == "sdpa" else "eager"

    def load(self, model_dir: str, attention: str = ATTENTION_DEFAULT,
             fast_path: bool = False) -> dict:
        """Load the checkpoint from one local directory, or keep the one loaded.

        A second ``load`` of what is loaded is a no-op reply, after the text
        encoder's attention is set (that needs no reload). A different
        directory, a different fast path, or a different attention for the
        decoders replaces what is loaded: the decoders' attention is chosen
        when the model is built, and the fast path's graphs are captured
        against it.
        """
        wanted = str(model_dir or "")
        attention = str(attention or ATTENTION_DEFAULT)
        if attention not in ATTENTION:
            raise Refusal("that attention is not one Breeze runs")
        if attention == "flash_attention_2" and not flash_attention_available():
            raise Refusal("Flash attention 2 needs the flash-attn package, which Breeze's "
                          "runtime does not have")
        fast = bool(fast_path)
        backbone = self._backbone_of(attention)
        if self.model is not None and wanted == self.model_dir and fast == self.fast_path \
                and backbone == self.backbone and (not fast or attention == self.attention):
            self._use_attention(attention)
            return {"resident_bytes": self.resident(), "weights_bytes": self.weights_bytes,
                    "load_seconds": 0.0, "already_loaded": True, "attention": self.attention,
                    "fast_path": self.fast_path}
        if not wanted or not os.path.isdir(wanted):
            raise Refusal("the model directory does not exist")
        if not os.path.isdir(os.path.join(wanted, "audio_tokenizer")):
            raise Refusal("the model directory has no audio_tokenizer folder; install the "
                          "model again")
        if self.model is not None:
            self.unload()
        torch, _numpy = self._frameworks()
        up = self._upstream()
        began = time.monotonic()
        device = "cuda:0" if self.device == "cuda" else self.device
        tokenizer, model, audio_tokenizer = up.load_runtime(
            up.Path(wanted), device=device, attn_implementation=backbone)
        up.update_generation_config_for_breeze(model)
        compiled = fast and compile_available()
        config = up.FastStreamingConfig(
            max_new_tokens=MAX_NEW_TOKENS, max_seq_len=MAX_SEQ_LEN, fast_all=None,
            fast_text_encoder=fast, fast_backbone_prefill=False, fast_backbone_decode=fast,
            fast_depth_decoder=compiled, fast_codec=compiled,
            repetition_penalty=REPETITION_PENALTY)
        runtime = up.FastBreezeStreamingRuntime(model, audio_tokenizer, config,
                                                tokenizer=tokenizer)
        weights = 0
        for owner in (model, getattr(audio_tokenizer, "model", None)):
            if owner is None:
                continue
            for parameter in owner.parameters():
                weights += int(parameter.numel()) * int(parameter.element_size())
        self.tokenizer, self.model, self.audio_tokenizer = tokenizer, model, audio_tokenizer
        self.runtime = runtime
        self.model_dir = wanted
        self.fast_path = fast
        self.compiled = compiled
        self.backbone = backbone
        self.attention = ""
        self.weights_bytes = weights
        self._use_attention(attention)
        elapsed = time.monotonic() - began
        _note(f"loaded in {elapsed:.1f} s — {weights / float(1 << 30):.1f} GiB of weights, "
              f"{ATTENTION[attention]}, decoders {backbone}"
              f"{', fast path' + ('' if compiled else ' (CUDA graphs only)') if fast else ''}")
        return {"resident_bytes": self.resident(), "weights_bytes": weights,
                "load_seconds": elapsed, "attention": attention, "fast_path": fast,
                "compiled": compiled}

    def _use_attention(self, attention: str) -> None:
        """Set the text encoder's attention: Flash attention 2, or the decoders' own.

        The text encoder reads ``_attn_implementation`` from its configuration
        at every forward pass, so this is an attribute on each configuration
        its modules hold, not a reload.
        """
        wanted = "flash_attention_2" if attention == "flash_attention_2" else self.backbone
        if attention == self.attention:
            return
        if wanted == "flash_attention_2" and not flash_attention_available():
            raise Refusal("Flash attention 2 needs the flash-attn package, which Breeze's "
                          "runtime does not have")
        encoder = getattr(self.model, "text_encoder", None)
        if encoder is not None:
            seen = set()
            modules = encoder.modules() if hasattr(encoder, "modules") else [encoder]
            for module in modules:
                config = getattr(module, "config", None)
                if config is None or id(config) in seen:
                    continue
                seen.add(id(config))
                try:
                    config._attn_implementation = wanted
                except Exception:  # noqa: BLE001 - a configuration without the field
                    continue
        self.attention = attention

    def unload(self) -> dict:
        """Let go of the model and give the card its memory back."""
        self.runtime = None
        self.model = None
        self.tokenizer = None
        self.audio_tokenizer = None
        self.model_dir = ""
        self.fast_path = False
        self.compiled = False
        self.backbone = ""
        self.attention = ""
        self.weights_bytes = 0
        gc.collect()
        if self.torch is not None and self._on_card():
            try:
                self.torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - nothing to empty
                pass
        return {"resident_bytes": self.resident()}

    # -- rendering --------------------------------------------------------- #

    def render(self, request: RenderRequest, cancelled: threading.Event,
               on_progress=None) -> "tuple[dict, bytes]":
        """One section, line by line, as one to four takes, take k at seed + k.

        Every line is its own call in its speaker's voice: a sample is cloned
        from its transcript (a direction, the line's own or the slot's, makes
        it a directed clone), a description designs a voice. A line longer
        than one call is split at its sentence ends first (:func:`split_text`)
        and every piece is cloned from the same reference. Between two lines
        of different speakers goes the request's gap; between two lines of one
        speaker, and between pieces, nothing.

        The cancel flag is read at every chunk the codec hands back, which is
        two frames -- 160 ms of audio -- on the eager path.
        """
        if self.model is None or self.runtime is None:
            raise Refusal("no model is loaded")
        torch, numpy = self._frameworks()
        if request.attention != self.attention:
            if self._backbone_of(request.attention) != self.backbone \
                    or (self.fast_path and request.attention != self.attention):
                raise Refusal("the model is loaded with another attention; load it again")
            self._use_attention(request.attention)
        takes = max(1, int(request.takes or 1))
        seeds = request.seeds
        gap = numpy.zeros(int(round(SAMPLE_RATE * request.gap_ms / 1000.0)), dtype=numpy.float32)
        progress = _Progress(on_progress, len(request.lines), takes)
        if self._on_card():
            torch.cuda.reset_peak_memory_stats(0)
        began = time.monotonic()
        made = [[] for _take in range(takes)]
        tokens = [0] * takes
        capped = [False] * takes
        try:
            with self._references(request) as keys:
                for take in range(takes):
                    seed = None if seeds is None else seeds[take]
                    previous = None
                    for index, (speaker, text, direction) in enumerate(request.lines):
                        if cancelled.is_set():
                            break
                        progress.at(index, take)
                        if previous is not None and previous != speaker and gap.size:
                            made[take].append(gap)
                            progress.add(gap.size)
                        previous = speaker
                        for piece in self._pieces(request, speaker, text, direction, keys):
                            if cancelled.is_set():
                                break
                            audio, frames, reached = self._speak(
                                request, piece, seed, cancelled, progress,
                                f"{request.job or 'render'}-{take + 1}-{index + 1}")
                            if audio is not None and audio.size:
                                made[take].append(audio)
                            tokens[take] += frames
                            capped[take] = capped[take] or reached
                    if cancelled.is_set():
                        break
            render_seconds = time.monotonic() - began
            peak = self.peak()
        finally:
            gc.collect()
            if self._on_card():
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001 - nothing to empty
                    pass
        arrays = [numpy.concatenate(pieces) if pieces else numpy.zeros(0, dtype=numpy.float32)
                  for pieces in made]
        answers = []
        for index, samples in enumerate(arrays):
            answers.append({"seed": None if seeds is None else seeds[index],
                            "samples": int(samples.size),
                            "seconds": samples.size / float(SAMPLE_RATE),
                            "tokens": int(tokens[index]), "capped": bool(capped[index])})
        if any(answer["capped"] for answer in answers):
            _note("a line reached the room one call allows before it ended")
        longest = max(answer["seconds"] for answer in answers)
        _note(f"rendered {len(answers)} take(s) of {len(request.lines)} line(s), the longest "
              f"{longest:.1f} s of audio, in {render_seconds:.1f} s — "
              f"{sum(tokens)} frame(s), peak {peak / float(1 << 30):.1f} GiB, "
              f"{ATTENTION[request.attention]}{', cancelled' if cancelled.is_set() else ''}")
        return {
            "format": "f32le",
            "sample_rate": SAMPLE_RATE,
            "takes": answers,
            "seconds": answers[0]["seconds"],
            "render_seconds": render_seconds,
            "peak_bytes": peak,
            "resident_bytes": self.resident(),
            "cancelled": bool(cancelled.is_set()),
            "tokens": sum(tokens),
            "capped": any(capped),
        }, b"".join(float32_bytes(samples) for samples in arrays)

    def _references(self, request: RenderRequest):
        """Each speaker's sample as codec frames, once, for the templates to find.

        A context manager: while it is open, upstream's
        ``templates._encode_prompt_audio`` answers this render's keys from
        frames encoded here, exactly as ``breeze_infer.audio.encode_prompt_audio``
        encodes a file -- the samples are 24 kHz float32 already, so the file
        it would have read is the array it has -- and anything else as before.
        """
        engine = self
        templates = self._upstream().templates

        class _Opened:
            def __enter__(self):
                torch, _numpy = engine._frameworks()
                self.keys = {}
                self.frames = {}
                for speaker, voice in sorted(request.voices.items()):
                    if voice.samples is None:
                        continue
                    encoded = engine.audio_tokenizer.encode(voice.samples, sr=SAMPLE_RATE)
                    codes = torch.as_tensor(encoded["audio_codes"][0], dtype=torch.int16)
                    if codes.ndim != 2:
                        raise Refusal(f"Speaker {speaker}'s sample could not be encoded")
                    key = f"model-chain-voice-{speaker}"
                    self.keys[speaker] = key
                    self.frames[key] = codes.cpu().contiguous()
                self.original = getattr(templates, "_encode_prompt_audio", None)
                frames = self.frames
                original = self.original

                def encode(audio_tokenizer, audio_path):
                    found = frames.get(audio_path) if isinstance(audio_path, str) else None
                    if found is not None:
                        return found
                    return original(audio_tokenizer, audio_path)

                templates._encode_prompt_audio = encode
                return self

            def __exit__(self, *_exc):
                templates._encode_prompt_audio = self.original
                return False

        return _Opened()

    def _request_for(self, voice: "VoiceEntry", text: str, direction: str,
                     keys) -> "tuple[dict, str]":
        """The request upstream's templates read for one piece, and its instruction."""
        request = {"id": "model-chain", "text": text, "speaker": SPEAKER_TAG}
        instruction = direction or voice.instruction
        if voice.samples is not None:
            request["ref_audio_path"] = keys.keys[voice.speaker]
            request["ref_text"] = voice.transcript
        else:
            # A designed voice: the description is the instruction, and a
            # line's own direction is said after it.
            instruction = f"{voice.description} {instruction}".strip() if instruction \
                else voice.description
        if instruction:
            request["instruction"] = instruction
        return request, instruction

    def _pieces(self, request: RenderRequest, speaker: int, text: str, direction: str,
                keys) -> list:
        """The line as the calls it takes: ``[(request, cfg, cap)]``.

        A call's prompt is the reference's frames, its transcript, the
        instruction and the piece's own words, and what it may generate is
        the rest of the 2,048-token sequence, at most the cap. A piece's own
        words are bounded by :data:`PIECE_TOKENS` -- a piece of a minute's
        speech is a fraction of that -- so the room is worked out once for the
        line and every piece is cut to fit it.
        """
        voice = request.voices[speaker]
        cap = min(int(request.max_new_tokens or MAX_NEW_TOKENS), MAX_NEW_TOKENS)
        _probe, instruction = self._request_for(voice, text, direction, keys)
        fixed = PROMPT_OVERHEAD + self._tokens(instruction)
        if voice.samples is not None:
            fixed += self._tokens(voice.transcript) + int(keys.frames[keys.keys[speaker]].shape[0])
        room = min(cap, MAX_SEQ_LEN - 1 - fixed - min(self._tokens(text), PIECE_TOKENS))
        if room <= 0:
            raise Refusal(f"Speaker {speaker}'s sample and transcript leave no room for a line; "
                          f"use a shorter sample")
        limit = max(1, int(room * HEADROOM))
        cfg = float(request.cfg_scale) if instruction else 1.0
        found = []
        for piece in split_text(text, limit) or [text]:
            entry, _instruction = self._request_for(voice, piece, direction, keys)
            found.append((entry, cfg, cap))
        return found

    def _tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:  # noqa: BLE001 - a tokenizer without the keyword
            return len(str(text).split())

    def _speak(self, request: RenderRequest, piece, seed, cancelled: threading.Event,
               progress: "_Progress", request_id: str):
        """One call: ``(float32 samples, frames generated, whether it ran out of room)``."""
        import dataclasses

        torch, numpy = self._frameworks()
        up = self._upstream()
        entry, cfg, cap = piece
        self.runtime.config = dataclasses.replace(
            self.runtime.config, max_new_tokens=cap, do_sample=bool(request.sampling),
            temperature=float(request.temperature) if request.sampling else None,
            top_p=float(request.top_p) if request.sampling else None)
        if seed is not None:
            up.set_all_seeds(int(seed))
        template = up.templates.get_template(up.templates.select_template_name(entry))
        inputs = up.templates.prepare_inputs(
            self.tokenizer, self.audio_tokenizer, self.model, [entry], template,
            guidance_scale=float(cfg), guidance_scale_ref=None, guidance_scale_ins=None)
        prompt = int(inputs["input_ids"].shape[-1])
        negative = inputs.get("cfg_negative_prompt_ids") if hasattr(inputs, "get") else None
        if negative is not None:
            prompt = max(prompt, int(negative.shape[-1]))
        counted = [0]

        def observe(_frame) -> None:
            counted[0] += 1

        chunks = []
        stream = self.runtime.iter_audio_chunks(inputs, request_id=request_id, seed=seed,
                                                token_observer=observe)
        try:
            for chunk in stream:
                audio = numpy.asarray(chunk.audio, dtype=numpy.float32).reshape(-1)
                chunks.append(audio)
                progress.add(audio.size)
                if cancelled.is_set():
                    break
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        frames = counted[0]
        reached = frames >= cap or prompt + frames >= MAX_SEQ_LEN - 1
        samples = numpy.concatenate(chunks) if chunks else numpy.zeros(0, dtype=numpy.float32)
        return samples, frames, bool(reached and not cancelled.is_set())


class _Progress:
    """Seconds of audio made in the take being rendered, sent every half second."""

    def __init__(self, report, lines: int, takes: int, interval: float = PROGRESS_INTERVAL):
        self._report = report
        self._interval = float(interval)
        self._last = 0.0
        self.lines = int(lines)
        self.takes = int(takes)
        self.line = 0
        self.take = 0
        self.samples = 0

    def at(self, line: int, take: int) -> None:
        if take != self.take:
            self.samples = 0
        self.line, self.take = int(line), int(take)
        self._send(force=True)

    def add(self, count: int) -> None:
        self.samples += int(count)
        self._send()

    def _send(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < self._interval:
            return
        self._last = now
        if self._report is None:
            return
        try:
            self._report({"seconds": round(self.samples / float(SAMPLE_RATE), 2),
                          "line": self.line + 1, "lines": self.lines,
                          "take": self.take + 1, "takes": self.takes})
        except Exception:  # noqa: BLE001 - progress is advisory
            pass


# --------------------------------------------------------------------------- #
# The worker: the lane, the writer lock, and the one render
# --------------------------------------------------------------------------- #


class Worker:
    """The lane and the render in flight. One model, one render at a time."""

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
        self._lane = threading.Thread(target=self._lane_loop, name="breeze-lane", daemon=True)
        self._lane.start()

    def send(self, header: dict, payload: bytes = b"") -> None:
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
            "attention": engine.attention if engine is not None else "",
            "fast_path": bool(engine.fast_path) if engine is not None else False,
        }


# --------------------------------------------------------------------------- #
# The command loop
# --------------------------------------------------------------------------- #


def serve(stdin, stdout, engine_factory=None) -> int:
    """Read frames until end of input. Returns the process exit status.

    ``status`` and ``cancel`` are answered on this thread while the lane is
    inside the model; ``load``, ``unload`` and ``render`` are handed to the
    lane, which runs them one at a time.
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
                        "breeze": _commit(),
                        "transformers": _package_version("transformers"),
                        "qwen_tts": _package_version("qwen-tts"),
                        "sample_rate": SAMPLE_RATE,
                        "attention": attention_available(),
                        "compile": compile_available(),
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
    """Put one request on the lane, and guarantee it is answered."""

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
                               str(header.get("attention") or ATTENTION_DEFAULT),
                               header.get("fast_path") is True)
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

    Freed *before* the reply goes out: the Voice Box sends a script's next
    section the moment it reads this one's reply, and would otherwise race this
    line and be refused "one render at a time" for a render that had ended.
    """
    try:
        def report(progress: dict) -> None:
            worker.send({"id": request_id, "progress": dict(progress)})

        found, audio = worker.engine.render(request, cancelled, report)
    finally:
        worker.end_render()
    answer = {"ok": True}
    answer.update(found)
    reply(request_id, answer, audio)


def _package_version(name: str) -> str:
    """One installed package's version, or an empty string. Never raises."""
    try:
        import importlib.metadata as metadata

        return str(metadata.version(name))
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Running this file directly
# --------------------------------------------------------------------------- #


def selftest() -> int:
    """Prove the staged runtime imports Breeze. One JSON line out.

    Run by the installer against a *staged* interpreter before anything is
    promoted, with no model: torch, transformers and ``qwen_tts`` are
    imported, Breeze's own entry points are resolved from the source folder
    the environment names, and the versions are printed. ``sox`` is reported
    rather than required: its Python wrapper only warns when the program is
    not installed, and nothing Breeze uses calls it.

    Flash attention 2 is *imported*, not just found: the community wheel is a
    compiled extension built against one torch, and a wheel that does not
    match this runtime's fails exactly here. A failure is reported
    (``flash_attn_loads`` false, with the reason) and is not the self-test's:
    the runtime still renders with SDPA or Eager.
    """
    report = {"ok": False, "error": "", "cuda": False}
    try:
        import torch

        report["torch"] = str(getattr(torch, "__version__", ""))
        report["transformers"] = _package_version("transformers")
        report["torchaudio"] = _package_version("torchaudio")
        report["qwen_tts"] = _package_version("qwen-tts")
        report["flash_attn"] = _package_version("flash-attn")
        report["flash_attn_loads"] = False
        report["flash_attn_error"] = ""
        if flash_attention_available():
            try:
                import flash_attn  # noqa: F401 - the import is the check

                report["flash_attn_loads"] = True
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                report["flash_attn_error"] = f"{exc.__class__.__name__}: {exc}"[:400]
        report["source"] = bool(_bootstrap_source())
        Upstream()
        report["breeze"] = _commit()
        report["cuda"] = bool(torch.cuda.is_available())
        report["attention"] = attention_available()
        report["compile"] = compile_available()
        report["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the report is the answer
        report["error"] = f"{exc.__class__.__name__}: {exc}"
    sys.stdout.write(json.dumps(report) + "\n")
    sys.stdout.flush()
    return 0 if report["ok"] else 1


def _claim_stdout():
    """The pipe to the parent, taken away from everything that prints.

    Breeze's own code prints to standard output while it loads and builds its
    graphs, and a line of text in the middle of the frame stream is a header
    length the parent cannot read. So descriptor 1 is duplicated for the
    protocol alone, and descriptor 1 itself -- with ``sys.stdout`` -- is pointed
    at standard error, which the parent reads line by line into its log.
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
