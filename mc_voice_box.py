"""Voice Box: the samples, prompts, configurations, pipelines and outputs of a
render, and the service that renders them with whichever engine is active.

The page (``javascript/voice_box.js``) is four stages in a row -- INPUT, PROMPT,
CONFIGURATION, OUTPUTS -- and a pipeline is one pass through them, kept like a
conversation is kept: a file of its own with its prompt, the configuration it
points at and the outputs it made. Everything a live page changes lives in
files under the voice data root, never in Forge's settings, so *Apply settings*
cannot write a stale copy over a folder the user chose a minute ago.

This module does the arithmetic and the files. It never touches a card: a
render asks for its card through the client :func:`use_turns` was handed
(:mod:`mc_turns_guests`), and speaks through the active engine's runtime
(:mod:`mc_voice_box_engines`: VibeVoice 7B or Breeze TTS 2), or the one
:func:`use_runtime` names in a test. Both are seams, because the independence
of the voice side from the memory side is an invariant
(``tests/test_voice_independence.py``) and because a render service whose rules
could only be tested with a GPU would be one whose rules nobody had tested.

What an engine renders with -- VibeVoice's diffusion steps and solver, Breeze's
line gap -- is its own vocabulary, and this module reads it as data from the
engine's adapter (``FIELDS``, ``clean``, ``refuse``, ``infotext``,
``capabilities``) rather than knowing it; docs/25-voice-box-engines.md is the
contract. A configuration keeps every engine's options side by side, so moving
between engines loses nothing.

Sound is 16-bit mono PCM at 24 kHz throughout -- the rate both engines speak at.
A sample arrives from the browser already trimmed to the selection the user
made, as a WAV, and :func:`mc_voice_reference.normalize` makes it canonical the
way it does for every engine that clones from a recording. An output is what
the worker returned, section by section, with the silences the prompt asked
for between them.
"""

from __future__ import annotations

import array
import json
import logging
import os
import re
import secrets
import shutil
import struct
import threading
import time
import wave
import io
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mc_voice_box_engines as engines
import mc_voice_box_fields as box_fields
import mc_voice_box_render as render_types
import mc_voice_paths as paths
import mc_voice_reference as reference

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""


class VoiceBoxError(RuntimeError):
    """A request the Voice Box will not carry out, in a sentence the page shows."""

    status = 400


class NotFound(VoiceBoxError):
    status = 404


class Unavailable(VoiceBoxError):
    """Something this machine lacks right now, which the page can say and offer to fix."""

    status = 409


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DIRNAME = "voice_box"
SAMPLES_DIRNAME = "samples"
CONFIGURATIONS_DIRNAME = "configurations"
PIPELINES_DIRNAME = "pipelines"
OUTPUTS_DIRNAME = "outputs"
PROMPTS_FILENAME = "prompts.json"
SETTINGS_FILENAME = "settings.json"
AUDIO_FILENAME = "audio.wav"
MP3_FILENAME = "audio.mp3"
"""An output's sound is an MP3 when this Forge can encode one and a WAV when it
cannot -- and for every render made before outputs were MP3s. A sample is
always a WAV: it is what the model is conditioned on, and never lossy."""
META_FILENAME = "meta.json"

SAMPLE_RATE = 24000
"""Both engines' rate, for samples and outputs alike."""

PEAKS = 240
"""Points in a waveform drawn from a list rather than from the audio."""

MAX_SPEAKERS = 4
MAX_SAMPLE_BYTES = 64 * 1024 * 1024
SAMPLE_MIN_SECONDS = 3.0
SAMPLE_MAX_SECONDS = 60.0
SAMPLE_MIN_PEAK = 0.01
MAX_TITLE_CHARS = 80
MAX_PROMPT_CHARS = 20_000
HISTORY_LIMIT = 100
"""Prompts remembered. Favourites are never counted against it."""
JOBS_KEPT = 50

PAUSE_DEFAULT_MS = 700
PAUSE_MIN_MS = 50
PAUSE_MAX_MS = 10_000

MAX_TAKES = 4
"""The most takes one render makes when an engine does not say (both say 4)."""
SEED_MAX = 2**31 - 1

MAX_TRANSCRIPT_CHARS = 2000
MAX_DESCRIPTION_CHARS = 1000
MAX_DIRECTION_CHARS = 300
MAX_DESIGN_TEXT_CHARS = 1000
"""A sample's transcript, a slot's voice description and standing direction,
and the sentence a designed voice speaks: each a few lines of text at most."""

TRANSCRIPT_SOURCES = ("typed", "whisper", "design")

DESIGN_TEXT = ("Hello, and welcome. This is the voice I will use from now on: clear, steady "
               "and easy to listen to, whether the story is quiet or full of surprises.")
DESIGN_TEXT_ZH = ("你好，欢迎收听。从现在开始，我会用这个声音为你讲述每一个故事，无论平静还是"
                  "充满惊喜，都清楚而自然。")
"""What a designed voice says when the page sends no sentence of its own: about
nine seconds, comfortably inside a sample's three to sixty, and the transcript
of the sample it becomes. Chinese when the description is mostly Chinese,
because Breeze's authors ask that the description's language match the text's."""

TRANSCRIBE_RATE = 16000
"""The rate Whisper hears at, which is the rate Voice Chat's dictation sends."""

KIND_RENDER, KIND_DESIGN = "render", "design"

MP3_BITRATE = 160_000
MP3_QUALITY = 0
"""The most an MP3 can hold of the model's sound: 160 kb/s is the ceiling of
MPEG-2 Layer III, the MP3 of a 24 kHz stream, and LAME's quality 0 is its
slowest and most careful search for where the bits go. Mono at 24 kHz because
that is what the model makes: a second channel would be a copy, and a higher
rate would be a resample of a sound with nothing above 12 kHz to keep. Made
straight from the model's 32-bit float samples, in memory -- no WAV is written
first. Constant bitrate so a seek lands where the page asks, and the encoder's
delay and padding are written into the file's own header, so a decoder gives
back exactly the samples that went in and a looped render has no gap. About
1.2 MB a minute, where a WAV of the same sound is 2.9."""
MEDIA_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav"}
SOFTWARE = "Voice Box (VibeVoice)"
"""What a VibeVoice render's file says made it; another engine's says its own
name the same way (:func:`software`)."""


def software(label: str = "VibeVoice") -> str:
    return f"Voice Box ({label or 'VibeVoice'})"


ENVELOPE = reference.Envelope(engine="VibeVoice", minimum_seconds=SAMPLE_MIN_SECONDS,
                              maximum_seconds=SAMPLE_MAX_SECONDS,
                              maximum_bytes=MAX_SAMPLE_BYTES, target_rate=SAMPLE_RATE,
                              minimum_peak=SAMPLE_MIN_PEAK)
"""What a sample may be. Three seconds is the least either engine conditions
on usefully; a minute is more than any speaker prompt needs and keeps a sample
a sample rather than an archive. A refusal names the engine that is active
(:func:`_envelope`), since samples serve both."""


def _envelope() -> reference.Envelope:
    return reference.Envelope(engine=engines.label(), minimum_seconds=SAMPLE_MIN_SECONDS,
                              maximum_seconds=SAMPLE_MAX_SECONDS,
                              maximum_bytes=MAX_SAMPLE_BYTES, target_rate=SAMPLE_RATE,
                              minimum_peak=SAMPLE_MIN_PEAK)

QUEUED, WAITING, LOADING, RENDERING, DONE, FAILED, CANCELLED = (
    "queued", "waiting", "loading", "rendering", "done", "failed", "cancelled")
LIVE = (QUEUED, WAITING, LOADING, RENDERING)

TURN_GRANTED, TURN_BLOCKED, TURN_CANCELLED = "granted", "blocked", "cancelled"
"""The turn system's words for the phases a render acts on. ``mc_turns_guests``
holds them equal to the constants, so this module need not import them."""

_lock = threading.RLock()
_turns = None
_runtime_module = None
_engine_module = None
_wait = 0.25


# --------------------------------------------------------------------------- #
# Seams
# --------------------------------------------------------------------------- #


def use_turns(client) -> None:
    """The turn client renders ask for their card through. Set by mc_turns_guests."""
    global _turns
    _turns = client


def turns():
    """The client :func:`use_turns` was handed, or None on a WebUI with no cards."""
    return _turns


def use_runtime(module) -> None:
    global _runtime_module
    _runtime_module = module


def use_engine(module) -> None:
    global _engine_module
    _engine_module = module


def _runtime():
    """The active engine's runtime, or the one a test handed :func:`use_runtime`."""
    if _runtime_module is not None:
        return _runtime_module
    return engines.runtime()


def _engine():
    """The active engine's adapter, or the one a test handed :func:`use_engine`."""
    if _engine_module is not None:
        return _engine_module
    return engines.adapter()


def _engine_id(engine=None) -> str:
    """The registry's id for ``engine``: its own ``ID``, else the active engine's."""
    found = str(getattr(engine if engine is not None else _engine(), "ID", "") or "")
    return found if engines.known(found) else engines.active_id()


def _vocabulary(engine_id: str):
    """The adapter whose fields, infotext and capabilities ``engine_id`` renders with."""
    return engines.adapter(engine_id)


def _capabilities(engine=None) -> dict:
    """What the engine can be asked for beyond a script and its samples."""
    engine = engine if engine is not None else _engine()
    ask = getattr(engine, "capabilities", None)
    if ask is None:
        ask = _vocabulary(_engine_id(engine)).capabilities
    try:
        found = ask()
    except Exception:
        logger.debug("Model Chain: the Voice Box could not read the engine's capabilities",
                     exc_info=True)
        return {}
    return found if isinstance(found, dict) else {}


def _label(engine=None) -> str:
    engine = engine if engine is not None else _engine()
    return str(getattr(engine, "LABEL", "") or engines.label(_engine_id(engine)))


def _max_takes(engine_id: str) -> int:
    try:
        return max(1, int(getattr(_vocabulary(engine_id), "MAX_TAKES", MAX_TAKES)))
    except Exception:
        return MAX_TAKES


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def root() -> Path:
    return paths.data_root() / DIRNAME


def _folder(name: str) -> Path:
    found = root() / name
    found.mkdir(parents=True, exist_ok=True)
    return found


def _read_json(path: Path, default):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            found = json.load(handle)
    except FileNotFoundError:
        return default
    except (OSError, ValueError):
        logger.warning("Model Chain: Voice Box could not read %s; starting it afresh",
                       path.name, exc_info=True)
        return default
    return found if isinstance(found, type(default)) else default


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + ".tmp")
    with open(scratch, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    os.replace(scratch, path)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + ".tmp")
    with open(scratch, "wb") as handle:
        handle.write(data)
    os.replace(scratch, path)


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return secrets.token_hex(8)


_ID = re.compile(r"^[0-9a-f]{16}$")


def _identifier(value) -> str:
    text = str(value or "").strip().lower()
    if not _ID.match(text):
        raise NotFound("That item is not in the Voice Box.")
    return text


def _title(value, fallback: str = "") -> str:
    text = " ".join(str(value or "").split())[:MAX_TITLE_CHARS]
    return text or fallback


_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _safe_filename(text: str, fallback: str = "render") -> str:
    cleaned = _UNSAFE.sub(" ", str(text or "")).strip(" .")
    cleaned = " ".join(cleaned.split())[:MAX_TITLE_CHARS]
    return cleaned or fallback


# --------------------------------------------------------------------------- #
# Sound
# --------------------------------------------------------------------------- #


def wav_bytes(pcm16: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """A mono 16-bit WAV around ``pcm16``."""
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(pcm16)
    return out.getvalue()


def pcm_of(data: bytes) -> tuple[bytes, int]:
    """``(pcm16, rate)`` of a mono 16-bit WAV this module wrote."""
    with wave.open(io.BytesIO(data), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise VoiceBoxError("That file is not the Voice Box's own mono 16-bit WAV.")
        return handle.readframes(handle.getnframes()), int(handle.getframerate())


def seconds_of(pcm16: bytes, rate: int = SAMPLE_RATE) -> float:
    return len(pcm16) / 2 / float(rate or SAMPLE_RATE)


def silence(milliseconds: int, rate: int = SAMPLE_RATE) -> bytes:
    return bytes(2 * int(round(rate * max(int(milliseconds), 0) / 1000.0)))


def peaks_of(pcm16: bytes, count: int = PEAKS) -> list[float]:
    """``count`` peak levels 0..1 across the sound, for a waveform drawn without decoding.

    NumPy when the host has it (Forge always does), the array module when it
    does not: a forty-minute render is sixty million samples, and the difference
    is the difference between a page that draws and one that waits.
    """
    total = len(pcm16) // 2
    if total <= 0:
        return [0.0] * count
    try:
        import numpy

        samples = numpy.frombuffer(pcm16[:total * 2], dtype="<i2")
        edges = numpy.linspace(0, total, count + 1, dtype=numpy.int64)
        found = []
        for start, stop in zip(edges[:-1], edges[1:]):
            chunk = samples[start:stop]
            found.append(float(numpy.abs(chunk.astype(numpy.int32)).max()) / 32768.0
                         if chunk.size else 0.0)
        return [min(max(level, 0.0), 1.0) for level in found]
    except ImportError:
        pass
    samples = array.array("h")
    samples.frombytes(pcm16[:total * 2])
    if struct.pack("<h", 1) != struct.pack("=h", 1):
        samples.byteswap()
    found = []
    for bucket in range(count):
        start = bucket * total // count
        stop = (bucket + 1) * total // count
        chunk = samples[start:stop]
        found.append(min(max(abs(value) for value in chunk) / 32768.0, 1.0) if len(chunk) else 0.0)
    return found


def pcm16_of(samples: bytes) -> bytes:
    """Little-endian 32-bit float samples as PCM16, clipped to -1..1: a WAV's, and the peaks'."""
    total = len(samples) // 4
    try:
        import numpy

        found = numpy.frombuffer(samples[:total * 4], dtype="<f4")
        return (numpy.clip(found, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    except ImportError:
        pass
    floats = array.array("f")
    floats.frombytes(samples[:total * 4])
    if struct.pack("<f", 1.0) != struct.pack("=f", 1.0):
        floats.byteswap()
    ints = array.array("h", (int(max(-1.0, min(1.0, value)) * 32767.0) for value in floats))
    if struct.pack("<h", 1) != struct.pack("=h", 1):
        ints.byteswap()
    return ints.tobytes()


def silence32(milliseconds: int, rate: int = SAMPLE_RATE) -> bytes:
    """A pause, as 32-bit float samples."""
    return bytes(4 * int(round(rate * max(int(milliseconds), 0) / 1000.0)))


def float32_of(pcm16: bytes) -> bytes:
    """The same sound as little-endian float32 in -1..1, the form the worker takes."""
    total = len(pcm16) // 2
    try:
        import numpy

        samples = numpy.frombuffer(pcm16[:total * 2], dtype="<i2").astype("<f4") / 32768.0
        return samples.astype("<f4").tobytes()
    except ImportError:
        pass
    ints = array.array("h")
    ints.frombytes(pcm16[:total * 2])
    if struct.pack("<h", 1) != struct.pack("=h", 1):
        ints.byteswap()
    floats = array.array("f", (value / 32768.0 for value in ints))
    if struct.pack("<f", 1.0) != struct.pack("=f", 1.0):
        floats.byteswap()
    return floats.tobytes()


_mp3_refused = False


def _encode_mp3(samples: bytes, rate: int, tags: dict) -> bytes | None:
    """32-bit float ``samples`` as a constant-bitrate MP3 carrying ``tags`` as ID3, or ``None``.

    Encoded in this process by PyAV, which Forge Neo installs for its own video
    work and which carries FFmpeg's LAME, so nothing is added to Forge's
    environment for it -- the voice side never adds anything there. The float
    samples go to LAME as they are, clipped to -1..1, so no 16-bit copy stands
    between the model and the file (see :data:`MP3_QUALITY`). ``None`` (no
    PyAV, a build without the encoder, a failure) keeps the output a WAV, said
    once in the log.
    """
    global _mp3_refused
    try:
        import av
        import numpy
    except ImportError:
        return None
    try:
        buffer = io.BytesIO()
        with av.open(buffer, mode="w", format="mp3") as container:
            for key, value in tags.items():
                container.metadata[key] = str(value)
            stream = container.add_stream("libmp3lame", rate=int(rate),
                                          options={"compression_level": str(MP3_QUALITY)})
            stream.bit_rate = MP3_BITRATE
            stream.layout = "mono"
            size = stream.codec_context.frame_size or 1152
            found = numpy.frombuffer(samples[:len(samples) // 4 * 4], dtype="<f4")
            found = numpy.clip(found, -1.0, 1.0).astype(numpy.float32)
            fifo = av.AudioFifo()
            if found.size:
                frame = av.AudioFrame.from_ndarray(found.reshape(1, -1), format="flt",
                                                   layout="mono")
                frame.sample_rate = int(rate)
                fifo.write(frame)
            while fifo.samples >= size:
                for packet in stream.encode(fifo.read(size)):
                    container.mux(packet)
            tail = fifo.read()
            if tail is not None:
                for packet in stream.encode(tail):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        return buffer.getvalue()
    except Exception:
        if not _mp3_refused:
            _mp3_refused = True
            logger.warning("Model Chain: Voice Box could not encode an MP3 on this Forge, so "
                           "its renders are kept as WAV", exc_info=True)
        return None


def _with_info(wav: bytes, fields: dict) -> bytes:
    """``wav`` with a ``LIST``/``INFO`` chunk after its sound: the tags a WAV can carry.

    After the sound, not before it, so every reader that stops at the ``data``
    chunk -- Python's :mod:`wave` among them -- reads the sound as it was.
    """
    body = b"INFO"
    for key, text in fields.items():
        data = str(text).encode("utf-8") + b"\x00"
        body += key.encode("ascii") + struct.pack("<I", len(data)) + data
        if len(data) % 2:
            body += b"\x00"
    tagged = wav + b"LIST" + struct.pack("<I", len(body)) + body
    return tagged[:4] + struct.pack("<I", len(tagged) - 8) + tagged[8:]


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #


def _samples_root() -> Path:
    return _folder(SAMPLES_DIRNAME)


def _sample_meta(identifier: str) -> Path:
    return _samples_root() / identifier / META_FILENAME


def _sample_audio(identifier: str) -> Path:
    return _samples_root() / identifier / AUDIO_FILENAME


def add_sample(wav: bytes, title: str = "", source: str = "file", transcript: str = "",
               design: dict | None = None) -> dict:
    """Take a recording -- the browser's trimmed selection -- into the library.

    ``source`` says where it came from: ``file``, ``microphone``,
    ``output:<id>`` for a piece of a render trimmed back into a sample, or
    ``design`` for a voice Breeze designed from a description. ``transcript``
    is the sample's exact words, which Breeze clones from and VibeVoice does
    not need; a designed sample is born with the sentence it spoke, and keeps
    ``design`` -- the description, seed, CFG and sentence that made it -- so
    the page can say where the voice came from.
    """
    canonical, _seconds = reference.normalize(wav, _envelope(), error=VoiceBoxError)
    pcm16, rate = pcm_of(canonical)
    identifier = _new_id()
    words = _text(transcript, MAX_TRANSCRIPT_CHARS, "A transcript")
    entry = {
        "id": identifier,
        "title": _title(title, f"Sample {len(samples()) + 1}"),
        "seconds": round(seconds_of(pcm16, rate), 2),
        "rate": int(rate),
        "peaks": peaks_of(pcm16),
        "source": str(source or "file")[:64],
        "created": _now(),
        "transcript": words,
        "transcript_source": ("design" if design else "typed") if words else "",
        "design": _design_record(design) if design else None,
    }
    with _lock:
        _write_bytes(_sample_audio(identifier), wav_bytes(pcm16, rate))
        _write_json(_sample_meta(identifier), entry)
    logger.info("Model Chain: Voice Box kept sample “%s” (%.1f s)", entry["title"],
                entry["seconds"])
    return entry


SAMPLE_FIELDS = {"transcript": "", "transcript_source": "", "design": None}
"""What a sample kept before transcripts existed reads with."""


def _sample_complete(entry: dict) -> dict:
    found = dict(entry)
    for key, value in SAMPLE_FIELDS.items():
        found.setdefault(key, value)
    return found


def _text(value, limit: int, what: str) -> str:
    """Words as one line: whitespace runs collapsed, refused past ``limit`` characters."""
    words = " ".join(str(value or "").split())
    if len(words) > limit:
        raise VoiceBoxError(f"{what} may be at most {limit} characters.")
    return words


def _design_record(design: dict) -> dict:
    found = dict(design or {})
    return {"description": _text(found.get("description"), MAX_DESCRIPTION_CHARS,
                                 "A voice description"),
            "seed": found.get("seed"),
            "cfg_scale": found.get("cfg_scale"),
            "text": _text(found.get("text"), MAX_DESIGN_TEXT_CHARS, "The design sentence")}


def samples() -> list[dict]:
    with _lock:
        found = []
        for child in sorted(_samples_root().iterdir()) if _samples_root().exists() else []:
            entry = _read_json(child / META_FILENAME, {})
            if entry.get("id") and (child / AUDIO_FILENAME).exists():
                found.append(_sample_complete(entry))
    found.sort(key=lambda entry: entry.get("created", 0))
    return found


def sample(identifier: str) -> dict:
    identifier = _identifier(identifier)
    entry = _read_json(_sample_meta(identifier), {})
    if not entry or not _sample_audio(identifier).exists():
        raise NotFound("That sample is no longer in the library.")
    return _sample_complete(entry)


def set_transcript(identifier: str, transcript: str, source: str = "typed") -> dict:
    """Keep a sample's exact words, or clear them with ``""``.

    Written only in the sample's own record, beside its sound: a transcript is
    as private as the recording it describes, and goes where it goes.
    """
    words = _text(transcript, MAX_TRANSCRIPT_CHARS, "A transcript")
    with _lock:
        entry = sample(identifier)
        entry["transcript"] = words
        entry["transcript_source"] = (source if source in TRANSCRIPT_SOURCES else "typed") \
            if words else ""
        _write_json(_sample_meta(entry["id"]), entry)
    return entry


def rename_sample(identifier: str, title: str) -> dict:
    with _lock:
        entry = sample(identifier)
        entry["title"] = _title(title, entry.get("title") or "Sample")
        _write_json(_sample_meta(entry["id"]), entry)
    return entry


def delete_sample(identifier: str) -> dict:
    """Remove a sample. A configuration that used it loses that speaker's voice."""
    with _lock:
        entry = sample(identifier)
        shutil.rmtree(_samples_root() / entry["id"], ignore_errors=True)
        for configuration in configurations():
            speakers = configuration.get("speakers") or {}
            if entry["id"] in speakers.values():
                configuration["speakers"] = {n: s for n, s in speakers.items() if s != entry["id"]}
                _write_json(_configuration_path(configuration["id"]), configuration)
    return {"deleted": entry["id"]}


def sample_wav(identifier: str) -> bytes:
    entry = sample(identifier)
    return _sample_audio(entry["id"]).read_bytes()


def sample_pcm(identifier: str) -> tuple[bytes, int]:
    return pcm_of(sample_wav(identifier))


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def _prompts_path() -> Path:
    return root() / PROMPTS_FILENAME


def _prompt_entries() -> list[dict]:
    found = _read_json(_prompts_path(), {})
    entries = found.get("entries") if isinstance(found, dict) else None
    return [entry for entry in (entries or []) if isinstance(entry, dict) and entry.get("id")]


def _write_prompts(entries: list[dict]) -> None:
    _write_json(_prompts_path(), {"schema": 1, "entries": entries})


def prompts() -> dict:
    """``{"history": [...newest first...], "favourites": [...]}``. A favourite is in both."""
    entries = sorted(_prompt_entries(), key=lambda entry: entry.get("used", 0), reverse=True)
    return {"history": entries,
            "favourites": [entry for entry in entries if entry.get("favourite")]}


def remember_prompt(text: str) -> dict:
    """Put ``text`` at the top of the history; the same words are one entry.

    The history keeps :data:`HISTORY_LIMIT` prompts and lets the oldest go;
    a favourite is never let go, whatever its age.
    """
    words = str(text or "").strip()
    if not words:
        raise VoiceBoxError("There is no prompt to remember.")
    if len(words) > MAX_PROMPT_CHARS:
        raise VoiceBoxError(f"A prompt may be at most {MAX_PROMPT_CHARS} characters.")
    with _lock:
        entries = _prompt_entries()
        found = next((entry for entry in entries if entry.get("text") == words), None)
        if found is None:
            found = {"id": _new_id(), "text": words, "created": _now(), "favourite": False}
            entries.append(found)
        found["used"] = _now()
        ordinary = [entry for entry in entries if not entry.get("favourite")]
        ordinary.sort(key=lambda entry: entry.get("used", 0), reverse=True)
        dropped = {entry["id"] for entry in ordinary[HISTORY_LIMIT:]}
        entries = [entry for entry in entries if entry["id"] not in dropped]
        _write_prompts(entries)
    return dict(found)


def favourite_prompt(identifier: str, on: bool = True) -> dict:
    identifier = _identifier(identifier)
    with _lock:
        entries = _prompt_entries()
        found = next((entry for entry in entries if entry["id"] == identifier), None)
        if found is None:
            raise NotFound("That prompt is no longer in the history.")
        found["favourite"] = bool(on)
        _write_prompts(entries)
    return dict(found)


def delete_prompt(identifier: str) -> dict:
    identifier = _identifier(identifier)
    with _lock:
        entries = _prompt_entries()
        kept = [entry for entry in entries if entry["id"] != identifier]
        if len(kept) == len(entries):
            raise NotFound("That prompt is no longer in the history.")
        _write_prompts(kept)
    return {"deleted": identifier}


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #

SHARED_DEFAULTS = {
    "name": "Default",
    "model_id": "",
    "card_uuid": "",
    "seed": None,
    "batch": 1,
    "speakers": {},
    "slots": {},
}
"""A configuration's own fields, whatever the engine. ``model_id`` and
``card_uuid`` empty mean the engine's default model and the card chosen in
Voice Box's settings; a model id that is not the rendering engine's is read as
its default. ``batch`` is how many takes one render makes, one output each,
the first at the seed and every next one at the seed after. ``speakers`` maps a
slot ("1".."4") to a sample; ``slots`` holds what a slot says beyond its sample
-- ``instruction``, a standing direction, and ``description``, a voice to
design -- for an engine that takes them (Breeze)."""

FIRST = engines.DEFAULT_ENGINE
"""The engine the Voice Box had before it had two (VibeVoice): its options are
what a configuration and an output held *flat*, beside the shared fields."""


def _flat_keys() -> tuple:
    """The first engine's options as a configuration held them before there was a
    second engine: flat, beside the shared fields -- its adapter's own ``FLAT``.
    Still read from a configuration or a page that sends them, and still written
    beside ``options`` so a reader that knows only them reads what it always
    read."""
    try:
        return tuple(getattr(_vocabulary(FIRST), "FLAT", ()) or ())
    except Exception:
        logger.debug("Model Chain: the Voice Box could not read the first engine's fields",
                     exc_info=True)
        return ()


def _flat_places() -> tuple:
    """:func:`_flat_keys` split as the configuration drew them: before the seed and
    the batch, and after (each field's ``place``)."""
    keys = _flat_keys()
    try:
        fields = list(_vocabulary(FIRST).FIELDS)
    except Exception:
        return keys, ()
    head = tuple(field["key"] for field in fields
                 if field["key"] in keys and field.get("place") == "head")
    return head, tuple(key for key in keys if key not in head)


CONFIGURATION_DEFAULTS = dict(SHARED_DEFAULTS, engine=FIRST, options={}, engines_saved=[])
"""The shape :func:`configurations` answers in. ``options`` is every engine's
own fields, by engine id, defaults where an engine was never saved; ``engine``
the engine it was last saved for; ``engines_saved`` the engines whose options
were ever saved in it."""


def _configurations_root() -> Path:
    return _folder(CONFIGURATIONS_DIRNAME)


def _configuration_path(identifier: str) -> Path:
    return _configurations_root() / f"{identifier}.json"


def _bounded_int(value, low: int, high: int, what: str, none_ok: bool = False):
    if value in (None, "") and none_ok:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise VoiceBoxError(f"{what} must be a whole number.") from None
    if not low <= number <= high:
        raise VoiceBoxError(f"{what} must be between {low} and {high}.")
    return number


def _clean_speakers(value) -> dict:
    found = {}
    if not isinstance(value, dict):
        return found
    known = {entry["id"] for entry in samples()}
    for number, sample_id in value.items():
        try:
            n = int(number)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= MAX_SPEAKERS and str(sample_id or "") in known:
            found[str(n)] = str(sample_id)
    return found


def _clean_slots(value) -> dict:
    """``{"1": {"instruction", "description"}}`` for the slots that say something."""
    found = {}
    if not isinstance(value, dict):
        return found
    for number, slot in value.items():
        try:
            n = int(number)
        except (TypeError, ValueError):
            continue
        if not 1 <= n <= MAX_SPEAKERS or not isinstance(slot, dict):
            continue
        instruction = _text(slot.get("instruction"), MAX_DIRECTION_CHARS, "A direction")
        description = _text(slot.get("description"), MAX_DESCRIPTION_CHARS,
                            "A voice description")
        if instruction or description:
            found[str(n)] = {"instruction": instruction, "description": description}
    return found


def _stored_options(entry: dict) -> dict:
    """An entry's options map as stored: ``options``, or the first engine's flat keys."""
    found = {key: dict(value) for key, value in dict(entry.get("options") or {}).items()
             if isinstance(value, dict) and engines.known(key)}
    if FIRST not in found:
        flat = {key: entry[key] for key in _flat_keys() if key in entry}
        if flat:
            found[FIRST] = flat
    return found


def _options_of(engine_id: str, stored: dict | None) -> dict:
    """An engine's options with its defaults where ``stored`` says nothing. Never raises."""
    try:
        found = dict(_vocabulary(engine_id).defaults())
    except Exception:
        logger.debug("Model Chain: the Voice Box could not read %s's fields", engine_id,
                     exc_info=True)
        return dict(stored or {})
    for key, value in dict(stored or {}).items():
        if key in found:
            found[key] = value
    return found


def _clean_options(engine_id: str, values: dict, previous: dict | None) -> dict:
    """One engine's options checked by its own adapter, in the Voice Box's sentences."""
    try:
        return _vocabulary(engine_id).clean(dict(values or {}), previous)
    except VoiceBoxError:
        raise
    except Exception as exc:
        if isinstance(exc, (ValueError, RuntimeError)):
            raise VoiceBoxError(str(exc)) from None
        raise


def _flat(found: dict) -> dict:
    """The configuration with the first engine's options mirrored flat, as before."""
    first = dict((found.get("options") or {}).get(FIRST) or {})
    for key in _flat_keys():
        if key in first:
            found[key] = first[key]
    return found


def _clean_configuration(values: dict, existing: dict | None = None,
                         engine_id: str = "") -> dict:
    """A configuration as it is saved or rendered, validated field by field.

    The shared fields here; each engine's options by its own adapter
    (:func:`_clean_options`). ``values["options"]`` changes only the engines
    it names, and the rest keep what ``existing`` had. The first engine's flat
    keys (VibeVoice's) are its options, as an older page sends them.
    """
    values = dict(values or {})
    engine_id = engine_id if engines.known(engine_id) else engines.active_id()
    base = _complete(dict(existing)) if existing else _complete({})
    merged = dict(base)
    for key in SHARED_DEFAULTS:
        if key in values:
            merged[key] = values[key]
    asked = {key: dict(value) for key, value in dict(values.get("options") or {}).items()
             if isinstance(value, dict) and engines.known(key)}
    flat = {key: values[key] for key in _flat_keys() if key in values}
    if flat:
        # An older page sends the first engine's options flat -- alone, or
        # beside the ``options`` map it was answered with and sends back
        # untouched. A flat value that differs from that map's is an edit only
        # such a page makes (this page never sends flat keys), so it wins; an
        # equal one is the server's own mirror and changes nothing.
        mine = asked.setdefault(FIRST, {})
        for key, value in flat.items():
            if key not in mine or mine[key] != value:
                mine[key] = value
    options = {}
    for known in engines.ENGINES:
        previous = base["options"].get(known)
        if known in asked:
            options[known] = _clean_options(known, asked[known], previous)
        else:
            options[known] = _options_of(known, previous)
    saved = [known for known in engines.ENGINES
             if known in asked or known in (base.get("engines_saved") or [])]
    found = {
        "name": _title(merged.get("name"), base.get("name") or "Configuration"),
        "engine": engine_id,
        "model_id": str(merged.get("model_id") or "")[:64],
        "card_uuid": str(merged.get("card_uuid") or "")[:64],
        "seed": _bounded_int(merged.get("seed"), 0, SEED_MAX, what="The seed", none_ok=True),
        "batch": _bounded_int(merged.get("batch"), 1, _max_takes(engine_id), what="The batch"),
        "speakers": _clean_speakers(merged.get("speakers")),
        "slots": _clean_slots(merged.get("slots")),
        "options": options,
        "engines_saved": saved,
    }
    return _flat(found)


def _complete(entry: dict) -> dict:
    """A stored configuration with every field, those it predates at their defaults.

    A configuration saved before there were two engines holds VibeVoice's
    options flat: they are VibeVoice's, it was saved for VibeVoice, and every
    other engine starts at its defaults in it.
    """
    stored = _stored_options(entry)
    found = dict(SHARED_DEFAULTS)
    found.update(entry)
    found["speakers"] = dict(found.get("speakers") or {})
    found["slots"] = {key: dict(value) for key, value in dict(found.get("slots") or {}).items()
                      if isinstance(value, dict)}
    found["options"] = {known: _options_of(known, stored.get(known))
                        for known in engines.ENGINES}
    if isinstance(entry.get("engines_saved"), list):
        found["engines_saved"] = [known for known in entry["engines_saved"]
                                  if engines.known(known)]
    else:
        found["engines_saved"] = [known for known in engines.ENGINES if known in stored]
    engine = str(entry.get("engine") or "")
    found["engine"] = engine if engines.known(engine) else (
        found["engines_saved"][0] if found["engines_saved"] else engines.DEFAULT_ENGINE)
    return _flat(found)


def configurations() -> list[dict]:
    with _lock:
        found = [_read_json(path, {}) for path in sorted(_configurations_root().glob("*.json"))]
    found = [_complete(entry) for entry in found if entry.get("id")]
    found.sort(key=lambda entry: (entry.get("name", "").lower(), entry.get("created", 0)))
    return found


def configuration(identifier: str) -> dict:
    identifier = _identifier(identifier)
    entry = _read_json(_configuration_path(identifier), {})
    if not entry.get("id"):
        raise NotFound("That configuration no longer exists.")
    return _complete(entry)


def save_configuration(values: dict) -> dict:
    """Create (no ``id``) or update a named configuration, validated field by field."""
    values = dict(values or {})
    with _lock:
        existing = configuration(values["id"]) if values.get("id") else None
        entry = _clean_configuration(values, existing)
        entry["id"] = existing["id"] if existing else _new_id()
        entry["created"] = existing.get("created", _now()) if existing else _now()
        entry["updated"] = _now()
        _write_json(_configuration_path(entry["id"]), entry)
    return entry


def delete_configuration(identifier: str) -> dict:
    with _lock:
        entry = configuration(identifier)
        try:
            _configuration_path(entry["id"]).unlink()
        except FileNotFoundError:
            pass
    return {"deleted": entry["id"]}


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #


def _pipelines_root() -> Path:
    return _folder(PIPELINES_DIRNAME)


def _pipeline_path(identifier: str) -> Path:
    return _pipelines_root() / f"{identifier}.json"


def pipelines() -> list[dict]:
    with _lock:
        found = [_read_json(path, {}) for path in sorted(_pipelines_root().glob("*.json"))]
    found = [entry for entry in found if entry.get("id")]
    found.sort(key=lambda entry: entry.get("updated", 0), reverse=True)
    return found


def pipeline(identifier: str) -> dict:
    identifier = _identifier(identifier)
    entry = _read_json(_pipeline_path(identifier), {})
    if not entry.get("id"):
        raise NotFound("That pipeline no longer exists.")
    return entry


def new_pipeline(name: str = "") -> dict:
    with _lock:
        count = len(pipelines())
        entry = {
            "id": _new_id(),
            "name": _title(name, f"Pipeline {count + 1}"),
            "prompt": "",
            "configuration_id": "",
            "outputs": [],
            "created": _now(),
            "updated": _now(),
        }
        _write_json(_pipeline_path(entry["id"]), entry)
    return entry


def save_pipeline(values: dict) -> dict:
    """Update a pipeline's name, prompt and configuration; its outputs are the service's."""
    values = dict(values or {})
    with _lock:
        entry = pipeline(values.get("id"))
        if "name" in values:
            entry["name"] = _title(values.get("name"), entry.get("name") or "Pipeline")
        if "prompt" in values:
            prompt = str(values.get("prompt") or "")
            if len(prompt) > MAX_PROMPT_CHARS:
                raise VoiceBoxError(f"A prompt may be at most {MAX_PROMPT_CHARS} characters.")
            entry["prompt"] = prompt
        if "configuration_id" in values:
            wanted = str(values.get("configuration_id") or "")
            entry["configuration_id"] = configuration(wanted)["id"] if wanted else ""
        entry["updated"] = _now()
        _write_json(_pipeline_path(entry["id"]), entry)
    return entry


def delete_pipeline(identifier: str) -> dict:
    """Forget a pipeline. Its outputs stay in the library, owned by no pipeline."""
    with _lock:
        entry = pipeline(identifier)
        for output_id in entry.get("outputs") or []:
            try:
                found = output(output_id)
            except NotFound:
                continue
            found["pipeline_id"] = ""
            _write_json(_output_meta(found["id"]), found)
        try:
            _pipeline_path(entry["id"]).unlink()
        except FileNotFoundError:
            pass
    return {"deleted": entry["id"]}


def _attach_output(pipeline_id: str, output_id: str) -> None:
    with _lock:
        try:
            entry = pipeline(pipeline_id)
        except NotFound:
            return
        entry.setdefault("outputs", [])
        if output_id not in entry["outputs"]:
            entry["outputs"].append(output_id)
        entry["updated"] = _now()
        _write_json(_pipeline_path(entry["id"]), entry)


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


def _outputs_root() -> Path:
    return _folder(OUTPUTS_DIRNAME)


def _output_meta(identifier: str) -> Path:
    return _outputs_root() / identifier / META_FILENAME


def _audio_in(folder: Path) -> Path | None:
    """The sound file of the output in ``folder``: its MP3, else its WAV, else none."""
    for name in (MP3_FILENAME, AUDIO_FILENAME):
        if (folder / name).exists():
            return folder / name
    return None


def _format_of(path: Path) -> str:
    return "mp3" if path.name == MP3_FILENAME else "wav"


def _quoted(value) -> str:
    """A parameter's value as a WebUI's infotext writes one: quoted only when it must be."""
    text = str(value)
    if not any(mark in text for mark in ',:"\n'):
        return text
    return json.dumps(text, ensure_ascii=False)


def infotext(entry: dict) -> str:
    """What made an output, the way a WebUI prints what made an image.

    The prompt as it was written, then one line of parameters -- enough to make
    the same render again: the seed is the one the render used, drawn when its
    configuration left the seed blank. Written into the file itself (the MP3's
    comment, the WAV's ``ICMT``) and shown on the page. An output made before
    seeds were recorded has none to give, and its line says nothing of one.

    The parameters are the engine's own (``adapter.infotext``): VibeVoice's
    line is exactly what it was before there was a second engine, and any other
    engine's starts by naming it -- ``Engine: Breeze TTS 2, …``. A render made
    before engines were recorded is VibeVoice's.
    """
    render = entry.get("render") if isinstance(entry.get("render"), dict) else {}
    engine_id = str(render.get("engine") or engines.DEFAULT_ENGINE)
    parts: list[str] = []

    def add(label: str, value) -> None:
        if value not in (None, ""):
            parts.append(f"{label}: {_quoted(value)}")

    speakers = []
    for speaker in render.get("speakers") or ():
        if isinstance(speaker, dict):
            speakers.append((f"Speaker {speaker.get('n')}",
                             speaker.get("title") or speaker.get("sample_id")))
            if speaker.get("instruction"):
                speakers.append((f"Speaker {speaker.get('n')} direction",
                                 speaker.get("instruction")))
    shared = {"seed": render.get("seed"), "model": render.get("model_id"), "speakers": speakers}
    try:
        lines = list(_vocabulary(engine_id).infotext(render, shared))
    except Exception:
        logger.debug("Model Chain: the Voice Box could not word an output's parameters",
                     exc_info=True)
        lines = [("Seed", shared["seed"]), ("Model", shared["model"])] + speakers
    if engine_id != engines.DEFAULT_ENGINE:
        add("Engine", engines.label(engine_id))
    for label, value in lines:
        add(label, value)
    sections = render.get("sections")
    add("Sections", sections if isinstance(sections, int) and sections > 1 else None)
    if entry.get("seconds") is not None:
        add("Length", f"{float(entry['seconds']):.1f} s")
    if render.get("render_seconds"):
        add("Render time", f"{float(render['render_seconds']):.1f} s")
    line = ", ".join(parts)
    prompt = str(render.get("prompt") or "").strip()
    return f"{prompt}\n{line}".strip() if prompt else line


def _shown(entry: dict, audio: Path) -> dict:
    """An output as it is handed out: the stored record, its sound's format and its infotext."""
    found = dict(entry)
    found["format"] = _format_of(audio)
    found["infotext"] = infotext(found)
    return found


def _stored_output(identifier: str) -> tuple[dict, Path]:
    identifier = _identifier(identifier)
    entry = _read_json(_output_meta(identifier), {})
    audio = _audio_in(_outputs_root() / identifier)
    if not entry.get("id") or audio is None:
        raise NotFound("That render is no longer in the Voice Box.")
    return entry, audio


def add_output(samples: bytes, rate: int, name: str, pipeline_id: str, render: dict) -> dict:
    """Keep a render: its sound as an MP3 (a WAV where MP3 cannot be made), its record beside it.

    ``samples`` is the model's sound as little-endian 32-bit float, which the
    MP3 is made from directly (:data:`MP3_QUALITY`); the peaks and a WAV, where
    one has to be made, are its 16-bit copy.

    The file carries its own infotext -- the MP3 in ID3 (the comment, and the
    whole record as JSON under ``voicebox``), the WAV in its ``INFO`` chunk --
    so a render downloaded or saved says what made it wherever it goes. The
    name is not written into the file: a rename would leave it saying the old
    one, and a player shows the file's name when there is no title.
    """
    identifier = _new_id()
    pcm16 = pcm16_of(samples)
    entry = {
        "id": identifier,
        "name": _title(name, "Render"),
        "pipeline_id": str(pipeline_id or ""),
        "seconds": round(seconds_of(pcm16, rate), 2),
        "rate": int(rate),
        "peaks": peaks_of(pcm16),
        "loop": False,
        "created": _now(),
        "render": dict(render or {}),
    }
    text = infotext(entry)
    maker = software(engines.label(str(entry["render"].get("engine") or
                                       engines.DEFAULT_ENGINE)))
    encoded = _encode_mp3(samples, rate, {
        "comment": text,
        "encoded_by": maker,
        "voicebox": json.dumps({"id": identifier, "render": entry["render"]},
                               ensure_ascii=False),
    })
    folder = _outputs_root() / identifier
    with _lock:
        if encoded is not None:
            audio = folder / MP3_FILENAME
            _write_bytes(audio, encoded)
        else:
            audio = folder / AUDIO_FILENAME
            _write_bytes(audio, _with_info(wav_bytes(pcm16, rate),
                                           {"ICMT": text, "ISFT": maker}))
        _write_json(_output_meta(identifier), entry)
    if pipeline_id:
        _attach_output(pipeline_id, identifier)
    return _shown(entry, audio)


def outputs(pipeline_id: str = "") -> list[dict]:
    with _lock:
        found = []
        for child in sorted(_outputs_root().iterdir()) if _outputs_root().exists() else []:
            entry = _read_json(child / META_FILENAME, {})
            audio = _audio_in(child)
            if entry.get("id") and audio is not None:
                found.append(_shown(entry, audio))
    if pipeline_id:
        found = [entry for entry in found if entry.get("pipeline_id") == pipeline_id]
    found.sort(key=lambda entry: entry.get("created", 0), reverse=True)
    return found


def output(identifier: str) -> dict:
    return _shown(*_stored_output(identifier))


def rename_output(identifier: str, name: str) -> dict:
    with _lock:
        entry, audio = _stored_output(identifier)
        entry["name"] = _title(name, entry.get("name") or "Render")
        _write_json(_output_meta(entry["id"]), entry)
    return _shown(entry, audio)


def set_loop(identifier: str, loop: bool) -> dict:
    with _lock:
        entry, audio = _stored_output(identifier)
        entry["loop"] = bool(loop)
        _write_json(_output_meta(entry["id"]), entry)
    return _shown(entry, audio)


def delete_output(identifier: str) -> dict:
    with _lock:
        entry, _audio = _stored_output(identifier)
        shutil.rmtree(_outputs_root() / entry["id"], ignore_errors=True)
        if entry.get("pipeline_id"):
            try:
                owner = pipeline(entry["pipeline_id"])
                owner["outputs"] = [o for o in owner.get("outputs") or [] if o != entry["id"]]
                _write_json(_pipeline_path(owner["id"]), owner)
            except NotFound:
                pass
    return {"deleted": entry["id"]}


def output_audio(identifier: str) -> tuple[bytes, str]:
    """An output's sound file as stored, and its format (``"mp3"`` or ``"wav"``)."""
    _entry, audio = _stored_output(identifier)
    return audio.read_bytes(), _format_of(audio)


def save_output(identifier: str) -> str:
    """Copy a render, with its metadata beside it, into the remembered folder.

    The folder is chosen once and kept (:func:`choose_save_folder`); a save
    with none chosen is refused with the sentence that tells the page to ask.
    A name already taken gets a number, never an overwrite.
    """
    stored, audio = _stored_output(identifier)
    entry = _shown(stored, audio)
    folder = str(settings().get("save_folder") or "")
    if not folder:
        raise VoiceBoxError("Choose a folder to save renders into first.")
    target = Path(folder)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VoiceBoxError(f"The save folder cannot be used: {exc.strerror or exc}") from None
    stem = _safe_filename(entry.get("name"))
    suffix = f".{entry['format']}"
    candidate, number = target / f"{stem}{suffix}", 2
    while candidate.exists() or candidate.with_suffix(".json").exists():
        candidate = target / f"{stem} ({number}){suffix}"
        number += 1
    shutil.copyfile(audio, candidate)
    sidecar = {key: value for key, value in entry.items() if key != "peaks"}
    _write_json(candidate.with_suffix(".json"), sidecar)
    logger.info("Model Chain: Voice Box saved “%s” to %s", entry["name"], candidate)
    return str(candidate)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

SETTINGS_DEFAULTS = {"save_folder": "", "card_uuid": "", "model_id": "", "keep_warm": True,
                     "engine": FIRST}
"""``engine`` is the engine every render uses, one at a time: a Voice Box
setting beside the card and the warm stay rather than a configuration's field.
It changes only through :func:`mc_voice_box_engines.select`, which makes sure
the other engine has let go of every card first."""


def _settings_path() -> Path:
    return root() / SETTINGS_FILENAME


def settings() -> dict:
    found = dict(SETTINGS_DEFAULTS)
    stored = _read_json(_settings_path(), {})
    for key in SETTINGS_DEFAULTS:
        if key in stored:
            found[key] = stored[key]
    found["keep_warm"] = bool(found.get("keep_warm", True))
    engine = str(found.get("engine") or "").strip().lower()
    found["engine"] = engine if engine in engines.ENGINES else engines.DEFAULT_ENGINE
    return found


def set_settings(values: dict) -> dict:
    values = dict(values or {})
    with _lock:
        found = settings()
        if "save_folder" in values:
            text = str(values.get("save_folder") or "").strip()
            if text:
                target = Path(text).expanduser()
                try:
                    target.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise VoiceBoxError(
                        f"That folder cannot be used: {exc.strerror or exc}") from None
                text = str(target)
            found["save_folder"] = text
        if "card_uuid" in values:
            found["card_uuid"] = str(values.get("card_uuid") or "")[:64]
        if "model_id" in values:
            found["model_id"] = str(values.get("model_id") or "")[:64]
        if "keep_warm" in values:
            found["keep_warm"] = bool(values.get("keep_warm"))
        if "engine" in values:
            wanted = str(values.get("engine") or "").strip().lower()
            if wanted not in engines.ENGINES:
                raise VoiceBoxError(f"{wanted or 'Nothing'} is not an engine the Voice Box has.")
            found["engine"] = wanted
        _write_json(_settings_path(), found)
    return found


def choose_save_folder() -> str:
    """Open the Forge PC's own folder dialog; remember what it says. ``""`` when cancelled."""
    import mc_llm_native

    current = settings().get("save_folder") or None
    chosen = mc_llm_native.choose_folder("Choose where Voice Box saves renders", current)
    if not chosen:
        return ""
    return set_settings({"save_folder": chosen})["save_folder"]


# --------------------------------------------------------------------------- #
# The script
# --------------------------------------------------------------------------- #


@dataclass
class Section:
    """One separately generated stretch of a script.

    ``pause_ms`` is the silence that goes *before* it -- what the ``[pause]``
    tag that ended the previous section asked for -- and is None for the first.
    """

    lines: list = field(default_factory=list)
    pause_ms: int | None = None
    directions: list = field(default_factory=list)
    """Each line's direction (``Speaker 2 (whispering): …``), ``""`` for none,
    in step with ``lines``. An engine that cannot be directed never sees them."""

    @property
    def speakers(self) -> list[int]:
        return sorted({number for number, _ in self.lines})

    def direction(self, index: int) -> str:
        return str(self.directions[index]) if index < len(self.directions) else ""


_PAUSE = re.compile(r"\[\s*pause(?:\s*[:=]\s*(\d{1,6})\s*(?:ms)?)?\s*\]", re.IGNORECASE)
_SPEAKER = re.compile(r"^\s*(?:speaker\s*(\d+)|\[\s*(\d+)\s*\])\s*(?:\(([^()]*)\))?\s*[:：]\s*(.*)$",
                      re.IGNORECASE)


def parse_script(text: str) -> list[Section]:
    """Sections of ``text``, each a list of ``(speaker, words)`` lines.

    ``Speaker 2:`` (or ``[2]:``) at the start of a line names its speaker; a
    line with no name continues the one before it, or is Speaker 1's when none
    has spoken. ``Speaker 2 (whispering, afraid):`` names its speaker and how
    the line is said: an engine that takes directions (Breeze) says it that
    way, and one that does not (VibeVoice) accepts the syntax and drops the
    direction, so a script moves between engines without edits. ``[pause]``
    and ``[pause:1500]`` end a section and put that much silence before the
    next -- the default :data:`PAUSE_DEFAULT_MS`, clamped to
    :data:`PAUSE_MIN_MS`..:data:`PAUSE_MAX_MS`. Breeze's vocal events --
    ``(laugh)`` in a line's words, ``[笑]`` in Chinese -- are words here, and
    pass through untouched: the tags this parser reads are matched exactly.
    """
    words = str(text or "")
    if len(words) > MAX_PROMPT_CHARS:
        raise VoiceBoxError(f"A prompt may be at most {MAX_PROMPT_CHARS} characters.")
    sections: list[Section] = []
    pending_pause: int | None = None
    cursor = 0
    # (chunk, the pause that ends it): "" for a bare [pause], digits for a
    # timed one, None after the last chunk, which nothing ends.
    pieces: list[tuple[str, str | None]] = []
    for match in _PAUSE.finditer(words):
        pieces.append((words[cursor:match.start()], match.group(1) or ""))
        cursor = match.end()
    pieces.append((words[cursor:], None))
    for chunk, pause in pieces:
        found = _lines_of(chunk)
        if found:
            sections.append(Section(lines=[(number, body) for number, body, _ in found],
                                    pause_ms=pending_pause if sections else None,
                                    directions=[direction for _, _, direction in found]))
            pending_pause = None
        if pause is None:
            continue
        asked = int(pause) if pause else PAUSE_DEFAULT_MS
        asked = min(max(asked, PAUSE_MIN_MS), PAUSE_MAX_MS)
        # Two pauses with nothing between them add up, to the cap.
        pending_pause = asked if pending_pause is None else min(pending_pause + asked,
                                                                PAUSE_MAX_MS)
    if not sections:
        raise VoiceBoxError("The prompt has no words to speak.")
    return sections


def _lines_of(chunk: str) -> list[tuple[int, str, str]]:
    """``(speaker, words, direction)`` per line of a section's text."""
    lines: list[tuple[int, str, str]] = []
    current: int | None = None
    for raw in chunk.splitlines():
        match = _SPEAKER.match(raw)
        if match:
            number = int(match.group(1) or match.group(2))
            if not 1 <= number <= MAX_SPEAKERS:
                raise VoiceBoxError(f"Speakers are numbered 1 to {MAX_SPEAKERS}; "
                                    f"the prompt names Speaker {number}.")
            current = number
            direction = " ".join(str(match.group(3) or "").split())[:MAX_DIRECTION_CHARS]
            body = " ".join(match.group(4).split())
            lines.append((number, body, direction))
            continue
        body = " ".join(raw.split())
        if not body:
            continue
        if current is None:
            current = 1
            lines.append((1, body, ""))
        else:
            number, previous, direction = lines[-1]
            lines[-1] = (number, f"{previous} {body}".strip(), direction)
    return [(number, body, direction) for number, body, direction in lines if body]


def script_summary(text: str) -> dict:
    """What the page says under the prompt: speakers, pauses, words. Never raises."""
    try:
        sections = parse_script(text)
    except VoiceBoxError as exc:
        return {"ok": False, "speakers": [], "pauses": 0, "words": 0, "sections": 0,
                "error": str(exc)}
    speakers = sorted({number for section in sections for number in section.speakers})
    words = sum(len(body.split()) for section in sections for _, body in section.lines)
    return {"ok": True, "speakers": speakers, "pauses": len(sections) - 1, "words": words,
            "sections": len(sections)}


# --------------------------------------------------------------------------- #
# The render service
# --------------------------------------------------------------------------- #


@dataclass
class Job:
    id: str
    name: str
    pipeline_id: str
    prompt: str
    configuration: dict
    card: str
    configuration_id: str = ""
    """The saved configuration the page was showing, kept for the output's record
    even when the render ran on the page's unsaved changes to it."""
    seed: int | None = None
    seed_drawn: bool = False
    """The seed every section renders with: the configuration's, or one drawn for
    this job when the configuration left it blank -- so every render has a seed
    that makes it again. A batch's first take; take k renders at ``seed + k``."""
    batch: int = 1
    names: list = field(default_factory=list)
    """Each take's output name, in take order."""
    origin: dict = field(default_factory=dict)
    """Who asked, when it was not the Voice Box's own Render: :data:`ORIGIN_KEYS`.

    LLM Studio's Send to Voice Box sends a message's text with the thread, the
    message and a digest of its words as one ``key``, and finds the render
    again after a reload by asking :func:`outputs` for that key. Kept on the
    output's record for that reason, with the engine that rendered it, and
    shown nowhere else.
    """
    kind: str = KIND_RENDER
    """``render``, which makes outputs, or ``design``, which makes one take a
    slot can keep as a sample (:func:`design`)."""
    engine: str = ""
    """The registry's id for the engine this job renders with."""
    options: dict = field(default_factory=dict)
    """The engine's options as the render uses them (:func:`mc_voice_box_fields.effective`)."""
    design: dict = field(default_factory=dict)
    """A design's description, sentence and CFG; once done, its take's length and seed."""
    design_pcm: bytes = field(default=b"", repr=False)
    """A design's take, 16-bit, until it is kept or the job is forgotten."""
    output_ids: list = field(default_factory=list)
    phase: str = QUEUED
    reason: str = ""
    warning: str = ""
    progress: dict = field(default_factory=dict)
    output_id: str = ""
    render_id: str = ""
    """The worker's id for the section being rendered, for a cancel to name."""
    created: float = field(default_factory=_now)
    started: float | None = None
    ended: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def elapsed(self) -> float | None:
        """Seconds since the job left the queue, to its end; ``None`` while it is queued.

        Worked out here, when the server answers, so a page counts on from a
        figure of the server's and never sets its own clock against the
        server's timestamps.
        """
        if self.started is None:
            return None
        return round(max((self.ended or _now()) - self.started, 0.0), 1)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "pipeline_id": self.pipeline_id,
                "phase": self.phase, "reason": self.reason, "warning": self.warning,
                "progress": dict(self.progress), "output_id": self.output_id,
                "origin": dict(self.origin),
                "output_ids": list(self.output_ids), "batch": self.batch,
                "created": self.created, "started": self.started, "ended": self.ended,
                "elapsed": self.elapsed(), "seed": self.seed, "seed_drawn": self.seed_drawn,
                "card": self.card, "live": self.phase in LIVE, "kind": self.kind,
                "engine": self.engine,
                "design": (dict(self.design, ready=bool(self.design_pcm))
                           if self.kind == KIND_DESIGN else None)}


class _Service:
    """One queue per card, one thread per card with work on it."""

    def __init__(self):
        self.lock = threading.RLock()
        self.queues: dict[str, deque] = {}
        self.threads: dict[str, threading.Thread] = {}
        self.jobs: deque = deque(maxlen=JOBS_KEPT)
        self.current: dict[str, Job] = {}

    def submit(self, job: Job) -> None:
        key = _card_key(job.card)
        with self.lock:
            self.jobs.append(job)
            self.queues.setdefault(key, deque()).append(job)
            thread = self.threads.get(key)
            if thread is None or not thread.is_alive():
                thread = threading.Thread(target=self._run_card, args=(key,),
                                          name=f"mc-voice-box-{key[-6:]}", daemon=True)
                self.threads[key] = thread
                thread.start()

    def _run_card(self, key: str) -> None:
        while True:
            with self.lock:
                queue = self.queues.get(key)
                if not queue:
                    self.threads.pop(key, None)
                    return
                job = queue.popleft()
                self.current[key] = job
            try:
                if job.cancelled:
                    _end(job, CANCELLED)
                else:
                    _perform(job)
            except Exception as exc:
                logger.warning("Model Chain: a Voice Box render failed", exc_info=True)
                _end(job, FAILED, warning=_sentence(exc))
            finally:
                with self.lock:
                    self.current.pop(key, None)

    def find(self, identifier: str) -> Job | None:
        with self.lock:
            return next((job for job in self.jobs if job.id == identifier), None)

    def listed(self) -> list[dict]:
        with self.lock:
            return [job.to_dict() for job in reversed(self.jobs)]

    def pending(self, pipeline_id: str) -> int:
        """Outputs a pipeline's live jobs will make: a take each."""
        with self.lock:
            return sum(max(1, int(job.batch or 1)) for job in self.jobs
                       if job.pipeline_id == pipeline_id and job.phase in LIVE
                       and job.kind == KIND_RENDER)

    def live(self) -> bool:
        """Whether anything -- a render or a design, on any card -- is queued or running."""
        with self.lock:
            return any(job.phase in LIVE for job in self.jobs) \
                or any(queue for queue in self.queues.values()) or bool(self.current)

    def clear(self) -> int:
        """Withdraw every job still queued, on every card; how many were. Running ones stay."""
        with self.lock:
            withdrawn = []
            for queue in self.queues.values():
                while queue:
                    withdrawn.append(queue.popleft())
        cleared = 0
        for job in withdrawn:
            if not job.cancelled and job.phase == QUEUED:
                cleared += 1
            job.cancel_event.set()
            if job.phase in LIVE:
                _end(job, CANCELLED)
        return cleared

    def stop(self) -> None:
        with self.lock:
            for queue in self.queues.values():
                for job in queue:
                    job.cancel_event.set()
            live = list(self.current.items())
        for key, job in live:
            _interrupt(job)

    def join(self, timeout: float) -> None:
        """Wait, up to ``timeout`` seconds in all, for every card's thread to end."""
        deadline = time.monotonic() + timeout
        with self.lock:
            threads = list(self.threads.values())
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))


def _interrupt(job: Job) -> None:
    """Withdraw or cut short ``job``, whatever it is doing right now."""
    job.cancel_event.set()
    if job.phase in (LOADING, RENDERING):
        try:
            _runtime().cancel(job.card, job.render_id or f"{job.id}:1")
        except Exception:
            logger.debug("Model Chain: the render could not be told to stop", exc_info=True)


_service = _Service()


def _card_key(uuid: str) -> str:
    text = str(uuid or "").strip().casefold()
    return "".join(character for character in text if character in "0123456789abcdef")


def _sentence(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text[:400] or exc.__class__.__name__


def _end(job: Job, phase: str, warning: str = "") -> None:
    job.phase = phase
    job.reason = ""
    if warning:
        job.warning = warning
    job.ended = _now()


ORIGIN_KEYS = ("kind", "key", "label")
ORIGIN_CHARS = 200
"""What a render's ``origin`` may carry: three short strings and nothing else.

Another tab on the page names itself (``kind``), the thing it rendered
(``key``, which it chooses and reads back) and a label for a person. A record
is kept on the job and the output, so it is held to a shape the file format
and the page can take whatever a caller sends: a key is cut to
:data:`ORIGIN_CHARS`, a value that is not a string is dropped, and anything
that is not a mapping is no origin at all. The Voice Box adds one word of its
own, ``engine`` -- which engine rendered it -- and only to an origin there is.
"""


def _clean_origin(value) -> dict:
    """The origin a caller sent, held to :data:`ORIGIN_KEYS`. Never raises."""
    if not isinstance(value, dict):
        return {}
    found = {}
    for key in ORIGIN_KEYS:
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            found[key] = text.strip()[:ORIGIN_CHARS]
    return found


def _chosen_configuration(configuration_id: str, inline: dict | None) -> tuple[dict, str]:
    """The configuration a press renders with, and the saved one it came from.

    ``inline`` -- the page's unsaved values -- validated the way a save is, and
    recorded against the saved configuration it changed while that still
    exists; otherwise the saved one; otherwise the defaults.
    """
    if inline is not None:
        chosen = _clean_configuration(dict(inline))
        source = ""
        if configuration_id:
            # Unsaved changes to a saved configuration: rendered as shown, and
            # the output still says which configuration they were made to.
            try:
                source = configuration(configuration_id)["id"]
            except NotFound:
                source = ""
        return chosen, source
    if configuration_id:
        stored = configuration(configuration_id)
        return dict(stored, **_clean_configuration(stored, stored)), stored["id"]
    return _clean_configuration({}), ""


def _engine_options(engine, chosen: dict) -> tuple[str, dict]:
    """``(engine id, the options a render sends)`` for the configuration chosen."""
    engine_id = _engine_id(engine)
    options = dict((chosen.get("options") or {}).get(engine_id) or {})
    try:
        fields = _vocabulary(engine_id).FIELDS
    except Exception:
        fields = ()
    return engine_id, box_fields.effective(fields, options)


def _refuse_options(engine, engine_id: str, options: dict) -> None:
    """Refuse at the press what the engine's runtime could not render (Flash attention 2)."""
    ask = getattr(engine, "refuse", None)
    if ask is None:
        ask = _vocabulary(engine_id).refuse
    refused = ask(options)
    if refused:
        raise VoiceBoxError(refused)


def _card_of(chosen: dict, engine) -> str:
    card = chosen.get("card_uuid") or settings().get("card_uuid") or ""
    if not card:
        raise VoiceBoxError(f"Choose the card {_label(engine)} renders on, in the configuration "
                            f"or in Voice Box's settings.")
    return card


def _seed_for(chosen: dict, batch: int, asked=None) -> tuple[int, bool]:
    """The seed a job renders with, and whether it was drawn for it."""
    seed = asked if asked is not None else chosen.get("seed")
    if seed is None:
        # Drawn so the whole batch fits: take k renders at seed + k.
        return secrets.randbelow(SEED_MAX - batch + 2), True
    seed = int(seed)
    if not 0 <= seed <= SEED_MAX:
        raise VoiceBoxError(f"The seed must be between 0 and {SEED_MAX}.")
    if seed + batch - 1 > SEED_MAX:
        raise VoiceBoxError(f"A batch of {batch} from seed {seed} would pass {SEED_MAX}, the "
                            f"largest seed: choose a smaller seed or a smaller batch.")
    return seed, False


def render(pipeline_id: str, prompt: str, configuration_id: str = "", name: str = "",
           inline: dict | None = None, origin: dict | None = None) -> dict:
    """Queue a render of ``prompt`` with a configuration, for a pipeline. Returns the job.

    The configuration is the saved one ``configuration_id`` names, or ``inline``
    -- the page's unsaved values -- validated the same way. Everything that can
    be refused before a card is asked for is refused here, to the caller: the
    prompt is parsed, the configuration read, the speakers checked against the
    samples, the card decided. What follows -- the turn, the load, the render --
    happens on the card's own thread and is read back from the job.

    The render uses the active engine and that engine's options in the
    configuration. An engine that clones from a transcript (Breeze) refuses a
    speaker whose sample has none, and a slot that only describes a voice it
    has not designed yet.

    A configuration with no seed gets one drawn here, so the job says from the
    start which seed it renders with and its output records it.

    ``origin`` is for a render asked for from another tab (:data:`ORIGIN_KEYS`):
    kept on the job and written into the output's record, so the tab that asked
    can find its render again, and otherwise changing nothing about the render.
    """
    if _turns is None:
        raise VoiceBoxError("Voice Box is not connected to the cards on this WebUI.")
    owner = pipeline(pipeline_id)
    sections = parse_script(prompt)
    chosen, source = _chosen_configuration(configuration_id, inline)
    engine = _engine()
    label = _label(engine)
    capabilities = _capabilities(engine)
    voice = capabilities.get("voice") or {}
    wanted = sorted({number for section in sections for number in section.speakers})
    for number in wanted:
        sample_id = chosen["speakers"].get(str(number))
        if not sample_id:
            slot = (chosen.get("slots") or {}).get(str(number)) or {}
            if voice.get("description") and slot.get("description"):
                raise VoiceBoxError(f"Speaker {number} has a description but no voice yet — "
                                    f"Design it and Keep it first.")
            raise VoiceBoxError(f"Speaker {number} has no sample — choose one in the "
                                f"configuration's speaker {number} slot.")
        if voice.get("transcript"):
            found = sample(sample_id)
            if not str(found.get("transcript") or "").strip():
                raise VoiceBoxError(f"Speaker {number}'s sample “{found.get('title') or 'Untitled'}” "
                                    f"has no transcript — {label} clones a voice only from its "
                                    f"exact words. Add one in the sample's row.")
    card = _card_of(chosen, engine)
    refused = engine.refusal()
    if refused:
        raise VoiceBoxError(refused)
    engine_id, options = _engine_options(engine, chosen)
    _refuse_options(engine, engine_id, options)
    batch = int(chosen["batch"])
    seed, drawn = _seed_for(chosen, batch)
    remember_prompt(prompt)
    if name:
        names = [_title(name, "Render")] if batch == 1 else \
            [_title(f"{name} · {take + 1}", "Render") for take in range(batch)]
    else:
        # Counted in this order, the pipeline read after the jobs: a job keeps
        # its outputs before it ends, so one that ends in between is found by
        # the second read. The other order let it slip between the two, and
        # two renders took one name. Every take is an output and takes its own
        # number, so a batch of four after the third output is the 4th to 7th.
        waiting = _service.pending(owner["id"])
        made = len(pipeline(owner["id"]).get("outputs") or [])
        names = [f"{owner.get('name') or 'Render'} {made + waiting + 1 + take}"
                 for take in range(batch)]
    origin = _clean_origin(origin)
    if origin:
        # Which engine answered the tab that asked: its record says so beside
        # the three strings it sent.
        origin["engine"] = engine_id
    job = Job(id=_new_id(), name=names[0], pipeline_id=owner["id"],
              prompt=str(prompt), configuration=chosen, card=card, configuration_id=source,
              seed=int(seed), seed_drawn=drawn, batch=batch, names=names,
              origin=origin, kind=KIND_RENDER, engine=engine_id, options=options)
    # The answer is the job as it was queued. Read after the card's thread has
    # it, it could already say the job had started, depending on who ran first.
    queued = job.to_dict()
    _service.submit(job)
    logger.info("Model Chain: Voice Box queued “%s” with %s (%d section%s, %d speaker%s)",
                job.name, label, len(sections), "" if len(sections) == 1 else "s", len(wanted),
                "" if len(wanted) == 1 else "s")
    return queued


def busy() -> bool:
    """Whether a render or a design is queued or running, on any card."""
    return _service.live()


def _voices_for(job: Job, sections) -> dict | None:
    """Each speaker's :class:`mc_voice_box_render.Voice`, or ``None`` after failing the job."""
    if job.kind == KIND_DESIGN:
        return {1: render_types.Voice(pcm=b"", rate=SAMPLE_RATE,
                                      description=str(job.design.get("description") or ""))}
    voices = {}
    slots = job.configuration.get("slots") or {}
    for number in sorted({n for section in sections for n in section.speakers}):
        sample_id = job.configuration["speakers"].get(str(number))
        try:
            pcm16, rate = sample_pcm(sample_id)
            entry = sample(sample_id)
        except VoiceBoxError:
            _end(job, FAILED, warning=f"Speaker {number}'s sample is no longer in the library.")
            return None
        slot = slots.get(str(number)) or {}
        voices[number] = render_types.Voice(
            pcm=float32_of(pcm16), rate=rate, sample_id=sample_id,
            title=str(entry.get("title") or ""), transcript=str(entry.get("transcript") or ""),
            instruction=str(slot.get("instruction") or ""))
    return voices


def _sections_of(job: Job) -> list:
    if job.kind == KIND_DESIGN:
        return [Section(lines=[(1, str(job.design.get("text") or ""))], directions=[""])]
    return parse_script(job.prompt)


def _model_for(job: Job, engine) -> str:
    """The configuration's model when it is one of the engine's, else the settings', else ``""``."""
    known = getattr(engine, "model_ids", None)
    try:
        listed = tuple(known()) if callable(known) else ()
    except Exception:
        listed = ()
    for candidate in (job.configuration.get("model_id"), settings().get("model_id")):
        candidate = str(candidate or "")
        if candidate and (not listed or candidate in listed):
            return candidate
    return ""


def _perform(job: Job) -> None:
    engine = _engine()
    runtime = _runtime()
    sections = _sections_of(job)
    voices = _voices_for(job, sections)
    if voices is None:
        return
    model_id = _model_for(job, engine)
    job.started = _now()
    job.phase = WAITING
    job.reason = "asking for the card"
    turn = _turns.request(job.card,
                          need_vram=int(engine.need_vram_bytes(model_id, takes=job.batch)),
                          need_ram=int(engine.need_ram_bytes(model_id)),
                          label=f"{_label(engine)} — {job.name}",
                          guest=str(getattr(engine, "GUEST", "") or ""))
    made = None
    try:
        while True:
            phase = turn.wait(timeout=_wait, cancelled=job.cancel_event)
            job.reason = str(getattr(turn, "reason", "") or "")
            if phase in (TURN_GRANTED, TURN_BLOCKED, TURN_CANCELLED) or job.cancelled:
                break
        if phase == TURN_BLOCKED:
            _end(job, FAILED, warning=str(getattr(turn, "warning", "") or
                                           "The card could not be made ready."))
            return
        if phase != TURN_GRANTED or job.cancelled:
            # Withdrawn by the user while waiting. Said to the turn system too:
            # a queued turn holds every image job on its card, and one this
            # loop stopped waiting on would otherwise stay queued.
            turn.cancel()
            _end(job, CANCELLED)
            return
        made = _render_granted(job, sections, voices, model_id, engine, runtime)
    finally:
        try:
            turn.finish(keep_warm=None if settings().get("keep_warm", True) else False)
        except Exception:
            logger.warning("Model Chain: could not hand the card back after a render",
                           exc_info=True)
    if made is not None:
        # The card is handed back first: making the files is work for the
        # processor, and an image job has no reason to wait for it.
        if job.kind == KIND_DESIGN:
            _keep_design_take(job, made)
        else:
            _keep(job, made)


def _render_granted(job: Job, sections, voices, model_id: str, engine, runtime):
    """Load, render every section, and return each take's ``(samples, rate, record)``.

    ``samples`` is the take's 32-bit float sound, its sections joined by their
    pauses. ``None`` when cancelled. Every section is one request to the
    engine's runtime for the whole batch, and take k is the same take in every
    section: VibeVoice makes a section's takes in one ``generate``, Breeze one
    after another, line by line, and the Voice Box never knows which.
    """
    job.phase = LOADING
    job.reason = "loading the model"
    if model_id:
        # The runtime loads the model the engine's settings name; the
        # configuration's choice is made that name before the load.
        engine.set_settings({"model_id": model_id})
    runtime.load(job.card, options=dict(job.options))
    takes = max(1, int(job.batch or 1))
    pieces: list[list[bytes]] = [[] for _take in range(takes)]
    total_seconds = 0.0
    peak = 0
    rate = SAMPLE_RATE
    render_seconds = 0.0
    tokens = [0] * takes
    capped = [False] * takes
    for index, section in enumerate(sections):
        if job.cancelled:
            break
        job.phase = RENDERING
        job.reason = ""
        job.progress = {"section": index + 1, "sections": len(sections),
                        "seconds": round(total_seconds, 1)}
        if section.pause_ms and pieces[0]:
            for kept in pieces:
                kept.append(silence32(section.pause_ms, rate))
            total_seconds += section.pause_ms / 1000.0

        def progressed(found: dict, _base=total_seconds, _index=index):
            progress = {"section": _index + 1, "sections": len(sections),
                        "seconds": round(_base + float(found.get("seconds") or 0.0), 1)}
            for key in ("line", "lines", "take", "takes"):
                if isinstance(found.get(key), int):
                    progress[key] = found[key]
            job.progress = progress

        job.render_id = f"{job.id}:{index + 1}"
        request = render_types.RenderJob(
            id=job.render_id,
            lines=[render_types.Line(number, body, section.direction(line))
                   for line, (number, body) in enumerate(section.lines)],
            voices={number: voice for number, voice in voices.items()
                    if number in section.speakers},
            seed=job.seed,
            takes=takes,
            options=dict(job.options))
        result = runtime.render(job.card, request, on_progress=progressed)
        peak = max(peak, int(getattr(result, "peak_bytes", 0) or 0))
        render_seconds += float(getattr(result, "render_seconds", 0.0) or 0.0)
        if getattr(result, "cancelled", False):
            job.cancel_event.set()
            break
        rate = int(getattr(result, "sample_rate", 0) or SAMPLE_RATE)
        made = list(result.takes)
        if len(made) != takes:
            raise VoiceBoxError(f"{_label(engine)} answered with a different number of takes "
                                f"than it was asked for.")
        for take, found in enumerate(made):
            pieces[take].append(bytes(found.pcm))
            tokens[take] += int(found.tokens or 0)
            capped[take] = capped[take] or bool(getattr(found, "capped", False))
        total_seconds += max(len(found.pcm) for found in made) / 4 / float(rate)
    if job.cancelled or not pieces[0]:
        _end(job, CANCELLED)
        return None
    # The calibration figure is not noted here: the runtime notes every
    # render's peak itself, and a second note would count each render twice.
    kept = []
    for take in range(takes):
        seed = int(job.seed) + take
        kept.append((b"".join(pieces[take]), rate,
                     _record(job, voices, model_id or _model_in_use(engine), seed, take,
                             takes, len(sections), render_seconds, peak, tokens[take],
                             capped[take])))
    return kept


def _record(job: Job, voices: dict, model_id: str, seed: int, take: int, takes: int,
            sections: int, render_seconds: float, peak: int, tokens: int,
            capped: bool) -> dict:
    """One take's ``render`` record: what made it, and the configuration to make it again.

    The configuration is the one the render used, for the page's Reuse
    settings: the seed is this take's own, so the same take comes out again
    first; model and card are the configuration's own (``""`` is the default),
    and every speaker slot is kept, used by this script or not. ``options`` is
    the engine's options as rendered; a VibeVoice render also writes them flat,
    exactly as every VibeVoice render before it did.
    """
    configured = job.configuration
    engine_id = job.engine or engines.DEFAULT_ENGINE
    speakers = []
    for number, voice in sorted(voices.items()):
        entry = {"n": number, "sample_id": voice.sample_id, "title": voice.title}
        if voice.instruction:
            entry["instruction"] = voice.instruction
        speakers.append(entry)
    stored = dict((configured.get("options") or {}).get(engine_id) or {})
    record = {
        "model_id": model_id,
        "card": job.card,
        "seed": seed,
        "seed_drawn": job.seed_drawn,
    }
    head, tail = _flat_places() if engine_id == FIRST else ((), ())
    if engine_id == FIRST:
        record.update({key: job.options.get(key) for key in head + tail})
    record.update({
        "speakers": speakers,
        "prompt": job.prompt,
        "origin": dict(job.origin),
        "sections": sections,
        "take": take + 1,
        "takes": takes,
        "render_seconds": round(render_seconds, 1),
        "peak_bytes": peak,
        "tokens": tokens,
        "engine": engine_id,
        "options": dict(job.options),
    })
    if capped:
        record["capped"] = True
    configuration = {
        "id": job.configuration_id,
        "name": str(configured.get("name") or ""),
        "model_id": str(configured.get("model_id") or ""),
        "card_uuid": str(configured.get("card_uuid") or ""),
    }
    configuration.update({key: stored.get(key) for key in head})
    configuration.update({"seed": seed, "batch": takes})
    configuration.update({key: stored.get(key) for key in tail})
    configuration.update({
        "speakers": dict(configured.get("speakers") or {}),
        "slots": {key: dict(value) for key, value in dict(configured.get("slots") or {}).items()},
        "engine": engine_id,
        "options": {engine_id: stored},
    })
    record["configuration"] = configuration
    return record


def _model_in_use(engine) -> str:
    """The model the engine loads when neither the configuration nor the settings name one."""
    ask = getattr(engine, "model_id", None)
    try:
        return str(ask() or "") if callable(ask) else ""
    except Exception:
        return ""


def _keep(job: Job, made: list) -> None:
    """Make every take's output, in take order, and end the job done."""
    job.reason = "saving the render" if len(made) == 1 else "saving the takes"
    entries = []
    for index, (samples, rate, record) in enumerate(made):
        name = job.names[index] if index < len(job.names) else job.name
        entry = add_output(samples, rate, name, job.pipeline_id, record)
        entries.append(entry)
        job.output_ids.append(entry["id"])
        if index == 0:
            job.output_id = entry["id"]
    sections = int(made[0][2].get("sections") or 1)
    job.progress = {"section": sections, "sections": sections,
                    "seconds": max(entry["seconds"] for entry in entries)}
    _end(job, DONE)
    logger.info("Model Chain: Voice Box rendered “%s” — %s of speech in %.0f s (%s)",
                job.name, ", ".join(f"{entry['seconds']:.1f} s" for entry in entries),
                float(made[0][2].get("render_seconds") or 0.0), entries[0]["format"].upper())


# --------------------------------------------------------------------------- #
# Voice design (Breeze): a voice from a description, made once and kept
# --------------------------------------------------------------------------- #

_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def design_text(description: str) -> str:
    """The sentence a designed voice speaks when the page sends none."""
    words = str(description or "")
    letters = [character for character in words if not character.isspace()]
    han = len(_HAN.findall(words))
    return DESIGN_TEXT_ZH if letters and han * 2 >= len(letters) else DESIGN_TEXT


def design(description: str, text: str = "", seed=None, configuration_id: str = "",
           inline: dict | None = None) -> dict:
    """Queue a design: one take of ``text`` in a voice made from ``description``.

    An engine that designs (Breeze) samples a new voice from a description on
    every call, so a voice is designed once, listened to, redrawn with the
    next seed until it is the one, and then kept as a sample with its
    transcript already filled (:func:`keep_design`); every line after that is
    a clone of it, and the voice holds across lines, sections and renders.

    The configuration supplies the card and the engine's options -- its CFG
    above all, which guides the description -- and ``seed`` overrides its
    seed (``None``: the configuration's, or one drawn). Nothing is made in any
    pipeline. Returns the job.
    """
    if _turns is None:
        raise VoiceBoxError("Voice Box is not connected to the cards on this WebUI.")
    engine = _engine()
    if not _capabilities(engine).get("design"):
        raise VoiceBoxError(f"{_label(engine)} does not design voices.")
    words = _text(description, MAX_DESCRIPTION_CHARS, "A voice description")
    if not words:
        raise VoiceBoxError("Describe the voice to design first.")
    sentence = _text(text, MAX_DESIGN_TEXT_CHARS, "The design sentence") or design_text(words)
    chosen, source = _chosen_configuration(configuration_id, inline)
    card = _card_of(chosen, engine)
    refused = engine.refusal()
    if refused:
        raise VoiceBoxError(refused)
    engine_id, options = _engine_options(engine, chosen)
    _refuse_options(engine, engine_id, options)
    if seed in ("", None):
        seed = None
    else:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise VoiceBoxError("The seed must be a whole number.") from None
    found_seed, drawn = _seed_for(chosen, 1, seed)
    job = Job(id=_new_id(), name=f"Design — {_title(words, 'a voice')[:40]}", pipeline_id="",
              prompt=sentence, configuration=dict(chosen, batch=1), card=card,
              configuration_id=source, seed=int(found_seed), seed_drawn=drawn, batch=1,
              names=["Designed voice"], kind=KIND_DESIGN, engine=engine_id, options=options,
              design={"description": words, "text": sentence,
                      "cfg_scale": options.get("cfg_scale"), "seed": int(found_seed)})
    queued = job.to_dict()
    _service.submit(job)
    logger.info("Model Chain: Voice Box queued a voice design with %s", _label(engine))
    return queued


def _keep_design_take(job: Job, made: list) -> None:
    samples, rate, _record_ = made[0]
    pcm16 = pcm16_of(samples)
    job.design_pcm = pcm16
    job.design.update({"seconds": round(seconds_of(pcm16, rate), 2), "rate": int(rate),
                       "seed": int(job.seed)})
    job.progress = {"section": 1, "sections": 1, "seconds": job.design["seconds"]}
    _end(job, DONE)
    logger.info("Model Chain: Voice Box designed a voice — %.1f s at seed %d",
                job.design["seconds"], int(job.seed))


def _design_job(identifier: str) -> Job:
    found = _service.find(_identifier(identifier))
    if found is None or found.kind != KIND_DESIGN:
        raise NotFound("That design is no longer listed.")
    if found.phase != DONE or not found.design_pcm:
        raise NotFound("That design has not finished.")
    return found


def design_audio(identifier: str) -> bytes:
    """A finished design's take, as a 16-bit WAV."""
    found = _design_job(identifier)
    return wav_bytes(found.design_pcm, int(found.design.get("rate") or SAMPLE_RATE))


def keep_design(identifier: str, title: str = "") -> dict:
    """Keep a finished design's take as a sample, its transcript the sentence it spoke."""
    found = _design_job(identifier)
    wav = wav_bytes(found.design_pcm, int(found.design.get("rate") or SAMPLE_RATE))
    entry = add_sample(wav, title or "Designed voice", source="design",
                       transcript=str(found.design.get("text") or ""),
                       design={"description": found.design.get("description"),
                               "seed": found.design.get("seed"),
                               "cfg_scale": found.design.get("cfg_scale"),
                               "text": found.design.get("text")})
    logger.info("Model Chain: Voice Box kept a designed voice as “%s”", entry["title"])
    return entry


# --------------------------------------------------------------------------- #
# Transcripts by Whisper (Voice Chat's speech recognition)
# --------------------------------------------------------------------------- #


def stt_status() -> dict:
    """``{"available", "message"}``: whether Whisper can fill a sample's transcript here.

    Voice Chat's recognizer runs in its own CPU worker beside Kokoro, and is
    there whatever Voice Chat engine is selected -- once Voice Chat's models
    are installed. Read from disk; starts nothing.
    """
    try:
        import mc_voice_models

        state = mc_voice_models.status()
    except Exception:
        logger.debug("Model Chain: the Voice Box could not read Voice Chat's installation",
                     exc_info=True)
        return {"available": False,
                "message": "Voice Chat's speech recognition could not be read."}
    if state.ready:
        return {"available": True, "message": ""}
    return {"available": False,
            "message": "Whisper is Voice Chat's speech recognition — install Voice Chat's "
                       "models in Settings → Voice Chat to fill transcripts with it."}


def transcribe(identifier: str) -> dict:
    """Whisper's reading of a sample, for the page to put in its transcript box. Not saved.

    The sample is resampled to the 16 kHz Voice Chat's dictation sends, and
    Voice Chat's worker transcribes it -- in thirty-second windows cut at the
    quietest moments when it is longer, because Whisper hears thirty seconds
    at a time. A result that is only one of Whisper's non-speech annotations
    is no transcript. The person checks the words and saves them: Breeze
    clones from the exact words, and a recognizer's guess is a draft.
    """
    entry = sample(identifier)
    state = stt_status()
    if not state["available"]:
        raise Unavailable(state["message"])
    import array as _array

    import mc_voice_hearing as hearing
    import mc_voice_runtime

    pcm16, rate = sample_pcm(entry["id"])
    samples = _array.array("h")
    samples.frombytes(pcm16[:len(pcm16) // 2 * 2])
    if struct.pack("<h", 1) != struct.pack("=h", 1):
        samples.byteswap()
    narrowed = reference.resample(samples, rate, TRANSCRIBE_RATE)
    if struct.pack("<h", 1) != struct.pack("=h", 1):
        narrowed.byteswap()
    try:
        found = mc_voice_runtime.transcribe(wav_bytes(narrowed.tobytes(), TRANSCRIBE_RATE))
    except mc_voice_runtime.VoiceRuntimeError as exc:
        raise Unavailable(str(exc)) from None
    words = " ".join(hearing.speech(str(found.get("text") or "")).split())
    if not words:
        raise VoiceBoxError("Whisper heard no words in that sample.")
    logger.info("Model Chain: Voice Box transcribed a sample — %.1f s of audio",
                float(entry.get("seconds") or 0.0))
    return {"text": words[:MAX_TRANSCRIPT_CHARS], "seconds": entry.get("seconds")}


def jobs() -> list[dict]:
    """The last :data:`JOBS_KEPT` jobs, newest first."""
    return _service.listed()


def job(identifier: str) -> dict:
    found = _service.find(_identifier(identifier))
    if found is None:
        raise NotFound("That render job is no longer listed.")
    return found.to_dict()


def cancel_job(identifier: str) -> dict:
    """Withdraw a queued job, or interrupt a waiting or rendering one."""
    found = _service.find(_identifier(identifier))
    if found is None:
        raise NotFound("That render job is no longer listed.")
    if found.phase not in LIVE:
        return found.to_dict()
    _interrupt(found)
    if found.phase == QUEUED:
        _end(found, CANCELLED)
    return found.to_dict()


def clear_queue() -> int:
    """Withdraw every queued job on every card; how many were. The running ones are left alone."""
    cleared = _service.clear()
    if cleared:
        logger.info("Model Chain: Voice Box withdrew %d queued render%s", cleared,
                    "" if cleared == 1 else "s")
    return cleared


def stop() -> None:
    """Cancel everything queued or running. For the WebUI's shutdown door."""
    _service.stop()


def forget() -> None:
    """Drop the service's memory of jobs, once its renders have ended. For tests.

    Waiting matters: a render still going when a test ends finishes a moment
    later and writes its output wherever the voice root points by then, which
    is the checkout once the test's own folder is put back.
    """
    global _service, _turns, _runtime_module, _engine_module
    _service.stop()
    _service.join(timeout=10.0)
    _service = _Service()
    _turns = None
    _runtime_module = None
    _engine_module = None
