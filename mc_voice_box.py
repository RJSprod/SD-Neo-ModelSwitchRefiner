"""Voice Box: the samples, prompts, configurations, pipelines and outputs of a
VibeVoice render, and the service that renders them.

The page (``javascript/voice_box.js``) is four stages in a row -- INPUT, PROMPT,
CONFIGURATION, OUTPUTS -- and a pipeline is one pass through them, kept like a
conversation is kept: a file of its own with its prompt, the configuration it
points at and the outputs it made. Everything a live page changes lives in
files under the voice data root, never in Forge's settings, so *Apply settings*
cannot write a stale copy over a folder the user chose a minute ago.

This module does the arithmetic and the files. It never touches a card: a
render asks for its card through the client :func:`use_turns` was handed
(:mod:`mc_turns_guests`), and speaks through the runtime :func:`use_runtime`
names (:mod:`mc_voice_vibevoice_runtime`). Both are seams, because the
independence of the voice side from the memory side is an invariant
(``tests/test_voice_independence.py``) and because a render service whose rules
could only be tested with a GPU would be one whose rules nobody had tested.

Sound is 16-bit mono PCM at 24 kHz throughout -- the rate VibeVoice speaks at.
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

import mc_voice_paths as paths
import mc_voice_reference as reference

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""


class VoiceBoxError(RuntimeError):
    """A request the Voice Box will not carry out, in a sentence the page shows."""

    status = 400


class NotFound(VoiceBoxError):
    status = 404


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
META_FILENAME = "meta.json"

SAMPLE_RATE = 24000
"""VibeVoice's rate, for samples and outputs alike."""

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

STEPS_RANGE = (1, 50)
CFG_RANGE = (1.0, 3.0)
SEED_MAX = 2**31 - 1
TOKENS_MAX = 65_536

ENVELOPE = reference.Envelope(engine="VibeVoice", minimum_seconds=SAMPLE_MIN_SECONDS,
                              maximum_seconds=SAMPLE_MAX_SECONDS,
                              maximum_bytes=MAX_SAMPLE_BYTES, target_rate=SAMPLE_RATE,
                              minimum_peak=SAMPLE_MIN_PEAK)
"""What a sample may be. Three seconds is the least VibeVoice conditions on
usefully; a minute is more than any speaker prompt needs and keeps a sample a
sample rather than an archive."""

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
    if _runtime_module is not None:
        return _runtime_module
    import mc_voice_vibevoice_runtime

    return mc_voice_vibevoice_runtime


def _engine():
    if _engine_module is not None:
        return _engine_module
    import mc_voice_vibevoice

    return mc_voice_vibevoice


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


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #


def _samples_root() -> Path:
    return _folder(SAMPLES_DIRNAME)


def _sample_meta(identifier: str) -> Path:
    return _samples_root() / identifier / META_FILENAME


def _sample_audio(identifier: str) -> Path:
    return _samples_root() / identifier / AUDIO_FILENAME


def add_sample(wav: bytes, title: str = "", source: str = "file") -> dict:
    """Take a recording -- the browser's trimmed selection -- into the library.

    ``source`` says where it came from: ``file``, ``microphone`` or
    ``output:<id>`` for a piece of a render trimmed back into a sample.
    """
    canonical, _seconds = reference.normalize(wav, ENVELOPE, error=VoiceBoxError)
    pcm16, rate = pcm_of(canonical)
    identifier = _new_id()
    entry = {
        "id": identifier,
        "title": _title(title, f"Sample {len(samples()) + 1}"),
        "seconds": round(seconds_of(pcm16, rate), 2),
        "rate": int(rate),
        "peaks": peaks_of(pcm16),
        "source": str(source or "file")[:64],
        "created": _now(),
    }
    with _lock:
        _write_bytes(_sample_audio(identifier), wav_bytes(pcm16, rate))
        _write_json(_sample_meta(identifier), entry)
    logger.info("Model Chain: Voice Box kept sample “%s” (%.1f s)", entry["title"],
                entry["seconds"])
    return entry


def samples() -> list[dict]:
    with _lock:
        found = []
        for child in sorted(_samples_root().iterdir()) if _samples_root().exists() else []:
            entry = _read_json(child / META_FILENAME, {})
            if entry.get("id") and (child / AUDIO_FILENAME).exists():
                found.append(entry)
    found.sort(key=lambda entry: entry.get("created", 0))
    return found


def sample(identifier: str) -> dict:
    identifier = _identifier(identifier)
    entry = _read_json(_sample_meta(identifier), {})
    if not entry or not _sample_audio(identifier).exists():
        raise NotFound("That sample is no longer in the library.")
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

CONFIGURATION_DEFAULTS = {
    "name": "Default",
    "model_id": "",
    "card_uuid": "",
    "steps": 10,
    "cfg_scale": 1.3,
    "seed": None,
    "max_new_tokens": None,
    "precision": "bf16",
    "lora_id": "",
    "lora_scale": 1.0,
    "speakers": {},
}
"""A configuration's fields. ``model_id`` and ``card_uuid`` empty mean the
engine's default model and the card chosen in Voice Box's settings.

``precision``, ``lora_id`` and ``lora_scale`` are the 7B's: how its language
model is held on the card (full, 8-bit or 4-bit) and which fine-tune, at what
strength, is applied to it. A model that takes neither -- the Realtime 0.5B --
is saved with ``bf16``, no LoRA and ``1.0``, and a configuration that asks it
for anything else is refused with a sentence rather than quietly changed.

``speakers`` maps a speaker number to a sample id for a model that clones
from samples, and to ``"preset:<stem>"`` for one that speaks with its own
preset voices; which of the two a model takes is the engine's to say
(:func:`_model_info`)."""

PRESET_PREFIX = "preset:"

LORA_SCALE_RANGE = (0.0, 2.0)

_FALLBACK_MODEL = {"id": "", "label": "VibeVoice", "kind": "longform", "precisions": ["bf16"],
                   "lora": False, "max_speakers": MAX_SPEAKERS, "voices": "samples",
                   "defaults": {"steps": 10, "cfg_scale": 1.3}, "presets": []}
"""What a model is taken to be when the engine cannot describe it: the 7B's
shape at full precision, which is what every configuration saved before the
engine could describe its models was made for."""


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


def _bounded_float(value, low: float, high: float, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise VoiceBoxError(f"{what} must be a number.") from None
    if not low <= number <= high:
        raise VoiceBoxError(f"{what} must be between {low:g} and {high:g}.")
    return round(number, 3)


def _model_info(model_id: str = "") -> dict:
    """What the engine says about ``model_id``: its kind, voices, precisions, LoRA.

    ``""`` is the model Voice Box's settings name, else the engine's default. An
    engine that cannot answer describes nothing, and :data:`_FALLBACK_MODEL` --
    the 7B at full precision -- stands in, so a configuration saved before models
    could be described still renders the way it always did.
    """
    engine = _engine()
    wanted = str(model_id or settings().get("model_id") or getattr(engine, "MODEL_DEFAULT", "")
                 or "")
    try:
        found = dict(engine.model_info(wanted) or {})
    except AttributeError:
        found = {}
    except Exception as exc:
        raise VoiceBoxError(str(exc) or "That model is not one this Voice Box knows.") from None
    merged = dict(_FALLBACK_MODEL, id=wanted)
    merged.update({key: value for key, value in found.items() if value is not None})
    return merged


def _model_ids() -> set:
    try:
        return {str(info.get("id")) for info in (_engine().models_info() or [])}
    except Exception:
        return set()


def _lora_ids() -> dict:
    try:
        return {str(entry.get("id")): entry for entry in (_engine().loras() or [])}
    except Exception:
        return {}


def _preset_ids(info: dict) -> dict:
    return {str(entry.get("id")): entry for entry in (info.get("presets") or [])}


def _clean_speakers(value, info: dict | None = None) -> dict:
    """The speaker slots ``info``'s model can use; anything else is dropped.

    Dropped rather than refused, because a slot goes stale by itself -- a sample
    deleted, a model switched from the 7B to the Realtime one -- and a
    configuration that could no longer be saved because of it would be one the
    page could not fix.
    """
    info = info or _FALLBACK_MODEL
    found = {}
    if not isinstance(value, dict):
        return found
    presets = info.get("voices") == "presets"
    limit = max(1, min(int(info.get("max_speakers") or MAX_SPEAKERS), MAX_SPEAKERS))
    known = _preset_ids(info) if presets else {entry["id"] for entry in samples()}
    for number, chosen in value.items():
        try:
            n = int(number)
        except (TypeError, ValueError):
            continue
        text = str(chosen or "")
        if not 1 <= n <= limit:
            continue
        if presets:
            if text.startswith(PRESET_PREFIX) and text[len(PRESET_PREFIX):] in known:
                found[str(n)] = text
        elif text in known:
            found[str(n)] = text
    return found


def _clean_configuration(values: dict, existing: dict | None = None) -> dict:
    base = dict(existing or CONFIGURATION_DEFAULTS)
    merged = dict(CONFIGURATION_DEFAULTS, **base)
    for key in CONFIGURATION_DEFAULTS:
        if key in values:
            merged[key] = values[key]
    model_id = str(merged.get("model_id") or "")[:64]
    known = _model_ids()
    if model_id and known and model_id not in known:
        raise VoiceBoxError("That model is not one this Voice Box knows.")
    info = _model_info(model_id)
    label = info.get("label") or "That model"
    precision = str(merged.get("precision") or "bf16")
    if precision not in (info.get("precisions") or ["bf16"]):
        raise VoiceBoxError(f"{label} cannot run at that precision.")
    lora_id = str(merged.get("lora_id") or "")
    if lora_id:
        if not info.get("lora"):
            raise VoiceBoxError(f"{label} does not take a LoRA.")
        if lora_id not in _lora_ids():
            raise VoiceBoxError("That LoRA is no longer in the library.")
    return {
        "name": _title(merged.get("name"), base.get("name") or "Configuration"),
        "model_id": model_id,
        "card_uuid": str(merged.get("card_uuid") or "")[:64],
        "steps": _bounded_int(merged.get("steps"), *STEPS_RANGE, what="Diffusion steps"),
        "cfg_scale": _bounded_float(merged.get("cfg_scale"), *CFG_RANGE, what="CFG"),
        "seed": _bounded_int(merged.get("seed"), 0, SEED_MAX, what="The seed", none_ok=True),
        "max_new_tokens": _bounded_int(merged.get("max_new_tokens"), 1, TOKENS_MAX,
                                       what="Max new tokens", none_ok=True),
        "precision": precision,
        "lora_id": lora_id,
        "lora_scale": _bounded_float(merged.get("lora_scale", 1.0), *LORA_SCALE_RANGE,
                                     what="The LoRA's strength"),
        "speakers": _clean_speakers(merged.get("speakers"), info),
    }


def configurations() -> list[dict]:
    with _lock:
        found = [_read_json(path, {}) for path in sorted(_configurations_root().glob("*.json"))]
    found = [entry for entry in found if entry.get("id")]
    found.sort(key=lambda entry: (entry.get("name", "").lower(), entry.get("created", 0)))
    return found


def configuration(identifier: str) -> dict:
    identifier = _identifier(identifier)
    entry = _read_json(_configuration_path(identifier), {})
    if not entry.get("id"):
        raise NotFound("That configuration no longer exists.")
    return entry


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


def forget_lora(identifier: str) -> list[str]:
    """Take a deleted LoRA out of every configuration that used it.

    The same treatment a deleted sample gets: the configuration keeps
    everything else and renders without a LoRA, rather than being left naming
    one that :func:`_clean_configuration` would refuse the next time it is
    saved. Returns the ids of the configurations that changed.
    """
    identifier = str(identifier or "")
    changed = []
    if not identifier:
        return changed
    with _lock:
        for entry in configurations():
            if entry.get("lora_id") != identifier:
                continue
            entry["lora_id"] = ""
            entry["lora_scale"] = 1.0
            _write_json(_configuration_path(entry["id"]), entry)
            changed.append(entry["id"])
    return changed


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


def _output_audio(identifier: str) -> Path:
    return _outputs_root() / identifier / AUDIO_FILENAME


def add_output(pcm16: bytes, rate: int, name: str, pipeline_id: str, render: dict) -> dict:
    identifier = _new_id()
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
    with _lock:
        _write_bytes(_output_audio(identifier), wav_bytes(pcm16, rate))
        _write_json(_output_meta(identifier), entry)
    if pipeline_id:
        _attach_output(pipeline_id, identifier)
    return entry


def outputs(pipeline_id: str = "") -> list[dict]:
    with _lock:
        found = []
        for child in sorted(_outputs_root().iterdir()) if _outputs_root().exists() else []:
            entry = _read_json(child / META_FILENAME, {})
            if entry.get("id") and (child / AUDIO_FILENAME).exists():
                found.append(entry)
    if pipeline_id:
        found = [entry for entry in found if entry.get("pipeline_id") == pipeline_id]
    found.sort(key=lambda entry: entry.get("created", 0), reverse=True)
    return found


def output(identifier: str) -> dict:
    identifier = _identifier(identifier)
    entry = _read_json(_output_meta(identifier), {})
    if not entry.get("id") or not _output_audio(identifier).exists():
        raise NotFound("That render is no longer in the Voice Box.")
    return entry


def rename_output(identifier: str, name: str) -> dict:
    with _lock:
        entry = output(identifier)
        entry["name"] = _title(name, entry.get("name") or "Render")
        _write_json(_output_meta(entry["id"]), entry)
    return entry


def set_loop(identifier: str, loop: bool) -> dict:
    with _lock:
        entry = output(identifier)
        entry["loop"] = bool(loop)
        _write_json(_output_meta(entry["id"]), entry)
    return entry


def delete_output(identifier: str) -> dict:
    with _lock:
        entry = output(identifier)
        shutil.rmtree(_outputs_root() / entry["id"], ignore_errors=True)
        if entry.get("pipeline_id"):
            try:
                owner = pipeline(entry["pipeline_id"])
                owner["outputs"] = [o for o in owner.get("outputs") or [] if o != entry["id"]]
                _write_json(_pipeline_path(owner["id"]), owner)
            except NotFound:
                pass
    return {"deleted": entry["id"]}


def output_wav(identifier: str) -> bytes:
    return _output_audio(output(identifier)["id"]).read_bytes()


def save_output(identifier: str) -> str:
    """Copy a render, with its metadata beside it, into the remembered folder.

    The folder is chosen once and kept (:func:`choose_save_folder`); a save
    with none chosen is refused with the sentence that tells the page to ask.
    A name already taken gets a number, never an overwrite.
    """
    entry = output(identifier)
    folder = str(settings().get("save_folder") or "")
    if not folder:
        raise VoiceBoxError("Choose a folder to save renders into first.")
    target = Path(folder)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VoiceBoxError(f"The save folder cannot be used: {exc.strerror or exc}") from None
    stem = _safe_filename(entry.get("name"))
    candidate, number = target / f"{stem}.wav", 2
    while candidate.exists():
        candidate = target / f"{stem} ({number}).wav"
        number += 1
    shutil.copyfile(_output_audio(entry["id"]), candidate)
    sidecar = {key: value for key, value in entry.items() if key != "peaks"}
    _write_json(candidate.with_suffix(".json"), sidecar)
    logger.info("Model Chain: Voice Box saved “%s” to %s", entry["name"], candidate)
    return str(candidate)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

SETTINGS_DEFAULTS = {"save_folder": "", "card_uuid": "", "model_id": "", "keep_warm": True}


def _settings_path() -> Path:
    return root() / SETTINGS_FILENAME


def settings() -> dict:
    found = dict(SETTINGS_DEFAULTS)
    stored = _read_json(_settings_path(), {})
    for key in SETTINGS_DEFAULTS:
        if key in stored:
            found[key] = stored[key]
    found["keep_warm"] = bool(found.get("keep_warm", True))
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

    @property
    def speakers(self) -> list[int]:
        return sorted({number for number, _ in self.lines})


_PAUSE = re.compile(r"\[\s*pause(?:\s*[:=]\s*(\d{1,6})\s*(?:ms)?)?\s*\]", re.IGNORECASE)
_SPEAKER = re.compile(r"^\s*(?:speaker\s*(\d+)|\[\s*(\d+)\s*\])\s*[:：]\s*(.*)$", re.IGNORECASE)


def parse_script(text: str) -> list[Section]:
    """Sections of ``text``, each a list of ``(speaker, words)`` lines.

    ``Speaker 2:`` (or ``[2]:``) at the start of a line names its speaker; a
    line with no name continues the one before it, or is Speaker 1's when none
    has spoken. ``[pause]`` and ``[pause:1500]`` end a section and put that much
    silence before the next -- the default :data:`PAUSE_DEFAULT_MS`, clamped to
    :data:`PAUSE_MIN_MS`..:data:`PAUSE_MAX_MS`.
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
        lines = _lines_of(chunk)
        if lines:
            sections.append(Section(lines=lines, pause_ms=pending_pause if sections else None))
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


def _lines_of(chunk: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    current: int | None = None
    for raw in chunk.splitlines():
        match = _SPEAKER.match(raw)
        if match:
            number = int(match.group(1) or match.group(2))
            if not 1 <= number <= MAX_SPEAKERS:
                raise VoiceBoxError(f"Speakers are numbered 1 to {MAX_SPEAKERS}; "
                                    f"the prompt names Speaker {number}.")
            current = number
            body = " ".join(match.group(3).split())
            lines.append((number, body))
            continue
        body = " ".join(raw.split())
        if not body:
            continue
        if current is None:
            current = 1
            lines.append((1, body))
        else:
            number, previous = lines[-1]
            lines[-1] = (number, f"{previous} {body}".strip())
    return [(number, body) for number, body in lines if body]


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

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "pipeline_id": self.pipeline_id,
                "phase": self.phase, "reason": self.reason, "warning": self.warning,
                "progress": dict(self.progress), "output_id": self.output_id,
                "created": self.created, "started": self.started, "ended": self.ended,
                "card": self.card, "live": self.phase in LIVE}


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
        """Jobs of a pipeline that have not made their output yet."""
        with self.lock:
            return sum(1 for job in self.jobs
                       if job.pipeline_id == pipeline_id and job.phase in LIVE)

    def stop(self) -> None:
        with self.lock:
            for queue in self.queues.values():
                for job in queue:
                    job.cancel_event.set()
            live = list(self.current.items())
        for key, job in live:
            _interrupt(job)


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


def render(pipeline_id: str, prompt: str, configuration_id: str = "", name: str = "",
           inline: dict | None = None) -> dict:
    """Queue a render of ``prompt`` with a configuration, for a pipeline. Returns the job.

    The configuration is the saved one ``configuration_id`` names, or ``inline``
    -- the page's unsaved values -- validated the same way. Everything that can
    be refused before a card is asked for is refused here, to the caller: the
    prompt is parsed, the configuration read, the speakers checked against the
    samples, the card decided. What follows -- the turn, the load, the render --
    happens on the card's own thread and is read back from the job.
    """
    if _turns is None:
        raise VoiceBoxError("Voice Box is not connected to the cards on this WebUI.")
    owner = pipeline(pipeline_id)
    sections = parse_script(prompt)
    if inline is not None:
        chosen = _clean_configuration(dict(inline))
    elif configuration_id:
        stored = configuration(configuration_id)
        chosen = dict(stored, **_clean_configuration(stored, stored))
    else:
        chosen = _clean_configuration({})
    wanted = sorted({number for section in sections for number in section.speakers})
    info = _model_info(chosen.get("model_id"))
    limit = int(info.get("max_speakers") or MAX_SPEAKERS)
    if wanted and wanted[-1] > limit:
        raise VoiceBoxError(
            f"{info.get('label') or 'That model'} speaks with one voice, and the script names "
            f"Speaker {wanted[-1]}." if limit == 1 else
            f"{info.get('label') or 'That model'} speaks with up to {limit} voices, and the "
            f"script names Speaker {wanted[-1]}.")
    for number in wanted:
        if not chosen["speakers"].get(str(number)):
            if info.get("voices") == "presets":
                raise VoiceBoxError(f"Choose a voice for Speaker {number} in the configuration.")
            raise VoiceBoxError(f"Speaker {number} has no sample — choose one in the "
                                f"configuration's speaker {number} slot.")
    card = chosen.get("card_uuid") or settings().get("card_uuid") or ""
    if not card:
        raise VoiceBoxError("Choose the card VibeVoice renders on, in the configuration "
                            "or in Voice Box's settings.")
    refused = _refusal(info.get("id") or "")
    if refused:
        raise VoiceBoxError(refused)
    remember_prompt(prompt)
    if not name:
        number = len(owner.get("outputs") or []) + _service.pending(owner["id"]) + 1
        name = f"{owner.get('name') or 'Render'} {number}"
    job = Job(id=_new_id(), name=_title(name, "Render"), pipeline_id=owner["id"],
              prompt=str(prompt), configuration=chosen, card=card)
    _service.submit(job)
    logger.info("Model Chain: Voice Box queued “%s” (%d section%s, %d speaker%s)", job.name,
                len(sections), "" if len(sections) == 1 else "s", len(wanted),
                "" if len(wanted) == 1 else "s")
    return job.to_dict()


def _refusal(model_id: str) -> str:
    """Why ``model_id`` cannot render now, in the engine's words, or ``""``."""
    engine = _engine()
    try:
        return str(engine.refusal(model_id=model_id) or "")
    except TypeError:
        return str(engine.refusal() or "")


def _perform(job: Job) -> None:
    engine = _engine()
    runtime = _runtime()
    sections = parse_script(job.prompt)
    info = _model_info(job.configuration.get("model_id"))
    presets = _preset_ids(info)
    voices = {}
    for number in sorted({n for section in sections for n in section.speakers}):
        chosen = str(job.configuration["speakers"].get(str(number)) or "")
        if chosen.startswith(PRESET_PREFIX):
            stem = chosen[len(PRESET_PREFIX):]
            voices[number] = {"preset": stem, "sample_id": "",
                              "title": str((presets.get(stem) or {}).get("name") or stem)}
            continue
        try:
            pcm16, rate = sample_pcm(chosen)
            title = sample(chosen).get("title", "")
        except VoiceBoxError:
            _end(job, FAILED, warning=f"Speaker {number}'s sample is no longer in the library.")
            return
        voices[number] = {"pcm": float32_of(pcm16), "rate": rate, "sample_id": chosen,
                          "title": title}
    model_id = str(info.get("id") or "")
    precision = str(job.configuration.get("precision") or "bf16")
    job.started = _now()
    job.phase = WAITING
    job.reason = "asking for the card"
    turn = _turns.request(job.card,
                          need_vram=int(engine.need_vram_bytes(model_id, precision)),
                          need_ram=int(engine.need_ram_bytes(model_id, precision)),
                          label=f"{engine.LABEL} — {job.name}")
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
        _render_granted(job, sections, voices, model_id, engine, runtime, precision)
    finally:
        try:
            turn.finish(keep_warm=None if settings().get("keep_warm", True) else False)
        except Exception:
            logger.warning("Model Chain: could not hand the card back after a render",
                           exc_info=True)


def _render_granted(job: Job, sections, voices, model_id: str, engine, runtime,
                    precision: str = "bf16") -> None:
    job.phase = LOADING
    job.reason = "loading the model"
    # The whole identity, named: a warm model loaded for another configuration --
    # another precision, another LoRA -- is replaced rather than reused.
    lora_id = str(job.configuration.get("lora_id") or "")
    lora_scale = float(job.configuration.get("lora_scale", 1.0))
    runtime.load(job.card, model_id=model_id, precision=precision, lora_id=lora_id,
                 lora_scale=lora_scale)
    pieces: list[bytes] = []
    total_seconds = 0.0
    peak = 0
    rate = SAMPLE_RATE
    render_seconds = 0.0
    tokens = 0
    for index, section in enumerate(sections):
        if job.cancelled:
            break
        job.phase = RENDERING
        job.reason = ""
        job.progress = {"section": index + 1, "sections": len(sections),
                        "seconds": round(total_seconds, 1)}
        if section.pause_ms and pieces:
            pieces.append(silence(section.pause_ms, rate))
            total_seconds += section.pause_ms / 1000.0

        def progressed(found: dict, _base=total_seconds, _index=index):
            job.progress = {"section": _index + 1, "sections": len(sections),
                            "seconds": round(_base + float(found.get("seconds") or 0.0), 1)}

        job.render_id = f"{job.id}:{index + 1}"
        request = runtime.RenderJob(
            id=job.render_id,
            script=[(number, body) for number, body in section.lines],
            voices={number: ({"preset": voice["preset"]} if voice.get("preset")
                             else (voice["pcm"], voice["rate"]))
                    for number, voice in voices.items() if number in section.speakers},
            cfg_scale=float(job.configuration["cfg_scale"]),
            steps=int(job.configuration["steps"]),
            seed=job.configuration.get("seed"),
            max_new_tokens=job.configuration.get("max_new_tokens"))
        result = runtime.render(job.card, request, on_progress=progressed)
        peak = max(peak, int(getattr(result, "peak_bytes", 0) or 0))
        render_seconds += float(getattr(result, "render_seconds", 0.0) or 0.0)
        tokens += int(getattr(result, "tokens", 0) or 0)
        if getattr(result, "cancelled", False):
            job.cancel_event.set()
            break
        pcm16, rate = pcm_of(result.wav)
        pieces.append(pcm16)
        total_seconds += seconds_of(pcm16, rate)
    if job.cancelled or not pieces:
        _end(job, CANCELLED)
        return
    lora = _lora_ids().get(lora_id) or {}
    entry = add_output(b"".join(pieces), rate, job.name, job.pipeline_id, {
        "model_id": model_id,
        "model": str(_model_info(model_id).get("label") or model_id),
        "precision": precision,
        "lora": ({"id": lora_id, "name": str(lora.get("name") or lora_id),
                  "scale": lora_scale} if lora_id else None),
        "card": job.card,
        "seed": job.configuration.get("seed"),
        "steps": job.configuration["steps"],
        "cfg_scale": job.configuration["cfg_scale"],
        "max_new_tokens": job.configuration.get("max_new_tokens"),
        "speakers": [{"n": number, "sample_id": voice["sample_id"], "title": voice["title"],
                      "preset": voice.get("preset", "")}
                     for number, voice in sorted(voices.items())],
        "prompt": job.prompt,
        "sections": len(sections),
        "render_seconds": round(render_seconds, 1),
        "peak_bytes": peak,
        "tokens": tokens,
    })
    job.output_id = entry["id"]
    job.progress = {"section": len(sections), "sections": len(sections),
                    "seconds": entry["seconds"]}
    # The peak is on the output for the record, and not noted with the engine
    # here: the runtime notes every render's peak itself, under the model and
    # precision it actually loaded, and a second note from here counted each
    # render twice in the calibration.
    _end(job, DONE)
    logger.info("Model Chain: Voice Box rendered “%s” — %.1f s of speech in %.0f s", job.name,
                entry["seconds"], render_seconds)


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


def stop() -> None:
    """Cancel everything queued or running. For the WebUI's shutdown door."""
    _service.stop()


def forget() -> None:
    """Drop the service's memory of jobs. For tests."""
    global _service, _turns, _runtime_module, _engine_module
    _service.stop()
    _service = _Service()
    _turns = None
    _runtime_module = None
    _engine_module = None
