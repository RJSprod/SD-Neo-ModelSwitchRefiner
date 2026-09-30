"""VibeVoice as a Voice Chat engine: which voices exist, which one speaks, and cloning.

The fourth text-to-speech engine's product side, beside :mod:`mc_voice_kokoro`,
:mod:`mc_voice_sopro` and :mod:`mc_voice_pocket`, answering the same contract
(``entries``, ``lookup``, ``default_id``, ``set_default``, ``resolve``,
``rename``, ``delete``, ``capabilities``, ``refusals``, ``public_status`` and
the rest). Speaking is :mod:`mc_voice_vibevoice_speech`'s; installing is the
Voice Box's (:mod:`mc_voice_vibevoice`), because VibeVoice is one installation
that two tabs speak through.

What makes this engine unlike the other three
---------------------------------------------
It runs on a graphics card, through a turn. Every reply it speaks asks the turn
system for its card (``mc_voice_box.turns()``), so a reply on a card that is
busy rendering an image or writing the reply itself starts when that work ends.
Nothing here asks for a card: this module answers questions about voices, and
answering them never starts a worker, loads a model or asks for a turn.

Its voices come from two models, and a voice says which:

    vibevoice:preset:<stem>     one of the Realtime 0.5B's preset voice prompts,
                                Microsoft's, installed with that model. It
                                cannot be renamed or deleted: it ships with
                                the model, and Microsoft withholds the code that
                                makes one from a recording.
    vibevoice:sample:<id>       a Voice Box sample, which the 7B clones at
                                render time from the recording itself. Renaming
                                or deleting it here is renaming or deleting it
                                in the Voice Box: it is the same sample.

A voice cloned here *is* a Voice Box sample
-------------------------------------------
The 7B does not prepare anything per voice -- it reads the sample on every
render -- so a voice made from a recording in Voice Chat is saved exactly where
the Voice Box keeps its own (``mc_voice_box.add_sample``, source ``voice-chat``),
and every sample made in the Voice Box is a voice here. Nothing goes stale when
a precision or a LoRA changes, which is why this engine has no Rebuild. The
preview transaction is Pocket's (:func:`prepare_preview`, :func:`save_preview`,
:func:`discard_preview`), with one difference that matters: the pending
recording lives in this process's memory and nowhere else until Save.

Where its choices live
----------------------
The default voice and the delivery are in Voice Chat's own VibeVoice file
(:func:`mc_voice_paths.vibevoice_chat_path`), written the moment they change and
never in a Forge option -- an option is also a component on the settings page,
and *Apply settings* writes the page's build-time copy back over what the panel
set since (I-PKT-19). The card, precision and LoRA a reply uses are the
installer's ``settings()["chat"]``, validated there.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field

import mc_voice_box as box
import mc_voice_engines as engines
import mc_voice_paths as paths
import mc_voice_reference as reference

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

ENGINE = engines.VIBEVOICE

LABEL = "VibeVoice"

TAB = "Voice Box"
"""The tab VibeVoice is installed from, by the name the WebUI shows it under."""

MODEL_7B = "vibevoice-7b"
MODEL_REALTIME = "vibevoice-realtime-0.5b"
"""The two models, by the ids the installer's manifest gives them. Spelled here
rather than read from :mod:`mc_voice_vibevoice` so that describing a voice never
depends on that module importing; ``tests/test_voice_vibevoice_chat.py`` holds
the two spellings equal."""

PRESET_PREFIX = f"{ENGINE}:preset:"
SAMPLE_PREFIX = f"{ENGINE}:sample:"

PREFERRED_PRESET = "en-Carter_man"
"""Where an installation with no chosen voice starts, when it is installed."""

CHAT_SCHEMA = 1
DEFAULT_KEY = "default"
"""The default voice's key in Voice Chat's VibeVoice file."""

CHAT_SETTINGS = ("card_uuid", "precision", "lora_id", "lora_scale")
"""The engine settings this panel may change: the installer's ``settings()["chat"]``."""

IDEAL_REFERENCE_SECONDS = 10.0
"""The length the clone form suggests. Not the model's limit -- a sample is 3 to
60 seconds -- but every sentence a cloned voice speaks re-reads its sample, so a
short clean one starts speaking sooner than a long one."""

MAX_NAME_CHARS = box.MAX_TITLE_CHARS

DEFAULT_TEST_TEXT = "This is a test of the VibeVoice voice."

NO_VOICE = (f"VibeVoice has no voice to speak with yet. Install the Realtime model for its "
            f"preset voices, or add a sample in the {TAB} tab.")


class VibeVoiceChatError(RuntimeError):
    """A VibeVoice request Voice Chat refused. A state somebody can act on, never a fault."""


_lock = threading.RLock()


# --------------------------------------------------------------------------- #
# The seams
# --------------------------------------------------------------------------- #


def _vibevoice():
    """The installer, imported when it is first needed rather than at start-up."""
    import mc_voice_vibevoice

    return mc_voice_vibevoice


def _speech():
    import mc_voice_vibevoice_speech

    return mc_voice_vibevoice_speech


# --------------------------------------------------------------------------- #
# Voice Chat's own VibeVoice file
# --------------------------------------------------------------------------- #


def _read() -> dict:
    try:
        found = json.loads(paths.vibevoice_chat_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _write(found: dict) -> None:
    """Replace the file atomically: written beside itself, then renamed."""
    path = paths.vibevoice_chat_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".json.new")
    staging.write_text(json.dumps(found, indent=2), encoding="utf-8")
    os.replace(staging, path)


def _stored(names) -> dict:
    """Some values out of the file, ``None`` for what it does not hold.

    Out of the file and out of nothing else. There is no older build whose
    Forge option this has to honour, so a host option under one of these names
    -- a stale copy *Apply settings* wrote, say -- is never read.
    """
    found = _read()
    return {name: found.get(name) for name in names}


def _store(values: dict) -> None:
    """Write some values into the file, in one atomic replacement."""
    with _lock:
        found = _read()
        found.update(values)
        found["schema"] = CHAT_SCHEMA
        _write(found)


# --------------------------------------------------------------------------- #
# What is installed
# --------------------------------------------------------------------------- #


@dataclass
class Status:
    """What VibeVoice can speak with, in one object the panel and the facade read."""

    supported: bool = False
    runtime_installed: bool = False
    runtime_message: str = ""
    realtime_installed: bool = False
    longform_installed: bool = False
    presets: list = field(default_factory=list)
    """The Realtime model's presets that are installed, in the installer's shape."""
    samples: list = field(default_factory=list)
    """The Voice Box's samples, in its shape."""
    models: list = field(default_factory=list)
    """Every model, as the panel may see it."""

    @property
    def ready(self) -> bool:
        """The runtime and at least one model with a voice to speak with."""
        return bool(self.runtime_installed
                    and ((self.realtime_installed and self.presets)
                         or (self.longform_installed and self.samples)))

    @property
    def cloning_ready(self) -> bool:
        """Whether a voice can be made from a recording: the runtime and the 7B."""
        return bool(self.runtime_installed and self.longform_installed)

    @property
    def cloning_message(self) -> str:
        if self.cloning_ready:
            return ""
        missing = []
        if not self.runtime_installed:
            missing.append("VibeVoice's runtime")
        if not self.longform_installed:
            missing.append("the VibeVoice 7B model")
        return (f"Making a voice from a recording needs {' and '.join(missing)}, installed in "
                f"the {TAB} tab. The Realtime 0.5B speaks Microsoft's preset voices only and "
                f"cannot make one.")

    @property
    def message(self) -> str:
        if self.ready:
            parts = []
            if self.realtime_installed and self.presets:
                count = len(self.presets)
                parts.append(f"{count} preset voice{'' if count == 1 else 's'} on the "
                             f"Realtime model")
            if self.longform_installed:
                count = len(self.samples)
                parts.append(f"{count} Voice Box sample{'' if count == 1 else 's'} on the 7B")
            return "Ready — " + " and ".join(parts) + "."
        if not self.supported and self.runtime_message:
            return self.runtime_message
        if not self.runtime_installed:
            return (f"Setup required — VibeVoice's runtime is not installed. Install it in "
                    f"the {TAB} tab.")
        if not (self.realtime_installed or self.longform_installed):
            return (f"Setup required — no VibeVoice model is installed. Install the Realtime "
                    f"0.5B for its preset voices, or the 7B to speak from samples, in the "
                    f"{TAB} tab.")
        if self.realtime_installed and not self.presets and not self.longform_installed:
            return (f"Setup required — the Realtime model is installed without its preset "
                    f"voices. Install it again in the {TAB} tab.")
        return (f"Setup required — VibeVoice 7B speaks from Voice Box samples and there are "
                f"none yet. Add one in the {TAB} tab, or make one from a recording in "
                f"Settings → Voice Chat.")


MODEL_FIELDS = ("id", "label", "kind", "installed", "runtime_installed", "precisions",
                "lora", "max_speakers", "voices", "defaults", "need_vram_bytes",
                "need_ram_bytes", "message", "download_bytes")
"""What of the installer's model record a page may see. Named rather than
filtered, so a field the installer adds later is absent until it is added here
on purpose (section 56)."""


def status() -> Status:
    """Read from the installer and the Voice Box's library. Starts nothing, never raises."""
    try:
        return _status()
    except Exception:
        logger.debug("Model Chain: could not read VibeVoice's installation", exc_info=True)
        return Status(runtime_message="VibeVoice's installation could not be read.")


def _status() -> Status:
    vibevoice = _vibevoice()
    found = Status()
    try:
        base = vibevoice.status()
        found.supported = bool(getattr(base, "supported", False))
        found.runtime_installed = bool(getattr(base, "runtime_installed", False))
        found.runtime_message = str(getattr(base, "runtime_message", "") or "")
    except Exception:
        logger.debug("Model Chain: the VibeVoice installer did not report", exc_info=True)
    try:
        infos = [dict(item) for item in (vibevoice.models_info() or [])
                 if isinstance(item, dict)]
    except Exception:
        logger.debug("Model Chain: the VibeVoice installer did not describe its models",
                     exc_info=True)
        infos = []
    for info in infos:
        identifier = str(info.get("id") or "")
        installed = bool(info.get("installed"))
        if info.get("runtime_installed"):
            found.runtime_installed = True
        if identifier == MODEL_REALTIME or (identifier != MODEL_7B
                                            and info.get("kind") == "realtime"):
            found.realtime_installed = found.realtime_installed or installed
            presets = info.get("presets")
            if not presets:
                try:
                    presets = vibevoice.presets(identifier or MODEL_REALTIME)
                except Exception:
                    presets = []
            found.presets = [dict(item) for item in (presets or ())
                             if isinstance(item, dict) and item.get("installed")
                             and str(item.get("id") or "")]
        elif identifier == MODEL_7B or info.get("kind") == "longform":
            found.longform_installed = found.longform_installed or installed
        found.models.append({name: info.get(name) for name in MODEL_FIELDS if name in info})
    try:
        found.samples = list(box.samples())
    except Exception:
        logger.debug("Model Chain: the Voice Box's samples could not be read", exc_info=True)
    return found


# --------------------------------------------------------------------------- #
# The voices
# --------------------------------------------------------------------------- #


def _created(value) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(value)))
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def _preset_entry(preset: dict, found: Status) -> dict:
    """One preset voice as the facade and the page see it, plus its private handle."""
    stem = str(preset.get("id") or "")
    name = str(preset.get("name") or stem)
    language = str(preset.get("language") or "")
    language_label = str(preset.get("language_label") or language or "")
    experimental = bool(preset.get("experimental", language not in ("", "en")))
    gender = str(preset.get("gender") or "")
    detail = ", ".join(part for part in (language_label, gender) if part)
    compatible = bool(found.runtime_installed and found.realtime_installed)
    entry = {
        "id": f"{PRESET_PREFIX}{stem}",
        "display_name": name,
        "label": f"{name} ({detail})" if detail else name,
        "type": "preset",
        "engine": ENGINE,
        "official": True,
        "editable": False,
        "deletable": False,
        "language": language,
        "accent": language_label + (" — experimental" if experimental else ""),
        "source_seconds": 0.0,
        "created_at": "",
        "compatible": compatible,
        "has_source": False,
        "_handle": {"model": MODEL_REALTIME, "preset": stem},
    }
    if not compatible:
        missing = "VibeVoice's runtime" if not found.runtime_installed else "that model"
        entry["_unprepared"] = (f"{name} is one of the Realtime model's preset voices, and "
                                f"{missing} is not installed. Install it in the {TAB} tab.")
    return entry


def _sample_entry(sample: dict, found: Status) -> dict:
    """One Voice Box sample as a Voice Chat voice, plus its private handle.

    ``has_source`` is False on purpose: the page offers Rebuild for a voice
    that is incompatible *and* has a retained recording, and this engine has
    nothing to rebuild -- the 7B reads the sample itself on every render.
    """
    identifier = str(sample.get("id") or "")
    title = str(sample.get("title") or "Sample")
    compatible = bool(found.runtime_installed and found.longform_installed)
    entry = {
        "id": f"{SAMPLE_PREFIX}{identifier}",
        "display_name": title,
        "label": title,
        "type": "sample",
        "engine": ENGINE,
        "official": False,
        "editable": True,
        "deletable": True,
        "language": "",
        "accent": "",
        "source_seconds": round(float(sample.get("seconds") or 0.0), 1),
        "created_at": _created(sample.get("created")),
        "compatible": compatible,
        "has_source": False,
        "_handle": {"model": MODEL_7B, "sample": identifier},
    }
    if not compatible:
        missing = "VibeVoice's runtime" if not found.runtime_installed else "VibeVoice 7B"
        entry["_unprepared"] = (f"{title} is a Voice Box sample, and speaking one needs "
                                f"{missing}, which is not installed. Install it in the "
                                f"{TAB} tab.")
    return entry


def _entries(found: Status) -> list:
    """Installed presets, English first and the experimental languages after, then samples."""
    english = [item for item in found.presets
               if str(item.get("language") or "") == "en"]
    others = sorted((item for item in found.presets
                     if str(item.get("language") or "") != "en"),
                    key=lambda item: str(item.get("language_label")
                                         or item.get("language") or ""))
    return ([_preset_entry(item, found) for item in english + others]
            + [_sample_entry(item, found) for item in found.samples
               if str(item.get("id") or "")])


def entries() -> list:
    """Every VibeVoice voice this installation can name. Reads; starts nothing."""
    return _entries(status())


def _find(found: list, voice_id: str):
    if not voice_id or not engines.belongs(voice_id, ENGINE):
        return None
    wanted = engines.native(voice_id)
    for entry in found:
        if engines.native(entry["id"]) == wanted:
            return entry
    return None


def lookup(voice_id: str):
    return _find(entries(), voice_id)


def _pick_default(found: list) -> str:
    """The stored default when it still names a voice, else a sensible first voice.

    Never another engine's voice (I-2): an installation with no VibeVoice voice
    answers with nothing, and the surface says why.
    """
    known = {entry["id"] for entry in found}
    stored = str(_stored((DEFAULT_KEY,)).get(DEFAULT_KEY) or "").strip()
    if stored in known:
        return stored
    preferred = f"{PRESET_PREFIX}{PREFERRED_PRESET}"
    if preferred in known:
        return preferred
    for kind in ("preset", "sample"):
        for entry in found:
            if entry["type"] == kind:
                return entry["id"]
    return ""


def default_id() -> str:
    """The VibeVoice default voice, qualified, or ``""`` when there is none."""
    return _pick_default(entries())


def default_entry():
    found = entries()
    return _find(found, _pick_default(found))


def set_default(voice_id: str) -> dict:
    """Make ``voice_id`` the default, now, in Voice Chat's own VibeVoice file."""
    entry = lookup(voice_id)
    if entry is None:
        raise VibeVoiceChatError("That is not a VibeVoice voice.")
    try:
        _store({DEFAULT_KEY: entry["id"]})
    except OSError:
        raise VibeVoiceChatError("The default voice could not be saved.") from None
    logger.info("Model Chain: the VibeVoice default voice is now %s", entry["id"])
    return entry


def resolve(voice_id: str = "") -> tuple:
    """``(qualified id, entry)`` for a voice, or for the default when it is not one.

    A voice that is known and cannot speak right now -- its model is not
    installed -- is still that voice: it comes back with ``compatible`` False
    and a sentence under ``_unprepared``, and the caller refuses with the
    sentence rather than quietly speaking another voice.
    """
    found = entries()
    entry = _find(found, voice_id) if voice_id else None
    if entry is None:
        entry = _find(found, _pick_default(found))
    if entry is None:
        raise VibeVoiceChatError(NO_VOICE)
    return entry["id"], entry


def check_name(name: str) -> str:
    """A display name, or a refusal that says what is wrong with this one."""
    text = " ".join(str(name or "").split())
    if not text:
        raise VibeVoiceChatError("Give the voice a name.")
    if len(text) > MAX_NAME_CHARS:
        raise VibeVoiceChatError(f"That name is longer than {MAX_NAME_CHARS} characters.")
    return text


def rename(voice_id: str, display_name: str) -> dict:
    """Rename a sample voice, which renames the Voice Box sample. Presets keep theirs."""
    entry = lookup(voice_id)
    if entry is None:
        raise VibeVoiceChatError("That is not a VibeVoice voice.")
    if entry["type"] != "sample":
        raise VibeVoiceChatError("A preset voice ships with the Realtime model and keeps its "
                                 "own name.")
    name = check_name(display_name)
    try:
        box.rename_sample(entry["_handle"]["sample"], name)
    except box.VoiceBoxError as exc:
        raise VibeVoiceChatError(str(exc)) from None
    logger.info("Model Chain: a VibeVoice voice (a Voice Box sample) was renamed")
    return lookup(voice_id)


def delete(voice_id: str) -> dict:
    """Delete a sample voice -- the Voice Box sample itself. Presets are refused.

    Exactly what the Voice Box's own Delete does, so a configuration there that
    used the sample loses that speaker. A preset ships with the Realtime model
    and is not this engine's to remove.
    """
    entry = lookup(voice_id)
    if entry is None:
        raise VibeVoiceChatError("That is not a VibeVoice voice.")
    if entry["type"] != "sample":
        raise VibeVoiceChatError("A preset voice ships with the Realtime model, so it cannot "
                                 "be deleted. Only Voice Box samples can.")
    try:
        box.delete_sample(entry["_handle"]["sample"])
    except box.VoiceBoxError as exc:
        raise VibeVoiceChatError(str(exc)) from None
    # A default that named it is left as it is: the Voice Box's own Delete
    # leaves it just the same, and a default that names no voice any more is
    # read as the first sensible one (:func:`_pick_default`).
    logger.info("Model Chain: a VibeVoice voice (a Voice Box sample) was deleted")
    return entry


def capacity() -> dict:
    """How many samples exist. There is no bank, and so no limit."""
    return {"used": len(status().samples), "total": None, "free": None}


def warnings() -> list:
    """What is wrong that somebody can see and act on. Never repairs anything."""
    found = status()
    out = []
    if not found.runtime_installed and (found.samples or found.realtime_installed
                                        or found.longform_installed):
        out.append(f"VibeVoice's runtime is not installed, so none of these voices can speak "
                   f"yet. Install it in the {TAB} tab.")
    if found.realtime_installed and not found.presets:
        out.append(f"The Realtime model is installed but its preset voices are not, so it has "
                   f"nothing to speak with. Install it again in the {TAB} tab.")
    if found.longform_installed and not found.samples:
        out.append(f"VibeVoice 7B is installed and the {TAB} has no samples yet, so there is "
                   f"no voice of your own to speak with. Add one in the {TAB} tab, or make "
                   f"one from a recording in the VibeVoice panel.")
    if found.samples and not found.longform_installed:
        out.append(f"The {TAB} samples are listed here, and speaking one needs VibeVoice 7B, "
                   f"which is not installed. Install it in the {TAB} tab.")
    if not found.presets and not found.samples:
        out.append(NO_VOICE)
    return out


def test_text() -> str:
    """What Test says. Shared with the other engines, as Pocket's is: it is a
    property of the control rather than of a model."""
    import mc_voice_registry as registry

    try:
        return registry.test_text() or DEFAULT_TEST_TEXT
    except Exception:
        return DEFAULT_TEST_TEXT


def set_test_text(text: str) -> str:
    import mc_voice_registry as registry

    return registry.set_test_text(text)


# --------------------------------------------------------------------------- #
# What the facade asks this engine about itself
# --------------------------------------------------------------------------- #


def capabilities() -> dict:
    """What VibeVoice can do, as behaviour. Section 8.

    ``clone_preview`` because the 7B clones from a recording; ``rebuild`` not,
    because nothing is prepared per voice that could go stale. Stop cancels:
    the worker checks a stop flag at every generation step.
    """
    mode = "cancel"
    try:
        mode = str(_speech().declared_interrupt_mode() or mode)
    except Exception:
        logger.debug("Model Chain: could not read VibeVoice's interrupt mode", exc_info=True)
    return {"clone_preview": True, "rebuild": False, "engine_settings": True,
            "starter_voices": False, "voice_lab": False, "interrupt_mode": mode}


def refusals() -> tuple:
    """The exception types this engine raises to *refuse* rather than fail."""
    found = [VibeVoiceChatError]
    try:
        found.append(_speech().VibeVoiceSpeechError)
    except Exception:
        logger.debug("Model Chain: VibeVoice's speech refusals could not be read",
                     exc_info=True)
    try:
        found.append(_vibevoice().VibeVoiceError)
    except Exception:
        logger.debug("Model Chain: VibeVoice's installer refusals could not be read",
                     exc_info=True)
    found.append(box.VoiceBoxError)
    return tuple(found)


def clone_hints() -> dict:
    """The clone form's window: the Voice Box's sample bounds, and ten seconds."""
    return {"min_seconds": float(box.SAMPLE_MIN_SECONDS),
            "ideal_seconds": IDEAL_REFERENCE_SECONDS,
            "max_seconds": float(box.SAMPLE_MAX_SECONDS)}


def engine_settings() -> dict:
    """The card, precision and LoRA a reply uses, and what each may be.

    Global to VibeVoice in Voice Chat, not per character: each one changes
    which model identity is loaded on which card, and a character that quietly
    moved the model between cards would be a character nobody could reason
    about. Precision and LoRA are the 7B's -- they apply to voices made from
    samples; the Realtime model always runs at full precision with none.
    """
    speech = _speech()
    chat = speech.chat_settings()
    card, source = speech.effective_card(chat)
    vibevoice = _vibevoice()
    labels = dict(getattr(vibevoice, "PRECISION_LABELS", {}) or {})
    try:
        needs = dict((vibevoice.model_info(MODEL_7B) or {}).get("need_vram_bytes") or {})
    except Exception:
        needs = {}
    precisions = [{"id": name, "label": str(labels.get(name) or name),
                   "need_vram_bytes": int(needs.get(name) or 0)}
                  for name in (getattr(vibevoice, "PRECISIONS", None) or ("bf16",))]
    return {
        "card_uuid": chat["card_uuid"],
        "precision": chat["precision"],
        "lora_id": chat["lora_id"],
        "lora_scale": chat["lora_scale"],
        "card_effective": card,
        "card_source": source,
        "precisions": precisions,
    }


def apply_engine_settings(values: dict = None) -> dict:
    """Change the card, precision, LoRA or strength, in the names the page sends.

    Validated by the installer, which owns ``settings()["chat"]``; a value it
    will not take comes back as its own sentence. An unknown name is refused
    here rather than dropped: a page with a control this build does not have is
    a stale surface, and answering it with silence would be answering "applied".

    Nothing is stopped. The next reply asks for its card and loads the model
    identity these settings name; the reply being spoken keeps the one it began
    with.
    """
    offered = {str(key): value for key, value in dict(values or {}).items()}
    unknown = sorted(set(offered) - set(CHAT_SETTINGS))
    if unknown:
        raise VibeVoiceChatError(f"{unknown[0]!r} is not a VibeVoice engine setting.")
    if not offered:
        return engine_settings()
    if "lora_scale" in offered:
        try:
            offered["lora_scale"] = float(offered["lora_scale"])
        except (TypeError, ValueError):
            raise VibeVoiceChatError("The LoRA's strength is a number from 0 to 2.") from None
    for name in ("card_uuid", "precision", "lora_id"):
        if name in offered:
            offered[name] = str(offered[name] or "").strip()
    merged = dict(_speech().chat_settings())
    merged.update(offered)
    _vibevoice().set_settings({"chat": merged})
    logger.info("Model Chain: VibeVoice's Voice Chat settings changed — %s",
                ", ".join(sorted(offered)))
    return engine_settings()


def cards() -> list:
    """The machine's cards, as the turn client lists them, with their roles in words."""
    client = box.turns()
    if client is None:
        return []
    try:
        found = client.cards() or []
    except Exception:
        logger.debug("Model Chain: the cards could not be listed", exc_info=True)
        return []
    rows = []
    for item in found:
        if not isinstance(item, dict) or not str(item.get("uuid") or ""):
            continue
        roles = []
        if item.get("image_card"):
            roles.append("the image model's card")
        if item.get("wangp_card"):
            roles.append("WanGP's card")
        rows.append({"uuid": str(item.get("uuid")),
                     "name": str(item.get("name") or f"GPU {item.get('index', '')}").strip(),
                     "index": item.get("index"),
                     "image_card": bool(item.get("image_card")),
                     "wangp_card": bool(item.get("wangp_card")),
                     "role": " and ".join(roles)})
    return rows


LORA_FIELDS = ("id", "name", "base", "bytes", "parts", "created")


def loras() -> list:
    """The installer's LoRA library, for the 7B, in the fields a page may see."""
    try:
        found = _vibevoice().loras() or []
    except Exception:
        logger.debug("Model Chain: VibeVoice's LoRA library could not be read", exc_info=True)
        return []
    return [{name: item.get(name) for name in LORA_FIELDS if name in item}
            for item in found if isinstance(item, dict) and item.get("id")]


def public_status() -> dict:
    """VibeVoice's operational state, in the shape every engine answers with.

    The common subset the status payload publishes, plus ``block``: the part
    only a page scoped to VibeVoice sees -- the models, the chat settings, the
    cards and LoRAs to choose from, the warnings, and whether a voice can be
    made from a recording here.
    """
    found = status()
    speech = _speech()
    try:
        live = speech.status() or {}
    except Exception:
        logger.debug("Model Chain: could not read VibeVoice's speech state", exc_info=True)
        live = {}
    try:
        state = speech.engine() or {}
    except Exception:
        logger.debug("Model Chain: could not read VibeVoice's residency", exc_info=True)
        state = {}
    try:
        chosen = engine_settings()
    except Exception:
        logger.debug("Model Chain: could not read VibeVoice's settings", exc_info=True)
        chosen = {}
    mode = str(live.get("interrupt_mode") or capabilities()["interrupt_mode"])
    return {
        "installed": found.ready,
        # Whether this engine *can* speak, never whether it is speaking: that
        # is ``engine_busy``, below (see mc_voice_pocket.public_status).
        "ready": found.ready,
        "tts_ready": found.ready,
        "message": found.message,
        "worker_resident": bool(state.get("loaded")),
        "engine_busy": bool(live.get("busy")),
        "draining": False,
        "interrupt_mode": mode,
        "block": {
            "installed": found.ready,
            "supported": found.supported,
            "runtime_installed": found.runtime_installed,
            "runtime_message": found.runtime_message,
            "message": found.message,
            "tab": TAB,
            "models": found.models,
            "presets_installed": len(found.presets),
            "samples": len(found.samples),
            "cloning_ready": found.cloning_ready,
            "cloning_message": found.cloning_message,
            "settings": chosen,
            "cards": cards(),
            "loras": loras(),
            "warnings": warnings(),
            "busy": bool(live.get("busy")),
            "draining": False,
            "interrupt_mode": mode,
            "state": state,
        },
    }


# --------------------------------------------------------------------------- #
# Making a voice from a recording
# --------------------------------------------------------------------------- #


_preview_lock = threading.Lock()
_preview: dict = {}
"""The one voice that has been auditioned but not kept.

In this process's memory and nowhere else: the recording, normalised, and the
name somebody gave it. Nothing is written to disk before Save, so Discard, a
second Create and a WebUI that closes all end it the same way -- by forgetting
it. One at a time, as Pocket's is: a second preview replaces the first.
"""


def preview_state() -> dict:
    """What is pending, without the audio. Safe to log and to poll."""
    with _preview_lock:
        if not _preview:
            return {"pending": False}
        return {"pending": True, "name": _preview.get("name", ""),
                "seconds": _preview.get("seconds", 0.0)}


def discard_preview(token: str = "") -> bool:
    """Forget the pending preview. An empty token forgets whatever is pending;
    a non-empty one has to match, so a stale tab cannot discard a newer preview."""
    with _preview_lock:
        if not _preview:
            return False
        if token and token != _preview.get("token"):
            return False
        name = _preview.get("name") or ""
        _preview.clear()
    logger.info("Model Chain: a VibeVoice voice preview was discarded — %s", name or "?")
    return True


def prepare_preview(display_name: str, wav: bytes) -> dict:
    """Audition a voice made from a recording, without keeping it. Section 26.2.

    In order, and every step can refuse without leaving anything behind:

        1  the name is checked;
        2  cloning must be possible here -- the runtime and the 7B installed;
        3  whatever was pending is forgotten, so a preview that fails halfway
           cannot leave the previous one's Save button live;
        4  the recording is normalised to the Voice Box's own sample form
           (3 to 60 seconds, mono, 24 kHz) and refused in its sentences;
        5  the 7B speaks the test line in that voice, through a card turn at
           the chat settings' precision and LoRA -- so what is heard is what
           Voice Chat would speak;
        6  only then is it pending, in memory.

    :func:`save_preview` is the only step that writes anything.
    """
    name = check_name(display_name)
    found = status()
    if not found.cloning_ready:
        raise VibeVoiceChatError(found.cloning_message)
    discard_preview()
    canonical, seconds = reference.normalize(bytes(wav or b""), box.ENVELOPE,
                                             error=VibeVoiceChatError)
    pcm16, _rate = box.pcm_of(canonical)
    audio = _speech().synthesize(test_text(), {"model": MODEL_7B, "reference": pcm16}, None)
    token = uuid.uuid4().hex
    with _preview_lock:
        _preview.clear()
        _preview.update({"token": token, "name": name, "wav": canonical,
                         "seconds": round(float(seconds), 2)})
    logger.info("Model Chain: a VibeVoice voice was auditioned from a %.1f s recording; it "
                "is not saved yet", seconds)
    return {"token": token, "name": name, "seconds": round(float(seconds), 2),
            "audio": audio}


def save_preview(token: str) -> dict:
    """Keep the pending preview as a Voice Box sample. The only step that writes.

    The token has to match, so a tab sitting on an old preview cannot save a
    voice somebody has since replaced. The record is cleared only after the
    sample exists: a Save that failed leaves the preview pending, to be saved
    again or discarded. Not made the default -- saving a voice is not the same
    decision as speaking with it.
    """
    with _preview_lock:
        if not _preview:
            raise VibeVoiceChatError("There is no voice waiting to be saved. Create one first.")
        if str(token or "") != _preview.get("token"):
            raise VibeVoiceChatError("That preview is no longer the one waiting to be saved. "
                                     "Create the voice again.")
        pending = dict(_preview)
        made = box.add_sample(pending["wav"], title=pending["name"], source="voice-chat")
        _preview.clear()
    logger.info("Model Chain: a VibeVoice voice was saved as a Voice Box sample — %.1f s",
                float(pending.get("seconds") or 0.0))
    voice = lookup(f"{SAMPLE_PREFIX}{made['id']}")
    if voice is None:
        voice = _sample_entry(made, status())
    return {"voice": voice}
