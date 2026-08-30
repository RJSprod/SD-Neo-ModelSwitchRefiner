"""How a Sopro voice is delivered, and which of it is Sopro's and which is ours.

The Sopro half of :mod:`mc_voice_profile`. Same shape, same inheritance-by-
absence, same clamping -- and deliberately a *separate module* with separate
storage, because common labels do not imply shared implementation (section 35).
A user may intentionally have

    Kokoro character speed = 1.10
    Sopro  character speed = 0.95

and switching the global engine restores the appropriate one.

What Sopro actually offers, stated plainly
-------------------------------------------
This matters more than it usually would, because the obvious expectation of a
modern TTS model -- a dial marked "emotion" -- is one Sopro V2 does not have,
and the one it looks like it has is not that.

``SoproTTS.stream`` takes ``lang``, ``temperature``, ``top_p``, ``top_k``,
``steps``, ``max_seconds``, ``min_seconds`` and ``chunk_frames``. There is no
speaking-rate argument, no pitch argument, no gain argument and no named emotion
input. So of the eight controls below, four are Sopro's own and four are ours:

    speed        Ours. Sopro has no rate control, so this is a pitch-preserving
                 time-scale on the model's output -- streaming SOLA, state
                 carried across chunks. Section 32 is explicit that a naive
                 resample that transposes the voice is not parity and does not
                 ship, and ``tests/test_voice_sopro_worker.py`` measures the
                 fundamental at every speed to prove it is not what happens.

    pitch        Ours. Resampling composed *after* the time-scale, so changing
                 speed at neutral pitch does not transpose and changing pitch
                 does not change the requested duration. Formants move with it,
                 which reads as a differently-sized speaker rather than as a
                 transposition -- the same honest description Kokoro's pitch
                 gets.

    volume       Ours. A scalar with a soft limiter, applied where the PCM is
                 built.

    pause        Ours. Silence between committed speech units.

    language     Sopro's. A pronunciation/language tag, not translation.

    temperature  Sopro's. Sampling variation, not warmth, not emotion.
    top_p        Sopro's. Advanced.
    top_k        Sopro's. Advanced.

The three sampling controls are named for what they are. Calling temperature
"Warmth" would be inventing a semantic the upstream model does not promise, and
section 37 says the help text is part of correctness.

What is deliberately not here
-----------------------------
Precision, solver steps, streaming chunk size, thread policy and model revision.
Those change compute and warm-cache compatibility for the whole worker rather
than how one character sounds, so they are global Sopro engine settings in
:mod:`mc_voice_sopro` (section 13, section 34). Seed is not here either: fixed
seeds belong to the Voice Lab's A/B comparisons, not to a character profile.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

OPT_SPEED = "model_chain_voice_sopro_speed"
OPT_PITCH = "model_chain_voice_sopro_pitch"
OPT_GAIN = "model_chain_voice_sopro_gain"
OPT_PAUSE = "model_chain_voice_sopro_pause"
OPT_LANGUAGE = "model_chain_voice_sopro_language"
OPT_TEMPERATURE = "model_chain_voice_sopro_temperature"
OPT_TOP_P = "model_chain_voice_sopro_top_p"
OPT_TOP_K = "model_chain_voice_sopro_top_k"

DELIVERY_FIELDS = ("speed", "pitch", "gain", "pause")
GENERATION_FIELDS = ("temperature", "top_p", "top_k")

FIELDS = DELIVERY_FIELDS + GENERATION_FIELDS
"""Every numeric field a character may override, in display order.

``language`` is not in here because it is a choice rather than a number and is
stored and resolved separately -- a slider list that had to special-case one
entry would be a list every caller has to remember to special-case too.
"""

OPTIONS = {"speed": OPT_SPEED, "pitch": OPT_PITCH, "gain": OPT_GAIN, "pause": OPT_PAUSE,
           "temperature": OPT_TEMPERATURE, "top_p": OPT_TOP_P, "top_k": OPT_TOP_K}

MODEL_DEFAULT = None
"""What a field holds when it should follow the pinned model's own value.

``None`` in a *stored default* means something different from ``None`` in a
character override, and both are correct. In a character it means "follow the
Sopro default"; in the Sopro default it means "follow the model configuration",
which is read from the running worker's handshake rather than repeated here. A
model revision that changed its temperature would otherwise not change anybody's
voice, because today's number would have been frozen into config.json.
"""

CONTROLS = {
    "speed": {
        "label": "Speed",
        "unit": "x",
        "minimum": 0.5,
        "maximum": 2.0,
        "step": 0.05,
        "default": 1.0,
        "decimals": 2,
        "group": "delivery",
        "owner": "voice-chat",
        "help": "How fast the voice speaks. Sopro has no speaking-rate input of its own, so "
                "Voice Chat time-scales the audio it produces without changing the pitch. "
                "Below about 0.7 and above about 1.5 the processing starts to be audible.",
    },
    "pitch": {
        "label": "Pitch",
        "unit": " semitones",
        "minimum": -12.0,
        "maximum": 12.0,
        "step": 0.5,
        "default": 0.0,
        "decimals": 1,
        "group": "delivery",
        "owner": "voice-chat",
        "help": "Shifts the whole voice, formants included, so it reads as a different-sized "
                "speaker rather than as a transposition. Independent of Speed: changing one "
                "does not change the other.",
    },
    "gain": {
        "label": "Volume",
        "unit": " dB",
        "minimum": -12.0,
        "maximum": 12.0,
        "step": 0.5,
        "default": 0.0,
        "decimals": 1,
        "group": "delivery",
        "owner": "voice-chat",
        "help": "Loudness relative to Sopro's own output, limited so a loud setting cannot "
                "clip into distortion.",
    },
    "pause": {
        "label": "Pause between sentences",
        "unit": " ms",
        "minimum": 0.0,
        "maximum": 1200.0,
        "step": 25.0,
        "default": 0.0,
        "decimals": 0,
        "group": "delivery",
        "owner": "voice-chat",
        "help": "Extra silence after each committed sentence, on top of Sopro's own. A "
                "measured character gets 200-400 ms; an urgent one gets none.",
    },
    "temperature": {
        "label": "Variation",
        "unit": "",
        "minimum": 0.1,
        "maximum": 1.5,
        "step": 0.05,
        "default": None,
        "decimals": 2,
        "group": "generation",
        "owner": "sopro",
        "advanced": False,
        "help": "Sopro's sampling temperature. Higher is more varied between takes and "
                "eventually less stable; lower is flatter and more repeatable. It is not "
                "an emotion or warmth control — the model has no such input. It also "
                "trades against how much the output sounds like the voice you cloned: "
                "the reference conditions the sampling, and turning this up lets each "
                "take wander further from it, so if a clone is not recognisable enough, "
                "lower this before anything else. Left alone it follows the model's own "
                "default.",
    },
    "top_p": {
        "label": "Top-p",
        "unit": "",
        "minimum": 0.1,
        "maximum": 1.0,
        "step": 0.05,
        "default": None,
        "decimals": 2,
        "group": "generation",
        "owner": "sopro",
        "advanced": True,
        "help": "Sopro's nucleus sampling cutoff. Most people should leave this alone; it is "
                "here because reproducing a particular run needs it.",
    },
    "top_k": {
        "label": "Top-k",
        "unit": "",
        "minimum": 1,
        "maximum": 200,
        "step": 1,
        "default": None,
        "decimals": 0,
        "group": "generation",
        "owner": "sopro",
        "advanced": True,
        "help": "Sopro's top-k sampling cutoff. As with top-p, this exists for "
                "reproducibility rather than for tuning a voice.",
    },
}

DEFAULTS = {name: CONTROLS[name]["default"] for name in FIELDS}
"""The neutral profile: Sopro's own delivery and the model's own sampling."""


class SoproProfileError(ValueError):
    """A delivery value that could not be used. Never fatal."""


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #


def _number(value, fallback):
    """``value`` as a float, or ``fallback`` for anything that is not one.

    Total, because every caller is either a settings store that may hold
    anything, a character file somebody edited by hand, or a JSON body from a
    browser. None of those is a reason for a reply to go unspoken.
    """
    if value is None or isinstance(value, bool):
        return fallback
    try:
        found = float(value)
    except (TypeError, ValueError):
        return fallback
    if found != found or found in (float("inf"), float("-inf")):
        return fallback
    return found


def clamp(values=None, **extra) -> dict:
    """A complete, in-range profile from whatever was offered.

    Never raises and never returns a partial dictionary. A missing delivery
    field takes the neutral default; a missing *generation* field stays ``None``,
    because "follow the model's own value" is a real state and substituting
    today's number for it would freeze it (see :data:`MODEL_DEFAULT`).
    """
    found = dict(values or {})
    found.update(extra)
    made = {}
    for name in FIELDS:
        spec = CONTROLS[name]
        offered = found.get(name)
        if spec["default"] is None and offered is None:
            made[name] = None
            continue
        wanted = _number(offered, spec["default"])
        if wanted is None:
            made[name] = None
            continue
        wanted = min(max(wanted, spec["minimum"]), spec["maximum"])
        made[name] = round(wanted, spec["decimals"]) if spec["decimals"] else int(wanted)
    made["language"] = _language(found.get("language"))
    return made


def _language_choices() -> tuple:
    """Sopro's language tags, read from the module that owns them.

    Read rather than repeated so that a model revision which added a language
    changes the settings row, the clone form and the character editor at once.
    Falls back to the empty hint alone if the Sopro module cannot be read, which
    is the honest answer on an installation where Sopro is not present at all.
    """
    try:
        import mc_voice_sopro as sopro

        return tuple(sopro.LANGUAGES)
    except Exception:
        return (("", "Auto"),)


def _language(value) -> str:
    wanted = str(value or "").strip().lower()
    return wanted if wanted in {item for item, _label in _language_choices()} else ""


def overrides(values=None, **extra) -> dict:
    """The same, but keeping ``None`` as ``None``.

    What a character stores. A field nobody has set stays unset, which is what
    makes it follow the Sopro default's profile rather than freezing today's
    value into the character file (I-4).
    """
    found = dict(values or {})
    found.update(extra)
    made = {}
    for name in FIELDS:
        offered = found.get(name)
        if offered is None or (isinstance(offered, str) and not offered.strip()):
            made[name] = None
            continue
        spec = CONTROLS[name]
        wanted = _number(offered, None)
        if wanted is None:
            made[name] = None
            continue
        wanted = min(max(wanted, spec["minimum"]), spec["maximum"])
        made[name] = round(wanted, spec["decimals"]) if spec["decimals"] else int(wanted)
    offered = found.get("language")
    made["language"] = (None if offered is None or (isinstance(offered, str)
                                                    and not offered.strip())
                        else _language(offered))
    return made


def neutral(profile=None) -> bool:
    """Whether this profile asks for nothing at all.

    Read to leave the shaping out of the request entirely, which is what keeps
    an installation that never touches these controls on the cheapest path
    through the worker's DSP.
    """
    found = clamp(profile)
    return (all(found[name] == CONTROLS[name]["default"] for name in DELIVERY_FIELDS)
            and all(found[name] is None for name in GENERATION_FIELDS)
            and not found["language"])


def value_label(name: str, value) -> str:
    """One control's value as it is written on screen, unit and sign included.

    Here rather than in each surface, because three of them draw it and a plus
    sign that appeared in two of the three would read as a different setting.
    """
    spec = CONTROLS.get(name)
    if spec is None:
        return str(value)
    if value is None:
        return "model default"
    found = _number(value, spec["default"] if spec["default"] is not None else 0.0)
    text = f"{found:.{spec['decimals']}f}"
    if spec["decimals"]:
        text = text.rstrip("0").rstrip(".") or "0"
    sign = "+" if name in ("pitch", "gain") and found > 0 else ""
    return f"{sign}{text}{spec['unit']}"


def describe(profile=None) -> str:
    """One line naming only what has been changed, or "Sopro's own delivery"."""
    found = clamp(profile)
    parts = []
    for name in FIELDS:
        spec = CONTROLS[name]
        value = found[name]
        if value == spec["default"]:
            continue
        parts.append(f"{spec['label'].casefold()} {value_label(name, value)}")
    if found.get("language"):
        import mc_voice_sopro as sopro

        parts.append(sopro._language_label(found["language"]).casefold())
    return ", ".join(parts) if parts else "Sopro's own delivery"


# --------------------------------------------------------------------------- #
# The default voice's profile
# --------------------------------------------------------------------------- #


def stored() -> dict:
    """The profile the Sopro default voice speaks with, from Sopro's own file.

    Not from the host's options, and that is a correction rather than a
    preference: an option is a component on the settings page as well as a
    stored value, so "Apply settings" wrote the page's build-time copy back over
    whatever this panel had just set. Two controls for one value, each able to
    overwrite the other. See :func:`mc_voice_sopro._setting`, which still reads
    the option when the file has nothing, so nobody loses what they configured.
    """
    import mc_voice_sopro as sopro

    found = {}
    try:
        for name, option in OPTIONS.items():
            found[name] = sopro._setting(option)
        found["language"] = sopro._setting(OPT_LANGUAGE)
    except Exception:
        return dict(DEFAULTS, language="")
    return clamp(found)


def remember(values=None, **extra) -> dict:
    """Write the Sopro default profile through to the host's store, and save it.

    Immediate, because this is an operational surface: somebody who slows the
    voice down while listening to it expects the next sentence to be slower, not
    to be told to press Apply.

    It writes only Sopro's own options. Kokoro's four remain untouched whatever
    happens here (I-3), which is enforced by the option names rather than by a
    check that could be forgotten.
    """
    import mc_voice_sopro as sopro

    wanted = clamp(values, **extra)
    try:
        for name, option in OPTIONS.items():
            value = wanted[name]
            # A generation field that is ``None`` is stored as an empty string,
            # which is what "follow the model" survives a round trip as.
            sopro._remember(option, "" if value is None else value)
        sopro._remember(OPT_LANGUAGE, wanted.get("language") or "")
    except Exception:
        logger.debug("Model Chain: could not persist the Sopro delivery profile",
                     exc_info=True)
    return stored()


def resolve(values=None) -> dict:
    """One profile to speak with: character overrides on top of the Sopro default.

    ``None`` in a character field means "follow the default", which is how a
    character that has never been given a Sopro delivery follows Settings, and
    how a character given only a slower speed keeps the default's pitch.
    """
    base = stored()
    offered = overrides(values)
    found = {name: (base[name] if offered[name] is None else offered[name])
             for name in FIELDS}
    found["language"] = (base.get("language") or "") if offered.get("language") is None \
        else offered["language"]
    return clamp(found)


# --------------------------------------------------------------------------- #
# What the worker is told
# --------------------------------------------------------------------------- #


def pitch_ratio(semitones) -> float:
    """A pitch shift in semitones as the frequency ratio the worker resamples by.

    Twelve semitones is an octave, so the ratio is ``2 ** (n / 12)``. Clamped
    again as a belt: this number divides the requested speed before it reaches
    the time-scaler, and a runaway value would ask for a stretch nothing can do.
    """
    found = _number(semitones, 0.0)
    found = min(max(found, CONTROLS["pitch"]["minimum"]), CONTROLS["pitch"]["maximum"])
    return float(2.0 ** (found / 12.0))


def gain_scale(decibels) -> float:
    """A volume in dB as the linear scalar the worker multiplies samples by."""
    found = _number(decibels, 0.0)
    found = min(max(found, CONTROLS["gain"]["minimum"]), CONTROLS["gain"]["maximum"])
    return float(10.0 ** (found / 20.0))


def request(profile=None) -> dict:
    """The profile as the worker's frame header carries it.

    Converted here rather than in the worker so that "what does +3 semitones
    mean" is answered once, in the process that has the settings, and the worker
    stays a thing that stretches, resamples and scales what it is told to.

    A generation field that is ``None`` is *omitted* rather than sent as null.
    The worker's :class:`Delivery` treats an absent key as "the model's own",
    which is the one representation of that state -- sending a number would make
    the model's default a value this process had to know.
    """
    found = clamp(profile)
    made = {
        "speed": float(found["speed"]),
        "pitch": pitch_ratio(found["pitch"]),
        "gain": gain_scale(found["gain"]),
        "pause_ms": int(found["pause"]),
    }
    if found.get("language"):
        made["language"] = found["language"]
    for name in GENERATION_FIELDS:
        if found[name] is not None:
            made[name] = found[name]
    return made
