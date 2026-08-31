"""How a PocketTTS voice is delivered, and which of it is Pocket's and which is ours.

The Pocket half of :mod:`mc_voice_profile` and :mod:`mc_voice_sopro_profile`.
Same shape, same inheritance-by-absence, same clamping -- and deliberately a
*separate module with separate storage*, because common labels do not imply
shared implementation (section 35, I-PKT-23). A user may intentionally have

    Kokoro character speed = 1.10
    Sopro  character speed = 0.95
    Pocket character speed = 1.05

and switching the global engine restores the appropriate one. Sharing an option
name between two of them would make the third engine's slider move the other's
voice, which is a bug nobody would look for in a profile module.

What Pocket actually offers, stated plainly
--------------------------------------------
The reviewed PocketTTS 3.0.2 generation API takes a model state, a text string
and sampling controls. There is **no speaking-rate argument**, no pitch
argument, no gain argument and no named emotion input (section 5.11). So of the
five controls below, four are ours and one is Pocket's:

    speed        Ours. Pocket has no rate control, so this is a pitch-preserving
                 time-scale on the model's output -- streaming SOLA, state
                 carried across the chunk boundaries Pocket streams at. Section
                 15 is explicit that a naive resample which transposes the voice
                 is not parity and does not ship, and
                 ``tests/test_voice_pocket_worker.py`` measures the fundamental
                 at every speed to prove it is not what happens.

    pitch        Ours. Resampling composed *after* the time-scale, so changing
                 speed at neutral pitch does not transpose and changing pitch
                 does not change the requested duration.

    volume       Ours. A scalar with a soft limiter, applied where the PCM is
                 built.

    pause        Ours. Silence between committed speech units.

    temperature  Pocket's. Sampling variation, and nothing else.

``temperature`` is labelled *Variation* for the reason section 14 gives at
length: higher values vary one take more and lower values are more repeatable,
and that is the whole of it. It is not an emotion, warmth, energy or identity
control. The model has no such input, and a slider claiming to be one would be
a product promise nobody has tested (I-PKT-26, section 29.5).

What is deliberately not here
-----------------------------
Precision, sampler decode steps, model/language revision and anything to do with
threads. Those change how the *runtime executes* rather than how one character
sounds, they are engine-global, and changing one stops the worker -- so a
character carrying them would be a character whose turn to speak restarted a
subprocess (I-PKT-23, section 13, section 49.11). They live in
:mod:`mc_voice_pocket`'s ``settings.json``.

Noise clamp, EOS threshold, frames-after-EOS, maximum tokens and a random seed
are not here either, and not because they are hard: section 16.5 says they may
appear in an experimental audition surface once repeatable tests give them a
user-facing purpose, and until then a slider for one would be a control whose
meaning nobody could state.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

OPT_SPEED = "model_chain_voice_pocket_speed"
OPT_PITCH = "model_chain_voice_pocket_pitch"
OPT_GAIN = "model_chain_voice_pocket_gain"
OPT_PAUSE = "model_chain_voice_pocket_pause"
OPT_TEMPERATURE = "model_chain_voice_pocket_temperature"

DELIVERY_FIELDS = ("speed", "pitch", "gain", "pause")
GENERATION_FIELDS = ("temperature",)

FIELDS = DELIVERY_FIELDS + GENERATION_FIELDS
"""Every field a character may override, in display order.

All five are numbers. Pocket has no per-character language: the model *is* the
language, the model is engine-global, and a character that could change it would
be a character whose turn reloaded the runtime and invalidated every warmed
voice state (section 13).
"""

OPTIONS = {"speed": OPT_SPEED, "pitch": OPT_PITCH, "gain": OPT_GAIN, "pause": OPT_PAUSE,
           "temperature": OPT_TEMPERATURE}
"""Field to stored name. Not one of them overlaps Kokoro's or Sopro's, which is
what makes I-PKT-3 a property of the spelling rather than a rule to remember."""

MODEL_DEFAULT = None
"""What a field holds when it should follow the selected model's own value.

``None`` in a *stored default* means something different from ``None`` in a
character override, and both are correct. In a character it means "follow the
Pocket default"; in the Pocket default it means "follow the model
configuration", which is read from the running worker's handshake rather than
repeated here. The English model currently recommends 0.3 and Pocket accepts
``temp=None`` to mean exactly that -- so freezing 0.3 into a settings file would
be freezing today's recommendation into every installation and calling it a
default (section 14, I-PKT-25).
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
        "help": "How fast the voice speaks. PocketTTS has no speaking-rate input of its "
                "own, so Voice Chat time-scales the audio it produces without changing the "
                "pitch. Below about 0.7 and above about 1.5 the processing starts to be "
                "audible.",
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
        "help": "Loudness relative to PocketTTS's own output, limited so a loud setting "
                "cannot clip into distortion.",
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
        "help": "Extra silence after each committed sentence, on top of PocketTTS's own. A "
                "measured character gets 200-400 ms; an urgent one gets none.",
    },
    "temperature": {
        "label": "Variation",
        "unit": "",
        "minimum": 0.1,
        "maximum": 1.2,
        "step": 0.05,
        "default": None,
        "decimals": 2,
        "group": "generation",
        "owner": "pocket",
        "advanced": False,
        "help": "PocketTTS's sampling temperature. Higher values vary one take more; lower "
                "values are more repeatable. It is not an emotion, warmth, energy or "
                "identity control — the model has no such input. It also trades against "
                "how much the output sounds like the voice you cloned: the reference "
                "conditions the sampling, and turning this up lets each take wander "
                "further from it, so if a clone is not recognisable enough, lower this "
                "before anything else. Left alone it follows the model's own default.",
    },
}
"""The five controls, key-for-key in the shape both other engines use.

The range on Variation is provisional and says so in the release notes: section
14 asks for a small sweep before a number is called shipped, and 0.1 to 1.2 is
an envelope that keeps the model's own recommended value reachable rather than a
measurement anybody has made (GATE P-4).
"""

DEFAULTS = {name: CONTROLS[name]["default"] for name in FIELDS}
"""The neutral profile: Pocket's own delivery and the model's own sampling."""


class PocketProfileError(ValueError):
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
    return made


def overrides(values=None, **extra) -> dict:
    """The same, but keeping ``None`` as ``None``.

    What a character stores. A field nobody has set stays unset, which is what
    makes it follow the Pocket default's profile rather than freezing today's
    value into the character file (I-4, I-PKT-4).
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
    return made


def neutral(profile=None) -> bool:
    """Whether this profile asks for nothing at all.

    Read to leave the shaping out of the request entirely, which is what keeps
    an installation that never touches these controls on the cheapest path
    through the worker's DSP -- and on Pocket that path matters more than usual,
    because the engine has one inference lane and every millisecond of it is a
    millisecond a Stop may have to wait for (section 21.3).
    """
    found = clamp(profile)
    return (all(found[name] == CONTROLS[name]["default"] for name in DELIVERY_FIELDS)
            and all(found[name] is None for name in GENERATION_FIELDS))


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
    """One line naming only what has been changed, or "PocketTTS's own delivery"."""
    found = clamp(profile)
    parts = []
    for name in FIELDS:
        spec = CONTROLS[name]
        value = found[name]
        if value == spec["default"]:
            continue
        parts.append(f"{spec['label'].casefold()} {value_label(name, value)}")
    return ", ".join(parts) if parts else "PocketTTS's own delivery"


# --------------------------------------------------------------------------- #
# The default voice's profile
# --------------------------------------------------------------------------- #


def stored() -> dict:
    """The profile the Pocket default voice speaks with, from Pocket's own file.

    Not from the host's options, and that is the correction Sopro already had to
    make rather than a preference: an option is a component on the settings page
    as well as a stored value, so "Apply settings" writes the page's build-time
    copy back over whatever the panel has since set. Two controls for one value,
    each able to overwrite the other (I-PKT-19). See
    :func:`mc_voice_pocket._setting`, which still reads an option when the file
    has nothing, so nothing an older build configured is lost.
    """
    import mc_voice_pocket as pocket

    found = {}
    try:
        for name, option in OPTIONS.items():
            found[name] = pocket._setting(option)
    except Exception:
        return dict(DEFAULTS)
    return clamp(found)


def remember(values=None, **extra) -> dict:
    """Write the Pocket default profile through to Pocket's file, and save it.

    Immediate, because this is an operational surface: somebody who slows the
    voice down while listening to it expects the next sentence to be slower, not
    to be told to press Apply.

    It writes only Pocket's own names. Kokoro's four and Sopro's eight remain
    untouched whatever happens here (I-3, I-PKT-3), which is enforced by the
    option names rather than by a check that could be forgotten.
    """
    import mc_voice_pocket as pocket

    wanted = clamp(values, **extra)
    try:
        for name, option in OPTIONS.items():
            value = wanted[name]
            # A generation field that is ``None`` is stored as an empty string,
            # which is what "follow the model" survives a round trip as.
            pocket._remember(option, "" if value is None else value)
    except Exception:
        logger.debug("Model Chain: could not persist the Pocket delivery profile",
                     exc_info=True)
    return stored()


def resolve(values=None) -> dict:
    """One profile to speak with: character overrides on top of the Pocket default.

    ``None`` in a character field means "follow the default", which is how a
    character that has never been given a Pocket delivery follows Settings, and
    how a character given only a slower speed keeps the default's pitch.

    ``None`` surviving all the way through -- character unset, default unset --
    is what reaches the worker as "no temperature key at all", and the worker
    reads that as the model's own. Three layers, one meaning, and no layer
    substitutes a number for it (T-PROFILE-P4).
    """
    base = stored()
    offered = overrides(values)
    found = {name: (base[name] if offered[name] is None else offered[name])
             for name in FIELDS}
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

    ``temperature`` is *omitted* rather than sent as null when it is ``None``.
    The worker's :class:`Delivery` reads an absent key as "the model's own",
    which is the one representation of that state -- sending a number would make
    the model's own recommendation a value this process had to know, and
    I-PKT-25 says a model revision is identified by its content rather than by
    whatever a settings file remembered about it.
    """
    found = clamp(profile)
    made = {
        "speed": float(found["speed"]),
        "pitch": pitch_ratio(found["pitch"]),
        "gain": gain_scale(found["gain"]),
        "pause_ms": int(found["pause"]),
    }
    for name in GENERATION_FIELDS:
        if found[name] is not None:
            made[name] = found[name]
    return made
