"""How a voice is delivered: speed, pitch, volume and pacing.

A *voice* is which speaker in the Kokoro bank is talking. A *delivery profile*
is everything about how that speaker says it, and this module is the one place
the four controls are defined, bounded, stored and resolved.

What Kokoro actually offers, stated plainly
-------------------------------------------
This matters more than it usually would, because the obvious expectation of a
modern TTS model -- a dial marked "emotion" -- is one Kokoro-82M does not have,
and inventing a slider that quietly did nothing would be worse than not offering
one.

``sherpa_onnx.OfflineTts.generate`` takes exactly two things that change the
sound: ``sid``, the speaker, and ``speed``. There is no emotion input, no style
vector beyond the speaker's own, no per-utterance prosody conditioning and no
SSML. The 1x256 style vector the graph is handed is selected by speaker id and
token length and is otherwise fixed -- ``mc_voice_bank`` documents the same
arithmetic from the other side.

So one of the four controls below is the model's and three are ours:

    speed   Kokoro's own. Passed straight to ``generate`` and changes how the
            model articulates, not just how fast the samples come out.

    pitch   Ours, and honest about how it works. Synthesis runs at
            ``speed x ratio`` and the audio is then resampled by ``ratio``,
            which shifts every frequency by the ratio and puts the duration
            back where the speed setting asked for it. Formants move with the
            pitch, so this reads as a different-sized speaker rather than as a
            transposed one -- which is what makes it useful for giving two
            characters the same voice and different bodies, and what makes
            large values sound like a cartoon.

    volume  Ours. A scalar on the samples, with a soft limiter, applied where
            the PCM is built.

    pacing  Ours. Silence inserted between sentences, which the segmenter has
            already found for streaming. It is the one control that changes
            delivery rather than timbre: a character who leaves a beat between
            sentences reads as measured, and one who leaves none reads as
            urgent.

Everything above is arithmetic on samples the model produced, and all of it is
done in the worker so that the streamed path and the completed-audio path
cannot drift apart. A profile that asks for nothing costs nothing: the worker
skips the shaping entirely when the numbers are neutral.

Two surfaces, one definition
----------------------------
Settings -> Voice Chat sets the profile the default voice speaks with. A
character carries its own, saved in the character file beside its sampling. Both
draw their controls from :data:`CONTROLS` and both clamp through :func:`clamp`,
so a value that is impossible in one place is impossible in the other, and a
range changed here changes both.

Inheritance is by absence. A character field that is ``None`` -- which is what a
character written before this existed, or never edited, has -- means "follow the
default", and :func:`resolve` is where that happens. It is deliberately not a
copy: somebody who slows the default voice down expects the characters they
never configured to slow down with it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

OPT_SPEED = "model_chain_voice_speed"
OPT_PITCH = "model_chain_voice_pitch"
OPT_GAIN = "model_chain_voice_gain"
OPT_PAUSE = "model_chain_voice_pause"

FIELDS = ("speed", "pitch", "gain", "pause")
"""Display order, and the order every list of four values in this feature is in."""

OPTIONS = {"speed": OPT_SPEED, "pitch": OPT_PITCH, "gain": OPT_GAIN, "pause": OPT_PAUSE}

CONTROLS = {
    "speed": {
        "label": "Speed",
        "unit": "x",
        "minimum": 0.5,
        "maximum": 2.0,
        "step": 0.05,
        "default": 1.0,
        "decimals": 2,
        "help": "Kokoro's own speaking rate. Below about 0.7 it starts to slur and above "
                "about 1.6 it starts to clip its own consonants.",
    },
    "pitch": {
        "label": "Pitch",
        "unit": " semitones",
        "minimum": -12.0,
        "maximum": 12.0,
        "step": 0.5,
        "default": 0.0,
        "decimals": 1,
        "help": "Shifts the whole voice, formants included, so it reads as a different-sized "
                "speaker. Two or three semitones is a noticeable change of character; a "
                "whole octave is a cartoon.",
    },
    "gain": {
        "label": "Volume",
        "unit": " dB",
        "minimum": -12.0,
        "maximum": 12.0,
        "step": 0.5,
        "default": 0.0,
        "decimals": 1,
        "help": "Loudness relative to the model's own output, limited so a loud setting "
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
        "help": "Extra silence after each sentence, on top of the model's own. A measured "
                "character gets 200-400 ms; an urgent one gets none.",
    },
}

DEFAULTS = {name: CONTROLS[name]["default"] for name in FIELDS}
"""The neutral profile: Kokoro exactly as it comes out of the model."""


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #


def _number(value, fallback):
    """``value`` as a float, or ``fallback`` for anything that is not one.

    Total, because every caller of this is either a settings store that may hold
    anything, a YAML file somebody edited by hand, or a JSON body from a
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

    Never raises and never returns a partial dictionary. A missing field takes
    the neutral default rather than the stored one, because this is the
    "make these numbers safe" function and inheritance is :func:`resolve`'s job.
    """
    found = dict(values or {})
    found.update(extra)
    made = {}
    for name in FIELDS:
        spec = CONTROLS[name]
        wanted = _number(found.get(name), spec["default"])
        wanted = min(max(wanted, spec["minimum"]), spec["maximum"])
        made[name] = round(wanted, spec["decimals"])
    return made


def overrides(values=None, **extra) -> dict:
    """The same, but keeping ``None`` as ``None``.

    What a character stores. A field nobody has set stays unset, which is what
    makes it follow the default voice's profile rather than freezing today's
    value into the character file.
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
        made[name] = round(min(max(wanted, spec["minimum"]), spec["maximum"]),
                           spec["decimals"])
    return made


def neutral(profile=None) -> bool:
    """Whether this profile asks for nothing at all.

    Read by the worker's caller to leave the shaping out of the request
    entirely, which is what keeps an installation that never touches these
    controls byte-for-byte on the path it had before they existed.
    """
    found = clamp(profile)
    return all(found[name] == CONTROLS[name]["default"] for name in FIELDS)


def value_label(name: str, value) -> str:
    """One control's value as it is written on screen, unit and sign included.

    Here rather than in each surface because three of them draw it -- the
    Settings sliders, the character screen's sliders and the status line
    :func:`describe` builds -- and a plus sign that appeared in two of the three
    would read as a different setting rather than as a different label.
    """
    spec = CONTROLS.get(name)
    if spec is None:
        return str(value)
    found = _number(value, spec["default"])
    text = f"{found:.{spec['decimals']}f}"
    if spec["decimals"]:
        text = text.rstrip("0").rstrip(".") or "0"
    sign = "+" if name in ("pitch", "gain") and found > 0 else ""
    return f"{sign}{text}{spec['unit']}"


def describe(profile=None) -> str:
    """One line naming only what has been changed, or "Kokoro's own delivery".

    Only the changes, because a status line reading "1.00x, 0.0 semitones,
    0.0 dB, 0 ms" says nothing four times over.
    """
    found = clamp(profile)
    parts = []
    for name in FIELDS:
        spec = CONTROLS[name]
        value = found[name]
        if value == spec["default"]:
            continue
        parts.append(f"{spec['label'].casefold()} {value_label(name, value)}")
    return ", ".join(parts) if parts else "Kokoro's own delivery"


# --------------------------------------------------------------------------- #
# The default voice's profile
# --------------------------------------------------------------------------- #


def stored() -> dict:
    """The profile the default voice speaks with, from the host's options."""
    found = {}
    try:
        from modules import shared

        for name, option in OPTIONS.items():
            found[name] = getattr(shared.opts, option, None)
    except Exception:
        return dict(DEFAULTS)
    return clamp(found)


def remember(values=None, **extra) -> dict:
    """Write the default profile through to the host's store, and save it.

    Immediate, like the two Voice switches and for the same reason: this is an
    operational surface. Somebody who slows the voice down while listening to it
    expects the next sentence to be slower, not to be told to press Apply.

    Best-effort against the host and never fatal. Returns what the store says
    afterwards, so a caller redraws from the truth rather than from what it
    hoped it had written.
    """
    wanted = clamp(values, **extra)
    try:
        from modules import shared

        for name, option in OPTIONS.items():
            shared.opts.set(option, wanted[name])
        shared.opts.save(shared.config_filename)
    except Exception:
        logger.debug("Model Chain: could not persist the Voice Chat delivery profile",
                     exc_info=True)
    return stored()


def resolve(values=None) -> dict:
    """One profile to speak with: overrides on top of the stored default.

    ``None`` in a field means "follow the default", which is how a character
    that has never been given a delivery follows Settings, and how a character
    given only a slower speed keeps the default's pitch.
    """
    base = stored()
    offered = overrides(values)
    return clamp({name: (base[name] if offered[name] is None else offered[name])
                  for name in FIELDS})


# --------------------------------------------------------------------------- #
# What the worker is told
# --------------------------------------------------------------------------- #


def pitch_ratio(semitones) -> float:
    """A pitch shift in semitones as the frequency ratio the worker resamples by.

    Twelve semitones is an octave, so the ratio is ``2 ** (n / 12)``. Bounded by
    :func:`clamp`'s own range before it gets here, and clamped again as a belt:
    this number multiplies a synthesis speed, and a runaway value would ask
    Kokoro for something it will refuse.
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
    stays a thing that multiplies and resamples what it is told to.
    """
    found = clamp(profile)
    return {
        "speed": float(found["speed"]),
        "pitch": pitch_ratio(found["pitch"]),
        "gain": gain_scale(found["gain"]),
        "pause_ms": int(found["pause"]),
    }
