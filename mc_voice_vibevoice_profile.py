"""How a VibeVoice voice is delivered in Voice Chat: two of the model's own controls.

The VibeVoice half of :mod:`mc_voice_profile`, :mod:`mc_voice_sopro_profile`
and :mod:`mc_voice_pocket_profile`. The same contract -- ``FIELDS``,
``CONTROLS``, ``DEFAULTS``, ``OPTIONS``, ``clamp``, ``overrides``, ``neutral``,
``value_label``, ``describe``, ``stored``, ``remember``, ``resolve``,
``request`` -- and deliberately a separate module with separate storage,
because common labels do not imply a shared implementation (section 35,
I-PKT-23).

What VibeVoice actually offers, stated plainly
----------------------------------------------
Both of its models generate speech with a language model that decides what
comes next and a diffusion head that draws each frame of sound. Two numbers
steer that and nothing else does:

    cfg_scale    Guidance. Classifier-free guidance on the diffusion head: how
                 strongly each frame is pulled towards the voice and the text
                 it was given rather than drifting. The model's.

    steps        Diffusion steps. How many denoising steps the head takes for
                 each frame. More costs more time on the card for every
                 sentence; fewer is faster and rougher. The model's.

There is no speed, pitch, volume or pause here, and not because they are
hard. The VibeVoice worker returns what the model produced and has no signal
processing of its own, and the Voice Pipeline serves PocketTTS only -- so a
Speed slider on this engine would be a control that did nothing, which is the
one kind of control this feature refuses to draw (section 37).

Both default to ``None``, "the model's own"
-------------------------------------------
The Realtime 0.5B and the 7B ship different defaults, and which model speaks
is a property of the *voice* (a preset is the Realtime model's, a sample is
the 7B's), not of this profile. So a stored ``None`` is resolved at render
time against whichever model is speaking, by :mod:`mc_voice_vibevoice_speech`
from the installer's own record of that model -- and freezing one model's
numbers into this file would hand them to the other model on the next reply.

Where the values live
---------------------
In Voice Chat's own VibeVoice file (:func:`mc_voice_paths.vibevoice_chat_path`),
written the moment a slider is let go, and never in a Forge option. That is
the lesson Sopro learned first and Pocket started from (I-PKT-19): a host
option is a component on the settings page as well as a stored value, and
*Apply settings* writes the page's build-time copy back over whatever this
panel set since. :data:`OPTIONS` are the names the two values are stored under
in that file -- inert names, the way Pocket's are, which nothing registers with
the host and nothing reads out of its options. They exist so that no two
engines' profiles can ever share a storage name, which
``tests/test_voice_engines.py`` holds across all four.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

OPT_CFG = "model_chain_voice_vibevoice_cfg"
OPT_STEPS = "model_chain_voice_vibevoice_steps"

DELIVERY_FIELDS = ()
"""None. See the module docstring: this engine has no signal processing to
apply a speed, a pitch, a gain or a pause with."""

GENERATION_FIELDS = ("cfg_scale", "steps")

FIELDS = DELIVERY_FIELDS + GENERATION_FIELDS
"""Every field a character may override, in display order."""

OPTIONS = {"cfg_scale": OPT_CFG, "steps": OPT_STEPS}
"""Field to storage name in Voice Chat's VibeVoice file. Not Forge options: see
the module docstring. Not one of them overlaps another engine's, which is what
makes I-PKT-3 a property of the spelling rather than a rule to remember."""

MODEL_DEFAULT = None
"""What a field holds when it should follow the speaking model's own value.

``None`` in the *stored default* means "follow the model"; ``None`` in a
character override means "follow the stored default". Both are states, and
neither layer substitutes a number for them.
"""

CONTROLS = {
    "cfg_scale": {
        "label": "Guidance",
        "unit": "",
        "minimum": 1.0,
        "maximum": 3.0,
        "step": 0.05,
        "default": None,
        "decimals": 2,
        "group": "generation",
        "owner": "vibevoice",
        "advanced": False,
        "help": "How strongly VibeVoice holds each frame of speech to the voice and the "
                "text it was given. Higher is tighter and can start to sound strained; "
                "lower is looser and can wander from the voice. It is not an emotion, "
                "warmth or energy control — the model has no such input. Left alone it "
                "follows the speaking model's own value.",
    },
    "steps": {
        "label": "Diffusion steps",
        "unit": "",
        "minimum": 1,
        "maximum": 30,
        "step": 1,
        "default": None,
        "decimals": 0,
        "group": "generation",
        "owner": "vibevoice",
        "advanced": False,
        "help": "How many steps VibeVoice's diffusion head takes for each frame of "
                "speech. More costs more time on the card for every sentence and is the "
                "first thing to lower if replies start pausing mid-sentence; fewer is "
                "faster and rougher. Left alone it follows the speaking model's own "
                "value.",
    },
}
"""The two controls, key for key in the shape the other engines use."""

DEFAULTS = {name: CONTROLS[name]["default"] for name in FIELDS}
"""The neutral profile: the speaking model's own guidance and steps."""


class VibeVoiceProfileError(ValueError):
    """A delivery value that could not be used. Never fatal."""


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #


def _number(value, fallback):
    """``value`` as a float, or ``fallback`` for anything that is not one.

    Total, because every caller is a settings file that may hold anything, a
    character file somebody edited by hand, or a JSON body from a browser, and
    none of those is a reason for a reply to go unspoken.
    """
    if value is None or isinstance(value, bool):
        return fallback
    if isinstance(value, str) and not value.strip():
        return fallback
    try:
        found = float(value)
    except (TypeError, ValueError):
        return fallback
    if found != found or found in (float("inf"), float("-inf")):
        return fallback
    return found


def _fit(name: str, value):
    """One offered value, in range and in the control's precision, or ``None``."""
    spec = CONTROLS[name]
    found = _number(value, None)
    if found is None:
        return None
    found = min(max(found, spec["minimum"]), spec["maximum"])
    return round(found, spec["decimals"]) if spec["decimals"] else int(round(found))


def clamp(values=None, **extra) -> dict:
    """A complete, in-range profile from whatever was offered.

    Never raises and never returns a partial dictionary. Both fields are the
    model's, so a missing or unreadable one is ``None`` -- "follow the speaking
    model" -- rather than today's number for one of the two models.
    """
    found = dict(values or {})
    found.update(extra)
    return {name: _fit(name, found.get(name)) for name in FIELDS}


def shown(values=None, voice_id: str = "") -> dict:
    """``values`` with each "model's own" written as that model's number, for a slider.

    Display only. A slider cannot show "the model's own": a Gradio slider handed
    ``None`` sits at its minimum -- one diffusion step, guidance 1.0 -- and a
    character whose *Own delivery* was ticked without moving anything was saved
    with those. So the character editor shows the numbers the voice's own model
    would use, the 7B's for a Voice Box sample and the Realtime model's for a
    preset, and a character saved untouched keeps sounding the way it did.
    What is stored is never changed here.
    """
    found = clamp(values)
    missing = [name for name in FIELDS if found[name] is None]
    if not missing:
        return found
    own = _model_defaults(voice_id)
    return clamp({name: (own.get(name) if name in missing else found[name])
                  for name in FIELDS})


def _model_defaults(voice_id: str = "") -> dict:
    """The speaking model's own guidance and steps for ``voice_id`` (or the default voice)."""
    try:
        import mc_voice_vibevoice as vibevoice
        import mc_voice_vibevoice_chat as chat

        entry = chat.lookup(voice_id) if voice_id else None
        entry = entry or chat.default_entry()
        model = str(((entry or {}).get("_handle") or {}).get("model") or vibevoice.MODEL_7B)
        found = dict((vibevoice.model_info(model) or {}).get("defaults") or {})
    except Exception:
        logger.debug("Model Chain: could not read a VibeVoice model's own delivery",
                     exc_info=True)
        return {}
    return {"cfg_scale": found.get("cfg_scale"), "steps": found.get("steps")}


def overrides(values=None, **extra) -> dict:
    """What a character stores: an unset field stays ``None``.

    The same arithmetic as :func:`clamp`, kept as its own name because the two
    mean different things to a caller -- a character's ``None`` follows the
    stored default, the stored default's ``None`` follows the model.
    """
    return clamp(values, **extra)


def neutral(profile=None) -> bool:
    """Whether this profile asks for nothing: the speaking model's own delivery."""
    return all(value is None for value in clamp(profile).values())


def value_label(name: str, value) -> str:
    """One control's value as it is written on screen."""
    spec = CONTROLS.get(name)
    if spec is None:
        return str(value)
    if value is None:
        return "model default"
    found = _number(value, None)
    if found is None:
        return "model default"
    text = f"{found:.{spec['decimals']}f}"
    if spec["decimals"]:
        text = text.rstrip("0").rstrip(".") or "0"
    return f"{text}{spec['unit']}"


def describe(profile=None) -> str:
    """One line naming only what has been changed, or "VibeVoice's own delivery"."""
    found = clamp(profile)
    parts = [f"{CONTROLS[name]['label'].casefold()} {value_label(name, found[name])}"
             for name in FIELDS if found[name] is not None]
    return ", ".join(parts) if parts else "VibeVoice's own delivery"


# --------------------------------------------------------------------------- #
# The default voice's profile
# --------------------------------------------------------------------------- #


def stored() -> dict:
    """The profile the default voice speaks with, from Voice Chat's VibeVoice file.

    Never from the host's options (see the module docstring), and never
    raises: a file that cannot be read is the model's own delivery.
    """
    import mc_voice_vibevoice_chat as chat

    try:
        found = chat._stored(tuple(OPTIONS.values()))
    except Exception:
        logger.debug("Model Chain: could not read the VibeVoice delivery", exc_info=True)
        return dict(DEFAULTS)
    return clamp({name: found.get(option) for name, option in OPTIONS.items()})


def remember(values=None, **extra) -> dict:
    """Write the default profile to Voice Chat's VibeVoice file, now.

    Immediate, because this is an operational surface: somebody who lowers the
    steps while listening expects the next sentence to be quicker, not to be
    told to press Apply. Both values in one atomic write, so a reader never sees
    one of them changed and the other not yet. Only VibeVoice's own names are
    written; no other engine's delivery can be reached from here (I-3).
    """
    import mc_voice_vibevoice_chat as chat

    wanted = clamp(values, **extra)
    try:
        chat._store({option: wanted[name] for name, option in OPTIONS.items()})
    except Exception:
        logger.debug("Model Chain: could not persist the VibeVoice delivery", exc_info=True)
    return stored()


def resolve(values=None) -> dict:
    """One profile to speak with: a character's overrides on the stored default.

    ``None`` surviving both layers -- character unset, default unset -- reaches
    :func:`request` as an absent key, and the speech runtime reads an absent
    key as the speaking model's own value.
    """
    base = stored()
    offered = overrides(values)
    return clamp({name: (base[name] if offered[name] is None else offered[name])
                  for name in FIELDS})


# --------------------------------------------------------------------------- #
# What the renderer is told
# --------------------------------------------------------------------------- #


def request(profile=None) -> dict:
    """The profile as a render carries it: only the fields somebody set.

    A ``None`` field is *omitted* rather than sent as null, so the one place
    that knows each model's own values -- :mod:`mc_voice_vibevoice_speech`,
    from the installer's record of the model speaking -- fills it in.
    """
    found = clamp(profile)
    made = {}
    if found["cfg_scale"] is not None:
        made["cfg_scale"] = float(found["cfg_scale"])
    if found["steps"] is not None:
        made["steps"] = int(found["steps"])
    return made
