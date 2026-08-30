"""The Sopro Voice Lab: an experiment that cannot become a setting.

Sopro's speaker encoder produces a richer conditioning structure than Kokoro's
speaker bank -- an identity embedding, a style embedding, and an eight-dimension
``style_ctrl`` vector. Those eight numbers are learned latents. Their meanings
are not documented as emotion, energy, breathiness, warmth or any other human
label, and this module is very deliberately built so that nobody can ship a
product claim about them by accident.

The design is one sentence: **the Lab has its own types, its own memory and its
own synthesis call, and there is no code path from any of them to Conversation.**

Why types rather than discipline
--------------------------------
Section 39 asks for production code to be unable to read Lab state *because the
types are separate*, rather than because a caller promised not to. So:

    a Lab session is :class:`Session`, which no production function accepts;
    Lab values are ``deltas`` and ``blend``, which no profile function reads;
    the synthesis is ``mc_voice_sopro_runtime.lab_audition``, which returns a
        WAV and never a turn;
    and nothing here writes a file, an option or a character.

There is no ``save``, no ``apply``, no ``promote`` and no ``to_profile``. Adding
one would be a design change with named semantics, bounds, migration and tests
-- which is what section 42 says a promotion has to be.

Session lifetime is a page, not a setting
------------------------------------------
Sessions live in this process's memory, keyed by an opaque token, and they are
dropped when the page stops asking, when the engine changes and when the WebUI
exits. Reloading Settings resets unsaved Lab state, which is a *requirement*
(section 39) rather than a limitation: a slider that survived a reload would be
a slider somebody could forget they had moved.

The eight sliders are deltas
----------------------------
Bounded offsets from the saved voice's own ``style_ctrl``, never a replacement
of it, so Reset All is exactly "every delta to zero" and the base vector is
whatever the voice was prepared with. :data:`DELTA_LIMIT` is a *conservative*
opening range chosen to be re-measured, not a claim about the model -- see its
own docstring.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

STYLE_CONTROLS = 8
"""How many style-control sliders the Lab shows.

The reviewed Sopro V2 ``style_ctrl_dim``. Checked against what the running
worker reports rather than assumed: a model revision with a different width
makes the Lab show that many and say so, instead of silently editing the first
eight of twelve.
"""

DELTA_LIMIT = 1.5
"""How far one style-control slider may move, either way.

Deliberately conservative and deliberately provisional. Section 41 says the
useful scale has to be *measured* against the pinned model before release --
swept positive and negative across several voices to find a range that does not
routinely explode the level, produce silence or destroy intelligibility -- and
that this UI then uses the tested range with zero at centre.

This is the opening bound for that sweep, not its result. The worker clamps to
its own wider hard limit independently, so a value from a stale page cannot
exceed what the model was shown to tolerate.
"""

BLEND_LIMIT = 1.0
"""The maximum weight of another voice's conditioning in a blend.

One means "entirely the other voice's chosen components", which is a legitimate
end of the experiment: it is what tells a listener whether the component being
substituted carries the thing they think it carries.
"""

BLEND_FIELDS = ("id_emb", "style_emb", "style_ctrl")
"""Which pre-projection components a blend may substitute.

The three the speaker encoder produces, and nothing else. A complete Sopro
Reference also holds semantic reference tokens and a reference mel, and this
never touches those -- which is exactly why section 42 calls the feature
Conditioning Blend and not identity or style transfer. Substituting one subset
of speaker conditioning while keeping another voice's reference context is not
proven disentanglement, and the UI does not say it is.
"""

MAX_TEXT_CHARS = 400
MAX_SESSIONS = 4
SESSION_TTL = 30 * 60.0
"""Four sessions, half an hour each. Bounded because a Lab session holds a
token and a handful of floats in this process, and an unbounded map of them is
a settings page that leaks a little every time somebody opens it."""

_lock = threading.RLock()
_sessions: dict = {}


class LabError(RuntimeError):
    """A Lab operation that could not be completed. Never fatal, never saved."""


class Session:
    """One Lab session's experimental state. Never production state.

    A class rather than a dict so that a production function handed one fails
    where the mistake is, rather than reading plausible-looking keys out of a
    mapping. Nothing in :mod:`mc_voice_sopro`, :mod:`mc_voice_sopro_profile` or
    :mod:`mc_voice_turn` accepts this type or any field of it.
    """

    __slots__ = ("token", "voice_id", "deltas", "blend", "seed", "text",
                 "temperature", "top_p", "top_k", "steps", "chunk_frames",
                 "base_style", "touched", "last")

    def __init__(self, token: str):
        self.token = str(token)
        self.voice_id = ""
        self.deltas = [0.0] * STYLE_CONTROLS
        self.blend = {}
        self.seed = None
        self.text = ""
        self.temperature = None
        self.top_p = None
        self.top_k = None
        self.steps = None
        self.chunk_frames = None
        self.base_style = []
        self.touched = time.monotonic()
        self.last = {}

    def reset(self) -> None:
        """Reset All. Mandatory, and it is every experimental value at once.

        The audition text and the chosen voice survive, because they are what
        the user is testing *with* rather than what they are testing.
        """
        self.deltas = [0.0] * STYLE_CONTROLS
        self.blend = {}
        self.seed = None
        self.temperature = None
        self.top_p = None
        self.top_k = None
        self.steps = None
        self.chunk_frames = None
        self.last = {}

    @property
    def neutral(self) -> bool:
        return (not any(abs(value) > 1e-9 for value in self.deltas)
                and not self.blend
                and self.temperature is None and self.top_p is None
                and self.top_k is None and self.steps is None
                and self.chunk_frames is None)

    def profile(self) -> dict:
        """The Lab's generation settings, in the profile module's own spelling.

        Built fresh each time and never stored anywhere a production resolve
        could reach. Delivery is deliberately neutral: the Lab is investigating
        conditioning, and time-scaling the result would put Voice Chat's own DSP
        between the user and the thing they are listening for.
        """
        found = {"speed": 1.0, "pitch": 0.0, "gain": 0.0, "pause": 0.0}
        for name in ("temperature", "top_p", "top_k"):
            value = getattr(self, name)
            if value is not None:
                found[name] = value
        return found

    def public(self) -> dict:
        """What the Lab surface draws. No tensors, no paths, no production ids."""
        return {
            "token": self.token,
            "voice_id": self.voice_id,
            "deltas": [round(float(value), 4) for value in self.deltas],
            "blend": dict(self.blend),
            "seed": self.seed,
            "text": self.text,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "steps": self.steps,
            "chunk_frames": self.chunk_frames,
            "base_style": [round(float(value), 4) for value in self.base_style],
            "neutral": self.neutral,
            "last": dict(self.last),
            "limits": {"delta": DELTA_LIMIT, "blend": BLEND_LIMIT,
                       "controls": len(self.deltas)},
        }


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #


def open_session(voice_id: str = "") -> dict:
    """Begin a Lab session on a registered Sopro voice.

    Refuses unless Sopro is the selected engine, which is section 38: the Lab
    exists only while Sopro is active, and it does not appear in the Voice
    overlay, the character editor, the ordinary voice picker or any Kokoro
    surface.
    """
    _require_sopro()
    import mc_voice_sopro as sopro

    entry = sopro.lookup(voice_id) if voice_id else sopro.default_entry()
    if entry is None:
        raise LabError("Create a Sopro voice before opening the Voice Lab.")

    found = Session(uuid.uuid4().hex[:16])
    found.voice_id = entry["id"]
    found.text = sopro.test_text()
    found.base_style = _base_style(entry["id"])
    if found.base_style and len(found.base_style) != STYLE_CONTROLS:
        # A model revision with a different width. Shown as what it is rather
        # than truncated: eight sliders over a twelve-dimension vector would be
        # a Lab quietly experimenting on a third of the thing it names.
        found.deltas = [0.0] * len(found.base_style)
    _remember(found)
    logger.info("Model Chain: a Sopro Voice Lab session was opened")
    return found.public()


def _base_style(voice_id: str) -> list:
    """The saved ``style_ctrl`` the sliders sit at zero against. Best effort.

    A worker that will not start is a Lab that shows sliders without their base
    values rather than a Lab that will not open: the deltas are still applied
    against whatever the voice was prepared with, because the *worker* adds them
    to the saved vector. Showing the numbers is a convenience, not the mechanism.
    """
    try:
        import mc_voice_sopro_runtime as runtime

        return list(runtime.lab_style(voice_id))
    except Exception:
        logger.debug("Model Chain: could not read a Sopro voice's style controls",
                     exc_info=True)
        return []


def session(token: str) -> Session:
    _expire()
    with _lock:
        found = _sessions.get(str(token or ""))
    if found is None:
        raise LabError("That Voice Lab session has expired. Reopen the Lab.")
    found.touched = time.monotonic()
    return found


def close(token: str) -> None:
    with _lock:
        _sessions.pop(str(token or ""), None)


def forget_all(reason: str = "") -> None:
    """Drop every Lab session. Called when the engine changes, and on shutdown.

    Section 39: switching away from Sopro destroys Lab session state. Not
    because the state is dangerous where it is, but because a Lab session that
    survived an engine switch would be a Lab session pointing at a voice library
    that is no longer the active one.
    """
    with _lock:
        count = len(_sessions)
        _sessions.clear()
    if count:
        logger.info("Model Chain: %d Sopro Voice Lab session(s) were discarded — %s",
                    count, reason or "no reason given")


def _remember(found: Session) -> None:
    with _lock:
        _sessions[found.token] = found
        while len(_sessions) > MAX_SESSIONS:
            oldest = min(_sessions.values(), key=lambda item: item.touched)
            _sessions.pop(oldest.token, None)


def _expire() -> None:
    now = time.monotonic()
    with _lock:
        for token in [token for token, found in _sessions.items()
                      if now - found.touched > SESSION_TTL]:
            _sessions.pop(token, None)


# --------------------------------------------------------------------------- #
# Editing an experiment
# --------------------------------------------------------------------------- #


def update(token: str, **values) -> dict:
    """Change one Lab session's experimental values. Writes nothing to disk.

    Every branch clamps rather than refuses, because this is a set of sliders
    somebody is dragging and an error toast per drag would be unusable. What it
    will not do is accept a shape it does not recognise: an unknown key is
    ignored, so a page from a later build cannot set a field this one would then
    pass to the worker unvalidated.
    """
    _require_sopro()
    found = session(token)

    if "voice_id" in values:
        import mc_voice_sopro as sopro

        entry = sopro.lookup(values["voice_id"])
        if entry is None:
            raise LabError("That Sopro voice does not exist.")
        if entry["id"] != found.voice_id:
            # A different voice is a different experiment. Keeping the deltas
            # would apply one voice's offsets to another's base vector, which is
            # the one comparison in this surface that means nothing at all.
            found.voice_id = entry["id"]
            found.reset()
            found.base_style = _base_style(entry["id"])

    if "deltas" in values:
        offered = list(values.get("deltas") or ())
        width = len(found.deltas)
        found.deltas = [_clamp(offered[index] if index < len(offered) else 0.0,
                               -DELTA_LIMIT, DELTA_LIMIT)
                        for index in range(width)]

    if "blend" in values:
        found.blend = _blend(values.get("blend"), found.voice_id)

    if "seed" in values:
        offered = values.get("seed")
        try:
            found.seed = None if offered in (None, "", False) else int(offered) & 0x7FFFFFFF
        except (TypeError, ValueError):
            found.seed = None

    if "text" in values:
        found.text = str(values.get("text") or "").strip()[:MAX_TEXT_CHARS]

    for name, low, high, whole in (("temperature", 0.1, 1.5, False),
                                   ("top_p", 0.1, 1.0, False),
                                   ("top_k", 1, 200, True)):
        if name in values:
            offered = values.get(name)
            if offered in (None, ""):
                setattr(found, name, None)
            else:
                value = _clamp(offered, low, high)
                setattr(found, name, int(value) if whole else round(value, 3))

    import mc_voice_sopro as sopro

    if "steps" in values:
        offered = values.get("steps")
        try:
            value = int(offered)
        except (TypeError, ValueError):
            value = 0
        found.steps = value if value in sopro.STEP_CHOICES else None
    if "chunk_frames" in values:
        offered = values.get("chunk_frames")
        try:
            value = int(offered)
        except (TypeError, ValueError):
            value = 0
        found.chunk_frames = value if value in sopro.CHUNK_CHOICES else None

    return found.public()


def _blend(offered, own: str) -> dict:
    """A Conditioning Blend request, validated. Empty means "no blend".

    A blend of a voice with itself is dropped rather than run: it is a null
    experiment that would still cost a full synthesis, and a UI that let
    somebody sit on it would be a UI where "nothing changed" has two meanings.
    """
    found = dict(offered or {})
    voice_id = str(found.get("voice_id") or "").strip()
    if not voice_id or voice_id == str(own or ""):
        return {}
    import mc_voice_sopro as sopro

    if sopro.lookup(voice_id) is None:
        raise LabError("That Sopro voice does not exist.")
    fields = [name for name in BLEND_FIELDS if found.get(name)]
    weight = _clamp(found.get("weight"), 0.0, BLEND_LIMIT)
    if not fields or weight <= 0.0:
        return {}
    return {"voice_id": voice_id, "weight": round(weight, 3),
            **{name: True for name in fields}}


def reset(token: str) -> dict:
    """Reset All. Mandatory (section 44), and it is one call."""
    found = session(token)
    found.reset()
    return found.public()


def _clamp(value, low, high) -> float:
    try:
        found = float(value)
    except (TypeError, ValueError):
        return 0.0 if low <= 0.0 <= high else float(low)
    if found != found or found in (float("inf"), float("-inf")):
        return 0.0 if low <= 0.0 <= high else float(low)
    return min(max(found, float(low)), float(high))


# --------------------------------------------------------------------------- #
# Auditioning an experiment
# --------------------------------------------------------------------------- #


def audition(token: str, side: str = "b") -> dict:
    """Play A or B. Returns a WAV and the run's own numbers. Saves nothing.

    ``A`` is the production voice exactly as Conversation would speak it with
    the *saved* profile; ``B`` is the same text through the experimental
    conditioning. Same text and, when a seed is fixed, the same seed -- which is
    what lets somebody attribute a difference to the control they moved rather
    than to sampling noise (section 40).

    The A side deliberately goes through the ordinary production audition rather
    than through a Lab call with zero deltas. If it went through the Lab path it
    would prove that the Lab at neutral sounds like the Lab at neutral, which is
    not the comparison anybody wants.
    """
    _require_sopro()
    found = session(token)
    if not found.voice_id:
        raise LabError("Choose a Sopro voice for the Lab first.")
    text = found.text or _default_text()

    import mc_voice_sopro_runtime as runtime

    started = time.monotonic()
    if str(side).lower() == "a":
        import mc_voice_sopro_profile as profiles

        audio = runtime.synthesize(text, found.voice_id,
                                   profiles.resolve(None))
        made = {"audio": audio, "elapsed_ms": int((time.monotonic() - started) * 1000)}
    else:
        made = runtime.lab_audition(found.voice_id, text, deltas=found.deltas,
                                    blend=found.blend or None,
                                    profile=_with_seed(found))
    found.last = {
        "side": "a" if str(side).lower() == "a" else "b",
        "first_audio_ms": int(made.get("first_audio_ms") or 0),
        "elapsed_ms": int(made.get("elapsed_ms") or 0),
        "audio_ms": int(made.get("audio_ms") or 0),
        "rtf": float(made.get("rtf") or 0.0),
        "chunks": int(made.get("chunks") or 0),
        "at": time.strftime("%H:%M:%S"),
    }
    return {"audio": made.get("audio") or b"", "state": found.public()}


def _with_seed(found: Session) -> dict:
    """The Lab's generation profile, plus its fixed seed if it has one.

    The seed rides in the profile dictionary rather than in a Lab-only argument
    because the worker's :class:`Delivery` already reads it from the header --
    and it is the one field of that class production never sets, which is why
    Conversation's sampling is unaffected by anything the Lab does here.
    """
    made = found.profile()
    if found.seed is not None:
        made["seed"] = int(found.seed)
    if found.steps is not None:
        made["steps"] = int(found.steps)
    if found.chunk_frames is not None:
        made["chunk_frames"] = int(found.chunk_frames)
    return made


def _default_text() -> str:
    try:
        import mc_voice_sopro as sopro

        return sopro.test_text()
    except Exception:
        return "This is a test of the Sopro voice."


def panel() -> dict:
    """What the Settings Lab surface needs before a session exists.

    Voices to choose between, the bounds, and the sentence that has to be on
    screen. The sentence is not decoration: section 44 requires the Lab to make
    it difficult to mistake a result for the character's saved voice, and this
    is where that text lives so that every surface drawing the Lab shows the
    same one.
    """
    _require_sopro()
    import mc_voice_sopro as sopro

    return {
        "voices": [{"id": entry["id"], "label": entry["display_name"]}
                   for entry in sopro.entries() if entry["compatible"]],
        "controls": STYLE_CONTROLS,
        "delta_limit": DELTA_LIMIT,
        "blend_limit": BLEND_LIMIT,
        "blend_fields": list(BLEND_FIELDS),
        "steps": list(sopro.STEP_CHOICES),
        "chunk_choices": list(sopro.CHUNK_CHOICES),
        "notice": ("These controls affect only this audition. They are not used in "
                   "Conversation and are not saved to characters or to the default voice."),
        "caution": ("The eight style controls are learned latent values. They are not "
                    "emotions, energy, warmth or breathiness — nobody has measured what "
                    "they mean, which is what this surface is for. Conditioning Blend "
                    "recombines speaker conditioning while keeping the first voice's "
                    "reference context; it is not proven identity or style transfer."),
    }


def _require_sopro() -> None:
    import mc_voice_engines as engines

    if engines.active() != engines.SOPRO:
        raise LabError("The Voice Lab is part of Sopro, and Sopro is not the selected "
                       "text-to-speech engine.")
