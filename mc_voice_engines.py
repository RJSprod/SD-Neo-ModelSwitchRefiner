"""Which text-to-speech engine is selected, and what a voice id means now.

Voice Chat had one TTS engine, so "voice" could be a Kokoro registry id and
"which engine" was not a question anybody had to ask. Sopro V2 ended both of
those assumptions, and this module is where they ended -- at the storage
boundary, once, rather than patched at every caller.

    ONE TTS ENGINE IS SELECTED FOR THE WHOLE WEBUI AT A TIME.

That is the product rule (I-1, I-PKT-1) and it is the only rule this module
enforces directly. Everything else here exists to make it survivable: a stable
id that says which engine owns a voice, an engine-scoped view of the default and
of a character, and a facade so that a caller who wants to speak asks *the active
engine* rather than asking Kokoro.

Why this is a registry and no longer a pair of branches
------------------------------------------------------
Two engines can be written as ``if sopro: ... else: ...`` and nobody notices,
because "the other one" is unambiguous. A third engine is where that stops being
a shortcut and becomes a defect: ``else`` silently means Kokoro, so a Pocket
voice resolved through a fallthrough would be resolved by the wrong bank, a
Pocket engine nobody had installed would report Kokoro's readiness, and a Pocket
worker would be left running by a ``_stop_all`` that only knew two names.

So the engines are data -- :data:`SPECS` -- and every operation below is a lookup
in it. I-PKT-30: after Pocket ships, a fourth engine registers a spec and an
adapter, and no shared fallthrough has to be found and corrected first.

The modules are named by string and imported lazily. Importing this module must
not import Torch, sherpa-onnx, Sopro or Pocket: it is read on the path that draws
a settings page, and an installation that has never selected an engine should not
pay that engine's import cost to find out it is not installed.

Speech-to-text is not in this selector and never will be. Whisper has its own
process, its own dependency closure and its own lifecycle, and I-7/I-PKT-5 say
switching engines must not reload it, change its quality tier or reset the
microphone. Nothing in this module touches :mod:`mc_voice_models`' STT state, and
``tests/test_voice_engines.py`` asserts that the switch below does not.

What a voice id is
------------------
    kokoro:official:af_heart
    kokoro:clone:<uuid>
    sopro:clone:<uuid>
    pocket:official:alba
    pocket:clone:<uuid>

Backend first, always, so that no caller outside an adapter can be handed a
voice and not know whose it is. A Sopro or Pocket id contains a server-generated
UUID and never a filesystem path, a display name or anything a browser supplied
(I-10, I-PKT-20, section 57).

Legacy ids -- ``official:af_heart``, ``clone:<uuid>``, and the bare speaker
names V1 wrote -- are Kokoro's, read as Kokoro's, and are *not* rewritten on
sight. Migration is by reading (section 12): a character file written in 2025
resolves correctly today and is only rewritten when somebody edits it. That is
what makes the upgrade lossless rather than a bulk rewrite of every character
in somebody's library. Adding Pocket changes none of it: no old id becomes a
Pocket id, ever.

Why the inactive engine is absent rather than hidden
----------------------------------------------------
Section 5 asks for the payload to be scoped, not the CSS. :func:`scope` and
:func:`refuse_mismatch` are how that is enforced on the server: a page that was
open when the engine changed, a theme script that re-rendered half a panel, or
a request replayed from a stale DOM cannot mutate the inactive engine's
operational settings. They get an active-engine mismatch answer instead, which
is a sentence rather than a silent write into state nobody is looking at.

Interruption is a capability, not an assumption
-----------------------------------------------
:func:`capabilities` and :func:`interrupt_mode` exist because the three engines
do not stop the same way. Kokoro and Sopro cancel; released PocketTTS 3.0.2 has
no safe cooperative cancellation for an abandoned stream, so it drains the one
native unit already in flight while the browser is silent (I-PKT-10, section 21).
Shared Voice Chat asks an engine to interrupt a turn and reads what that engine
says interruption means. It does not decide for it, and it does not infer the
answer from a version number.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

OPT_ENGINE = "model_chain_tts_engine"
"""The stored engine id. The display label may change; this may not.

Section 4 names this option explicitly, so it is spelled exactly as the design
intent spells it rather than following this feature's ``model_chain_voice_``
prefix. A rename here is a migration, not a tidy-up.
"""

KOKORO = "kokoro"
SOPRO = "sopro"
POCKET = "pocket"


@dataclass(frozen=True)
class EngineSpec:
    """One engine, as data: what it is called and which modules own it.

    Frozen because it is a table rather than state. The three module names are
    strings and not modules: this file is imported to answer "which engine is
    selected", which is a question a settings page asks before anything has been
    installed, and importing three runtimes to answer it would make an
    installation pay for engines it does not use.
    """

    id: str
    label: str
    blurb: str
    adapter: str
    runtime: str
    profiles: str


SPECS = (
    EngineSpec(
        id=KOKORO,
        label="Kokoro",
        blurb="The built-in speaker bank, through sherpa-onnx. Fast, small, and installed "
              "already if you have been using Voice Chat.",
        adapter="mc_voice_kokoro",
        runtime="mc_voice_runtime",
        profiles="mc_voice_profile",
    ),
    EngineSpec(
        id=SOPRO,
        label="Sopro V2",
        blurb="A streaming model that clones a voice from a short recording you make here. "
              "Installed separately, runs on the CPU, and brings its own runtime.",
        adapter="mc_voice_sopro",
        runtime="mc_voice_sopro_runtime",
        profiles="mc_voice_sopro_profile",
    ),
    EngineSpec(
        id=POCKET,
        label="PocketTTS",
        blurb="A CPU text-to-speech model with streaming and reference voice cloning. "
              "Installed separately and kept isolated from Kokoro and Sopro.",
        adapter="mc_voice_pocket",
        runtime="mc_voice_pocket_runtime",
        profiles="mc_voice_pocket_profile",
    ),
)
"""Every engine this build knows, in selector order.

Kokoro first because it is the default and the one an upgrade lands on. A stored
value outside this table means Kokoro -- see :func:`active`.
"""

ENGINES = tuple(found.id for found in SPECS)
"""Every engine id, derived. Kept as a tuple because callers iterate it and
because ``in ENGINES`` is the validity test the whole module rests on."""

DEFAULT_ENGINE = KOKORO
"""Where an installation with no stored choice starts.

Section 12: if no active engine setting exists, the default is Kokoro, so an
upgrade does not change anybody's voice -- and adding a third engine does not
change an installation that had already chosen one.
"""

LABELS = {found.id: found.label for found in SPECS}
"""What each engine is called on screen. Derived from the one table."""

BLURBS = {found.id: found.blurb for found in SPECS}
"""The sentence the selector card carries. Factual, and never a superlative."""

INTERRUPT_MODES = ("cancel", "drain_unit", "cooperative")
"""What an engine can mean by "stop".

``cancel``       synthesis is actually abandoned (Kokoro, Sopro).
``drain_unit``   playback is silent immediately and the one native unit already
                 in flight is drained and discarded (PocketTTS 3.0.2).
``cooperative``  the engine is asked to stop and does, from reviewed upstream
                 support. Nothing declares it yet; it is here so that adopting
                 an upstream cancellation implementation is a capability change
                 rather than a new command (section 21.7).
"""


class EngineError(RuntimeError):
    """An engine-scoped operation that was refused. Never fatal."""


class ActiveEngineMismatch(EngineError):
    """A request that named an engine which is not the selected one.

    Its own class because the API answers it differently: this is not "that
    failed", it is "the page you are looking at is out of date", and the browser
    reloads its panel rather than showing an error.
    """


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


def spec(engine: str = "") -> EngineSpec:
    """The :class:`EngineSpec` for ``engine``, or a refusal. The one lookup.

    Every operation below that used to ask "is this Sopro?" asks this instead,
    which is what makes a fourth engine a row in :data:`SPECS` rather than a
    search for every place two engines were assumed.
    """
    wanted = check(engine or active())
    for found in SPECS:
        if found.id == wanted:
            return found
    raise EngineError(f"{engine!r} is not a text-to-speech engine this build has.")


def _module(name: str):
    """One engine's module, imported now rather than at start-up.

    By string, through :func:`importlib.import_module`, because the alternative
    is a table of imports at module scope -- and that would make importing *this*
    module import Torch, sherpa-onnx and every model registry in the build, on
    the path that draws a settings page.
    """
    return importlib.import_module(str(name))


# --------------------------------------------------------------------------- #
# The selector
# --------------------------------------------------------------------------- #


def active() -> str:
    """The selected TTS engine id. Never raises, never anything but a known id.

    Read on the path that decides whether a reply is spoken and by every surface
    that decides what to draw, so a host that will not answer, an option that
    was never registered and a value somebody hand-edited into config.json all
    mean the same thing: Kokoro, which is what this installation had before any
    of the other engines existed.
    """
    try:
        from modules import shared

        found = str(getattr(shared.opts, OPT_ENGINE, "") or "").strip().casefold()
    except Exception:
        return DEFAULT_ENGINE
    return found if found in ENGINES else DEFAULT_ENGINE


def label(engine: str = "") -> str:
    return LABELS.get(str(engine or active()), LABELS[DEFAULT_ENGINE])


def check(engine: str) -> str:
    """``engine`` as a known id, or a refusal. The one validator."""
    wanted = str(engine or "").strip().casefold()
    if wanted not in ENGINES:
        raise EngineError(f"{engine!r} is not a text-to-speech engine this build has.")
    return wanted


def select(engine: str) -> dict:
    """Change the active engine. The whole runtime boundary, in order.

    Section 4's seven steps, and the order is the design rather than
    convenience:

        1  cancel any speech turn that is running, because a reply half spoken
           in Kokoro must not finish in Sopro;
        2  stop the TTS worker that is running, so the inactive engine consumes
           no RAM (section 58);
        3  persist the new id;
        4  leave every inactive-engine setting on disk untouched (I-3, I-PKT-3),
           which is achieved by not writing any;
        5  never start a download, and
        6  never start a model load -- selecting an uninstalled engine is
           allowed, and shows that engine's install surface (section 17);
        7  never switch back on its own.

    STT is deliberately absent from all seven. Nothing below touches Whisper,
    its tier, or the microphone -- I-7 and I-PKT-5, and
    ``tests/test_voice_engines.py`` proves it by watching the STT lifecycle
    across a switch.

    A switch is a *lifecycle* boundary and not a user Stop, which matters now
    that one engine's Stop is a drain: the turn is retired and the worker is
    stopped outright rather than waited for (I-PKT-13, section 21.6).

    Returns the state the surfaces should redraw from, so a caller paints the
    truth rather than what it hoped it had written.
    """
    wanted = check(engine)
    current = active()
    if wanted == current:
        return state()

    # Cancel first, stop second. A worker stopped while a turn still believes
    # it is being spoken leaves a browser waiting on a stream that will never
    # produce another byte -- which is the bug ``unload`` was written around.
    _quiet_all(f"the text-to-speech engine changed to {LABELS[wanted]}")
    _stop_all()
    _forget_lab()

    _remember(OPT_ENGINE, wanted)
    logger.info("Model Chain: the text-to-speech engine is now %s", LABELS[wanted])
    return state()


def _quiet_all(reason: str) -> None:
    try:
        import mc_voice_turn as turns

        turns.forget_all(reason)
    except Exception:
        logger.debug("Model Chain: could not cancel voice turns for an engine switch",
                     exc_info=True)


def _stop_all() -> None:
    """Stop every TTS worker, whichever is running. Never raises.

    Every registered engine rather than "the one that was active", because the
    state this has to reach is *no TTS worker running* and asking which one it
    was is one more thing that can be wrong. Driven by :data:`SPECS` so that an
    engine added later is stopped by having been registered, not by somebody
    remembering to add a third module name here. None of the calls starts
    anything.
    """
    for found in SPECS:
        try:
            _module(found.runtime).stop("the text-to-speech engine changed")
        except Exception:
            logger.debug("Model Chain: could not stop %s for an engine switch",
                         found.runtime, exc_info=True)


def _forget_lab() -> None:
    """Discard every Voice Lab session. Section 39.

    Switching engines destroys Lab state -- not because the state is dangerous
    where it is, but because a Lab session that survived would be one pointing at
    a voice library that is no longer the active engine's. The Lab belongs to
    Sopro (it is the only engine that declares ``voice_lab``), and it is cleared
    on any switch rather than on switches away from Sopro specifically, because
    "no Lab session outlives a selector change" is the simpler thing to be sure
    of. Imported lazily so that an installation which has never selected Sopro
    does not load the module to find out it has nothing to clear.
    """
    try:
        import mc_voice_lab as lab

        lab.forget_all("the text-to-speech engine changed")
    except Exception:
        logger.debug("Model Chain: could not discard the Voice Lab sessions", exc_info=True)


def state() -> dict:
    """What every engine-scoped surface draws its frame from.

    One shape for every engine: the id, the label, whether it is installed, and
    the two lines a panel needs before it knows anything else. Deliberately
    small -- the operational detail belongs to the active engine's own module,
    and putting it here would be the cross-engine payload section 5 forbids.

    The selector is the one neutral surface allowed to name every engine
    (T-ENG-P8), which is why this is the one payload that carries all three.
    """
    chosen = active()
    return {
        "active": chosen,
        "label": LABELS[chosen],
        "engines": [{"id": found.id, "label": found.label, "blurb": found.blurb,
                     "active": found.id == chosen, "installed": installed(found.id)}
                    for found in SPECS],
        "installed": installed(chosen),
    }


def installed(engine: str = "") -> bool:
    """Whether ``engine`` is installed. Reads disk; never starts anything.

    Asked of the engine's own adapter rather than decided here, because "is it
    installed" is a different question for each of them -- a bundled ONNX bank,
    an unpacked PyTorch closure, a pinned Pocket runtime plus its model plus its
    official voice states -- and a facade that answered it would be a facade
    that had to be edited every time one of those changed.

    Section 17: reading status must never start a download or a model load. Every
    adapter's ``status()`` is a file-system question with no side effects, and an
    engine whose module will not even import -- which is the ordinary state of an
    engine this build has registered but nobody has installed -- answers False
    rather than raising.
    """
    wanted = str(engine or active())
    try:
        return bool(adapter(wanted).status().ready)
    except Exception:
        logger.debug("Model Chain: could not read whether %s is installed", wanted,
                     exc_info=True)
        return False


def capabilities(engine: str = "") -> dict:
    """What ``engine`` can actually do, as the engine itself declares it.

    Behaviour, not decoration (section 8). It is read to decide whether a route
    exists, whether a panel is drawn, and -- for ``interrupt_mode`` -- what Stop
    is allowed to promise. Every key is always present, so a caller reads a
    value rather than guessing from an absence.

    An engine that cannot be imported answers the conservative shape: nothing it
    can do, and ``cancel`` for interruption, because a caller that believed an
    uninstalled engine drains would show a waiting state nothing will ever clear.
    """
    wanted = str(engine or active())
    found = {"clone_preview": False, "rebuild": False, "engine_settings": False,
             "starter_voices": False, "voice_lab": False, "interrupt_mode": "cancel"}
    try:
        offered = adapter(wanted).capabilities()
    except Exception:
        logger.debug("Model Chain: could not read %s's capabilities", wanted, exc_info=True)
        return found
    for key in found:
        if key in (offered or {}):
            found[key] = offered[key]
    if found["interrupt_mode"] not in INTERRUPT_MODES:
        logger.debug("Model Chain: %s declared an interrupt mode this build does not "
                     "implement", wanted)
        found["interrupt_mode"] = "cancel"
    return found


def interrupt_mode(engine: str = "") -> str:
    """What Stop means on ``engine``. One of :data:`INTERRUPT_MODES`.

    Frozen onto a turn when the turn is created, like the voice and the profile,
    so that a turn started under one engine is never interrupted under another
    engine's rules (I-PKT-10).
    """
    return capabilities(engine)["interrupt_mode"]


def refusals(engine: str = "") -> tuple:
    """The exception types ``engine``'s adapter raises to *refuse* rather than fail.

    A refusal is a state: no voice has been created on this engine yet, or the
    character's voice was deleted. It is ordinary, it is what the design's own
    failure table says happens, and it stays true on every turn until somebody
    acts on it. A fault is none of those things.

    Keeping the two apart is what stops either burying the other. Logged as one
    kind, a broken voice bank is indistinguishable from an engine nobody has
    made a voice for yet -- and the first user to meet the second got a full
    traceback per assistant reply, which is how a log stops being readable.

    Asked of the adapter, because which exception an engine refuses with is a
    fact about that engine. "Not Sopro means the Kokoro registry" was true while
    there were two engines and became wrong the moment there were three.
    """
    wanted = str(engine or active())
    found = [EngineError]
    try:
        found.extend(adapter(wanted).refusals())
    except Exception:
        logger.debug("Model Chain: could not read %s's refusal types", wanted,
                     exc_info=True)
    return tuple(found)


def refuse_mismatch(engine: str) -> str:
    """``engine`` if it is the active one, or raise. Section 5, server-side.

    An empty ``engine`` means "whatever is active" and is how every ordinary
    caller reaches this -- the check exists for the request that *names* one,
    which is the request that came from a page drawn before somebody switched.
    """
    wanted = str(engine or "").strip().casefold()
    if not wanted:
        return active()
    if wanted not in ENGINES:
        raise EngineError(f"{engine!r} is not a text-to-speech engine this build has.")
    if wanted != active():
        raise ActiveEngineMismatch(
            f"{LABELS[wanted]} is no longer the selected text-to-speech engine, so that "
            f"was not applied. Reload the page to see {LABELS[active()]}.")
    return wanted


def scope(payload: dict, engine: str = "") -> dict:
    """A status payload with only the active engine's operational state in it.

    The mechanical half of section 5. A caller builds whatever it likes and
    passes it through here, and what leaves carries the active engine's block
    under ``engine`` plus the neutral fields -- never an inactive engine's
    voices, clone controls, precision, sampling, languages, Lab, drain state or
    status.

    Written as a filter rather than as a rule each caller follows, because
    "remember not to include the other engine" is the kind of instruction that
    survives exactly one new field -- and it now has two other engines to
    survive rather than one.
    """
    chosen = str(engine or active())
    found = {key: value for key, value in dict(payload or {}).items()
             if key not in ENGINES}
    found["engine"] = chosen
    found["engine_label"] = LABELS.get(chosen, chosen)
    if chosen in (payload or {}):
        found[chosen] = payload[chosen]
    return found


# --------------------------------------------------------------------------- #
# Voice identity
# --------------------------------------------------------------------------- #


def qualify(voice_id: str, engine: str = "") -> str:
    """A stable id with its backend in front of it, adding one if it is missing.

    An id that already names an engine is returned untouched, including one
    naming the *other* engine -- this is a spelling function, not a policy one,
    and silently re-badging a Kokoro id as Sopro because Sopro happens to be
    selected is precisely the cross-engine confusion I-2 forbids.
    """
    text = str(voice_id or "").strip()
    if not text:
        return ""
    head, _, rest = text.partition(":")
    head = head.casefold()
    if head in ENGINES:
        # An engine name with nothing after it is not a voice, and there is no
        # honest way to make one out of it: it can arrive from a hand-edited
        # option or an older build's stored value, and every caller here already
        # treats "" as "nothing is chosen" and falls back. Producing "kokoro:"
        # instead would be producing an id that matches no voice and looks like
        # it should.
        return f"{head}:{rest}" if rest else ""
    return f"{str(engine or DEFAULT_ENGINE)}:{text}"


def engine_of(voice_id: str) -> str:
    """Which engine owns ``voice_id``. Unqualified means Kokoro (section 12)."""
    head = str(voice_id or "").strip().split(":", 1)[0].casefold()
    return head if head in ENGINES else DEFAULT_ENGINE


def native(voice_id: str) -> str:
    """The engine's own half of a qualified id, with the backend removed.

    What an adapter is handed. ``kokoro:official:af_heart`` becomes
    ``official:af_heart``, which is exactly what :mod:`mc_voice_registry` has
    always spoken, so the Kokoro registry did not have to learn a new dialect
    to gain a prefix.
    """
    text = str(voice_id or "").strip()
    head = text.split(":", 1)[0].casefold()
    if head in ENGINES and ":" in text:
        return text.split(":", 1)[1]
    return text


def belongs(voice_id: str, engine: str = "") -> bool:
    """Whether ``voice_id`` is one of ``engine``'s. Used before resolving one."""
    return engine_of(voice_id) == str(engine or active())


# --------------------------------------------------------------------------- #
# Engine-scoped character state
# --------------------------------------------------------------------------- #


def character_voice(character, engine: str = "") -> str:
    """The stable voice id ``character`` asks for on ``engine``, or ``""``.

    Two shapes are read and only one is written, which is what section 12's
    "lossless and not a bulk rewrite" means in practice:

        voices: {kokoro: ..., sopro: ...}   the engine-aware shape
        voice: "official:af_heart"          every character file written before

    A legacy ``voice`` is Kokoro's and answers only for Kokoro. Asked for Sopro,
    a character that has never been given a Sopro voice answers ``""`` -- which
    is inheritance by absence (I-4) and means "follow the Sopro default", not
    "translate the Kokoro one".

    Read defensively off whatever the panel handed over, because this runs
    inside the generator that produces a reply and a character object from an
    older build must not raise there.

    Reading only. The *writing* side lives on the character dataclass itself --
    ``Character.voice_fields`` -- because which fields a character file has is a
    fact about the format rather than about the engine facade, and two functions
    that both knew how to write one would be two functions to keep in step.
    """
    wanted = str(engine or active())
    try:
        found = getattr(character, "voices", None)
        if isinstance(found, dict):
            chosen = str(found.get(wanted) or "").strip()
            if chosen:
                return qualify(chosen, wanted)
    except Exception:
        logger.debug("Model Chain: could not read a character's engine voices", exc_info=True)
    if wanted != KOKORO:
        return ""
    try:
        legacy = str(getattr(character, "voice", "") or "").strip()
    except Exception:
        return ""
    return qualify(legacy, KOKORO) if legacy else ""


def character_profile(character, engine: str = "") -> dict:
    """``character``'s delivery overrides for ``engine``, or an empty set.

    Empty is the ordinary case and is not a failure: it is what every character
    written before this existed has, and what makes them follow that engine's
    current defaults rather than freezing today's values into a file.
    """
    wanted = str(engine or active())
    try:
        found = getattr(character, "voice_profiles", None)
        if isinstance(found, dict) and isinstance(found.get(wanted), dict):
            return dict(found[wanted])
    except Exception:
        logger.debug("Model Chain: could not read a character's engine profiles",
                     exc_info=True)
    if wanted != KOKORO:
        return {}
    try:
        legacy = getattr(character, "voice_profile", None)
        return dict(legacy) if isinstance(legacy, dict) else {}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# The facade
# --------------------------------------------------------------------------- #


def adapter(engine: str = ""):
    """The module that owns ``engine``'s voices, defaults and synthesis.

    The contract section 8 names -- ``entries``, ``lookup``, ``default_id``,
    ``set_default``, ``resolve``, ``rename``, ``delete``, ``capabilities``,
    ``refusals``, ``public_status`` -- with every function speaking qualified
    ids. Looked up in :data:`SPECS` and imported lazily rather than branched on:
    :mod:`mc_voice_sopro` reads a manifest and a data root and
    :mod:`mc_voice_pocket` reads three, and a module that did that at import
    would make an installation with neither slower to start for a feature it
    does not use.
    """
    return _module(spec(engine).adapter)


def runtime(engine: str = ""):
    """The module that owns ``engine``'s worker process and its lifecycle.

    Every runtime answers the same small set of questions -- start, speak a
    turn, interrupt a turn, stop, and say what it is -- and answers them
    differently enough that they are three processes rather than one with
    branches (section 49.1).
    """
    return _module(spec(engine).runtime)


def profiles(engine: str = ""):
    """The module that defines ``engine``'s delivery controls and their ranges.

    One module per engine rather than one with branches, because common labels
    do not imply shared storage (section 35, I-PKT-23). Speed means a Kokoro
    ``generate`` argument on one side and a pitch-preserving time-stretch on the
    other two, and a single ``CONTROLS`` table would have had to lie about at
    least one of them.
    """
    return _module(spec(engine).profiles)


def resolve(voice_id: str = "", engine: str = ""):
    """``(qualified id, entry)`` for the active engine. Never crosses engines.

    An id belonging to another engine is not resolved and is not translated:
    it is treated as absent, so the caller falls back to *this* engine's default
    and the surface says the character's voice is missing. I-2, I-PKT-2 and
    section 7 in one branch -- there is no path from here to another engine's
    bank, and a third engine does not add one.
    """
    wanted = check(engine or active())
    chosen = str(voice_id or "").strip()
    if chosen and not belongs(chosen, wanted):
        logger.info("Model Chain: a %s voice was asked for while %s is selected, so the "
                    "%s default was used instead", engine_of(chosen), LABELS[wanted],
                    LABELS[wanted])
        chosen = ""
    return adapter(wanted).resolve(chosen)


def _remember(name: str, value) -> None:
    try:
        from modules import shared

        shared.opts.set(name, value)
        shared.opts.save(shared.config_filename)
    except Exception:
        logger.debug("Model Chain: could not persist the text-to-speech engine",
                     exc_info=True)
