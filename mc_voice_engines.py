"""Which text-to-speech engine is selected, and what a voice id means now.

Voice Chat had one TTS engine, so "voice" could be a Kokoro registry id and
"which engine" was not a question anybody had to ask. Sopro V2 ends both of
those assumptions, and this module is where they end -- at the storage boundary,
once, rather than patched at every caller.

    ONE TTS ENGINE IS SELECTED FOR THE WHOLE WEBUI AT A TIME.

That is the product rule (I-1) and it is the only rule this module enforces
directly. Everything else here exists to make it survivable: a stable id that
says which engine owns a voice, an engine-scoped view of the default and of a
character, and a facade so that a caller who wants to speak asks *the active
engine* rather than asking Kokoro.

Speech-to-text is not in this selector and never will be. Whisper has its own
process, its own dependency closure and its own lifecycle, and I-7 says
switching Kokoro to Sopro must not reload it, change its quality tier or reset
the microphone. Nothing in this module touches :mod:`mc_voice_models`' STT
state, and ``tests/test_voice_engines.py`` asserts that the switch below does
not.

What a voice id is
------------------
    kokoro:official:af_heart
    kokoro:clone:<uuid>
    sopro:clone:<uuid>

Backend first, always, so that no caller outside an adapter can be handed a
voice and not know whose it is. A Sopro id contains a server-generated UUID and
never a filesystem path, a display name or anything a browser supplied
(I-10, section 57).

Legacy ids -- ``official:af_heart``, ``clone:<uuid>``, and the bare speaker
names V1 wrote -- are Kokoro's, read as Kokoro's, and are *not* rewritten on
sight. Migration is by reading (section 12): a character file written in 2025
resolves correctly today and is only rewritten when somebody edits it. That is
what makes the upgrade lossless rather than a bulk rewrite of every character
in somebody's library.

Why the inactive engine is absent rather than hidden
----------------------------------------------------
Section 5 asks for the payload to be scoped, not the CSS. :func:`scope` and
:func:`refuse_mismatch` are how that is enforced on the server: a page that was
open when the engine changed, a theme script that re-rendered half a panel, or
a request replayed from a stale DOM cannot mutate the inactive engine's
operational settings. They get an active-engine mismatch answer instead, which
is a sentence rather than a silent write into state nobody is looking at.
"""

from __future__ import annotations

import logging

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

ENGINES = (KOKORO, SOPRO)
"""Every engine id this build knows, in selector order.

Kokoro first because it is the default and the one an upgrade lands on. A
stored value outside this tuple means Kokoro -- see :func:`active`.
"""

DEFAULT_ENGINE = KOKORO
"""Where an installation with no stored choice starts.

Section 12: if no active engine setting exists, the default is Kokoro, so an
upgrade does not change anybody's voice.
"""

LABELS = {KOKORO: "Kokoro", SOPRO: "Sopro V2"}
"""What each engine is called on screen. The one place both names appear."""

BLURBS = {
    KOKORO: "The built-in speaker bank, through sherpa-onnx. Fast, small, and installed "
            "already if you have been using Voice Chat.",
    SOPRO: "A streaming model that clones a voice from a short recording you make here. "
           "Installed separately, runs on the CPU, and brings its own runtime.",
}


class EngineError(RuntimeError):
    """An engine-scoped operation that was refused. Never fatal."""


class ActiveEngineMismatch(EngineError):
    """A request that named an engine which is not the selected one.

    Its own class because the API answers it differently: this is not "that
    failed", it is "the page you are looking at is out of date", and the browser
    reloads its panel rather than showing an error.
    """


# --------------------------------------------------------------------------- #
# The selector
# --------------------------------------------------------------------------- #


def active() -> str:
    """The selected TTS engine id. Never raises, never anything but a known id.

    Read on the path that decides whether a reply is spoken and by every surface
    that decides what to draw, so a host that will not answer, an option that
    was never registered and a value somebody hand-edited into config.json all
    mean the same thing: Kokoro, which is what this installation had before
    Sopro existed.
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
        4  leave every inactive-engine setting on disk untouched (I-3), which
           is achieved by not writing any;
        5  never start a download, and
        6  never start a model load -- selecting an uninstalled engine is
           allowed, and shows that engine's install surface (section 17);
        7  never switch back on its own.

    STT is deliberately absent from all seven. Nothing below touches Whisper,
    its tier, or the microphone -- I-7, and ``tests/test_voice_engines.py``
    proves it by watching the STT lifecycle across a switch.

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

    Both rather than "the one that was active", because the state this has to
    reach is *no TTS worker running* and asking which one it was is one more
    thing that can be wrong. Neither call starts anything.
    """
    for module in ("mc_voice_runtime", "mc_voice_sopro_runtime"):
        try:
            __import__(module).stop("the text-to-speech engine changed")
        except Exception:
            logger.debug("Model Chain: could not stop %s for an engine switch", module,
                         exc_info=True)


def _forget_lab() -> None:
    """Discard every Voice Lab session. Section 39.

    Switching away from Sopro destroys Lab state -- not because the state is
    dangerous where it is, but because a Lab session that survived would be one
    pointing at a voice library that is no longer the active engine's. Imported
    lazily so that an installation which has never selected Sopro does not load
    the module to find out it has nothing to clear.
    """
    try:
        import mc_voice_lab as lab

        lab.forget_all("the text-to-speech engine changed")
    except Exception:
        logger.debug("Model Chain: could not discard the Voice Lab sessions", exc_info=True)


def state() -> dict:
    """What every engine-scoped surface draws its frame from.

    One shape for both engines: the id, the label, whether it is installed, and
    the two lines a panel needs before it knows anything else. Deliberately
    small -- the operational detail belongs to the active engine's own module,
    and putting it here would be the cross-engine payload section 5 forbids.
    """
    chosen = active()
    return {
        "active": chosen,
        "label": LABELS[chosen],
        "engines": [{"id": name, "label": LABELS[name], "blurb": BLURBS[name],
                     "active": name == chosen, "installed": installed(name)}
                    for name in ENGINES],
        "installed": installed(chosen),
    }


def installed(engine: str = "") -> bool:
    """Whether ``engine`` is installed. Reads disk; never starts anything.

    Section 17: reading status must never start a download or a model load. Both
    branches below are file-system questions with no side effects, and both
    answer False rather than raising on an installation that is mid-upgrade.
    """
    wanted = str(engine or active())
    try:
        if wanted == SOPRO:
            import mc_voice_sopro as sopro

            return bool(sopro.status().ready)
        import mc_voice_models as models

        return bool(models.status().tts_ready)
    except Exception:
        logger.debug("Model Chain: could not read whether %s is installed", wanted,
                     exc_info=True)
        return False


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
    """
    wanted = str(engine or active())
    found = [EngineError]
    try:
        if wanted == SOPRO:
            import mc_voice_sopro as sopro

            found.append(sopro.SoproError)
        else:
            import mc_voice_registry as registry

            found.append(registry.RegistryError)
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
    under ``engine`` plus the neutral fields -- never the inactive engine's
    voices, clone controls, precision, sampling, languages, Lab or status.

    Written as a filter rather than as a rule each caller follows, because
    "remember not to include the other engine" is the kind of instruction that
    survives exactly one new field.
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
    head = text.split(":", 1)[0].casefold()
    if head in ENGINES:
        return f"{head}:{text.split(':', 1)[1]}"
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

    The contract section 11 names -- ``entries``, ``lookup``, ``default_id``,
    ``set_default``, ``resolve``, ``rename``, ``delete`` -- with every function
    speaking qualified ids. Importing lazily rather than at module load is not
    tidiness: :mod:`mc_voice_sopro` reads a manifest and a data root, and a
    module that did that at import would make an installation with no Sopro
    slower to start for a feature it does not use.
    """
    wanted = check(engine or active())
    if wanted == SOPRO:
        import mc_voice_sopro as sopro

        return sopro
    import mc_voice_kokoro as kokoro

    return kokoro


def runtime(engine: str = ""):
    """The module that owns ``engine``'s worker process and its lifecycle."""
    wanted = check(engine or active())
    if wanted == SOPRO:
        import mc_voice_sopro_runtime as sopro_runtime

        return sopro_runtime
    import mc_voice_runtime as kokoro_runtime

    return kokoro_runtime


def profiles(engine: str = ""):
    """The module that defines ``engine``'s delivery controls and their ranges.

    Two modules rather than one with a branch, because common labels do not
    imply shared storage (section 35). Speed means a Kokoro ``generate``
    argument on one side and a pitch-preserving time-stretch on the other, and a
    single ``CONTROLS`` table would have had to lie about one of them.
    """
    wanted = check(engine or active())
    if wanted == SOPRO:
        import mc_voice_sopro_profile as sopro_profile

        return sopro_profile
    import mc_voice_profile as kokoro_profile

    return kokoro_profile


def resolve(voice_id: str = "", engine: str = ""):
    """``(qualified id, entry)`` for the active engine. Never crosses engines.

    An id belonging to the other engine is not resolved and is not translated:
    it is treated as absent, so the caller falls back to *this* engine's default
    and the surface says the character's voice is missing. I-2 and section 7 in
    one branch -- there is no path from here to the other engine's bank.
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
