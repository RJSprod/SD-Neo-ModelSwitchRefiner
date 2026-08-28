"""The two things Voice Chat remembers, and the one place they are stored.

Voice Chat V1 persists user *intent* and nothing else:

    model_chain_voice_auto_send   -- send a dictated message without asking
    model_chain_voice_auto_speak  -- read completed replies aloud

Both default off, and both live in the host's own options store. That is the
whole of the state design, and the deliberate part is what is missing from it.
Microphone permission, whether the browser has unlocked its AudioContext, which
audio source is playing, the worker's PID and the id of the request in flight
are all facts about *this page, right now*; writing any of them to a settings
file would produce a setting that is wrong the moment the page is closed and
misleading the moment it is opened somewhere else.

One store, two surfaces
-----------------------
Settings -> Voice Chat and the Voice flyout in Conversation edit the same two
options. There is no second copy: :func:`auto_send` and :func:`auto_speak` read
the host, :func:`remember` writes the host and saves the config file, and the
flyout's checkboxes are drawn from those readers every time it opens. A pair of
hidden Gradio components kept in sync with the Settings page would have been
the other design, and it is the one where the operational surface and the
settings page quietly disagree about what is switched on.

Forward compatibility
---------------------
:data:`DEFAULTS` names every key a later version may add -- model ids, a voice,
a language, thread counts -- so that UI and runtime code asks *this* module for
"which STT model" rather than repeating a literal. V1 answers with the
manifest's default in every case; V2 can make the same call answer with a
stored choice without a single call site changing.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

OPT_AUTO_SEND = "model_chain_voice_auto_send"
OPT_AUTO_SPEAK = "model_chain_voice_auto_speak"

OPTIONS = (OPT_AUTO_SEND, OPT_AUTO_SPEAK)
"""Every option this feature persists. The settings section registers exactly
these two booleans, and ``tests/test_voice_ui.py`` asserts the section and this
tuple agree -- an option registered and never read, or read and never
registered, is the shape of bug that makes a checkbox do nothing."""

DEFAULTS = {
    "auto_send": False,
    "auto_speak": False,
    # Not settings in V1 and deliberately named anyway: see the module
    # docstring. ``None`` means "whatever the manifest says is default", which
    # is the answer every V1 caller gets and the answer a V2 selector replaces.
    "stt_model_id": None,
    "tts_model_id": None,
    "tts_voice_id": None,
    "stt_language": None,
}


def auto_send() -> bool:
    """Whether a successful dictation presses Send by itself. Default off."""
    return _flag(OPT_AUTO_SEND)


def auto_speak() -> bool:
    """Whether a completed reply is read aloud. Default off."""
    return _flag(OPT_AUTO_SPEAK)


def settings() -> dict:
    """Both switches, as the flyout and the status route report them."""
    return {"auto_send": auto_send(), "auto_speak": auto_speak()}


def remember(**values) -> dict:
    """Write switches through to the host's options store, and save it.

    Called by the flyout, which is an operational surface rather than a form
    with an Apply button: somebody who turns on "Speak replies automatically"
    while talking to a character expects the next reply to be spoken, not to be
    told to visit Settings. So the write is immediate and it is a *real* write
    -- ``opts.set`` plus ``opts.save`` -- rather than a value held in a Gradio
    State that a page reload would lose.

    Best-effort against the host and never fatal: an installation whose options
    object refuses the write keeps the switch it had, and Voice Chat keeps
    working at that setting. Returns what the store says afterwards, so a caller
    redraws from the truth rather than from what it hoped it had written.
    """
    wanted = {name: bool(values[key])
              for key, name in (("auto_send", OPT_AUTO_SEND), ("auto_speak", OPT_AUTO_SPEAK))
              if values.get(key) is not None}
    if wanted:
        try:
            from modules import shared

            for name, value in wanted.items():
                shared.opts.set(name, value)
            shared.opts.save(shared.config_filename)
        except Exception:
            logger.debug("Model Chain: could not persist a Voice Chat switch", exc_info=True)
    return settings()


def _flag(name: str) -> bool:
    """One boolean option, read live, defaulting to off on any doubt at all.

    "On any doubt" is the important half. This is read on the path that decides
    whether to synthesize speech and whether to press Send for somebody, and
    both of those are things a user has to have asked for. A host that will not
    answer, an option that was never registered, and a value that is not a
    boolean all mean the same thing here: no.
    """
    try:
        from modules import shared

        value = getattr(shared.opts, name, None)
    except Exception:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() in ("true", "1", "yes", "on")
    return False
