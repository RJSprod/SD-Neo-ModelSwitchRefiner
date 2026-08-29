"""Voice Chat's controls in Conversation, and its row on the Settings page.

Three things go into a panel that already works, and the constraint on all
three is that Conversation must look and behave exactly as it did for somebody
who never installs a speech model:

    a "Voice" chip in the header, beside Threads / Character / You / Model;
    a Voice overlay, which is a fourth screen in the panel's existing one-at-a-
        time surface machinery rather than a new mechanism;
    a microphone button in the composer row, beside Send.

Nothing here does inference, downloads anything, or touches a process. It builds
components and holds the handful of handlers that read a switch or draw a status
line. The capture gesture, the audio, the playback and the composer insertion
are all in ``javascript/voice_chat.js``, where a browser is the only thing that
can do them.

Why the overlay and not a drawer
--------------------------------
I-9, and the panel's own rule. Conversation's surfaces are absolutely positioned
inside the workspace: an open one takes *no room in the layout*, so nothing it
covers can be pushed anywhere -- least of all the composer, off the bottom of a
phone. A voice drawer would have been the one control that broke the geometry
every other control in this panel obeys.

The speech marker, and why it is a closure
------------------------------------------
:func:`speech_marker` takes a function that hands over the text of a reply that
*completed*, and returns a Gradio handler. The chat panel wires that handler with
``.success(...)`` on each of the six reply-producing runs, and passes its own
"what did the last run finish with" reader in. Two things fall out of that shape:

    the dependency points one way -- the panel does not import a voice module to
    ask about conversations, and this module does not import the panel to ask
    about replies; and

    R2-1 is enforced twice. ``.success`` is Gradio's own "only if the preceding
    event did not raise", and the reader is only ever given a value by the branch
    of the streaming handler that reached a completed reply -- so a cancelled,
    failed or interrupted run produces no target even on a host whose event
    framework reports a nominally successful terminal callback.
"""

from __future__ import annotations

import logging
import time

import gradio as gr

import mc_llm_ui as ui
import mc_voice_api as api
import mc_voice_models as models
import mc_voice_runtime as runtime
import mc_voice_state as state
import mc_voice_turn as turns

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

SCREEN = "voice"
"""The name this overlay answers to in the panel's ``SCREENS`` tuple."""

MIC_GLYPH = "\U0001f3a4"

MIC_PX = 44
"""The handle, at the same 44px the paperclip beside it establishes.

The size a finger can be relied on to hit, and the size the composer row is
already built around.
"""

MIC_TRACK_PX = 2 * MIC_PX
"""The track: two handles wide, which is the "2 x 1 area" this gesture asked for.

Deliberately not wider. A slide has to be long enough that nothing does it by
accident and short enough that a thumb reaching across a phone completes it in
one movement, and one handle's travel is both.
"""

NOT_READY = "Voice Chat is not set up. Install both models in Settings → Voice Chat."


def chip():
    """The header control. A chip, because it opens configuration like the rest.

    Not a permanent on/off toggle: the two switches behind it are settings, and
    a header full of live toggles is a header where one stray tap changes what
    the next reply does.
    """
    return gr.Button("Voice", size="sm", scale=0, min_width=0,
                     elem_id=ui.ident("chat", "to-voice"),
                     elem_classes=ui.classes("chip-button"))


def microphone():
    """The composer control: a two-slot track with the microphone at its left.

    Slide the microphone to the right-hand end of the track and hold it there to
    record; let go to stop and transcribe. Press-and-hold used to do this, and it
    was the wrong gesture on the device most likely to be dictating: on Android a
    long press belongs to the operating system before it belongs to a web page,
    and it raises the context menu or the selection callout over the composer.
    A slide is a gesture no platform wants, and opening a microphone stops being
    something anybody does by brushing against a button.

    The track is drawn by this extension and the handle is an ordinary Gradio
    button, which is what keeps the failure mode gentle: a page whose JavaScript
    never ran still has a labelled control that explains what is wrong when it
    is pressed, rather than a dead patch of composer.

    Never disabled, even with nothing installed. The requested interaction is
    that using it *explains what is wrong* -- a control that is simply dead is a
    control somebody presses three times and then reports as broken.
    """
    with gr.Column(scale=0, min_width=MIC_TRACK_PX,
                   elem_id=ui.ident("chat", "voice-track"),
                   elem_classes=ui.classes("voice-track")):
        return gr.Button(MIC_GLYPH, size="sm", scale=0, min_width=MIC_PX,
                         elem_id=ui.ident("chat", "voice-mic"),
                         elem_classes=ui.classes("icon-button", "voice-mic"))


def plumbing() -> dict:
    """The four hidden boxes the browser reads. None of them carries content.

    ``turn`` is the opaque id of the reply being spoken *now*. It is what makes
    streaming speech possible without the browser ever reading the transcript:
    the browser exchanges the id for audio, and what that id stands for was
    decided on the server (section 24).

    ``token`` is the V1 completed-reply target, and it is still here as the
    explicit non-streaming fallback (section 20). Python writes at most one of
    the two for any one reply, which is what stops both mechanisms firing for
    the same answer.

    ``run_state`` is Python's half of "is the composer busy" -- ``"llm"`` while
    a reply is being generated and ``"idle"`` otherwise. The browser holds the
    other half and combines them; see :data:`mc_llm_chat_panel.LLM_RUNNING`.

    ``key`` is this process's page token, which the browser sends back on every
    voice request. Put in the page by Python because that is the one channel a
    cross-site page cannot read.
    """
    token = gr.Textbox(value="", visible=False, container=False,
                       elem_id=ui.ident("chat", "voice-token"))
    turn = gr.Textbox(value="", visible=False, container=False,
                      elem_id=ui.ident("chat", "voice-turn"))
    run_state = gr.Textbox(value="idle", visible=False, container=False,
                           elem_id=ui.ident("chat", "voice-run-state"))
    key = gr.Textbox(value=api.session_token(), visible=False, container=False,
                     elem_id=ui.ident("chat", "voice-key"))
    return {"token": token, "turn": turn, "run_state": run_state, "key": key}


# --------------------------------------------------------------------------- #
# Speech for one assistant run
# --------------------------------------------------------------------------- #


_last_run: dict = {"turn": ""}
"""Whether the run that is starting created a speech turn.

Read by :func:`speech_marker`, which is what keeps section 20's rule true: at
most one automatic-speech mechanism per reply. A run that streams produces a
turn and no completed-reply target; a run that could not stream produces a
target and no turn.
"""


def begin_speech(character=None, persona=None, opening: str = ""):
    """Create the VoiceTurn for a run that is about to start, or ``None``.

    Called from inside the Conversation generator, so every refusal here is a
    reason the reply is simply not spoken -- never a reason it is not written.
    The refusals, in order:

        Auto Speak is off                 -> nothing
        the speech models are not there   -> nothing
        the default voice does not resolve-> nothing, and Settings says why
        anything at all went wrong        -> nothing, logged, never raised

    ``opening`` is the text a continuation is extending. It is not passed to
    the segmenter as text to speak -- it is the reason the turn starts from the
    tail (section 7) -- and the caller feeds only newly generated chunks.

    The voice and the delivery are the character's where it has them and the
    default where it does not, resolved here and frozen onto the turn. A
    character that names a voice which is no longer installed falls back to the
    default rather than going unspoken, for the reason
    :func:`mc_voice_registry.default_id` gives: a deleted clone is not a reason
    for a character to stop talking.
    """
    _last_run["turn"] = ""
    try:
        if not state.auto_speak():
            # The commonest reason by far, and the one that used to be
            # indistinguishable from a broken feature: both produce a reply
            # that is simply not spoken and nothing written down anywhere.
            _quietly("\"Speak replies automatically\" is off")
            return None
        if not models.status().tts_ready:
            _quietly("the text-to-speech model is not installed")
            return None
        import mc_voice_profile as profiles
        import mc_voice_registry as registry

        sid, entry = registry.resolve(voice_of(character))
        delivery = profiles.resolve(profile_of(character))
        found = turns.create(voice_id=entry["id"], sid=sid, labels=_labels(character, persona),
                             profile=delivery)
        found.base_chars = len(str(opening or ""))
        found.start()
        _last_run["turn"] = found.id
        logger.info("Model Chain: Voice will read this reply aloud — %s, speaker %d, %s, "
                    "turn %s", entry["id"], sid, profiles.describe(delivery), found.id[:8])
        return found
    except Exception:
        # Warning rather than debug, and this is the correction that matters:
        # a failure here disables the whole feature for that reply and used to
        # leave no trace at all in a log at the default level. "Voice went
        # silent and nothing was written down" is not a diagnosable state.
        logger.warning("Model Chain: Voice Chat could not start speaking this reply",
                       exc_info=True)
        return None


def voice_of(character) -> str:
    """The stable voice id a character asks for, or ``""`` for the default.

    Read defensively off whatever the panel handed over. This runs inside the
    generator that produces a reply, and a character object from an older build,
    a ``None``, or a hand-edited file with a number where an id belongs are all
    the same answer: use the default voice.
    """
    try:
        return str(getattr(character, "voice", "") or "").strip()
    except Exception:
        return ""


def profile_of(character) -> dict:
    """A character's delivery overrides, or an empty set of them.

    Empty is the ordinary case and is not a failure: it is what every character
    written before this existed has, and it is what makes them follow the
    default voice's delivery.
    """
    try:
        found = getattr(character, "voice_profile", None)
        return dict(found) if isinstance(found, dict) else {}
    except Exception:
        return {}


def _quietly(reason: str) -> None:
    """Say once, at most every ten minutes, why a reply was not read aloud.

    Throttled because it is on the path of every assistant turn and the reason
    does not change between them -- but said at all, because the alternative is
    the silence this function was added to end.
    """
    now = time.monotonic()
    last = _quiet.get(reason)
    # ``None`` rather than a zero default, which is not the same thing: on a
    # machine whose monotonic clock is still under ten minutes -- a WebUI
    # started shortly after boot -- "last said at zero" reads as "said
    # recently", and the first and most useful line is the one that gets
    # thrown away.
    if last is not None and now - last < 600.0:
        return
    _quiet[reason] = now
    logger.info("Model Chain: Voice is not reading replies aloud — %s", reason)


_quiet: dict = {}


def _labels(character, persona) -> tuple:
    """The names ``clean_reply`` would strip from the front of a reply.

    Shared with the segmenter so that streaming cannot speak a label the panel
    is about to remove from the screen (section 9). Read defensively because
    this runs inside a generator that must not raise for a voice reason.
    """
    found = []
    for candidate in (getattr(character, "name", ""), getattr(persona, "display", ""),
                      "Assistant"):
        text = str(candidate or "").strip()
        if text:
            found.append(text)
    return tuple(found)


def cancel_speech(reason: str = "user") -> bool:
    """Stop whatever is being spoken. Idempotent; never raises.

    Called by the Conversation Stop handler, which is the server half of one
    unified Stop -- see :func:`mc_llm_chat_panel._cancel`.
    """
    try:
        return turns.cancel_active(reason)
    except Exception:
        logger.debug("Model Chain: Voice Chat could not cancel the active turn",
                     exc_info=True)
        return False


def sheet() -> dict:
    """The Voice overlay: what is ready, two switches, and a way out."""
    with gr.Column(visible=False, elem_id=ui.ident("chat", "voice"),
                   elem_classes=ui.classes("sheet", "sheet-screen")) as screen:
        with gr.Row(elem_classes=ui.classes("sheet-head")):
            back = gr.Button("‹ Back", size="sm", scale=0, min_width=76,
                             elem_classes=ui.classes("sheet-back"))
            gr.Markdown("#### Voice")
        engine = gr.HTML(engine_panel(), elem_id=ui.ident("chat", "voice-engine"))
        readiness = gr.HTML(readiness_notice(),
                            elem_id=ui.ident("chat", "voice-readiness"))
        auto_send = gr.Checkbox(label="Automatically send dictation",
                                value=state.auto_send(),
                                elem_id=ui.ident("chat", "voice-auto-send"))
        auto_speak = gr.Checkbox(label="Speak replies automatically",
                                 value=state.auto_speak(),
                                 elem_id=ui.ident("chat", "voice-auto-speak"))
        gr.Markdown(
            "Speech is transcribed and spoken on this PC, on the CPU. Nothing is sent to an "
            "online service. From a phone, the recording crosses your own connection to this "
            "WebUI. Most browsers only open a microphone on a page they consider secure — "
            "Voice Chat does not add a rule of its own, and will tell you what your browser "
            "actually said if it refuses.",
            elem_classes=ui.classes("hint"))
    return {"screen": screen, "back": back, "readiness": readiness, "engine": engine,
            "auto_send": auto_send, "auto_speak": auto_speak}


# --------------------------------------------------------------------------- #
# The character's own voice
# --------------------------------------------------------------------------- #


def character_panel() -> dict:
    """A character's voice and delivery, inside the character editor.

    The same list Settings shows, in a third of the room. That is the whole
    design constraint: the character screen is an overlay that has to work on a
    phone, and fifty-three voices laid out as settings rows would be a screen
    somebody scrolls past rather than reads. So the rows are one line each --
    name, accent, audition -- in a grid that reflows, and the group headings
    Settings uses are kept because "American" and "British" is the distinction
    somebody is actually choosing between.

    The list itself is painted by ``javascript/voice_chat.js`` from
    ``/voice/voices``, for the reason the Settings row is: it changes when a
    clone finishes and when a voice is renamed, and a Gradio dropdown rebuilt
    from Python would be a list that goes stale the moment either happens. What
    Gradio owns is the *value* -- a hidden textbox the browser writes and
    ``_save_character`` reads -- so the selection is saved with the rest of the
    character by the ordinary Save, with no second store and no second button.

    The four sliders are Gradio's own, because they are ordinary values with no
    liveness to them. They are behind a checkbox rather than always in force:
    unchecked means this character has no delivery of its own and follows
    Settings → Voice Chat, which is what every character written before this
    existed does and what most of them should keep doing.
    """
    controls = {}
    with gr.Accordion("Voice", open=False, elem_classes=ui.classes("advanced")):
        gr.Markdown(
            "Which voice reads this character's replies, and how. Saved with the character "
            "by **Save character** below.",
            elem_classes=ui.classes("hint"))
        picker = gr.HTML(character_voices_html(),
                         elem_id=ui.ident("chat", "character-voice-list"))
        # Never shown and never a control. It is where the browser puts the row
        # somebody tapped, so that Gradio's own Save reads a value rather than
        # this panel needing a save button of its own.
        chosen = gr.Textbox(value="", visible=False, container=False,
                            elem_id=ui.ident("chat", "character-voice"))
        custom = gr.Checkbox(
            label="Give this character its own delivery", value=False,
            elem_id=ui.ident("chat", "character-voice-custom"),
            info="Off, it speaks the way Settings → Voice Chat says. On, the four settings "
                 "below are this character's and nothing else changes them.")
        with gr.Group(visible=False,
                      elem_id=ui.ident("chat", "character-voice-delivery")) as delivery:
            for control in delivery_controls():
                name = control["name"]
                controls[name] = gr.Slider(
                    label=_slider_label(control),
                    minimum=control["minimum"], maximum=control["maximum"],
                    step=control["step"], value=control["default"],
                    elem_id=ui.ident("chat", f"character-voice-{name}"))
    return {"picker": picker, "chosen": chosen, "custom": custom, "delivery": delivery,
            "sliders": [controls[name] for name in _field_names() if name in controls],
            "names": [name for name in _field_names() if name in controls]}


def _slider_label(control: dict) -> str:
    """``Pitch (semitones)``. The unit goes in the label because a Gradio slider
    shows a bare number beside it and "-3" on its own is not a pitch."""
    unit = str(control.get("unit") or "").strip()
    if not unit or unit == "x":
        return str(control.get("label") or "")
    return f'{control.get("label") or ""} ({unit})'


def _field_names() -> tuple:
    try:
        import mc_voice_profile as profiles

        return tuple(profiles.FIELDS)
    except Exception:
        return ()


def character_voices_html() -> str:
    """The first frame of the compact list, and the shape the browser fills in.

    Static markup here says something true on a page whose JavaScript has not
    run and on an installation with no text-to-speech model at all; everything
    live is painted from ``/voice/voices``. No speaker id is put in the
    document, here or there -- section 56.
    """
    return (
        f'<div class="mc-voice-picker" data-mc-voice-picker '
        f'data-mc-voice-key="{ui.escape(api.session_token())}">'
        '<div class="mc-voice-picker-current" data-mc-voice-picker-current>Loading…</div>'
        '<div class="mc-voice-picker-list" data-mc-voice-picker-list></div>'
        '</div>')


def character_state(character) -> dict:
    """What the editor's voice controls should hold for ``character``.

    One function so the four places that fill the editor -- opening it, opening
    it on a new character, cancelling, and switching who you are talking to --
    cannot disagree about whether a character has a delivery of its own.
    """
    overrides = profile_of(character)
    has_own = any(value is not None for value in overrides.values())
    try:
        import mc_voice_profile as profiles

        # ``resolve`` answers with the default for every field the character
        # does not set, so this is the effective delivery in both cases -- the
        # sliders open where the sound the user is listening to actually is,
        # rather than at a neutral they are not.
        effective = profiles.resolve(overrides)
        names = profiles.FIELDS
    except Exception:
        logger.debug("Model Chain: could not read a character's delivery", exc_info=True)
        effective, names = {}, ()
    return {"voice": voice_of(character), "custom": has_own,
            "values": [effective.get(name) for name in names]}


def character_profile(custom, values) -> dict:
    """The four overrides to save, from the checkbox and the four sliders.

    Unchecked is four ``None``s rather than four defaults, and that is the
    difference the whole inheritance model rests on: a character that follows
    Settings has to keep following it when Settings changes.
    """
    names = _field_names()
    if not custom or not names:
        return {name: None for name in names}
    offered = dict(zip(names, list(values or [])))
    try:
        import mc_voice_profile as profiles

        return profiles.overrides(offered)
    except Exception:
        logger.debug("Model Chain: could not read the delivery sliders", exc_info=True)
        return {name: None for name in names}


def engine_panel() -> str:
    """The live Loaded/Unloaded block, and the button that changes it.

    Section 32. Static HTML with data attributes rather than Gradio controls,
    for the same reason the Settings install row is: this has to *poll*, and a
    Gradio component that re-rendered every second while the flyout was open
    would fight the panel for the surface it is drawn on. The browser paints it
    from ``/voice/status`` and stops asking the moment the flyout closes.

    What is rendered here is only the first frame, so a flyout opened on a page
    whose JavaScript has not run yet still says something true.
    """
    found = runtime.engine()
    labels = {
        "unloaded": "\u25cb Unloaded — loads automatically on next voice use",
        "loading": "\u25cc Loading speech models…",
        "idle": "\u25cf Loaded — CPU, idle",
        "stt": "\u25cf Loaded — Listening",
        "tts": "\u25cf Loaded — Speaking",
        "stopping": "\u25cc Unloading…",
        "error": "\u25cf The speech engine could not start",
    }
    action = "unload" if found.get("loaded") else "load"
    voice = ""
    try:
        import mc_voice_registry as registry

        entry = registry.default_entry()
        voice = entry["label"] if entry else ""
    except Exception:
        logger.debug("Model Chain: could not read the default voice", exc_info=True)
    return (
        f'<div class="mc-voice-engine">'
        f'<div class="mc-voice-engine-head">Voice engine</div>'
        f'<div class="mc-voice-engine-state" data-mc-voice-engine-line>'
        f'{ui.escape(labels.get(found.get("state") or "unloaded", labels["unloaded"]))}</div>'
        f'<button type="button" class="mc-voice-runtime" data-mc-voice-runtime="{action}">'
        f'{action.title()}</button>'
        f'<div class="mc-voice-engine-voice">Default voice: '
        f'<span data-mc-voice-default>{ui.escape(voice)}</span></div>'
        f'</div>')


def readiness_notice() -> str:
    """One line for the overlay, drawn from the same status Settings reads."""
    found = models.status()
    if found.ready:
        return ui.notice("Ready.")
    if not found.platform_supported:
        return ui.notice(found.runtime_message, "warn")
    missing = [name for name, ok in (("speech to text", found.stt_ready),
                                     ("text to speech", found.tts_ready)) if not ok]
    if not found.runtime_ready and not missing:
        return ui.notice("Setup required — install the voice models in Settings → Voice Chat.",
                         "warn")
    return ui.notice(f"Setup required — {' and '.join(missing) or 'the voice runtime'} "
                     f"still to install. Settings → Voice Chat.", "warn")


def open_sheet(screens) -> list:
    """Open the overlay and refresh what it says.

    Read live rather than from what the checkboxes happened to hold: Settings
    may have changed either switch since this panel was built, and a flyout that
    shows a stale value is a flyout that turns a setting off by being tapped.
    """
    current = state.settings()
    return screens(SCREEN) + [readiness_notice(), engine_panel(),
                              gr.update(value=current["auto_send"]),
                              gr.update(value=current["auto_speak"])]


def set_auto_send(value):
    """Persist immediately. Section 43: no Apply, no second copy of the truth."""
    return _remember(auto_send=bool(value), key="auto_send")


def set_auto_speak(value):
    """Persist immediately -- and stop a reply that is being read aloud.

    Section 28. Turning the switch off while the speaker is talking has to be
    believed at once: a switch that says "do not read replies aloud" while a
    reply is being read aloud is a switch nobody trusts again. The browser does
    the same thing from its side for the audible half; this is the server half,
    and both are idempotent.

    Turning it *on* deliberately does nothing to the reply in flight. That turn
    was created without speech and cannot grow it, and starting mid-answer would
    speak from the middle of a sentence.
    """
    if not value:
        cancel_speech("auto speak off")
    return _remember(auto_speak=bool(value), key="auto_speak")


def _remember(*, key: str, **values):
    stored = state.remember(**values)
    # Answered from the store rather than echoed back, so a write the host
    # refused shows the checkbox snapping back instead of lying about it.
    return gr.update(value=stored[key])


def speech_marker(take_reply, character_named=None):
    """A success-only handler that turns a completed reply into a target token.

    ``take_reply`` is the chat panel's own record of what the run that just
    finished produced -- consumed, so one run can only ever create one target.
    Four separate refusals, and each is one of the ways this feature could
    otherwise speak something nobody asked it to:

        the run did not complete with a reply   -> nothing (R2-1, I-11)
        "Speak replies automatically" is off    -> nothing
        the TTS model is not installed          -> nothing, and the switch is
                                                   left alone (section 44)
        the snapshot is empty                   -> nothing

    Returns the empty string in every one of those cases, which the browser
    ignores, and never raises: a voice failure must not be able to cancel a
    reply that has already arrived.

    ``character_named`` is the panel's own reader -- a name in, a character out.
    It is a closure rather than an import for the reason this module's docstring
    gives: the dependency points one way, and a voice module that reached into
    the character store to ask who was talking would be pointing it both ways.
    Without one, a reply is remembered against the default voice, which is what
    this path did for every reply before characters had voices of their own.
    """

    def marker(who=""):
        try:
            text = take_reply()
            if not text or not str(text).strip():
                return ""
            if _last_run.get("turn"):
                # This reply was streamed. Producing a completed-reply target as
                # well would speak it twice -- section 20 forbids two automatic
                # mechanisms that can both fire for one reply, and the server is
                # where that is decided rather than in the browser.
                _last_run["turn"] = ""
                return ""
            if not state.auto_speak():
                return ""
            if not models.status().tts_ready:
                return ""
            # Which voice, and how, snapshotted with the words -- see
            # :func:`mc_voice_api.remember_reply`. Resolved now rather than when
            # the browser asks for audio, because "which character was this"
            # is a question only this moment is certain of the answer to.
            character = None
            if character_named is not None:
                try:
                    character = character_named(who)
                except Exception:
                    logger.debug("Model Chain: could not read the character to speak as",
                                 exc_info=True)
            import mc_voice_profile as profiles

            return api.remember_reply(str(text), voice_id=voice_of(character),
                                      profile=profiles.resolve(profile_of(character)))
        except Exception:
            logger.debug("Model Chain: Voice Chat could not prepare a spoken reply",
                         exc_info=True)
            return ""

    return marker


# --------------------------------------------------------------------------- #
# The Settings page row
# --------------------------------------------------------------------------- #


def _manual_section(kind: str, addresses: list, blurb: str, placeholder: str,
                    model: str = "") -> str:
    """The "or install from files you download yourself" half of a row.

    Every row has one now, the engine included: the failure that made the engine
    row necessary in the first place was an automatic install that could not be
    completed, and an escape hatch that covers two of the three things a person
    needs is not an escape hatch.

    ``model`` names a particular bundle where a kind has more than one -- the
    three speech-to-text qualities -- and becomes the *scope* the folder box and
    its button answer to. The scope has to be per bundle rather than per kind or
    the three folder boxes on the page would all be found by the same selector,
    and pressing Install under the high tier would read whatever was typed under
    the low one. The request still carries the kind, because that is what the
    install route installs; the model rides beside it.
    """
    if not addresses:
        return ""
    scope = str(model or kind)
    links = "".join(
        f'<li><a href="{ui.escape(item["url"])}" target="_blank" rel="noreferrer">'
        f'{ui.escape(item["filename"])}</a>'
        + ("" if item.get("archive") or item["filename"] == item["save_as"]
           else f' <span class="mc-voice-rename">→ save as '
                f'{ui.escape(item["save_as"])}</span>')
        + "</li>"
        for item in addresses)
    return (
        f'<details class="mc-voice-manual">'
        f'<summary>Or install from files you download yourself</summary>'
        f'<p>{blurb}</p>'
        f'<ul class="mc-voice-links">{links}</ul>'
        f'<div class="mc-voice-folder-row">'
        f'<input type="text" class="mc-voice-folder" '
        f'data-mc-voice-folder="{ui.escape(scope)}" spellcheck="false" '
        f'placeholder="{ui.escape(placeholder)}" />'
        f'<button type="button" class="mc-voice-install-local" '
        f'data-mc-voice-local="{ui.escape(kind)}" '
        f'data-mc-voice-scope="{ui.escape(scope)}" '
        f'data-mc-voice-model="{ui.escape(str(model or ""))}">'
        f'Install from this folder</button>'
        f'</div>'
        f'<p class="mc-voice-note">Nothing is installed on trust: a file under the right '
        f'name with the wrong contents is refused exactly as a bad download is.</p>'
        f'</details>')


def _tier_row(found) -> str:
    """The speech-to-text row: three qualities, and one of them in use.

    A card each rather than a drop-down, because the choice is not between three
    names -- it is between a fast one that mishears, a heavy one that does not,
    and the one in the middle, and nobody can make that choice from a list of
    labels. So each card carries what it costs to download, what it costs in
    memory while it is loaded, and a sentence about what it is good and bad at.

    Two buttons and they are deliberately separate. **Download** fetches that
    tier. **Use this** points Voice Chat at it. Splitting them is what lets
    somebody keep all three on disk and switch between them without a download,
    and what lets somebody choose the high tier and *then* start the download
    for it rather than being made to do it in the other order.

    The sizes are approximate and say so. The model bundles are not pinned in
    this repository -- ``mc_voice_models`` explains why at length -- so the
    exact figure is not known until the publisher is asked at install time, and
    a number presented as exact that turned out to be 12 MB off would be a
    worse answer than "about".
    """
    if not found.platform_supported:
        return (f'<div class="mc-voice-row" data-mc-voice-kind="stt">'
                f'<div class="mc-voice-head">'
                f'<div class="mc-voice-heading">Speech to text</div>'
                f'<div class="mc-voice-status" data-mc-voice-status="stt">'
                f'{ui.escape(found.stt_message)}</div></div></div>')
    try:
        tiers = models.catalogue("stt")
    except Exception:
        logger.debug("Model Chain: could not describe the speech-to-text tiers",
                     exc_info=True)
        tiers = []

    cards = "".join(_tier_card(entry) for entry in tiers)
    return (
        f'<div class="mc-voice-row" data-mc-voice-kind="stt">'
        f'<div class="mc-voice-head">'
        f'<div class="mc-voice-heading">Speech to text</div>'
        f'<div class="mc-voice-default" data-mc-voice-chosen="stt">'
        f'{ui.escape(found.stt_label)}</div>'
        f'<div class="mc-voice-status" data-mc-voice-status="stt">'
        f'{ui.escape(found.stt_message)}</div>'
        f'</div>'
        f'<p class="mc-voice-note">Three qualities of the same transcriber. They differ in '
        f'how often they mishear a name or an accent, in how long a sentence takes on the '
        f'CPU, and in how much memory they hold while they are loaded — all of which is on '
        f'top of whatever the language and image models are already using. Download as many '
        f'as you like; only the one marked <em>In use</em> is loaded. Sizes are approximate '
        f'until the download starts, when the publisher is asked for the exact figure.</p>'
        f'<div class="mc-voice-tiers" data-mc-voice-tiers="stt">{cards}</div>'
        f'</div>')


def _tier_card(entry: dict) -> str:
    """One quality, everything it costs, and the two buttons for it."""
    identifier = str(entry.get("id") or "")
    facts = []
    if entry.get("about_label"):
        facts.append(f"about {entry['about_label']} to download")
    if entry.get("ram_label"):
        facts.append(f"about {entry['ram_label']} of memory while it is loaded")
    return (
        f'<div class="mc-voice-tier{" mc-voice-tier-chosen" if entry.get("chosen") else ""}" '
        f'data-mc-voice-tier="{ui.escape(identifier)}">'
        f'<div class="mc-voice-tier-head">'
        f'<span class="mc-voice-tier-rank">{ui.escape(entry.get("tier_label") or "")}</span>'
        f'<span class="mc-voice-tier-name">{ui.escape(entry.get("label") or "")}</span>'
        f'<span class="mc-voice-tier-mark" data-mc-voice-tier-mark>'
        f'{"In use" if entry.get("chosen") else ""}</span>'
        f'</div>'
        f'<div class="mc-voice-tier-summary">{ui.escape(entry.get("summary") or "")}</div>'
        f'<div class="mc-voice-tier-facts">{ui.escape(" · ".join(facts))}</div>'
        f'<div class="mc-voice-tier-notes">{ui.escape(entry.get("notes") or "")}</div>'
        f'<div class="mc-voice-tier-state" data-mc-voice-tier-state>'
        f'{ui.escape(entry.get("message") or "")}</div>'
        f'<div class="mc-voice-tier-actions">'
        f'<button type="button" class="mc-voice-install" '
        f'data-mc-voice-tier-install="{ui.escape(identifier)}">'
        f'{"Installed" if entry.get("installed") else "Download"}</button>'
        f'<button type="button" class="mc-voice-entry-action" '
        f'data-mc-voice-tier-use="{ui.escape(identifier)}">Use this</button>'
        f'</div>'
        + _manual_section(
            "stt", entry.get("sources") or [],
            "The Download button above does all of this for you. This is here for a machine "
            "that cannot reach the publisher — no Internet, or a proxy that will not pass a "
            "large binary. Download these three files into one folder, then give Voice Chat "
            "that folder. The original filenames are fine — nothing needs renaming, and no "
            "account or access token is needed.",
            "C:\\Users\\you\\Downloads\\voice-stt", model=identifier)
        + '</div>')


def settings_html() -> str:
    """The Voice Chat install row, as static HTML the browser makes live.

    Forge's settings system stores options; it does not host arbitrary Gradio
    controls with handlers. So the buttons are ordinary buttons in an HTML
    block, ``javascript/voice_chat.js`` wires them to the install route, and the
    same script polls the status route to redraw the rows. That is the
    mechanism the design intent recommends, and it keeps "download" from
    becoming a fake persistent boolean that a settings backup would restore as
    an instruction to download something.

    Three rows -- the engine and the two models -- and every one of them has two
    ways in. The button fetches the pinned artifacts. The folder box installs
    from files somebody already has, which is the answer for a proxy that will
    not pass a large binary, an air-gapped machine, and an automatic install
    that failed for a reason nobody has got to the bottom of yet. The addresses
    are printed beside it rather than left for somebody to search for: "a
    Whisper small ONNX export" describes several files and only one of them is
    the one this runtime wants.

    The page token is an attribute on the container because this row is not
    inside Conversation and has no hidden Gradio component of its own to read
    it from.
    """
    found = models.status()
    parts = [f'<div class="mc-voice-settings" '
             f'data-mc-voice-key="{ui.escape(api.session_token())}">']

    # The engine gets a row of its own with a button of its own. It used to be
    # a line of text, on the reasoning that it is an implementation detail of
    # the two models and is installed by whichever is downloaded first -- which
    # is true right up until somebody installs both models from files they
    # already had. Then nothing was downloaded, the engine is still missing,
    # both model buttons read "Installed", and there is nothing left to press.
    try:
        wheels = models.runtime_sources()
    except Exception:
        logger.debug("Model Chain: could not describe the voice engine", exc_info=True)
        wheels = []
    parts.append(
        f'<div class="mc-voice-row" data-mc-voice-kind="runtime">'
        f'<div class="mc-voice-head">'
        f'<div class="mc-voice-heading">Voice engine</div>'
        f'<div class="mc-voice-default">sherpa-onnx, CPU only</div>'
        f'<div class="mc-voice-status" data-mc-voice-status="runtime">'
        f'{ui.escape(found.runtime_message)}</div>'
        f'<button type="button" class="mc-voice-install" '
        f'data-mc-voice-install="runtime">Install voice engine</button>'
        f'</div>'
        + _manual_section(
            "runtime", wheels,
            "Two wheels from PyPI, unpacked into a folder of their own. Both models need "
            "them, and downloading either model installs them too. Put both files in one "
            "folder and give Voice Chat that folder — they are pinned in this extension, "
            "so what you supply is checked against a hash committed here.",
            "C:\\Users\\you\\Downloads\\voice-engine")
        + '</div>')
    parts.append(f'<div class="mc-voice-runtime">{ui.escape(found.summary)}</div>')

    parts.append(_tier_row(found))

    for kind, heading in (("tts", "Text to speech"),):
        label, addresses = "", []
        if found.platform_supported:
            try:
                entry = models.default_model(kind)
                label = entry.label
                addresses = models.sources(kind)
            except Exception:
                logger.debug("Model Chain: could not describe the %s bundle", kind,
                             exc_info=True)
        message = found.tts_message

        parts.append(f'<div class="mc-voice-row" data-mc-voice-kind="{kind}">')
        parts.append(
            f'<div class="mc-voice-head">'
            f'<div class="mc-voice-heading">{ui.escape(heading)}</div>'
            f'<div class="mc-voice-default">{ui.escape(label)}</div>'
            f'<div class="mc-voice-status" data-mc-voice-status="{kind}">'
            f'{ui.escape(message)}</div>'
            f'<button type="button" class="mc-voice-install" '
            f'data-mc-voice-install="{kind}">Download default {kind.upper()}</button>'
            f'</div>')
        parts.append(_manual_section(
            kind, addresses,
            "The button above does all of this for you. This is here for a machine that "
            "cannot reach the publishers — no Internet, or a proxy that will not pass a "
            "large binary. Download these into one folder, then give Voice Chat that "
            "folder. The original filenames are fine — nothing needs renaming, and no "
            "account or access token is needed for either site.",
            f"C:\\Users\\you\\Downloads\\voice-{kind}"))
        parts.append("</div>")

    parts.append(
        '<div class="mc-voice-note">Voice Chat runs on the CPU and never uses the graphics '
        'card. It has no sign-in of its own and needs no account, API key or access token '
        '— not for this WebUI, and not for the sites these files come from. After they are '
        'installed it needs no Internet connection at all.</div>')
    parts.append("</div>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# The Settings voice management row
# --------------------------------------------------------------------------- #


def delivery_controls() -> list:
    """The four sliders, as data, in the order both surfaces draw them.

    Read from :mod:`mc_voice_profile` rather than repeated here, because a range
    that differed between the settings page and the character screen would be a
    setting that silently changed when it was edited in the other place.
    """
    try:
        import mc_voice_profile as profiles

        return [dict(profiles.CONTROLS[name], name=name) for name in profiles.FIELDS]
    except Exception:
        logger.debug("Model Chain: could not read the voice delivery controls", exc_info=True)
        return []


def _delivery_block() -> str:
    """How the default voice is delivered: four sliders and what they do.

    Full width and fully labelled here, because this is the settings page and
    there is room. The same four controls appear in the character screen in a
    compact form -- one grid, no help text -- for the same reason the voice list
    does: that surface is a flyout on a phone.

    Painted by the browser from ``/voice/profile`` like everything else in this
    row. The first frame is drawn here so that a page whose JavaScript has not
    run yet still says something true.
    """
    try:
        import mc_voice_profile as profiles

        current = profiles.stored()
        summary = profiles.describe(current)
    except Exception:
        logger.debug("Model Chain: could not read the voice delivery profile", exc_info=True)
        current, summary = {}, ""

    rows = []
    for control in delivery_controls():
        name = control["name"]
        value = current.get(name, control["default"])
        rows.append(
            f'<div class="mc-voice-slider" data-mc-voice-slider="{ui.escape(name)}">'
            f'<label for="mc-voice-slider-{ui.escape(name)}">'
            f'{ui.escape(control["label"])}</label>'
            f'<input type="range" id="mc-voice-slider-{ui.escape(name)}" '
            f'min="{control["minimum"]}" max="{control["maximum"]}" '
            f'step="{control["step"]}" value="{value}" '
            f'data-mc-voice-slider-input="{ui.escape(name)}" />'
            f'<output data-mc-voice-slider-value="{ui.escape(name)}">'
            f'{ui.escape(_value_label(name, value))}</output>'
            f'<div class="mc-voice-slider-help">{ui.escape(control["help"])}</div>'
            f'</div>')

    return (
        '<div class="mc-voice-row" data-mc-voice-delivery>'
        '<div class="mc-voice-head">'
        '<div class="mc-voice-heading">Delivery</div>'
        '<div class="mc-voice-default" data-mc-voice-delivery-summary>'
        f'{ui.escape(summary)}</div>'
        '</div>'
        '<p class="mc-voice-note">How the default voice speaks. A character with a delivery '
        'of its own overrides these; a character without one follows them, so changing a '
        'slider here changes every character you have not configured separately.</p>'
        + "".join(rows)
        + '<div class="mc-voice-slider-actions">'
        '<button type="button" class="mc-voice-entry-action" data-mc-voice-delivery-test>'
        'Test</button>'
        '<button type="button" class="mc-voice-entry-action" data-mc-voice-delivery-reset>'
        'Reset</button>'
        '</div>'
        '<p class="mc-voice-note">Kokoro exposes one of these itself — speed, which changes '
        'how the model articulates rather than only how fast it plays. Pitch, volume and '
        'pacing are applied by Voice Chat to the audio the model produced: pitch by '
        'resynthesising faster and reading the result back slower, which moves the formants '
        'with it and reads as a different-sized speaker. There is no emotion control, '
        'because Kokoro-82M has no emotion input — a slider for one would do nothing.</p>'
        '</div>')


def _value_label(name: str, value) -> str:
    try:
        import mc_voice_profile as profiles

        return profiles.value_label(name, value)
    except Exception:
        return str(value)


def voices_html() -> str:
    """Voice selection, auditioning, renaming, deleting, and cloning.

    A second HTML block on the Settings page, drawn and redrawn by
    ``javascript/voice_chat.js`` from ``/voice/voices`` and
    ``/voice/cloning/status``. Static markup here is the first frame and the
    shape; everything live is painted by the browser, for the same reason the
    install row is -- Forge's settings system stores options, it does not host
    Gradio controls with handlers.

    What is *not* here is any voice data. The list is fetched, so a page that
    was open when a clone finished shows it on its next paint rather than
    needing a reload, and no speaker id is ever put in the document (section 56).
    """
    return (
        f'<div class="mc-voice-voices" data-mc-voice-key="{ui.escape(api.session_token())}">'
        '<div class="mc-voice-row">'
        '<div class="mc-voice-head">'
        '<div class="mc-voice-heading">Voices</div>'
        '<div class="mc-voice-default" data-mc-voice-current>Loading…</div>'
        '</div>'
        '<div class="mc-voice-testline">'
        '<label for="mc-voice-test-text">Test text</label>'
        '<input type="text" id="mc-voice-test-text" data-mc-voice-test-text '
        'spellcheck="false" maxlength="400" />'
        '</div>'
        '<div class="mc-voice-warnings" data-mc-voice-warnings></div>'
        '<div class="mc-voice-list" data-mc-voice-list></div>'
        '</div>'
        + _delivery_block()
        + '<div class="mc-voice-row" data-mc-voice-cloning>'
        '<div class="mc-voice-head">'
        '<div class="mc-voice-heading">Voice cloning</div>'
        '<div class="mc-voice-status" data-mc-voice-cloning-status>Checking…</div>'
        '</div>'
        '<p class="mc-voice-note">Optional. Voice Chat speaks perfectly well without it, '
        'and a voice cloned once is used from then on by the ordinary speech engine — '
        'nothing extra runs while you are talking.</p>'
        '<p class="mc-voice-note mc-voice-consent">Only clone a voice you own or have '
        'permission to clone. The recording stays on this PC and is deleted when the clone '
        'finishes.</p>'
        '<div class="mc-voice-cloning-checks" data-mc-voice-cloning-checks></div>'
        '<details class="mc-voice-manual">'
        '<summary>Manual setup</summary>'
        '<p>Prepare a Storytime folder with <code>bin/storytime</code>, '
        '<code>assets/kokoro.onnx</code>, <code>assets/tokens.json</code>, '
        '<code>assets/spk_encoder.onnx</code> and <code>assets/voices/*.bin</code>, then '
        'give Voice Chat that folder.</p>'
        '<ul class="mc-voice-links" data-mc-voice-cloning-links></ul>'
        '<div class="mc-voice-folder-row">'
        '<input type="text" class="mc-voice-folder" data-mc-voice-cloning-folder '
        'spellcheck="false" placeholder="/home/you/storytime" />'
        '<button type="button" class="mc-voice-install-local" '
        'data-mc-voice-cloning-adopt>Use this folder</button>'
        '</div>'
        '</details>'
        '<div class="mc-voice-clone-form" data-mc-voice-clone-form hidden>'
        '<div class="mc-voice-field">'
        '<label for="mc-voice-clone-name">Name</label>'
        '<input type="text" id="mc-voice-clone-name" data-mc-voice-clone-name '
        'maxlength="48" spellcheck="false" />'
        '</div>'
        '<div class="mc-voice-field">'
        '<label for="mc-voice-clone-language">English</label>'
        '<select id="mc-voice-clone-language" data-mc-voice-clone-language>'
        '<option value="en-US">American</option>'
        '<option value="en-GB">British</option>'
        '</select>'
        '</div>'
        '<div class="mc-voice-field">'
        '<label for="mc-voice-clone-file">Reference recording</label>'
        '<input type="file" id="mc-voice-clone-file" accept=".wav,audio/wav,audio/x-wav" '
        'data-mc-voice-clone-file />'
        '</div>'
        '<button type="button" class="mc-voice-install" data-mc-voice-clone-start>'
        'Start cloning</button>'
        '<p class="mc-voice-note">Ten to twenty seconds, read at your normal pace in a quiet '
        'room. Cloning runs on the CPU and takes a long time — you can leave this page, and '
        'speech and images keep working while it runs.</p>'
        '</div>'
        '<div class="mc-voice-clone-job" data-mc-voice-clone-job hidden>'
        '<div class="mc-voice-clone-name" data-mc-voice-job-name></div>'
        '<div class="mc-voice-clone-state" data-mc-voice-job-state></div>'
        '<div class="mc-voice-progress"><div class="mc-voice-progress-bar" '
        'data-mc-voice-job-bar></div></div>'
        '<div class="mc-voice-clone-step" data-mc-voice-job-step></div>'
        '<button type="button" class="mc-voice-abort" data-mc-voice-clone-abort>'
        'Abort clone</button>'
        '</div>'
        '</div>'
        '</div>')
