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
        import mc_voice_engines as engines

        # Resolved once, here, and frozen onto the turn: the engine, the voice
        # and the delivery. Section 47 -- changing Settings while a reply is
        # already speaking affects the next turn, not half of the current one.
        active = engines.active()
        if not engines.installed(active):
            _quietly(f"{engines.label(active)} is not installed")
            return None
        profiles = engines.profiles(active)
        try:
            voice_id, entry = engines.resolve(voice_of(character, active), active)
        except engines.refusals(active) as exc:
            # A voice the adapter *refuses* to resolve is a state, not a fault:
            # no voice has been created on this engine yet, or the character's
            # one was deleted. Both are ordinary, both stay true on every reply
            # until somebody acts, and both used to raise into the catch-all
            # below -- so the first user to meet one got a full traceback at
            # WARNING per assistant turn, burying the failures that warning is
            # there to surface. Anything the adapter did *not* declare still
            # goes to the catch-all, because a broken voice bank is a fault and
            # has to keep reading like one.
            logger.debug("Model Chain: the voice could not be resolved", exc_info=True)
            _quietly(str(exc) or f"{engines.label(active)} has no voice to speak with")
            return None
        delivery = profiles.resolve(profile_of(character, active))
        found = turns.create(voice_id=voice_id, sid=int(entry.get("_sid") or 0),
                             handle=_handle(active, entry), engine=active,
                             labels=_labels(character, persona), profile=delivery)
        found.base_chars = len(str(opening or ""))
        found.start()
        _last_run["turn"] = found.id
        logger.info("Model Chain: Voice will read this reply aloud — %s on %s, %s, turn %s",
                    voice_id, engines.label(active), profiles.describe(delivery),
                    found.id[:8])
        return found
    except Exception:
        # Warning rather than debug, and this is the correction that matters:
        # a failure here disables the whole feature for that reply and used to
        # leave no trace at all in a log at the default level. "Voice went
        # silent and nothing was written down" is not a diagnosable state.
        logger.warning("Model Chain: Voice Chat could not start speaking this reply",
                       exc_info=True)
        return None


def _handle(engine: str, entry) -> object:
    """What the engine's adapter needs to speak this voice, and nothing more.

    A sherpa speaker number for Kokoro, the qualified stable id for Sopro. It
    is built here, at the one place that has just resolved the voice, and the
    turn carries it without ever looking inside -- which is what keeps a numeric
    SID out of the shared turn contract (I-10).
    """
    import mc_voice_engines as engines

    if str(engine) == engines.SOPRO:
        return str((entry or {}).get("id") or "")
    return int((entry or {}).get("_sid") or 0)


def voice_of(character, engine: str = "") -> str:
    """The stable voice id a character asks for on ``engine``, or ``""``.

    Read defensively off whatever the panel handed over. This runs inside the
    generator that produces a reply, and a character object from an older build,
    a ``None``, or a hand-edited file with a number where an id belongs are all
    the same answer: use that engine's default voice.

    A character configured for the other engine only answers ``""`` here, which
    is inheritance by absence rather than translation (I-4): a Kokoro speaker
    number means nothing to Sopro and pretending otherwise would be the
    cross-engine confusion I-2 forbids.
    """
    try:
        import mc_voice_engines as engines

        return engines.character_voice(character, engine or engines.active())
    except Exception:
        logger.debug("Model Chain: could not read a character's voice", exc_info=True)
        return ""


def profile_of(character, engine: str = "") -> dict:
    """A character's delivery overrides for ``engine``, or an empty set of them.

    Empty is the ordinary case and is not a failure: it is what every character
    written before this existed has, and what makes them follow that engine's
    current default delivery.
    """
    try:
        import mc_voice_engines as engines

        return engines.character_profile(character, engine or engines.active())
    except Exception:
        logger.debug("Model Chain: could not read a character's delivery", exc_info=True)
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
        import mc_voice_engines as engines

        return tuple(engines.profiles().FIELDS)
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

    Drawn from the *active* engine's saved state and from nothing else (section
    7). Switching the global engine redraws this section from that engine's own
    values; it does not translate a Kokoro pitch into a Sopro one, and a
    character with a Kokoro voice and no Sopro voice opens with no voice
    selected rather than with the wrong one.
    """
    import mc_voice_engines as engines

    active = engines.active()
    overrides = profile_of(character, active)
    has_own = any(value is not None for value in overrides.values())
    try:
        profiles = engines.profiles(active)
        # ``resolve`` answers with the default for every field the character
        # does not set, so this is the effective delivery in both cases -- the
        # sliders open where the sound the user is listening to actually is,
        # rather than at a neutral they are not.
        effective = profiles.resolve(overrides)
        names = profiles.FIELDS
    except Exception:
        logger.debug("Model Chain: could not read a character's delivery", exc_info=True)
        effective, names = {}, ()
    return {"voice": voice_of(character, active), "custom": has_own, "engine": active,
            "values": [effective.get(name) for name in names]}


def character_profile(custom, values) -> dict:
    """The overrides to save, from the checkbox and the sliders.

    Unchecked is every field ``None`` rather than every field at today's
    default, and that is the difference the whole inheritance model rests on: a
    character that follows Settings has to keep following it when Settings
    changes (I-4).
    """
    names = _field_names()
    if not custom or not names:
        return {name: None for name in names}
    offered = dict(zip(names, list(values or [])))
    try:
        import mc_voice_engines as engines

        return engines.profiles().overrides(offered)
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
    import mc_voice_engines as engines

    active = engines.active()
    label = engines.label(active)
    if active == engines.SOPRO:
        import mc_voice_sopro_runtime as sopro_runtime

        found = sopro_runtime.engine()
    else:
        found = runtime.engine()
    labels = {
        "unloaded": f"\u25cb Unloaded — loads automatically on next voice use",
        "loading": "\u25cc Loading speech models…",
        "idle": "\u25cf Loaded — CPU, idle",
        "stt": "\u25cf Loaded — Listening",
        "tts": "\u25cf Loaded — Speaking",
        "speaking": "\u25cf Loaded — Speaking",
        "preparing": "\u25cf Preparing a voice…",
        "stopping": "\u25cc Unloading…",
        "error": f"\u25cf {label} could not start",
    }
    action = "unload" if found.get("loaded") else "load"
    voice = ""
    try:
        entry = engines.adapter(active).default_entry()
        voice = entry["label"] if entry else ""
    except Exception:
        logger.debug("Model Chain: could not read the default voice", exc_info=True)
    # The engine's *name* is here and its settings are not. Section 9: the
    # overlay stays operational rather than becoming a second settings page, and
    # it must never be a place where both engines' configurations are displayed.
    # The only way from here to the other engine is the link to Settings.
    return (
        f'<div class="mc-voice-engine" data-mc-voice-engine-id="{ui.escape(active)}">'
        f'<div class="mc-voice-engine-head">Voice engine</div>'
        f'<div class="mc-voice-engine-name" data-mc-voice-engine-name>'
        f'{ui.escape(label)}</div>'
        f'<div class="mc-voice-engine-state" data-mc-voice-engine-line>'
        f'{ui.escape(labels.get(found.get("state") or "unloaded", labels["unloaded"]))}</div>'
        f'<button type="button" class="mc-voice-runtime" data-mc-voice-runtime="{action}">'
        f'{action.title()}</button>'
        f'<div class="mc-voice-engine-voice">Default voice: '
        f'<span data-mc-voice-default>{ui.escape(voice)}</span></div>'
        f'</div>')


def readiness_notice() -> str:
    """One line for the overlay, drawn from the same status Settings reads."""
    import mc_voice_engines as engines

    found = models.status()
    active = engines.active()
    speaks = engines.installed(active)
    if found.stt_ready and speaks:
        return ui.notice("Ready.")
    if not found.platform_supported:
        return ui.notice(found.runtime_message, "warn")
    # Named separately because they are separate installations with separate
    # lifecycles (I-6): dictation can work perfectly while the selected speech
    # engine is not installed, and a single "not set up" would hide that.
    missing = []
    if not found.stt_ready:
        missing.append("speech to text")
    if not speaks:
        missing.append(f"{engines.label(active)} text to speech")
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
            import mc_voice_engines as engines

            active = engines.active()
            if not engines.installed(active):
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
            profiles = engines.profiles(active)
            return api.remember_reply(
                str(text), voice_id=voice_of(character, active),
                profile=profiles.resolve(profile_of(character, active)), engine=active)
        except Exception:
            logger.debug("Model Chain: Voice Chat could not prepare a spoken reply",
                         exc_info=True)
            return ""

    return marker


# --------------------------------------------------------------------------- #
# The Settings page row
# --------------------------------------------------------------------------- #


def _manual_section(kind: str, addresses: list, blurb: str, placeholder: str,
                    model: str = "", title: str = "") -> str:
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

    ``title`` names what this section installs, for a row that has more than one
    of them. Sopro's row has two -- the runtime and the model artifacts -- and
    with the default text both collapsed sections read "Or install from files
    you download yourself", one directly above the other, which says nothing
    about which is which.
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
        f'<summary>{ui.escape(title or "Or install from files you download yourself")}'
        f'</summary>'
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
    import mc_voice_engines as engines

    found = models.status()
    active = engines.active()
    parts = [f'<div class="mc-voice-settings" '
             f'data-mc-voice-key="{ui.escape(api.session_token())}" '
             f'data-mc-voice-engine-id="{ui.escape(active)}">',
             engine_selector_html()]

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

    if active != engines.KOKORO:
        # Section 5: the inactive engine's operational settings are *absent*,
        # not collapsed. The Kokoro install row, its bundle name and its manual
        # section are not rendered at all while Sopro is selected -- so a stale
        # DOM, a theme script or a partial Gradio re-render has nothing to
        # expose, and no request can be built from markup that is not there.
        parts.append(sopro_html())
        parts.append(cleanup_html())
        parts.append(
            '<div class="mc-voice-note">Voice Chat runs on the CPU and never uses the '
            'graphics card. Sopro brings its own isolated PyTorch runtime, which is kept '
            'separate from Forge\'s and from Kokoro\'s. After it is installed it needs no '
            'Internet connection at all.</div>')
        parts.append("</div>")
        return "".join(parts)

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


def engine_selector_html() -> str:
    """The one place both engine names are meant to be visible at once. Section 4.

    A pair of cards rather than a drop-down, because the choice is not between
    two labels: it is between a built-in speaker bank that is probably already
    installed and a streaming model that clones a voice from a recording and
    brings a hundred and forty megabytes of PyTorch with it. Nobody can make
    that choice from a list of names.

    Selecting an engine that is not installed is allowed and is *not* an error
    state (section 17). The selected engine's own page then says it is not
    installed and offers to install it; Kokoro does not reappear as an
    operational panel because Sopro is not ready yet, and nothing switches back
    on its own.
    """
    import mc_voice_engines as engines

    found = engines.state()
    cards = []
    for entry in found["engines"]:
        mark = "mc-voice-engine-card-active" if entry["active"] else ""
        cards.append(
            f'<button type="button" class="mc-voice-engine-card {mark}" '
            f'data-mc-voice-engine-pick="{ui.escape(entry["id"])}"'
            f'{" disabled" if entry["active"] else ""}>'
            f'<span class="mc-voice-engine-card-name">{ui.escape(entry["label"])}</span>'
            f'<span class="mc-voice-engine-card-state">'
            f'{"Selected" if entry["active"] else ("Installed" if entry["installed"] else "Not installed")}'
            f'</span>'
            f'<span class="mc-voice-engine-card-blurb">{ui.escape(entry["blurb"])}</span>'
            f'</button>')
    return (
        '<div class="mc-voice-row" data-mc-voice-engines>'
        '<div class="mc-voice-head">'
        '<div class="mc-voice-heading">Text-to-speech engine</div>'
        f'<div class="mc-voice-default" data-mc-voice-engine-current>'
        f'{ui.escape(found["label"])}</div>'
        '</div>'
        '<p class="mc-voice-note">One engine speaks for the whole WebUI at a time. '
        'Choosing one changes which voice settings appear everywhere — in this page, in '
        'the Voice menu and in every character. The other engine keeps its own voices, '
        'default and character settings exactly as they were, and they come back when you '
        'choose it again.</p>'
        f'<div class="mc-voice-engine-cards">{"".join(cards)}</div>'
        '<p class="mc-voice-note">Dictation is not part of this choice. Whisper has its '
        'own model and its own process, and switching between Kokoro and Sopro does not '
        'reload it, change its quality or touch your microphone settings.</p>'
        '</div>')


def sopro_html() -> str:
    """The whole Sopro settings surface: install, engine settings, and nothing else.

    Drawn only when Sopro is the selected engine, and the voice library, the
    clone form and the Voice Lab live in :func:`voices_html` beneath it -- the
    same split Kokoro has, so the two engines' pages have the same shape even
    though almost nothing in them is shared.

    Static markup here is the first frame; ``javascript/voice_chat.js`` repaints
    it from ``/voice/sopro``. A page whose JavaScript has not run still says
    something true.
    """
    import mc_voice_sopro as sopro

    try:
        found = sopro.status()
        runtime_sources = sopro.sources("runtime")
        model_sources = sopro.sources("model")
        settings = sopro.engine_settings()
        entry = sopro.bundle()
    except Exception:
        logger.debug("Model Chain: could not describe Sopro", exc_info=True)
        return ('<div class="mc-voice-row" data-mc-voice-kind="sopro">'
                '<div class="mc-voice-head"><div class="mc-voice-heading">Sopro V2</div>'
                '<div class="mc-voice-status">Sopro could not be described. This is a '
                'problem with the extension rather than with your installation.</div>'
                '</div></div>')

    return (
        f'<div class="mc-voice-row" data-mc-voice-kind="sopro">'
        f'<div class="mc-voice-head">'
        f'<div class="mc-voice-heading">Sopro V2 Turbo</div>'
        f'<div class="mc-voice-default">{ui.escape(entry.label)}, CPU only</div>'
        f'<div class="mc-voice-status" data-mc-voice-status="sopro">'
        f'{ui.escape(found.message)}</div>'
        f'<button type="button" class="mc-voice-install" data-mc-voice-sopro-install>'
        f'{"Installed" if found.ready else "Install Sopro"}</button>'
        f'</div>'
        f'<p class="mc-voice-note">A streaming model that makes a voice from a short '
        f'recording you make here — no separate cloning tool, no training job. It brings '
        f'its own isolated PyTorch runtime '
        f'({ui.escape(models._bytes_label(sum(int(item.size or 0) for item in (sopro.platform().artifacts if sopro.platform() else ())) or 0))}) '
        f'and its model artifacts '
        f'({ui.escape(models._bytes_label(entry.download_bytes))}, approximate until the '
        f'download starts). Both are separate from Kokoro and from Forge: installing this '
        f'changes nothing about either, and removing it changes nothing either.</p>'
        f'<div class="mc-voice-sopro-parts">'
        f'<div class="mc-voice-check" data-mc-voice-sopro-runtime>'
        f'{ui.escape(found.runtime_message)}</div>'
        f'<div class="mc-voice-check" data-mc-voice-sopro-model>'
        f'{ui.escape(found.model_message)}</div>'
        f'</div>'
        + _manual_section(
            "sopro-runtime", runtime_sources,
            "The Install button above does all of this for you. This is here for a machine "
            "that cannot reach PyPI — no Internet, or a proxy that will not pass a "
            "hundred-megabyte binary. Download every file into one folder and give Voice "
            "Chat that folder. The original filenames are fine, and each one is checked "
            "against a hash committed in this extension.",
            "C:\\Users\\you\\Downloads\\sopro-runtime",
            title="Or install the PyTorch runtime from files you download yourself")
        + _manual_section(
            "sopro-model", model_sources,
            "The model artifacts. Download these seven files into one folder and give Voice "
            "Chat that folder. No account or access token is needed.",
            "C:\\Users\\you\\Downloads\\sopro-model",
            title="Or install the model artifacts from files you download yourself")
        + _sopro_engine_settings(settings, found)
        + '</div>')


def cleanup_html() -> str:
    """The recording-cleanup installer, and what it costs.

    Its own row rather than a line in Sopro's, because it is not a
    text-to-speech engine: it takes a recording and gives back a quieter one,
    and the engine selector has no opinion about it. It sits here because this
    is where the recording is made.

    The price is on the row rather than behind it. A quarter of a gigabyte to
    tidy a twenty-second clip is a real decision and somebody should be able to
    make it before pressing anything.
    """
    import mc_voice_cleanup as cleanup

    try:
        found = cleanup.status()
        size = models._bytes_label(found.download_bytes)
    except Exception:
        logger.debug("Model Chain: could not describe recording cleanup", exc_info=True)
        return ('<div class="mc-voice-row" data-mc-voice-kind="cleanup">'
                '<div class="mc-voice-head"><div class="mc-voice-heading">Recording '
                'cleanup</div><div class="mc-voice-status">Recording cleanup could not be '
                'described. This is a problem with the extension rather than with your '
                'installation.</div></div></div>')

    return (
        f'<div class="mc-voice-row" data-mc-voice-kind="cleanup">'
        f'<div class="mc-voice-head">'
        f'<div class="mc-voice-heading">Recording cleanup</div>'
        f'<div class="mc-voice-default">{ui.escape(cleanup.LABEL)}, CPU only</div>'
        f'<div class="mc-voice-status" data-mc-voice-status="cleanup">'
        f'{ui.escape(found.message)}</div>'
        f'<button type="button" class="mc-voice-install" data-mc-voice-cleanup-install'
        f'{" disabled" if not found.platform_supported else ""}>'
        f'{"Installed" if found.ready else "Install cleanup"}</button>'
        f'</div>'
        f'<p class="mc-voice-note">Optional. A learned denoiser that takes background '
        f'noise out of a recording before it becomes a voice — a fan, a room, traffic, '
        f'hiss. The page already does a simpler cleanup with no download at all, and this '
        f'is the better one.</p>'
        f'<p class="mc-voice-note">It costs about {ui.escape(size)}, most of it a second '
        f'copy of PyTorch: DeepFilterNet only publishes its Rust library for Python 3.10 '
        f'and 3.11, so it brings an interpreter of its own rather than sharing this '
        f'WebUI\'s. It runs only while a recording is being cleaned, stops itself two '
        f'minutes after the last one, and is ended with the WebUI whatever happens.</p>'
        f'<div class="mc-voice-sopro-parts">'
        f'<div class="mc-voice-check" data-mc-voice-cleanup-runtime>'
        f'{ui.escape(found.runtime_message)}</div>'
        f'<div class="mc-voice-check" data-mc-voice-cleanup-model>'
        f'{ui.escape(found.model_message)}</div>'
        f'</div>'
        f'<div class="mc-voice-progress" data-mc-voice-progress="cleanup" hidden>'
        f'<div class="mc-voice-progress-bar" data-mc-voice-progress-bar="cleanup"></div>'
        f'</div>'
        f'</div>')


def _sopro_engine_settings(settings: dict, found) -> str:
    """Precision, solver steps and streaming chunk size. Global to Sopro.

    Not per character and not per voice (section 34): each of these changes
    compute, memory and which warmed streaming caches are still valid for the
    whole worker, and a character setting that quietly reloaded the model would
    be a character setting nobody could reason about. Changing one stops the
    worker; the next reply starts it again.
    """
    def choices(name: str, current, values, labels=None) -> str:
        options = "".join(
            f'<option value="{ui.escape(str(value))}"'
            f'{" selected" if str(value) == str(current) else ""}>'
            f'{ui.escape(str((labels or {}).get(value, value)))}</option>'
            for value in values)
        return (f'<select data-mc-voice-sopro-setting="{ui.escape(name)}">{options}</select>')

    precision_labels = {item["id"]: item["label"] for item in settings["precisions"]}
    return (
        '<details class="mc-voice-manual" data-mc-voice-sopro-settings>'
        '<summary>Engine settings</summary>'
        '<p class="mc-voice-note">These change how Sopro runs rather than how a character '
        'sounds, so they apply to every Sopro voice. Changing one unloads Sopro; the next '
        'reply loads it again.</p>'
        '<div class="mc-voice-field">'
        '<label>Precision</label>'
        + choices("precision", settings["precision"],
                  [item["id"] for item in settings["precisions"]], precision_labels)
        + '<p class="mc-voice-note">INT8 quantizes the autoregressive blocks and is faster '
          'and lighter on the CPU. Your saved voices stay valid either way — only the '
          'warmed streaming caches are rebuilt.</p>'
        '</div>'
        '<div class="mc-voice-field">'
        '<label>Solver steps</label>'
        + choices("steps", settings["steps"], settings["step_choices"])
        + '<p class="mc-voice-note">How many steps the acoustic solver takes. More is '
          'slower. This is a compute setting, not a character trait.</p>'
        '</div>'
        '<div class="mc-voice-field">'
        '<label>Streaming chunk size</label>'
        + choices("chunk_frames", settings["chunk_frames"], settings["chunk_choices"])
        + '<p class="mc-voice-note">How much audio Sopro produces before handing a piece '
          'over. Smaller may start sooner and finish later; larger may do the reverse. '
          'Only benchmarked values are offered.</p>'
        '</div>'
        f'<p class="mc-voice-note">CPU threads are fixed at four working threads and one '
        f'coordinating thread for this build, chosen from measurements and reported in the '
        f'log rather than tuned here. Build fingerprint '
        f'<code>{ui.escape(found.fingerprint or "not installed")}</code>.</p>'
        '</details>')


def delivery_controls() -> list:
    """The four sliders, as data, in the order both surfaces draw them.

    Read from :mod:`mc_voice_profile` rather than repeated here, because a range
    that differed between the settings page and the character screen would be a
    setting that silently changed when it was edited in the other place.
    """
    try:
        import mc_voice_engines as engines

        profiles = engines.profiles()
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
    import mc_voice_engines as engines

    active = engines.active()
    try:
        profiles = engines.profiles(active)
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
        + _delivery_note(active)
        + '</div>')


def _delivery_note(engine: str) -> str:
    """The paragraph that says which of these controls is the model's own.

    Section 37 makes this part of correctness rather than decoration, and the
    two engines need different sentences because the same four labels mean
    different things: Kokoro takes a speed argument and Sopro does not, so
    Sopro's Speed is Voice Chat's own time-scaling and the text has to say so
    rather than let somebody assume a model control.
    """
    import mc_voice_engines as engines

    if engine == engines.SOPRO:
        return (
            '<p class="mc-voice-note">Sopro has no speaking-rate input of its own, so '
            'Speed is applied by Voice Chat: the audio is time-scaled without changing '
            'the pitch, and the processing carries across the pieces Sopro streams so '
            'there is no click between them. Pitch is separate and composes with it — '
            'changing Speed at Pitch 0 does not transpose the voice, and changing Pitch '
            'does not change how long the sentence takes. Volume and pacing are Voice '
            'Chat\'s too.</p>'
            '<p class="mc-voice-note">Variation is Sopro\'s own sampling temperature, and '
            'Top-p and Top-k are its sampling cut-offs. They control how much a take '
            'varies from another take. They are not emotion, warmth or energy controls — '
            'the model has no such input, and a slider that claimed to be one would be '
            'making a promise nobody has tested. Left alone they follow the model\'s own '
            'configuration.</p>')
    return (
        '<p class="mc-voice-note">Kokoro exposes one of these itself — speed, which changes '
        'how the model articulates rather than only how fast it plays. Pitch, volume and '
        'pacing are applied by Voice Chat to the audio the model produced: pitch by '
        'resynthesising faster and reading the result back slower, which moves the formants '
        'with it and reads as a different-sized speaker. There is no emotion control, '
        'because Kokoro-82M has no emotion input — a slider for one would do nothing.</p>')


def _sopro_voices_html() -> str:
    """Sopro's voice library, clone form, delivery controls and Voice Lab.

    The same product operations Kokoro's block offers -- list, audition, set as
    default, assign to a character, rename, delete -- plus the two Sopro has
    that Kokoro does not: making a voice from a recording taken here, and
    rebuilding a voice whose preparation no longer matches the installed build.

    The Lab is last and is a ``<details>`` that starts closed, because it is
    experimental and section 44 asks for it to be hard to mistake for the
    ordinary controls. It is in this document only because Sopro is selected;
    it does not exist in Kokoro's.
    """
    import mc_voice_sopro as sopro

    languages = "".join(
        f'<option value="{ui.escape(code)}">{ui.escape(label)}</option>'
        for code, label in sopro.LANGUAGES)
    return (
        f'<div class="mc-voice-voices" data-mc-voice-key="{ui.escape(api.session_token())}" '
        f'data-mc-voice-engine-id="{ui.escape(sopro.ENGINE)}">'
        '<div class="mc-voice-row">'
        '<div class="mc-voice-head">'
        '<div class="mc-voice-heading">Sopro voices</div>'
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
        # Above the clone form on purpose. Sopro has no speaker bank, so an
        # installation with nothing in it used to offer exactly one way forward:
        # record yourself. That is a wall in front of somebody who only wants to
        # hear whether the engine works, and the way past it is cheap.
        + '<div class="mc-voice-row" data-mc-voice-sopro-starter>'
        '<div class="mc-voice-head">'
        '<div class="mc-voice-heading">Starter voices</div>'
        '<div class="mc-voice-status" data-mc-voice-starter-status></div>'
        '<button type="button" class="mc-voice-install" data-mc-voice-starter-make>'
        'Add starter voices</button>'
        '</div>'
        '<p class="mc-voice-note">Four voices made here, on this PC, by having '
        'Kokoro read a short passage and cloning that. They take a few seconds '
        'each and need no recording and no download. Nobody\'s voice is copied — '
        'a Kokoro speaker is synthetic — and what you get is an ordinary Sopro '
        'voice you can rename, audition, assign or delete like any other.</p>'
        '</div>'
        + f'<div class="mc-voice-row" data-mc-voice-sopro-clone>'
        f'<div class="mc-voice-head">'
        f'<div class="mc-voice-heading">Clone voice</div>'
        f'<div class="mc-voice-status" data-mc-voice-sopro-clone-status></div>'
        f'</div>'
        f'<p class="mc-voice-note">Sopro makes a voice from a short reference recording. '
        f'This is part of how the model normally works — there is no training job and '
        f'nothing extra to install. It runs on the CPU in the same process that will '
        f'later speak the voice, and takes seconds rather than hours.</p>'
        f'<p class="mc-voice-note mc-voice-consent">Only clone a voice you own or have '
        f'permission to clone. The recording stays on this PC, beside the voice, so that a '
        f'future Sopro update can rebuild the voice without asking you to record again. '
        f'Deleting the voice deletes the recording.</p>'
        f'<div class="mc-voice-clone-form" data-mc-voice-sopro-form>'
        f'<div class="mc-voice-field">'
        f'<label for="mc-voice-sopro-name">Name</label>'
        f'<input type="text" id="mc-voice-sopro-name" data-mc-voice-sopro-name '
        f'maxlength="48" spellcheck="false" />'
        f'</div>'
        f'<div class="mc-voice-field">'
        f'<label for="mc-voice-sopro-language">Language hint</label>'
        f'<select id="mc-voice-sopro-language" data-mc-voice-sopro-language>{languages}'
        f'</select>'
        f'<p class="mc-voice-note">A pronunciation hint, not a translation. Auto is the '
        f'right answer unless you know otherwise.</p>'
        f'</div>'
        f'<div class="mc-voice-field">'
        f'<label for="mc-voice-sopro-file">Recording</label>'
        f'<input type="file" id="mc-voice-sopro-file" '
        f'accept="audio/*,.wav,.mp3,.m4a,.aac,.ogg,.oga,.opus,.flac,.webm,.mp4" '
        f'data-mc-voice-sopro-file />'
        f'<button type="button" class="mc-voice-entry-action" data-mc-voice-sopro-record>'
        f'Record here</button>'
        f'<span class="mc-voice-sopro-recording" data-mc-voice-sopro-recording></span>'
        f'</div>'
        # The browser already decodes every format it can play, and already has
        # a WAV encoder in this file for dictation. So "bring me a 16-bit PCM
        # WAV of between five and twenty seconds" -- which is a real constraint
        # of the model and was being handed to the user as homework -- becomes
        # something the page does: drop in an MP3, drag the part you want, press
        # Create. Nothing here reaches the network; the file is read, decoded,
        # trimmed and encoded in the tab.
        + f'<div class="mc-voice-trim" data-mc-voice-trim hidden>'
        f'<canvas class="mc-voice-wave" data-mc-voice-wave height="96" '
        f'aria-label="Waveform of the chosen recording. Drag to choose the part to '
        f'clone, or use the start and end boxes below."></canvas>'
        f'<div class="mc-voice-trim-row">'
        f'<button type="button" class="mc-voice-entry-action" data-mc-voice-trim-play>'
        f'Play selection</button>'
        f'<button type="button" class="mc-voice-entry-action" data-mc-voice-trim-best>'
        f'Pick 15 s for me</button>'
        f'<label class="mc-voice-lab-check">'
        f'<input type="checkbox" data-mc-voice-clean /> Clean up the recording</label>'
        # Shown only when the engine is installed, because a control offering a
        # choice between one thing and one thing that is not there is not a
        # choice. The page-side pass is always available and is the default.
        f'<select data-mc-voice-clean-how class="mc-voice-clean-how" hidden>'
        f'<option value="page">in this page (fast)</option>'
        f'<option value="deepfilternet">with DeepFilterNet (better)</option>'
        f'</select>'
        f'<label for="mc-voice-trim-start">Start</label>'
        f'<input type="number" id="mc-voice-trim-start" data-mc-voice-trim-start '
        f'min="0" step="0.1" inputmode="decimal" />'
        f'<label for="mc-voice-trim-end">End</label>'
        f'<input type="number" id="mc-voice-trim-end" data-mc-voice-trim-end '
        f'min="0" step="0.1" inputmode="decimal" />'
        f'</div>'
        # Written into an aria-live region rather than only drawn on the canvas:
        # the length is the one thing that decides whether Create will work, and
        # a canvas says it to nobody using a screen reader.
        f'<div class="mc-voice-trim-state" data-mc-voice-trim-state role="status" '
        f'aria-live="polite"></div>'
        f'<p class="mc-voice-note">Cleaning takes out steady background noise — hiss, '
        f'hum, a fan, room tone — and lifts the level. It is ordinary spectral '
        f'subtraction rather than a learned denoiser, it runs here in the page, and '
        f'you can hear it: tick it and press Play selection, then untick and play '
        f'again. What is uploaded is whatever you can hear.</p>'
        f'</div>'
        f'<button type="button" class="mc-voice-install" data-mc-voice-sopro-create>'
        f'Create voice</button>'
        f'<p class="mc-voice-note">'
        f'{int(sopro.MIN_REFERENCE_SECONDS)} to {int(sopro.MAX_REFERENCE_SECONDS)} seconds '
        f'of one clear speaker, at a natural speaking pace, in a room without much '
        f'background noise. The recording is checked, normalised and prepared here; you '
        f'will hear the finished voice before it is saved.</p>'
        f'</div>'
        f'</div>'
        + _lab_html()
        + '</div>')


def _lab_html() -> str:
    """The Voice Lab. Experimental, closed by default, and labelled as both.

    Section 38: it exists only while Sopro is the active engine, and it does not
    appear in the Conversation overlay, the character editor or any ordinary
    voice picker. Section 44: it has to be difficult to mistake a Lab result for
    a character's saved voice, so the notice is the first thing in it and Reset
    All is always present.

    The sliders are numbered rather than named. Naming them after emotions or
    vocal traits before repeatable tests justify those names would be inventing
    a product claim about eight learned latent values, which is precisely what
    section 41 forbids.
    """
    import mc_voice_lab as lab

    sliders = "".join(
        f'<div class="mc-voice-slider" data-mc-voice-lab-slider="{index}">'
        f'<label for="mc-voice-lab-{index}">Style control {index + 1}</label>'
        f'<input type="range" id="mc-voice-lab-{index}" '
        f'min="{-lab.DELTA_LIMIT}" max="{lab.DELTA_LIMIT}" step="0.05" value="0" '
        f'data-mc-voice-lab-input="{index}" />'
        f'<output data-mc-voice-lab-value="{index}">0</output>'
        f'</div>'
        for index in range(lab.STYLE_CONTROLS))
    return (
        '<details class="mc-voice-row mc-voice-lab" data-mc-voice-lab>'
        '<summary>Voice Lab (experimental)</summary>'
        '<div class="mc-voice-lab-notice">'
        'These controls affect only this audition. They are not used in Conversation and '
        'are not saved to characters or to the default voice.'
        '</div>'
        '<p class="mc-voice-note">The eight style controls below are learned latent values '
        'inside Sopro. They are not emotions, energy, warmth or breathiness — nobody has '
        'measured what they mean, which is what this surface is for. Conditioning Blend '
        'recombines speaker conditioning while keeping the first voice\'s reference '
        'context; it is not proven identity or style transfer.</p>'
        '<div class="mc-voice-field">'
        '<label for="mc-voice-lab-voice">Voice</label>'
        '<select id="mc-voice-lab-voice" data-mc-voice-lab-voice></select>'
        '</div>'
        '<div class="mc-voice-field">'
        '<label for="mc-voice-lab-text">Audition text</label>'
        '<input type="text" id="mc-voice-lab-text" data-mc-voice-lab-text maxlength="400" />'
        '</div>'
        f'<div class="mc-voice-lab-sliders">{sliders}</div>'
        '<div class="mc-voice-field">'
        '<label>Conditioning Blend</label>'
        '<select data-mc-voice-lab-blend-voice><option value="">No blend</option></select>'
        '<label class="mc-voice-lab-check">'
        '<input type="checkbox" data-mc-voice-lab-blend-field="id_emb" /> identity</label>'
        '<label class="mc-voice-lab-check">'
        '<input type="checkbox" data-mc-voice-lab-blend-field="style_emb" /> style</label>'
        '<label class="mc-voice-lab-check">'
        '<input type="checkbox" data-mc-voice-lab-blend-field="style_ctrl" />'
        ' style controls</label>'
        '<input type="range" min="0" max="1" step="0.05" value="0" '
        'data-mc-voice-lab-blend-weight />'
        '<output data-mc-voice-lab-blend-value>0</output>'
        '</div>'
        '<div class="mc-voice-field">'
        '<label class="mc-voice-lab-check">'
        '<input type="checkbox" data-mc-voice-lab-fixed-seed /> Fixed seed</label>'
        '<input type="number" data-mc-voice-lab-seed value="1234" min="0" '
        'max="2147483647" />'
        '<p class="mc-voice-note">A fixed seed makes A and B differ because of the control '
        'you moved rather than because of sampling noise.</p>'
        '</div>'
        '<div class="mc-voice-slider-actions">'
        '<button type="button" class="mc-voice-entry-action" data-mc-voice-lab-play="a">'
        'Play A (saved voice)</button>'
        '<button type="button" class="mc-voice-entry-action" data-mc-voice-lab-play="b">'
        'Play B (experiment)</button>'
        '<button type="button" class="mc-voice-entry-action" data-mc-voice-lab-reset>'
        'Reset all</button>'
        '</div>'
        '<div class="mc-voice-lab-metrics" data-mc-voice-lab-metrics></div>'
        '</details>')


def _value_label(name: str, value) -> str:
    try:
        import mc_voice_engines as engines

        return engines.profiles().value_label(name, value)
    except Exception:
        return str(value)


def voices_html() -> str:
    """Voice selection, auditioning, renaming, deleting, and cloning.

    A second HTML block on the Settings page, drawn and redrawn by
    ``javascript/voice_chat.js`` from ``/voice/voices``. Static markup here is
    the first frame and the shape; everything live is painted by the browser,
    for the same reason the install row is -- Forge's settings system stores
    options, it does not host Gradio controls with handlers.

    Engine-scoped from the first byte. When Kokoro is selected this renders the
    Kokoro list, the Kokoro delivery sliders and the Storytime cloning panel;
    when Sopro is selected it renders Sopro's list, Sopro's delivery and
    generation controls, Sopro's clone form and the Voice Lab -- and the other
    engine's markup is *not in the document* (section 5). Switching engines
    reloads this page, which is how the browser gets a document with the other
    engine's controls genuinely absent rather than hidden.

    What is *not* here on either branch is any voice data. The list is fetched,
    so a page that was open when a clone finished shows it on its next paint
    rather than needing a reload, and no engine-native address is ever put in
    the document (section 56).
    """
    import mc_voice_engines as engines

    active = engines.active()
    if active == engines.SOPRO:
        return _sopro_voices_html()
    return (
        f'<div class="mc-voice-voices" data-mc-voice-key="{ui.escape(api.session_token())}" '
        f'data-mc-voice-engine-id="{ui.escape(active)}">'
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
