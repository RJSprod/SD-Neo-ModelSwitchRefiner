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
        import mc_voice_registry as registry

        sid, entry = registry.resolve()
        found = turns.create(voice_id=entry["id"], sid=sid, labels=_labels(character, persona))
        found.base_chars = len(str(opening or ""))
        found.start()
        _last_run["turn"] = found.id
        logger.info("Model Chain: Voice will read this reply aloud — %s, speaker %d, turn %s",
                    entry["id"], sid, found.id[:8])
        return found
    except Exception:
        # Warning rather than debug, and this is the correction that matters:
        # a failure here disables the whole feature for that reply and used to
        # leave no trace at all in a log at the default level. "Voice went
        # silent and nothing was written down" is not a diagnosable state.
        logger.warning("Model Chain: Voice Chat could not start speaking this reply",
                       exc_info=True)
        return None


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


def speech_marker(take_reply):
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
    """

    def marker():
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
            return api.remember_reply(str(text))
        except Exception:
            logger.debug("Model Chain: Voice Chat could not prepare a spoken reply",
                         exc_info=True)
            return ""

    return marker


# --------------------------------------------------------------------------- #
# The Settings page row
# --------------------------------------------------------------------------- #


def _manual_section(kind: str, addresses: list, blurb: str, placeholder: str) -> str:
    """The "or install from files you download yourself" half of a row.

    Every row has one now, the engine included: the failure that made the engine
    row necessary in the first place was an automatic install that could not be
    completed, and an escape hatch that covers two of the three things a person
    needs is not an escape hatch.
    """
    if not addresses:
        return ""
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
        f'data-mc-voice-folder="{kind}" spellcheck="false" '
        f'placeholder="{ui.escape(placeholder)}" />'
        f'<button type="button" class="mc-voice-install-local" '
        f'data-mc-voice-local="{kind}">Install from this folder</button>'
        f'</div>'
        f'<p class="mc-voice-note">Nothing is installed on trust: a file under the right '
        f'name with the wrong contents is refused exactly as a bad download is.</p>'
        f'</details>')


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

    for kind, heading in (("stt", "Speech to text"), ("tts", "Text to speech")):
        label, addresses = "", []
        if found.platform_supported:
            try:
                entry = models.default_model(kind)
                label = entry.label
                addresses = models.sources(kind)
            except Exception:
                logger.debug("Model Chain: could not describe the %s bundle", kind,
                             exc_info=True)
        message = found.stt_message if kind == "stt" else found.tts_message

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
        '<div class="mc-voice-row" data-mc-voice-cloning>'
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
