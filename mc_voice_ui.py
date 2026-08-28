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

import gradio as gr

import mc_llm_ui as ui
import mc_voice_api as api
import mc_voice_models as models
import mc_voice_state as state

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

SCREEN = "voice"
"""The name this overlay answers to in the panel's ``SCREENS`` tuple."""

MIC_GLYPH = "\U0001f3a4"

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
    """The composer control. One compact button, the same 44px as the paperclip.

    Never disabled, even with nothing installed. The requested interaction is
    that pressing it *explains what is wrong* -- a control that is simply dead
    is a control somebody presses three times and then reports as broken.
    """
    return gr.Button(MIC_GLYPH, size="sm", scale=0, min_width=44,
                     elem_id=ui.ident("chat", "voice-mic"),
                     elem_classes=ui.classes("icon-button", "voice-mic"))


def plumbing() -> dict:
    """The two hidden boxes the browser reads. Neither carries any content.

    ``token`` is an opaque one-shot handle to a speech target held in this
    process's RAM -- never the reply text, which is the difference between this
    and copying an assistant message through a hidden DOM field where a page
    extension or a screen reader would find it.

    ``key`` is this process's page token, which the browser sends back on every
    voice request. Put in the page by Python because that is the one channel a
    cross-site page cannot read.
    """
    token = gr.Textbox(value="", visible=False, container=False,
                       elem_id=ui.ident("chat", "voice-token"))
    key = gr.Textbox(value=api.session_token(), visible=False, container=False,
                     elem_id=ui.ident("chat", "voice-key"))
    return {"token": token, "key": key}


def sheet() -> dict:
    """The Voice overlay: what is ready, two switches, and a way out."""
    with gr.Column(visible=False, elem_id=ui.ident("chat", "voice"),
                   elem_classes=ui.classes("sheet", "sheet-screen")) as screen:
        with gr.Row(elem_classes=ui.classes("sheet-head")):
            back = gr.Button("‹ Back", size="sm", scale=0, min_width=76,
                             elem_classes=ui.classes("sheet-back"))
            gr.Markdown("#### Voice")
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
            "WebUI — which needs to be HTTPS for the browser to open the microphone at all.",
            elem_classes=ui.classes("hint"))
    return {"screen": screen, "back": back, "readiness": readiness,
            "auto_send": auto_send, "auto_speak": auto_speak}


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
    return screens(SCREEN) + [readiness_notice(),
                              gr.update(value=current["auto_send"]),
                              gr.update(value=current["auto_speak"])]


def set_auto_send(value):
    """Persist immediately. Section 43: no Apply, no second copy of the truth."""
    return _remember(auto_send=bool(value), key="auto_send")


def set_auto_speak(value):
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


def settings_html() -> str:
    """The Voice Chat install row, as static HTML the browser makes live.

    Forge's settings system stores options; it does not host arbitrary Gradio
    controls with handlers. So the buttons are ordinary buttons in an HTML
    block, ``javascript/voice_chat.js`` wires them to the install route, and the
    same script polls the status route to redraw the rows. That is the
    mechanism the design intent recommends, and it keeps "download" from
    becoming a fake persistent boolean that a settings backup would restore as
    an instruction to download something.

    Each row has two ways in, because one of them is not always available. The
    button downloads the pinned artifacts. The folder box installs from files
    somebody already has -- which is the answer for a build whose artifacts are
    not pinned, a proxy that will not pass a 300 MB binary, and an air-gapped
    machine. The addresses are printed beside it rather than left for somebody
    to search for: "a Whisper small ONNX export" describes several files and
    only one of them is the one this runtime wants.

    The page token is an attribute on the container because this row is not
    inside Conversation and has no hidden Gradio component of its own to read
    it from.
    """
    found = models.status()
    parts = [f'<div class="mc-voice-settings" '
             f'data-mc-voice-key="{ui.escape(api.session_token())}">']
    parts.append(f'<div class="mc-voice-runtime">{ui.escape(found.runtime_message)}</div>')

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

        if addresses:
            links = "".join(
                f'<li><a href="{ui.escape(item["url"])}" target="_blank" rel="noreferrer">'
                f'{ui.escape(item["filename"])}</a>'
                + ("" if item["archive"] or item["filename"] == item["save_as"]
                   else f' <span class="mc-voice-rename">→ save as '
                        f'{ui.escape(item["save_as"])}</span>')
                + "</li>"
                for item in addresses)
            parts.append(
                f'<details class="mc-voice-manual">'
                f'<summary>Install from files you download yourself</summary>'
                f'<p>Download these into one folder, then give Voice Chat that folder. '
                f'The original filenames are fine — nothing needs renaming, and no '
                f'account or access token is needed for either site.</p>'
                f'<ul class="mc-voice-links">{links}</ul>'
                f'<div class="mc-voice-folder-row">'
                f'<input type="text" class="mc-voice-folder" '
                f'data-mc-voice-folder="{kind}" spellcheck="false" '
                f'placeholder="C:\\Users\\you\\Downloads\\voice-{kind}" />'
                f'<button type="button" class="mc-voice-install-local" '
                f'data-mc-voice-local="{kind}">Install from this folder</button>'
                f'</div>'
                f'<p class="mc-voice-note">Files you supply are checked for shape and '
                f'completeness, not against a hash in this repository — there is none '
                f'for them. Voice Chat records what it installed so later changes still '
                f'show.</p>'
                f'</details>')
        parts.append("</div>")

    parts.append(
        '<div class="mc-voice-note">Voice Chat runs on the CPU and never uses the graphics '
        'card. It has no sign-in of its own and needs no account, API key or access token '
        '— not for this WebUI, and not for the sites these files come from. After they are '
        'installed it needs no Internet connection at all.</div>')
    parts.append("</div>")
    return "".join(parts)
