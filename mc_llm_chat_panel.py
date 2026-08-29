"""Conversation: a messaging application first, a control surface second.

Section 4.3 asks for the conversation area to receive most of the visual space
and for configuration to be secondary and collapsible. The first version of
this panel answered that with a drawer beside the transcript; this one answers
it the way a messaging application does, because that is what the thing is:
there is a compact header, a transcript that owns the scrolling, and a composer
on the bottom edge, and *nothing else is in the layout at all*. Threads, the
character, the persona, the model and the per-message actions are surfaces that
open over the conversation and close again, leaving the transcript exactly
where it was.

Why not a drawer
----------------
A drawer is a column, and a column is in the layout. On a desktop that costs a
fifth of the width; on a phone Gradio wraps it to a full-width block *above* the
stage, which pushes the composer off the bottom of the window -- the one thing
a chat window must never do. An overlay costs nothing until it is opened and
nothing after it is closed, at every width, which is why every configuration
surface here is one.

The state lives in the vendored ``prompt_master.chat`` package, unchanged:
characters are files in a folder other tools already understand, chats are
documents filed under the character they belong to, and both are read and
written by ``CharacterStore`` and ``ChatStore`` rather than by anything here.
That is section 16's separation kept by construction -- Conversation's history
cannot leak into Prompt Studio's, because they are not in the same files and
never pass through the same code.

View states
-----------
The panel is built once and the surfaces are shown and hidden, because Gradio
cannot create a control in response to a click. The states are:

``CHAT_HOME``
    Header, transcript, status, composer. Everything else hidden.
``NAV_SHEET``
    Threads, Character, You, and the secondary route to the model, to Setup and
    to the other workspaces.
``THREADS_SCREEN`` / ``CHARACTER_SCREEN`` / ``PERSONA_SCREEN``
    One destination each, opened from the nav sheet.
``MESSAGE_ACTION_SHEET``
    A bottom sheet applying to the message that was tapped.
``MESSAGE_EDIT_MODE``
    The composer, temporarily replaced by an editor for the selected message.
``ATTACHMENT_PREVIEW``
    The picture for the next message, in a compact row above the composer.

Only one of the four screens is ever open: :func:`_screens` is the single
function that says so, and every handler that opens one returns its whole
answer rather than toggling a component of its own.

Per-message actions, and why they are a sheet
---------------------------------------------
Gradio 4.40's ``Chatbot`` renders the value it is given and nothing else, so
there is nowhere to hang a ``⋯`` on a bubble. The component's own ``select``
event is what nominates a message -- tap a bubble and it says which -- and the
actions are drawn once, in a sheet that overlays the bottom of the transcript,
applying to whichever message is nominated. Everything the standalone menu
offers is there: edit, regenerate, continue, send again from here, branch from
here, delete, delete from here, and the version pager a regenerate leaves
behind. A sheet rather than a row in the flow, because a row inserted between
the transcript and the composer moves both of them every time you tap a
message.

**Edit** means two different things and is two different things. A reply is text
on the transcript, so editing it rewrites that text where it sits. One of your
own messages is a *prompt that was sent*, so editing it takes it back out of the
thread and into the composer, where it is an unsent message again -- and if
there were replies after it, that happens in a branch, so the thread it came
from keeps every one of them. The same rule regenerating follows, for the same
reason: nothing a reader typed or was told is thrown away to make room for a
second attempt.

Regenerate is the one action that also has an icon on the bubble itself, because
it is the one asked for often enough that three taps is two too many. The icon
is drawn in the browser -- there is nowhere in a Gradio 4.40 bubble to put a
component -- and all it does is nominate a reply: the handler behind it is the
sheet's, so the two can never mean different things. It is polish in the strict
sense, and the tab is complete without it.

``Chatbot`` pairs turns, so the row and column the click reports is not the
index of a message. The map between them is built with the transcript, in one
pass, and carried in a ``gr.State`` -- deriving it a second time somewhere else
is how the two would come to disagree.

Multimodal attachment follows the same rule the standalone application used:
an image is offered only when the running model has a projector, and a request
that carries one is refused with a sentence rather than quietly sent blind.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import gradio as gr

import mc_llm_attachments
import mc_llm_paths
import mc_llm_runtime
import mc_llm_sessions as sessions
import mc_llm_state
import mc_llm_ui as ui
import mc_voice_ui

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

NO_SELECTION = -1
"""What ``selected`` holds when the action sheet applies to nothing."""

SCREENS = ("threads", "character", "persona", "voice")
"""The overlay surfaces, in the order :func:`_screens` returns them.

One at a time, always. They occupy the same space -- the whole conversation
workspace on a narrow display -- so two open at once is two half-drawn panels
rather than one usable one, and the way that is guaranteed is that there is one
function which decides, and it decides for all of them at once.

There is no menu among them any more. Threads, Character and You were behind
one, and a menu whose entire contents are three destinations is a tap in front
of every one of them: they are buttons in the header now, and the header wraps
on a narrow display rather than folding back into a menu. What ``\u2630`` opens
is the shell's own workspace chooser, which is what it opens in every other
mode -- one button, one behaviour, everywhere in LLM Studio.

Voice is the fourth and joined the tuple rather than inventing a mechanism of
its own. Everything that follows from being in this list is exactly what a voice
flyout needs: it takes no room when closed, it closes when another opens, and
``\u2630`` puts it away with the rest.
"""

SELECTION_ORDER = ("sheet", "heading", "back", "pager", "forward", "drop",
                   "regenerate", "continue", "resend", "edit", "edit_box", "edit_image",
                   "composer")
"""The order :func:`_selection_updates` answers in.

The action sheet is redrawn from that one function by every handler that
changes the transcript, so the list of controls it writes into has to be
written down once. ``tests/test_llm_panels.py`` asserts the two are the same
length: a handler one value short would put a label into a visibility and
nothing would raise.

The last four are not in the sheet. An edit borrows the composer's space rather
than opening a panel of its own, so "which message is selected" decides the
state of the edit row, of the picture in it, and of the composer -- and because
they are in this list, every refresh returns the panel to CHAT_HOME without a
second handler having to remember to.

Every message is edited the same way, yours and the character's alike. Yours
used to be *taken back* into the composer instead, which meant editing one in
the middle of a thread had to branch, because sending it again would have
destroyed the replies that followed. That made Edit a second Branch. It is now
what it says: the words in the thread change, the thread stays where it is, and
what follows is a conversation whose earlier turn now says something else.
"""


def build() -> dict:
    """Assemble the panel. Returns the handles the shell needs."""
    from prompt_master.chat.characters import (DEFAULT_MAX_REPLY_TOKENS, DEFAULT_TEMPERATURE,
                                               DEFAULT_TOP_P)
    from prompt_master.core.models import RANDOM_SEED

    prefs = mc_llm_state.preferences()
    who = prefs.get("character") or ""
    # Populated at build rather than left for the first character change: the
    # panel opens on the thread it was left on, and a list that starts empty
    # reads as "you have no threads" rather than as "pick a character".
    initial_threads = _thread_choices(who)
    initial_thread = prefs.get("thread") or (initial_threads[0][1] if initial_threads else "")
    initial_persona = _persona()
    opened = _load(who, initial_thread)
    initial_rows, initial_map = _view(opened)

    cancellation = gr.State(None)
    thread_state = gr.State(initial_thread)
    # Which message the action sheet applies to, and how to get from the
    # component's (row, column) to that index. Both are State rather than
    # recomputed, because a transcript that has been edited since the click is
    # a different transcript and the click has to be read against the one it
    # was made on.
    selected = gr.State(NO_SELECTION)
    positions = gr.State(initial_map)
    # Which overlay is open, by name. Held here rather than read back off the
    # components because a Column has no value a handler can be given -- and
    # without it the menu button could only ever open the menu, never close it.
    surface = gr.State("")

    # position: relative in the stylesheet, and every sheet below is absolutely
    # positioned inside it. That is the whole of the responsive strategy: a
    # surface that opens takes no room in the layout, so nothing it opens over
    # can be pushed anywhere -- least of all the composer, off the bottom of a
    # phone.
    with gr.Column(elem_id=ui.ident("chat"),
                   elem_classes=ui.classes("workspace", "chat-workspace")):

        # -- CHAT_HOME ------------------------------------------------------ #

        with gr.Column(elem_id=ui.ident("chat", "stage"),
                       elem_classes=ui.classes("chat-stage")):

            with gr.Row(elem_classes=ui.classes("chat-header")):
                # The same button LLM Studio's own bar carries, doing the same
                # thing: it opens the shell's workspace chooser. It used to
                # open a menu of this panel's own, which meant the control in
                # the top-left corner did one thing in Conversation and another
                # everywhere else.
                menu = gr.Button("☰", size="sm", scale=0, min_width=44,
                                 elem_id=ui.ident("chat", "menu"),
                                 elem_classes=ui.classes("icon-button"))
                header = gr.HTML(_heading(who, opened),
                                 elem_id=ui.ident("chat", "title"),
                                 elem_classes=ui.classes("chat-title"))
                # Out of the menu and onto the bar. Three destinations behind a
                # menu is a tap in front of each of them, and they are the three
                # this mode is navigated by. The row wraps on a narrow display
                # rather than folding back into a menu -- a second line of
                # chips is a smaller loss than a hidden one.
                to_threads = gr.Button("Threads", size="sm", scale=0, min_width=0,
                                       elem_id=ui.ident("chat", "to-threads"),
                                       elem_classes=ui.classes("chip-button"))
                to_character = gr.Button("Character", size="sm", scale=0, min_width=0,
                                         elem_id=ui.ident("chat", "to-character"),
                                         elem_classes=ui.classes("chip-button"))
                to_persona = gr.Button("You", size="sm", scale=0, min_width=0,
                                       elem_id=ui.ident("chat", "to-persona"),
                                       elem_classes=ui.classes("chip-button"))
                # Voice sits with the other secondary surfaces because that is
                # what it is: two switches and a readiness line, behind a tap.
                # The row wraps on a narrow display exactly as it already does
                # -- losing a little of the title is a smaller loss than a chip
                # that is not there.
                to_voice = mc_voice_ui.chip()
                # The runtime, as one tappable word. What it opens -- the model
                # chooser, Load, Unload, the route to Setup -- is the shell's,
                # and the shell wires this button to it: Conversation is not
                # the place a model filename, a rescan and two buttons belong
                # on screen at all times.
                model = gr.Button(_MODEL_LABEL, size="sm", scale=0, min_width=0,
                                  elem_id=ui.ident("chat", "model"),
                                  elem_classes=ui.classes("chip-button"))

            # No height= at all. The height is the space the tab has, worked
            # out in the browser and applied in style.css -- a number here
            # would be an inline style and would win over the CSS that makes
            # the transcript fit the window instead of the other way round.
            # No copy button: it put an icon under every message in a
            # transcript whose whole job is to be read, to do something a
            # selection and Ctrl+C already do.
            transcript = gr.Chatbot(
                label=None, show_label=False, show_copy_button=False, render_markdown=True,
                value=initial_rows,
                # Yours on the right of your messages and the character's on the
                # left of its replies, which is where Gradio already puts them
                # -- a reply and a question are otherwise the same grey block,
                # and in a long thread there is nothing to scan for. See
                # :func:`_faces`.
                avatar_images=_faces(who),
                elem_id=ui.ident("chat", "transcript"),
                elem_classes=ui.classes("transcript"))

            # THE PER-REPLY REGENERATE ICON, as far as Python is concerned.
            #
            # A Gradio 4.40 Chatbot draws its own bubbles and there is nowhere
            # to put a component on one, so the icon itself is drawn in the
            # browser by javascript/llm_studio.js. What it does when tapped is
            # entirely here: it writes which reply was tapped into this box and
            # presses this button, and the handler below is the same one the
            # action sheet's Regenerate uses. The browser nominates; Python
            # decides, loads, streams and saves.
            #
            # Both are invisible and neither is anything to do without the
            # other. The icons are polish: with the script absent, or a theme
            # whose bubbles it cannot recognise, they are simply not drawn and
            # Regenerate is where it has always been -- on the sheet a tap on
            # the bubble opens.
            regenerate_at = gr.Textbox(value="", visible=False, container=False,
                                       elem_id=ui.ident("chat", "regenerate-at"))
            regenerate_now = gr.Button("Regenerate this reply", visible=False,
                                       elem_id=ui.ident("chat", "regenerate-now"))

            # One line, always the same height, immediately above the composer.
            # "Ready." is quiet; a warning or an error is not, and neither of
            # them moves the transcript by a pixel when it arrives.
            status = gr.HTML(ui.notice("Ready."), elem_id=ui.ident("chat", "status"),
                             elem_classes=ui.classes("chat-status"))

            with gr.Row(elem_id=ui.ident("chat", "composer"),
                        elem_classes=ui.classes("composer")) as composer:
                attach = gr.Button("\U0001f4ce", size="sm", scale=0, min_width=44,
                                   elem_id=ui.ident("chat", "attach"),
                                   elem_classes=ui.classes("icon-button"))
                # A chip beside the composer, and nothing at all until there is
                # a picture in it. It was a full-width drop target above the
                # composer, opened by the paperclip -- a panel's worth of empty
                # dashed border to say "no picture yet", on the one surface
                # that must not grow.
                #
                # The paperclip opens the browser's own file picker, which is
                # what tapping the drop target did anyway; javascript/
                # llm_studio.js forwards the press to this component's file
                # input. The Python handler beside it makes the chip visible,
                # so a browser that never ran the script still gets a target it
                # can click rather than nothing at all.
                #
                # type="pil" and not "filepath": Gradio's filepath preprocess
                # calls ``processing_utils.save_pil_to_cache`` with a ``name``
                # argument, and this WebUI replaces that function with an older
                # one that has no such parameter -- so every filepath image
                # input in the host raises ``TypeError`` before a handler is
                # ever reached.
                # ``sources`` is one, and that is what makes the paperclip
                # work: with more than one, Gradio draws a chooser and the file
                # input is not in the DOM until somebody has picked "upload".
                # A chip this size has room for a picture and nothing else
                # anyway.
                attachment = gr.Image(label="Next message", type="pil", visible=False,
                                      sources=["upload"], show_label=False,
                                      show_download_button=False, container=False,
                                      scale=0, min_width=0,
                                      elem_id=ui.ident("chat", "image"),
                                      elem_classes=ui.classes("attachment"))
                # One line, and never more than six: the composer is the one
                # thing that must stay on screen, so it grows with what is
                # being written and then stops, and a longer message scrolls
                # inside its own box. Gradio's own rule for a box declared one
                # line tall is that Enter submits and Shift+Enter breaks the
                # line, which is what a messaging application does and what the
                # placeholder therefore says -- and the box is bound to
                # ``submit`` as well as to the button below, so the two agree.
                message = gr.Textbox(
                    label=None, lines=1, max_lines=6, show_label=False, container=False,
                    scale=1, placeholder="Message…  Enter sends, Shift+Enter for a new line.",
                    elem_id=ui.ident("chat", "message"))
                # Slide right and hold to dictate. Everything it does happens
                # in the browser -- the gesture, the capture, the encoding -- so
                # there is no click handler here at all: a Gradio round trip to
                # start a recording would put a network hop between the gesture
                # and the microphone opening.
                # Built and not held: everything it does happens in the
                # browser, so there is no Python handler to wire it to and
                # nothing here needs a reference to it afterwards.
                mc_voice_ui.microphone()
                # Send and Stop are one control in two states, in one place.
                # Two components rather than one because a single button cannot
                # carry two click handlers without both of them firing; only
                # ever one of them is on screen.
                send = gr.Button("Send", variant="primary", size="sm", scale=0, min_width=88,
                                 elem_id=ui.ident("chat", "send"))
                stop = gr.Button("Stop", variant="stop", size="sm", scale=0, min_width=88,
                                 visible=False, interactive=False,
                                 elem_id=ui.ident("chat", "stop"))

            # MESSAGE_EDIT_MODE. The composer's space, borrowed: the transcript
            # stays visible above it, which is the whole reason an edit does not
            # open a panel of its own.
            with gr.Row(visible=False, elem_id=ui.ident("chat", "edit"),
                        elem_classes=ui.classes("composer", "composer-edit")) as edit_row:
                # Said on the row rather than left to the buttons: a composer
                # that has quietly filled itself with a message from halfway up
                # the transcript is a composer somebody sends by accident.
                gr.HTML(f'<div class="{ui.PREFIX}-editing">\u270e Editing</div>',
                        elem_classes=ui.classes("editing"))
                # The same paperclip and the same chip as the composer, for
                # the same reason: a message that was sent with a picture is
                # edited as a whole, so the picture has to be changeable and
                # removable here rather than only at the moment it was first
                # attached. Empty for a message that has none, which is also
                # how one is added to a message that never had one.
                edit_attach = gr.Button("\U0001f4ce", size="sm", scale=0, min_width=44,
                                        elem_id=ui.ident("chat", "edit-attach"),
                                        elem_classes=ui.classes("icon-button"))
                edit_image = gr.Image(label="This message", type="pil", visible=False,
                                      sources=["upload"], show_label=False,
                                      show_download_button=False, container=False,
                                      scale=0, min_width=0,
                                      elem_id=ui.ident("chat", "edit-image"),
                                      elem_classes=ui.classes("attachment"))
                edit_box = gr.Textbox(label="Editing message", lines=2, max_lines=8,
                                      show_label=False, container=False, scale=1,
                                      placeholder="Editing this message…",
                                      elem_id=ui.ident("chat", "editor"))
                save_edit = gr.Button("Save", variant="primary", size="sm", scale=0,
                                      min_width=80)
                cancel_edit = gr.Button("Cancel", size="sm", scale=0, min_width=80)

        # -- THREADS_SCREEN ------------------------------------------------- #

        with gr.Column(visible=False, elem_id=ui.ident("chat", "threads"),
                       elem_classes=ui.classes("sheet", "sheet-screen")) as threads_screen:
            with gr.Row(elem_classes=ui.classes("sheet-head")):
                threads_back = gr.Button("‹ Back", size="sm", scale=0, min_width=76,
                                         elem_classes=ui.classes("sheet-back"))
                gr.Markdown("#### Threads")
                new_thread = gr.Button("New", variant="primary", size="sm", scale=0,
                                       min_width=68)
            search = gr.Textbox(label="Find a thread", placeholder="Filter by title…",
                                show_label=False, container=False,
                                elem_id=ui.ident("chat", "search"))
            threads = gr.Radio(label=None, show_label=False, choices=initial_threads,
                               value=initial_thread or None, container=False,
                               elem_id=ui.ident("chat", "threads"),
                               elem_classes=ui.classes("threads"))
            rename = gr.Button("Rename this thread", size="sm")
            rename_box = gr.Textbox(label="New title", visible=False, container=False,
                                    placeholder="New title…",
                                    elem_id=ui.ident("chat", "rename"))
            with gr.Row(visible=False) as rename_row:
                save_rename = gr.Button("Save", variant="primary", size="sm")
                cancel_rename = gr.Button("Cancel", size="sm")
            # Away from the primary controls and behind its own heading, which
            # is the only confirmation Gradio can honestly offer: there is no
            # dialog to put in front of it, so the distance is the deliberation.
            with gr.Group(elem_classes=ui.classes("destructive")):
                gr.Markdown("##### Danger zone", elem_classes=ui.classes("sheet-label"))
                delete = gr.Button("Delete this thread", size="sm", variant="stop")

        # -- CHARACTER_SCREEN ----------------------------------------------- #

        with gr.Column(visible=False, elem_id=ui.ident("chat", "character"),
                       elem_classes=ui.classes("sheet", "sheet-screen")) as character_screen:
            with gr.Row(elem_classes=ui.classes("sheet-head")):
                character_back = gr.Button("‹ Back", size="sm", scale=0, min_width=76,
                                           elem_classes=ui.classes("sheet-back"))
                gr.Markdown("#### Character")
            character = gr.Dropdown(
                label="Talking to", choices=_character_choices(),
                value=who or None, interactive=True,
                elem_id=ui.ident("chat", "who"))
            with gr.Row():
                edit_character = gr.Button("Edit", size="sm")
                new_character = gr.Button("New", size="sm")
                refresh_characters = gr.Button("↻ Refresh", size="sm")
            # Choosing, editing and creating a character are three things done
            # to the same object, so they are one screen: the drop-down is who
            # you are talking to, and the editor under it is that same
            # character, opened only when it is being changed.
            #
            # Which character the editor is *bound to* is this State and never
            # the drop-down above. They are the same name while an existing
            # character is being edited and deliberately different while a new
            # one is being written, and reading the drop-down for it is what
            # made New behave as Rename. See the note above ``_open_character``.
            editing = gr.State(NOT_EDITING)
            with gr.Group(visible=False,
                          elem_id=ui.ident("chat", "character-editor")) as character_editor:
                name = gr.Textbox(label="Name", elem_id=ui.ident("chat", "name"))
                # Beside the character file as ``<name>.png``, which is where
                # oobabooga keeps one and where importing a card already writes
                # it -- so a character imported with a face already has one and
                # this box is only for the ones that did not.
                character_face = gr.Image(label="Picture", type="pil", sources=["upload"],
                                          height=96, width=96, show_download_button=False,
                                          elem_id=ui.ident("chat", "character-face"),
                                          elem_classes=ui.classes("face"))
                context = gr.Textbox(label="Context", lines=6,
                                     placeholder="Who the character is. Shown to the model "
                                                 "before the first line of dialogue.")
                greeting = gr.Textbox(label="Greeting", lines=3)
                # What the model is actually told, built from the boxes above
                # and from the persona, and updated as they are typed into. It
                # is the answer to "what is the system prompt right now", which
                # nothing on this screen could answer before.
                system_preview = gr.Textbox(
                    label="System prompt in force", lines=6, interactive=False,
                    elem_id=ui.ident("chat", "system-preview"),
                    info="Built from the name, the Context and your persona. Read-only — "
                         "press Edit this to take a copy into the box below.")
                edit_system = gr.Button("Edit this system prompt", size="sm")
                system = gr.Textbox(label="System prompt override", lines=6,
                                    placeholder="Leave empty to use the built prompt above.",
                                    info="Set, this replaces the built prompt entirely and "
                                         "stops following the Context and the persona.")
                gr.Markdown(
                    "Save writes the override and the Advanced generation settings below "
                    "with the rest of the character, so a seed of −1 there is what makes "
                    "its replies vary.",
                    elem_classes=ui.classes("hint"))
                # Which voice reads this character aloud, and how, beside the
                # rest of what makes it that character. Built by the voice
                # module so the compact list here and the full one in Settings
                # cannot drift apart -- see mc_voice_ui.character_panel.
                character_voice = mc_voice_ui.character_panel()
                with gr.Row():
                    save_character = gr.Button("Save character", variant="primary", size="sm")
                    close_editor = gr.Button("Cancel", size="sm")
            import_card = gr.File(label="Import a character card",
                                  file_types=[".json", ".yaml", ".yml", ".png"],
                                  elem_id=ui.ident("chat", "import"))
            # Nested, because they are the answer to a question almost nobody
            # asks -- and every default here is the vendored package's own,
            # named rather than copied: a second set of literals in the UI is
            # how a panel quietly stops matching the engine behind it.
            with gr.Accordion("Advanced generation settings", open=False,
                              elem_classes=ui.classes("advanced")):
                with gr.Row():
                    temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0,
                                            step=0.05, value=DEFAULT_TEMPERATURE)
                    top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, step=0.01,
                                      value=DEFAULT_TOP_P)
                reply_tokens = gr.Slider(label="Reply tokens", minimum=64, maximum=4096,
                                         step=64, value=DEFAULT_MAX_REPLY_TOKENS)
                seed = gr.Number(label="Seed", value=RANDOM_SEED, precision=0,
                                 info=f"{RANDOM_SEED} draws a fresh seed for every reply.")
            with gr.Group(elem_classes=ui.classes("destructive")):
                gr.Markdown("##### Danger zone", elem_classes=ui.classes("sheet-label"))
                delete_character = gr.Button("Delete this character", size="sm",
                                             variant="stop")

        # -- PERSONA_SCREEN ------------------------------------------------- #

        with gr.Column(visible=False, elem_id=ui.ident("chat", "persona"),
                       elem_classes=ui.classes("sheet", "sheet-screen")) as persona_screen:
            with gr.Row(elem_classes=ui.classes("sheet-head")):
                persona_back = gr.Button("‹ Back", size="sm", scale=0, min_width=76,
                                         elem_classes=ui.classes("sheet-back"))
                gr.Markdown("#### You")
            persona_name = gr.Textbox(label="Your name", value=initial_persona.name)
            persona_face = gr.Image(label="Your picture", type="pil", sources=["upload"],
                                    height=96, width=96, show_download_button=False,
                                    value=_face_value(_persona_face()),
                                    elem_id=ui.ident("chat", "persona-face"),
                                    elem_classes=ui.classes("face"))
            persona_description = gr.Textbox(label="About you", lines=4,
                                             value=initial_persona.description)
            save_persona = gr.Button("Save", variant="primary", size="sm")

        # -- VOICE_SCREEN --------------------------------------------------- #

        voice = mc_voice_ui.sheet()
        # Two hidden boxes rather than a component with a value anybody can
        # read: one carries this process's page token into the page, and the
        # other carries an opaque one-shot handle to a reply that has finished.
        # Neither ever holds a transcript or a reply.
        voice_plumbing = mc_voice_ui.plumbing()

        # -- MESSAGE_ACTION_SHEET ------------------------------------------- #

        actions = _action_sheet()

    # -- wiring ----------------------------------------------------------- #
    #
    # Everything that changes what the transcript is goes through one output
    # list, so a handler cannot leave the transcript, the position map, the
    # header and the action sheet describing four different conversations.

    selection = {"sheet": actions["sheet"], "heading": actions["heading"],
                 "back": actions["back"], "pager": actions["pager"],
                 "forward": actions["forward"], "drop": actions["drop"],
                 "regenerate": actions["regenerate"], "continue": actions["continue"],
                 "resend": actions["resend"],
                 "edit": edit_row, "edit_box": edit_box, "edit_image": edit_image,
                 "composer": composer}
    view = ([transcript, positions, selected, status, header]
            + [selection[key] for key in SELECTION_ORDER])
    # Two more outputs than V1, and both are hidden values rather than
    # visibilities: the opaque id of the reply being spoken, and Python's half
    # of "is the composer busy". Appended at the end so every existing yield
    # keeps its meaning -- see :data:`LLM_RUNNING`.
    stream = [cancellation, transcript, positions, message, attachment, status, send, stop,
              voice_plumbing["turn"], voice_plumbing["run_state"]]
    sampling = [temperature, top_p, reply_tokens, seed]
    # The State first, then one visibility per surface: the order
    # :func:`_screens` answers in.
    screens = [surface, threads_screen, character_screen, persona_screen, voice["screen"]]

    # -- getting about ---------------------------------------------------- #

    # The workspace chooser is the shell's, so all this panel does is get its
    # own surfaces out of the way; the shell adds the second handler that opens
    # it. Two handlers on one button, which is what makes one control mean the
    # same thing in Conversation as it does in every other mode.
    menu.click(fn=_leave, inputs=[character, thread_state],
               outputs=screens + view, queue=False)
    for control in (threads_back, character_back, persona_back, voice["back"]):
        control.click(fn=_close_screens, outputs=screens, queue=False)

    to_threads.click(fn=_open_threads, inputs=[character, search],
                     outputs=screens + [threads], queue=False)
    to_character.click(fn=_open_character_screen, inputs=[character],
                       outputs=screens + [character], queue=False)
    to_persona.click(fn=_open_persona,
                     outputs=screens + [persona_name, persona_description, persona_face],
                     queue=False)

    # -- voice ------------------------------------------------------------- #
    #
    # The two switches are Forge settings and are written the moment they are
    # tapped, so the flyout and Settings -> Voice Chat are two views of one
    # value rather than two values that have to be kept in step.
    to_voice.click(fn=lambda: mc_voice_ui.open_sheet(_screens),
                   outputs=screens + [voice["readiness"], voice["engine"],
                                      voice["auto_send"], voice["auto_speak"]], queue=False)
    # ``input`` and not ``change``, for the reason the thread list gives above:
    # ``change`` also fires when the server puts a value in, and opening the
    # flyout puts both stored values in. Listening to it would write the
    # settings file every time somebody looked at the menu.
    voice["auto_send"].input(fn=mc_voice_ui.set_auto_send, inputs=[voice["auto_send"]],
                             outputs=[voice["auto_send"]], queue=False)
    voice["auto_speak"].input(fn=mc_voice_ui.set_auto_speak, inputs=[voice["auto_speak"]],
                              outputs=[voice["auto_speak"]], queue=False)

    # -- threads ---------------------------------------------------------- #

    # ``input`` and not ``change``: this code refills the list whenever a
    # thread is created, deleted or branched, and ``change`` fires on the
    # refill -- which would re-open the thread the handler had just opened, and
    # send the reader home from the screen they were working in. Every handler
    # that moves the value returns the transcript with it, so nothing is lost
    # by listening only to the tap.
    _picked(threads)(fn=_open_thread_home, inputs=[character, threads, message],
                     outputs=[thread_state, message] + view + screens, queue=False)
    search.change(fn=lambda person, text: gr.update(choices=_thread_choices(person, text)),
                  inputs=[character, search], outputs=[threads], queue=False)
    new_thread.click(fn=_new_thread, inputs=[character, search],
                     outputs=[threads, thread_state] + view + screens, queue=False)
    delete.click(fn=_delete_thread, inputs=[character, thread_state, search],
                 outputs=[threads, thread_state] + view, queue=False)

    rename.click(fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
                 outputs=[rename_box, rename_row], queue=False)
    cancel_rename.click(fn=lambda: (gr.update(value="", visible=False), gr.update(visible=False)),
                        outputs=[rename_box, rename_row], queue=False)
    for control in (rename_box.submit, save_rename.click):
        control(fn=_rename_thread, inputs=[character, thread_state, rename_box, search],
                outputs=[threads, rename_box, rename_row, status, header], queue=False)

    # -- the character and the persona ------------------------------------ #

    # One output list for every handler that touches the editor, in the order
    # ``_editor_fields`` answers in, so none of them can leave the boxes and
    # the character being written to describing different characters.
    # The voice controls, in the order ``_editor_fields`` answers in. Held as a
    # list of its own so the two places that build an editor state -- the fields
    # and the closed-editor no-op -- agree about how many there are.
    voice_fields = ([character_voice["chosen"], character_voice["custom"],
                     character_voice["delivery"]] + character_voice["sliders"])

    editor = ([character_editor, editing, name, context, greeting, system]
              + sampling + [system_preview, status, character_face] + voice_fields)

    switched = character.change(fn=_select_character, inputs=[character, search, message],
                                outputs=[threads, thread_state, message] + view
                                + [name, context, greeting, system] + sampling
                                + [editing, system_preview, character_face]
                                + voice_fields, queue=False)
    switched.then(fn=_faces_update, inputs=[character], outputs=[transcript], queue=False)
    edit_character.click(fn=_open_character, inputs=[character], outputs=editor, queue=False)
    new_character.click(fn=_new_character, outputs=editor, queue=False)
    refresh_characters.click(fn=_refresh_characters, inputs=[character],
                             outputs=[character, status], queue=False)
    close_editor.click(fn=_cancel_character, inputs=[character], outputs=editor, queue=False)
    # The preview follows what is being typed rather than what is on disk, so
    # the three boxes it is built from all refresh it.
    for box in (name, context, system):
        box.change(fn=_system_preview, inputs=[name, context, system],
                   outputs=[system_preview], queue=False)
    edit_system.click(fn=_adopt_system_prompt, inputs=[name, context, system],
                      outputs=[system, status], queue=False)
    saving = save_character.click(
        fn=_save_character,
        inputs=([editing, name, context, greeting, system] + sampling + [character_face]
                + [character_voice["chosen"], character_voice["custom"]]
                + character_voice["sliders"]),
        outputs=[character, editing, status], queue=False)
    saving.then(fn=_faces_update, inputs=[character], outputs=[transcript], queue=False)
    # Shown and hidden by the checkbox above it, on ``change`` rather than
    # ``input`` so that loading a character which has its own delivery reveals
    # the sliders holding it -- a group left shut over four values that are in
    # force would be four settings nobody can see.
    character_voice["custom"].change(
        fn=lambda wanted: gr.update(visible=bool(wanted)),
        inputs=[character_voice["custom"]], outputs=[character_voice["delivery"]],
        queue=False)
    delete_character.click(fn=_delete_character, inputs=[character],
                           outputs=[character, character_editor, editing, status], queue=False)
    imported = import_card.upload(
        fn=_import_character, inputs=[import_card],
        outputs=[character, character_editor, editing, status], queue=False)
    # A ``.png`` card carries its own picture, and importing one writes it
    # beside the character. The transcript is told, because the character it
    # has just landed on is the one being talked to.
    imported.then(fn=_faces_update, inputs=[character], outputs=[transcript], queue=False)

    persona_saved = save_persona.click(
        fn=_save_persona, inputs=[persona_name, persona_description, persona_face],
        outputs=[status], queue=False)
    persona_saved.then(fn=_faces_update, inputs=[character], outputs=[transcript],
                       queue=False)

    # -- the per-message actions ------------------------------------------ #

    transcript.select(fn=_select_message,
                      inputs=[character, thread_state, positions, selected],
                      outputs=view, queue=False)
    actions["close"].click(fn=_close_selection, inputs=[character, thread_state],
                           outputs=view, queue=False)

    actions["back"].click(fn=_page_version(-1), inputs=[character, thread_state, selected],
                          outputs=view, queue=False)
    actions["forward"].click(fn=_page_version(1), inputs=[character, thread_state, selected],
                             outputs=view, queue=False)
    actions["drop"].click(fn=_drop_version, inputs=[character, thread_state, selected],
                          outputs=view, queue=False)

    # One shape for both roles now. Edit used to be able to move the panel onto
    # another thread -- editing one of your own messages branched -- and
    # answered with the thread list and the open thread in front of the view to
    # say so. It edits in place, so there is nothing in front of the view.
    actions["edit"].click(fn=_open_editor,
                          inputs=[character, thread_state, selected],
                          outputs=view, queue=False)
    save_edit.click(fn=_commit_edit,
                    inputs=[character, thread_state, selected, edit_box, edit_image],
                    outputs=view, queue=False)
    cancel_edit.click(fn=lambda: (gr.update(visible=False), gr.update(value=None, visible=False),
                                  gr.update(visible=True)),
                      outputs=[edit_row, edit_image, composer], queue=False)

    actions["branch"].click(fn=_branch_here, inputs=[character, thread_state, selected, search],
                            outputs=[threads, thread_state] + view, queue=False)
    actions["delete"].click(fn=_delete_message, inputs=[character, thread_state, selected],
                            outputs=view, queue=False)
    actions["delete_from"].click(fn=_delete_from, inputs=[character, thread_state, selected],
                                 outputs=view, queue=False)

    # -- the attachment ---------------------------------------------------- #
    #
    # The paperclip only makes the chip visible. Opening the file picker is the
    # browser's own, forwarded to this component's file input by
    # javascript/llm_studio.js -- a press that reaches Python and comes back is
    # a round trip to open a dialog that was already one tap away.
    attach.click(fn=_offer_attachment, outputs=[attachment, status], queue=False)
    attachment.clear(fn=_cleared_attachment, outputs=[attachment, status], queue=False)
    edit_attach.click(fn=lambda: gr.update(visible=True), outputs=[edit_image], queue=False)
    edit_image.clear(fn=lambda: gr.update(visible=False), outputs=[edit_image], queue=False)

    # -- sending, and the three ways of asking again ---------------------- #

    sent = [character, thread_state, message, attachment] + sampling
    replying = send.click(fn=_send, inputs=sent, outputs=stream, show_progress="minimal")
    # The same pathway from the keyboard. Gradio fires ``submit`` on Enter for
    # a composer that is one line tall, which is what this one starts as.
    submitted = message.submit(fn=_send, inputs=sent, outputs=stream,
                               show_progress="minimal")
    # The only streaming handler whose outputs carry the thread list and the
    # open thread: regenerating in the middle of a thread branches, and a panel
    # left pointing at the thread it came from would apply the next action to
    # the wrong conversation.
    regenerating = actions["regenerate"].click(
        fn=_regenerate, inputs=[character, thread_state, selected] + sampling + [search],
        outputs=[threads, thread_state] + stream, show_progress="minimal")
    # The same handler, nominated from the bubble instead of from the sheet.
    # One tap rather than three, which is the whole of what the icon is for.
    again = regenerate_now.click(
        fn=_regenerate_reply,
        inputs=[character, thread_state, positions, regenerate_at] + sampling + [search],
        outputs=[threads, thread_state] + stream, show_progress="minimal")
    continuing = actions["continue"].click(
        fn=_continue, inputs=[character, thread_state, selected] + sampling,
        outputs=stream, show_progress="minimal")
    resending = actions["resend"].click(
        fn=_resend, inputs=[character, thread_state, selected] + sampling,
        outputs=stream, show_progress="minimal")

    # The three that start from the action sheet put it away as they go: the
    # reply is arriving in the transcript behind it, and Stop is in the
    # composer, which the sheet is covering.
    for control in (actions["regenerate"], actions["continue"], actions["resend"]):
        control.click(fn=lambda: gr.update(visible=False), outputs=[actions["sheet"]],
                      queue=False)

    # Built once and attached to every run below, so a seventh way of producing
    # a reply cannot be added without either joining this loop or being visibly
    # absent from it. Section 49's requirement is exactly that the registration
    # be structurally shared.
    speech_marker = mc_voice_ui.speech_marker(take_completed_reply, _character_named)

    for run in (replying, submitted, regenerating, again, continuing, resending):
        # The thread list is refreshed because an untitled thread has just been
        # named, and the selection is dropped because the message it pointed at
        # may not be the message that is there now.
        run.then(fn=lambda person, text: gr.update(choices=_thread_choices(person, text)),
                 inputs=[character, search], outputs=[threads])
        run.then(fn=_close_selection, inputs=[character, thread_state], outputs=view)
        # ``success`` and deliberately not ``then``. Gradio runs a ``then``
        # continuation whether or not the event before it raised, which would
        # make "the run reached its terminal callback" the trigger for speaking
        # -- and a run that failed reaches it too. ``success`` runs only after
        # an event that did not raise, and the handler still checks what the run
        # actually left behind, because a nominally successful terminal callback
        # is not the same claim as a whole reply.
        run.success(fn=speech_marker, inputs=[character],
                    outputs=[voice_plumbing["token"]])

    stop.click(fn=_cancel, inputs=[cancellation],
               outputs=[status, send, stop, voice_plumbing["turn"],
                        voice_plumbing["run_state"]],
               cancels=[replying, submitted, regenerating, again, continuing, resending],
               queue=False)

    return {"status": status, "transcript": transcript, "header": header,
            "persona": (persona_name, persona_description),
            # What the shell wires: the state chip opens the model sheet, and
            # the menu opens the workspace chooser -- the same two sheets its
            # own bar opens, from the same corner of the screen.
            "model": [model], "modes": menu, "chip": model}


_MODEL_LABEL = "● Model"
"""What the header's state control says before the shell has told it anything."""


def _action_sheet() -> dict:
    """The per-message actions, as a sheet over the bottom of the transcript.

    Built once and re-labelled, because Gradio cannot create a control in
    response to a click: which of these apply to the message in hand is said by
    hiding the ones that do not, and by :func:`_selection_updates`.
    """
    with gr.Column(visible=False, elem_id=ui.ident("chat", "actions"),
                   elem_classes=ui.classes("sheet", "sheet-bottom",
                                           "message-actions")) as sheet:
        with gr.Row(elem_classes=ui.classes("sheet-head")):
            heading = gr.HTML(elem_id=ui.ident("chat", "selection"))
            close = gr.Button("✕", size="sm", scale=0, min_width=44,
                              elem_classes=ui.classes("icon-button"))
        with gr.Row(elem_classes=ui.classes("message-actions-row")):
            back = gr.Button("◀", size="sm", min_width=44, visible=False)
            pager = gr.HTML(visible=False, elem_classes=ui.classes("pager"))
            forward = gr.Button("▶", size="sm", min_width=44, visible=False)
            drop = gr.Button("Delete this version", size="sm", visible=False)
        with gr.Row(elem_classes=ui.classes("message-actions-row")):
            edit = gr.Button("Edit", size="sm")
            regenerate = gr.Button("Regenerate", size="sm", visible=False,
                                   elem_id=ui.ident("chat", "regenerate"))
            carry_on = gr.Button("Continue", size="sm", visible=False)
            resend = gr.Button("Send again from here", size="sm", visible=False)
            branch = gr.Button("Branch from here", size="sm")
        # Last, and behind a rule of its own: nothing above this line loses
        # anything, and everything below it does.
        with gr.Group(elem_classes=ui.classes("destructive")):
            with gr.Row(elem_classes=ui.classes("message-actions-row")):
                delete = gr.Button("Delete message", size="sm", variant="stop")
                delete_from = gr.Button("Delete from here", size="sm", variant="stop")

    return {
        "sheet": sheet, "heading": heading, "close": close,
        "back": back, "pager": pager, "forward": forward, "drop": drop,
        "edit": edit, "regenerate": regenerate, "continue": carry_on, "resend": resend,
        "branch": branch, "delete": delete, "delete_from": delete_from,
    }


def _picked(component):
    """The event that fires when a *user* picks, not when this code refills.

    The same helper the shell's model chooser needs and for the same reason,
    restated rather than imported: this module builds a panel and must not
    import the shell to do it.
    """
    return getattr(component, "input", None) or component.change


# --------------------------------------------------------------------------- #
# Stores
# --------------------------------------------------------------------------- #


def _characters():
    from prompt_master.chat.characters import CharacterStore

    return CharacterStore.from_paths(mc_llm_paths.app_paths())


def _chats():
    from prompt_master.chat.history import ChatStore

    return ChatStore.from_paths(mc_llm_paths.app_paths())


def _persona():
    from prompt_master.chat.characters import load_persona

    return load_persona(mc_llm_paths.app_paths())


def _persona_face():
    from prompt_master.chat.characters import persona_avatar

    try:
        return persona_avatar(mc_llm_paths.app_paths())
    except Exception:
        logger.debug("Model Chain: could not read your picture", exc_info=True)
        return None


def _character_face(who: str):
    try:
        return _characters().avatar_for(who) if who else None
    except Exception:
        logger.debug("Model Chain: could not read %s's picture", who, exc_info=True)
        return None


def _face_value(path):
    """A picture for a picker to show, or nothing.

    The host is asked to serve it on the way past. Gradio shows a path-valued
    Image by handing the browser a ``file=`` URL, and a URL for a file Gradio
    was never told about is a broken picture in a box that is supposed to be
    showing somebody what they already chose.
    """
    if path is None:
        return None
    mc_llm_attachments.allow(Path(path))
    return str(path)


def _faces(who: str) -> list:
    """``[you, the character]`` -- what the transcript draws beside each message.

    In that order because that is the order Gradio's Chatbot takes them, and it
    already puts the first on the right of your messages and the second on the
    left of the replies. Which is the whole request: a long thread of grey
    blocks has nothing to scan for, and "which of these did I write" should not
    be a question the reader has to answer by reading.

    Whoever has not chosen a picture gets a drawn one rather than nothing --
    a face on one side and a gap on the other is worse than either, because the
    gap reads as a message that failed to load. See
    :func:`mc_llm_attachments.default_avatar`, and note the two are drawn at
    opposite hues so they cannot come out the same colour.

    Never raises: a transcript that cannot draw a face is a transcript, and a
    panel that would not build because of one is not.
    """
    try:
        person = _persona()
        yours = _persona_face() or mc_llm_attachments.default_avatar(
            person.display, mc_llm_attachments.OPPOSITE)
        theirs = _character_face(who) or mc_llm_attachments.default_avatar(
            who or "Character")
        return [mc_llm_attachments.file_data(yours), mc_llm_attachments.file_data(theirs)]
    except Exception:
        logger.debug("Model Chain: could not work out the transcript's faces", exc_info=True)
        return [None, None]


def _faces_update(who: str):
    """The transcript, told about a face that has just changed.

    Only the faces: an update carrying no value leaves the messages exactly as
    they are, which matters because this is chained after handlers that have
    already put the right thread on screen.
    """
    return gr.update(avatar_images=_faces(who))


def _character_choices() -> list[str]:
    try:
        return _characters().names()
    except Exception:
        logger.debug("Model Chain: could not list characters", exc_info=True)
        return []


def _thread_choices(who: str, filter_text: str = "") -> list[tuple[str, str]]:
    if not who:
        return []
    try:
        listing = _chats().listing(who)
    except Exception:
        logger.debug("Model Chain: could not list threads", exc_info=True)
        return []
    needle = (filter_text or "").strip().casefold()
    return [(info.title, info.identifier) for info in listing
            if not needle or needle in info.title.casefold()]


def _load(who: str, identifier: str):
    """One thread, with its pictures where pictures now live.

    The migration is here rather than in a script somebody has to run: a chat
    written before there was an attachment folder carries its stills inline,
    and the first time it is opened they are written out to files and the chat
    is saved without them. Once, per chat, invisibly -- and a chat that has
    nothing to move is not written at all.
    """
    if not who or not identifier:
        return None
    try:
        conversation = _chats().load(who, identifier)
    except Exception:
        logger.debug("Model Chain: could not load thread %s", identifier, exc_info=True)
        return None
    try:
        if mc_llm_attachments.adopt(conversation, who):
            _chats().save(conversation)
            logger.info("Model Chain: moved this thread's attachments into %s",
                        mc_llm_attachments.folder(who))
    except Exception:
        logger.warning("Model Chain: could not move this thread's attachments onto disk; "
                       "they stay inside the chat file", exc_info=True)
    return conversation


# --------------------------------------------------------------------------- #
# The transcript, the header, and the map back to the conversation
# --------------------------------------------------------------------------- #


def _view(conversation) -> tuple[list[list[str | None]], list[list[int]]]:
    """The Chatbot value, and where each message ended up in it.

    Gradio 4.40 predates the message-shaped Chatbot value, and section 5 says
    to target the components the host actually has rather than newer Gradio
    assumptions. So a conversation is a list of ``[user, bot]`` pairs, and the
    pairing is what makes a click's ``(row, column)`` not a message index: two
    replies in a row become two rows with an empty left side, and one exchange
    is one row holding two messages.

    Both are produced here, in one pass over the messages, because they are one
    fact. Deriving the map somewhere else -- from the rows, or by pairing again
    -- is how a click would come to name the message beside the one clicked.
    """
    from prompt_master.chat.history import ASSISTANT, USER

    rows: list[list[str | None]] = []
    positions: list[list[int]] = []
    if conversation is None:
        return rows, positions
    for index, message in enumerate(conversation.messages):
        body = _body(message)
        if message.role == USER:
            rows.append([body, None])
            positions.append([len(rows) - 1, 0, index])
        elif rows and rows[-1][1] is None and message.role == ASSISTANT:
            rows[-1][1] = body
            positions.append([len(rows) - 1, 1, index])
        else:
            rows.append([None, body])
            positions.append([len(rows) - 1, 1, index])
    return rows, positions


def _body(message) -> str:
    """One message as the transcript draws it: the picture, then the words.

    The picture itself rather than its name. A conversation about a photograph
    that shows the reader a line of italic text saying a photograph was
    attached is a conversation missing half of itself -- most of all when it is
    reopened a week later, which is the case this was reported from.

    An image the file for has gone answers with a sentence saying so, because a
    message that was sent with a picture is not the same message without one.
    """
    body = message.text
    if not getattr(message, "attached", False):
        return body
    shown = (mc_llm_attachments.markup(message.image_path, message.image_name)
             if message.image_path
             # A chat whose pictures could not be moved onto disk still shows
             # them; there is simply nothing to point the browser at, so the
             # bytes go inline as they always did. Rare, and better than a
             # transcript that has lost a picture the file still holds.
             else f'<img src="{message.image}" alt="{ui.escape(message.image_name or "attached image")}"'
                  f' class="mc-llm-attached">')
    return f"{shown}\n\n{body}" if body else shown


def _transcript(conversation) -> list[list[str | None]]:
    """The Chatbot value alone, for callers that do not need the map."""
    return _view(conversation)[0]


def _heading(who: str | None, conversation) -> str:
    """The header's two lines: who you are talking to, and about what.

    Two lines rather than one sentence, because they are answers to two
    different questions and only the first is asked often. On a narrow display
    the thread title is the one that truncates.

    ``who`` is only consulted when there is no conversation to ask: a thread
    knows which character it belongs to, and a header built from the panel's
    idea of that rather than the thread's is a header that can disagree with
    the transcript under it.
    """
    person = (getattr(conversation, "character", None) or who or "No character").strip()
    title = getattr(conversation, "title", None) or "No thread"
    return (f'<div class="{ui.PREFIX}-heading">'
            f'<span class="{ui.PREFIX}-heading-who">{ui.escape(person)}</span>'
            f'<span class="{ui.PREFIX}-heading-thread">{ui.escape(title)}</span>'
            f'</div>')


def _message_at(positions, row, column) -> int:
    """The message a click on ``(row, column)`` names, or :data:`NO_SELECTION`."""
    for entry in positions or ():
        try:
            at_row, at_column, index = entry
        except (TypeError, ValueError):
            continue
        if int(at_row) == int(row) and int(at_column) == int(column):
            return int(index)
    return NO_SELECTION


def _reply_at(positions, ordinal) -> int:
    """The message the ``ordinal``-th reply in the transcript is.

    What the regenerate icon on a bubble sends back. The browser cannot know a
    message index -- ``_view`` pairs turns, so the row a bubble is on is not
    one -- and it should not be taught to: all it can honestly report is *which
    reply this is*, counted down the transcript, and the map that answers what
    that means is already in the panel's hands.

    Counted against ``positions`` rather than against the conversation, for the
    same reason the click map exists at all: the transcript the icons were
    drawn on is the one ``positions`` describes, and a reply is only ever the
    ``n``-th thing on screen.
    """
    try:
        wanted = int(str(ordinal).strip())
    except (TypeError, ValueError):
        return NO_SELECTION
    if wanted < 0:
        return NO_SELECTION
    replies = []
    for entry in positions or ():
        try:
            _, at_column, index = entry
        except (TypeError, ValueError):
            continue
        if int(at_column) == 1:
            replies.append(int(index))
    return replies[wanted] if wanted < len(replies) else NO_SELECTION


def _regenerate_reply(who, identifier, positions, ordinal, temperature, top_p,
                      reply_tokens, seed, filter_text=""):
    """Regenerate the reply whose icon was tapped.

    A translation and nothing else: the ordinal becomes a message index and
    :func:`_regenerate` does the rest, so the icon and the sheet's Regenerate
    cannot come to mean two different things -- including the branching, which
    is the part it would be worst to have two of.
    """
    index = _reply_at(positions, ordinal)
    if index == NO_SELECTION:
        yield _here(identifier, _idle(_load(who, identifier), "", None,
                                      "That reply is no longer in the thread. Reopen it "
                                      "and try again.", "warn"))
        return
    yield from _regenerate(who, identifier, index, temperature, top_p, reply_tokens,
                           seed, filter_text)


def _selection_updates(conversation, index: int, editing: bool = False) -> list:
    """What the action sheet shows for message ``index``.

    Returned in the order :data:`SELECTION_ORDER` lists, which is the one thing
    about this function that has to be kept in step with the layout.

    ``editing`` is the one state that is not a property of the message: the
    in-place editor is open on it. Everywhere else this is left alone, and the
    last four answers put the editor away, empty its picture chip and give the
    composer back.
    """
    from prompt_master.chat.history import ASSISTANT

    hidden = gr.update(visible=False)
    home = [gr.update(visible=False), gr.update(), gr.update(value=None, visible=False),
            gr.update(visible=True)]
    if (conversation is None or index < 0 or index >= len(conversation.messages)):
        return [gr.update(visible=False), gr.update(value=""), hidden, hidden, hidden, hidden,
                hidden, hidden, hidden] + home

    message = conversation.messages[index]
    reply = message.role == ASSISTANT
    last = index == len(conversation.messages) - 1
    versions = len(message.versions)
    speaker = "the character" if reply else "you"
    opening = " ".join(message.text.split())[:80] or "(empty)"

    return [
        # Put away while the editor is open: the sheet covers the bottom of the
        # transcript and the editor is under it, so leaving both up is two
        # panels arguing over the same corner of the screen.
        gr.update(visible=not editing),
        gr.update(value=ui.notice(f"Message {index + 1} of {len(conversation.messages)}, "
                                  f"from {speaker}: {opening}"
                                  + ("…" if len(message.text) > 80 else ""))),
        gr.update(visible=versions > 1, interactive=message.active > 0),
        gr.update(visible=versions > 1,
                  value=f'<div class="{ui.PREFIX}-pager">'
                        f'{message.active + 1}/{versions}</div>'),
        gr.update(visible=versions > 1, interactive=message.active < versions - 1),
        gr.update(visible=versions > 1),
        # Regenerate writes this reply again and keeps the one it had as a
        # version; continuing only makes sense for the reply still at the end,
        # because anything after it would be answering a question that has
        # already been answered again.
        gr.update(visible=reply),
        gr.update(visible=reply and last and bool(message.text.strip())),
        gr.update(visible=not reply),
        # The edit row is put away and the composer comes back: a refresh is a
        # return to CHAT_HOME, whatever the panel was doing before it -- unless
        # this refresh is the one that opened the editor.
        gr.update(visible=editing),
        gr.update(value=message.text),
        _editable_picture(message) if editing else gr.update(value=None, visible=False),
        gr.update(visible=not editing),
    ]


def _editable_picture(message):
    """The editor's chip, holding this message's picture when there is one.

    A path rather than the bytes: the component is being given a value to show,
    and handing it a file it can fetch costs a URL where handing it a decoded
    photograph would cost the photograph, twice -- once to send and once when
    the edit comes back.

    Nothing to show for a message with no attachment, and nothing to show for
    one whose file has gone. The second is deliberate rather than an oversight:
    an empty chip means "this message has no picture now", and offering to keep
    one that is not there would be offering to keep nothing.
    """
    found = mc_llm_attachments.locate(getattr(message, "image_path", ""))
    if found is None:
        return gr.update(value=None, visible=False)
    mc_llm_attachments.allow(found)
    return gr.update(value=str(found), visible=True)


def _refresh(conversation, note: str, kind: str = "info",
             index: int = NO_SELECTION, editing: bool = False) -> list:
    """Every output the ``view`` list names, for one state of one thread.

    One function for all of them because they are one fact: the rows, the map a
    click is read against, the header, the message the action sheet applies to
    and what that sheet shows all come from the same conversation, and a
    handler that returned four of the five would leave the fifth describing a
    thread that is no longer on screen.
    """
    rows, positions = _view(conversation)
    return ([rows, positions, index, ui.notice(note, kind), _heading(None, conversation)]
            + _selection_updates(conversation, index, editing))


def _reopen(who, identifier, note: str, kind: str = "info",
            index: int = NO_SELECTION, editing: bool = False) -> list:
    """:func:`_refresh` for a thread read back off disk."""
    return _refresh(_load(who, identifier), note, kind, index, editing)


# --------------------------------------------------------------------------- #
# The surfaces
# --------------------------------------------------------------------------- #


def _screens(name: str = "") -> list:
    """Which surface is open, and one visibility per surface.

    The name comes back first because it is the answer the ``surface`` State
    holds, and the State is what makes ``\u2630`` a toggle rather than a
    control that can only ever open something. Everything else is derived from
    it, so "only one open at a time" is a property of this function rather than
    a rule every handler has to remember.
    """
    wanted = name if name in SCREENS else ""
    return [wanted] + [gr.update(visible=(wanted == key)) for key in SCREENS]


def _close_screens() -> list:
    return _screens("")


def _leave(who, identifier) -> list:
    """Put this panel's own surfaces away, for a control that opens the shell's.

    The message selection goes with them: the action sheet applies to a message
    the reader is about to stop looking at, and a sheet left open underneath
    another sheet is the second half of every "why is this still here?".
    """
    return _close_screens() + _close_selection(who, identifier)


def _open_threads(who, filter_text) -> list:
    return _screens("threads") + [gr.update(choices=_thread_choices(who, filter_text))]


def _open_character_screen(who) -> list:
    return _screens("character") + [gr.update(choices=_character_choices(),
                                              value=who or None)]


def _open_persona() -> list:
    person = _persona()
    return _screens("persona") + [person.name, person.description,
                                  _face_value(_persona_face())]


def _offer_attachment():
    """Show the picture chip, and say when sending one would be pointless.

    Warned here rather than when the reply fails: whether a picture can be sent
    depends on the model that is running, and finding that out after writing a
    message is finding it out too late.

    All this does is make the chip visible. The file picker is the browser's
    own and is opened by javascript/llm_studio.js, which forwards this press to
    the component's file input -- so the ordinary case is a picker, and the
    case where that script did not run is a small target to click instead of
    nothing at all.
    """
    try:
        sees = mc_llm_runtime.config().sees
    except Exception:
        logger.debug("Model Chain: could not read the vision configuration", exc_info=True)
        sees = True
    if sees:
        return gr.update(visible=True), ui.notice("Choose a picture for the next message.")
    return gr.update(visible=True), ui.notice(
        "The model running has no vision projector, so an attached image cannot be sent to "
        "it. Choose a multimodal backbone in Setup, or send the message without a picture.",
        "warn")


def _cleared_attachment():
    """The component's own ✕. The chip goes away with the picture it held."""
    return gr.update(visible=False), ui.notice("Ready.")


def _select_character(who, filter_text, typed=""):
    """Switch character: reload its threads, open the newest, fill the editor."""
    from prompt_master.chat.characters import Character

    choices = _thread_choices(who, filter_text)
    identifier = choices[0][1] if choices else ""
    conversation = _load(who, identifier)
    lifted = gr.update()
    waiting = (_unanswered(conversation) if conversation is not None
               and not (typed or "").strip() else NO_SELECTION)
    if waiting != NO_SELECTION:
        lifted = conversation.messages[waiting].text
        conversation.delete_from(waiting)
        _chats().save(conversation)
    try:
        loaded = _characters().load(who) if who else Character(name="")
    except Exception:
        loaded = Character(name=who or "")
    mc_llm_state.remember(character=who or "", thread=identifier)
    note = f"{len(choices)} thread{'s' if len(choices) != 1 else ''}."
    # The editor follows the selection. It is usually shut, and when it is not,
    # leaving it bound to the character that *was* selected would have the next
    # Save write this character's boxes over that character's file.
    return ([gr.update(choices=choices, value=identifier or None), identifier, lifted]
            + _refresh(conversation, note)
            + [loaded.name, loaded.context, loaded.greeting, loaded.system,
               loaded.temperature, loaded.top_p, loaded.max_reply_tokens, loaded.seed,
               loaded.name or NOT_EDITING,
               _system_preview(loaded.name, loaded.context, loaded.system),
               _face_value(_character_face(loaded.name))]
            + _voice_fields(loaded))


def _open_thread(who, identifier, typed=""):
    """Open a thread. Two answers in front of the refresh: which, and the box.

    A thread that ends in a message of yours nobody ever answered opens with
    that message back in the composer -- see :func:`_unanswered`. Never over
    something half-written: ``typed`` is what is in the box now, and a box with
    anything in it is left exactly as it is. The message stays in the thread in
    that case, where **Edit** will still take it back once the box is free.
    """
    conversation = _load(who, identifier)
    if conversation is None:
        return [identifier or "", gr.update()] + _refresh(None, "Choose a thread.", "warn")
    mc_llm_state.remember(character=who or "", thread=identifier)
    waiting = _unanswered(conversation) if not (typed or "").strip() else NO_SELECTION
    if waiting != NO_SELECTION:
        lifted = conversation.messages[waiting].text
        conversation.delete_from(waiting)
        _chats().save(conversation)
        return ([identifier, lifted]
                + _refresh(conversation, "This message never got a reply, so it is back in "
                                         "the box. Send it again when you are ready."))
    return [identifier, gr.update()] + _refresh(conversation, conversation.title)


def _open_thread_home(who, identifier, typed=""):
    """Tapping a thread opens it and returns to the conversation."""
    return _open_thread(who, identifier, typed) + _close_screens()


def _new_thread(who, filter_text):
    if not who:
        return ([gr.update(), ""]
                + _refresh(None, "Choose a character first.", "warn")
                + _screens("threads"))  # stay where the message can be read
    store = _chats()
    conversation = store.new(who)
    _greet(conversation, who)
    store.save(conversation)
    mc_llm_state.remember(character=who, thread=conversation.identifier)
    return ([gr.update(choices=_thread_choices(who, filter_text),
                       value=conversation.identifier), conversation.identifier]
            + _refresh(conversation, "New thread.")
            + _close_screens())


def _greet(conversation, who: str) -> None:
    """Open a new thread with the character's greeting, if it has one."""
    from prompt_master.chat.history import ASSISTANT
    from prompt_master.chat.prompt import greeting_text

    try:
        character = _characters().load(who)
    except Exception:
        return
    text = greeting_text(character, _persona())
    if text.strip():
        conversation.append(ASSISTANT, text)


def _delete_thread(who, identifier, filter_text):
    if not (who and identifier):
        return [gr.update(), ""] + _refresh(None, "Choose a thread first.", "warn")
    _chats().delete(who, identifier)
    choices = _thread_choices(who, filter_text)
    following = choices[0][1] if choices else ""
    return ([gr.update(choices=choices, value=following or None), following]
            + _reopen(who, following, "Deleted."))


def _rename_thread(who, identifier, title, filter_text):
    conversation = _load(who, identifier)
    if conversation is None or not (title or "").strip():
        return (gr.update(), gr.update(visible=False), gr.update(visible=False),
                ui.notice("Nothing to rename.", "warn"), gr.update())
    conversation.title = title.strip()
    _chats().save(conversation)
    return (gr.update(choices=_thread_choices(who, filter_text), value=identifier),
            gr.update(value="", visible=False), gr.update(visible=False),
            ui.notice("Renamed."), _heading(who, conversation))


# --------------------------------------------------------------------------- #
# Per-message actions
# --------------------------------------------------------------------------- #


def _select_message(who, identifier, positions, current, event: gr.SelectData = None):
    """A tap on a bubble nominates the message the action sheet applies to.

    Tapping the message that is already nominated puts the sheet away again.
    The same gesture opens and closes it because there is only one gesture: a
    Chatbot bubble has no second affordance to dismiss from.
    """
    conversation = _load(who, identifier)
    index = NO_SELECTION
    where = getattr(event, "index", None)
    if isinstance(where, (list, tuple)) and len(where) >= 2:
        index = _message_at(positions, where[0], where[1])
    elif isinstance(where, int):
        # Some hosts report a flat index. Read it as the row, and take the
        # reply half of it, which is what a flat index counts.
        index = _message_at(positions, where, 1)
    if index == NO_SELECTION:
        return _refresh(conversation,
                        "That part of the transcript is not a message.", "warn")
    if _selection(current) == index:
        return _refresh(conversation, "Ready.")
    return _refresh(conversation, "Ready.", index=index)


def _selection(value) -> int:
    """``value`` as a message index. A State that has never been set is None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return NO_SELECTION


def _close_selection(who, identifier):
    return _reopen(who, identifier, "Ready.")


def _page_version(step: int):
    """Show the previous or next version of the selected reply."""
    def page(who, identifier, index):
        conversation = _load(who, identifier)
        if conversation is None or not (0 <= index < len(conversation.messages)):
            return _refresh(conversation, "Choose a message first.", "warn")
        message = conversation.messages[index]
        message.show(message.active + step)
        _chats().save(conversation)
        # The sheet stays open on the message being paged: reading three
        # variants is three taps on one control, not three round trips through
        # the transcript.
        return _refresh(conversation,
                        f"Showing version {message.active + 1} of {len(message.versions)}.",
                        index=index)

    return page


def _drop_version(who, identifier, index):
    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return _refresh(conversation, "Choose a message first.", "warn")
    message = conversation.messages[index]
    if len(message.versions) <= 1:
        return _refresh(conversation,
                        "This message has only one version — deleting it would delete the "
                        "message.", "warn", index=index)
    message.drop_version()
    _chats().save(conversation)
    return _refresh(conversation, "Version deleted.", index=index)


def _open_editor(who, identifier, index):
    """Open the editor on one message -- yours or the character’s, alike.

    It used to be two different things wearing one word. A reply was rewritten
    where it sat; one of your own messages was *taken back* into the composer,
    and because sending it again would have destroyed the replies that followed
    it, doing that in the middle of a thread had to branch.

    Which made Edit a second Branch, and that is what was wrong with it. Branch
    already exists, one button along, and says what it does.

    So Edit edits. The words in the thread change and the thread stays where it
    is, replies included. That is not a side effect to be apologised for -- it
    is the feature: a conversation whose second turn now asks about the sun,
    with a reply under it that says "blue", is a conversation you can then ask
    about. What was said is what the file says was said, and the file is the
    only record there is.
    """
    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return _refresh(conversation, "Choose a message first.", "warn")
    return _refresh(conversation, "Editing this message. Save when you are done.",
                    index=index, editing=True)


def _unanswered(conversation) -> int:
    """The last message, when it is one of yours that never got a reply.

    The state a cancelled or failed reply leaves behind: your message is saved
    before the request goes out -- deliberately, so it is not lost when the
    reply is -- and ``_tidy`` then takes away the empty bubble that was going to
    hold the answer. What is left is a thread ending in a question nobody
    answered, and the only thing to do with it is ask again.

    So it goes back into the composer when the thread is opened, which is what
    was asked for: *"if the last message in a thread is a user message, it
    should just appear in the user prompt field for me to submit."*

    Not one carrying a picture. The composer's chip is a picture the browser
    uploaded and a saved one is a file beside the chat, and putting the second
    back into the first is machinery for the rarest path here -- so a message
    with an attachment stays in the thread, where **Edit** changes it in place
    and **Send again from here** re-asks it without disturbing it.
    """
    from prompt_master.chat.history import USER

    messages = getattr(conversation, "messages", None) or ()
    if not messages:
        return NO_SELECTION
    index = len(messages) - 1
    message = messages[index]
    if message.role != USER or message.attached or not message.text.strip():
        return NO_SELECTION
    return index


def _commit_edit(who, identifier, index, text, picture):
    """Save one edited message -- its words and its picture -- and go home.

    Home rather than back to the action sheet: the sheet covers the bottom of
    the transcript, and reopening it over the message just saved is the panel
    looking stuck on a thing that has finished.

    The picture is taken from the editor's own chip, so removing it there
    removes it from the message and putting a different one there replaces it.
    The one case where the chip is *not* the message's picture is a chat old
    enough to still be carrying it inline and too broken to have been moved
    onto disk when it was opened -- there is nothing to show in the chip then,
    so an empty chip must not be read as "the user took the picture away".
    """
    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return _refresh(conversation, "Choose a message first.", "warn")
    message = conversation.messages[index]
    message.text = (text or "").strip()
    shown = bool(message.image_path) or not message.image
    if shown:
        _attach_to(message, picture, who)
    _chats().save(conversation)
    return _refresh(conversation, "Edited.")


def _attach_to(message, picture, who: str) -> None:
    """Make ``message``'s attachment agree with what the editor's chip holds."""
    if picture is None:
        message.image_path, message.image, message.image_name = "", "", ""
        return
    try:
        kept = mc_llm_attachments.store(picture, who)
    except Exception:
        logger.warning("Model Chain: could not keep the edited attachment; the message keeps "
                       "the picture it had", exc_info=True)
        return
    if kept != message.image_path:
        # A different picture, so the old name is no longer about it. The
        # chip's own round trip through Gradio drops the filename it was
        # uploaded under, which is why there is a name for the case at all.
        message.image_name = ui.ATTACHED
    message.image_path, message.image = kept, ""


def _branch_here(who, identifier, index, filter_text):
    """Copy the thread up to this message, so an alternative can be explored.

    A branch is a copy, which is the vendored store's own semantics: the two
    conversations then diverge as ordinary threads with nothing shared, rather
    than as a tree every later operation would have to understand.
    """
    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return ([gr.update(), identifier or ""]
                + _refresh(conversation, "Choose a message first.", "warn"))
    branched = _chats().branch(conversation, index)
    mc_llm_state.remember(character=who or "", thread=branched.identifier)
    return ([gr.update(choices=_thread_choices(who, filter_text), value=branched.identifier),
             branched.identifier]
            + _refresh(branched,
                       "Branched — this is a new thread, and the one it came from is "
                       "untouched."))


def _delete_message(who, identifier, index):
    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return _refresh(conversation, "Choose a message first.", "warn")
    conversation.delete(index)
    _chats().save(conversation)
    return _refresh(conversation, "Message deleted.")


def _delete_from(who, identifier, index):
    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return _refresh(conversation, "Choose a message first.", "warn")
    removed = len(conversation.messages) - index
    conversation.delete_from(index)
    _chats().save(conversation)
    return _refresh(conversation,
                    f"Deleted {removed} message{'s' if removed != 1 else ''}.")


# --------------------------------------------------------------------------- #
# Replies
# --------------------------------------------------------------------------- #


def _send(who, identifier, text, picture, temperature, top_p, reply_tokens, seed):
    """Add your message to the thread, then stream the reply to it."""
    from prompt_master.chat.history import ASSISTANT, USER

    conversation = _load(who, identifier)
    if conversation is None:
        yield _idle(None, text, gr.update(), "Choose a character and a thread first.", "warn")
        return
    if not (text or "").strip() and picture is None:
        yield _idle(conversation, text, gr.update(), "Write a message first.", "warn")
        return

    kept, attachment_name = "", ""
    if picture is not None:
        if not mc_llm_runtime.config().sees:
            yield _idle(conversation, text, gr.update(),
                        "The model running has no vision projector, so the attached image "
                        "cannot be sent to it. Choose one in Setup, or remove the image.",
                        "error")
            return
        try:
            kept = mc_llm_attachments.store(picture, who)
            attachment_name = ui.picked_name(picture)
        except Exception as exc:
            yield _idle(conversation, text, gr.update(), ui.failure(exc), "error")
            return

    conversation.append(USER, (text or "").strip(), image_name=attachment_name,
                        image_path=kept)
    _chats().save(conversation)
    conversation.append(ASSISTANT, "")
    yield from _stream(who, conversation, len(conversation.messages) - 1,
                       temperature, top_p, reply_tokens, seed)


def _into_thread(events, threads, identifier):
    """``events`` from :func:`_stream`, prefixed with the thread they are in.

    Regenerate is the one streaming handler that can change which thread the
    panel is on, so it is the one whose outputs carry the thread list and the
    open thread in front of everything :func:`_stream` yields.
    """
    for event in events:
        yield (threads, identifier) + tuple(event)


def _here(identifier, event):
    """One :func:`_stream`-shaped event, in the thread that is already open."""
    return (gr.update(), identifier or "") + tuple(event)


def _regenerate(who, identifier, index, temperature, top_p, reply_tokens, seed,
                filter_text=""):
    """Write this reply again — as a version at the end, as a branch in the middle.

    Two behaviours because there are two situations, and one of them used to be
    handled by throwing away the user's conversation.

    **At the end of the thread**, where nothing follows, regenerating appends a
    version and ``◀ 2/3 ▶`` pages between the attempts. That is what makes it
    reversible: a reply that came back worse is undone rather than re-rolled
    until luck returns, and nothing is lost because there was nothing after it.

    **In the middle**, it branches. The old behaviour was
    ``truncate_after(index)`` — every message after the one being regenerated
    was *deleted*, so paging back to the first version showed that reply with
    the rest of the conversation gone for ever. Reported as exactly that: "if I
    go back to the original response, I expect the entire thread to load."

    Versions cannot answer that. A version is one string; the thing that has to
    come back is every message that followed it, and the store's own word for a
    conversation that diverges is a *branch*. So the thread up to the message
    before this reply is copied into a new one, the new reply is written there,
    and the thread it came from keeps every word of what followed — reachable
    in the thread list, whole.

    The two rules together are also what makes paging safe: versions now only
    ever exist where nothing follows them.
    """
    from prompt_master.chat.history import ASSISTANT

    conversation = _load(who, identifier)
    index = _last_reply(conversation, index)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        yield _here(identifier, _idle(conversation, "", None,
                                      "There is no reply to regenerate.", "warn"))
        return
    message = conversation.messages[index]
    if message.role != ASSISTANT:
        yield _here(identifier, _idle(conversation, "", None,
                                      "Regenerate applies to a reply. For one of your own "
                                      "messages, use Send again from here.", "warn"))
        return

    if index < len(conversation.messages) - 1:
        # Branch at the message *before* this reply, so the branch ends on the
        # turn the reply was answering and the new one is written to it. An
        # opening reply has no turn before it, and ``branch(-1)`` is the empty
        # copy that says so -- still a branch, so the thread it came from is
        # still whole, which is the whole point.
        branched = _chats().branch(conversation, index - 1)
        mc_llm_state.remember(character=who or "", thread=branched.identifier)
        branched.append(ASSISTANT, "")
        yield from _into_thread(
            _stream(who, branched, len(branched.messages) - 1,
                    temperature, top_p, reply_tokens, seed),
            gr.update(choices=_thread_choices(who, filter_text),
                      value=branched.identifier),
            branched.identifier)
        return

    conversation.truncate_after(index)
    message.add_version("")
    yield from _into_thread(
        _stream(who, conversation, index, temperature, top_p, reply_tokens, seed),
        gr.update(), identifier)


def _continue(who, identifier, index, temperature, top_p, reply_tokens, seed):
    """Carry the reply on from where it stopped."""
    from prompt_master.chat.history import ASSISTANT
    from prompt_master.chat.prompt import continue_instruction

    conversation = _load(who, identifier)
    index = _last_reply(conversation, index)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        yield _idle(conversation, "", None, "There is no reply to continue.", "warn")
        return
    message = conversation.messages[index]
    if message.role != ASSISTANT or not message.text.strip():
        yield _idle(conversation, "", None, "There is nothing to carry on from.", "warn")
        return

    try:
        character = _characters().load(who)
    except Exception as exc:
        yield _idle(conversation, "", None, ui.failure(exc), "error")
        return
    # ``upto`` includes the reply itself: the model cannot carry on from text
    # it was not shown.
    yield from _stream(who, conversation, index, temperature, top_p, reply_tokens, seed,
                       instruction=continue_instruction(character), upto=index + 1,
                       opening=message.text)


def _resend(who, identifier, index, temperature, top_p, reply_tokens, seed):
    """Answer this message of yours again, dropping everything after it."""
    from prompt_master.chat.history import ASSISTANT, USER

    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        yield _idle(conversation, "", None, "Choose one of your messages first.", "warn")
        return
    if conversation.messages[index].role != USER:
        yield _idle(conversation, "", None,
                    "Send again from here applies to one of your own messages. For a reply, "
                    "use Regenerate.", "warn")
        return

    conversation.truncate_after(index)
    _chats().save(conversation)
    conversation.append(ASSISTANT, "")
    yield from _stream(who, conversation, len(conversation.messages) - 1,
                       temperature, top_p, reply_tokens, seed)


def _last_reply(conversation, index) -> int:
    """``index``, or the last reply in the thread when nothing is selected.

    So Regenerate and Continue still mean what they mean with no message
    nominated -- "again" and "more" are about the end of the thread unless
    somebody has said otherwise.
    """
    from prompt_master.chat.history import ASSISTANT

    try:
        index = int(index)
    except (TypeError, ValueError):
        index = NO_SELECTION
    if index >= 0 or conversation is None:
        return index
    return conversation.last_index(ASSISTANT)


# The composer's primary action, in its two states. Visibility *and*
# interactivity: only one of the two is ever on screen, and the one that is not
# is also disabled, so a keyboard shortcut aimed at a hidden Stop finds a
# control that refuses rather than one that quietly cancels nothing.
SENT = gr.update(value=None, visible=False)
"""What the composer's picture chip becomes once the message carrying it has gone.

Emptied *and* taken out of the layout, which is one update rather than two
because they are one fact: a chip with nothing in it is a gap beside the
composer, and a chip still holding the picture that has just been sent is a
picture the next message would carry again without anybody asking for it.
"""

BUSY = (gr.update(visible=False, interactive=False), gr.update(visible=True, interactive=True))
IDLE = (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False))

LLM_RUNNING = "llm"
LLM_IDLE = "idle"
"""What the hidden run-state component holds, and why it exists.

Section 25. Send/Stop visibility used to be a property of the last Gradio
update, which is correct exactly while the language model is the only thing
that can be busy. Streaming speech breaks that: the reply finishes, this
generator yields IDLE, Gradio puts Send back -- and the speaker is still
talking with nothing on screen to stop it. So Python publishes *its* half of
the answer as a value rather than as a visibility, ``javascript/voice_chat.js``
holds the other half, and one function in the browser combines them. Neither
side sets visibility the other could contradict.
"""


_completed: dict = {}
_completed_lock = threading.Lock()
"""What the run that just finished produced, if it produced anything.

The one authoritative answer to "did this run leave a completed new assistant
reply, and what was its text". Written by :func:`_stream` and nowhere else, and
read exactly once by :func:`take_completed_reply`.

It is a *snapshot*, and that is the whole point of it (R2-5). By the time
anything reads this, the reader may have edited that message, regenerated it,
branched away, or opened another thread -- so the answer to "what did this run
say" cannot be a message index, a thread id, or anything else that is re-resolved
later. It is the string, copied at the moment the run reached its completed
branch.

Cleared at the start of every run, so a run that fails, is Stopped, or raises
cannot inherit the previous run's answer -- which is one of the two mechanisms
that make automatic speech success-only. The other is that this dictionary is
only ever *filled in* by the branch that reached a whole reply.
"""


def _begin_run() -> None:
    with _completed_lock:
        _completed.clear()


def _completed_reply(text: str) -> None:
    """Record a reply that finished. Only ever called on the completed path."""
    with _completed_lock:
        _completed["text"] = str(text or "")


def take_completed_reply() -> str:
    """The completed reply, consumed. Empty when the last run produced none.

    Consumed rather than read, so one run can produce at most one of whatever
    is downstream of it -- a duplicate terminal callback, which a host is
    entitled to deliver, gets nothing the second time.
    """
    with _completed_lock:
        return _completed.pop("text", "")


def _with_pictures(messages):
    """Read the attached stills back for the request about to be built.

    The conversation on disk holds paths; the vendored prompt builder wants the
    embedded bytes, and knows nothing about a folder. Read here, at the one
    moment they are needed, and never written back -- a message that has an
    ``image_path`` never saves an inline copy beside it.

    Only the newest few, because that is all a request can carry: the builder
    keeps at most ``MAX_IMAGES`` of the stills that survive trimming, and
    decoding forty photographs to send four would be forty reads a message.
    """
    from prompt_master.chat.prompt import MAX_IMAGES

    allowance = MAX_IMAGES
    for message in reversed(messages or ()):
        if not message.image_path or message.image:
            continue
        if allowance <= 0:
            break
        message.image = mc_llm_attachments.data_url(message.image_path)
        allowance -= 1
    return messages


def _idle(conversation, text, attachment, note: str, kind: str = "info") -> tuple:
    """One yield that changes nothing and says why.

    ``attachment`` is what the picture slot should become: ``None`` to empty it,
    and ``gr.update()`` to leave whatever is in it alone. The second is what a
    refusal wants -- the user is about to fix the message and press Send again,
    and taking their picture away while telling them why the message could not
    be sent would make them attach it twice. It is also the cheap answer: the
    slot holds a decoded image now, and handing one back is a full re-encode.
    """
    rows, positions = _view(conversation)
    return (None, rows, positions, text, attachment, ui.notice(note, kind)) + IDLE \
        + ("", LLM_IDLE)


def _stream(who, conversation, index, temperature, top_p, reply_tokens, seed,
            instruction=None, upto=None, opening: str = ""):
    """Stream one reply into ``conversation.messages[index]``, and save it.

    ``opening`` is what is already in that message -- the text a continuation
    extends -- and ``upto`` bounds the history the model is asked from, so an
    ordinary reply is not written from the empty message being written into
    while a continuation is written from the reply it continues.
    """
    from prompt_master.chat.characters import (DEFAULT_MAX_REPLY_TOKENS, DEFAULT_TEMPERATURE,
                                               DEFAULT_TOP_P)
    from prompt_master.chat.prompt import build, clean_reply, needs_vision
    from prompt_master.core.models import RANDOM_SEED, draw_seed

    store = _chats()
    # Whatever the previous run left behind stops being true here. See
    # :data:`_completed`.
    _begin_run()

    try:
        character = _characters().load(who)
    except Exception as exc:
        yield _idle(conversation, "", None, ui.failure(exc), "error")
        return
    persona = _persona()
    history = conversation.messages[:upto if upto is not None else index]

    # Every control below falls back to the character's own value, and the
    # character's own value falls back to the vendored default. A cleared
    # number box therefore runs at the settings the standalone application
    # would have used, rather than at a literal typed into this file.
    tokens = _number(reply_tokens, character.max_reply_tokens, DEFAULT_MAX_REPLY_TOKENS)
    resolved = _number(character.seed, RANDOM_SEED)
    if resolved == RANDOM_SEED:
        resolved = draw_seed()
    asked = _number(seed, RANDOM_SEED)

    # Built once and then asked about, rather than asked about the history and
    # built afterwards. The two answers differ exactly when trimming has dropped
    # the only still, and the request that is actually sent is the one whose
    # needs decide whether a projector has to be loaded for it.
    wire = build(character, persona, _with_pictures(history),
                 context_size=_context_size(), reply_tokens=tokens, instruction=instruction)
    request = sessions.ChatRequest(
        messages=wire,
        needs_vision=needs_vision(wire),
        temperature=_decimal(temperature, character.temperature, DEFAULT_TEMPERATURE),
        top_p=_decimal(top_p, character.top_p, DEFAULT_TOP_P),
        max_tokens=tokens,
        seed=asked if asked != RANDOM_SEED else resolved,
    )

    message = conversation.messages[index]
    cancel = sessions.Cancellation()
    streamed = opening
    # A continuation is joined to what is already there, and the model is not
    # reliable about starting with the space that needs.
    join_space = bool(opening) and not opening[-1].isspace()

    # The speech turn for this run, created before the first chunk arrives so
    # that the browser has its id in the very first yield and can open the audio
    # stream while the model is still thinking. ``opening`` is passed because a
    # continuation must speak only the newly generated tail: the existing
    # opening is already on screen and may already have been read aloud, and
    # re-speaking it is the surprising behaviour section 7 rules out.
    #
    # Everything about this is failure-tolerant on purpose. A voice turn that
    # cannot be created is a run that streams text exactly as it always did --
    # invariant: Voice never takes Conversation down with it.
    turn = mc_voice_ui.begin_speech(character=character, persona=persona, opening=opening)
    turn_token = turn.id if turn is not None else ""
    busy = BUSY + (turn_token, LLM_RUNNING)
    # The token stays in the field when the run ends rather than being cleared:
    # the browser reads it by polling, and a short reply whose terminal yield
    # follows the first one immediately would otherwise blank it before anybody
    # looked. It is superseded by the next run's token, and a token already
    # spoken is ignored.
    idle = IDLE + (turn_token, LLM_IDLE)

    kept = False

    def keep() -> None:
        """Put what the message holds on to disk. Idempotent, and never raises.

        Every exit from the loop below goes through this, including the one
        that cannot yield anything afterwards, which is why it is a function
        rather than three copies of the same three lines.
        """
        nonlocal kept
        try:
            _tidy(conversation, index)
            conversation.retitle()
            store.save(conversation)
            kept = True
        except Exception:
            logger.warning("Model Chain: could not save the conversation", exc_info=True)

    rows, positions = _view(conversation)
    yield cancel, rows, positions, "", SENT, ui.working("Starting…"), *busy

    try:
        for event in sessions.conversation(request, cancel):
            if event.kind == sessions.CHUNK:
                if join_space and event.text and not event.text[0].isspace():
                    streamed += " "
                join_space = False
                streamed += event.text
                message.text = streamed
                # Non-blocking, always: this is inside the generator that draws
                # the reply, and section 6 is explicit that it must never call
                # Kokoro. The most this does is run a segmenter over a few
                # hundred characters and put the result on a queue.
                if turn is not None:
                    turn.add_text(event.text)
                rows, positions = _view(conversation)
                yield cancel, rows, positions, "", SENT, gr.update(), *busy
            elif event.kind == sessions.STATUS:
                rows, positions = _view(conversation)
                yield cancel, rows, positions, "", SENT, ui.working(event.text), *busy
            elif event.kind in (sessions.DONE, sessions.CANCELLED):
                whole = event.text if event.kind == sessions.DONE and not opening else streamed
                message.text = clean_reply(whole or streamed, character, persona)
                keep()
                if event.kind == sessions.DONE:
                    # The one line that makes a reply eligible to be spoken, in
                    # the one branch that means the reply is whole. Stopped goes
                    # to the same place on screen and deliberately not to here.
                    _completed_reply(message.text)
                    # The authoritative text, which is what finally decides what
                    # is spoken: the panel's own ``clean_reply`` has just run,
                    # and the tail nobody has heard yet is flushed against *that*
                    # rather than against the concatenated chunks.
                    if turn is not None:
                        turn.complete(_spoken_tail(message.text, opening))
                elif turn is not None:
                    # Stopped. Nothing further is spoken -- what has already been
                    # heard cannot be unheard, and section 4 says so plainly.
                    turn.cancel("stopped")
                note = "Stopped." if event.kind == sessions.CANCELLED else "Reply complete."
                rows, positions = _view(conversation)
                yield (cancel, rows, positions, "", None,
                       ui.notice(note, "warn" if event.kind == sessions.CANCELLED else "info"),
                       *idle)
                return
            elif event.kind == sessions.FAILED:
                # The half-written reply goes; your turn stays, so the message
                # you typed is not lost to a server that would not start.
                keep()
                if turn is not None:
                    turn.cancel("failed")
                rows, positions = _view(conversation)
                yield cancel, rows, positions, "", SENT, ui.notice(event.text, "error"), *idle
                return
    except Exception as exc:
        keep()
        if turn is not None:
            turn.cancel("failed")
        rows, positions = _view(conversation)
        yield cancel, rows, positions, "", SENT, ui.notice(ui.failure(exc), "error"), *idle
        return
    finally:
        # The reply you were reading has to survive whatever ended this
        # generator, and one of the things that ends it cannot be caught above.
        #
        # Stop is wired as ``cancels=``, and what that does is *close* the
        # generator where it stands -- which raises GeneratorExit, a
        # BaseException, straight past the ``except Exception``. So the branch
        # that saves never ran: the reply stayed on screen, because Gradio
        # keeps the rows it was last given, and was never written to the
        # thread. It came back missing the next time the thread was opened,
        # and the transcript showed a message of yours with nothing under it.
        # The same hole swallowed a browser refresh and a dropped queue entry.
        #
        # A ``finally`` runs on all of them. Saving is not yielding, so it is
        # safe here: yielding during a GeneratorExit would be a RuntimeError,
        # writing a file is not. What is saved is what the message holds --
        # a partial reply is a real reply, which is what the CANCELLED branch
        # already decided.
        if not kept:
            keep()
        # GeneratorExit lands here and nowhere else -- Stop is wired as
        # ``cancels=``, which closes this generator rather than raising inside
        # it. A turn left running after that would keep speaking a reply the
        # user has already stopped, so the cancel goes in the one block that
        # runs on every way out. Idempotent, so the branches above that already
        # cancelled cost nothing.
        if turn is not None and turn.busy and not turn.source_done:
            turn.cancel("interrupted")

    # The event stream ended without a terminal event, which is the other way
    # a reply finishes whole: everything that arrived, arrived.
    _completed_reply(message.text)
    if turn is not None:
        turn.complete(_spoken_tail(message.text, opening))
    rows, positions = _view(conversation)
    yield cancel, rows, positions, "", SENT, ui.notice("Reply complete."), *idle


def _spoken_tail(whole: str, opening: str) -> str:
    """The part of a finished reply Voice is responsible for.

    For an ordinary reply that is all of it. For a continuation it is what was
    newly generated: the opening is already on screen and may already have been
    read aloud, and re-speaking it is the surprising behaviour section 7 rules
    out. Checked rather than sliced blindly, because ``clean_reply`` may have
    taken whitespace off the front -- and a slice that was one character out
    would cut a word in half at the join.
    """
    if opening and whole.startswith(opening):
        return whole[len(opening):]
    if opening and whole.strip().startswith(opening.strip()):
        return whole.strip()[len(opening.strip()):]
    return whole


def _tidy(conversation, index: int) -> None:
    """Take away a reply that never arrived, and nothing else.

    "Never arrived" is the message being empty. A continuation that came back
    with nothing still has the reply it was continuing in it and is left
    exactly where it was -- there is a difference between a generation that
    produced no new text and one that produced no reply, and only the second is
    something to clear up after.

    A message that has other versions loses only the empty one, because the
    versions are what a regenerate was risking, and losing them all is not what
    "that attempt failed" should cost.
    """
    if not (0 <= index < len(conversation.messages)):
        return
    message = conversation.messages[index]
    if message.text.strip():
        return
    if len(message.versions) > 1:
        message.drop_version()
    elif index == len(conversation.messages) - 1:
        conversation.delete(index)


def _number(value, *fallbacks) -> int:
    """``value`` as a whole number, or the first fallback that is one.

    Exists because a Gradio number box that has been cleared hands back
    ``None`` and a slider that has never been touched can hand back whatever
    the browser last had -- and the answer to either is the value the character
    was saved with, never a crash and never a literal invented here.
    """
    for candidate in (value,) + fallbacks:
        try:
            return int(float(candidate))
        except (TypeError, ValueError):
            continue
    return 0


def _decimal(value, *fallbacks) -> float:
    for candidate in (value,) + fallbacks:
        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue
    return 0.0


def _cancel(cancel):
    """Stop the run, and put the controls back.

    The buttons are the point. ``cancels=`` closes the generator where it
    stands, which is what makes a stop immediate -- and a generator that is
    closed never reaches the yield that would have put Send back in the
    composer. So the run stopped, the partial reply stayed, and the panel was
    left permanently busy with no way to ask for anything else. Whatever
    restores those controls has to be *this* handler, because it is the only
    one that still runs.
    """
    if cancel is not None:
        cancel.cancel()
    # The other half of one Stop. The browser has already silenced the speaker
    # by the time this arrives -- it is on the same button, in the capture phase
    # -- and this is what guarantees the *backend* stops: Kokoro stops being
    # asked for samples, and the turn cannot produce any more audio even if the
    # browser's own request never reached us. Section 26 wants both, and
    # cancelling twice is defined behaviour, so neither has to know about the
    # other.
    mc_voice_ui.cancel_speech("stop")
    return (ui.notice("Stopped.", "warn"),) + IDLE + ("", LLM_IDLE)


def _context_size() -> int:
    """The context the runtime is actually placed at, not the one requested.

    Trimming a conversation against a number llama.cpp was not started with is
    how a thread ends up truncated by the server instead of by the fitter --
    and the fitter is the one that knows which turns matter.

    Three answers, in order of how much they know. What llama.cpp *reported*
    beats what it was asked for, because a build with its own memory fitter
    adjusts the context downwards and then answers with 6,912 tokens against a
    conversation trimmed to fit 7,168. What it was asked for beats the setting,
    for the same reason one step earlier.
    """
    report = mc_llm_runtime.runtime.report
    granted = report.offload.granted_context if report.offload else 0
    if granted:
        return int(granted)
    if report.placement is not None:
        return int(report.placement.context)
    return int(mc_llm_runtime.config().context_size)


# --------------------------------------------------------------------------- #
# Characters and persona
# --------------------------------------------------------------------------- #


# Creating a character and editing one are the same screen and *not* the same
# operation, and the difference is one field: which character on disk the
# editor is bound to. That is ``editing`` -- the empty string while a new one
# is being written, and a name once there is a file behind it.
#
# It has to be its own State because the obvious alternative is what shipped
# and what lost somebody's work: Save read the "Talking to" drop-down for the
# name to write over. So pressing New, typing a name and pressing Save renamed
# the character that happened to be selected -- moving its file, taking its
# picture with it, and leaving the user with a list that no longer had the
# character they started from. The drop-down says who you are talking to. It
# has never said what the editor is editing, and it does not decide it now.

NOT_EDITING = ""
"""``editing`` while the editor holds a character that is not on disk yet."""


def _character_named(who):
    """The character being talked to, or ``None``. Never raises.

    Passed to :func:`mc_voice_ui.speech_marker` so that a completed reply is
    remembered against the voice its character asked for. Total, because it is
    read on a success handler attached to every reply-producing run: a character
    file that has gone missing is a reply spoken in the default voice, not a
    reply that fails after it has been written.
    """
    name = str(who or "").strip()
    if not name:
        return None
    try:
        return _characters().load(name)
    except Exception:
        logger.debug("Model Chain: could not read the character %r", name, exc_info=True)
        return None


def _blank_character():
    """A character with the vendored package's own defaults and no name.

    Named rather than copied, for the reason the panel already gives about the
    Advanced accordion: a second set of literals in the UI is how a panel
    quietly stops matching the engine behind it. The seed among them is
    :data:`RANDOM_SEED`, so a character nobody gives a seed to draws a fresh
    one for every reply.
    """
    from prompt_master.chat.characters import Character

    return Character(name="")


def _editor_fields(character, editing: str, note: str, kind: str = "info") -> list:
    """One state of the editor, in the order the outputs are wired.

    Every handler that touches the editor returns this, so none of them can
    leave the name box, the sampling and the character being written to
    describing three different characters.
    """
    blank = _blank_character()
    return [gr.update(visible=True), editing,
            character.name, character.context, character.greeting, character.system or "",
            _decimal(character.temperature, blank.temperature),
            _decimal(character.top_p, blank.top_p),
            _number(character.max_reply_tokens, blank.max_reply_tokens),
            _number(character.seed, blank.seed),
            _system_preview(character.name, character.context, character.system),
            ui.notice(note, kind),
            # The picture this character has, if it has one. Read off disk with
            # everything else so the box cannot end up showing the face of the
            # character that *was* open in the editor.
            _face_value(_character_face(character.name))] + _voice_fields(character)


def _voice_fields(character) -> list:
    """The voice controls for ``character``, in the order they are wired.

    The hidden id, the "its own delivery" checkbox, that group's visibility, and
    then the four sliders. Visibility is in the list rather than left to the
    checkbox's own handler because loading a character is a server-side write:
    Gradio does not fire ``change`` for a value Python put there, so a character
    with its own delivery would open with the box ticked and the sliders hidden
    under it.
    """
    found = mc_voice_ui.character_state(character)
    return ([found["voice"], found["custom"], gr.update(visible=found["custom"])]
            + list(found["values"]))


def _blank_voice_fields() -> list:
    """The same list, all of it left alone. One per control, counted here."""
    return [gr.update()] * (3 + len(mc_voice_ui.delivery_controls()))


def _closed_editor(note: str, kind: str = "info") -> list:
    """The editor shut, with everything in it left exactly as it was."""
    return ([gr.update(visible=False), gr.update()] + [gr.update()] * 9
            + [ui.notice(note, kind), gr.update()] + _blank_voice_fields())


def _system_preview(name, context, system) -> str:
    """The system message this character would actually be given, as it stands.

    Composed rather than described. A character's system prompt is *built* --
    the wrapper, the character's own context folded into it, and the persona's
    name and description if there is one -- and the only honest way to answer
    "what is it currently?" is to build the thing and show it. It updates as
    the boxes above it are typed into, so what is on screen is the prompt for
    the character being written rather than for the one on disk.

    Read-only, and the override box under it is where an edit goes. Those are
    two different objects on purpose: this one follows the persona and the name
    for ever, and an override is frozen text that stops following anything. A
    panel that let somebody type into the preview would be converting every
    character they looked at into the second kind.
    """
    try:
        from prompt_master.chat import prompt as chat_prompt
        from prompt_master.chat.characters import Character

        character = Character(name=(name or "").strip(), context=context or "",
                              system=system or "")
        return chat_prompt.system_text(character, _persona())
    except Exception:
        logger.debug("Model Chain: could not compose the system prompt", exc_info=True)
        return ""


def _adopt_system_prompt(name, context, system):
    """Put the prompt in force into the override box, to be edited from there.

    The route from "let me see it" to "let me change it", and one press because
    the alternative is selecting eight lines of read-only text and pasting them
    into the box underneath. What lands there is exactly what was running a
    moment ago, so saving without touching it changes nothing about how the
    character behaves -- it only stops the wrapper following the persona.
    """
    if (system or "").strip():
        return gr.update(), ui.notice(
            "This character already has a system prompt of its own — the box below is it, "
            "and the preview is showing what it produces.", "warn")
    composed = _system_preview(name, context, system)
    if not composed:
        return gr.update(), ui.notice("There is no system prompt to copy yet.", "warn")
    return composed, ui.notice(
        "Copied into the box below. It is the character's own now: editing the Context or "
        "the persona will no longer change it. Clear the box to go back to the built one.")


def _open_character(who):
    """Open the editor on the character being talked to."""
    if not who:
        return _closed_editor("Choose a character to edit, or press New.", "warn")
    try:
        loaded = _characters().load(who)
    except Exception as exc:
        return _closed_editor(ui.failure(exc), "error")
    return _editor_fields(loaded, loaded.name,
                          f"Editing {loaded.name}. Save writes it back; changing the name "
                          f"renames it.")


def _new_character():
    """Clear the editor for a character that does not exist yet.

    The sampling is reset with the rest of it. Leaving it alone would have a
    new character silently inherit the settings of whichever one was selected
    when New was pressed -- which is how a character nobody gave a seed to ends
    up with somebody else's.
    """
    return _editor_fields(_blank_character(), NOT_EDITING,
                          "Fill in a name and press Save character to create it. It will not "
                          "touch the character you are talking to.")


def _cancel_character(who):
    """Shut the editor and put the sampling back to the selected character's.

    Cancel has to undo what New did to the boxes outside the editor, or a user
    who thought better of creating a character is left talking to their old one
    at a new one's settings.
    """
    try:
        loaded = _characters().load(who) if who else None
    except Exception:
        loaded = None
    if loaded is None:
        return _closed_editor("Ready.")
    fields = _editor_fields(loaded, NOT_EDITING, "Ready.")
    fields[0] = gr.update(visible=False)
    return fields


def _save_character(editing, name, context, greeting, system, temperature, top_p,
                    reply_tokens, seed, face=None, voice="", voice_custom=False,
                    *voice_values):
    """Write the editor to disk, creating or renaming as ``editing`` says.

    ``editing`` and not the drop-down, which is the whole fix. Creating refuses
    a name that is already taken rather than writing over it -- overwriting a
    character silently is the same loss by a shorter road.
    """
    from prompt_master.chat.characters import Character

    wanted = (name or "").strip()
    if not wanted:
        return [gr.update(), gr.update(), ui.notice("A character needs a name.", "warn")]

    store = _characters()
    creating = not (editing or "").strip()
    if creating and store.exists(wanted):
        return [gr.update(), gr.update(),
                ui.notice(f"There is already a character called {wanted}. Give this one "
                          f"another name, or press Edit to change that one.", "warn")]

    blank = _blank_character()
    # Four values or four ``None``s, and the difference is the checkbox: a
    # character with no delivery of its own follows Settings → Voice Chat, and
    # keeps following it when that changes. See mc_voice_ui.character_profile.
    delivery = mc_voice_ui.character_profile(voice_custom, voice_values)
    character = Character(
        name=wanted, context=context or "", greeting=greeting or "",
        temperature=_decimal(temperature, blank.temperature),
        top_p=_decimal(top_p, blank.top_p),
        max_reply_tokens=_number(reply_tokens, blank.max_reply_tokens),
        seed=_number(seed, blank.seed), system=system or "",
        voice=str(voice or "").strip(),
        voice_speed=delivery.get("speed"), voice_pitch=delivery.get("pitch"),
        voice_gain=delivery.get("gain"), voice_pause=delivery.get("pause"))
    try:
        store.save(character, previous_name=None if creating else editing)
    except Exception as exc:
        return [gr.update(), gr.update(), ui.notice(ui.failure(exc), "error")]

    # After the save, so a rename has already moved the old picture onto the new
    # stem and this writes over the right file. ``face`` is None when the box
    # was emptied, which is how a character's picture is taken away.
    try:
        store.set_avatar(character.name, face)
    except Exception:
        logger.warning("Model Chain: %s was saved but its picture could not be",
                       character.name, exc_info=True)

    # ``editing`` becomes the saved name, so pressing Save a second time edits
    # what was just created rather than trying to create it again.
    return [gr.update(choices=_character_choices(), value=character.name), character.name,
            ui.notice(f"{'Created' if creating else 'Saved'} {character.name}.")]


def _delete_character(who):
    """Remove a character, and land on whichever one is left."""
    if not who:
        return [gr.update(), gr.update(visible=False), NOT_EDITING,
                ui.notice("Choose a character first.", "warn")]
    try:
        _characters().delete(who)
    except Exception as exc:
        return [gr.update(), gr.update(), gr.update(), ui.notice(ui.failure(exc), "error")]
    left = _character_choices()
    return [gr.update(choices=left, value=left[0] if left else None),
            gr.update(visible=False), NOT_EDITING,
            ui.notice(f"Deleted {who}." if left else
                      f"Deleted {who}. There are no characters left — press New to write one.")]


def _refresh_characters(who):
    """Re-read the characters folder without changing anything in it.

    For the case the panel cannot see on its own: a ``.yaml`` copied in from an
    oobabooga install while the tab was open, or a file deleted by hand.
    """
    found = _character_choices()
    keep = who if who in found else (found[0] if found else None)
    return [gr.update(choices=found, value=keep),
            ui.notice(f"{len(found)} character{'' if len(found) == 1 else 's'}."
                      if found else "No characters yet — press New to write one.")]


def _import_character(upload):
    if not upload:
        return [gr.update(), gr.update(visible=False), NOT_EDITING,
                ui.notice("Choose a card to import.", "warn")]
    try:
        imported = _characters().import_file(Path(getattr(upload, "name", upload)))
    except Exception as exc:
        return [gr.update(), gr.update(), gr.update(), ui.notice(ui.failure(exc), "error")]
    return [gr.update(choices=_character_choices(), value=imported.name),
            gr.update(visible=False), NOT_EDITING,
            ui.notice(f"Imported {imported.name}.")]


def _save_persona(name, description, face=None):
    """Your name, what you are like, and your face. ``face`` of None removes it."""
    from prompt_master.chat.characters import Persona, save_persona, set_persona_avatar

    paths = mc_llm_paths.app_paths()
    try:
        save_persona(paths, Persona(name=(name or "").strip(),
                                    description=description or ""))
    except Exception as exc:
        return ui.notice(ui.failure(exc), "error")
    try:
        set_persona_avatar(paths, face)
    except Exception:
        logger.warning("Model Chain: your persona was saved but your picture could not be",
                       exc_info=True)
        return ui.notice("Saved, but your picture could not be written.", "warn")
    return ui.notice("Saved.")
