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
from pathlib import Path

import gradio as gr

import mc_llm_paths
import mc_llm_runtime
import mc_llm_sessions as sessions
import mc_llm_state
import mc_llm_ui as ui

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

NO_SELECTION = -1
"""What ``selected`` holds when the action sheet applies to nothing."""

SCREENS = ("nav", "threads", "character", "persona")
"""The overlay surfaces, in the order :func:`_screens` returns them.

One at a time, always. They occupy the same space -- the whole conversation
workspace on a narrow display -- so two open at once is two half-drawn panels
rather than one usable one, and the way that is guaranteed is that there is one
function which decides, and it decides for all of them at once.
"""

SELECTION_ORDER = ("sheet", "heading", "back", "pager", "forward", "drop",
                   "regenerate", "continue", "resend", "edit", "edit_box", "composer")
"""The order :func:`_selection_updates` answers in.

The action sheet is redrawn from that one function by every handler that
changes the transcript, so the list of controls it writes into has to be
written down once. ``tests/test_llm_panels.py`` asserts the two are the same
length: a handler one value short would put a label into a visibility and
nothing would raise.

The last three are not in the sheet. Editing replaces the composer rather than
opening an editor between the transcript and the composer, so "which message is
selected" decides the state of the edit row and of the composer too -- and
because they are in this list, every refresh returns the panel to CHAT_HOME
without a second handler having to remember to.
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
    # Whether the attachment row is open. Held here rather than read back off
    # the component because an Image has no value for a handler to be given --
    # only a State can be an input to the click that flips it.
    attachment_open = gr.State(False)
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
                menu = gr.Button("☰", size="sm", scale=0, min_width=44,
                                 elem_id=ui.ident("chat", "menu"),
                                 elem_classes=ui.classes("icon-button"))
                header = gr.HTML(_heading(who, opened),
                                 elem_id=ui.ident("chat", "title"),
                                 elem_classes=ui.classes("chat-title"))
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
                elem_id=ui.ident("chat", "transcript"),
                elem_classes=ui.classes("transcript"))

            # ATTACHMENT_PREVIEW. Not in the layout until Attach opens it, and
            # small when it is: the picture for the next message is a chip, not
            # a panel.
            with gr.Row(visible=False, elem_id=ui.ident("chat", "attachment"),
                        elem_classes=ui.classes("attachment")) as attachment_row:
                attachment = gr.Image(label="Next message", type="filepath", height=96,
                                      show_label=False, scale=1,
                                      elem_id=ui.ident("chat", "image"))
                remove_attachment = gr.Button("Remove", size="sm", scale=0, min_width=88)

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
                edit_box = gr.Textbox(label="Editing message", lines=2, max_lines=8,
                                      show_label=False, container=False, scale=1,
                                      placeholder="Editing this message…",
                                      elem_id=ui.ident("chat", "editor"))
                save_edit = gr.Button("Save", variant="primary", size="sm", scale=0,
                                      min_width=80)
                cancel_edit = gr.Button("Cancel", size="sm", scale=0, min_width=80)

        # -- NAV_SHEET ------------------------------------------------------ #

        with gr.Column(visible=False, elem_id=ui.ident("chat", "nav"),
                       elem_classes=ui.classes("sheet", "sheet-side")) as nav:
            with gr.Row(elem_classes=ui.classes("sheet-head")):
                gr.Markdown("#### Menu")
                close_nav = gr.Button("✕", size="sm", scale=0, min_width=44,
                                      elem_classes=ui.classes("icon-button"))
            to_threads = gr.Button("Threads", elem_classes=ui.classes("nav-entry"))
            to_character = gr.Button("Character", elem_classes=ui.classes("nav-entry"))
            to_persona = gr.Button("You", elem_classes=ui.classes("nav-entry"))
            gr.Markdown("##### Elsewhere", elem_classes=ui.classes("sheet-label"))
            to_model = gr.Button("Model / Runtime", size="sm",
                                 elem_classes=ui.classes("nav-entry"))
            to_setup = gr.Button("Setup", size="sm", elem_classes=ui.classes("nav-entry"))
            to_modes = gr.Button("Switch mode", size="sm",
                                 elem_classes=ui.classes("nav-entry"))

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
                context = gr.Textbox(label="Context", lines=6,
                                     placeholder="Who the character is. Shown to the model "
                                                 "before the first line of dialogue.")
                greeting = gr.Textbox(label="Greeting", lines=3)
                system = gr.Textbox(label="System prompt override", lines=3,
                                    placeholder="Leave empty to use the built prompt.")
                gr.Markdown(
                    "Save writes the Advanced generation settings below with the rest of the "
                    "character, so a seed of −1 there is what makes its replies vary.",
                    elem_classes=ui.classes("hint"))
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
            persona_description = gr.Textbox(label="About you", lines=4,
                                             value=initial_persona.description)
            save_persona = gr.Button("Save", variant="primary", size="sm")

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
                 "edit": edit_row, "edit_box": edit_box, "composer": composer}
    view = ([transcript, positions, selected, status, header]
            + [selection[key] for key in SELECTION_ORDER])
    stream = [cancellation, transcript, positions, message, attachment, status, send, stop]
    sampling = [temperature, top_p, reply_tokens, seed]
    # The State first, then one visibility per surface: the order
    # :func:`_screens` answers in.
    screens = [surface, nav, threads_screen, character_screen, persona_screen]

    # -- getting about ---------------------------------------------------- #

    menu.click(fn=_toggle_nav, inputs=[surface, character, thread_state],
               outputs=screens + view, queue=False)
    close_nav.click(fn=_close_screens, outputs=screens, queue=False)
    for control in (threads_back, character_back, persona_back):
        control.click(fn=_close_screens, outputs=screens, queue=False)

    to_threads.click(fn=_open_threads, inputs=[character, search],
                     outputs=screens + [threads], queue=False)
    to_character.click(fn=_open_character_screen, inputs=[character],
                       outputs=screens + [character], queue=False)
    to_persona.click(fn=_open_persona,
                     outputs=screens + [persona_name, persona_description], queue=False)
    # The three secondary entries belong to the shell -- the model sheet, the
    # Setup workspace and the mode chooser are not Conversation's to own -- so
    # all this panel does is get out of the way, and the shell adds the second
    # handler that does the rest.
    for control in (to_model, to_setup, to_modes):
        control.click(fn=_close_screens, outputs=screens, queue=False)

    # -- threads ---------------------------------------------------------- #

    # ``input`` and not ``change``: this code refills the list whenever a
    # thread is created, deleted or branched, and ``change`` fires on the
    # refill -- which would re-open the thread the handler had just opened, and
    # send the reader home from the screen they were working in. Every handler
    # that moves the value returns the transcript with it, so nothing is lost
    # by listening only to the tap.
    _picked(threads)(fn=_open_thread_home, inputs=[character, threads],
                     outputs=[thread_state] + view + screens, queue=False)
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
    editor = ([character_editor, editing, name, context, greeting, system]
              + sampling + [status])

    character.change(fn=_select_character, inputs=[character, search],
                     outputs=[threads, thread_state] + view
                     + [name, context, greeting, system] + sampling + [editing], queue=False)
    edit_character.click(fn=_open_character, inputs=[character], outputs=editor, queue=False)
    new_character.click(fn=_new_character, outputs=editor, queue=False)
    refresh_characters.click(fn=_refresh_characters, inputs=[character],
                             outputs=[character, status], queue=False)
    close_editor.click(fn=_cancel_character, inputs=[character], outputs=editor, queue=False)
    save_character.click(
        fn=_save_character,
        inputs=[editing, name, context, greeting, system] + sampling,
        outputs=[character, editing, status], queue=False)
    delete_character.click(fn=_delete_character, inputs=[character],
                           outputs=[character, character_editor, editing, status], queue=False)
    import_card.upload(fn=_import_character, inputs=[import_card],
                       outputs=[character, character_editor, editing, status], queue=False)

    save_persona.click(fn=_save_persona, inputs=[persona_name, persona_description],
                       outputs=[status], queue=False)

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

    actions["edit"].click(fn=_open_editor, inputs=[character, thread_state, selected],
                          outputs=[edit_row, edit_box, composer, actions["sheet"]],
                          queue=False)
    save_edit.click(fn=_commit_edit, inputs=[character, thread_state, selected, edit_box],
                    outputs=view, queue=False)
    cancel_edit.click(fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                      outputs=[edit_row, composer], queue=False)

    actions["branch"].click(fn=_branch_here, inputs=[character, thread_state, selected, search],
                            outputs=[threads, thread_state] + view, queue=False)
    actions["delete"].click(fn=_delete_message, inputs=[character, thread_state, selected],
                            outputs=view, queue=False)
    actions["delete_from"].click(fn=_delete_from, inputs=[character, thread_state, selected],
                                 outputs=view, queue=False)

    # -- the attachment ---------------------------------------------------- #

    attach.click(fn=_toggle_attachment, inputs=[attachment_open],
                 outputs=[attachment_open, attachment_row, status], queue=False)
    remove_attachment.click(fn=_clear_attachment,
                            outputs=[attachment_open, attachment_row, attachment, status],
                            queue=False)

    # -- sending, and the three ways of asking again ---------------------- #

    sent = [character, thread_state, message, attachment] + sampling
    replying = send.click(fn=_send, inputs=sent, outputs=stream, show_progress="minimal")
    # The same pathway from the keyboard. Gradio fires ``submit`` on Enter for
    # a composer that is one line tall, which is what this one starts as.
    submitted = message.submit(fn=_send, inputs=sent, outputs=stream,
                               show_progress="minimal")
    regenerating = actions["regenerate"].click(
        fn=_regenerate, inputs=[character, thread_state, selected] + sampling,
        outputs=stream, show_progress="minimal")
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

    for run in (replying, submitted, regenerating, continuing, resending):
        # The thread list is refreshed because an untitled thread has just been
        # named, and the selection is dropped because the message it pointed at
        # may not be the message that is there now.
        run.then(fn=lambda person, text: gr.update(choices=_thread_choices(person, text)),
                 inputs=[character, search], outputs=[threads])
        run.then(fn=_close_selection, inputs=[character, thread_state], outputs=view)

    stop.click(fn=_cancel, inputs=[cancellation], outputs=[status, send, stop],
               cancels=[replying, submitted, regenerating, continuing, resending],
               queue=False)

    return {"status": status, "transcript": transcript, "header": header,
            "persona": (persona_name, persona_description),
            # What the shell wires: the state chip and the nav entry both open
            # the model sheet, Setup switches workspace, and Switch mode opens
            # the workspace chooser.
            "model": [model, to_model], "setup": to_setup, "modes": to_modes,
            "chip": model}


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
    if not who or not identifier:
        return None
    try:
        return _chats().load(who, identifier)
    except Exception:
        logger.debug("Model Chain: could not load thread %s", identifier, exc_info=True)
        return None


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
        body = message.text
        if message.image_name:
            body = f"*[{message.image_name}]*\n\n{body}" if body else f"*[{message.image_name}]*"
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


def _selection_updates(conversation, index: int) -> list:
    """What the action sheet shows for message ``index``.

    Returned in the order :data:`SELECTION_ORDER` lists, which is the one thing
    about this function that has to be kept in step with the layout.
    """
    from prompt_master.chat.history import ASSISTANT

    hidden = gr.update(visible=False)
    home = [gr.update(visible=False), gr.update(), gr.update(visible=True)]
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
        gr.update(visible=True),
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
        # return to CHAT_HOME, whatever the panel was doing before it.
        gr.update(visible=False),
        gr.update(value=message.text),
        gr.update(visible=True),
    ]


def _refresh(conversation, note: str, kind: str = "info",
             index: int = NO_SELECTION) -> list:
    """Every output the ``view`` list names, for one state of one thread.

    One function for all of them because they are one fact: the rows, the map a
    click is read against, the header, the message the action sheet applies to
    and what that sheet shows all come from the same conversation, and a
    handler that returned four of the five would leave the fifth describing a
    thread that is no longer on screen.
    """
    rows, positions = _view(conversation)
    return ([rows, positions, index, ui.notice(note, kind), _heading(None, conversation)]
            + _selection_updates(conversation, index))


def _reopen(who, identifier, note: str, kind: str = "info",
            index: int = NO_SELECTION) -> list:
    """:func:`_refresh` for a thread read back off disk."""
    return _refresh(_load(who, identifier), note, kind, index)


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


def _toggle_nav(open_now, who, identifier) -> list:
    """The menu button: open the menu, or put away whatever is open.

    A menu that only opens is a menu you cannot dismiss from the control you
    opened it with -- which is exactly what "the menus are not toggling open
    and close" was. The state it is toggling against is a State rather than the
    component's own visibility, because a Column has no value a handler can be
    given.

    Opening also drops the message selection: the action sheet applies to a
    message the reader can no longer see, and a sheet left open under another
    sheet is the second half of every "why is this still here?".
    """
    wanted = "" if open_now == "nav" else "nav"
    return _screens(wanted) + _close_selection(who, identifier)


def _open_threads(who, filter_text) -> list:
    return _screens("threads") + [gr.update(choices=_thread_choices(who, filter_text))]


def _open_character_screen(who) -> list:
    return _screens("character") + [gr.update(choices=_character_choices(),
                                              value=who or None)]


def _open_persona() -> list:
    person = _persona()
    return _screens("persona") + [person.name, person.description]


def _toggle_attachment(open_now):
    """Open the image row, or close it — and say when it would be pointless.

    Warned when it is opened rather than when the reply fails: whether a
    picture can be sent depends on the model running, and finding that out
    after writing a message is finding it out too late.
    """
    showing = not bool(open_now)
    if not showing:
        return False, gr.update(visible=False), ui.notice("Ready.")
    try:
        sees = mc_llm_runtime.config().sees
    except Exception:
        logger.debug("Model Chain: could not read the vision configuration", exc_info=True)
        sees = True
    if sees:
        return True, gr.update(visible=True), ui.notice(
            "The attached image goes with your next message.")
    return True, gr.update(visible=True), ui.notice(
        "The model running has no vision projector, so an attached image cannot be sent to "
        "it. Choose one in LLM Studio’s Setup mode, or send the message without a "
        "picture.", "warn")


def _clear_attachment():
    """Take the picture off the next message, and put the row away with it."""
    return False, gr.update(visible=False), None, ui.notice("Ready.")


# --------------------------------------------------------------------------- #
# Threads
# --------------------------------------------------------------------------- #


def _select_character(who, filter_text):
    """Switch character: reload its threads, open the newest, fill the editor."""
    from prompt_master.chat.characters import Character

    choices = _thread_choices(who, filter_text)
    identifier = choices[0][1] if choices else ""
    conversation = _load(who, identifier)
    try:
        loaded = _characters().load(who) if who else Character(name="")
    except Exception:
        loaded = Character(name=who or "")
    mc_llm_state.remember(character=who or "", thread=identifier)
    note = f"{len(choices)} thread{'s' if len(choices) != 1 else ''}."
    # The editor follows the selection. It is usually shut, and when it is not,
    # leaving it bound to the character that *was* selected would have the next
    # Save write this character's boxes over that character's file.
    return ([gr.update(choices=choices, value=identifier or None), identifier]
            + _refresh(conversation, note)
            + [loaded.name, loaded.context, loaded.greeting, loaded.system,
               loaded.temperature, loaded.top_p, loaded.max_reply_tokens, loaded.seed,
               loaded.name or NOT_EDITING])


def _open_thread(who, identifier):
    conversation = _load(who, identifier)
    if conversation is None:
        return [identifier or ""] + _refresh(None, "Choose a thread.", "warn")
    mc_llm_state.remember(character=who or "", thread=identifier)
    return [identifier] + _refresh(conversation, conversation.title)


def _open_thread_home(who, identifier):
    """Tapping a thread opens it and returns to the conversation."""
    return _open_thread(who, identifier) + _close_screens()


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
    """Borrow the composer's space for the selected message.

    Four answers: the edit row, its text, the composer it replaces and the
    action sheet it was opened from. The transcript is untouched and stays
    where it was, which is the whole reason an edit is not a panel of its own.
    """
    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return (gr.update(visible=False), gr.update(), gr.update(visible=True),
                gr.update(visible=False))
    return (gr.update(visible=True), gr.update(value=conversation.messages[index].text),
            gr.update(visible=False), gr.update(visible=False))


def _commit_edit(who, identifier, index, text):
    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return _refresh(conversation, "Choose a message first.", "warn")
    conversation.messages[index].text = (text or "").strip()
    _chats().save(conversation)
    return _refresh(conversation, "Edited.", index=index)


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


def _send(who, identifier, text, image_path, temperature, top_p, reply_tokens, seed):
    """Add your message to the thread, then stream the reply to it."""
    from prompt_master.chat.history import ASSISTANT, USER

    conversation = _load(who, identifier)
    if conversation is None:
        yield _idle(None, text, image_path, "Choose a character and a thread first.", "warn")
        return
    if not (text or "").strip() and not image_path:
        yield _idle(conversation, text, image_path, "Write a message first.", "warn")
        return

    attachment, attachment_name = "", ""
    if image_path:
        if not mc_llm_runtime.config().sees:
            yield _idle(conversation, text, image_path,
                        "The model running has no vision projector, so the attached image "
                        "cannot be sent to it. Choose one in Setup, or remove the image.",
                        "error")
            return
        try:
            attachment = ui.data_url(image_path) or ""
            # Path, not a string split on "/": a Windows temporary file arrives
            # with backslashes and the whole path would end up as the caption.
            attachment_name = Path(image_path).name
        except Exception as exc:
            yield _idle(conversation, text, image_path, ui.failure(exc), "error")
            return

    conversation.append(USER, (text or "").strip(), attachment, attachment_name)
    _chats().save(conversation)
    conversation.append(ASSISTANT, "")
    yield from _stream(who, conversation, len(conversation.messages) - 1,
                       temperature, top_p, reply_tokens, seed)


def _regenerate(who, identifier, index, temperature, top_p, reply_tokens, seed):
    """Write this reply again, keeping the one it had as a version.

    Paging rather than replacing, which is what makes a regenerate reversible:
    the reply that came back worse is undone with ``◀`` rather than by
    regenerating until luck returns.
    """
    from prompt_master.chat.history import ASSISTANT

    conversation = _load(who, identifier)
    index = _last_reply(conversation, index)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        yield _idle(conversation, "", None, "There is no reply to regenerate.", "warn")
        return
    message = conversation.messages[index]
    if message.role != ASSISTANT:
        yield _idle(conversation, "", None,
                    "Regenerate applies to a reply. For one of your own messages, use Send "
                    "again from here.", "warn")
        return

    conversation.truncate_after(index)
    message.add_version("")
    yield from _stream(who, conversation, index, temperature, top_p, reply_tokens, seed)


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
BUSY = (gr.update(visible=False, interactive=False), gr.update(visible=True, interactive=True))
IDLE = (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False))


def _idle(conversation, text, image_path, note: str, kind: str = "info") -> tuple:
    """One yield that changes nothing and says why."""
    rows, positions = _view(conversation)
    return (None, rows, positions, text, image_path, ui.notice(note, kind)) + IDLE


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
    from prompt_master.chat.prompt import build, clean_reply, has_image
    from prompt_master.core.models import RANDOM_SEED, draw_seed

    busy, idle = BUSY, IDLE
    store = _chats()

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

    request = sessions.ChatRequest(
        messages=build(character, persona, history, context_size=_context_size(),
                       reply_tokens=tokens, instruction=instruction),
        needs_vision=has_image(history),
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
    yield cancel, rows, positions, "", None, ui.working("Starting…"), *busy

    try:
        for event in sessions.conversation(request, cancel):
            if event.kind == sessions.CHUNK:
                if join_space and event.text and not event.text[0].isspace():
                    streamed += " "
                join_space = False
                streamed += event.text
                message.text = streamed
                rows, positions = _view(conversation)
                yield cancel, rows, positions, "", None, gr.update(), *busy
            elif event.kind == sessions.STATUS:
                rows, positions = _view(conversation)
                yield cancel, rows, positions, "", None, ui.working(event.text), *busy
            elif event.kind in (sessions.DONE, sessions.CANCELLED):
                whole = event.text if event.kind == sessions.DONE and not opening else streamed
                message.text = clean_reply(whole or streamed, character, persona)
                keep()
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
                rows, positions = _view(conversation)
                yield cancel, rows, positions, "", None, ui.notice(event.text, "error"), *idle
                return
    except Exception as exc:
        keep()
        rows, positions = _view(conversation)
        yield cancel, rows, positions, "", None, ui.notice(ui.failure(exc), "error"), *idle
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

    rows, positions = _view(conversation)
    yield cancel, rows, positions, "", None, ui.notice("Reply complete."), *idle


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
    return (ui.notice("Stopped.", "warn"),) + IDLE


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
            ui.notice(note, kind)]


def _closed_editor(note: str, kind: str = "info") -> list:
    """The editor shut, with everything in it left exactly as it was."""
    return [gr.update(visible=False), gr.update()] + [gr.update()] * 8 + [ui.notice(note, kind)]


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
                    reply_tokens, seed):
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
    character = Character(
        name=wanted, context=context or "", greeting=greeting or "",
        temperature=_decimal(temperature, blank.temperature),
        top_p=_decimal(top_p, blank.top_p),
        max_reply_tokens=_number(reply_tokens, blank.max_reply_tokens),
        seed=_number(seed, blank.seed), system=system or "")
    try:
        store.save(character, previous_name=None if creating else editing)
    except Exception as exc:
        return [gr.update(), gr.update(), ui.notice(ui.failure(exc), "error")]

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


def _save_persona(name, description):
    from prompt_master.chat.characters import Persona, save_persona

    try:
        save_persona(mc_llm_paths.app_paths(),
                     Persona(name=(name or "").strip(), description=description or ""))
    except Exception as exc:
        return ui.notice(ui.failure(exc), "error")
    return ui.notice("Saved.")
