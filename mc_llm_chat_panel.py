"""Conversation: threads, characters and streamed replies (section 4.3).

Designed around reading. Section 4.3 asks for the conversation area to receive
most of the visual space and for configuration to be secondary and collapsible,
and this panel takes that further than the first version did: there is one
column, it is the transcript, and everything else -- threads, the character,
the persona -- lives in a single drawer that is *not in the layout at all*
until it is asked for. A rail that is always there is not secondary, it is
narrow; a drawer that is not rendered is what actually gives the transcript the
window.

The state lives in the vendored ``prompt_master.chat`` package, unchanged:
characters are files in a folder other tools already understand, chats are
documents filed under the character they belong to, and both are read and
written by ``CharacterStore`` and ``ChatStore`` rather than by anything here.
That is section 16's separation kept by construction -- Conversation's history
cannot leak into Prompt Studio's, because they are not in the same files and
never pass through the same code.

Per-message actions, and why they are a bar rather than a menu
--------------------------------------------------------------
The standalone application hangs a ``⋯`` on every bubble. Gradio 4.40's
``Chatbot`` has nowhere to put one -- it renders the value it is given and
nothing else -- and that was originally recorded as "no per-message
affordances", with branching reduced to "copy the whole thread and delete back".
That is the wrong trade: the actions are the feature, and the component only
decides where they are drawn.

So the component's own ``select`` event is what nominates a message -- click a
bubble and it says which -- and the actions are drawn once, in a bar under the
transcript, applying to whichever message is nominated. Everything the
standalone menu offers is there: edit, regenerate, continue, send again from
here, branch from here, delete, delete from here, and the version pager a
regenerate leaves behind. One bar rather than sixty menus is also what keeps
this cheap: the transcript is re-rendered from the conversation on every
change, which is the standalone application's own rule and the only way a dozen
in-place mutations cannot drift from what is on disk.

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
"""What ``selected`` holds when the action bar applies to nothing."""


def build() -> dict:
    """Assemble the panel. Returns the handles the shell needs."""
    from prompt_master.chat.characters import (DEFAULT_MAX_REPLY_TOKENS, DEFAULT_TEMPERATURE,
                                               DEFAULT_TOP_P)
    from prompt_master.core.models import RANDOM_SEED

    prefs = mc_llm_state.preferences()
    who = prefs.get("character") or ""
    # Populated at build rather than left for the first character change: the
    # panel opens on the thread it was left on, and a rail that starts empty
    # reads as "you have no threads" rather than as "pick a character".
    initial_threads = _thread_choices(who)
    initial_thread = prefs.get("thread") or (initial_threads[0][1] if initial_threads else "")
    initial_persona = _persona()
    initial_rows, initial_map = _view(_load(who, initial_thread))

    cancellation = gr.State(None)
    thread_state = gr.State(initial_thread)
    # Which message the action bar applies to, and how to get from the
    # component's (row, column) to that index. Both are State rather than
    # recomputed, because a transcript that has been edited since the click is
    # a different transcript and the click has to be read against the one it
    # was made on.
    selected = gr.State(NO_SELECTION)
    positions = gr.State(initial_map)
    # Whether the drawer and the image box are open. Held here rather than read
    # back off the components because a Column and an Image have no value for a
    # handler to be given -- only a State can be an input to the click that
    # flips it.
    drawer_open = gr.State(False)
    attachment_open = gr.State(False)

    with gr.Row(elem_id=ui.ident("chat"),
                elem_classes=ui.classes("workspace", "chat-workspace")):

        # -- the drawer: everything that is not the conversation ----------- #
        #
        # visible=False and not merely narrow. Gradio removes a hidden column
        # from the layout entirely, which is what lets the transcript have the
        # whole window rather than the whole window minus two rails.
        with gr.Column(scale=1, min_width=260, visible=False,
                       elem_id=ui.ident("chat", "drawer"),
                       elem_classes=ui.classes("drawer")) as drawer:

            with gr.Accordion("Threads", open=True):
                search = gr.Textbox(label="Find a thread", placeholder="Filter by title…",
                                    elem_id=ui.ident("chat", "search"))
                threads = gr.Radio(label=None, show_label=False, choices=initial_threads,
                                   value=initial_thread or None,
                                   elem_id=ui.ident("chat", "threads"),
                                   elem_classes=ui.classes("threads"))
                with gr.Row():
                    new_thread = gr.Button("New", size="sm", variant="primary")
                    rename = gr.Button("Rename", size="sm")
                    delete = gr.Button("Delete", size="sm", variant="stop")
                rename_box = gr.Textbox(label="New title", visible=False,
                                        elem_id=ui.ident("chat", "rename"))

            # Choosing, editing and creating a character are three things done
            # to the same object, so they are one section: the drop-down is
            # who you are talking to, and the editor under it is that same
            # character, opened only when it is being changed.
            with gr.Accordion("Character", open=False):
                character = gr.Dropdown(
                    label="Talking to", choices=_character_choices(),
                    value=who or None, interactive=True,
                    elem_id=ui.ident("chat", "character"))
                with gr.Row():
                    edit_character = gr.Button("Edit", size="sm")
                    new_character = gr.Button("New", size="sm")
                    delete_character = gr.Button("Delete", size="sm", variant="stop")
                with gr.Group(visible=False,
                              elem_id=ui.ident("chat", "character-editor")) as character_editor:
                    name = gr.Textbox(label="Name", elem_id=ui.ident("chat", "name"))
                    context = gr.Textbox(label="Context", lines=8,
                                         placeholder="Who the character is. Shown to the model "
                                                     "before the first line of dialogue.")
                    greeting = gr.Textbox(label="Greeting", lines=3)
                    system = gr.Textbox(label="System prompt override", lines=3,
                                        placeholder="Leave empty to use the built prompt.")
                    # Every default here is the vendored package's own, named
                    # rather than copied: these are the numbers the standalone
                    # application ships with, and a second set of literals in
                    # the UI is how a panel quietly stops matching the engine
                    # behind it.
                    with gr.Row():
                        temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0,
                                                step=0.05, value=DEFAULT_TEMPERATURE)
                        top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, step=0.01,
                                          value=DEFAULT_TOP_P)
                    reply_tokens = gr.Slider(label="Reply tokens", minimum=64, maximum=4096,
                                             step=64, value=DEFAULT_MAX_REPLY_TOKENS)
                    seed = gr.Number(label="Seed", value=RANDOM_SEED, precision=0,
                                     info=f"{RANDOM_SEED} draws a fresh seed for every reply.")
                    with gr.Row():
                        save_character = gr.Button("Save character", variant="primary", size="sm")
                        close_editor = gr.Button("Close", size="sm")
                import_card = gr.File(label="Import a character card",
                                      file_types=[".json", ".yaml", ".yml", ".png"],
                                      elem_id=ui.ident("chat", "import"))

            with gr.Accordion("You", open=False):
                persona_name = gr.Textbox(label="Your name", value=initial_persona.name)
                persona_description = gr.Textbox(label="About you", lines=4,
                                                 value=initial_persona.description)
                save_persona = gr.Button("Save", size="sm")

        # -- the stage: status, transcript, actions, composer -------------- #
        with gr.Column(scale=4, min_width=360,
                       elem_id=ui.ident("chat", "stage"),
                       elem_classes=ui.classes("stage", "chat-stage")):

            with gr.Row(elem_classes=ui.classes("stage-head")):
                panel = gr.Button("☰ Threads & character", size="sm",
                                  elem_id=ui.ident("chat", "panel"),
                                  elem_classes=ui.classes("drawer-toggle"))
                status = gr.HTML(ui.notice("Ready."), elem_id=ui.ident("chat", "status"))

            # No height= at all. The height is the space the tab has, worked
            # out in the browser and applied in style.css -- a number here
            # would be an inline style and would win over the CSS that makes
            # the transcript fit the window instead of the other way round.
            transcript = gr.Chatbot(
                label=None, show_copy_button=True, render_markdown=True,
                value=initial_rows,
                elem_id=ui.ident("chat", "transcript"),
                elem_classes=ui.classes("transcript"))

            actions = _action_bar()

            with gr.Row(visible=False, elem_id=ui.ident("chat", "attachment"),
                        elem_classes=ui.classes("attachment")) as attachment_row:
                attachment = gr.Image(label="Attach an image", type="filepath", height=140,
                                      elem_id=ui.ident("chat", "image"))

            with gr.Row(elem_classes=ui.classes("composer")):
                message = gr.Textbox(
                    label=None, lines=3, max_lines=12, show_label=False, scale=6,
                    placeholder="Write a message. Shift+Enter for a new line.",
                    elem_id=ui.ident("chat", "message"))
                with gr.Column(scale=1, min_width=120,
                               elem_classes=ui.classes("composer-side")):
                    send = gr.Button("Send", variant="primary",
                                     elem_id=ui.ident("chat", "send"))
                    stop = gr.Button("Stop", variant="stop", interactive=False,
                                     elem_id=ui.ident("chat", "stop"))
                    attach = gr.Button("Attach…", size="sm",
                                       elem_id=ui.ident("chat", "attach"))

    # -- wiring ----------------------------------------------------------- #
    #
    # Everything that changes what the transcript is goes through one output
    # list, so a handler cannot leave the transcript, the position map and the
    # action bar describing three different conversations.

    view = [transcript, positions, selected, status] + actions["outputs"]
    stream = [cancellation, transcript, positions, message, attachment, status, send, stop]
    sampling = [temperature, top_p, reply_tokens, seed]

    panel.click(fn=_toggle_drawer, inputs=[drawer_open],
                outputs=[drawer_open, drawer], queue=False)
    attach.click(fn=_toggle_attachment, inputs=[attachment_open],
                 outputs=[attachment_open, attachment_row, status], queue=False)

    character.change(fn=_select_character, inputs=[character, search],
                     outputs=[threads, thread_state] + view
                     + [name, context, greeting, system] + sampling, queue=False)
    search.change(fn=lambda person, text: gr.update(choices=_thread_choices(person, text)),
                  inputs=[character, search], outputs=[threads], queue=False)
    threads.change(fn=_open_thread, inputs=[character, threads],
                   outputs=[thread_state] + view, queue=False)

    new_thread.click(fn=_new_thread, inputs=[character, search],
                     outputs=[threads, thread_state] + view, queue=False)
    delete.click(fn=_delete_thread, inputs=[character, thread_state, search],
                 outputs=[threads, thread_state] + view, queue=False)
    rename.click(fn=lambda: gr.update(visible=True), outputs=[rename_box], queue=False)
    rename_box.submit(fn=_rename_thread, inputs=[character, thread_state, rename_box, search],
                      outputs=[threads, rename_box, status], queue=False)

    # -- the per-message actions ------------------------------------------ #

    transcript.select(fn=_select_message, inputs=[character, thread_state, positions],
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
                          outputs=[actions["editor"], actions["editor_box"]], queue=False)
    actions["save_edit"].click(
        fn=_commit_edit, inputs=[character, thread_state, selected, actions["editor_box"]],
        outputs=view, queue=False)
    actions["cancel_edit"].click(fn=lambda: gr.update(visible=False),
                                 outputs=[actions["editor"]], queue=False)

    actions["branch"].click(fn=_branch_here, inputs=[character, thread_state, selected, search],
                            outputs=[threads, thread_state] + view, queue=False)
    actions["delete"].click(fn=_delete_message, inputs=[character, thread_state, selected],
                            outputs=view, queue=False)
    actions["delete_from"].click(fn=_delete_from, inputs=[character, thread_state, selected],
                                 outputs=view, queue=False)

    # -- sending, and the three ways of asking again ---------------------- #

    replying = send.click(
        fn=_send, inputs=[character, thread_state, message, attachment] + sampling,
        outputs=stream, show_progress="minimal")
    regenerating = actions["regenerate"].click(
        fn=_regenerate, inputs=[character, thread_state, selected] + sampling,
        outputs=stream, show_progress="minimal")
    continuing = actions["continue"].click(
        fn=_continue, inputs=[character, thread_state, selected] + sampling,
        outputs=stream, show_progress="minimal")
    resending = actions["resend"].click(
        fn=_resend, inputs=[character, thread_state, selected] + sampling,
        outputs=stream, show_progress="minimal")

    for run in (replying, regenerating, continuing, resending):
        # The thread list is refreshed because an untitled thread has just been
        # named, and the selection is dropped because the message it pointed at
        # may not be the message that is there now.
        run.then(fn=lambda person, text: gr.update(choices=_thread_choices(person, text)),
                 inputs=[character, search], outputs=[threads])
        run.then(fn=_close_selection, inputs=[character, thread_state], outputs=view)

    stop.click(fn=_cancel, inputs=[cancellation], outputs=[status],
               cancels=[replying, regenerating, continuing, resending], queue=False)

    # -- characters and persona ------------------------------------------- #

    edit_character.click(fn=_open_character, inputs=[character],
                         outputs=[character_editor, name, context, greeting, system]
                         + sampling + [status], queue=False)
    new_character.click(fn=_new_character,
                        outputs=[character_editor, name, context, greeting, system, status],
                        queue=False)
    close_editor.click(fn=lambda: gr.update(visible=False), outputs=[character_editor],
                       queue=False)
    save_character.click(
        fn=_save_character,
        inputs=[character, name, context, greeting, system] + sampling,
        outputs=[character, status], queue=False)
    delete_character.click(fn=_delete_character, inputs=[character],
                           outputs=[character, status], queue=False)
    import_card.upload(fn=_import_character, inputs=[import_card],
                       outputs=[character, status], queue=False)

    save_persona.click(fn=_save_persona, inputs=[persona_name, persona_description],
                       outputs=[status], queue=False)

    return {"status": status, "transcript": transcript, "drawer": drawer,
            "persona": (persona_name, persona_description)}


def _action_bar() -> dict:
    """The one row of per-message actions, and the editor it can open.

    Built once and re-labelled, because Gradio cannot create a control in
    response to a click: which of these apply to the message in hand is said by
    hiding the ones that do not, and by :func:`_selection_updates`.
    """
    with gr.Group(visible=False, elem_id=ui.ident("chat", "actions"),
                  elem_classes=ui.classes("message-actions")) as bar:
        with gr.Row(elem_classes=ui.classes("message-actions-head")):
            heading = gr.HTML(elem_id=ui.ident("chat", "selection"))
            close = gr.Button("✕", size="sm", min_width=44,
                              elem_classes=ui.classes("message-actions-close"))
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
            delete = gr.Button("Delete message", size="sm", variant="stop")
            delete_from = gr.Button("Delete from here", size="sm", variant="stop")
        with gr.Group(visible=False, elem_classes=ui.classes("message-editor")) as editor:
            editor_box = gr.Textbox(label="Edit this message", lines=6, show_label=True,
                                    elem_id=ui.ident("chat", "editor"))
            with gr.Row():
                save_edit = gr.Button("Save", size="sm", variant="primary")
                cancel_edit = gr.Button("Cancel", size="sm")

    return {
        "bar": bar, "heading": heading, "close": close,
        "back": back, "pager": pager, "forward": forward, "drop": drop,
        "edit": edit, "regenerate": regenerate, "continue": carry_on, "resend": resend,
        "branch": branch, "delete": delete, "delete_from": delete_from,
        "editor": editor, "editor_box": editor_box,
        "save_edit": save_edit, "cancel_edit": cancel_edit,
        # The order here is the order :func:`_selection_updates` returns, and
        # the two are asserted against each other in tests/test_llm_panels.py:
        # a handler that returned them in a different order would silently put
        # a button's label in another button's visibility.
        "outputs": [bar, heading, back, pager, forward, drop,
                    regenerate, carry_on, resend, editor, editor_box],
    }


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
# The transcript, and the map back to the conversation
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
    """What the action bar shows for message ``index``.

    Returned in the order ``_action_bar()["outputs"]`` lists, which is the one
    thing about this function that has to be kept in step with the layout.
    """
    from prompt_master.chat.history import ASSISTANT

    hidden = gr.update(visible=False)
    if (conversation is None or index < 0 or index >= len(conversation.messages)):
        return [gr.update(visible=False), gr.update(value=""), hidden, hidden, hidden, hidden,
                hidden, hidden, hidden, hidden, gr.update(value="")]

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
        gr.update(visible=False),
        gr.update(value=message.text),
    ]


def _refresh(conversation, note: str, kind: str = "info",
             index: int = NO_SELECTION) -> list:
    """Every output the ``view`` list names, for one state of one thread.

    One function for all of them because they are one fact: the rows, the map
    a click is read against, the message the action bar applies to and what
    that bar shows all come from the same conversation, and a handler that
    returned three of the four would leave the fourth describing a thread that
    is no longer on screen.
    """
    rows, positions = _view(conversation)
    return ([rows, positions, index, ui.notice(note, kind)]
            + _selection_updates(conversation, index))


def _reopen(who, identifier, note: str, kind: str = "info",
            index: int = NO_SELECTION) -> list:
    """:func:`_refresh` for a thread read back off disk."""
    return _refresh(_load(who, identifier), note, kind, index)


# --------------------------------------------------------------------------- #
# The drawer and the attachment
# --------------------------------------------------------------------------- #


def _toggle_drawer(open_now):
    """Show or hide the drawer."""
    showing = not bool(open_now)
    return showing, gr.update(visible=showing)


def _toggle_attachment(open_now):
    """Open the image box, or close it — and say when it would be pointless.

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
        "it. Choose one in LLM Studio\u2019s Setup mode, or send the message without a "
        "picture.", "warn")


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
    return ([gr.update(choices=choices, value=identifier or None), identifier]
            + _refresh(conversation, note)
            + [loaded.name, loaded.context, loaded.greeting, loaded.system,
               loaded.temperature, loaded.top_p, loaded.max_reply_tokens, loaded.seed])


def _open_thread(who, identifier):
    conversation = _load(who, identifier)
    if conversation is None:
        return [identifier or ""] + _refresh(None, "Choose a thread.", "warn")
    mc_llm_state.remember(character=who or "", thread=identifier)
    return [identifier] + _refresh(conversation, conversation.title)


def _new_thread(who, filter_text):
    if not who:
        return ([gr.update(), ""]
                + _refresh(None, "Choose a character first.", "warn"))
    store = _chats()
    conversation = store.new(who)
    _greet(conversation, who)
    store.save(conversation)
    mc_llm_state.remember(character=who, thread=conversation.identifier)
    return ([gr.update(choices=_thread_choices(who, filter_text),
                       value=conversation.identifier), conversation.identifier]
            + _refresh(conversation, "New thread."))


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
        return gr.update(), gr.update(visible=False), ui.notice("Nothing to rename.", "warn")
    conversation.title = title.strip()
    _chats().save(conversation)
    return (gr.update(choices=_thread_choices(who, filter_text), value=identifier),
            gr.update(value="", visible=False), ui.notice("Renamed."))


# --------------------------------------------------------------------------- #
# Per-message actions
# --------------------------------------------------------------------------- #


def _select_message(who, identifier, positions, event: gr.SelectData = None):
    """A click on a bubble nominates the message the action bar applies to."""
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
    return _refresh(conversation, "Ready.", index=index)


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
    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return gr.update(visible=False), gr.update()
    return gr.update(visible=True), gr.update(value=conversation.messages[index].text)


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


def _idle(conversation, text, image_path, note: str, kind: str = "info") -> tuple:
    """One yield that changes nothing and says why."""
    rows, positions = _view(conversation)
    return (None, rows, positions, text, image_path, ui.notice(note, kind),
            gr.update(interactive=True), gr.update(interactive=False))


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

    busy = (gr.update(interactive=False), gr.update(interactive=True))
    idle = (gr.update(interactive=True), gr.update(interactive=False))
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
    rows, positions = _view(conversation)
    yield cancel, rows, positions, "", None, ui.notice("Starting…"), *busy

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
                yield cancel, rows, positions, "", None, ui.notice(event.text), *busy
            elif event.kind in (sessions.DONE, sessions.CANCELLED):
                whole = event.text if event.kind == sessions.DONE and not opening else streamed
                message.text = clean_reply(whole or streamed, character, persona)
                _tidy(conversation, index)
                conversation.retitle()
                store.save(conversation)
                note = "Stopped." if event.kind == sessions.CANCELLED else "Reply complete."
                rows, positions = _view(conversation)
                yield (cancel, rows, positions, "", None,
                       ui.notice(note, "warn" if event.kind == sessions.CANCELLED else "info"),
                       *idle)
                return
            elif event.kind == sessions.FAILED:
                # The half-written reply goes; your turn stays, so the message
                # you typed is not lost to a server that would not start.
                _tidy(conversation, index)
                store.save(conversation)
                rows, positions = _view(conversation)
                yield cancel, rows, positions, "", None, ui.notice(event.text, "error"), *idle
                return
    except Exception as exc:
        _tidy(conversation, index)
        store.save(conversation)
        rows, positions = _view(conversation)
        yield cancel, rows, positions, "", None, ui.notice(ui.failure(exc), "error"), *idle
        return

    store.save(conversation)
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
    if cancel is not None:
        cancel.cancel()
    return ui.notice("Stopping…", "warn")


def _context_size() -> int:
    """The context the runtime is actually placed at, not the one requested.

    Trimming a conversation against a number llama.cpp was not started with is
    how a thread ends up truncated by the server instead of by the fitter --
    and the fitter is the one that knows which turns matter.
    """
    report = mc_llm_runtime.runtime.report
    if report.placement is not None:
        return int(report.placement.context)
    return int(mc_llm_runtime.config().context_size)


# --------------------------------------------------------------------------- #
# Characters and persona
# --------------------------------------------------------------------------- #


def _open_character(who):
    """Open the editor on the character being talked to."""
    from prompt_master.chat.characters import Character

    if not who:
        return ([gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(),
                 gr.update(), gr.update(), gr.update(), gr.update()]
                + [ui.notice("Choose a character to edit, or press New.", "warn")])
    try:
        loaded = _characters().load(who)
    except Exception as exc:
        return ([gr.update(visible=False)] + [gr.update()] * 8
                + [ui.notice(ui.failure(exc), "error")])
    blank = Character(name="")
    return [gr.update(visible=True), loaded.name, loaded.context, loaded.greeting,
            loaded.system or "",
            _decimal(loaded.temperature, blank.temperature),
            _decimal(loaded.top_p, blank.top_p),
            _number(loaded.max_reply_tokens, blank.max_reply_tokens),
            _number(loaded.seed, blank.seed),
            ui.notice(f"Editing {loaded.name}. Save writes it back; changing the name saves it "
                      f"under the new one.")]


def _new_character():
    return (gr.update(visible=True), "", "", "", "",
            ui.notice("Fill in a name and press Save character to create it."))


def _save_character(previous, name, context, greeting, system, temperature, top_p,
                    reply_tokens, seed):
    from prompt_master.chat.characters import Character

    if not (name or "").strip():
        return gr.update(), ui.notice("A character needs a name.", "warn")
    blank = Character(name="")
    character = Character(
        name=name.strip(), context=context or "", greeting=greeting or "",
        temperature=_decimal(temperature, blank.temperature),
        top_p=_decimal(top_p, blank.top_p),
        max_reply_tokens=_number(reply_tokens, blank.max_reply_tokens),
        seed=_number(seed, blank.seed), system=system or "")
    try:
        _characters().save(character, previous_name=previous or None)
    except Exception as exc:
        return gr.update(), ui.notice(ui.failure(exc), "error")
    return (gr.update(choices=_character_choices(), value=character.name),
            ui.notice(f"Saved {character.name}."))


def _delete_character(who):
    if not who:
        return gr.update(), ui.notice("Choose a character first.", "warn")
    try:
        _characters().delete(who)
    except Exception as exc:
        return gr.update(), ui.notice(ui.failure(exc), "error")
    return gr.update(choices=_character_choices(), value=None), ui.notice(f"Deleted {who}.")


def _import_character(upload):
    if not upload:
        return gr.update(), ui.notice("Choose a card to import.", "warn")
    try:
        imported = _characters().import_file(Path(getattr(upload, "name", upload)))
    except Exception as exc:
        return gr.update(), ui.notice(ui.failure(exc), "error")
    return (gr.update(choices=_character_choices(), value=imported.name),
            ui.notice(f"Imported {imported.name}."))


def _save_persona(name, description):
    from prompt_master.chat.characters import Persona, save_persona

    try:
        save_persona(mc_llm_paths.app_paths(),
                     Persona(name=(name or "").strip(), description=description or ""))
    except Exception as exc:
        return ui.notice(ui.failure(exc), "error")
    return ui.notice("Saved.")
