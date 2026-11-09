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
``CONVERSATION_SHEET``
    The one pop-up the header's **Conversation** opens (2026-11-06): your
    favourite threads with any character, who you are talking to and that
    character's threads, each with a star, and folded under them Emotion,
    Start the next reply with, Generation, Voice, Character, This thread and
    You. It replaced six chips -- Threads, Character, Emotion, Start with, You
    and Voice -- each opening a menu of its own, some of them inside others:
    "It's time to clean up chat controls ... What we need is a single pop up
    that allows me to configure the CONVERSATION".
``MESSAGE_EDIT_MODE``
    The composer, temporarily replaced by an editor for the selected message.
``ATTACHMENT_PREVIEW``
    The picture for the next message, in a compact row above the composer.

Only one surface is ever open: :func:`_screens` is the single function that
says so, and every handler that opens one returns its whole answer rather than
toggling a component of its own.

Per-message actions, and why they are a row in the bubble
---------------------------------------------------------
Gradio 4.40's ``Chatbot`` renders the value it is given and nothing else, so
there is nowhere to hang a component on a bubble. The actions used to be a
sheet over the bottom of the transcript, opened by the component's ``select``
event, with a line saying which message it applied to. They are a row of icon
buttons *inside* the bubble now, shown by a tap on the message and put away by
the next, drawn by ``javascript/llm_studio.js`` the way the Forge Assistant
draws its own -- "I like how in conversation mode flyout, I can access the
action right in the message" -- for both the character's replies and your own
prompts. Everything the sheet offered is there: edit, regenerate, continue,
send again from here, branch from here, delete, delete from here and the
version pager a regenerate leaves behind; and three the sheet did not have,
Copy, Send to VibeVoice and Play, which are the browser's and the Voice Box's
and never reach this file.

What reaches this file is a **nomination**. A button in a row writes which
message it is on -- ``assistant:2``, the third reply counted down the
transcript, or ``user:0`` -- into one hidden box and presses one hidden button
per action. Each handler turns that into a message index through the
``positions`` map the transcript was drawn with (:func:`_nominated`) and calls
the same function the sheet's button called. The browser nominates; Python
decides, loads, streams and saves. There is no round trip on the tap itself,
which is why the row opens at once.

The row needs two things only Python knows: how many versions a reply has and
which is showing, and a key the Voice Box can file a render under. Both ride
inside the bubble as a hidden marker the browser reads (:func:`_meta`), so
every write of the transcript -- a refresh, a streamed token -- carries them.

**Edit** means two different things and is two different things. A reply is text
on the transcript, so editing it rewrites that text where it sits. One of your
own messages is a *prompt that was sent*, so editing it takes it back out of the
thread and into the composer, where it is an unsent message again -- and if
there were replies after it, that happens in a branch, so the thread it came
from keeps every one of them. The same rule regenerating follows, for the same
reason: nothing a reader typed or was told is thrown away to make room for a
second attempt.

``Chatbot`` pairs turns, so a row of the component is not a message index, and
a reply's ordinal down the transcript is not one either. The map between them
is built with the transcript, in one pass, and carried in a ``gr.State`` --
deriving it a second time somewhere else is how the two would come to disagree.

Multimodal attachment follows the same rule the standalone application used:
an image is offered only when the running model has a projector, and a request
that carries one is refused with a sentence rather than quietly sent blind.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

import mc_llm_attachments
import mc_llm_compact
import mc_llm_format
import mc_llm_generation
import mc_llm_overlays
import mc_llm_paths
import mc_llm_runtime
import mc_llm_sessions as sessions  # noqa: F401  -- see below
import mc_llm_state
import mc_llm_ui as ui
import mc_voice_ui

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

# ``sessions`` is imported and not called. This panel does not run a model any
# more -- a reply belongs to :mod:`mc_llm_conversation_ops` and this file
# follows it -- but the name is part of the module's surface: the tests reach
# ``mc_llm_chat_panel.sessions`` to replace ``conversation`` with a list of
# events, and doing that through the alias keeps them pointing at whichever
# module this panel actually talks to rather than at a second copy of that
# decision.

NO_SELECTION = -1
"""What ``selected`` holds when the action sheet applies to nothing."""

OPEN_SYSTEM_EDITOR = """(who) => {
    const editor = window.forgeAssistant && window.forgeAssistant.systemEditor;
    if (editor) editor.open(who || "");
    else console.warn("Model Chain: the system prompt editor is not on this page");
    return [];
}"""
"""The character screen's ⤢ System prompt, run in the browser and nowhere else.

The full-page editor is ``javascript/forge_assistant_system.js``, shared with
the Forge Assistant's ⋯ menu; this hands it the character in "Talking to". A
Gradio round trip would add nothing: the editor reads and writes through the
assistant's routes itself.
"""

SCREENS = ("conversation",)
"""The overlay surfaces, in the order :func:`_screens` returns them.

One: the Conversation pop-up. There were four -- Threads, Character, You and
Voice, each opened by a chip of its own beside Emotion's and Start with's
pickers -- until 2026-11-06, when the user asked for "a single pop up that
allows me to configure the CONVERSATION", the rest nested. A tuple still,
because the overlay registry (:mod:`mc_llm_overlays`) answers for every
surface in LLM Studio at once: opening this one puts the shell's sheets away,
and ``\u2630``, which opens the shell's own workspace chooser as it does in
every other mode, puts this one away.
"""

STREAM_ORDER = ("operation", "transcript", "positions", "composer", "attachment", "status",
                "send", "stop", "speech_turn", "run_state", "revision")
"""What a reply's follower writes, in order.

Written down for the same reason :data:`SELECTION_ORDER` is: the list is long,
every frame has to be the same length, and a frame one value short would put a
status line into a visibility with nothing raising. The two at the end are the
newest -- the id of the reply being spoken and Python's half of "is the
composer busy" -- and the last is the revision of the thread the frame draws,
so a page whose stream was cut short still holds the revision it last saw.

``operation`` is the first, and it changed meaning rather than moving: it held
a ``Cancellation`` object, and it holds the id of the operation the server is
running. Stop is a request about that id now, not the closing of a generator.
"""

SELECTION_ORDER = ("edit", "edit_box", "edit_image", "composer")
"""The order :func:`_selection_updates` answers in.

The editor is redrawn from that one function by every handler that changes
the transcript, so the list of controls it writes into has to be written down
once. ``tests/test_llm_panels.py`` asserts the two are the same length: a
handler one value short would put a label into a visibility and nothing would
raise.

Four, since the action sheet went: an edit borrows the composer's space rather
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
    prefs = mc_llm_state.preferences()
    who = prefs.get("character") or ""
    # Populated at build rather than left for the first character change: the
    # panel opens on the thread it was left on, and a list that starts empty
    # reads as "you have no threads" rather than as "pick a character".
    initial_threads = _threads(who)
    initial_thread = prefs.get("thread") or (initial_threads[0].identifier if initial_threads
                                             else "")
    initial_persona = _persona()
    opened = _load(who, initial_thread)
    initial_rows, initial_map = _view(opened)

    # What Stop is aimed at. It held a ``Cancellation`` object; it holds the id
    # of the operation the server is running, because the reply is the server's
    # now and Stop is a request rather than the closing of a generator.
    cancellation = gr.State("")
    # The revision of the thread as this page last drew it. Every mutation this
    # panel sends carries it, and the service validates it together with the
    # message index under the conversation's own lock -- which is what makes
    # "delete message 4" mean the message the person was looking at rather than
    # whatever is at index 4 by the time it arrives.
    revision_state = gr.State(_token(opened))
    thread_state = gr.State(initial_thread)
    # Which message the editor is open on, and how to get from a nomination
    # to a message index. Both are State rather than recomputed, because a
    # transcript that has been edited since the tap is a different transcript
    # and the nomination has to be read against the one it was made on.
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
                # One button for everything about the conversation (2026-11-06):
                # "make these all just one "Conversation" button". It opens the
                # one pop-up (CONVERSATION_SHEET above); the dot the stylesheet
                # puts on it while this thread has an emotion or a start set is
                # read off the heading beside it, as the two chips' dots were.
                to_conversation = gr.Button("Conversation", size="sm", scale=0, min_width=0,
                                            elem_id=ui.ident("chat", "to-conversation"),
                                            elem_classes=ui.classes("chip-button",
                                                                    "conversation-chip"))
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
                # Following a streamed reply is javascript/llm_studio.js's
                # alone: Gradio's own autoscroll counts the last 100 px as the
                # end and put a reader who had scrolled up less than that back
                # at the end with every chunk.
                autoscroll=False,
                # Math, by the Chatbot's own KaTeX: every piece the rules in
                # mc_llm_format find -- $…$, $$…$$, \(…\), \[…\] and the
                # environments -- reaches it between one of these two
                # invisible pairs (:func:`_body`), and nothing else is a
                # delimiter, so money and code never are.
                latex_delimiters=mc_llm_format.latex_delimiters(),
                elem_id=ui.ident("chat", "transcript"),
                elem_classes=ui.classes("transcript"))

            # THE ACTION ROW IN EVERY BUBBLE, as far as Python is concerned.
            #
            # A Gradio 4.40 Chatbot draws its own bubbles and there is nowhere
            # to put a component on one, so the row itself is drawn in the
            # browser by javascript/llm_studio.js. What a button does when
            # tapped is entirely here: it writes which message it is on into
            # this box -- "assistant:2", "user:0" -- and presses the hidden
            # button for its action, and the handler behind that button is the
            # one the old action sheet's button called. The browser nominates;
            # Python decides, loads, streams and saves.
            #
            # All invisible, and none of them anything to do without the
            # script. One box rather than one per action, because a press is
            # one nomination and one action, and the box's value rides in the
            # same request as the press.
            action_at = gr.Textbox(value="", visible=False, container=False,
                                   elem_id=ui.ident("chat", "action-at"))
            nominated = _nominated_buttons()
            # The Emotion section's rows, the same way: the browser writes
            # which line and what to do with it ({"do": "choose"|"delete",
            # "line": ...}) and presses the button; Python reads it against the
            # list as it is now (:func:`_emotion_pressed`).
            emotion_at = gr.Textbox(value="", visible=False, container=False,
                                    elem_id=ui.ident("chat", "emotion-at"))
            emotion_now = gr.Button("Emotion", visible=False,
                                    elem_id=ui.ident("chat", "emotion-now"))
            # Favourites' and Threads' rows, the same way again: which thread,
            # whose, and whether to open it or star it ({"do": "open"|"star",
            # "character": ..., "thread": ...}), read against what is on disk
            # now (:func:`_thread_pressed`).
            thread_at = gr.Textbox(value="", visible=False, container=False,
                                   elem_id=ui.ident("chat", "thread-at"))
            thread_now = gr.Button("Open or star a thread", visible=False,
                                   elem_id=ui.ident("chat", "thread-now"))

            # THE FORGE ASSISTANT'S PLUMBING, as far as Gradio is concerned.
            #
            # The assistant is a DOM shell appended to document.body, because
            # Forge exposes no callback that mounts a Gradio component outside
            # the tab tree. So the three things it *does* need from Gradio live
            # here, inside the tab, and that asymmetry is accepted rather than
            # worked around.
            #
            # The key is the capability for its HTTP routes, put in the page by
            # Python because that is the one channel a cross-site page cannot
            # read -- the same mechanism, and the same reasoning, as the Voice
            # Chat key beside it.
            # Built and not held: the browser finds it by its id and Python
            # never reads it back, so a reference here would be a reference
            # nothing uses.
            gr.Textbox(value=_conversation_key(), visible=False, container=False,
                       elem_id=ui.ident("chat", "conversation-key"))
            # The refresh bridge. When another window writes to the thread this
            # tab is showing, the page store presses this, and the handler
            # re-reads the conversation and redraws the rows. It is READ-ONLY
            # and deliberately does not go through ``_open_thread``: that lifts
            # an unanswered last message out of the thread and deletes it,
            # which is fine as something somebody asked for and catastrophic as
            # something an event triggers.
            refresh_at = gr.Textbox(value="", visible=False, container=False,
                                    elem_id=ui.ident("chat", "refresh-at"))
            refresh_now = gr.Button("Refresh this conversation", visible=False,
                                    elem_id=ui.ident("chat", "refresh-now"))
            # The open bridge. A thread started or chosen in the Forge
            # Assistant's panel is this tab's thread too -- "a new thread
            # should make the flyout and conversation mode start from scratch
            # because they are in sync" -- and the tab hears nothing on its
            # own, so the panel writes the thread's id here and presses this.
            # The handler opens that thread the way choosing it in Threads
            # does and remembers it, so the next page and the panel's own looks
            # agree with it. Read-only like the refresh bridge: it never lifts
            # a message out of a thread, which is ``_open_thread``'s business
            # and not something a press from elsewhere may do.
            open_at = gr.Textbox(value="", visible=False, container=False,
                                 elem_id=ui.ident("chat", "open-at"))
            open_now = gr.Button("Open the thread the assistant chose", visible=False,
                                 elem_id=ui.ident("chat", "open-now"))
            # The paste bridge. A picture pasted into this composer is staged
            # over the assistant's own upload route by the browser, which hands
            # back a token; this puts the staged picture into the chip so the
            # tab shows what the flyout shows. A token rather than a synthesised
            # file drop, because a drop is a guess about what the installed
            # Gradio does with a DataTransfer and a token is an answer.
            paste_token = gr.Textbox(value="", visible=False, container=False,
                                     elem_id=ui.ident("chat", "paste-token"))
            paste_now = gr.Button("Attach the pasted picture", visible=False,
                                  elem_id=ui.ident("chat", "paste-now"))
            # The presentation settings, as custom properties. A stylesheet
            # variable rather than a class per value: changing the column or the
            # bubble width reloads nothing, resets no scroll and interrupts no
            # reply, because all that changes is one number.
            gr.HTML(_presentation_style(), elem_id=ui.ident("chat", "presentation"))

            # One line, always the same height, immediately above the composer.
            # "Ready." is quiet; a warning or an error is not, and neither of
            # them moves the transcript by a pixel when it arrives.
            status = gr.HTML(ui.notice("Ready."), elem_id=ui.ident("chat", "status"),
                             elem_classes=ui.classes("chat-status"))
            # How full the thread's context is (mc_llm_window), drawn here and
            # shown by javascript/llm_studio.js at the right end of the status
            # line, which it asks for again whenever that line changes -- after a
            # reply, a trim, a compaction, a change of thread. A bridge rather
            # than one more output of every handler that writes the line.
            context_meter = gr.HTML(_context_meter(who, initial_thread), visible=False,
                                    elem_id=ui.ident("chat", "context"))
            context_now = gr.Button("Read how full the context is", visible=False,
                                    elem_id=ui.ident("chat", "context-now"))

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
                # Hidden, and interactive anyway. Those are two different
                # questions and building Stop with both answers set to "no" was
                # the bug: the browser reveals Stop for a Voice-only phase, but
                # a Gradio button built ``interactive=False`` is rendered
                # ``disabled``, so what appeared was a Stop that could not be
                # pressed while the speaker was still talking. Visibility is
                # shared between Python and the browser; enablement is not
                # something either of them has a reason to withdraw.
                stop = gr.Button("Stop", variant="stop", size="sm", scale=0, min_width=88,
                                 visible=False, interactive=True,
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
                # Named for the browser: javascript/llm_studio.js opens the
                # message editor's dialog over this row and presses these two
                # with the dialog's answer. Without the script the row is the
                # editor, as it always was.
                save_edit = gr.Button("Save", variant="primary", size="sm", scale=0,
                                      min_width=80, elem_id=ui.ident("chat", "edit-save"))
                cancel_edit = gr.Button("Cancel", size="sm", scale=0, min_width=80,
                                        elem_id=ui.ident("chat", "edit-cancel"))

        # -- CONVERSATION_SHEET --------------------------------------------- #
        #
        # Everything about the conversation, in one pop-up (2026-11-06): "make
        # these all just one "Conversation" button ... The most important top
        # level menu should still be there, but everything else can be nested
        # ... top level is just select from existing or +new ... make it easy
        # to favorite an existing thread i have with one of my characters, so
        # i can quickly load and continue". On top: Favourites, the starred
        # threads of every character; who you are talking to, with + New; and
        # that character's threads, each with its star. Under them, folded:
        # what shapes this thread's replies (Emotion, Start the next reply
        # with, Generation, Voice), then the character, this thread and you.

        with gr.Column(visible=False, elem_id=ui.ident("chat", "conversation"),
                       elem_classes=ui.classes("sheet", "sheet-screen",
                                               "conversation-sheet")) as conversation_screen:
            with gr.Row(elem_classes=ui.classes("sheet-head", "conversation-head")):
                conversation_back = gr.Button("‹ Back", size="sm", scale=0, min_width=76,
                                              elem_classes=ui.classes("sheet-back"))
                gr.Markdown("#### Conversation")
                # On top, unfolded: starting a thread is the one thing about
                # threads that needs no list (2026-11-06: "top level is just
                # select from existing or +new").
                new_thread = gr.Button("+ New thread", variant="primary", size="sm", scale=0,
                                       min_width=0, elem_id=ui.ident("chat", "new-thread"))
            # Every starred thread, whoever it is with, most recently used
            # first. Folded, and folded again whenever the pop-up opens
            # (2026-11-09: "the favorite threads needs to be collapsed on open
            # ... i should have to open up a menu to see those thread"); its
            # label counts them. Rows the browser presses (thread_at above) and
            # Python draws (_favourite_rows).
            starred = len(_favourites_of_everyone())
            with gr.Accordion(_favourites_label(starred), open=False, visible=bool(starred),
                              elem_id=ui.ident("chat", "favourites-section"),
                              elem_classes=ui.classes("section",
                                                      "threads-section")) as favourites_section:
                favourites = gr.HTML(_favourite_rows(who, initial_thread),
                                     elem_id=ui.ident("chat", "favourites"),
                                     elem_classes=ui.classes("favourites-block"))
            with gr.Row(elem_classes=ui.classes("talking-to")):
                character = gr.Dropdown(
                    label="Talking to", choices=_character_choices(),
                    value=who or None, interactive=True, scale=1, min_width=0,
                    elem_id=ui.ident("chat", "who"))
                new_character = gr.Button("+ New", size="sm", scale=0, min_width=72,
                                          elem_id=ui.ident("chat", "new-character"))
            # Choosing, editing and creating a character are three things done
            # to the same object, so they are in one place: the drop-down is
            # who you are talking to, and the editor under it is that same
            # character, opened only when it is being changed -- by + New
            # beside the drop-down, or by Edit in the Character section below
            # (javascript/llm_studio.js scrolls it into view either way).
            #
            # Which character the editor is *bound to* is this State and never
            # the drop-down above. They are the same name while an existing
            # character is being edited and deliberately different while a new
            # one is being written, and reading the drop-down for it is what
            # made New behave as Rename. See the note above ``_open_character``.
            editing = gr.State(NOT_EDITING)
            # The editor, in the order somebody writes a character: who they
            # are, what they say first, what they are told, how they sound.
            # Labels and no prose: the people on this screen know what a
            # system prompt is, and a paragraph under every box was what made
            # it a page to scroll rather than a form to fill.
            with gr.Group(visible=False,
                          elem_id=ui.ident("chat", "character-editor")) as character_editor:
                with gr.Row(elem_classes=ui.classes("identity")):
                    # Beside the character file as ``<name>.png``, which is
                    # where oobabooga keeps one and where importing a card
                    # already writes it -- so a character imported with a face
                    # already has one and this box is only for the ones that
                    # did not.
                    character_face = gr.Image(label="Picture", type="pil", sources=["upload"],
                                              height=96, width=96, show_download_button=False,
                                              scale=0, min_width=112,
                                              elem_id=ui.ident("chat", "character-face"),
                                              elem_classes=ui.classes("face"))
                    with gr.Column(scale=1, min_width=160):
                        name = gr.Textbox(label="Name", elem_id=ui.ident("chat", "name"))
                        greeting = gr.Textbox(label="Greeting", lines=2)
                context = gr.Textbox(label="Context", lines=6,
                                     placeholder="Who the character is.")
                gr.Markdown("##### System prompt", elem_classes=ui.classes("sheet-label"))
                # What the model is actually told, built from the boxes above
                # and from the persona, and updated as they are typed into.
                system_preview = gr.Textbox(
                    label="In force", lines=4, interactive=False,
                    elem_id=ui.ident("chat", "system-preview"))
                edit_system = gr.Button("Copy into override", size="sm")
                # Named, because the full-page editor writes this box when it
                # saves the character this editor is open on -- otherwise the
                # next Save here would put the old prompt back.
                system = gr.Textbox(label="System prompt override", lines=4,
                                    elem_id=ui.ident("chat", "system"),
                                    placeholder="Empty: built from the Context and your persona.")
                # Which voice reads this character aloud, and how. Built by the
                # voice module so the compact list here and the full one in
                # Settings cannot drift apart -- see mc_voice_ui.character_panel.
                character_voice = mc_voice_ui.character_panel()
                with gr.Row():
                    save_character = gr.Button("Save character", variant="primary", size="sm")
                    close_editor = gr.Button("Cancel", size="sm")
            # The character's threads, folded as Favourites are and for the same
            # word, folded again whenever the pop-up opens.
            with gr.Accordion("Threads", open=False,
                              elem_id=ui.ident("chat", "threads-section"),
                              elem_classes=ui.classes("section",
                                                      "threads-section")) as threads_section:
                # One line: Gradio draws a box of one line at most as an <input>,
                # with no resize grip in its corner.
                search = gr.Textbox(label="Find a thread", placeholder="Find a thread…",
                                    show_label=False, container=False, lines=1, max_lines=1,
                                    elem_id=ui.ident("chat", "search"))
                # A thread is opened by a tap on it and kept at the top by its
                # star -- "I need a star or a pin or something to keep it at top"
                # -- which is the 2026-11-06 morning's pin (ChatStore.PINS), a tap
                # away on every row rather than a button for the open thread.
                threads = gr.HTML(_thread_rows(who, "", initial_thread),
                                  elem_id=ui.ident("chat", "threads"),
                                  elem_classes=ui.classes("threads"))

            # What shapes this thread's replies. Each section's label says what
            # is in force (_emotion_label, _start_label), drawn again as the
            # pop-up opens and as the thread changes.
            with gr.Accordion(_emotion_label(who, initial_thread), open=False,
                              elem_id=ui.ident("chat", "emotion"),
                              elem_classes=ui.classes("section")) as emotion_section:
                emotion_list = gr.HTML(_emotion_rows(who, getattr(opened, "identifier", "")))
                with gr.Row(elem_classes=ui.classes("emotion-add")):
                    emotion_new = gr.Textbox(label=None, show_label=False, container=False,
                                             lines=1, max_lines=1, scale=1,
                                             placeholder="*I am so nervous*",
                                             elem_id=ui.ident("chat", "emotion-new"))
                    emotion_add = gr.Button("Add", size="sm", scale=0, min_width=64)
            # The words the next reply starts with (mc_llm_start_with), for one
            # reply: Set keeps them, the reply that starts from them spends them.
            with gr.Accordion(_start_label(who, initial_thread), open=False,
                              elem_id=ui.ident("chat", "start"),
                              elem_classes=ui.classes("section")) as start_section:
                start_text = gr.Textbox(value=_start_of(who, getattr(opened, "identifier", "")),
                                        label=None, show_label=False, container=False,
                                        lines=1, max_lines=4, placeholder="The color is blue",
                                        elem_id=ui.ident("chat", "start-text"))
                with gr.Row(elem_classes=ui.classes("start-actions")):
                    start_clear = gr.Button("Clear", size="sm", scale=0, min_width=72,
                                            elem_id=ui.ident("chat", "start-clear"))
                    start_set = gr.Button("Set", size="sm", scale=0, min_width=72,
                                          variant="primary",
                                          elem_id=ui.ident("chat", "start-set"))
            # The thread's context window (mc_llm_window, mc_llm_compact,
            # 2026-11-09: "the ability to see how full my context window is ...
            # adjust the context window before my next reply ... keep the 5,000
            # latest ... a compacting option as an alternative"). Its label says
            # how full it is; Trim keeps the latest messages and leaves the older
            # ones out, Compact has the model summarize them into a hidden
            # summary sent in their place, Clear puts it back. Between whole
            # messages, never into the last exchange.
            initial_reading = _context_reading(who, initial_thread)
            with gr.Accordion(_context_label(initial_reading), open=False,
                              elem_id=ui.ident("chat", "context-section"),
                              elem_classes=ui.classes("section")) as context_section:
                context_info = gr.HTML(_context_info(initial_reading),
                                       elem_id=ui.ident("chat", "context-info"))
                context_keep = gr.Slider(**_keep_slider(initial_reading),
                                         label="Keep the latest (tokens)",
                                         elem_id=ui.ident("chat", "context-keep"))
                context_length = gr.Slider(
                    minimum=mc_llm_compact.SUMMARY_MIN, maximum=mc_llm_compact.SUMMARY_MAX,
                    step=50, value=mc_llm_compact.SUMMARY_DEFAULT,
                    label="Summary length (tokens)", info="For Compact",
                    elem_id=ui.ident("chat", "context-length"))
                with gr.Row(elem_classes=ui.classes("context-actions")):
                    context_trim = gr.Button("Trim", size="sm", scale=0, min_width=72,
                                             elem_id=ui.ident("chat", "context-trim"))
                    context_compact = gr.Button("Compact", size="sm", scale=0, min_width=88,
                                                variant="primary",
                                                elem_id=ui.ident("chat", "context-compact"))
                    context_clear = gr.Button("Clear", size="sm", scale=0, min_width=72,
                                              elem_id=ui.ident("chat", "context-clear"))
                    context_cancel = gr.Button("Cancel compacting", size="sm", scale=0,
                                               min_width=72, visible=False,
                                               elem_id=ui.ident("chat", "context-cancel"))
                context_summary = gr.Textbox(**_summary_box(initial_reading),
                                             label="Summary sent in place of the older "
                                                   "messages", interactive=False, lines=4,
                                             max_lines=12,
                                             elem_id=ui.ident("chat", "context-summary"))
            # How this thread's replies are sampled, kept with the thread
            # (mc_llm_generation, 2026-11-09: "These should be kept per
            # conversation ... It should be default until I override for a
            # thread"). Drawn from what the thread chose -- and, for what it did
            # not, from the default -- as the pop-up opens and as the thread
            # changes under it, and written the moment a control moves. Its
            # label says what the thread chose.
            initial_generation = _generation_values(who, initial_thread)
            with gr.Accordion(_generation_label(who, initial_thread), open=False,
                              elem_id=ui.ident("chat", "generation"),
                              elem_classes=ui.classes("section")) as generation_section:
                generation = {}
                for group, heading in mc_llm_generation.GROUPS:
                    gr.Markdown(f"##### {heading}", elem_classes=ui.classes("sheet-label"))
                    members = [setting for setting in mc_llm_generation.SETTINGS
                               if setting.group == group]
                    for start in range(0, len(members), 2):
                        with gr.Row(elem_classes=ui.classes("generation-row")):
                            for setting in members[start:start + 2]:
                                generation[setting.key] = _generation_control(
                                    setting, initial_generation[setting.key])
                generation_reset = gr.Button("Reset to defaults", size="sm",
                                             elem_id=ui.ident("chat", "generation-reset"))
            # Voice: what is ready, and its two switches (mc_voice_ui.section).
            voice = mc_voice_ui.section()

            # Then what is changed less often: the character, this thread, you.
            with gr.Accordion("Character", open=False, elem_id=ui.ident("chat", "character"),
                              elem_classes=ui.classes("section")):
                with gr.Row():
                    edit_character = gr.Button("Edit", size="sm",
                                               elem_id=ui.ident("chat", "edit-character"))
                    refresh_characters = gr.Button("↻ Refresh", size="sm")
                    # The character's system prompt on a page of its own: the
                    # editor the Forge Assistant's ⋯ menu opens, over the whole
                    # window. Browser-only -- it reads and writes through the
                    # assistant's routes -- so the press never reaches the queue.
                    system_open = gr.Button("⤢ System prompt", size="sm",
                                            elem_id=ui.ident("chat", "system-open"))
                with gr.Accordion("Import a character card", open=False,
                                  elem_classes=ui.classes("advanced")):
                    import_card = gr.File(label="Card: .yaml, .json or a .png with its JSON inside",
                                          file_types=[".json", ".yaml", ".yml", ".png"],
                                          elem_id=ui.ident("chat", "import"))
                with gr.Group(elem_classes=ui.classes("destructive")):
                    gr.Markdown("##### Danger zone", elem_classes=ui.classes("sheet-label"))
                    delete_character = gr.Button("Delete this character", size="sm",
                                                 variant="stop")
            with gr.Accordion("This thread", open=False, elem_id=ui.ident("chat", "thread"),
                              elem_classes=ui.classes("section")):
                rename = gr.Button("Rename this thread", size="sm")
                rename_box = gr.Textbox(label="New title", visible=False, container=False,
                                        placeholder="New title…",
                                        elem_id=ui.ident("chat", "rename"))
                with gr.Row(visible=False) as rename_row:
                    save_rename = gr.Button("Save", variant="primary", size="sm")
                    cancel_rename = gr.Button("Cancel", size="sm")
                # Away from the primary controls and behind its own heading,
                # which is the only confirmation Gradio can honestly offer:
                # there is no dialog to put in front of it, so the distance is
                # the deliberation.
                with gr.Group(elem_classes=ui.classes("destructive")):
                    gr.Markdown("##### Danger zone", elem_classes=ui.classes("sheet-label"))
                    delete = gr.Button("Delete this thread", size="sm", variant="stop")
            with gr.Accordion("You", open=False, elem_id=ui.ident("chat", "persona"),
                              elem_classes=ui.classes("section")):
                persona_name = gr.Textbox(label="Your name", value=initial_persona.name)
                persona_face = gr.Image(label="Your picture", type="pil", sources=["upload"],
                                        height=96, width=96, show_download_button=False,
                                        value=_face_value(_persona_face()),
                                        elem_id=ui.ident("chat", "persona-face"),
                                        elem_classes=ui.classes("face"))
                persona_description = gr.Textbox(label="About you", lines=4,
                                                 value=initial_persona.description)
                save_persona = gr.Button("Save", variant="primary", size="sm")

        # Two hidden boxes rather than a component with a value anybody can
        # read: one carries this process's page token into the page, and the
        # other carries an opaque one-shot handle to a reply that has finished.
        # Neither ever holds a transcript or a reply.
        voice_plumbing = mc_voice_ui.plumbing()

    # -- wiring ----------------------------------------------------------- #
    #
    # Everything that changes what the transcript is goes through one output
    # list, so a handler cannot leave the transcript, the position map, the
    # header and the editor describing four different conversations.

    selection = {"edit": edit_row, "edit_box": edit_box, "edit_image": edit_image,
                 "composer": composer}
    view = ([transcript, positions, selected, status, header]
            + [selection[key] for key in SELECTION_ORDER] + [revision_state])
    # Two more outputs than V1, and both are hidden values rather than
    # visibilities: the opaque id of the reply being spoken, and Python's half
    # of "is the composer busy". Appended at the end so every existing yield
    # keeps its meaning -- see :data:`LLM_RUNNING`.
    streamed = {"operation": cancellation, "transcript": transcript, "positions": positions,
                "composer": message, "attachment": attachment, "status": status,
                "send": send, "stop": stop, "speech_turn": voice_plumbing["turn"],
                "run_state": voice_plumbing["run_state"], "revision": revision_state}
    stream = [streamed[name] for name in STREAM_ORDER]
    # The Generation section's label and controls, in mc_llm_generation.KEYS
    # order: what ``_generation_view`` answers.
    generation_drawn = ([generation_section]
                        + [generation[key] for key in mc_llm_generation.KEYS])
    # The Context section and the meter: what ``_context_view`` answers.
    context_drawn = [context_section, context_info, context_keep, context_summary,
                     context_cancel, context_meter]
    # The State first, then one visibility per surface: the order
    # :func:`_screens` answers in. The tail is the surfaces this panel does not
    # own and closes anyway -- see :mod:`mc_llm_overlays`. It is built from the
    # registry rather than listed, because what is in it depends on what the
    # shell managed to build.
    mc_llm_overlays.register("conversation", conversation_screen)
    mc_llm_overlays.register_state(OVERLAY_OWNER, surface)
    screens = _screen_outputs(surface)

    # -- getting about ---------------------------------------------------- #

    # The workspace chooser is the shell's, so all this panel does is get its
    # own surfaces out of the way; the shell adds the second handler that opens
    # it. Two handlers on one button, which is what makes one control mean the
    # same thing in Conversation as it does in every other mode.
    menu.click(fn=_close_screens, outputs=screens, queue=False)
    conversation_back.click(fn=_close_screens, outputs=screens, queue=False)

    # The one button opens the one pop-up, and each part of it is drawn again
    # from what is stored by a handler of its own on the same press -- the
    # lists and who you are talking to, the sections' labels, Voice's switches,
    # you -- so what it shows is what is in force, whatever the other window or
    # Settings changed meanwhile.
    emotions_drawn = [emotion_list, emotion_section, header]
    starts_drawn = [start_section, start_text, header]
    voice_drawn = [voice[name] for name in mc_voice_ui.SECTION_VALUES]
    persona_drawn = [persona_name, persona_description, persona_face]
    to_conversation.click(fn=_open_conversation, inputs=[character, thread_state, search],
                          outputs=screens + [threads, favourites, character], queue=False)
    # Favourites and Threads folded again on every open: their lists are opened
    # to be chosen from (2026-11-09).
    to_conversation.click(fn=_thread_lists_folded,
                          outputs=[favourites_section, threads_section], queue=False)
    to_conversation.click(fn=_generation_view, inputs=[character, thread_state],
                          outputs=generation_drawn, queue=False)
    to_conversation.click(fn=_context_view, inputs=[character, thread_state],
                          outputs=context_drawn, queue=False)
    to_conversation.click(fn=_emotion_view, inputs=[character, thread_state],
                          outputs=emotions_drawn, queue=False)
    to_conversation.click(fn=_start_view, inputs=[character, thread_state],
                          outputs=starts_drawn, queue=False)
    to_conversation.click(fn=mc_voice_ui.section_values, outputs=voice_drawn, queue=False)
    to_conversation.click(fn=_persona_view, outputs=persona_drawn, queue=False)

    # -- the thread's emotion ---------------------------------------------- #
    #
    # The rows, the section's label and the bar's heading (whose mark fills
    # the Conversation chip's dot) are drawn again together, so they never
    # disagree about which line is on.
    #
    # A change of emotion is a new end to the thread's system prompt, and the
    # conversation is read again under it in the background at once
    # (mc_llm_cache_update): the status line says so quietly until it is done,
    # as it does after a run.
    chosen = emotion_now.click(fn=_emotion_pressed, inputs=[character, thread_state, emotion_at],
                               outputs=emotions_drawn + [status], queue=False)
    chosen.then(fn=_cache_status, inputs=[character, thread_state], outputs=[status],
                show_progress="hidden")
    for event in (emotion_add.click, emotion_new.submit):
        added = event(fn=_emotion_added, inputs=[character, thread_state, emotion_new],
                      outputs=emotions_drawn + [status, emotion_new], queue=False)
        added.then(fn=_cache_status, inputs=[character, thread_state], outputs=[status],
                   show_progress="hidden")
    # -- the start of the next reply --------------------------------------- #
    #
    # The box, the section's label and the bar's heading together; the reply
    # that starts from it is drawn with the heading again by every run's
    # settle, which empties the dot, and the section is drawn again from what
    # is kept as the pop-up next opens.
    for event in (start_set.click, start_text.submit):
        event(fn=_start_written, inputs=[character, thread_state, start_text],
              outputs=starts_drawn + [status], queue=False)
    start_clear.click(fn=lambda who, identifier: _start_written(who, identifier, ""),
                      inputs=[character, thread_state], outputs=starts_drawn + [status],
                      queue=False)

    # -- the thread's generation settings ----------------------------------- #
    #
    # Each control keeps what it says for the open thread the moment it moves
    # (``input``: a person moving it, never the server drawing it) and again as
    # it is let go (``release``, a slider's), so the value kept is the last one
    # whichever request lands last. Reset forgets them all.
    for key in mc_llm_generation.KEYS:
        control = generation[key]
        keep = _generation_chosen(key)
        control.input(fn=keep, inputs=[character, thread_state, control],
                      outputs=[generation_section, status], queue=False)
        if hasattr(control, "release"):
            control.release(fn=keep, inputs=[character, thread_state, control],
                            outputs=[generation_section, status], queue=False)
    generation_reset.click(fn=_generation_reset, inputs=[character, thread_state],
                           outputs=generation_drawn + [status], queue=False)

    # -- the thread's context window ----------------------------------------- #
    #
    # The meter is asked for again by the page whenever the status line changes
    # (javascript/llm_studio.js); the section's actions draw it themselves.
    # Compact follows the summary being written until it is done, on the
    # status line, and draws the section again after it.
    context_now.click(fn=_context_meter, inputs=[character, thread_state],
                      outputs=[context_meter], queue=False, show_progress="hidden")
    context_trim.click(fn=_context_trim, inputs=[character, thread_state, context_keep],
                       outputs=context_drawn + [status], queue=False)
    context_clear.click(fn=_context_clear, inputs=[character, thread_state, context_keep],
                        outputs=context_drawn + [status], queue=False)
    context_cancel.click(fn=_context_cancel, inputs=[character, thread_state, context_keep],
                         outputs=context_drawn + [status], queue=False)
    compacted = context_compact.click(
        fn=_context_compact, inputs=[character, thread_state, context_keep, context_length],
        outputs=context_drawn + [status], show_progress="hidden")
    for done in (compacted,):
        done.then(fn=_cache_status, inputs=[character, thread_state], outputs=[status],
                  show_progress="hidden")

    # -- voice ------------------------------------------------------------- #
    #
    # The two switches are Forge settings and are written the moment they are
    # tapped, so the pop-up and Settings -> Voice Chat are two views of one
    # value rather than two values that have to be kept in step.
    #
    # ``input`` and not ``change``: ``change`` also fires when the server puts
    # a value in, and opening the pop-up puts both stored values in. Listening
    # to it would write the settings file every time somebody looked.
    voice["auto_send"].input(fn=mc_voice_ui.set_auto_send, inputs=[voice["auto_send"]],
                             outputs=[voice["auto_send"]], queue=False)
    voice["auto_speak"].input(fn=mc_voice_ui.set_auto_speak, inputs=[voice["auto_speak"]],
                              outputs=[voice["auto_speak"]], queue=False)

    # -- threads ---------------------------------------------------------- #
    #
    # Favourites and Threads are rows Python draws and the browser presses: a
    # row opens its thread, its star keeps it at the top (javascript/
    # llm_studio.js nominates into ``thread_at`` and presses ``thread_now``).
    # A favourite with another character is opened by remembering it and
    # choosing that character, whose own change handler below opens the
    # thread remembered for it (:func:`_select_character`).
    pressed = thread_now.click(fn=_thread_pressed,
                               inputs=[character, thread_at, search, message, thread_state],
                               outputs=([threads, favourites, character, thread_state, message]
                                        + view + screens), queue=False)
    # A star changes how many favourites there are, and the first makes the
    # section appear.
    pressed.then(fn=_favourites_section, outputs=[favourites_section], queue=False)
    pressed.then(fn=_context_view, inputs=[character, thread_state], outputs=context_drawn,
                 queue=False)
    search.change(fn=_thread_rows, inputs=[character, search, thread_state],
                  outputs=[threads], queue=False)
    made = new_thread.click(fn=_new_thread, inputs=[character, search],
                            outputs=[threads, thread_state] + view + screens, queue=False)
    made.then(fn=_context_view, inputs=[character, thread_state], outputs=context_drawn,
              queue=False)
    removed = delete.click(fn=_delete_thread,
                           inputs=[character, thread_state, search, revision_state],
                           outputs=[threads, thread_state] + view, queue=False)
    # The pop-up stays open over a delete: what it shows of the thread that is
    # open now -- a star, its emotion, its start -- is drawn again.
    removed.then(fn=_favourite_rows, inputs=[character, thread_state], outputs=[favourites],
                 queue=False)
    removed.then(fn=_favourites_section, outputs=[favourites_section], queue=False)
    removed.then(fn=_generation_view, inputs=[character, thread_state],
                 outputs=generation_drawn, queue=False)
    removed.then(fn=_context_view, inputs=[character, thread_state], outputs=context_drawn,
                 queue=False)
    removed.then(fn=_emotion_view, inputs=[character, thread_state], outputs=emotions_drawn,
                 queue=False)
    removed.then(fn=_start_view, inputs=[character, thread_state], outputs=starts_drawn,
                 queue=False)

    rename.click(fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
                 outputs=[rename_box, rename_row], queue=False)
    cancel_rename.click(fn=lambda: (gr.update(value="", visible=False), gr.update(visible=False)),
                        outputs=[rename_box, rename_row], queue=False)
    for control in (rename_box.submit, save_rename.click):
        renamed = control(fn=_rename_thread,
                          inputs=[character, thread_state, rename_box, search, revision_state],
                          outputs=[threads, rename_box, rename_row, status, header], queue=False)
        renamed.then(fn=_favourite_rows, inputs=[character, thread_state],
                     outputs=[favourites], queue=False)

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
              + [system_preview, status, character_face] + voice_fields)

    switched = character.change(fn=_select_character, inputs=[character, search, message],
                                outputs=[threads, thread_state, message] + view
                                + [name, context, greeting, system]
                                + [editing, system_preview, character_face]
                                + voice_fields, queue=False)
    switched.then(fn=_faces_update, inputs=[character], outputs=[transcript], queue=False)
    # Chosen in the pop-up, which stays open: its lists, and the sections of
    # the thread the change opened, are drawn again.
    switched.then(fn=_favourite_rows, inputs=[character, thread_state], outputs=[favourites],
                  queue=False)
    switched.then(fn=_emotion_view, inputs=[character, thread_state], outputs=emotions_drawn,
                  queue=False)
    switched.then(fn=_start_view, inputs=[character, thread_state], outputs=starts_drawn,
                  queue=False)
    switched.then(fn=_generation_view, inputs=[character, thread_state],
                  outputs=generation_drawn, queue=False)
    switched.then(fn=_context_view, inputs=[character, thread_state], outputs=context_drawn,
                  queue=False)
    edit_character.click(fn=_open_character, inputs=[character], outputs=editor, queue=False)
    new_character.click(fn=_new_character, outputs=editor, queue=False)
    refresh_characters.click(fn=_refresh_characters, inputs=[character],
                             outputs=[character, status], queue=False)
    system_open.click(fn=None, inputs=[character], outputs=None, js=OPEN_SYSTEM_EDITOR,
                      queue=False)
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
        inputs=([editing, name, context, greeting, system] + [character_face]
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
    #
    # Every one of these is pressed by the row in a bubble, with the message
    # it is on nominated in ``action_at``. The nomination rides in the same
    # request as the press, so there is nothing to race: the handler reads it
    # against the map the transcript was drawn with and refuses one that no
    # longer names a message rather than acting on the message now at that
    # place.
    where = [character, thread_state, positions, action_at]

    nominated["back"].click(fn=_page_at(-1), inputs=where + [revision_state],
                            outputs=view, queue=False)
    nominated["forward"].click(fn=_page_at(1), inputs=where + [revision_state],
                               outputs=view, queue=False)
    nominated["drop"].click(fn=_drop_at, inputs=where + [revision_state],
                            outputs=view, queue=False)

    # One shape for both roles now. Edit used to be able to move the panel onto
    # another thread -- editing one of your own messages branched -- and
    # answered with the thread list and the open thread in front of the view to
    # say so. It edits in place, so there is nothing in front of the view. The
    # editor's Save still reads ``selected``, which the open answer set.
    nominated["edit"].click(fn=_edit_at, inputs=where, outputs=view, queue=False)
    save_edit.click(fn=_commit_edit,
                    inputs=[character, thread_state, selected, edit_box, edit_image,
                            revision_state],
                    outputs=view, queue=False)
    cancel_edit.click(fn=lambda: (gr.update(visible=False), gr.update(value=None, visible=False),
                                  gr.update(visible=True)),
                      outputs=[edit_row, edit_image, composer], queue=False)

    nominated["branch"].click(fn=_branch_at, inputs=where + [search, revision_state],
                              outputs=[threads, thread_state] + view, queue=False)
    nominated["delete"].click(fn=_delete_at, inputs=where + [revision_state],
                              outputs=view, queue=False)
    nominated["delete_from"].click(fn=_delete_from_at, inputs=where + [revision_state],
                                   outputs=view, queue=False)

    # -- the attachment ---------------------------------------------------- #
    #
    # The paperclip only makes the chip visible. Opening the file picker is the
    # browser's own, forwarded to this component's file input by
    # javascript/llm_studio.js -- a press that reaches Python and comes back is
    # a round trip to open a dialog that was already one tap away.
    refresh_now.click(fn=_refresh_from_event, inputs=[character, thread_state, refresh_at],
                      outputs=view, queue=False)
    open_now.click(fn=_open_from_event, inputs=[character, open_at, search],
                   outputs=[threads, thread_state] + view, queue=False)
    paste_now.click(fn=_attach_staged, inputs=[paste_token],
                    outputs=[attachment, status], queue=False)

    attach.click(fn=_offer_attachment, outputs=[attachment, status], queue=False)
    attachment.clear(fn=_cleared_attachment, outputs=[attachment, status], queue=False)
    edit_attach.click(fn=lambda: gr.update(visible=True), outputs=[edit_image], queue=False)
    edit_image.clear(fn=lambda: gr.update(visible=False), outputs=[edit_image], queue=False)

    # -- sending, and the three ways of asking again ---------------------- #

    # No sampling controls among them: a reply is sampled at the thread's own
    # settings, read where its request is built (mc_llm_generation).
    sent = [character, thread_state, message, attachment, revision_state]
    replying = send.click(fn=_send, inputs=sent, outputs=stream, show_progress="minimal")
    # The same pathway from the keyboard. Gradio fires ``submit`` on Enter for
    # a composer that is one line tall, which is what this one starts as.
    submitted = message.submit(fn=_send, inputs=sent, outputs=stream,
                               show_progress="minimal")
    # The streaming handlers whose outputs carry the thread list and the open
    # thread: regenerating in the middle of a thread branches, and a panel
    # left pointing at the thread it came from would apply the next action to
    # the wrong conversation. Continue and Send again from here can land in a
    # branch too -- Continue on an earlier reply, and Send again from here
    # always (specification S8).
    asking = where + [search, revision_state]
    regenerating = nominated["regenerate"].click(
        fn=_regenerate_reply, inputs=asking,
        outputs=[threads, thread_state] + stream, show_progress="minimal")
    continuing = nominated["continue"].click(
        fn=_continue_at, inputs=asking,
        outputs=[threads, thread_state] + stream, show_progress="minimal")
    resending = nominated["resend"].click(
        fn=_resend_at, inputs=asking,
        outputs=[threads, thread_state] + stream, show_progress="minimal")

    # Built once and attached to every run below, so a sixth way of producing
    # a reply cannot be added without either joining this loop or being visibly
    # absent from it. (Five since the action row: the sheet's Regenerate and the
    # reply's own ↻ were two paths to one action.) Section 49's requirement is exactly that the registration
    # be structurally shared.
    speech_marker = mc_voice_ui.speech_marker(completed_reply, _character_named)

    for run in (replying, submitted, regenerating, continuing, resending):
        # The thread list is refreshed because an untitled thread has just been
        # named, and the view is read back off disk because the message the
        # editor pointed at may not be the message that is there now.
        run.then(fn=_thread_rows, inputs=[character, search, thread_state], outputs=[threads])
        settled = run.then(fn=_settle, inputs=[character, thread_state], outputs=view)
        # After the view is read back, and on its own: a reply started from
        # your words is read again in the background (mc_llm_cache_update), and
        # the status line says so quietly until it is done. Its own event, so
        # Send is never held behind it -- a message sent meanwhile says it is
        # waiting, from its own run.
        settled.then(fn=_cache_status, inputs=[character, thread_state], outputs=[status],
                     show_progress="hidden")
        # ``success`` and deliberately not ``then``. Gradio runs a ``then``
        # continuation whether or not the event before it raised, which would
        # make "the run reached its terminal callback" the trigger for speaking
        # -- and a run that failed reaches it too. ``success`` runs only after
        # an event that did not raise, and the handler still checks what the run
        # actually left behind, because a nominally successful terminal callback
        # is not the same claim as a whole reply.
        # The operation id rides with the character now. The record it reads is
        # that operation's own, so two replies finishing at once cannot take
        # each other's answer -- which one process-global slot could, and did.
        run.success(fn=speech_marker, inputs=[character, cancellation],
                    outputs=[voice_plumbing["token"]])

    # No ``cancels=``. Closing the follower would stop nothing -- the reply is
    # on a service thread -- and closing it used to be how a stopped reply lost
    # the yield that put Send back. Stop asks the service to cancel the
    # operation this page started, which is the same cancellation token this
    # button always set; the follower ends when it sees the terminal phase.
    stop.click(fn=_cancel, inputs=[cancellation],
               outputs=[status, send, stop, voice_plumbing["turn"],
                        voice_plumbing["run_state"]],
               queue=False)

    return {"status": status, "transcript": transcript, "header": header,
            "persona": (persona_name, persona_description),
            # What the shell wires: the state chip opens the model sheet, and
            # the menu opens the workspace chooser -- the same two sheets its
            # own bar opens, from the same corner of the screen.
            "model": [model], "modes": menu, "chip": model}


_MODEL_LABEL = "● Model"
"""What the header's state control says before the shell has told it anything."""


NOMINATED_ACTIONS = (
    ("regenerate", "Regenerate this reply"),
    ("edit", "Edit this message"),
    ("continue", "Continue this reply"),
    ("resend", "Send again from here"),
    ("branch", "Branch from here"),
    ("delete", "Delete this message"),
    ("delete_from", "Delete from here"),
    ("back", "Show the previous version"),
    ("forward", "Show the next version"),
    ("drop", "Delete this version"),
)
"""The hidden buttons the row in a bubble presses, by name, with the label each is built with.

One per action the old sheet had, and the regenerate icon's one before it,
now all of a kind. The id each gets -- ``mc-llm-chat-<name>-now``, with the
underscore of ``delete_from`` a dash -- is what ``javascript/llm_studio.js``
presses; ``tests/test_llm_panels.py`` compares the two files on it.
"""


def _nominated_buttons() -> dict:
    """The hidden buttons of :data:`NOMINATED_ACTIONS`, built where the transcript is."""
    return {name: gr.Button(label, visible=False,
                            elem_id=ui.ident("chat", f"{name.replace('_', '-')}-now"))
            for name, label in NOMINATED_ACTIONS}


# --------------------------------------------------------------------------- #
# Stores
# --------------------------------------------------------------------------- #


def _characters():
    from prompt_master.chat.characters import CharacterStore

    return CharacterStore.from_paths(mc_llm_paths.app_paths())


def _chats():
    from prompt_master.chat.history import ChatStore

    return ChatStore.from_paths(mc_llm_paths.app_paths())


# --------------------------------------------------------------------------- #
# Every change to a conversation, through the one door
# --------------------------------------------------------------------------- #
#
# Fifteen call sites in this file wrote chat files directly. Each was correct on
# its own and none of them could see the others, which is exactly the shape of
# the defect: two windows open on one thread, the second one saving the copy it
# loaded ten seconds ago, every reply that arrived in between gone with nothing
# on screen to say so.
#
# So none of them writes any more. Each builds a command -- what to do, to which
# message, and *which revision the person was looking at* -- and hands it to
# :mod:`mc_llm_conversation_service`, which validates the three together under
# the conversation's own lock. What comes back is an outcome, and a refusal is
# a sentence rather than a silent overwrite.
#
# The revision travels in a ``gr.State`` that :func:`_refresh` keeps current, so
# it is always the revision of the thread as this page last drew it.


def _expected(token):
    """One comparison token, in the shape the envelope takes.

    Two shapes because there are two kinds of file: one this store has written,
    which compares by a counter, and one written before there were counters,
    which compares by the hash of its bytes. ``None`` is for the commands that
    have nothing to compare against yet.
    """
    if isinstance(token, bool) or token is None:
        return None
    if isinstance(token, int):
        return {"kind": "revision", "value": token}
    return {"kind": "legacy", "fingerprint": str(token)}


def _command(action, who, identifier, *, expected=None, index=None, version=None,
             payload=None):
    """Build one command from this panel's inputs and run it. Never raises."""
    import uuid

    import mc_llm_conversation_service as service

    body = {"protocol_version": service.PROTOCOL_VERSION,
            "server_epoch": service.SERVER_EPOCH,
            # Minted here, once per gesture. A retry of *this* press would
            # reuse it; a second press is a second decision and gets its own.
            "operation_id": uuid.uuid4().hex,
            "action": action,
            "conversation": {"character": who or "", "thread_id": identifier or ""},
            "expected_revision": _expected(expected),
            "payload": payload or {}}
    if index is not None:
        body["target"] = {"index": int(index), "version": version}
    return service.submit(body)


def _refused(outcome) -> tuple[str, str]:
    """``(sentence, kind)`` for a command that did not happen."""
    import mc_llm_conversation_service as service

    error = outcome.get("error") or {}
    code = error.get("code") or ""
    message = error.get("message") or "That could not be done."
    kind = "warn" if code in (service.STALE_REVISION, service.THREAD_BUSY,
                              service.INVALID_INPUT, service.NOT_FOUND) else "error"
    return message, kind


def _token(conversation):
    """What the thread on screen compares as, for the next command from it."""
    return getattr(conversation, "comparison", None) if conversation is not None else None


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


STAR = ('<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" focusable="false" '
        'fill="{fill}" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 '
        '18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>')
"""A thread's star, filled while it is one of your favourites. Drawn, never
typed: the host's theme swaps an emoji in a button for an icon and takes the
button's words with it (the Lobe trap)."""


def _threads(who: str, filter_text: str = "") -> list:
    """The character's threads: starred first, then the rest, each most recent first.

    ``ChatStore.listing``'s order; the search matches the title alone.
    """
    if not who:
        return []
    try:
        listing = _chats().listing(who)
    except Exception:
        logger.debug("Model Chain: could not list threads", exc_info=True)
        return []
    needle = (filter_text or "").strip().casefold()
    return [info for info in listing if not needle or needle in info.title.casefold()]


def _thread_row(who: str, info, current: bool, with_who: bool = False) -> str:
    """One thread: a tap on it opens it, its star keeps it at the top.

    Two buttons the browser reads (``data-character``, ``data-thread``) and
    Python decides about: see :func:`_thread_pressed`. Favourites say whose
    thread it is as well (``with_who``).
    """
    prefix = ui.PREFIX
    whose, identifier = ui.escape(who), ui.escape(info.identifier)
    title = ui.escape(info.title)
    pinned = bool(getattr(info, "pinned", False))
    flags = (f'data-current="{"true" if current else "false"}" '
             f'data-pinned="{"true" if pinned else "false"}"')
    named = (f'<span class="{prefix}-thread-who">{whose}</span>' if with_who else "")
    marked = ' aria-current="true"' if current else ""
    star_says = (f"Unstar {title}: off the top of the list" if pinned
                 else f"Star {title}: keep it at the top")
    return (f'<div class="{prefix}-thread-row" role="listitem" {flags}>'
            f'<button type="button" class="{prefix}-thread-open" data-character="{whose}" '
            f'data-thread="{identifier}"{marked}>'
            f'<span class="{prefix}-thread-title">{title}</span>{named}</button>'
            f'<button type="button" class="{prefix}-thread-star" data-character="{whose}" '
            f'data-thread="{identifier}" aria-pressed="{"true" if pinned else "false"}" '
            f'aria-label="{star_says}" title="{star_says}">'
            f'{STAR.format(fill="currentColor" if pinned else "none")}</button></div>')


def _thread_rows(who: str, filter_text: str = "", current: str = "") -> str:
    """The Threads list: the character's threads as rows, the open one marked."""
    prefix = ui.PREFIX
    if not who:
        return f'<div class="{prefix}-thread-empty">Choose who you are talking to first.</div>'
    found = _threads(who, filter_text)
    if not found:
        typed = (filter_text or "").strip()
        note = (f"No thread matches \u201c{ui.escape(typed)}\u201d." if typed
                else "No threads yet: + New thread starts one.")
        return f'<div class="{prefix}-thread-empty">{note}</div>'
    rows = "".join(_thread_row(who, info, info.identifier == (current or "")) for info in found)
    return f'<div class="{prefix}-thread-list" role="list" aria-label="Threads">{rows}</div>'


def _favourites_of_everyone() -> list:
    """Every starred thread with every character, ``(character, info)``, most recent first."""
    found = []
    for name in _character_choices():
        try:
            found.extend((name, info) for info in _chats().favourites(name))
        except Exception:
            logger.debug("Model Chain: could not read a character's starred threads",
                         exc_info=True)
    found.sort(key=lambda pair: -pair[1].updated)
    return found


def _favourite_rows(who: str = "", current: str = "") -> str:
    """Favourites, on top of the pop-up: the starred threads of every character.

    "make it easy to favorite an existing thread i have with one of my
    characters, so i can quickly load and continue" (2026-11-06): one tap from
    the pop-up to any of them, whoever it is with, once the section is opened
    -- it is folded, and its label says how many (:func:`_favourites_label`).
    Nothing at all while none is starred, and no section either
    (:func:`_favourites_section`): the stars on the Threads rows say how one
    gets here.
    """
    found = _favourites_of_everyone()
    if not found:
        return ""
    prefix = ui.PREFIX
    rows = "".join(_thread_row(name, info, name == who and info.identifier == (current or ""),
                               with_who=True)
                   for name, info in found)
    return (f'<div class="{prefix}-favourites">'
            f'<div class="{prefix}-thread-list" role="list" aria-label="Favourite threads">'
            f'{rows}</div></div>')


THREAD_ACTIONS = ("open", "star")
"""What a press on a row of Favourites or Threads may ask for."""


def _view_saying(note: str, kind: str = "info") -> list:
    """The view's outputs left as they are but for the status line, which says ``note``.

    The status line is the view's fourth output, as :func:`_refresh` builds it.
    """
    view = [gr.update()] * len(_refresh(None, ""))
    view[3] = ui.notice(note, kind)
    return view


def _thread_pressed(who, nomination, filter_text="", typed="", current=""):
    """A row of Favourites or Threads was pressed: open its thread, or star it.

    Answers threads, favourites, the character, the thread open, the composer,
    the view and the surfaces. The nomination names the thread and whose it
    is, read against what is on disk now: a thread deleted in the other window
    meanwhile is said, never opened or starred into a pin to nothing.

    A star keeps the pop-up open and changes nothing else. A thread of the
    character you are talking to opens as Threads always opened one, and the
    pop-up goes. A favourite with another character is remembered and that
    character chosen: its change handler (:func:`_select_character`) opens the
    thread remembered for it, as it opens everything else about a character.
    """
    import json

    surfaces = len(_close_screens())
    unchanged = [gr.update()] * (5 + len(_refresh(None, "")) + surfaces)
    try:
        asked = json.loads(nomination or "{}")
    except ValueError:
        return unchanged
    if not isinstance(asked, dict) or asked.get("do") not in THREAD_ACTIONS:
        return unchanged
    wanted = str(asked.get("thread") or "").strip()
    whose = str(asked.get("character") or who or "").strip()
    if not wanted or not whose:
        return unchanged
    lists = [_thread_rows(who, filter_text, current), _favourite_rows(who, current)]
    try:
        # Another character's only if it is one of yours: Favourites lists no
        # other, so any other name is not a press this page made.
        known = whose == (who or "") or whose in _character_choices()
        there = known and _chats().path_for(whose, wanted).is_file()
    except Exception:
        there = False
    if not there:
        return (lists + [gr.update()] * 3 + _view_saying("That thread is no longer there.", "warn")
                + [gr.update()] * surfaces)
    if asked["do"] == "star":
        try:
            _chats().toggle_pinned(whose, wanted)
        except Exception:
            logger.warning("Model Chain: a thread could not be starred", exc_info=True)
            return (lists + [gr.update()] * 3
                    + _view_saying("That thread could not be starred.", "warn")
                    + [gr.update()] * surfaces)
        return ([_thread_rows(who, filter_text, current), _favourite_rows(who, current)]
                + [gr.update()] * (3 + len(_refresh(None, ""))) + [gr.update()] * surfaces)
    if whose != (who or ""):
        _selected(whose, wanted)
        return ([gr.update(), gr.update(), gr.update(value=whose), gr.update(), gr.update()]
                + [gr.update()] * len(_refresh(None, "")) + _close_screens())
    return ([_thread_rows(who, filter_text, wanted), _favourite_rows(who, wanted), gr.update()]
            + _open_thread(who, wanted, typed) + _close_screens())


def _load(who: str, identifier: str):
    """One thread, with its pictures where pictures now live.

    The migration is here rather than in a script somebody has to run: a chat
    written before there was an attachment folder carries its stills inline,
    and the first time it is opened they are written out to files and the chat
    is saved without them. Once, per chat, invisibly -- and a chat that has
    nothing to move is not written at all.
    """
    import mc_llm_conversation_store as guarded

    if not who or not identifier:
        return None
    key = guarded.Key(character=who, thread_id=identifier)
    store = _chats()
    try:
        conversation, _ = guarded.read(store, key)
    except Exception:
        logger.debug("Model Chain: could not load thread %s", identifier, exc_info=True)
        return None
    if not _needs_adoption(conversation):
        return conversation
    try:
        # A guarded maintenance transaction rather than a save inside a read.
        # It takes the same lock every other write takes, so it cannot land on
        # top of another window's edit, and it does nothing at all while a
        # reply is being generated -- a background tidy-up must never be the
        # reason a reply cannot be saved.
        committed = guarded.maintain(store, key,
                                     lambda copy: mc_llm_attachments.adopt(copy, who))
        if committed.written:
            logger.info("Model Chain: moved this thread's attachments onto disk, beside the "
                        "conversation")
            return committed.conversation
    except Exception:
        logger.warning("Model Chain: could not move this thread's attachments onto disk; "
                       "they stay inside the chat file", exc_info=True)
    return conversation


def _needs_adoption(conversation) -> bool:
    """Whether this thread still carries a picture inside the chat file.

    Asked before the transaction rather than inside it so that opening an
    already-migrated thread -- which is every thread, after the first time --
    takes a lock for exactly as long as it takes to answer "no": not at all.
    """
    return any(message.image and not message.image_path
               for message in getattr(conversation, "messages", None) or ())


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
        body = _body(message, _meta(conversation, message))
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


META_CLASS = "mc-llm-meta"
"""The class of the hidden marker in every bubble, which the page script reads.

Four attributes: ``data-mc-role``, ``data-mc-versions`` and ``data-mc-active``
for the row's version pager, and ``data-mc-key`` for the Voice Box. A span
rather than anything cleverer because a span with ``data-`` attributes is what
Gradio's sanitiser lets through a Chatbot message; it holds no text, so it
reads as nothing, and ``style.css`` keeps the paragraph it sits in out of the
layout. In the markdown it is a line of its own at the top, because a line
appended after a code fence would keep the fence open and a span joined to a
heading would unmake it.
"""


def _meta(conversation, message) -> str:
    """The marker for one message, as the markdown the transcript draws.

    The key is the thread and a digest of the role and the words, and not the
    message's index: a message deleted above moves every index below it, and
    the render the Voice Box filed under a key has to stay that message's. Two
    messages with the same words in one thread share a key, and so share a
    render, which is the right answer for the same words. A different version
    of a reply is different words, so it is a key of its own -- the render the
    Play button offers is of the version showing.
    """
    import hashlib

    from prompt_master.chat.history import ASSISTANT

    thread = str(getattr(conversation, "identifier", "") or "")
    digest = hashlib.sha1(f"{message.role}\n{message.text}".encode("utf-8")).hexdigest()[:12]
    versions = max(1, len(getattr(message, "versions", None) or [""]))
    return (f'<span class="{META_CLASS}" '
            f'data-mc-role="{"assistant" if message.role == ASSISTANT else "user"}" '
            f'data-mc-versions="{versions}" '
            f'data-mc-active="{max(0, min(int(getattr(message, "active", 0) or 0), versions - 1))}" '
            f'data-mc-key="{ui.escape(thread)}:{digest}" hidden></span>')


def _body(message, meta: str = "") -> str:
    """One message as the transcript draws it: the marker, the picture, then the words.

    The picture itself rather than its name. A conversation about a photograph
    that shows the reader a line of italic text saying a photograph was
    attached is a conversation missing half of itself -- most of all when it is
    reopened a week later, which is the case this was reported from.

    An image the file for has gone answers with a sentence saying so, because a
    message that was sent with a picture is not the same message without one.

    The words are drawn, not saved, by :func:`mc_llm_format.for_chatbot`: their
    math put between the Chatbot's own delimiters, their quotes marked, a lone
    tilde escaped -- and the thread keeps exactly what was written.
    """
    body = mc_llm_format.for_chatbot(message.text)
    if getattr(message, "attached", False):
        shown = (mc_llm_attachments.markup(message.image_path, message.image_name)
                 if message.image_path
                 # A chat whose pictures could not be moved onto disk still shows
                 # them; there is simply nothing to point the browser at, so the
                 # bytes go inline as they always did. Rare, and better than a
                 # transcript that has lost a picture the file still holds.
                 else f'<img src="{message.image}" alt="{ui.escape(message.image_name or "attached image")}"'
                      f' class="mc-llm-attached">')
        body = f"{shown}\n\n{body}" if body else shown
    return f"{meta}\n\n{body}" if meta and body else (meta or body)


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
    # Read by the stylesheet: the Conversation chip's dot is filled while the
    # thread has an emotion or a start set. Here because every view is drawn
    # with this heading.
    identifier = getattr(conversation, "identifier", "")
    on = "on" if _emotion_of(person, identifier) else "off"
    start = "on" if _start_of(person, identifier) else "off"
    return (f'<div class="{ui.PREFIX}-heading" data-emotion="{on}" data-start="{start}">'
            f'<span class="{ui.PREFIX}-heading-who">{ui.escape(person)}</span>'
            f'<span class="{ui.PREFIX}-heading-thread">{ui.escape(title)}</span>'
            f'</div>')


# --------------------------------------------------------------------------- #
# The thread's emotion
# --------------------------------------------------------------------------- #


def _emotion_of(who, identifier) -> str:
    """The thread's emotion, ``""`` for Off -- and Off when it cannot be read."""
    if not who or not identifier:
        return ""
    try:
        import mc_llm_emotions

        return mc_llm_emotions.for_thread(who, identifier)
    except Exception:
        logger.debug("Model Chain: could not read this thread's emotion", exc_info=True)
        return ""


def _short(text, limit: int = 40) -> str:
    """``text`` on one line, cut to ``limit`` characters with an ellipsis."""
    words = " ".join(str(text or "").split())
    return words if len(words) <= limit else words[:limit - 1].rstrip() + "\u2026"


def _emotion_label(who, identifier) -> str:
    """The Emotion section's label, saying what is on while it is folded."""
    line = _emotion_of(who, identifier)
    return f"Emotion · {_short(line)}" if line else "Emotion · Off"


EMOTION_TRASH = ('<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true" '
                 'focusable="false" fill="none" stroke="currentColor" stroke-width="2" '
                 'stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/>'
                 '<path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/></svg>')
"""Drawn, never typed: a cross in a span is what Lobe's icon swap replaces."""


def _emotion_rows(who, identifier) -> str:
    """Off and every line, the thread's own checked, each line with Delete.

    Buttons the browser reads (``data-emotion``) and Python decides about: see
    :func:`_emotion_pressed`. With no thread open there is nothing to choose
    for, and only the list can be changed.
    """
    try:
        import mc_llm_emotions

        lines = mc_llm_emotions.listed()
    except Exception:
        logger.debug("Model Chain: could not read the emotion list", exc_info=True)
        lines = []
    chosen = _emotion_of(who, identifier)
    usable = bool(who and identifier)
    rows = [_emotion_row("", "Off", chosen == "", usable)]
    rows += [_emotion_row(line, line, chosen == line, usable) for line in lines]
    return (_emotion_modes()
            + f'<div class="{ui.PREFIX}-emotion-list" role="radiogroup" aria-label="Emotion">'
            + "".join(rows) + "</div>")


EMOTION_MODES = (("system", "System prompt"), ("prefill", "Starts the reply"))
"""How every thread's emotion steers (mc_llm_emotions.MODES), as the picker
names them: at the end of the thread's system prompt, or as the first words of
a new reply, Oobabooga's way."""


def _emotion_modes() -> str:
    """The switch between the two ways, above the list: one choice for every
    thread, nominated as ``{"do": "mode", "line": ...}`` (:func:`_emotion_pressed`)."""
    try:
        import mc_llm_emotions

        current = mc_llm_emotions.mode()
    except Exception:
        logger.debug("Model Chain: could not read how emotions steer", exc_info=True)
        current = "system"
    prefix = ui.PREFIX
    buttons = "".join(
        f'<button type="button" class="{prefix}-emotion-mode" role="radio" '
        f'aria-checked="{"true" if mode == current else "false"}" '
        f'data-emotion-mode="{mode}">{label}</button>'
        for mode, label in EMOTION_MODES)
    return (f'<div class="{prefix}-emotion-modes" role="radiogroup" '
            f'aria-label="How the emotion steers">{buttons}</div>')


def _emotion_row(line: str, label: str, on: bool, usable: bool) -> str:
    prefix, value, checked = ui.PREFIX, ui.escape(line), "true" if on else "false"
    choice = (f'<button type="button" class="{prefix}-emotion-choice" role="radio" '
              f'aria-checked="{checked}" data-emotion="{value}"'
              f'{"" if usable else " disabled"}>'
              f'<span class="{prefix}-emotion-check" aria-hidden="true">{"✓" if on else ""}</span>'
              f'<span class="{prefix}-emotion-text">{ui.escape(label)}</span></button>')
    delete = "" if not line else (
        f'<button type="button" class="{prefix}-emotion-delete" data-emotion="{value}" '
        f'aria-label="Delete {value}" title="Delete {value}">{EMOTION_TRASH}</button>')
    return f'<div class="{prefix}-emotion-row" data-chosen="{checked}">{choice}{delete}</div>'


def _emotion_view(who, identifier) -> tuple:
    """The rows, the section's label and the bar's heading: everything that shows it."""
    conversation = _load(who, identifier) if who and identifier else None
    return (_emotion_rows(who, identifier), gr.update(label=_emotion_label(who, identifier)),
            _heading(who, conversation))


def _emotion_choose(who, identifier, line: str) -> tuple[str, str]:
    """Set the thread's emotion through the service. ``("", "")`` when it was
    set -- the check and the chip say so -- or a sentence and its kind."""
    import mc_llm_conversation_service as service

    if not who or not identifier:
        return "Choose a character and a thread first.", "warn"
    outcome = _command(service.SET_EMOTION, who, identifier, payload={"emotion": line})
    return ("", "") if outcome.get("ok") else _refused(outcome)


def _emotion_steer(who, identifier, chosen) -> tuple[str, str]:
    """Choose how every thread's emotion steers. Through the service when a thread
    is open, so it is read again in the background under its new front; straight
    to the store when none is, since there is then nothing to read."""
    import mc_llm_conversation_service as service
    import mc_llm_emotions

    if who and identifier:
        outcome = _command(service.SET_EMOTION_MODE, who, identifier,
                           payload={"emotion_mode": str(chosen or "")})
        return ("", "") if outcome.get("ok") else _refused(outcome)
    try:
        mc_llm_emotions.set_mode(chosen)
    except mc_llm_emotions.Refused as refusal:
        return str(refusal), "warn"
    return "", ""


def _emotion_pressed(who, identifier, nomination) -> tuple:
    """A row of the Emotion section was pressed: choose its line (or Off), or
    delete it -- or one of the two ways it steers was.

    The nomination names the line by its words, not its place, so a list that
    changed in the other window meanwhile cannot turn a press on one line into
    an answer about another.
    """
    import json

    import mc_llm_emotions

    try:
        asked = json.loads(nomination or "{}")
    except ValueError:
        asked = {}
    if not isinstance(asked, dict) or asked.get("do") not in ("choose", "delete", "mode"):
        return _emotion_view(who, identifier) + (gr.update(),)
    if asked["do"] == "mode":
        note, kind = _emotion_steer(who, identifier, asked.get("line"))
        return _emotion_view(who, identifier) + (ui.notice(note, kind) if note else gr.update(),)
    line = mc_llm_emotions.clean(asked.get("line"))
    note, kind = "", "info"
    if asked["do"] == "delete":
        if line:
            mc_llm_emotions.remove(line)
    else:
        note, kind = _emotion_choose(who, identifier, line)
    return _emotion_view(who, identifier) + (ui.notice(note, kind) if note else gr.update(),)


def _emotion_added(who, identifier, text) -> tuple:
    """A new line, added to the end of the list and chosen for this thread."""
    import mc_llm_emotions

    try:
        mc_llm_emotions.add(text)
    except mc_llm_emotions.Refused as refusal:
        return _emotion_view(who, identifier) + (ui.notice(str(refusal), "warn"), gr.update())
    note, kind = _emotion_choose(who, identifier, mc_llm_emotions.clean(text))
    return _emotion_view(who, identifier) + (ui.notice(note, kind) if note else gr.update(), "")


# --------------------------------------------------------------------------- #
# The start of the next reply
# --------------------------------------------------------------------------- #


def _start_of(who, identifier) -> str:
    """The words the thread's next reply starts with, ``""`` for none."""
    if not who or not identifier:
        return ""
    try:
        import mc_llm_start_with

        return mc_llm_start_with.for_thread(who, identifier)
    except Exception:
        logger.debug("Model Chain: could not read this thread's reply start", exc_info=True)
        return ""


def _start_label(who, identifier) -> str:
    """The Start section's label, saying what is kept while it is folded."""
    words = _start_of(who, identifier)
    return (f"Start the next reply with · {_short(words)}" if words
            else "Start the next reply with")


def _start_view(who, identifier) -> tuple:
    """The section's label, the box with what is kept, and the bar's heading."""
    conversation = _load(who, identifier) if who and identifier else None
    return (gr.update(label=_start_label(who, identifier)), _start_of(who, identifier),
            _heading(who, conversation))


def _start_written(who, identifier, text) -> tuple:
    """Keep ``text`` as the start of the thread's next reply, or clear it with ``""``."""
    import mc_llm_conversation_service as service

    if not who or not identifier:
        return _start_view(who, identifier) + (
            ui.notice("Choose a character and a thread first.", "warn"),)
    outcome = _command(service.SET_START_WITH, who, identifier,
                       payload={"start_with": text or ""})
    if not outcome.get("ok"):
        note, kind = _refused(outcome)
        label, _, heading = _start_view(who, identifier)
        return label, gr.update(), heading, ui.notice(note, kind)
    note = ("The next reply starts with your words." if outcome.get("start_with")
            else "The next reply starts freely.")
    return _start_view(who, identifier) + (ui.notice(note),)


# -- the thread's generation settings ----------------------------------------- #
#
# Kept with the thread (mc_llm_generation) and read where every request in it
# is built, so these controls only have to show what is kept and keep what is
# moved: nothing a reply is sent with comes from them.


def _generation_control(setting, value):
    """One control of the Generation section, opened on ``value``.

    Out of Forge's ``ui-config.json``, as the seed box is (:func:`ui.seed_box`):
    the host writes a labelled control's stored value -- and a slider's range --
    back over what is built, and these share their labels with other controls
    in LLM Studio's tab. Their values are the thread's and nothing else's.
    """
    ident = ui.ident("chat", "generation-" + setting.key.replace("_", "-"))
    if setting.key == "seed":
        return ui.seed_box(info=setting.info, value=value, elem_id=ident)
    control = gr.Slider(label=setting.label, minimum=setting.minimum,
                        maximum=setting.maximum, step=setting.step, value=value,
                        info=setting.info or None, elem_id=ident)
    control.do_not_save_to_config = True
    return control


def _generation_record(who, identifier) -> dict:
    """What the thread chose; ``{}`` for none, or when it cannot be read."""
    import mc_llm_generation

    if not who or not identifier:
        return {}
    try:
        return mc_llm_generation.for_thread(who, identifier)
    except Exception:
        logger.debug("Model Chain: could not read this thread's generation settings",
                     exc_info=True)
        return {}


def _generation_values(who, identifier) -> dict:
    """Every setting's value for the thread, by key: its choice, else the default."""
    import mc_llm_generation

    return mc_llm_generation.in_force(_generation_record(who, identifier),
                                      mc_llm_generation.base())


def _generation_label(who, identifier) -> str:
    import mc_llm_generation

    return mc_llm_generation.label(_generation_record(who, identifier))


def _generation_view(who, identifier) -> list:
    """The section's label and every control, in ``mc_llm_generation.KEYS`` order."""
    import mc_llm_generation

    values = _generation_values(who, identifier)
    return ([gr.update(label=_generation_label(who, identifier))]
            + [values[key] for key in mc_llm_generation.KEYS])


def _generation_chosen(key: str):
    """The handler for one control: keep what it says for the open thread.

    Answers the section's label and the status line, which stays as it is
    unless the choice could not be kept. Fired as the control moves and again
    as it is let go (a slider's ``release``), so the last value is the one kept
    whichever request arrives last.
    """
    import mc_llm_generation

    def chosen(who, identifier, value):
        try:
            record = mc_llm_generation.choose(who, identifier, key, value)
        except mc_llm_generation.Refused as refusal:
            return gr.update(), ui.notice(str(refusal), "warn")
        except Exception:
            logger.warning("Model Chain: a thread's generation setting could not be kept",
                           exc_info=True)
            return gr.update(), ui.notice("That setting could not be kept.", "error")
        return gr.update(label=mc_llm_generation.label(record)), gr.update()

    chosen.__name__ = f"_generation_chosen_{key}"
    return chosen


def _generation_reset(who, identifier) -> list:
    """Forget the thread's choices: every control back to its default."""
    import mc_llm_generation

    if not who or not identifier:
        return _generation_view(who, identifier) + [
            ui.notice("Choose a character and a thread first.", "warn")]
    try:
        mc_llm_generation.reset(who, identifier)
    except Exception:
        logger.warning("Model Chain: a thread's generation settings could not be reset",
                       exc_info=True)
        return _generation_view(who, identifier) + [
            ui.notice("The settings could not be reset.", "error")]
    return _generation_view(who, identifier) + [
        ui.notice("This thread is back to its defaults.")]


# -- the context window ---------------------------------------------------------- #


def _context_reading(who, identifier):
    """How full the thread's context is (``mc_llm_conversation_ops.context_reading``),
    or None for no thread, or when it cannot be said."""
    if not who or not identifier:
        return None
    try:
        import mc_llm_conversation_ops as ops
        from mc_llm_conversation_store import Key

        return ops.context_reading(Key(character=who, thread_id=identifier))
    except Exception:
        logger.debug("Model Chain: could not read how full this thread's context is",
                     exc_info=True)
        return None


def _compacting(reading) -> bool:
    progress = (reading or {}).get("compacting") or {}
    return progress.get("phase") in ("waiting", "preparing", "writing")


def _context_state(reading) -> str:
    """``ok``, ``near`` (within a tenth of where the window moves by itself),
    ``full`` (past it: the next message moves it) or ``compacting``."""
    if _compacting(reading):
        return "compacting"
    used, room = int(reading.get("used") or 0), int(reading.get("room") or 0)
    if room and used > room:
        return "full"
    if room and used > room * 0.9:
        return "near"
    return "ok"


def _context_numbers(reading) -> str:
    """``4,401 / 8,192``, with ``≈`` when it is an estimate."""
    mark = "" if reading.get("exact") else "≈"
    return f"{mark}{int(reading.get('used') or 0):,} / {int(reading.get('limit') or 0):,}"


def _context_title(reading) -> str:
    """The meter's longer reading, for its tooltip and the section's first line."""
    sent, total = int(reading.get("sent") or 0), int(reading.get("messages") or 0)
    parts = [f"{int(reading.get('used') or 0):,} of {int(reading.get('limit') or 0):,} "
             f"tokens in context{'' if reading.get('exact') else ' (estimated)'}",
             f"{sent} of {total} messages sent"]
    summary = reading.get("summary")
    if summary:
        parts.append(f"a summary of the {int(summary.get('covers') or 0)} before them "
                     f"({int(summary.get('tokens') or 0):,} tokens)")
    parts.append(f"{int(reading.get('reserve') or 0):,} kept for a reply")
    parts.append(f"older messages leave past {int(reading.get('room') or 0):,}")
    if _compacting(reading):
        parts.append("compacting now")
    return " · ".join(parts)


def _context_meter(who, identifier=None, reading=None) -> str:
    """The meter's markup for the status line: quiet numbers, nothing for no thread."""
    from html import escape

    if reading is None:
        reading = _context_reading(who, identifier)
    if not reading:
        return ""
    return (f'<span class="mc-llm-context" data-state="{_context_state(reading)}" '
            f'title="{escape(_context_title(reading), quote=True)}">'
            f'{escape(_context_numbers(reading))}</span>')


def _context_label(reading) -> str:
    """``Context · 4,401 / 8,192``, and ``· summary`` for a compacted thread."""
    if not reading:
        return "Context"
    label = f"Context · {_context_numbers(reading)}"
    if reading.get("summary"):
        label += " · summary"
    return label


def _context_info(reading) -> str:
    if not reading:
        return ui.notice("Choose a thread to see its context.")
    return ui.notice(_context_title(reading) + ".")


def _keep_slider(reading) -> dict:
    """Keep the latest: up to the context, starting at three fifths of it -- the
    5,000 of an 8,192 the request named."""
    limit = int((reading or {}).get("limit") or 8192)
    top = max((limit // 100) * 100, 500)
    return {"minimum": 0, "maximum": top, "step": 100,
            "value": min(max(int(limit * 0.6) // 100 * 100, 500), top)}


def _summary_box(reading) -> dict:
    summary = (reading or {}).get("summary") or {}
    return {"value": summary.get("text") or "", "visible": bool(summary)}


def _context_view(who, identifier, keep=None) -> list:
    """The section's label, its reading, the Keep slider's range (and the value
    already chosen), the summary, Cancel's visibility -- and the meter."""
    reading = _context_reading(who, identifier)
    slider = _keep_slider(reading)
    if keep is not None:
        slider["value"] = min(max(int(keep), 0), slider["maximum"])
    return [gr.update(label=_context_label(reading)), _context_info(reading),
            gr.update(**slider), gr.update(**_summary_box(reading)),
            gr.update(visible=_compacting(reading)),
            _context_meter(who, identifier, reading)]


def _context_trim(who, identifier, keep) -> list:
    """Keep the latest messages that fit in ``keep`` tokens and leave the rest out."""
    import mc_llm_conversation_service as service

    if not who or not identifier:
        return _context_view(who, identifier, keep) + [
            ui.notice("Choose a character and a thread first.", "warn")]
    outcome = _command(service.SET_WINDOW, who, identifier,
                       payload={"keep_tokens": int(keep or 0)})
    if not outcome.get("ok"):
        note, kind = _refused(outcome)
        return _context_view(who, identifier, keep) + [ui.notice(note, kind)]
    kept = outcome.get("kept") or {}
    reading = outcome.get("context") or {}
    note = (f"Sending the latest {int(kept.get('messages') or 0)} of "
            f"{int(reading.get('messages') or 0)} messages "
            f"({int(kept.get('tokens') or 0):,} tokens).")
    return _context_view(who, identifier, keep) + [ui.notice(note)]


def _context_clear(who, identifier, keep) -> list:
    """No summary, and the newest messages that fit again."""
    import mc_llm_conversation_service as service

    if not who or not identifier:
        return _context_view(who, identifier, keep) + [
            ui.notice("Choose a character and a thread first.", "warn")]
    outcome = _command(service.CLEAR_WINDOW, who, identifier)
    if not outcome.get("ok"):
        note, kind = _refused(outcome)
        return _context_view(who, identifier, keep) + [ui.notice(note, kind)]
    return _context_view(who, identifier, keep) + [
        ui.notice("The context sends the newest messages that fit again.")]


def _context_cancel(who, identifier, keep) -> list:
    import mc_llm_conversation_service as service

    if who and identifier:
        _command(service.CANCEL_COMPACT, who, identifier)
    return _context_view(who, identifier, keep) + [gr.update()]


def _context_compact(who, identifier, keep, length):
    """Have the model summarize what goes, and say how it is getting on until it
    is done. A generator: the summary is written on the server's own thread, and
    this only follows it, so leaving the page loses nothing."""
    import time

    import mc_llm_conversation_service as service

    if not who or not identifier:
        yield _context_view(who, identifier, keep) + [
            ui.notice("Choose a character and a thread first.", "warn")]
        return
    outcome = _command(service.COMPACT, who, identifier,
                       payload={"keep_tokens": int(keep or 0),
                                "summary_tokens": int(length or mc_llm_compact.SUMMARY_DEFAULT)})
    if not outcome.get("ok"):
        note, kind = _refused(outcome)
        yield _context_view(who, identifier, keep) + [ui.notice(note, kind)]
        return
    asked = outcome.get("compacting") or {}
    yield _context_view(who, identifier, keep) + [
        ui.working(f"Compacting {int(asked.get('messages') or 0)} messages…")]
    from mc_llm_conversation_store import Key

    key = Key(character=who, thread_id=identifier)
    began = time.monotonic()
    while mc_llm_compact.running(key) and time.monotonic() - began < 1800:
        progress = mc_llm_compact.progress(key) or {}
        status = progress.get("status") or "Compacting…"
        written = int(progress.get("written") or 0)
        if progress.get("phase") == "writing" and written:
            status = f"Summarizing… {written:,} characters"
        yield [gr.update()] * 6 + [ui.working(status)]
        time.sleep(0.5)
    ended = mc_llm_compact.progress(key) or {}
    kind = {"done": "info", "cancelled": "warn"}.get(ended.get("phase"), "error")
    yield _context_view(who, identifier, keep) + [
        ui.notice(ended.get("status") or "Compacting ended.", kind)]


# -- the thread lists, folded -------------------------------------------------- #


def _favourites_label(count: int | None = None) -> str:
    """``Favourites · N``: how many are starred, whoever they are with."""
    if count is None:
        count = len(_favourites_of_everyone())
    return f"Favourites · {count}" if count else "Favourites"


def _favourites_section() -> dict:
    """The Favourites section's label and whether it is there: not until one is starred."""
    count = len(_favourites_of_everyone())
    return gr.update(label=_favourites_label(count), visible=bool(count))


def _thread_lists_folded() -> tuple:
    """Favourites and Threads folded, as the pop-up opens: a list is opened to be
    chosen from, never shown at the top level (2026-11-09)."""
    count = len(_favourites_of_everyone())
    return (gr.update(open=False, label=_favourites_label(count), visible=bool(count)),
            gr.update(open=False))


def _cache_status(who, identifier):
    """The status line while the conversation is read again, and Ready after.

    Quiet -- the info line with a slow dot, no bar -- because nothing waits on
    it unless a message is sent meanwhile, and that message's own run says
    *Waiting for the cache update…* in the busy style. This stops writing the
    moment such a run is going, so the two never take turns on the line.
    """
    import time

    import mc_llm_cache_update as cache_update

    if not who or not identifier:
        yield gr.update()
        return
    name = f"{who}/{identifier}"
    if not cache_update.pending(name):
        yield gr.update()
        return
    yield ui.updating("Updating cache…")
    began = time.monotonic()
    while cache_update.pending(name) and time.monotonic() - began < cache_update.WAIT_LIMIT:
        if _replying(who, identifier):
            yield gr.update()
            return
        time.sleep(0.25)
    yield gr.update() if _replying(who, identifier) else ui.notice("Ready.")


def _replying(who, identifier) -> bool:
    """Whether a reply is under way in this thread now."""
    try:
        import mc_llm_conversation_ops as ops
        from mc_llm_conversation_store import Key

        return ops.for_conversation(Key(character=who, thread_id=identifier)) is not None
    except Exception:
        return False


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

    What a button in a reply's row sends back. The browser cannot know a
    message index -- ``_view`` pairs turns, so the row a bubble is on is not
    one -- and it should not be taught to: all it can honestly report is *which
    reply this is*, counted down the transcript, and the map that answers what
    that means is already in the panel's hands.

    Counted against ``positions`` rather than against the conversation, for the
    same reason the map exists at all: the transcript the row was drawn on is
    the one ``positions`` describes, and a reply is only ever the ``n``-th
    thing on screen.
    """
    return _ordinal_at(positions, ordinal, 1)


def _prompt_at(positions, ordinal) -> int:
    """The message the ``ordinal``-th prompt of yours in the transcript is."""
    return _ordinal_at(positions, ordinal, 0)


def _ordinal_at(positions, ordinal, column: int) -> int:
    try:
        wanted = int(str(ordinal).strip())
    except (TypeError, ValueError):
        return NO_SELECTION
    if wanted < 0:
        return NO_SELECTION
    found = []
    for entry in positions or ():
        try:
            _, at_column, index = entry
        except (TypeError, ValueError):
            continue
        if int(at_column) == column:
            found.append(int(index))
    return found[wanted] if wanted < len(found) else NO_SELECTION


def _nominated(positions, text) -> int:
    """The message a row's button named, as an index into the thread it was drawn on.

    ``assistant:2`` is the third reply counted down the transcript, ``user:0``
    your first prompt; a bare number is a reply's ordinal, which is what the
    regenerate icon sent before the row existed. Anything else names nothing,
    and nothing is :data:`NO_SELECTION`, which every handler refuses rather
    than reading as "the message now at that place": a transcript the browser
    is a moment behind on must never cost somebody a message.
    """
    value = str(text or "").strip()
    role, _, ordinal = value.partition(":")
    if not _:
        return _reply_at(positions, value)
    role = role.strip().lower()
    if role == "assistant":
        return _reply_at(positions, ordinal)
    if role == "user":
        return _prompt_at(positions, ordinal)
    return NO_SELECTION


STALE_NOMINATION = "That message is no longer in the thread. Reopen it and try again."
"""What a row says when its message has moved out from under it."""


def _regenerate_reply(who, identifier, positions, at, filter_text="", revision=None):
    """Regenerate the reply whose row was pressed.

    A translation and nothing else: the nomination becomes a message index and
    :func:`_regenerate` does the rest, so the row and the service cannot come
    to mean two different things -- including the branching, which is the
    part it would be worst to have two of.
    """
    index = _nominated(positions, at)
    if index == NO_SELECTION:
        yield _here(identifier, _idle(_load(who, identifier), "", None,
                                      "That reply is no longer in the thread. Reopen it "
                                      "and try again.", "warn"))
        return
    yield from _regenerate(who, identifier, index, filter_text, revision)


def _continue_at(who, identifier, positions, at, filter_text="", revision=None):
    """Continue the reply whose row was pressed; :func:`_continue` does the rest."""
    index = _nominated(positions, at)
    if index == NO_SELECTION:
        yield _here(identifier, _idle(_load(who, identifier), "", None, STALE_NOMINATION,
                                      "warn"))
        return
    yield from _continue(who, identifier, index, filter_text, revision)


def _resend_at(who, identifier, positions, at, filter_text="", revision=None):
    """Answer again the prompt whose row was pressed; :func:`_resend` does the rest."""
    index = _nominated(positions, at)
    if index == NO_SELECTION:
        yield _here(identifier, _idle(_load(who, identifier), "", None, STALE_NOMINATION,
                                      "warn"))
        return
    yield from _resend(who, identifier, index, filter_text, revision)


def _stale(who, identifier) -> list:
    """The view, unchanged, saying the row's message is gone."""
    return _refresh(_load(who, identifier), STALE_NOMINATION, "warn")


def _edit_at(who, identifier, positions, at):
    index = _nominated(positions, at)
    return _stale(who, identifier) if index == NO_SELECTION \
        else _open_editor(who, identifier, index)


def _branch_at(who, identifier, positions, at, filter_text, revision=None):
    index = _nominated(positions, at)
    if index == NO_SELECTION:
        return [gr.update(), identifier or ""] + _stale(who, identifier)
    return _branch_here(who, identifier, index, filter_text, revision)


def _delete_at(who, identifier, positions, at, revision=None):
    index = _nominated(positions, at)
    return _stale(who, identifier) if index == NO_SELECTION \
        else _delete_message(who, identifier, index, revision)


def _delete_from_at(who, identifier, positions, at, revision=None):
    index = _nominated(positions, at)
    return _stale(who, identifier) if index == NO_SELECTION \
        else _delete_from(who, identifier, index, revision)


def _page_at(step: int):
    """Show the previous or next version of the reply whose row was pressed."""
    page = _page_version(step)

    def at_row(who, identifier, positions, at, revision=None):
        index = _nominated(positions, at)
        return _stale(who, identifier) if index == NO_SELECTION \
            else page(who, identifier, index, revision)

    return at_row


def _drop_at(who, identifier, positions, at, revision=None):
    index = _nominated(positions, at)
    return _stale(who, identifier) if index == NO_SELECTION \
        else _drop_version(who, identifier, index, revision)


def _selection_updates(conversation, index: int, editing: bool = False) -> list:
    """What the editor shows for message ``index``.

    Returned in the order :data:`SELECTION_ORDER` lists, which is the one thing
    about this function that has to be kept in step with the layout.

    ``editing`` is the one state that is not a property of the message: the
    in-place editor is open on it. Everywhere else this is left alone, and the
    four answers put the editor away, empty its picture chip and give the
    composer back. Which actions apply to a message is the row's business now,
    drawn in the browser from the marker :func:`_meta` puts in the bubble.
    """
    home = [gr.update(visible=False), gr.update(), gr.update(value=None, visible=False),
            gr.update(visible=True)]
    if (conversation is None or index < 0 or index >= len(conversation.messages)):
        return home

    message = conversation.messages[index]
    return [
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
    nomination is read against, the header, the message the editor is open on
    and what the editor shows all come from the same conversation, and a
    handler that returned four of the five would leave the fifth describing a
    thread that is no longer on screen.

    The revision is the newest of them and belongs in the same list for the
    same reason: it is what the *next* command from this page will claim to
    have been looking at, so a redraw that left it behind would be a page
    quietly claiming to have seen a thread it has not.
    """
    rows, positions = _view(conversation)
    return ([rows, positions, index, ui.notice(note, kind), _heading(None, conversation)]
            + _selection_updates(conversation, index, editing)
            + [_token(conversation)])


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
    control that can only ever open something.

    The decision is :mod:`mc_llm_overlays`' rather than this function's now,
    and the answer is longer than it was: the four screens this panel owns,
    then every other pop surface in LLM Studio that the build registered --
    the shell's two sheets and the message action sheet. Those used to be
    three separate ideas of "only one at a time" that did not know about each
    other, so opening the character editor left the workspace chooser under it.
    Callers see the same first five values they always did, which is why the
    wiring below simply appends components rather than being rewritten.
    """
    wanted = name if name in SCREENS else ""
    return ([wanted] + mc_llm_overlays.updates(wanted, SCREENS)
            + mc_llm_overlays.foreign(SCREENS, OVERLAY_OWNER)[1])


OVERLAY_OWNER = "chat"
"""What this panel's open-surface State is called in the overlay registry."""


def _screen_outputs(surface) -> list:
    """The components :func:`_screens`'s answer lands in, in its order."""
    return ([surface] + mc_llm_overlays.components(SCREENS)
            + mc_llm_overlays.foreign(SCREENS, OVERLAY_OWNER)[0])


def _close_screens() -> list:
    return _screens("")


def _open_conversation(who, identifier, filter_text="") -> list:
    """The pop-up, with its lists and who you are talking to read again."""
    return (_screens("conversation")
            + [_thread_rows(who, filter_text, identifier), _favourite_rows(who, identifier),
               gr.update(choices=_character_choices(), value=who or None)])


def _persona_view() -> list:
    """You, as saved, as the pop-up opens: Settings or the other window may have
    changed it, and a box showing a stale name is a Save that puts it back."""
    person = _persona()
    return [person.name, person.description, _face_value(_persona_face())]


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


def _presentation_style() -> str:
    """The column width, bubble width and text size, as one ``<style>``.

    Read from the host's own settings through :mod:`mc_assistant_settings`,
    which validates them -- a Gradio radio stores whatever string it displayed
    and a hand-edited config can hold anything, so what reaches the page is
    always one of the values that module names.
    """
    try:
        import mc_assistant_settings

        return f"<style>{mc_assistant_settings.style()}</style>"
    except Exception:
        logger.debug("Model Chain: could not read the presentation settings", exc_info=True)
        return ""


def _selected(who, identifier) -> None:
    """Remember which conversation this tab is on, and say so.

    Two things, because they are one fact with two audiences. The preference is
    where the *next* page starts; the event is how the panel already open on
    this one follows along. Before the event existed the flyout seeded itself
    once and then never moved again, so choosing a thread in the tab left the
    two views looking at different conversations with nothing to say they were.

    Publishing fails open, as all publishing does: a window that cannot be told
    shows a thread that is one selection behind, which is a refresh away.
    """
    mc_llm_state.remember(character=who or "", thread=identifier or "")
    try:
        import mc_llm_conversation_service as service

        service.publish(service.CHARACTER_CHANGED,
                        None, character=who or "", thread_id=identifier or "")
    except Exception:
        logger.debug("Model Chain: could not publish the selection", exc_info=True)


def _conversation_key() -> str:
    """This process's capability for the assistant's routes. Never logged.

    Read through a function rather than at import so that a host without
    FastAPI -- or a test -- gets an empty field and a panel that works, rather
    than an import error in a module the conversation does not otherwise need.
    """
    try:
        import mc_llm_conversation_api

        return mc_llm_conversation_api.session_token()
    except Exception:
        logger.debug("Model Chain: the conversation routes have no key to publish",
                     exc_info=True)
        return ""


def _refresh_from_event(who, identifier, wanted=""):
    """Redraw the thread this tab is showing. Reads; never writes.

    What keeps one window current when the other one writes, without a polling
    loop: an event arrives, the page store presses the hidden button, this runs
    once. ``wanted`` is the thread the event was about, and a mismatch is not an
    error -- the tab has moved on since, and redrawing it with somebody else's
    conversation is the bug this check exists to prevent.

    Attachment adoption is deliberately not done here. It is a write, it is a
    write nobody asked for, and a refresh is the one path that must be
    incapable of changing a file.
    """
    import mc_llm_conversation_store as guarded

    if wanted and str(wanted).strip() and str(wanted).strip() != str(identifier or ""):
        return [gr.update()] * len(_refresh(None, ""))
    if not (who and identifier):
        return _refresh(None, "Ready.")
    try:
        conversation, _ = guarded.read(_chats(),
                                       guarded.Key(character=who, thread_id=identifier))
    except Exception:
        logger.debug("Model Chain: could not re-read %s for a refresh", identifier,
                     exc_info=True)
        return [gr.update()] * len(_refresh(None, ""))
    return _refresh(conversation, conversation.title)


def _open_from_event(who, wanted, filter_text=""):
    """Open the thread the Forge Assistant's panel chose, and remember it.

    The panel and this tab are one conversation, so a thread started or chosen
    there is this tab's thread too. Opened the way Threads opens one -- the
    list refilled with it chosen, the thread state moved, the transcript
    redrawn, the selection remembered and published -- and never more than
    that: the composer is left exactly as it is, and nothing is lifted out of
    the thread, which :func:`_open_thread` does for a press that asked for it
    and a press from elsewhere may not. A thread that cannot be read changes
    nothing, and says nothing: the panel has already said what it did.
    """
    identifier = str(wanted or "").strip()
    conversation = _load(who, identifier) if who and identifier else None
    if conversation is None:
        return [gr.update(), gr.update()] + [gr.update()] * len(_refresh(None, ""))
    _selected(who, identifier)
    return ([_thread_rows(who, filter_text, identifier), identifier]
            + _refresh(conversation, conversation.title))


def _attach_staged(token):
    """Put a staged picture into the composer's chip.

    The browser has already uploaded it and had it decoded, checked and
    re-encoded by :mod:`mc_llm_attachment_staging`; what arrives here is a
    token for a picture that is *ready*. So the chip shows exactly what will be
    sent, which is the whole reason staging exists -- a chip filled from a file
    the server has not looked at is a chip that can disagree with the message.
    """
    import mc_llm_attachment_staging as staging

    staged = staging.resolve(token)
    if staged is None:
        return gr.update(), ui.notice("That picture is no longer available. Attach it "
                                      "again.", "warn")
    if not staged.ready:
        return gr.update(), ui.notice(staged.reason or "That picture could not be used.",
                                      "warn")
    return (gr.update(value=staged.image, visible=True),
            ui.notice(f"{staged.name or 'Picture'} is attached to the next message."))


def _select_character(who, filter_text, typed=""):
    """Switch character: reload its threads, open one, fill the editor.

    The newest -- or the thread remembered for this character, when the
    selection remembered is this character's: a favourite with another
    character is remembered before that character is chosen
    (:func:`_thread_pressed`), so choosing it opens the favourite. A choice in
    the drop-down remembers nothing first, so it opens the newest, as it did.
    """
    from prompt_master.chat.characters import Character

    choices = _threads(who, filter_text)
    identifier = choices[0].identifier if choices else ""
    remembered = mc_llm_state.preferences()
    wanted = str(remembered.get("thread") or "")
    if who and wanted and remembered.get("character") == who:
        try:
            if _chats().path_for(who, wanted).is_file():
                identifier = wanted
        except Exception:
            logger.debug("Model Chain: could not look for the thread remembered", exc_info=True)
    conversation = _load(who, identifier)
    lifted, conversation = _lift(who, identifier, conversation, typed)
    try:
        loaded = _characters().load(who) if who else Character(name="")
    except Exception:
        loaded = Character(name=who or "")
    _selected(who, identifier)
    note = f"{len(choices)} thread{'s' if len(choices) != 1 else ''}."
    # The editor follows the selection. It is usually shut, and when it is not,
    # leaving it bound to the character that *was* selected would have the next
    # Save write this character's boxes over that character's file.
    return ([_thread_rows(who, filter_text, identifier), identifier, lifted]
            + _refresh(conversation, note)
            + [loaded.name, loaded.context, loaded.greeting, loaded.system,
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
    _selected(who, identifier)
    lifted, conversation = _lift(who, identifier, conversation, typed)
    if lifted is not _UNTOUCHED:
        return ([identifier, lifted]
                + _refresh(conversation, "This message never got a reply, so it is back in "
                                         "the box. Send it again when you are ready."))
    return [identifier, gr.update()] + _refresh(conversation, conversation.title)


_UNTOUCHED = gr.update()
"""What ``lifted`` is when nothing was taken back into the composer.

A single shared update rather than a fresh one per call, so that "did the lift
happen" is an identity test rather than a guess about what an update means.
"""


def _lift(who, identifier, conversation, typed):
    """Take an unanswered last message back into the composer. Guarded.

    Navigation writes, and always did: opening a thread that ends in a message
    of yours nobody answered takes it out of the thread and puts it back in the
    box. That is what the owner asked for and it stays -- but it is a *command*
    now (``restore_to_draft``), which means two things it was not before.

    It takes the conversation's lock, so it cannot land on top of another
    window's edit. And it is reachable only from a handler somebody triggered:
    the read-only refresh that keeps a second window current calls the snapshot
    path, which cannot delete anything. That mattered, because the refresh and
    the lift were the same function.
    """
    waiting = (_unanswered(conversation) if conversation is not None
               and not (typed or "").strip() else NO_SELECTION)
    if waiting == NO_SELECTION:
        return _UNTOUCHED, conversation
    import mc_llm_conversation_service as service

    text = conversation.messages[waiting].text
    outcome = _command(service.RESTORE_TO_DRAFT, who, identifier,
                       expected=_token(conversation), index=waiting)
    if not outcome.get("ok"):
        # The thread moved, or a reply is arriving in it. The message stays
        # where it is, which is exactly where Edit and Send again from here
        # still reach it.
        logger.debug("Model Chain: the unanswered message stayed in the thread: %s",
                     (outcome.get("error") or {}).get("message"))
        return _UNTOUCHED, conversation
    return text, _load(who, identifier)


def _new_thread(who, filter_text):
    """Start a thread with ``who``: greeted, opened, the pop-up put away.

    At the default generation settings, whichever thread was open (2026-11-09:
    "It should be default until I override for a thread").
    """
    if not who:
        return ([gr.update(), ""]
                + _refresh(None, "Choose a character first.", "warn")
                + _screens("threads"))  # stay where the message can be read
    import mc_llm_conversation_service as service

    outcome = _command(service.CREATE_THREAD, who, "")
    if not outcome.get("ok"):
        note, kind = _refused(outcome)
        return ([gr.update(), ""] + _refresh(None, note, kind) + _screens("threads"))
    identifier = (outcome.get("resulting_conversation") or {}).get("thread_id", "")
    conversation = _load(who, identifier)
    _selected(who, identifier)
    return ([_thread_rows(who, filter_text, identifier), identifier]
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


def _delete_thread(who, identifier, filter_text, revision=None):
    import mc_llm_conversation_service as service

    if not (who and identifier):
        return [gr.update(), ""] + _refresh(None, "Choose a thread first.", "warn")
    outcome = _command(service.DELETE_THREAD, who, identifier,
                       expected=_revision_or_read(revision, who, identifier))
    if not outcome.get("ok"):
        note, kind = _refused(outcome)
        return ([_thread_rows(who, filter_text, identifier), identifier]
                + _reopen(who, identifier, note, kind))
    left = _threads(who, filter_text)
    following = left[0].identifier if left else ""
    return ([_thread_rows(who, filter_text, following), following]
            + _reopen(who, following, "Deleted."))


def _rename_thread(who, identifier, title, filter_text, revision=None):
    import mc_llm_conversation_service as service

    conversation = _load(who, identifier)
    if conversation is None or not (title or "").strip():
        return (gr.update(), gr.update(visible=False), gr.update(visible=False),
                ui.notice("Nothing to rename.", "warn"), gr.update())
    outcome = _command(service.RENAME_THREAD, who, identifier,
                       expected=_revision_or(revision, conversation),
                       payload={"title": title.strip()})
    if not outcome.get("ok"):
        note, kind = _refused(outcome)
        return (gr.update(), gr.update(visible=False), gr.update(visible=False),
                ui.notice(note, kind), gr.update())
    renamed = _load(who, identifier)
    return (_thread_rows(who, filter_text, identifier),
            gr.update(value="", visible=False), gr.update(visible=False),
            ui.notice("Renamed."), _heading(who, renamed))


def _revision_or(revision, conversation):
    """The revision this page last drew, or the one the thread carries now.

    A handler wired before the revision State existed hands ``None``, and the
    honest answer for it is the token of the copy it is working from -- which
    is what it read a moment ago, under the same gesture. It is a weaker claim
    than a State the browser round-tripped and a far stronger one than no
    comparison at all, and it is never a *silent* overwrite: the transaction
    still refuses if the file moved between that read and this write.
    """
    if revision is None or revision == "":
        return _token(conversation)
    return revision


def _revision_or_read(revision, who, identifier):
    if revision is None or revision == "":
        return _token(_load(who, identifier))
    return revision


# --------------------------------------------------------------------------- #
# Per-message actions
# --------------------------------------------------------------------------- #


def _selection(value) -> int:
    """``value`` as a message index. A State that has never been set is None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return NO_SELECTION


def _settle(who, identifier):
    """The view read back off disk, with nothing selected: what every run ends on."""
    return _reopen(who, identifier, "Ready.")


def _page_version(step: int):
    """Show the previous or next version of the selected reply."""
    def page(who, identifier, index, revision=None):
        import mc_llm_conversation_service as service

        conversation = _load(who, identifier)
        if conversation is None or not (0 <= index < len(conversation.messages)):
            return _refresh(conversation, "Choose a message first.", "warn")
        message = conversation.messages[index]
        wanted = max(0, min(message.active + step, len(message.versions) - 1))
        outcome = _command(service.SELECT_VERSION, who, identifier,
                           expected=_revision_or(revision, conversation),
                           index=index, version=wanted)
        if not outcome.get("ok"):
            note, kind = _refused(outcome)
            return _reopen(who, identifier, note, kind, index=index)
        # The message stays the one being paged: the row in its bubble reads
        # the version from the marker the refresh draws, so reading three
        # variants is three taps on one control.
        return _reopen(who, identifier,
                       f"Showing version {wanted + 1} of {len(message.versions)}.",
                       index=index)

    return page


def _drop_version(who, identifier, index, revision=None):
    import mc_llm_conversation_service as service

    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return _refresh(conversation, "Choose a message first.", "warn")
    message = conversation.messages[index]
    if len(message.versions) <= 1:
        return _refresh(conversation,
                        "This message has only one version — deleting it would delete the "
                        "message.", "warn", index=index)
    outcome = _command(service.DROP_VERSION, who, identifier,
                       expected=_revision_or(revision, conversation),
                       index=index, version=message.active)
    if not outcome.get("ok"):
        note, kind = _refused(outcome)
        return _reopen(who, identifier, note, kind, index=index)
    return _reopen(who, identifier, "Version deleted.", index=index)


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


def _commit_edit(who, identifier, index, text, picture, revision=None):
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
    import mc_llm_conversation_service as service

    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return _refresh(conversation, "Choose a message first.", "warn")
    message = conversation.messages[index]
    # The editor's chip *is* the message's picture, except for one case: a chat
    # old enough to still be carrying it inline and too broken to have been
    # moved onto disk when it was opened. There is nothing to show in the chip
    # then, so an empty chip must not be read as "the user took the picture
    # away" -- which is what ``keep`` says and what it is the default for.
    shown = bool(message.image_path) or not message.image
    payload = {"text": (text or "").strip(), "image_action": service.KEEP}
    if shown:
        payload["image_action"] = service.REMOVE if picture is None else service.REPLACE
        if picture is not None:
            staged = _stage_picture(picture, message.image_name)
            if staged is None:
                return _refresh(conversation, "That picture could not be kept.", "error",
                                index=index, editing=True)
            payload["attachment_token"] = staged
    outcome = _command(service.EDIT_MESSAGE, who, identifier,
                       expected=_revision_or(revision, conversation),
                       index=index, version=message.active, payload=payload)
    if not outcome.get("ok"):
        note, kind = _refused(outcome)
        return _reopen(who, identifier, note, kind, index=index, editing=True)
    return _reopen(who, identifier, "Edited.")


def _stage_picture(picture, name: str = "") -> str:
    """One decoded picture from a Gradio chip, staged. Its token, or ``None``.

    The chip hands back a PIL image, because asking Gradio for a path raises on
    this host -- the composer's own note has the mechanism, and the note is
    where it stays: a second copy of that sentence is a second thing to update
    when it stops being true. Staging wants bytes, so the image is encoded once
    here, losslessly, rather than losing a generation of quality on the way
    through a second door.
    """
    import io

    import mc_llm_attachment_staging as staging

    try:
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
    except Exception:
        logger.warning("Model Chain: could not read the chosen picture", exc_info=True)
        return None
    staged = staging.stage(buffer.getvalue(), name or ui.ATTACHED)
    return staged.token if staged.ready else None


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


def _branch_here(who, identifier, index, filter_text, revision=None):
    """Copy the thread up to this message, so an alternative can be explored.

    A branch is a copy, which is the vendored store's own semantics: the two
    conversations then diverge as ordinary threads with nothing shared, rather
    than as a tree every later operation would have to understand.
    """
    import mc_llm_conversation_service as service

    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return ([gr.update(), identifier or ""]
                + _refresh(conversation, "Choose a message first.", "warn"))
    outcome = _command(service.BRANCH, who, identifier,
                       expected=_revision_or(revision, conversation), index=index)
    if not outcome.get("ok"):
        note, kind = _refused(outcome)
        return ([gr.update(), identifier or ""]
                + _reopen(who, identifier, note, kind, index=index))
    made = (outcome.get("resulting_conversation") or {}).get("thread_id", "")
    branched = _load(who, made)
    _selected(who, made)
    return ([_thread_rows(who, filter_text, made), made]
            + _refresh(branched,
                       "Branched — this is a new thread, and the one it came from is "
                       "untouched."))


def _delete_message(who, identifier, index, revision=None):
    import mc_llm_conversation_service as service

    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return _refresh(conversation, "Choose a message first.", "warn")
    outcome = _command(service.DELETE_MESSAGE, who, identifier,
                       expected=_revision_or(revision, conversation), index=index)
    if not outcome.get("ok"):
        note, kind = _refused(outcome)
        return _reopen(who, identifier, note, kind, index=index)
    return _reopen(who, identifier, "Message deleted.")


def _delete_from(who, identifier, index, revision=None):
    import mc_llm_conversation_service as service

    conversation = _load(who, identifier)
    if conversation is None or not (0 <= index < len(conversation.messages)):
        return _refresh(conversation, "Choose a message first.", "warn")
    removed = len(conversation.messages) - index
    # The count travels with the command and is checked under the lock. A
    # "delete 4 messages" that arrives when there are now six is not a delete
    # of six -- it is a question the person has to be asked again.
    outcome = _command(service.DELETE_FROM, who, identifier,
                       expected=_revision_or(revision, conversation), index=index,
                       payload={"confirm_count": removed})
    if not outcome.get("ok"):
        note, kind = _refused(outcome)
        return _reopen(who, identifier, note, kind, index=index)
    return _reopen(who, identifier,
                   f"Deleted {removed} message{'s' if removed != 1 else ''}.")


# --------------------------------------------------------------------------- #
# Replies
# --------------------------------------------------------------------------- #


def _send(who, identifier, text, picture, revision=None):
    """Add your message to the thread, then follow the reply to it.

    Nothing here streams any more, and nothing here saves. The command goes to
    the service, which appends your turn under the conversation's lock and
    starts the reply on a thread of its own; this generator subscribes and
    draws what it sees.

    That is the whole of specification 7: the reply used to exist *inside* this
    generator, so closing it -- a refresh, a dropped queue entry, Stop wired as
    ``cancels=`` -- was the reply ending. Now closing it loses a subscription.
    """
    import mc_llm_conversation_service as service

    conversation = _load(who, identifier)
    if conversation is None:
        yield _idle(None, text, gr.update(), "Choose a character and a thread first.", "warn")
        return
    if not (text or "").strip() and picture is None:
        yield _idle(conversation, text, gr.update(), "Write a message first.", "warn")
        return

    # No sampling: the thread's own generation settings are read where the
    # request is built (mc_llm_generation), so what is sent is what is kept --
    # and a page drawn before they changed cannot send the old ones.
    payload = {"text": text or ""}
    if picture is not None:
        if not mc_llm_runtime.config().sees:
            yield _idle(conversation, text, gr.update(),
                        "The model running has no vision projector, so the attached image "
                        "cannot be sent to it. Choose one in Setup, or remove the image.",
                        "error")
            return
        staged = _stage_picture(picture, ui.picked_name(picture))
        if staged is None:
            yield _idle(conversation, text, gr.update(),
                        "That picture could not be read. Choose another, or send the "
                        "message without one.", "error")
            return
        payload["attachment_token"] = staged

    outcome = _command(service.SEND, who, identifier,
                       expected=_revision_or(revision, conversation), payload=payload)
    yield from _follow(outcome, who, identifier, text, clearing=True)


def _regenerate(who, identifier, index, filter_text="", revision=None):
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

    Both rules are the service's now, and identical from the flyout. What is
    left here is the panel half: which thread the list should be showing when a
    branch is where the reply went.
    """
    import mc_llm_conversation_service as service

    conversation = _load(who, identifier)
    if conversation is None:
        yield _here(identifier, _idle(None, "", None,
                                      "There is no reply to regenerate.", "warn"))
        return
    outcome = _command(service.REGENERATE, who, identifier,
                       expected=_revision_or(revision, conversation),
                       index=_last_reply(conversation, index))
    yield from _follow_thread(outcome, who, identifier, filter_text)


def _continue(who, identifier, index, filter_text="", revision=None):
    """Carry the reply on from where it stopped.

    On the final reply that is what it always was. On an earlier one it now
    *branches* first (specification S8): continuing in place mutated a message
    that had descendants, so the replies under it went on answering a paragraph
    that no longer said what they were answering, and nothing recorded that it
    had changed.
    """
    import mc_llm_conversation_service as service

    conversation = _load(who, identifier)
    if conversation is None:
        yield _here(identifier, _idle(None, "", None,
                                      "There is no reply to continue.", "warn"))
        return
    outcome = _command(service.CONTINUE, who, identifier,
                       expected=_revision_or(revision, conversation),
                       index=_last_reply(conversation, index))
    yield from _follow_thread(outcome, who, identifier, filter_text)


def _resend(who, identifier, index, filter_text="", revision=None):
    """Answer this message of yours again, in a branch.

    It used to drop everything after it. ``truncate_after(index)`` then
    ``save()``: every reply that followed the message, permanently deleted, with
    no branch and nothing to page back to. That is the same defect Regenerate
    was fixed for and the fix was never applied here — specification S8 calls it
    live data loss at HEAD, and it is.

    So it branches, exactly as a mid-thread Regenerate does: the thread up to
    and including your message is copied, the new answer is written in the copy,
    and the conversation it came from keeps every word of what followed.

    Your *last* message has nothing following it to keep, so that one is
    answered in place -- the thread stays the thread, as the Forge Assistant's
    SEND AGAIN expects. See ``mc_llm_conversation_ops._plan_resend``.
    """
    import mc_llm_conversation_service as service

    conversation = _load(who, identifier)
    if conversation is None or not (0 <= _selection(index) < len(conversation.messages)):
        yield _here(identifier, _idle(conversation, "", None,
                                      "Choose one of your messages first.", "warn"))
        return
    outcome = _command(service.RESEND_FROM_USER, who, identifier,
                       expected=_revision_or(revision, conversation), index=index)
    yield from _follow_thread(outcome, who, identifier, filter_text)


# --------------------------------------------------------------------------- #
# Following a reply that belongs to the server
# --------------------------------------------------------------------------- #


def _here(identifier, event):
    """One follower-shaped event, in the thread that is already open.

    The three actions that can land in a branch prefix every yield with the
    thread list and the thread they are in. A refusal never moves, so it says
    so: no change to the list, and the thread that was already open.
    """
    return (gr.update(), identifier or "") + tuple(event)


def _follow(outcome, who, identifier, typed="", clearing: bool = False):
    """Draw one operation as it happens. Writes nothing, saves nothing.

    Its ``finally`` closes a subscription and that is all, which is the point:
    ``cancels=`` used to close this generator with ``GeneratorExit`` -- a
    ``BaseException``, straight past every ``except Exception`` -- and the only
    reason a stopped reply survived was a ``finally`` block written after a bug
    report that said replies were disappearing. A browser refresh went through
    the same hole. There is no hole to go through now.

    ``clearing`` is the composer. It is emptied in the acceptance yield and
    never again: every yield used to write an unconditional ``""`` into it, so
    anything typed while a reply arrived was wiped by the next token.
    """
    import mc_llm_conversation_ops as ops

    if not outcome.get("ok"):
        note, kind = _refused(outcome)
        yield _idle(_load(who, identifier), typed, gr.update(), note, kind)
        return

    operation_id = outcome.get("operation_id", "")
    landed = (outcome.get("resulting_conversation") or {}).get("thread_id") or identifier
    conversation = _load(who, landed)
    working = _working(conversation, outcome.get("target") or {})
    target = (outcome.get("target") or {}).get("index", -1)

    first = True
    for state in ops.listen(operation_id):
        phase = state.get("phase")
        text = state.get("generated_text") or ""
        if phase in ops.TERMINAL:
            # The authority is the file, not the last patch: a completion that
            # was refused because the conversation moved must not leave the
            # reply drawn as though it had been saved.
            settled = _load(who, landed)
            rows, positions = _view(settled)
            note = state.get("status") or "Reply complete."
            kind = "warn" if phase in (ops.STOPPED, ops.INTERRUPTED) else (
                "error" if phase in (ops.FAILED, ops.SAVE_FAILED) else "info")
            if state.get("recovery_status"):
                note = ("Your reply is ready, but the conversation changed. "
                        "It is kept — open the assistant to save or discard it.")
                kind = "warn"
            yield (operation_id, rows, positions,
                   "" if first and clearing else gr.update(),
                   None if first and clearing else gr.update(),
                   ui.notice(note, kind), *IDLE, state.get("speech_turn_id") or "",
                   LLM_IDLE, _token(settled))
            return
        _provisional(working, target, text)
        rows, positions = _view(working)
        yield (operation_id, rows, positions,
               "" if first and clearing else gr.update(),
               SENT if first and clearing else gr.update(),
               ui.working(state.get("status") or "Starting…"), *BUSY,
               state.get("speech_turn_id") or "", LLM_RUNNING,
               # The revision does not move while a reply is provisional. It is
               # carried anyway so the State is written by every yield rather
               # than only by the last one -- a page whose stream was cut short
               # still holds the revision it was drawing.
               _token(conversation))
        first = False


def _follow_thread(outcome, who, identifier, filter_text=""):
    """:func:`_follow`, for the three actions that can land in another thread.

    Regenerate mid-thread, Continue on an earlier reply and Send again from an
    earlier message all branch, so the panel has to move with the reply: a
    thread list still pointing at the conversation it came from would apply the
    next action to the wrong one. Send again from the *last* message answers in
    place, lands where it started, and moves nothing.
    """
    landed = (outcome.get("resulting_conversation") or {}).get("thread_id") or identifier
    if outcome.get("ok") and landed != identifier:
        _selected(who, landed)
        moved = _thread_rows(who, filter_text, landed)
    else:
        moved = gr.update()
    for event in _follow(outcome, who, landed if outcome.get("ok") else identifier):
        yield (moved, landed if outcome.get("ok") else (identifier or "")) + tuple(event)


def _working(conversation, target: dict):
    """A copy of the thread with room for the reply that is about to arrive.

    A copy, because the reply is *provisional* until the service writes it: it
    is drawn so the reader can watch it, and it is never presented as a saved
    version. The authoritative thread is re-read when the operation ends.
    """
    if conversation is None:
        return None
    from prompt_master.chat.history import ASSISTANT, Conversation

    import mc_llm_conversation_ops as ops

    copy = Conversation.from_dict(conversation.to_dict())
    kind, index = target.get("kind"), target.get("index", -1)
    if kind == ops.APPEND and index == len(copy.messages):
        copy.append(ASSISTANT, "")
    elif kind == ops.VERSION and 0 <= index < len(copy.messages):
        copy.messages[index].add_version("")
    return copy


def _provisional(working, index: int, text: str) -> None:
    if working is None or not (0 <= index < len(working.messages)):
        return
    working.messages[index].text = text


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
IDLE = (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=True))
"""Send and Stop, as ``(send, stop)`` pairs, for the two states Python knows about.

Stop is left *interactive* in both. Hiding a control and disabling it are
different statements, and only the first is Python's to make here: the language
model going idle is not the end of the response, because the speaker may still
be reading it aloud. The browser reveals Stop for that phase -- and used to
reveal a button Gradio had already rendered ``disabled``, which is a Stop that
is visible during exactly the phase it cannot stop.

So Python owns visibility for its own half and never withdraws enablement.
``javascript/voice_chat.js`` owns the other half of visibility and clears any
``disabled`` a re-render puts back.
"""

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


def completed_reply(operation_id: str = "") -> str:
    """What one finished operation produced, read rather than consumed.

    This used to be ``take_completed_reply()`` over a process-global one-shot
    slot: one dictionary, one string, popped by whoever asked first. Two pages
    could pop each other's answer, and the page that lost spoke somebody else's
    reply into somebody else's conversation. Two threads replying at once could
    do it to each other with one page open.

    It is an operation's own record now (:mod:`mc_llm_conversation_ops`), so
    there is nothing to take and nothing to lose. Reading rather than consuming
    is safe for the same reason: a duplicate terminal callback, which a host is
    entitled to deliver, gets the same answer twice instead of somebody else's
    once, and speaking twice is prevented where it always was -- by the
    streamed-turn check in :func:`mc_voice_ui.speech_marker`.
    """
    import mc_llm_conversation_ops as ops

    found = ops.completion(operation_id)
    if found is None or found.status != ops.COMPLETED:
        return ""
    return found.final_text


def _with_pictures(messages, every_picture: bool = False):
    """Read the attached stills back for the request about to be built.

    The conversation on disk holds paths; the vendored prompt builder wants the
    embedded bytes, and knows nothing about a folder. Read here, at the one
    moment they are needed, and never written back -- a message that has an
    ``image_path`` never saves an inline copy beside it.

    Only the newest, unless ``every_picture`` (the *Show the model every
    picture* setting) asks for all of them, because that is all a request
    carries: the builder keeps ``stills_carried(n, every_picture)`` of the
    stills that survive trimming, and decoding forty photographs to send one
    would be forty reads a message. The same rule as the builder's, on the
    same count, so that the two never disagree about which stills go -- the
    builder dropping one this read had loaded would rewrite a message the
    cache had, for nothing.

    A picture a message still carries inline -- an old chat whose picture
    could not be moved onto disk -- is sized here to the same cap as the rest,
    so nothing larger than :data:`mc_llm_attachments.VISION_MAX_SIDE` reaches
    the model by any route.
    """
    from prompt_master.chat.prompt import stills_carried
    from prompt_master.imaging.preprocess import fit_data_url

    pictures = [message for message in (messages or ())
                if getattr(message, "image_path", None) or getattr(message, "image", None)]
    allowance = stills_carried(len(pictures), every_picture)
    for message in reversed(pictures):
        if allowance <= 0:
            break
        if message.image_path and not message.image:
            message.image = mc_llm_attachments.data_url(message.image_path)
        elif message.image:
            message.image = fit_data_url(message.image)
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
        + ("", LLM_IDLE, _token(conversation))


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


def _cancel(operation_id):
    """Stop the reply, and put the controls back.

    A plain callback now, and ``cancels=`` is gone from the button (7.9). It
    used to close the streaming generator where it stood, which is what made
    Stop immediate -- and also what made it dangerous: a closed generator never
    reached the yield that would have put Send back, so the panel was left
    permanently busy, and the reply survived only because of a ``finally``
    written after a bug report about replies disappearing.

    The reply is not in a generator any more. Stop sets the operation's own
    cancellation, which is the same :class:`mc_llm_sessions.Cancellation` this
    button always set and which llama.cpp honours between tokens; the follower
    ends when it sees the terminal phase. Pressing it twice is harmless, and
    pressing it after the reply finished leaves the reply and stops the audio.
    """
    import mc_llm_conversation_ops as ops

    if operation_id:
        ops.stop(str(operation_id))
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
    leave the name box and the character being written to describing two
    different characters. The generation settings are not in it: they are the
    thread's (:mod:`mc_llm_generation`), drawn by the Generation section.
    """
    return [gr.update(visible=True), editing,
            character.name, character.context, character.greeting, character.system or "",
            _system_preview(character.name, character.context, character.system),
            ui.notice(note, kind),
            # The picture this character has, if it has one. Read off disk with
            # everything else so the box cannot end up showing the face of the
            # character that *was* open in the editor.
            _face_value(_character_face(character.name))] + _voice_fields(character)


def _voice_engine_fields(existing_name: str, voice, delivery: dict) -> dict:
    """The character's voice fields after an edit, scoped to the active engine.

    The character being edited is read back from the store first, so the engine
    the user is *not* looking at keeps exactly what it had. A save that built
    these from the editor alone would silently clear the other engine's voice
    every time somebody changed a name -- which is I-3 broken by an ordinary,
    invisible action nobody would connect to the cause.

    Never fatal: a store that will not answer produces a character with only the
    engine that was edited, which is the same thing a brand-new character has.
    """
    import mc_voice_engines as engines
    # Every engine's fields, from the character format itself
    # (``characters.VOICE_FIELDS``): the list that lived here never gained
    # PocketTTS's, so each Save dropped a character's Pocket voice.
    from prompt_master.chat.characters import VOICE_FIELDS, Character

    found = {}
    try:
        character = _characters().load(str(existing_name or "").strip())
    except Exception:
        character = None
    if character is not None:
        for name in VOICE_FIELDS:
            found[name] = getattr(character, name, None)
    try:
        active = engines.active()
        # Qualified here, at the one place that knows which engine is selected.
        # A picker sends whatever the voice list gave it, and a list painted
        # before this build -- or a character imported from an oobabooga file --
        # sends the engine's own spelling. Both become the backend-qualified id
        # the shared turn contract carries (I-10).
        found.update(Character.voice_fields(
            active, engines.qualify(str(voice or "").strip(), active), delivery))
    except Exception:
        logger.warning("Model Chain: could not scope a character's voice to the active "
                       "engine", exc_info=True)
        found.update(Character.voice_fields("kokoro", str(voice or "").strip(), delivery))
    return {name: found.get(name) if name in found else None
            for name in VOICE_FIELDS}


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
    return ([gr.update(visible=False), gr.update()] + [gr.update()] * 5
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
            "This character already has a system prompt of its own: the override box.", "warn")
    composed = _system_preview(name, context, system)
    if not composed:
        return gr.update(), ui.notice("Nothing to copy yet.", "warn")
    return composed, ui.notice(
        "Copied. The Context and your persona no longer change it; empty the box to go back.")


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
    """Clear the editor for a character that does not exist yet."""
    return _editor_fields(_blank_character(), NOT_EDITING,
                          "New character: a name and Save character create it.")


def _cancel_character(who):
    """Shut the editor and put its boxes back to the selected character's.

    Cancel has to undo what New did to them, or the next Edit opens on a
    character nobody created.
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


def _save_character(editing, name, context, greeting, system, face=None, voice="",
                    voice_custom=False, *voice_values):
    """Write the editor to disk, creating or renaming as ``editing`` says.

    ``editing`` and not the drop-down, which is the whole fix. Creating refuses
    a name that is already taken rather than writing over it -- overwriting a
    character silently is the same loss by a shorter road.

    The sampling in the character's file is left as the file has it (a new one
    gets the package's): the Generation controls are each thread's now, over
    the defaults (:mod:`mc_llm_generation`), and Save is about the boxes the
    editor shows -- it does not drop what it does not show.
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

    kept = _blank_character()
    if not creating:
        try:
            kept = store.load(editing)
        except Exception:
            logger.debug("Model Chain: could not read %r's sampling to keep it", editing,
                         exc_info=True)
    # Values or ``None``s, and the difference is the checkbox: a character with
    # no delivery of its own follows Settings → Voice Chat, and keeps following
    # it when that changes. See mc_voice_ui.character_profile.
    delivery = mc_voice_ui.character_profile(voice_custom, voice_values)
    # Scoped to the *selected* engine, and the other engine's voice and profile
    # are copied through unread. Editing Ada while Sopro is selected cannot
    # change what Ada sounds like on Kokoro (I-3), and it does not give her a
    # Sopro profile she never asked for either -- absence is what inheritance is
    # made of. See Character.voice_fields and _voice_engine_fields below.
    engine_fields = _voice_engine_fields(editing or wanted, voice, delivery)
    character = Character(
        name=wanted, context=context or "", greeting=greeting or "",
        temperature=kept.temperature, top_p=kept.top_p,
        max_reply_tokens=kept.max_reply_tokens, seed=kept.seed, system=system or "",
        **engine_fields)
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
