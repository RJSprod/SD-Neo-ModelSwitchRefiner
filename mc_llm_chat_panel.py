"""Conversation: threads, characters and streamed replies (section 4.3).

Designed around reading. Section 4.3 asks for the conversation area to receive
most of the visual space and for configuration to be secondary and collapsible,
so the transcript is the centre column at full height, the thread list is a
narrow rail beside it, and everything about characters, personas and sampling
is behind accordions that open when wanted.

The state lives in the vendored ``prompt_master.chat`` package, unchanged:
characters are files in a folder other tools already understand, chats are
documents filed under the character they belong to, and both are read and
written by ``CharacterStore`` and ``ChatStore`` rather than by anything here.
That is section 16's separation kept by construction -- Conversation's history
cannot leak into Prompt Studio's, because they are not in the same files and
never pass through the same code.

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
    cancellation = gr.State(None)
    thread_state = gr.State(initial_thread)

    with gr.Row(elem_id=ui.ident("chat"), elem_classes=ui.classes("workspace")):

        # -- left rail: characters and threads (section 4.5, item 2) ------- #
        with gr.Column(scale=1, min_width=220, elem_classes=ui.classes("rail")):
            character = gr.Dropdown(
                label="Character", choices=_character_choices(),
                value=who or None, interactive=True,
                elem_id=ui.ident("chat", "character"))
            search = gr.Textbox(label="Find a thread", placeholder="Filter by title…",
                                elem_id=ui.ident("chat", "search"))
            threads = gr.Radio(label="Threads", choices=initial_threads,
                               value=initial_thread or None,
                               elem_id=ui.ident("chat", "threads"),
                               elem_classes=ui.classes("threads"))
            with gr.Row():
                new_thread = gr.Button("New", size="sm", variant="primary")
                branch = gr.Button("Branch", size="sm")
            with gr.Row():
                rename = gr.Button("Rename", size="sm")
                delete = gr.Button("Delete", size="sm", variant="stop")
            rename_box = gr.Textbox(label="New title", visible=False,
                                    elem_id=ui.ident("chat", "rename"))

        # -- centre: the transcript, then the composer --------------------- #
        with gr.Column(scale=4, min_width=420, elem_classes=ui.classes("stage")):
            status = gr.HTML(ui.notice("Ready."), elem_id=ui.ident("chat", "status"))
            transcript = gr.Chatbot(
                label=None, height=560, show_copy_button=True, render_markdown=True,
                value=_transcript(_load(who, initial_thread)),
                elem_id=ui.ident("chat", "transcript"),
                elem_classes=ui.classes("transcript"))

            with gr.Row(elem_classes=ui.classes("composer")):
                with gr.Column(scale=5):
                    message = gr.Textbox(
                        label=None, lines=3, max_lines=12, show_label=False,
                        placeholder="Write a message. Shift+Enter for a new line.",
                        elem_id=ui.ident("chat", "message"))
                with gr.Column(scale=1, min_width=140):
                    attachment = gr.Image(label="Attach", type="filepath", height=110,
                                          elem_id=ui.ident("chat", "image"))

            with gr.Row(elem_classes=ui.classes("actions")):
                send = gr.Button("Send", variant="primary", elem_id=ui.ident("chat", "send"))
                stop = gr.Button("Stop", variant="stop", interactive=False,
                                 elem_id=ui.ident("chat", "stop"))
                regenerate = gr.Button("Regenerate")
                undo = gr.Button("Undo")
                clear = gr.Button("Clear thread")

        # -- right: who is talking, and how (section 4.3: secondary) ------- #
        with gr.Column(scale=2, min_width=250, elem_classes=ui.classes("inspector")):
            with gr.Accordion("Character", open=False):
                name = gr.Textbox(label="Name", elem_id=ui.ident("chat", "name"))
                context = gr.Textbox(label="Context", lines=8,
                                     placeholder="Who the character is. Shown to the model "
                                                 "before the first line of dialogue.")
                greeting = gr.Textbox(label="Greeting", lines=3)
                system = gr.Textbox(label="System prompt override", lines=3,
                                    placeholder="Leave empty to use the built prompt.")
                # Every default here is the vendored package's own, named rather
                # than copied: these are the numbers the standalone application
                # ships with, and a second set of literals in the UI is how a
                # panel quietly stops matching the engine behind it.
                with gr.Row():
                    temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0,
                                            step=0.05, value=DEFAULT_TEMPERATURE)
                    top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, step=0.01,
                                      value=DEFAULT_TOP_P)
                reply_tokens = gr.Slider(label="Reply tokens", minimum=64, maximum=4096, step=64,
                                         value=DEFAULT_MAX_REPLY_TOKENS)
                seed = gr.Number(label="Seed", value=RANDOM_SEED, precision=0,
                                 info=f"{RANDOM_SEED} draws a fresh seed for every reply.")
                with gr.Row():
                    save_character = gr.Button("Save character", variant="primary", size="sm")
                    new_character = gr.Button("New", size="sm")
                    delete_character = gr.Button("Delete", size="sm", variant="stop")
                import_card = gr.File(label="Import a character card",
                                      file_types=[".json", ".yaml", ".yml", ".png"],
                                      elem_id=ui.ident("chat", "import"))

            with gr.Accordion("You", open=False):
                persona_name = gr.Textbox(label="Your name", value=initial_persona.name)
                persona_description = gr.Textbox(label="About you", lines=4,
                                                 value=initial_persona.description)
                save_persona = gr.Button("Save", size="sm")

    # -- wiring ----------------------------------------------------------- #

    character.change(fn=_select_character, inputs=[character, search],
                     outputs=[threads, thread_state, transcript, status, name, context, greeting,
                              system, temperature, top_p, reply_tokens, seed], queue=False)
    search.change(fn=lambda who, text: gr.update(choices=_thread_choices(who, text)),
                  inputs=[character, search], outputs=[threads], queue=False)
    threads.change(fn=_open_thread, inputs=[character, threads],
                   outputs=[transcript, thread_state, status], queue=False)

    new_thread.click(fn=_new_thread, inputs=[character, search],
                     outputs=[threads, thread_state, transcript, status], queue=False)
    branch.click(fn=_branch_thread, inputs=[character, thread_state, search],
                 outputs=[threads, thread_state, transcript, status], queue=False)
    delete.click(fn=_delete_thread, inputs=[character, thread_state, search],
                 outputs=[threads, thread_state, transcript, status], queue=False)
    rename.click(fn=lambda: gr.update(visible=True), outputs=[rename_box], queue=False)
    rename_box.submit(fn=_rename_thread, inputs=[character, thread_state, rename_box, search],
                      outputs=[threads, rename_box, status], queue=False)

    replying = send.click(
        fn=_send, inputs=[character, thread_state, message, attachment, temperature, top_p,
                          reply_tokens, seed],
        outputs=[cancellation, transcript, message, attachment, status, send, stop],
        show_progress="minimal")
    replying.then(fn=lambda who, text: gr.update(choices=_thread_choices(who, text)),
                  inputs=[character, search], outputs=[threads])

    regenerating = regenerate.click(
        fn=_regenerate, inputs=[character, thread_state, temperature, top_p, reply_tokens, seed],
        outputs=[cancellation, transcript, message, attachment, status, send, stop],
        show_progress="minimal")

    stop.click(fn=_cancel, inputs=[cancellation], outputs=[status],
               cancels=[replying, regenerating], queue=False)

    undo.click(fn=_undo, inputs=[character, thread_state], outputs=[transcript, status],
               queue=False)
    clear.click(fn=_clear, inputs=[character, thread_state], outputs=[transcript, status],
                queue=False)

    save_character.click(
        fn=_save_character,
        inputs=[character, name, context, greeting, system, temperature, top_p, reply_tokens, seed],
        outputs=[character, status], queue=False)
    new_character.click(fn=_new_character,
                        outputs=[name, context, greeting, system, status], queue=False)
    delete_character.click(fn=_delete_character, inputs=[character],
                           outputs=[character, status], queue=False)
    import_card.upload(fn=_import_character, inputs=[import_card],
                       outputs=[character, status], queue=False)

    save_persona.click(fn=_save_persona, inputs=[persona_name, persona_description],
                       outputs=[status], queue=False)

    return {"status": status, "transcript": transcript,
            "persona": (persona_name, persona_description)}


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


def _transcript(conversation) -> list[list[str | None]]:
    """The conversation as Gradio 4's Chatbot value: a list of ``[user, bot]`` pairs.

    Gradio 4.40 predates the message-shaped Chatbot value, and section 5 says
    to target the components the host actually has rather than newer Gradio
    assumptions. The pairing is the only lossy part -- two replies in a row
    become two rows with an empty left side, which is what the component draws
    correctly anyway.
    """
    from prompt_master.chat.history import ASSISTANT, USER

    if conversation is None:
        return []
    rows: list[list[str | None]] = []
    for message in conversation.messages:
        body = message.text
        if message.image_name:
            body = f"*[{message.image_name}]*\n\n{body}" if body else f"*[{message.image_name}]*"
        if message.role == USER:
            rows.append([body, None])
        elif rows and rows[-1][1] is None and message.role == ASSISTANT:
            rows[-1][1] = body
        else:
            rows.append([None, body])
    return rows


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
    return (gr.update(choices=choices, value=identifier or None), identifier,
            _transcript(conversation),
            ui.notice(f"{len(choices)} thread{'s' if len(choices) != 1 else ''}."),
            loaded.name, loaded.context, loaded.greeting, loaded.system,
            loaded.temperature, loaded.top_p, loaded.max_reply_tokens, loaded.seed)


def _open_thread(who, identifier):
    conversation = _load(who, identifier)
    if conversation is None:
        return [], identifier or "", ui.notice("Choose a thread.", "warn")
    mc_llm_state.remember(character=who or "", thread=identifier)
    return _transcript(conversation), identifier, ui.notice(conversation.title)


def _new_thread(who, filter_text):
    if not who:
        return gr.update(), "", [], ui.notice("Choose a character first.", "warn")
    store = _chats()
    conversation = store.new(who)
    _greet(conversation, who)
    store.save(conversation)
    mc_llm_state.remember(character=who, thread=conversation.identifier)
    return (gr.update(choices=_thread_choices(who, filter_text), value=conversation.identifier),
            conversation.identifier, _transcript(conversation), ui.notice("New thread."))


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


def _branch_thread(who, identifier, filter_text):
    """Copy a thread from its last turn, so an alternative can be explored.

    Branching from the end rather than from a chosen turn: picking a point is
    a per-message action and Gradio 4's Chatbot has no per-message affordance
    to hang one off. Copy then delete back is the same result in two moves.
    """
    conversation = _load(who, identifier)
    if conversation is None:
        return gr.update(), identifier or "", [], ui.notice("Choose a thread first.", "warn")
    store = _chats()
    branched = store.branch(conversation, len(conversation.messages) - 1)
    store.save(branched)
    return (gr.update(choices=_thread_choices(who, filter_text), value=branched.identifier),
            branched.identifier, _transcript(branched), ui.notice("Branched."))


def _delete_thread(who, identifier, filter_text):
    if not (who and identifier):
        return gr.update(), "", [], ui.notice("Choose a thread first.", "warn")
    _chats().delete(who, identifier)
    choices = _thread_choices(who, filter_text)
    following = choices[0][1] if choices else ""
    return (gr.update(choices=choices, value=following or None), following,
            _transcript(_load(who, following)), ui.notice("Deleted."))


def _rename_thread(who, identifier, title, filter_text):
    conversation = _load(who, identifier)
    if conversation is None or not (title or "").strip():
        return gr.update(), gr.update(visible=False), ui.notice("Nothing to rename.", "warn")
    conversation.title = title.strip()
    _chats().save(conversation)
    return (gr.update(choices=_thread_choices(who, filter_text), value=identifier),
            gr.update(value="", visible=False), ui.notice("Renamed."))


# --------------------------------------------------------------------------- #
# Replies
# --------------------------------------------------------------------------- #


def _send(who, identifier, text, image_path, temperature, top_p, reply_tokens, seed):
    yield from _reply(who, identifier, text, image_path, temperature, top_p, reply_tokens, seed,
                      regenerate=False)


def _regenerate(who, identifier, temperature, top_p, reply_tokens, seed):
    yield from _reply(who, identifier, "", None, temperature, top_p, reply_tokens, seed,
                      regenerate=True)


def _reply(who, identifier, text, image_path, temperature, top_p, reply_tokens, seed,
           regenerate: bool):
    """Stream one reply into the thread, and save the thread when it lands."""
    from prompt_master.chat.characters import (DEFAULT_MAX_REPLY_TOKENS, DEFAULT_TEMPERATURE,
                                               DEFAULT_TOP_P)
    from prompt_master.chat.history import ASSISTANT, USER
    from prompt_master.chat.prompt import build, clean_reply
    from prompt_master.core.models import RANDOM_SEED, draw_seed

    busy = (gr.update(interactive=False), gr.update(interactive=True))
    idle = (gr.update(interactive=True), gr.update(interactive=False))

    conversation = _load(who, identifier)
    if conversation is None:
        yield (None, [], text, image_path,
               ui.notice("Choose a character and a thread first.", "warn"), *idle)
        return
    if not regenerate and not (text or "").strip() and not image_path:
        yield (None, _transcript(conversation), text, image_path,
               ui.notice("Write a message first.", "warn"), *idle)
        return

    store = _chats()
    try:
        character = _characters().load(who)
    except Exception as exc:
        yield (None, _transcript(conversation), text, image_path,
               ui.notice(ui.failure(exc), "error"), *idle)
        return

    attachment, attachment_name = "", ""
    if image_path:
        if not mc_llm_runtime.config().sees:
            yield (None, _transcript(conversation), text, image_path,
                   ui.notice("The model running has no vision projector, so the attached image "
                             "cannot be sent to it. Choose one under Models and Hardware, or "
                             "remove the image.", "error"), *idle)
            return
        try:
            attachment = ui.data_url(image_path) or ""
            # Path, not a string split on "/": a Windows temporary file arrives
            # with backslashes and the whole path would end up as the caption.
            attachment_name = Path(image_path).name
        except Exception as exc:
            yield (None, _transcript(conversation), text, image_path,
                   ui.notice(ui.failure(exc), "error"), *idle)
            return

    if regenerate:
        # Drop the reply being replaced, and everything after it, so the model
        # is asked the same question rather than a longer one.
        last = conversation.last_index(ASSISTANT)
        if last < 0:
            yield (None, _transcript(conversation), text, image_path,
                   ui.notice("There is no reply to regenerate.", "warn"), *idle)
            return
        conversation.delete_from(last)
    else:
        conversation.append(USER, (text or "").strip(), attachment, attachment_name)

    persona = _persona()
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
        messages=build(character, persona, conversation.messages,
                       context_size=_context_size(), reply_tokens=tokens),
        needs_vision=bool(attachment) or _has_image(conversation),
        temperature=_decimal(temperature, character.temperature, DEFAULT_TEMPERATURE),
        top_p=_decimal(top_p, character.top_p, DEFAULT_TOP_P),
        max_tokens=tokens,
        seed=asked if asked != RANDOM_SEED else resolved,
    )

    cancel = sessions.Cancellation()
    reply = conversation.append(ASSISTANT, "")
    rows = _transcript(conversation)
    yield cancel, rows, "", None, ui.notice("Starting…"), *busy

    streamed = ""
    try:
        for event in sessions.conversation(request, cancel):
            if event.kind == sessions.CHUNK:
                streamed += event.text
                reply.text = streamed
                yield cancel, _transcript(conversation), "", None, gr.update(), *busy
            elif event.kind == sessions.STATUS:
                yield (cancel, _transcript(conversation), "", None,
                       ui.notice(event.text), *busy)
            elif event.kind in (sessions.DONE, sessions.CANCELLED):
                reply.text = clean_reply(event.text or streamed, character, persona)
                if not reply.text.strip():
                    conversation.delete(len(conversation.messages) - 1)
                conversation.retitle()
                store.save(conversation)
                message = ("Stopped." if event.kind == sessions.CANCELLED else "Reply complete.")
                yield (cancel, _transcript(conversation), "", None,
                       ui.notice(message, "warn" if event.kind == sessions.CANCELLED else "info"),
                       *idle)
                return
            elif event.kind == sessions.FAILED:
                # The half-written reply goes; the user's turn stays, so the
                # message they typed is not lost to a server that would not start.
                conversation.delete(len(conversation.messages) - 1)
                store.save(conversation)
                yield (cancel, _transcript(conversation), "", None,
                       ui.notice(event.text, "error"), *idle)
                return
    except Exception as exc:
        yield (cancel, _transcript(conversation), "", None,
               ui.notice(ui.failure(exc), "error"), *idle)
        return

    store.save(conversation)
    yield cancel, _transcript(conversation), "", None, ui.notice("Reply complete."), *idle


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


def _has_image(conversation) -> bool:
    from prompt_master.chat.prompt import has_image

    return has_image(conversation.messages)


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


def _undo(who, identifier):
    """Remove the last exchange."""
    from prompt_master.chat.history import USER

    conversation = _load(who, identifier)
    if conversation is None or not conversation.messages:
        return [], ui.notice("Nothing to undo.", "warn")
    last_user = conversation.last_index(USER)
    conversation.delete_from(last_user if last_user >= 0 else len(conversation.messages) - 1)
    _chats().save(conversation)
    return _transcript(conversation), ui.notice("Undone.")


def _clear(who, identifier):
    conversation = _load(who, identifier)
    if conversation is None:
        return [], ui.notice("Choose a thread first.", "warn")
    conversation.messages.clear()
    _greet(conversation, who)
    _chats().save(conversation)
    return _transcript(conversation), ui.notice("Thread cleared.")


# --------------------------------------------------------------------------- #
# Characters and persona
# --------------------------------------------------------------------------- #


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


def _new_character():
    return "", "", "", "", ui.notice("Fill in a name and save to create the character.")


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
