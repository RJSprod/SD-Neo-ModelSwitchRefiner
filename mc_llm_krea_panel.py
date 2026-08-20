"""Krea 2: a prompt-authoring workspace, not a chat and not an image backend.

One task in, one finished Krea prompt out -- structurally MiniMax's shape
rather than Conversation's, and deliberately so. What makes it its own mode
rather than a variant of MiniMax is the references: up to four pictures whose
*order on screen* is part of the request. "Replace the face of the woman in
image 1 with the woman from image 2" is a sentence that means nothing at all if
image 1 and image 2 can trade places between the upload control and the prompt.

So the references are four numbered slots and not a multi-file upload. A Gradio
file list reorders itself when an entry is deleted and replaced, and the day it
does that is the day somebody's face swap comes back with the wrong face and
nothing in the interface explains why. A slot labelled "Image 1" is Image 1 for
as long as it holds a picture, and the panel refuses to run rather than close a
gap in the numbering behind the user's back.

Everything about how the prompt is actually written -- Krea's own expansion
instruction, the reference addendum, the captioner, the sampling, the cleaning
-- belongs to ``prompt_master.krea``, whose ``expansion.txt`` is vendored from
Krea's repository with its provenance in ``UPSTREAM_SOURCE.txt``. Nothing here
writes prompt text.

This mode generates no images and settles nothing about how they would be
generated: there is no sampler here, no CFG, no LoRA strength, no mask and no
negative prompt, because those belong to an image-generation integration and
this is the thing that writes what such an integration would be given.
"""

from __future__ import annotations

import logging
import time

import gradio as gr

import mc_llm_runtime
import mc_llm_sessions as sessions
import mc_llm_state
import mc_llm_ui as ui

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

NUMBERING_NOTE = (
    "References are numbered by the slot they are in: the first is **Image 1**, "
    "the second **Image 2**, and so on. Refer to them that way in your "
    "instruction — *“use the woman from image 1”*, *“keep the "
    "composition from image 1”*, *“use image 3 only as a style "
    "reference”*. Fill the slots in order; a gap is refused rather than "
    "closed up, because closing it would renumber the pictures you are "
    "describing. This numbering is LLM Studio's convention for talking to the "
    "prompt writer, not Krea syntax.")
"""What the numbers mean, said once, where the pictures are.

It is on the page rather than in documentation because the whole feature rests
on the user and the model agreeing about which picture is which, and an
agreement one side has never been told about is not one.
"""


def build() -> dict:
    """Assemble the panel. Returns the handles the shell needs."""
    from prompt_master.krea import enhancer

    cancellation = gr.State(None)

    with gr.Row(elem_id=ui.ident("krea"), elem_classes=ui.classes("workspace")):

        # -- left rail: this mode's own history ---------------------------- #
        with gr.Column(scale=1, min_width=200, elem_classes=ui.classes("rail")):
            gr.Markdown("### Sessions")
            history = gr.Dropdown(label="Saved prompts", choices=_history_choices(), value=None,
                                  elem_id=ui.ident("krea", "history"))
            with gr.Row():
                load = gr.Button("Load", size="sm")
                drop = gr.Button("Delete", size="sm", variant="stop")
            refresh = gr.Button("Refresh", size="sm")
            seed = gr.Number(label="Seed", value=-1, precision=0,
                             info="-1 draws a fresh seed for every prompt.",
                             elem_id=ui.ident("krea", "seed"))
            gr.Markdown("Krea keeps its own history, separate from Prompt Studio, "
                        "Conversation and MiniMax. The pictures are not saved with it — "
                        "their names and descriptions are.",
                        elem_classes=ui.classes("hint"))

        # -- centre: the written prompt, then what it was written from ----- #
        with gr.Column(scale=4, min_width=420, elem_classes=ui.classes("stage")):
            status = gr.HTML(ui.notice("Ready."), elem_id=ui.ident("krea", "status"))
            # Fixed height, for the reason MiniMax's output box is fixed: a
            # Gradio Textbox grows from `lines` towards `max_lines` as text
            # arrives, so a box that can grow walks Stop off the bottom of the
            # window at the exact moment somebody wants to press it.
            written = gr.Textbox(
                label="Krea 2 prompt", lines=14, max_lines=14, show_copy_button=True,
                elem_id=ui.ident("krea", "output"),
                elem_classes=ui.classes("output", "output-primary"))

            prompt = gr.Textbox(
                label="What you want", lines=4, max_lines=4,
                placeholder="An image idea, or an edit described against the references — "
                            "“replace the face of the woman in image 1 with the woman "
                            "from image 2, keeping image 1’s pose, outfit, lighting "
                            "and background”.",
                elem_id=ui.ident("krea", "prompt"))

            with gr.Row(elem_id=ui.ident("krea", "references")):
                images = [
                    gr.Image(label=f"Image {position}", type="filepath", height=140,
                             elem_id=ui.ident("krea", "image", str(position)))
                    for position in range(1, enhancer.MAX_REFERENCES + 1)]

            with gr.Row(elem_classes=ui.classes("actions")):
                generate = gr.Button(enhancer.BUTTON_LABEL, variant="primary",
                                     elem_id=ui.ident("krea", "generate"))
                stop = gr.Button("Stop", variant="stop", interactive=False,
                                 elem_id=ui.ident("krea", "stop"))
                clear = gr.Button("Clear")

            with gr.Accordion("Reference numbering and descriptions", open=False):
                gr.Markdown(NUMBERING_NOTE, elem_classes=ui.classes("hint"))
                captions = gr.Textbox(
                    label="What the model saw (used to write the prompt)",
                    lines=8, max_lines=8, show_copy_button=True, visible=False,
                    elem_id=ui.ident("krea", "captions"))

    # -- wiring ----------------------------------------------------------- #

    running = generate.click(
        fn=_generate, inputs=[prompt, seed] + images,
        outputs=[cancellation, written, captions, status, generate, stop],
        show_progress="minimal")
    running.then(fn=lambda: gr.update(choices=_history_choices()), outputs=[history])

    stop.click(fn=_cancel, inputs=[cancellation], outputs=[status, generate, stop],
               cancels=[running], queue=False)
    clear.click(fn=_clear, outputs=[prompt, written, captions, status] + images, queue=False)

    refresh.click(fn=lambda: gr.update(choices=_history_choices()), outputs=[history],
                  queue=False)
    load.click(fn=_load_session, inputs=[history],
               outputs=[prompt, written, captions, status], queue=False)
    drop.click(fn=_delete_session, inputs=[history], outputs=[history, status], queue=False)

    return {"status": status, "output": written, "stop": stop}


# --------------------------------------------------------------------------- #
# The references
# --------------------------------------------------------------------------- #


def references(paths) -> tuple[list, str]:
    """The filled slots as numbered references, or a sentence saying why not.

    The order handed back is the order the slots are in on screen, and the
    number on each reference is the number printed on its slot. Nothing here
    consults a filename, a timestamp, a temporary path or the contents of a
    picture, because every one of those is a way for Image 2 to quietly become
    Image 1.

    A gap -- slot 2 empty with slot 3 filled -- is refused. Closing it up would
    be the silent renumbering §8 forbids: the user has written "image 3" in
    their instruction and would get a prompt about a picture the writer is
    calling Image 2.
    """
    from prompt_master.krea.references import Reference

    slots = list(paths or [])
    filled = [(position, path) for position, path in enumerate(slots, start=1) if path]
    if not filled:
        return [], ""

    empty = [position for position in range(1, filled[-1][0]) if not slots[position - 1]]
    if empty:
        missing = ", ".join(f"Image {position}" for position in empty)
        return [], (f"{missing} is empty, but a later slot has a picture in it. "
                    "Fill the reference slots in order — moving the pictures up would "
                    "change the numbers you are describing them by.")

    return [Reference(ui_index=position, path=str(path)) for position, path in filled], ""


def _encoded(found) -> str:
    """Attach each reference's picture, as the data URL a vision model is sent.

    Done here rather than in the session so that a picture that cannot be read
    is reported against the slot it is in, while the run has not started and
    nothing holds the GPU.
    """
    for reference in found:
        try:
            reference.data_url = ui.data_url(reference.path) or ""
        except Exception as exc:
            # Named by slot, not by path. "Image 2 could not be read" is
            # something a user can act on; a temporary upload path is not, and
            # is somebody's home directory besides.
            return f"{reference.label} could not be read: {ui.failure(exc)}"
        if not reference.data_url:
            return f"{reference.label} could not be read."
    return ""


def _described(captions) -> str:
    """The captions as one numbered block, in the order they arrived."""
    return "\n\n".join(f"Image {position}: {caption}"
                       for position, caption in enumerate(captions, start=1))


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


def _generate(prompt, seed, *paths):
    """Stream one Krea prompt. Describe every reference first, in slot order."""
    from prompt_master.core.models import RANDOM_SEED, draw_seed

    busy = (gr.update(interactive=False), gr.update(interactive=True))
    idle = (gr.update(interactive=True), gr.update(interactive=False))
    hidden = gr.update(value="", visible=False)

    if not (prompt or "").strip():
        yield None, "", hidden, ui.notice("Describe the image you want first.", "warn"), *idle
        return

    found, complaint = references(paths)
    if complaint:
        yield None, "", hidden, ui.notice(complaint, "warn"), *idle
        return

    if found:
        # Refused before anything is started, and never worked around. A run
        # that quietly dropped the references and wrote a text-only prompt
        # would hand back a plausible paragraph about the wrong request (§8).
        if not mc_llm_runtime.config().sees:
            yield (None, "", hidden,
                   ui.notice("The model running has no vision projector, so the reference "
                             "images cannot be read. Choose a model with one in Setup, or "
                             "remove the references and write from text alone.", "error"), *idle)
            return
        unreadable = _encoded(found)
        if unreadable:
            yield None, "", hidden, ui.notice(unreadable, "error"), *idle
            return

    resolved = int(seed or RANDOM_SEED)
    if resolved == RANDOM_SEED:
        resolved = draw_seed()

    cancel = sessions.Cancellation()
    text, described = "", []
    yield cancel, "", hidden, ui.working("Starting…"), *busy

    try:
        for event in sessions.krea(prompt.strip(), found, resolved, cancel):
            if event.kind == sessions.CHUNK:
                text += event.text
                yield cancel, text, gr.update(), gr.update(), *busy
            elif event.kind == sessions.CAPTION:
                # The captions arrive in slot order, one per reference, which
                # is what lets this pair the first with Image 1 without either
                # side carrying an index around.
                described.append(event.text)
                yield (cancel, text, gr.update(value=_described(described), visible=True),
                       ui.working(f"Image {len(described)} described."), *busy)
            elif event.kind == sessions.STATUS:
                yield cancel, text, gr.update(), ui.working(event.text), *busy
            elif event.kind == sessions.DONE:
                text = event.text
                _remember(prompt, text, resolved, found, described)
                yield (cancel, text, gr.update(),
                       ui.notice(f"Complete · Seed: {resolved}"), *idle)
                return
            elif event.kind == sessions.CANCELLED:
                yield cancel, text, gr.update(), ui.notice("Cancelled.", "warn"), *idle
                return
            elif event.kind == sessions.FAILED:
                yield cancel, text, gr.update(), ui.notice(event.text, "error"), *idle
                return
    except Exception as exc:
        yield cancel, text, gr.update(), ui.notice(ui.failure(exc), "error"), *idle
        return

    yield cancel, text, gr.update(), ui.notice("Finished."), *idle


def _cancel(cancel):
    """Stop the run, and put the controls back.

    Whatever restores the controls has to be this handler: ``cancels=`` closes
    the generator where it stands, and a closed generator never reaches the
    yield that would have re-enabled Generate and greyed out Stop.
    """
    if cancel is not None:
        cancel.cancel()
    return (ui.notice("Stopped.", "warn"),
            gr.update(interactive=True), gr.update(interactive=False))


def _clear():
    """Empty the whole workspace: the request, the prompt, the pictures, the captions."""
    from prompt_master.krea import enhancer

    return ("", "", gr.update(value="", visible=False), ui.notice("Cleared."),
            *[None] * enhancer.MAX_REFERENCES)


def _remember(prompt, result, seed, found, captions) -> None:
    """Save the session -- names and descriptions, never the pictures (§11, §14)."""
    try:
        mc_llm_state.save_krea_session(mc_llm_state.KreaSession(
            prompt=(prompt or "").strip(), result=result, seed=int(seed),
            reference_names=[reference.name for reference in found],
            reference_captions=list(captions)))
    except Exception:
        logger.debug("Model Chain: could not save the Krea session", exc_info=True)


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #


def _history_choices() -> list[tuple[str, str]]:
    try:
        return [(session.label, session.identifier)
                for session in reversed(mc_llm_state.krea_sessions())]
    except Exception:
        return []


def _load_session(identifier):
    """Restore a saved prompt: the text, the result and what the pictures were.

    The image slots are left alone on purpose. The files are not saved and may
    not exist any more, so filling the slots would be a claim the panel cannot
    back; what comes back is the names and the descriptions, as information.
    Regenerating a reference-aware prompt means attaching the pictures again,
    and the status line says so rather than leaving it to be discovered.
    """
    if not identifier:
        return "", "", gr.update(), ui.notice("Choose a saved prompt first.", "warn")
    found = next((s for s in mc_llm_state.krea_sessions() if s.identifier == identifier), None)
    if found is None:
        return "", "", gr.update(), ui.notice("That prompt is no longer saved.", "warn")

    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(found.created))
    captions = list(found.reference_captions or [])
    names = list(found.reference_names or [])
    note = f"Loaded the prompt from {stamp} (seed {found.seed})."
    if names:
        listed = ", ".join(f"Image {position}: {name}"
                           for position, name in enumerate(names, start=1))
        note += (f" It was written from {listed}. The pictures are not saved with a "
                 "session — attach them again to write another prompt from them.")
    return (found.prompt, found.result,
            gr.update(value=_described(captions), visible=bool(captions)),
            ui.notice(note))


def _delete_session(identifier):
    if not identifier:
        return gr.update(), ui.notice("Choose a saved prompt first.", "warn")
    mc_llm_state.delete_krea_session(identifier)
    return gr.update(choices=_history_choices(), value=None), ui.notice("Deleted.")
