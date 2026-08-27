"""MiniMax H3: a dedicated prompt enhancer, not a chat (section 4.4).

Kept a separate mode with its own input, its own output and its own history,
because that is what section 4.4 asks for and because the workflow really is
different: one box of rough prompt goes in, one finished H3 prompt comes out,
and when a reference image is attached there is a captioning step in between
whose result is worth seeing on its own. Presenting any of that as an
assistant's chat turn would lose the caption, lose the variant, and lose the
reason the enhancer exists.

The instructions, the sampling, the token budgets, the caption pass and the
output cleaning are all the vendored ``prompt_master.minimax`` package's, which
in turn carries WanGP's own prompt text with its provenance recorded beside it
in ``UPSTREAM_SOURCE.txt``. Nothing here writes prompt text.
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


def build() -> dict:
    """Assemble the panel. Returns the handles the shell needs."""
    from prompt_master.core.models import RANDOM_SEED
    from prompt_master.minimax import enhancer

    prefs = mc_llm_state.preferences()
    cancellation = gr.State(None)

    with gr.Row(elem_id=ui.ident("minimax"), elem_classes=ui.classes("workspace")):

        # -- left rail: this mode's own history ---------------------------- #
        with gr.Column(scale=1, min_width=200, elem_classes=ui.classes("rail")):
            gr.Markdown("### Sessions")
            history = gr.Dropdown(label="Saved prompts", choices=_history_choices(), value=None,
                                  elem_id=ui.ident("minimax", "history"))
            with gr.Row():
                load = gr.Button("Load", size="sm")
                drop = gr.Button("Delete", size="sm", variant="stop")
            refresh = gr.Button("Refresh", size="sm")
            gr.Markdown("MiniMax keeps its own history, separate from Prompt Studio and "
                        "Conversation.", elem_classes=ui.classes("hint"))

        # -- centre: the written prompt, then what it was written from ----- #
        with gr.Column(scale=4, min_width=420, elem_classes=ui.classes("stage")):
            status = gr.HTML(ui.notice("Ready."), elem_id=ui.ident("minimax", "status"))
            # max_lines == lines, and that equality is the whole point: a
            # Gradio Textbox grows from `lines` towards `max_lines` as text
            # arrives, so an output box that could reach 44 lines grew by
            # hundreds of pixels *while a generation was streaming into it* --
            # and everything below it, including Stop, was pushed off the
            # bottom of the window at the one moment somebody wants to press
            # it. A box that cannot change size cannot move the button, and a
            # prompt longer than the box is read by scrolling inside it.
            written = gr.Textbox(
                label="H3 prompt", lines=18, max_lines=18, show_copy_button=True,
                elem_id=ui.ident("minimax", "output"),
                elem_classes=ui.classes("output", "output-primary"))
            caption = gr.Textbox(
                label="Image description (used to write the prompt)", lines=3, visible=False,
                show_copy_button=True, elem_id=ui.ident("minimax", "caption"))

            with gr.Row(elem_classes=ui.classes("composer")):
                with gr.Column(scale=4):
                    prompt = gr.Textbox(
                        label="Your prompt", lines=4, max_lines=4,
                        placeholder="A rough description of the shot. The enhancer writes the "
                                    "structured H3 prompt from it.",
                        elem_id=ui.ident("minimax", "prompt"))
                with gr.Column(scale=1, min_width=160):
                    # type="pil" and not "filepath": Gradio's filepath
                    # preprocess calls ``processing_utils.save_pil_to_cache``
                    # with a ``name`` argument, and this WebUI replaces that
                    # function with an older one that has no such parameter --
                    # so every filepath image input in the host raises
                    # ``TypeError`` before a handler is ever reached. Asking
                    # for the picture itself skips that call entirely, and is
                    # the better answer anyway: one decode instead of a second
                    # copy of somebody's photograph written into a cache we
                    # then read back.
                    image = gr.Image(label="Reference frame", type="pil", height=140,
                                     elem_id=ui.ident("minimax", "image"))

            with gr.Row(elem_classes=ui.classes("actions")):
                enhance = gr.Button(enhancer.BUTTON_LABEL, variant="primary",
                                    elem_id=ui.ident("minimax", "enhance"))
                stop = gr.Button("Stop", variant="stop", interactive=False,
                                 elem_id=ui.ident("minimax", "stop"))
                clear = gr.Button("Clear")

        # -- right: variant, seed, and MiniMax's own structure guide -------- #
        with gr.Column(scale=2, min_width=250, elem_classes=ui.classes("inspector")):
            variant = gr.Radio(
                label="Variant", choices=ui.choices(enhancer.VARIANTS),
                value=prefs.get("minimax_variant") or enhancer.FL2VA,
                elem_id=ui.ident("minimax", "variant"))
            seed = gr.Number(label="Seed", value=RANDOM_SEED, precision=0,
                             info=f"{RANDOM_SEED} draws a fresh seed for every prompt.")
            with gr.Accordion("What an H3 prompt is made of", open=False):
                structure = gr.Markdown(enhancer.infos(prefs.get("minimax_variant")
                                                       or enhancer.FL2VA))

    # -- wiring ----------------------------------------------------------- #

    variant.change(fn=_structure, inputs=[variant], outputs=[structure], queue=False)

    running = enhance.click(
        fn=_enhance, inputs=[prompt, variant, image, seed],
        outputs=[cancellation, written, caption, status, enhance, stop],
        show_progress="minimal")
    running.then(fn=lambda: gr.update(choices=_history_choices()), outputs=[history])

    stop.click(fn=_cancel, inputs=[cancellation], outputs=[status, enhance, stop],
               cancels=[running], queue=False)
    clear.click(fn=lambda: ("", "", gr.update(value="", visible=False), ui.notice("Cleared.")),
                outputs=[prompt, written, caption, status], queue=False)

    refresh.click(fn=lambda: gr.update(choices=_history_choices()), outputs=[history],
                  queue=False)
    load.click(fn=_load_session, inputs=[history],
               outputs=[prompt, written, caption, variant, status], queue=False)
    drop.click(fn=_delete_session, inputs=[history], outputs=[history, status], queue=False)

    return {"status": status, "output": written, "stop": stop}


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


def _enhance(prompt, variant, picture, seed):
    """Stream one H3 prompt. Caption first when there is a picture."""
    from prompt_master.core.models import RANDOM_SEED, draw_seed

    busy = (gr.update(interactive=False), gr.update(interactive=True))
    idle = (gr.update(interactive=True), gr.update(interactive=False))
    hidden = gr.update(value="", visible=False)

    if not (prompt or "").strip():
        yield None, "", hidden, ui.notice("Write a prompt to enhance first.", "warn"), *idle
        return

    attachment = None
    # ``is None`` and not falsiness: the slot holds a decoded picture now, and
    # asking the truthiness of an image is asking a question its class is free
    # to answer for reasons of its own.
    if picture is not None:
        if not mc_llm_runtime.config().sees:
            yield (None, "", hidden,
                   ui.notice("The model running has no vision projector, so the reference frame "
                             "cannot be sent to it. Choose one in Setup, or "
                             "remove the image.", "error"), *idle)
            return
        try:
            attachment = ui.data_url(picture)
        except Exception as exc:
            yield None, "", hidden, ui.notice(ui.failure(exc), "error"), *idle
            return

    resolved = int(seed or RANDOM_SEED)
    if resolved == RANDOM_SEED:
        resolved = draw_seed()

    cancel = sessions.Cancellation()
    text, described = "", ""
    yield cancel, "", hidden, ui.working("Starting…"), *busy

    try:
        for event in sessions.minimax(prompt.strip(), variant, attachment, resolved, cancel):
            if event.kind == sessions.CHUNK:
                text += event.text
                yield cancel, text, gr.update(), gr.update(), *busy
            elif event.kind == sessions.CAPTION:
                described = event.text
                yield (cancel, text, gr.update(value=described, visible=True),
                       ui.working("Image described."), *busy)
            elif event.kind == sessions.STATUS:
                yield cancel, text, gr.update(), ui.working(event.text), *busy
            elif event.kind == sessions.DONE:
                text = event.text
                _remember(prompt, variant, described, text, resolved, picture)
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

    The buttons are the point. ``cancels=`` closes the generator where it
    stands, which is what makes a stop immediate -- and a generator that is
    closed never reaches the yield that would have re-enabled Send and greyed
    out Stop. So the run stopped, the partial text stayed, and the panel was
    left permanently busy with no way to ask for anything else. Whatever
    restores those controls has to be *this* handler, because it is the only
    one that still runs.
    """
    if cancel is not None:
        cancel.cancel()
    return (ui.notice("Stopped.", "warn"),
            gr.update(interactive=True), gr.update(interactive=False))


def _structure(variant):
    """MiniMax's own guide to the chosen variant's format.

    The same text WanGP shows beside its prompt box, behind an accordion rather
    than on the page: it is read once while learning the format and is in the
    way every time after that.
    """
    from prompt_master.minimax import enhancer

    return enhancer.infos(variant or enhancer.FL2VA)


def _remember(prompt, variant, caption, result, seed, picture) -> None:
    try:
        mc_llm_state.save_minimax_session(mc_llm_state.MinimaxSession(
            variant=variant, prompt=(prompt or "").strip(), caption=caption or "",
            result=result, seed=int(seed),
            image_name=ui.picked_name(picture)))
        mc_llm_state.remember(minimax_variant=variant)
    except Exception:
        logger.debug("Model Chain: could not save the MiniMax session", exc_info=True)


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #


def _history_choices() -> list[tuple[str, str]]:
    try:
        return [(session.label, session.identifier)
                for session in reversed(mc_llm_state.minimax_sessions())]
    except Exception:
        return []


def _load_session(identifier):
    if not identifier:
        return ("", "", gr.update(), gr.update(),
                ui.notice("Choose a saved prompt first.", "warn"))
    found = next((s for s in mc_llm_state.minimax_sessions() if s.identifier == identifier), None)
    if found is None:
        return ("", "", gr.update(), gr.update(),
                ui.notice("That prompt is no longer saved.", "warn"))
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(found.created))
    return (found.prompt, found.result,
            gr.update(value=found.caption, visible=bool(found.caption)),
            gr.update(value=found.variant),
            ui.notice(f"Loaded the prompt from {stamp} (seed {found.seed})."))


def _delete_session(identifier):
    if not identifier:
        return gr.update(), ui.notice("Choose a saved prompt first.", "warn")
    mc_llm_state.delete_minimax_session(identifier)
    return gr.update(choices=_history_choices(), value=None), ui.notice("Deleted.")
