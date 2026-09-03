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

Creative Mode is the one thing here that decides anything about content, and it
decides it *before* the model is asked and without asking a model. The local
Creative Director in ``prompt_master.krea.director`` picks an art-direction
brief out of a vendored vocabulary using a seeded PRNG, that brief is appended
to the user turn under its own label, and the writer is then called exactly
once. Turning the slider up does not mean asking the model more times or asking
it more loudly; it means handing it a different brief.

The whole engine is shared with txt2img's Creative Mode -- the same library, the
same Director, the same settings file -- so a Creativity position and an axis
configuration mean the same thing in both places. What is *not* shared is any
notion of images: this mode still generates none, and Creative Mode adds no
image controls, no seed that means a diffusion seed, and no gallery.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import gradio as gr

import mc_creative_krea
import mc_creative_panel
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

CREATIVE_NOTE = (
    "**Natural** leaves the axis out of the brief entirely — the model decides as it "
    "would without Creative Mode, and a Natural axis has no row below. **Vary** lets "
    "the local director choose, and the Creativity slider decides whether the axis "
    "activates at all, how strongly it is expressed, and how hard recent choices are "
    "pushed away; exclude any treatments you never want it to pick. **Fixed** repeats "
    "one chosen value every roll.\n\n"
    "Your own words always win: type *oil painting of a car* and Medium stays oil "
    "painting however Medium is set. The direction is chosen here, in Python, from a "
    "vendored vocabulary — no model is asked what to vary, and each press is still "
    "exactly one Krea writer request.\n\n"
    "These settings are shared with txt2img's Creative Mode.")
"""The three modes, explained where they are set rather than in documentation.

Natural is the one that needs saying out loud: it means *no line at all* for
that axis, not a line telling the model to please itself. A brief that says
"Texture: your choice" has put texture in the model's foreground, which is the
opposite of leaving it alone.
"""


def build() -> dict:
    """Assemble the panel. Returns the handles the shell wires, and Creative Mode's.

    The Creative Mode controls are in the map without the shell reading them,
    because their *configuration* is a promise -- the range, the default, and
    that the axis table is built from the library rather than from a list here
    -- and a promise nothing can ask about is a promise nothing can hold you to.
    """
    from prompt_master.core.models import RANDOM_SEED
    from prompt_master.krea import enhancer, variation

    stored = mc_creative_krea.settings()
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
            seed = gr.Number(label="Seed", value=RANDOM_SEED, precision=0,
                             info=f"{RANDOM_SEED} draws a fresh seed for every prompt.",
                             elem_id=ui.ident("krea", "seed"))

            creative = gr.Checkbox(
                value=bool(stored["enabled"]), label="Creative Mode",
                info="direct the prompt locally before expanding it",
                elem_id=ui.ident("krea", "creative"))
            creativity = gr.Slider(
                label=variation.LABEL, minimum=variation.MINIMUM,
                maximum=variation.MAXIMUM, step=1, value=stored["creativity"],
                info=variation.HELP, visible=bool(stored["enabled"]),
                elem_id=ui.ident("krea", "creativity"))

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
                # type="pil" and not "filepath": Gradio's filepath preprocess
                # calls ``processing_utils.save_pil_to_cache`` with a ``name``
                # argument, and this WebUI replaces that function with an older
                # one that has no such parameter -- so every filepath image
                # input in the host raises ``TypeError`` before a handler is
                # ever reached. Asking for the picture itself skips that call.
                images = [
                    gr.Image(label=f"Image {position}", type="pil", height=140,
                             elem_id=ui.ident("krea", "image", str(position)))
                    for position in range(1, enhancer.MAX_REFERENCES + 1)]

            with gr.Row(elem_classes=ui.classes("actions")):
                generate = gr.Button(enhancer.BUTTON_LABEL, variant="primary",
                                     elem_id=ui.ident("krea", "generate"))
                stop = gr.Button("Stop", variant="stop", interactive=False,
                                 elem_id=ui.ident("krea", "stop"))
                clear = gr.Button("Clear")

            with gr.Accordion("Creative Controls", open=False,
                               visible=bool(stored["enabled"]),
                               elem_id=ui.ident("krea", "controls")) as controls:
                # The same panel txt2img draws, from the same module: one layout,
                # one set of handlers, one settings file. Two implementations of
                # a ten-axis editor would disagree within a release, and the
                # first thing they would disagree about is what a fresh install
                # does.
                panel = mc_creative_panel.build(
                    lambda *parts: ui.ident("krea", *parts), ui.notice, status,
                    creativity=creativity, stored=stored)
                axis_controls = list(panel.axis_controls) if panel is not None else []
                if panel is not None:
                    creative_seed, anti = panel.seed, panel.anti
                else:
                    # The library would not load, so there is no panel and no
                    # vocabulary to direct with -- but the generate handler's
                    # input list is fixed at build time and cannot have holes in
                    # it. These stand in, hold the stored values, and let the
                    # mode go on writing undirected prompts.
                    creative_seed = gr.Number(value=stored["seed"], precision=0,
                                              visible=False,
                                              elem_id=ui.ident("krea", "creative-seed"))
                    anti = gr.Checkbox(value=bool(stored["anti_repetition"]),
                                       visible=False, elem_id=ui.ident("krea", "anti"))

                gr.Markdown(CREATIVE_NOTE, elem_classes=ui.classes("hint"))

                recipe = gr.Textbox(
                    label="Last creative recipe", lines=10, max_lines=10,
                    interactive=False, show_copy_button=True, visible=False,
                    elem_id=ui.ident("krea", "recipe"))

            with gr.Accordion("Reference numbering and descriptions", open=False):
                gr.Markdown(NUMBERING_NOTE, elem_classes=ui.classes("hint"))
                captions = gr.Textbox(
                    label="What the model saw (used to write the prompt)",
                    lines=8, max_lines=8, show_copy_button=True, visible=False,
                    elem_id=ui.ident("krea", "captions"))

    # -- wiring ----------------------------------------------------------- #

    running = generate.click(
        fn=_generate,
        inputs=[prompt, seed, creative, creativity, creative_seed, anti] + list(axis_controls)
               + images,
        outputs=[cancellation, written, captions, recipe, status, generate, stop],
        show_progress="minimal")
    running.then(fn=lambda: gr.update(choices=_history_choices()), outputs=[history])

    stop.click(fn=_cancel, inputs=[cancellation], outputs=[status, generate, stop],
               cancels=[running], queue=False)
    clear.click(fn=_clear, outputs=[prompt, written, captions, status] + images, queue=False)

    creative.change(fn=_toggled, inputs=[creative], outputs=[creativity, controls, status],
                    queue=False)
    # Remembered on release rather than on every pixel of the drag: the value
    # is a preference, and a preferences file rewritten forty times as a slider
    # travels from 1 to 10 is forty writes for one decision.
    if panel is not None:
        # With the cost line, so the number beside the directions cannot be one
        # action behind the slider that changes it.
        creativity.release(
            fn=lambda value: (_remember_creativity(value),
                              gr.update(value=mc_creative_panel.describe_cost())),
            inputs=[creativity], outputs=[status, panel.cost], queue=False)
    else:
        creativity.release(fn=_remember_creativity, inputs=[creativity],
                           outputs=[status], queue=False)
    refresh.click(fn=lambda: gr.update(choices=_history_choices()), outputs=[history],
                  queue=False)
    load.click(fn=_load_session, inputs=[history],
               outputs=[prompt, written, captions, status], queue=False)
    drop.click(fn=_delete_session, inputs=[history], outputs=[history, status], queue=False)

    found = {"status": status, "output": written, "stop": stop, "creativity": creativity,
             "creative": creative, "controls": controls, "recipe": recipe,
             "axes": axis_controls}
    if panel is not None:
        found["panel"] = panel
        found.update({f"creative_{name}": component
                      for name, component in panel.components().items()})
    return found


# --------------------------------------------------------------------------- #
# The references
# --------------------------------------------------------------------------- #


def references(picked) -> tuple[list, str]:
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

    slots = list(picked or [])
    filled = [(position, found) for position, found in enumerate(slots, start=1)
              if _in_a_slot(found)]
    if not filled:
        return [], ""

    empty = [position for position in range(1, filled[-1][0])
             if not _in_a_slot(slots[position - 1])]
    if empty:
        missing = ", ".join(f"Image {position}" for position in empty)
        return [], (f"{missing} is empty, but a later slot has a picture in it. "
                    "Fill the reference slots in order — moving the pictures up would "
                    "change the numbers you are describing them by.")

    return [Reference(ui_index=position,
                      path=str(found) if _is_a_path(found) else "",
                      picture=None if _is_a_path(found) else found)
            for position, found in filled], ""


def _in_a_slot(found) -> bool:
    """Whether a reference slot has something in it.

    Not ``bool(found)``. A slot holds a decoded picture now, and a picture that
    happens to be entirely black is still a picture -- while an empty slot is
    ``None`` and an empty path is ``""``. Asking the truthiness of whatever
    arrived would one day be asking it of something that answers False for a
    reason of its own.
    """
    return found is not None and not (_is_a_path(found) and not str(found))


def _is_a_path(found) -> bool:
    return isinstance(found, (str, Path))


def _encoded(found) -> str:
    """Attach each reference's picture, as the data URL a vision model is sent.

    Done here rather than in the session so that a picture that cannot be read
    is reported against the slot it is in, while the run has not started and
    nothing holds the GPU.
    """
    for reference in found:
        try:
            reference.data_url = ui.data_url(reference.source) or ""
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


def _recipe_view(recipe) -> str:
    """The last recipe as something a person can read and argue with."""
    items = getattr(recipe, "items", ())
    notes = getattr(recipe, "notes", ())
    if not items:
        head = (f"No creative direction at Creativity {getattr(recipe, 'creativity', 0)}.\n"
                "Creativity 0 and 1 direct nothing by design; above that, add at least "
                "one direction.")
        return "\n".join([head, *notes]) if notes else head
    lines = [f"Creative seed: {recipe.creative_seed}   ·   writer seed: {recipe.llm_seed}"
             f"   ·   library {recipe.library_version}"]
    if getattr(recipe, "replayed", False):
        lines.append("Replayed from a recorded recipe: nothing was drawn.")
    if recipe.locked:
        lines.append("Locked by your prompt: " + ", ".join(recipe.locked))
    for note in notes:
        lines.append(f"Note: {note}")
    lines.append("")
    for item in items:
        lines.append(f"{item.label} [{item.source}] {item.variant_id} — {item.variant_label}")
    lines.append("")
    lines.append(recipe.brief)
    return "\n".join(lines)


def _toggled(enabled):
    """Show or hide the Creative controls, and remember the toggle."""
    mc_creative_krea.remember(**{mc_creative_krea.ENABLED: bool(enabled)})
    shown = gr.update(visible=bool(enabled))
    if enabled:
        told = ("Creative Mode is on. Each press directs the prompt locally, then asks "
                f"the writer once. {mc_creative_panel.active_note()}")
    else:
        told = "Creative Mode is off."
    return shown, shown, ui.notice(told)


def _remember_creativity(value):
    """Keep the slider's position, shared with txt2img's Creative Mode.

    One position for both surfaces, not two. The axes, the seed and the slider
    describe how this installation does art direction, and somebody who has
    spent five minutes configuring ten axes here should not have to do it again
    in txt2img.
    """
    from prompt_master.krea.variation import clamp

    mc_creative_krea.remember(**{mc_creative_krea.CREATIVITY: clamp(value)})
    stored = mc_creative_krea.settings()
    told = mc_creative_panel.describe_creativity(value, stored)
    return ui.notice(told, "warn" if "nothing to scale" in told else "info")


def _generate(prompt, seed, creative, creativity, creative_seed, anti, *rest):
    """Stream one Krea prompt: directed locally first, then written once.

    The tail of the arguments is the axis table followed by the reference slots,
    both variable in length, so they are split by the library's axis count
    rather than unpacked positionally.

    Literal commands are honoured here for the same reason they are honoured in
    txt2img: the prompt this tab produces is a prompt somebody pastes into a
    prompt box, and ``[[<lora:krea2_edit:1>]]`` has to survive the round trip.
    They are lifted out before the Director reads the source, and put back
    around the finished text -- so the writer never sees them and the box the
    user copies from already has them in it.
    """
    from prompt_master.core.models import RANDOM_SEED, draw_seed
    from prompt_master.krea import extra_networks, literals
    from prompt_master.krea.variation import clamp

    busy = (gr.update(interactive=False), gr.update(interactive=True))
    idle = (gr.update(interactive=True), gr.update(interactive=False))
    hidden = gr.update(value="", visible=False)
    keep = gr.update()

    axis_values, paths = _split_tail(rest)

    if not (prompt or "").strip():
        yield (None, "", hidden, keep,
               ui.notice("Describe the image you want first.", "warn"), *idle)
        return

    # Bracketed before the parse, so the panel protects a bare LoRA tag exactly
    # as the generation hook does. A preview that handed the writer a tag the
    # real run would have lifted out would be showing the user a prompt their
    # own pipeline cannot produce.
    parsed = literals.parse(extra_networks.protect(prompt))
    written_source = parsed.clean_text.strip()
    if not written_source:
        # Nothing transformable, and nothing to write from. The commands are
        # handed back exactly as typed rather than sent to a model that would
        # expand them into prose about a LoRA filename.
        yield (None, literals.restore("", parsed), hidden, keep,
               ui.notice("That prompt is nothing but literal commands, so there was "
                         "nothing for the writer to expand. They are returned "
                         "unchanged.", "warn"), *idle)
        return
    for warning in parsed.warnings:
        logger.warning("Model Chain: the Krea source prompt — %s", warning)

    found, complaint = references(paths)
    if complaint:
        yield None, "", hidden, keep, ui.notice(complaint, "warn"), *idle
        return

    if found:
        # Refused before anything is started, and never worked around. A run
        # that quietly dropped the references and wrote a text-only prompt
        # would hand back a plausible paragraph about the wrong request (§8).
        if not mc_llm_runtime.config().sees:
            yield (None, "", hidden, keep,
                   ui.notice("The model running has no vision projector, so the reference "
                             "images cannot be read. Choose a model with one in Setup, or "
                             "remove the references and write from text alone.", "error"), *idle)
            return
        unreadable = _encoded(found)
        if unreadable:
            yield None, "", hidden, keep, ui.notice(unreadable, "error"), *idle
            return

    resolved = int(seed or RANDOM_SEED)
    if resolved == RANDOM_SEED:
        resolved = draw_seed()
    position = clamp(creativity)

    # The Director runs here, before anything is streamed and before the GPU is
    # asked for: it is ordinary Python over a vendored vocabulary, it cannot
    # fail slowly, and doing it first means the recipe is on screen while the
    # model is still being waited for.
    recipe, complaint = _direct(written_source, creative, position, creative_seed, anti,
                                axis_values)
    if complaint:
        yield None, "", hidden, keep, ui.notice(complaint, "error"), *idle
        return
    direction = getattr(recipe, "brief", "") if recipe is not None else ""
    shown = (gr.update(value=_recipe_view(recipe), visible=True)
             if recipe is not None else gr.update(value="", visible=False))

    cancel = sessions.Cancellation()
    text, described = "", []
    yield cancel, "", hidden, shown, ui.working("Starting…"), *busy

    try:
        for event in sessions.krea(written_source, found, resolved, cancel, position,
                                   direction):
            if event.kind == sessions.CHUNK:
                text += event.text
                yield cancel, text, keep, keep, keep, *busy
            elif event.kind == sessions.CAPTION:
                # The captions arrive in slot order, one per reference, which
                # is what lets this pair the first with Image 1 without either
                # side carrying an index around.
                described.append(event.text)
                yield (cancel, text, gr.update(value=_described(described), visible=True),
                       keep, ui.working(f"Image {len(described)} described."), *busy)
            elif event.kind == sessions.STATUS:
                yield cancel, text, keep, keep, ui.working(event.text), *busy
            elif event.kind == sessions.DONE:
                # Restored once, here, on the finished text -- the same rule the
                # txt2img hook follows and for the same reason: one assembly
                # point per prompt is how "exactly once" stays true across every
                # path out of this loop.
                text = literals.restore(event.text, parsed)
                _remember(prompt, text, resolved, found, described, position, recipe)
                if recipe is not None and anti:
                    mc_creative_krea.note_roll(recipe)
                yield (cancel, text, keep, keep,
                       ui.notice(_completed(resolved, position, recipe)), *idle)
                return
            elif event.kind == sessions.CANCELLED:
                yield cancel, text, keep, keep, ui.notice("Cancelled.", "warn"), *idle
                return
            elif event.kind == sessions.FAILED:
                yield cancel, text, keep, keep, ui.notice(event.text, "error"), *idle
                return
    except Exception as exc:
        yield cancel, text, keep, keep, ui.notice(ui.failure(exc), "error"), *idle
        return

    yield cancel, text, keep, keep, ui.notice("Finished."), *idle


def _split_tail(rest) -> tuple[tuple, tuple]:
    """The variable argument tail, as ``(axis controls, reference slots)``.

    Two variable-length groups arrive as one flat tuple because Gradio has no
    other shape for them. The split is by the library's axis count -- three
    controls per axis, mode, fixed value and exclusions -- with the references
    taking whatever is left, so a library that grows an axis moves the boundary
    without this being edited.
    """
    try:
        from prompt_master.krea import library as library_module

        width = len(library_module.library().axis_keys) * 3
    except Exception:
        width = 0
    return tuple(rest[:width]), tuple(rest[width:])


def _direct(prompt, creative, position, creative_seed, anti, axis_values):
    """The recipe for this press, or ``(None, "")`` when Creative Mode is off.

    A failure here is fatal to the run and deliberately so: somebody who turned
    Creative Mode on and pressed Generate asked for art direction, and quietly
    writing an undirected prompt instead would hand back a plausible paragraph
    that answers a question they did not ask.
    """
    if not creative:
        return None, ""
    from prompt_master.krea import director

    modes, fixed, excluded = mc_creative_panel.axes_from(axis_values)
    settings = {key: director.AxisSetting(mode=mode, fixed_id=fixed.get(key),
                                          excluded_ids=frozenset(excluded.get(key) or ()))
                for key, mode in modes.items()}
    try:
        return director.roll(
            source=prompt.strip(), creativity=position,
            creative_seed=mc_creative_krea._seed(creative_seed),
            settings=settings or None,
            history=mc_creative_krea.recent_ids() if anti else ()), ""
    except Exception as exc:
        logger.debug("Model Chain: the Creative Director failed", exc_info=True)
        return None, f"The creativity library could not be used: {exc}"


def _completed(seed, position, recipe) -> str:
    """The finished line: what the writer ran at, and what directed it."""
    line = f"Complete · Seed: {seed} · Creativity: {position}"
    if recipe is None:
        return line
    return (f"{line} · Creative seed: {recipe.creative_seed} · "
            f"{len(recipe.items)} {'axis' if len(recipe.items) == 1 else 'axes'} directed")


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


def _remember(prompt, result, seed, found, captions, creativity, recipe=None) -> None:
    """Save the session -- names and descriptions, never the pictures (§11, §14).

    Creativity, the Creative seed and the recipe ids are stored beside the
    writer's seed because together they answer the only interesting question
    about a saved prompt: what would have to be set to write this again. The
    recipe goes in as ids rather than as its rendered sentences, because ids are
    stable across library versions by contract and a paragraph of English is not.
    """
    try:
        mc_llm_state.save_krea_session(mc_llm_state.KreaSession(
            prompt=(prompt or "").strip(), result=result, seed=int(seed),
            creativity=int(creativity),
            creative_seed=int(getattr(recipe, "creative_seed", -1)),
            recipe=str(getattr(recipe, "compact", "")),
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
    note = (f"Loaded the prompt from {stamp} (seed {found.seed}, "
            f"creativity {found.creativity}).")
    if found.recipe:
        note += (f" It was directed by creative seed {found.creative_seed}: "
                 f"{found.recipe}.")
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
