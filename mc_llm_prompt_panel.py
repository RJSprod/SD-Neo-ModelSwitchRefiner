"""Prompt Studio: the LTX video-prompt workspace (section 4.2).

This is not a chat window and is not built like one. The positive prompt is the
product, so it gets the largest surface; the negative is beside it because it is
a second output rather than an afterthought; the intent composer sits under
them; and the thirty-odd shot controls live in a column that can be collapsed
out of the way. Section 4.2's instruction is explicit -- the positive prompt,
the negative prompt, the status and the structured generation information must
not be forced into one conversational response blob -- and keeping them as four
separate components is how that is honoured rather than described.

Every control the standalone application had is here, reading its options from
``prompt_engine.options`` so the lists cannot drift from the engine that
consumes them. What has changed is only the arrangement: section 4.5 asks for
output over controls, and the Qt window put a dense control grid beside a
smaller output pane.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import gradio as gr

import mc_llm_sessions as sessions
import mc_llm_state
import mc_llm_ui as ui

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

DIMENSIONS = [
    ("704 × 1216 (portrait)", "704x1216"),
    ("1216 × 704 (landscape)", "1216x704"),
    ("768 × 768 (square)", "768x768"),
    ("1920 × 1080", "1920x1080"),
    ("1080 × 1920", "1080x1920"),
]
"""The standalone application's own list, unchanged -- these are LTX output
sizes rather than arbitrary numbers, and inventing more would invite sizes the
model was not trained to write for."""


def build() -> dict:
    """Assemble the panel. Returns the handles the shell needs."""
    from prompt_master.core.models import RANDOM_SEED
    from prompt_master.prompt_engine import motion, options as opt

    defaults = opt.DEFAULTS
    stored = mc_llm_state.preferences().get("prompt_defaults") or {}

    def initial(key, fallback=None):
        return stored.get(key, defaults.get(key, fallback))

    controls: dict = {}
    cancellation = gr.State(None)

    with gr.Row(elem_id=ui.ident("prompt"), elem_classes=ui.classes("workspace")):

        # -- left rail: prompt sessions (section 4.5, item 2) -------------- #
        with gr.Column(scale=1, min_width=200, elem_classes=ui.classes("rail")):
            gr.Markdown("### Sessions")
            history = gr.Dropdown(
                label="Saved generations", choices=_history_choices(), value=None,
                interactive=True, elem_id=ui.ident("prompt", "history"))
            with gr.Row():
                load = gr.Button("Load", size="sm")
                drop = gr.Button("Delete", size="sm", variant="stop")
            refresh = gr.Button("Refresh", size="sm")
            gr.Markdown(
                "Saved automatically after every successful generation, with the "
                "controls that produced it.", elem_classes=ui.classes("hint"))

        # -- centre: the output, then the composer ------------------------ #
        with gr.Column(scale=4, min_width=420, elem_classes=ui.classes("stage")):
            status = gr.HTML(ui.notice("Ready."), elem_id=ui.ident("prompt", "status"))

            with gr.Row(elem_classes=ui.classes("outputs")):
                with gr.Column(scale=3):
                    positive = gr.Textbox(
                        label="Positive prompt", lines=16, max_lines=40, show_copy_button=True,
                        elem_id=ui.ident("prompt", "positive"),
                        elem_classes=ui.classes("output", "output-primary"))
                with gr.Column(scale=2):
                    negative = gr.Textbox(
                        label="Negative prompt", lines=16, max_lines=40, show_copy_button=True,
                        elem_id=ui.ident("prompt", "negative"),
                        elem_classes=ui.classes("output"))

            with gr.Row():
                saved = gr.File(label="Saved prompts", visible=False,
                                elem_id=ui.ident("prompt", "file"))

            with gr.Row(elem_classes=ui.classes("composer")):
                with gr.Column(scale=4):
                    intent = gr.Textbox(
                        label="Video intent", lines=4, max_lines=12,
                        placeholder="What happens in the shot. Quoted lines become spoken dialogue.",
                        elem_id=ui.ident("prompt", "intent"))
                with gr.Column(scale=1, min_width=160):
                    image = gr.Image(
                        label="Start frame (I2V)", type="filepath", height=140,
                        elem_id=ui.ident("prompt", "image"))

            with gr.Row(elem_classes=ui.classes("actions")):
                generate = gr.Button("Generate prompt", variant="primary",
                                     elem_id=ui.ident("prompt", "generate"))
                stop = gr.Button("Stop", variant="stop", interactive=False,
                                 elem_id=ui.ident("prompt", "stop"))
                save = gr.Button("Save to file")
                clear = gr.Button("Clear")

        # -- right: the shot controls (section 4.5, items 4 and 5) --------- #
        with gr.Column(scale=2, min_width=260, elem_classes=ui.classes("inspector")):
            with gr.Accordion("Shot", open=True):
                controls["video_mode"] = gr.Radio(
                    label="Video mode", choices=ui.choices(opt.VIDEO_MODES),
                    value=initial("video_mode"), elem_id=ui.ident("prompt", "mode"))
                controls["seconds"] = gr.Slider(
                    label="Duration (seconds)", minimum=1, maximum=60, step=0.5,
                    value=initial("seconds", 12.0))
                controls["fps"] = gr.Slider(label="FPS", minimum=8, maximum=60, step=1,
                                            value=initial("fps", 24))
                controls["dimensions"] = gr.Dropdown(
                    label="Dimensions", choices=DIMENSIONS,
                    value=initial("dimensions",
                                  f"{defaults['output_width']}x{defaults['output_height']}"))
                controls["seed"] = gr.Number(
                    label="Seed", value=initial("seed", 7), precision=0,
                    info=f"{RANDOM_SEED} draws a new seed for every generation.")

            with gr.Accordion("Look", open=False):
                controls["style"] = gr.Dropdown(
                    label="Visual style", choices=ui.grouped_choices(opt.STYLES_GROUPED),
                    value=initial("style"))
                controls["motion"] = gr.Dropdown(
                    label="Motion", choices=ui.choices(motion.OPTIONS),
                    value=initial("motion", motion.DEFAULT))
                controls["camera"] = gr.Dropdown(
                    label="Camera", choices=ui.choices(opt.CAMERAS), value=initial("camera"))
                controls["transition"] = gr.Dropdown(
                    label="Transition", choices=ui.choices(opt.TRANSITIONS),
                    value=initial("transition"))
                controls["pov"] = gr.Dropdown(
                    label="First person", choices=ui.choices(opt.POV), value=initial("pov"))
                controls["wardrobe"] = gr.Dropdown(
                    label="Wardrobe", choices=ui.choices(opt.WARDROBE), value=initial("wardrobe"))
                controls["undress"] = gr.Checkbox(label="Undress sequence",
                                                  value=initial("undress", False))

            with gr.Accordion("Voice and music", open=False):
                controls["accent"] = gr.Dropdown(
                    label="Accent", choices=ui.choices(opt.ACCENTS), value=initial("accent"))
                controls["accent_strength"] = gr.Dropdown(
                    label="Accent strength", choices=ui.choices(opt.ACCENT_STRENGTHS),
                    value=initial("accent_strength"))
                controls["dialogue"] = gr.Slider(
                    label="Dialogue / talk (%)", minimum=0, maximum=100, step=1,
                    value=initial("dialogue", 20))
                controls["music"] = gr.Dropdown(
                    label="Music", choices=ui.choices(opt.MUSIC), value=initial("music"))
                controls["music_bg"] = gr.Checkbox(label="Music plays low under the scene",
                                                   value=initial("music_bg", False))
                controls["speech"] = gr.Slider(
                    label="Extra speech", minimum=1, maximum=10, step=1,
                    value=initial("speech", 1))
                speech_note = gr.Markdown(_speech_note(initial("speech", 1),
                                                       initial("dialogue", 20)),
                                          elem_classes=ui.classes("hint"))

            with gr.Accordion("Wording", open=False):
                controls["fmt"] = gr.Dropdown(
                    label="Output format", choices=ui.choices(opt.OUTPUT_FORMATS),
                    value=initial("fmt"))
                controls["smart_negative"] = gr.Checkbox(
                    label="Smart negative — a second pass over the finished script",
                    value=initial("smart_negative", False))
                controls["lexicon"] = gr.Textbox(
                    label="Lexicon", lines=3,
                    placeholder="Name = description, one per line. Only names present in the "
                                "intent are used.")
                controls["negative_extra"] = gr.Textbox(
                    label="Extra negative terms", lines=2,
                    placeholder="Extra terms to keep out of the shot, comma separated.")

    # -- wiring ----------------------------------------------------------- #

    # One list, used for the inputs, for what is persisted, and for what a
    # loaded session restores -- so a control cannot be added to the panel and
    # forgotten by the history.
    control_inputs = [controls[name] for name in _ORDER]

    for control in (controls["speech"], controls["dialogue"]):
        control.change(fn=_speech_note, inputs=[controls["speech"], controls["dialogue"]],
                       outputs=[speech_note], show_progress="hidden")

    running = generate.click(
        fn=_generate, inputs=[intent, image] + control_inputs,
        outputs=[cancellation, positive, negative, status, generate, stop],
        show_progress="minimal",
    )
    running.then(fn=lambda: gr.update(choices=_history_choices()), outputs=[history])

    stop.click(fn=_cancel, inputs=[cancellation], outputs=[status], cancels=[running],
               queue=False)

    save.click(fn=_save_file, inputs=[positive, negative], outputs=[saved])
    clear.click(fn=lambda: ("", "", None, ui.notice("Cleared.")),
                outputs=[intent, positive, image, status], queue=False)

    refresh.click(fn=lambda: gr.update(choices=_history_choices()), outputs=[history],
                  queue=False)
    load.click(fn=_load_session, inputs=[history],
               outputs=[intent, positive, negative, status] + control_inputs, queue=False)
    drop.click(fn=_delete_session, inputs=[history], outputs=[history, status], queue=False)

    return {"status": status, "positive": positive, "negative": negative}


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


def _request(intent, image_path, values: dict):
    """A ``PromptRequest`` from the panel, with the seed resolved.

    Resolved here for the same reason the standalone app resolved it here: a
    request carrying ``-1`` would seed the engine's casting and llama.cpp's
    sampler with two different numbers, and the generation would not be
    reproducible from the seed it reported.
    """
    from prompt_master.core.models import RANDOM_SEED, PromptRequest, draw_seed

    width, height = (int(part) for part in str(values["dimensions"]).split("x"))
    seed = int(values["seed"] or 0)
    return PromptRequest(
        intent=(intent or "").strip(),
        image_data_url=ui.data_url(image_path),
        image_name=Path(image_path).name if image_path else "",
        video_mode=values["video_mode"],
        seconds=float(values["seconds"]),
        fps=int(values["fps"]),
        style=values["style"],
        motion=values["motion"],
        speech=int(values["speech"]),
        camera=values["camera"],
        transition=values["transition"],
        pov=values["pov"],
        accent=values["accent"],
        accent_strength=values["accent_strength"],
        dialogue=int(values["dialogue"]),
        music=values["music"],
        music_bg=bool(values["music_bg"]),
        wardrobe=values["wardrobe"],
        undress=bool(values["undress"]),
        lexicon=values["lexicon"] or "",
        fmt=values["fmt"],
        negative_extra=values["negative_extra"] or "",
        seed=draw_seed() if seed == RANDOM_SEED else seed,
        smart_negative=bool(values["smart_negative"]),
        output_width=width,
        output_height=height,
    )


_ORDER = ["video_mode", "seconds", "fps", "dimensions", "seed", "style", "motion", "camera",
          "transition", "pov", "wardrobe", "undress", "accent", "accent_strength", "dialogue",
          "music", "music_bg", "speech", "fmt", "smart_negative", "lexicon", "negative_extra"]


def _generate(intent, image_path, *values):
    """Stream one generation. A Gradio generator, so every yield is a repaint."""
    settings = dict(zip(_ORDER, values))
    busy = (gr.update(interactive=False), gr.update(interactive=True))
    idle = (gr.update(interactive=True), gr.update(interactive=False))

    if not (intent or "").strip():
        yield None, "", "", ui.notice("Enter a video intent first.", "warn"), *idle
        return

    if settings["video_mode"] == "i2v" and not image_path:
        yield (None, "", "",
               ui.notice("Image to video needs an attached image. Attach one, or switch to "
                         "text to video.", "warn"), *idle)
        return

    try:
        request = _request(intent, image_path, settings)
    except Exception as exc:
        yield None, "", "", ui.notice(ui.failure(exc), "error"), *idle
        return

    cancel = sessions.Cancellation()
    positive_text, negative_text = "", ""
    yield cancel, "", "", ui.working("Starting…"), *busy

    try:
        for event in sessions.prompt_studio(request, cancel):
            if event.kind == sessions.CHUNK:
                positive_text += event.text
                yield cancel, positive_text, negative_text, gr.update(), *busy
            elif event.kind == sessions.POSITIVE:
                positive_text = event.text
                yield cancel, positive_text, negative_text, gr.update(), *busy
            elif event.kind == sessions.NEGATIVE:
                negative_text = event.text
                yield cancel, positive_text, negative_text, gr.update(), *busy
            elif event.kind == sessions.STATUS:
                yield cancel, positive_text, negative_text, ui.working(event.text), *busy
            elif event.kind == sessions.DONE:
                _remember(settings, request, positive_text, negative_text)
                yield cancel, positive_text, negative_text, ui.notice(event.text), *idle
                return
            elif event.kind == sessions.CANCELLED:
                yield (cancel, positive_text, negative_text,
                       ui.notice(event.text or "Cancelled.", "warn"), *idle)
                return
            elif event.kind == sessions.FAILED:
                yield (cancel, positive_text, negative_text,
                       ui.notice(event.text, "error"), *idle)
                return
    except Exception as exc:
        yield cancel, positive_text, negative_text, ui.notice(ui.failure(exc), "error"), *idle
        return

    yield cancel, positive_text, negative_text, ui.notice("Finished."), *idle


def _cancel(cancel):
    """Stop the run. Cheap, queue-free, and safe to press twice."""
    if cancel is not None:
        cancel.cancel()
    return ui.working("Stopping…", "warn")


def _remember(settings: dict, request, positive: str, negative: str) -> None:
    """Persist the generation and the controls that produced it (section 16)."""
    try:
        mc_llm_state.save_prompt_session(mc_llm_state.PromptSession(
            title=request.intent[:60], intent=request.intent, positive=positive,
            negative=negative, seed=int(request.seed), image_name=request.image_name,
            controls=dict(settings)))
        mc_llm_state.remember(prompt_defaults=dict(settings))
    except Exception:
        logger.debug("Model Chain: could not save the Prompt Studio session", exc_info=True)


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #


def _history_choices() -> list[tuple[str, str]]:
    try:
        return [(session.label, session.identifier)
                for session in reversed(mc_llm_state.prompt_sessions())]
    except Exception:
        return []


def _load_session(identifier):
    """Put a saved generation back on screen, controls and all."""
    blanks = [gr.update() for _ in _ORDER]
    if not identifier:
        return ["", "", "", ui.notice("Choose a saved generation first.", "warn")] + blanks

    found = next((s for s in mc_llm_state.prompt_sessions() if s.identifier == identifier), None)
    if found is None:
        return ["", "", "", ui.notice("That generation is no longer saved.", "warn")] + blanks

    controls = found.controls or {}
    restored = [gr.update(value=controls[name]) if name in controls else gr.update()
                for name in _ORDER]
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(found.created))
    return ([found.intent, found.positive, found.negative,
             ui.notice(f"Loaded the generation from {stamp} (seed {found.seed}).")] + restored)


def _delete_session(identifier):
    if not identifier:
        return gr.update(), ui.notice("Choose a saved generation first.", "warn")
    mc_llm_state.delete_prompt_session(identifier)
    return gr.update(choices=_history_choices(), value=None), ui.notice("Deleted.")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _speech_note(multiplier, dialogue) -> str:
    """The sentence that says what a slider position means.

    Ten unlabelled positions say nothing; "3× the lines" says all of it. The
    lifted dialogue budget is quoted from ``speech.dialogue_floor`` rather than
    described, because that function is what actually decides it.
    """
    from prompt_master.prompt_engine import speech

    value = int(multiplier or speech.NONE)
    if value <= speech.NONE:
        return "Speech exactly as the intent quotes it."
    floor = speech.dialogue_floor(value, int(dialogue or 0))
    return (f"{value}× the lines — extra speech written to match, and the dialogue "
            f"budget raised to {floor}%.")


def _save_file(positive, negative):
    """Write the pair to a file the browser can download."""
    if not (positive or negative):
        return gr.update(visible=False)
    directory = Path(tempfile.mkdtemp(prefix="model-chain-prompts-"))
    target = directory / "prompts.txt"
    target.write_text(f"POSITIVE\n{positive or ''}\n\nNEGATIVE\n{negative or ''}\n",
                      encoding="utf-8")
    return gr.update(value=str(target), visible=True)
