"""Krea Creative Mode in txt2img: two controls, and a drawer nobody has to open.

The default surface is one checkbox, one slider and a collapsed accordion:

    [ Creative Mode ]   Creativity [0----5----10]   ▸ Creative Controls

That is the whole footprint until somebody opens the drawer. Ten axes with three
modes and a value each is a lot of interface for a feature whose good answer is
usually "leave it on Vary and move the slider", so the ten axes are there for
the person who wants them and invisible to the person who does not.

A separate always-on script rather than another accordion inside Model Chain,
for three reasons that are all about blast radius. Model Chain's ``ui()`` returns
a long argument list that travels in presets and infotexts, and Creative Mode's
controls have no business in either. Model Chain is a large, long-settled thing
and a Creative gate that failed to build must not be able to take the two-stage
chain down with it. And Creative Mode is txt2img-only for a different reason than
Model Chain is, so the two ``show()`` methods happen to agree today without that
being one decision.

Where the model is called, and where it is not
----------------------------------------------
:func:`_gate` calls it, once, from a Gradio handler, before any native
generation has started. :meth:`ScriptKreaCreative.before_process` never does --
it applies an expansion that already exists and consumes the token that
permitted it. That split is not stylistic: an LLM run waits for the host to stop
generating, so an expansion requested from inside a running image job would be
waiting on the job that is waiting on it.

What is not here
----------------
No idle delay, no typing watcher, no repeat toggle, no reroll scheduler and no
status machine. A roll happens because somebody pressed Generate. Pressing it
again is how you get another one.
"""

from __future__ import annotations

import gradio as gr

import mc_creative_krea
import mc_llm_sessions as sessions
import mc_memory
from modules import errors, scripts

logger = mc_memory.logger
"""Shared with the helper modules; mc_memory attaches the console handler."""

PREFIX = "mc-krea-creative"
"""Every id this script puts in the page starts here.

The browser gate finds its elements by these ids and by nothing else -- no
Gradio-generated class, no DOM shape. A theme that replaces Gradio's internals
can change how this looks and cannot stop it working.
"""


def ident(*parts: str) -> str:
    """A stable, extension-owned element id."""
    return "-".join((PREFIX,) + tuple(str(part) for part in parts if part))


def notice(text: str, kind: str = "info") -> str:
    """One line of Creative Mode status, as scoped HTML.

    Its own classes rather than LLM Studio's, because ``style.css`` scopes those
    under ``#mc-llm-studio`` and this line is in txt2img. Same idea, same
    reliance on the host's custom properties for colour, different neighbourhood.
    """
    import html

    return (f'<div class="{PREFIX}-notice {PREFIX}-notice-{kind}">'
            f'{html.escape(str(text or ""))}</div>')


READY = "ready"
FAILED = "failed"


def _signal(kind: str, detail: str = "") -> str:
    """One line for the hidden token box: ``ready:<nonce>:`` or ``failed:<nonce>:``.

    Why a textbox and not a Gradio event the browser could subscribe to: the
    browser has to know *when* the expansion is available so it can let one
    native Generate click through, and a value it can poll for a moment after it
    asked is the smallest mechanism that does that without this script growing
    an endpoint of its own. The nonce makes each answer distinguishable from the
    last, so two identical outcomes in a row are still two events.
    """
    import secrets

    return f"{kind}:{secrets.token_hex(4)}:{detail}"


# --------------------------------------------------------------------------- #
# The handlers
# --------------------------------------------------------------------------- #


def _gate(task_id, source, creativity, seed, anti_repetition, loras, *axis_values):
    """One creative roll, made available to one native generation.

    The only place in txt2img that asks the language model for anything, and it
    asks exactly once. Everything creative has already been decided in Python by
    the time the request goes out: the Director picks the art direction from a
    vendored vocabulary with a seeded PRNG, and the model's whole job is to turn
    one brief into one Krea prompt.

    ``task_id`` is minted in the browser by :js:func:`mcKreaCreativeSubmit`,
    which has already asked Forge to draw and poll its progress bar for it --
    the same two steps the host's own Generate does. Passing it as the first
    argument, from a ``js=`` hook rather than out of a hidden textbox, is the
    host's idiom and is race-free: the value is put into the request as it is
    built rather than into a component and hoped for.

    Streams so the panel can say what is happening while a cold llama-server
    loads, which is the part of the wait that looks like a hang.
    """
    session = mc_creative_krea.creative
    stored = _stored(creativity, seed, anti_repetition, loras, axis_values)
    expanded = ""

    try:
        for event in session.roll(source, stored, guard_checkpoint=True,
                                  task_id=task_id):
            if event.kind == sessions.CHUNK:
                expanded += event.text
                yield (notice("Writing the Krea prompt…"), gr.update(),
                       gr.update(), gr.update(value=expanded))
            elif event.kind == sessions.STATUS:
                yield notice(event.text), gr.update(), gr.update(), gr.update()
            elif event.kind == sessions.DONE:
                last = session.last
                if last is None:
                    yield (notice("The prompt was written but is no longer current.",
                                  "warn"), _signal(FAILED), gr.update(), gr.update())
                    return
                armed = session.arm(last, loras)
                yield (notice(f"Prompt ready · Creativity {last.creativity} · "
                              f"Creative seed {last.creative_seed}"),
                       _signal(READY, armed.token), _recipe_view(last.recipe),
                       last.expanded)
                return
            elif event.kind == sessions.CANCELLED:
                yield (notice(event.text or "Stopped.", "warn"), _signal(FAILED),
                       gr.update(), gr.update())
                return
            elif event.kind == sessions.FAILED:
                yield (notice(event.text, "error"), _signal(FAILED),
                       gr.update(), gr.update())
                return
    except Exception as exc:
        errors.report("Model Chain: the Creative Mode gate failed", exc_info=True)
        yield notice(str(exc), "error"), _signal(FAILED), gr.update(), gr.update()
        return

    yield notice("Nothing was written.", "warn"), _signal(FAILED), gr.update(), gr.update()


def _stored(creativity, seed, anti_repetition, loras, axis_values) -> dict:
    """The settings for this roll, taken from the controls rather than the file.

    Read off the panel and not out of preferences, because the panel is what the
    user is looking at. A roll that used the last *saved* value would silently
    ignore the slider somebody had just moved, which is the sort of bug that
    only shows up as "the creativity control does not seem to do anything".
    """
    from prompt_master.krea import director, variation

    stored = mc_creative_krea.settings()
    stored["creativity"] = variation.clamp(creativity)
    stored["anti_repetition"] = bool(anti_repetition)
    stored["loras"] = mc_creative_krea.lora_suffix(loras)
    try:
        stored["seed"] = int(seed)
    except (TypeError, ValueError):
        stored["seed"] = director.RANDOM_SEED

    modes, fixed = _axes_from(axis_values)
    if modes:
        stored["axis_modes"] = modes
        stored["fixed_values"] = fixed
    return stored


def _axes_from(axis_values) -> tuple[dict, dict]:
    """The axis controls, as ``(modes, fixed ids)``.

    The values arrive as one flat tuple -- Gradio has no other shape for a
    variable number of inputs -- laid out mode, fixed, mode, fixed in the
    library's own axis order. That pairing is the one thing this function knows
    and the one thing that could go quietly wrong, so it is done in one place
    rather than at each of the handlers that needs it.
    """
    from prompt_master.krea import director

    try:
        from prompt_master.krea import library as library_module

        keys = library_module.library().axis_keys
    except Exception:
        return {}, {}

    modes, fixed = {}, {}
    for position, key in enumerate(keys):
        mode = str(axis_values[position * 2] if position * 2 < len(axis_values)
                   else director.VARY).casefold()
        chosen = axis_values[position * 2 + 1] if position * 2 + 1 < len(axis_values) else None
        modes[key] = mode if mode in director.MODES else director.VARY
        if chosen:
            fixed[key] = str(chosen)
    return modes, fixed


def _recipe_view(recipe) -> str:
    """The last recipe as something a person can read and argue with.

    Deliberately the ids *and* the labels. The ids are what the metadata records
    and what a Fixed selection stores, so somebody who liked a roll needs to see
    them; the labels are what makes the list mean anything at a glance.
    """
    items = getattr(recipe, "items", ())
    if not items:
        return (f"No creative direction at Creativity {getattr(recipe, 'creativity', 0)}.\n"
                "Creativity 0 and 1 direct nothing by design; above that, set at least "
                "one axis to Vary.")
    lines = [f"Creative seed: {recipe.creative_seed}   ·   LLM seed: {recipe.llm_seed}"
             f"   ·   library {recipe.library_version}"]
    if recipe.locked:
        lines.append("Locked by your prompt: " + ", ".join(recipe.locked))
    lines.append("")
    for item in items:
        lines.append(f"{item.label} [{item.source}] {item.variant_id} — {item.variant_label}")
    lines.append("")
    lines.append(recipe.brief)
    return "\n".join(lines)


def _toggled(enabled):
    """Show or hide the controls, and remember the toggle."""
    if not enabled:
        mc_creative_krea.creative.disarm()
    mc_creative_krea.remember(**{mc_creative_krea.ENABLED: bool(enabled)})
    shown = gr.update(visible=bool(enabled))
    if enabled:
        objection = mc_creative_krea.checkpoint_objection()
        told = notice(objection or
                      "Creative Mode is on. Press Generate: the prompt is directed "
                      "locally and expanded once, then Forge makes the image.",
                      "warn" if objection else "info")
    else:
        told = notice("Creative Mode is off.")
    return shown, shown, gr.update(value=told, visible=bool(enabled))


def _remember_creativity(value):
    from prompt_master.krea.variation import clamp, describe

    mc_creative_krea.remember(**{mc_creative_krea.CREATIVITY: clamp(value)})
    return notice(describe(value))


def _remember_seed(value):
    mc_creative_krea.remember(**{mc_creative_krea.SEED: mc_creative_krea._seed(value)})
    return gr.update()


def _remember_anti(value):
    mc_creative_krea.remember(**{mc_creative_krea.ANTI_REPETITION: bool(value)})
    return gr.update()


def _remember_loras(value):
    """Keep the pinned tags, and answer with what was actually kept.

    The box is rewritten with the parsed tags rather than with what was typed,
    which is the visible half of the rule that this field contributes networks
    and never prompt text: prose typed here disappears in front of the person
    who typed it, instead of silently reaching the image model.
    """
    suffix = mc_creative_krea.lora_suffix(value)
    mc_creative_krea.remember(**{mc_creative_krea.LORAS: suffix})
    counted = len(mc_creative_krea.pinned_tags(suffix))
    return suffix, notice(f"{counted} pinned LoRA{'' if counted == 1 else 's'}.")


def _remember_axes(*axis_values):
    modes, fixed = _axes_from(axis_values)
    mc_creative_krea.remember(**{mc_creative_krea.AXIS_MODES: modes,
                                 mc_creative_krea.FIXED_VALUES: fixed})
    return gr.update()


def _forget_history():
    mc_creative_krea.forget_history()
    return notice("Recent-roll memory cleared; every treatment is available again.")


# --------------------------------------------------------------------------- #
# The script
# --------------------------------------------------------------------------- #


class ScriptKreaCreative(scripts.Script):
    """The Creative Mode controls, and the hook that applies what the gate wrote."""

    def __init__(self):
        super().__init__()
        # The native txt2img control Creative Mode reads rather than duplicates.
        self._prompt_component = None
        # Filled in by ui(); the shell never reads it, and a test asking what
        # the panel is made of has something to ask.
        self.components: dict = {}

    def title(self):
        return "Krea Creative Mode"

    def show(self, is_img2img):
        # txt2img only. Creative Mode's whole shape is "the positive prompt is a
        # short idea and the image is made from an expansion of it", and
        # img2img's prompt describes an edit to a picture that already exists.
        return scripts.AlwaysVisible if not is_img2img else None

    def after_component(self, component, **kwargs):
        """Capture the native positive prompt: the gate's only input from the tab."""
        if (kwargs.get("elem_id") or "") == "txt2img_prompt":
            self._prompt_component = component

    # -- UI ---------------------------------------------------------------- #

    def ui(self, is_img2img):
        from prompt_master.krea import variation

        stored = mc_creative_krea.settings()

        with gr.Group(elem_id=ident("group")):
            with gr.Row(elem_id=ident("bar")):
                enabled = gr.Checkbox(
                    value=bool(stored["enabled"]), label="Creative Mode", scale=1,
                    elem_id=ident("toggle"),
                    info="direct the prompt locally, then expand it with Krea 2")
                creativity = gr.Slider(
                    label=variation.LABEL, minimum=variation.MINIMUM,
                    maximum=variation.MAXIMUM, step=1, value=stored["creativity"],
                    scale=3, visible=bool(stored["enabled"]), info=variation.HELP,
                    elem_id=ident("creativity"))

            status = gr.HTML(notice("Creative Mode is off."),
                             visible=bool(stored["enabled"]), elem_id=ident("status"))

            with gr.Accordion("Creative Controls", open=False,
                              visible=bool(stored["enabled"]),
                              elem_id=ident("controls")) as controls:
                axis_controls = self._axis_table(stored)

                with gr.Row():
                    seed = gr.Number(
                        label="Creative seed", value=stored["seed"], precision=0,
                        scale=2, elem_id=ident("seed"),
                        info="-1 rolls a new one each time; a fixed value repeats "
                             "the same art direction. Not the image seed.")
                    anti = gr.Checkbox(
                        value=bool(stored["anti_repetition"]), scale=1,
                        label="Avoid recent treatments", elem_id=ident("anti"),
                        info="pushes the last few rolls' choices away at high Creativity")
                    forget = gr.Button("Clear recent memory", size="sm", scale=1,
                                       elem_id=ident("forget"))

                loras = gr.Textbox(
                    label="Pinned LoRAs", value=stored["loras"],
                    placeholder="<lora:name:0.8> <lora:other:0.5>",
                    elem_id=ident("loras"),
                    info="appended to the generated prompt; never sent to the language model")

                gr.Markdown(
                    "**Natural** leaves the axis out of the brief entirely — the model "
                    "decides as it would without Creative Mode. **Vary** lets the local "
                    "director choose, and the Creativity slider decides whether the axis "
                    "activates at all, how strongly it is expressed, and how hard recent "
                    "choices are pushed away. **Fixed** repeats your chosen value every "
                    "roll.\n\n"
                    "Your own words always win. Type *oil painting of a car* and Medium "
                    "stays oil painting however Medium is set.\n\n"
                    "Creative Mode changes the positive prompt only. The negative prompt, "
                    "the checkpoint, the sampler, the size, Steps, the image seed and "
                    "every other setting stay exactly where Forge puts them, and the "
                    "image itself is generated by Forge.")

                with gr.Accordion("Last creative roll", open=False,
                                  elem_id=ident("diagnostics")):
                    recipe = gr.Textbox(
                        label="Recipe and brief", lines=12, max_lines=12,
                        interactive=False, show_copy_button=True,
                        elem_id=ident("recipe"))
                    expanded = gr.Textbox(
                        label="Expanded Krea prompt", lines=6, max_lines=6,
                        interactive=False, show_copy_button=True,
                        elem_id=ident("expanded"))

            # -- plumbing the browser drives, and the user never sees -------- #
            run = gr.Button("Creative Mode: roll", visible=False, elem_id=ident("run"))
            token = gr.Textbox(value="", visible=False, elem_id=ident("token"))
            # Never read as a component. It is here because a Gradio event needs
            # an input to put the browser's task id into, and the ``js=`` hook
            # below overwrites the value on its way past -- exactly as the
            # host's own submit() puts its id_task into argument zero.
            task = gr.Textbox(value="", visible=False, elem_id=ident("task"))

        self.components = {
            "enabled": enabled, "creativity": creativity, "status": status,
            "controls": controls, "seed": seed, "anti": anti, "forget": forget,
            "loras": loras, "recipe": recipe, "expanded": expanded, "run": run,
            "token": token, "task": task, "axes": axis_controls}

        self._wire(enabled, creativity, status, controls, seed, anti, forget, loras,
                   recipe, expanded, run, token, task, axis_controls)
        return [enabled]

    def _axis_table(self, stored) -> list:
        """One row per axis: what it is called, how it behaves, and its pinned value.

        Built from the library rather than from a list here, so a package that
        adds an axis grows a row without this file being edited. A library that
        will not load leaves the table empty and says why -- Creative Mode is
        then refused at the gate with the same message, rather than silently
        directing nothing.
        """
        try:
            from prompt_master.krea import library as library_module

            lib = library_module.library()
        except Exception as exc:
            gr.Markdown(f"The creativity library could not be read, so Creative Mode "
                        f"has no vocabulary to direct with: {exc}")
            return []

        from prompt_master.krea import director

        modes = stored.get("axis_modes") or {}
        fixed = stored.get("fixed_values") or {}
        rows = []
        for key in lib.axis_keys:
            axis = lib.axis(key)
            with gr.Row(elem_id=ident("axis", key)):
                mode = gr.Radio(
                    label=axis.label, choices=[("Natural", director.NATURAL),
                                               ("Vary", director.VARY),
                                               ("Fixed", director.FIXED)],
                    value=modes.get(key, director.VARY), scale=2,
                    elem_id=ident("axis", key, "mode"))
                value = gr.Dropdown(
                    label="Fixed value", scale=3, value=fixed.get(key),
                    choices=[(variant.label, variant.identifier)
                             for variant in axis.variants],
                    elem_id=ident("axis", key, "value"))
            rows.extend([mode, value])
        return rows

    def _wire(self, enabled, creativity, status, controls, seed, anti, forget, loras,
              recipe, expanded, run, token, task, axis_controls):
        """Every handler, in one place, so the component list above stays readable."""
        enabled.change(fn=_toggled, inputs=[enabled],
                       outputs=[creativity, controls, status], queue=False)
        creativity.release(fn=_remember_creativity, inputs=[creativity],
                           outputs=[status], queue=False)
        seed.input(fn=_remember_seed, inputs=[seed], outputs=[seed], queue=False)
        anti.change(fn=_remember_anti, inputs=[anti], outputs=[anti], queue=False)
        loras.blur(fn=_remember_loras, inputs=[loras], outputs=[loras, status],
                   queue=False)
        forget.click(fn=_forget_history, outputs=[status], queue=False)

        for control in axis_controls:
            control.change(fn=_remember_axes, inputs=list(axis_controls),
                           outputs=[status], queue=False)

        if self._prompt_component is not None:
            # show_progress="hidden": the roll reports itself on the host's own
            # progress bar, in the gallery where image progress appears, and
            # Gradio's little spinner over the status line would be a second
            # thing claiming to describe the same wait.
            run.click(fn=_gate,
                      inputs=[task, self._prompt_component, creativity, seed, anti,
                              loras] + list(axis_controls),
                      outputs=[status, token, recipe, expanded],
                      show_progress="hidden", js="mcKreaCreativeSubmit")
        else:
            # A UI so heavily customised that the positive prompt could not be
            # found. Creative Mode says so once, here, rather than half-working:
            # the gate has nothing to read and the browser would be arming
            # generations from an empty source.
            run.click(fn=lambda: (notice("The txt2img positive prompt could not be "
                                         "found, so Creative Mode cannot read your "
                                         "source text.", "error"), _signal(FAILED)),
                      outputs=[status, token], queue=False)

    # -- generation --------------------------------------------------------- #

    def before_process(self, p, enabled=False, *args, **kwargs):
        """Substitute the expanded prompt, if one was armed for this generation.

        ``before_process`` and not ``process``: this runs before Forge builds
        ``all_prompts`` from ``p.prompt``, so one assignment reaches the batch,
        the styles pass, the infotext and Stage 2's inherited prompt without any
        of them having to be told about Creative Mode.

        No inference happens here, ever. The roll was made by a Gradio handler
        that finished before the Generate click was allowed through, and
        consuming its token is the whole of this hook's work. It is consumed
        unconditionally so that a nested or queued generation cannot make a
        second image out of one roll's permission.
        """
        try:
            armed = mc_creative_krea.creative.consume()
        except Exception:
            errors.report("Model Chain: Creative Mode could not read its arming token",
                          exc_info=True)
            return
        if armed is None:
            return
        if not enabled:
            # Armed, then switched off before the click landed. The safe reading
            # of that is the user's most recent instruction, which was "off".
            logger.info("Model Chain: Creative Mode was disarmed before the generation "
                        "started")
            return

        p.prompt = armed.generation
        try:
            p.extra_generation_params.update(armed.metadata)
        except Exception:
            logger.debug("Model Chain: could not record the Creative Mode metadata",
                         exc_info=True)
        logger.info("Model Chain: Creative Mode prompt applied — %s characters from a "
                    "%s-character source at creativity %s, creative seed %s",
                    f"{len(armed.generation):,}", f"{len(armed.roll.source):,}",
                    armed.roll.creativity, armed.roll.creative_seed)
