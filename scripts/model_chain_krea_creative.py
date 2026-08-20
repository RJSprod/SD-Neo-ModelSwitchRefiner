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

Where the model is called
-------------------------
:meth:`ScriptKreaCreative.before_process` calls it, once, at the top of the
generation the user just started -- before Forge builds ``all_prompts`` and
before the checkpoint is (re)loaded, which is the same ordering the roll always
had and the reason the writer can still size itself against a card the image
model has not taken yet.

It used to be called from a Gradio handler instead, ahead of the click, because
an LLM run waits for the host to stop generating and a roll asked for from
inside a running image job would have been waiting on the job that was waiting
on it. The browser enforced that ordering: it swallowed the Generate click, ran
the roll, then clicked Generate again itself.

The cost of that arrangement was that a generation could not finish without a
live page. The press did not start an image; a ``setInterval`` in the tab did,
once the roll came back -- and browsers throttle those to one tick a second in a
hidden tab and one a minute in a frozen one, so a Creative generation was late
if you changed windows and never happened at all if you closed the tab.

So the deadlock is now answered where it lives: :class:`mc_broker.host_job` lets
this hook say that the image job is blocked waiting for the roll, which is
precisely the case in which waiting for the image job is the wrong thing to do.
One press does everything, and nothing after the press needs a browser.

What is not here
----------------
No idle delay, no typing watcher, no repeat toggle, no reroll scheduler, no
status machine and no click gate. A roll happens because somebody pressed
Generate. Pressing it again is how you get another one.
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


# --------------------------------------------------------------------------- #
# The settings one roll runs with
# --------------------------------------------------------------------------- #


def _settings_for(values) -> dict:
    """This generation's Creative settings, from the panel when it sent them.

    ``values`` is what Forge handed ``before_process`` after the enabled flag:
    the four scalars and then the axis table, in the order :meth:`ui` returned
    them. A UI that could not build its axis table sends fewer, and an API
    request sends none at all, so the length is checked rather than assumed and
    the saved preferences answer for anything absent.
    """
    values = tuple(values or ())
    if len(values) < 4:
        return mc_creative_krea.settings()
    creativity, seed, anti_repetition, loras = values[:4]
    return _stored(creativity, seed, anti_repetition, loras, values[4:])


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

    if not axis_values:
        # No table to read. Answering with every axis on Vary would be this
        # function inventing a configuration and handing it to _stored, which
        # writes it over whatever the user actually set.
        return {}, {}

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


def _last_roll():
    """The most recent roll, for the diagnostics drawer.

    Reads :attr:`mc_creative_krea.Creative.last`, which the roll writes as its
    final act and which outlives the page: a generation started before the tab
    was closed can be inspected in the tab that opens after it.
    """
    last = mc_creative_krea.creative.last
    if last is None:
        return ("No roll has been made yet in this session.", "")
    return _recipe_view(last.recipe), last.expanded


# --------------------------------------------------------------------------- #
# The script
# --------------------------------------------------------------------------- #


class ScriptKreaCreative(scripts.Script):
    """The Creative Mode controls, and the hook that writes the prompt."""

    def __init__(self):
        super().__init__()
        # Filled in by ui(); the shell never reads it, and a test asking what
        # the panel is made of has something to ask.
        self.components: dict = {}
        # True while this hook is inside its own roll. A nested process_images()
        # -- Stage 2's, or any extension's -- must not start a second one, and
        # the cost of getting that wrong is not a duplicate image but an LLM
        # request that begins while the first is still streaming.
        self._rolling = False

    def title(self):
        return "Krea Creative Mode"

    def show(self, is_img2img):
        # txt2img only. Creative Mode's whole shape is "the positive prompt is a
        # short idea and the image is made from an expansion of it", and
        # img2img's prompt describes an edit to a picture that already exists.
        return scripts.AlwaysVisible if not is_img2img else None

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

                # Filled on request rather than streamed. The roll happens
                # inside the generation now, where there is no open Gradio
                # event to push anything down -- and a drawer nobody opened is
                # the wrong thing to hold a websocket open for anyway. The
                # button reads whatever the last roll left behind, which is
                # still there after the tab has been closed and reopened.
                with gr.Accordion("Last creative roll", open=False,
                                  elem_id=ident("diagnostics")):
                    show = gr.Button("Show the last roll", size="sm",
                                     elem_id=ident("show"))
                    recipe = gr.Textbox(
                        label="Recipe and brief", lines=12, max_lines=12,
                        interactive=False, show_copy_button=True,
                        elem_id=ident("recipe"))
                    expanded = gr.Textbox(
                        label="Expanded Krea prompt", lines=6, max_lines=6,
                        interactive=False, show_copy_button=True,
                        elem_id=ident("expanded"))

        self.components = {
            "enabled": enabled, "creativity": creativity, "status": status,
            "controls": controls, "seed": seed, "anti": anti, "forget": forget,
            "loras": loras, "show": show, "recipe": recipe, "expanded": expanded,
            "axes": axis_controls}

        self._wire(enabled, creativity, status, controls, seed, anti, forget, loras,
                   show, recipe, expanded, axis_controls)

        # Every control travels to before_process, because that is where the
        # roll happens and the panel is what the user is looking at. They are
        # this script's own arguments and reach neither Model Chain's preset
        # list nor its infotext.
        return [enabled, creativity, seed, anti, loras] + list(axis_controls)

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
              show, recipe, expanded, axis_controls):
        """Every handler, in one place, so the component list above stays readable.

        All of them ``queue=False``: not one of these does any work worth
        queueing, and none of them starts, stops or waits for a generation. The
        panel is settings and nothing else now -- the roll is in
        :meth:`before_process`, which is not reachable from here.
        """
        enabled.change(fn=_toggled, inputs=[enabled],
                       outputs=[creativity, controls, status], queue=False)
        creativity.release(fn=_remember_creativity, inputs=[creativity],
                           outputs=[status], queue=False)
        seed.input(fn=_remember_seed, inputs=[seed], outputs=[seed], queue=False)
        anti.change(fn=_remember_anti, inputs=[anti], outputs=[anti], queue=False)
        loras.blur(fn=_remember_loras, inputs=[loras], outputs=[loras, status],
                   queue=False)
        forget.click(fn=_forget_history, outputs=[status], queue=False)
        show.click(fn=_last_roll, outputs=[recipe, expanded], queue=False)

        for control in axis_controls:
            control.change(fn=_remember_axes, inputs=list(axis_controls),
                           outputs=[status], queue=False)

    # -- generation --------------------------------------------------------- #

    def before_process(self, p, enabled=False, *args, **kwargs):
        """Write this generation's prompt, then substitute it.

        ``before_process`` and not ``process``: this runs before Forge builds
        ``all_prompts`` from ``p.prompt``, so one assignment reaches the batch,
        the styles pass, the infotext and Stage 2's inherited prompt without any
        of them having to be told about Creative Mode. It also runs before the
        checkpoint is loaded, which is what lets the writer size itself against
        a card the image model has not taken yet.

        Everything one press needs happens here, on the thread the host is
        already running the job on. Nothing is waited for in a browser and
        nothing is picked up from one: close the tab after pressing Generate and
        this still finishes, and Forge writes the files.

        Failure is always "generate what the user typed". A library that will not
        load, a checkpoint that is not Krea 2, a language model that will not
        answer, an Interrupt during the roll -- none of them is a reason to
        refuse a generation the user asked for, and all of them say so in the
        log.

        Nothing is carried in from before the press. The prompt this generation
        uses is written during this generation, from the flag and the panel
        values Forge just handed over, so there is no earlier state to be stale
        and none to be spent by the wrong image.
        """
        if not enabled:
            return
        if self._rolling:
            # A process_images() nested inside our own roll. There is nothing to
            # do for it and a great deal to get wrong.
            logger.debug("Model Chain: Creative Mode is already rolling; the nested "
                         "generation is left alone")
            return

        written = self._roll(p, args)
        if written is None:
            return

        p.prompt = written.generation
        try:
            p.extra_generation_params.update(written.metadata)
        except Exception:
            logger.debug("Model Chain: could not record the Creative Mode metadata",
                         exc_info=True)
        logger.info("Model Chain: Creative Mode prompt applied — %s characters from a "
                    "%s-character source at creativity %s, creative seed %s",
                    f"{len(written.generation):,}", f"{len(written.roll.source):,}",
                    written.roll.creativity, written.roll.creative_seed)

    def _roll(self, p, values):
        """One creative roll for this generation, or ``None`` to leave it alone.

        The whole of the deadlock fix is the two lines that matter here: the roll
        runs inside :class:`mc_broker.host_job`, which is how
        ``mc_llm_sessions._Gpu.acquire`` is told that the image job is blocked
        waiting for this request rather than competing with it, and the bar is
        borrowed rather than claimed, because the host already started one for
        the generation this is the first part of.

        The events are drained rather than forwarded. There is no open Gradio
        event to forward them to -- the press became a native generation, not a
        handler with an output list -- so the progress bar carries the phase and
        the log carries the rest.
        """
        import mc_broker

        source = str(getattr(p, "prompt", "") or "").strip()
        if not source:
            logger.info("Model Chain: Creative Mode has no source prompt to work from")
            return None

        session = mc_creative_krea.creative
        settings = _settings_for(values)
        loras = settings.get("loras", "")

        # Held by name and closed explicitly, the way the roll itself holds the
        # LLM run: every path out of the loop below leaves the generator
        # suspended, and it is its ``finally`` that gives the progress bar and
        # the workload lock back. Closing it is what runs that now rather than
        # whenever the interpreter next collects the frame.
        events = session.roll(source, settings, guard_checkpoint=True, own_bar=False)
        written = False
        self._rolling = True
        try:
            with mc_broker.host_job():
                for event in events:
                    if event.kind == sessions.STATUS:
                        logger.debug("Model Chain: Creative Mode — %s", event.text)
                    elif event.kind == sessions.CANCELLED:
                        logger.info("Model Chain: the Creative Mode roll was stopped; "
                                    "the generation continues with the typed prompt")
                        break
                    elif event.kind == sessions.FAILED:
                        logger.warning("Model Chain: the Creative Mode roll failed (%s); "
                                       "generating from the typed prompt instead",
                                       event.text)
                        break
                    elif event.kind == sessions.DONE:
                        written = True
                        break
        except Exception:
            errors.report("Model Chain: the Creative Mode roll failed", exc_info=True)
            return None
        finally:
            events.close()
            self._rolling = False

        if not written:
            return None

        last = session.last
        if last is None or not last.expanded.strip():
            logger.warning("Model Chain: the Creative Mode roll produced nothing; "
                           "generating from the typed prompt instead")
            return None
        return mc_creative_krea.prepare(last, loras)
