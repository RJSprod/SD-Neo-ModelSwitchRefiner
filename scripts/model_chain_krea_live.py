"""Krea Live: the txt2img half, kept as small as the feature allows.

A separate always-on script rather than another accordion inside Model Chain,
for three reasons that are all about blast radius. It is txt2img-only and Model
Chain's own panel is a large, long-settled thing whose arguments are recorded
in presets and infotexts; Live's controls belong to a different feature and
have no business travelling in those; and a Live gate that failed to build must
not be able to take the two-stage chain down with it.

What is on screen
-----------------
With Live off, one checkbox. That is the whole footprint: the positive prompt
must not be able to change meaning without something visible saying so, and one
checkbox is the least that can honestly say it.

With Live on, one compact strip -- Creativity, the idle delay, a quick Steps
handle, the pinned LoRA count, reroll, and what Live is currently doing -- plus
a collapsed configuration area and a collapsed view of the last written prompt.
Width, height, sampler, scheduler, CFG, batch and checkpoint are all left
exactly where Forge already put them. Nothing here duplicates a native control
except Steps, and that one is a handle on the native component rather than a
second value: changing either changes both, so the number in the metadata is
the number that was used.

What is not on screen
---------------------
The expanded prompt never goes into the positive prompt box. The box keeps the
short phrase the user is iterating on; the paragraph the writer produced is
substituted at the processing boundary and shown, read-only, in the collapsed
view. That is the difference between a workflow you can type in and one where
every generation eats your source text.

Where the LLM is called, and where it is not
--------------------------------------------
:func:`_gate` calls it, from a Gradio handler, before any native generation has
started. :meth:`ScriptKreaLive.before_process` never does -- it applies an
expansion that already exists and consumes the token that permitted it. That
split is not stylistic: an LLM run waits for the host to stop generating, so an
expansion requested from inside a running image job would be waiting on the job
that is waiting on it.
"""

from __future__ import annotations

import gradio as gr

import mc_live_krea
import mc_llm_sessions as sessions
import mc_memory
from modules import errors, scripts

logger = mc_memory.logger
"""Shared with the helper modules; mc_memory attaches the console handler."""

PREFIX = "mc-krea-live"
"""Every id this script puts in the page starts here.

The browser controller finds every one of its elements by these ids and by
nothing else -- no Gradio-generated class, no DOM shape. A theme that replaces
Gradio's internals can change how the strip looks and cannot stop it working.
"""


def ident(*parts: str) -> str:
    """A stable, extension-owned element id."""
    return "-".join((PREFIX,) + tuple(str(part) for part in parts if part))


def notice(text: str, kind: str = "info") -> str:
    """One line of Live status, as scoped HTML.

    Its own classes rather than LLM Studio's, because ``style.css`` scopes those
    under ``#mc-llm-studio`` and this line is in txt2img. Same idea, same
    reliance on the host's custom properties for colour, different neighbourhood.
    """
    import html

    return (f'<div class="{PREFIX}-notice {PREFIX}-notice-{kind}">'
            f'{html.escape(str(text or ""))}</div>')


# Values the browser controller reads out of the token box. They are short and
# literal on purpose: this is the one string crossing from Python to JavaScript
# and a structured payload would be a second format to keep in step.
READY = "ready"
FAILED = "failed"


def _signal(kind: str, detail: str = "") -> str:
    """One line for the hidden token box: ``ready:<token>`` or ``failed:``.

    Why a textbox and not a Gradio event the browser could subscribe to: the
    browser has to know *when* the expansion is available so it can let one
    native Generate click through, and a value it can poll for a moment after
    it asked is the smallest mechanism that does that without this script
    growing an endpoint of its own. The nonce makes each answer distinguishable
    from the last, so two identical outcomes in a row are still two events.
    """
    import secrets

    return f"{kind}:{secrets.token_hex(4)}:{detail}"


# --------------------------------------------------------------------------- #
# The handlers
# --------------------------------------------------------------------------- #


def _gate(source, creativity, loras, prompt_seed):
    """Make one expansion available for one native generation.

    The one place in txt2img that may ask the language model for anything, and
    it asks at most once: :meth:`mc_live_krea.Live.prepare` reuses the cached
    expansion whenever the prompt-authoring state is unchanged, which is what
    every reroll, every Steps change and every new image seed hits.

    Streams so the strip can say what is happening while a cold llama-server
    loads, which is the part of the wait that looks like a hang.
    """
    live = mc_live_krea.live
    expanded, generation = "", ""

    try:
        for event in live.prepare(source, creativity, loras, prompt_seed):
            if event.kind == sessions.CHUNK:
                expanded += event.text
                yield (notice("Writing the Krea prompt…"), gr.update(),
                       gr.update(value=expanded), gr.update())
            elif event.kind == sessions.STATUS:
                yield notice(event.text), gr.update(), gr.update(), gr.update()
            elif event.kind == sessions.DONE:
                # Read off the arming rather than off the cache: the cache is a
                # mutable field that a keystroke in the next millisecond may
                # legitimately clear, and this line is describing the expansion
                # that was just armed, which cannot change under it.
                armed = live.armed
                if armed is None:
                    yield (notice("The prompt was written but is no longer current.",
                                  "warn"), _signal(FAILED), gr.update(), gr.update())
                    return
                yield (notice(f"Prompt ready · Creativity {armed.expansion.creativity} · "
                              f"Prompt seed {armed.expansion.prompt_seed}"),
                       _signal(READY, event.text), armed.expansion.expanded,
                       armed.generation)
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
        errors.report("Model Chain: the Krea Live gate failed", exc_info=True)
        yield notice(str(exc), "error"), _signal(FAILED), gr.update(), gr.update()
        return

    yield notice("Nothing was written.", "warn"), _signal(FAILED), gr.update(), gr.update()


def _revise():
    """The source text has moved on: drop the cached expansion, stop the writer.

    Called by the browser once per typing burst -- when an edit lands while an
    expansion is in flight -- and not once per keystroke. A late answer is
    discarded by the controller whether or not the cancel arrives in time, so
    this is an optimisation on top of a correctness property rather than the
    property itself.
    """
    mc_live_krea.live.revise()
    return notice("New text pending — waiting for you to stop typing.")


def _stop():
    """Stop Live. Leaves a running diffusion pass alone; Forge owns that."""
    mc_live_krea.live.stop()
    return notice("Krea Live stopped.")


def _toggled(enabled, creativity, delay, loras, prompt_seed):
    """Show or hide the strip, and remember where its controls were left."""
    if not enabled:
        mc_live_krea.live.stop()
    else:
        mc_live_krea.live.reset_failures()
        mc_live_krea.live.say(mc_live_krea.WAITING)
    mc_live_krea.remember(creativity=creativity, delay=delay, loras=loras,
                          seed=prompt_seed)
    shown = gr.update(visible=bool(enabled))
    if enabled:
        objection = mc_live_krea.checkpoint_objection()
        said = objection or (
            "Krea Live is on. Type in the positive prompt and stop; the prompt is "
            "expanded once and every image after that reuses it.")
        told = notice(said, "warn" if objection else "info")
    else:
        told = notice("Krea Live is off.")
    return shown, shown, gr.update(value=told, visible=bool(enabled))


def _remember_creativity(value):
    mc_live_krea.remember(creativity=value)
    from prompt_master.krea.variation import describe

    return notice(describe(value))


def _remember_delay(value):
    mc_live_krea.remember(delay=value)
    return gr.update()


def _remember_loras(value):
    """Keep the pinned tags, and answer with what was actually kept.

    The box is rewritten with the parsed tags rather than with what was typed,
    which is the visible half of the rule that this field contributes networks
    and never prompt text: prose typed here disappears in front of the person
    who typed it, instead of silently reaching the image model.
    """
    suffix = mc_live_krea.lora_suffix(value)
    mc_live_krea.remember(loras=suffix)
    counted = len(mc_live_krea.pinned_tags(suffix))
    return suffix, notice(f"{counted} pinned LoRA{'' if counted == 1 else 's'}.")


def _remember_seed(value):
    mc_live_krea.remember(seed=value)
    return gr.update()


# --------------------------------------------------------------------------- #
# The script
# --------------------------------------------------------------------------- #


class ScriptKreaLive(scripts.Script):
    """The Live strip, and the hook that applies what the strip's gate wrote."""

    def __init__(self):
        super().__init__()
        # The native txt2img controls Live borrows rather than duplicates.
        self._prompt_component = None
        self._steps_component = None
        # Filled in by ui(); the shell never reads it, and a test asking what
        # the strip is made of has something to ask.
        self.components: dict = {}

    def title(self):
        return "Krea Live"

    def show(self, is_img2img):
        # txt2img only. Live's whole shape is "the positive prompt is a short
        # idea and the image is made from an expansion of it", and img2img's
        # prompt is a description of an edit to a picture that already exists.
        return scripts.AlwaysVisible if not is_img2img else None

    _TRACKED_COMPONENTS = {
        "txt2img_prompt": "_prompt_component",
        "txt2img_steps": "_steps_component",
    }

    def after_component(self, component, **kwargs):
        """Capture the native prompt and Steps controls."""
        attribute = self._TRACKED_COMPONENTS.get(kwargs.get("elem_id") or "")
        if attribute is not None:
            setattr(self, attribute, component)

    # -- UI ---------------------------------------------------------------- #

    def ui(self, is_img2img):
        from prompt_master.krea import variation

        stored = mc_live_krea.remembered()

        with gr.Group(elem_id=ident("group")):
            with gr.Row(elem_id=ident("bar")):
                enabled = gr.Checkbox(
                    value=False, label="Krea Live", elem_id=ident("toggle"),
                    info="expand the positive prompt with Krea 2 before generating")

            with gr.Row(visible=False, elem_id=ident("strip")) as strip:
                creativity = gr.Slider(
                    label=variation.LABEL, minimum=variation.MINIMUM,
                    maximum=variation.MAXIMUM, step=1, value=stored["creativity"],
                    scale=3, elem_id=ident("creativity"))
                delay = gr.Number(
                    label="Idle delay (s)", value=stored["delay"],
                    minimum=mc_live_krea.MIN_DELAY, maximum=mc_live_krea.MAX_DELAY,
                    step=0.5, scale=1, elem_id=ident("delay"))
                steps = gr.Number(
                    label="Steps", value=getattr(self._steps_component, "value", None),
                    precision=0, scale=1, elem_id=ident("steps"),
                    info="the same value as the Steps slider above")
                reroll = gr.Checkbox(
                    value=False, label="Reroll", scale=1, elem_id=ident("reroll"),
                    info="keep drawing new seeds from the same prompt")

            status = gr.HTML(notice("Krea Live is off."), visible=False,
                             elem_id=ident("status"))

            with gr.Accordion("Krea Live configuration", open=False, visible=False,
                              elem_id=ident("config")) as config:
                loras = gr.Textbox(
                    label="Pinned LoRAs", value=stored["loras"],
                    placeholder="<lora:name:0.8> <lora:other:0.5>",
                    elem_id=ident("loras"),
                    info="appended to the generated prompt; never sent to the language model")
                prompt_seed = gr.Number(
                    label="Prompt seed", value=stored["seed"], precision=0,
                    elem_id=ident("seed"),
                    info="-1 draws a fresh seed for each new prompt; it is not the image seed")
                gr.Markdown(
                    "Krea Live changes the positive prompt only. The negative prompt, "
                    "the checkpoint, the sampler, the size, the image seed and every "
                    "other setting stay exactly where Forge puts them, and the image "
                    "itself is generated by Forge.\n\n"
                    "References are not part of Live: a reference image needs a "
                    "captioning pass of its own before the prompt can be written, and "
                    "Live is built around one model request per prompt. Use "
                    "**LLM Studio → Krea 2** for prompts written from reference images.")

                expanded = gr.Textbox(
                    label="Last Krea prompt (as written)", lines=6, max_lines=6,
                    interactive=False, show_copy_button=True, elem_id=ident("expanded"))
                generation = gr.Textbox(
                    label="Last generation prompt (with pinned LoRAs)", lines=4,
                    max_lines=4, interactive=False, show_copy_button=True,
                    elem_id=ident("generation"))

            # -- plumbing the browser drives, and the user never sees ------- #
            run = gr.Button("Krea Live: expand", visible=False, elem_id=ident("run"))
            revise = gr.Button("Krea Live: revise", visible=False, elem_id=ident("revise"))
            halt = gr.Button("Krea Live: stop", visible=False, elem_id=ident("halt"))
            token = gr.Textbox(value="", visible=False, elem_id=ident("token"))

        # Kept as a named map, not returned as script arguments. Only ``enabled``
        # is an argument: the rest are wired to their own handlers, and every
        # extra argument here would be another value travelling in presets and
        # infotexts that has nothing to do with the image.
        self.components = {
            "enabled": enabled, "creativity": creativity, "delay": delay, "steps": steps,
            "reroll": reroll, "status": status, "strip": strip, "config": config,
            "loras": loras, "prompt_seed": prompt_seed, "expanded": expanded,
            "generation": generation, "run": run, "revise": revise, "halt": halt,
            "token": token}

        self._wire(enabled, creativity, delay, steps, reroll, status, strip, config,
                   loras, prompt_seed, expanded, generation, run, revise, halt, token)
        return [enabled]

    def _wire(self, enabled, creativity, delay, steps, reroll, status, strip, config,
              loras, prompt_seed, expanded, generation, run, revise, halt, token):
        """Every handler, in one place, so the component list above stays readable."""
        enabled.change(fn=_toggled, inputs=[enabled, creativity, delay, loras, prompt_seed],
                       outputs=[strip, config, status], queue=False)

        creativity.release(fn=_remember_creativity, inputs=[creativity], outputs=[status],
                           queue=False)
        delay.input(fn=_remember_delay, inputs=[delay], outputs=[delay], queue=False)
        loras.blur(fn=_remember_loras, inputs=[loras], outputs=[loras, status], queue=False)
        prompt_seed.input(fn=_remember_seed, inputs=[prompt_seed], outputs=[prompt_seed],
                          queue=False)

        if self._prompt_component is not None:
            run.click(fn=_gate,
                      inputs=[self._prompt_component, creativity, loras, prompt_seed],
                      outputs=[status, token, expanded, generation],
                      show_progress="minimal")
        else:
            # A UI so heavily customised that the positive prompt could not be
            # found. Live says so once, here, rather than half-working: the gate
            # has nothing to read and the browser would be arming generations
            # from an empty source.
            run.click(fn=lambda: (notice("The txt2img positive prompt could not be found, "
                                         "so Krea Live cannot read your source text.",
                                         "error"), _signal(FAILED)),
                      outputs=[status, token], queue=False)

        revise.click(fn=_revise, outputs=[status], queue=False)
        halt.click(fn=_stop, outputs=[status], queue=False)

        self._wire_steps(steps)

    def _wire_steps(self, steps):
        """One Steps value, reachable from two places.

        Both directions are bound to user-input events -- ``release`` on the
        native slider, ``input`` on the Live box -- and never to ``change``,
        which also fires when the server writes a value. With ``change`` the two
        controls would answer each other in a loop; with these, a value only
        moves when a person moved it.

        Deliberately not a "Live Steps" setting. A hidden override that made the
        native slider say 20 while the generation ran 8 would put a number in
        the PNG metadata that never happened.
        """
        native = self._steps_component
        if native is None:
            return
        steps.input(fn=lambda value: value, inputs=[steps], outputs=[native], queue=False)
        native.release(fn=lambda value: value, inputs=[native], outputs=[steps],
                       queue=False)

    # -- generation --------------------------------------------------------- #

    def before_process(self, p, enabled=False, *args, **kwargs):
        """Substitute the expanded prompt, if one was armed for this generation.

        ``before_process`` and not ``process``: this runs before Forge builds
        ``all_prompts`` from ``p.prompt``, so one assignment reaches the batch,
        the styles pass, the infotext and Stage 2's inherited prompt without any
        of them having to be told about Live.

        No inference happens here, ever. The token was written by a Gradio
        handler that finished before the Generate click was allowed through, and
        consuming it is the whole of this hook's work. It is consumed
        unconditionally so that a nested or queued generation cannot make a
        second image out of one expansion's permission.
        """
        try:
            armed = mc_live_krea.live.consume()
        except Exception:
            errors.report("Model Chain: Krea Live could not read its arming token",
                          exc_info=True)
            return
        if armed is None:
            return
        if not enabled:
            # Armed, then switched off before the click landed. The safe reading
            # of that is the user's most recent instruction, which was "off".
            logger.info("Model Chain: Krea Live was disarmed before the generation started")
            return

        p.prompt = armed.generation
        try:
            p.extra_generation_params.update(armed.metadata)
        except Exception:
            logger.debug("Model Chain: could not record the Krea Live metadata",
                         exc_info=True)
        logger.info("Model Chain: Krea Live prompt applied — %s characters from a "
                    "%s-character source at creativity %s",
                    f"{len(armed.generation):,}", f"{len(armed.expansion.source):,}",
                    armed.expansion.creativity)
